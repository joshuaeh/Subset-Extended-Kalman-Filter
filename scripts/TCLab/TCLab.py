#!
#########################################################
# Imports
#########################################################
import datetime
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import Sobol
import torch

from gekko import GEKKO
from jtoolbox.logger import H5Logger
from sekf.modeling import AbstractNN, mask_fn
from sekf.optimizers import maskedAdam

#########################################################
# Config
#########################################################
rng = np.random.default_rng(42)
logger = H5Logger("TCLab.h5", overwrite=False)

M_TRAINING_EXAMPLES = 16
N_TRAINING_EXAMPLES = 2**M_TRAINING_EXAMPLES
M_TESTING_EXAMPLES = 14
N_TESTING_EXAMPLES = 2**M_TESTING_EXAMPLES
M_TRANSFER_EXAMPLES = 10
N_TRANSFER_EXAMPLES = 2**M_TRANSFER_EXAMPLES
M_TRANSFER_TEST_EXAMPLES = 10
N_TRANSFER_TEST_EXAMPLES = 2**M_TRANSFER_TEST_EXAMPLES

T_MIN = 23
T_MAX = 46
T_RANGE = T_MAX - T_MIN
Q_MIN = 0
Q_MAX = 100
Q_RANGE = Q_MAX - Q_MIN


#########################################################
# Sobol Sampling
#########################################################
def simulate_TCLab_Data(data_prefix, m_examples, **kwargs):
    if not data_prefix.endswith("/"):
        data_prefix += "/"

    sampler = Sobol(d=8, scramble=True, rng=rng)
    if not logger.check_key(data_prefix + "sobol_sampling"):
        sobol_samples = sampler.random_base2(m=m_examples)
        logger.log_attribute(data_prefix + "sobol_sampling", sobol_samples)
    else:
        sobol_samples = logger.get_dataset(data_prefix + "sobol_sampling")

    # Initialize GEKKO system
    # must define:
    #     TH1, TH2, TC1, TC2 (initial values)
    #     TC1sp, TC2sp (setpoints across all discretization)
    tf_min = 30  # time in minutes
    tf = tf_min * 60  # (sec)
    n = tf_min * 2 + 1  # one point every 5 seconds

    m = GEKKO(name="tclab-mpc", remote=False)

    m.time = np.linspace(0, tf, n)  # time points for simulation

    # Parameters from Estimation
    K1 = m.FV(value=kwargs.get("k1", 0.607))
    K2 = m.FV(value=kwargs.get("k2", 0.293))
    K3 = m.FV(value=kwargs.get("k3", 0.24))
    tau12 = m.FV(value=kwargs.get("tau12", 192))
    tau3 = m.FV(value=kwargs.get("tau3", 15))

    # don't update parameters with optimizer
    K1.STATUS = 0
    K2.STATUS = 0
    K3.STATUS = 0
    tau12.STATUS = 0
    tau3.STATUS = 0

    # Manipulated variables
    Q1 = m.MV(value=0, name="q1")
    Q1.STATUS = 1  # manipulated by optimizer
    # Q1.DMAX = 20.0
    # Q1.DCOST = 0.1
    Q1.UPPER = 100.0
    Q1.LOWER = 0.0

    Q2 = m.MV(value=0, name="q2")
    Q2.STATUS = 1  # manipulated by optimizer
    # Q2.DMAX = 30.0
    # Q2.DCOST = 0.1
    Q2.UPPER = 100.0
    Q2.LOWER = 0.0

    # State variables
    TH1 = m.SV(value=23.0)
    TH2 = m.SV(value=23.0)

    # Controlled variables (not implemented as CV)
    # TC1 = m.CV(value=T1m[0],name='tc1')
    # TC1.STATUS = 1     # drive to set point
    # TC1.FSTATUS = 0    # receive measurement
    # TC1.TAU = 40       # response speed (time constant)
    # TC1.TR_INIT = 1    # reference trajectory
    # TC1.TR_OPEN = 0
    # TC1.SP = T1sp

    # TC2 = m.CV(value=T2m[0],name='tc2')
    # TC2.STATUS = 1     # drive to set point
    # TC2.FSTATUS = 0   # receive measurement
    # TC2.TAU = 0        # response speed (time constant)
    # TC2.TR_INIT = 0    # dead-band
    # TC2.TR_OPEN = 1
    # TC2.SP = T2sp

    TC1 = m.Var(value=23.0, name="tc1")  # use Var to initialize with T1m[0]
    TC2 = m.Var(value=23.0, name="tc2")  # use Var to initialize with T2m[0]
    TC1sp = m.Param(value=np.ones(n) * 23.0)
    TC2sp = m.Param(value=np.ones(n) * 23.0)
    Ta = m.Param(value=23.0)  # degC

    # Heat transfer between two heaters
    DT = m.Intermediate(TH2 - TH1)

    # Empirical correlations
    m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
    m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
    m.Equation(tau3 * TC1.dt() + TC1 == TH1)
    m.Equation(tau3 * TC2.dt() + TC2 == TH2)

    m.Obj(
        (TC1 - TC1sp) ** 2  # minimize error from setpoint
        + (TC2 - TC2sp) ** 2  # minimize error from setpoint
    )

    # Global Options
    m.options.IMODE = 6  # MPC

    try:
        s = logger.get_dataset(data_prefix + "Q1").shape[0]
    except KeyError:
        s = 0

    for i in range(s, sobol_samples.shape[0]):
        try:
            sobol_sample = sobol_samples[i]
            T10, T20, T1sp_1, T1sp_2, T1sp_3, T2sp_1, T2sp_2, T2sp_3 = T_RANGE * sobol_sample + T_MIN
            print(
                f"Run {i + 1}/{sobol_samples.shape[0]}: T10: {T10:.2f}, T20: {T20:.2f}, T1sp_1: {T1sp_1:.2f}, T1sp_2: {T1sp_2:.2f}, T1sp_3: {T1sp_3:.2f}, T2sp_1: {T2sp_1:.2f}, T2sp_2: {T2sp_2:.2f}, T2sp_3: {T2sp_3:.2f}",
                end="\r",
            )
            T1sp = np.ones(n) * T1sp_1
            T2sp = np.ones(n) * T2sp_1
            T1sp[20:] = T1sp_2
            T1sp[40:] = T1sp_3
            T2sp[20:] = T2sp_2
            T2sp[40:] = T2sp_3
            # print(f"T10: {T10}, T20: {T20}, T1sp: {T1sp}, T2sp: {T2sp}")

            #########################################################
            # update Model
            #########################################################

            # State variables
            TH1.value = T10
            TH2.value = T20

            TC1.value = T10
            TC2.value = T20
            TC1sp.value = T1sp
            TC2sp.value = T2sp

            sol = m.solve(disp=False)

            logger.log_dict(
                {
                    data_prefix + "T10": T10,
                    data_prefix + "T20": T20,
                    data_prefix + "T1sp": T1sp,
                    data_prefix + "T2sp": T2sp,
                    data_prefix + "TH1": TH1.value,
                    data_prefix + "TH2": TH2.value,
                    data_prefix + "TC1": TC1.value,
                    data_prefix + "TC2": TC2.value,
                    data_prefix + "Q1": Q1.value,
                    data_prefix + "Q2": Q2.value,
                    data_prefix + "SSE": np.array((T1sp - TC1.value) ** 2 + (T2sp - TC2.value) ** 2),
                }
            )
        except Exception as e:
            print(e)
            logger.log_dict(
                {
                    data_prefix + "T10": T10,
                    data_prefix + "T20": T20,
                    data_prefix + "T1sp": T1sp,
                    data_prefix + "T2sp": T2sp,
                    data_prefix + "TH1": np.nan * np.ones(61),
                    data_prefix + "TH2": np.nan * np.ones(61),
                    data_prefix + "TC1": np.nan * np.ones(61),
                    data_prefix + "TC2": np.nan * np.ones(61),
                    data_prefix + "Q1": np.nan * np.ones(61),
                    data_prefix + "Q2": np.nan * np.ones(61),
                    data_prefix + "SSE": np.nan,
                }
            )
    m.cleanup()
    return


def TCLab_mpc_result(t10, t20, t1sp, t2sp, **kwargs):
    ### Simulate
    # MPC Prediction
    tf_min = 30  # time in minutes
    tf = tf_min * 60  # (sec)
    n = tf_min * 2 + 1  # one point every 5 seconds

    m = GEKKO(name="tclab-mpc", remote=False)

    m.time = np.linspace(0, tf, n)  # time points for simulation

    # Parameters from Estimation
    K1 = m.FV(value=kwargs.get("k1", 0.607))
    K2 = m.FV(value=kwargs.get("k2", 0.293))
    K3 = m.FV(value=kwargs.get("k3", 0.24))
    tau12 = m.FV(value=kwargs.get("tau12", 192))
    tau3 = m.FV(value=kwargs.get("tau3", 15))

    # don't update parameters with optimizer
    K1.STATUS = 0
    K2.STATUS = 0
    K3.STATUS = 0
    tau12.STATUS = 0
    tau3.STATUS = 0

    # Manipulated variables
    Q1 = m.MV(value=0, name="q1")
    Q1.STATUS = 1  # manipulated by optimizer
    # Q1.DMAX = 20.0
    # Q1.DCOST = 0.1
    Q1.UPPER = 100.0
    Q1.LOWER = 0.0

    Q2 = m.MV(value=0, name="q2")
    Q2.STATUS = 1  # manipulated by optimizer
    # Q2.DMAX = 30.0
    # Q2.DCOST = 0.1
    Q2.UPPER = 100.0
    Q2.LOWER = 0.0

    # State variables
    TH1 = m.SV(value=23.0)
    TH2 = m.SV(value=23.0)

    # Controlled variables (not implemented as CV)
    # TC1 = m.CV(value=T1m[0],name='tc1')
    # TC1.STATUS = 1     # drive to set point
    # TC1.FSTATUS = 0    # receive measurement
    # TC1.TAU = 40       # response speed (time constant)
    # TC1.TR_INIT = 1    # reference trajectory
    # TC1.TR_OPEN = 0
    # TC1.SP = T1sp

    # TC2 = m.CV(value=T2m[0],name='tc2')
    # TC2.STATUS = 1     # drive to set point
    # TC2.FSTATUS = 0   # receive measurement
    # TC2.TAU = 0        # response speed (time constant)
    # TC2.TR_INIT = 0    # dead-band
    # TC2.TR_OPEN = 1
    # TC2.SP = T2sp

    TC1 = m.Var(value=23.0, name="tc1")  # use Var to initialize with T1m[0]
    TC2 = m.Var(value=23.0, name="tc2")  # use Var to initialize with T2m[0]
    TC1sp = m.Param(value=np.ones(n) * 23.0)
    TC2sp = m.Param(value=np.ones(n) * 23.0)
    Ta = m.Param(value=23.0)  # degC

    # Heat transfer between two heaters
    DT = m.Intermediate(TH2 - TH1)

    # Empirical correlations
    m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
    m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
    m.Equation(tau3 * TC1.dt() + TC1 == TH1)
    m.Equation(tau3 * TC2.dt() + TC2 == TH2)

    m.Obj(
        (TC1 - TC1sp) ** 2  # minimize error from setpoint
        + (TC2 - TC2sp) ** 2  # minimize error from setpoint
    )

    # Global Options
    m.options.IMODE = 6  # MPC

    # different inputs
    # State variables
    TH1.value = t10
    TH2.value = t20

    TC1.value = t10
    TC2.value = t20
    TC1sp.value = t1sp
    TC2sp.value = t2sp

    try:
        sol = m.solve(disp=False)
        m.cleanup()
        return np.array(Q1.value), np.array(Q2.value), np.array(TC1.value), np.array(TC2.value)
    except Exception as e:
        print(e)
        m.cleanup()
        return t1sp * np.nan, t1sp * np.nan, t1sp * np.nan, t1sp * np.nan


def tclab_sim(t10, t20, q1, q2, **kwargs):
    ### Simulate
    # MPC Prediction
    tf_min = 30  # time in minutes
    tf = tf_min * 60  # (sec)
    n = tf_min * 2 + 1  # one point every 5 seconds

    #########################################################
    # Initialize Model
    #########################################################
    m = GEKKO(remote=False)

    # with a local server
    # m = GEKKO(name='tclab-mpc',server='http://127.0.0.1',remote=True)

    m.time = np.linspace(0, tf, n)  # time points for simulation

    # Parameters from Estimation
    K1 = m.FV(value=kwargs.get("k1", 0.607))
    K2 = m.FV(value=kwargs.get("k2", 0.293))
    K3 = m.FV(value=kwargs.get("k3", 0.24))
    tau12 = m.FV(value=kwargs.get("tau12", 192))
    tau3 = m.FV(value=kwargs.get("tau3", 15))

    # don't update parameters with optimizer
    # K1.STATUS = 0
    # K2.STATUS = 0
    # K3.STATUS = 0
    # tau12.STATUS = 0
    # tau3.STATUS = 0

    # heater setting
    Q1 = m.Param(name="Q1")
    Q1.value = q1
    Q2 = m.Param(name="Q2")
    Q2.value = q2

    # State variables
    TH1 = m.Var(value=t10, name="th1")  # use Var to initialize with T1m[0]
    TH2 = m.Var(value=t20, name="th2")  # use Var to initialize with T2m[0])
    TC1 = m.Var(value=t10, name="tc1")  # use Var to initialize with T1m[0]
    TC2 = m.Var(value=t20, name="tc2")  # use Var to initialize with T2m[0]
    Ta = m.Param(value=23.0)  # degC

    # Heat transfer between two heaters
    DT = m.Intermediate(TH2 - TH1)

    # Empirical correlations
    m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
    m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
    m.Equation(tau3 * TC1.dt() + TC1 == TH1)
    m.Equation(tau3 * TC2.dt() + TC2 == TH2)
    m.options.IMODE = 4  # simulataneous simulation
    m.solve(disp=False)

    m.cleanup()

    return np.array(TC1.value), np.array(TC2.value)


def tclab_sim(t10, t20, q1, q2, **kwargs):
    ### Simulate
    # MPC Prediction
    tf_min = 30  # time in minutes
    tf = tf_min * 60  # (sec)
    n = tf_min * 2 + 1  # one point every 5 seconds

    #########################################################
    # Initialize Model
    #########################################################
    m = GEKKO(name="tclab-sim", remote=True)

    # with a local server
    # m = GEKKO(name='tclab-mpc',server='http://127.0.0.1',remote=True)

    m.time = np.linspace(0, tf, n)  # time points for simulation

    # Parameters from Estimation
    K1 = m.FV(value=kwargs.get("k1", 0.607))
    K2 = m.FV(value=kwargs.get("k2", 0.293))
    K3 = m.FV(value=kwargs.get("k3", 0.24))
    tau12 = m.FV(value=kwargs.get("tau12", 192))
    tau3 = m.FV(value=kwargs.get("tau3", 15))

    # don't update parameters with optimizer
    # K1.STATUS = 0
    # K2.STATUS = 0
    # K3.STATUS = 0
    # tau12.STATUS = 0
    # tau3.STATUS = 0

    # heater setting
    Q1 = m.Param(name="Q1")
    Q1.value = q1
    Q2 = m.Param(name="Q2")
    Q2.value = q2

    # State variables
    TH1 = m.Var(value=t10, name="th1")  # use Var to initialize with T1m[0]
    TH2 = m.Var(value=t20, name="th2")  # use Var to initialize with T2m[0])
    TC1 = m.Var(value=t10, name="tc1")  # use Var to initialize with T1m[0]
    TC2 = m.Var(value=t20, name="tc2")  # use Var to initialize with T2m[0]
    Ta = m.Param(value=23.0)  # degC

    # Heat transfer between two heaters
    DT = m.Intermediate(TH2 - TH1)

    # Empirical correlations
    m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
    m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
    m.Equation(tau3 * TC1.dt() + TC1 == TH1)
    m.Equation(tau3 * TC2.dt() + TC2 == TH2)
    m.options.IMODE = 4  # simulataneous simulation
    m.solve(disp=False)

    m.cleanup()

    return TC1.value, TC2.value


#########################################################
# Datasets and NN Training
#########################################################
class TCLabDataset(torch.utils.data.Dataset):
    def __init__(self, logger, prefix="data/train/", begin_index=0, end_index=-1):
        self.logger = logger
        self.begin_index = begin_index
        self.end_index = end_index
        self.T_range = 46 - 23
        self.Q_range = 100 - 0
        self.Tmin = 23
        self.Qmin = 0

        # ensure prefix ends with /
        if not prefix.endswith("/"):
            prefix += "/"

        # TODO: Remove indexes with any nan values
        t10 = logger.get_dataset(prefix + "T10")[begin_index:end_index]
        t20 = logger.get_dataset(prefix + "T20")[begin_index:end_index]
        t1sp = logger.get_dataset(prefix + "T1sp")[begin_index:end_index]
        t2sp = logger.get_dataset(prefix + "T2sp")[begin_index:end_index]
        q1 = logger.get_dataset(prefix + "Q1")[begin_index:end_index]
        q2 = logger.get_dataset(prefix + "Q2")[begin_index:end_index]

        combined_array = np.concatenate((t10[:, None], t20[:, None], t1sp, t2sp, q1, q2), axis=1)

        valid_mask = np.isnan(combined_array).sum(axis=1) == 0

        self.T10 = torch.tensor(
            self._Tscale(t10[valid_mask]),
            dtype=torch.float32,
        )
        self.T20 = torch.tensor(
            self._Tscale(t20[valid_mask]),
            dtype=torch.float32,
        )
        self.T1sp = torch.tensor(
            self._Tscale(t1sp[valid_mask]),
            dtype=torch.float32,
        )
        self.T2sp = torch.tensor(
            self._Tscale(t2sp[valid_mask]),
            dtype=torch.float32,
        )
        self.Q1 = torch.tensor(
            self._Qscale(q1[valid_mask]),
            dtype=torch.float32,
        )
        self.Q2 = torch.tensor(
            self._Qscale(q2[valid_mask]),
            dtype=torch.float32,
        )

    def __len__(self):
        return self.T10.shape[0]

    def _Tscale(self, T):
        return (T - self.Tmin) / self.T_range

    def _Qscale(self, Q):
        return (Q - self.Qmin) / self.Q_range

    def _Tunscale(self, T):
        return T * self.T_range + self.Tmin

    def _Qunscale(self, Q):
        return Q * self.Q_range + self.Qmin

    def __getitem__(self, idx):
        T10i = self.T10[idx : idx + 1]
        T20i = self.T20[idx : idx + 1]
        T1spi = self.T1sp[idx]
        T2spi = self.T2sp[idx]
        Q1i = self.Q1[idx]
        Q2i = self.Q2[idx]

        return T10i, T20i, T1spi, T2spi, Q1i, Q2i

    def __getitem__np(self, idx):
        T10i, T20i, T1spi, T2spi, Q1i, Q2i = self.__getitem__(idx)
        t10i = self._Tunscale(T10i).numpy()
        t20i = self._Tunscale(T20i).numpy()
        t1spi = self._Tunscale(T1spi).numpy()
        t2spi = self._Tunscale(T2spi).numpy()
        q1i = self._Qunscale(Q1i).numpy()
        q2i = self._Qunscale(Q2i).numpy()

    def get_all(self):
        return (
            self.T10.reshape(-1, 1),
            self.T20.reshape(-1, 1),
            self.T1sp.reshape(-1, 61),
            self.T2sp.reshape(-1, 61),
            self.Q1.reshape(-1, 61),
            self.Q2.reshape(-1, 61),
        )


def compare_SSE(index, NN, dataset):
    t10, t20, t1sp, t2sp, q1, q2 = dataset.__getitem__(index)
    q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
    q1_p = q1_p.detach().numpy()
    q2_p = q2_p.detach().numpy()

    t10 = dataset._Tunscale(t10).numpy()
    t20 = dataset._Tunscale(t20).numpy()
    t1sp = dataset._Tunscale(t1sp).numpy()
    t2sp = dataset._Tunscale(t2sp).numpy()
    q1 = dataset._Qunscale(q1).numpy()
    q2 = dataset._Qunscale(q2).numpy()
    q1_p = dataset._Qunscale(q1_p)
    q2_p = dataset._Qunscale(q2_p)

    t1_sim_mpc, t2_sim_mpc = tclab_sim(t10, t20, q1, q2)
    t1_sim_nn, t2_sim_nn = tclab_sim(t10, t20, q1_p, q2_p)

    SSE_mpc = np.sum((t1_sim_mpc - t1sp) ** 2) + np.sum((t2_sim_mpc - t2sp) ** 2)
    SSE_nn = np.sum((t1_sim_nn - t1sp) ** 2) + np.sum((t2_sim_nn - t2sp) ** 2)
    return SSE_mpc, SSE_nn


def sample_SSE(n_samples, NN, dataset):
    # Sample n_samples from the dataset
    indices = rng.choice(len(dataset), n_samples, replace=False)
    SSE_mpc = []
    SSE_nn = []
    for i in indices:
        sse_mpc, sse_nn = compare_SSE(i, NN, dataset)
        SSE_mpc.append(sse_mpc)
        SSE_nn.append(sse_nn)
    return np.array(SSE_mpc), np.array(SSE_nn)


def compare_SSE(index, NN, dataset):
    t10, t20, t1sp, t2sp, q1, q2 = dataset.__getitem__(index)
    q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
    q1_p = q1_p.detach().numpy()
    q2_p = q2_p.detach().numpy()

    t10 = dataset._Tunscale(t10).numpy()
    t20 = dataset._Tunscale(t20).numpy()
    t1sp = dataset._Tunscale(t1sp).numpy()
    t2sp = dataset._Tunscale(t2sp).numpy()
    q1 = dataset._Qunscale(q1).numpy()
    q2 = dataset._Qunscale(q2).numpy()
    q1_p = dataset._Qunscale(q1_p)
    q2_p = dataset._Qunscale(q2_p)

    t1_sim_mpc, t2_sim_mpc = tclab_sim(t10, t20, q1, q2)
    t1_sim_nn, t2_sim_nn = tclab_sim(t10, t20, q1_p, q2_p)

    SSE_mpc = np.sum((t1_sim_mpc - t1sp) ** 2) + np.sum((t2_sim_mpc - t2sp) ** 2)
    SSE_nn = np.sum((t1_sim_nn - t1sp) ** 2) + np.sum((t2_sim_nn - t2sp) ** 2)
    return SSE_mpc, SSE_nn


def sample_SSE(n_samples, NN, dataset):
    # Sample n_samples from the dataset
    indices = rng.choice(len(dataset), n_samples, replace=False)
    SSE_mpc = []
    SSE_nn = []
    for i in indices:
        sse_mpc, sse_nn = compare_SSE(i, NN, dataset)
        SSE_mpc.append(sse_mpc)
        SSE_nn.append(sse_nn)
    return np.array(SSE_mpc), np.array(SSE_nn)


class SimpleNN(AbstractNN):
    def __init__(self, embedding_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1a = torch.nn.Linear(1, embedding_dim)
        self.fc1b = torch.nn.Linear(1, embedding_dim)
        self.fc1c = torch.nn.Linear(61, embedding_dim)
        self.fc1d = torch.nn.Linear(61, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc2a = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc2b = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc3a = torch.nn.Linear(embedding_dim, 61)
        self.fc3b = torch.nn.Linear(embedding_dim, 61)
        self.hard_sigmoid = torch.nn.Hardsigmoid()

    def forward(self, T10, T20, T1sp, T2sp):
        x = torch.relu(self.fc1a(T10) + self.fc1b(T20) + self.fc1c(T1sp) + self.fc1d(T2sp))
        x = torch.relu(self.fc2(x))
        xa = torch.relu(self.fc2a(x))
        xb = torch.relu(self.fc2b(x))
        q1 = self.hard_sigmoid(self.fc3a(xa))
        q2 = self.hard_sigmoid(self.fc3b(xb))
        return q1, q2


def log_trial_info():
    pass


def check_duplicate_trial_info():
    pass


def identify_best_trial(base_key, logger=logger, rescrape=False):
    with logger.open_log() as f:
        base = f[base_key]
        # No best trial found, so we need to find the best one
        # Find all groups in the base key
        group_keys = [k for k in base.keys() if isinstance(base[k], h5py.Group)]
        best_trial = None
        best_loss = None
        for group_key in group_keys:
            group = base[group_key]
            try:
                train_loss = group["train_loss"][:].min()
                test_loss = group["test_loss"][:].min()
            except KeyError:
                continue
            if best_loss is None or (train_loss + test_loss) < best_loss:
                best_loss = train_loss + test_loss
                best_trial = group_key
            base.attrs["best_trial"] = best_trial
            base.attrs["best_train_loss"] = base[best_trial]["train_loss"][:].min()
            base.attrs["best_test_loss"] = base[best_trial]["test_loss"][:].min()

        return base.attrs["best_trial"], base.attrs["best_train_loss"], base.attrs["best_test_loss"]


if __name__ == "__main__":
    ############################################################################
    # simulate data as needed
    ############################################################################
    simulate_TCLab_Data("data/train/", M_TRAINING_EXAMPLES)
    simulate_TCLab_Data("data/test/", M_TESTING_EXAMPLES)
    simulate_TCLab_Data("data/transfer/", M_TRANSFER_EXAMPLES, k1=0.2, k2=0.5, tau12=60)
    simulate_TCLab_Data("data/transfer_test/", M_TRANSFER_TEST_EXAMPLES, k1=0.2, k2=0.5, tau12=60)
    simulate_TCLab_Data("data/transfer-tau12(60)/", M_TRANSFER_EXAMPLES, tau12=60)
    simulate_TCLab_Data("data/transfer-tau12(60)_test/", M_TRANSFER_TEST_EXAMPLES, tau12=60)

    ############################################################################
    # training
    ############################################################################
    train_dataset = TCLabDataset(logger, prefix="data/train/")
    test_dataset = TCLabDataset(logger, prefix="data/test/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
    model_version = "v0"

    if os.path.exists(f"TCLab_NN{model_version}.pt"):
        NN = torch.load(f"TCLab_NN{model_version}.pt", weights_only=False)
    else:
        NN = SimpleNN()
        NN._init_params()

        opt = torch.optim.Adam(NN.parameters(), lr=0.01)
        loss_fn = lambda q1, q2, Q1, Q2: torch.mean((q1 - Q1) ** 2 + (q2 - Q2) ** 2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=20)

        epochs = 1000

        training_losses = []
        test_losses = []
        learning_rates = []

        for epoch in range(epochs):
            train_loss = 0
            for i, (t10, t20, t1sp, t2sp, q1, q2) in enumerate(train_loader):
                opt.zero_grad()
                q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
                loss = loss_fn(q1_p, q2_p, q1, q2)
                loss.backward()
                opt.step()

            with torch.no_grad():
                # train loss
                t10, t20, t1sp, t2sp, q1, q2 = train_dataset.get_all()
                q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
                train_loss = loss_fn(q1_p, q2_p, q1, q2)
                training_losses.append(train_loss.item())
                # Test loss
                t10, t20, t1sp, t2sp, q1, q2 = test_dataset.get_all()
                q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
                test_loss = loss_fn(q1_p, q2_p, q1, q2)
                test_losses.append(test_loss.item())

            lr_scheduler.step(train_loss)

            learning_rates.append(lr_scheduler.get_last_lr()[0])

            logger.log_dict(
                {
                    f"training/{model_version}/loss/train": train_loss,
                    f"training/{model_version}/loss/test": test_loss,
                    f"training/{model_version}/lr": lr_scheduler.get_last_lr()[0],
                }
            )

            print(
                f"Epoch {epoch}/{epochs}, Train loss: {train_loss:.4e}, Test loss: {test_loss:.4e}, learning rate: {lr_scheduler.get_last_lr()[0]:.4e}",
                end="\r",
            )

            torch.save(NN, f"TCLab_NN{model_version}.pt")

    ############################################################################
    # transfer learning
    ############################################################################
    TRANSFER_SCENARIO = "transfer-tau12(60)"
    tl_dataset = TCLabDataset(logger, prefix=f"data/{TRANSFER_SCENARIO}/")
    tl_test_dataset = TCLabDataset(logger, prefix=f"data/{TRANSFER_SCENARIO}_test/")
    t10_, t20_, t1sp_, t2sp_, q1_, q2_ = tl_dataset.get_all()
    t10_t, t20_t, t1sp_t, t2sp_t, q1_t, q2_t = tl_test_dataset.get_all()
    loss_fn = lambda q1, q2, Q1, Q2: torch.mean((q1 - Q1) ** 2 + (q2 - Q2) ** 2)

    # NN predictions on transferred dataset
    if not logger.check_key(f"{TRANSFER_SCENARIO}/No Maintenance/q1_p"):
        with torch.no_grad():
            q1_p_, q2_p_ = NN(t10_, t20_, t1sp_, t2sp_)
            loss = (q1_p_ - q1_) ** 2 + (q2_p_ - q2_) ** 2
            logger.log_dict(
                {
                    f"{TRANSFER_SCENARIO}/No Maintenance/train/q1_p": q1_p_.detach().numpy(),
                    f"{TRANSFER_SCENARIO}/No Maintenance/train/q2_p": q2_p_.detach().numpy(),
                    f"{TRANSFER_SCENARIO}/No Maintenance/train/loss": loss.detach().numpy(),
                }
            )
            q1_p_, q2_p_ = NN(t10_t, t20_t, t1sp_t, t2sp_t)
            loss = (q1_p_ - q1_t) ** 2 + (q2_p_ - q2_t) ** 2
            logger.log_dict(
                {
                    f"{TRANSFER_SCENARIO}/No Maintenance/test/q1_p": q1_p_.detach().numpy(),
                    f"{TRANSFER_SCENARIO}/No Maintenance/test/q2_p": q2_p_.detach().numpy(),
                    f"{TRANSFER_SCENARIO}/No Maintenance/test/loss": loss.detach().numpy(),
                }
            )

    # online Adam - vanilla  (did not do well. LR may need to be changed, or just not good)
    # for t10, t20, t1sp, t2sp, q1, q2 in tl_dataloader:
    #     q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
    #     loss = loss_fn(q1_p, q2_p, q1, q2)
    #     loss.backward()
    #     opt.step()

    #     with torch.no_grad():
    #         q1_p_, q2_p_ = NN(t10_, t20_, t1sp_, t2sp_)
    #         total_loss = loss_fn(q1_p_, q2_p_, q1_, q2_)

    #     logger.log_dict({
    #         "transfer/Online Adam/Vanilla/q1_p": q1_p.detach().numpy(),
    #         "transfer/Online Adam/Vanilla/q2_p": q2_p.detach().numpy(),
    #         "transfer/Online Adam/Vanilla/loss": np.array([loss.item()]),
    #         "transfer/Online Adam/Vanilla/global_loss": np.array([total_loss.item()]),
    #     })

    # TODO: Rename trial mechanics so that if it exists, it will move on (keep info in name)

    # Mini-Batch Adam
    # for LR in [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
    #     for BATCH_SIZE in [5, 25, 50, 100, 500, 1000]:
    #         # LR = 0.01
    #         EPOCHS = 1000
    #         NN = torch.load(f"TCLab_NN{model_version}.pt", weights_only=False)
    #         dataloader = torch.utils.data.DataLoader(tl_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #         opt = torch.optim.Adam(NN.parameters(), lr=LR)
    #         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=20)
    #         trail_key = logger.get_unique_key(f"{TRANSFER_SCENARIO}/Batched Adam/trial/")
    #         logger.log_attribute(trail_key + "batch size", BATCH_SIZE)
    #         logger.log_attribute(trail_key + "lr", LR)
    #         logger.log_attribute(trail_key + "epochs", EPOCHS)
    #         for epoch in range(EPOCHS):
    #             for t10, t20, t1sp, t2sp, q1, q2 in dataloader:
    #                 q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
    #                 loss = loss_fn(q1_p, q2_p, q1, q2)
    #                 loss.backward()
    #                 opt.step()

    #             with torch.no_grad():
    #                 q1_p_, q2_p_ = NN(t10_, t20_, t1sp_, t2sp_)
    #                 train_loss = loss_fn(q1_p_, q2_p_, q1_, q2_)
    #                 q1_p_t, q2_p_t = NN(t10_t, t20_t, t1sp_t, t2sp_t)
    #                 test_loss = loss_fn(q1_p_t, q2_p_t, q1_t, q2_t)

    #             lr_scheduler.step(train_loss)

    #             logger.log_dict(
    #                 {
    #                     trail_key + "train_loss": train_loss.item(),
    #                     trail_key + "test_loss": test_loss.item(),
    #                     # trail_key + "lr": lr_scheduler.get_last_lr()[0],
    #                 }
    #             )

    # # compare with traning a new model
    # i = 0
    # for LR in [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
    #     for BATCH_SIZE in [5, 25, 50, 100, 500, 1000]:
    #         i += 1
    #         if i < 37:
    #             continue
    #         # LR = 0.01
    #         EPOCHS = 2000
    #         NN = torch.load(f"TCLab_NN{model_version}.pt", weights_only=False)
    #         NN._init_params()
    #         dataloader = torch.utils.data.DataLoader(tl_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #         opt = torch.optim.Adam(NN.parameters(), lr=LR)
    #         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=20)
    #         trail_key = logger.get_unique_key(f"{TRANSFER_SCENARIO}/New NN - Batched Adam/trial/")
    #         logger.log_attribute(trail_key + "batch size", BATCH_SIZE)
    #         logger.log_attribute(trail_key + "lr", LR)
    #         logger.log_attribute(trail_key + "epochs", EPOCHS)
    #         for epoch in range(EPOCHS):
    #             for t10, t20, t1sp, t2sp, q1, q2 in dataloader:
    #                 q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
    #                 loss = loss_fn(q1_p, q2_p, q1, q2)
    #                 loss.backward()
    #                 opt.step()

    #             with torch.no_grad():
    #                 q1_p_, q2_p_ = NN(t10_, t20_, t1sp_, t2sp_)
    #                 train_loss = loss_fn(q1_p_, q2_p_, q1_, q2_)
    #                 q1_p_t, q2_p_t = NN(t10_t, t20_t, t1sp_t, t2sp_t)
    #                 test_loss = loss_fn(q1_p_t, q2_p_t, q1_t, q2_t)

    #             lr_scheduler.step(train_loss)

    #             logger.log_dict(
    #                 {
    #                     trail_key + "train_loss": train_loss.item(),
    #                     trail_key + "test_loss": test_loss.item(),
    #                     # trail_key + "lr": lr_scheduler.get_last_lr()[0],
    #                 }
    #             )

    # compare with TL + masked Adam
    i = 0
    # for LR in [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
    for LR in [3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]:
        for BATCH_SIZE in [5, 25, 50, 100, 500, 1000]:
            for MASK_RATIO in [0.5, 0.8, 0.9, 0.95, 0.99]:
                i += 1
                if i < 160:
                    continue
                # LR = 0.01
                EPOCHS = 2000
                NN = torch.load(f"TCLab_NN{model_version}.pt", weights_only=False)
                # NN._init_params()
                dataloader = torch.utils.data.DataLoader(tl_dataset, batch_size=BATCH_SIZE, shuffle=True)
                opt = maskedAdam(NN.parameters(), lr=LR)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=20)
                trail_key = logger.get_unique_key(f"{TRANSFER_SCENARIO}/Masked Adam/trialv2/")
                logger.log_attribute(trail_key + "batch size", BATCH_SIZE)
                logger.log_attribute(trail_key + "lr", LR)
                logger.log_attribute(trail_key + "epochs", EPOCHS)
                logger.log_attribute(trail_key + "mask ratio", MASK_RATIO)
                for epoch in range(EPOCHS):
                    for t10, t20, t1sp, t2sp, q1, q2 in dataloader:
                        q1_p, q2_p = NN(t10, t20, t1sp, t2sp)
                        loss = loss_fn(q1_p, q2_p, q1, q2)
                        loss.backward()
                        grads = opt._get_flat_grads()
                        mask = mask_fn(grads, quantile_thresh=MASK_RATIO)
                        opt.masked_step(mask=mask)

                    with torch.no_grad():
                        q1_p_, q2_p_ = NN(t10_, t20_, t1sp_, t2sp_)
                        train_loss = loss_fn(q1_p_, q2_p_, q1_, q2_)
                        q1_p_t, q2_p_t = NN(t10_t, t20_t, t1sp_t, t2sp_t)
                        test_loss = loss_fn(q1_p_t, q2_p_t, q1_t, q2_t)

                    lr_scheduler.step(train_loss)

                    logger.log_dict(
                        {
                            trail_key + "train_loss": train_loss.item(),
                            trail_key + "test_loss": test_loss.item(),
                            # trail_key + "lr": lr_scheduler.get_last_lr()[0],
                        }
                    )
