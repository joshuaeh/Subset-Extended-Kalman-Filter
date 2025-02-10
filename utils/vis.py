""" """

import imageio

from .__init__ import *


def x_ticks_pi_units(ax):
    ax.set_xticks(
        np.linspace(0, np.pi, 5, endpoint=True),
        labels=[
            "0",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
        ],
    )
    return


def giffer(image_paths, gif_path, duration=0.15, n_loops=3):
    # with imageio.get_writer(gif_path, mode="I", duration=duration) as writer:
    #     for image_path in image_paths:
    #         image = imageio.imread(image_path)
    #         writer.append_data(image)
    images = []
    for image_path in image_paths:
        images.append(imageio.imread(image_path))

    # add more frames to the end so there is a pause
    imageio.mimsave(gif_path, images, duration=duration, loop=n_loops)
    return


def graph_predictions(
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    xscaler,
    yscaler,
    epoch=None,
    save_prefix=None,
    writer=None,
    close_fig=True,
):
    x_train_plot = xscaler.inverse_transform(x_train.detach().numpy())
    x_test_plot = xscaler.inverse_transform(x_test.detach().numpy())
    y_pred_plot = yscaler.inverse_transform(model(x_train).detach().numpy())
    y_pred_test_plot = yscaler.inverse_transform(model(x_test).detach().numpy())
    y_train_plot = yscaler.inverse_transform(y_train.detach().numpy())
    y_test_plot = yscaler.inverse_transform(y_test.detach().numpy())

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=1_000)
    ax.scatter(
        x_train_plot,
        y_pred_plot,
        5,
        color="green",
        marker=".",
        label="Train Predictions",
    )
    ax.scatter(
        x_test_plot,
        y_pred_test_plot,
        5,
        color="blue",
        marker=".",
        label="Test Predictions",
    )
    ax.scatter(
        x_train_plot,
        y_train_plot,
        2,
        color="orange",
        marker=".",
        label="Train True",
        alpha=0.7,
    )
    ax.scatter(
        x_test_plot,
        y_test_plot,
        2,
        color="red",
        marker=".",
        label="Test True",
        alpha=0.7,
    )
    if epoch is not None:
        ax.text(0.1, -0.9, f"Epoch: {epoch:,}")

    ax.legend(loc="upper right")
    ax.set_ylabel("Output")
    ax.set_xlabel("Input")

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_{epoch}.png")

    if writer is not None:
        writer.add_figure(f"Test Predictions", fig, epoch)

    if close_fig:
        plt.close()
    return fig, ax


def plot_loss_l2_gradients(drift_parameters, drifted_losses, parameter_grads, retrain):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=1_000)
    ax.plot(drift_parameters, drifted_losses, "r-", linewidth=0.5, label="Drifted Loss")
    ax.hlines(
        LOSS_THRESHOLD,
        DRIFT_MIN,
        DRIFT_MAX,
        color="black",
        linewidth=0.5,
        label="Loss Threshold",
    )
    ax.scatter(
        drift_parameters[retrain],
        drifted_losses[retrain] + 0.01,
        color="black",
        s=5,
        marker="*",
        label="Retrain",
    )
    ax2 = ax.twinx()
    ax2.plot(
        drift_parameters,
        parameter_grads.mean(axis=1),
        linestyle="-",
        color="gray",
        linewidth=0.5,
        label="Mean",
    )
    ax2.plot(
        drift_parameters,
        parameter_grads.max(axis=1),
        linestyle="--",
        color="gray",
        linewidth=0.5,
        label="Max/Min",
    )
    ax2.plot(
        drift_parameters,
        parameter_grads.min(axis=1),
        linestyle="--",
        color="gray",
        linewidth=0.5,
    )
    ax2.plot(
        drift_parameters,
        np.linalg.norm(parameter_grads, axis=1),
        linestyle="-",
        color="orange",
        linewidth=0.5,
        label="Gradient L2 Norm",
    )
    ax.legend(bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax2.legend(bbox_to_anchor=(1.0, -0.15), ncol=2)
    ax.set_xticks(
        np.linspace(0, np.pi, 5, endpoint=True),
        labels=[
            "0",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
        ],
    )
    ax.set_xlabel("Parameter drift")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Parameter Gradient")

    return fig, ax


def plot_gradient_heatmap(parameter_grads, param_info):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=1_000)
    sns.heatmap(
        parameter_grads.T,
        cmap="bwr",
        center=0.00,
        # norm=Norm(vmin=parameter_grads.min(), vmax=parameter_grads.max()),
        # norm=SymLogNorm(linthresh=0.001),
        cbar_kws={"label": "dLoss/dParam"},
        ax=ax,
    )
    n, info = get_parameter_info(model)

    major_labels = []
    major_indices = []
    i = 0
    for label, n_p in param_info.items():
        major_labels.append(label + "(1)")
        major_indices.append(i + 0.5)
        i += n_p
    minor_labels = []
    minor_indices = []
    j = 2
    for i in range(n):
        if i + 0.5 not in major_indices:
            minor_labels.append(f"({j})")
            minor_indices.append(i + 0.5)
            j += 1
        else:
            j = 2
    ax.set_yticks(major_indices, labels=major_labels, rotation=0)
    ax.set_yticks(minor_indices, labels=minor_labels, minor=True)
    ax.set_xticks(
        [0, 250, 500, 750, 1000],
        labels=["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
        rotation=0,
    )
    ax.set_title("Parameter Gradient as drift (c) increases for Sin(x+c)")
    ax.set_ylabel("Parameters")
    ax.set_xlabel("Drift (c) value")
    plt.show()
    return fig, ax


def plot_gradients_layerwise(parameter_grads):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), dpi=1_000, sharex=True)
    axs[0].hlines(0.0, -1, 1001, color="black", linewidth=0.5, linestyle="-")
    axs[1].hlines(0.0, -1, 1001, color="black", linewidth=0.5, linestyle="-")
    for i in range(10):  # linear1.weights
        axs[0].plot(
            abs(parameter_grads[:, i]),
            linewidth=0.5,
            color="black",
            linestyle="-",
            alpha=0.8,
        )
    for i in range(10, 20):  # linear1.weights
        axs[0].plot(
            abs(parameter_grads[:, i]),
            linewidth=0.5,
            color="red",
            linestyle="-",
            alpha=0.8,
        )
    for i in range(20, 30):  # linear1.bias
        axs[1].plot(
            abs(parameter_grads[:, i]),
            linewidth=0.5,
            color="blue",
            linestyle="-",
            alpha=0.8,
        )
    i = 30  # linear2.weights
    axs[1].plot(
        abs(parameter_grads[:, i]),
        linewidth=0.5,
        color="orange",
        linestyle="-",
        alpha=0.8,
    )

    axs[0].hlines(0.0, -1, 1001, color="black", linewidth=0.5, linestyle="-")
    axs[1].hlines(0.0, -1, 1001, color="black", linewidth=0.5, linestyle="-")

    axs[0].set_ylim([-0.05, 0.25])
    axs[1].set_ylim([-0.05, 0.25])
    axs[0].set_xlim([-1, 1001])
    axs[1].set_xlim([-1, 1001])

    axs[1].legend(
        [
            Line2D([0], [0], color="black", linewidth=1, label="Linear1 Weights"),
            Line2D([0], [0], color="red", linewidth=1, label="Linear1 Biases"),
            Line2D([0], [0], color="blue", linewidth=1, label="Linear2 Weights"),
            Line2D([0], [0], color="orange", linewidth=1, label="Linear2 Biases"),
        ],
        ["Linear1 Weights", "Linear1 Biases", "Linear2 Weights", "Linear2 Biases"],
        bbox_to_anchor=(1, -0.13),
        ncol=4,
        loc="upper right",
    )
    axs[0].set_ylabel("Layer 1 Parameter Gradient Magnitudes")
    axs[1].set_ylabel("Layer 2 Parameter Gradient Magnitudes")
    axs[1].set_xlabel("Drift (c) value")
    axs[1].set_xticks(
        [0, 250, 500, 750, 1000],
        labels=["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    fig.tight_layout()
    return fig, ax


def plot_gradient_figs(drift_parameters, parameter_grads, drifted_losses, retrain, param_info):
    fig, ax = plot_loss_l2_gradients(drift_parameters, drifted_losses, parameter_grads, retrain)
    plt.show()
    fig, ax = plot_loss_l2_gradients(drift_parameters, drifted_losses, abs(parameter_grads), retrain)
    plt.show()
    fig, ax = plot_gradient_heatmap(parameter_grads, param_info)
    plt.show()
    fig, ax = plot_gradients_layerwise(parameter_grads)


def plot_IQR(drifted_parameter_grads, median_grad, std_dev_grad, n_params=31):
    parameter_idx = np.arange(n_params)
    outside_IQR = abs(drifted_parameter_grads) > (median_grad + 1.5 * std_dev_grad)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=1_000)
    ax.plot(
        parameter_idx[~outside_IQR],
        np.abs(drifted_parameter_grads[~outside_IQR]),
        "b.",
        label="In IQR",
    )
    ax.plot(
        parameter_idx[outside_IQR],
        np.abs(drifted_parameter_grads[outside_IQR]),
        "r.",
        label="Out IQR",
    )
    ax.hlines(
        [median_grad + 1.5 * std_dev_grad],
        0,
        n_params,
        color="k",
        linestyle="--",
        label="IQR Bounds",
    )
    ax.hlines(median_grad, 0, n_params, color="k", linestyle="-", label="Median")
    ax.legend()
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Gradient Value")

    return fig, ax
