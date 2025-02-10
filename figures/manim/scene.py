from manim import *
import random

# config
config["background_color"] = WHITE

class GenScene(Scene):
    def construct(self):
        # Prompt: 
        
        # Set background color to white
        self.camera.background_color = WHITE

        # Create a 20x20 matrix of squares
        matrix_size = 15
        square_size = 0.25
        selected_elements = 6
        selected_indices = [2, 5, 6, 7, 8, 13]
        squares = VGroup(*[
            VGroup(*[
                Square(side_length=square_size, color=BLACK, fill_color=WHITE, fill_opacity=0.9)
                for _ in range(matrix_size)
            ]).arrange(RIGHT, buff=0)
            for _ in range(matrix_size)
        ]).arrange(DOWN, buff=0)

        # Center the matrix on the left half of the screen
        squares.to_edge(LEFT, buff=1)
        
        # Label the matrix "P"
        label_p = Tex("P", color=BLACK).next_to(squares, DOWN)
        self.play(Create(squares), FadeIn(label_p))
        
        # About a square away, show the vector H which is a 15x1 matrix
        H_group = VGroup(*[
            Square(side_length=square_size, color=BLACK, fill_color=WHITE, fill_opacity=0.9)
            for _ in range(matrix_size)
        ]).arrange(DOWN, buff=0).next_to(squares, LEFT)
        label_H = Tex(r"$\nabla_\pi\mathcal{L}$", color=BLACK).next_to(H_group, DOWN)
        self.play(Create(H_group), FadeIn(label_H))
        
        # create HT by creating a copy of H and rotating it to the top of P
        HT_group = H_group.copy()
        HT_group.generate_target()
        HT_group.target.rotate(PI/2).next_to(squares, UP)
        label_HT = Tex(r"$\nabla_\pi\mathcal{L}^T$", color=BLACK).next_to(HT_group.target, RIGHT)
        self.play(MoveToTarget(HT_group), FadeIn(label_HT))
        
        # Highlight H and HT elements
        H_elements_group = VGroup(*[H_group[row] for row in selected_indices])
        HT_elements_group = VGroup(*[HT_group[col] for col in selected_indices])
        self.play(H_elements_group.animate.set_fill(GREY), HT_elements_group.animate.set_fill(GREY))

        grey_elements = []
        green_elements = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i in selected_indices and j in selected_indices:
                    green_elements.append(squares[i][j])
                elif i in selected_indices or j in selected_indices:
                    grey_elements.append(squares[i][j])
        grey_elements = VGroup(*grey_elements)
        green_elements = VGroup(*green_elements)
        self.play(grey_elements.animate.set_fill(GREY), green_elements.animate.set_fill(GREEN))
        new_green_elements = VGroup(*[squares[row][col].copy() for col in selected_indices for row in selected_indices])

        # Move green elements to a new matrix on the right half of the screen
        new_green_elements.generate_target()
        new_green_elements.target.arrange_in_grid(rows=selected_elements, cols=selected_elements, buff=0).to_edge(RIGHT, buff=1)

        # Label the new matrix "P'"
        label_p_prime = Tex("P$^\prime$", color=BLACK).next_to(new_green_elements.target, DOWN)

        # Animate the movement of green elements and add the new label
        self.play(MoveToTarget(new_green_elements), FadeIn(label_p_prime))

        # Add the original label
        self.add(label_p)
        
        self.wait(2)

# class GenScene2(Scene):
#     def construct(self):
#         # Create neural network nodes
#         input_nodes = [Circle(radius=0.2, color=WHITE).shift(LEFT*3 + UP*i) for i in range(3)]
#         hidden_nodes = [Circle(radius=0.2, color=WHITE).shift(UP*i) for i in range(3)]
#         output_node = Circle(radius=0.2, color=WHITE).shift(RIGHT*3)

#         # Fill nodes with random share of red
#         for node in input_nodes + hidden_nodes + [output_node]:
#             node.set_fill(RED, opacity=random.uniform(0.1, 1.0))

#         # Create edges
#         edges = [
#             Line(start=input_node.get_center(), end=hidden_node.get_center(), color=WHITE)
#             for input_node in input_nodes for hidden_node in hidden_nodes
#         ] + [
#             Line(start=hidden_node.get_center(), end=output_node.get_center(), color=WHITE)
#             for hidden_node in hidden_nodes
#         ]

#         # Create true value and loss
#         true_value = Text("True Value", font_size=24).next_to(output_node, UP)
#         loss_arrow = Arrow(start=true_value.get_bottom(), end=output_node.get_top(), buff=0.1, color=WHITE)
#         loss_text = Text("Loss", font_size=24, color=RED).next_to(loss_arrow, RIGHT)

#         # Add all elements to the scene
#         self.play(*[Create(node) for node in input_nodes + hidden_nodes + [output_node]])
#         self.play(*[Create(edge) for edge in edges])
#         self.play(Write(true_value), Create(loss_arrow), Write(loss_text))