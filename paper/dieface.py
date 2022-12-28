"""dieface.py - Credits to 3B1B"""

from manim import *


class DieFace(VGroup):
    def __init__(
        self,
        value: int,
        side_length: float = 1.0,
        corner_radius: float = 0.15,
        stroke_color: str = WHITE,
        stroke_width: float = 2.0,
        fill_color: str = GREY_E,
        dot_radius: float = 0.08,
        dot_color: str = BLUE_B,
        dot_coalesce_factor: float = 0.5,
    ):
        dot = Dot(radius=dot_radius, fill_color=dot_color)
        square = Square(
            side_length=side_length,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=1.0,
        )
        square.round_corners(corner_radius)

        if not (1 <= value <= 6):
            raise Exception("DieFace only accepts integer inputs between 1 and 6")

        edge_group = [
            (ORIGIN,),
            (UL, DR),
            (UL, ORIGIN, DR),
            (UL, UR, DL, DR),
            (UL, UR, ORIGIN, DL, DR),
            (UL, UR, LEFT, RIGHT, DL, DR),
        ][value - 1]

        arrangement = VGroup(
            *(
                dot.copy().move_to(square.get_critical_point(vect))
                for vect in edge_group
            )
        )
        arrangement.space_out_submobjects(dot_coalesce_factor)

        super().__init__(square, arrangement)
        self.value = value
        self.index = value
