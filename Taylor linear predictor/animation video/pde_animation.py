from __future__ import annotations

import numpy as np

from manim import *


config.background_color = BLACK
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 60
config.frame_width = 16
config.frame_height = 9


GRID_COLOR = ManimColor("#FF6B6B")
FD_NODE_COLOR = ManimColor("#FFD700")
RL_HOT = ManimColor("#FF4500")
RL_COOL = ManimColor("#ADFF2F")
AXIS_COLOR = WHITE
ANNOTATION_COLOR = ManimColor("#00BFFF")
GOLD = ManimColor("#FFD700")


def latex_label(symbol: str, scale: float = 0.65) -> Mobject:
    """Use MathTex for axis labels, with a text fallback if LaTeX is absent."""
    try:
        label = MathTex(symbol, color=AXIS_COLOR)
        label.scale(scale)
        return label
    except Exception:
        return Text(symbol, color=AXIS_COLOR, font_size=int(36 * scale))


class DomainPanel:
    def __init__(
        self,
        center: np.ndarray = ORIGIN,
        width: float = 6.0,
        height: float = 6.4,
        stroke_width: float = 2.0,
        label_scale: float = 0.65,
    ):
        self.center = np.array(center)
        self.width = width
        self.height = height
        self.left = self.center[0] - width / 2
        self.right = self.center[0] + width / 2
        self.bottom = self.center[1] - height / 2
        self.top = self.center[1] + height / 2

        self.box = Rectangle(
            width=width,
            height=height,
            stroke_color=AXIS_COLOR,
            stroke_width=stroke_width,
            fill_opacity=0,
        ).move_to(center)

        self.x_label = latex_label("x", scale=label_scale).next_to(self.box, DOWN, buff=0.18)
        self.t_label = latex_label("t", scale=label_scale).next_to(self.box, LEFT, buff=0.18)

        x_ticks = []
        for i in range(9):
            p = self.point(i / 8, 0)
            x_ticks.append(Line(p + DOWN * 0.07, p + UP * 0.07, color=AXIS_COLOR, stroke_width=1.4))

        t_ticks = []
        for i in range(11):
            p = self.point(0, i / 10)
            t_ticks.append(Line(p + LEFT * 0.07, p + RIGHT * 0.07, color=AXIS_COLOR, stroke_width=1.4))

        self.t_zero = Text("0", color=AXIS_COLOR, font_size=18).next_to(
            self.point(0, 0), LEFT + DOWN, buff=0.08
        )
        self.t_final = Text("T", color=AXIS_COLOR, font_size=18).next_to(
            self.point(0, 1), LEFT + UP, buff=0.08
        )

        self.axes = VGroup(
            self.box,
            *x_ticks,
            *t_ticks,
            self.x_label,
            self.t_label,
            self.t_zero,
            self.t_final,
        )

    def point(self, x_frac: float, t_frac: float) -> np.ndarray:
        return np.array(
            [
                self.left + x_frac * self.width,
                self.bottom + t_frac * self.height,
                0,
            ]
        )


class PDESolvingComparison(Scene):
    def construct(self) -> None:
        self.camera.background_color = BLACK

        self.title_card()
        domain, grid_lines, grid_nodes = self.fixed_grid_scene()
        self.amr_scene(domain, grid_lines, grid_nodes)
        self.rl_sampling_scene()
        self.summary_scene()

    def fade_all(self, run_time: float = 0.8) -> None:
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[FadeOut(mob) for mob in mobs], run_time=run_time)

    def title_card(self) -> None:
        title = Text(
            "How Should a PDE Solver Decide Where to Look?",
            color=WHITE,
            font="Times New Roman",
            font_size=36,
        )
        subtitle = Text("Three Approaches", color=WHITE, font_size=28).next_to(title, DOWN, buff=0.35)
        card = VGroup(title, subtitle).move_to(ORIGIN)

        self.play(FadeIn(title, shift=UP * 0.15), run_time=0.8)
        self.play(FadeIn(subtitle, shift=UP * 0.12), run_time=0.5)
        self.wait(2.1)
        self.play(FadeOut(card), run_time=0.6)

    def make_uniform_grid(
        self,
        domain: DomainPanel,
        x_nodes: int = 8,
        t_nodes: int = 10,
        line_opacity: float = 1.0,
        node_opacity: float = 1.0,
        node_radius: float = 0.035,
    ) -> tuple[VGroup, VGroup, VGroup, VGroup]:
        vertical = VGroup()
        for i in range(x_nodes):
            alpha = i / (x_nodes - 1)
            line = Line(
                domain.point(alpha, 0),
                domain.point(alpha, 1),
                color=GRID_COLOR,
                stroke_width=1.45,
                stroke_opacity=line_opacity,
            )
            vertical.add(line)

        horizontal = VGroup()
        for j in range(t_nodes):
            beta = j / (t_nodes - 1)
            line = Line(
                domain.point(0, beta),
                domain.point(1, beta),
                color=GRID_COLOR,
                stroke_width=1.45,
                stroke_opacity=line_opacity,
            )
            horizontal.add(line)

        dots = VGroup()
        for j in range(t_nodes):
            beta = j / (t_nodes - 1)
            for i in range(x_nodes):
                alpha = i / (x_nodes - 1)
                dots.add(
                    Dot(
                        domain.point(alpha, beta),
                        radius=node_radius,
                        color=FD_NODE_COLOR,
                        fill_opacity=node_opacity,
                    )
                )

        return vertical, horizontal, VGroup(vertical, horizontal), dots

    def fixed_grid_scene(self) -> tuple[DomainPanel, VGroup, VGroup]:
        domain = DomainPanel(center=DOWN * 0.12, width=6.0, height=7.05, label_scale=0.62)
        title = Text("Traditional Fixed Grid", color=WHITE, font_size=32).to_corner(UL, buff=0.45)
        vertical, horizontal, grid_lines, grid_nodes = self.make_uniform_grid(domain)

        self.play(FadeIn(title), Create(domain.axes), run_time=0.9)
        self.play(
            LaggedStart(*[Create(line) for line in vertical], lag_ratio=0.13),
            run_time=4.0,
        )
        row_animations = []
        x_nodes = len(vertical)
        for row_index, line in enumerate(horizontal):
            row_dots = [
                grid_nodes[row_index * x_nodes + col_index]
                for col_index in range(x_nodes)
            ]
            row_animations.append(
                AnimationGroup(
                    Create(line),
                    LaggedStart(
                        *[GrowFromCenter(dot) for dot in row_dots],
                        lag_ratio=0.045,
                    ),
                    lag_ratio=0.0,
                )
            )
        self.play(
            LaggedStart(*row_animations, lag_ratio=0.16),
            run_time=4.4,
        )

        annotation = Text(
            "Every point evaluated —\neven where nothing interesting happens",
            color=WHITE,
            font_size=23,
            line_spacing=0.85,
        )
        annotation.next_to(domain.box, RIGHT, buff=0.35).shift(UP * 1.2)
        arrow = Arrow(
            annotation.get_left() + LEFT * 0.05,
            domain.point(0.74, 0.57),
            color=ANNOTATION_COLOR,
            buff=0.08,
            stroke_width=5,
        )

        self.play(GrowArrow(arrow), FadeIn(annotation), run_time=1.2)
        self.wait(3.0)
        self.play(FadeOut(annotation), FadeOut(arrow), FadeOut(title), run_time=0.7)
        return domain, grid_lines, grid_nodes

    def amr_scene(self, domain: DomainPanel, grid_lines: VGroup, grid_nodes: VGroup) -> None:
        title = Text("Adaptive Mesh Refinement (AMR)", color=WHITE, font_size=32).to_corner(UL, buff=0.45)
        self.play(FadeIn(title), grid_nodes.animate.set_opacity(0.35), run_time=0.8)

        x0, x1 = 3 / 7, 5 / 7
        t0, t1 = 4 / 9, 6 / 9
        fine_lines = VGroup()
        for i in range(5):
            alpha = interpolate(x0, x1, i / 4)
            fine_lines.add(
                Line(
                    domain.point(alpha, t0),
                    domain.point(alpha, t1),
                    color=FD_NODE_COLOR,
                    stroke_width=3.0,
                )
            )
        for j in range(5):
            beta = interpolate(t0, t1, j / 4)
            fine_lines.add(
                Line(
                    domain.point(x0, beta),
                    domain.point(x1, beta),
                    color=FD_NODE_COLOR,
                    stroke_width=3.0,
                )
            )

        fine_nodes = VGroup()
        for j in range(5):
            beta = interpolate(t0, t1, j / 4)
            for i in range(5):
                alpha = interpolate(x0, x1, i / 4)
                fine_nodes.add(Dot(domain.point(alpha, beta), radius=0.04, color=FD_NODE_COLOR))

        refined_region = SurroundingRectangle(
            fine_lines,
            color=FD_NODE_COLOR,
            buff=0.03,
            stroke_width=2,
        )

        self.play(Create(refined_region), run_time=0.8)
        self.play(
            LaggedStart(*[Create(line) for line in fine_lines], lag_ratio=0.08),
            run_time=4.0,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in fine_nodes], lag_ratio=0.02),
            run_time=1.2,
        )

        note_1 = Text("Refinement triggered\nby error estimate", color=WHITE, font_size=22, line_spacing=0.85)
        note_1.next_to(domain.box, RIGHT, buff=0.35).shift(UP * 0.9)
        arrow = Arrow(
            note_1.get_left() + LEFT * 0.05,
            domain.point(0.56, 0.56),
            color=ANNOTATION_COLOR,
            buff=0.08,
            stroke_width=5,
        )
        note_2 = Text("Still requires\na mesh structure", color=WHITE, font_size=22, line_spacing=0.85).next_to(
            note_1, DOWN, aligned_edge=LEFT, buff=0.45
        )

        self.play(GrowArrow(arrow), FadeIn(note_1), run_time=1.1)
        self.play(FadeIn(note_2, shift=UP * 0.1), run_time=1.0)
        self.wait(3.0)
        self.fade_all(run_time=0.9)

    def rl_sampling_scene(self) -> None:
        left_domain = DomainPanel(center=LEFT * 3.75 + UP * 0.05, width=5.0, height=5.8, label_scale=0.5)
        right_domain = DomainPanel(center=RIGHT * 3.75 + UP * 0.05, width=5.0, height=5.8, label_scale=0.5)

        _, _, left_grid_lines, left_grid_nodes = self.make_uniform_grid(
            left_domain,
            line_opacity=0.4,
            node_opacity=0.4,
            node_radius=0.025,
        )
        left_label = Text("FD Reference", color=WHITE, font_size=23).next_to(left_domain.box, UP, buff=0.2)
        right_label = Text("RL-based Adaptive Sampling", color=WHITE, font_size=32).next_to(
            right_domain.box, UP, buff=0.2
        )

        ic_points = [
            (0.05, 0.0),
            (0.15, 0.0),
            (0.25, 0.0),
            (0.35, 0.0),
            (0.45, 0.0),
            (0.55, 0.0),
            (0.65, 0.0),
            (0.75, 0.0),
            (0.85, 0.0),
            (0.95, 0.0),
        ]
        ic_dots = VGroup(
            *[
                Dot(right_domain.point(x, t), radius=0.035, color=FD_NODE_COLOR)
                for x, t in ic_points
            ]
        )

        self.play(
            FadeIn(left_label),
            FadeIn(right_label),
            Create(left_domain.axes),
            Create(right_domain.axes),
            Create(left_grid_lines),
            FadeIn(left_grid_nodes),
            LaggedStart(*[GrowFromCenter(dot) for dot in ic_dots], lag_ratio=0.05),
            run_time=1.8,
        )

        rl_points = [
            (0.10, 0.15),
            (0.25, 0.15),
            (0.45, 0.15),
            (0.65, 0.15),
            (0.80, 0.15),
            (0.95, 0.15),
            (0.15, 0.30),
            (0.28, 0.32),
            (0.38, 0.33),
            (0.48, 0.32),
            (0.58, 0.30),
            (0.80, 0.30),
            (0.10, 0.50),
            (0.20, 0.49),
            (0.30, 0.51),
            (0.42, 0.50),
            (0.60, 0.48),
            (0.85, 0.48),
            (0.12, 0.67),
            (0.22, 0.65),
            (0.35, 0.68),
            (0.50, 0.66),
            (0.15, 0.82),
            (0.28, 0.80),
            (0.42, 0.83),
        ]

        known_points = list(ic_points)
        rl_dots = VGroup()

        for index, target in enumerate(rl_points, start=1):
            neighbours = self.nearest_known_points(known_points, target, count=5)
            arrows = VGroup()
            for source in neighbours:
                arrows.add(
                    Arrow(
                        right_domain.point(*source),
                        right_domain.point(*target),
                        buff=0.07,
                        color=ANNOTATION_COLOR,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.20,
                    ).set_z_index(3)
                )
            dot = Dot(
                right_domain.point(*target),
                radius=0.045,
                color=self.rl_point_color(index, *target),
            ).set_z_index(4)

            self.play(GrowFromCenter(dot), run_time=0.12)
            self.play(dot.animate.scale(1.35), rate_func=there_and_back, run_time=0.16)
            self.play(Create(arrows), run_time=0.30)
            self.wait(0.45)
            self.play(FadeOut(arrows), run_time=0.20)

            rl_dots.add(dot)
            known_points.append(target)


        self.wait(1.0)
        self.play(FadeIn(final_note), run_time=0.7)
        self.wait(3.0)
        self.fade_all(run_time=0.9)

    def nearest_known_points(
        self,
        known_points: list[tuple[float, float]],
        target: tuple[float, float],
        count: int = 5,
    ) -> list[tuple[float, float]]:
        tx, tt = target
        return sorted(
            known_points,
            key=lambda point: (point[0] - tx) ** 2 + (point[1] - tt) ** 2,
        )[:count]

    def rl_point_color(self, index: int, x_frac: float, t_frac: float) -> ManimColor:
        if index <= 6:
            heat = 0.08
        elif index <= 12:
            heat = 1.0 - abs(x_frac - 0.38) / 0.34
        elif index <= 18:
            heat = 1.0 - abs(x_frac - 0.28) / 0.34
        elif index <= 22:
            heat = 1.0 - abs(x_frac - 0.25) / 0.30
        else:
            heat = 1.0 - abs(x_frac - 0.28) / 0.28
        return interpolate_color(RL_COOL, RL_HOT, float(np.clip(heat, 0.0, 1.0)))

    def summary_scene(self) -> None:
        header = Text(
            "Method              | Mesh-free? | Adaptive?",
            color=WHITE,
            font="Consolas",
            font_size=28,
        )
        row_1 = Text(
            "Fixed Grid (FD)     |    No      |    No",
            color=WHITE,
            font="Consolas",
            font_size=28,
        )
        row_2 = Text(
            "AMR                 |    No      |    Yes",
            color=WHITE,
            font="Consolas",
            font_size=28,
        )
        row_3 = Text(
            "RL Sampling (ours)  |    Yes     |    Yes",
            color=GOLD,
            font="Consolas",
            font_size=28,
        )

        table = VGroup(header, row_1, row_2, row_3).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        table.move_to(ORIGIN + DOWN * 0.1)
        highlight = SurroundingRectangle(row_3, color=GOLD, buff=0.14, stroke_width=2)

        self.play(FadeIn(header), run_time=0.8)
        self.play(FadeIn(row_1), FadeIn(row_2), FadeIn(row_3), Create(highlight), run_time=1.0)
        self.wait(4.0)
        self.fade_all(run_time=0.8)
