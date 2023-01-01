from __future__ import annotations

import itertools as it

import networkx as nx
from manim import *

from dieface import DieFace


def get_die_faces(
    values: list[int] = range(1, 7),
    buff: float = LARGE_BUFF,
    side_length: float = 1.0,
    corner_radius: float = 0.15,
    stroke_color: str = WHITE,
    stroke_width: float = 2.0,
    fill_color: str = GREY_E,
    dot_radius: float = 0.08,
    dot_color: str = BLUE_B,
    dot_coalesce_factor: float = 0.5,
):
    """Return a line of six die faces from 1 to 6."""
    die_faces = VGroup(
        *(
            DieFace(
                n,
                side_length=side_length,
                corner_radius=corner_radius,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fill_color=fill_color,
                dot_radius=dot_radius,
                dot_color=dot_color,
                dot_coalesce_factor=dot_coalesce_factor,
            )
            for n in values
        )
    )
    die_faces.arrange(RIGHT, buff=buff)
    return die_faces


class Main(Scene):
    """Robust SPNs animation (Sum-Product Networks).

    1. Title + credits sequence
    2. Joint probability distributions: 6-faces dices
        1. Discrete probability distribution
        2. Adding probabilities: different realisations
        3. Multiplying probabilities: different events (variables)
        4. Why using the joint probability is a bad idea
            1. Space: memory needed 6^N with N dices
            2. Time: inference/marginalization is expensive (exponential)
    3. Sum-product networks
        1. Graphical representation: complete, decomposable, normalised
            1. Completeness: same events for children of a sum node
            2. Decomposition: different events for children of a product node
            3. Normalisation: probabilities for an event sum to 1
        2. More efficient representation
            1. Space: the memory needed is only 6 + 6
            2. Time: inference/marginalization is cheap (linear)
    4. Robust sum-product networks
        1. What if your knowledge is even more uncertain?
            1. If a probability varies, then the SPN might give completely wrong results
            2. How to include imprecision? use imprecise probabilities!
            3. Probabilities belong to a range, and sum nodes are constrained
        2. What do you infer then? Bounds (lower/upper)
    5. Further concepts:
        1. Independence / Conditional independence
        2. SPN have to satisfy certain properties to represent probabilities
        3. Imprecise probabilities search space
    """

    def fade_in_out(self, mobject, delay=1, **kwargs):
        """Fade in, wait for delay, fade out."""
        self.play(FadeIn(mobject), **kwargs)
        self.wait(delay)
        self.play(FadeOut(mobject))

    def construct(self):
        """Call sub-sequences."""
        self.title()
        self.next_section()
        self.dices()
        self.next_section()
        self.sum_product_networks()
        self.next_section()
        self.uncertainty()
        self.next_section()
        self.further()
        self.wait(2)

    def title(self):
        """Title sequence."""
        title = Title(
            r"AOS4 - Paper illustration: ", r"Robustifying sum-product networks"
        )
        credits = Text(
            "D. Deratani Mauá, D. Conaty, F. Gagliardi Cozman, K. Poppenhaeger, C. Polpo de Campos",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        ).next_to(title, DOWN)
        author = Text(
            "Pascal Quach",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        ).shift(ORIGIN)
        self.play(FadeIn(title, credits, author))

        self.wait(1)

        self.play(FadeOut(title, credits, author))

    def dices(self):
        """Joint probability distribution 6-faces dices."""
        # Intro quote
        t0 = Paragraph(
            "If you remember something about probability theory,\n "
            "then these dices might haunt your dreams.",
            font_size=DEFAULT_FONT_SIZE * 0.6,
            alignment="center",
        )

        # Discrete probability distribution
        t1 = Text(
            "Probabilities are introduced using numbers",
            font_size=DEFAULT_FONT_SIZE * 0.75,
        )
        probabilities = VGroup(*[MathTex(r"\frac16") for _ in range(6)])
        blue_die_faces = get_die_faces()
        probabilities_table = self.get_probabilities_table(r"\frac{1}{6}")
        t2 = Tex(r"$D$ is the value obtained by rolling a 6-sided fair die")

        # Adding probabilities
        t3a = VGroup(
            MarkupText(
                f"Adding probabilities is as simple as summing them\n",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
            MarkupText(
                f"<span fgcolor='{BLUE_B}'>At least</span> one event must be true",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
        )
        prob_1_or_2 = VGroup(
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r"\cup"),
            *get_die_faces([2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"+"),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
        )
        frac13 = MathTex(r"\frac13")

        # Multiplying probabilities
        t3b = VGroup(
            MarkupText(
                f"Multiplying probabilities from independent events means\n",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
            MarkupText(
                f"<span fgcolor='{BLUE_B}'>all</span> events must be true",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
        )
        prob_1_and_2 = VGroup(
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r"\cap"),
            *get_die_faces([2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"\times"),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
        )
        frac136 = MathTex(r"\frac1{36}")

        # Joint probability distribution
        t4a = Paragraph(
            "What if you had two independent dices?\n"
            "How do you combine probability distributions?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
            alignment="center",
        )
        t4b = Text(
            "You simply need to multiply every combination of the probabilities",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t4c = Text(
            "If we replace every probability with a variable, we get 36 different values",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        joint_dist_table_16 = self.dice_joint_table(
            [[MathTex(r"\frac16\times\frac16") for _ in range(6)] for _ in range(6)]
        )
        joint_dist_table_136 = self.dice_joint_table(
            [[MathTex(r"\frac{1}{36}") for _ in range(6)] for _ in range(6)]
        )
        joint_dist_table_ab_1 = self.dice_joint_table(
            [
                [
                    MathTex(
                        r"p_" + str(i),
                        r"\times ",
                        r"p'_" + str(j),
                        substrings_to_isolate=["a", "b"],
                    )
                    for j in range(1, 7)
                ]
                for i in range(1, 7)
            ]
        )
        for e in joint_dist_table_ab_1.get_entries_without_labels():
            e.set_color_by_tex("p", BLUE_B)
            e.set_color_by_tex("p'", RED_B)

        # Space complexity
        t5 = MarkupText(
            "How many values do you need to know?", font_size=DEFAULT_FONT_SIZE * 0.5
        )
        t5a = MarkupText(
            f"For a single die, you need <span fgcolor='{BLUE_B}'>6</span> values",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        t5b = MarkupText(
            f"For two dices, you need <span fgcolor='{BLUE_B}'>36</span> values",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        t5c = MarkupText(
            f"The number of values increases "
            f"<span fgcolor='{BLUE_B}'>exponentially</span>",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        space_complexity_equations = [
            MathTex(f"6^{{{i}}}={6 ** i:,}") for i in range(1, 11)
        ]

        # Time complexity
        joint_dist_table_136b = self.dice_joint_table(
            [[MathTex(r"1/36") for _ in range(6)] for _ in range(6)]
        )
        t6a = MarkupText(
            f"It is possible to reverse the combination by "
            f"<span fgcolor='{BLUE_B}'>flattening</span> the table",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        t6b = MarkupText(
            f"This process is called <span fgcolor='{BLUE_B}'>marginalization</span>",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        t6c = MarkupText(
            f"The more dices you have, the longer it "
            f"takes to marginalize a joint distribution.",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        t6d = MarkupText(
            f"But we can do better using "
            f"<span fgcolor='{BLUE_B}'>computational graphs</span>",
            font_size=DEFAULT_FONT_SIZE * 0.45,
        )
        row_marginal_636 = self.get_probabilities_table(
            probability=r"6/36",
            labels=False,
            flip=True,
            dot_color=RED_B,
            side_length=0.4,
            dot_radius=0.04,
        )
        row_marginal_16 = self.get_probabilities_table(
            probability=r"1/6",
            labels=False,
            flip=True,
            dot_color=RED_B,
            side_length=0.4,
            dot_radius=0.04,
        )
        col_marginal_16 = self.get_probabilities_table(
            probability=r"1/6",
            labels=False,
            dot_color=BLUE_B,
            transpose=True,
            flip=True,
            side_length=0.4,
            dot_radius=0.04,
        )

        ## --- Animation ---- ##

        # Intro
        self.fade_in_out(t0.to_edge(ORIGIN))

        # Discrete probability distributions
        self.play(FadeIn(blue_die_faces))
        self.play(blue_die_faces.animate.to_edge(UP))
        self.play(
            *(
                prob.animate.next_to(blue_die_face, DOWN)
                for prob, blue_die_face in zip(probabilities, blue_die_faces)
            )
        )
        self.fade_in_out(t1.to_edge(ORIGIN))
        t2.to_edge(UP)
        probabilities_table.next_to(t2, DOWN)
        self.play(
            ReplacementTransform(
                probabilities + blue_die_faces,
                probabilities_table,
            )
        )
        self.play(FadeIn(t2))
        self.wait(1)

        # Adding probabilities
        prob_1_or_2.arrange_submobjects().next_to(probabilities_table, DOWN * 1.5)
        t3a.arrange(DOWN).to_edge(DOWN)
        self.play(FadeIn(t3a))
        self.play(FadeIn(prob_1_or_2))
        self.wait(0.5)
        frac13_1 = probabilities_table.get_entries(pos=(2, 2)).copy()
        frac13_2 = probabilities_table.get_entries(pos=(2, 3)).copy()
        frac13_1.move_to(VGroup(prob_1_or_2[6:9]).get_center())
        frac13_2.move_to(VGroup(prob_1_or_2[10:13]).get_center())

        self.wait(1)
        self.play(
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 2)).copy(), frac13_1
            ),
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 3)).copy(), frac13_2
            ),
            FadeOut(prob_1_or_2[6:9]),
            FadeOut(prob_1_or_2[10:13]),
        )
        self.wait(0.5)
        frac13.move_to(prob_1_or_2[9].get_center())
        self.play(
            ReplacementTransform(VGroup(frac13_1, frac13_2, prob_1_or_2[9]), frac13)
        )
        self.wait(0.25)
        self.play(frac13.animate.next_to(prob_1_or_2[5], RIGHT))
        self.wait(1)
        self.play(FadeOut(prob_1_or_2[:6]), FadeOut(frac13), FadeOut(t3a))

        # Multiplying probabilities
        prob_1_and_2.arrange_submobjects().next_to(probabilities_table, DOWN * 1.5)
        t3b.arrange(DOWN).to_edge(DOWN)
        self.play(FadeIn(t3b))
        self.play(FadeIn(prob_1_and_2))
        frac136_1 = probabilities_table.get_entries(pos=(2, 2)).copy()
        frac136_2 = probabilities_table.get_entries(pos=(2, 3)).copy()
        frac136_1.move_to(VGroup(prob_1_and_2[6:9]).get_center())
        frac136_2.move_to(VGroup(prob_1_and_2[10:13]).get_center())

        self.wait(1)
        self.play(
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 2)).copy(), frac136_1
            ),
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 3)).copy(), frac136_2
            ),
            FadeOut(prob_1_and_2[6:9]),
            FadeOut(prob_1_and_2[10:13]),
        )
        self.wait(0.5)
        frac136.move_to(prob_1_and_2[9].get_center())
        self.play(
            ReplacementTransform(VGroup(frac136_1, frac136_2, prob_1_and_2[9]), frac136)
        )
        self.wait(0.25)
        self.play(frac136.animate.next_to(prob_1_and_2[5], RIGHT))
        self.wait(1)
        self.play(FadeOut(prob_1_and_2[:6]), FadeOut(frac136), FadeOut(t3b))
        self.wait(1)

        # Joint probabibility distributions
        for t in (t4a, t4b, t4c):
            t.to_edge(UP)
        self.play(ReplacementTransform(t2, t4a))
        for table in (joint_dist_table_16, joint_dist_table_136, joint_dist_table_ab_1):
            table.scale(scale_factor=0.6)
            table.next_to(t4a, DOWN).to_edge(LEFT)
        joint_dist_table_ab_2 = joint_dist_table_ab_1.copy()

        for e1, e2, e3 in zip(
            joint_dist_table_16.get_entries_without_labels(),
            joint_dist_table_136.get_entries_without_labels(),
            joint_dist_table_ab_1.get_entries_without_labels(),
        ):
            e2.move_to(e1.get_center())
            e3.move_to(e1.get_center())
        self.wait(1)

        self.play(
            ReplacementTransform(probabilities_table, joint_dist_table_16),
        )
        self.wait(1)

        self.play(ReplacementTransform(t4a, t4b))
        self.wait(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_16.get_entries_without_labels(),
                joint_dist_table_136.get_entries_without_labels(),
            )
        )
        self.wait(1)

        self.play(ReplacementTransform(t4b, t4c))
        self.wait(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_136.get_entries_without_labels(),
                joint_dist_table_ab_1.get_entries_without_labels(),
            )
        )
        self.wait(1)
        self.play(
            FadeOut(t4c),
        )

        # Space complexity
        t5.to_edge(UP)
        self.play(FadeIn(t5))
        joint_dist_table_16.fade(1)
        joint_dist_table_136.fade(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_ab_1.get_entries_without_labels(),
                joint_dist_table_ab_2,
            )
        )
        self.wait(1)

        self.play(joint_dist_table_ab_2.animate.to_edge(LEFT))

        t5a.next_to(joint_dist_table_ab_2, RIGHT)
        self.play(FadeIn(t5a))
        self.wait(0.5)
        box_first_column = SurroundingRectangle(
            VGroup(
                *it.chain(
                    [VGroup(a[0]) for a in joint_dist_table_ab_2.get_columns()[1][1:]]
                )
            )
        )
        self.play(Create(box_first_column))
        self.wait(2)

        t5b.move_to(t5a.get_center())
        self.play(ReplacementTransform(t5a, t5b), Uncreate(box_first_column))
        self.wait(0.5)
        box_entries = SurroundingRectangle(
            joint_dist_table_ab_2.get_entries_without_labels()
        )
        self.play(Create(box_entries))
        self.wait(2)

        self.wait(1)
        self.play(FadeOut(t5b))
        t5c.to_edge(UP)
        self.play(ReplacementTransform(t5, t5c))
        self.wait(0.5)

        for eq in space_complexity_equations:
            eq.next_to(joint_dist_table_ab_2, RIGHT * 1.5)
        for i in range(1, len(space_complexity_equations)):
            self.play(
                ReplacementTransform(
                    space_complexity_equations[i - 1], space_complexity_equations[i]
                )
            )
            self.wait(0.01)
        self.wait(2)
        self.play(Uncreate(box_entries), FadeOut(space_complexity_equations[-1]))
        self.wait(2)

        # Time Complexity
        t6a.to_edge(UP)
        for e in (
            joint_dist_table_136b,
            row_marginal_636,
            row_marginal_16,
            col_marginal_16,
        ):
            e.scale(scale_factor=0.6)
        joint_dist_table_136b.next_to(t6a, DOWN)
        row_marginal_636.next_to(joint_dist_table_136b, DOWN)
        row_marginal_16.next_to(joint_dist_table_136b, DOWN)
        col_marginal_16.move_to(joint_dist_table_136b, RIGHT).shift(RIGHT * 1.5)
        self.play(ReplacementTransform(t5c, t6a))
        self.wait(1)
        self.play(ReplacementTransform(joint_dist_table_ab_2, joint_dist_table_136b))
        self.wait(0.5)
        self.play(FadeIn(row_marginal_636))
        for anim in (
            ReplacementTransform(col.copy(), row_cell)
            for col, row_cell in zip(
                joint_dist_table_136b.get_columns()[1:],
                row_marginal_636.get_entries_without_labels()[:6],
            )
        ):
            self.play(anim)
            self.wait(0.01)
        self.wait(1)

        self.play(
            ReplacementTransform(
                row_marginal_636,
                row_marginal_16,
            )
        )
        self.wait(1)

        self.play(FadeIn(col_marginal_16))
        for anim in (
            ReplacementTransform(row.copy(), col_cell)
            for row, col_cell in zip(
                joint_dist_table_136b.get_rows()[1:],
                col_marginal_16.get_entries_without_labels()[::2],
            )
        ):
            self.play(anim)
            self.wait(0.01)
        self.wait(1)
        self.play(FadeOut(joint_dist_table_136b, row_marginal_16, col_marginal_16))

        t6b.next_to(t6a, DOWN)
        self.play(FadeIn(t6b))
        self.wait(1)

        t6c.next_to(t6b, DOWN)
        self.play(FadeIn(t6c))
        self.wait(1)

        t6d.next_to(t6c, DOWN)
        self.play(FadeIn(t6d))
        self.wait(3)
        self.play(FadeOut(t6a, t6b, t6c, t6d))

    def sum_product_networks(self):
        """
        1. Graphical representation (complete, decomposable, normalised)
            1. Completeness: same events for children of a sum node
            2. Decomposition: different events for children of a product node
            3. Normalisation: probabilities for an event sum to 1
        2. More efficient representation
            1. Space: the memory needed is only 6 + 6
            2. Time: inference/marginalization is cheap (linear)
        """
        # Graphical Representation
        t1a = MarkupText(
            f"A <span fgcolor='{BLUE_B}'>computational graph</span> "
            f"is a drawing showing how to compute values",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1b = MarkupText(
            f"For probabilities, we only need "
            f"<span fgcolor='{BLUE_B}'>sums</span>"
            f" and "
            f"<span fgcolor='{BLUE_B}'>products</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1c = VGroup(
            MarkupText(
                f"If we consider independents events, a simple\n",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
            MarkupText(
                f"<span fgcolor='{BLUE_B}'>computational graph</span> "
                f"would look like this",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            ),
        )
        t1 = VGroup(t1a, t1b, t1c)
        g = nx.Graph()
        #           x
        #       +       +
        #     1...6   1...6
        g.add_edges_from(
            (
                (r"\times", r"+'"),
                (r"\times", r"+"),
                *((r"+'", r"a_" + str(i)) for i in range(1, 7)),
                *((r"+", r"b_" + str(i)) for i in range(1, 7)),
            )
        )
        graph = Graph(
            vertices=list(g.nodes),
            edges=list(g.edges),
            layout="tree",
            layout_scale=5,
            root_vertex=r"\times",
            labels=True,
            layout_config=dict(
                vertex_spacing=(1, 1.5),
            ),
        )
        t2a = MarkupText(
            f"A probability distribution is the "
            f"<span fgcolor='{BLUE_B}'>sum</span>-<span fgcolor='{RED_B}'>product</span> "
            f"of its events",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2b = MarkupText(
            f"A joint probability distribution is the "
            f"<span fgcolor='{RED_B}'>product</span> "
            f"of its variables",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        sum_product_24 = VGroup(
            MathTex(r"\mathbb{P}("),
            MathTex(r"D_1="),
            *get_die_faces(
                [2],
                side_length=0.5,
                dot_radius=0.04,
                dot_color=BLUE_B,
            ),
            MathTex(r")="),
            MathTex(r"\mathbb{P}(D_1="),
            *get_die_faces(
                [2],
                side_length=0.5,
                dot_radius=0.04,
                dot_color=BLUE_B,
            ),
            MathTex(r")"),
            MathTex(r"\times 1", color=RED_B),
            MathTex(r"+", color=BLUE_B),  # 8
            MathTex(r"\mathbb{P}(D_1="),
            *get_die_faces(
                [4],
                side_length=0.5,
                dot_radius=0.04,
                dot_color=BLUE_B,
            ),
            MathTex(r")"),
            MathTex(r"\times 0", color=RED_B),
        )
        sum_product_6_6 = VGroup(
            MathTex(r"\mathbb{P}("),
            MathTex(r"D_1="),
            *get_die_faces(
                [6],
                side_length=0.5,
                dot_radius=0.04,
                dot_color=BLUE_B,
            ),
            MathTex(r", D_2="),
            *get_die_faces(
                [6],
                side_length=0.5,
                dot_radius=0.04,
                dot_color=RED_B,
            ),
            MathTex(r")="),
            MathTex(r"\mathbb{P}(D_1="),
            *get_die_faces([6], side_length=0.5, dot_radius=0.04),
            MathTex(r")\times 1"),
            MathTex(r"\times", color=RED_B),
            MathTex(r"\mathbb{P}(D_2="),
            *get_die_faces([6], side_length=0.5, dot_radius=0.04),
            MathTex(r")\times 1"),
        )

        # Complexity
        t3a = MarkupText(
            f"In contrary to the full table, the "
            f"<span fgcolor='{BLUE_B}'>sum</span>-<span fgcolor='{RED_B}'>product</span> "
            f"network (SPN) requires fewer computations",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3b = MarkupText(
            f"However, this kind of network has drawbacks",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )

        ## -- Animation -- ##
        t1c.arrange(DOWN)
        t1.to_edge(UP)
        t1.arrange(DOWN)
        self.play(FadeIn(t1a))
        self.wait(2)
        self.play(FadeIn(t1b))
        self.wait(2)
        self.play(FadeIn(t1c))
        self.wait(2)
        self.play(FadeOut(t1))
        self.wait(2)

        # Sum-product
        t2a.to_edge(UP)
        graph.next_to(t2a, DOWN)
        _graph = graph.copy()
        sum_product_24.arrange_submobjects()
        sum_product_24.next_to(graph, DOWN)
        sum_product_24.scale(scale_factor=0.9)
        self.play(Create(graph))
        self.wait(1)

        self.play(FadeIn(t2a))
        self.play(
            FadeIn(sum_product_24[0:4]),
            graph.vertices[r"a_2"].animate.set_color(BLUE_B),
        )
        self.wait(1)

        self.play(
            FadeIn(sum_product_24[4:8]),
            graph.edges[(r"+'", r"a_2")].animate.set_color(BLUE_B),
        )
        self.wait(1)

        self.play(
            graph.vertices[r"+'"].animate.set_color(BLUE_B),
            FadeIn(sum_product_24[8]),
        )
        self.wait(1)

        self.play(
            FadeIn(sum_product_24[9:]),
            graph.vertices[r"a_4"].animate.set_color(GREY),
            graph.edges[(r"+'", r"a_4")].animate.set_color(GREY),
        )
        self.wait(3)
        self.play(FadeOut(sum_product_24), FadeOut(t2a))
        self.play(ReplacementTransform(graph, _graph))
        graph = _graph

        # Joint distribution
        t2b.to_edge(UP)
        graph.next_to(t2b, DOWN)
        sum_product_6_6.arrange_submobjects()
        sum_product_6_6.next_to(graph, DOWN)
        sum_product_6_6.scale(scale_factor=0.9)
        self.wait(1)

        self.play(FadeIn(t2b))
        self.play(
            FadeIn(sum_product_6_6[0:6]),
            graph.vertices[r"a_6"].animate.set_color(BLUE_B),
            graph.vertices[r"b_6"].animate.set_color(BLUE_B),
        )
        self.wait(1)

        self.play(
            graph.edges[(r"+'", r"a_6")].animate.set_color(BLUE_B),
            FadeIn(sum_product_6_6[6:9]),
        )
        self.wait(1)

        self.play(
            graph.edges[(r"+", r"b_6")].animate.set_color(BLUE_B),
            FadeIn(sum_product_6_6[10:]),
        )
        self.wait(1)

        self.play(
            FadeIn(sum_product_6_6[9]),
            graph.edges[(r"\times", r"+'")].animate.set_color(RED_B),
            graph.edges[(r"\times", r"+")].animate.set_color(RED_B),
        )
        self.wait(5)
        self.play(FadeOut(t2b), Uncreate(graph), FadeOut(sum_product_6_6))

        # Complexity
        t3a.to_edge(UP)
        t3b.next_to(t3a, DOWN)
        self.play(FadeIn(t3a))
        self.wait(1)
        self.play(ReplacementTransform(t3a, t3b))
        self.wait(3)
        self.play(FadeOut(t3a), FadeOut(t3b))

    def uncertainty(self):
        """
        4. Robust sum-product networks
            1. What if your knowledge is even more uncertain?
                1. If a probability varies, then the SPN might give completely wrong results
                2. How to include imprecision? use imprecise probabilities!
                3. Probabilities belong to a range, and sum nodes are constrained
            2. What do you infer then? Bounds (lower/upper)
        5. Further concepts:
            1. Independence / Conditional independence
            2. Learning SPNs
        """
        t1a = MarkupText(
            f"Probabilities are "
            f"<span fgcolor='{BLUE_B}'>subjective</span> "
            f"and often "
            f"<span fgcolor='{BLUE_B}'>unknown</span> ",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1b = MarkupText(
            f"What if they were also " f"<span fgcolor='{BLUE_B}'>uncertain</span> ?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1c = MarkupText(
            f"If the die's fairness is "
            f"<span fgcolor='{BLUE_B}'>uncertain</span>"
            f", then computed probabilities are often wrong",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1d = MarkupText(
            f"We say that the SPN is not " f"<span fgcolor='{BLUE_B}'>robust</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1 = VGroup(t1a, t1b, t1c, t1d)

        t2a = MarkupText(
            f"Let us consider a die, but its fairness is "
            f"<span fgcolor='{BLUE_B}'>uncertain</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2b = MarkupText(
            f"The probabilities are <span fgcolor='{RED_B}'>imprecise</span>"
            f", meaning that instead "
            f"of <span fgcolor='{BLUE_B}'>real-valued</span> probabilities, ",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2c = MarkupText(
            f"we have <span fgcolor='{RED_B}'>value sets</span> probabilities",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2 = VGroup(t2a, t2b, t2c)
        real_probabilities = VGroup(
            *[MathTex(r"p_" + str(i), color=BLUE_B) for i in range(6)]
        )
        _real_probabilities = MathTex(
            r"p_1 &=1/6\\"
            r"p_2 &=1/6\\"
            r"p_3 &=1/6\\"
            r"p_4 &=1/6\\"
            r"p_5 &=1/6\\"
            r"p_6 &=1/6\\",
            font_size=DEFAULT_FONT_SIZE * 0.7,
        )
        set_probabilities = VGroup(
            *[MathTex(r"p_" + str(i), color=RED_B) for i in range(6)]
        )
        _set_probabilities = MathTex(
            r"&L_1 \leq p_1 \leq U_1\\",
            r"&L_2 \leq p_2 \leq U_2\\",
            r"&L_3 \leq p_3 \leq U_3\\",
            r"&L_4 \leq p_4 \leq U_4\\",
            r"&L_5 \leq p_5 \leq U_5\\",
            r"&L_6 \leq p_6 \leq U_6\\",
            font_size=DEFAULT_FONT_SIZE * 0.7,
        )
        real_prob_table = self.get_probabilities_table(real_probabilities)
        set_prob_table = self.get_probabilities_table(
            set_probabilities, dot_color=RED_B
        )

        t3a = MarkupText(
            f"What robustness you gain from using <span fgcolor='{RED_B}'>imprecise</span>, "
            f"you lose in computational complexity",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3b = MarkupText(
            "This is a " f"<span fgcolor='{BLUE_B}'>trade-off</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3c = MarkupText(
            f"Instead, we are more interested in finding the "
            f"<span fgcolor='{BLUE_B}'>minimum</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3d = MarkupText(
            f" and "
            f"<span fgcolor='{BLUE_B}'>maximum</span> "
            f"values for a given probability",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3 = VGroup(t3a, t3b, t3c, t3d)
        real_prob_1 = VGroup(
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"p_1"),
        )
        min_prob_1 = VGroup(
            MathTex(r"\min"),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"?"),
        )
        max_prob_1 = VGroup(
            MathTex(r"\max"),
            MathTex(r"\mathbb{P}(D="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"?"),
        )

        ## --- Animation ---- ##
        t1.arrange(DOWN)
        t1.to_edge(UP)
        for t in t1:
            self.play(FadeIn(t))
            self.wait(1)
        self.wait(2)

        t2.arrange(DOWN)
        t2.to_edge(UP)
        for table in (real_prob_table, set_prob_table):
            table.scale(scale_factor=0.75)
            table.next_to(t2, DOWN).shift(LEFT * 2)
        for prob in (_real_probabilities, _set_probabilities):
            prob.next_to(real_prob_table, RIGHT).shift(RIGHT * 2)
        self.play(ReplacementTransform(t1, t2a))
        self.wait(1)
        self.play(FadeIn(t2b))
        self.wait(1)
        self.play(FadeIn(real_prob_table, _real_probabilities))
        self.wait(2)
        self.play(
            FadeIn(t2c),
            ReplacementTransform(real_prob_table, set_prob_table),
            ReplacementTransform(_real_probabilities, _set_probabilities),
        )
        self.wait(3)
        self.play(FadeOut(set_prob_table, _set_probabilities))

        t3.arrange(DOWN)
        t3.to_edge(UP)
        for prob in (real_prob_1, min_prob_1, max_prob_1):
            prob.arrange()
        real_prob_1.next_to(t3, DOWN * 2).shift(LEFT * 3)
        min_prob_1.move_to(real_prob_1)
        max_prob_1.next_to(real_prob_1, RIGHT).shift(RIGHT)
        self.play(ReplacementTransform(t2, t3a))
        self.wait(2)
        self.play(FadeIn(real_prob_1))
        self.wait(1)
        self.play(FadeIn(t3b))
        self.wait(1)
        self.play(FadeIn(t3c))
        self.wait(0.5)
        self.play(
            ReplacementTransform(real_prob_1, min_prob_1),
        )
        self.wait(1)
        self.play(
            FadeIn(t3d),
            FadeIn(max_prob_1),
        )
        self.wait(1)
        self.play(
            FadeOut(t3),
            FadeOut(min_prob_1),
            FadeOut(max_prob_1),
        )

    def further(self):
        """
        5. Further concepts:
            1. Independence / Conditional independence
            2. SPN have to satisfy certain properties to represent probabilities
            3. Imprecise probabilities search space
            4. Learning SPNs
        """
        t1a = Paragraph(
            f"Over the course of this video, we have skipped essential concepts\n"
            f"for the sake of clarity.",
            font_size=DEFAULT_FONT_SIZE * 0.5,
            alignment="center",
        )
        t1b = MarkupText(
            f"For example, how do you represent "
            f"<span fgcolor='{BLUE_B}'>non-independence</span> "
            f"in SPNs?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1c = MarkupText(
            f"What would the graphs look like?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t1 = VGroup(t1a, t1b, t1c)

        t2a = MarkupText(
            f"SPNs have to satisfy a number of "
            f"<span fgcolor='{BLUE_B}'>properties</span>",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2b = MarkupText(
            f"in order to accurately represent probability distributions",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2c = MarkupText(
            f"What would they be?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t2 = VGroup(t2a, t2b, t2c)

        t3a = MarkupText(
            f"In addition, using "
            f"<span fgcolor='{RED_B}'>imprecise</span> "
            f"probabilities",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3b = MarkupText(
            f"How would you find the extrema of probabilities?",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        t3 = VGroup(t3a, t3b)

        ## --- Animation ---- ##
        t1.arrange(DOWN)
        t1.to_edge(UP)
        for t in t1:
            self.play(FadeIn(t))
            self.wait(1)
        self.wait(2)

        t2.arrange(DOWN)
        t2.to_edge(UP)
        self.play(ReplacementTransform(t1, t2[0]))
        self.wait(1)
        for t in t2[1:]:
            self.play(FadeIn(t))
            self.wait(1)
        self.wait(2)

        t3.arrange(DOWN)
        t3.to_edge(UP)
        self.play(ReplacementTransform(t2, t3[0]))
        self.wait(1)
        for t in t3[1:]:
            self.play(FadeIn(t))
            self.wait(1)
        self.wait(3)

    def get_probabilities_table(
        self,
        probability: str | VGroup[MathTex] = r"\frac{1}{6}",
        font_size: float = DEFAULT_FONT_SIZE,
        values: list[int] = range(1, 7),
        buff: float = LARGE_BUFF,
        transpose: bool = False,
        flip: bool = False,
        labels: bool = True,
        side_length: float = 1.0,
        corner_radius: float = 0.15,
        stroke_color: str = WHITE,
        stroke_width: float = 2.0,
        fill_color: str = GREY_E,
        dot_radius: float = 0.08,
        dot_color: str = BLUE_B,
        dot_coalesce_factor: float = 0.5,
    ):
        """Return an MobjectTable containing a fair die faces and probabilities."""
        if isinstance(probability, str):
            probabilities = VGroup(
                *[MathTex(probability, font_size=font_size) for _ in range(6)]
            )
        elif isinstance(probability, VGroup):
            probabilities = probability
        die_faces = get_die_faces(
            values=values,
            buff=buff,
            side_length=side_length,
            corner_radius=corner_radius,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            dot_radius=dot_radius,
            dot_color=dot_color,
            dot_coalesce_factor=dot_coalesce_factor,
        )
        if transpose:
            if flip:
                values = [[p, df] for p, df in zip(probabilities, die_faces)]
            else:
                values = [[df, p] for df, p in zip(die_faces, probabilities)]
            if labels:
                values = [[MathTex("d"), MathTex(r"\mathbb{P}(D=d)")]] + values
        else:
            if flip:
                values = [
                    [p.copy() for p in probabilities],
                    [df for df in die_faces],
                ]
            else:
                values = [
                    [df for df in die_faces],
                    [p.copy() for p in probabilities],
                ]
            if labels:
                values[0] = [MathTex("d")] + values[0]
                values[1] = [MathTex(r"\mathbb{P}(D=d)")] + values[1]
        probabilities_table = MobjectTable(
            values,
            h_buff=MED_LARGE_BUFF,
            v_buff=MED_LARGE_BUFF,
            line_config=dict(stroke_width=1),
        )
        return probabilities_table

    def dice_joint_table(self, table_values):
        """Joint probability distribution with 6-sided dice labels."""
        return MobjectTable(
            table_values,
            top_left_entry=Tex(r"$d_1$\textbackslash $d_2$"),
            row_labels=[
                df.copy()
                for df in get_die_faces(
                    side_length=0.5,
                    dot_radius=0.04,
                )
            ],
            col_labels=[
                df.copy()
                for df in get_die_faces(
                    side_length=0.5,
                    dot_color=RED_B,
                    dot_radius=0.04,
                )
            ],
            v_buff=MED_LARGE_BUFF,
            h_buff=MED_LARGE_BUFF,
            line_config=dict(stroke_width=1),
        )
