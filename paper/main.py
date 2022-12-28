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
        4. Graphical representation: complete, decomposable, normalised
            1. Completeness: same events for children of a sum node
            2. Decomposition: different events for children of a product node
            3. Normalisation: probabilities for an event sum to 1
        5. More efficient representation
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
        2. Learning SPNs
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
        blue_die_faces = get_die_faces()
        probabilities = VGroup(*[MathTex(r"\frac{1}{6}") for _ in range(6)])
        probabilities_table = MobjectTable(
            [
                [MathTex("f")] + [df for df in get_die_faces()],
                [MathTex(r"\mathbb{P}(F=f)")] + [p.copy() for p in probabilities],
            ],
            h_buff=MED_LARGE_BUFF,
            v_buff=MED_LARGE_BUFF,
        )
        t2 = Tex(r"$F$ is the value obtained by rolling a 6-sided fair die")

        # Adding probabilities
        t3 = Text(
            "Adding probabilities is as simple as summing them",
            font_size=DEFAULT_FONT_SIZE * 0.5,
        )
        prob_12 = VGroup(
            MathTex(r"\mathbb{P}(F="),
            *get_die_faces([1, 2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"="),
            MathTex(r"\mathbb{P}(F="),
            *get_die_faces([1], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
            MathTex(r"+"),
            MathTex(r"\mathbb{P}(F="),
            *get_die_faces([2], side_length=0.5, dot_radius=0.04),
            MathTex(r")"),
        )
        frac13 = MathTex(r"\frac13")

        # Multiplying probabilities

        # Joint probability distribution
        t4 = Tex(r"What if you have multiple independent dices?")
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
                        r"a_" + str(i) + r"\times " + r"b_" + str(j),
                        substrings_to_isolate=["a", "b"],
                    )
                    for j in range(1, 7)
                ]
                for i in range(1, 7)
            ]
        )
        for e in joint_dist_table_ab_1.get_entries_without_labels():
            e.set_color_by_tex("a", BLUE_B)
            e.set_color_by_tex("b", RED_B)

        # Space complexity
        t5 = Text(
            "For a single die, you need 6 values", font_size=DEFAULT_FONT_SIZE * 0.45
        )
        variables = VGroup(MathTex("6"), MathTex("1"))
        space_complexity_eq1 = MathTex("6^{{1}}", "=", "{{6}}")
        space_complexity_eq2 = MathTex("6^{{2}}", "=", "{{36}}")

        # Time complexity

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
        prob_12.arrange_submobjects().next_to(probabilities_table, DOWN)
        t3.shift(DOWN * 2)
        self.play(t3.animate.to_edge(DOWN))
        self.play(FadeIn(prob_12))
        self.wait(0.5)
        frac13_1 = probabilities_table.get_entries(pos=(2, 2)).copy()
        frac13_2 = probabilities_table.get_entries(pos=(2, 3)).copy()
        frac13_1.move_to(VGroup(prob_12[5:8]).get_center())
        frac13_2.move_to(VGroup(prob_12[9:12]).get_center())

        self.play(
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 2)).copy(), frac13_1
            ),
            ReplacementTransform(
                probabilities_table.get_entries(pos=(2, 3)).copy(), frac13_2
            ),
            FadeOut(prob_12[5:8]),
            FadeOut(prob_12[9:12]),
        )
        self.wait(0.5)
        frac13.move_to(prob_12[8].get_center())
        self.play(ReplacementTransform(VGroup(frac13_1, frac13_2, prob_12[8]), frac13))
        self.wait(0.25)
        self.play(frac13.animate.next_to(prob_12[4], RIGHT))
        self.wait(1)
        self.play(FadeOut(prob_12[:5]), FadeOut(frac13), FadeOut(t3))

        # Multiplying probabilities
        self.play(ReplacementTransform(t2, t4.to_edge(UP)))
        self.wait(1)

        # Joint probabibility distributions
        for table in (joint_dist_table_16, joint_dist_table_136, joint_dist_table_ab_1):
            table.scale(scale_factor=0.6)
            table.next_to(t4, DOWN)
        joint_dist_table_ab_2 = joint_dist_table_ab_1.copy()

        for e1, e2, e3 in zip(
            joint_dist_table_16.get_entries_without_labels(),
            joint_dist_table_136.get_entries_without_labels(),
            joint_dist_table_ab_1.get_entries_without_labels(),
        ):
            e2.move_to(e1.get_center())
            e3.move_to(e1.get_center())

        self.play(
            ReplacementTransform(probabilities_table, joint_dist_table_16),
        )
        self.wait(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_16.get_entries_without_labels(),
                joint_dist_table_136.get_entries_without_labels(),
            )
        )
        self.wait(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_136.get_entries_without_labels(),
                joint_dist_table_ab_1.get_entries_without_labels(),
            )
        )
        self.wait(1)
        joint_dist_table_16.fade(1)
        joint_dist_table_136.fade(1)
        self.play(
            ReplacementTransform(
                joint_dist_table_ab_1,
                joint_dist_table_ab_2,
            ),
        )
        self.play(joint_dist_table_ab_2.animate.to_edge(LEFT))
        t5.next_to(joint_dist_table_ab_2, RIGHT * 2)
        self.play(FadeIn(t5))

        self.wait(5)

    def dice_joint_table(self, table_values):
        """Joint probability distribution with 6-sided dice labels."""
        return MobjectTable(
            table_values,
            top_left_entry=Tex(r"$f_1$\textbackslash $f_2$"),
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
