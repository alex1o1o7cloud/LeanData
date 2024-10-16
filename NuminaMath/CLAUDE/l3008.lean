import Mathlib

namespace NUMINAMATH_CALUDE_area_ratio_EFPH_ABCD_l3008_300850

-- Define the parallelogram ABCD
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define points E, F, P, H on the sides of ABCD
def OnSide (P Q R : Point) : Prop := sorry

-- Define the ratio of line segments
def RatioOfSegments (P Q R : Point) (r : ℚ) : Prop := sorry

-- Define the area of a polygon
def Area (polygon : Set Point) : ℝ := sorry

theorem area_ratio_EFPH_ABCD (A B C D E F P H : Point) :
  Parallelogram A B C D →
  OnSide E A B →
  OnSide F B C →
  OnSide P C D →
  OnSide H A D →
  RatioOfSegments A E B (1/3) →
  RatioOfSegments B F C (1/3) →
  RatioOfSegments C P D (1/2) →
  RatioOfSegments A H D (1/2) →
  (Area {E, F, P, H}) / (Area {A, B, C, D}) = 37/72 := by sorry

end NUMINAMATH_CALUDE_area_ratio_EFPH_ABCD_l3008_300850


namespace NUMINAMATH_CALUDE_contact_list_count_is_45_l3008_300801

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let extended_family_added : ℕ := 5
  let acquaintances_added : ℕ := 7
  let coworkers_added : ℕ := 10
  let extended_family_removed : ℕ := 3
  let acquaintances_removed : ℕ := 4
  let coworkers_removed : ℕ := (coworkers_added * 3) / 10

  let total_added : ℕ := classmates + out_of_school_friends + immediate_family + 
                         extended_family_added + acquaintances_added + coworkers_added
  let total_removed : ℕ := extended_family_removed + acquaintances_removed + coworkers_removed

  total_added - total_removed

theorem contact_list_count_is_45 : contact_list_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_contact_list_count_is_45_l3008_300801


namespace NUMINAMATH_CALUDE_room_width_l3008_300863

/-- 
Given a rectangular room with length 20 feet and 1 foot longer than its width,
prove that the width of the room is 19 feet.
-/
theorem room_width (length width : ℕ) : 
  length = 20 ∧ length = width + 1 → width = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l3008_300863


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l3008_300846

-- Define the number of options for each category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4
def num_drinks : ℕ := 3

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 3

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats) * (Nat.choose num_vegetables vegetables_to_choose) * (num_desserts) * (num_drinks) = 360 := by
  sorry


end NUMINAMATH_CALUDE_tyler_meal_choices_l3008_300846


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l3008_300876

-- Define the concept of a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to get the angles of a triangle
def angles (t : Triangle) : ℝ × ℝ × ℝ :=
  sorry

-- Define the property that corresponding angles are either equal or sum to 180°
def corresponding_angles_property (t1 t2 : Triangle) : Prop :=
  let (α1, β1, γ1) := angles t1
  let (α2, β2, γ2) := angles t2
  (α1 = α2 ∨ α1 + α2 = 180) ∧
  (β1 = β2 ∨ β1 + β2 = 180) ∧
  (γ1 = γ2 ∨ γ1 + γ2 = 180)

-- Theorem statement
theorem corresponding_angles_equal (t1 t2 : Triangle) 
  (h : corresponding_angles_property t1 t2) : 
  angles t1 = angles t2 :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l3008_300876


namespace NUMINAMATH_CALUDE_triangle_angle_from_side_ratio_l3008_300825

theorem triangle_angle_from_side_ratio :
  ∀ (a b c : ℝ) (A B C : ℝ),
    a > 0 → b > 0 → c > 0 →
    A > 0 → B > 0 → C > 0 →
    a / b = 1 / Real.sqrt 3 →
    b / c = Real.sqrt 3 / 2 →
    A + B + C = π →
    c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
    B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_side_ratio_l3008_300825


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3008_300802

theorem root_sum_theorem (a b c d r : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (r - a) * (r - b) * (r - c) * (r - d) = 4 →
  4 * r = a + b + c + d := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3008_300802


namespace NUMINAMATH_CALUDE_crayon_cost_l3008_300895

theorem crayon_cost (total_students : ℕ) (buyers : ℕ) (crayons_per_student : ℕ) (crayon_cost : ℕ) :
  total_students = 50 →
  buyers > total_students / 2 →
  buyers * crayons_per_student * crayon_cost = 1998 →
  crayon_cost > crayons_per_student →
  crayon_cost = 37 :=
by sorry

end NUMINAMATH_CALUDE_crayon_cost_l3008_300895


namespace NUMINAMATH_CALUDE_unseen_corner_color_code_l3008_300829

/-- Represents the colors of a Rubik's Cube -/
inductive Color
  | White
  | Yellow
  | Green
  | Blue
  | Orange
  | Red

/-- Represents a corner piece of a Rubik's Cube -/
structure Corner :=
  (c1 c2 c3 : Color)

/-- Assigns a numeric code to each color -/
def color_code (c : Color) : ℕ :=
  match c with
  | Color.White => 1
  | Color.Yellow => 2
  | Color.Green => 3
  | Color.Blue => 4
  | Color.Orange => 5
  | Color.Red => 6

/-- Represents the state of a Rubik's Cube -/
structure RubiksCube :=
  (corners : List Corner)

/-- Represents a solved Rubik's Cube -/
def solved_cube : RubiksCube := sorry

/-- Represents a scrambled Rubik's Cube with 7 visible corners -/
def scrambled_cube : RubiksCube := sorry

theorem unseen_corner_color_code :
  ∀ (cube : RubiksCube),
    (cube.corners.length = 8) →
    (∃ (visible_corners : List Corner), visible_corners.length = 7 ∧ visible_corners ⊆ cube.corners) →
    ∃ (unseen_corner : Corner),
      unseen_corner ∈ cube.corners ∧
      unseen_corner ∉ (visible_corners : List Corner) ∧
      color_code (unseen_corner.c1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_unseen_corner_color_code_l3008_300829


namespace NUMINAMATH_CALUDE_range_x_when_a_zero_range_a_for_p_sufficient_q_l3008_300878

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_x_when_a_zero :
  ∀ x : ℝ, p x ∧ ¬(q 0 x) ↔ -7/2 ≤ x ∧ x < -3 :=
sorry

-- Theorem for the second question
theorem range_a_for_p_sufficient_q :
  (∀ x : ℝ, p x → ∀ a : ℝ, q a x) ↔ ∀ a : ℝ, -5/2 ≤ a ∧ a ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_zero_range_a_for_p_sufficient_q_l3008_300878


namespace NUMINAMATH_CALUDE_perimeter_relations_l3008_300884

variable (n : ℕ+) (r : ℝ) (hr : r > 0)

/-- Perimeter of regular n-gon circumscribed around a circle with radius r -/
noncomputable def K (n : ℕ+) (r : ℝ) : ℝ := sorry

/-- Perimeter of regular n-gon inscribed in a circle with radius r -/
noncomputable def k (n : ℕ+) (r : ℝ) : ℝ := sorry

theorem perimeter_relations (n : ℕ+) (r : ℝ) (hr : r > 0) :
  (K (2 * n) r = (2 * K n r * k n r) / (K n r + k n r)) ∧
  (k (2 * n) r = Real.sqrt ((k n r) * (K (2 * n) r))) := by sorry

end NUMINAMATH_CALUDE_perimeter_relations_l3008_300884


namespace NUMINAMATH_CALUDE_negation_is_returning_transformation_l3008_300898

theorem negation_is_returning_transformation (a : ℝ) : -(-a) = a := by
  sorry

end NUMINAMATH_CALUDE_negation_is_returning_transformation_l3008_300898


namespace NUMINAMATH_CALUDE_probability_jack_queen_king_hearts_l3008_300837

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (face_cards_per_suit : Nat)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    suits := 4,
    face_cards_per_suit := 3 }

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

theorem probability_jack_queen_king_hearts (d : Deck := standard_deck) :
  probability d d.face_cards_per_suit = 3 / 52 := by
  sorry

#eval probability standard_deck standard_deck.face_cards_per_suit

end NUMINAMATH_CALUDE_probability_jack_queen_king_hearts_l3008_300837


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l3008_300831

theorem sum_of_square_roots (x : ℝ) 
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15) 
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) : 
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l3008_300831


namespace NUMINAMATH_CALUDE_recruitment_probabilities_l3008_300887

/-- Represents the recruitment scenario -/
structure RecruitmentScenario where
  totalQuestions : Nat
  drawnQuestions : Nat
  knownQuestions : Nat
  minCorrect : Nat

/-- Calculates the probability of proceeding to the interview stage -/
def probabilityToInterview (scenario : RecruitmentScenario) : Rat :=
  sorry

/-- Represents the probability distribution of correctly answerable questions -/
structure ProbabilityDistribution where
  p0 : Rat
  p1 : Rat
  p2 : Rat
  p3 : Rat

/-- Calculates the probability distribution of correctly answerable questions -/
def probabilityDistribution (scenario : RecruitmentScenario) : ProbabilityDistribution :=
  sorry

theorem recruitment_probabilities 
  (scenario : RecruitmentScenario)
  (h1 : scenario.totalQuestions = 10)
  (h2 : scenario.drawnQuestions = 3)
  (h3 : scenario.knownQuestions = 6)
  (h4 : scenario.minCorrect = 2) :
  probabilityToInterview scenario = 2/3 ∧
  let dist := probabilityDistribution scenario
  dist.p0 = 1/30 ∧ dist.p1 = 3/10 ∧ dist.p2 = 1/2 ∧ dist.p3 = 1/6 :=
sorry

end NUMINAMATH_CALUDE_recruitment_probabilities_l3008_300887


namespace NUMINAMATH_CALUDE_linear_regression_not_guaranteed_point_l3008_300892

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  dataPoints : List Point

/-- Checks if a point is in the list of data points -/
def isDataPoint (p : Point) (lr : LinearRegression) : Prop :=
  p ∈ lr.dataPoints

/-- Theorem: The linear regression line is not guaranteed to pass through (6.5, 8) -/
theorem linear_regression_not_guaranteed_point (lr : LinearRegression) 
  (h1 : isDataPoint ⟨2, 3⟩ lr)
  (h2 : isDataPoint ⟨5, 7⟩ lr)
  (h3 : isDataPoint ⟨8, 9⟩ lr)
  (h4 : isDataPoint ⟨11, 13⟩ lr) :
  ¬ ∀ (regression_line : Point → Prop), 
    (∀ p, isDataPoint p lr → regression_line p) → 
    regression_line ⟨6.5, 8⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_linear_regression_not_guaranteed_point_l3008_300892


namespace NUMINAMATH_CALUDE_bisected_tangents_iff_parabola_l3008_300858

/-- A curve in the xy-plane -/
structure Curve where
  -- The equation of the curve
  equation : ℝ → ℝ → Prop

/-- Property that any tangent line segment between the point of tangency and the x-axis is bisected by the y-axis -/
def has_bisected_tangents (c : Curve) : Prop :=
  ∀ (x y : ℝ), c.equation x y →
    ∃ (slope : ℝ), 
      -- The tangent line at (x, y) intersects the x-axis at (-x, 0)
      y = slope * (x - (-x))

/-- A parabola of the form y^2 = Cx -/
def is_parabola (c : Curve) : Prop :=
  ∃ (C : ℝ), ∀ (x y : ℝ), c.equation x y ↔ y^2 = C * x

/-- Theorem stating the equivalence between the bisected tangents property and being a parabola -/
theorem bisected_tangents_iff_parabola (c : Curve) :
  has_bisected_tangents c ↔ is_parabola c :=
sorry

end NUMINAMATH_CALUDE_bisected_tangents_iff_parabola_l3008_300858


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3008_300849

/-- The number of knights at the round table -/
def n : ℕ := 30

/-- The probability that at least two of three randomly chosen knights are adjacent -/
def P : ℚ := 572 / 203

/-- The main theorem stating the probability of adjacent knights -/
theorem adjacent_knights_probability : 
  (1 : ℚ) - (n - 3 : ℚ) * (n - 5 : ℚ) * (n - 7 : ℚ) / (n.choose 3 : ℚ) = P := by
  sorry


end NUMINAMATH_CALUDE_adjacent_knights_probability_l3008_300849


namespace NUMINAMATH_CALUDE_non_defective_engines_count_l3008_300830

def total_engines (num_batches : ℕ) (engines_per_batch : ℕ) : ℕ :=
  num_batches * engines_per_batch

def non_defective_fraction : ℚ := 3/4

theorem non_defective_engines_count 
  (num_batches : ℕ) 
  (engines_per_batch : ℕ) 
  (h1 : num_batches = 5) 
  (h2 : engines_per_batch = 80) :
  ↑(total_engines num_batches engines_per_batch) * non_defective_fraction = 300 := by
  sorry

end NUMINAMATH_CALUDE_non_defective_engines_count_l3008_300830


namespace NUMINAMATH_CALUDE_triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l3008_300838

/-- The area of a triangle with two sides of length 17 and one side of length 16 is 120 -/
theorem triangle_area_with_sides_17_17_16 : ℝ → Prop :=
  fun area =>
    ∀ (D E F : ℝ × ℝ),
      let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
      let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
      let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
      de = 17 ∧ ef = 17 ∧ df = 16 →
      area = 120

/-- Proof of the theorem -/
theorem prove_triangle_area_with_sides_17_17_16 : triangle_area_with_sides_17_17_16 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l3008_300838


namespace NUMINAMATH_CALUDE_square_side_length_l3008_300864

theorem square_side_length (diagonal : ℝ) (h : diagonal = 4) : 
  ∃ side : ℝ, side = 2 * Real.sqrt 2 ∧ side ^ 2 + side ^ 2 = diagonal ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3008_300864


namespace NUMINAMATH_CALUDE_marbles_ratio_l3008_300834

def total_marbles : ℕ := 63
def your_initial_marbles : ℕ := 16

def brother_marbles : ℕ → ℕ → ℕ
  | your_marbles, marbles_given => 
    (total_marbles - your_marbles - (3 * (your_marbles - marbles_given))) / 2 + marbles_given

def your_final_marbles : ℕ := your_initial_marbles - 2

theorem marbles_ratio : 
  ∃ (m : ℕ), m > 0 ∧ your_final_marbles = m * (brother_marbles your_initial_marbles 2) ∧
  (your_final_marbles : ℚ) / (brother_marbles your_initial_marbles 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_marbles_ratio_l3008_300834


namespace NUMINAMATH_CALUDE_marble_theorem_l3008_300891

def marble_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo : ℕ := wolfgang_marbles + ludo_marbles
  let michael_marbles : ℕ := (2 * total_wolfgang_ludo) / 3
  let total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles
  wolfgang_marbles = 16 →
  total_marbles / 3 = 20

theorem marble_theorem : marble_problem 16 := by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l3008_300891


namespace NUMINAMATH_CALUDE_sin_2010_degrees_l3008_300806

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2010_degrees_l3008_300806


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_empty_iff_l3008_300883

def A (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 5}

theorem union_of_A_and_B (m : ℝ) :
  m = -3 → A m ∪ B = {x | -7 < x ∧ x ≤ 5} := by sorry

theorem intersection_empty_iff (m : ℝ) :
  A m ∩ B = ∅ ↔ m ≤ -4 ∨ 1 ≤ m := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_empty_iff_l3008_300883


namespace NUMINAMATH_CALUDE_adjacent_removal_unequal_sums_l3008_300867

theorem adjacent_removal_unequal_sums (arrangement : List ℕ) : 
  arrangement.length = 2005 → 
  ∃ (i : Fin 2005), 
    ¬∃ (partition : List ℕ → List ℕ × List ℕ), 
      let remaining := arrangement.removeNth i.val ++ arrangement.removeNth ((i.val + 1) % 2005)
      (partition remaining).1.sum = (partition remaining).2.sum :=
by sorry

end NUMINAMATH_CALUDE_adjacent_removal_unequal_sums_l3008_300867


namespace NUMINAMATH_CALUDE_circle_center_coord_sum_l3008_300888

theorem circle_center_coord_sum (x y : ℝ) :
  x^2 + y^2 = 4*x - 6*y + 9 →
  ∃ (center_x center_y : ℝ), center_x + center_y = -1 ∧
    ∀ (point_x point_y : ℝ),
      (point_x - center_x)^2 + (point_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coord_sum_l3008_300888


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l3008_300823

theorem sin_2alpha_plus_pi_6 (α : ℝ) (h : Real.sin (α - π/3) = 2/3 + Real.sin α) :
  Real.sin (2*α + π/6) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l3008_300823


namespace NUMINAMATH_CALUDE_chloe_carrot_count_l3008_300808

/-- Given Chloe's carrot picking scenario, prove the final number of carrots. -/
theorem chloe_carrot_count (initial_carrots : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) 
  (h1 : initial_carrots = 48)
  (h2 : thrown_out = 45)
  (h3 : picked_next_day = 42) :
  initial_carrots - thrown_out + picked_next_day = 45 :=
by sorry

end NUMINAMATH_CALUDE_chloe_carrot_count_l3008_300808


namespace NUMINAMATH_CALUDE_triangle_problem_l3008_300810

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : 2 * abc.b * Real.cos abc.A = abc.c * Real.cos abc.A + abc.a * Real.cos abc.C)
  (h2 : abc.a = Real.sqrt 7)
  (h3 : abc.b + abc.c = 4) :
  abc.A = π / 3 ∧ abc.b * abc.c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3008_300810


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l3008_300833

/-- Represents the scale of the blueprint in meters per inch -/
def scale : ℝ := 50

/-- Represents the length of the line segment on the blueprint in inches -/
def blueprint_length : ℝ := 7.5

/-- Represents the actual length in meters that the blueprint line segment represents -/
def actual_length : ℝ := blueprint_length * scale

theorem blueprint_to_actual_length : actual_length = 375 := by
  sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l3008_300833


namespace NUMINAMATH_CALUDE_code_cracker_combinations_l3008_300890

/-- The number of different colors of pegs in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots for pegs in the CodeCracker game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the CodeCracker game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes in the CodeCracker game is 7776 -/
theorem code_cracker_combinations : total_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_code_cracker_combinations_l3008_300890


namespace NUMINAMATH_CALUDE_intersection_chord_length_l3008_300843

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop :=
  2 * x - 2 * Real.sqrt 3 * y + 2 * Real.sqrt 3 - 1 = 0

/-- The circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2

/-- The theorem stating that the length of the chord formed by the intersection of line l and circle C is √5/2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l3008_300843


namespace NUMINAMATH_CALUDE_roundness_of_250000_l3008_300862

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 250,000 is 10. -/
theorem roundness_of_250000 : roundness 250000 = 10 := by sorry

end NUMINAMATH_CALUDE_roundness_of_250000_l3008_300862


namespace NUMINAMATH_CALUDE_total_sweaters_61_l3008_300816

def sweaters_fortnight (day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ) : Prop :=
  day1 = 8 ∧
  day2 = day1 + 2 ∧
  day3_4 = day2 - 4 ∧
  day5 = day3_4 ∧
  day6 = day1 / 2 ∧
  day7 = 0 ∧
  day8_10 = (day1 + day2 + day3_4 * 2 + day5 + day6) * 3 * 3 / (4 * 6) ∧
  day11 = day8_10 / 3 / 3 ∧
  day12_13 = day8_10 / 2 / 3 ∧
  day14 = 1

theorem total_sweaters_61 :
  ∀ day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ,
  sweaters_fortnight day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 →
  day1 + day2 + day3_4 * 2 + day5 + day6 + day7 + day8_10 + day11 + day12_13 * 2 + day14 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_total_sweaters_61_l3008_300816


namespace NUMINAMATH_CALUDE_hundred_with_five_threes_l3008_300811

-- Define a custom type for our arithmetic expressions
inductive Expr
  | const : ℕ → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.const n => if n = 3 then 1 else 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.const n => n
  | Expr.add e1 e2 => evaluate e1 + evaluate e2
  | Expr.sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_five_threes : 
  ∃ e : Expr, countThrees e = 5 ∧ evaluate e = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_five_threes_l3008_300811


namespace NUMINAMATH_CALUDE_jake_debt_work_hours_l3008_300881

def total_hours_worked (initial_debt_A initial_debt_B initial_debt_C : ℕ)
                       (payment_A payment_B payment_C : ℕ)
                       (rate_A rate_B rate_C : ℕ) : ℕ :=
  let remaining_debt_A := initial_debt_A - payment_A
  let remaining_debt_B := initial_debt_B - payment_B
  let remaining_debt_C := initial_debt_C - payment_C
  let hours_A := remaining_debt_A / rate_A
  let hours_B := remaining_debt_B / rate_B
  let hours_C := remaining_debt_C / rate_C
  hours_A + hours_B + hours_C

theorem jake_debt_work_hours :
  total_hours_worked 150 200 250 60 80 100 15 20 25 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jake_debt_work_hours_l3008_300881


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3008_300879

/-- A real-valued function that satisfies the given functional equation. -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2

/-- The theorem stating the only two functions that satisfy the functional equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ 
    (∀ x : ℝ, f x = x^2 / 2002) ∨ 
    (∀ x : ℝ, f x = x^2 / 2002 + 2 * x + 2002) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3008_300879


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3008_300824

theorem negation_of_existence_proposition :
  (¬ ∃ (n : ℕ), n^2 > 2^n) ↔ (∀ (n : ℕ), n^2 ≤ 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3008_300824


namespace NUMINAMATH_CALUDE_mess_expenditure_original_mess_expenditure_l3008_300893

/-- Calculates the original daily expenditure of a mess given initial conditions. -/
theorem mess_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let original_expenditure : ℕ := initial_students * (total_students * expense_increase) / (total_students * avg_decrease)
  original_expenditure

/-- Proves that the original daily expenditure of the mess was 420 given the specified conditions. -/
theorem original_mess_expenditure :
  mess_expenditure 35 7 42 1 = 420 := by
  sorry

end NUMINAMATH_CALUDE_mess_expenditure_original_mess_expenditure_l3008_300893


namespace NUMINAMATH_CALUDE_optimal_strategy_l3008_300821

/-- Represents the price of bananas on each day of Marina's trip. -/
def banana_prices : List ℝ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

/-- Represents the optimal buying strategy for bananas. -/
def buying_strategy : List ℕ := [1, 1, 1, 4, 0, 0, 1, 0, 1, 4, 1, 0, 0, 0, 3, 0, 0, 2, 0]

/-- The number of days in Marina's trip. -/
def trip_length : ℕ := 18

/-- The maximum number of days a banana can be eaten after purchase. -/
def max_banana_freshness : ℕ := 4

/-- Calculates the total cost of bananas based on a given buying strategy. -/
def total_cost (strategy : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) strategy banana_prices)

/-- Checks if a given buying strategy is valid according to the problem constraints. -/
def is_valid_strategy (strategy : List ℕ) : Prop :=
  strategy.length = trip_length + 1 ∧
  List.sum strategy = trip_length ∧
  ∀ i, i < trip_length → List.sum (List.take (min max_banana_freshness (trip_length - i)) (List.drop i strategy)) ≥ 1

/-- Theorem stating that the given buying strategy is optimal. -/
theorem optimal_strategy :
  is_valid_strategy buying_strategy ∧
  ∀ other_strategy, is_valid_strategy other_strategy →
    total_cost buying_strategy ≤ total_cost other_strategy :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l3008_300821


namespace NUMINAMATH_CALUDE_complex_vector_sum_l3008_300818

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) : 
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l3008_300818


namespace NUMINAMATH_CALUDE_magazine_page_height_l3008_300894

/-- Given advertising costs and dimensions, calculate the height of a magazine page -/
theorem magazine_page_height 
  (cost_per_sq_inch : ℝ) 
  (ad_fraction : ℝ) 
  (page_width : ℝ) 
  (total_cost : ℝ) 
  (h : cost_per_sq_inch = 8)
  (h₁ : ad_fraction = 1/2)
  (h₂ : page_width = 12)
  (h₃ : total_cost = 432) :
  ∃ (page_height : ℝ), 
    page_height = 9 ∧ 
    ad_fraction * page_height * page_width * cost_per_sq_inch = total_cost := by
  sorry

end NUMINAMATH_CALUDE_magazine_page_height_l3008_300894


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3008_300807

theorem trigonometric_expression_value :
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) /
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3008_300807


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_24_l3008_300800

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- A number is divisible by another number if the remainder of their division is zero -/
def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem smallest_four_digit_divisible_by_24 :
  is_four_digit 1104 ∧ 
  is_divisible_by 1104 24 ∧ 
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 24 → 1104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_24_l3008_300800


namespace NUMINAMATH_CALUDE_garden_length_perimeter_ratio_l3008_300897

/-- Proves that for a rectangular garden with length 24 feet and width 18 feet, 
    the ratio of its length to its perimeter is 2:7. -/
theorem garden_length_perimeter_ratio :
  let length : ℕ := 24
  let width : ℕ := 18
  let perimeter : ℕ := 2 * (length + width)
  (length : ℚ) / perimeter = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_perimeter_ratio_l3008_300897


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3008_300866

/-- Given four points in 3D space, this theorem proves that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (8, -5, 5) →
  B = (18, -15, 10) →
  C = (1, 5, -7) →
  D = (3, -3, 13) →
  ∃ t s : ℝ, 
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (1 + 2*s, 5 - 8*s, -7 + 20*s) ∧
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (-16, 7, -7) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3008_300866


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3008_300841

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 6)
  (sum_cube_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3008_300841


namespace NUMINAMATH_CALUDE_fractional_square_gt_floor_square_l3008_300813

theorem fractional_square_gt_floor_square (x : ℝ) (hx : x > 0) :
  (x ^ 2 - ⌊x ^ 2⌋) > (⌊x⌋ ^ 2) ↔ ∃ n : ℤ, Real.sqrt (n ^ 2 + 1) ≤ x ∧ x < n + 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_square_gt_floor_square_l3008_300813


namespace NUMINAMATH_CALUDE_class_average_problem_l3008_300822

theorem class_average_problem (class_size : ℝ) (h_positive : class_size > 0) :
  let group1_size := 0.2 * class_size
  let group2_size := 0.5 * class_size
  let group3_size := class_size - group1_size - group2_size
  let group1_avg := 80
  let group2_avg := 60
  let overall_avg := 58
  ∃ (group3_avg : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) / class_size = overall_avg ∧
    group3_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l3008_300822


namespace NUMINAMATH_CALUDE_meaningful_expression_l3008_300860

/-- The expression (x+3)/(x-1) + (x-2)^0 is meaningful if and only if x ≠ 1 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 1) + (x - 2)^0) ↔ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3008_300860


namespace NUMINAMATH_CALUDE_function_properties_l3008_300804

/-- A function type that represents the relationship between x and y --/
def Function := ℝ → ℝ

/-- The given values in the table --/
structure TableValues where
  y_neg5 : ℝ
  y_neg2 : ℝ
  y_2 : ℝ
  y_5 : ℝ

/-- Proposition: If y is an inverse proportion function of x, then 2m + 5n = 0 --/
def inverse_proportion_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ k : ℝ, ∀ x : ℝ, f x * x = k) →
  2 * tv.y_neg2 + 5 * tv.y_5 = 0

/-- Proposition: If y is a linear function of x, then n - m = 7 --/
def linear_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) →
  tv.y_5 - tv.y_neg2 = 7

/-- Proposition: If y is a quadratic function of x and the graph opens downwards, 
    then m > n is not necessarily true --/
def quadratic_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b c : ℝ, a < 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c) →
  ¬(tv.y_neg2 > tv.y_5)

/-- The main theorem that combines all three propositions --/
theorem function_properties (f : Function) (tv : TableValues) : 
  inverse_proportion_prop f tv ∧ 
  linear_function_prop f tv ∧ 
  quadratic_function_prop f tv := by sorry

end NUMINAMATH_CALUDE_function_properties_l3008_300804


namespace NUMINAMATH_CALUDE_complex_number_equality_l3008_300832

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  ∃ (r : ℝ), r > 0 ∧ z - (z - 6) / (z - 1) = r →
  z = 2 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3008_300832


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3008_300873

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3008_300873


namespace NUMINAMATH_CALUDE_magnitude_v_l3008_300899

/-- Given complex numbers u and v, prove that |v| = 5.2 under the given conditions -/
theorem magnitude_v (u v : ℂ) : 
  u * v = 24 - 10 * Complex.I → Complex.abs u = 5 → Complex.abs v = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l3008_300899


namespace NUMINAMATH_CALUDE_intersection_sum_l3008_300820

/-- 
Given two lines y = 2x + c and y = 4x + d that intersect at the point (8, 12),
prove that c + d = -24
-/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2*x + c) →
  (∀ x y : ℝ, y = 4*x + d) →
  12 = 2*8 + c →
  12 = 4*8 + d →
  c + d = -24 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3008_300820


namespace NUMINAMATH_CALUDE_jimmy_and_irene_payment_l3008_300870

/-- The amount paid by Jimmy and Irene for their clothing purchases with a senior citizen discount --/
def amountPaid (jimmyShorts : ℕ) (jimmyShortPrice : ℚ) (ireneShirts : ℕ) (ireneShirtPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := jimmyShorts * jimmyShortPrice + ireneShirts * ireneShirtPrice
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that Jimmy and Irene pay $117 for their purchases --/
theorem jimmy_and_irene_payment :
  amountPaid 3 15 5 17 10 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_and_irene_payment_l3008_300870


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l3008_300817

/-- Represents a date with a day and a month -/
structure Date where
  day : ℕ
  month : ℕ

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month (assuming non-leap year) -/
def daysInMonth (month : ℕ) : ℕ :=
  if month == 2 then 28 else if month ∈ [4, 6, 9, 11] then 30 else 31

/-- Returns the weekday of a given date, assuming February 1 is a Tuesday -/
def weekdayOfDate (d : Date) : Weekday :=
  sorry

/-- Returns true if the given date is a Tuesday -/
def isTuesday (d : Date) : Prop :=
  weekdayOfDate d = Weekday.Tuesday

/-- Returns true if the given date is a Terrific Tuesday (5th Tuesday of the month) -/
def isTerrificTuesday (d : Date) : Prop :=
  isTuesday d ∧ d.day > 28

/-- The main theorem: The first Terrific Tuesday after February 1 is March 29 -/
theorem first_terrific_tuesday : 
  ∃ (d : Date), d.month = 3 ∧ d.day = 29 ∧ isTerrificTuesday d ∧
  ∀ (d' : Date), (d'.month < 3 ∨ (d'.month = 3 ∧ d'.day < 29)) → ¬isTerrificTuesday d' :=
  sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l3008_300817


namespace NUMINAMATH_CALUDE_max_cows_is_correct_l3008_300889

/-- Represents the maximum number of cows a rancher can buy given specific constraints. -/
def max_cows : ℕ :=
  let budget : ℕ := 1300
  let steer_cost : ℕ := 30
  let cow_cost : ℕ := 33
  30

/-- Theorem stating that max_cows is indeed the maximum number of cows the rancher can buy. -/
theorem max_cows_is_correct :
  ∀ s c : ℕ,
  s > 0 →
  c > 0 →
  c > 2 * s →
  s * 30 + c * 33 ≤ 1300 →
  c ≤ max_cows :=
by sorry

#eval max_cows  -- Should output 30

end NUMINAMATH_CALUDE_max_cows_is_correct_l3008_300889


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l3008_300875

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_selection :
  let total_players : ℕ := 14
  let triplets : ℕ := 3
  let starters : ℕ := 6
  let non_triplet_players : ℕ := total_players - triplets
  let lineups_without_triplets : ℕ := choose non_triplet_players starters
  let lineups_with_one_triplet : ℕ := triplets * (choose non_triplet_players (starters - 1))
  lineups_without_triplets + lineups_with_one_triplet = 1848 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l3008_300875


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3008_300844

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3008_300844


namespace NUMINAMATH_CALUDE_percentage_difference_l3008_300885

theorem percentage_difference (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3008_300885


namespace NUMINAMATH_CALUDE_brians_largest_integer_l3008_300852

theorem brians_largest_integer (x : ℤ) : 
  (∀ y : ℤ, 10 ≤ 8*y - 70 ∧ 8*y - 70 ≤ 99 → y ≤ x) ↔ x = 21 :=
by sorry

end NUMINAMATH_CALUDE_brians_largest_integer_l3008_300852


namespace NUMINAMATH_CALUDE_smartphone_price_difference_l3008_300847

/-- Calculate the final price after applying a discount --/
def final_price (initial_price : ℚ) (discount_percent : ℚ) : ℚ :=
  initial_price * (1 - discount_percent / 100)

/-- The problem statement --/
theorem smartphone_price_difference :
  let store_a_initial_price : ℚ := 125
  let store_a_discount : ℚ := 8
  let store_b_initial_price : ℚ := 130
  let store_b_discount : ℚ := 10
  
  final_price store_b_initial_price store_b_discount -
  final_price store_a_initial_price store_a_discount = 2 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l3008_300847


namespace NUMINAMATH_CALUDE_vector_linear_combination_l3008_300882

/-- Given vectors a, b, and c in ℝ², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l3008_300882


namespace NUMINAMATH_CALUDE_division_theorem_l3008_300819

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 131 → divisor = 14 → remainder = 5 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3008_300819


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3008_300835

theorem basketball_free_throws 
  (two_point_shots : ℕ) 
  (three_point_shots : ℕ) 
  (free_throws : ℕ) : 
  (3 * three_point_shots = 2 * two_point_shots) → 
  (free_throws = two_point_shots + 1) → 
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 84) → 
  free_throws = 16 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3008_300835


namespace NUMINAMATH_CALUDE_karl_process_preserves_swapped_pairs_l3008_300809

/-- Represents a permutation of cards -/
def Permutation := List Nat

/-- Counts the number of swapped pairs (inversions) in a permutation -/
def countSwappedPairs (p : Permutation) : Nat :=
  sorry

/-- Karl's process of rearranging cards -/
def karlProcess (p : Permutation) : Permutation :=
  sorry

theorem karl_process_preserves_swapped_pairs (n : Nat) (initial : Permutation) :
  initial.length = n →
  initial.toFinset = Finset.range n →
  countSwappedPairs initial = countSwappedPairs (karlProcess initial) :=
sorry

end NUMINAMATH_CALUDE_karl_process_preserves_swapped_pairs_l3008_300809


namespace NUMINAMATH_CALUDE_can_cut_one_more_square_l3008_300857

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square that can be cut from the grid -/
structure Square :=
  (size : ℕ)

/-- Function to calculate the number of cells in a grid -/
def grid_cells (g : Grid) : ℕ := g.rows * g.cols

/-- Function to calculate the number of cells in a square -/
def square_cells (s : Square) : ℕ := s.size * s.size

/-- Function to calculate the number of cells remaining after cutting squares -/
def remaining_cells (g : Grid) (s : Square) (n : ℕ) : ℕ :=
  grid_cells g - n * square_cells s

/-- Theorem stating that after cutting 99 2x2 squares from a 29x29 grid, 
    at least one more 2x2 square can be cut -/
theorem can_cut_one_more_square (g : Grid) (s : Square) :
  g.rows = 29 → g.cols = 29 → s.size = 2 →
  ∃ (m : ℕ), m > 99 ∧ remaining_cells g s m ≥ square_cells s :=
by sorry

end NUMINAMATH_CALUDE_can_cut_one_more_square_l3008_300857


namespace NUMINAMATH_CALUDE_water_level_accurate_l3008_300842

/-- Represents the water level function for a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  water_level x = 6 + 0.3 * x ∧ water_level x ≥ 6 ∧ water_level x ≤ 7.5 := by
  sorry

end NUMINAMATH_CALUDE_water_level_accurate_l3008_300842


namespace NUMINAMATH_CALUDE_expression_equalities_l3008_300812

theorem expression_equalities : 
  (1 / (Real.sqrt 2 - 1) + Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 6) + Real.sqrt 8 = 4) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_equalities_l3008_300812


namespace NUMINAMATH_CALUDE_rubber_duck_race_l3008_300872

theorem rubber_duck_race (regular_price large_price large_count total : ℕ) :
  regular_price = 3 →
  large_price = 5 →
  large_count = 185 →
  total = 1588 →
  ∃ regular_count : ℕ, 
    regular_count * regular_price + large_count * large_price = total ∧
    regular_count = 221 := by
  sorry

end NUMINAMATH_CALUDE_rubber_duck_race_l3008_300872


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3008_300853

theorem complex_equation_solution :
  ∀ z : ℂ, z + 5 - 6*I = 3 + 4*I → z = -2 + 10*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3008_300853


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3008_300855

theorem inequality_holds_iff (x : ℝ) : 
  (∀ y : ℝ, y^2 - (5^x - 1)*(y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3008_300855


namespace NUMINAMATH_CALUDE_zeros_of_f_l3008_300856

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 2

-- State the theorem
theorem zeros_of_f : 
  {x : ℝ | f x = 0} = {2, -1} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3008_300856


namespace NUMINAMATH_CALUDE_probability_three_primes_l3008_300851

-- Define a 12-sided die
def Die := Finset (Fin 12)

-- Define the set of prime numbers on a 12-sided die
def PrimeNumbers : Finset (Fin 12) := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on a single die
def ProbPrime : ℚ := (PrimeNumbers.card : ℚ) / 12

-- Define the probability of not rolling a prime number on a single die
def ProbNotPrime : ℚ := 1 - ProbPrime

-- Define the number of dice
def NumDice : ℕ := 4

-- Define the number of dice that should show a prime
def NumPrimeDice : ℕ := 3

-- Theorem statement
theorem probability_three_primes :
  (NumDice.choose NumPrimeDice : ℚ) * ProbPrime ^ NumPrimeDice * ProbNotPrime ^ (NumDice - NumPrimeDice) = 875 / 5184 :=
sorry

end NUMINAMATH_CALUDE_probability_three_primes_l3008_300851


namespace NUMINAMATH_CALUDE_complex_inequality_l3008_300826

theorem complex_inequality : ∀ (i : ℂ), i^2 = -1 → Complex.abs (2 - i) > 2 * (i^4).re :=
fun i h =>
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l3008_300826


namespace NUMINAMATH_CALUDE_solve_run_problem_l3008_300874

def run_problem (speed2 : ℝ) : Prop :=
  let time1 : ℝ := 0.5
  let speed1 : ℝ := 10
  let time2 : ℝ := 0.5
  let time3 : ℝ := 0.25
  let speed3 : ℝ := 8
  let total_distance : ℝ := 17
  (speed1 * time1) + (speed2 * time2) + (speed3 * time3) = total_distance

theorem solve_run_problem : 
  ∃ (speed2 : ℝ), run_problem speed2 ∧ speed2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_run_problem_l3008_300874


namespace NUMINAMATH_CALUDE_alternating_walk_forms_cycle_l3008_300845

/-- Represents a direction of turn -/
inductive Direction
| Left
| Right

/-- Represents the island as a graph -/
structure Island where
  -- The set of vertices (junctions)
  vertices : Type
  -- The edges (roads) between vertices
  edges : vertices → vertices → Prop
  -- Every vertex has exactly three edges
  three_roads : ∀ v : vertices, ∃! (n : Nat), n = 3 ∧ (∃ (adjacent : Finset vertices), adjacent.card = n ∧ ∀ u ∈ adjacent, edges v u)

/-- Represents a walk on the island -/
def Walk (island : Island) : Type :=
  Nat → island.vertices × Direction

/-- A walk is alternating if it alternates between left and right turns -/
def IsAlternating (walk : Walk island) : Prop :=
  ∀ n : Nat, 
    (walk n).2 ≠ (walk (n + 1)).2

/-- The main theorem: any alternating walk on a finite island will eventually form a cycle -/
theorem alternating_walk_forms_cycle (island : Island) (walk : Walk island) 
    (finite_island : Finite island.vertices) (alternating : IsAlternating walk) : 
    ∃ (start finish : Nat), start < finish ∧ (walk start).1 = (walk finish).1 := by
  sorry


end NUMINAMATH_CALUDE_alternating_walk_forms_cycle_l3008_300845


namespace NUMINAMATH_CALUDE_find_a9_l3008_300827

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a9 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 5) (h3 : a 4 + a 8 = 22) : 
  ∃ x : ℝ, a 9 = x :=
sorry

end NUMINAMATH_CALUDE_find_a9_l3008_300827


namespace NUMINAMATH_CALUDE_product_mod_seven_l3008_300886

theorem product_mod_seven : (2031 * 2032 * 2033 * 2034) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3008_300886


namespace NUMINAMATH_CALUDE_father_age_three_times_marika_l3008_300848

/-- Marika's age in 2004 -/
def marika_age_2004 : ℕ := 8

/-- The year of reference -/
def reference_year : ℕ := 2004

/-- Marika's father's age in 2004 -/
def father_age_2004 : ℕ := 4 * marika_age_2004

/-- The year we're looking for -/
def target_year : ℕ := 2008

theorem father_age_three_times_marika : 
  ∃ (years_passed : ℕ), 
    target_year = reference_year + years_passed ∧
    father_age_2004 + years_passed = 3 * (marika_age_2004 + years_passed) := by
  sorry

end NUMINAMATH_CALUDE_father_age_three_times_marika_l3008_300848


namespace NUMINAMATH_CALUDE_probability_three_divisible_by_3_l3008_300865

/-- The probability of a single 12-sided die showing a number divisible by 3 -/
def p_divisible_by_3 : ℚ := 1 / 3

/-- The probability of a single 12-sided die not showing a number divisible by 3 -/
def p_not_divisible_by_3 : ℚ := 2 / 3

/-- The number of dice rolled -/
def total_dice : ℕ := 7

/-- The number of dice that should show a number divisible by 3 -/
def target_dice : ℕ := 3

/-- The theorem stating the probability of exactly three out of seven fair 12-sided dice 
    showing a number divisible by 3 -/
theorem probability_three_divisible_by_3 : 
  (Nat.choose total_dice target_dice : ℚ) * 
  p_divisible_by_3 ^ target_dice * 
  p_not_divisible_by_3 ^ (total_dice - target_dice) = 560 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_divisible_by_3_l3008_300865


namespace NUMINAMATH_CALUDE_function_max_min_sum_l3008_300814

theorem function_max_min_sum (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f := fun x : ℝ => (5 * a^x + 1) / (a^x - 1) + Real.log (Real.sqrt (1 + x^2) - x)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, N ≤ f x) ∧ M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l3008_300814


namespace NUMINAMATH_CALUDE_coin_toss_recurrence_l3008_300877

/-- The probability of having a group of length k or more in n tosses of a symmetric coin. -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem coin_toss_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_coin_toss_recurrence_l3008_300877


namespace NUMINAMATH_CALUDE_leilas_savings_leilas_savings_proof_l3008_300861

theorem leilas_savings : ℝ → Prop :=
  fun savings =>
    let makeup_fraction : ℝ := 3/5
    let sweater_fraction : ℝ := 1/3
    let sweater_cost : ℝ := 40
    let shoes_cost : ℝ := 30
    let remaining_fraction : ℝ := 1 - makeup_fraction - sweater_fraction
    
    (sweater_fraction * savings = sweater_cost) ∧
    (remaining_fraction * savings = shoes_cost) ∧
    (savings = 175)

-- The proof goes here
theorem leilas_savings_proof : ∃ (s : ℝ), leilas_savings s :=
sorry

end NUMINAMATH_CALUDE_leilas_savings_leilas_savings_proof_l3008_300861


namespace NUMINAMATH_CALUDE_square_IJKL_side_length_l3008_300871

-- Define the side lengths of squares ABCD and EFGH
def side_ABCD : ℝ := 3
def side_EFGH : ℝ := 9

-- Define the right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the arrangement of triangles
def triangle_arrangement (t : RightTriangle) : Prop :=
  t.leg1 - t.leg2 = side_ABCD ∧ t.leg1 + t.leg2 = side_EFGH

-- Theorem statement
theorem square_IJKL_side_length 
  (t : RightTriangle) 
  (h : triangle_arrangement t) : 
  t.hypotenuse = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_IJKL_side_length_l3008_300871


namespace NUMINAMATH_CALUDE_fraction_decimal_comparison_l3008_300896

theorem fraction_decimal_comparison : (1 : ℚ) / 4 - 0.250000025 = 1 / (4 * 10^7) := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_comparison_l3008_300896


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3008_300805

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def original_number : ℕ := 12910000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation :=
  { coefficient := 1.291
    exponent := 7
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3008_300805


namespace NUMINAMATH_CALUDE_tan_225_degrees_l3008_300840

theorem tan_225_degrees : Real.tan (225 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_225_degrees_l3008_300840


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3008_300880

theorem negative_fraction_comparison : -1/3 < -1/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3008_300880


namespace NUMINAMATH_CALUDE_min_value_theorem_l3008_300839

theorem min_value_theorem (C D x : ℝ) (hC : C > 0) (hD : D > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 + 3/2 ∧ ∀ y, y = C/(D-2) → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3008_300839


namespace NUMINAMATH_CALUDE_unique_exaggeration_combination_l3008_300868

/-- Represents the number of people who exaggerated the wolf's tail length --/
structure TailExaggeration where
  simple : Nat
  creative : Nat

/-- Calculates the final tail length given the number of simple and creative people --/
def finalTailLength (e : TailExaggeration) : Nat :=
  (2 ^ e.simple) * (3 ^ e.creative)

/-- Theorem stating that there is a unique combination of simple and creative people
    that results in a tail length of 864 meters --/
theorem unique_exaggeration_combination :
  ∃! e : TailExaggeration, finalTailLength e = 864 :=
sorry

end NUMINAMATH_CALUDE_unique_exaggeration_combination_l3008_300868


namespace NUMINAMATH_CALUDE_document_download_income_increase_sales_target_increase_basketball_success_rate_l3008_300836

-- Define percentages as real numbers between 0 and 1
def Percentage := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

-- 1. Document download percentages
theorem document_download (a b : Percentage) :
  (a.val + b.val = 1) → ((a.val = 0.58 ∧ b.val = 0.42) ∨ (a.val = 0.42 ∧ b.val = 0.58)) :=
sorry

-- 2. Xiao Ming's income increase
theorem income_increase (last_year current_year : ℝ) (h : current_year = 1.24 * last_year) :
  current_year > last_year :=
sorry

-- 3. Shopping mall sales target
theorem sales_target_increase (august_target september_target : ℝ) 
  (h : september_target = 1.5 * august_target) :
  september_target > 0.5 * august_target :=
sorry

-- 4. Luo Luo's basketball shot success rate
theorem basketball_success_rate (attempts successes : ℕ) :
  attempts = 5 ∧ successes = 5 → (successes : ℝ) / attempts = 1 :=
sorry

end NUMINAMATH_CALUDE_document_download_income_increase_sales_target_increase_basketball_success_rate_l3008_300836


namespace NUMINAMATH_CALUDE_sum_of_geometric_not_necessarily_geometric_l3008_300854

/-- A sequence is geometric if there exists a constant ratio between consecutive terms. -/
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

/-- The sum of two sequences -/
def SeqSum (s₁ s₂ : ℕ → ℝ) : ℕ → ℝ :=
  fun n ↦ s₁ n + s₂ n

/-- Theorem: The sum of two geometric sequences is not necessarily a geometric sequence -/
theorem sum_of_geometric_not_necessarily_geometric :
  ¬ ∀ (a b : ℕ → ℝ), IsGeometric a → IsGeometric b → IsGeometric (SeqSum a b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_geometric_not_necessarily_geometric_l3008_300854


namespace NUMINAMATH_CALUDE_no_real_roots_l3008_300803

theorem no_real_roots : ¬ ∃ (x : ℝ), x^2 + 3*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3008_300803


namespace NUMINAMATH_CALUDE_set_equality_l3008_300869

-- Define the sets A, B, and C as subsets of ℝ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}
def C : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- State the theorem
theorem set_equality : C = (Set.univ \ A) ∩ B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3008_300869


namespace NUMINAMATH_CALUDE_farthest_line_from_origin_l3008_300815

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The point A(1,2) -/
def pointA : Point := ⟨1, 2⟩

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The line x + 2y - 5 = 0 -/
def targetLine : Line := ⟨1, 2, -5⟩

theorem farthest_line_from_origin : 
  (pointOnLine pointA targetLine) ∧ 
  (∀ l : Line, pointOnLine pointA l → distancePointToLine origin targetLine ≥ distancePointToLine origin l) :=
sorry

end NUMINAMATH_CALUDE_farthest_line_from_origin_l3008_300815


namespace NUMINAMATH_CALUDE_choose_with_mandatory_l3008_300828

theorem choose_with_mandatory (n m k : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 1) :
  (Nat.choose (n - k) (m - k)) = 84 :=
sorry

end NUMINAMATH_CALUDE_choose_with_mandatory_l3008_300828


namespace NUMINAMATH_CALUDE_food_court_combinations_l3008_300859

/-- Represents the number of options for each meal component -/
structure MealOptions where
  entrees : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of meal combinations -/
def mealCombinations (options : MealOptions) : Nat :=
  options.entrees * options.drinks * options.desserts

/-- The given meal options in the food court -/
def foodCourtOptions : MealOptions :=
  { entrees := 4, drinks := 4, desserts := 2 }

/-- Theorem: The number of distinct meal combinations in the food court is 32 -/
theorem food_court_combinations :
  mealCombinations foodCourtOptions = 32 := by
  sorry

end NUMINAMATH_CALUDE_food_court_combinations_l3008_300859
