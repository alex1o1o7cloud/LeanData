import Mathlib

namespace NUMINAMATH_CALUDE_no_real_intersection_l450_45068

theorem no_real_intersection : ¬∃ x : ℝ, 3 * x^2 - 6 * x + 5 = 0 := by sorry

end NUMINAMATH_CALUDE_no_real_intersection_l450_45068


namespace NUMINAMATH_CALUDE_log_579_between_consecutive_integers_l450_45074

theorem log_579_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 579 / Real.log 10 ∧ Real.log 579 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_579_between_consecutive_integers_l450_45074


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l450_45072

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l450_45072


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l450_45088

theorem gain_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 1080 → 
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l450_45088


namespace NUMINAMATH_CALUDE_grid_arrangements_eq_six_l450_45023

/-- The number of ways to arrange 3 distinct elements in 3 positions -/
def arrangements_of_three : ℕ := 3 * 2 * 1

/-- The number of ways to arrange digits 1, 2, and 3 in three boxes of a 2x2 grid,
    with the fourth box fixed -/
def grid_arrangements : ℕ := arrangements_of_three

theorem grid_arrangements_eq_six :
  grid_arrangements = 6 := by sorry

end NUMINAMATH_CALUDE_grid_arrangements_eq_six_l450_45023


namespace NUMINAMATH_CALUDE_intersection_point_l450_45079

/-- The polar equation of curve l₁ -/
def l₁ (ρ θ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

/-- The parametric equations of line l₂ -/
def l₂ (x y t : ℝ) : Prop :=
  x = 1 - 2 * t ∧ y = 2 * t + 2

/-- The theorem stating that (1, 2) is the unique intersection point of l₁ and l₂ -/
theorem intersection_point :
  ∃! p : ℝ × ℝ, (∃ ρ θ : ℝ, l₁ ρ θ ∧ p.1 = ρ * Real.cos θ ∧ p.2 = ρ * Real.sin θ) ∧
                (∃ t : ℝ, l₂ p.1 p.2 t) ∧
                p = (1, 2) :=
  sorry


end NUMINAMATH_CALUDE_intersection_point_l450_45079


namespace NUMINAMATH_CALUDE_largest_circle_area_in_square_l450_45061

/-- The area of the largest circle inside a square of side length 70 cm -/
theorem largest_circle_area_in_square : 
  let square_side : ℝ := 70
  let circle_area : ℝ := Real.pi * (square_side / 2)^2
  circle_area = 1225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_area_in_square_l450_45061


namespace NUMINAMATH_CALUDE_cards_after_home_count_l450_45051

/-- The number of get well cards Mariela received in the hospital -/
def cards_in_hospital : ℕ := 403

/-- The total number of get well cards Mariela received -/
def total_cards : ℕ := 690

/-- The number of get well cards Mariela received after coming home -/
def cards_after_home : ℕ := total_cards - cards_in_hospital

theorem cards_after_home_count : cards_after_home = 287 := by
  sorry

end NUMINAMATH_CALUDE_cards_after_home_count_l450_45051


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l450_45032

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l450_45032


namespace NUMINAMATH_CALUDE_certain_number_problem_l450_45034

theorem certain_number_problem : ∃ x : ℝ, 0.12 * x - 0.1 * 14.2 = 1.484 ∧ x = 24.2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l450_45034


namespace NUMINAMATH_CALUDE_andre_flowers_l450_45002

/-- Given Rosa's initial and final number of flowers, prove that the number of flowers
    Andre gave to Rosa is the difference between the final and initial counts. -/
theorem andre_flowers (initial final andre : ℕ) 
  (h1 : initial = 67)
  (h2 : final = 90)
  (h3 : final = initial + andre) : 
  andre = final - initial := by
  sorry

end NUMINAMATH_CALUDE_andre_flowers_l450_45002


namespace NUMINAMATH_CALUDE_area_of_connected_paper_l450_45026

/-- The area of connected colored paper sheets -/
theorem area_of_connected_paper (n : ℕ) (side_length overlap : ℝ) :
  n > 0 →
  side_length > 0 →
  overlap ≥ 0 →
  overlap < side_length →
  let total_length := side_length + (n - 1 : ℝ) * (side_length - overlap)
  let area := total_length * side_length
  n = 6 ∧ side_length = 30 ∧ overlap = 7 →
  area = 4350 := by
  sorry

end NUMINAMATH_CALUDE_area_of_connected_paper_l450_45026


namespace NUMINAMATH_CALUDE_min_value_sum_l450_45050

theorem min_value_sum (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  1/a + 9/b + 16/c + 25/d + 36/e + 49/f ≥ 67.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l450_45050


namespace NUMINAMATH_CALUDE_opposite_face_is_B_l450_45025

-- Define the faces of the cube
inductive Face : Type
| X | A | B | C | D | E

-- Define the net structure
structure Net :=
  (faces : Finset Face)
  (center : Face)
  (surrounding : List Face)
  (adjacent_to_A : Face)
  (adjacent_to_D : Face)

-- Define the property of being opposite in a cube
def is_opposite (f1 f2 : Face) : Prop := sorry

-- Define the cube folding function
def fold_to_cube (n : Net) : Prop := sorry

-- Theorem statement
theorem opposite_face_is_B (n : Net) : 
  n.faces.card = 6 ∧ 
  n.center = Face.X ∧ 
  n.surrounding = [Face.A, Face.B, Face.D] ∧
  n.adjacent_to_A = Face.C ∧
  n.adjacent_to_D = Face.E ∧
  fold_to_cube n →
  is_opposite Face.X Face.B :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_B_l450_45025


namespace NUMINAMATH_CALUDE_rooster_weight_problem_l450_45083

theorem rooster_weight_problem (price_per_kg : ℝ) (weight_rooster1 : ℝ) (total_earnings : ℝ) :
  price_per_kg = 0.5 →
  weight_rooster1 = 30 →
  total_earnings = 35 →
  ∃ weight_rooster2 : ℝ,
    weight_rooster2 = 40 ∧
    total_earnings = price_per_kg * (weight_rooster1 + weight_rooster2) :=
by sorry

end NUMINAMATH_CALUDE_rooster_weight_problem_l450_45083


namespace NUMINAMATH_CALUDE_sin_A_value_side_c_value_l450_45015

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3 ∧ t.a = 6

-- Theorem 1
theorem sin_A_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 14) :
  Real.sin t.A = (3 / 14) * Real.sqrt 3 := by
  sorry

-- Theorem 2
theorem side_c_value (t : Triangle) (h : triangle_conditions t) (harea : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3) :
  t.c = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_side_c_value_l450_45015


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l450_45091

theorem average_of_six_numbers (numbers : List ℕ) :
  numbers = [12, 412, 812, 1212, 1612, 2012] →
  (numbers.sum / numbers.length : ℚ) = 1012 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l450_45091


namespace NUMINAMATH_CALUDE_intersection_midpoint_l450_45090

theorem intersection_midpoint (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 - k ∧ A.1^2 = A.2) ∧ 
    (B.2 = B.1 - k ∧ B.1^2 = B.2) ∧ 
    A ≠ B ∧
    (A.2 + B.2) / 2 = 1) →
  k = -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_l450_45090


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l450_45067

theorem arithmetic_geometric_harmonic_means (a b c : ℝ) :
  (a + b + c) / 3 = 9 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l450_45067


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l450_45082

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (drama_or_science : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : drama_or_science = 210) :
  drama_club + science_club - drama_or_science = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l450_45082


namespace NUMINAMATH_CALUDE_part_one_evaluation_part_two_evaluation_l450_45046

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part I
theorem part_one_evaluation : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Part II
theorem part_two_evaluation :
  lg 14 - 2 * lg (7/3) + lg 7 - lg 18 = 0 := by sorry

end NUMINAMATH_CALUDE_part_one_evaluation_part_two_evaluation_l450_45046


namespace NUMINAMATH_CALUDE_fraction_simplification_l450_45071

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + x*y) / (x*y) * y^2 / (x + y) = y :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l450_45071


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l450_45044

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 4 / x - 1 ≥ 3 :=
by sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ x + 4 / x - 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l450_45044


namespace NUMINAMATH_CALUDE_coach_sunscreen_fraction_is_correct_l450_45059

/-- The fraction of sunscreen transferred to a person's forehead when heading the ball -/
def transfer_fraction : ℚ := 1 / 10

/-- The fraction of sunscreen remaining on the ball after a header -/
def remaining_fraction : ℚ := 1 - transfer_fraction

/-- The sequence of headers -/
inductive Header
| C : Header  -- Coach
| A : Header  -- Player A
| B : Header  -- Player B

/-- The repeating sequence of headers -/
def header_sequence : List Header := [Header.C, Header.A, Header.C, Header.B]

/-- The fraction of original sunscreen on Coach C's forehead after infinite headers -/
def coach_sunscreen_fraction : ℚ := 10 / 19

/-- Theorem stating that the fraction of original sunscreen on Coach C's forehead
    after infinite headers is 10/19 -/
theorem coach_sunscreen_fraction_is_correct :
  coach_sunscreen_fraction = 
    (transfer_fraction * (1 / (1 - remaining_fraction^2))) := by sorry

end NUMINAMATH_CALUDE_coach_sunscreen_fraction_is_correct_l450_45059


namespace NUMINAMATH_CALUDE_min_value_of_z_l450_45099

/-- Given a set of constraints on x and y, prove that the minimum value of z = 3x - 4y is -1 -/
theorem min_value_of_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x - 4 * y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = 3 * x - 4 * y → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l450_45099


namespace NUMINAMATH_CALUDE_equations_represent_scenario_l450_45018

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℝ
  sheep : ℝ

/-- The system of equations representing the livestock values -/
def livestock_equations (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- The given scenario of livestock values -/
def livestock_scenario (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- Theorem stating that the system of equations correctly represents the scenario -/
theorem equations_represent_scenario :
  ∀ v : LivestockValue, livestock_equations v ↔ livestock_scenario v :=
by sorry

end NUMINAMATH_CALUDE_equations_represent_scenario_l450_45018


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l450_45019

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 :=
by sorry

theorem min_value_reciprocal_sum_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l450_45019


namespace NUMINAMATH_CALUDE_light_reflection_l450_45012

-- Define the points and lines
def M : ℝ × ℝ := (-1, 3)
def P : ℝ × ℝ := (1, 0)

def x_axis (x y : ℝ) : Prop := y = 0
def reflecting_line (x y : ℝ) : Prop := x + y = 4

-- Define the equations of l2 and l3
def l2_equation (x y : ℝ) : Prop := y = 3/2 * (x - 1)
def l3_equation (x y : ℝ) : Prop := 2*x - 3*y + 1 = 0

-- State the theorem
theorem light_reflection :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), (1 - t) * M.1 + t * P.1 = x ∧ (1 - t) * M.2 + t * P.2 = y) →  -- l1 passes through M and P
  (x_axis P.1 P.2) →  -- P is on x-axis
  (∃ (s : ℝ), reflecting_line (P.1 + s) (P.2 + s)) →  -- l2 intersects reflecting_line
  (l2_equation x y ∧ l3_equation x y) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l450_45012


namespace NUMINAMATH_CALUDE_angle_bisector_construction_with_two_sided_ruler_l450_45070

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (vertex : Point)
  (line1 : Line)
  (line2 : Line)

/-- A two-sided ruler -/
structure TwoSidedRuler :=
  (length : ℝ)

/-- Definition of an angle bisector -/
def is_angle_bisector (a : Angle) (l : Line) : Prop :=
  sorry

/-- Definition of an inaccessible point -/
def is_inaccessible (p : Point) : Prop :=
  sorry

/-- Main theorem: It is possible to construct the bisector of an angle with an inaccessible vertex using only a two-sided ruler -/
theorem angle_bisector_construction_with_two_sided_ruler 
  (a : Angle) (r : TwoSidedRuler) (h : is_inaccessible a.vertex) : 
  ∃ (l : Line), is_angle_bisector a l :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_construction_with_two_sided_ruler_l450_45070


namespace NUMINAMATH_CALUDE_circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l450_45013

/-- A circle defined by two inequalities -/
def SpecialCircle (x y : ℝ) : Prop :=
  (abs x + abs y ≤ (3/2) * Real.sqrt (2 * (x^2 + y^2))) ∧
  (Real.sqrt (2 * (x^2 + y^2)) ≤ 3 * max (abs x) (abs y))

/-- The circle is larger than a standard unit circle -/
theorem circle_larger_than_unit : ∃ (x y : ℝ), SpecialCircle x y ∧ x^2 + y^2 > 1 := by sorry

/-- The circle is contained within a square larger than the standard unit square -/
theorem circle_in_larger_square : ∃ (s : ℝ), s > 1 ∧ ∀ (x y : ℝ), SpecialCircle x y → max (abs x) (abs y) ≤ s := by sorry

/-- The circle may not touch all points of a diamond inscribed in the square -/
theorem circle_may_not_touch_diamond : ∃ (x y : ℝ), abs x + abs y = 1 ∧ ¬(SpecialCircle x y) := by sorry

end NUMINAMATH_CALUDE_circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l450_45013


namespace NUMINAMATH_CALUDE_triangle_properties_l450_45093

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.tan t.A * Real.tan t.B - Real.tan t.A - Real.tan t.B = Real.sqrt 3 ∧
  t.c = 2 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ 20 / 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l450_45093


namespace NUMINAMATH_CALUDE_abs_opposite_equal_l450_45094

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_opposite_equal_l450_45094


namespace NUMINAMATH_CALUDE_hexagon_enclosed_by_polygons_l450_45028

/-- A regular hexagon is enclosed by m regular n-sided polygons, where three polygons meet at each vertex of the hexagon. -/
theorem hexagon_enclosed_by_polygons (m : ℕ) (n : ℕ) : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_enclosed_by_polygons_l450_45028


namespace NUMINAMATH_CALUDE_tangent_line_equation_l450_45038

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := 3*x - 2*x^3

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 - 6*x^2

/-- The x-coordinate of the point of tangency -/
def a : ℝ := -1

/-- Theorem: The equation of the tangent line to y = 3x - 2x^3 at x = -1 is 3x + y + 4 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - f a) = f' a * (x - a) ↔ 3*x + y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l450_45038


namespace NUMINAMATH_CALUDE_association_and_likelihood_ratio_l450_45000

-- Define the contingency table
def excellent_math_excellent_chinese : ℕ := 45
def excellent_math_not_excellent_chinese : ℕ := 35
def not_excellent_math_excellent_chinese : ℕ := 45
def not_excellent_math_not_excellent_chinese : ℕ := 75

def total_sample_size : ℕ := 200

-- Define the chi-square test statistic
def chi_square_statistic : ℚ :=
  (total_sample_size * (excellent_math_excellent_chinese * not_excellent_math_not_excellent_chinese - 
  excellent_math_not_excellent_chinese * not_excellent_math_excellent_chinese)^2) / 
  ((excellent_math_excellent_chinese + excellent_math_not_excellent_chinese) * 
  (not_excellent_math_excellent_chinese + not_excellent_math_not_excellent_chinese) * 
  (excellent_math_excellent_chinese + not_excellent_math_excellent_chinese) * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

-- Define the critical value at α = 0.01
def critical_value : ℚ := 6635 / 1000

-- Define the likelihood ratio L(B|A)
def likelihood_ratio : ℚ := 
  (not_excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese)) / 
  (excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

theorem association_and_likelihood_ratio : 
  chi_square_statistic > critical_value ∧ likelihood_ratio = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_association_and_likelihood_ratio_l450_45000


namespace NUMINAMATH_CALUDE_like_terms_imply_exponent_one_l450_45064

theorem like_terms_imply_exponent_one (a b : ℝ) (m n x : ℕ) :
  (∃ (k : ℝ), 2 * a^x * b^(n+1) = k * (-3 * a * b^(2*m))) →
  (2*m - n)^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponent_one_l450_45064


namespace NUMINAMATH_CALUDE_exists_trapezoid_in_selected_vertices_l450_45098

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of selected vertices from a regular polygon -/
def SelectedVertices (n k : ℕ) (p : RegularPolygon n) :=
  {s : Finset (Fin n) // s.card = k}

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  (v1.1 - v2.1) * (v3.2 - v4.2) = (v1.2 - v2.2) * (v3.1 - v4.1) ∨
  (v1.1 - v3.1) * (v2.2 - v4.2) = (v1.2 - v3.2) * (v2.1 - v4.1) ∨
  (v1.1 - v4.1) * (v2.2 - v3.2) = (v1.2 - v4.2) * (v2.1 - v3.1)

/-- Main theorem: There exists a trapezoid among 64 selected vertices of a regular 1981-gon -/
theorem exists_trapezoid_in_selected_vertices 
  (p : RegularPolygon 1981) (s : SelectedVertices 1981 64 p) :
  ∃ (a b c d : Fin 1981), a ∈ s.val ∧ b ∈ s.val ∧ c ∈ s.val ∧ d ∈ s.val ∧
    IsTrapezoid (p.vertices a) (p.vertices b) (p.vertices c) (p.vertices d) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_trapezoid_in_selected_vertices_l450_45098


namespace NUMINAMATH_CALUDE_price_reduction_l450_45084

theorem price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 1 - 0.08
  let second_reduction := 1 - 0.10
  let final_price := original_price * first_reduction * second_reduction
  final_price / original_price = 0.828 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_l450_45084


namespace NUMINAMATH_CALUDE_math_club_minimum_size_l450_45035

theorem math_club_minimum_size :
  ∀ (boys girls : ℕ),
  (boys : ℝ) / (boys + girls : ℝ) > 0.6 →
  girls = 5 →
  boys + girls ≥ 13 ∧
  ∀ (total : ℕ), total < 13 →
    ¬(∃ (b g : ℕ), b + g = total ∧ (b : ℝ) / (total : ℝ) > 0.6 ∧ g = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_math_club_minimum_size_l450_45035


namespace NUMINAMATH_CALUDE_g_of_3_l450_45047

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem g_of_3 : g 3 = 113 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l450_45047


namespace NUMINAMATH_CALUDE_extremum_implies_f_of_2_l450_45008

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem extremum_implies_f_of_2 (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≥ f a b 1) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = -2 →
  f a b 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_f_of_2_l450_45008


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_k_l450_45004

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -2} := by sorry

-- Theorem for the second part of the problem
theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3) (-1), f x ≤ k * x + 1) ↔ k ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_k_l450_45004


namespace NUMINAMATH_CALUDE_gerald_pie_purchase_l450_45017

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

/-- The cost of the meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def remaining_pfennigs : ℕ := 7

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

theorem gerald_pie_purchase :
  initial_farthings - pie_cost * farthings_per_pfennig = remaining_pfennigs * farthings_per_pfennig :=
sorry

end NUMINAMATH_CALUDE_gerald_pie_purchase_l450_45017


namespace NUMINAMATH_CALUDE_year_2023_ad_representation_l450_45060

/-- Represents a year in the Gregorian calendar. -/
structure Year where
  value : Int
  is_ad : Bool

/-- Converts a Year to its numerical representation. -/
def Year.to_int (y : Year) : Int :=
  if y.is_ad then y.value else -y.value

/-- The year 500 BC -/
def year_500_bc : Year := { value := 500, is_ad := false }

/-- The year 2023 AD -/
def year_2023_ad : Year := { value := 2023, is_ad := true }

/-- Theorem stating that given 500 BC is denoted as -500, 2023 AD is denoted as +2023 -/
theorem year_2023_ad_representation :
  (year_500_bc.to_int = -500) → (year_2023_ad.to_int = 2023) := by
  sorry

end NUMINAMATH_CALUDE_year_2023_ad_representation_l450_45060


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_132_l450_45062

theorem complex_arithmetic_expression_equals_132 :
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_132_l450_45062


namespace NUMINAMATH_CALUDE_min_factors_to_remove_for_2_l450_45022

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def endsIn2 (n : ℕ) : Prop := n % 10 = 2

def factorsToRemove (n : ℕ) : ℕ := 
  let multiples_of_5 := n / 5
  multiples_of_5 + 1

theorem min_factors_to_remove_for_2 : 
  ∃ (removed : Finset ℕ), 
    removed.card = factorsToRemove 99 ∧ 
    endsIn2 ((factorial 99) / (removed.prod id)) ∧
    ∀ (other : Finset ℕ), other.card < factorsToRemove 99 → 
      ¬(endsIn2 ((factorial 99) / (other.prod id))) :=
sorry

end NUMINAMATH_CALUDE_min_factors_to_remove_for_2_l450_45022


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l450_45056

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  a * b ≠ 0 → 
  k ≥ 1 → 
  a = 2 * k * b → 
  (Nat.choose n 2 * (2 * b)^(n - 2) * (k - 1)^2 + 
   Nat.choose n 3 * (2 * b)^(n - 3) * (k - 1)^3 = 0) → 
  n = 3 * k - 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l450_45056


namespace NUMINAMATH_CALUDE_greatest_x_value_l450_45011

theorem greatest_x_value (x : ℤ) (h : (3.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (3.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l450_45011


namespace NUMINAMATH_CALUDE_f_simplification_and_result_l450_45053

noncomputable def f (α : ℝ) : ℝ :=
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi) ^ 2) /
  (Real.sin (α - Real.pi / 2) * Real.cos (Real.pi / 2 + α) * Real.tan (Real.pi - α))

theorem f_simplification_and_result (α : ℝ) :
  f α = Real.tan α ∧
  (f α = 2 → (3 * Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_f_simplification_and_result_l450_45053


namespace NUMINAMATH_CALUDE_max_intersections_circles_lines_l450_45065

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to count the number of intersection points -/
def count_intersections (circles : List Circle) (lines : List Line) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem max_intersections_circles_lines :
  ∀ (circles : List Circle) (lines : List Line),
    circles.length = 2 →
    lines.length = 3 →
    (∃ (l : Line) (c : Circle), l ∈ lines ∧ c ∈ circles ∧
      (∀ (c' : Circle) (l' : Line), c' ∈ circles → l' ∈ lines →
        c' ≠ c → l' ≠ l → ¬ (count_intersections [c'] [l'] > 0))) →
    count_intersections circles lines ≤ 12 :=
  sorry

end NUMINAMATH_CALUDE_max_intersections_circles_lines_l450_45065


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l450_45020

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Proves that the loss percentage for a radio with cost price 2400 and selling price 2100 is 12.5%. -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 2400
  let selling_price : ℚ := 2100
  loss_percentage cost_price selling_price = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l450_45020


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l450_45001

theorem intersection_nonempty_condition (m n : ℝ) :
  let A : Set ℝ := {x | m - 1 < x ∧ x < m + 1}
  let B : Set ℝ := {x | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ (2 < m + n ∧ m + n < 5) := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l450_45001


namespace NUMINAMATH_CALUDE_smallest_three_digit_candy_count_l450_45006

theorem smallest_three_digit_candy_count (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) →  -- n is a three-digit number
  ((n + 7) % 9 = 0) →     -- if Alicia gains 7 candies, she'll have a multiple of 9
  ((n - 9) % 7 = 0) →     -- if Alicia loses 9 candies, she'll have a multiple of 7
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) → False) →  -- n is the smallest such number
  n = 101 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_candy_count_l450_45006


namespace NUMINAMATH_CALUDE_square_of_negative_cube_l450_45048

theorem square_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_cube_l450_45048


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l450_45027

/-- A right triangle with sides 6, 8, and 10 (hypotenuse) -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1 = 6 ∧ side2 = 8 ∧ hypotenuse = 10

/-- A square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x < t.side1 ∧ x < t.side2

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y < t.side1 ∧ y < t.side2

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_at_right_angle t x)
  (h2 : inscribed_square_on_hypotenuse t y) :
  x / y = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l450_45027


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l450_45054

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l450_45054


namespace NUMINAMATH_CALUDE_total_ways_eq_7464_l450_45003

def num_oreo_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4
def total_products : ℕ := 5

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def alpha_choices (k : ℕ) : ℕ := ways_to_choose (num_oreo_flavors + num_milk_flavors) k

def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then num_oreo_flavors
  else if k = 2 then ways_to_choose num_oreo_flavors 2 + num_oreo_flavors
  else if k = 3 then ways_to_choose num_oreo_flavors 3 + num_oreo_flavors * (num_oreo_flavors - 1) + num_oreo_flavors
  else if k = 4 then ways_to_choose num_oreo_flavors 4 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + num_oreo_flavors
  else ways_to_choose num_oreo_flavors 5 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + 
       num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 2 + num_oreo_flavors

def total_ways : ℕ := 
  (Finset.range (total_products + 1)).sum (λ k => alpha_choices k * beta_choices (total_products - k))

theorem total_ways_eq_7464 : total_ways = 7464 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_7464_l450_45003


namespace NUMINAMATH_CALUDE_beth_has_winning_strategy_l450_45042

/-- Represents the state of a wall of bricks -/
structure Wall :=
  (bricks : ℕ)

/-- Represents the game state with multiple walls -/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall -/
def nimValue (w : Wall) : ℕ :=
  sorry

/-- Calculates the combined nim-value of a game state -/
def combinedNimValue (gs : GameState) : ℕ :=
  sorry

/-- Determines if a given game state is a winning position for the current player -/
def isWinningPosition (gs : GameState) : Prop :=
  combinedNimValue gs ≠ 0

/-- The initial game state -/
def initialState : GameState :=
  { walls := [{ bricks := 7 }, { bricks := 3 }, { bricks := 2 }] }

theorem beth_has_winning_strategy :
  ¬ isWinningPosition initialState :=
sorry

end NUMINAMATH_CALUDE_beth_has_winning_strategy_l450_45042


namespace NUMINAMATH_CALUDE_savings_amount_correct_l450_45041

def calculate_savings (lightweight_price medium_price heavyweight_price : ℚ)
  (home_lightweight grandparents_medium_factor neighbor_heavyweight : ℕ)
  (dad_total dad_lightweight_percent dad_medium_percent dad_heavyweight_percent : ℚ) : ℚ :=
  let home_medium := home_lightweight * grandparents_medium_factor
  let dad_lightweight := dad_total * dad_lightweight_percent
  let dad_medium := dad_total * dad_medium_percent
  let dad_heavyweight := dad_total * dad_heavyweight_percent
  let total_amount := 
    lightweight_price * (home_lightweight + dad_lightweight) +
    medium_price * (home_medium + dad_medium) +
    heavyweight_price * (neighbor_heavyweight + dad_heavyweight)
  total_amount / 2

theorem savings_amount_correct :
  calculate_savings 0.15 0.25 0.35 12 3 46 250 0.5 0.3 0.2 = 41.45 :=
by sorry

end NUMINAMATH_CALUDE_savings_amount_correct_l450_45041


namespace NUMINAMATH_CALUDE_not_perfect_square_special_number_l450_45055

/-- A 100-digit number with all digits as fives except one is not a perfect square. -/
theorem not_perfect_square_special_number : 
  ∀ n : ℕ, 
  (n ≥ 10^99 ∧ n < 10^100) →  -- 100-digit number
  (∃! d : ℕ, d < 10 ∧ d ≠ 5 ∧ 
    ∀ i : ℕ, i < 100 → 
      (n / 10^i) % 10 = if (n / 10^i) % 10 = d then d else 5) →  -- All digits are fives except one
  ¬∃ m : ℕ, n = m^2 :=  -- Not a perfect square
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_special_number_l450_45055


namespace NUMINAMATH_CALUDE_repacking_books_leftover_l450_45080

/-- The number of books left over when repacking from boxes of 42 to boxes of 45 -/
def books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ) : ℕ :=
  (initial_boxes * books_per_initial_box) % books_per_new_box

/-- Theorem stating that repacking 1573 boxes of 42 books into boxes of 45 books leaves 6 books over -/
theorem repacking_books_leftover :
  books_left_over 1573 42 45 = 6 := by
  sorry

#eval books_left_over 1573 42 45

end NUMINAMATH_CALUDE_repacking_books_leftover_l450_45080


namespace NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l450_45031

theorem sufficient_to_necessary_contrapositive (A B : Prop) 
  (h : A → B) : ¬B → ¬A := by sorry

end NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l450_45031


namespace NUMINAMATH_CALUDE_evaluate_expression_l450_45010

theorem evaluate_expression : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l450_45010


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l450_45037

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Main theorem about the quadratic equation x^2 + mx + m - 2 = 0 -/
theorem quadratic_equation_properties (p : QuadraticParams) :
  -- If -2 is one root, the other root is 0
  (∃ (r : QuadraticRoots), r.x₁ = -2 ∧ r.x₂ = 0 ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- The equation always has two distinct real roots
  (∀ (x : ℝ), x^2 + p.m * x + p.m - 2 = 0 → 
    ∃ (r : QuadraticRoots), r.x₁ ≠ r.x₂ ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- If x₁^2 + x₂^2 + m(x₁ + x₂) = m^2 + 1, then m = -3 or m = 1
  (∀ (r : QuadraticRoots), 
    r.x₁^2 + r.x₂^2 + p.m * (r.x₁ + r.x₂) = p.m^2 + 1 →
    p.m = -3 ∨ p.m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l450_45037


namespace NUMINAMATH_CALUDE_polynomial_with_three_equal_roots_l450_45077

theorem polynomial_with_three_equal_roots (a b : ℤ) : 
  (∃ r : ℤ, (∀ x : ℝ, x^4 + x^3 - 18*x^2 + a*x + b = 0 ↔ 
    (x = r ∨ x = r ∨ x = r ∨ x = ((-1 : ℝ) - 3*r)))) → 
  (a = -52 ∧ b = -40) := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_three_equal_roots_l450_45077


namespace NUMINAMATH_CALUDE_three_digit_number_property_l450_45052

theorem three_digit_number_property (a b c : ℕ) : 
  a ≠ 0 → 
  a < 10 → b < 10 → c < 10 →
  10 * b + c = 8 * a →
  10 * a + b = 8 * c →
  (10 * a + c) / b = 17 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l450_45052


namespace NUMINAMATH_CALUDE_emilys_necklaces_l450_45078

/-- Emily's necklace-making problem -/
theorem emilys_necklaces (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) 
  (h1 : necklaces = 26)
  (h2 : beads_per_necklace = 2)
  (h3 : total_beads = 52)
  (h4 : necklaces * beads_per_necklace = total_beads) :
  necklaces = total_beads / beads_per_necklace :=
by sorry

end NUMINAMATH_CALUDE_emilys_necklaces_l450_45078


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l450_45036

def cannot_form_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem min_cut_length_for_non_triangle : ∃ (x : ℝ),
  (x > 0) ∧
  (cannot_form_triangle (9 - x) (12 - x) (15 - x)) ∧
  (∀ y, 0 < y ∧ y < x → ¬(cannot_form_triangle (9 - y) (12 - y) (15 - y))) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l450_45036


namespace NUMINAMATH_CALUDE_bridge_length_l450_45085

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l450_45085


namespace NUMINAMATH_CALUDE_single_color_subgraph_exists_l450_45045

/-- A graph where each pair of vertices is connected by exactly one of two types of edges -/
structure TwoColorGraph (α : Type*) where
  vertices : Set α
  edge_type1 : α → α → Prop
  edge_type2 : α → α → Prop
  edge_exists : ∀ (v w : α), v ∈ vertices → w ∈ vertices → v ≠ w → 
    (edge_type1 v w ∧ ¬edge_type2 v w) ∨ (edge_type2 v w ∧ ¬edge_type1 v w)

/-- A subgraph that includes all vertices and uses only one type of edge -/
def SingleColorSubgraph {α : Type*} (G : TwoColorGraph α) :=
  {H : Set (α × α) // 
    (∀ v ∈ G.vertices, ∃ w, (v, w) ∈ H ∨ (w, v) ∈ H) ∧
    (∀ (v w : α), (v, w) ∈ H → G.edge_type1 v w) ∨
    (∀ (v w : α), (v, w) ∈ H → G.edge_type2 v w)}

/-- The main theorem: there always exists a single-color subgraph -/
theorem single_color_subgraph_exists {α : Type*} (G : TwoColorGraph α) :
  Nonempty (SingleColorSubgraph G) := by
  sorry

end NUMINAMATH_CALUDE_single_color_subgraph_exists_l450_45045


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l450_45030

/-- Given three numbers 2, m, 8 forming a geometric sequence, 
    the eccentricity of the conic section x^2/m + y^2/2 = 1 is either √2/2 or √3 -/
theorem conic_section_eccentricity (m : ℝ) :
  (2 * m = m * 8) →
  let e := if m > 0 then Real.sqrt (1 - 2 / m) else Real.sqrt (1 + m / 2)
  e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l450_45030


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l450_45043

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 15/8) →
  (a 6 * a 7 = -9/8) →
  (1 / a 5 + 1 / a 6 + 1 / a 7 + 1 / a 8 = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l450_45043


namespace NUMINAMATH_CALUDE_age_difference_l450_45089

/-- Given three people a, b, and c, with their ages satisfying certain conditions,
    prove that a is 2 years older than b. -/
theorem age_difference (a b c : ℕ) : 
  b = 28 →                  -- b is 28 years old
  b = 2 * c →               -- b is twice as old as c
  a + b + c = 72 →          -- The total of the ages of a, b, and c is 72
  a = b + 2 :=              -- a is 2 years older than b
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l450_45089


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l450_45016

/-- The repeating decimal 0.215215215... -/
def repeating_decimal : ℚ := 0.215215215

/-- The fraction 215/999 -/
def fraction : ℚ := 215 / 999

/-- Theorem stating that the repeating decimal 0.215215215... is equal to the fraction 215/999 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l450_45016


namespace NUMINAMATH_CALUDE_unique_prime_ending_l450_45058

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def number (A : ℕ) : ℕ := 130400 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_ending_l450_45058


namespace NUMINAMATH_CALUDE_ratio_evaluation_l450_45087

theorem ratio_evaluation : 
  (5^3003 * 2^3005) / 10^3004 = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l450_45087


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l450_45057

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |x - 4| ≤ 2}
def B : Set ℝ := {x : ℝ | (5 - x) / (x + 1) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: A ∩ (Uᶜ B) = [5,6]
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 5 6 := by sorry

-- Theorem 2: If A ∩ C ≠ ∅, then a ∈ (2, +∞)
theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a ∈ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l450_45057


namespace NUMINAMATH_CALUDE_trent_travel_distance_l450_45040

/-- The total distance Trent traveled -/
def total_distance (house_to_bus bus_to_library : ℕ) : ℕ :=
  2 * (house_to_bus + bus_to_library)

/-- Theorem stating that Trent's total travel distance is 22 blocks -/
theorem trent_travel_distance :
  ∃ (house_to_bus bus_to_library : ℕ),
    house_to_bus = 4 ∧
    bus_to_library = 7 ∧
    total_distance house_to_bus bus_to_library = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_trent_travel_distance_l450_45040


namespace NUMINAMATH_CALUDE_common_chord_length_is_10_l450_45076

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

/-- The length of the common chord of two intersecting circles -/
def common_chord_length : ℝ := 10

/-- Theorem: The length of the common chord of the given intersecting circles is 10 -/
theorem common_chord_length_is_10 :
  ∃ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_is_10_l450_45076


namespace NUMINAMATH_CALUDE_two_word_sentences_count_correct_count_l450_45005

def word : String := "YARIŞMA"

theorem two_word_sentences_count : ℕ :=
  let n : ℕ := word.length
  let repeated_letter_count : ℕ := 2  -- 'A' appears twice
  let permutations : ℕ := n.factorial / repeated_letter_count.factorial
  let space_positions : ℕ := n + 1
  permutations * space_positions

theorem correct_count : two_word_sentences_count = 20160 := by
  sorry

end NUMINAMATH_CALUDE_two_word_sentences_count_correct_count_l450_45005


namespace NUMINAMATH_CALUDE_cascade_properties_l450_45069

/-- A cascade generated by a natural number r -/
def Cascade (r : ℕ) : Finset ℕ :=
  Finset.image (λ i => i * r) (Finset.range 12)

/-- The property that a pair of natural numbers belongs to exactly six cascades -/
def BelongsToSixCascades (a b : ℕ) : Prop :=
  ∃ (cascades : Finset ℕ), cascades.card = 6 ∧
    ∀ r ∈ cascades, a ∈ Cascade r ∧ b ∈ Cascade r

/-- A coloring function from natural numbers to 12 colors -/
def ColoringFunction := ℕ → Fin 12

/-- The property that a coloring function assigns different colors to all elements in any cascade -/
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ r : ℕ, ∀ i j : Fin 12, i ≠ j → f (r * (i.val + 1)) ≠ f (r * (j.val + 1))

theorem cascade_properties :
  (∃ a b : ℕ, BelongsToSixCascades a b) ∧
  (∃ f : ColoringFunction, ValidColoring f) := by sorry

end NUMINAMATH_CALUDE_cascade_properties_l450_45069


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l450_45073

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_sales_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l450_45073


namespace NUMINAMATH_CALUDE_min_digits_theorem_l450_45029

/-- The minimum number of digits to the right of the decimal point needed to express the given fraction as a decimal -/
def min_decimal_digits : ℕ := 30

/-- The numerator of the fraction -/
def numerator : ℕ := 987654321

/-- The denominator of the fraction -/
def denominator : ℕ := 2^30 * 5^6

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction numerator/denominator as a decimal is min_decimal_digits -/
theorem min_digits_theorem :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, m * denominator ≠ numerator * 10^n) ∧
  (∃ m : ℕ, m * denominator = numerator * 10^min_decimal_digits) :=
sorry

end NUMINAMATH_CALUDE_min_digits_theorem_l450_45029


namespace NUMINAMATH_CALUDE_circle_tangent_and_intersections_l450_45039

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*Real.sqrt 3*y + 3 = 0

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define line l₁
def l₁ (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define line l₂
def l₂ (x : ℝ) : Prop := x = 1

-- Define the condition for points R, M, and N
def RMN_condition (R M N : ℝ × ℝ) : Prop :=
  C R.1 R.2 → l₂ M.1 → l₂ N.1 → 
  (R.1 - N.1)^2 + (R.2 - N.2)^2 = 3 * ((R.1 - M.1)^2 + (R.2 - M.2)^2)

-- Main theorem
theorem circle_tangent_and_intersections :
  -- Length of tangent line from A to C is √6
  (∃ T : ℝ × ℝ, C T.1 T.2 ∧ (T.1 - A.1)^2 + (T.2 - A.2)^2 = 6) ∧
  -- Slope k of l₁ satisfies k = √3/3 or k = 11√3/15
  (∃ k : ℝ, (k = Real.sqrt 3 / 3 ∨ k = 11 * Real.sqrt 3 / 15) ∧
    ∃ P Q : ℝ × ℝ, C P.1 P.2 ∧ C Q.1 Q.2 ∧ l₁ k P.1 P.2 ∧ l₁ k Q.1 Q.2 ∧
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2) ∧
  -- Coordinates of M and N
  ((∃ M N : ℝ × ℝ, M = (1, 4 * Real.sqrt 3 / 3) ∧ N = (1, 2 * Real.sqrt 3) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N) ∨
   (∃ M N : ℝ × ℝ, M = (1, 2 * Real.sqrt 3 / 3) ∧ N = (1, 0) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_intersections_l450_45039


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l450_45021

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x ≤ 1} ∪ {x | x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l450_45021


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l450_45096

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((7 : ℚ) / 12) = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l450_45096


namespace NUMINAMATH_CALUDE_sqrt_cube_equivalence_l450_45066

theorem sqrt_cube_equivalence (x : ℝ) (h : x ≤ 0) :
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_equivalence_l450_45066


namespace NUMINAMATH_CALUDE_largest_k_inequality_l450_45081

theorem largest_k_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 ∧ 
  ∀ k : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (a * b + 1) * (b + 1) ≥ k * a * b^2) → k ≤ 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l450_45081


namespace NUMINAMATH_CALUDE_negation_equivalence_l450_45007

theorem negation_equivalence (S : Set ℕ) :
  (¬ ∀ x ∈ S, x^2 ≠ 4) ↔ (∃ x ∈ S, x^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l450_45007


namespace NUMINAMATH_CALUDE_train_passing_time_l450_45097

/-- Time for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 110 →
  train_speed = 90 * (1000 / 3600) →
  man_speed = 9 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 4 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l450_45097


namespace NUMINAMATH_CALUDE_sequence_length_problem_solution_l450_45014

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ + d * i)

theorem sequence_length (a₁ a_n d : ℤ) (h : d ≠ 0) :
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           a₁ = a_n + (n - 1) * d :=
by sorry

theorem problem_solution :
  let a₁ := 160
  let a_n := 28
  let d := -4
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           n = 34 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_problem_solution_l450_45014


namespace NUMINAMATH_CALUDE_toy_sale_analysis_l450_45033

-- Define the cost price
def cost_price : ℝ := 20

-- Define the maximum profit percentage
def max_profit_percentage : ℝ := 0.3

-- Define the linear relationship between weekly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -10 * x + 300

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem statement
theorem toy_sale_analysis :
  -- Part 1: Verify the linear relationship
  (sales_volume 22 = 80 ∧ sales_volume 24 = 60) ∧
  -- Part 2: Verify the selling price for 210 yuan profit
  (∃ x : ℝ, x ≤ cost_price * (1 + max_profit_percentage) ∧ profit x = 210 ∧ x = 23) ∧
  -- Part 3: Verify the maximum profit
  (∃ x : ℝ, x = 25 ∧ profit x = 250 ∧ ∀ y : ℝ, profit y ≤ profit x) := by
  sorry


end NUMINAMATH_CALUDE_toy_sale_analysis_l450_45033


namespace NUMINAMATH_CALUDE_max_volume_cube_l450_45024

/-- Given a constant sum of edges, the volume of a rectangular prism is maximized when it is a cube -/
theorem max_volume_cube (s : ℝ) (hs : s > 0) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 * s →
  a * b * c ≤ s^3 ∧ (a * b * c = s^3 ↔ a = s ∧ b = s ∧ c = s) :=
by sorry

end NUMINAMATH_CALUDE_max_volume_cube_l450_45024


namespace NUMINAMATH_CALUDE_flashlight_visibility_difference_l450_45092

/-- Flashlight visibility problem -/
theorem flashlight_visibility_difference (veronica_visibility : ℝ) :
  veronica_visibility = 1000 →
  let freddie_visibility := 3 * veronica_visibility
  let velma_visibility := 5 * freddie_visibility - 2000
  let daphne_visibility := (veronica_visibility + freddie_visibility + velma_visibility) / 3
  let total_visibility := veronica_visibility + freddie_visibility + velma_visibility + daphne_visibility
  total_visibility = 40000 →
  ∃ ε > 0, |velma_visibility - daphne_visibility - 7666.67| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_flashlight_visibility_difference_l450_45092


namespace NUMINAMATH_CALUDE_ellipse_and_point_theorem_l450_45086

-- Define the ellipse M
structure Ellipse :=
  (a b : ℝ)
  (center_x center_y : ℝ)
  (eccentricity : ℝ)

-- Define the line l
structure Line :=
  (m : ℝ)
  (b : ℝ)

-- Define a point
structure Point :=
  (x y : ℝ)

-- Define the problem
theorem ellipse_and_point_theorem 
  (M : Ellipse)
  (l : Line)
  (N : ℝ → Point) :
  M.center_x = 0 ∧ 
  M.center_y = 0 ∧ 
  M.a = 2 ∧ 
  M.eccentricity = 1/2 ∧
  l.m ≠ 0 ∧
  (∀ t, N t = Point.mk t 0) →
  (M.a^2 * M.b^2 = 12 ∧ 
   (∀ t, (0 < t ∧ t < 1/4) ↔ 
     ∃ A B : Point, 
       A.x^2 / 4 + A.y^2 / 3 = 1 ∧
       B.x^2 / 4 + B.y^2 / 3 = 1 ∧
       A.x = l.m * A.y + l.b ∧
       B.x = l.m * B.y + l.b ∧
       ((A.x - t)^2 + A.y^2 = (B.x - t)^2 + B.y^2) ∧
       ((A.x - (N t).x) * (B.y - A.y) = (A.y - (N t).y) * (B.x - A.x)))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_point_theorem_l450_45086


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l450_45049

-- Factorization of -4a²x + 12ax - 9x
theorem factorization_1 (a x : ℝ) : -4 * a^2 * x + 12 * a * x - 9 * x = -x * (2*a - 3)^2 := by
  sorry

-- Factorization of (2x + y)² - (x + 2y)²
theorem factorization_2 (x y : ℝ) : (2*x + y)^2 - (x + 2*y)^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l450_45049


namespace NUMINAMATH_CALUDE_count_five_or_six_base_eight_l450_45075

/-- 
Given a positive integer n and a base b, returns true if n (when expressed in base b)
contains at least one digit that is either 5 or 6.
-/
def contains_five_or_six (n : ℕ+) (b : ℕ) : Prop := sorry

/-- 
Counts the number of positive integers up to n (inclusive) that contain
at least one 5 or 6 when expressed in base b.
-/
def count_with_five_or_six (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 
Theorem: The number of integers from 1 to 256 (inclusive) in base 8
that contain at least one 5 or 6 digit is equal to 220.
-/
theorem count_five_or_six_base_eight : 
  count_with_five_or_six 256 8 = 220 := by sorry

end NUMINAMATH_CALUDE_count_five_or_six_base_eight_l450_45075


namespace NUMINAMATH_CALUDE_power_mod_seven_l450_45009

theorem power_mod_seven : 5^1986 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l450_45009


namespace NUMINAMATH_CALUDE_min_value_of_f_l450_45095

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l450_45095


namespace NUMINAMATH_CALUDE_probability_calculation_l450_45063

def total_silverware : ℕ := 8 + 7 + 5

def probability_2forks_1spoon_1knife (forks spoons knives total : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose forks 2 * Nat.choose spoons 1 * Nat.choose knives 1
  let total_outcomes := Nat.choose total 4
  favorable_outcomes / total_outcomes

theorem probability_calculation :
  probability_2forks_1spoon_1knife 8 7 5 total_silverware = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l450_45063
