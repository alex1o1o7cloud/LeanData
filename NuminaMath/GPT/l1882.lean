import Mathlib

namespace NUMINAMATH_GPT_truck_distance_and_efficiency_l1882_188288

theorem truck_distance_and_efficiency (m d g1 g2 : ℕ) (h1 : d = 300) (h2 : g1 = 10) (h3 : g2 = 15) :
  (d * (g2 / g1) = 450) ∧ (d / g1 = 30) :=
by
  sorry

end NUMINAMATH_GPT_truck_distance_and_efficiency_l1882_188288


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1882_188225

theorem solution_set_of_inequality (x : ℝ) (h : |x - 1| < 1) : 0 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1882_188225


namespace NUMINAMATH_GPT_train_length_eq_1800_l1882_188232

theorem train_length_eq_1800 (speed_kmh : ℕ) (time_sec : ℕ) (distance : ℕ) (L : ℕ)
  (h_speed : speed_kmh = 216)
  (h_time : time_sec = 60)
  (h_distance : distance = 60 * time_sec)
  (h_total_distance : distance = 2 * L) :
  L = 1800 := by
  sorry

end NUMINAMATH_GPT_train_length_eq_1800_l1882_188232


namespace NUMINAMATH_GPT_mutually_exclusive_any_two_l1882_188253

variables (A B C : Prop)
axiom all_not_defective : A
axiom all_defective : B
axiom not_all_defective : C

theorem mutually_exclusive_any_two :
  (¬(A ∧ B)) ∧ (¬(A ∧ C)) ∧ (¬(B ∧ C)) :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_any_two_l1882_188253


namespace NUMINAMATH_GPT_range_of_m_l1882_188276

theorem range_of_m (m y1 y2 k : ℝ) (h1 : y1 = -2 * (m - 2) ^ 2 + k) (h2 : y2 = -2 * (m - 1) ^ 2 + k) (h3 : y1 > y2) : m > 3 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1882_188276


namespace NUMINAMATH_GPT_total_boxes_sold_is_189_l1882_188218

-- Define the conditions
def boxes_sold_friday : ℕ := 40
def boxes_sold_saturday := 2 * boxes_sold_friday - 10
def boxes_sold_sunday := boxes_sold_saturday / 2
def boxes_sold_monday := boxes_sold_sunday + (boxes_sold_sunday / 4)

-- Define the total boxes sold over the four days
def total_boxes_sold := boxes_sold_friday + boxes_sold_saturday + boxes_sold_sunday + boxes_sold_monday

-- Theorem to prove the total number of boxes sold is 189
theorem total_boxes_sold_is_189 : total_boxes_sold = 189 := by
  sorry

end NUMINAMATH_GPT_total_boxes_sold_is_189_l1882_188218


namespace NUMINAMATH_GPT_least_perimeter_of_triangle_l1882_188296

-- Define the sides of the triangle
def side1 : ℕ := 40
def side2 : ℕ := 48

-- Given condition for the third side
def valid_third_side (x : ℕ) : Prop :=
  8 < x ∧ x < 88

-- The least possible perimeter given the conditions
def least_possible_perimeter : ℕ :=
  side1 + side2 + 9

theorem least_perimeter_of_triangle (x : ℕ) (h : valid_third_side x) (hx : x = 9) : least_possible_perimeter = 97 :=
by
  rw [least_possible_perimeter]
  exact rfl

end NUMINAMATH_GPT_least_perimeter_of_triangle_l1882_188296


namespace NUMINAMATH_GPT_no_fractional_solution_l1882_188267

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_no_fractional_solution_l1882_188267


namespace NUMINAMATH_GPT_max_ab_min_reciprocal_sum_l1882_188291

noncomputable section

-- Definitions for conditions
def is_positive_real (x : ℝ) : Prop := x > 0

def condition (a b : ℝ) : Prop := is_positive_real a ∧ is_positive_real b ∧ (a + 10 * b = 1)

-- Maximum value of ab
theorem max_ab (a b : ℝ) (h : condition a b) : a * b ≤ 1 / 40 :=
sorry

-- Minimum value of 1/a + 1/b
theorem min_reciprocal_sum (a b : ℝ) (h : condition a b) : 1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_max_ab_min_reciprocal_sum_l1882_188291


namespace NUMINAMATH_GPT_problem_statement_l1882_188255

-- Definitions of the conditions
variables (x y z w : ℕ)

-- The proof problem
theorem problem_statement
  (hx : x^3 = y^2)
  (hz : z^4 = w^3)
  (hzx : z - x = 17)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hw_pos : w > 0) :
  w - y = 229 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1882_188255


namespace NUMINAMATH_GPT_roots_quadratic_l1882_188221

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_l1882_188221


namespace NUMINAMATH_GPT_sqrt_product_l1882_188229

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_product_l1882_188229


namespace NUMINAMATH_GPT_seashells_needed_to_reach_target_l1882_188266

-- Definitions based on the conditions
def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

-- Statement to prove
theorem seashells_needed_to_reach_target : target_seashells - current_seashells = 6 :=
by
  sorry

end NUMINAMATH_GPT_seashells_needed_to_reach_target_l1882_188266


namespace NUMINAMATH_GPT_converse_implication_l1882_188278

theorem converse_implication (a : ℝ) : (a^2 = 1 → a = 1) → (a = 1 → a^2 = 1) :=
sorry

end NUMINAMATH_GPT_converse_implication_l1882_188278


namespace NUMINAMATH_GPT_cost_of_basic_calculator_l1882_188214

variable (B S G : ℕ)

theorem cost_of_basic_calculator 
  (h₁ : S = 2 * B)
  (h₂ : G = 3 * S)
  (h₃ : B + S + G = 72) : 
  B = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_basic_calculator_l1882_188214


namespace NUMINAMATH_GPT_intersection_points_l1882_188283

-- Define parameters: number of sides for each polygon
def n₆ := 6
def n₇ := 7
def n₈ := 8
def n₉ := 9

-- Condition: polygons are inscribed in the same circle, no shared vertices, no three sides intersect at a common point
def polygons_are_disjoint (n₁ n₂ : ℕ) (n₃ n₄ : ℕ) (n₅ : ℕ) : Prop :=
  true -- Assume this is a primitive condition encapsulating given constraints

-- Prove the number of intersection points is 80
theorem intersection_points : polygons_are_disjoint n₆ n₇ n₈ n₉ n₅ → 
  2 * (n₆ + n₇ + n₇ + n₈) + 2 * (n₇ + n₈) + 2 * n₉ = 80 :=
by  
  sorry

end NUMINAMATH_GPT_intersection_points_l1882_188283


namespace NUMINAMATH_GPT_min_x_plus_9y_l1882_188254

variable {x y : ℝ}

theorem min_x_plus_9y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / y = 1) : x + 9 * y ≥ 16 :=
  sorry

end NUMINAMATH_GPT_min_x_plus_9y_l1882_188254


namespace NUMINAMATH_GPT_brock_buys_7_cookies_l1882_188293

variable (cookies_total : ℕ)
variable (sold_to_stone : ℕ)
variable (left_after_sale : ℕ)
variable (cookies_brock_buys : ℕ)
variable (cookies_katy_buys : ℕ)

theorem brock_buys_7_cookies
  (h1 : cookies_total = 5 * 12)
  (h2 : sold_to_stone = 2 * 12)
  (h3 : left_after_sale = 15)
  (h4 : cookies_total - sold_to_stone - (cookies_brock_buys + cookies_katy_buys) = left_after_sale)
  (h5 : cookies_katy_buys = 2 * cookies_brock_buys) :
  cookies_brock_buys = 7 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_brock_buys_7_cookies_l1882_188293


namespace NUMINAMATH_GPT_mark_reads_1750_pages_per_week_l1882_188212

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_mark_reads_1750_pages_per_week_l1882_188212


namespace NUMINAMATH_GPT_pradeep_failure_marks_l1882_188234

theorem pradeep_failure_marks :
  let total_marks := 925
  let pradeep_score := 160
  let passing_percentage := 20
  let passing_marks := (passing_percentage / 100) * total_marks
  let failed_by := passing_marks - pradeep_score
  failed_by = 25 :=
by
  sorry

end NUMINAMATH_GPT_pradeep_failure_marks_l1882_188234


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1882_188209

def set_A : Set ℝ := {x | -x^2 - x + 6 > 0}
def set_B : Set ℝ := {x | 5 / (x - 3) ≤ -1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1882_188209


namespace NUMINAMATH_GPT_sphere_radius_and_volume_l1882_188231

theorem sphere_radius_and_volume (A : ℝ) (d : ℝ) (π : ℝ) (r : ℝ) (R : ℝ) (V : ℝ) 
  (h_cross_section : A = π) (h_distance : d = 1) (h_radius : r = 1) :
  R = Real.sqrt (r^2 + d^2) ∧ V = (4 / 3) * π * R^3 := 
by
  sorry

end NUMINAMATH_GPT_sphere_radius_and_volume_l1882_188231


namespace NUMINAMATH_GPT_complex_number_properties_l1882_188285

open Complex

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Given conditions in Lean: \( z \) satisfies \( z(2+i) = i^{10} \)
def satisfies_condition (z : ℂ) : Prop :=
  z * (2 + i) = i^10

-- Theorem stating the required proofs
theorem complex_number_properties (z : ℂ) (hc : satisfies_condition z) :
  Complex.abs z = Real.sqrt 5 / 5 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  -- Placeholders for the proof steps
  sorry

end NUMINAMATH_GPT_complex_number_properties_l1882_188285


namespace NUMINAMATH_GPT_difference_in_circumferences_l1882_188298

def r_inner : ℝ := 25
def r_outer : ℝ := r_inner + 15

theorem difference_in_circumferences : 2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi := by
  sorry

end NUMINAMATH_GPT_difference_in_circumferences_l1882_188298


namespace NUMINAMATH_GPT_max_remainder_when_divided_by_8_l1882_188243

-- Define the problem: greatest possible remainder when apples divided by 8.
theorem max_remainder_when_divided_by_8 (n : ℕ) : ∃ r : ℕ, r < 8 ∧ r = 7 ∧ n % 8 = r := 
sorry

end NUMINAMATH_GPT_max_remainder_when_divided_by_8_l1882_188243


namespace NUMINAMATH_GPT_real_value_of_m_pure_imaginary_value_of_m_l1882_188205

open Complex

-- Given condition
def z (m : ℝ) : ℂ := (m^2 - m : ℂ) - (m^2 - 1 : ℂ) * I

-- Part (I)
theorem real_value_of_m (m : ℝ) (h : im (z m) = 0) : m = 1 ∨ m = -1 := by
  sorry

-- Part (II)
theorem pure_imaginary_value_of_m (m : ℝ) (h1 : re (z m) = 0) (h2 : im (z m) ≠ 0) : m = 0 := by
  sorry

end NUMINAMATH_GPT_real_value_of_m_pure_imaginary_value_of_m_l1882_188205


namespace NUMINAMATH_GPT_minimum_number_of_different_numbers_l1882_188211

theorem minimum_number_of_different_numbers (total_numbers : ℕ) (frequent_count : ℕ) (frequent_occurrences : ℕ) (less_frequent_occurrences : ℕ) (h1 : total_numbers = 2019) (h2 : frequent_count = 10) (h3 : less_frequent_occurrences = 9) : ∃ k : ℕ, k ≥ 225 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_number_of_different_numbers_l1882_188211


namespace NUMINAMATH_GPT_sqrt_four_squared_l1882_188208

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end NUMINAMATH_GPT_sqrt_four_squared_l1882_188208


namespace NUMINAMATH_GPT_min_ab_correct_l1882_188252

noncomputable def min_ab (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) : ℝ :=
  (6 - 2 * Real.sqrt 3) / 3

theorem min_ab_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) :
  a + b ≥ min_ab a b c h1 h2 :=
sorry

end NUMINAMATH_GPT_min_ab_correct_l1882_188252


namespace NUMINAMATH_GPT_twice_midpoint_l1882_188224

open Complex

def z1 : ℂ := -7 + 5 * I
def z2 : ℂ := 9 - 11 * I

theorem twice_midpoint : 2 * ((z1 + z2) / 2) = 2 - 6 * I := 
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_twice_midpoint_l1882_188224


namespace NUMINAMATH_GPT_smallest_positive_integer_ends_6996_l1882_188249

theorem smallest_positive_integer_ends_6996 :
  ∃ m : ℕ, (m % 4 = 0 ∧ m % 9 = 0 ∧ ∀ d ∈ m.digits 10, d = 6 ∨ d = 9 ∧ m.digits 10 ∩ {6, 9} ≠ ∅ ∧ m % 10000 = 6996) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_ends_6996_l1882_188249


namespace NUMINAMATH_GPT_beetle_number_of_routes_128_l1882_188250

noncomputable def beetle_routes (A B : Type) : Nat :=
  let choices_at_first_step := 4
  let choices_at_second_step := 4
  let choices_at_third_step := 4
  let choices_at_final_step := 2
  choices_at_first_step * choices_at_second_step * choices_at_third_step * choices_at_final_step

theorem beetle_number_of_routes_128 (A B : Type) :
  beetle_routes A B = 128 :=
  by sorry

end NUMINAMATH_GPT_beetle_number_of_routes_128_l1882_188250


namespace NUMINAMATH_GPT_curve_properties_l1882_188223

noncomputable def curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

theorem curve_properties :
  curve 1 0 ∧ curve 0 1 ∧ curve (1/4) (1/4) ∧ 
  (∀ p : ℝ × ℝ, curve p.1 p.2 → curve p.2 p.1) :=
by
  sorry

end NUMINAMATH_GPT_curve_properties_l1882_188223


namespace NUMINAMATH_GPT_alice_basketball_probability_l1882_188207

/-- Alice and Bob play a game with a basketball. On each turn, if Alice has the basketball,
 there is a 5/8 chance that she will toss it to Bob and a 3/8 chance that she will keep the basketball.
 If Bob has the basketball, there is a 1/4 chance that he will toss it to Alice, and if he doesn't toss it to Alice,
 he keeps it. Alice starts with the basketball. What is the probability that Alice has the basketball again after two turns? -/
theorem alice_basketball_probability :
  (5 / 8) * (1 / 4) + (3 / 8) * (3 / 8) = 19 / 64 := 
by
  sorry

end NUMINAMATH_GPT_alice_basketball_probability_l1882_188207


namespace NUMINAMATH_GPT_hyperbola_imaginary_axis_twice_real_axis_l1882_188246

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) : 
  (exists (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0), mx^2 + b^2 * y^2 = b^2) ∧
  (b = 2 * a) ∧ (m < 0) → 
  m = -1 / 4 := 
sorry

end NUMINAMATH_GPT_hyperbola_imaginary_axis_twice_real_axis_l1882_188246


namespace NUMINAMATH_GPT_constant_abs_difference_l1882_188217

variable (a : ℕ → ℝ)

-- Define the condition for the recurrence relation
def recurrence_relation : Prop := ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

-- State the theorem
theorem constant_abs_difference (h : recurrence_relation a) : ∃ C : ℝ, ∀ n ≥ 2, |(a n)^2 - (a (n-1)) * (a (n+1))| = C :=
    sorry

end NUMINAMATH_GPT_constant_abs_difference_l1882_188217


namespace NUMINAMATH_GPT_min_value_3x_4y_l1882_188275

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / x + 1 / y = 1) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_3x_4y_l1882_188275


namespace NUMINAMATH_GPT_opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l1882_188297

theorem opposite_of_neg23_eq_23 : -(-23) = 23 := 
by sorry

theorem reciprocal_of_neg23_eq_neg_1_div_23 : (1 : ℚ) / (-23) = -(1 / 23 : ℚ) :=
by sorry

theorem abs_value_of_neg23_eq_23 : abs (-23) = 23 :=
by sorry

end NUMINAMATH_GPT_opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l1882_188297


namespace NUMINAMATH_GPT_reciprocal_neg_half_l1882_188244

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_half_l1882_188244


namespace NUMINAMATH_GPT_probability_of_5_pieces_of_candy_l1882_188287

-- Define the conditions
def total_eggs : ℕ := 100 -- Assume total number of eggs is 100 for simplicity
def blue_eggs : ℕ := 4 * total_eggs / 5
def purple_eggs : ℕ := total_eggs / 5
def blue_eggs_with_5_candies : ℕ := blue_eggs / 4
def purple_eggs_with_5_candies : ℕ := purple_eggs / 2
def total_eggs_with_5_candies : ℕ := blue_eggs_with_5_candies + purple_eggs_with_5_candies

-- The proof problem
theorem probability_of_5_pieces_of_candy : (total_eggs_with_5_candies : ℚ) / (total_eggs : ℚ) = 3 / 10 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_5_pieces_of_candy_l1882_188287


namespace NUMINAMATH_GPT_remainder_div_x_minus_4_l1882_188295

def f (x : ℕ) : ℕ := x^5 - 8 * x^4 + 16 * x^3 + 25 * x^2 - 50 * x + 24

theorem remainder_div_x_minus_4 : 
  (f 4) = 224 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_div_x_minus_4_l1882_188295


namespace NUMINAMATH_GPT_min_y_value_l1882_188269

noncomputable def y (x : ℝ) : ℝ :=
  (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

theorem min_y_value : 
  ∃ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, abs (x' - 5.92) < δ → abs (y x' - y 5.92) < ε) :=
sorry

end NUMINAMATH_GPT_min_y_value_l1882_188269


namespace NUMINAMATH_GPT_hari_joined_after_5_months_l1882_188277

noncomputable def praveen_investment := 3780 * 12
noncomputable def hari_investment (x : ℕ) := 9720 * (12 - x)

theorem hari_joined_after_5_months :
  ∃ (x : ℕ), (praveen_investment : ℝ) / (hari_investment x) = (2:ℝ) / 3 ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_hari_joined_after_5_months_l1882_188277


namespace NUMINAMATH_GPT_area_ratio_problem_l1882_188289

theorem area_ratio_problem
  (A B C : ℝ) -- Areas of the corresponding regions
  (m n : ℕ)  -- Given ratios
  (PQR_is_right_triangle : true)  -- PQR is a right-angled triangle (placeholder condition)
  (RSTU_is_rectangle : true)  -- RSTU is a rectangle (placeholder condition)
  (ratio_A_B : A / B = m / 2)  -- Ratio condition 1
  (ratio_A_C : A / C = n / 1)  -- Ratio condition 2
  (PTS_sim_TQU_sim_PQR : true)  -- Similar triangles (placeholder condition)
  : n = 9 := 
sorry

end NUMINAMATH_GPT_area_ratio_problem_l1882_188289


namespace NUMINAMATH_GPT_work_completion_days_l1882_188215

theorem work_completion_days (a b : ℕ) (h1 : a + b = 6) (h2 : a + b = 15 / 4) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l1882_188215


namespace NUMINAMATH_GPT_short_side_is_7_l1882_188241

variable (L S : ℕ)

-- Given conditions
def perimeter : ℕ := 38
def long_side : ℕ := 12

-- In Lean, prove that the short side is 7 given L and P
theorem short_side_is_7 (h1 : 2 * L + 2 * S = perimeter) (h2 : L = long_side) : S = 7 := by
  sorry

end NUMINAMATH_GPT_short_side_is_7_l1882_188241


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1882_188260

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1882_188260


namespace NUMINAMATH_GPT_max_writers_and_editors_l1882_188279

theorem max_writers_and_editors (T W : ℕ) (E : ℕ) (x : ℕ) (hT : T = 100) (hW : W = 35) (hE : E > 38) (h_comb : W + E + x = T)
    (h_neither : T = W + E + x) : x = 26 := by
  sorry

end NUMINAMATH_GPT_max_writers_and_editors_l1882_188279


namespace NUMINAMATH_GPT_big_bea_bananas_l1882_188226

theorem big_bea_bananas :
  ∃ (b : ℕ), (b + (b + 8) + (b + 16) + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 196) ∧ (b + 48 = 52) := by
  sorry

end NUMINAMATH_GPT_big_bea_bananas_l1882_188226


namespace NUMINAMATH_GPT_find_value_of_reciprocal_sin_double_angle_l1882_188237

open Real

noncomputable def point := ℝ × ℝ

def term_side_angle_passes_through (α : ℝ) (P : point) :=
  ∃ (r : ℝ), P = (r * cos α, r * sin α)

theorem find_value_of_reciprocal_sin_double_angle (α : ℝ) (P : point) (h : term_side_angle_passes_through α P) :
  P = (-2, 1) → (1 / sin (2 * α)) = -5 / 4 :=
by
  intro hP
  sorry

end NUMINAMATH_GPT_find_value_of_reciprocal_sin_double_angle_l1882_188237


namespace NUMINAMATH_GPT_pond_depth_l1882_188236

theorem pond_depth (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) :
    V = L * W * D ↔ D = 5 := 
by
  rw [hL, hW, hV]
  constructor
  · intro h1
    linarith
  · intro h2
    rw [h2]
    linarith

#check pond_depth

end NUMINAMATH_GPT_pond_depth_l1882_188236


namespace NUMINAMATH_GPT_blue_eyes_blonde_hair_logic_l1882_188248

theorem blue_eyes_blonde_hair_logic :
  ∀ (a b c d : ℝ), 
  (a / (a + b) > (a + c) / (a + b + c + d)) →
  (a / (a + c) > (a + b) / (a + b + c + d)) :=
by
  intro a b c d h
  sorry

end NUMINAMATH_GPT_blue_eyes_blonde_hair_logic_l1882_188248


namespace NUMINAMATH_GPT_intersection_M_P_l1882_188210

variable {x a : ℝ}

def M (a : ℝ) : Set ℝ := { x | x > a ∧ a^2 - 12*a + 20 < 0 }
def P : Set ℝ := { x | x ≤ 10 }

theorem intersection_M_P (a : ℝ) (h : 2 < a ∧ a < 10) : 
  M a ∩ P = { x | a < x ∧ x ≤ 10 } :=
sorry

end NUMINAMATH_GPT_intersection_M_P_l1882_188210


namespace NUMINAMATH_GPT_find_f_of_five_thirds_l1882_188239

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_f_of_five_thirds (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_fun : ∀ x : ℝ, f (1 + x) = f (-x))
  (h_val : f (-1 / 3) = 1 / 3) : 
  f (5 / 3) = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_find_f_of_five_thirds_l1882_188239


namespace NUMINAMATH_GPT_cos_double_angle_l1882_188247

variable (α : ℝ)
variable (h : Real.cos α = 2/3)

theorem cos_double_angle : Real.cos (2 * α) = -1/9 :=
  by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1882_188247


namespace NUMINAMATH_GPT_value_of_coins_l1882_188240

theorem value_of_coins (n d : ℕ) (hn : n + d = 30)
    (hv : 10 * n + 5 * d = 5 * n + 10 * d + 90) :
    300 - 5 * n = 180 := by
  sorry

end NUMINAMATH_GPT_value_of_coins_l1882_188240


namespace NUMINAMATH_GPT_pictures_per_day_calc_l1882_188245

def years : ℕ := 3
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

def number_of_cards : ℕ := total_spent / cost_per_card
def total_images : ℕ := number_of_cards * images_per_card
def days_in_year : ℕ := 365
def total_days : ℕ := years * days_in_year

theorem pictures_per_day_calc : 
  (total_images / total_days) = 10 := 
by
  sorry

end NUMINAMATH_GPT_pictures_per_day_calc_l1882_188245


namespace NUMINAMATH_GPT_right_obtuse_triangle_impossible_l1882_188292

def triangle_interior_angles_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_right_angle (α : ℝ) : Prop :=
  α = 90

def is_obtuse_angle (α : ℝ) : Prop :=
  α > 90

theorem right_obtuse_triangle_impossible (α β γ : ℝ) (h1 : triangle_interior_angles_sum α β γ) (h2 : is_right_angle α) (h3 : is_obtuse_angle β) : false :=
  sorry

end NUMINAMATH_GPT_right_obtuse_triangle_impossible_l1882_188292


namespace NUMINAMATH_GPT_minimum_number_of_gloves_l1882_188282

theorem minimum_number_of_gloves (participants : ℕ) (gloves_per_participant : ℕ) (total_participants : participants = 63) (each_participant_needs_2_gloves : gloves_per_participant = 2) : 
  participants * gloves_per_participant = 126 :=
by
  rcases participants, gloves_per_participant, total_participants, each_participant_needs_2_gloves
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_minimum_number_of_gloves_l1882_188282


namespace NUMINAMATH_GPT_num_solutions_x_squared_minus_y_squared_eq_2001_l1882_188299

theorem num_solutions_x_squared_minus_y_squared_eq_2001 
  (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2 - y^2 = 2001 ↔ (x, y) = (1001, 1000) ∨ (x, y) = (335, 332) := sorry

end NUMINAMATH_GPT_num_solutions_x_squared_minus_y_squared_eq_2001_l1882_188299


namespace NUMINAMATH_GPT_correlation_coefficient_correct_option_l1882_188294

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end NUMINAMATH_GPT_correlation_coefficient_correct_option_l1882_188294


namespace NUMINAMATH_GPT_hyperbola_range_l1882_188264

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (|m| - 1) - y^2 / (m - 2) = 1)) → (-1 < m ∧ m < 1) ∨ (m > 2) := by
  sorry

end NUMINAMATH_GPT_hyperbola_range_l1882_188264


namespace NUMINAMATH_GPT_circles_coincide_l1882_188272

-- Definitions for circle being inscribed in an angle and touching each other
structure Circle :=
  (radius : ℝ)
  (center: ℝ × ℝ)

def inscribed_in_angle (c : Circle) (θ: ℝ) : Prop :=
  -- Placeholder definition for circle inscribed in an angle
  sorry

def touches (c₁ c₂ : Circle) : Prop :=
  -- Placeholder definition for circles touching each other
  sorry

-- The angles of the triangle ABC are A, B, and C.
-- We are given the following conditions:
variables (A B C : ℝ) -- angles
variables (S1 S2 S3 S4 S5 S6 S7: Circle) -- circles

-- Circle S1 is inscribed in angle A
axiom S1_condition : inscribed_in_angle S1 A

-- Circle S2 is inscribed in angle B and touches S1 externally
axiom S2_condition : inscribed_in_angle S2 B ∧ touches S2 S1

-- Circle S3 is inscribed in angle C and touches S2
axiom S3_condition : inscribed_in_angle S3 C ∧ touches S3 S2

-- Circle S4 is inscribed in angle A and touches S3
axiom S4_condition : inscribed_in_angle S4 A ∧ touches S4 S3

-- We repeat this pattern up to circle S7
axiom S5_condition : inscribed_in_angle S5 B ∧ touches S5 S4
axiom S6_condition : inscribed_in_angle S6 C ∧ touches S6 S5
axiom S7_condition : inscribed_in_angle S7 A ∧ touches S7 S6

-- We need to prove the circle S7 coincides with S1
theorem circles_coincide : S7 = S1 :=
by
  -- Proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_circles_coincide_l1882_188272


namespace NUMINAMATH_GPT_smallest_integer_y_l1882_188242

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_l1882_188242


namespace NUMINAMATH_GPT_max_k_value_l1882_188219

theorem max_k_value (m : ℝ) (h : 0 < m ∧ m < 1/2) : 
  ∃ k : ℝ, (∀ m, 0 < m ∧ m < 1/2 → (1 / m + 2 / (1 - 2 * m)) ≥ k) ∧ k = 8 :=
by sorry

end NUMINAMATH_GPT_max_k_value_l1882_188219


namespace NUMINAMATH_GPT_range_of_a_nonempty_intersection_range_of_a_subset_intersection_l1882_188204

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) ≤ 0}

-- Define set B in terms of variable a
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Statement 1: Proving the range of a when A ∩ B ≠ ∅
theorem range_of_a_nonempty_intersection (a : ℝ) : (A ∩ B a ≠ ∅) → (-1 / 2 ≤ a ∧ a ≤ 2) :=
by
  sorry

-- Statement 2: Proving the range of a when A ∩ B = B
theorem range_of_a_subset_intersection (a : ℝ) : (A ∩ B a = B a) → (a ≥ 2 ∨ a ≤ -3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_nonempty_intersection_range_of_a_subset_intersection_l1882_188204


namespace NUMINAMATH_GPT_min_value_of_x2_plus_y2_l1882_188258

open Real

theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  x^2 + y^2 ≥ 7 - 4 * sqrt 3 := sorry

end NUMINAMATH_GPT_min_value_of_x2_plus_y2_l1882_188258


namespace NUMINAMATH_GPT_nancy_indian_food_freq_l1882_188265

-- Definitions based on the problem
def antacids_per_indian_day := 3
def antacids_per_mexican_day := 2
def antacids_per_other_day := 1
def mexican_per_week := 2
def total_antacids_per_month := 60
def weeks_per_month := 4
def days_per_week := 7

-- The proof statement
theorem nancy_indian_food_freq :
  ∃ (I : ℕ), (total_antacids_per_month = 
    weeks_per_month * (antacids_per_indian_day * I + 
    antacids_per_mexican_day * mexican_per_week + 
    antacids_per_other_day * (days_per_week - I - mexican_per_week))) ∧ I = 3 :=
by
  sorry

end NUMINAMATH_GPT_nancy_indian_food_freq_l1882_188265


namespace NUMINAMATH_GPT_find_base_17_digit_l1882_188227

theorem find_base_17_digit (a : ℕ) (h1 : 0 ≤ a ∧ a < 17) 
  (h2 : (25 + a) % 16 = 0) : a = 7 :=
sorry

end NUMINAMATH_GPT_find_base_17_digit_l1882_188227


namespace NUMINAMATH_GPT_oysters_eaten_l1882_188216

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end NUMINAMATH_GPT_oysters_eaten_l1882_188216


namespace NUMINAMATH_GPT_relationship_between_k_and_a_l1882_188200

theorem relationship_between_k_and_a (a k : ℝ) (h_a : 0 < a ∧ a < 1) :
  (k^2 + 1) * a^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_relationship_between_k_and_a_l1882_188200


namespace NUMINAMATH_GPT_range_of_x_l1882_188202

theorem range_of_x (x : ℝ) : (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → x^2 - (t^2 + t - 3) * x + t^2 * (t - 3) > 0) ↔ (x < -4 ∨ x > 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1882_188202


namespace NUMINAMATH_GPT_train_speed_and_length_l1882_188263

theorem train_speed_and_length (V l : ℝ) 
  (h1 : 7 * V = l) 
  (h2 : 25 * V = 378 + l) : 
  V = 21 ∧ l = 147 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_and_length_l1882_188263


namespace NUMINAMATH_GPT_value_is_100_l1882_188259

theorem value_is_100 (number : ℕ) (h : number = 20) : 5 * number = 100 :=
by
  sorry

end NUMINAMATH_GPT_value_is_100_l1882_188259


namespace NUMINAMATH_GPT_exists_increasing_sequences_l1882_188280

theorem exists_increasing_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, b n < b (n + 1)) ∧
  (∀ n : ℕ, a n * (a n + 1) ∣ b n ^ 2 + 1) :=
sorry

end NUMINAMATH_GPT_exists_increasing_sequences_l1882_188280


namespace NUMINAMATH_GPT_expected_digits_die_l1882_188228

noncomputable def expected_number_of_digits (numbers : List ℕ) : ℚ :=
  let one_digit_numbers := numbers.filter (λ n => n < 10)
  let two_digit_numbers := numbers.filter (λ n => n >= 10)
  let p_one_digit := (one_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  let p_two_digit := (two_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  p_one_digit * 1 + p_two_digit * 2

theorem expected_digits_die :
  expected_number_of_digits [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 1.5833 := 
by
  sorry

end NUMINAMATH_GPT_expected_digits_die_l1882_188228


namespace NUMINAMATH_GPT_ratio_of_neighborhood_to_gina_l1882_188203

variable (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ)

def neighborhood_to_gina_ratio (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ) := 
  (Total_weight_collected - Gina_bags * Weight_per_bag) / (Gina_bags * Weight_per_bag)

theorem ratio_of_neighborhood_to_gina 
  (h₁ : Gina_bags = 2) 
  (h₂ : Weight_per_bag = 4) 
  (h₃ : Total_weight_collected = 664) :
  neighborhood_to_gina_ratio Gina_bags Weight_per_bag Total_weight_collected = 82 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_neighborhood_to_gina_l1882_188203


namespace NUMINAMATH_GPT_probability_of_black_yellow_green_probability_of_not_red_or_green_l1882_188256

namespace ProbabilityProof

/- Definitions of events A, B, C, D representing probabilities as real numbers -/
variables (P_A P_B P_C P_D : ℝ)

/- Conditions stated in the problem -/
def conditions (h1 : P_A = 1 / 3)
               (h2 : P_B + P_C = 5 / 12)
               (h3 : P_C + P_D = 5 / 12)
               (h4 : P_A + P_B + P_C + P_D = 1) :=
  true

/- Proof that P(B) = 1/4, P(C) = 1/6, and P(D) = 1/4 given the conditions -/
theorem probability_of_black_yellow_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1) :
  P_B = 1 / 4 ∧ P_C = 1 / 6 ∧ P_D = 1 / 4 :=
by
  sorry

/- Proof that the probability of not drawing a red or green ball is 5/12 -/
theorem probability_of_not_red_or_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1)
  (h5 : P_B = 1 / 4)
  (h6 : P_C = 1 / 6)
  (h7 : P_D = 1 / 4) :
  1 - (P_A + P_D) = 5 / 12 :=
by
  sorry

end ProbabilityProof

end NUMINAMATH_GPT_probability_of_black_yellow_green_probability_of_not_red_or_green_l1882_188256


namespace NUMINAMATH_GPT_scientific_notation_equivalence_l1882_188201

theorem scientific_notation_equivalence : 3 * 10^(-7) = 0.0000003 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_equivalence_l1882_188201


namespace NUMINAMATH_GPT_water_usage_l1882_188238

theorem water_usage (payment : ℝ) (usage : ℝ) : 
  payment = 7.2 → (usage ≤ 6 → payment = usage * 0.8) → (usage > 6 → payment = 4.8 + (usage - 6) * 1.2) → usage = 8 :=
by
  sorry

end NUMINAMATH_GPT_water_usage_l1882_188238


namespace NUMINAMATH_GPT_smallest_positive_integer_in_form_l1882_188262

theorem smallest_positive_integer_in_form :
  ∃ (m n p : ℤ), 1234 * m + 56789 * n + 345 * p = 1 := sorry

end NUMINAMATH_GPT_smallest_positive_integer_in_form_l1882_188262


namespace NUMINAMATH_GPT_no_solution_bills_l1882_188257

theorem no_solution_bills (x y z : ℕ) (h1 : x + y + z = 10) (h2 : x + 3 * y + 5 * z = 25) : false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_bills_l1882_188257


namespace NUMINAMATH_GPT_sum_of_legs_is_104_l1882_188220

theorem sum_of_legs_is_104 (x : ℕ) (h₁ : x^2 + (x + 2)^2 = 53^2) : x + (x + 2) = 104 := sorry

end NUMINAMATH_GPT_sum_of_legs_is_104_l1882_188220


namespace NUMINAMATH_GPT_binom_n_n_minus_1_l1882_188286

theorem binom_n_n_minus_1 (n : ℕ) (h : 0 < n) : (Nat.choose n (n-1)) = n :=
  sorry

end NUMINAMATH_GPT_binom_n_n_minus_1_l1882_188286


namespace NUMINAMATH_GPT_find_a_n_find_b_n_find_T_n_l1882_188230

-- definitions of sequences and common ratios
variable (a_n b_n : ℕ → ℕ)
variable (S_n T_n : ℕ → ℕ)
variable (q : ℝ)
variable (n : ℕ)

-- conditions
axiom a1 : a_n 1 = 1
axiom S3 : S_n 3 = 9
axiom b1 : b_n 1 = 1
axiom b3 : b_n 3 = 20
axiom q_pos : q > 0
axiom geo_seq : (∀ n, b_n n / a_n n = q ^ (n - 1))

-- goals to prove
theorem find_a_n : ∀ n, a_n n = 2 * n - 1 := 
by sorry

theorem find_b_n : ∀ n, b_n n = (2 * n - 1) * 2 ^ (n - 1) := 
by sorry

theorem find_T_n : ∀ n, T_n n = (2 * n - 3) * 2 ^ n + 3 :=
by sorry

end NUMINAMATH_GPT_find_a_n_find_b_n_find_T_n_l1882_188230


namespace NUMINAMATH_GPT_hannah_final_pay_l1882_188290

theorem hannah_final_pay : (30 * 18) - (5 * 3) + (15 * 4) - (((30 * 18) - (5 * 3) + (15 * 4)) * 0.10 + ((30 * 18) - (5 * 3) + (15 * 4)) * 0.05) = 497.25 :=
by
  sorry

end NUMINAMATH_GPT_hannah_final_pay_l1882_188290


namespace NUMINAMATH_GPT_cost_price_of_apple_is_18_l1882_188271

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_apple_is_18_l1882_188271


namespace NUMINAMATH_GPT_repaved_inches_before_today_l1882_188233

theorem repaved_inches_before_today :
  let A := 4000
  let B := 3500
  let C := 2500
  let repaved_A := 0.70 * A
  let repaved_B := 0.60 * B
  let repaved_C := 0.80 * C
  let total_repaved_before := repaved_A + repaved_B + repaved_C
  let repaved_today := 950
  let new_total_repaved := total_repaved_before + repaved_today
  new_total_repaved - repaved_today = 6900 :=
by
  sorry

end NUMINAMATH_GPT_repaved_inches_before_today_l1882_188233


namespace NUMINAMATH_GPT_min_value_of_a_squared_plus_b_squared_l1882_188213

-- Problem definition and condition
def is_on_circle (a b : ℝ) : Prop :=
  (a^2 + b^2 - 2*a + 4*b - 20) = 0

-- Theorem statement
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) (h : is_on_circle a b) :
  a^2 + b^2 = 30 - 10 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_squared_plus_b_squared_l1882_188213


namespace NUMINAMATH_GPT_sequence_G_51_l1882_188251

theorem sequence_G_51 :
  ∀ G : ℕ → ℚ, 
  (∀ n : ℕ, G (n + 1) = (3 * G n + 2) / 2) → 
  G 1 = 3 → 
  G 51 = (3^51 + 1) / 2 := by 
  sorry

end NUMINAMATH_GPT_sequence_G_51_l1882_188251


namespace NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1882_188261

theorem sixth_graders_more_than_seventh (c_pencil : ℕ) (h_cents : c_pencil > 0)
    (h_cond : ∀ n : ℕ, n * c_pencil = 221 ∨ n * c_pencil = 286)
    (h_sixth_graders : 35 > 0) :
    ∃ n6 n7 : ℕ, n6 > n7 ∧ n6 - n7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1882_188261


namespace NUMINAMATH_GPT_map_scale_to_yards_l1882_188206

theorem map_scale_to_yards :
  (6.25 * 500) / 3 = 1041 + 2 / 3 := 
by sorry

end NUMINAMATH_GPT_map_scale_to_yards_l1882_188206


namespace NUMINAMATH_GPT_sum_of_11378_and_121_is_odd_l1882_188235

theorem sum_of_11378_and_121_is_odd (h1 : Even 11378) (h2 : Odd 121) : Odd (11378 + 121) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_11378_and_121_is_odd_l1882_188235


namespace NUMINAMATH_GPT_men_in_second_group_l1882_188222

theorem men_in_second_group (M : ℕ) : 
    (18 * 20 = M * 24) → M = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_men_in_second_group_l1882_188222


namespace NUMINAMATH_GPT_p_eval_at_neg_one_l1882_188273

noncomputable def p (x : ℝ) : ℝ :=
  x^2 - 2*x + 9

theorem p_eval_at_neg_one : p (-1) = 12 := by
  sorry

end NUMINAMATH_GPT_p_eval_at_neg_one_l1882_188273


namespace NUMINAMATH_GPT_cone_cannot_have_rectangular_cross_section_l1882_188284

noncomputable def solid := Type

def is_cylinder (s : solid) : Prop := sorry
def is_cone (s : solid) : Prop := sorry
def is_rectangular_prism (s : solid) : Prop := sorry
def is_cube (s : solid) : Prop := sorry

def has_rectangular_cross_section (s : solid) : Prop := sorry

axiom cylinder_has_rectangular_cross_section (s : solid) : is_cylinder s → has_rectangular_cross_section s
axiom rectangular_prism_has_rectangular_cross_section (s : solid) : is_rectangular_prism s → has_rectangular_cross_section s
axiom cube_has_rectangular_cross_section (s : solid) : is_cube s → has_rectangular_cross_section s

theorem cone_cannot_have_rectangular_cross_section (s : solid) : is_cone s → ¬has_rectangular_cross_section s := 
sorry

end NUMINAMATH_GPT_cone_cannot_have_rectangular_cross_section_l1882_188284


namespace NUMINAMATH_GPT_expected_value_correct_prob_abs_diff_ge_1_correct_l1882_188281

/-- Probability distribution for a single die roll -/
def prob_score (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else if n = 2 then 1/3 else if n = 3 then 1/6 else 0

/-- Expected value based on the given probability distribution -/
def expected_value : ℚ := 
  (1 * prob_score 1) + (2 * prob_score 2) + (3 * prob_score 3)

/-- Proving the expected value calculation -/
theorem expected_value_correct : expected_value = 7/6 :=
  by sorry

/-- Calculate the probability of score difference being at least 1 between two players -/
def prob_abs_diff_ge_1 (x y : ℕ) : ℚ :=
  -- Implementation would involve detailed probability combinations that result in diff >= 1
  sorry

/-- Prove the probability of |x - y| being at least 1 -/
theorem prob_abs_diff_ge_1_correct : 
  ∀ (x y : ℕ), prob_abs_diff_ge_1 x y < 1 :=
  by sorry

end NUMINAMATH_GPT_expected_value_correct_prob_abs_diff_ge_1_correct_l1882_188281


namespace NUMINAMATH_GPT_largest_eight_digit_number_contains_even_digits_l1882_188268

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end NUMINAMATH_GPT_largest_eight_digit_number_contains_even_digits_l1882_188268


namespace NUMINAMATH_GPT_exists_infinitely_many_n_l1882_188274

def digit_sum (m : ℕ) : ℕ := sorry  -- Define the digit sum function

theorem exists_infinitely_many_n (S : ℕ → ℕ)
  (hS : ∀ m : ℕ, S m = digit_sum m) :
  ∃ᶠ n in at_top, S (3^n) ≥ S (3^(n + 1)) := 
sorry

end NUMINAMATH_GPT_exists_infinitely_many_n_l1882_188274


namespace NUMINAMATH_GPT_parabola_vertex_y_axis_opens_upwards_l1882_188270

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end NUMINAMATH_GPT_parabola_vertex_y_axis_opens_upwards_l1882_188270
