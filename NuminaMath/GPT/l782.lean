import Mathlib

namespace NUMINAMATH_GPT_people_with_diploma_percentage_l782_78289

-- Definitions of the given conditions
def P_j_and_not_d := 0.12
def P_not_j_and_d := 0.15
def P_j := 0.40

-- Definitions for intermediate values
def P_not_j := 1 - P_j
def P_not_j_d := P_not_j * P_not_j_and_d

-- Definition of the result to prove
def P_d := (P_j - P_j_and_not_d) + P_not_j_d

theorem people_with_diploma_percentage : P_d = 0.43 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_people_with_diploma_percentage_l782_78289


namespace NUMINAMATH_GPT_f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l782_78280

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Part (1)
theorem f_positive_for_all_x (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k > -2 := sorry

-- Part (2)
theorem f_min_value_negative_two (k : ℝ) : (∀ x : ℝ, f x k ≥ -2) → k = -8 := sorry

-- Part (3)
theorem f_triangle_sides (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, (f x1 k + f x2 k > f x3 k) ∧ (f x2 k + f x3 k > f x1 k) ∧ (f x3 k + f x1 k > f x2 k)) ↔ (-1/2 ≤ k ∧ k ≤ 4) := sorry

end NUMINAMATH_GPT_f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l782_78280


namespace NUMINAMATH_GPT_parabola_equation_through_origin_point_l782_78295

-- Define the conditions
def vertex_origin := (0, 0)
def point_on_parabola := (-2, 4)

-- Define what it means to be a standard equation of a parabola passing through a point
def standard_equation_passing_through (p : ℝ) (x y : ℝ) : Prop :=
  (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)

-- The theorem stating the conclusion
theorem parabola_equation_through_origin_point :
  ∃ p > 0, standard_equation_passing_through p (-2) 4 ∧
  (4^2 = -8 * (-2) ∨ (-2)^2 = 4) := 
sorry

end NUMINAMATH_GPT_parabola_equation_through_origin_point_l782_78295


namespace NUMINAMATH_GPT_factorial_units_digit_l782_78263

theorem factorial_units_digit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hba : a < b) : 
  ¬ (∃ k : ℕ, (b! - a!) % 10 = 7) := 
sorry

end NUMINAMATH_GPT_factorial_units_digit_l782_78263


namespace NUMINAMATH_GPT_base10_to_base7_conversion_l782_78258

theorem base10_to_base7_conversion :
  ∃ b1 b2 b3 b4 b5 : ℕ, 3 * 7^3 + 1 * 7^2 + 6 * 7^1 + 6 * 7^0 = 3527 ∧ 
  b1 = 1 ∧ b2 = 3 ∧ b3 = 1 ∧ b4 = 6 ∧ b5 = 6 ∧ (3527:ℕ) = (1*7^4 + b1*7^3 + b2*7^2 + b3*7^1 + b4*7^0) := by
sorry

end NUMINAMATH_GPT_base10_to_base7_conversion_l782_78258


namespace NUMINAMATH_GPT_lcm_gcf_ratio_280_450_l782_78286

open Nat

theorem lcm_gcf_ratio_280_450 :
  let a := 280
  let b := 450
  lcm a b / gcd a b = 1260 :=
by
  let a := 280
  let b := 450
  sorry

end NUMINAMATH_GPT_lcm_gcf_ratio_280_450_l782_78286


namespace NUMINAMATH_GPT_painting_problem_l782_78278

theorem painting_problem (initial_painters : ℕ) (initial_days : ℚ) (initial_rate : ℚ) (new_days : ℚ) (new_rate : ℚ) : 
  initial_painters = 6 ∧ initial_days = 5/2 ∧ initial_rate = 2 ∧ new_days = 2 ∧ new_rate = 2.5 →
  ∃ additional_painters : ℕ, additional_painters = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_painting_problem_l782_78278


namespace NUMINAMATH_GPT_Dorottya_should_go_first_l782_78285

def probability_roll_1_or_2 : ℚ := 2 / 10

def probability_no_roll_1_or_2 : ℚ := 1 - probability_roll_1_or_2

variables {P_1 P_2 P_3 P_4 P_5 P_6 : ℚ}
  (hP1 : P_1 = probability_roll_1_or_2 * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP2 : P_2 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 1) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP3 : P_3 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 2) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP4 : P_4 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 3) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP5 : P_5 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 4) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP6 : P_6 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 5) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))

theorem Dorottya_should_go_first : P_1 > P_2 ∧ P_2 > P_3 ∧ P_3 > P_4 ∧ P_4 > P_5 ∧ P_5 > P_6 :=
by {
  -- Skipping actual proof steps
  sorry
}

end NUMINAMATH_GPT_Dorottya_should_go_first_l782_78285


namespace NUMINAMATH_GPT_evaluate_integral_l782_78212

noncomputable def integral_problem : Real :=
  ∫ x in (-2 : Real)..(2 : Real), (Real.sqrt (4 - x^2) - x^2017)

theorem evaluate_integral :
  integral_problem = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_evaluate_integral_l782_78212


namespace NUMINAMATH_GPT_average_of_three_quantities_l782_78221

theorem average_of_three_quantities 
  (five_avg : ℚ) (three_avg : ℚ) (two_avg : ℚ) 
  (h_five_avg : five_avg = 10) 
  (h_two_avg : two_avg = 19) : 
  three_avg = 4 := 
by 
  let sum_5 := 5 * 10
  let sum_2 := 2 * 19
  let sum_3 := sum_5 - sum_2
  let three_avg := sum_3 / 3
  sorry

end NUMINAMATH_GPT_average_of_three_quantities_l782_78221


namespace NUMINAMATH_GPT_max_snacks_l782_78205

-- Define the conditions and the main statement we want to prove

def single_snack_cost : ℕ := 2
def four_snack_pack_cost : ℕ := 6
def six_snack_pack_cost : ℕ := 8
def budget : ℕ := 20

def max_snacks_purchased : ℕ := 14

theorem max_snacks (h1 : single_snack_cost = 2) 
                   (h2 : four_snack_pack_cost = 6) 
                   (h3 : six_snack_pack_cost = 8) 
                   (h4 : budget = 20) : 
                   max_snacks_purchased = 14 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_snacks_l782_78205


namespace NUMINAMATH_GPT_cloud_ratio_l782_78261

theorem cloud_ratio (D Carson Total : ℕ) (h1 : Carson = 6) (h2 : Total = 24) (h3 : Carson + D = Total) :
  (D / Carson) = 3 := by
  sorry

end NUMINAMATH_GPT_cloud_ratio_l782_78261


namespace NUMINAMATH_GPT_ratio_transformation_l782_78291

theorem ratio_transformation (x1 y1 x2 y2 : ℚ) (h₁ : x1 / y1 = 7 / 5) (h₂ : x2 = x1 * y1) (h₃ : y2 = y1 * x1) : x2 / y2 = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_transformation_l782_78291


namespace NUMINAMATH_GPT_total_cost_correct_l782_78268

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4

theorem total_cost_correct :
  sandwich_quantity * sandwich_cost + soda_quantity * soda_cost = 8.38 := 
  by
    sorry

end NUMINAMATH_GPT_total_cost_correct_l782_78268


namespace NUMINAMATH_GPT_abs_x_minus_y_l782_78217

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_minus_y_l782_78217


namespace NUMINAMATH_GPT_geom_seq_arith_seq_l782_78218

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def isGeomSeq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

theorem geom_seq_arith_seq (h1 : ∀ n, 0 < a n) 
  (h2 : isGeomSeq a q)
  (h3 : 2 * (1 / 2 * a 5) = a 3 + a 4)
  : (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := 
sorry

end NUMINAMATH_GPT_geom_seq_arith_seq_l782_78218


namespace NUMINAMATH_GPT_find_m_l782_78272

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

-- The intersection condition
def intersect_condition (m : ℝ) : Prop := A m ∩ B = {3}

-- The statement to prove
theorem find_m : ∃ m : ℝ, intersect_condition m → m = 3 :=
by {
  use 3,
  sorry
}

end NUMINAMATH_GPT_find_m_l782_78272


namespace NUMINAMATH_GPT_smaller_cube_edge_length_l782_78255

-- Given conditions
variables (s : ℝ) (volume_large_cube : ℝ) (n : ℝ)
-- n = 8 (number of smaller cubes), volume_large_cube = 1000 cm³

theorem smaller_cube_edge_length (h1 : n = 8) (h2 : volume_large_cube = 1000) :
  s^3 = volume_large_cube / n → s = 5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_cube_edge_length_l782_78255


namespace NUMINAMATH_GPT_classroom_activity_solution_l782_78249

theorem classroom_activity_solution 
  (x y : ℕ) 
  (h1 : x - y = 6) 
  (h2 : x * y = 45) : 
  x = 11 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_classroom_activity_solution_l782_78249


namespace NUMINAMATH_GPT_smallest_whole_number_above_perimeter_triangle_l782_78270

theorem smallest_whole_number_above_perimeter_triangle (s : ℕ) (h1 : 12 < s) (h2 : s < 26) :
  53 = Nat.ceil ((7 + 19 + s : ℕ) / 1) := by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_above_perimeter_triangle_l782_78270


namespace NUMINAMATH_GPT_max_roses_l782_78275

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 7.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  ∃ n, n = 316 :=
by
  sorry

end NUMINAMATH_GPT_max_roses_l782_78275


namespace NUMINAMATH_GPT_perpendicular_line_through_center_l782_78253

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * x + y^2 - 3 = 0

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line we want to prove passes through the center of the circle and is perpendicular to the given line
def wanted_line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Main statement: Prove that the line that passes through the center of the circle and is perpendicular to the given line has the equation x - y + 1 = 0
theorem perpendicular_line_through_center (x y : ℝ) :
  (circle_eq (-1) 0) ∧ (line_eq x y) → wanted_line_eq x y :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_center_l782_78253


namespace NUMINAMATH_GPT_tangent_range_of_a_l782_78230

theorem tangent_range_of_a 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2 = 0)
  (A : ℝ × ℝ) 
  (A_eq : A = (1, 2)) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_range_of_a_l782_78230


namespace NUMINAMATH_GPT_two_pow_1000_mod_17_l782_78232

theorem two_pow_1000_mod_17 : 2^1000 % 17 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_pow_1000_mod_17_l782_78232


namespace NUMINAMATH_GPT_find_k_l782_78238

theorem find_k (k : ℝ) : (∀ x y : ℝ, y = k * x + 3 → (x, y) = (2, 5)) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l782_78238


namespace NUMINAMATH_GPT_solve_linear_system_l782_78292

/-- Let x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈ be real numbers that satisfy the following system of equations:
1. x₁ + x₂ + x₃ = 6
2. x₂ + x₃ + x₄ = 9
3. x₃ + x₄ + x₅ = 3
4. x₄ + x₅ + x₆ = -3
5. x₅ + x₆ + x₇ = -9
6. x₆ + x₇ + x₈ = -6
7. x₇ + x₈ + x₁ = -2
8. x₈ + x₁ + x₂ = 2
Prove that the solution is
  x₁ = 1, x₂ = 2, x₃ = 3, x₄ = 4, x₅ = -4, x₆ = -3, x₇ = -2, x₈ = -1
-/
theorem solve_linear_system :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
  x₁ + x₂ + x₃ = 6 →
  x₂ + x₃ + x₄ = 9 →
  x₃ + x₄ + x₅ = 3 →
  x₄ + x₅ + x₆ = -3 →
  x₅ + x₆ + x₇ = -9 →
  x₆ + x₇ + x₈ = -6 →
  x₇ + x₈ + x₁ = -2 →
  x₈ + x₁ + x₂ = 2 →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  intros x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, the proof steps would go
  sorry

end NUMINAMATH_GPT_solve_linear_system_l782_78292


namespace NUMINAMATH_GPT_four_genuine_coin_probability_l782_78209

noncomputable def probability_all_genuine_given_equal_weight : ℚ :=
  let total_coins := 20
  let genuine_coins := 12
  let counterfeit_coins := 8

  -- Calculate the probability of selecting two genuine coins from total coins
  let prob_first_pair_genuine := (genuine_coins / total_coins) * 
                                    ((genuine_coins - 1) / (total_coins - 1))

  -- Updating remaining counts after selecting the first pair
  let remaining_genuine_coins := genuine_coins - 2
  let remaining_total_coins := total_coins - 2

  -- Calculate the probability of selecting another two genuine coins
  let prob_second_pair_genuine := (remaining_genuine_coins / remaining_total_coins) * 
                                    ((remaining_genuine_coins - 1) / (remaining_total_coins - 1))

  -- Probability of A ∩ B
  let prob_A_inter_B := prob_first_pair_genuine * prob_second_pair_genuine

  -- Assuming prob_B represents the weighted probabilities including complexities
  let prob_B := (110 / 1077) -- This is an estimated combined probability for the purpose of this definition

  -- Conditional probability P(A | B)
  prob_A_inter_B / prob_B

theorem four_genuine_coin_probability :
  probability_all_genuine_given_equal_weight = 110 / 1077 := sorry

end NUMINAMATH_GPT_four_genuine_coin_probability_l782_78209


namespace NUMINAMATH_GPT_triangle_area_ratio_l782_78227

noncomputable def area_ratio (AD DC : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * AD * h / ((1 / 2) * DC * h)

theorem triangle_area_ratio (AD DC : ℝ) (h : ℝ) (condition1 : AD = 5) (condition2 : DC = 7) :
  area_ratio AD DC h = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l782_78227


namespace NUMINAMATH_GPT_abc_perfect_ratio_l782_78251

theorem abc_perfect_ratio {a b c : ℚ} (h1 : ∃ t : ℤ, a + b + c = t ∧ a^2 + b^2 + c^2 = t) :
  ∃ (p q : ℤ), (abc = p^3 / q^2) ∧ (IsCoprime p q) := 
sorry

end NUMINAMATH_GPT_abc_perfect_ratio_l782_78251


namespace NUMINAMATH_GPT_car_travel_time_l782_78214

theorem car_travel_time (speed distance : ℝ) (h₁ : speed = 65) (h₂ : distance = 455) :
  distance / speed = 7 :=
by
  -- We will invoke the conditions h₁ and h₂ to conclude the theorem
  sorry

end NUMINAMATH_GPT_car_travel_time_l782_78214


namespace NUMINAMATH_GPT_symmetric_point_of_A_is_correct_l782_78225

def symmetric_point_with_respect_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_of_A_is_correct :
  symmetric_point_with_respect_to_x_axis (3, 4) = (3, -4) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_of_A_is_correct_l782_78225


namespace NUMINAMATH_GPT_line_equation_passing_through_point_and_opposite_intercepts_l782_78201

theorem line_equation_passing_through_point_and_opposite_intercepts 
  : ∃ (a b : ℝ), (y = a * x) ∨ (x - y = b) :=
by
  use (3/2), (-1)
  sorry

end NUMINAMATH_GPT_line_equation_passing_through_point_and_opposite_intercepts_l782_78201


namespace NUMINAMATH_GPT_max_knights_between_knights_l782_78240

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end NUMINAMATH_GPT_max_knights_between_knights_l782_78240


namespace NUMINAMATH_GPT_Sarah_score_l782_78245

theorem Sarah_score (G S : ℕ) (h1 : S = G + 60) (h2 : (S + G) / 2 = 108) : S = 138 :=
by
  sorry

end NUMINAMATH_GPT_Sarah_score_l782_78245


namespace NUMINAMATH_GPT_number_of_participants_eq_14_l782_78260

theorem number_of_participants_eq_14 (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_participants_eq_14_l782_78260


namespace NUMINAMATH_GPT_measure_of_arc_BD_l782_78271

-- Definitions for conditions
def diameter (A B M : Type) : Prop := sorry -- Placeholder definition for diameter
def chord (C D M : Type) : Prop := sorry -- Placeholder definition for chord intersecting at point M
def angle_measure (A B C : Type) (angle_deg: ℝ) : Prop := sorry -- Placeholder for angle measure
def arc_measure (C B : Type) (arc_deg: ℝ) : Prop := sorry -- Placeholder for arc measure

-- Main theorem to prove
theorem measure_of_arc_BD
  (A B C D M : Type)
  (h_diameter : diameter A B M)
  (h_chord : chord C D M)
  (h_angle_CMB : angle_measure C M B 73)
  (h_arc_BC : arc_measure B C 110) :
  ∃ (arc_BD : ℝ), arc_BD = 144 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_arc_BD_l782_78271


namespace NUMINAMATH_GPT_frequency_of_a_is_3_l782_78222

def sentence : String := "Happy Teachers'Day!"

def frequency_of_a_in_sentence (s : String) : Nat :=
  s.foldl (λ acc c => if c = 'a' then acc + 1 else acc) 0

theorem frequency_of_a_is_3 : frequency_of_a_in_sentence sentence = 3 :=
  by
    sorry

end NUMINAMATH_GPT_frequency_of_a_is_3_l782_78222


namespace NUMINAMATH_GPT_remainder_mod_5_l782_78244

theorem remainder_mod_5 :
  let a := 1492
  let b := 1776
  let c := 1812
  let d := 1996
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_mod_5_l782_78244


namespace NUMINAMATH_GPT_polygon_sides_l782_78241

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1980) : n = 13 := 
by sorry

end NUMINAMATH_GPT_polygon_sides_l782_78241


namespace NUMINAMATH_GPT_find_x_in_inches_l782_78206

noncomputable def x_value (x : ℝ) : Prop :=
  let area_larger_square := (4 * x) ^ 2
  let area_smaller_square := (3 * x) ^ 2
  let area_triangle := (1 / 2) * (3 * x) * (4 * x)
  let total_area := area_larger_square + area_smaller_square + area_triangle
  total_area = 1100 ∧ x = Real.sqrt (1100 / 31)

theorem find_x_in_inches (x : ℝ) : x_value x :=
by sorry

end NUMINAMATH_GPT_find_x_in_inches_l782_78206


namespace NUMINAMATH_GPT_initial_masses_l782_78296

def area_of_base : ℝ := 15
def density_water : ℝ := 1
def density_ice : ℝ := 0.92
def change_in_water_level : ℝ := 5
def final_height_of_water : ℝ := 115

theorem initial_masses (m_ice m_water : ℝ) :
  m_ice = 675 ∧ m_water = 1050 :=
by
  -- Calculate the change in volume of water
  let delta_v := area_of_base * change_in_water_level

  -- Relate this volume change to the volume difference between ice and water
  let lhs := m_ice / density_ice - m_ice / density_water
  let eq1 := delta_v

  -- Solve for the mass of ice
  have h_ice : m_ice = 675 := 
  sorry

  -- Determine the final volume of water
  let final_volume_of_water := final_height_of_water * area_of_base

  -- Determine the initial mass of water
  let mass_of_water_total := density_water * final_volume_of_water
  let initial_mass_of_water :=
    mass_of_water_total - m_ice

  have h_water : m_water = 1050 := 
  sorry

  exact ⟨h_ice, h_water⟩

end NUMINAMATH_GPT_initial_masses_l782_78296


namespace NUMINAMATH_GPT_arith_seq_ratio_l782_78248

theorem arith_seq_ratio {a b : ℕ → ℕ} {S T : ℕ → ℕ}
  (h₁ : ∀ n, S n = (n * (2 * a n - a 1)) / 2)
  (h₂ : ∀ n, T n = (n * (2 * b n - b 1)) / 2)
  (h₃ : ∀ n, S n / T n = (5 * n + 3) / (2 * n + 7)) :
  (a 9 / b 9 = 88 / 41) :=
sorry

end NUMINAMATH_GPT_arith_seq_ratio_l782_78248


namespace NUMINAMATH_GPT_cost_per_person_l782_78298

-- Definitions based on conditions
def totalCost : ℕ := 13500
def numberOfFriends : ℕ := 15

-- Main statement
theorem cost_per_person : totalCost / numberOfFriends = 900 :=
by sorry

end NUMINAMATH_GPT_cost_per_person_l782_78298


namespace NUMINAMATH_GPT_exactly_two_talents_l782_78237

open Nat

def total_students : Nat := 50
def cannot_sing_students : Nat := 20
def cannot_dance_students : Nat := 35
def cannot_act_students : Nat := 15

theorem exactly_two_talents : 
  (total_students - cannot_sing_students) + 
  (total_students - cannot_dance_students) + 
  (total_students - cannot_act_students) - total_students = 30 := by
  sorry

end NUMINAMATH_GPT_exactly_two_talents_l782_78237


namespace NUMINAMATH_GPT_CarlYardAreaIsCorrect_l782_78247

noncomputable def CarlRectangularYardArea (post_count : ℕ) (distance_between_posts : ℕ) (long_side_factor : ℕ) :=
  let x := post_count / (2 * (1 + long_side_factor))
  let short_side := (x - 1) * distance_between_posts
  let long_side := (long_side_factor * x - 1) * distance_between_posts
  short_side * long_side

theorem CarlYardAreaIsCorrect :
  CarlRectangularYardArea 24 5 3 = 825 := 
by
  -- calculation steps if needed or
  sorry

end NUMINAMATH_GPT_CarlYardAreaIsCorrect_l782_78247


namespace NUMINAMATH_GPT_total_tagged_numbers_l782_78236

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end NUMINAMATH_GPT_total_tagged_numbers_l782_78236


namespace NUMINAMATH_GPT_length_of_major_axis_l782_78226

theorem length_of_major_axis (x y : ℝ) (h : (x^2 / 25) + (y^2 / 16) = 1) : 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_of_major_axis_l782_78226


namespace NUMINAMATH_GPT_unique_ab_for_interval_condition_l782_78259

theorem unique_ab_for_interval_condition : 
  ∃! (a b : ℝ), (∀ x, (0 ≤ x ∧ x ≤ 1) → |x^2 - a * x - b| ≤ 1 / 8) ∧ a = 1 ∧ b = -1 / 8 := by
  sorry

end NUMINAMATH_GPT_unique_ab_for_interval_condition_l782_78259


namespace NUMINAMATH_GPT_maciek_total_purchase_cost_l782_78252

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_maciek_total_purchase_cost_l782_78252


namespace NUMINAMATH_GPT_side_lengths_sum_eq_225_l782_78203

noncomputable def GX (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - x

noncomputable def GY (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - y

noncomputable def GZ (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - z

theorem side_lengths_sum_eq_225
  (x y z : ℝ)
  (h : GX x y z ^ 2 + GY x y z ^ 2 + GZ x y z ^ 2 = 75) :
  (x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2 = 225 := by {
  sorry
}

end NUMINAMATH_GPT_side_lengths_sum_eq_225_l782_78203


namespace NUMINAMATH_GPT_max_neg_integers_l782_78233

-- Definitions for the conditions
def areIntegers (a b c d e f : Int) : Prop := True
def sumOfProductsNeg (a b c d e f : Int) : Prop := (a * b + c * d * e * f) < 0

-- The theorem to prove
theorem max_neg_integers (a b c d e f : Int) (h1 : areIntegers a b c d e f) (h2 : sumOfProductsNeg a b c d e f) : 
  ∃ s : Nat, s = 4 := 
sorry

end NUMINAMATH_GPT_max_neg_integers_l782_78233


namespace NUMINAMATH_GPT_square_area_l782_78234

theorem square_area (x : ℝ) (A B C D E F : ℝ)
  (h1 : E = x / 3)
  (h2 : F = (2 * x) / 3)
  (h3 : abs (B - E) = 40)
  (h4 : abs (E - F) = 40)
  (h5 : abs (F - D) = 40) :
  x^2 = 2880 :=
by
  -- Main proof here
  sorry

end NUMINAMATH_GPT_square_area_l782_78234


namespace NUMINAMATH_GPT_total_jumps_l782_78243

def taehyung_jumps_per_day : ℕ := 56
def taehyung_days : ℕ := 3
def namjoon_jumps_per_day : ℕ := 35
def namjoon_days : ℕ := 4

theorem total_jumps : taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end NUMINAMATH_GPT_total_jumps_l782_78243


namespace NUMINAMATH_GPT_cooks_number_l782_78299

variable (C W : ℕ)

theorem cooks_number (h1 : 10 * C = 3 * W) (h2 : 14 * C = 3 * (W + 12)) : C = 9 :=
by
  sorry

end NUMINAMATH_GPT_cooks_number_l782_78299


namespace NUMINAMATH_GPT_factor_polynomial_sum_l782_78297

theorem factor_polynomial_sum (P Q : ℤ) :
  (∀ x : ℂ, (x^2 + 4*x + 5) ∣ (x^4 + P*x^2 + Q)) → P + Q = 19 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_factor_polynomial_sum_l782_78297


namespace NUMINAMATH_GPT_represent_259BC_as_neg259_l782_78215

def year_AD (n: ℤ) : ℤ := n

def year_BC (n: ℕ) : ℤ := -(n : ℤ)

theorem represent_259BC_as_neg259 : year_BC 259 = -259 := 
by 
  rw [year_BC]
  norm_num

end NUMINAMATH_GPT_represent_259BC_as_neg259_l782_78215


namespace NUMINAMATH_GPT_joel_age_when_dad_is_twice_l782_78246

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end NUMINAMATH_GPT_joel_age_when_dad_is_twice_l782_78246


namespace NUMINAMATH_GPT_remainder_when_M_divided_by_52_l782_78264

def M : Nat := 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

theorem remainder_when_M_divided_by_52 : M % 52 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_M_divided_by_52_l782_78264


namespace NUMINAMATH_GPT_each_niece_gets_fifty_ice_cream_sandwiches_l782_78266

theorem each_niece_gets_fifty_ice_cream_sandwiches
  (total_sandwiches : ℕ)
  (total_nieces : ℕ)
  (h1 : total_sandwiches = 1857)
  (h2 : total_nieces = 37) :
  (total_sandwiches / total_nieces) = 50 :=
by
  sorry

end NUMINAMATH_GPT_each_niece_gets_fifty_ice_cream_sandwiches_l782_78266


namespace NUMINAMATH_GPT_xiao_wang_fourth_place_l782_78250

section Competition
  -- Define the participants and positions
  inductive Participant
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

  inductive Position
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

  open Participant Position

  -- Conditions given in the problem
  variables
    (place : Participant → Position)
    (hA1 : place XiaoWang = First → place XiaoZhang = Third)
    (hA2 : place XiaoWang = First → place XiaoZhang ≠ Third)
    (hB1 : place XiaoLi = First → place XiaoZhao = Fourth)
    (hB2 : place XiaoLi = First → place XiaoZhao ≠ Fourth)
    (hC1 : place XiaoZhao = Second → place XiaoWang = Third)
    (hC2 : place XiaoZhao = Second → place XiaoWang ≠ Third)
    (no_ties : ∀ x y, place x = place y → x = y)
    (half_correct : ∀ p, (p = A → ((place XiaoWang = First ∨ place XiaoZhang = Third) ∧ (place XiaoWang ≠ First ∨ place XiaoZhang ≠ Third)))
                          ∧ (p = B → ((place XiaoLi = First ∨ place XiaoZhao = Fourth) ∧ (place XiaoLi ≠ First ∨ place XiaoZhao ≠ Fourth)))
                          ∧ (p = C → ((place XiaoZhao = Second ∨ place XiaoWang = Third) ∧ (place XiaoZhao ≠ Second ∨ place XiaoWang ≠ Third)))) 

  -- The goal to prove
  theorem xiao_wang_fourth_place : place XiaoWang = Fourth :=
  sorry
end Competition

end NUMINAMATH_GPT_xiao_wang_fourth_place_l782_78250


namespace NUMINAMATH_GPT_increase_in_surface_area_l782_78288

-- Define the edge length of the original cube and other conditions
variable (a : ℝ)

-- Define the increase in surface area problem
theorem increase_in_surface_area (h : 1 ≤ 27) : 
  let original_surface_area := 6 * a^2
  let smaller_cube_edge := a / 3
  let smaller_surface_area := 6 * (smaller_cube_edge)^2
  let total_smaller_surface_area := 27 * smaller_surface_area
  total_smaller_surface_area - original_surface_area = 12 * a^2 :=
by
  -- Provided the proof to satisfy Lean 4 syntax requirements to check for correctness
  sorry

end NUMINAMATH_GPT_increase_in_surface_area_l782_78288


namespace NUMINAMATH_GPT_samir_climbed_318_stairs_l782_78274

theorem samir_climbed_318_stairs 
  (S : ℕ)
  (h1 : ∀ {V : ℕ}, V = (S / 2) + 18 → S + V = 495) 
  (half_S : ∃ k : ℕ, S = k * 2) -- assumes S is even 
  : S = 318 := 
by
  sorry

end NUMINAMATH_GPT_samir_climbed_318_stairs_l782_78274


namespace NUMINAMATH_GPT_triangle_semicircle_l782_78242

noncomputable def triangle_semicircle_ratio : ℝ :=
  let AB := 8
  let BC := 6
  let CA := 2 * Real.sqrt 7
  let radius_AB := AB / 2
  let radius_BC := BC / 2
  let radius_CA := CA / 2
  let area_semicircle_AB := (1 / 2) * Real.pi * radius_AB ^ 2
  let area_semicircle_BC := (1 / 2) * Real.pi * radius_BC ^ 2
  let area_semicircle_CA := (1 / 2) * Real.pi * radius_CA ^ 2
  let area_triangle := AB * BC / 2
  let total_shaded_area := (area_semicircle_AB + area_semicircle_BC + area_semicircle_CA) - area_triangle
  let area_circle_CA := Real.pi * (radius_CA ^ 2)
  total_shaded_area / area_circle_CA

theorem triangle_semicircle : triangle_semicircle_ratio = 2 - (12 * Real.sqrt 3) / (7 * Real.pi) := by
  sorry

end NUMINAMATH_GPT_triangle_semicircle_l782_78242


namespace NUMINAMATH_GPT_solve_for_A_plus_B_l782_78211

theorem solve_for_A_plus_B (A B : ℤ) (h : ∀ ω, ω^2 + ω + 1 = 0 → ω^103 + A * ω + B = 0) : A + B = -1 :=
sorry

end NUMINAMATH_GPT_solve_for_A_plus_B_l782_78211


namespace NUMINAMATH_GPT_total_bill_l782_78279

def number_of_adults := 2
def number_of_children := 5
def meal_cost := 3

theorem total_bill : number_of_adults * meal_cost + number_of_children * meal_cost = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_bill_l782_78279


namespace NUMINAMATH_GPT_trajectory_eq_find_m_l782_78235

-- First problem: Trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  A = (1, 0) → B = (-1, 0) → 
  (dist P A) * (dist A B) = (dist P B) * (dist A B) → 
  P.snd ^ 2 = 4 * P.fst :=
by sorry

-- Second problem: Value of m
theorem find_m (P : ℝ × ℝ) (M N : ℝ × ℝ) (m : ℝ) :
  P.snd ^ 2 = 4 * P.fst → 
  M.snd = M.fst + m → 
  N.snd = N.fst + m →
  (M.fst - N.fst) * (M.snd - N.snd) + (N.snd - M.snd) * (N.fst - M.fst) = 0 →
  m ≠ 0 →
  m < 1 →
  m = -4 :=
by sorry

end NUMINAMATH_GPT_trajectory_eq_find_m_l782_78235


namespace NUMINAMATH_GPT_pages_needed_l782_78262

def new_cards : ℕ := 2
def old_cards : ℕ := 10
def cards_per_page : ℕ := 3
def total_cards : ℕ := new_cards + old_cards

theorem pages_needed : total_cards / cards_per_page = 4 := by
  sorry

end NUMINAMATH_GPT_pages_needed_l782_78262


namespace NUMINAMATH_GPT_f_2202_minus_f_2022_l782_78267

-- Definitions and conditions
def f : ℕ+ → ℕ+ := sorry -- The exact function is provided through conditions and will be proven property-wise.

axiom f_increasing {a b : ℕ+} : a < b → f a < f b
axiom f_range (n : ℕ+) : ∃ m : ℕ+, f n = ⟨m, sorry⟩ -- ensuring f maps to ℕ+
axiom f_property (n : ℕ+) : f (f n) = 3 * n

-- Prove the statement
theorem f_2202_minus_f_2022 : f 2202 - f 2022 = 1638 :=
by sorry

end NUMINAMATH_GPT_f_2202_minus_f_2022_l782_78267


namespace NUMINAMATH_GPT_heads_at_least_once_in_three_tosses_l782_78216

theorem heads_at_least_once_in_three_tosses :
  let total_outcomes := 8
  let all_tails_outcome := 1
  (1 - (all_tails_outcome / total_outcomes) = (7 / 8)) :=
by
  let total_outcomes := 8
  let all_tails_outcome := 1
  sorry

end NUMINAMATH_GPT_heads_at_least_once_in_three_tosses_l782_78216


namespace NUMINAMATH_GPT_find_certain_number_l782_78290

theorem find_certain_number : 
  ∃ (certain_number : ℕ), 1038 * certain_number = 173 * 240 ∧ certain_number = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l782_78290


namespace NUMINAMATH_GPT_speed_difference_valid_l782_78208

-- Definitions of the conditions
def speed (s : ℕ) : ℕ := s^2 + 2 * s

-- Theorem statement that needs to be proven
theorem speed_difference_valid : 
  (speed 5 - speed 3) = 20 :=
  sorry

end NUMINAMATH_GPT_speed_difference_valid_l782_78208


namespace NUMINAMATH_GPT_range_of_m_l782_78224

open Set

variable (f : ℝ → ℝ) (m : ℝ)

theorem range_of_m (h1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h2 : f (2 * m) > f (1 + m)) : m < 1 :=
by {
  -- The proof would go here.
  sorry
}

end NUMINAMATH_GPT_range_of_m_l782_78224


namespace NUMINAMATH_GPT_tips_earned_l782_78223

theorem tips_earned
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (tip_customers := total_customers - no_tip_customers)
  (total_tips := tip_customers * tip_amount)
  (h1 : total_customers = 9)
  (h2 : no_tip_customers = 5)
  (h3 : tip_amount = 8) :
  total_tips = 32 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tips_earned_l782_78223


namespace NUMINAMATH_GPT_unbroken_seashells_left_l782_78200

-- Definitions based on given conditions
def total_seashells : ℕ := 6
def cone_shells : ℕ := 3
def conch_shells : ℕ := 3
def broken_cone_shells : ℕ := 2
def broken_conch_shells : ℕ := 2
def given_away_conch_shells : ℕ := 1

-- Mathematical statement to prove the final count of unbroken seashells
theorem unbroken_seashells_left : 
  (cone_shells - broken_cone_shells) + (conch_shells - broken_conch_shells - given_away_conch_shells) = 1 :=
by 
  -- Calculation (steps omitted per instructions)
  sorry

end NUMINAMATH_GPT_unbroken_seashells_left_l782_78200


namespace NUMINAMATH_GPT_division_theorem_l782_78276

theorem division_theorem (k : ℕ) (h : k = 6) : 24 / k = 4 := by
  sorry

end NUMINAMATH_GPT_division_theorem_l782_78276


namespace NUMINAMATH_GPT_ratio_of_pens_to_notebooks_is_5_to_4_l782_78284

theorem ratio_of_pens_to_notebooks_is_5_to_4 (P N : ℕ) (hP : P = 50) (hN : N = 40) :
  (P / Nat.gcd P N) = 5 ∧ (N / Nat.gcd P N) = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_pens_to_notebooks_is_5_to_4_l782_78284


namespace NUMINAMATH_GPT_gcd_of_36_and_54_l782_78277

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end NUMINAMATH_GPT_gcd_of_36_and_54_l782_78277


namespace NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l782_78273

theorem sum_of_altitudes_of_triangle : 
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  altitude1 + altitude2 + altitude3 = (22 * Real.sqrt 73 + 48) / Real.sqrt 73 :=
by
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l782_78273


namespace NUMINAMATH_GPT_equal_water_and_alcohol_l782_78239

variable (a m : ℝ)

-- Conditions:
-- Cup B initially contains m liters of water.
-- Transfers as specified in the problem.

theorem equal_water_and_alcohol (h : m > 0) :
  (a * (m / (m + a)) = a * (m / (m + a))) :=
by
  sorry

end NUMINAMATH_GPT_equal_water_and_alcohol_l782_78239


namespace NUMINAMATH_GPT_bc_fraction_of_ad_l782_78210

theorem bc_fraction_of_ad
  {A B D C : Type}
  (length_AB length_BD length_AC length_CD length_AD length_BC : ℝ)
  (h1 : length_AB = 3 * length_BD)
  (h2 : length_AC = 4 * length_CD)
  (h3 : length_AD = length_AB + length_BD + length_CD)
  (h4 : length_BC = length_AC - length_AB) :
  length_BC / length_AD = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_bc_fraction_of_ad_l782_78210


namespace NUMINAMATH_GPT_seventieth_even_integer_l782_78265

theorem seventieth_even_integer : 2 * 70 = 140 :=
by
  sorry

end NUMINAMATH_GPT_seventieth_even_integer_l782_78265


namespace NUMINAMATH_GPT_find_f_neg_6_l782_78228

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 2) / Real.log 2 + (a - 1) * x + b else -(Real.log (-x + 2) / Real.log 2 + (a - 1) * -x + b)

theorem find_f_neg_6 (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = -f (-x) a b) 
                     (h2 : ∀ x : ℝ, x ≥ 0 → f x a b = Real.log (x + 2) / Real.log 2 + (a - 1) * x + b)
                     (h3 : f 2 a b = -1) : f (-6) 0 (-1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_6_l782_78228


namespace NUMINAMATH_GPT_max_contribution_l782_78257

theorem max_contribution (n : ℕ) (total : ℝ) (min_contribution : ℝ)
  (h1 : n = 12) (h2 : total = 20) (h3 : min_contribution = 1)
  (h4 : ∀ i : ℕ, i < n → min_contribution ≤ min_contribution) :
  ∃ max_contrib : ℝ, max_contrib = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_contribution_l782_78257


namespace NUMINAMATH_GPT_Yoongi_has_smaller_number_l782_78283

def Jungkook_number : ℕ := 6 + 3
def Yoongi_number : ℕ := 4

theorem Yoongi_has_smaller_number : Yoongi_number < Jungkook_number :=
by
  exact sorry

end NUMINAMATH_GPT_Yoongi_has_smaller_number_l782_78283


namespace NUMINAMATH_GPT_translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l782_78213

def f_translation : ℝ → ℝ :=
  fun x => (x - 1)^2 - 2

def f_quad (a x : ℝ) : ℝ :=
  x^2 - 2*a*x - 1

theorem translated_quadratic :
  ∀ x, f_translation x = (x - 1)^2 - 2 :=
by
  intro x
  simp [f_translation]

theorem range_of_translated_quadratic :
  ∀ x, 0 ≤ x ∧ x ≤ 4 → -2 ≤ f_translation x ∧ f_translation x ≤ 7 :=
by
  sorry

theorem min_value_on_interval :
  ∀ a, 
    (a ≤ 0 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a x ≥ -1)) ∧
    (0 < a ∧ a < 2 → f_quad a a = -a^2 - 1) ∧
    (a ≥ 2 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a 2 = -4*a + 3)) :=
by
  sorry

end NUMINAMATH_GPT_translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l782_78213


namespace NUMINAMATH_GPT_mary_sheep_purchase_l782_78204

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end NUMINAMATH_GPT_mary_sheep_purchase_l782_78204


namespace NUMINAMATH_GPT_algebra_problem_l782_78202

-- Definition of variable y
variable (y : ℝ)

-- Given the condition
axiom h : 2 * y^2 + 3 * y + 7 = 8

-- We need to prove that 4 * y^2 + 6 * y - 9 = -7 given the condition
theorem algebra_problem : 4 * y^2 + 6 * y - 9 = -7 :=
by sorry

end NUMINAMATH_GPT_algebra_problem_l782_78202


namespace NUMINAMATH_GPT_distance_from_dormitory_to_city_l782_78220

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h : (1/5) * D + (2/3) * D + 4 = D) : 
  D = 30 :=
sorry

end NUMINAMATH_GPT_distance_from_dormitory_to_city_l782_78220


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l782_78231

def M : Set ℝ := { x | (x - 3) / (x - 1) ≤ 0 }
def N : Set ℝ := { x | -6 * x^2 + 11 * x - 4 > 0 }

theorem intersection_of_M_and_N : M ∩ N = { x | 1 < x ∧ x < 4 / 3 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l782_78231


namespace NUMINAMATH_GPT_english_class_students_l782_78294

variables (e f s u v w : ℕ)

theorem english_class_students
  (h1 : e + u + v + w + f + s + 2 = 40)
  (h2 : e + u + v = 3 * (f + w))
  (h3 : e + u + w = 2 * (s + v)) : 
  e = 30 := 
sorry

end NUMINAMATH_GPT_english_class_students_l782_78294


namespace NUMINAMATH_GPT_prob_KH_then_Ace_l782_78254

noncomputable def probability_KH_then_Ace_drawn_in_sequence : ℚ :=
  let prob_first_card_is_KH := 1 / 52
  let prob_second_card_is_Ace := 4 / 51
  prob_first_card_is_KH * prob_second_card_is_Ace

theorem prob_KH_then_Ace : probability_KH_then_Ace_drawn_in_sequence = 1 / 663 := by
  sorry

end NUMINAMATH_GPT_prob_KH_then_Ace_l782_78254


namespace NUMINAMATH_GPT_Jonah_paid_commensurate_l782_78207

def price_per_pineapple (P : ℝ) :=
  let number_of_pineapples := 6
  let rings_per_pineapple := 12
  let total_rings := number_of_pineapples * rings_per_pineapple
  let price_per_4_rings := 5
  let price_per_ring := price_per_4_rings / 4
  let total_revenue := total_rings * price_per_ring
  let profit := 72
  total_revenue - number_of_pineapples * P = profit

theorem Jonah_paid_commensurate {P : ℝ} (h : price_per_pineapple P) :
  P = 3 :=
  sorry

end NUMINAMATH_GPT_Jonah_paid_commensurate_l782_78207


namespace NUMINAMATH_GPT_solve_for_A_l782_78293

def f (A B x : ℝ) : ℝ := A * x ^ 2 - 3 * B ^ 3
def g (B x : ℝ) : ℝ := 2 * B * x + B ^ 2

theorem solve_for_A (B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) :
  A = 3 / (16 / B + 8 + B ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l782_78293


namespace NUMINAMATH_GPT_shopkeeper_percentage_gain_l782_78219

theorem shopkeeper_percentage_gain 
    (original_price : ℝ) 
    (price_increase : ℝ) 
    (first_discount : ℝ) 
    (second_discount : ℝ)
    (new_price : ℝ) 
    (discounted_price1 : ℝ) 
    (final_price : ℝ) 
    (percentage_gain : ℝ) 
    (h1 : original_price = 100)
    (h2 : price_increase = original_price * 0.34)
    (h3 : new_price = original_price + price_increase)
    (h4 : first_discount = new_price * 0.10)
    (h5 : discounted_price1 = new_price - first_discount)
    (h6 : second_discount = discounted_price1 * 0.15)
    (h7 : final_price = discounted_price1 - second_discount)
    (h8 : percentage_gain = ((final_price - original_price) / original_price) * 100) :
    percentage_gain = 2.51 :=
by sorry

end NUMINAMATH_GPT_shopkeeper_percentage_gain_l782_78219


namespace NUMINAMATH_GPT_find_triplet_x_y_z_l782_78282

theorem find_triplet_x_y_z :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + 1 / (y + 1 / z : ℝ) = (10 : ℝ) / 7) ∧ (x = 1 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_triplet_x_y_z_l782_78282


namespace NUMINAMATH_GPT_relationship_abc_l782_78229

noncomputable def a (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin x) / x
noncomputable def b (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin (x^3)) / (x^3)
noncomputable def c (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := ((Real.sin x)^3) / (x^3)

theorem relationship_abc (x : ℝ) (hx : 0 < x ∧ x < 1) : b x hx > a x hx ∧ a x hx > c x hx :=
by
  sorry

end NUMINAMATH_GPT_relationship_abc_l782_78229


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l782_78256

-- Problem 1
theorem problem1 :
  -11 - (-8) + (-13) + 12 = -4 :=
  sorry

-- Problem 2
theorem problem2 :
  3 + 1 / 4 + (- (2 + 3 / 5)) + (5 + 3 / 4) - (8 + 2 / 5) = -2 :=
  sorry

-- Problem 3
theorem problem3 :
  -36 * (5 / 6 - 4 / 9 + 11 / 12) = -47 :=
  sorry

-- Problem 4
theorem problem4 :
  12 * (-1 / 6) + 27 / abs (3 ^ 2) + (-2) ^ 3 = -7 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l782_78256


namespace NUMINAMATH_GPT_smallest_constant_l782_78269

theorem smallest_constant (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + b^2 + a * b) / c^2 ≥ 3 / 4 :=
sorry

end NUMINAMATH_GPT_smallest_constant_l782_78269


namespace NUMINAMATH_GPT_product_of_two_numbers_l782_78287

theorem product_of_two_numbers (a b : ℤ) (h1 : Int.gcd a b = 10) (h2 : Int.lcm a b = 90) : a * b = 900 := 
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l782_78287


namespace NUMINAMATH_GPT_intervals_of_monotonicity_minimum_value_l782_78281

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem intervals_of_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x, 0 < x ∧ x ≤ 1 / a → f a x ≤ f a (1 / a)) ∧
  (∀ x, x ≥ 1 / a → f a x ≥ f a (1 / a)) :=
sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (a < Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = -a) ∧
  (a ≥ Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = Real.log 2 - 2 * a) :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_minimum_value_l782_78281
