import Mathlib

namespace NUMINAMATH_GPT_third_derivative_y_l1083_108379

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5 * x - 3)

theorem third_derivative_y (x : ℝ) : 
  (deriv^[3] y x) = -150 * x * Real.sin (5 * x - 3) + (30 - 125 * x^2) * Real.cos (5 * x - 3) :=
by
  sorry

end NUMINAMATH_GPT_third_derivative_y_l1083_108379


namespace NUMINAMATH_GPT_distance_between_circle_centers_l1083_108384

open Real

theorem distance_between_circle_centers :
  let center1 := (1 / 2, 0)
  let center2 := (0, 1 / 2)
  dist center1 center2 = sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_circle_centers_l1083_108384


namespace NUMINAMATH_GPT_number_of_77s_l1083_108356

theorem number_of_77s (a b : ℕ) :
  (∃ a : ℕ, 1015 = a + 3 * 77 ∧ a + 21 = 10)
  ∧ (∃ b : ℕ, 2023 = b + 6 * 77 + 2 * 777 ∧ b = 7)
  → 6 = 6 := 
by
    sorry

end NUMINAMATH_GPT_number_of_77s_l1083_108356


namespace NUMINAMATH_GPT_intersection_A_B_l1083_108354

def setA : Set ℝ := {x | x^2 - 1 > 0}
def setB : Set ℝ := {x | Real.log x / Real.log 2 < 1}

theorem intersection_A_B :
  {x | x ∈ setA ∧ x ∈ setB} = {x | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1083_108354


namespace NUMINAMATH_GPT_factorize_first_poly_factorize_second_poly_l1083_108395

variable (x m n : ℝ)

-- Proof statement for the first polynomial
theorem factorize_first_poly : x^2 + 14*x + 49 = (x + 7)^2 := 
by sorry

-- Proof statement for the second polynomial
theorem factorize_second_poly : (m - 1) + n^2 * (1 - m) = (m - 1) * (1 - n) * (1 + n) := 
by sorry

end NUMINAMATH_GPT_factorize_first_poly_factorize_second_poly_l1083_108395


namespace NUMINAMATH_GPT_negation_of_proposition_l1083_108334

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1083_108334


namespace NUMINAMATH_GPT_counter_example_not_power_of_4_for_25_l1083_108355

theorem counter_example_not_power_of_4_for_25 : ∃ n ≥ 2, n = 25 ∧ ¬ ∃ k : ℕ, 2 ^ (2 ^ n) % (2 ^ n - 1) = 4 ^ k :=
by {
  sorry
}

end NUMINAMATH_GPT_counter_example_not_power_of_4_for_25_l1083_108355


namespace NUMINAMATH_GPT_max_sum_two_digit_primes_l1083_108374

theorem max_sum_two_digit_primes : (89 + 97) = 186 := 
by
  sorry

end NUMINAMATH_GPT_max_sum_two_digit_primes_l1083_108374


namespace NUMINAMATH_GPT_captain_age_eq_your_age_l1083_108370

-- Represent the conditions as assumptions
variables (your_age : ℕ) -- You, the captain, have an age as a natural number

-- Define the statement
theorem captain_age_eq_your_age (H_cap : ∀ captain, captain = your_age) : ∀ captain, captain = your_age := by
  sorry

end NUMINAMATH_GPT_captain_age_eq_your_age_l1083_108370


namespace NUMINAMATH_GPT_students_not_in_either_l1083_108320

theorem students_not_in_either (total_students chemistry_students biology_students both_subjects neither_subjects : ℕ) 
  (h1 : total_students = 120) 
  (h2 : chemistry_students = 75) 
  (h3 : biology_students = 50) 
  (h4 : both_subjects = 15) 
  (h5 : neither_subjects = total_students - (chemistry_students - both_subjects + biology_students - both_subjects + both_subjects)) : 
  neither_subjects = 10 := 
by 
  sorry

end NUMINAMATH_GPT_students_not_in_either_l1083_108320


namespace NUMINAMATH_GPT_counties_rained_on_monday_l1083_108361

theorem counties_rained_on_monday : 
  ∀ (M T R_no_both R_both : ℝ),
    T = 0.55 → 
    R_no_both = 0.35 →
    R_both = 0.60 →
    (M + T - R_both = 1 - R_no_both) →
    M = 0.70 :=
by
  intros M T R_no_both R_both hT hR_no_both hR_both hInclusionExclusion
  sorry

end NUMINAMATH_GPT_counties_rained_on_monday_l1083_108361


namespace NUMINAMATH_GPT_mean_of_other_two_l1083_108332

theorem mean_of_other_two (a b c d e f : ℕ) (h : a = 1867 ∧ b = 1993 ∧ c = 2019 ∧ d = 2025 ∧ e = 2109 ∧ f = 2121):
  ((a + b + c + d + e + f) - (4 * 2008)) / 2 = 2051 := by
  sorry

end NUMINAMATH_GPT_mean_of_other_two_l1083_108332


namespace NUMINAMATH_GPT_interior_surface_area_is_correct_l1083_108318

-- Define the original dimensions of the rectangular sheet
def original_length : ℕ := 40
def original_width : ℕ := 50

-- Define the side length of the square corners
def corner_side : ℕ := 10

-- Define the area of the original sheet
def area_original : ℕ := original_length * original_width

-- Define the area of one square corner
def area_corner : ℕ := corner_side * corner_side

-- Define the total area removed by all four corners
def area_removed : ℕ := 4 * area_corner

-- Define the remaining area after the corners are removed
def area_remaining : ℕ := area_original - area_removed

-- The theorem to be proved
theorem interior_surface_area_is_correct : area_remaining = 1600 := by
  sorry

end NUMINAMATH_GPT_interior_surface_area_is_correct_l1083_108318


namespace NUMINAMATH_GPT_f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l1083_108369

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x / (1 + x) else 1 / (1 + x)

theorem f_property (x : ℝ) (hx : 0 < x) : 
  f x = f (1 / x) :=
by
  sorry

theorem f_equals_when_x_lt_1 (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1) : 
  f x = 1 / (1 + x) :=
by
  sorry

theorem f_equals_when_x_gt_1 (x : ℝ) (hx : 1 < x) : 
  f x = x / (1 + x) :=
by
  sorry

end NUMINAMATH_GPT_f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l1083_108369


namespace NUMINAMATH_GPT_maximize_angle_l1083_108345

structure Point where
  x : ℝ
  y : ℝ

def A (a : ℝ) : Point := ⟨0, a⟩
def B (b : ℝ) : Point := ⟨0, b⟩

theorem maximize_angle
  (a b : ℝ)
  (h : a > b)
  (h₁ : b > 0)
  : ∃ (C : Point), C = ⟨Real.sqrt (a * b), 0⟩ :=
sorry

end NUMINAMATH_GPT_maximize_angle_l1083_108345


namespace NUMINAMATH_GPT_compute_expression_l1083_108327

theorem compute_expression : 45 * (28 + 72) + 55 * 45 = 6975 := 
  by
  sorry

end NUMINAMATH_GPT_compute_expression_l1083_108327


namespace NUMINAMATH_GPT_solve_for_x_l1083_108393

theorem solve_for_x : ∃ x : ℤ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1083_108393


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1083_108375

noncomputable def common_difference (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

theorem arithmetic_sequence_properties :
  ∃ d : ℚ, d = 5 / 9 ∧ ∃ S : ℚ, S = -29 / 3 ∧
  ∀ n : ℕ, ∃ a₁ a₅ a₈ : ℚ, a₁ = -3 ∧
    a₅ = common_difference a₁ d 5 ∧
    a₈ = common_difference a₁ d 8 ∧ 
    11 * a₅ = 5 * a₈ - 13 ∧
    S = (n / 2) * (2 * a₁ + (n - 1) * d) ∧
    n = 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1083_108375


namespace NUMINAMATH_GPT_no_sol_for_eq_xn_minus_yn_eq_2k_l1083_108397

theorem no_sol_for_eq_xn_minus_yn_eq_2k (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_n : n > 2) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^n - y^n = 2^k := 
sorry

end NUMINAMATH_GPT_no_sol_for_eq_xn_minus_yn_eq_2k_l1083_108397


namespace NUMINAMATH_GPT_chicken_rabbit_problem_l1083_108360

theorem chicken_rabbit_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_chicken_rabbit_problem_l1083_108360


namespace NUMINAMATH_GPT_union_of_A_and_B_l1083_108316

-- Define the sets A and B as given in the problem
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 4}

-- State the theorem to prove that A ∪ B = {0, 1, 2, 4}
theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1083_108316


namespace NUMINAMATH_GPT_min_value_of_sum_l1083_108330

theorem min_value_of_sum (a b : ℝ) (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 6) :
  a + b ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l1083_108330


namespace NUMINAMATH_GPT_cara_neighbors_l1083_108357

def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem cara_neighbors : number_of_pairs 7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_cara_neighbors_l1083_108357


namespace NUMINAMATH_GPT_sum_of_x_values_l1083_108342

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_values_l1083_108342


namespace NUMINAMATH_GPT_find_S6_l1083_108326

-- sum of the first n terms of an arithmetic sequence
variable (S : ℕ → ℕ)

-- Given conditions
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- Theorem statement
theorem find_S6 : S 6 = 36 := sorry

end NUMINAMATH_GPT_find_S6_l1083_108326


namespace NUMINAMATH_GPT_power_expression_l1083_108376

variable {x : ℂ} -- Define x as a complex number

theorem power_expression (
  h : x - 1/x = 2 * Complex.I * Real.sqrt 2
) : x^(2187:ℕ) - 1/x^(2187:ℕ) = -22 * Complex.I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_power_expression_l1083_108376


namespace NUMINAMATH_GPT_sum_adjacent_angles_pentagon_l1083_108347

theorem sum_adjacent_angles_pentagon (n : ℕ) (θ : ℕ) (hn : n = 5) (hθ : θ = 40) :
  let exterior_angle := 360 / n
  let new_adjacent_angle := 180 - (exterior_angle + θ)
  let sum_adjacent_angles := n * new_adjacent_angle
  sum_adjacent_angles = 340 := by
  sorry

end NUMINAMATH_GPT_sum_adjacent_angles_pentagon_l1083_108347


namespace NUMINAMATH_GPT_ravi_money_l1083_108324

theorem ravi_money (n q d : ℕ) (h1 : q = n + 2) (h2 : d = q + 4) (h3 : n = 6) :
  (n * 5 + q * 25 + d * 10) = 350 := by
  sorry

end NUMINAMATH_GPT_ravi_money_l1083_108324


namespace NUMINAMATH_GPT_angle_double_of_supplementary_l1083_108364

theorem angle_double_of_supplementary (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 2 * (180 - x)) : x = 120 :=
sorry

end NUMINAMATH_GPT_angle_double_of_supplementary_l1083_108364


namespace NUMINAMATH_GPT_simplify_P_eq_l1083_108377

noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

theorem simplify_P_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy: x ≠ y) : P x y = x / y := 
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_simplify_P_eq_l1083_108377


namespace NUMINAMATH_GPT_max_min_values_monotonocity_l1083_108350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - (1 / 2) * x ^ 2

theorem max_min_values (a : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (ha : a = 1) : 
  f a 0 = 0 ∧ f a 1 = 1 / 2 ∧ f a (1 / 3) = -1 / 54 :=
sorry

theorem monotonocity (a : ℝ) (hx : 0 < x ∧ x < (1 / (6 * a))) (ha : 0 < a) : 
  (3 * a * x ^ 2 - x) < 0 → (f a x) < (f a 0) :=
sorry

end NUMINAMATH_GPT_max_min_values_monotonocity_l1083_108350


namespace NUMINAMATH_GPT_intersection_complement_l1083_108343

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end NUMINAMATH_GPT_intersection_complement_l1083_108343


namespace NUMINAMATH_GPT_sin_480_eq_sqrt3_div_2_l1083_108358

theorem sin_480_eq_sqrt3_div_2 : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_480_eq_sqrt3_div_2_l1083_108358


namespace NUMINAMATH_GPT_lcm_9_14_l1083_108367

/-- Given the definition of the least common multiple (LCM) and the prime factorizations,
    prove that the LCM of 9 and 14 is 126. -/
theorem lcm_9_14 : Int.lcm 9 14 = 126 := by
  sorry

end NUMINAMATH_GPT_lcm_9_14_l1083_108367


namespace NUMINAMATH_GPT_john_increased_bench_press_factor_l1083_108346

theorem john_increased_bench_press_factor (initial current : ℝ) (decrease_percent : ℝ) 
  (h_initial : initial = 500) 
  (h_current : current = 300) 
  (h_decrease : decrease_percent = 0.80) : 
  current / (initial * (1 - decrease_percent)) = 3 := 
by
  -- We'll provide the proof here later
  sorry

end NUMINAMATH_GPT_john_increased_bench_press_factor_l1083_108346


namespace NUMINAMATH_GPT_fraction_identity_l1083_108328

theorem fraction_identity (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x / y := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_identity_l1083_108328


namespace NUMINAMATH_GPT_interest_difference_l1083_108385

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P * (1 + r)^t - P

theorem interest_difference : 
  simple_interest 500 0.20 2 - (500 * (1 + 0.20)^2 - 500) = 20 := by
  sorry

end NUMINAMATH_GPT_interest_difference_l1083_108385


namespace NUMINAMATH_GPT_find_possible_values_a_l1083_108371

theorem find_possible_values_a :
  ∃ a : ℤ, ∃ b : ℤ, ∃ c : ℤ, 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ∧
  ((b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) ↔ 
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_a_l1083_108371


namespace NUMINAMATH_GPT_correct_operation_l1083_108380

variables (a b : ℝ)

theorem correct_operation : (3 * a + b) * (3 * a - b) = 9 * a^2 - b^2 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1083_108380


namespace NUMINAMATH_GPT_quadratic_discriminant_l1083_108313

theorem quadratic_discriminant (k : ℝ) :
  (∃ x : ℝ, k*x^2 + 2*x - 1 = 0) ∧ (∀ a b, (a*x + b) ^ 2 = a^2 * x^2 + 2 * a * b * x + b^2) ∧
  (a = k) ∧ (b = 2) ∧ (c = -1) ∧ ((b^2 - 4 * a * c = 0) → (4 + 4 * k = 0)) → k = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1083_108313


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l1083_108300

open Nat

def arithmetic_sequence (a : ℕ → ℚ) (a1 d : ℚ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_seq_problem :
  ∃ (a : ℕ → ℚ) (a1 d : ℚ),
    (arithmetic_sequence a a1 d) ∧
    (a 2 + a 3 + a 4 = 3) ∧
    (a 7 = 8) ∧
    (a 11 = 15) :=
  sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l1083_108300


namespace NUMINAMATH_GPT_unit_digit_of_fourth_number_l1083_108340

theorem unit_digit_of_fourth_number
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 % 10 = 4)
  (h2 : n2 % 10 = 8)
  (h3 : n3 % 10 = 3)
  (h4 : (n1 * n2 * n3 * n4) % 10 = 8) : 
  n4 % 10 = 3 :=
sorry

end NUMINAMATH_GPT_unit_digit_of_fourth_number_l1083_108340


namespace NUMINAMATH_GPT_find_x_l1083_108378

-- Define the mean of three numbers
def mean_three (a b c : ℕ) : ℚ := (a + b + c) / 3

-- Define the mean of two numbers
def mean_two (x y : ℕ) : ℚ := (x + y) / 2

-- Main theorem: value of x that satisfies the given condition
theorem find_x : 
  (mean_three 6 9 18) = (mean_two x 15) → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1083_108378


namespace NUMINAMATH_GPT_vectors_opposite_directions_l1083_108310

variable {V : Type*} [AddCommGroup V]

theorem vectors_opposite_directions (a b : V) (h : a + 4 • b = 0) (ha : a ≠ 0) (hb : b ≠ 0) : a = -4 • b :=
by sorry

end NUMINAMATH_GPT_vectors_opposite_directions_l1083_108310


namespace NUMINAMATH_GPT_profit_percentage_l1083_108398

theorem profit_percentage (cost_price selling_price marked_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : selling_price = 0.90 * marked_price)
  (h3 : selling_price = 65.97) :
  ((selling_price - cost_price) / cost_price) * 100 = 38.88 := 
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1083_108398


namespace NUMINAMATH_GPT_circle_properties_l1083_108359

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

theorem circle_properties (x y : ℝ) :
  circle_center x y ↔ ((x - 2)^2 + y^2 = 2^2) ∧ ((2, 0) = (2, 0)) :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l1083_108359


namespace NUMINAMATH_GPT_least_adjacent_probability_l1083_108368

theorem least_adjacent_probability (n : ℕ) 
    (h₀ : 0 < n)
    (h₁ : (∀ m : ℕ, 0 < m ∧ m < n → (4 * m^2 - 4 * m + 8) / (m^2 * (m^2 - 1)) ≥ 1 / 2015)) : 
    (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1)) < 1 / 2015 := by
  sorry

end NUMINAMATH_GPT_least_adjacent_probability_l1083_108368


namespace NUMINAMATH_GPT_tiles_with_no_gaps_l1083_108344

-- Define the condition that the tiling consists of regular octagons
def regular_octagon_internal_angle := 135

-- Define the other regular polygons
def regular_triangle_internal_angle := 60
def regular_square_internal_angle := 90
def regular_pentagon_internal_angle := 108
def regular_hexagon_internal_angle := 120

-- The proposition to be proved: A flat surface without gaps
-- can be achieved using regular squares and regular octagons.
theorem tiles_with_no_gaps :
  ∃ (m n : ℕ), regular_octagon_internal_angle * m + regular_square_internal_angle * n = 360 :=
sorry

end NUMINAMATH_GPT_tiles_with_no_gaps_l1083_108344


namespace NUMINAMATH_GPT_verify_final_weights_l1083_108303

-- Define the initial weights
def initial_bench_press : ℝ := 500
def initial_squat : ℝ := 400
def initial_deadlift : ℝ := 600

-- Define the weight adjustment transformations for each exercise
def transform_bench_press (w : ℝ) : ℝ :=
  let w1 := w * 0.20
  let w2 := w1 * 1.60
  let w3 := w2 * 0.80
  let w4 := w3 * 3
  w4

def transform_squat (w : ℝ) : ℝ :=
  let w1 := w * 0.50
  let w2 := w1 * 1.40
  let w3 := w2 * 2
  w3

def transform_deadlift (w : ℝ) : ℝ :=
  let w1 := w * 0.70
  let w2 := w1 * 1.80
  let w3 := w2 * 0.60
  let w4 := w3 * 1.50
  w4

-- The final calculated weights for verification
def final_bench_press : ℝ := 384
def final_squat : ℝ := 560
def final_deadlift : ℝ := 680.4

-- Statement of the problem: prove that the transformed weights are as calculated
theorem verify_final_weights : 
  transform_bench_press initial_bench_press = final_bench_press ∧ 
  transform_squat initial_squat = final_squat ∧ 
  transform_deadlift initial_deadlift = final_deadlift := 
by 
  sorry

end NUMINAMATH_GPT_verify_final_weights_l1083_108303


namespace NUMINAMATH_GPT_floor_div_eq_floor_div_floor_l1083_108394

theorem floor_div_eq_floor_div_floor {α : ℝ} {d : ℕ} (h₁ : 0 < α) : 
  (⌊α / d⌋ = ⌊⌊α⌋ / d⌋) := 
sorry

end NUMINAMATH_GPT_floor_div_eq_floor_div_floor_l1083_108394


namespace NUMINAMATH_GPT_solve_for_a_l1083_108341

-- Define the lines
def l1 (x y : ℝ) := x + y - 2 = 0
def l2 (x y a : ℝ) := 2 * x + a * y - 3 = 0

-- Define orthogonality condition
def perpendicular (m₁ m₂ : ℝ) := m₁ * m₂ = -1

-- The theorem to prove
theorem solve_for_a (a : ℝ) :
  (∀ x y : ℝ, l1 x y → ∀ x y : ℝ, l2 x y a → perpendicular (-1) (-2 / a)) → a = 2 := 
sorry

end NUMINAMATH_GPT_solve_for_a_l1083_108341


namespace NUMINAMATH_GPT_total_crackers_l1083_108396

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end NUMINAMATH_GPT_total_crackers_l1083_108396


namespace NUMINAMATH_GPT_trig_identity_solution_l1083_108323

open Real

theorem trig_identity_solution :
  sin (15 * (π / 180)) * cos (45 * (π / 180)) + sin (105 * (π / 180)) * sin (135 * (π / 180)) = sqrt 3 / 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_trig_identity_solution_l1083_108323


namespace NUMINAMATH_GPT_temperature_range_l1083_108391

-- Conditions: highest temperature and lowest temperature
def highest_temp : ℝ := 5
def lowest_temp : ℝ := -2
variable (t : ℝ) -- given temperature on February 1, 2018

-- Proof problem statement
theorem temperature_range : lowest_temp ≤ t ∧ t ≤ highest_temp :=
sorry

end NUMINAMATH_GPT_temperature_range_l1083_108391


namespace NUMINAMATH_GPT_matrix_vector_subtraction_l1083_108325

open Matrix

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def matrix_mul_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

theorem matrix_vector_subtraction (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ)
  (hv : matrix_mul_vector M v = ![4, 6])
  (hw : matrix_mul_vector M w = ![5, -4]) :
  matrix_mul_vector M (v - (2 : ℝ) • w) = ![-6, 14] :=
sorry

end NUMINAMATH_GPT_matrix_vector_subtraction_l1083_108325


namespace NUMINAMATH_GPT_workout_goal_l1083_108322

def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19
def wednesday_situps_needed : ℕ := 59

theorem workout_goal : monday_situps + tuesday_situps + wednesday_situps_needed = 90 := by
  sorry

end NUMINAMATH_GPT_workout_goal_l1083_108322


namespace NUMINAMATH_GPT_range_of_y_when_x_3_l1083_108386

variable (a c : ℝ)

theorem range_of_y_when_x_3 (h1 : -4 ≤ a + c ∧ a + c ≤ -1) (h2 : -1 ≤ 4 * a + c ∧ 4 * a + c ≤ 5) :
  -1 ≤ 9 * a + c ∧ 9 * a + c ≤ 20 :=
sorry

end NUMINAMATH_GPT_range_of_y_when_x_3_l1083_108386


namespace NUMINAMATH_GPT_difference_of_squares_example_l1083_108302

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end NUMINAMATH_GPT_difference_of_squares_example_l1083_108302


namespace NUMINAMATH_GPT_set_intersection_l1083_108363

def U : Set ℝ := Set.univ
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}
def C_U_B : Set ℝ := {x | x < 2}

theorem set_intersection :
  A ∩ C_U_B = {-1, 0, 1} :=
sorry

end NUMINAMATH_GPT_set_intersection_l1083_108363


namespace NUMINAMATH_GPT_sum_of_squares_eq_l1083_108353

theorem sum_of_squares_eq :
  ∀ (M G D : ℝ), 
  (M = G / 3) → 
  (G = 450) → 
  (D = 2 * G) → 
  (M^2 + G^2 + D^2 = 1035000) :=
by
  intros M G D hM hG hD
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_l1083_108353


namespace NUMINAMATH_GPT_petya_prevents_vasya_l1083_108390

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end NUMINAMATH_GPT_petya_prevents_vasya_l1083_108390


namespace NUMINAMATH_GPT_parallel_perpendicular_trans_l1083_108337

variables {Plane Line : Type}

-- Definitions in terms of lines and planes
variables (α β γ : Plane) (a b : Line)

-- Definitions of parallel and perpendicular
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- The mathematical statement to prove
theorem parallel_perpendicular_trans :
  (parallel a b) → (perpendicular b α) → (perpendicular a α) :=
by sorry

end NUMINAMATH_GPT_parallel_perpendicular_trans_l1083_108337


namespace NUMINAMATH_GPT_day_of_week_100_days_from_wednesday_l1083_108307

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end NUMINAMATH_GPT_day_of_week_100_days_from_wednesday_l1083_108307


namespace NUMINAMATH_GPT_no_positive_n_l1083_108329

theorem no_positive_n :
  ¬ ∃ (n : ℕ) (n_pos : n > 0) (a b : ℕ) (a_sd : a < 10) (b_sd : b < 10), 
    (1234 - n) * b = (6789 - n) * a :=
by 
  sorry

end NUMINAMATH_GPT_no_positive_n_l1083_108329


namespace NUMINAMATH_GPT_math_proof_problem_l1083_108305

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ), (x > 12) ∧ ((x - 5) / 12 = 5 / (x - 12)) ∧ (x = 17)

theorem math_proof_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1083_108305


namespace NUMINAMATH_GPT_find_multiplying_number_l1083_108314

variable (a b : ℤ)

theorem find_multiplying_number (h : a^2 * b = 3 * (4 * a + 2)) (ha : a = 1) :
  b = 18 := by
  sorry

end NUMINAMATH_GPT_find_multiplying_number_l1083_108314


namespace NUMINAMATH_GPT_value_of_a_l1083_108362

theorem value_of_a (a : ℕ) : (∃ (x1 x2 x3 : ℤ),
  abs (abs (x1 - 3) - 1) = a ∧
  abs (abs (x2 - 3) - 1) = a ∧
  abs (abs (x3 - 3) - 1) = a ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1083_108362


namespace NUMINAMATH_GPT_base_b_representation_l1083_108308

theorem base_b_representation (b : ℕ) : (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 → b = 14 := 
sorry

end NUMINAMATH_GPT_base_b_representation_l1083_108308


namespace NUMINAMATH_GPT_total_shaded_area_of_square_carpet_l1083_108399

theorem total_shaded_area_of_square_carpet :
  ∀ (S T : ℝ),
    (9 / S = 3) →
    (S / T = 3) →
    (8 * T^2 + S^2 = 17) :=
by
  intros S T h1 h2
  sorry

end NUMINAMATH_GPT_total_shaded_area_of_square_carpet_l1083_108399


namespace NUMINAMATH_GPT_savings_after_increase_l1083_108301

-- Conditions
def salary : ℕ := 5000
def initial_savings_ratio : ℚ := 0.20
def expense_increase_ratio : ℚ := 1.20

-- Derived initial values
def initial_savings : ℚ := initial_savings_ratio * salary
def initial_expenses : ℚ := ((1 : ℚ) - initial_savings_ratio) * salary

-- New expenses after increase
def new_expenses : ℚ := expense_increase_ratio * initial_expenses

-- Savings after expense increase
def final_savings : ℚ := salary - new_expenses

theorem savings_after_increase : final_savings = 200 := by
  sorry

end NUMINAMATH_GPT_savings_after_increase_l1083_108301


namespace NUMINAMATH_GPT_trevor_quarters_counted_l1083_108319

-- Define the conditions from the problem
variable (Q D : ℕ) 
variable (total_coins : ℕ := 77)
variable (excess : ℕ := 48)

-- Use the conditions to assert the existence of quarters and dimes such that the totals align with the given constraints
theorem trevor_quarters_counted : (Q + D = total_coins) ∧ (D = Q + excess) → Q = 29 :=
by
  -- Add sorry to skip the actual proof, as we are only writing the statement
  sorry

end NUMINAMATH_GPT_trevor_quarters_counted_l1083_108319


namespace NUMINAMATH_GPT_program1_values_program2_values_l1083_108335

theorem program1_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧
  a = -5 ∧ b = 8 ∧ c = 8 :=
by sorry

theorem program2_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧ c = a ∧
  a = -5 ∧ b = 8 ∧ c = -5 :=
by sorry

end NUMINAMATH_GPT_program1_values_program2_values_l1083_108335


namespace NUMINAMATH_GPT_find_B_from_period_l1083_108309

theorem find_B_from_period (A B C D : ℝ) (h : B ≠ 0) (period_condition : 2 * |2 * π / B| = 4 * π) : B = 1 := sorry

end NUMINAMATH_GPT_find_B_from_period_l1083_108309


namespace NUMINAMATH_GPT_empty_square_exists_in_4x4_l1083_108315

theorem empty_square_exists_in_4x4  :
  ∀ (points: Finset (Fin 4 × Fin 4)), points.card = 15 → 
  ∃ (i j : Fin 4), (i, j) ∉ points :=
by
  sorry

end NUMINAMATH_GPT_empty_square_exists_in_4x4_l1083_108315


namespace NUMINAMATH_GPT_sum_of_solutions_l1083_108387

theorem sum_of_solutions (a : ℝ) (h : 0 < a ∧ a < 1) :
  let x1 := 3 + a
  let x2 := 3 - a
  let x3 := 1 + a
  let x4 := 1 - a
  x1 + x2 + x3 + x4 = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1083_108387


namespace NUMINAMATH_GPT_find_x0_l1083_108348

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x0_l1083_108348


namespace NUMINAMATH_GPT_chapters_page_difference_l1083_108365

def chapter1_pages : ℕ := 37
def chapter2_pages : ℕ := 80

theorem chapters_page_difference : chapter2_pages - chapter1_pages = 43 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_chapters_page_difference_l1083_108365


namespace NUMINAMATH_GPT_percentage_increase_correct_l1083_108312

-- Define the highest and lowest scores as given conditions.
def highest_score : ℕ := 92
def lowest_score : ℕ := 65

-- State that the percentage increase calculation will result in 41.54%
theorem percentage_increase_correct :
  ((highest_score - lowest_score) * 100) / lowest_score = 4154 / 100 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_correct_l1083_108312


namespace NUMINAMATH_GPT_problem_solution_l1083_108331

theorem problem_solution (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x^2 + (b - 3) * x + 3) →
  (∀ x : ℝ, f x = f (-x)) →
  (a^2 - 2 = -a) →
  a + b = 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_problem_solution_l1083_108331


namespace NUMINAMATH_GPT_smallest_satisfying_N_is_2520_l1083_108383

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end NUMINAMATH_GPT_smallest_satisfying_N_is_2520_l1083_108383


namespace NUMINAMATH_GPT_cardinals_count_l1083_108306

theorem cardinals_count (C R B S : ℕ) 
  (hR : R = 4 * C)
  (hB : B = 2 * C)
  (hS : S = 3 * C + 1)
  (h_total : C + R + B + S = 31) :
  C = 3 :=
by
  sorry

end NUMINAMATH_GPT_cardinals_count_l1083_108306


namespace NUMINAMATH_GPT_sum_of_coordinates_of_X_l1083_108366

theorem sum_of_coordinates_of_X 
  (X Y Z : ℝ × ℝ)
  (h1 : dist X Z / dist X Y = 1 / 2)
  (h2 : dist Z Y / dist X Y = 1 / 2)
  (hY : Y = (1, 7))
  (hZ : Z = (-1, -7)) :
  (X.1 + X.2) = -24 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_X_l1083_108366


namespace NUMINAMATH_GPT_chess_tournament_num_players_l1083_108321

theorem chess_tournament_num_players (n : ℕ) :
  (∀ k, k ≠ n → exists m, m ≠ n ∧ (k = m)) ∧ 
  ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))) = (1 / 13 * ((1 / 2 * n * (n - 1)) - ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))))) →
  n = 21 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_num_players_l1083_108321


namespace NUMINAMATH_GPT_license_plates_count_l1083_108351

/--
Define the conditions and constants.
-/
def num_letters := 26
def num_first_digit := 5  -- Odd digits
def num_second_digit := 5 -- Even digits

theorem license_plates_count : num_letters ^ 3 * num_first_digit * num_second_digit = 439400 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1083_108351


namespace NUMINAMATH_GPT_factor_expression_l1083_108311

theorem factor_expression (x a b c : ℝ) :
  (x - a) ^ 2 * (b - c) + (x - b) ^ 2 * (c - a) + (x - c) ^ 2 * (a - b) = -(a - b) * (b - c) * (c - a) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1083_108311


namespace NUMINAMATH_GPT_smallest_palindrome_divisible_by_6_l1083_108352

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end NUMINAMATH_GPT_smallest_palindrome_divisible_by_6_l1083_108352


namespace NUMINAMATH_GPT_find_angle_C_range_of_a_plus_b_l1083_108373

variables {A B C a b c : ℝ}

-- Define the conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Proof problem 1: show angle C is π/3
theorem find_angle_C (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b c A B C) : 
  C = π / 3 :=
sorry

-- Proof problem 2: if c = 2, then show the range of a + b
theorem range_of_a_plus_b (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b 2 A B C) :
  2 < a + b ∧ a + b ≤ 4 :=
sorry

end NUMINAMATH_GPT_find_angle_C_range_of_a_plus_b_l1083_108373


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l1083_108336

theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) 
  (h1 : 360 / exterior_angle = n) (h2 : 20 = exterior_angle)
  (h3 : 10 = side_length) : 180 = n * side_length :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l1083_108336


namespace NUMINAMATH_GPT_proof_problem_l1083_108372

variables (p q : Prop)

theorem proof_problem (hpq : p ∨ q) (hnp : ¬p) : q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1083_108372


namespace NUMINAMATH_GPT_scientific_notation_of_845_billion_l1083_108389

/-- Express 845 billion yuan in scientific notation. -/
theorem scientific_notation_of_845_billion :
  (845 * (10^9 : ℝ)) / (10^9 : ℝ) = 8.45 * 10^3 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_845_billion_l1083_108389


namespace NUMINAMATH_GPT_distinct_real_roots_l1083_108388

open Real

theorem distinct_real_roots (n : ℕ) (hn : n > 0) (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (2 * n - 1 < x1 ∧ x1 ≤ 2 * n + 1) ∧ 
  (2 * n - 1 < x2 ∧ x2 ≤ 2 * n + 1) ∧ |x1 - 2 * n| = k ∧ |x2 - 2 * n| = k) ↔ (0 < k ∧ k ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_l1083_108388


namespace NUMINAMATH_GPT_calculate_value_l1083_108382

theorem calculate_value (a b c x : ℕ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) (h_x : x = 3) :
  x^(a * (b + c)) - (x^a + x^b + x^c) = 204 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l1083_108382


namespace NUMINAMATH_GPT_fourth_person_height_l1083_108304

theorem fourth_person_height 
  (height1 height2 height3 height4 : ℝ)
  (diff12 : height2 = height1 + 2)
  (diff23 : height3 = height2 + 2)
  (diff34 : height4 = height3 + 6)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 76) :
  height4 = 82 :=
by
  sorry

end NUMINAMATH_GPT_fourth_person_height_l1083_108304


namespace NUMINAMATH_GPT_relationship_between_p_and_q_l1083_108392

variables {x y : ℝ}

def p (x y : ℝ) := (x^2 + y^2) * (x - y)
def q (x y : ℝ) := (x^2 - y^2) * (x + y)

theorem relationship_between_p_and_q (h1 : x < y) (h2 : y < 0) : p x y > q x y := 
  by sorry

end NUMINAMATH_GPT_relationship_between_p_and_q_l1083_108392


namespace NUMINAMATH_GPT_difference_twice_cecil_and_catherine_l1083_108338

theorem difference_twice_cecil_and_catherine
  (Cecil Catherine Carmela : ℕ)
  (h1 : Cecil = 600)
  (h2 : Carmela = 2 * 600 + 50)
  (h3 : 600 + (2 * 600 - Catherine) + Carmela = 2800) :
  2 * 600 - Catherine = 250 := by
  sorry

end NUMINAMATH_GPT_difference_twice_cecil_and_catherine_l1083_108338


namespace NUMINAMATH_GPT_boys_contributions_l1083_108333

theorem boys_contributions (x y z : ℝ) (h1 : z = x + 6.4) (h2 : (1 / 2) * x = (1 / 3) * y) (h3 : (1 / 2) * x = (1 / 4) * z) :
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_boys_contributions_l1083_108333


namespace NUMINAMATH_GPT_range_of_2a_plus_3b_l1083_108381

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_2a_plus_3b_l1083_108381


namespace NUMINAMATH_GPT_problem1_problem2_l1083_108349

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x - 1
  else if 0 < x ∧ x ≤ 1 then -2 * x + 1
  else 0 -- considering the function is not defined outside the given range

-- Statement to prove that f(f(-1)) = -1
theorem problem1 : f (f (-1)) = -1 :=
by
  sorry

-- Statements to prove the solution set for |f(x)| < 1/2
theorem problem2 : { x : ℝ | |f x| < 1 / 2 } = { x : ℝ | -3/4 < x ∧ x < -1/4 } ∪ { x : ℝ | 1/4 < x ∧ x < 3/4 } :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1083_108349


namespace NUMINAMATH_GPT_gcd_polynomial_l1083_108317

theorem gcd_polynomial {b : ℤ} (h1 : ∃ k : ℤ, b = 2 * 7786 * k) : 
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l1083_108317


namespace NUMINAMATH_GPT_donuts_count_is_correct_l1083_108339

-- Define the initial number of donuts
def initial_donuts : ℕ := 50

-- Define the number of donuts Bill eats
def eaten_by_bill : ℕ := 2

-- Define the number of donuts taken by the secretary
def taken_by_secretary : ℕ := 4

-- Calculate the remaining donuts after Bill and the secretary take their portions
def remaining_after_bill_and_secretary : ℕ := initial_donuts - eaten_by_bill - taken_by_secretary

-- Define the number of donuts stolen by coworkers (half of the remaining donuts)
def stolen_by_coworkers : ℕ := remaining_after_bill_and_secretary / 2

-- Define the number of donuts left for the meeting
def donuts_left_for_meeting : ℕ := remaining_after_bill_and_secretary - stolen_by_coworkers

-- The theorem to prove
theorem donuts_count_is_correct : donuts_left_for_meeting = 22 :=
by
  sorry

end NUMINAMATH_GPT_donuts_count_is_correct_l1083_108339
