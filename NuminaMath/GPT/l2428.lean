import Mathlib

namespace NUMINAMATH_GPT_arithmetic_geometric_sequences_sequence_sum_first_terms_l2428_242808

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 1 + (n * (n + 1)) / 2

theorem arithmetic_geometric_sequences
  (a b S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 0 = 1)
  (h4 : b 0 = 1)
  (h5 : b 2 * S 2 = 36)
  (h6 : b 1 * S 1 = 8) :
  ((∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ n)) ∨
  ((∀ n, a n = -(2 * n / 3) + 5 / 3) ∧ (∀ n, b n = 6 ^ n)) :=
sorry

theorem sequence_sum_first_terms
  (a : ℕ → ℤ)
  (h : ∀ n, a n = 2 * n + 1)
  (S : ℕ → ℤ)
  (T : ℕ → ℚ)
  (hS : sequence_sum a S)
  (n : ℕ) :
  T n = n / (2 * n + 1) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequences_sequence_sum_first_terms_l2428_242808


namespace NUMINAMATH_GPT_find_remainder_proof_l2428_242819

def div_remainder_problem :=
  let number := 220050
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  let divisor := sum
  let quotient_correct := quotient = 220
  let division_formula := number = divisor * quotient + 50
  quotient_correct ∧ division_formula

theorem find_remainder_proof : div_remainder_problem := by
  sorry

end NUMINAMATH_GPT_find_remainder_proof_l2428_242819


namespace NUMINAMATH_GPT_shadow_stretch_rate_is_5_feet_per_hour_l2428_242827

-- Given conditions
def shadow_length_in_inches (hours_past_noon : ℕ) : ℕ := 360
def hours_past_noon : ℕ := 6

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

-- Calculate rate of increase of shadow length per hour
def rate_of_shadow_stretch_per_hour : ℕ := inches_to_feet (shadow_length_in_inches hours_past_noon) / hours_past_noon

theorem shadow_stretch_rate_is_5_feet_per_hour :
  rate_of_shadow_stretch_per_hour = 5 := by
  sorry

end NUMINAMATH_GPT_shadow_stretch_rate_is_5_feet_per_hour_l2428_242827


namespace NUMINAMATH_GPT_correct_polynomial_and_result_l2428_242812

theorem correct_polynomial_and_result :
  ∃ p q r : Polynomial ℝ,
    q = X^2 - 3 * X + 5 ∧
    p + q = 5 * X^2 - 2 * X + 4 ∧
    p = 4 * X^2 + X - 1 ∧
    r = p - q ∧
    r = 3 * X^2 + 4 * X - 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_polynomial_and_result_l2428_242812


namespace NUMINAMATH_GPT_problem_statement_l2428_242810

variables {R : Type*} [LinearOrderedField R]

theorem problem_statement (a b c : R) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : (b - a) ^ 2 - 4 * (b - c) * (c - a) = 0) : (b - c) / (c - a) = -1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2428_242810


namespace NUMINAMATH_GPT_find_last_three_digits_of_9_pow_107_l2428_242830

theorem find_last_three_digits_of_9_pow_107 : (9 ^ 107) % 1000 = 969 := 
by 
  sorry

end NUMINAMATH_GPT_find_last_three_digits_of_9_pow_107_l2428_242830


namespace NUMINAMATH_GPT_range_of_m_l2428_242835

theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2428_242835


namespace NUMINAMATH_GPT_fraction_halfway_between_one_fourth_and_one_sixth_l2428_242816

theorem fraction_halfway_between_one_fourth_and_one_sixth :
  (1/4 + 1/6) / 2 = 5 / 24 :=
by
  sorry

end NUMINAMATH_GPT_fraction_halfway_between_one_fourth_and_one_sixth_l2428_242816


namespace NUMINAMATH_GPT_g_of_5_l2428_242802

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 20 = 30 →
  g 5 = 7.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_g_of_5_l2428_242802


namespace NUMINAMATH_GPT_roundTripAverageSpeed_l2428_242824

noncomputable def averageSpeed (distAB distBC speedAB speedBC speedCB totalTime : ℝ) : ℝ :=
  let timeAB := distAB / speedAB
  let timeBC := distBC / speedBC
  let timeCB := distBC / speedCB
  let timeBA := totalTime - (timeAB + timeBC + timeCB)
  let totalDistance := 2 * (distAB + distBC)
  totalDistance / totalTime

theorem roundTripAverageSpeed :
  averageSpeed 150 230 80 88 100 9 = 84.44 :=
by
  -- The actual proof will go here, which is not required for this task.
  sorry

end NUMINAMATH_GPT_roundTripAverageSpeed_l2428_242824


namespace NUMINAMATH_GPT_calculate_expression_l2428_242814

theorem calculate_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2428_242814


namespace NUMINAMATH_GPT_gcd_sixPn_n_minus_2_l2428_242803

def nthSquarePyramidalNumber (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

def sixPn (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1)

theorem gcd_sixPn_n_minus_2 (n : ℕ) (h_pos : 0 < n) : Int.gcd (sixPn n) (n - 2) ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_gcd_sixPn_n_minus_2_l2428_242803


namespace NUMINAMATH_GPT_missing_digit_divisibility_by_13_l2428_242820

theorem missing_digit_divisibility_by_13 (B : ℕ) (H : 0 ≤ B ∧ B ≤ 9) : 
  (13 ∣ (200 + 10 * B + 5)) ↔ B = 12 :=
by sorry

end NUMINAMATH_GPT_missing_digit_divisibility_by_13_l2428_242820


namespace NUMINAMATH_GPT_ratio_B_over_A_eq_one_l2428_242825

theorem ratio_B_over_A_eq_one (A B : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 3 → 
  (A : ℝ) / (x + 3) + (B : ℝ) / (x * (x - 3)) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) :
  (B : ℝ) / (A : ℝ) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_B_over_A_eq_one_l2428_242825


namespace NUMINAMATH_GPT_larger_acute_angle_right_triangle_l2428_242805

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end NUMINAMATH_GPT_larger_acute_angle_right_triangle_l2428_242805


namespace NUMINAMATH_GPT_evaluate_expression_l2428_242831

noncomputable def expression := 
  (Real.sqrt 3 * Real.tan (Real.pi / 15) - 3) / 
  (4 * (Real.cos (Real.pi / 15))^2 * Real.sin (Real.pi / 15) - 2 * Real.sin (Real.pi / 15))

theorem evaluate_expression : expression = -4 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2428_242831


namespace NUMINAMATH_GPT_minimum_value_of_f_roots_sum_gt_2_l2428_242815

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f 1 = 1 := by
  exists 1
  sorry

theorem roots_sum_gt_2 (a x₁ x₂ : ℝ) (h_f_x₁ : f x₁ = a) (h_f_x₂ : f x₂ = a) (h_x₁_lt_x₂ : x₁ < x₂) :
    x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_roots_sum_gt_2_l2428_242815


namespace NUMINAMATH_GPT_closest_fraction_l2428_242817

theorem closest_fraction (n : ℤ) : 
  let frac1 := 37 / 57 
  let closest := 15 / 23
  n = 15 ∧ abs (851 - 57 * n) = min (abs (851 - 57 * 14)) (abs (851 - 57 * 15)) :=
by
  let frac1 := (37 : ℚ) / 57
  let closest := (15 : ℚ) / 23
  have h : 37 * 23 = 851 := by norm_num
  have denom : 57 * 23 = 1311 := by norm_num
  let num := 851
  sorry

end NUMINAMATH_GPT_closest_fraction_l2428_242817


namespace NUMINAMATH_GPT_number_of_toys_gained_l2428_242832

theorem number_of_toys_gained
  (num_toys : ℕ) (selling_price : ℕ) (cost_price_one_toy : ℕ)
  (total_cp := num_toys * cost_price_one_toy)
  (profit := selling_price - total_cp)
  (num_toys_equiv_to_profit := profit / cost_price_one_toy) :
  num_toys = 18 → selling_price = 23100 → cost_price_one_toy = 1100 → num_toys_equiv_to_profit = 3 :=
by
  intros h1 h2 h3
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_number_of_toys_gained_l2428_242832


namespace NUMINAMATH_GPT_hyperbola_no_common_point_l2428_242811

theorem hyperbola_no_common_point (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (y_line : ∀ x : ℝ, y = 2 * x) : 
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e ≤ Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_no_common_point_l2428_242811


namespace NUMINAMATH_GPT_import_tax_l2428_242822

theorem import_tax (total_value : ℝ) (tax_rate : ℝ) (excess_limit : ℝ) (correct_tax : ℝ)
  (h1 : total_value = 2560) (h2 : tax_rate = 0.07) (h3 : excess_limit = 1000) : 
  correct_tax = tax_rate * (total_value - excess_limit) :=
by
  sorry

end NUMINAMATH_GPT_import_tax_l2428_242822


namespace NUMINAMATH_GPT_sum_of_squares_l2428_242821

theorem sum_of_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2428_242821


namespace NUMINAMATH_GPT_brenda_age_l2428_242828

-- Define ages of Addison, Brenda, Carlos, and Janet
variables (A B C J : ℕ)

-- Formalize the conditions from the problem
def condition1 := A = 4 * B
def condition2 := C = 2 * B
def condition3 := A = J

-- State the theorem we aim to prove
theorem brenda_age (A B C J : ℕ) (h1 : condition1 A B)
                                (h2 : condition2 C B)
                                (h3 : condition3 A J) :
  B = J / 4 :=
sorry

end NUMINAMATH_GPT_brenda_age_l2428_242828


namespace NUMINAMATH_GPT_product_of_last_two_digits_div_by_6_and_sum_15_l2428_242806

theorem product_of_last_two_digits_div_by_6_and_sum_15
  (n : ℕ)
  (h1 : n % 6 = 0)
  (A B : ℕ)
  (h2 : n % 100 = 10 * A + B)
  (h3 : A + B = 15)
  (h4 : B % 2 = 0) : 
  A * B = 54 := 
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_div_by_6_and_sum_15_l2428_242806


namespace NUMINAMATH_GPT_find_solutions_equation_l2428_242801

theorem find_solutions_equation :
  {x : ℝ | 1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 11 * x - 12) = 0}
  = {1, -12, 4, -3} :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_equation_l2428_242801


namespace NUMINAMATH_GPT_num_three_digit_integers_sum_to_seven_l2428_242829

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end NUMINAMATH_GPT_num_three_digit_integers_sum_to_seven_l2428_242829


namespace NUMINAMATH_GPT_teapot_volume_proof_l2428_242833

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem teapot_volume_proof (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0.5)
  (h2 : arithmetic_sequence a d 7 + arithmetic_sequence a d 8 + arithmetic_sequence a d 9 = 2.5) :
  arithmetic_sequence a d 5 = 0.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_teapot_volume_proof_l2428_242833


namespace NUMINAMATH_GPT_bus_passenger_count_l2428_242800

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_bus_passenger_count_l2428_242800


namespace NUMINAMATH_GPT_quadratic_solution_linear_factor_solution_l2428_242804

theorem quadratic_solution (x : ℝ) : (5 * x^2 + 2 * x - 1 = 0) ↔ (x = (-1 + Real.sqrt 6) / 5 ∨ x = (-1 - Real.sqrt 6) / 5) := by
  sorry

theorem linear_factor_solution (x : ℝ) : (x * (x - 3) - 4 * (3 - x) = 0) ↔ (x = 3 ∨ x = -4) := by
  sorry

end NUMINAMATH_GPT_quadratic_solution_linear_factor_solution_l2428_242804


namespace NUMINAMATH_GPT_sum_reciprocals_square_l2428_242813

theorem sum_reciprocals_square (x y : ℕ) (h : x * y = 11) : (1 : ℚ) / (↑x ^ 2) + (1 : ℚ) / (↑y ^ 2) = 122 / 121 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_square_l2428_242813


namespace NUMINAMATH_GPT_sheena_weeks_to_complete_l2428_242834

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end NUMINAMATH_GPT_sheena_weeks_to_complete_l2428_242834


namespace NUMINAMATH_GPT_calculation_correct_l2428_242809

theorem calculation_correct : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l2428_242809


namespace NUMINAMATH_GPT_work_completion_l2428_242823

theorem work_completion (W : ℕ) (n : ℕ) (h1 : 0 < n) (H1 : 0 < W) :
  (∀ w : ℕ, w ≤ W / n) → 
  (∀ k : ℕ, k = (7 * n) / 10 → k * (3 * W) / (10 * n) ≥ W / 3) → 
  (∀ m : ℕ, m = (3 * n) / 10 → m * (7 * W) / (10 * n) ≥ W / 3) → 
  ∃ g1 g2 g3 : ℕ, g1 + g2 + g3 < W / 3 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l2428_242823


namespace NUMINAMATH_GPT_percentage_difference_l2428_242818

theorem percentage_difference : (0.4 * 60 - (4/5 * 25)) = 4 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l2428_242818


namespace NUMINAMATH_GPT_simplify_correct_l2428_242807

def simplify_polynomial (x : Real) : Real :=
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9)

theorem simplify_correct (x : Real) :
  simplify_polynomial x = 2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_correct_l2428_242807


namespace NUMINAMATH_GPT_zachary_pushups_l2428_242826

variable {P : ℕ}
variable {C : ℕ}

theorem zachary_pushups :
  C = 58 → C = P + 12 → P = 46 :=
by 
  intros hC1 hC2
  rw [hC2] at hC1
  linarith

end NUMINAMATH_GPT_zachary_pushups_l2428_242826
