import Mathlib

namespace NUMINAMATH_GPT_suff_not_nec_for_abs_eq_one_l1798_179865

variable (m : ℝ)

theorem suff_not_nec_for_abs_eq_one (hm : m = 1) : |m| = 1 ∧ (¬(|m| = 1 → m = 1)) := by
  sorry

end NUMINAMATH_GPT_suff_not_nec_for_abs_eq_one_l1798_179865


namespace NUMINAMATH_GPT_range_of_m_l1798_179840

theorem range_of_m (m : ℝ)
  (h₁ : (m^2 - 4) ≥ 0)
  (h₂ : (4 * (m - 2)^2 - 16) < 0) :
  1 < m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1798_179840


namespace NUMINAMATH_GPT_fish_lifespan_is_12_l1798_179801

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end NUMINAMATH_GPT_fish_lifespan_is_12_l1798_179801


namespace NUMINAMATH_GPT_sum_abc_is_eight_l1798_179825

theorem sum_abc_is_eight (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_abc_is_eight_l1798_179825


namespace NUMINAMATH_GPT_div_by_3kp1_iff_div_by_3k_l1798_179837

theorem div_by_3kp1_iff_div_by_3k (m n k : ℕ) (h1 : m > n) :
  (3 ^ (k + 1)) ∣ (4 ^ m - 4 ^ n) ↔ (3 ^ k) ∣ (m - n) := 
sorry

end NUMINAMATH_GPT_div_by_3kp1_iff_div_by_3k_l1798_179837


namespace NUMINAMATH_GPT_ducks_and_dogs_total_l1798_179863

theorem ducks_and_dogs_total (d g : ℕ) (h1 : d = g + 2) (h2 : 4 * g - 2 * d = 10) : d + g = 16 :=
  sorry

end NUMINAMATH_GPT_ducks_and_dogs_total_l1798_179863


namespace NUMINAMATH_GPT_remainder_sum_first_six_primes_div_seventh_prime_l1798_179832

-- Define the first six prime numbers
def firstSixPrimes : List ℕ := [2, 3, 5, 7, 11, 13]

-- Define the sum of the first six prime numbers
def sumOfFirstSixPrimes : ℕ := firstSixPrimes.sum

-- Define the seventh prime number
def seventhPrime : ℕ := 17

-- Proof statement that the remainder of the division is 7
theorem remainder_sum_first_six_primes_div_seventh_prime :
  (sumOfFirstSixPrimes % seventhPrime) = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_first_six_primes_div_seventh_prime_l1798_179832


namespace NUMINAMATH_GPT_magnitude_of_z_l1798_179833

open Complex

theorem magnitude_of_z {z : ℂ} (h : z * (1 + I) = 1 - I) : abs z = 1 :=
sorry

end NUMINAMATH_GPT_magnitude_of_z_l1798_179833


namespace NUMINAMATH_GPT_g_function_property_l1798_179858

variable {g : ℝ → ℝ}
variable {a b : ℝ}

theorem g_function_property 
  (h1 : ∀ a c : ℝ, c^3 * g a = a^3 * g c)
  (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 :=
  sorry

end NUMINAMATH_GPT_g_function_property_l1798_179858


namespace NUMINAMATH_GPT_john_makes_200_profit_l1798_179857

noncomputable def john_profit (num_woodburnings : ℕ) (price_per_woodburning : ℕ) (cost_of_wood : ℕ) : ℕ :=
  (num_woodburnings * price_per_woodburning) - cost_of_wood

theorem john_makes_200_profit :
  john_profit 20 15 100 = 200 :=
by
  sorry

end NUMINAMATH_GPT_john_makes_200_profit_l1798_179857


namespace NUMINAMATH_GPT_fraction_income_spent_on_rent_l1798_179897

theorem fraction_income_spent_on_rent
  (hourly_wage : ℕ)
  (work_hours_per_week : ℕ)
  (weeks_in_month : ℕ)
  (food_expense : ℕ)
  (tax_expense : ℕ)
  (remaining_income : ℕ) :
  hourly_wage = 30 →
  work_hours_per_week = 48 →
  weeks_in_month = 4 →
  food_expense = 500 →
  tax_expense = 1000 →
  remaining_income = 2340 →
  ((hourly_wage * work_hours_per_week * weeks_in_month - remaining_income - (food_expense + tax_expense)) / (hourly_wage * work_hours_per_week * weeks_in_month) = 1/3) :=
by
  intros h_wage h_hours h_weeks h_food h_taxes h_remaining
  sorry

end NUMINAMATH_GPT_fraction_income_spent_on_rent_l1798_179897


namespace NUMINAMATH_GPT_johns_videos_weekly_minutes_l1798_179819

theorem johns_videos_weekly_minutes (daily_minutes weekly_minutes : ℕ) (short_video_length long_factor: ℕ) (short_videos_per_day long_videos_per_day days : ℕ)
  (h1 : daily_minutes = short_videos_per_day * short_video_length + long_videos_per_day * (long_factor * short_video_length))
  (h2 : weekly_minutes = daily_minutes * days)
  (h_short_videos_per_day : short_videos_per_day = 2)
  (h_long_videos_per_day : long_videos_per_day = 1)
  (h_short_video_length : short_video_length = 2)
  (h_long_factor : long_factor = 6)
  (h_weekly_minutes : weekly_minutes = 112):
  days = 7 :=
by
  sorry

end NUMINAMATH_GPT_johns_videos_weekly_minutes_l1798_179819


namespace NUMINAMATH_GPT_divides_power_diff_l1798_179861

theorem divides_power_diff (x : ℤ) (y z w : ℕ) (hy : y % 2 = 1) (hz : z % 2 = 1) (hw : w % 2 = 1) : 17 ∣ x^(y^(z^w)) - x^(y^z) := 
by
  sorry

end NUMINAMATH_GPT_divides_power_diff_l1798_179861


namespace NUMINAMATH_GPT_candidate_percentage_l1798_179809

theorem candidate_percentage (P : ℝ) (h : (P / 100) * 7800 + 2340 = 7800) : P = 70 :=
sorry

end NUMINAMATH_GPT_candidate_percentage_l1798_179809


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l1798_179817

theorem sum_of_cubes_of_consecutive_integers :
  ∃ (a b c d : ℕ), a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a^2 + b^2 + c^2 + d^2 = 9340) ∧ (a^3 + b^3 + c^3 + d^3 = 457064) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l1798_179817


namespace NUMINAMATH_GPT_problem_1_1_and_2_problem_1_2_l1798_179806

section Sequence

variables (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom a_1 : a 1 = 3
axiom a_n_recurr : ∀ n ≥ 2, a n = 2 * a (n - 1) + (n - 2)

-- Prove that {a_n + n} is a geometric sequence and find the general term formula for {a_n}
theorem problem_1_1_and_2 :
  (∀ n ≥ 2, (a (n - 1) + (n - 1) ≠ 0)) ∧ ((a 1 + 1) * 2^(n - 1) = a n + n) ∧
  (∀ n, a n = 2^(n + 1) - n) :=
sorry

-- Find the sum of the first n terms, S_n, of the sequence {a_n}
theorem problem_1_2 (n : ℕ) : S n = 2^(n + 2) - 4 - (n^2 + n) / 2 :=
sorry

end Sequence

end NUMINAMATH_GPT_problem_1_1_and_2_problem_1_2_l1798_179806


namespace NUMINAMATH_GPT_stella_toilet_paper_packs_l1798_179894

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end NUMINAMATH_GPT_stella_toilet_paper_packs_l1798_179894


namespace NUMINAMATH_GPT_sum_roots_of_quadratic_l1798_179876

theorem sum_roots_of_quadratic (a b : ℝ) (h₁ : a^2 - a - 6 = 0) (h₂ : b^2 - b - 6 = 0) (h₃ : a ≠ b) :
  a + b = 1 :=
sorry

end NUMINAMATH_GPT_sum_roots_of_quadratic_l1798_179876


namespace NUMINAMATH_GPT_survey_support_percentage_l1798_179872

theorem survey_support_percentage 
  (num_men : ℕ) (percent_men_support : ℝ)
  (num_women : ℕ) (percent_women_support : ℝ)
  (h_men : num_men = 200)
  (h_percent_men_support : percent_men_support = 0.7)
  (h_women : num_women = 500)
  (h_percent_women_support : percent_women_support = 0.75) :
  (num_men * percent_men_support + num_women * percent_women_support) / (num_men + num_women) * 100 = 74 := 
by
  sorry

end NUMINAMATH_GPT_survey_support_percentage_l1798_179872


namespace NUMINAMATH_GPT_fraction_calculation_l1798_179880

theorem fraction_calculation :
  ( (12^4 + 324) * (26^4 + 324) * (38^4 + 324) * (50^4 + 324) * (62^4 + 324)) /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) =
  73.481 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l1798_179880


namespace NUMINAMATH_GPT_correct_operation_l1798_179836

variable (a b : ℝ)

theorem correct_operation : 
  ¬ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧
  ¬ ((a^3) ^ 2 = a ^ 5) ∧
  (a ^ 5 / a ^ 3 = a ^ 2) ∧
  ¬ (a ^ 3 + a ^ 2 = a ^ 5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1798_179836


namespace NUMINAMATH_GPT_condition_for_M_eq_N_l1798_179860

theorem condition_for_M_eq_N (a1 b1 c1 a2 b2 c2 : ℝ) 
    (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0) 
    (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) → 
    (M = {x : ℝ | a1 * x ^ 2 + b1 * x + c1 > 0} ∧
     N = {x : ℝ | a2 * x ^ 2 + b2 * x + c2 > 0} →
    (¬ (M = N))) ∨ (¬ (N = {} ↔ (M = N))) :=
sorry

end NUMINAMATH_GPT_condition_for_M_eq_N_l1798_179860


namespace NUMINAMATH_GPT_new_average_page_count_l1798_179874

theorem new_average_page_count
  (n : ℕ) (a : ℕ) (p1 p2 : ℕ)
  (h_n : n = 80) (h_a : a = 120)
  (h_p1 : p1 = 150) (h_p2 : p2 = 170) :
  (n - 2) ≠ 0 → 
  ((n * a - (p1 + p2)) / (n - 2) = 119) := 
by sorry

end NUMINAMATH_GPT_new_average_page_count_l1798_179874


namespace NUMINAMATH_GPT_probability_standard_bulb_l1798_179814

structure FactoryConditions :=
  (P_H1 : ℝ)
  (P_H2 : ℝ)
  (P_H3 : ℝ)
  (P_A_H1 : ℝ)
  (P_A_H2 : ℝ)
  (P_A_H3 : ℝ)

theorem probability_standard_bulb (conditions : FactoryConditions) : 
  conditions.P_H1 = 0.45 → 
  conditions.P_H2 = 0.40 → 
  conditions.P_H3 = 0.15 →
  conditions.P_A_H1 = 0.70 → 
  conditions.P_A_H2 = 0.80 → 
  conditions.P_A_H3 = 0.81 → 
  (conditions.P_H1 * conditions.P_A_H1 + 
   conditions.P_H2 * conditions.P_A_H2 + 
   conditions.P_H3 * conditions.P_A_H3) = 0.7565 :=
by 
  intros h1 h2 h3 a_h1 a_h2 a_h3 
  sorry

end NUMINAMATH_GPT_probability_standard_bulb_l1798_179814


namespace NUMINAMATH_GPT_sum_of_different_roots_eq_six_l1798_179882

theorem sum_of_different_roots_eq_six (a b : ℝ) (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_different_roots_eq_six_l1798_179882


namespace NUMINAMATH_GPT_winning_votes_cast_l1798_179879

theorem winning_votes_cast (V : ℝ) (h1 : 0.40 * V = 280) : 0.70 * V = 490 :=
by
  sorry

end NUMINAMATH_GPT_winning_votes_cast_l1798_179879


namespace NUMINAMATH_GPT_price_reduction_l1798_179855

theorem price_reduction (x : ℝ) 
  (initial_price : ℝ := 60) 
  (final_price : ℝ := 48.6) :
  initial_price * (1 - x) * (1 - x) = final_price :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_l1798_179855


namespace NUMINAMATH_GPT_average_of_quantities_l1798_179877

theorem average_of_quantities (a1 a2 a3 a4 a5 : ℝ) :
  ((a1 + a2 + a3) / 3 = 4) →
  ((a4 + a5) / 2 = 21.5) →
  ((a1 + a2 + a3 + a4 + a5) / 5 = 11) :=
by
  intros h3 h2
  sorry

end NUMINAMATH_GPT_average_of_quantities_l1798_179877


namespace NUMINAMATH_GPT_total_points_l1798_179823

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end NUMINAMATH_GPT_total_points_l1798_179823


namespace NUMINAMATH_GPT_problem1_problem2_l1798_179891

noncomputable def f (x a c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem 1: Prove that for c = 19, the inequality f(1, a, 19) > 0 holds for -2 < a < 8
theorem problem1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by sorry

-- Problem 2: Given that f(x) > 0 has solution set (-1, 3), find a and c
theorem problem2 (a c : ℝ) (hx : ∀ x, -1 < x ∧ x < 3 → f x a c > 0) : 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1798_179891


namespace NUMINAMATH_GPT_negation_proposition_true_l1798_179851

theorem negation_proposition_true (x : ℝ) : (¬ (|x| > 1 → x > 1)) ↔ (|x| ≤ 1 → x ≤ 1) :=
by sorry

end NUMINAMATH_GPT_negation_proposition_true_l1798_179851


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1798_179868

-- Conditions provided
def first_term (x y : ℝ) := x + y^2
def second_term (x y : ℝ) := x - y^2
def third_term (x y : ℝ) := x - 3*y^2
def fourth_term (x y : ℝ) := x - 5*y^2

-- Proof to determine the fifth term
theorem fifth_term_arithmetic_sequence (x y : ℝ) :
  (fourth_term x y) - (third_term x y) = -2*y^2 →
  (x - 5 * y^2) - 2 * y^2 = x - 7 * y^2 :=
by sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1798_179868


namespace NUMINAMATH_GPT_cost_two_cones_l1798_179888

-- Definition for the cost of a single ice cream cone
def cost_one_cone : ℕ := 99

-- The theorem to prove the cost of two ice cream cones
theorem cost_two_cones : 2 * cost_one_cone = 198 := 
by 
  sorry

end NUMINAMATH_GPT_cost_two_cones_l1798_179888


namespace NUMINAMATH_GPT_evaluate_expression_l1798_179887

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end NUMINAMATH_GPT_evaluate_expression_l1798_179887


namespace NUMINAMATH_GPT_product_of_binaries_l1798_179889

-- Step a) Define the binary numbers as Lean 4 terms.
def bin_11011 : ℕ := 0b11011
def bin_111 : ℕ := 0b111
def bin_101 : ℕ := 0b101

-- Step c) Define the goal to be proven.
theorem product_of_binaries :
  bin_11011 * bin_111 * bin_101 = 0b1110110001 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_product_of_binaries_l1798_179889


namespace NUMINAMATH_GPT_tens_digit_of_8_pow_1234_l1798_179886

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_8_pow_1234_l1798_179886


namespace NUMINAMATH_GPT_fraction_to_decimal_l1798_179818

theorem fraction_to_decimal : (5 / 50) = 0.10 := 
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1798_179818


namespace NUMINAMATH_GPT_bob_mean_score_l1798_179892

-- Conditions
def scores : List ℝ := [68, 72, 76, 80, 85, 90]
def alice_scores (a1 a2 a3 : ℝ) : Prop := a1 < a2 ∧ a2 < a3 ∧ a1 + a2 + a3 = 225
def bob_scores (b1 b2 b3 : ℝ) : Prop := b1 + b2 + b3 = 246

-- Theorem statement proving Bob's mean score
theorem bob_mean_score (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1 ∈ scores) (h2 : a2 ∈ scores) (h3 : a3 ∈ scores)
  (h4 : b1 ∈ scores) (h5 : b2 ∈ scores) (h6 : b3 ∈ scores)
  (h7 : alice_scores a1 a2 a3)
  (h8 : bob_scores b1 b2 b3)
  (h9 : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3)
  : (b1 + b2 + b3) / 3 = 82 :=
sorry

end NUMINAMATH_GPT_bob_mean_score_l1798_179892


namespace NUMINAMATH_GPT_simplify_expression_l1798_179830

theorem simplify_expression :
  15 * (18 / 5) * (-42 / 45) = -50.4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1798_179830


namespace NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_l1798_179866

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_l1798_179866


namespace NUMINAMATH_GPT_fraction_division_l1798_179870

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end NUMINAMATH_GPT_fraction_division_l1798_179870


namespace NUMINAMATH_GPT_baseball_card_decrease_l1798_179827

theorem baseball_card_decrease (x : ℝ) :
  (0 < x) ∧ (x < 100) ∧ (100 - x) * 0.9 = 45 → x = 50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_baseball_card_decrease_l1798_179827


namespace NUMINAMATH_GPT_range_of_a_l1798_179885

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1798_179885


namespace NUMINAMATH_GPT_evaluate_exp_power_l1798_179875

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_exp_power_l1798_179875


namespace NUMINAMATH_GPT_min_rows_512_l1798_179816

theorem min_rows_512 (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (H : ∀ A (i j : ℕ), i < 10 → j < 10 → i ≠ j → ∃ B, B < n ∧ (table B i ≠ table A i) ∧ (table B j ≠ table A j) ∧ ∀ k, k ≠ i ∧ k ≠ j → table B k = table A k) : 
  n ≥ 512 :=
sorry

end NUMINAMATH_GPT_min_rows_512_l1798_179816


namespace NUMINAMATH_GPT_work_problem_l1798_179805

theorem work_problem (P Q R W t_q : ℝ) (h1 : P = Q + R) 
    (h2 : (P + Q) * 10 = W) 
    (h3 : R * 35 = W) 
    (h4 : Q * t_q = W) : 
    t_q = 28 := 
by
    sorry

end NUMINAMATH_GPT_work_problem_l1798_179805


namespace NUMINAMATH_GPT_at_least_one_two_prob_l1798_179859

-- Definitions and conditions corresponding to the problem
def total_outcomes (n : ℕ) : ℕ := n * n
def no_twos_outcomes (n : ℕ) : ℕ := (n - 1) * (n - 1)

-- The probability calculation
def probability_at_least_one_two (n : ℕ) : ℚ := 
  let tot_outs := total_outcomes n
  let no_twos := no_twos_outcomes n
  (tot_outs - no_twos : ℚ) / tot_outs

-- Our main theorem to be proved
theorem at_least_one_two_prob : 
  probability_at_least_one_two 6 = 11 / 36 := 
by
  sorry

end NUMINAMATH_GPT_at_least_one_two_prob_l1798_179859


namespace NUMINAMATH_GPT_rectangle_perimeter_eq_circle_circumference_l1798_179843

theorem rectangle_perimeter_eq_circle_circumference (l : ℝ) :
  2 * (l + 3) = 10 * Real.pi -> l = 5 * Real.pi - 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_eq_circle_circumference_l1798_179843


namespace NUMINAMATH_GPT_baseball_game_earnings_l1798_179856

theorem baseball_game_earnings (W S : ℝ) 
  (h1 : W + S = 4994.50) 
  (h2 : W = S - 1330.50) : 
  S = 3162.50 := 
by 
  sorry

end NUMINAMATH_GPT_baseball_game_earnings_l1798_179856


namespace NUMINAMATH_GPT_not_all_x_ne_1_imp_x2_ne_0_l1798_179896

theorem not_all_x_ne_1_imp_x2_ne_0 : ¬ (∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) :=
sorry

end NUMINAMATH_GPT_not_all_x_ne_1_imp_x2_ne_0_l1798_179896


namespace NUMINAMATH_GPT_gummies_remain_l1798_179881

theorem gummies_remain
  (initial_candies : ℕ)
  (sibling_candies_per : ℕ)
  (num_siblings : ℕ)
  (best_friend_fraction : ℝ)
  (cousin_fraction : ℝ)
  (kept_candies : ℕ)
  (result : ℕ)
  (h_initial : initial_candies = 500)
  (h_sibling_candies_per : sibling_candies_per = 35)
  (h_num_siblings : num_siblings = 3)
  (h_best_friend_fraction : best_friend_fraction = 0.5)
  (h_cousin_fraction : cousin_fraction = 0.25)
  (h_kept_candies : kept_candies = 50)
  (h_result : result = 99) : 
  (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋ - 
  ⌊cousin_fraction * (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋)⌋ 
  - kept_candies) = result := 
by {
  sorry
}

end NUMINAMATH_GPT_gummies_remain_l1798_179881


namespace NUMINAMATH_GPT_negation_P_l1798_179831

-- Define the proposition P
def P (m : ℤ) : Prop := ∃ x : ℤ, 2 * x^2 + x + m ≤ 0

-- Define the negation of the proposition P
theorem negation_P (m : ℤ) : ¬P m ↔ ∀ x : ℤ, 2 * x^2 + x + m > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_P_l1798_179831


namespace NUMINAMATH_GPT_earnings_per_widget_l1798_179873

/-
Theorem:
Given:
1. Hourly wage is $12.50.
2. Hours worked in a week is 40.
3. Total weekly earnings are $580.
4. Number of widgets produced in a week is 500.

We want to prove:
The earnings per widget are $0.16.
-/

theorem earnings_per_widget (hourly_wage : ℝ) (hours_worked : ℝ)
  (total_weekly_earnings : ℝ) (widgets_produced : ℝ) :
  (hourly_wage = 12.50) →
  (hours_worked = 40) →
  (total_weekly_earnings = 580) →
  (widgets_produced = 500) →
  ( (total_weekly_earnings - hourly_wage * hours_worked) / widgets_produced = 0.16) :=
by
  intros h_wage h_hours h_earnings h_widgets
  sorry

end NUMINAMATH_GPT_earnings_per_widget_l1798_179873


namespace NUMINAMATH_GPT_solve_a_for_pure_imaginary_l1798_179829

theorem solve_a_for_pure_imaginary (a : ℝ) : (1 - a^2 = 0) ∧ (2 * a ≠ 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_a_for_pure_imaginary_l1798_179829


namespace NUMINAMATH_GPT_resistance_of_one_rod_l1798_179898

section RodResistance

variables (R_0 R : ℝ)

-- Given: the resistance of the entire construction is 8 Ω
def entire_construction_resistance : Prop := R = 8

-- Given: formula for the equivalent resistance
def equivalent_resistance_formula : Prop := R = 4 / 10 * R_0

-- To prove: the resistance of one rod is 20 Ω
theorem resistance_of_one_rod 
  (h1 : entire_construction_resistance R)
  (h2 : equivalent_resistance_formula R_0 R) :
  R_0 = 20 :=
sorry

end RodResistance

end NUMINAMATH_GPT_resistance_of_one_rod_l1798_179898


namespace NUMINAMATH_GPT_other_carton_racket_count_l1798_179867

def num_total_cartons : Nat := 38
def num_total_rackets : Nat := 100
def num_specific_cartons : Nat := 24
def num_rackets_per_specific_carton : Nat := 3

def num_remaining_cartons := num_total_cartons - num_specific_cartons
def num_remaining_rackets := num_total_rackets - (num_specific_cartons * num_rackets_per_specific_carton)

theorem other_carton_racket_count :
  (num_remaining_rackets / num_remaining_cartons) = 2 :=
by
  sorry

end NUMINAMATH_GPT_other_carton_racket_count_l1798_179867


namespace NUMINAMATH_GPT_jeans_to_tshirt_ratio_l1798_179815

noncomputable def socks_price := 5
noncomputable def tshirt_price := socks_price + 10
noncomputable def jeans_price := 30

theorem jeans_to_tshirt_ratio :
  jeans_price / tshirt_price = (2 : ℝ) :=
by sorry

end NUMINAMATH_GPT_jeans_to_tshirt_ratio_l1798_179815


namespace NUMINAMATH_GPT_domain_of_f_l1798_179854

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : domain f = {x | x > 1} := sorry

end NUMINAMATH_GPT_domain_of_f_l1798_179854


namespace NUMINAMATH_GPT_five_digit_number_l1798_179808

open Nat

noncomputable def problem_statement : Prop :=
  ∃ A B C D E F : ℕ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A + B + C + D + E + F = 25 ∧
    (A, B, C, D, E, F) = (3, 4, 2, 1, 6, 9)

theorem five_digit_number : problem_statement := 
  sorry

end NUMINAMATH_GPT_five_digit_number_l1798_179808


namespace NUMINAMATH_GPT_rectangle_dimensions_l1798_179800

theorem rectangle_dimensions (x : ℝ) (h : 4 * x * x = 120) : x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1798_179800


namespace NUMINAMATH_GPT_linear_combination_of_matrices_l1798_179813

variable (A B : Matrix (Fin 3) (Fin 3) ℤ) 

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, -4, 0],
    ![-1, 5, 1],
    ![0, 3, -7]
  ]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![4, -1, -2],
    ![0, -3, 5],
    ![2, 0, -4]
  ]

theorem linear_combination_of_matrices :
  3 • matrixA - 2 • matrixB = 
  ![
    ![-2, -10, 4],
    ![-3, 21, -7],
    ![-4, 9, -13]
  ] :=
sorry

end NUMINAMATH_GPT_linear_combination_of_matrices_l1798_179813


namespace NUMINAMATH_GPT_incorrect_statement_l1798_179850

-- Definitions based on the given conditions
def tripling_triangle_altitude_triples_area (b h : ℝ) : Prop :=
  3 * (1/2 * b * h) = 1/2 * b * (3 * h)

def halving_rectangle_base_halves_area (b h : ℝ) : Prop :=
  1/2 * b * h = 1/2 * (b * h)

def tripling_circle_radius_triples_area (r : ℝ) : Prop :=
  3 * (Real.pi * r^2) = Real.pi * (3 * r)^2

def tripling_divisor_and_numerator_leaves_quotient_unchanged (a b : ℝ) (hb : b ≠ 0) : Prop :=
  a / b = 3 * a / (3 * b)

def halving_negative_quantity_makes_it_greater (x : ℝ) : Prop :=
  x < 0 → (x / 2) > x

-- The incorrect statement is that tripling the radius of a circle triples the area
theorem incorrect_statement : ∃ r : ℝ, tripling_circle_radius_triples_area r → False :=
by
  use 1
  simp [tripling_circle_radius_triples_area]
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1798_179850


namespace NUMINAMATH_GPT_tangent_condition_sum_f_l1798_179807

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

theorem tangent_condition (a : ℝ) (h : f a 1 = f a 1) (m : ℝ) : 
    (3 * a + 1 = (7 - (f a 1)) / 2) := 
    sorry

theorem sum_f (a : ℝ) (h : a = 3/7) : 
    f a (-4) + f a (-3) + f a (-2) + f a (-1) + f a 0 + 
    f a 1 + f a 2 + f a 3 + f a 4 = 9 := 
    sorry

end NUMINAMATH_GPT_tangent_condition_sum_f_l1798_179807


namespace NUMINAMATH_GPT_divisor_is_20_l1798_179847

theorem divisor_is_20 (D q1 q2 q3 : ℕ) :
  (242 = D * q1 + 11) ∧
  (698 = D * q2 + 18) ∧
  (940 = D * q3 + 9) →
  D = 20 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_20_l1798_179847


namespace NUMINAMATH_GPT_find_t_when_perpendicular_l1798_179839

variable {t : ℝ}

def vector_m (t : ℝ) : ℝ × ℝ := (t + 1, 1)
def vector_n (t : ℝ) : ℝ × ℝ := (t + 2, 2)
def add_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def sub_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem find_t_when_perpendicular : 
  (dot_product (add_vectors (vector_m t) (vector_n t)) (sub_vectors (vector_m t) (vector_n t)) = 0) ↔ t = -3 := by
  sorry

end NUMINAMATH_GPT_find_t_when_perpendicular_l1798_179839


namespace NUMINAMATH_GPT_inverse_proportion_value_scientific_notation_l1798_179838

-- Statement to prove for Question 1:
theorem inverse_proportion_value (m : ℤ) (x : ℝ) :
  (m - 2) * x ^ (m ^ 2 - 5) = 0 ↔ m = -2 := by
  sorry

-- Statement to prove for Question 2:
theorem scientific_notation : -0.00000032 = -3.2 * 10 ^ (-7) := by
  sorry

end NUMINAMATH_GPT_inverse_proportion_value_scientific_notation_l1798_179838


namespace NUMINAMATH_GPT_percentage_seeds_germinated_l1798_179824

/-- There were 300 seeds planted in the first plot and 200 seeds planted in the second plot. 
    30% of the seeds in the first plot germinated and 32% of the total seeds germinated.
    Prove that 35% of the seeds in the second plot germinated. -/
theorem percentage_seeds_germinated 
  (s1 s2 : ℕ) (p1 p2 t : ℚ)
  (h1 : s1 = 300) 
  (h2 : s2 = 200) 
  (h3 : p1 = 30) 
  (h4 : t = 32) 
  (h5 : 0.30 * s1 + p2 * s2 = 0.32 * (s1 + s2)) :
  p2 = 35 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_seeds_germinated_l1798_179824


namespace NUMINAMATH_GPT_john_bought_packs_l1798_179869

def students_in_classes : List ℕ := [20, 25, 18, 22, 15]
def packs_per_student : ℕ := 3

theorem john_bought_packs :
  (students_in_classes.sum) * packs_per_student = 300 := by
  sorry

end NUMINAMATH_GPT_john_bought_packs_l1798_179869


namespace NUMINAMATH_GPT_sequence_last_number_is_one_l1798_179893

theorem sequence_last_number_is_one :
  ∃ (a : ℕ → ℤ), (a 1 = 1) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a (n + 1) = a n + a (n + 2)) ∧ (a 1999 = 1) := sorry

end NUMINAMATH_GPT_sequence_last_number_is_one_l1798_179893


namespace NUMINAMATH_GPT_range_of_a_l1798_179895

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1798_179895


namespace NUMINAMATH_GPT_find_k_l1798_179820

-- Define the problem's conditions and constants
variables (S x y : ℝ)

-- Define the main theorem to prove k = 8 given the conditions
theorem find_k (h1 : 0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y) = 18) :
  (x * y / 3) / (x + y) = 8 := by 
  sorry

end NUMINAMATH_GPT_find_k_l1798_179820


namespace NUMINAMATH_GPT_gcd_linear_combination_l1798_179810

theorem gcd_linear_combination (a b : ℤ) (h : Int.gcd a b = 1) : 
    Int.gcd (11 * a + 2 * b) (18 * a + 5 * b) = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_linear_combination_l1798_179810


namespace NUMINAMATH_GPT_unsold_tomatoes_l1798_179848

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end NUMINAMATH_GPT_unsold_tomatoes_l1798_179848


namespace NUMINAMATH_GPT_find_special_n_l1798_179899

open Nat

def is_divisor (d n : ℕ) : Prop := n % d = 0

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def special_primes_condition (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem find_special_n (n : ℕ) (h : n > 1) :
  special_primes_condition n → n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_GPT_find_special_n_l1798_179899


namespace NUMINAMATH_GPT_cubic_identity_l1798_179871

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := 
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l1798_179871


namespace NUMINAMATH_GPT_part1_part2_l1798_179890

noncomputable def inverse_function_constant (k : ℝ) : Prop :=
  (∀ x : ℝ, 0 < x → (x, 3) ∈ {p : ℝ × ℝ | p.snd = k / p.fst})

noncomputable def range_m (m : ℝ) : Prop :=
  0 < m → m < 3

theorem part1 (k : ℝ) (hk : k ≠ 0) (h : (1, 3).snd = k / (1, 3).fst) :
  k = 3 := by
  sorry

theorem part2 (m : ℝ) (hm : m ≠ 0) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1798_179890


namespace NUMINAMATH_GPT_complement_P_inter_Q_l1798_179822

def P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}
def complement_P : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_P_inter_Q : (complement_P ∩ Q) = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_complement_P_inter_Q_l1798_179822


namespace NUMINAMATH_GPT_carnival_ring_toss_l1798_179864

theorem carnival_ring_toss (total_amount : ℕ) (days : ℕ) (amount_per_day : ℕ) 
  (h1 : total_amount = 420) 
  (h2 : days = 3) 
  (h3 : total_amount = days * amount_per_day) : amount_per_day = 140 :=
by
  sorry

end NUMINAMATH_GPT_carnival_ring_toss_l1798_179864


namespace NUMINAMATH_GPT_marge_funds_for_fun_l1798_179828

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end NUMINAMATH_GPT_marge_funds_for_fun_l1798_179828


namespace NUMINAMATH_GPT_original_laborers_l1798_179853

theorem original_laborers (x : ℕ) : (x * 8 = (x - 3) * 14) → x = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_laborers_l1798_179853


namespace NUMINAMATH_GPT_identical_solutions_k_value_l1798_179852

theorem identical_solutions_k_value (k : ℝ) :
  (∀ (x y : ℝ), y = x^2 ∧ y = 4 * x + k → (x - 2)^2 = 0) → k = -4 :=
by
  sorry

end NUMINAMATH_GPT_identical_solutions_k_value_l1798_179852


namespace NUMINAMATH_GPT_closest_integer_to_cuberoot_of_200_l1798_179862

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end NUMINAMATH_GPT_closest_integer_to_cuberoot_of_200_l1798_179862


namespace NUMINAMATH_GPT_geometric_and_arithmetic_sequences_l1798_179884

theorem geometric_and_arithmetic_sequences (a b c x y : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : 2 * x = a + b)
  (h3 : 2 * y = b + c) :
  (a / x + c / y) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_and_arithmetic_sequences_l1798_179884


namespace NUMINAMATH_GPT_part1_part2_l1798_179812

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1798_179812


namespace NUMINAMATH_GPT_arithmetic_sequence_minimum_value_S_n_l1798_179841

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_minimum_value_S_n_l1798_179841


namespace NUMINAMATH_GPT_boxes_containing_neither_l1798_179844

theorem boxes_containing_neither (total_boxes markers erasers both : ℕ) 
  (h_total : total_boxes = 15) (h_markers : markers = 8) (h_erasers : erasers = 5) (h_both : both = 4) :
  total_boxes - (markers + erasers - both) = 6 :=
by
  sorry

end NUMINAMATH_GPT_boxes_containing_neither_l1798_179844


namespace NUMINAMATH_GPT_selecting_elements_l1798_179835

theorem selecting_elements (P Q S : ℕ) (a : ℕ) 
    (h1 : P = Nat.choose 17 (2 * a - 1))
    (h2 : Q = Nat.choose 17 (2 * a))
    (h3 : S = Nat.choose 18 12) :
    P + Q = S → (a = 3 ∨ a = 6) :=
by
  sorry

end NUMINAMATH_GPT_selecting_elements_l1798_179835


namespace NUMINAMATH_GPT_fraction_diff_equals_7_over_12_l1798_179802

noncomputable def fraction_diff : ℚ :=
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6)

theorem fraction_diff_equals_7_over_12 : fraction_diff = 7 / 12 := by
  sorry

end NUMINAMATH_GPT_fraction_diff_equals_7_over_12_l1798_179802


namespace NUMINAMATH_GPT_width_of_room_l1798_179849

-- Definitions from conditions
def length : ℝ := 8
def total_cost : ℝ := 34200
def cost_per_sqm : ℝ := 900

-- Theorem stating the width of the room
theorem width_of_room : (total_cost / cost_per_sqm) / length = 4.75 := by 
  sorry

end NUMINAMATH_GPT_width_of_room_l1798_179849


namespace NUMINAMATH_GPT_gcd_min_val_l1798_179821

theorem gcd_min_val (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 1155) : ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 105 :=
by
  sorry

end NUMINAMATH_GPT_gcd_min_val_l1798_179821


namespace NUMINAMATH_GPT_problem_statement_l1798_179842

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1798_179842


namespace NUMINAMATH_GPT_find_e_l1798_179804

-- Define values for a, b, c, d
def a := 2
def b := 3
def c := 4
def d := 5

-- State the problem
theorem find_e (e : ℚ) : a + b + c + d + e = a + (b + (c - (d * e))) → e = -5/6 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l1798_179804


namespace NUMINAMATH_GPT_infinite_zeros_in_S_l1798_179834

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n % 4 = 0 then -↑(n + 1) else
  if n % 4 = 1 then ↑n else
  if n % 4 = 2 then ↑n else
  -↑(n + 1)

-- Define the sequence S_k as partial sum of a_n
def S : ℕ → ℤ
| 0       => a 0
| (n + 1) => S n + a (n + 1)

-- Proposition: S_k contains infinitely many zeros
theorem infinite_zeros_in_S : ∀ n : ℕ, ∃ m > n, S m = 0 := sorry

end NUMINAMATH_GPT_infinite_zeros_in_S_l1798_179834


namespace NUMINAMATH_GPT_Kira_was_away_for_8_hours_l1798_179803

theorem Kira_was_away_for_8_hours
  (kibble_rate: ℕ)
  (initial_kibble: ℕ)
  (remaining_kibble: ℕ)
  (hours_per_pound: ℕ) 
  (kibble_eaten: ℕ)
  (kira_was_away: ℕ)
  (h1: kibble_rate = 1)
  (h2: initial_kibble = 3)
  (h3: remaining_kibble = 1)
  (h4: hours_per_pound = 4)
  (h5: kibble_eaten = initial_kibble - remaining_kibble)
  (h6: kira_was_away = hours_per_pound * kibble_eaten) : 
  kira_was_away = 8 :=
by
  sorry

end NUMINAMATH_GPT_Kira_was_away_for_8_hours_l1798_179803


namespace NUMINAMATH_GPT_greatest_value_q_minus_r_l1798_179826

theorem greatest_value_q_minus_r {x y : ℕ} (hx : x < 10) (hy : y < 10) (hqr : 9 * (x - y) < 70) :
  9 * (x - y) = 63 :=
sorry

end NUMINAMATH_GPT_greatest_value_q_minus_r_l1798_179826


namespace NUMINAMATH_GPT_shopkeeper_gain_l1798_179846

noncomputable def overall_percentage_gain (P : ℝ) (increase_percentage : ℝ) (discount1_percentage : ℝ) (discount2_percentage : ℝ) : ℝ :=
  let increased_price := P * (1 + increase_percentage)
  let price_after_first_discount := increased_price * (1 - discount1_percentage)
  let final_price := price_after_first_discount * (1 - discount2_percentage)
  ((final_price - P) / P) * 100

theorem shopkeeper_gain : 
  overall_percentage_gain 100 0.32 0.10 0.15 = 0.98 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_gain_l1798_179846


namespace NUMINAMATH_GPT_poster_height_proportion_l1798_179845

-- Defining the given conditions
def original_width : ℕ := 3
def original_height : ℕ := 2
def new_width : ℕ := 12
def scale_factor := new_width / original_width

-- The statement to prove the new height
theorem poster_height_proportion :
  scale_factor = 4 → (original_height * scale_factor) = 8 :=
by
  sorry

end NUMINAMATH_GPT_poster_height_proportion_l1798_179845


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l1798_179883

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l1798_179883


namespace NUMINAMATH_GPT_quadratic_graphs_intersect_at_one_point_l1798_179878

theorem quadratic_graphs_intersect_at_one_point
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3)
  (h_intersect_fg : ∃ x₀ : ℝ, (a1 - a2) * x₀^2 + (b1 - b2) * x₀ + (c1 - c2) = 0 ∧ (b1 - b2)^2 - 4 * (a1 - a2) * (c1 - c2) = 0)
  (h_intersect_gh : ∃ x₁ : ℝ, (a2 - a3) * x₁^2 + (b2 - b3) * x₁ + (c2 - c3) = 0 ∧ (b2 - b3)^2 - 4 * (a2 - a3) * (c2 - c3) = 0)
  (h_intersect_fh : ∃ x₂ : ℝ, (a1 - a3) * x₂^2 + (b1 - b3) * x₂ + (c1 - c3) = 0 ∧ (b1 - b3)^2 - 4 * (a1 - a3) * (c1 - c3) = 0) :
  ∃ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0) ∧ (a2 * x^2 + b2 * x + c2 = 0) ∧ (a3 * x^2 + b3 * x + c3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_graphs_intersect_at_one_point_l1798_179878


namespace NUMINAMATH_GPT_least_four_digit_with_factors_3_5_7_l1798_179811

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end NUMINAMATH_GPT_least_four_digit_with_factors_3_5_7_l1798_179811
