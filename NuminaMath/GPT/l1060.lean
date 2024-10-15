import Mathlib

namespace NUMINAMATH_GPT_complete_the_square_d_l1060_106024

theorem complete_the_square_d (x : ℝ) :
  ∃ c d, (x^2 + 10 * x + 9 = 0 → (x + c)^2 = d) ∧ d = 16 :=
sorry

end NUMINAMATH_GPT_complete_the_square_d_l1060_106024


namespace NUMINAMATH_GPT_find_simple_annual_rate_l1060_106038

-- Conditions from part a).
-- 1. Principal initial amount (P) is $5,000.
-- 2. Annual interest rate for compounded interest (r) is 0.06.
-- 3. Number of times it compounds per year (n) is 2 (semi-annually).
-- 4. Time period (t) is 1 year.
-- 5. The interest earned after one year for simple interest is $6 less than compound interest.

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate_compound : ℝ := 0.06
noncomputable def times_compounded : ℕ := 2
noncomputable def time_years : ℝ := 1
noncomputable def compound_interest : ℝ := principal * (1 + annual_rate_compound / times_compounded) ^ (times_compounded * time_years) - principal
noncomputable def simple_interest : ℝ := compound_interest - 6

-- Question from part a) translated to Lean statement using the condition that simple interest satisfaction
theorem find_simple_annual_rate : 
    ∃ r : ℝ, principal * r * time_years = simple_interest :=
by
  exists (0.0597)
  sorry

end NUMINAMATH_GPT_find_simple_annual_rate_l1060_106038


namespace NUMINAMATH_GPT_inequality_holds_l1060_106011

noncomputable def f (x : ℝ) := x^2 + 2 * Real.cos x

theorem inequality_holds (x1 x2 : ℝ) : 
  f x1 > f x2 → x1 > |x2| := 
sorry

end NUMINAMATH_GPT_inequality_holds_l1060_106011


namespace NUMINAMATH_GPT_simplify_expression_l1060_106053

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x⁻¹ - x + 2) = (1 - (x - 1)^2) / x := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1060_106053


namespace NUMINAMATH_GPT_min_value_of_n_l1060_106031

def is_prime (p : ℕ) : Prop := p ≥ 2 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop := ¬ is_prime n

def decomposable_into_primes_leq_10 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≤ 10 ∧ q ≤ 10 ∧ n = p * q

theorem min_value_of_n : ∃ n : ℕ, is_not_prime n ∧ decomposable_into_primes_leq_10 n ∧ n = 6 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_min_value_of_n_l1060_106031


namespace NUMINAMATH_GPT_part1_exists_n_part2_not_exists_n_l1060_106042

open Nat

def is_prime (p : Nat) : Prop := p > 1 ∧ ∀ m : Nat, m ∣ p → m = 1 ∨ m = p

-- Part 1: Prove there exists an n such that n-96, n, n+96 are all primes
theorem part1_exists_n :
  ∃ (n : Nat), is_prime (n - 96) ∧ is_prime n ∧ is_prime (n + 96) :=
sorry

-- Part 2: Prove there does not exist an n such that n-1996, n, n+1996 are all primes
theorem part2_not_exists_n :
  ¬ (∃ (n : Nat), is_prime (n - 1996) ∧ is_prime n ∧ is_prime (n + 1996)) :=
sorry

end NUMINAMATH_GPT_part1_exists_n_part2_not_exists_n_l1060_106042


namespace NUMINAMATH_GPT_fraction_equality_l1060_106057

theorem fraction_equality (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := 
by
  -- Use the hypthesis to derive that a = 2k, b = 3k, c = 4k and show the equality.
  sorry

end NUMINAMATH_GPT_fraction_equality_l1060_106057


namespace NUMINAMATH_GPT_adele_age_fraction_l1060_106005

theorem adele_age_fraction 
  (jackson_age : ℕ) 
  (mandy_age : ℕ) 
  (adele_age_fraction : ℚ) 
  (total_age_10_years : ℕ)
  (H1 : jackson_age = 20)
  (H2 : mandy_age = jackson_age + 10)
  (H3 : total_age_10_years = (jackson_age + 10) + (mandy_age + 10) + (jackson_age * adele_age_fraction + 10))
  (H4 : total_age_10_years = 95) : 
  adele_age_fraction = 3 / 4 := 
sorry

end NUMINAMATH_GPT_adele_age_fraction_l1060_106005


namespace NUMINAMATH_GPT_abs_diff_26th_term_l1060_106032

def C (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def D (n : ℕ) : ℤ := 85 - 20 * (n - 1)

theorem abs_diff_26th_term :
  |(C 26) - (D 26)| = 840 := by
  sorry

end NUMINAMATH_GPT_abs_diff_26th_term_l1060_106032


namespace NUMINAMATH_GPT_parallel_lines_l1060_106015

theorem parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, a * x + 2 * y - 1 = k * (2 * x + a * y + 2)) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l1060_106015


namespace NUMINAMATH_GPT_computation_l1060_106099

theorem computation :
  52 * 46 + 104 * 52 = 7800 := by
  sorry

end NUMINAMATH_GPT_computation_l1060_106099


namespace NUMINAMATH_GPT_sum_of_variables_l1060_106087

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : 
  x + y + z = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_variables_l1060_106087


namespace NUMINAMATH_GPT_jack_lap_time_improvement_l1060_106029

/-!
Jack practices running in a stadium. Initially, he completed 15 laps in 45 minutes.
After a month of training, he completed 18 laps in 42 minutes. By how many minutes 
has he improved his lap time?
-/

theorem jack_lap_time_improvement:
  ∀ (initial_laps current_laps : ℕ) 
    (initial_time current_time : ℝ), 
    initial_laps = 15 → 
    current_laps = 18 → 
    initial_time = 45 → 
    current_time = 42 → 
    (initial_time / initial_laps - current_time / current_laps = 2/3) :=
by 
  intros _ _ _ _ h_initial_laps h_current_laps h_initial_time h_current_time
  rw [h_initial_laps, h_current_laps, h_initial_time, h_current_time]
  sorry

end NUMINAMATH_GPT_jack_lap_time_improvement_l1060_106029


namespace NUMINAMATH_GPT_log2_square_eq_37_l1060_106016

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_square_eq_37
  {x y : ℝ}
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_log : log2 x = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (log2 (x / y))^2 = 37 := by
  sorry

end NUMINAMATH_GPT_log2_square_eq_37_l1060_106016


namespace NUMINAMATH_GPT_necessary_condition_ac_eq_bc_l1060_106094

theorem necessary_condition_ac_eq_bc {a b c : ℝ} (hc : c ≠ 0) : (ac = bc ↔ a = b) := by
  sorry

end NUMINAMATH_GPT_necessary_condition_ac_eq_bc_l1060_106094


namespace NUMINAMATH_GPT_other_root_l1060_106052

theorem other_root (x : ℚ) (h : 48 * x^2 + 29 = 35 * x + 12) : x = 3 / 4 ∨ x = 1 / 3 := 
by {
  -- Proof can be filled in here
  sorry
}

end NUMINAMATH_GPT_other_root_l1060_106052


namespace NUMINAMATH_GPT_calculate_expr_l1060_106076

theorem calculate_expr (h1 : Real.sin (30 * Real.pi / 180) = 1 / 2)
    (h2 : Real.cos (30 * Real.pi / 180) = Real.sqrt (3) / 2) :
    3 * Real.tan (30 * Real.pi / 180) + 6 * Real.sin (30 * Real.pi / 180) = 3 + Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_calculate_expr_l1060_106076


namespace NUMINAMATH_GPT_decreasing_interval_of_function_l1060_106025

noncomputable def y (x : ℝ) : ℝ := (3 / Real.pi) ^ (x ^ 2 + 2 * x - 3)

theorem decreasing_interval_of_function :
  ∀ x ∈ Set.Ioi (-1 : ℝ), ∃ ε > 0, ∀ δ > 0, δ ≤ ε → y (x - δ) > y x :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_of_function_l1060_106025


namespace NUMINAMATH_GPT_product_of_solutions_l1060_106027

theorem product_of_solutions : 
  (∃ x1 x2 : ℝ, |5 * x1 - 1| + 4 = 54 ∧ |5 * x2 - 1| + 4 = 54 ∧ x1 * x2 = -99.96) :=
  by sorry

end NUMINAMATH_GPT_product_of_solutions_l1060_106027


namespace NUMINAMATH_GPT_circle_center_l1060_106092

theorem circle_center : ∃ (a b : ℝ), (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y - 4 = 0 ↔ (x - a)^2 + (y - b)^2 = 9) ∧ a = 1 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_circle_center_l1060_106092


namespace NUMINAMATH_GPT_prove_scientific_notation_l1060_106023

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end NUMINAMATH_GPT_prove_scientific_notation_l1060_106023


namespace NUMINAMATH_GPT_common_chord_eq_l1060_106049

theorem common_chord_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 8*y - 8 = 0) → (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
    x + 2*y - 1 = 0 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_eq_l1060_106049


namespace NUMINAMATH_GPT_hawks_total_points_l1060_106079

/-- 
  Define the number of points per touchdown 
  and the number of touchdowns scored by the Hawks. 
-/
def points_per_touchdown : ℕ := 7
def touchdowns : ℕ := 3

/-- 
  Prove that the total number of points the Hawks have is 21. 
-/
theorem hawks_total_points : touchdowns * points_per_touchdown = 21 :=
by
  sorry

end NUMINAMATH_GPT_hawks_total_points_l1060_106079


namespace NUMINAMATH_GPT_new_supervisor_salary_correct_l1060_106033

noncomputable def salary_new_supervisor
  (avg_salary_old : ℝ)
  (old_supervisor_salary : ℝ)
  (avg_salary_new : ℝ)
  (workers_count : ℝ)
  (total_salary_workers : ℝ := (avg_salary_old * (workers_count + 1)) - old_supervisor_salary)
  (new_supervisor_salary : ℝ := (avg_salary_new * (workers_count + 1)) - total_salary_workers)
  : ℝ :=
  new_supervisor_salary

theorem new_supervisor_salary_correct :
  salary_new_supervisor 430 870 420 8 = 780 :=
by
  simp [salary_new_supervisor]
  sorry

end NUMINAMATH_GPT_new_supervisor_salary_correct_l1060_106033


namespace NUMINAMATH_GPT_percent_notebooks_staplers_clips_l1060_106009

def percent_not_special (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) : ℝ :=
  100 - (n + s + c)

theorem percent_notebooks_staplers_clips (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) :
  percent_not_special n s c h_n h_s h_c = 25 :=
by
  unfold percent_not_special
  rw [h_n, h_s, h_c]
  norm_num

end NUMINAMATH_GPT_percent_notebooks_staplers_clips_l1060_106009


namespace NUMINAMATH_GPT_sequence_term_formula_l1060_106001

open Real

def sequence_sum_condition (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S n + a n = 4 - 1 / (2 ^ (n - 2))

theorem sequence_term_formula 
  (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : sequence_sum_condition S a) :
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_term_formula_l1060_106001


namespace NUMINAMATH_GPT_Okeydokey_should_receive_25_earthworms_l1060_106036

def applesOkeydokey : ℕ := 5
def applesArtichokey : ℕ := 7
def totalEarthworms : ℕ := 60
def totalApples : ℕ := applesOkeydokey + applesArtichokey
def okeydokeyProportion : ℚ := applesOkeydokey / totalApples
def okeydokeyEarthworms : ℚ := okeydokeyProportion * totalEarthworms

theorem Okeydokey_should_receive_25_earthworms : okeydokeyEarthworms = 25 := by
  sorry

end NUMINAMATH_GPT_Okeydokey_should_receive_25_earthworms_l1060_106036


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_y_l1060_106058

theorem inequality_holds_for_all_x_y (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ x + y + x * y := 
by sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_y_l1060_106058


namespace NUMINAMATH_GPT_repeating_decimal_division_l1060_106026

theorem repeating_decimal_division:
  let x := (54 / 99 : ℚ)
  let y := (18 / 99 : ℚ)
  (x / y) * (1 / 2) = (3 / 2 : ℚ) := by
    sorry

end NUMINAMATH_GPT_repeating_decimal_division_l1060_106026


namespace NUMINAMATH_GPT_seashells_total_now_l1060_106040

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end NUMINAMATH_GPT_seashells_total_now_l1060_106040


namespace NUMINAMATH_GPT_quadratic_roots_l1060_106063

theorem quadratic_roots (x : ℝ) : 
  (2 * x^2 - 4 * x - 5 = 0) ↔ 
  (x = (2 + Real.sqrt 14) / 2 ∨ x = (2 - Real.sqrt 14) / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1060_106063


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l1060_106044

-- Definitions of the conditions
def total_tickets (f h : ℕ) : Prop := f + h = 180
def total_revenue (f h p : ℕ) : Prop := f * p + h * (p / 2) = 2750

-- Theorem statement
theorem revenue_from_full_price_tickets (f h p : ℕ) 
  (h_total_tickets : total_tickets f h) 
  (h_total_revenue : total_revenue f h p) : 
  f * p = 1000 :=
  sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l1060_106044


namespace NUMINAMATH_GPT_domain_of_log_function_l1060_106051

-- Define the problematic quadratic function
def quadratic_fn (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Define the domain condition for our function
def domain_condition (x : ℝ) : Prop := quadratic_fn x > 0

-- The actual statement to prove, stating that the domain is (1, 3)
theorem domain_of_log_function :
  {x : ℝ | domain_condition x} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_domain_of_log_function_l1060_106051


namespace NUMINAMATH_GPT_Mark_same_color_opposite_foot_l1060_106077

variable (shoes : Finset (Σ _ : Fin (14), Bool))

def same_color_opposite_foot_probability (shoes : Finset (Σ _ : Fin (14), Bool)) : ℚ := 
  let total_shoes : ℚ := 28
  let num_black_pairs := 7
  let num_brown_pairs := 4
  let num_gray_pairs := 2
  let num_white_pairs := 1
  let black_pair_prob  := (14 / total_shoes) * (7 / (total_shoes - 1))
  let brown_pair_prob  := (8 / total_shoes) * (4 / (total_shoes - 1))
  let gray_pair_prob   := (4 / total_shoes) * (2 / (total_shoes - 1))
  let white_pair_prob  := (2 / total_shoes) * (1 / (total_shoes - 1))
  black_pair_prob + brown_pair_prob + gray_pair_prob + white_pair_prob

theorem Mark_same_color_opposite_foot (shoes : Finset (Σ _ : Fin (14), Bool)) :
  same_color_opposite_foot_probability shoes = 35 / 189 := 
sorry

end NUMINAMATH_GPT_Mark_same_color_opposite_foot_l1060_106077


namespace NUMINAMATH_GPT_correct_operation_l1060_106067

-- Define the operations given in the conditions
def optionA (m : ℝ) := m^2 + m^2 = 2 * m^4
def optionB (a : ℝ) := a^2 * a^3 = a^5
def optionC (m n : ℝ) := (m * n^2) ^ 3 = m * n^6
def optionD (m : ℝ) := m^6 / m^2 = m^3

-- Theorem stating that option B is the correct operation
theorem correct_operation (a m n : ℝ) : optionB a :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1060_106067


namespace NUMINAMATH_GPT_number_of_bowls_l1060_106081

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_bowls_l1060_106081


namespace NUMINAMATH_GPT_abs_neg_four_squared_plus_six_l1060_106086

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end NUMINAMATH_GPT_abs_neg_four_squared_plus_six_l1060_106086


namespace NUMINAMATH_GPT_find_f_and_q_l1060_106019

theorem find_f_and_q (m : ℤ) (q : ℝ) :
  (∀ x > 0, (x : ℝ)^(-m^2 + 2*m + 3) = (x : ℝ)^4) ∧
  (∀ x ∈ [-1, 1], 2 * (x^2) - 8 * x + q - 1 > 0) →
  q > 7 :=
by
  sorry

end NUMINAMATH_GPT_find_f_and_q_l1060_106019


namespace NUMINAMATH_GPT_wendy_third_day_miles_l1060_106068

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end NUMINAMATH_GPT_wendy_third_day_miles_l1060_106068


namespace NUMINAMATH_GPT_factor_x4_plus_64_l1060_106028

theorem factor_x4_plus_64 (x : ℝ) : 
  (x^4 + 64) = (x^2 - 4 * x + 8) * (x^2 + 4 * x + 8) :=
sorry

end NUMINAMATH_GPT_factor_x4_plus_64_l1060_106028


namespace NUMINAMATH_GPT_complex_root_seventh_power_l1060_106084

theorem complex_root_seventh_power (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end NUMINAMATH_GPT_complex_root_seventh_power_l1060_106084


namespace NUMINAMATH_GPT_multiply_exponents_l1060_106055

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end NUMINAMATH_GPT_multiply_exponents_l1060_106055


namespace NUMINAMATH_GPT_calculate_dividend_l1060_106072

def divisor : ℕ := 21
def quotient : ℕ := 14
def remainder : ℕ := 7
def expected_dividend : ℕ := 301

theorem calculate_dividend : (divisor * quotient + remainder = expected_dividend) := 
by
  sorry

end NUMINAMATH_GPT_calculate_dividend_l1060_106072


namespace NUMINAMATH_GPT_charity_tickets_l1060_106020

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end NUMINAMATH_GPT_charity_tickets_l1060_106020


namespace NUMINAMATH_GPT_calculate_fg_l1060_106037

def f (x : ℝ) : ℝ := x - 4

def g (x : ℝ) : ℝ := x^2 + 5

theorem calculate_fg : f (g (-3)) = 10 := by
  sorry

end NUMINAMATH_GPT_calculate_fg_l1060_106037


namespace NUMINAMATH_GPT_range_of_a_l1060_106000

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2 * t - a < 0) ↔ a ≤ -1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1060_106000


namespace NUMINAMATH_GPT_lion_turn_angles_l1060_106047

-- Define the radius of the circle
def radius (r : ℝ) := r = 10

-- Define the path length the lion runs in meters
def path_length (d : ℝ) := d = 30000

-- Define the final goal: The sum of all the angles of its turns is at least 2998 radians
theorem lion_turn_angles (r d : ℝ) (α : ℝ) (hr : radius r) (hd : path_length d) (hα : d ≤ 10 * α) : α ≥ 2998 := 
sorry

end NUMINAMATH_GPT_lion_turn_angles_l1060_106047


namespace NUMINAMATH_GPT_intersection_A_B_l1060_106075

-- Define sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

-- The theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry -- proof is skipped as instructed

end NUMINAMATH_GPT_intersection_A_B_l1060_106075


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_and_difference_l1060_106014

theorem arithmetic_sequence_first_term_and_difference
  (a1 d : ℤ)
  (h1 : (a1 + 2 * d) * (a1 + 5 * d) = 406)
  (h2 : a1 + 8 * d = 2 * (a1 + 3 * d) + 6) : 
  a1 = 4 ∧ d = 5 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_and_difference_l1060_106014


namespace NUMINAMATH_GPT_seq_ratio_l1060_106007

theorem seq_ratio (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n, a n * a (n + 1) = 2^n) : 
  a 7 / a 3 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_seq_ratio_l1060_106007


namespace NUMINAMATH_GPT_tamika_greater_probability_l1060_106095

-- Definitions for the conditions
def tamika_results : Set ℕ := {11 * 12, 11 * 13, 12 * 13}
def carlos_result : ℕ := 2 + 3 + 4

-- Theorem stating the problem
theorem tamika_greater_probability : 
  (∀ r ∈ tamika_results, r > carlos_result) → (1 : ℚ) = 1 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_tamika_greater_probability_l1060_106095


namespace NUMINAMATH_GPT_value_of_abs_div_sum_l1060_106054

theorem value_of_abs_div_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|a| / a + |b| / b = 2) ∨ (|a| / a + |b| / b = -2) ∨ (|a| / a + |b| / b = 0) := 
by
  sorry

end NUMINAMATH_GPT_value_of_abs_div_sum_l1060_106054


namespace NUMINAMATH_GPT_min_value_x1_squared_plus_x2_squared_plus_x3_squared_l1060_106043

theorem min_value_x1_squared_plus_x2_squared_plus_x3_squared
    (x1 x2 x3 : ℝ) 
    (h1 : 3 * x1 + 2 * x2 + x3 = 30) 
    (h2 : x1 > 0) 
    (h3 : x2 > 0) 
    (h4 : x3 > 0) : 
    x1^2 + x2^2 + x3^2 ≥ 125 := 
  by sorry

end NUMINAMATH_GPT_min_value_x1_squared_plus_x2_squared_plus_x3_squared_l1060_106043


namespace NUMINAMATH_GPT_avg_annual_growth_rate_l1060_106098

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_l1060_106098


namespace NUMINAMATH_GPT_total_students_in_lab_l1060_106062

def total_workstations : Nat := 16
def workstations_for_2_students : Nat := 10
def students_per_workstation_2 : Nat := 2
def students_per_workstation_3 : Nat := 3

theorem total_students_in_lab :
  let workstations_with_2_students := workstations_for_2_students
  let workstations_with_3_students := total_workstations - workstations_for_2_students
  let students_in_2_student_workstations := workstations_with_2_students * students_per_workstation_2
  let students_in_3_student_workstations := workstations_with_3_students * students_per_workstation_3
  students_in_2_student_workstations + students_in_3_student_workstations = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_lab_l1060_106062


namespace NUMINAMATH_GPT_equation_proof_l1060_106089

theorem equation_proof :
  (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_equation_proof_l1060_106089


namespace NUMINAMATH_GPT_banana_unique_permutations_l1060_106041

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end NUMINAMATH_GPT_banana_unique_permutations_l1060_106041


namespace NUMINAMATH_GPT_james_total_oop_correct_l1060_106006

-- Define the costs and insurance coverage percentages as given conditions.
def cost_consultation : ℝ := 300
def coverage_consultation : ℝ := 0.80

def cost_xray : ℝ := 150
def coverage_xray : ℝ := 0.70

def cost_prescription : ℝ := 75
def coverage_prescription : ℝ := 0.50

def cost_therapy : ℝ := 120
def coverage_therapy : ℝ := 0.60

-- Define the out-of-pocket calculation for each service
def oop_consultation := cost_consultation * (1 - coverage_consultation)
def oop_xray := cost_xray * (1 - coverage_xray)
def oop_prescription := cost_prescription * (1 - coverage_prescription)
def oop_therapy := cost_therapy * (1 - coverage_therapy)

-- Define the total out-of-pocket cost
def total_oop : ℝ := oop_consultation + oop_xray + oop_prescription + oop_therapy

-- Proof statement
theorem james_total_oop_correct : total_oop = 190.50 := by
  sorry

end NUMINAMATH_GPT_james_total_oop_correct_l1060_106006


namespace NUMINAMATH_GPT_part1_part2_l1060_106064

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x - 4 ≤ 0}

-- Problem 1
theorem part1 (m : ℝ) : 
  (A ∩ B m = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) → m = 3 :=
by sorry

-- Problem 2
theorem part2 (m : ℝ) : 
  (A ⊆ (B m)ᶜ) → (m < -3 ∨ m > 5) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1060_106064


namespace NUMINAMATH_GPT_arithmetic_mean_of_two_numbers_l1060_106061

def is_arithmetic_mean (x y z : ℚ) : Prop :=
  (x + z) / 2 = y

theorem arithmetic_mean_of_two_numbers :
  is_arithmetic_mean (9 / 12) (5 / 6) (7 / 8) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_two_numbers_l1060_106061


namespace NUMINAMATH_GPT_students_taking_history_but_not_statistics_l1060_106045

theorem students_taking_history_but_not_statistics (H S U : ℕ) (total_students : ℕ) 
  (H_val : H = 36) (S_val : S = 30) (U_val : U = 59) (total_students_val : total_students = 90) :
  H - (H + S - U) = 29 := 
by
  sorry

end NUMINAMATH_GPT_students_taking_history_but_not_statistics_l1060_106045


namespace NUMINAMATH_GPT_area_shaded_region_in_hexagon_l1060_106059

theorem area_shaded_region_in_hexagon (s : ℝ) (r : ℝ) (h_s : s = 4) (h_r : r = 2) :
  let area_hexagon := ((3 * Real.sqrt 3) / 2) * s^2
  let area_semicircle := (π * r^2) / 2
  let total_area_semicircles := 8 * area_semicircle
  let area_shaded_region := area_hexagon - total_area_semicircles
  area_shaded_region = 24 * Real.sqrt 3 - 16 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_area_shaded_region_in_hexagon_l1060_106059


namespace NUMINAMATH_GPT_find_a_minus_c_l1060_106039

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 80) (h2 : (b + c) / 2 = 180) : a - c = -200 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_minus_c_l1060_106039


namespace NUMINAMATH_GPT_find_a_14_l1060_106022

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence sum formula
def arithmetic_seq_sum (a_1 d : α) (n : ℕ) : α :=
  n * a_1 + n * (n - 1) / 2 * d

-- Define the nth term of an arithmetic sequence
def arithmetic_seq_nth (a_1 d : α) (n : ℕ) : α :=
  a_1 + (n - 1 : ℕ) * d

theorem find_a_14
  (a_1 d : α)
  (h1 : arithmetic_seq_sum a_1 d 11 = 55)
  (h2 : arithmetic_seq_nth a_1 d 10 = 9) :
  arithmetic_seq_nth a_1 d 14 = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_a_14_l1060_106022


namespace NUMINAMATH_GPT_face_value_of_ticket_l1060_106004
noncomputable def face_value_each_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) : ℝ :=
  total_price / (group_size * (1 + tax_rate))

theorem face_value_of_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) :
  total_price = 945 →
  group_size = 25 →
  tax_rate = 0.05 →
  face_value_each_ticket total_price group_size tax_rate = 36 := 
by
  intros h_total_price h_group_size h_tax_rate
  rw [h_total_price, h_group_size, h_tax_rate]
  simp [face_value_each_ticket]
  sorry

end NUMINAMATH_GPT_face_value_of_ticket_l1060_106004


namespace NUMINAMATH_GPT_cost_per_box_l1060_106030

theorem cost_per_box (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℕ) (box_cost : ℝ) 
  (h1 : trays = 3) 
  (h2 : cookies_per_tray = 80) 
  (h3 : cookies_per_box = 60)
  (h4 : total_cost = 14) 
  (h5 : (trays * cookies_per_tray) = 240)
  (h6 : (240 / cookies_per_box : ℕ) = 4) 
  (h7 : (total_cost / 4 : ℝ) = box_cost) : 
  box_cost = 3.5 := 
by sorry

end NUMINAMATH_GPT_cost_per_box_l1060_106030


namespace NUMINAMATH_GPT_student_ticket_cost_l1060_106008

theorem student_ticket_cost 
  (total_tickets_sold : ℕ) 
  (total_revenue : ℕ) 
  (nonstudent_ticket_cost : ℕ) 
  (student_tickets_sold : ℕ) 
  (cost_per_student_ticket : ℕ) 
  (nonstudent_tickets_sold : ℕ) 
  (H1 : total_tickets_sold = 821) 
  (H2 : total_revenue = 1933)
  (H3 : nonstudent_ticket_cost = 3)
  (H4 : student_tickets_sold = 530) 
  (H5 : nonstudent_tickets_sold = total_tickets_sold - student_tickets_sold)
  (H6 : 530 * cost_per_student_ticket + nonstudent_tickets_sold * 3 = 1933) : 
  cost_per_student_ticket = 2 := 
by
  sorry

end NUMINAMATH_GPT_student_ticket_cost_l1060_106008


namespace NUMINAMATH_GPT_complex_numbers_equation_l1060_106046

theorem complex_numbers_equation {a b : ℂ} (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := 
by sorry

end NUMINAMATH_GPT_complex_numbers_equation_l1060_106046


namespace NUMINAMATH_GPT_integer_solutions_k_l1060_106021

theorem integer_solutions_k (k n m : ℤ) (h1 : k + 1 = n^2) (h2 : 16 * k + 1 = m^2) :
  k = 0 ∨ k = 3 :=
by sorry

end NUMINAMATH_GPT_integer_solutions_k_l1060_106021


namespace NUMINAMATH_GPT_quadratic_root_expression_l1060_106088

theorem quadratic_root_expression (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 + x - 2023 = 0 → (x = a ∨ x = b)) 
  (ha_neq_b : a ≠ b) :
  a^2 + 2*a + b = 2022 :=
sorry

end NUMINAMATH_GPT_quadratic_root_expression_l1060_106088


namespace NUMINAMATH_GPT_subscriptions_sold_to_parents_l1060_106035

-- Definitions for the conditions
variable (P : Nat) -- subscriptions sold to parents
def grandfather := 1
def next_door_neighbor := 2
def other_neighbor := 2 * next_door_neighbor
def subscriptions_other_than_parents := grandfather + next_door_neighbor + other_neighbor
def total_earnings := 55
def earnings_from_others := 5 * subscriptions_other_than_parents
def earnings_from_parents := total_earnings - earnings_from_others
def subscription_price := 5

-- Theorem stating the equivalent math proof
theorem subscriptions_sold_to_parents : P = earnings_from_parents / subscription_price :=
by
  sorry

end NUMINAMATH_GPT_subscriptions_sold_to_parents_l1060_106035


namespace NUMINAMATH_GPT_complex_z_solution_l1060_106002

theorem complex_z_solution (z : ℂ) (i : ℂ) (h : i * z = 1 - i) (hi : i * i = -1) : z = -1 - i :=
by sorry

end NUMINAMATH_GPT_complex_z_solution_l1060_106002


namespace NUMINAMATH_GPT_find_smallest_d_l1060_106070

noncomputable def smallest_possible_d (c d : ℕ) : ℕ :=
  if c - d = 8 ∧ Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 then d else 0

-- Proving the smallest possible value of d given the conditions
theorem find_smallest_d :
  ∀ c d : ℕ, (0 < c) → (0 < d) → (c - d = 8) → 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 → d = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_d_l1060_106070


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1060_106060

noncomputable def f (x a : ℝ) := |x - a| + x

theorem problem_part1 (a : ℝ) (h_a : a = 1) :
  {x : ℝ | f x a ≥ x + 2} = {x : ℝ | x ≥ 3} ∪ {x : ℝ | x ≤ -1} :=
by 
  simp [h_a, f]
  sorry

theorem problem_part2 (a : ℝ) (h_solution : {x : ℝ | f x a ≤ 3 * x} = {x : ℝ | x ≥ 2}) :
  a = 6 :=
by
  simp [f] at h_solution
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1060_106060


namespace NUMINAMATH_GPT_game_24_set1_game_24_set2_l1060_106071

-- Equivalent proof problem for set {3, 2, 6, 7}
theorem game_24_set1 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 6) (h₄ : d = 7) :
  ((d / b) * c + a) = 24 := by
  subst_vars
  sorry

-- Equivalent proof problem for set {3, 4, -6, 10}
theorem game_24_set2 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = -6) (h₄ : d = 10) :
  ((b + c + d) * a) = 24 := by
  subst_vars
  sorry

end NUMINAMATH_GPT_game_24_set1_game_24_set2_l1060_106071


namespace NUMINAMATH_GPT_no_2007_in_display_can_2008_appear_in_display_l1060_106066

-- Definitions of the operations as functions on the display number.
def button1 (n : ℕ) : ℕ := 1
def button2 (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n
def button3 (n : ℕ) : ℕ := if n >= 3 then n - 3 else n
def button4 (n : ℕ) : ℕ := 4 * n

-- Initial condition
def initial_display : ℕ := 0

-- Define can_appear as a recursive function to determine if a number can appear on the display.
def can_appear (target : ℕ) : Prop :=
  ∃ n : ℕ, n = target ∧ (∃ f : (ℕ → ℕ) → ℕ, f initial_display = target)

-- Prove the statements:
theorem no_2007_in_display : ¬ can_appear 2007 :=
  sorry

theorem can_2008_appear_in_display : can_appear 2008 :=
  sorry

end NUMINAMATH_GPT_no_2007_in_display_can_2008_appear_in_display_l1060_106066


namespace NUMINAMATH_GPT_imperative_sentence_structure_l1060_106010

theorem imperative_sentence_structure (word : String) (is_base_form : word = "Surround") :
  (word = "Surround" ∨ word = "Surrounding" ∨ word = "Surrounded" ∨ word = "Have surrounded") →
  (∃ sentence : String, sentence = word ++ " yourself with positive people, and you will keep focused on what you can do instead of what you can’t.") →
  word = "Surround" :=
by
  intros H_choice H_sentence
  cases H_choice
  case inl H1 => assumption
  case inr H2_1 =>
    cases H2_1
    case inl H2_1_1 => sorry
    case inr H2_1_2 =>
      cases H2_1_2
      case inl H2_1_2_1 => sorry
      case inr H2_1_2_2 => sorry

end NUMINAMATH_GPT_imperative_sentence_structure_l1060_106010


namespace NUMINAMATH_GPT_evaluate_expression_l1060_106003

theorem evaluate_expression :
  (2:ℝ) ^ ((0:ℝ) ^ (Real.sin (Real.pi / 2)) ^ 2) + ((3:ℝ) ^ 0) ^ 1 ^ 4 = 2 := by
  -- Given conditions
  have h1 : Real.sin (Real.pi / 2) = 1 := by sorry
  have h2 : (3:ℝ) ^ 0 = 1 := by sorry
  have h3 : (0:ℝ) ^ 1 = 0 := by sorry
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1060_106003


namespace NUMINAMATH_GPT_Mr_Kishore_saved_10_percent_l1060_106096

-- Define the costs and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

-- Define the total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage saved
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- The statement to prove
theorem Mr_Kishore_saved_10_percent : percentage_saved = 10 := by
  sorry

end NUMINAMATH_GPT_Mr_Kishore_saved_10_percent_l1060_106096


namespace NUMINAMATH_GPT_find_v5_l1060_106082

noncomputable def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + v n + 1

theorem find_v5 :
  ∃ (v : ℕ → ℝ), sequence v ∧ v 3 = 11 ∧ v 6 = 242 ∧ v 5 = 73.5 :=
by
  sorry

end NUMINAMATH_GPT_find_v5_l1060_106082


namespace NUMINAMATH_GPT_sea_lions_count_l1060_106017

theorem sea_lions_count (S P : ℕ) (h1 : 11 * S = 4 * P) (h2 : P = S + 84) : S = 48 := 
by {
  sorry
}

end NUMINAMATH_GPT_sea_lions_count_l1060_106017


namespace NUMINAMATH_GPT_max_f_l1060_106065

noncomputable def f (x : ℝ) : ℝ :=
  min (min (2 * x + 2) (1 / 2 * x + 1)) (-3 / 4 * x + 7)

theorem max_f : ∃ x : ℝ, f x = 17 / 5 :=
by
  sorry

end NUMINAMATH_GPT_max_f_l1060_106065


namespace NUMINAMATH_GPT_number_consisting_of_11_hundreds_11_tens_and_11_units_l1060_106056

theorem number_consisting_of_11_hundreds_11_tens_and_11_units :
  11 * 100 + 11 * 10 + 11 = 1221 :=
by
  sorry

end NUMINAMATH_GPT_number_consisting_of_11_hundreds_11_tens_and_11_units_l1060_106056


namespace NUMINAMATH_GPT_solve_system_l1060_106085

theorem solve_system (x y z a : ℝ) 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = a^2) 
  (h3 : x^3 + y^3 + z^3 = a^3) : 
  (x = 0 ∧ y = 0 ∧ z = a) ∨ 
  (x = 0 ∧ y = a ∧ z = 0) ∨ 
  (x = a ∧ y = 0 ∧ z = 0) := 
sorry

end NUMINAMATH_GPT_solve_system_l1060_106085


namespace NUMINAMATH_GPT_intersection_P_Q_l1060_106012

section set_intersection

variable (x : ℝ)

def P := { x : ℝ | x ≤ 1 }
def Q := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : { x | x ∈ P ∧ x ∈ Q } = { x | -1 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end set_intersection

end NUMINAMATH_GPT_intersection_P_Q_l1060_106012


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_fraction_l1060_106050

theorem arithmetic_geometric_sequence_fraction 
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 + a2 = 10)
  (h2 : 1 * b3 = 9)
  (h3 : b2 ^ 2 = 9) : 
  b2 / (a1 + a2) = 3 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_fraction_l1060_106050


namespace NUMINAMATH_GPT_find_slope_of_line_l1060_106048

theorem find_slope_of_line (k m x0 : ℝ) (P Q : ℝ × ℝ) 
  (hP : P.2^2 = 4 * P.1) 
  (hQ : Q.2^2 = 4 * Q.1) 
  (hMid : (P.1 + Q.1) / 2 = x0 ∧ (P.2 + Q.2) / 2 = 2) 
  (hLineP : P.2 = k * P.1 + m) 
  (hLineQ : Q.2 = k * Q.1 + m) : k = 1 :=
by sorry

end NUMINAMATH_GPT_find_slope_of_line_l1060_106048


namespace NUMINAMATH_GPT_geom_seq_property_l1060_106078

noncomputable def a_n : ℕ → ℝ := sorry  -- The definition of the geometric sequence

theorem geom_seq_property (a_n : ℕ → ℝ) (h : a_n 6 + a_n 8 = 4) :
  a_n 8 * (a_n 4 + 2 * a_n 6 + a_n 8) = 16 := by
sorry

end NUMINAMATH_GPT_geom_seq_property_l1060_106078


namespace NUMINAMATH_GPT_kylie_beads_total_l1060_106091

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end NUMINAMATH_GPT_kylie_beads_total_l1060_106091


namespace NUMINAMATH_GPT_total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l1060_106083

-- Initial conditions
def cost_4_oranges : Nat := 12
def cost_7_oranges : Nat := 28
def total_oranges : Nat := 28

-- Calculate the total cost for 28 oranges
theorem total_cost_28_oranges
  (x y : Nat) 
  (h1 : 4 * x + 7 * y = total_oranges) 
  (h2 : total_oranges = 28) 
  (h3 : x = 7) 
  (h4 : y = 0) : 
  7 * cost_4_oranges = 84 := 
by sorry

-- Calculate the average cost per orange
theorem avg_cost_per_orange 
  (total_cost : Nat) 
  (h1 : total_cost = 84)
  (h2 : total_oranges = 28) : 
  total_cost / total_oranges = 3 := 
by sorry

-- Calculate the cost for 6 oranges
theorem cost_6_oranges 
  (avg_cost : Nat)
  (h1 : avg_cost = 3)
  (n : Nat) 
  (h2 : n = 6) : 
  n * avg_cost = 18 := 
by sorry

end NUMINAMATH_GPT_total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l1060_106083


namespace NUMINAMATH_GPT_tangent_line_exists_l1060_106069

noncomputable def tangent_line_problem := ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Int.gcd (Int.gcd a b) c = 1 ∧ 
  (∀ x y : ℝ, a * x + b * (x^2 + 52 / 25) = c ∧ a * (y^2 + 81 / 16) + b * y = c) ∧ 
  a + b + c = 168

theorem tangent_line_exists : tangent_line_problem := by
  sorry

end NUMINAMATH_GPT_tangent_line_exists_l1060_106069


namespace NUMINAMATH_GPT_total_goals_is_50_l1060_106013

def team_a_first_half_goals := 8
def team_b_first_half_goals := team_a_first_half_goals / 2
def team_c_first_half_goals := 2 * team_b_first_half_goals
def team_a_first_half_missed_penalty := 1
def team_c_first_half_missed_penalty := 2

def team_a_second_half_goals := team_c_first_half_goals
def team_b_second_half_goals := team_a_first_half_goals
def team_c_second_half_goals := team_b_second_half_goals + 3
def team_a_second_half_successful_penalty := 1
def team_b_second_half_successful_penalty := 2

def total_team_a_goals := team_a_first_half_goals + team_a_second_half_goals + team_a_second_half_successful_penalty
def total_team_b_goals := team_b_first_half_goals + team_b_second_half_goals + team_b_second_half_successful_penalty
def total_team_c_goals := team_c_first_half_goals + team_c_second_half_goals

def total_goals := total_team_a_goals + total_team_b_goals + total_team_c_goals

theorem total_goals_is_50 : total_goals = 50 := by
  unfold total_goals
  unfold total_team_a_goals total_team_b_goals total_team_c_goals
  unfold team_a_first_half_goals team_b_first_half_goals team_c_first_half_goals
  unfold team_a_second_half_goals team_b_second_half_goals team_c_second_half_goals
  unfold team_a_second_half_successful_penalty team_b_second_half_successful_penalty
  sorry

end NUMINAMATH_GPT_total_goals_is_50_l1060_106013


namespace NUMINAMATH_GPT_gran_age_indeterminate_l1060_106090

theorem gran_age_indeterminate
(gran_age : ℤ) -- Let Gran's age be denoted by gran_age
(guess1 : ℤ := 75) -- The first grandchild guessed 75
(guess2 : ℤ := 78) -- The second grandchild guessed 78
(guess3 : ℤ := 81) -- The third grandchild guessed 81
-- One guess is mistaken by 1 year
(h1 : (abs (gran_age - guess1) = 1) ∨ (abs (gran_age - guess2) = 1) ∨ (abs (gran_age - guess3) = 1))
-- Another guess is mistaken by 2 years
(h2 : (abs (gran_age - guess1) = 2) ∨ (abs (gran_age - guess2) = 2) ∨ (abs (gran_age - guess3) = 2))
-- Another guess is mistaken by 4 years
(h3 : (abs (gran_age - guess1) = 4) ∨ (abs (gran_age - guess2) = 4) ∨ (abs (gran_age - guess3) = 4)) :
  False := sorry

end NUMINAMATH_GPT_gran_age_indeterminate_l1060_106090


namespace NUMINAMATH_GPT_julian_owes_jenny_l1060_106093

-- Define the initial debt and the additional borrowed amount
def initial_debt : ℕ := 20
def additional_borrowed : ℕ := 8

-- Define the total debt
def total_debt : ℕ := initial_debt + additional_borrowed

-- Statement of the problem: Prove that total_debt equals 28
theorem julian_owes_jenny : total_debt = 28 :=
by
  sorry

end NUMINAMATH_GPT_julian_owes_jenny_l1060_106093


namespace NUMINAMATH_GPT_remainder_when_dividing_by_y_minus_4_l1060_106074

def g (y : ℤ) : ℤ := y^5 - 8 * y^4 + 12 * y^3 + 25 * y^2 - 40 * y + 24

theorem remainder_when_dividing_by_y_minus_4 : g 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_by_y_minus_4_l1060_106074


namespace NUMINAMATH_GPT_hypotenuse_of_right_triangle_l1060_106097

theorem hypotenuse_of_right_triangle (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_of_right_triangle_l1060_106097


namespace NUMINAMATH_GPT_non_deg_ellipse_condition_l1060_106080

theorem non_deg_ellipse_condition (k : ℝ) : k > -19 ↔ 
  (∃ x y : ℝ, 3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k) :=
sorry

end NUMINAMATH_GPT_non_deg_ellipse_condition_l1060_106080


namespace NUMINAMATH_GPT_arithmetic_sequence_m_value_l1060_106034

theorem arithmetic_sequence_m_value 
  (a : ℕ → ℝ) (d : ℝ) (h₁ : d ≠ 0) 
  (h₂ : a 3 + a 6 + a 10 + a 13 = 32) 
  (m : ℕ) (h₃ : a m = 8) : 
  m = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_m_value_l1060_106034


namespace NUMINAMATH_GPT_trigonometric_identity_l1060_106073

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1060_106073


namespace NUMINAMATH_GPT_ShepherdProblem_l1060_106018

theorem ShepherdProblem (x y : ℕ) :
  (x + 9 = 2 * (y - 9) ∧ y + 9 = x - 9) ↔
  ((x + 9 = 2 * (y - 9)) ∧ (y + 9 = x - 9)) :=
by
  sorry

end NUMINAMATH_GPT_ShepherdProblem_l1060_106018
