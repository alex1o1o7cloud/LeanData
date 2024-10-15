import Mathlib

namespace NUMINAMATH_GPT_count_positive_n_l2062_206296

def is_factorable (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = -2) ∧ (a * b = - (n:ℤ))

theorem count_positive_n : 
  (∃ (S : Finset ℕ), S.card = 45 ∧ ∀ n ∈ S, (1 ≤ n ∧ n ≤ 2000) ∧ is_factorable n) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_count_positive_n_l2062_206296


namespace NUMINAMATH_GPT_popsicles_eaten_l2062_206246

theorem popsicles_eaten (total_minutes : ℕ) (minutes_per_popsicle : ℕ) (h : total_minutes = 405) (k : minutes_per_popsicle = 12) :
  (total_minutes / minutes_per_popsicle) = 33 :=
by
  sorry

end NUMINAMATH_GPT_popsicles_eaten_l2062_206246


namespace NUMINAMATH_GPT_arithmetic_sequence_term_12_l2062_206241

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_term_12 (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 :=
by
  -- The following line ensures the theorem compiles correctly.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_12_l2062_206241


namespace NUMINAMATH_GPT_contractor_total_amount_l2062_206279

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end NUMINAMATH_GPT_contractor_total_amount_l2062_206279


namespace NUMINAMATH_GPT_candies_for_50_rubles_l2062_206228

theorem candies_for_50_rubles : 
  ∀ (x : ℕ), (45 * x = 45) → (50 / x = 50) := 
by
  intros x h
  sorry

end NUMINAMATH_GPT_candies_for_50_rubles_l2062_206228


namespace NUMINAMATH_GPT_total_reams_of_paper_l2062_206285

def reams_for_haley : ℕ := 2
def reams_for_sister : ℕ := 3

theorem total_reams_of_paper : reams_for_haley + reams_for_sister = 5 := by
  sorry

end NUMINAMATH_GPT_total_reams_of_paper_l2062_206285


namespace NUMINAMATH_GPT_tangent_circles_locus_l2062_206242

theorem tangent_circles_locus :
  ∃ (a b : ℝ), ∀ (C1_center : ℝ × ℝ) (C2_center : ℝ × ℝ) (C1_radius : ℝ) (C2_radius : ℝ),
    C1_center = (0, 0) ∧ C2_center = (2, 0) ∧ C1_radius = 1 ∧ C2_radius = 3 ∧
    (∀ (r : ℝ), (a - 0)^2 + (b - 0)^2 = (r + C1_radius)^2 ∧ (a - 2)^2 + (b - 0)^2 = (C2_radius - r)^2) →
    84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 := sorry

end NUMINAMATH_GPT_tangent_circles_locus_l2062_206242


namespace NUMINAMATH_GPT_combination_equality_l2062_206232

theorem combination_equality : 
  Nat.choose 5 2 + Nat.choose 5 3 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_combination_equality_l2062_206232


namespace NUMINAMATH_GPT_overall_gain_percent_l2062_206298

theorem overall_gain_percent {initial_cost first_repair second_repair third_repair sell_price : ℝ} 
  (h1 : initial_cost = 800) 
  (h2 : first_repair = 150) 
  (h3 : second_repair = 75) 
  (h4 : third_repair = 225) 
  (h5 : sell_price = 1600) :
  (sell_price - (initial_cost + first_repair + second_repair + third_repair)) / 
  (initial_cost + first_repair + second_repair + third_repair) * 100 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_overall_gain_percent_l2062_206298


namespace NUMINAMATH_GPT_inequality_solution_set_min_value_of_x_plus_y_l2062_206201

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem inequality_solution_set (a : ℝ) :
  (if a < 0 then (∀ x : ℝ, f a x > 0 ↔ (1/a < x ∧ x < 2))
   else if a = 0 then (∀ x : ℝ, f a x > 0 ↔ x < 2)
   else if 0 < a ∧ a < 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 2 ∨ 1/a < x))
   else if a = 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x ≠ 2))
   else if a > 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 1/a ∨ x > 2))
   else false) := 
sorry

theorem min_value_of_x_plus_y (a : ℝ) (h : 0 < a) (x y : ℝ) (hx : y ≥ f a (|x|)) :
  x + y ≥ -a - (1/a) := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_min_value_of_x_plus_y_l2062_206201


namespace NUMINAMATH_GPT_slower_plane_speed_l2062_206262

-- Let's define the initial conditions and state the theorem in Lean 4
theorem slower_plane_speed 
    (x : ℕ) -- speed of the slower plane
    (h1 : x + 2*x = 900) : -- based on the total distance after 3 hours
    x = 300 :=
by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_slower_plane_speed_l2062_206262


namespace NUMINAMATH_GPT_quadratic_average_of_roots_l2062_206282

theorem quadratic_average_of_roots (a b c : ℝ) (h_eq : a ≠ 0) (h_b : b = -6) (h_c : c = 3) 
  (discriminant : (b^2 - 4 * a * c) = 12) : 
  (b^2 - 4 * a * c = 12) → ((-b / (2 * a)) / 2 = 1.5) :=
by
  have a_val : a = 2 := sorry
  sorry

end NUMINAMATH_GPT_quadratic_average_of_roots_l2062_206282


namespace NUMINAMATH_GPT_xyz_sum_sqrt14_l2062_206220

theorem xyz_sum_sqrt14 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 1) (h2 : x + 2 * y + 3 * z = Real.sqrt 14) :
  x + y + z = (3 * Real.sqrt 14) / 7 :=
sorry

end NUMINAMATH_GPT_xyz_sum_sqrt14_l2062_206220


namespace NUMINAMATH_GPT_power_function_general_form_l2062_206229

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_general_form (α : ℝ) :
  ∃ y : ℝ, ∃ α : ℝ, f 3 α = y ∧ ∀ x : ℝ, f x α = x ^ α :=
by
  sorry

end NUMINAMATH_GPT_power_function_general_form_l2062_206229


namespace NUMINAMATH_GPT_pq_combined_work_rate_10_days_l2062_206259

/-- Conditions: 
1. wr_p = wr_qr, where wr_qr is the combined work rate of q and r
2. wr_r allows completing the work in 30 days
3. wr_q allows completing the work in 30 days

We need to prove that the combined work rate of p and q allows them to complete the work in 10 days.
-/
theorem pq_combined_work_rate_10_days
  (wr_p wr_q wr_r wr_qr : ℝ)
  (h1 : wr_p = wr_qr)
  (h2 : wr_r = 1/30)
  (h3 : wr_q = 1/30) :
  wr_p + wr_q = 1/10 := by
  sorry

end NUMINAMATH_GPT_pq_combined_work_rate_10_days_l2062_206259


namespace NUMINAMATH_GPT_solution_set_l2062_206238

variable (f : ℝ → ℝ)

def cond1 := ∀ x, f x = f (-x)
def cond2 := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def cond3 := f (1/3) = 0

theorem solution_set (hf1 : cond1 f) (hf2 : cond2 f) (hf3 : cond3 f) :
  { x : ℝ | f (Real.log x / Real.log (1/8)) > 0 } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | 2 < x } :=
sorry

end NUMINAMATH_GPT_solution_set_l2062_206238


namespace NUMINAMATH_GPT_ted_cookies_eaten_l2062_206245

def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def days_baking : ℕ := 6
def cookies_per_day : ℕ := trays_per_day * cookies_per_tray
def total_cookies_baked : ℕ := days_baking * cookies_per_day
def cookies_eaten_by_frank : ℕ := days_baking
def cookies_before_ted : ℕ := total_cookies_baked - cookies_eaten_by_frank
def cookies_left_after_ted : ℕ := 134

theorem ted_cookies_eaten : cookies_before_ted - cookies_left_after_ted = 4 := by
  sorry

end NUMINAMATH_GPT_ted_cookies_eaten_l2062_206245


namespace NUMINAMATH_GPT_probability_of_red_jelly_bean_l2062_206289

-- Definitions based on conditions
def total_jelly_beans := 7 + 9 + 4 + 10
def red_jelly_beans := 7

-- Statement we want to prove
theorem probability_of_red_jelly_bean : (red_jelly_beans : ℚ) / total_jelly_beans = 7 / 30 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_probability_of_red_jelly_bean_l2062_206289


namespace NUMINAMATH_GPT_logical_impossibility_of_thoughts_l2062_206217

variable (K Q : Prop)

/-- Assume that King and Queen are sane (sane is represented by them not believing they're insane) -/
def sane (p : Prop) : Prop :=
  ¬(p = true)

/-- Define the nested thoughts -/
def KingThinksQueenThinksKingThinksQueenOutOfMind (K Q : Prop) :=
  K ∧ Q ∧ K ∧ Q = ¬sane Q

/-- The main proposition -/
theorem logical_impossibility_of_thoughts (hK : sane K) (hQ : sane Q) : 
  ¬KingThinksQueenThinksKingThinksQueenOutOfMind K Q :=
by sorry

end NUMINAMATH_GPT_logical_impossibility_of_thoughts_l2062_206217


namespace NUMINAMATH_GPT_jill_spent_50_percent_on_clothing_l2062_206236

theorem jill_spent_50_percent_on_clothing (
  T : ℝ) (hT : T ≠ 0)
  (h : 0.05 * T * C + 0.10 * 0.30 * T = 0.055 * T):
  C = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_jill_spent_50_percent_on_clothing_l2062_206236


namespace NUMINAMATH_GPT_smallest_b_for_factorization_l2062_206250

theorem smallest_b_for_factorization : ∃ (p q : ℕ), p * q = 2007 ∧ p + q = 232 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_factorization_l2062_206250


namespace NUMINAMATH_GPT_linear_expressions_constant_multiple_l2062_206257

theorem linear_expressions_constant_multiple 
    (a b c p q r : ℝ)
    (h : (a*x + p)^2 + (b*x + q)^2 = (c*x + r)^2) : 
    a*b ≠ 0 → p*q ≠ 0 → (a / b = p / q) :=
by
  -- Given: (ax + p)^2 + (bx + q)^2 = (cx + r)^2
  -- Prove: a / b = p / q, implying that A(x) and B(x) can be expressed as the constant times C(x)
  sorry

end NUMINAMATH_GPT_linear_expressions_constant_multiple_l2062_206257


namespace NUMINAMATH_GPT_no_upper_bound_l2062_206200

-- Given Conditions
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {M : ℝ}

-- Condition: widths and lengths of plates are 1 and a1, a2, a3, ..., respectively
axiom width_1 : ∀ n, (S n > 0)

-- Condition: a1 ≠ 1
axiom a1_neq_1 : a 1 ≠ 1

-- Condition: plates are similar but not congruent starting from the second
axiom similar_not_congruent : ∀ n > 1, (a (n+1) > a n)

-- Condition: S_n denotes the length covered after placing n plates
axiom Sn_length : ∀ n, S (n+1) = S n + a (n+1)

-- Condition: a_{n+1} = 1 / S_n
axiom an_reciprocal : ∀ n, a (n+1) = 1 / S n

-- The final goal: no such real number exists that S_n does not exceed
theorem no_upper_bound : ∀ M : ℝ, ∃ n : ℕ, S n > M := 
sorry

end NUMINAMATH_GPT_no_upper_bound_l2062_206200


namespace NUMINAMATH_GPT_positive_integers_satisfy_inequality_l2062_206216

theorem positive_integers_satisfy_inequality :
  ∀ (n : ℕ), 2 * n - 5 < 5 - 2 * n ↔ n = 1 ∨ n = 2 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_positive_integers_satisfy_inequality_l2062_206216


namespace NUMINAMATH_GPT_diameter_increase_l2062_206240

theorem diameter_increase (D D' : ℝ) (h : π * (D' / 2) ^ 2 = 2.4336 * π * (D / 2) ^ 2) : D' / D = 1.56 :=
by
  -- Statement only, proof is omitted
  sorry

end NUMINAMATH_GPT_diameter_increase_l2062_206240


namespace NUMINAMATH_GPT_total_fruits_correct_l2062_206206

def total_fruits 
  (Jason_watermelons : Nat) (Jason_pineapples : Nat)
  (Mark_watermelons : Nat) (Mark_pineapples : Nat)
  (Sandy_watermelons : Nat) (Sandy_pineapples : Nat) : Nat :=
  Jason_watermelons + Jason_pineapples +
  Mark_watermelons + Mark_pineapples +
  Sandy_watermelons + Sandy_pineapples

theorem total_fruits_correct :
  total_fruits 37 56 68 27 11 14 = 213 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_correct_l2062_206206


namespace NUMINAMATH_GPT_number_plus_273_l2062_206297

theorem number_plus_273 (x : ℤ) (h : x - 477 = 273) : x + 273 = 1023 := by
  sorry

end NUMINAMATH_GPT_number_plus_273_l2062_206297


namespace NUMINAMATH_GPT_day_of_18th_day_of_month_is_tuesday_l2062_206251

theorem day_of_18th_day_of_month_is_tuesday
  (day_of_24th_is_monday : ℕ → ℕ)
  (mod_seven : ∀ n, n % 7 = n)
  (h24 : day_of_24th_is_monday 24 = 1) : day_of_24th_is_monday 18 = 2 :=
by
  sorry

end NUMINAMATH_GPT_day_of_18th_day_of_month_is_tuesday_l2062_206251


namespace NUMINAMATH_GPT_no_minimum_value_l2062_206202

noncomputable def f (x : ℝ) : ℝ :=
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) *
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem no_minimum_value : ¬ ∃ x, (0 < x ∧ x < 4.5) ∧ (∀ y, (0 < y ∧ y < 4.5) → f x ≤ f y) :=
sorry

end NUMINAMATH_GPT_no_minimum_value_l2062_206202


namespace NUMINAMATH_GPT_range_of_a_l2062_206247

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2062_206247


namespace NUMINAMATH_GPT_find_r_given_conditions_l2062_206268

theorem find_r_given_conditions (p c r : ℝ) (h1 : p * r = 360) (h2 : 6 * c * r = 15) (h3 : r = 4) : r = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_r_given_conditions_l2062_206268


namespace NUMINAMATH_GPT_computer_hardware_contract_prob_l2062_206276

theorem computer_hardware_contract_prob :
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  ∃ P_H : ℝ, P_at_least_one = P_H + P_S - P_H_and_S ∧ P_H = 0.8 :=
by
  -- Let definitions and initial conditions
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  -- Solve for P(H)
  let P_H := 0.8
  -- Show the proof of the calculation
  sorry

end NUMINAMATH_GPT_computer_hardware_contract_prob_l2062_206276


namespace NUMINAMATH_GPT_sisters_work_together_days_l2062_206253

-- Definitions based on conditions
def task_completion_rate_older_sister : ℚ := 1/10
def task_completion_rate_younger_sister : ℚ := 1/20
def work_done_by_older_sister_alone : ℚ := 4 * task_completion_rate_older_sister
def remaining_task_after_older_sister : ℚ := 1 - work_done_by_older_sister_alone
def combined_work_rate : ℚ := task_completion_rate_older_sister + task_completion_rate_younger_sister

-- Statement of the proof problem
theorem sisters_work_together_days : 
  (combined_work_rate * x = remaining_task_after_older_sister) → 
  (x = 4) :=
by
  sorry

end NUMINAMATH_GPT_sisters_work_together_days_l2062_206253


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2062_206274

-- Problem 1: Prove that if the inequality |x-1| - |x-2| < a holds for all x in ℝ, then a > 1.
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a) → a > 1 :=
sorry

-- Problem 2: Prove that if the inequality |x-1| - |x-2| < a has at least one real solution, then a > -1.
theorem problem_2 (a : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| < a) → a > -1 :=
sorry

-- Problem 3: Prove that if the solution set of the inequality |x-1| - |x-2| < a is empty, then a ≤ -1.
theorem problem_3 (a : ℝ) :
  (¬∃ x : ℝ, |x - 1| - |x - 2| < a) → a ≤ -1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2062_206274


namespace NUMINAMATH_GPT_triangle_area_difference_l2062_206291

theorem triangle_area_difference 
  (b h : ℝ)
  (hb : 0 < b)
  (hh : 0 < h)
  (A_base : ℝ) (A_height : ℝ)
  (hA_base: A_base = 1.20 * b)
  (hA_height: A_height = 0.80 * h)
  (A_area: ℝ) (B_area: ℝ)
  (hA_area: A_area = 0.5 * A_base * A_height)
  (hB_area: B_area = 0.5 * b * h) :
  (B_area - A_area) / B_area = 0.04 := 
by sorry

end NUMINAMATH_GPT_triangle_area_difference_l2062_206291


namespace NUMINAMATH_GPT_find_a100_find_a1983_l2062_206258

open Nat

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k, a (a k) = 3 * k

theorem find_a100 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 100 = 181 := 
sorry

theorem find_a1983 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 1983 = 3762 := 
sorry

end NUMINAMATH_GPT_find_a100_find_a1983_l2062_206258


namespace NUMINAMATH_GPT_Julie_monthly_salary_l2062_206252

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end NUMINAMATH_GPT_Julie_monthly_salary_l2062_206252


namespace NUMINAMATH_GPT_find_real_solutions_l2062_206266

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_find_real_solutions_l2062_206266


namespace NUMINAMATH_GPT_quadratic_positive_range_l2062_206230

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) ↔ ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_positive_range_l2062_206230


namespace NUMINAMATH_GPT_smallest_divisible_by_15_18_20_is_180_l2062_206213

theorem smallest_divisible_by_15_18_20_is_180 :
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (20 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (20 ∣ m)) → n ≤ m ∧ n = 180 := by
  sorry

end NUMINAMATH_GPT_smallest_divisible_by_15_18_20_is_180_l2062_206213


namespace NUMINAMATH_GPT_absolute_value_neg_2022_l2062_206243

theorem absolute_value_neg_2022 : abs (-2022) = 2022 :=
by sorry

end NUMINAMATH_GPT_absolute_value_neg_2022_l2062_206243


namespace NUMINAMATH_GPT_passenger_waiting_time_probability_l2062_206249

def bus_arrival_interval : ℕ := 5

def waiting_time_limit : ℕ := 3

/-- 
  Prove that for a bus arriving every 5 minutes,
  the probability that a passenger's waiting time 
  is no more than 3 minutes, given the passenger 
  arrives at a random time, is 3/5. 
--/
theorem passenger_waiting_time_probability 
  (bus_interval : ℕ) (time_limit : ℕ) 
  (random_arrival : ℝ) :
  bus_interval = 5 →
  time_limit = 3 →
  0 ≤ random_arrival ∧ random_arrival < bus_interval →
  (random_arrival ≤ time_limit) →
  (random_arrival / ↑bus_interval) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_passenger_waiting_time_probability_l2062_206249


namespace NUMINAMATH_GPT_adam_spent_on_ferris_wheel_l2062_206293

-- Define the conditions
def ticketsBought : Nat := 13
def ticketsLeft : Nat := 4
def costPerTicket : Nat := 9

-- Define the question and correct answer as a proof goal
theorem adam_spent_on_ferris_wheel : (ticketsBought - ticketsLeft) * costPerTicket = 81 := by
  sorry

end NUMINAMATH_GPT_adam_spent_on_ferris_wheel_l2062_206293


namespace NUMINAMATH_GPT_find_n_find_m_constant_term_find_m_max_coefficients_l2062_206271

-- 1. Prove that if the sum of the binomial coefficients is 256, then n = 8.
theorem find_n (n : ℕ) (h : 2^n = 256) : n = 8 :=
by sorry

-- 2. Prove that if the constant term is 35/8, then m = ±1/2.
theorem find_m_constant_term (m : ℚ) (h : m^4 * (Nat.choose 8 4) = 35/8) : m = 1/2 ∨ m = -1/2 :=
by sorry

-- 3. Prove that if only the 6th and 7th terms have the maximum coefficients, then m = 2.
theorem find_m_max_coefficients (m : ℚ) (h1 : m ≠ 0) (h2 : m^5 * (Nat.choose 8 5) = m^6 * (Nat.choose 8 6)) : m = 2 :=
by sorry

end NUMINAMATH_GPT_find_n_find_m_constant_term_find_m_max_coefficients_l2062_206271


namespace NUMINAMATH_GPT_ratio_of_kids_to_adult_meals_l2062_206287

theorem ratio_of_kids_to_adult_meals (k a : ℕ) (h1 : k = 8) (h2 : k + a = 12) : k / a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_kids_to_adult_meals_l2062_206287


namespace NUMINAMATH_GPT_equal_sums_arithmetic_sequences_l2062_206264

-- Define the arithmetic sequences and their sums
def s₁ (n : ℕ) : ℕ := n * (5 * n + 13) / 2
def s₂ (n : ℕ) : ℕ := n * (3 * n + 37) / 2

-- State the theorem: for given n != 0, prove s₁ n = s₂ n implies n = 12
theorem equal_sums_arithmetic_sequences (n : ℕ) (h : n ≠ 0) : 
  s₁ n = s₂ n → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_equal_sums_arithmetic_sequences_l2062_206264


namespace NUMINAMATH_GPT_men_in_first_scenario_l2062_206280

theorem men_in_first_scenario 
  (M : ℕ) 
  (daily_hours_first weekly_earning_first daily_hours_second weekly_earning_second : ℝ) 
  (number_of_men_second : ℕ)
  (days_per_week : ℕ := 7) 
  (h1 : M * daily_hours_first * days_per_week = weekly_earning_first)
  (h2 : number_of_men_second * daily_hours_second * days_per_week = weekly_earning_second) 
  (h1_value : daily_hours_first = 10) 
  (w1_value : weekly_earning_first = 1400) 
  (h2_value : daily_hours_second = 6) 
  (w2_value : weekly_earning_second = 1890)
  (second_scenario_men : number_of_men_second = 9) : 
  M = 4 :=
by
  sorry

end NUMINAMATH_GPT_men_in_first_scenario_l2062_206280


namespace NUMINAMATH_GPT_average_selling_price_is_86_l2062_206210

def selling_prices := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

def average (prices : List Nat) : Nat :=
  (prices.sum) / prices.length

theorem average_selling_price_is_86 :
  average selling_prices = 86 :=
by
  sorry

end NUMINAMATH_GPT_average_selling_price_is_86_l2062_206210


namespace NUMINAMATH_GPT_earnings_difference_l2062_206288

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_earnings_difference_l2062_206288


namespace NUMINAMATH_GPT_meter_to_skips_l2062_206221

/-!
# Math Proof Problem
Suppose hops, skips and jumps are specific units of length. Given the following conditions:
1. \( b \) hops equals \( c \) skips.
2. \( d \) jumps equals \( e \) hops.
3. \( f \) jumps equals \( g \) meters.

Prove that one meter equals \( \frac{cef}{bdg} \) skips.
-/

theorem meter_to_skips (b c d e f g : ℝ) (h1 : b ≠ 0) (h2 : c ≠ 0) (h3 : d ≠ 0) (h4 : e ≠ 0) (h5 : f ≠ 0) (h6 : g ≠ 0) :
  (1 : ℝ) = (cef) / (bdg) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_meter_to_skips_l2062_206221


namespace NUMINAMATH_GPT_derivative_of_f_l2062_206215

noncomputable def f (x : ℝ) : ℝ :=
  (Nat.choose 4 0 : ℝ) - (Nat.choose 4 1 : ℝ) * x + (Nat.choose 4 2 : ℝ) * x^2 - (Nat.choose 4 3 : ℝ) * x^3 + (Nat.choose 4 4 : ℝ) * x^4

theorem derivative_of_f : 
  ∀ (x : ℝ), (deriv f x) = 4 * (-1 + x)^3 :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_f_l2062_206215


namespace NUMINAMATH_GPT_geom_seq_prod_of_terms_l2062_206295

theorem geom_seq_prod_of_terms (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end NUMINAMATH_GPT_geom_seq_prod_of_terms_l2062_206295


namespace NUMINAMATH_GPT_total_hours_worked_l2062_206244

def hours_per_day : ℕ := 8 -- Frank worked 8 hours on each day
def number_of_days : ℕ := 4 -- First 4 days of the week

theorem total_hours_worked : hours_per_day * number_of_days = 32 := by
  sorry

end NUMINAMATH_GPT_total_hours_worked_l2062_206244


namespace NUMINAMATH_GPT_range_of_t_l2062_206224

def ellipse (x y t : ℝ) : Prop := (x^2) / 4 + (y^2) / t = 1

def distance_greater_than_one (x y t : ℝ) : Prop := 
  let a := if t > 4 then Real.sqrt t else 2
  let b := if t > 4 then 2 else Real.sqrt t
  let c := if t > 4 then Real.sqrt (t - 4) else Real.sqrt (4 - t)
  a - c > 1

theorem range_of_t (t : ℝ) : 
  (∀ x y, ellipse x y t → distance_greater_than_one x y t) ↔ 
  (3 < t ∧ t < 4) ∨ (4 < t ∧ t < 25 / 4) := 
sorry

end NUMINAMATH_GPT_range_of_t_l2062_206224


namespace NUMINAMATH_GPT_president_savings_l2062_206275

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end NUMINAMATH_GPT_president_savings_l2062_206275


namespace NUMINAMATH_GPT_f_36_l2062_206256

variable {R : Type*} [CommRing R]
variable (f : R → R) (p q : R)

-- Conditions
axiom f_mult_add : ∀ x y, f (x * y) = f x + f y
axiom f_2 : f 2 = p
axiom f_3 : f 3 = q

-- Statement to prove
theorem f_36 : f 36 = 2 * (p + q) :=
by
  sorry

end NUMINAMATH_GPT_f_36_l2062_206256


namespace NUMINAMATH_GPT_find_A_l2062_206233

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 10 * A + 3 + 610 + B = 695) : A = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_A_l2062_206233


namespace NUMINAMATH_GPT_seokgi_money_l2062_206223

open Classical

variable (S Y : ℕ)

theorem seokgi_money (h1 : ∃ S, S + 2000 < S + Y + 2000)
                     (h2 : ∃ Y, Y + 1500 < S + Y + 1500)
                     (h3 : 3500 + (S + Y + 2000) = (S + Y) + 3500)
                     (boat_price1: ∀ S, S + 2000 = S + 2000)
                     (boat_price2: ∀ Y, Y + 1500 = Y + 1500) :
  S = 5000 :=
by sorry

end NUMINAMATH_GPT_seokgi_money_l2062_206223


namespace NUMINAMATH_GPT_fraction_e_over_d_l2062_206248

theorem fraction_e_over_d :
  ∃ (d e : ℝ), (∀ (x : ℝ), x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ e / d = -1298 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_e_over_d_l2062_206248


namespace NUMINAMATH_GPT_Jasmine_shopping_time_l2062_206209

-- Define the variables for the times in minutes
def T_start := 960  -- 4:00 pm in minutes (4*60)
def T_commute := 30
def T_dryClean := 10
def T_dog := 20
def T_cooking := 90
def T_dinner := 1140  -- 7:00 pm in minutes (19*60)

-- The calculated start time for cooking in minutes
def T_startCooking := T_dinner - T_cooking

-- The time Jasmine has between arriving home and starting cooking
def T_groceryShopping := T_startCooking - (T_start + T_commute + T_dryClean + T_dog)

theorem Jasmine_shopping_time :
  T_groceryShopping = 30 := by
  sorry

end NUMINAMATH_GPT_Jasmine_shopping_time_l2062_206209


namespace NUMINAMATH_GPT_simplify_expression_l2062_206227

theorem simplify_expression :
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2062_206227


namespace NUMINAMATH_GPT_quadrilateral_angles_l2062_206222

theorem quadrilateral_angles 
  (A B C D : Type) 
  (a d b c : Float)
  (hAD : a = d ∧ d = c) 
  (hBDC_twice_BDA : ∃ x : Float, b = 2 * x) 
  (hBDA_CAD_ratio : ∃ x : Float, d = 2/3 * x) :
  (∃ α β γ δ : Float, 
    α = 75 ∧ 
    β = 135 ∧ 
    γ = 60 ∧ 
    δ = 90) := 
sorry

end NUMINAMATH_GPT_quadrilateral_angles_l2062_206222


namespace NUMINAMATH_GPT_magnitude_of_power_l2062_206284

noncomputable def z : ℂ := 4 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_power :
  Complex.abs (z ^ 4) = 576 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_power_l2062_206284


namespace NUMINAMATH_GPT_quadratic_vertex_coordinates_l2062_206234

theorem quadratic_vertex_coordinates : ∀ x : ℝ,
  (∃ y : ℝ, y = 2 * x^2 - 4 * x + 5) →
  (1, 3) = (1, 3) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_vertex_coordinates_l2062_206234


namespace NUMINAMATH_GPT_prime_p_geq_5_div_24_l2062_206208

theorem prime_p_geq_5_div_24 (p : ℕ) (hp : Nat.Prime p) (hp_geq_5 : p ≥ 5) : 24 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_GPT_prime_p_geq_5_div_24_l2062_206208


namespace NUMINAMATH_GPT_fill_pool_with_B_only_l2062_206203

theorem fill_pool_with_B_only
    (time_AB : ℝ)
    (R_AB : time_AB = 30)
    (time_A_B_then_B : ℝ)
    (R_A_B_then_B : (10 / 30 + (time_A_B_then_B - 10) / time_A_B_then_B) = 1)
    (only_B_time : ℝ)
    (R_B : only_B_time = 60) :
    only_B_time = 60 :=
by
    sorry

end NUMINAMATH_GPT_fill_pool_with_B_only_l2062_206203


namespace NUMINAMATH_GPT_profit_increase_l2062_206231

theorem profit_increase (x y : ℝ) (a : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (profit_eq : y - x = x * (a / 100))
  (new_profit_eq : y - 0.95 * x = 0.95 * x * (a / 100) + 0.95 * x * (15 / 100)) :
  a = 185 :=
by
  sorry

end NUMINAMATH_GPT_profit_increase_l2062_206231


namespace NUMINAMATH_GPT_triangle_area_example_l2062_206218

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_example :
  triangle_area (-2, 3) (7, -1) (4, 6) = 25.5 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_triangle_area_example_l2062_206218


namespace NUMINAMATH_GPT_geometric_series_sum_l2062_206292

theorem geometric_series_sum (n : ℕ) : 
  let a₁ := 2
  let q := 2
  let S_n := a₁ * (1 - q^n) / (1 - q)
  S_n = 2 - 2^(n + 1) := 
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2062_206292


namespace NUMINAMATH_GPT_probability_sum_equals_6_l2062_206207

theorem probability_sum_equals_6 : 
  let possible_outcomes := 36
  let favorable_outcomes := 5
  (favorable_outcomes / possible_outcomes : ℚ) = 5 / 36 := 
by 
  sorry

end NUMINAMATH_GPT_probability_sum_equals_6_l2062_206207


namespace NUMINAMATH_GPT_total_concrete_weight_l2062_206283

theorem total_concrete_weight (w1 w2 : ℝ) (c1 c2 : ℝ) (total_weight : ℝ)
  (h1 : w1 = 1125)
  (h2 : w2 = 1125)
  (h3 : c1 = 0.093)
  (h4 : c2 = 0.113)
  (h5 : (w1 * c1 + w2 * c2) / (w1 + w2) = 0.108) :
  total_weight = w1 + w2 :=
by
  sorry

end NUMINAMATH_GPT_total_concrete_weight_l2062_206283


namespace NUMINAMATH_GPT_minimum_black_edges_5x5_l2062_206255

noncomputable def minimum_black_edges_on_border (n : ℕ) : ℕ :=
if n = 5 then 5 else 0

theorem minimum_black_edges_5x5 : 
  minimum_black_edges_on_border 5 = 5 :=
by sorry

end NUMINAMATH_GPT_minimum_black_edges_5x5_l2062_206255


namespace NUMINAMATH_GPT_third_number_is_42_l2062_206272

variable (x : ℕ)

def number1 : ℕ := 5 * x
def number2 : ℕ := 6 * x
def number3 : ℕ := 8 * x

theorem third_number_is_42 (h : number1 x + number3 x = number2 x + 49) : number2 x = 42 :=
by
  sorry

end NUMINAMATH_GPT_third_number_is_42_l2062_206272


namespace NUMINAMATH_GPT_scientific_notation_3080000_l2062_206260

theorem scientific_notation_3080000 : (3080000 : ℝ) = 3.08 * 10^6 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_3080000_l2062_206260


namespace NUMINAMATH_GPT_maries_trip_distance_l2062_206235

theorem maries_trip_distance (x : ℚ)
  (h1 : x = x / 4 + 15 + x / 6) :
  x = 180 / 7 :=
by
  sorry

end NUMINAMATH_GPT_maries_trip_distance_l2062_206235


namespace NUMINAMATH_GPT_three_digit_odds_factors_count_l2062_206211

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end NUMINAMATH_GPT_three_digit_odds_factors_count_l2062_206211


namespace NUMINAMATH_GPT_find_value_of_a_l2062_206205

noncomputable def value_of_a (a : ℝ) (hyp_asymptotes_tangent_circle : Prop) : Prop :=
  a = (Real.sqrt 3) / 3 → hyp_asymptotes_tangent_circle

theorem find_value_of_a (a : ℝ) (condition1 : 0 < a)
  (condition_hyperbola : ∀ x y, x^2 / a^2 - y^2 = 1)
  (condition_circle : ∀ x y, x^2 + y^2 - 4*y + 3 = 0)
  (hyp_asymptotes_tangent_circle : Prop) :
  value_of_a a hyp_asymptotes_tangent_circle := 
sorry

end NUMINAMATH_GPT_find_value_of_a_l2062_206205


namespace NUMINAMATH_GPT_real_no_impure_l2062_206273

theorem real_no_impure {x : ℝ} (h1 : x^2 - 1 = 0) (h2 : x^2 + 3 * x + 2 ≠ 0) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_real_no_impure_l2062_206273


namespace NUMINAMATH_GPT_smallest_x_y_sum_l2062_206270

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≠ y) (h_fraction : 1/x + 1/y = 1/20) : x + y = 90 :=
sorry

end NUMINAMATH_GPT_smallest_x_y_sum_l2062_206270


namespace NUMINAMATH_GPT_cristina_pace_correct_l2062_206261

-- Definitions of the conditions
def head_start : ℕ := 30
def nicky_pace : ℕ := 3  -- meters per second
def time_for_catch_up : ℕ := 15  -- seconds

-- Distance covers by Nicky
def nicky_distance : ℕ := nicky_pace * time_for_catch_up

-- Total distance covered by Cristina to catch up Nicky
def cristina_distance : ℕ := nicky_distance + head_start

-- Cristina's pace
def cristina_pace : ℕ := cristina_distance / time_for_catch_up

-- Theorem statement
theorem cristina_pace_correct : cristina_pace = 5 := by 
  sorry

end NUMINAMATH_GPT_cristina_pace_correct_l2062_206261


namespace NUMINAMATH_GPT_books_arrangement_l2062_206225

-- All conditions provided in Lean as necessary definitions
def num_arrangements (math_books english_books science_books : ℕ) : ℕ :=
  if math_books = 4 ∧ english_books = 6 ∧ science_books = 2 then
    let arrangements_groups := 2 * 3  -- Number of valid group placements
    let arrangements_math := Nat.factorial math_books
    let arrangements_english := Nat.factorial english_books
    let arrangements_science := Nat.factorial science_books
    arrangements_groups * arrangements_math * arrangements_english * arrangements_science
  else
    0

theorem books_arrangement : num_arrangements 4 6 2 = 207360 :=
by
  sorry

end NUMINAMATH_GPT_books_arrangement_l2062_206225


namespace NUMINAMATH_GPT_even_function_maximum_l2062_206263

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def has_maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ ∀ y : ℝ, a ≤ y ∧ y ≤ b → f y ≤ f x

theorem even_function_maximum 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_max_1_7 : has_maximum_on_interval f 1 7) :
  has_maximum_on_interval f (-7) (-1) :=
sorry

end NUMINAMATH_GPT_even_function_maximum_l2062_206263


namespace NUMINAMATH_GPT_train_crossing_time_l2062_206265

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end NUMINAMATH_GPT_train_crossing_time_l2062_206265


namespace NUMINAMATH_GPT_add_decimal_l2062_206212

theorem add_decimal (a b : ℝ) (h1 : a = 0.35) (h2 : b = 124.75) : a + b = 125.10 :=
by sorry

end NUMINAMATH_GPT_add_decimal_l2062_206212


namespace NUMINAMATH_GPT_integer_values_of_a_l2062_206281

theorem integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^4 + 4 * x^3 + a * x^2 + 8 = 0) ↔ (a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_integer_values_of_a_l2062_206281


namespace NUMINAMATH_GPT_ticket_price_increase_l2062_206278

noncomputable def y (x : ℕ) : ℝ :=
  if x ≤ 100 then
    30 * x - 50 * Real.sqrt x - 500
  else
    30 * x - 50 * Real.sqrt x - 700

theorem ticket_price_increase (m : ℝ) : 
  m * 20 - 50 * Real.sqrt 20 - 500 ≥ 0 → m ≥ 37 := sorry

end NUMINAMATH_GPT_ticket_price_increase_l2062_206278


namespace NUMINAMATH_GPT_min_value_fraction_sum_l2062_206219

theorem min_value_fraction_sum (p q r a b : ℝ) (hpq : 0 < p) (hq : p < q) (hr : q < r)
  (h_sum : p + q + r = a) (h_prod_sum : p * q + q * r + r * p = b) (h_prod : p * q * r = 48) :
  ∃ (min_val : ℝ), min_val = (1 / p) + (2 / q) + (3 / r) ∧ min_val = 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l2062_206219


namespace NUMINAMATH_GPT_Jakes_weight_is_198_l2062_206267

variable (Jake Kendra : ℕ)

-- Conditions
variable (h1 : Jake - 8 = 2 * Kendra)
variable (h2 : Jake + Kendra = 293)

theorem Jakes_weight_is_198 : Jake = 198 :=
by
  sorry

end NUMINAMATH_GPT_Jakes_weight_is_198_l2062_206267


namespace NUMINAMATH_GPT_cost_of_apples_is_2_l2062_206254

variable (A : ℝ)

def cost_of_apples (A : ℝ) : ℝ := 5 * A
def cost_of_sugar (A : ℝ) : ℝ := 3 * (A - 1)
def cost_of_walnuts : ℝ := 0.5 * 6
def total_cost (A : ℝ) : ℝ := cost_of_apples A + cost_of_sugar A + cost_of_walnuts

theorem cost_of_apples_is_2 (A : ℝ) (h : total_cost A = 16) : A = 2 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_apples_is_2_l2062_206254


namespace NUMINAMATH_GPT_geom_prog_roots_a_eq_22_l2062_206226

theorem geom_prog_roots_a_eq_22 (x1 x2 x3 a : ℝ) :
  (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → 
  (∃ b q, (x1 = b ∧ x2 = b * q ∧ x3 = b * q^2) ∧ (x1 + x2 + x3 = 11) ∧ (x1 * x2 * x3 = 8) ∧ (x1*x2 + x2*x3 + x3*x1 = a)) → 
  a = 22 :=
sorry

end NUMINAMATH_GPT_geom_prog_roots_a_eq_22_l2062_206226


namespace NUMINAMATH_GPT_problem_nine_chapters_l2062_206237

theorem problem_nine_chapters (x y : ℝ) :
  (x + (1 / 2) * y = 50) →
  (y + (2 / 3) * x = 50) →
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_problem_nine_chapters_l2062_206237


namespace NUMINAMATH_GPT_num_distinct_triangles_in_octahedron_l2062_206214

theorem num_distinct_triangles_in_octahedron : ∃ n : ℕ, n = 48 ∧ ∀ (V : Finset (Fin 8)), 
  V.card = 3 → (∀ {a b c : Fin 8}, a ∈ V ∧ b ∈ V ∧ c ∈ V → 
  ¬((a = 0 ∧ b = 1 ∧ c = 2) ∨ (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 6 ∧ b = 7 ∧ c = 8)
  ∨ (a = 7 ∧ b = 0 ∧ c = 1) ∨ (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 6 ∧ c = 7))) :=
by sorry

end NUMINAMATH_GPT_num_distinct_triangles_in_octahedron_l2062_206214


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2062_206277

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2024 * x ↔ x = 0 ∨ x = 2024 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2062_206277


namespace NUMINAMATH_GPT_pipes_fill_cistern_in_12_minutes_l2062_206290

noncomputable def time_to_fill_cistern_with_pipes (A_fill : ℝ) (B_fill : ℝ) (C_empty : ℝ) : ℝ :=
  let A_rate := 1 / (12 * 3)          -- Pipe A's rate
  let B_rate := 1 / (8 * 3)           -- Pipe B's rate
  let C_rate := -1 / 24               -- Pipe C's rate
  let combined_rate := A_rate + B_rate - C_rate
  (1 / 3) / combined_rate             -- Time to fill remaining one-third

theorem pipes_fill_cistern_in_12_minutes :
  time_to_fill_cistern_with_pipes 12 8 24 = 12 :=
by
  sorry

end NUMINAMATH_GPT_pipes_fill_cistern_in_12_minutes_l2062_206290


namespace NUMINAMATH_GPT_peculiar_looking_less_than_500_l2062_206269

def is_composite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def peculiar_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 7 = 0 ∨ n % 11 = 0)

theorem peculiar_looking_less_than_500 :
  ∃ n, n = 33 ∧ ∀ k, k < 500 → peculiar_looking k → k = n :=
sorry

end NUMINAMATH_GPT_peculiar_looking_less_than_500_l2062_206269


namespace NUMINAMATH_GPT_sandy_age_l2062_206286

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 :=
sorry

end NUMINAMATH_GPT_sandy_age_l2062_206286


namespace NUMINAMATH_GPT_negation_correct_l2062_206299

-- Define the initial proposition
def initial_proposition : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 - 2 * x > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 - 2 * x ≤ 0

-- Statement of the theorem
theorem negation_correct :
  (¬ initial_proposition) = negated_proposition :=
by
  sorry

end NUMINAMATH_GPT_negation_correct_l2062_206299


namespace NUMINAMATH_GPT_biff_break_even_hours_l2062_206204

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end NUMINAMATH_GPT_biff_break_even_hours_l2062_206204


namespace NUMINAMATH_GPT_S_5_value_l2062_206239

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a2a4 (h : geometric_sequence a) : a 1 * a 3 = 16
axiom S3 : S 3 = 7

theorem S_5_value 
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = a 0 * (1 - (a 1)^(n)) / (1 - a 1)) :
  S 5 = 31 :=
sorry

end NUMINAMATH_GPT_S_5_value_l2062_206239


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l2062_206294

theorem largest_divisor_of_expression 
  (x : ℤ) (h_odd : x % 2 = 1) :
  384 ∣ (8*x + 4) * (8*x + 8) * (4*x + 2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l2062_206294
