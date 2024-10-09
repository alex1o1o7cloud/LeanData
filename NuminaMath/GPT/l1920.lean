import Mathlib

namespace local_minimum_value_of_f_l1920_192086

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem local_minimum_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x) ∧ f x = 1 :=
by
  sorry

end local_minimum_value_of_f_l1920_192086


namespace winner_lifted_weight_l1920_192010

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l1920_192010


namespace min_value_fraction_l1920_192063

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) : (x + y) / x = 0.8 :=
by
  sorry

end min_value_fraction_l1920_192063


namespace add_fractions_l1920_192023

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l1920_192023


namespace quadratic_inequality_l1920_192097

theorem quadratic_inequality (a : ℝ) :
  (¬ (∃ x : ℝ, a * x^2 + 2 * x + 3 ≤ 0)) ↔ (a > 1 / 3) :=
by 
  sorry

end quadratic_inequality_l1920_192097


namespace solve_for_x_l1920_192087

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2 * x - 25) : x = -20 :=
by
  sorry

end solve_for_x_l1920_192087


namespace tom_pie_share_l1920_192048

theorem tom_pie_share :
  (∃ (x : ℚ), 4 * x = (5 / 8) ∧ x = 5 / 32) :=
by
  sorry

end tom_pie_share_l1920_192048


namespace cistern_fill_time_l1920_192019

theorem cistern_fill_time (fillA emptyB : ℕ) (hA : fillA = 8) (hB : emptyB = 12) : (24 : ℕ) = 24 :=
by
  sorry

end cistern_fill_time_l1920_192019


namespace find_a5_l1920_192032

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = 2 * n * (n + 1))
  (ha : ∀ n ≥ 2, a n = S n - S (n - 1)) : 
  a 5 = 20 := 
sorry

end find_a5_l1920_192032


namespace factory_production_schedule_l1920_192080

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l1920_192080


namespace calculate_revolutions_l1920_192089

def wheel_diameter : ℝ := 8
def distance_traveled_miles : ℝ := 0.5
def feet_per_mile : ℝ := 5280
def distance_traveled_feet : ℝ := distance_traveled_miles * feet_per_mile

theorem calculate_revolutions :
  let radius : ℝ := wheel_diameter / 2
  let circumference : ℝ := 2 * Real.pi * radius
  let revolutions : ℝ := distance_traveled_feet / circumference
  revolutions = 330 / Real.pi := by
  sorry

end calculate_revolutions_l1920_192089


namespace work_done_by_gas_l1920_192030

def gas_constant : ℝ := 8.31 -- J/(mol·K)
def temperature_change : ℝ := 100 -- K (since 100°C increase is equivalent to 100 K in Kelvin)
def moles_of_gas : ℝ := 1 -- one mole of gas

theorem work_done_by_gas :
  (1/2) * gas_constant * temperature_change = 415.5 :=
by sorry

end work_done_by_gas_l1920_192030


namespace earphone_cost_l1920_192067

/-- 
The cost of the earphone purchased on Friday can be calculated given:
1. The mean expenditure over 7 days is 500.
2. The expenditures for Monday, Tuesday, Wednesday, Thursday, Saturday, and Sunday are 450, 600, 400, 500, 550, and 300, respectively.
3. On Friday, the expenditures include a pen costing 30 and a notebook costing 50.
-/
theorem earphone_cost
  (mean_expenditure : ℕ)
  (mon tue wed thur sat sun : ℕ)
  (pen_cost notebook_cost : ℕ)
  (mean_expenditure_eq : mean_expenditure = 500)
  (mon_eq : mon = 450)
  (tue_eq : tue = 600)
  (wed_eq : wed = 400)
  (thur_eq : thur = 500)
  (sat_eq : sat = 550)
  (sun_eq : sun = 300)
  (pen_cost_eq : pen_cost = 30)
  (notebook_cost_eq : notebook_cost = 50)
  : ∃ (earphone_cost : ℕ), earphone_cost = 620 := 
by
  sorry

end earphone_cost_l1920_192067


namespace least_sum_four_primes_gt_10_l1920_192037

theorem least_sum_four_primes_gt_10 : 
  ∃ (p1 p2 p3 p4 : ℕ), 
    p1 > 10 ∧ p2 > 10 ∧ p3 > 10 ∧ p4 > 10 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1 + p2 + p3 + p4 = 60 ∧
    ∀ (q1 q2 q3 q4 : ℕ), 
      q1 > 10 ∧ q2 > 10 ∧ q3 > 10 ∧ q4 > 10 ∧ 
      Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ Nat.Prime q4 ∧
      q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
      q1 + q2 + q3 + q4 ≥ 60 :=
by
  sorry

end least_sum_four_primes_gt_10_l1920_192037


namespace Mike_profit_l1920_192022

def total_cost (acres : ℕ) (cost_per_acre : ℕ) : ℕ :=
  acres * cost_per_acre

def revenue (acres_sold : ℕ) (price_per_acre : ℕ) : ℕ :=
  acres_sold * price_per_acre

def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem Mike_profit :
  let acres := 200
  let cost_per_acre := 70
  let acres_sold := acres / 2
  let price_per_acre := 200
  let cost := total_cost acres cost_per_acre
  let rev := revenue acres_sold price_per_acre
  profit rev cost = 6000 :=
by
  sorry

end Mike_profit_l1920_192022


namespace find_f_7_l1920_192004

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 4) = f x
axiom piecewise_function (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : f x = 2 * x^3

theorem find_f_7 : f 7 = -2 := by
  sorry

end find_f_7_l1920_192004


namespace problem_l1920_192070

noncomputable def key_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : Real.sqrt (x * y) ≤ 1) 
    : Prop := ∃ z : ℝ, 0 < z ∧ z = 2 * (x + y) / (x + y + 2)^2

theorem problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 2) :
    (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16 / 25 := 
sorry

end problem_l1920_192070


namespace probability_sin_in_interval_half_l1920_192065

noncomputable def probability_sin_interval : ℝ :=
  let a := - (Real.pi / 2)
  let b := Real.pi / 2
  let interval_length := b - a
  (b - 0) / interval_length

theorem probability_sin_in_interval_half :
  probability_sin_interval = 1 / 2 := by
  sorry

end probability_sin_in_interval_half_l1920_192065


namespace area_of_triangle_ABC_l1920_192043

theorem area_of_triangle_ABC 
  (BD DC : ℕ) 
  (h_ratio : BD / DC = 4 / 3)
  (S_BEC : ℕ) 
  (h_BEC : S_BEC = 105) :
  ∃ (S_ABC : ℕ), S_ABC = 315 := 
sorry

end area_of_triangle_ABC_l1920_192043


namespace smallest_nonneg_int_mod_15_l1920_192034

theorem smallest_nonneg_int_mod_15 :
  ∃ x : ℕ, x + 7263 ≡ 3507 [MOD 15] ∧ ∀ y : ℕ, y + 7263 ≡ 3507 [MOD 15] → x ≤ y :=
by
  sorry

end smallest_nonneg_int_mod_15_l1920_192034


namespace first_shaded_square_each_column_l1920_192025

/-- A rectangular board with 10 columns, numbered starting from 
    1 to the nth square left-to-right and top-to-bottom. The student shades squares 
    that are perfect squares. Prove that the first shaded square ensuring there's at least 
    one shaded square in each of the 10 columns is 400. -/
theorem first_shaded_square_each_column : 
  (∃ n, (∀ k, 1 ≤ k ∧ k ≤ 10 → ∃ m, m^2 ≡ k [MOD 10] ∧ m^2 ≤ n) ∧ n = 400) :=
sorry

end first_shaded_square_each_column_l1920_192025


namespace count_positive_integers_satisfying_properties_l1920_192024

theorem count_positive_integers_satisfying_properties :
  (∃ n : ℕ, ∀ N < 2007,
    (N % 2 = 1) ∧
    (N % 3 = 2) ∧
    (N % 4 = 3) ∧
    (N % 5 = 4) ∧
    (N % 6 = 5) → n = 33) :=
by
  sorry

end count_positive_integers_satisfying_properties_l1920_192024


namespace probability_blue_or_green_l1920_192036

def faces : Type := {faces : ℕ // faces = 6}
noncomputable def blue_faces : ℕ := 3
noncomputable def red_faces : ℕ := 2
noncomputable def green_faces : ℕ := 1

theorem probability_blue_or_green :
  (blue_faces + green_faces) / 6 = (2 / 3) := by
  sorry

end probability_blue_or_green_l1920_192036


namespace exists_integer_root_l1920_192076

theorem exists_integer_root (a b c d : ℤ) (ha : a ≠ 0)
  (h : ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * (a * x^3 + b * x^2 + c * x + d) = y * (a * y^3 + b * y^2 + c * y + d)) :
  ∃ z : ℤ, a * z^3 + b * z^2 + c * z + d = 0 :=
by
  sorry

end exists_integer_root_l1920_192076


namespace tangent_line_of_f_eq_kx_l1920_192077

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

theorem tangent_line_of_f_eq_kx (k : ℝ) : 
    (∃ x₀, tangent_line k x₀ = f x₀ ∧ deriv f x₀ = k) → 
    (k = 0 ∨ k = 1 ∨ k = -1) := 
  sorry

end tangent_line_of_f_eq_kx_l1920_192077


namespace bigger_part_of_sum_54_l1920_192098

theorem bigger_part_of_sum_54 (x y : ℕ) (h₁ : x + y = 54) (h₂ : 10 * x + 22 * y = 780) : x = 34 :=
sorry

end bigger_part_of_sum_54_l1920_192098


namespace probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l1920_192069

-- Definitions for balls and initial conditions
def totalBalls : ℕ := 10
def redBalls : ℕ := 2
def whiteBalls : ℕ := 3
def yellowBalls : ℕ := 5

-- Drawing without replacement
noncomputable def probability_second_ball_red : ℚ :=
  (2/10) * (1/9) + (8/10) * (2/9)

-- Probabilities for each case
noncomputable def probability_first_prize : ℚ := 
  (redBalls.choose 1 * whiteBalls.choose 1) / (totalBalls.choose 2)

noncomputable def probability_second_prize : ℚ := 
  (redBalls.choose 2) / (totalBalls.choose 2)

noncomputable def probability_third_prize : ℚ := 
  (whiteBalls.choose 2) / (totalBalls.choose 2)

-- Probability of at least one yellow ball (no prize)
noncomputable def probability_no_prize : ℚ := 
  1 - probability_first_prize - probability_second_prize - probability_third_prize

-- Probability distribution and expectation for number of winners X
noncomputable def winning_probability : ℚ := probability_first_prize + probability_second_prize + probability_third_prize

noncomputable def P_X (n : ℕ) : ℚ :=
  if n = 0 then (7/9)^3
  else if n = 1 then 3 * (2/9) * (7/9)^2
  else if n = 2 then 3 * (2/9)^2 * (7/9)
  else if n = 3 then (2/9)^3
  else 0

noncomputable def expectation_X : ℚ := 
  3 * winning_probability

-- Lean statements
theorem probability_of_second_ball_red_is_correct :
  probability_second_ball_red = 1 / 5 := by
  sorry

theorem probabilities_of_winning_prizes :
  probability_first_prize = 2 / 15 ∧
  probability_second_prize = 1 / 45 ∧
  probability_third_prize = 1 / 15 := by
  sorry

theorem distribution_and_expectation_of_X :
  P_X 0 = 343 / 729 ∧
  P_X 1 = 294 / 729 ∧
  P_X 2 = 84 / 729 ∧
  P_X 3 = 8 / 729 ∧
  expectation_X = 2 / 3 := by
  sorry

end probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l1920_192069


namespace M_is_infinite_l1920_192074

variable (M : Set ℝ)

def has_properties (M : Set ℝ) : Prop :=
  (∃ x y : ℝ, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ ∀ x ∈ M, (3*x - 2 ∈ M ∨ -4*x + 5 ∈ M)

theorem M_is_infinite (M : Set ℝ) (h : has_properties M) : ¬Finite M := by
  sorry

end M_is_infinite_l1920_192074


namespace part_I_part_II_l1920_192015

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem part_I (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 0) →
  -1 < a ∧ a ≤ 11/5 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := 
  if abs x ≥ 1 then 2 * x^2 - 2 * a * x + a + 1 
  else -2 * a * x + a + 3

theorem part_II (a : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) →
  1 + Real.sqrt 3 < a ∧ a ≤ 19/5 :=
sorry

end part_I_part_II_l1920_192015


namespace exponent_fraction_simplification_l1920_192014

theorem exponent_fraction_simplification : 
  (2 ^ 2016 + 2 ^ 2014) / (2 ^ 2016 - 2 ^ 2014) = 5 / 3 := 
by {
  -- proof steps would go here
  sorry
}

end exponent_fraction_simplification_l1920_192014


namespace range_of_a_l1920_192021

/-- Definitions for propositions p and q --/
def p (a : ℝ) : Prop := a > 0 ∧ a < 1
def q (a : ℝ) : Prop := (2 * a - 3) ^ 2 - 4 > 0

/-- Theorem stating the range of possible values for a given conditions --/
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ¬(p a) ∧ ¬(q a) = false) (h4 : p a ∨ q a) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a ≥ 5 / 2) :=
sorry

end range_of_a_l1920_192021


namespace ab_value_l1920_192031

theorem ab_value (a b c : ℤ) (h1 : a^2 = 16) (h2 : 2 * a * b = -40) : a * b = -20 := 
sorry

end ab_value_l1920_192031


namespace max_pens_given_budget_l1920_192052

-- Define the conditions.
def max_pens (x y : ℕ) := 12 * x + 20 * y

-- Define the main theorem stating the proof problem.
theorem max_pens_given_budget : ∃ (x y : ℕ), (10 * x + 15 * y ≤ 173) ∧ (max_pens x y = 224) :=
  sorry

end max_pens_given_budget_l1920_192052


namespace legos_in_box_at_end_l1920_192040

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l1920_192040


namespace other_root_of_quadratic_l1920_192099

theorem other_root_of_quadratic (m : ℝ) :
  (∃ t : ℝ, (x^2 + m * x - 20 = 0) ∧ (x = -4 ∨ x = t)) → (t = 5) :=
by
  sorry

end other_root_of_quadratic_l1920_192099


namespace remaining_bottle_caps_l1920_192039

-- Definitions based on conditions
def initial_bottle_caps : ℕ := 65
def eaten_bottle_caps : ℕ := 4

-- Theorem
theorem remaining_bottle_caps : initial_bottle_caps - eaten_bottle_caps = 61 :=
by
  sorry

end remaining_bottle_caps_l1920_192039


namespace central_angle_of_sector_l1920_192094

noncomputable def sector_area (α r : ℝ) : ℝ := (1/2) * α * r^2

theorem central_angle_of_sector :
  sector_area 3 2 = 6 :=
by
  unfold sector_area
  norm_num
  done

end central_angle_of_sector_l1920_192094


namespace derek_lowest_score_l1920_192058

theorem derek_lowest_score:
  ∀ (score1 score2 max_points target_avg min_score tests_needed last_test1 last_test2 : ℕ),
  score1 = 85 →
  score2 = 78 →
  max_points = 100 →
  target_avg = 84 →
  min_score = 60 →
  tests_needed = 4 →
  last_test1 >= min_score →
  last_test2 >= min_score →
  last_test1 <= max_points →
  last_test2 <= max_points →
  (score1 + score2 + last_test1 + last_test2) = target_avg * tests_needed →
  min last_test1 last_test2 = 73 :=
by
  sorry

end derek_lowest_score_l1920_192058


namespace arithmetic_seq_sin_identity_l1920_192028

theorem arithmetic_seq_sin_identity:
  ∀ (a : ℕ → ℝ), (a 2 + a 6 = (3/2) * Real.pi) → (Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2) :=
by
  sorry

end arithmetic_seq_sin_identity_l1920_192028


namespace capital_after_18_years_l1920_192068

noncomputable def initial_investment : ℝ := 2000
def rate_of_increase : ℝ := 0.50
def period : ℕ := 3
def total_time : ℕ := 18

theorem capital_after_18_years :
  (initial_investment * (1 + rate_of_increase) ^ (total_time / period)) = 22781.25 :=
by
  sorry

end capital_after_18_years_l1920_192068


namespace solution_is_111_l1920_192054

-- Define the system of equations
def system_of_equations (x y z : ℝ) :=
  (x^2 + 7 * y + 2 = 2 * z + 4 * Real.sqrt (7 * x - 3)) ∧
  (y^2 + 7 * z + 2 = 2 * x + 4 * Real.sqrt (7 * y - 3)) ∧
  (z^2 + 7 * x + 2 = 2 * y + 4 * Real.sqrt (7 * z - 3))

-- Prove that x = 1, y = 1, z = 1 is a solution to the system of equations
theorem solution_is_111 : system_of_equations 1 1 1 :=
by
  sorry

end solution_is_111_l1920_192054


namespace find_q_l1920_192061

noncomputable def p : ℝ := -(5 / 6)
noncomputable def g (x : ℝ) : ℝ := p * x^2 + (5 / 6) * x + 5

theorem find_q :
  (∀ x : ℝ, g x = p * x^2 + q * x + r) ∧ 
  (g (-2) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 1 = 5) 
  → q = 5 / 6 :=
sorry

end find_q_l1920_192061


namespace value_of_K_l1920_192042

theorem value_of_K (K: ℕ) : 4^5 * 2^3 = 2^K → K = 13 := by
  sorry

end value_of_K_l1920_192042


namespace lynne_total_spending_l1920_192051

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l1920_192051


namespace number_of_rows_l1920_192020

theorem number_of_rows (r : ℕ) (h1 : ∀ bus : ℕ, bus * (4 * r) = 240) : r = 10 :=
sorry

end number_of_rows_l1920_192020


namespace additional_income_needed_to_meet_goal_l1920_192049

def monthly_current_income : ℤ := 4000
def annual_goal : ℤ := 60000
def additional_amount_per_month (monthly_current_income annual_goal : ℤ) : ℤ :=
  (annual_goal - (monthly_current_income * 12)) / 12

theorem additional_income_needed_to_meet_goal :
  additional_amount_per_month monthly_current_income annual_goal = 1000 :=
by
  sorry

end additional_income_needed_to_meet_goal_l1920_192049


namespace P_parity_Q_div_by_3_l1920_192066

-- Define polynomial P(x)
def P (x p q : ℤ) : ℤ := x*x + p*x + q

-- Define polynomial Q(x)
def Q (x p q : ℤ) : ℤ := x*x*x + p*x + q

-- Part (a) proof statement
theorem P_parity (p q : ℤ) (h1 : Odd p) (h2 : Even q ∨ Odd q) :
  (∀ x : ℤ, Even (P x p q)) ∨ (∀ x : ℤ, Odd (P x p q)) :=
sorry

-- Part (b) proof statement
theorem Q_div_by_3 (p q : ℤ) (h1 : q % 3 = 0) (h2 : p % 3 = 2) :
  ∀ x : ℤ, Q x p q % 3 = 0 :=
sorry

end P_parity_Q_div_by_3_l1920_192066


namespace expenditure_increase_l1920_192072

theorem expenditure_increase
  (current_expenditure : ℝ)
  (future_expenditure : ℝ)
  (years : ℕ)
  (r : ℝ)
  (h₁ : current_expenditure = 1000)
  (h₂ : future_expenditure = 2197)
  (h₃ : years = 3)
  (h₄ : future_expenditure = current_expenditure * (1 + r / 100) ^ years) :
  r = 30 :=
sorry

end expenditure_increase_l1920_192072


namespace range_of_a_l1920_192029

open Set

theorem range_of_a (a x : ℝ) (h : x^2 - 2 * x + 1 - a^2 < 0) (h2 : 0 < x) (h3 : x < 4) :
  a < -3 ∨ a > 3 :=
sorry

end range_of_a_l1920_192029


namespace log_one_fifth_25_eq_neg2_l1920_192044

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l1920_192044


namespace original_employees_229_l1920_192013

noncomputable def original_number_of_employees (reduced_employees : ℕ) (reduction_percentage : ℝ) : ℝ := 
  reduced_employees / (1 - reduction_percentage)

theorem original_employees_229 : original_number_of_employees 195 0.15 = 229 := 
by
  sorry

end original_employees_229_l1920_192013


namespace simplify_expression_l1920_192007

theorem simplify_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x / y) ^ (y - x) :=
by
  sorry

end simplify_expression_l1920_192007


namespace max_discount_benefit_l1920_192000

theorem max_discount_benefit {S X : ℕ} (P : ℕ → Prop) :
  S = 1000 →
  X = 99 →
  (∀ s1 s2 s3 s4 : ℕ, s1 ≥ s2 ∧ s2 ≥ s3 ∧ s3 ≥ s4 ∧ s4 ≥ X ∧ s1 + s2 + s3 + s4 = S →
  ∃ N : ℕ, P N ∧ N = 504) := 
by
  intros hS hX
  sorry

end max_discount_benefit_l1920_192000


namespace total_weight_is_correct_l1920_192090

-- Define the variables
def envelope_weight : ℝ := 8.5
def additional_weight_per_envelope : ℝ := 2
def num_envelopes : ℝ := 880

-- Define the total weight calculation
def total_weight : ℝ := num_envelopes * (envelope_weight + additional_weight_per_envelope)

-- State the theorem to prove that the total weight is as expected
theorem total_weight_is_correct : total_weight = 9240 :=
by
  sorry

end total_weight_is_correct_l1920_192090


namespace number_of_acute_triangles_l1920_192006

def num_triangles : ℕ := 7
def right_triangles : ℕ := 2
def obtuse_triangles : ℕ := 3

theorem number_of_acute_triangles :
  num_triangles - right_triangles - obtuse_triangles = 2 := by
  sorry

end number_of_acute_triangles_l1920_192006


namespace no_positive_x_for_volume_l1920_192062

noncomputable def volume (x : ℤ) : ℤ :=
  (x + 5) * (x - 7) * (x^2 + x + 30)

theorem no_positive_x_for_volume : ¬ ∃ x : ℕ, 0 < x ∧ volume x < 800 := by
  sorry

end no_positive_x_for_volume_l1920_192062


namespace inequality_proof_l1920_192096

theorem inequality_proof 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
  sorry

end inequality_proof_l1920_192096


namespace least_pebbles_2021_l1920_192092

noncomputable def least_pebbles (n : ℕ) : ℕ :=
  n + n / 2

theorem least_pebbles_2021 :
  least_pebbles 2021 = 3031 :=
by
  sorry

end least_pebbles_2021_l1920_192092


namespace masha_can_generate_all_integers_up_to_1093_l1920_192016

theorem masha_can_generate_all_integers_up_to_1093 :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n → n ≤ 1093 → f n ∈ {k | ∃ (a b c d e f g : ℤ), a * 1 + b * 3 + c * 9 + d * 27 + e * 81 + f * 243 + g * 729 = k}) :=
sorry

end masha_can_generate_all_integers_up_to_1093_l1920_192016


namespace percentage_is_4_l1920_192084

-- Define the problem conditions
def percentage_condition (p : ℝ) : Prop := p * 50 = 200

-- State the theorem with the given conditions and the correct answer
theorem percentage_is_4 (p : ℝ) (h : percentage_condition p) : p = 4 := sorry

end percentage_is_4_l1920_192084


namespace tangent_line_computation_l1920_192056

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end tangent_line_computation_l1920_192056


namespace club_additional_members_l1920_192085

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l1920_192085


namespace quadratic_expression_l1920_192027

theorem quadratic_expression (b c : ℤ) : 
  (∀ x : ℝ, (x^2 - 20*x + 49 = (x + b)^2 + c)) → (b + c = -61) :=
by
  sorry

end quadratic_expression_l1920_192027


namespace function_odd_and_decreasing_l1920_192071

noncomputable def f (a x : ℝ) : ℝ := (1 / a) ^ x - a ^ x

theorem function_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -f a x) ∧ (∀ x y, x < y → f a x > f a y) :=
by
  sorry

end function_odd_and_decreasing_l1920_192071


namespace triangle_area_l1920_192005

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end triangle_area_l1920_192005


namespace scientific_notation_of_million_l1920_192057

theorem scientific_notation_of_million : 1000000 = 10^6 :=
by
  sorry

end scientific_notation_of_million_l1920_192057


namespace part1_part2_l1920_192075

def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part1 (x : ℝ) : (f x 2 > 2) ↔ (x > 3 / 2) :=
sorry

theorem part2 (a : ℝ) (ha : a > 0) : (∀ x, f x a < 2 * a) ↔ (1 < a) :=
sorry

end part1_part2_l1920_192075


namespace equal_sum_squares_l1920_192002

open BigOperators

-- Definitions
def n := 10

-- Assuming x and y to be arrays that hold the number of victories and losses for each player respectively.
variables {x y : Fin n → ℝ}

-- Conditions
axiom pair_meet_once : ∀ i : Fin n, x i + y i = (n - 1)

-- Theorem to be proved
theorem equal_sum_squares : ∑ i : Fin n, x i ^ 2 = ∑ i : Fin n, y i ^ 2 :=
by
  sorry

end equal_sum_squares_l1920_192002


namespace find_a_to_make_f_odd_l1920_192008

noncomputable def f (a : ℝ) (x : ℝ): ℝ := x^3 * (Real.log (Real.exp x + 1) + a * x)

theorem find_a_to_make_f_odd :
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1/2 :=
by 
  sorry

end find_a_to_make_f_odd_l1920_192008


namespace quadratic_solutions_l1920_192009

theorem quadratic_solutions (x : ℝ) : (2 * x^2 + 5 * x + 3 = 0) → (x = -1 ∨ x = -3 / 2) :=
by {
  sorry
}

end quadratic_solutions_l1920_192009


namespace olivia_correct_answers_l1920_192041

theorem olivia_correct_answers (c w : ℕ) 
  (h1 : c + w = 15) 
  (h2 : 6 * c - 3 * w = 45) : 
  c = 10 := 
  sorry

end olivia_correct_answers_l1920_192041


namespace solve_x2_plus_4y2_l1920_192011

theorem solve_x2_plus_4y2 (x y : ℝ) (h₁ : x + 2 * y = 6) (h₂ : x * y = -6) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end solve_x2_plus_4y2_l1920_192011


namespace ratio_of_numbers_l1920_192095

theorem ratio_of_numbers (x : ℝ) (h_sum : x + 3.5 = 14) : x / 3.5 = 3 :=
by
  sorry

end ratio_of_numbers_l1920_192095


namespace find_prices_min_cost_l1920_192093

-- Definitions based on conditions
def price_difference (x y : ℕ) : Prop := x - y = 50
def total_cost (x y : ℕ) : Prop := 2 * x + 3 * y = 250
def cost_function (a : ℕ) : ℕ := 50 * a + 6000
def min_items (a : ℕ) : Prop := a ≥ 80
def total_items : ℕ := 200

-- Lean 4 statements for the proof problem
theorem find_prices (x y : ℕ) (h1 : price_difference x y) (h2 : total_cost x y) :
  (x = 80) ∧ (y = 30) :=
sorry

theorem min_cost (a : ℕ) (h1 : min_items a) :
  cost_function a ≥ 10000 :=
sorry

#check find_prices
#check min_cost

end find_prices_min_cost_l1920_192093


namespace percentage_less_than_y_is_70_percent_less_than_z_l1920_192001

variable {x y z : ℝ}

theorem percentage_less_than (h1 : x = 1.20 * y) (h2 : x = 0.36 * z) : y = 0.3 * z :=
by
  sorry

theorem y_is_70_percent_less_than_z (h : y = 0.3 * z) : (1 - y / z) * 100 = 70 :=
by
  sorry

end percentage_less_than_y_is_70_percent_less_than_z_l1920_192001


namespace number_of_flags_l1920_192081

theorem number_of_flags (colors : Finset ℕ) (stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : stripes = 3) : 
  (colors.card ^ stripes) = 27 := 
by
  sorry

end number_of_flags_l1920_192081


namespace kimberly_loan_l1920_192064

theorem kimberly_loan :
  ∃ (t : ℕ), (1.06 : ℝ)^t > 3 ∧ ∀ (t' : ℕ), t' < t → (1.06 : ℝ)^t' ≤ 3 :=
by
sorry

end kimberly_loan_l1920_192064


namespace calculate_value_l1920_192078

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l1920_192078


namespace rectangle_dimensions_l1920_192018

theorem rectangle_dimensions (l w : ℝ) : 
  (∃ x : ℝ, x = l - 3 ∧ x = w - 2 ∧ x^2 = (1 / 2) * l * w) → (l = 9 ∧ w = 8) :=
by
  sorry

end rectangle_dimensions_l1920_192018


namespace second_option_cost_per_day_l1920_192053

theorem second_option_cost_per_day :
  let distance_one_way := 150
  let rental_first_option := 50
  let kilometers_per_liter := 15
  let cost_per_liter := 0.9
  let savings := 22
  let total_distance := distance_one_way * 2
  let total_liters := total_distance / kilometers_per_liter
  let gasoline_cost := total_liters * cost_per_liter
  let total_cost_first_option := rental_first_option + gasoline_cost
  let second_option_cost := total_cost_first_option + savings
  second_option_cost = 90 :=
by
  sorry

end second_option_cost_per_day_l1920_192053


namespace ab_leq_one_l1920_192055

theorem ab_leq_one (a b x : ℝ) (h1 : (x + a) * (x + b) = 9) (h2 : x = a + b) : a * b ≤ 1 := 
sorry

end ab_leq_one_l1920_192055


namespace student_weighted_avg_larger_l1920_192079

variable {u v w : ℚ}

theorem student_weighted_avg_larger (h1 : u < v) (h2 : v < w) :
  (4 * u + 6 * v + 20 * w) / 30 > (2 * u + 3 * v + 4 * w) / 9 := by
  sorry

end student_weighted_avg_larger_l1920_192079


namespace tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l1920_192059

open Real

theorem tan_alpha_minus_pi_over_4_eq_neg_3_over_4 (α β : ℝ) 
  (h1 : tan (α + β) = 1 / 2) 
  (h2 : tan β = 1 / 3) : 
  tan (α - π / 4) = -3 / 4 :=
sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l1920_192059


namespace part1_part2_l1920_192083

-- Part 1
theorem part1 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

-- Part 2
theorem part2 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

end part1_part2_l1920_192083


namespace prime_list_count_l1920_192082

theorem prime_list_count {L : ℕ → ℕ} 
  (hL₀ : L 0 = 29)
  (hL : ∀ (n : ℕ), L (n + 1) = L n * 101 + L 0) :
  (∃! n, n = 0 ∧ Prime (L n)) ∧ ∀ m > 0, ¬ Prime (L m) := 
by
  sorry

end prime_list_count_l1920_192082


namespace cups_of_ketchup_l1920_192073

-- Define variables and conditions
variables (k : ℕ)
def vinegar : ℕ := 1
def honey : ℕ := 1
def sauce_per_burger : ℚ := 1 / 4
def sauce_per_pulled_pork : ℚ := 1 / 6
def burgers : ℕ := 8
def pulled_pork_sandwiches : ℕ := 18

-- Main theorem statement
theorem cups_of_ketchup (h : 8 * sauce_per_burger + 18 * sauce_per_pulled_pork = k + vinegar + honey) : k = 3 :=
  by
    sorry

end cups_of_ketchup_l1920_192073


namespace length_AB_of_parallelogram_l1920_192047

theorem length_AB_of_parallelogram
  (AD BC : ℝ) (AB CD : ℝ) 
  (h1 : AD = 5) 
  (h2 : BC = 5) 
  (h3 : AB = CD)
  (h4 : AD + BC + AB + CD = 14) : 
  AB = 2 :=
by
  sorry

end length_AB_of_parallelogram_l1920_192047


namespace num_small_boxes_l1920_192088

-- Conditions
def chocolates_per_small_box := 25
def total_chocolates := 400

-- Claim: Prove that the number of small boxes is 16
theorem num_small_boxes : (total_chocolates / chocolates_per_small_box) = 16 := 
by sorry

end num_small_boxes_l1920_192088


namespace difference_in_savings_correct_l1920_192038

def S_last_year : ℝ := 45000
def saved_last_year_pct : ℝ := 0.083
def raise_pct : ℝ := 0.115
def saved_this_year_pct : ℝ := 0.056

noncomputable def saved_last_year_amount : ℝ := saved_last_year_pct * S_last_year
noncomputable def S_this_year : ℝ := S_last_year * (1 + raise_pct)
noncomputable def saved_this_year_amount : ℝ := saved_this_year_pct * S_this_year
noncomputable def difference_in_savings : ℝ := saved_last_year_amount - saved_this_year_amount

theorem difference_in_savings_correct :
  difference_in_savings = 925.20 := by
  sorry

end difference_in_savings_correct_l1920_192038


namespace correct_answer_l1920_192091

theorem correct_answer (x : ℝ) (h : 3 * x - 10 = 50) : 3 * x + 10 = 70 :=
sorry

end correct_answer_l1920_192091


namespace sand_exchange_impossible_l1920_192050

/-- Given initial conditions for g and p, the goal is to determine if 
the banker can have at least 2 kg of each type of sand in the end. -/
theorem sand_exchange_impossible (g p : ℕ) (G P : ℕ) 
  (initial_g : g = 1001) (initial_p : p = 1001) 
  (initial_G : G = 1) (initial_P : P = 1)
  (exchange_rule : ∀ x y : ℚ, x * p = y * g) 
  (decrement_rule : ∀ k, 1 ≤ k ∧ k ≤ 2000 → 
    (g = 1001 - k ∨ p = 1001 - k)) :
  ¬(G ≥ 2 ∧ P ≥ 2) :=
by
  -- Add a placeholder to skip the proof
  sorry

end sand_exchange_impossible_l1920_192050


namespace cosine_between_vectors_l1920_192046

noncomputable def vector_cos_angle (a b : ℝ × ℝ) := 
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / (norm_a * norm_b)

theorem cosine_between_vectors (t : ℝ) 
  (ht : let a := (1, t); let b := (-1, 2 * t);
        (3 * a.1 - b.1) * b.1 + (3 * a.2 - b.2) * b.2 = 0) :
  vector_cos_angle (1, t) (-1, 2 * t) = Real.sqrt 3 / 3 := 
by
  sorry

end cosine_between_vectors_l1920_192046


namespace variables_and_unknowns_l1920_192035

theorem variables_and_unknowns (f_1 f_2: ℝ → ℝ → ℝ) (f: ℝ → ℝ → ℝ) :
  (∀ x y, f_1 x y = 0 ∧ f_2 x y = 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (∀ x y, f x y = 0 → (∃ a b, x = a ∧ y = b)) :=
by sorry

end variables_and_unknowns_l1920_192035


namespace rachel_envelopes_first_hour_l1920_192045

theorem rachel_envelopes_first_hour (total_envelopes : ℕ) (hours : ℕ) (e2 : ℕ) (e_per_hour : ℕ) :
  total_envelopes = 1500 → hours = 8 → e2 = 141 → e_per_hour = 204 →
  ∃ e1 : ℕ, e1 = 135 :=
by
  sorry

end rachel_envelopes_first_hour_l1920_192045


namespace slopes_and_angles_l1920_192033

theorem slopes_and_angles (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : θ₁ = 3 * θ₂)
  (h2 : m = 5 * n)
  (h3 : m = Real.tan θ₁)
  (h4 : n = Real.tan θ₂)
  (h5 : m ≠ 0) :
  m * n = 5 / 7 :=
by {
  sorry
}

end slopes_and_angles_l1920_192033


namespace consecutive_triples_with_product_divisible_by_1001_l1920_192012

theorem consecutive_triples_with_product_divisible_by_1001 :
  ∃ (a b c : ℕ), 
    (a = 76 ∧ b = 77 ∧ c = 78) ∨ 
    (a = 77 ∧ b = 78 ∧ c = 79) ∧ 
    (a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧ 
    (b = a + 1 ∧ c = b + 1) ∧ 
    (1001 ∣ (a * b * c)) :=
by sorry

end consecutive_triples_with_product_divisible_by_1001_l1920_192012


namespace common_ratio_q_l1920_192060

noncomputable def Sn (n : ℕ) (a1 q : ℝ) := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q (a1 : ℝ) (q : ℝ) (h : q ≠ 1) (h1 : 6 * Sn 4 a1 q = Sn 5 a1 q + 5 * Sn 6 a1 q) : q = -6/5 := by
  sorry

end common_ratio_q_l1920_192060


namespace identity_problem_l1920_192026

theorem identity_problem
  (a b : ℝ)
  (h₁ : a * b = 2)
  (h₂ : a + b = 3) :
  (a - b)^2 = 1 :=
by
  sorry

end identity_problem_l1920_192026


namespace length_BA_correct_area_ABCDE_correct_l1920_192003

variables {BE CD CE CA : ℝ}
axiom BE_eq : BE = 13
axiom CD_eq : CD = 3
axiom CE_eq : CE = 10
axiom CA_eq : CA = 10

noncomputable def length_BA : ℝ := 3
noncomputable def area_ABCDE : ℝ := 4098 / 61

theorem length_BA_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  length_BA = 3 := 
by { sorry }

theorem area_ABCDE_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  area_ABCDE = 4098 / 61 := 
by { sorry }

end length_BA_correct_area_ABCDE_correct_l1920_192003


namespace cosine_product_identity_l1920_192017

open Real

theorem cosine_product_identity (α : ℝ) (n : ℕ) :
  (List.foldr (· * ·) 1 (List.map (λ k => cos (2^k * α)) (List.range (n + 1)))) =
  sin (2^(n + 1) * α) / (2^(n + 1) * sin α) :=
sorry

end cosine_product_identity_l1920_192017
