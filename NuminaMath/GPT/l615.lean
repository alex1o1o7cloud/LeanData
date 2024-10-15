import Mathlib

namespace NUMINAMATH_GPT_maxOccursAt2_l615_61576

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

theorem maxOccursAt2 {m : ℝ} :
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ f m) ∧ 0 ≤ m ∧ m ≤ 2 → (0 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_GPT_maxOccursAt2_l615_61576


namespace NUMINAMATH_GPT_required_run_rate_l615_61512

/-
In the first 10 overs of a cricket game, the run rate was 3.5. 
What should be the run rate in the remaining 40 overs to reach the target of 320 runs?
-/

def run_rate_in_10_overs : ℝ := 3.5
def overs_played : ℕ := 10
def target_runs : ℕ := 320 
def remaining_overs : ℕ := 40

theorem required_run_rate : 
  (target_runs - (run_rate_in_10_overs * overs_played)) / remaining_overs = 7.125 := by 
sorry

end NUMINAMATH_GPT_required_run_rate_l615_61512


namespace NUMINAMATH_GPT_Malou_score_third_quiz_l615_61521

-- Defining the conditions as Lean definitions
def score1 : ℕ := 91
def score2 : ℕ := 92
def average : ℕ := 91
def num_quizzes : ℕ := 3

-- Proving that score3 equals 90
theorem Malou_score_third_quiz :
  ∃ score3 : ℕ, (score1 + score2 + score3) / num_quizzes = average ∧ score3 = 90 :=
by
  use (90 : ℕ)
  sorry

end NUMINAMATH_GPT_Malou_score_third_quiz_l615_61521


namespace NUMINAMATH_GPT_smallest_number_l615_61557

theorem smallest_number (a b c d e: ℕ) (h1: a = 5) (h2: b = 8) (h3: c = 1) (h4: d = 2) (h5: e = 6) :
  min (min (min (min a b) c) d) e = 1 :=
by
  -- Proof skipped using sorry
  sorry

end NUMINAMATH_GPT_smallest_number_l615_61557


namespace NUMINAMATH_GPT_value_of_b_prod_l615_61500

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 2 ^ (n - 1)

-- The goal is to prove that b_{a_1} * b_{a_3} * b_{a_5} = 4096
theorem value_of_b_prod : b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end NUMINAMATH_GPT_value_of_b_prod_l615_61500


namespace NUMINAMATH_GPT_expression_value_l615_61552

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end NUMINAMATH_GPT_expression_value_l615_61552


namespace NUMINAMATH_GPT_grandpa_tomatoes_before_vacation_l615_61551

theorem grandpa_tomatoes_before_vacation 
  (tomatoes_after_vacation : ℕ) 
  (growth_factor : ℕ) 
  (actual_number : ℕ) 
  (h1 : growth_factor = 100) 
  (h2 : tomatoes_after_vacation = 3564) 
  (h3 : actual_number = tomatoes_after_vacation / growth_factor) : 
  actual_number = 36 := 
by
  -- Here would be the step-by-step proof, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_grandpa_tomatoes_before_vacation_l615_61551


namespace NUMINAMATH_GPT_math_problem_l615_61503

open Function

noncomputable def rotate_90_ccw (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

noncomputable def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem math_problem (a b : ℝ) :
  reflect_over_y_eq_x (rotate_90_ccw (a, b) (2, 3)) = (4, -5) → b - a = -5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_math_problem_l615_61503


namespace NUMINAMATH_GPT_number_of_buyers_l615_61588

theorem number_of_buyers 
  (today yesterday day_before : ℕ) 
  (h1 : today = yesterday + 40) 
  (h2 : yesterday = day_before / 2) 
  (h3 : day_before + yesterday + today = 140) : 
  day_before = 67 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_number_of_buyers_l615_61588


namespace NUMINAMATH_GPT_percentage_female_officers_on_duty_l615_61578

theorem percentage_female_officers_on_duty:
  ∀ (total_on_duty female_on_duty total_female_officers : ℕ),
    total_on_duty = 160 →
    female_on_duty = total_on_duty / 2 →
    total_female_officers = 500 →
    female_on_duty / total_female_officers * 100 = 16 :=
by
  intros total_on_duty female_on_duty total_female_officers h1 h2 h3
  -- Ensure types are correct
  change total_on_duty = 160 at h1
  change female_on_duty = total_on_duty / 2 at h2
  change total_female_officers = 500 at h3
  sorry

end NUMINAMATH_GPT_percentage_female_officers_on_duty_l615_61578


namespace NUMINAMATH_GPT_range_of_a_l615_61513

theorem range_of_a (a : ℝ) (h₀ : a > 0) : (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l615_61513


namespace NUMINAMATH_GPT_one_third_12x_plus_5_l615_61541

-- Define x as a real number
variable (x : ℝ)

-- Define the hypothesis
def h := 12 * x + 5

-- State the theorem
theorem one_third_12x_plus_5 : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 :=
  by 
    sorry -- Proof is omitted

end NUMINAMATH_GPT_one_third_12x_plus_5_l615_61541


namespace NUMINAMATH_GPT_muffin_half_as_expensive_as_banana_l615_61595

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end NUMINAMATH_GPT_muffin_half_as_expensive_as_banana_l615_61595


namespace NUMINAMATH_GPT_inequality_smallest_integer_solution_l615_61524

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end NUMINAMATH_GPT_inequality_smallest_integer_solution_l615_61524


namespace NUMINAMATH_GPT_blue_eyes_count_l615_61555

theorem blue_eyes_count (total_students students_both students_neither : ℕ)
  (ratio_blond_to_blue : ℕ → ℕ)
  (h_total : total_students = 40)
  (h_ratio : ratio_blond_to_blue 3 = 2)
  (h_both : students_both = 8)
  (h_neither : students_neither = 5) :
  ∃ y : ℕ, y = 18 :=
by
  sorry

end NUMINAMATH_GPT_blue_eyes_count_l615_61555


namespace NUMINAMATH_GPT_integer_points_between_A_B_l615_61571

/-- 
Prove that the number of integer coordinate points strictly between 
A(2, 3) and B(50, 80) on the line passing through A and B is c.
-/
theorem integer_points_between_A_B 
  (A B : ℤ × ℤ) (hA : A = (2, 3)) (hB : B = (50, 80)) 
  (c : ℕ) :
  ∃ (n : ℕ), n = c ∧ ∀ (x y : ℤ), (A.1 < x ∧ x < B.1) → (A.2 < y ∧ y < B.2) → 
              (y = ((A.2 - B.2) / (A.1 - B.1) * x + 3 - (A.2 - B.2) / (A.1 - B.1) * 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_points_between_A_B_l615_61571


namespace NUMINAMATH_GPT_distance_between_A_and_B_l615_61514

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l615_61514


namespace NUMINAMATH_GPT_probability_heads_exactly_8_in_10_l615_61596

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end NUMINAMATH_GPT_probability_heads_exactly_8_in_10_l615_61596


namespace NUMINAMATH_GPT_math_proof_problems_l615_61549

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  (sin (π - α) - 2 * sin (π / 2 + α) = 0) → (sin α * cos α + sin α ^ 2 = 6 / 5)

noncomputable def problem2 (α β : ℝ) : Prop :=
  (tan (α + β) = -1) → (tan α = 2) → (tan β = 3)

-- Example of how to state these problems as a theorem
theorem math_proof_problems (α β : ℝ) : problem1 α ∧ problem2 α β := by
  sorry

end NUMINAMATH_GPT_math_proof_problems_l615_61549


namespace NUMINAMATH_GPT_teal_bakery_pumpkin_pie_l615_61570

theorem teal_bakery_pumpkin_pie (P : ℕ) 
    (pumpkin_price_per_slice : ℕ := 5)
    (custard_price_per_slice : ℕ := 6)
    (pumpkin_pies_sold : ℕ := 4)
    (custard_pies_sold : ℕ := 5)
    (custard_pieces_per_pie : ℕ := 6)
    (total_revenue : ℕ := 340) :
    4 * P * pumpkin_price_per_slice + custard_pies_sold * custard_pieces_per_pie * custard_price_per_slice = total_revenue → P = 8 := 
by
  sorry

end NUMINAMATH_GPT_teal_bakery_pumpkin_pie_l615_61570


namespace NUMINAMATH_GPT_solution_l615_61580

namespace Proof

open Set

def proof_problem : Prop :=
  let U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5, 6}
  A ∩ (U \ B) = {1, 2}

theorem solution : proof_problem := by
  -- The pre-defined proof_problem must be shown here
  -- Proof: sorry
  sorry

end Proof

end NUMINAMATH_GPT_solution_l615_61580


namespace NUMINAMATH_GPT_linear_equation_solution_l615_61519

theorem linear_equation_solution (a b : ℤ) (x y : ℤ) (h1 : x = 2) (h2 : y = -1) (h3 : a * x + b * y = -1) : 
  1 + 2 * a - b = 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l615_61519


namespace NUMINAMATH_GPT_bushels_needed_l615_61502

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end NUMINAMATH_GPT_bushels_needed_l615_61502


namespace NUMINAMATH_GPT_Cathy_and_Chris_worked_months_l615_61546

theorem Cathy_and_Chris_worked_months (Cathy_hours : ℕ) (weekly_hours : ℕ) (weeks_in_month : ℕ) (extra_weekly_hours : ℕ) (weeks_for_Chris_sick : ℕ) : 
  Cathy_hours = 180 →
  weekly_hours = 20 →
  weeks_in_month = 4 →
  extra_weekly_hours = weekly_hours →
  weeks_for_Chris_sick = 1 →
  (Cathy_hours - extra_weekly_hours * weeks_for_Chris_sick) / weekly_hours / weeks_in_month = (2 : ℕ) :=
by
  intros hCathy_hours hweekly_hours hweeks_in_month hextra_weekly_hours hweeks_for_Chris_sick
  rw [hCathy_hours, hweekly_hours, hweeks_in_month, hextra_weekly_hours, hweeks_for_Chris_sick]
  norm_num
  sorry

end NUMINAMATH_GPT_Cathy_and_Chris_worked_months_l615_61546


namespace NUMINAMATH_GPT_cricket_average_increase_l615_61558

theorem cricket_average_increase
    (A : ℝ) -- average score after 18 innings
    (score19 : ℝ) -- runs scored in 19th inning
    (new_average : ℝ) -- new average after 19 innings
    (score19_def : score19 = 97)
    (new_average_def :  new_average = 25)
    (total_runs_def : 19 * new_average = 18 * A + 97) : 
    new_average - (18 * A + score19) / 19 = 4 := 
by
  sorry

end NUMINAMATH_GPT_cricket_average_increase_l615_61558


namespace NUMINAMATH_GPT_break_even_production_volume_l615_61586

theorem break_even_production_volume
  (Q : ℕ) 
  (ATC : ℕ → ℚ)
  (P : ℚ)
  (h1 : ∀ Q, ATC Q = 100 + 100000 / Q)
  (h2 : P = 300) :
  ATC 500 = P :=
by
  sorry

end NUMINAMATH_GPT_break_even_production_volume_l615_61586


namespace NUMINAMATH_GPT_area_of_sector_l615_61581

theorem area_of_sector (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 60) : (θ / 360 * π * r^2 = 6 * π) :=
by sorry

end NUMINAMATH_GPT_area_of_sector_l615_61581


namespace NUMINAMATH_GPT_Kenny_running_to_basketball_ratio_l615_61582

theorem Kenny_running_to_basketball_ratio (basketball_hours trumpet_hours running_hours : ℕ) 
    (h1 : basketball_hours = 10)
    (h2 : trumpet_hours = 2 * running_hours)
    (h3 : trumpet_hours = 40) :
    running_hours = 20 ∧ basketball_hours = 10 ∧ (running_hours / basketball_hours = 2) :=
by
  sorry

end NUMINAMATH_GPT_Kenny_running_to_basketball_ratio_l615_61582


namespace NUMINAMATH_GPT_most_stable_performance_l615_61523

-- Given variances for the four people
def S_A_var : ℝ := 0.56
def S_B_var : ℝ := 0.60
def S_C_var : ℝ := 0.50
def S_D_var : ℝ := 0.45

-- We need to prove that the variance for D is the smallest
theorem most_stable_performance :
  S_D_var < S_C_var ∧ S_D_var < S_A_var ∧ S_D_var < S_B_var :=
by
  sorry

end NUMINAMATH_GPT_most_stable_performance_l615_61523


namespace NUMINAMATH_GPT_mass_percentage_of_N_in_NH4Br_l615_61536

theorem mass_percentage_of_N_in_NH4Br :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Br := 79.90
  let molar_mass_NH4Br := (1 * molar_mass_N) + (4 * molar_mass_H) + (1 * molar_mass_Br)
  let mass_percentage_N := (molar_mass_N / molar_mass_NH4Br) * 100
  mass_percentage_N = 14.30 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_N_in_NH4Br_l615_61536


namespace NUMINAMATH_GPT_line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l615_61592

-- Define the point (M) and the properties of the line
def point_M : ℝ × ℝ := (-1, 3)

def parallel_to_y_axis (line : ℝ × ℝ → Prop) : Prop :=
  ∃ b : ℝ, ∀ y : ℝ, line (b, y)

-- Statement we need to prove
theorem line_through_point_parallel_to_y_axis_eq_x_eq_neg1 :
  (∃ line : ℝ × ℝ → Prop, line point_M ∧ parallel_to_y_axis line) → ∀ p : ℝ × ℝ, (p.1 = -1 ↔ (∃ line : ℝ × ℝ → Prop, line p ∧ line point_M ∧ parallel_to_y_axis line)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l615_61592


namespace NUMINAMATH_GPT_m_range_satisfies_inequality_l615_61540

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x + sin x

theorem m_range_satisfies_inequality :
  ∀ (m : ℝ), f (2 * m ^ 2 - m + π - 1) ≥ -2 * π ↔ -1 / 2 ≤ m ∧ m ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_m_range_satisfies_inequality_l615_61540


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l615_61568

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : Real.sin α = 3 / 5) : Real.tan (α - π / 4) = -1 / 7 ∨ Real.tan (α - π / 4) = -7 := 
sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l615_61568


namespace NUMINAMATH_GPT_equal_rental_costs_l615_61543

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end NUMINAMATH_GPT_equal_rental_costs_l615_61543


namespace NUMINAMATH_GPT_solve_for_y_l615_61584

theorem solve_for_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l615_61584


namespace NUMINAMATH_GPT_minimum_value_of_f_l615_61569

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / Real.sqrt (x^2 + 5)

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 6 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l615_61569


namespace NUMINAMATH_GPT_highest_probability_white_ball_l615_61597

theorem highest_probability_white_ball :
  let red_balls := 2
  let black_balls := 3
  let white_balls := 4
  let total_balls := red_balls + black_balls + white_balls
  let prob_red := red_balls / total_balls
  let prob_black := black_balls / total_balls
  let prob_white := white_balls / total_balls
  prob_white > prob_black ∧ prob_black > prob_red :=
by
  sorry

end NUMINAMATH_GPT_highest_probability_white_ball_l615_61597


namespace NUMINAMATH_GPT_problem_statement_l615_61517

noncomputable def equation_of_altitude (A B C: (ℝ × ℝ)): (ℝ × ℝ × ℝ) :=
by
  sorry

theorem problem_statement :
  let A := (-1, 4)
  let B := (-2, -1)
  let C := (2, 3)
  equation_of_altitude A B C = (1, 1, -3) ∧
  |1 / 2 * (4 - (-1)) * 4| = 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l615_61517


namespace NUMINAMATH_GPT_seashells_given_l615_61526

theorem seashells_given (original_seashells : ℕ) (current_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 35) 
  (h2 : current_seashells = 17) 
  (h3 : given_seashells = original_seashells - current_seashells) : 
  given_seashells = 18 := 
by 
  sorry

end NUMINAMATH_GPT_seashells_given_l615_61526


namespace NUMINAMATH_GPT_train_stops_15_min_per_hour_l615_61573

/-
Without stoppages, a train travels a certain distance with an average speed of 80 km/h,
and with stoppages, it covers the same distance with an average speed of 60 km/h.
Prove that the train stops for 15 minutes per hour.
-/
theorem train_stops_15_min_per_hour (D : ℝ) (h1 : 0 < D) :
  let T_no_stop := D / 80
  let T_stop := D / 60
  let T_lost := T_stop - T_no_stop
  let mins_per_hour := T_lost * 60
  mins_per_hour = 15 := by
  sorry

end NUMINAMATH_GPT_train_stops_15_min_per_hour_l615_61573


namespace NUMINAMATH_GPT_intersect_at_2d_l615_61566

def g (x : ℝ) (c : ℝ) : ℝ := 4 * x + c

theorem intersect_at_2d (c d : ℤ) (h₁ : d = 8 + c) (h₂ : 2 = g d c) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersect_at_2d_l615_61566


namespace NUMINAMATH_GPT_calculate_expression_l615_61504

theorem calculate_expression : |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l615_61504


namespace NUMINAMATH_GPT_train_speed_first_part_l615_61531

theorem train_speed_first_part (x v : ℝ) (h1 : 0 < x) (h2 : 0 < v) 
  (h_avg_speed : (3 * x) / (x / v + 2 * x / 20) = 22.5) : v = 30 :=
sorry

end NUMINAMATH_GPT_train_speed_first_part_l615_61531


namespace NUMINAMATH_GPT_eval_expression_at_values_l615_61547

theorem eval_expression_at_values : 
  ∀ x y : ℕ, x = 3 ∧ y = 4 → 
  5 * (x^(y+1)) + 6 * (y^(x+1)) + 2 * x * y = 2775 :=
by
  intros x y hxy
  cases hxy
  sorry

end NUMINAMATH_GPT_eval_expression_at_values_l615_61547


namespace NUMINAMATH_GPT_other_number_of_given_conditions_l615_61528

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end NUMINAMATH_GPT_other_number_of_given_conditions_l615_61528


namespace NUMINAMATH_GPT_part1_part2_l615_61518

namespace Problem

open Real

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

theorem part1 (h : p 1 x ∧ q x) : 2 < x ∧ x < 3:= 
sorry

theorem part2 (hpq : ∀ x, ¬ p a x → ¬ q x) : 
   1 < a ∧ a ≤ 2 := 
sorry

end Problem

end NUMINAMATH_GPT_part1_part2_l615_61518


namespace NUMINAMATH_GPT_number_of_paths_l615_61598

theorem number_of_paths (n : ℕ) (h1 : n > 3) : 
  (2 * (8 * n^3 - 48 * n^2 + 88 * n - 48) + (4 * n^2 - 12 * n + 8) + (2 * n - 2)) = 16 * n^3 - 92 * n^2 + 166 * n - 90 :=
by
  sorry

end NUMINAMATH_GPT_number_of_paths_l615_61598


namespace NUMINAMATH_GPT_prime_difference_condition_l615_61590

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_difference_condition :
  ∃ (x y : ℕ), is_prime x ∧ is_prime y ∧ 4 < x ∧ x < 18 ∧ 4 < y ∧ y < 18 ∧ x ≠ y ∧ (x * y - (x + y)) = 119 :=
by
  sorry

end NUMINAMATH_GPT_prime_difference_condition_l615_61590


namespace NUMINAMATH_GPT_angles_does_not_exist_l615_61575

theorem angles_does_not_exist (a1 a2 a3 : ℝ) 
  (h1 : a1 + a2 = 90) 
  (h2 : a2 + a3 = 180) 
  (h3 : a3 = 18) : False :=
by
  sorry

end NUMINAMATH_GPT_angles_does_not_exist_l615_61575


namespace NUMINAMATH_GPT_mean_equality_l615_61565

theorem mean_equality (z : ℝ) :
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mean_equality_l615_61565


namespace NUMINAMATH_GPT_minimum_f_value_l615_61577

noncomputable def f (x : ℝ) : ℝ :=
   Real.sqrt (2 * x ^ 2 - 4 * x + 4) + 
   Real.sqrt (2 * x ^ 2 - 16 * x + (Real.log x / Real.log 2) ^ 2 - 2 * x * (Real.log x / Real.log 2) + 
              2 * (Real.log x / Real.log 2) + 50)

theorem minimum_f_value : ∀ x : ℝ, x > 0 → f x ≥ 7 ∧ f 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_minimum_f_value_l615_61577


namespace NUMINAMATH_GPT_determine_b_l615_61520

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iio 2 ∪ Set.Ioi 6 → -x^2 + b * x - 7 < 0) ∧ 
  (∀ x : ℝ, ¬(x ∈ Set.Iio 2 ∪ Set.Ioi 6) → ¬(-x^2 + b * x - 7 < 0)) → 
  b = 8 :=
sorry

end NUMINAMATH_GPT_determine_b_l615_61520


namespace NUMINAMATH_GPT_vertex_of_parabola_l615_61527

theorem vertex_of_parabola : ∀ x y : ℝ, y = 2 * (x - 1) ^ 2 + 2 → (1, 2) = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l615_61527


namespace NUMINAMATH_GPT_middle_number_between_52_and_certain_number_l615_61556

theorem middle_number_between_52_and_certain_number :
  ∃ n, n > 52 ∧ (∀ k, 52 ≤ k ∧ k ≤ n → ∃ l, k = 52 + l) ∧ (n = 52 + 16) :=
sorry

end NUMINAMATH_GPT_middle_number_between_52_and_certain_number_l615_61556


namespace NUMINAMATH_GPT_find_value_of_a2004_b2004_l615_61509

-- Given Definitions and Conditions
def a : ℝ := sorry
def b : ℝ := sorry
def A : Set ℝ := {a, a^2, a * b}
def B : Set ℝ := {1, a, b}

-- The theorem statement
theorem find_value_of_a2004_b2004 (h : A = B) : a ^ 2004 + b ^ 2004 = 1 :=
sorry

end NUMINAMATH_GPT_find_value_of_a2004_b2004_l615_61509


namespace NUMINAMATH_GPT_find_diagonal_length_l615_61510

theorem find_diagonal_length (d : ℝ) (offset1 offset2 : ℝ) (area : ℝ)
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 300) :
  (1/2) * d * (offset1 + offset2) = area → d = 40 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_find_diagonal_length_l615_61510


namespace NUMINAMATH_GPT_final_amounts_calculation_l615_61587

noncomputable def article_A_original_cost : ℚ := 200
noncomputable def article_B_original_cost : ℚ := 300
noncomputable def article_C_original_cost : ℚ := 400
noncomputable def exchange_rate_euro_to_usd : ℚ := 1.10
noncomputable def exchange_rate_gbp_to_usd : ℚ := 1.30
noncomputable def discount_A : ℚ := 0.50
noncomputable def discount_B : ℚ := 0.30
noncomputable def discount_C : ℚ := 0.40
noncomputable def sales_tax_rate : ℚ := 0.05
noncomputable def reward_points : ℚ := 100
noncomputable def reward_point_value : ℚ := 0.05

theorem final_amounts_calculation :
  let discounted_A := article_A_original_cost * discount_A
  let final_A := (article_A_original_cost - discounted_A) * exchange_rate_euro_to_usd
  let discounted_B := article_B_original_cost * discount_B
  let final_B := (article_B_original_cost - discounted_B) * exchange_rate_gbp_to_usd
  let discounted_C := article_C_original_cost * discount_C
  let final_C := article_C_original_cost - discounted_C
  let total_discounted_cost_usd := final_A + final_B + final_C
  let sales_tax := total_discounted_cost_usd * sales_tax_rate
  let reward := reward_points * reward_point_value
  let final_amount_usd := total_discounted_cost_usd + sales_tax - reward
  let final_amount_euro := final_amount_usd / exchange_rate_euro_to_usd
  final_amount_usd = 649.15 ∧ final_amount_euro = 590.14 :=
by
  sorry

end NUMINAMATH_GPT_final_amounts_calculation_l615_61587


namespace NUMINAMATH_GPT_bill_spots_l615_61544

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end NUMINAMATH_GPT_bill_spots_l615_61544


namespace NUMINAMATH_GPT_proof_valid_set_exists_l615_61532

noncomputable def valid_set_exists : Prop :=
∃ (s : Finset ℕ), s.card = 10 ∧ 
(∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → a ≠ b) ∧ 
(∃ (t1 : Finset ℕ), t1 ⊆ s ∧ t1.card = 3 ∧ ∀ n ∈ t1, 5 ∣ n) ∧
(∃ (t2 : Finset ℕ), t2 ⊆ s ∧ t2.card = 4 ∧ ∀ n ∈ t2, 4 ∣ n) ∧
s.sum id < 75

theorem proof_valid_set_exists : valid_set_exists :=
sorry

end NUMINAMATH_GPT_proof_valid_set_exists_l615_61532


namespace NUMINAMATH_GPT_average_children_l615_61537

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_children_l615_61537


namespace NUMINAMATH_GPT_complex_division_product_l615_61589

theorem complex_division_product
  (i : ℂ)
  (h_exp: i * i = -1)
  (a b : ℝ)
  (h_div: (1 + 7 * i) / (2 - i) = a + b * i)
  : a * b = -3 := 
sorry

end NUMINAMATH_GPT_complex_division_product_l615_61589


namespace NUMINAMATH_GPT_part1_part2_l615_61574

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l615_61574


namespace NUMINAMATH_GPT_estimate_diff_and_prod_l615_61525

variable {x y : ℝ}
variable (hx : x > y) (hy : y > 0)

theorem estimate_diff_and_prod :
  (1.1*x) - (y - 2) = (x - y) + 0.1 * x + 2 ∧ (1.1 * x) * (y - 2) = 1.1 * (x * y) - 2.2 * x :=
by 
  sorry -- Proof details go here

end NUMINAMATH_GPT_estimate_diff_and_prod_l615_61525


namespace NUMINAMATH_GPT_speed_difference_is_36_l615_61538

open Real

noncomputable def alex_speed : ℝ := 8 / (40 / 60)
noncomputable def jordan_speed : ℝ := 12 / (15 / 60)
noncomputable def speed_difference : ℝ := jordan_speed - alex_speed

theorem speed_difference_is_36 : speed_difference = 36 := by
  have hs1 : alex_speed = 8 / (40 / 60) := rfl
  have hs2 : jordan_speed = 12 / (15 / 60) := rfl
  have hd : speed_difference = jordan_speed - alex_speed := rfl
  rw [hs1, hs2] at hd
  simp [alex_speed, jordan_speed, speed_difference] at hd
  sorry

end NUMINAMATH_GPT_speed_difference_is_36_l615_61538


namespace NUMINAMATH_GPT_solve_for_k_l615_61561

theorem solve_for_k (x k : ℝ) (h : k ≠ 0) 
(h_eq : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 7)) : k = 7 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_solve_for_k_l615_61561


namespace NUMINAMATH_GPT_spinner_prob_C_l615_61579

theorem spinner_prob_C (P_A P_B P_C : ℚ) (h_A : P_A = 1/3) (h_B : P_B = 5/12) (h_total : P_A + P_B + P_C = 1) : 
  P_C = 1/4 := 
sorry

end NUMINAMATH_GPT_spinner_prob_C_l615_61579


namespace NUMINAMATH_GPT_chocolate_difference_l615_61548

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_difference_l615_61548


namespace NUMINAMATH_GPT_root_quadratic_l615_61534

theorem root_quadratic (m : ℝ) (h : m^2 - 2*m - 1 = 0) : m^2 + 1/m^2 = 6 :=
sorry

end NUMINAMATH_GPT_root_quadratic_l615_61534


namespace NUMINAMATH_GPT_exists_duplicate_in_grid_of_differences_bounded_l615_61530

theorem exists_duplicate_in_grid_of_differences_bounded :
  ∀ (f : ℕ × ℕ → ℤ), 
  (∀ i j, i < 10 → j < 10 → (i + 1 < 10 → (abs (f (i, j) - f (i + 1, j)) ≤ 5)) 
                             ∧ (j + 1 < 10 → (abs (f (i, j) - f (i, j + 1)) ≤ 5))) → 
  ∃ x y : ℕ × ℕ, x ≠ y ∧ f x = f y :=
by
  intros
  sorry -- Proof goes here

end NUMINAMATH_GPT_exists_duplicate_in_grid_of_differences_bounded_l615_61530


namespace NUMINAMATH_GPT_linda_original_savings_l615_61539

theorem linda_original_savings (S : ℝ) (f : ℝ) (a : ℝ) (t : ℝ) 
  (h1 : f = 7 / 13 * S) (h2 : a = 3 / 13 * S) 
  (h3 : t = S - f - a) (h4 : t = 180) (h5 : a = 360) : 
  S = 1560 :=
by 
  sorry

end NUMINAMATH_GPT_linda_original_savings_l615_61539


namespace NUMINAMATH_GPT_same_terminal_side_angles_l615_61508

theorem same_terminal_side_angles (k : ℤ) :
  ∃ (k1 k2 : ℤ), k1 * 360 - 1560 = -120 ∧ k2 * 360 - 1560 = 240 :=
by
  -- Conditions and property definitions can be added here if needed
  sorry

end NUMINAMATH_GPT_same_terminal_side_angles_l615_61508


namespace NUMINAMATH_GPT_simplify_expression_l615_61591

variable (y : ℝ)

theorem simplify_expression : 3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := 
  by
  sorry

end NUMINAMATH_GPT_simplify_expression_l615_61591


namespace NUMINAMATH_GPT_alpha_beta_working_together_time_l615_61562

theorem alpha_beta_working_together_time
  (A B C : ℝ)
  (h : ℝ)
  (hA : A = B + 5)
  (work_together_A : A > 0)
  (work_together_B : B > 0)
  (work_together_C : C > 0)
  (combined_work : 1/A + 1/B + 1/C = 1/(A - 6))
  (combined_work2 : 1/A + 1/B + 1/C = 1/(B - 1))
  (time_gamma : 1/A + 1/B + 1/C = 2/C) :
  h = 4/3 :=
sorry

end NUMINAMATH_GPT_alpha_beta_working_together_time_l615_61562


namespace NUMINAMATH_GPT_larry_correct_evaluation_l615_61516

theorem larry_correct_evaluation (a b c d e : ℝ) 
(Ha : a = 5) (Hb : b = 3) (Hc : c = 6) (Hd : d = 4) :
a - b + c + d - e = a - (b - (c + (d - e))) → e = 0 :=
by
  -- Not providing the actual proof
  sorry

end NUMINAMATH_GPT_larry_correct_evaluation_l615_61516


namespace NUMINAMATH_GPT_orthocenter_of_triangle_ABC_l615_61553

def point : Type := ℝ × ℝ × ℝ

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)

def orthocenter (A B C : point) : point := sorry -- We'll skip the function implementation here

theorem orthocenter_of_triangle_ABC :
  orthocenter A B C = (13/7, 41/14, 55/7) :=
sorry

end NUMINAMATH_GPT_orthocenter_of_triangle_ABC_l615_61553


namespace NUMINAMATH_GPT_ordered_triples_count_l615_61506

noncomputable def count_valid_triples (n : ℕ) :=
  ∃ x y z : ℕ, ∃ k : ℕ, x * y * z = k ∧ k = 5 ∧ lcm x y = 48 ∧ lcm x z = 450 ∧ lcm y z = 600

theorem ordered_triples_count : count_valid_triples 5 := by
  sorry

end NUMINAMATH_GPT_ordered_triples_count_l615_61506


namespace NUMINAMATH_GPT_relationship_y1_y2_l615_61594

variables {x1 x2 : ℝ}

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 + 6 * x - 5

theorem relationship_y1_y2 (hx1 : 0 ≤ x1) (hx1_lt : x1 < 1) (hx2 : 2 ≤ x2) (hx2_lt : x2 < 3) :
  f x1 ≥ f x2 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_l615_61594


namespace NUMINAMATH_GPT_negation_of_conditional_l615_61593

-- Define the propositions
def P (x : ℝ) : Prop := x > 2015
def Q (x : ℝ) : Prop := x > 0

-- Negate the propositions
def notP (x : ℝ) : Prop := x <= 2015
def notQ (x : ℝ) : Prop := x <= 0

-- Theorem: Negation of the conditional statement
theorem negation_of_conditional (x : ℝ) : ¬ (P x → Q x) ↔ (notP x → notQ x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_conditional_l615_61593


namespace NUMINAMATH_GPT_kishore_savings_l615_61511

noncomputable def rent := 5000
noncomputable def milk := 1500
noncomputable def groceries := 4500
noncomputable def education := 2500
noncomputable def petrol := 2000
noncomputable def miscellaneous := 700
noncomputable def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def salary : ℝ := total_expenses / 0.9 -- given that savings is 10% of salary

theorem kishore_savings : (salary * 0.1) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_kishore_savings_l615_61511


namespace NUMINAMATH_GPT_normal_level_shortage_l615_61522

variable (T : ℝ) (normal_capacity : ℝ) (end_of_month_reservoir : ℝ)
variable (h1 : end_of_month_reservoir = 6)
variable (h2 : end_of_month_reservoir = 2 * normal_capacity)
variable (h3 : end_of_month_reservoir = 0.60 * T)

theorem normal_level_shortage :
  normal_capacity = 7 :=
by
  sorry

end NUMINAMATH_GPT_normal_level_shortage_l615_61522


namespace NUMINAMATH_GPT_snow_globes_in_box_l615_61533

theorem snow_globes_in_box (S : ℕ) 
  (h1 : ∀ (box_decorations : ℕ), box_decorations = 4 + 1 + S)
  (h2 : ∀ (num_boxes : ℕ), num_boxes = 12)
  (h3 : ∀ (total_decorations : ℕ), total_decorations = 120) :
  S = 5 :=
by
  sorry

end NUMINAMATH_GPT_snow_globes_in_box_l615_61533


namespace NUMINAMATH_GPT_cost_of_jeans_l615_61583

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end NUMINAMATH_GPT_cost_of_jeans_l615_61583


namespace NUMINAMATH_GPT_milk_butterfat_problem_l615_61501

variable (x : ℝ)

def butterfat_10_percent (x : ℝ) := 0.10 * x
def butterfat_35_percent_in_8_gallons : ℝ := 0.35 * 8
def total_milk (x : ℝ) := x + 8
def total_butterfat (x : ℝ) := 0.20 * (x + 8)

theorem milk_butterfat_problem 
    (h : butterfat_10_percent x + butterfat_35_percent_in_8_gallons = total_butterfat x) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_milk_butterfat_problem_l615_61501


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l615_61559

open Real

theorem relationship_between_a_and_b
  (a b x : ℝ)
  (h1 : a ≠ 1)
  (h2 : b ≠ 1)
  (h3 : 4 * (log x / log a)^3 + 5 * (log x / log b)^3 = 7 * (log x)^3) :
  b = a ^ (3 / 5)^(1 / 3) := 
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l615_61559


namespace NUMINAMATH_GPT_girls_attending_event_l615_61567

theorem girls_attending_event (total_students girls_attending boys_attending : ℕ) 
    (h1 : total_students = 1500) 
    (h2 : girls_attending = 3 / 5 * girls) 
    (h3 : boys_attending = 2 / 3 * (total_students - girls)) 
    (h4 : girls_attending + boys_attending = 900) : 
    girls_attending = 900 := 
by 
    sorry

end NUMINAMATH_GPT_girls_attending_event_l615_61567


namespace NUMINAMATH_GPT_geom_seq_42_l615_61545

variable {α : Type*} [Field α] [CharZero α]

noncomputable def a_n (n : ℕ) (a1 q : α) : α := a1 * q ^ n

theorem geom_seq_42 (a1 q : α) (h1 : a1 = 3) (h2 : a1 * (1 + q^2 + q^4) = 21) :
  a1 * (q^2 + q^4 + q^6) = 42 := 
by
  sorry

end NUMINAMATH_GPT_geom_seq_42_l615_61545


namespace NUMINAMATH_GPT_solve_for_k_l615_61572

theorem solve_for_k (a k : ℝ) (h : a ^ 10 / (a ^ k) ^ 4 = a ^ 2) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l615_61572


namespace NUMINAMATH_GPT_silver_coin_value_l615_61585

--- Definitions from the conditions
def total_value_hoard (value_silver : ℕ) := 100 * 3 * value_silver + 60 * value_silver + 33

--- Statement of the theorem to prove
theorem silver_coin_value (x : ℕ) (h : total_value_hoard x = 2913) : x = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_silver_coin_value_l615_61585


namespace NUMINAMATH_GPT_piecewise_function_identity_l615_61529

theorem piecewise_function_identity (x : ℝ) : 
  (3 * x + abs (5 * x - 10)) = if x < 2 then -2 * x + 10 else 8 * x - 10 := by
  sorry

end NUMINAMATH_GPT_piecewise_function_identity_l615_61529


namespace NUMINAMATH_GPT_no_perfect_square_solution_l615_61554

theorem no_perfect_square_solution (n : ℕ) (x : ℕ) (hx : x < 10^n) :
  ¬ (∀ y, 0 ≤ y ∧ y ≤ 9 → ∃ z : ℤ, ∃ k : ℤ, 10^(n+1) * z + 10 * x + y = k^2) :=
sorry

end NUMINAMATH_GPT_no_perfect_square_solution_l615_61554


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_not_perfect_square_l615_61550

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_five_consecutive_integers_not_perfect_square_l615_61550


namespace NUMINAMATH_GPT_complex_combination_l615_61535

open Complex

def a : ℂ := 2 - I
def b : ℂ := -1 + I

theorem complex_combination : 2 * a + 3 * b = 1 + I :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_complex_combination_l615_61535


namespace NUMINAMATH_GPT_total_miles_ran_l615_61515

theorem total_miles_ran (miles_monday miles_wednesday miles_friday : ℕ)
  (h1 : miles_monday = 3)
  (h2 : miles_wednesday = 2)
  (h3 : miles_friday = 7) :
  miles_monday + miles_wednesday + miles_friday = 12 := 
by
  sorry

end NUMINAMATH_GPT_total_miles_ran_l615_61515


namespace NUMINAMATH_GPT_find_initial_avg_height_l615_61560

noncomputable def initially_calculated_avg_height (A : ℚ) (boys : ℕ) (wrong_height right_height : ℚ) (actual_avg_height : ℚ) :=
  boys = 35 ∧
  wrong_height = 166 ∧
  right_height = 106 ∧
  actual_avg_height = 182 ∧
  35 * A - (wrong_height - right_height) = 35 * actual_avg_height

theorem find_initial_avg_height : ∃ A : ℚ, initially_calculated_avg_height A 35 166 106 182 ∧ A = 183.71 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_avg_height_l615_61560


namespace NUMINAMATH_GPT_scientific_notation_correct_l615_61507

-- Define the given number
def given_number : ℕ := 138000

-- Define the scientific notation expression
def scientific_notation : ℝ := 1.38 * 10^5

-- The proof goal: Prove that 138,000 expressed in scientific notation is 1.38 * 10^5
theorem scientific_notation_correct : (given_number : ℝ) = scientific_notation := by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l615_61507


namespace NUMINAMATH_GPT_problem_statement_l615_61563

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, 8 < x → f (x) > f (x + 1))
variable (h2 : ∀ x, f (x + 8) = f (-x + 8))

theorem problem_statement : f 7 > f 10 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l615_61563


namespace NUMINAMATH_GPT_increase_in_average_weight_l615_61564

theorem increase_in_average_weight 
    (A : ℝ) 
    (weight_left : ℝ)
    (weight_new : ℝ)
    (h_weight_left : weight_left = 67)
    (h_weight_new : weight_new = 87) : 
    ((8 * A - weight_left + weight_new) / 8 - A) = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_increase_in_average_weight_l615_61564


namespace NUMINAMATH_GPT_b_minus_a_eq_two_l615_61505

theorem b_minus_a_eq_two (a b : ℤ) (h1 : b = 7) (h2 : a * b = 2 * (a + b) + 11) : b - a = 2 :=
by
  sorry

end NUMINAMATH_GPT_b_minus_a_eq_two_l615_61505


namespace NUMINAMATH_GPT_calculation_result_l615_61542

theorem calculation_result : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by
  sorry

end NUMINAMATH_GPT_calculation_result_l615_61542


namespace NUMINAMATH_GPT_games_bought_l615_61599

/-- 
Given:
1. Geoffrey received €20 from his grandmother.
2. Geoffrey received €25 from his aunt.
3. Geoffrey received €30 from his uncle.
4. Geoffrey now has €125 in his wallet.
5. Geoffrey has €20 left after buying games.
6. Each game costs €35.

Prove that Geoffrey bought 3 games.
-/
theorem games_bought 
  (grandmother_money aunt_money uncle_money total_money left_money game_cost spent_money games_bought : ℤ)
  (h1 : grandmother_money = 20)
  (h2 : aunt_money = 25)
  (h3 : uncle_money = 30)
  (h4 : total_money = 125)
  (h5 : left_money = 20)
  (h6 : game_cost = 35)
  (h7 : spent_money = total_money - left_money)
  (h8 : games_bought = spent_money / game_cost) :
  games_bought = 3 := 
sorry

end NUMINAMATH_GPT_games_bought_l615_61599
