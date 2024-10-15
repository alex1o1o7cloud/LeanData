import Mathlib

namespace NUMINAMATH_GPT_correlation_index_l283_28352

variable (height_variation_weight_explained : ℝ)
variable (random_errors_contribution : ℝ)

def R_squared : ℝ := height_variation_weight_explained

theorem correlation_index (h1 : height_variation_weight_explained = 0.64) (h2 : random_errors_contribution = 0.36) : R_squared height_variation_weight_explained = 0.64 :=
by
  exact h1  -- Placeholder for actual proof, since only statement is required

end NUMINAMATH_GPT_correlation_index_l283_28352


namespace NUMINAMATH_GPT_seq_periodic_l283_28311

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1/4
  else ite (n > 1) (1 - (1 / (seq (n-1)))) 0 -- handle invalid cases with a default zero

theorem seq_periodic {n : ℕ} (h : seq 1 = 1/4) (h2 : ∀ k ≥ 2, seq k = 1 - (1 / (seq (k-1)))) :
  seq 2014 = 1/4 :=
sorry

end NUMINAMATH_GPT_seq_periodic_l283_28311


namespace NUMINAMATH_GPT_problem_statement_l283_28310

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x / Real.log 2 else sorry

theorem problem_statement : f (1 / 2) < f (1 / 3) ∧ f (1 / 3) < f 2 :=
by
  -- Definitions based on given conditions
  have h1 : ∀ x : ℝ, f (2 - x) = f x := sorry
  have h2 : ∀ x : ℝ, 1 ≤ x → f x = Real.log x / Real.log 2 := sorry
  -- Proof of the statement based on h1 and h2
  sorry

end NUMINAMATH_GPT_problem_statement_l283_28310


namespace NUMINAMATH_GPT_polynomial_remainder_l283_28355

theorem polynomial_remainder (a b : ℝ) (h : ∀ x : ℝ, (x^3 - 2*x^2 + a*x + b) % ((x - 1)*(x - 2)) = 2*x + 1) : 
  a = 1 ∧ b = 3 := 
sorry

end NUMINAMATH_GPT_polynomial_remainder_l283_28355


namespace NUMINAMATH_GPT_more_ducks_than_four_times_chickens_l283_28342

def number_of_chickens (C : ℕ) : Prop :=
  185 = 150 + C

def number_of_ducks (C : ℕ) (MoreDucks : ℕ) : Prop :=
  150 = 4 * C + MoreDucks

theorem more_ducks_than_four_times_chickens (C MoreDucks : ℕ) (h1 : number_of_chickens C) (h2 : number_of_ducks C MoreDucks) : MoreDucks = 10 := by
  sorry

end NUMINAMATH_GPT_more_ducks_than_four_times_chickens_l283_28342


namespace NUMINAMATH_GPT_quadratic_expression_value_l283_28317

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l283_28317


namespace NUMINAMATH_GPT_graph_of_equation_is_shifted_hyperbola_l283_28383

-- Definitions
def given_equation (x y : ℝ) : Prop := x^2 - 4*y^2 - 2*x = 0

-- Theorem statement
theorem graph_of_equation_is_shifted_hyperbola :
  ∀ x y : ℝ, given_equation x y = ((x - 1)^2 = 1 + 4*y^2) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_shifted_hyperbola_l283_28383


namespace NUMINAMATH_GPT_solution_set_of_inequality_l283_28369

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : 
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio (0 : ℝ) ∪ Set.Ici (1 / 2) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l283_28369


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l283_28321

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 2*x < 0) → (|x - 2| < 2) ∧ ¬(|x - 2| < 2) → (x^2 - 2*x < 0 ↔ |x-2| < 2) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l283_28321


namespace NUMINAMATH_GPT_solve_equation_l283_28333

theorem solve_equation (x : ℝ) (h : x ≠ 1) : -x^2 = (2 * x + 4) / (x - 1) → (x = -2 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l283_28333


namespace NUMINAMATH_GPT_unique_n_value_l283_28346

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (h1 : 1 = d 1) (h2 : ∀ i, d i ≤ n) (h3 : ∀ i j, i < j → d i < d j) 
                       (h4 : d (n - 1) = n) (h5 : ∃ k, k ≥ 4 ∧ ∀ i ≤ k, d i ∣ n)
                       (h6 : ∃ d1 d2 d3 d4, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ n = d1^2 + d2^2 + d3^2 + d4^2) : 
                       n = 130 := sorry

end NUMINAMATH_GPT_unique_n_value_l283_28346


namespace NUMINAMATH_GPT_highest_temperature_l283_28390

theorem highest_temperature
  (initial_temp : ℝ := 60)
  (final_temp : ℝ := 170)
  (heating_rate : ℝ := 5)
  (cooling_rate : ℝ := 7)
  (total_time : ℝ := 46) :
  ∃ T : ℝ, (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time ∧ T = 240 :=
by
  sorry

end NUMINAMATH_GPT_highest_temperature_l283_28390


namespace NUMINAMATH_GPT_total_earnings_from_selling_working_games_l283_28305

-- Conditions definition
def total_games : ℕ := 16
def broken_games : ℕ := 8
def working_games : ℕ := total_games - broken_games
def game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

-- Proof problem statement
theorem total_earnings_from_selling_working_games : List.sum game_prices = 68 := by
  sorry

end NUMINAMATH_GPT_total_earnings_from_selling_working_games_l283_28305


namespace NUMINAMATH_GPT_acute_triangle_l283_28397

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ 0 < A ∧ 0 < B ∧ 0 < C

def each_angle_less_than_sum_of_others (A B C : ℝ) : Prop :=
  A < B + C ∧ B < A + C ∧ C < A + B

theorem acute_triangle (A B C : ℝ) 
  (h1 : is_triangle A B C) 
  (h2 : each_angle_less_than_sum_of_others A B C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := 
sorry

end NUMINAMATH_GPT_acute_triangle_l283_28397


namespace NUMINAMATH_GPT_michael_earnings_l283_28389

-- Define variables for pay rates and hours.
def regular_pay_rate : ℝ := 7.00
def overtime_multiplier : ℝ := 2
def regular_hours : ℝ := 40
def overtime_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours

-- Define the earnings functions.
def regular_earnings (hourly_rate : ℝ) (hours : ℝ) : ℝ := hourly_rate * hours
def overtime_earnings (hourly_rate : ℝ) (multiplier : ℝ) (hours : ℝ) : ℝ := hourly_rate * multiplier * hours

-- Total earnings calculation.
def total_earnings (total_hours : ℝ) : ℝ := 
regular_earnings regular_pay_rate regular_hours + 
overtime_earnings regular_pay_rate overtime_multiplier (overtime_hours total_hours)

-- The theorem to prove the correct earnings for 42.857142857142854 hours worked.
theorem michael_earnings : total_earnings 42.857142857142854 = 320 := by
  sorry

end NUMINAMATH_GPT_michael_earnings_l283_28389


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l283_28320

theorem solve_eq1 (x : ℝ) : (x^2 - 2 * x - 8 = 0) ↔ (x = 4 ∨ x = -2) :=
sorry

theorem solve_eq2 (x : ℝ) : (2 * x^2 - 4 * x + 1 = 0) ↔ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l283_28320


namespace NUMINAMATH_GPT_cat_total_birds_caught_l283_28387

theorem cat_total_birds_caught (day_birds night_birds : ℕ) 
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) :
  day_birds + night_birds = 24 :=
sorry

end NUMINAMATH_GPT_cat_total_birds_caught_l283_28387


namespace NUMINAMATH_GPT_cupric_cyanide_formation_l283_28399

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cupric_cyanide_formation_l283_28399


namespace NUMINAMATH_GPT_triangle_square_ratio_l283_28319

theorem triangle_square_ratio (s_t s_s : ℕ) (h : 3 * s_t = 4 * s_s) : (s_t : ℚ) / s_s = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_triangle_square_ratio_l283_28319


namespace NUMINAMATH_GPT_overall_gain_loss_percent_zero_l283_28370

theorem overall_gain_loss_percent_zero (CP_A CP_B CP_C SP_A SP_B SP_C : ℝ)
  (h1 : CP_A = 600) (h2 : CP_B = 700) (h3 : CP_C = 800)
  (h4 : SP_A = 450) (h5 : SP_B = 750) (h6 : SP_C = 900) :
  ((SP_A + SP_B + SP_C) - (CP_A + CP_B + CP_C)) / (CP_A + CP_B + CP_C) * 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_overall_gain_loss_percent_zero_l283_28370


namespace NUMINAMATH_GPT_find_roots_combination_l283_28373

theorem find_roots_combination 
  (α β : ℝ)
  (hα : α^2 - 3 * α + 1 = 0)
  (hβ : β^2 - 3 * β + 1 = 0) :
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end NUMINAMATH_GPT_find_roots_combination_l283_28373


namespace NUMINAMATH_GPT_part1_part2_l283_28398

open Set

def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

def M : Set ℝ := { x | f x > 0 }

theorem part1 :
  M = { x | - (1 / 3 : ℝ) < x ∧ x < 3 } :=
sorry

theorem part2 :
  ∀ (x y : ℝ), x ∈ M → y ∈ M → abs (x + y + x * y) < 15 :=
sorry

end NUMINAMATH_GPT_part1_part2_l283_28398


namespace NUMINAMATH_GPT_emails_difference_l283_28374

theorem emails_difference
  (emails_morning : ℕ)
  (emails_afternoon : ℕ)
  (h_morning : emails_morning = 10)
  (h_afternoon : emails_afternoon = 3)
  : emails_morning - emails_afternoon = 7 := by
  sorry

end NUMINAMATH_GPT_emails_difference_l283_28374


namespace NUMINAMATH_GPT_find_k_l283_28385

theorem find_k (d : ℤ) (h : d ≠ 0) (a : ℤ → ℤ) 
  (a_def : ∀ n, a n = 4 * d + (n - 1) * d) 
  (geom_mean_condition : ∃ k, a k * a k = a 1 * a 6) : 
  ∃ k, k = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l283_28385


namespace NUMINAMATH_GPT_solve_for_n_l283_28349

theorem solve_for_n (n : ℤ) : (3 : ℝ)^(2 * n + 2) = 1 / 9 ↔ n = -2 := by
  sorry

end NUMINAMATH_GPT_solve_for_n_l283_28349


namespace NUMINAMATH_GPT_half_angle_in_second_and_fourth_quadrants_l283_28308

theorem half_angle_in_second_and_fourth_quadrants
  (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  (∃ m : ℤ, m * π + π / 2 < α / 2 ∧ α / 2 < m * π + 3 * π / 4) :=
by sorry

end NUMINAMATH_GPT_half_angle_in_second_and_fourth_quadrants_l283_28308


namespace NUMINAMATH_GPT_locus_of_tangency_centers_l283_28384

def locus_of_centers (a b : ℝ) : Prop := 8 * a ^ 2 + 9 * b ^ 2 - 16 * a - 64 = 0

theorem locus_of_tangency_centers (a b : ℝ)
  (hx1 : ∃ x y : ℝ, x ^ 2 + y ^ 2 = 1) 
  (hx2 : ∃ x y : ℝ, (x - 2) ^ 2 + y ^ 2 = 25) 
  (hcent : ∃ r : ℝ, a^2 + b^2 = (r + 1)^2 ∧ (a - 2)^2 + b^2 = (5 - r)^2) : 
  locus_of_centers a b :=
sorry

end NUMINAMATH_GPT_locus_of_tangency_centers_l283_28384


namespace NUMINAMATH_GPT_sale_in_2nd_month_l283_28368

-- Defining the variables for the sales in the months
def sale_in_1st_month : ℝ := 6435
def sale_in_3rd_month : ℝ := 7230
def sale_in_4th_month : ℝ := 6562
def sale_in_5th_month : ℝ := 6855
def required_sale_in_6th_month : ℝ := 5591
def required_average_sale : ℝ := 6600
def number_of_months : ℝ := 6
def total_sales_needed : ℝ := required_average_sale * number_of_months

-- Proof statement
theorem sale_in_2nd_month : sale_in_1st_month + x + sale_in_3rd_month + sale_in_4th_month + sale_in_5th_month + required_sale_in_6th_month = total_sales_needed → x = 6927 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_2nd_month_l283_28368


namespace NUMINAMATH_GPT_ratio_calculation_l283_28300

theorem ratio_calculation (A B C : ℚ)
  (h_ratio : (A / B = 3 / 2) ∧ (B / C = 2 / 5)) :
  (4 * A + 3 * B) / (5 * C - 2 * B) = 15 / 23 := by
  sorry

end NUMINAMATH_GPT_ratio_calculation_l283_28300


namespace NUMINAMATH_GPT_calculation_of_expression_l283_28325

theorem calculation_of_expression :
  (1.99 ^ 2 - 1.98 * 1.99 + 0.99 ^ 2) = 1 := 
by sorry

end NUMINAMATH_GPT_calculation_of_expression_l283_28325


namespace NUMINAMATH_GPT_complement_and_intersection_l283_28377

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {0, 1, 2}

theorem complement_and_intersection :
  ((U \ A) ∩ B) = {1, 2} := 
by
  sorry

end NUMINAMATH_GPT_complement_and_intersection_l283_28377


namespace NUMINAMATH_GPT_exists_k_tastrophic_function_l283_28360

noncomputable def k_tastrophic (f : ℕ+ → ℕ+) (k : ℕ) (n : ℕ+) : Prop :=
(f^[k] n) = n^k

theorem exists_k_tastrophic_function (k : ℕ) (h : k > 1) : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, k_tastrophic f k n :=
by sorry

end NUMINAMATH_GPT_exists_k_tastrophic_function_l283_28360


namespace NUMINAMATH_GPT_probability_of_red_light_l283_28395

-- Definitions based on the conditions
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Statement of the problem to prove the probability of seeing red light
theorem probability_of_red_light : (red_duration : ℚ) / total_cycle_time = 2 / 5 := 
by sorry

end NUMINAMATH_GPT_probability_of_red_light_l283_28395


namespace NUMINAMATH_GPT_bees_population_reduction_l283_28307

theorem bees_population_reduction :
  ∀ (initial_population loss_per_day : ℕ),
  initial_population = 80000 → 
  loss_per_day = 1200 → 
  ∃ days : ℕ, initial_population - days * loss_per_day = initial_population / 4 ∧ days = 50 :=
by
  intros initial_population loss_per_day h_initial h_loss
  use 50
  sorry

end NUMINAMATH_GPT_bees_population_reduction_l283_28307


namespace NUMINAMATH_GPT_quadratic_root_exists_l283_28326

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_exists_l283_28326


namespace NUMINAMATH_GPT_cricketer_average_after_19_innings_l283_28315

theorem cricketer_average_after_19_innings
  (A : ℝ) 
  (total_runs_after_18 : ℝ := 18 * A) 
  (runs_in_19th : ℝ := 99) 
  (new_avg : ℝ := A + 4) 
  (total_runs_after_19 : ℝ := total_runs_after_18 + runs_in_19th) 
  (equation : 19 * new_avg = total_runs_after_19) : 
  new_avg = 27 :=
by
  sorry

end NUMINAMATH_GPT_cricketer_average_after_19_innings_l283_28315


namespace NUMINAMATH_GPT_peter_speed_l283_28309

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end NUMINAMATH_GPT_peter_speed_l283_28309


namespace NUMINAMATH_GPT_water_left_l283_28376

theorem water_left (initial_water: ℚ) (science_experiment_use: ℚ) (plant_watering_use: ℚ)
  (h1: initial_water = 3)
  (h2: science_experiment_use = 5 / 4)
  (h3: plant_watering_use = 1 / 2) :
  (initial_water - science_experiment_use - plant_watering_use = 5 / 4) :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_water_left_l283_28376


namespace NUMINAMATH_GPT_area_increase_by_nine_l283_28354

theorem area_increase_by_nine (a : ℝ) :
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := extended_side_length^2;
  extended_area / original_area = 9 :=
by
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := (extended_side_length)^2;
  sorry

end NUMINAMATH_GPT_area_increase_by_nine_l283_28354


namespace NUMINAMATH_GPT_Hallie_earnings_l283_28361

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end NUMINAMATH_GPT_Hallie_earnings_l283_28361


namespace NUMINAMATH_GPT_sum_of_two_integers_l283_28365

theorem sum_of_two_integers (x y : ℝ) (h₁ : x^2 + y^2 = 130) (h₂ : x * y = 45) : x + y = 2 * Real.sqrt 55 :=
sorry

end NUMINAMATH_GPT_sum_of_two_integers_l283_28365


namespace NUMINAMATH_GPT_find_g_values_l283_28335

variables (f g : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y, g (x - y) = g x * g y + f x * f y
axiom cond2 : f (-1) = -1
axiom cond3 : f 0 = 0
axiom cond4 : f 1 = 1

-- Goal
theorem find_g_values : g 0 = 1 ∧ g 1 = 0 ∧ g 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_g_values_l283_28335


namespace NUMINAMATH_GPT_parabola_vertex_l283_28327

theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, t^2 + 2 * t - 2 ≥ y) ∧ (x^2 + 2 * x - 2 = y) ∧ (x = -1) ∧ (y = -3) :=
by sorry

end NUMINAMATH_GPT_parabola_vertex_l283_28327


namespace NUMINAMATH_GPT_A_is_false_l283_28332

variables {a b : ℝ}

-- Condition: Proposition B - The sum of the roots of the equation is 2
axiom sum_of_roots : ∀ (x1 x2 : ℝ), x1 + x2 = -a

-- Condition: Proposition C - x = 3 is a root of the equation
axiom root3 : ∃ (x1 x2 : ℝ), (x1 = 3 ∨ x2 = 3)

-- Condition: Proposition D - The two roots have opposite signs
axiom opposite_sign_roots : ∀ (x1 x2 : ℝ), x1 * x2 < 0

-- Prove: Proposition A is false
theorem A_is_false : ¬ (∃ x1 x2 : ℝ, x1 = 1 ∨ x2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_A_is_false_l283_28332


namespace NUMINAMATH_GPT_exists_xy_nat_divisible_l283_28364

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end NUMINAMATH_GPT_exists_xy_nat_divisible_l283_28364


namespace NUMINAMATH_GPT_complex_subtraction_l283_28334

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * I) (h2 : z2 = 3 + I) :
  z1 - z2 = -1 + 2 * I := 
by
  sorry

end NUMINAMATH_GPT_complex_subtraction_l283_28334


namespace NUMINAMATH_GPT_rachel_picked_total_apples_l283_28351

-- Define the conditions
def num_trees : ℕ := 4
def apples_per_tree_picked : ℕ := 7
def apples_remaining : ℕ := 29

-- Define the total apples picked
def total_apples_picked : ℕ := num_trees * apples_per_tree_picked

-- Formal statement of the goal
theorem rachel_picked_total_apples : total_apples_picked = 28 := 
by
  sorry

end NUMINAMATH_GPT_rachel_picked_total_apples_l283_28351


namespace NUMINAMATH_GPT_black_squares_covered_by_trominoes_l283_28336

theorem black_squares_covered_by_trominoes (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (k : ℕ), k * k = (n + 1) / 2 ∧ n ≥ 7) ↔ n ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_black_squares_covered_by_trominoes_l283_28336


namespace NUMINAMATH_GPT_total_apples_collected_l283_28314

-- Definitions based on conditions
def number_of_green_apples : ℕ := 124
def number_of_red_apples : ℕ := 3 * number_of_green_apples

-- Proof statement
theorem total_apples_collected : number_of_red_apples + number_of_green_apples = 496 := by
  sorry

end NUMINAMATH_GPT_total_apples_collected_l283_28314


namespace NUMINAMATH_GPT_correct_negation_of_p_l283_28358

open Real

def proposition_p (x : ℝ) := x > 0 → sin x ≥ -1

theorem correct_negation_of_p :
  ¬ (∀ x, proposition_p x) ↔ (∃ x, x > 0 ∧ sin x < -1) :=
by
  sorry

end NUMINAMATH_GPT_correct_negation_of_p_l283_28358


namespace NUMINAMATH_GPT_solve_equation_l283_28301

-- Defining the original equation as a Lean function
def equation (x : ℝ) : Prop :=
  (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2))

theorem solve_equation :
  ∃ x : ℝ, equation x ∧ x = -13 / 2 :=
by
  -- Equation specification and transformations
  sorry

end NUMINAMATH_GPT_solve_equation_l283_28301


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l283_28394

noncomputable def perimeter_of_isosceles_triangle : ℝ :=
  let BC := 10
  let height := 6
  let half_base := BC / 2
  let side := Real.sqrt (height^2 + half_base^2)
  let perimeter := 2 * side + BC
  perimeter

theorem isosceles_triangle_perimeter :
  let BC := 10
  let height := 6
  perimeter_of_isosceles_triangle = 2 * Real.sqrt (height^2 + (BC / 2)^2) + BC := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l283_28394


namespace NUMINAMATH_GPT_triangles_with_two_colors_l283_28380

theorem triangles_with_two_colors {n : ℕ} 
  (h1 : ∀ (p : Finset ℝ) (hn : p.card = n) 
      (e : p → p → Prop), 
      (∀ (x y : p), e x y → e x y = red ∨ e x y = yellow ∨ e x y = green) /\
      (∀ (a b c : p), 
        (e a b = red ∨ e a b = yellow ∨ e a b = green) ∧ 
        (e b c = red ∨ e b c = yellow ∨ e b c = green) ∧ 
        (e a c = red ∨ e a c = yellow ∨ e a c = green) → 
        (e a b ≠ e b c ∨ e b c ≠ e a c ∨ e a b ≠ e a c))) :
  n < 13 := 
sorry

end NUMINAMATH_GPT_triangles_with_two_colors_l283_28380


namespace NUMINAMATH_GPT_product_of_p_r_s_l283_28363

theorem product_of_p_r_s :
  ∃ p r s : ℕ, 3^p + 3^5 = 252 ∧ 2^r + 58 = 122 ∧ 5^3 * 6^s = 117000 ∧ p * r * s = 36 :=
by
  sorry

end NUMINAMATH_GPT_product_of_p_r_s_l283_28363


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l283_28330

theorem molecular_weight_of_one_mole 
  (molicular_weight_9_moles : ℕ) 
  (weight_9_moles : ℕ)
  (h : molicular_weight_9_moles = 972 ∧ weight_9_moles = 9) : 
  molicular_weight_9_moles / weight_9_moles = 108 := 
  by
    sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l283_28330


namespace NUMINAMATH_GPT_original_avg_expenditure_correct_l283_28396

variables (A B C a b c X Y Z : ℝ)
variables (hA : A > 0) (hB : B > 0) (hC : C > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem original_avg_expenditure_correct
    (h_orig_exp : (A * X + B * Y + C * Z) / (A + B + C) - 1 
    = ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42):
    True := 
sorry

end NUMINAMATH_GPT_original_avg_expenditure_correct_l283_28396


namespace NUMINAMATH_GPT_circle_areas_sum_l283_28302

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end NUMINAMATH_GPT_circle_areas_sum_l283_28302


namespace NUMINAMATH_GPT_ribbon_per_gift_l283_28357

-- Definitions for the conditions in the problem
def total_ribbon_used : ℚ := 4/15
def num_gifts: ℕ := 5

-- Statement to prove
theorem ribbon_per_gift : total_ribbon_used / num_gifts = 4 / 75 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_per_gift_l283_28357


namespace NUMINAMATH_GPT_max_value_of_expression_l283_28343

noncomputable def max_expression_value (x y : ℝ) :=
  x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  max_expression_value x y ≤ 961 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l283_28343


namespace NUMINAMATH_GPT_sides_of_polygon_l283_28367

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_sides_of_polygon_l283_28367


namespace NUMINAMATH_GPT_rectangle_area_function_relationship_l283_28353

theorem rectangle_area_function_relationship (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_function_relationship_l283_28353


namespace NUMINAMATH_GPT_cat_food_inequality_l283_28345

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end NUMINAMATH_GPT_cat_food_inequality_l283_28345


namespace NUMINAMATH_GPT_annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l283_28339

-- Define principal amounts for Paul, Emma and Harry
def principalPaul : ℚ := 5000
def principalEmma : ℚ := 3000
def principalHarry : ℚ := 7000

-- Define time periods for Paul, Emma and Harry
def timePaul : ℚ := 2
def timeEmma : ℚ := 4
def timeHarry : ℚ := 3

-- Define interests received from Paul, Emma and Harry
def interestPaul : ℚ := 2200
def interestEmma : ℚ := 3400
def interestHarry : ℚ := 3900

-- Define the simple interest formula 
def simpleInterest (P : ℚ) (R : ℚ) (T : ℚ) : ℚ := P * R * T

-- Prove the annual interest rates for each loan 
theorem annual_interest_rate_Paul : 
  ∃ (R : ℚ), simpleInterest principalPaul R timePaul = interestPaul ∧ R = 0.22 := 
by
  sorry

theorem annual_interest_rate_Emma : 
  ∃ (R : ℚ), simpleInterest principalEmma R timeEmma = interestEmma ∧ R = 0.2833 := 
by
  sorry

theorem annual_interest_rate_Harry : 
  ∃ (R : ℚ), simpleInterest principalHarry R timeHarry = interestHarry ∧ R = 0.1857 := 
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l283_28339


namespace NUMINAMATH_GPT_percent_not_filler_l283_28356

theorem percent_not_filler (total_weight filler_weight : ℕ) (h1 : total_weight = 180) (h2 : filler_weight = 45) : 
  ((total_weight - filler_weight) * 100 / total_weight = 75) :=
by 
  sorry

end NUMINAMATH_GPT_percent_not_filler_l283_28356


namespace NUMINAMATH_GPT_inequality_inequality_l283_28331

theorem inequality_inequality (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) :
  ac + bd ≤ 8 :=
sorry

end NUMINAMATH_GPT_inequality_inequality_l283_28331


namespace NUMINAMATH_GPT_smallest_positive_integer_solution_l283_28328

theorem smallest_positive_integer_solution : ∃ n : ℕ, 23 * n % 9 = 310 % 9 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_solution_l283_28328


namespace NUMINAMATH_GPT_meaningful_fraction_iff_l283_28306

theorem meaningful_fraction_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (2 - x)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_iff_l283_28306


namespace NUMINAMATH_GPT_sequence_properties_l283_28366

/-- Theorem setup:
Assume a sequence {a_n} with a_1 = 1 and a_{n+1} = 2a_n / (a_n + 2)
Also, define b_n = 1 / a_n
-/
theorem sequence_properties 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  -- Prove that {b_n} (b n = 1 / a n) is arithmetic with common difference 1/2
  (∃ b : ℕ → ℝ, (∀ n : ℕ, b n = 1 / a n) ∧ (∀ n : ℕ, b (n + 1) = b n + 1 / 2)) ∧ 
  -- Prove the general formula for a_n
  (∀ n : ℕ, a (n + 1) = 2 / (n + 1)) := 
sorry


end NUMINAMATH_GPT_sequence_properties_l283_28366


namespace NUMINAMATH_GPT_rowing_upstream_speed_l283_28371

-- Definitions based on conditions
def V_m : ℝ := 45 -- speed of the man in still water
def V_downstream : ℝ := 53 -- speed of the man rowing downstream
def V_s : ℝ := V_downstream - V_m -- speed of the stream
def V_upstream : ℝ := V_m - V_s -- speed of the man rowing upstream

-- The goal is to prove that the speed of the man rowing upstream is 37 kmph
theorem rowing_upstream_speed :
  V_upstream = 37 := by
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l283_28371


namespace NUMINAMATH_GPT_field_area_l283_28312

theorem field_area
  (L : ℕ) (W : ℕ) (A : ℕ)
  (h₁ : L = 20)
  (h₂ : 2 * W + L = 100)
  (h₃ : A = L * W) :
  A = 800 := by
  sorry

end NUMINAMATH_GPT_field_area_l283_28312


namespace NUMINAMATH_GPT_triangle_altitude_from_equal_area_l283_28372

variable (x : ℝ)

theorem triangle_altitude_from_equal_area (h : x^2 = (1 / 2) * x * altitude) :
  altitude = 2 * x := by
  sorry

end NUMINAMATH_GPT_triangle_altitude_from_equal_area_l283_28372


namespace NUMINAMATH_GPT_maximum_unique_walks_l283_28316

-- Define the conditions
def starts_at_A : Prop := true
def crosses_bridge_1_first : Prop := true
def finishes_at_B : Prop := true
def six_bridges_linking_two_islands_and_banks : Prop := true

-- Define the theorem to prove the maximum number of unique walks is 6
theorem maximum_unique_walks : starts_at_A ∧ crosses_bridge_1_first ∧ finishes_at_B ∧ six_bridges_linking_two_islands_and_banks → ∃ n, n = 6 :=
by
  intros
  existsi 6
  sorry

end NUMINAMATH_GPT_maximum_unique_walks_l283_28316


namespace NUMINAMATH_GPT_find_x_l283_28341

theorem find_x (x : ℝ) (h: 0.8 * 90 = 70 / 100 * x + 30) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l283_28341


namespace NUMINAMATH_GPT_total_marbles_l283_28359

theorem total_marbles (r b g : ℕ) (total : ℕ) 
  (h_ratio : 2 * g = 4 * b) 
  (h_blue_marbles : b = 36) 
  (h_total_formula : total = r + b + g) 
  : total = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l283_28359


namespace NUMINAMATH_GPT_greatest_common_divisor_of_120_and_m_l283_28362

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_of_120_and_m_l283_28362


namespace NUMINAMATH_GPT_part_a_part_b_l283_28329

variable {A : Type} [Ring A] (h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6)

-- Part (a)
theorem part_a (x : A) (n : Nat) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 :=
sorry

-- Part (b)
theorem part_b (x : A) : x^4 = x :=
by
  have h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6 := h
  sorry

end NUMINAMATH_GPT_part_a_part_b_l283_28329


namespace NUMINAMATH_GPT_max_value_of_quadratic_at_2_l283_28378

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

theorem max_value_of_quadratic_at_2 : ∃ (x : ℝ), x = 2 ∧ ∀ y : ℝ, f y ≤ f x :=
by
  use 2
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_at_2_l283_28378


namespace NUMINAMATH_GPT_choir_members_correct_l283_28318

noncomputable def choir_membership : ℕ :=
  let n := 226
  n

theorem choir_members_correct (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
by
  sorry

end NUMINAMATH_GPT_choir_members_correct_l283_28318


namespace NUMINAMATH_GPT_scientific_notation_of_distance_l283_28344

theorem scientific_notation_of_distance :
  ∃ a n, (1 ≤ a ∧ a < 10) ∧ 384000 = a * 10^n ∧ a = 3.84 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_distance_l283_28344


namespace NUMINAMATH_GPT_milk_cost_is_3_l283_28348

def Banana_cost : ℝ := 2
def Sales_tax_rate : ℝ := 0.20
def Total_spent : ℝ := 6

theorem milk_cost_is_3 (Milk_cost : ℝ) :
  Total_spent = (Milk_cost + Banana_cost) + Sales_tax_rate * (Milk_cost + Banana_cost) → 
  Milk_cost = 3 :=
by
  simp [Banana_cost, Sales_tax_rate, Total_spent]
  sorry

end NUMINAMATH_GPT_milk_cost_is_3_l283_28348


namespace NUMINAMATH_GPT_smallest_non_factor_product_of_48_l283_28340

theorem smallest_non_factor_product_of_48 :
  ∃ (x y : ℕ), x ≠ y ∧ x * y ≤ 48 ∧ (x ∣ 48) ∧ (y ∣ 48) ∧ ¬ (x * y ∣ 48) ∧ x * y = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_non_factor_product_of_48_l283_28340


namespace NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l283_28303

open Set

-- Definitions
def U : Set ℤ := {-1, 1, 3}
def A : Set ℤ := {-1}

-- Theorem statement
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l283_28303


namespace NUMINAMATH_GPT_Uncle_Fyodor_age_l283_28388

variable (age : ℕ)

-- Conditions from the problem
def Sharik_statement : Prop := age > 11
def Matroskin_statement : Prop := age > 10

-- The theorem stating the problem to be proved
theorem Uncle_Fyodor_age
  (H : (Sharik_statement age ∧ ¬Matroskin_statement age) ∨ (¬Sharik_statement age ∧ Matroskin_statement age)) :
  age = 11 :=
by
  sorry

end NUMINAMATH_GPT_Uncle_Fyodor_age_l283_28388


namespace NUMINAMATH_GPT_graph_comparison_l283_28338

theorem graph_comparison :
  (∀ x : ℝ, (x^2 - x + 3) < (x^2 - x + 5)) :=
by
  sorry

end NUMINAMATH_GPT_graph_comparison_l283_28338


namespace NUMINAMATH_GPT_meaningful_expr_iff_x_ne_neg_5_l283_28375

theorem meaningful_expr_iff_x_ne_neg_5 (x : ℝ) : (x + 5 ≠ 0) ↔ (x ≠ -5) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expr_iff_x_ne_neg_5_l283_28375


namespace NUMINAMATH_GPT_total_number_of_animals_l283_28304

-- Define the data and conditions
def total_legs : ℕ := 38
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the proof problem
theorem total_number_of_animals (h1 : total_legs = 38) 
                                (h2 : chickens = 5) 
                                (h3 : chicken_legs = 2) 
                                (h4 : sheep_legs = 4) : 
  (∃ sheep : ℕ, chickens + sheep = 12) :=
by 
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l283_28304


namespace NUMINAMATH_GPT_lcm_24_36_40_l283_28324

-- Define the natural numbers 24, 36, and 40
def n1 : ℕ := 24
def n2 : ℕ := 36
def n3 : ℕ := 40

-- Define the prime factorization of each number
def factors_n1 := [2^3, 3^1] -- 24 = 2^3 * 3^1
def factors_n2 := [2^2, 3^2] -- 36 = 2^2 * 3^2
def factors_n3 := [2^3, 5^1] -- 40 = 2^3 * 5^1

-- Prove that the LCM of n1, n2, n3 is 360
theorem lcm_24_36_40 : Nat.lcm (Nat.lcm n1 n2) n3 = 360 := sorry

end NUMINAMATH_GPT_lcm_24_36_40_l283_28324


namespace NUMINAMATH_GPT_tan_add_sin_l283_28379

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end NUMINAMATH_GPT_tan_add_sin_l283_28379


namespace NUMINAMATH_GPT_percentage_difference_l283_28322

theorem percentage_difference (X : ℝ) (h1 : first_num = 0.70 * X) (h2 : second_num = 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l283_28322


namespace NUMINAMATH_GPT_fraction_identity_l283_28382

theorem fraction_identity (x y z v : ℝ) (hy : y ≠ 0) (hv : v ≠ 0)
    (h : x / y + z / v = 1) : x / y - z / v = (x / y) ^ 2 - (z / v) ^ 2 := by
  sorry

end NUMINAMATH_GPT_fraction_identity_l283_28382


namespace NUMINAMATH_GPT_problem_b_is_proposition_l283_28386

def is_proposition (s : String) : Prop :=
  s = "sin 45° = 1" ∨ s = "x^2 + 2x - 1 > 0"

theorem problem_b_is_proposition : is_proposition "sin 45° = 1" :=
by
  -- insert proof steps to establish that "sin 45° = 1" is a proposition
  sorry

end NUMINAMATH_GPT_problem_b_is_proposition_l283_28386


namespace NUMINAMATH_GPT_solve_inequality_l283_28392

theorem solve_inequality (x : ℝ) : (1 + x) / 3 < x / 2 → x > 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l283_28392


namespace NUMINAMATH_GPT_digit_swap_division_l283_28393

theorem digit_swap_division (ab ba : ℕ) (k1 k2 : ℤ) (a b : ℕ) :
  (ab = 10 * a + b) ∧ (ba = 10 * b + a) →
  (ab % 7 = 1) ∧ (ba % 7 = 1) →
  ∃ n, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_swap_division_l283_28393


namespace NUMINAMATH_GPT_chosen_number_l283_28391

theorem chosen_number (x: ℤ) (h: 2 * x - 152 = 102) : x = 127 :=
by
  sorry

end NUMINAMATH_GPT_chosen_number_l283_28391


namespace NUMINAMATH_GPT_student_total_marks_l283_28347

theorem student_total_marks (total_questions correct_answers incorrect_answer_score correct_answer_score : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_answers = 38)
    (h3 : correct_answer_score = 4)
    (h4 : incorrect_answer_score = 1)
    (incorrect_answers := total_questions - correct_answers) 
    : (correct_answers * correct_answer_score - incorrect_answers * incorrect_answer_score) = 130 :=
by
  -- proof to be provided here
  sorry

end NUMINAMATH_GPT_student_total_marks_l283_28347


namespace NUMINAMATH_GPT_cubic_roots_natural_numbers_l283_28337

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end NUMINAMATH_GPT_cubic_roots_natural_numbers_l283_28337


namespace NUMINAMATH_GPT_average_and_variance_of_original_data_l283_28350

theorem average_and_variance_of_original_data (μ σ_sq : ℝ)
  (h1 : 2 * μ - 80 = 1.2)
  (h2 : 4 * σ_sq = 4.4) :
  μ = 40.6 ∧ σ_sq = 1.1 :=
by
  sorry

end NUMINAMATH_GPT_average_and_variance_of_original_data_l283_28350


namespace NUMINAMATH_GPT_angle_C_eq_pi_div_3_find_ab_values_l283_28323

noncomputable def find_angle_C (A B C : ℝ) (a b c : ℝ) : ℝ :=
  if c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C then C else 0

noncomputable def find_sides_ab (A B C : ℝ) (c S : ℝ) : Set (ℝ × ℝ) :=
  if C = Real.pi / 3 ∧ c = 2 * Real.sqrt 3 ∧ S = 2 * Real.sqrt 3 then
    { (a, b) | a^4 - 20 * a^2 + 64 = 0 ∧ b = 8 / a } else
    ∅

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (a b c : ℝ) :
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C)
  ↔ (C = Real.pi / 3) :=
sorry

theorem find_ab_values (A B C : ℝ) (c S a b : ℝ) :
  (C = Real.pi / 3) ∧ (c = 2 * Real.sqrt 3) ∧ (S = 2 * Real.sqrt 3) ∧ (a^4 - 20 * a^2 + 64 = 0) ∧ (b = 8 / a)
  ↔ ((a, b) = (2, 4) ∨ (a, b) = (4, 2)) :=
sorry

end NUMINAMATH_GPT_angle_C_eq_pi_div_3_find_ab_values_l283_28323


namespace NUMINAMATH_GPT_sum_of_distances_from_circumcenter_to_sides_l283_28313

theorem sum_of_distances_from_circumcenter_to_sides :
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let a := r1 + r2
  let b := r1 + r3
  let c := r2 + r3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r_incircle := area / s
  r_incircle = Real.sqrt 7 →
  let sum_distances := (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
  sum_distances = (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
:= sorry

end NUMINAMATH_GPT_sum_of_distances_from_circumcenter_to_sides_l283_28313


namespace NUMINAMATH_GPT_johnson_and_martinez_tied_at_may_l283_28381

def home_runs_johnson (m : String) : ℕ :=
  if m = "January" then 2 else
  if m = "February" then 12 else
  if m = "March" then 20 else
  if m = "April" then 15 else
  if m = "May" then 9 else 0

def home_runs_martinez (m : String) : ℕ :=
  if m = "January" then 5 else
  if m = "February" then 9 else
  if m = "March" then 15 else
  if m = "April" then 20 else
  if m = "May" then 9 else 0

def cumulative_home_runs (player_home_runs : String → ℕ) (months : List String) : ℕ :=
  months.foldl (λ acc m => acc + player_home_runs m) 0

def months_up_to_may : List String :=
  ["January", "February", "March", "April", "May"]

theorem johnson_and_martinez_tied_at_may :
  cumulative_home_runs home_runs_johnson months_up_to_may
  = cumulative_home_runs home_runs_martinez months_up_to_may :=
by
    sorry

end NUMINAMATH_GPT_johnson_and_martinez_tied_at_may_l283_28381
