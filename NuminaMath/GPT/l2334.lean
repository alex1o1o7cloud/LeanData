import Mathlib

namespace range_of_m_l2334_233421

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + Real.log x + 3)) :
  ∃ m, m ∈ Set.Icc (1 / (2 * Real.exp 1)) (1 + Real.log 3 / 6) :=
sorry

end range_of_m_l2334_233421


namespace difference_between_advertised_and_actual_mileage_l2334_233466

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_l2334_233466


namespace find_ages_l2334_233411

theorem find_ages (P J G : ℕ)
  (h1 : P - 10 = 1 / 3 * (J - 10))
  (h2 : J = P + 12)
  (h3 : G = 1 / 2 * (P + J)) :
  P = 16 ∧ G = 22 :=
by
  sorry

end find_ages_l2334_233411


namespace fraction_expression_l2334_233472

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end fraction_expression_l2334_233472


namespace maximum_value_expression_l2334_233444

-- Definitions
def f (x : ℝ) := -3 * x^2 + 18 * x - 1

-- Lean statement to prove that the maximum value of the function f is 26.
theorem maximum_value_expression : ∃ x : ℝ, f x = 26 :=
sorry

end maximum_value_expression_l2334_233444


namespace find_cost_of_two_enchiladas_and_five_tacos_l2334_233410

noncomputable def cost_of_two_enchiladas_and_five_tacos (e t : ℝ) : ℝ :=
  2 * e + 5 * t

theorem find_cost_of_two_enchiladas_and_five_tacos (e t : ℝ):
  (e + 4 * t = 3.50) → (4 * e + t = 4.20) → cost_of_two_enchiladas_and_five_tacos e t = 5.04 :=
by
  intro h1 h2
  sorry

end find_cost_of_two_enchiladas_and_five_tacos_l2334_233410


namespace max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l2334_233439

open Real

theorem max_min_2sinx_minus_3 : 
  ∀ x : ℝ, 
    -5 ≤ 2 * sin x - 3 ∧ 
    2 * sin x - 3 ≤ -1 :=
by sorry

theorem max_min_7_fourth_sinx_minus_sinx_squared : 
  ∀ x : ℝ, 
    -1/4 ≤ (7/4 + sin x - sin x ^ 2) ∧ 
    (7/4 + sin x - sin x ^ 2) ≤ 2 :=
by sorry

end max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l2334_233439


namespace negation_of_proposition_l2334_233438

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0)) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_of_proposition_l2334_233438


namespace jason_gave_seashells_to_tim_l2334_233426

-- Defining the conditions
def original_seashells : ℕ := 49
def current_seashells : ℕ := 36

-- The proof statement
theorem jason_gave_seashells_to_tim :
  original_seashells - current_seashells = 13 :=
by
  sorry

end jason_gave_seashells_to_tim_l2334_233426


namespace solve_system_equations_l2334_233460

theorem solve_system_equations (x y : ℝ) :
  (5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8) ↔
  (x = -1 ∧ y = 2) ∨ (x = 11 ∧ y = -7) ∨ (x = -11 ∧ y = 7) ∨ (x = 1 ∧ y = -2) := 
sorry

end solve_system_equations_l2334_233460


namespace hyperbola_eccentricity_range_l2334_233452

-- Lean 4 statement for the given problem.
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (x y : ℝ), y = x * Real.sqrt 3 → y^2 / b^2 - x^2 / a^2 = 1 ∨ ∃ (z : ℝ), y = x * Real.sqrt 3 ∧ z^2 / b^2 - x^2 / a^2 = 1) :
  1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a < 2 :=
by
  sorry

end hyperbola_eccentricity_range_l2334_233452


namespace Anna_s_wear_size_l2334_233464

theorem Anna_s_wear_size
  (A : ℕ)
  (Becky_size : ℕ)
  (Ginger_size : ℕ)
  (h1 : Becky_size = 3 * A)
  (h2 : Ginger_size = 2 * Becky_size - 4)
  (h3 : Ginger_size = 8) :
  A = 2 :=
by
  sorry

end Anna_s_wear_size_l2334_233464


namespace find_n_l2334_233448

-- Define the values of quarters and dimes in cents
def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10

-- Define the number of quarters and dimes
def num_quarters : ℕ := 15
def num_dimes : ℕ := 25

-- Define the total value in cents corresponding to the quarters
def total_value_quarters : ℕ := num_quarters * value_of_quarter

-- Define the condition where total value by quarters equals total value by n dimes
def equivalent_dimes (n : ℕ) : Prop := total_value_quarters = n * value_of_dime

-- The theorem to prove
theorem find_n : ∃ n : ℕ, equivalent_dimes n ∧ n = 38 := 
by {
  use 38,
  sorry
}

end find_n_l2334_233448


namespace jesse_money_left_l2334_233489

def initial_money : ℝ := 500
def novel_cost_pounds : ℝ := 13
def num_novels : ℕ := 10
def bookstore_discount : ℝ := 0.20
def exchange_rate_usd_to_pounds : ℝ := 0.7
def lunch_cost_multiplier : ℝ := 3
def lunch_tax_rate : ℝ := 0.12
def lunch_tip_rate : ℝ := 0.18
def jacket_original_euros : ℝ := 120
def jacket_discount : ℝ := 0.30
def jacket_expense_multiplier : ℝ := 2
def exchange_rate_pounds_to_euros : ℝ := 1.15

theorem jesse_money_left : 
  initial_money - (
    ((novel_cost_pounds * num_novels * (1 - bookstore_discount)) / exchange_rate_usd_to_pounds)
    + ((novel_cost_pounds * lunch_cost_multiplier * (1 + lunch_tax_rate + lunch_tip_rate)) / exchange_rate_usd_to_pounds)
    + ((((jacket_original_euros * (1 - jacket_discount)) / exchange_rate_pounds_to_euros) / exchange_rate_usd_to_pounds))
  ) = 174.66 := by
  sorry

end jesse_money_left_l2334_233489


namespace min_ab_min_expr_min_a_b_l2334_233485

-- Define the conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hln : Real.log a + Real.log b = Real.log (a + 9 * b))

-- 1. The minimum value of ab
theorem min_ab : ab = 36 :=
sorry

-- 2. The minimum value of (81 / a^2) + (1 / b^2)
theorem min_expr : (81 / a^2) + (1 / b^2) = (1 / 2) :=
sorry

-- 3. The minimum value of a + b
theorem min_a_b : a + b = 16 :=
sorry

end min_ab_min_expr_min_a_b_l2334_233485


namespace isosceles_right_triangle_third_angle_l2334_233458

/-- In an isosceles right triangle where one of the angles opposite the equal sides measures 45 degrees, 
    the measure of the third angle is 90 degrees. -/
theorem isosceles_right_triangle_third_angle (θ : ℝ) 
  (h1 : θ = 45)
  (h2 : ∀ (a b c : ℝ), a + b + c = 180) : θ + θ + 90 = 180 :=
by
  sorry

end isosceles_right_triangle_third_angle_l2334_233458


namespace scooter_travel_time_l2334_233470

variable (x : ℝ)
variable (h_speed : x > 0)
variable (h_travel_time : (50 / (x - 1/2)) - (50 / x) = 3/4)

theorem scooter_travel_time : 50 / x = 50 / x := 
  sorry

end scooter_travel_time_l2334_233470


namespace n_equal_three_l2334_233404

variable (m n : ℝ)

-- Conditions
def in_second_quadrant (m n : ℝ) : Prop := m < 0 ∧ n > 0
def distance_to_x_axis_eq_three (n : ℝ) : Prop := abs n = 3

-- Proof problem statement
theorem n_equal_three 
  (h1 : in_second_quadrant m n) 
  (h2 : distance_to_x_axis_eq_three n) : 
  n = 3 := 
sorry

end n_equal_three_l2334_233404


namespace gain_percent_calculation_l2334_233401

def gain : ℝ := 0.70
def cost_price : ℝ := 70.0

theorem gain_percent_calculation : (gain / cost_price) * 100 = 1 := by
  sorry

end gain_percent_calculation_l2334_233401


namespace line_circle_no_intersection_l2334_233405

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l2334_233405


namespace lucky_sum_probability_eq_l2334_233414

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l2334_233414


namespace water_volume_correct_l2334_233473

-- Define the conditions
def ratio_water_juice : ℕ := 5
def ratio_juice_water : ℕ := 3
def total_punch_volume : ℚ := 3  -- in liters

-- Define the question and the correct answer
def volume_of_water (ratio_water_juice ratio_juice_water : ℕ) (total_punch_volume : ℚ) : ℚ :=
  (ratio_water_juice * total_punch_volume) / (ratio_water_juice + ratio_juice_water)

-- The proof problem
theorem water_volume_correct : volume_of_water ratio_water_juice ratio_juice_water total_punch_volume = 15 / 8 :=
by
  sorry

end water_volume_correct_l2334_233473


namespace problem_solved_by_half_participants_l2334_233463

variables (n m : ℕ)
variable (solve : ℕ → ℕ → Prop)  -- solve i j means participant i solved problem j

axiom half_n_problems_solved : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)

theorem problem_solved_by_half_participants (h : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)) : 
  ∃ j, j < n ∧ (∃ count, count ≥ m / 2 ∧ (∃ i, i < m → solve i j)) :=
  sorry

end problem_solved_by_half_participants_l2334_233463


namespace find_f_neg_19_div_3_l2334_233446

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then 
    8^x 
  else 
    sorry -- The full definition is complex and not needed for the statement

-- Define the properties of f
lemma f_periodic (x : ℝ) : f (x + 2) = f x := 
  sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := 
  sorry

theorem find_f_neg_19_div_3 : f (-19/3) = -2 :=
  sorry

end find_f_neg_19_div_3_l2334_233446


namespace right_handed_players_count_l2334_233425

theorem right_handed_players_count (total_players throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : left_handed_non_throwers = (total_players - throwers) / 3)
  (h4 : right_handed_non_throwers = total_players - throwers - left_handed_non_throwers)
  (h5 : ∀ n, n = throwers + right_handed_non_throwers) : 
  (throwers + right_handed_non_throwers) = 62 := 
by 
  sorry

end right_handed_players_count_l2334_233425


namespace total_winning_team_points_l2334_233459

/-!
# Lean 4 Math Proof Problem

Prove that the total points scored by the winning team at the end of the game is 50 points given the conditions provided.
-/

-- Definitions
def losing_team_points_first_quarter : ℕ := 10
def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter
def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + 10
def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

-- Theorem statement
theorem total_winning_team_points : winning_team_points_third_quarter = 50 :=
by
  sorry

end total_winning_team_points_l2334_233459


namespace sum_quotient_remainder_div9_l2334_233456

theorem sum_quotient_remainder_div9 (n : ℕ) (h₁ : n = 248 * 5 + 4) :
  let q := n / 9
  let r := n % 9
  q + r = 140 :=
by
  sorry

end sum_quotient_remainder_div9_l2334_233456


namespace sufficient_condition_l2334_233434

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem sufficient_condition (a : ℝ) : (∀ x y : ℝ, N x y a → M x y) ↔ (a ≥ 5 / 4) := 
sorry

end sufficient_condition_l2334_233434


namespace A_P_not_76_l2334_233407

theorem A_P_not_76 :
    ∀ (w : ℕ), w > 0 → (2 * w^2 + 6 * w) ≠ 76 :=
by
  intro w hw
  sorry

end A_P_not_76_l2334_233407


namespace number_of_white_balls_l2334_233432

-- Definition of the conditions
def total_balls : ℕ := 40
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45
def prob_white := 1 - prob_red - prob_black

-- The statement that needs to be proved
theorem number_of_white_balls : (total_balls : ℝ) * prob_white = 16 :=
by
  sorry

end number_of_white_balls_l2334_233432


namespace oranges_apples_bananas_equiv_l2334_233445

-- Define weights
variable (w_orange w_apple w_banana : ℝ)

-- Conditions
def condition1 : Prop := 9 * w_orange = 6 * w_apple
def condition2 : Prop := 4 * w_banana = 3 * w_apple

-- Main problem
theorem oranges_apples_bananas_equiv :
  ∀ (w_orange w_apple w_banana : ℝ),
  (9 * w_orange = 6 * w_apple) →
  (4 * w_banana = 3 * w_apple) →
  ∃ (a b : ℕ), a = 17 ∧ b = 13 ∧ (a + 3/4 * b = (45/9) * 6) :=
by
  intros w_orange w_apple w_banana h1 h2
  -- note: actual proof would go here
  sorry

end oranges_apples_bananas_equiv_l2334_233445


namespace yeri_change_l2334_233488

theorem yeri_change :
  let cost_candies := 5 * 120
  let cost_chocolates := 3 * 350
  let total_cost := cost_candies + cost_chocolates
  let amount_handed_over := 2500
  amount_handed_over - total_cost = 850 :=
by
  sorry

end yeri_change_l2334_233488


namespace min_value_of_a_l2334_233474

theorem min_value_of_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y))) : 
  a ≥ Real.sqrt 2 :=
sorry -- Proof is omitted

end min_value_of_a_l2334_233474


namespace approx_ineq_l2334_233457

noncomputable def approx (x : ℝ) : ℝ := 1 + 6 * (-0.002 : ℝ)

theorem approx_ineq (x : ℝ) (h : x = 0.998) : 
  abs ((x^6) - approx x) < 0.001 :=
by
  sorry

end approx_ineq_l2334_233457


namespace sin_identity_l2334_233483

theorem sin_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (60 * Real.pi / 180 + 2 * α) = 7 / 9 :=
by
  sorry

end sin_identity_l2334_233483


namespace people_needed_to_mow_lawn_in_4_hours_l2334_233499

-- Define the given constants and conditions
def n := 4
def t := 6
def c := n * t -- The total work that can be done in constant hours
def t' := 4

-- Define the new number of people required to complete the work in t' hours
def n' := c / t'

-- Define the problem statement
theorem people_needed_to_mow_lawn_in_4_hours : n' - n = 2 := 
sorry

end people_needed_to_mow_lawn_in_4_hours_l2334_233499


namespace smallest_multiple_of_3_l2334_233431

theorem smallest_multiple_of_3 (a : ℕ) (h : ∀ i j : ℕ, i < 6 → j < 6 → 3 * (a + i) = 3 * (a + 10 + j) → a = 50) : 3 * a = 150 :=
by
  sorry

end smallest_multiple_of_3_l2334_233431


namespace aaron_pages_sixth_day_l2334_233428

theorem aaron_pages_sixth_day 
  (h1 : 18 + 12 + 23 + 10 + 17 + y = 6 * 15) : 
  y = 10 :=
by
  sorry

end aaron_pages_sixth_day_l2334_233428


namespace Connie_correct_result_l2334_233478

theorem Connie_correct_result :
  ∀ x: ℝ, (200 - x = 100) → (200 + x = 300) :=
by
  intros x h
  have h1 : x = 100 := by linarith [h]
  rw [h1]
  linarith

end Connie_correct_result_l2334_233478


namespace divisible_by_7_imp_coefficients_divisible_by_7_l2334_233400

theorem divisible_by_7_imp_coefficients_divisible_by_7
  (a0 a1 a2 a3 a4 a5 a6 : ℤ)
  (h : ∀ x : ℤ, 7 ∣ (a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)) :
  7 ∣ a0 ∧ 7 ∣ a1 ∧ 7 ∣ a2 ∧ 7 ∣ a3 ∧ 7 ∣ a4 ∧ 7 ∣ a5 ∧ 7 ∣ a6 :=
sorry

end divisible_by_7_imp_coefficients_divisible_by_7_l2334_233400


namespace calc_sum_of_digits_l2334_233433

theorem calc_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10) 
(hm : 10 * 3 + x = 34) (hmy : 34 * (10 * y + 4) = 136) : x + y = 7 :=
sorry

end calc_sum_of_digits_l2334_233433


namespace teamX_total_games_l2334_233436

variables (x : ℕ)

-- Conditions
def teamX_wins := (3/4) * x
def teamX_loses := (1/4) * x

def teamY_wins := (2/3) * (x + 10)
def teamY_loses := (1/3) * (x + 10)

-- Question: Prove team X played 20 games
theorem teamX_total_games :
  teamY_wins - teamX_wins = 5 ∧ teamY_loses - teamX_loses = 5 → x = 20 := by
sorry

end teamX_total_games_l2334_233436


namespace find_f2_plus_g2_l2334_233443

-- Functions f and g are defined
variable (f g : ℝ → ℝ)

-- Conditions based on the problem
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function : Prop := ∀ x : ℝ, g (-x) = g x
def function_equation : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

-- Lean Theorem Statement
theorem find_f2_plus_g2 (h1 : even_function f) (h2 : odd_function g) (h3 : function_equation f g) :
  f 2 + g 2 = -4 :=
by
  sorry

end find_f2_plus_g2_l2334_233443


namespace acute_angle_coincidence_l2334_233406

theorem acute_angle_coincidence (α : ℝ) (k : ℤ) :
  0 < α ∧ α < 180 ∧ 9 * α = k * 360 + α → α = 45 ∨ α = 90 ∨ α = 135 :=
by
  sorry

end acute_angle_coincidence_l2334_233406


namespace root_is_neg_one_then_m_eq_neg_3_l2334_233450

theorem root_is_neg_one_then_m_eq_neg_3 (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ x = -1) : m = -3 :=
sorry

end root_is_neg_one_then_m_eq_neg_3_l2334_233450


namespace fraction_equation_l2334_233493

theorem fraction_equation (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end fraction_equation_l2334_233493


namespace area_increase_l2334_233477

theorem area_increase (l w : ℝ) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_original := l * w
  let A_new := l_new * w_new
  ((A_new - A_original) / A_original) * 100 = 56 := 
by
  sorry

end area_increase_l2334_233477


namespace shop_conditions_l2334_233408

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l2334_233408


namespace farmer_land_owned_l2334_233412

def total_land (farmer_land : ℝ) (cleared_land : ℝ) : Prop :=
  cleared_land = 0.9 * farmer_land

def cleared_with_tomato (cleared_land : ℝ) (tomato_land : ℝ) : Prop :=
  tomato_land = 0.1 * cleared_land
  
def tomato_land_given (tomato_land : ℝ) : Prop :=
  tomato_land = 90

theorem farmer_land_owned (T : ℝ) :
  (∃ cleared : ℝ, total_land T cleared ∧ cleared_with_tomato cleared 90) → T = 1000 :=
by
  sorry

end farmer_land_owned_l2334_233412


namespace final_apples_count_l2334_233467

-- Define the initial conditions
def initial_apples : Nat := 128

def percent_25 (n : Nat) : Nat := n * 25 / 100

def apples_after_selling_to_jill (n : Nat) : Nat := n - percent_25 n

def apples_after_selling_to_june (n : Nat) : Nat := apples_after_selling_to_jill n - percent_25 (apples_after_selling_to_jill n)

def apples_after_giving_to_teacher (n : Nat) : Nat := apples_after_selling_to_june n - 1

-- The theorem stating the problem to be proved
theorem final_apples_count : apples_after_giving_to_teacher initial_apples = 71 := by
  sorry

end final_apples_count_l2334_233467


namespace coefficient_a_must_be_zero_l2334_233468

noncomputable def all_real_and_positive_roots (a b c : ℝ) : Prop :=
∀ p : ℝ, p > 0 → ∀ x : ℝ, (a * x^2 + b * x + c + p = 0) → x > 0

theorem coefficient_a_must_be_zero (a b c : ℝ) :
  (all_real_and_positive_roots a b c) → (a = 0) :=
by sorry

end coefficient_a_must_be_zero_l2334_233468


namespace range_of_m_l2334_233427

noncomputable def proof_problem (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) : Prop :=
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (1/x + 2/y = 1) ∧ (x + y / 2 < m^2 + 3 * m) ↔ (m < -4 ∨ m > 1)

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) :
  proof_problem x y m hx hy hxy :=
sorry

end range_of_m_l2334_233427


namespace alyssa_limes_picked_l2334_233490

-- Definitions for the conditions
def total_limes : ℕ := 57
def mike_limes : ℕ := 32

-- The statement to be proved
theorem alyssa_limes_picked :
  ∃ (alyssa_limes : ℕ), total_limes - mike_limes = alyssa_limes ∧ alyssa_limes = 25 :=
by
  have alyssa_limes : ℕ := total_limes - mike_limes
  use alyssa_limes
  sorry

end alyssa_limes_picked_l2334_233490


namespace problem_tiles_count_l2334_233418

theorem problem_tiles_count (T B : ℕ) (h: 2 * T + 3 * B = 301) (hB: B = 3) : T = 146 := 
by
  sorry

end problem_tiles_count_l2334_233418


namespace original_number_of_girls_l2334_233486

theorem original_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 4 * (b - 60) = g - 20) : 
  g = 460 / 11 :=
by
  sorry

end original_number_of_girls_l2334_233486


namespace common_point_geometric_progression_passing_l2334_233441

theorem common_point_geometric_progression_passing
  (a b c : ℝ) (r : ℝ) (h_b : b = a * r) (h_c : c = a * r^2) :
  ∃ x y : ℝ, (∀ a ≠ 0, a * x + (a * r) * y = a * r^2) → (x = 0 ∧ y = 1) :=
by
  sorry

end common_point_geometric_progression_passing_l2334_233441


namespace number_of_pairs_l2334_233495

theorem number_of_pairs (n : Nat) : 
  (∃ n, n > 2 ∧ ∀ x y : ℝ, (5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16) → True) :=
sorry

end number_of_pairs_l2334_233495


namespace Arman_total_earnings_two_weeks_l2334_233454

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end Arman_total_earnings_two_weeks_l2334_233454


namespace number_of_valid_consecutive_sum_sets_l2334_233429

-- Definition of what it means to be a set of consecutive integers summing to 225
def sum_of_consecutive_integers (n a : ℕ) : Prop :=
  ∃ k : ℕ, (k = (n * (2 * a + n - 1)) / 2) ∧ (k = 225)

-- Prove that there are exactly 4 sets of two or more consecutive positive integers that sum to 225
theorem number_of_valid_consecutive_sum_sets : 
  ∃ (sets : Finset (ℕ × ℕ)), 
    (∀ (n a : ℕ), (n, a) ∈ sets ↔ sum_of_consecutive_integers n a) ∧ 
    (2 ≤ n) ∧ 
    sets.card = 4 := sorry

end number_of_valid_consecutive_sum_sets_l2334_233429


namespace g_of_f_three_l2334_233476

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3*x^2 + 3*x + 2

theorem g_of_f_three : g (f 3) = 1952 := by
  sorry

end g_of_f_three_l2334_233476


namespace triangular_number_19_l2334_233491

def triangular_number (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

theorem triangular_number_19 : triangular_number 19 = 210 := by
  sorry

end triangular_number_19_l2334_233491


namespace find_a_l2334_233423

theorem find_a (a : ℝ) : (-2 * a + 3 = -4) -> (a = 7 / 2) :=
by
  intro h
  sorry

end find_a_l2334_233423


namespace find_a2023_l2334_233417

theorem find_a2023 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a n + a (n + 1) = n) : a 2023 = 1012 :=
sorry

end find_a2023_l2334_233417


namespace rightmost_four_digits_of_5_pow_2023_l2334_233442

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l2334_233442


namespace Daria_vacuum_cleaner_problem_l2334_233487

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l2334_233487


namespace perfect_square_trinomial_l2334_233422

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, a^2 = 1 ∧ b^2 = 1 ∧ x^2 + m * x * y + y^2 = (a * x + b * y)^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l2334_233422


namespace no_linear_term_in_product_l2334_233461

theorem no_linear_term_in_product (m : ℝ) :
  (∀ (x : ℝ), (x - 3) * (3 * x + m) - (3 * x^2 - 3 * m) = 0) → m = 9 :=
by
  intro h
  sorry

end no_linear_term_in_product_l2334_233461


namespace find_sum_l2334_233475

variable (a b c d : ℝ)

theorem find_sum :
  (ab + bc + cd + da = 20) →
  (b + d = 4) →
  (a + c = 5) := by
  sorry

end find_sum_l2334_233475


namespace fraction_calculation_l2334_233435

-- Define the initial values of x and y
def x : ℚ := 4 / 6
def y : ℚ := 8 / 10

-- Statement to prove
theorem fraction_calculation : (6 * x^2 + 10 * y) / (60 * x * y) = 11 / 36 := by
  sorry

end fraction_calculation_l2334_233435


namespace total_shaded_cubes_l2334_233440

/-
The large cube consists of 27 smaller cubes, each face is a 3x3 grid.
Opposite faces are shaded in an identical manner, with each face having 5 shaded smaller cubes.
-/

theorem total_shaded_cubes (number_of_smaller_cubes : ℕ)
  (face_shade_pattern : ∀ (face : ℕ), ℕ)
  (opposite_face_same_shade : ∀ (face1 face2 : ℕ), face1 = face2 → face_shade_pattern face1 = face_shade_pattern face2)
  (faces_possible : ∀ (face : ℕ), face < 6)
  (each_face_shaded_squares : ∀ (face : ℕ), face_shade_pattern face = 5)
  : ∃ (n : ℕ), n = 20 :=
by
  sorry

end total_shaded_cubes_l2334_233440


namespace arithmetic_seq_n_possible_values_l2334_233403

theorem arithmetic_seq_n_possible_values
  (a1 : ℕ) (a_n : ℕ → ℕ) (d : ℕ) (n : ℕ):
  a1 = 1 → 
  (∀ n, n ≥ 3 → a_n n = 100) → 
  (∃ d : ℕ, ∀ n, n ≥ 3 → a_n n = a1 + (n - 1) * d) → 
  (n = 4 ∨ n = 10 ∨ n = 12 ∨ n = 34 ∨ n = 100) := by
  sorry

end arithmetic_seq_n_possible_values_l2334_233403


namespace simplify_expression_l2334_233451

theorem simplify_expression (w : ℝ) : (5 - 2 * w) - (4 + 5 * w) = 1 - 7 * w := by 
  sorry

end simplify_expression_l2334_233451


namespace remainder_expression_div_10_l2334_233496

theorem remainder_expression_div_10 (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^p + t + 11^t * 6^(p * t)) % 10 = 1 :=
by
  sorry

end remainder_expression_div_10_l2334_233496


namespace not_possible_2020_parts_possible_2023_parts_l2334_233437

-- Define the initial number of parts and the operation that adds two parts
def initial_parts : Nat := 1
def operation (n : Nat) : Nat := n + 2

theorem not_possible_2020_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2020) : False :=
sorry

theorem possible_2023_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2023) : True :=
sorry

end not_possible_2020_parts_possible_2023_parts_l2334_233437


namespace probability_of_digit_six_l2334_233465

theorem probability_of_digit_six :
  let total_numbers := 90
  let favorable_numbers := 18
  0 < total_numbers ∧ 0 < favorable_numbers →
  (favorable_numbers / total_numbers : ℚ) = 1 / 5 :=
by
  intros total_numbers favorable_numbers h
  sorry

end probability_of_digit_six_l2334_233465


namespace row_length_in_feet_l2334_233462

theorem row_length_in_feet (seeds_per_row : ℕ) (space_per_seed : ℕ) (inches_per_foot : ℕ) (H1 : seeds_per_row = 80) (H2 : space_per_seed = 18) (H3 : inches_per_foot = 12) : 
  seeds_per_row * space_per_seed / inches_per_foot = 120 :=
by
  sorry

end row_length_in_feet_l2334_233462


namespace num_ways_express_2009_as_diff_of_squares_l2334_233492

theorem num_ways_express_2009_as_diff_of_squares : 
  ∃ (n : Nat), n = 12 ∧ 
  ∃ (a b : Int), ∀ c, 2009 = a^2 - b^2 ∧ 
  (c = 1 ∨ c = -1) ∧ (2009 = (c * a)^2 - (c * b)^2) :=
sorry

end num_ways_express_2009_as_diff_of_squares_l2334_233492


namespace A_completes_job_alone_l2334_233481

theorem A_completes_job_alone (efficiency_B efficiency_A total_work days_A : ℝ) :
  efficiency_A = 1.3 * efficiency_B → 
  total_work = (efficiency_A + efficiency_B) * 13 → 
  days_A = total_work / efficiency_A → 
  days_A = 23 :=
by
  intros h1 h2 h3
  sorry

end A_completes_job_alone_l2334_233481


namespace max_pieces_four_cuts_l2334_233409

theorem max_pieces_four_cuts (n : ℕ) (h : n = 4) : (by sorry : ℕ) = 14 := 
by sorry

end max_pieces_four_cuts_l2334_233409


namespace at_least_two_equal_l2334_233420

theorem at_least_two_equal (x y z : ℝ) (h1 : x * y + z = y * z + x) (h2 : y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l2334_233420


namespace coconuts_for_crab_l2334_233484

theorem coconuts_for_crab (C : ℕ) (H1 : 6 * C * 19 = 342) : C = 3 :=
sorry

end coconuts_for_crab_l2334_233484


namespace parallel_lines_not_coincident_l2334_233449

theorem parallel_lines_not_coincident (x y : ℝ) (m : ℝ) :
  (∀ y, x + (1 + m) * y = 2 - m ∧ ∀ y, m * x + 2 * y + 8 = 0) → (m =1) := 
sorry

end parallel_lines_not_coincident_l2334_233449


namespace number_of_bugs_l2334_233480

def flowers_per_bug := 2
def total_flowers_eaten := 6

theorem number_of_bugs : total_flowers_eaten / flowers_per_bug = 3 := 
by sorry

end number_of_bugs_l2334_233480


namespace map_distance_to_actual_distance_l2334_233471

theorem map_distance_to_actual_distance :
  ∀ (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ), 
    d_map = 15 → scale_inch = 0.25 → scale_mile = 3 →
    (d_map / scale_inch) * scale_mile = 180 :=
by
  intros d_map scale_inch scale_mile h1 h2 h3
  rw [h1, h2, h3]
  sorry

end map_distance_to_actual_distance_l2334_233471


namespace james_total_payment_l2334_233497

noncomputable def first_pair_cost : ℝ := 40
noncomputable def second_pair_cost : ℝ := 60
noncomputable def discount_applied_to : ℝ := min first_pair_cost second_pair_cost
noncomputable def discount_amount := discount_applied_to / 2
noncomputable def total_before_extra_discount := first_pair_cost + (second_pair_cost - discount_amount)
noncomputable def extra_discount := total_before_extra_discount / 4
noncomputable def final_amount := total_before_extra_discount - extra_discount

theorem james_total_payment : final_amount = 60 := by
  sorry

end james_total_payment_l2334_233497


namespace ellie_sam_in_photo_probability_l2334_233479

-- Definitions of the conditions
def lap_time_ellie := 120 -- seconds
def lap_time_sam := 75 -- seconds
def start_time := 10 * 60 -- 10 minutes in seconds
def photo_duration := 60 -- 1 minute in seconds
def photo_section := 1 / 3 -- fraction of the track captured in the photo

-- The probability that both Ellie and Sam are in the photo section between 10 to 11 minutes
theorem ellie_sam_in_photo_probability :
  let ellie_time := start_time;
  let sam_time := start_time;
  let ellie_range := (ellie_time - (photo_section * lap_time_ellie / 2), ellie_time + (photo_section * lap_time_ellie / 2));
  let sam_range := (sam_time - (photo_section * lap_time_sam / 2), sam_time + (photo_section * lap_time_sam / 2));
  let overlap_start := max ellie_range.1 sam_range.1;
  let overlap_end := min ellie_range.2 sam_range.2;
  let overlap_duration := max 0 (overlap_end - overlap_start);
  let overlap_probability := overlap_duration / photo_duration;
  overlap_probability = 5 / 12 :=
by
  sorry

end ellie_sam_in_photo_probability_l2334_233479


namespace david_age_uniq_l2334_233455

theorem david_age_uniq (C D E : ℚ) (h1 : C = 4 * D) (h2 : E = D + 7) (h3 : C = E + 1) : D = 8 / 3 := 
by 
  sorry

end david_age_uniq_l2334_233455


namespace roots_of_polynomial_l2334_233494

noncomputable def P (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 - 6 * x

theorem roots_of_polynomial : ∀ x : ℝ, P x = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by 
  -- Here you would provide the proof, but we use sorry to indicate it is left out
  sorry

end roots_of_polynomial_l2334_233494


namespace arithmetic_proof_l2334_233413

theorem arithmetic_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end arithmetic_proof_l2334_233413


namespace possible_values_of_b_l2334_233415

theorem possible_values_of_b (b : ℝ) (h : ∃ x y : ℝ, y = 2 * x + b ∧ y > 0 ∧ x = 0) : b > 0 :=
sorry

end possible_values_of_b_l2334_233415


namespace max_rabbits_with_long_ears_and_jumping_far_l2334_233498

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l2334_233498


namespace infinite_colored_points_l2334_233469

theorem infinite_colored_points
(P : ℤ → Prop) (red blue : ℤ → Prop)
(h_color : ∀ n : ℤ, (red n ∨ blue n))
(h_red_blue_partition : ∀ n : ℤ, ¬(red n ∧ blue n)) :
  ∃ (C : ℤ → Prop) (k : ℕ), (C = red ∨ C = blue) ∧ ∀ n : ℕ, ∃ m : ℤ, C m ∧ (m % n) = 0 :=
by
  sorry

end infinite_colored_points_l2334_233469


namespace fifth_inequality_l2334_233402

theorem fifth_inequality (h1: 1 / Real.sqrt 2 < 1)
                         (h2: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
                         (h3: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
                         1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := 
sorry

end fifth_inequality_l2334_233402


namespace factorize_expr1_factorize_expr2_l2334_233416

-- Proof Problem 1
theorem factorize_expr1 (a : ℝ) : 
  (a^2 - 4 * a + 4 - 4 * (a - 2) + 4) = (a - 4)^2 :=
sorry

-- Proof Problem 2
theorem factorize_expr2 (x y : ℝ) : 
  16 * x^4 - 81 * y^4 = (4 * x^2 + 9 * y^2) * (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

end factorize_expr1_factorize_expr2_l2334_233416


namespace solve_abs_inequality_l2334_233453

theorem solve_abs_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8) :=
sorry

end solve_abs_inequality_l2334_233453


namespace complex_quadrant_l2334_233419

open Complex

theorem complex_quadrant (z : ℂ) (h : (1 + I) * z = 2 * I) : 
  z.re > 0 ∧ z.im < 0 :=
  sorry

end complex_quadrant_l2334_233419


namespace smallest_sum_B_c_l2334_233482

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l2334_233482


namespace solve_inequality_l2334_233430

theorem solve_inequality (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ x ∈ Set.Ioo (-2) (3) :=
sorry

end solve_inequality_l2334_233430


namespace impossible_list_10_numbers_with_given_conditions_l2334_233447

theorem impossible_list_10_numbers_with_given_conditions :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ 7 → (a i * a (i + 1) * a (i + 2)) % 6 = 0) ∧
    (∀ i, 0 ≤ i ∧ i ≤ 8 → (a i * a (i + 1)) % 6 ≠ 0) :=
by
  sorry

end impossible_list_10_numbers_with_given_conditions_l2334_233447


namespace total_cost_for_seven_hard_drives_l2334_233424

-- Condition: Two identical hard drives cost $50.
def cost_of_two_hard_drives : ℝ := 50

-- Condition: There is a 10% discount if you buy more than four hard drives.
def discount_rate : ℝ := 0.10

-- Question: What is the total cost in dollars for buying seven of these hard drives?
theorem total_cost_for_seven_hard_drives : (7 * (cost_of_two_hard_drives / 2)) * (1 - discount_rate) = 157.5 := 
by 
  -- def cost_of_one_hard_drive
  let cost_of_one_hard_drive := cost_of_two_hard_drives / 2
  -- def cost_of_seven_hard_drives
  let cost_of_seven_hard_drives := 7 * cost_of_one_hard_drive
  have h₁ : 7 * (cost_of_two_hard_drives / 2) = cost_of_seven_hard_drives := by sorry
  have h₂ : cost_of_seven_hard_drives * (1 - discount_rate) = 157.5 := by sorry
  exact h₂

end total_cost_for_seven_hard_drives_l2334_233424
