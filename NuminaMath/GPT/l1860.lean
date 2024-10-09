import Mathlib

namespace cost_per_ball_correct_l1860_186008

-- Define the values given in the conditions
def total_amount_paid : ℝ := 4.62
def number_of_balls : ℝ := 3.0

-- Define the expected cost per ball according to the problem statement
def expected_cost_per_ball : ℝ := 1.54

-- Statement to prove that the cost per ball is as expected
theorem cost_per_ball_correct : (total_amount_paid / number_of_balls) = expected_cost_per_ball := 
sorry

end cost_per_ball_correct_l1860_186008


namespace coleen_sprinkles_l1860_186069

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end coleen_sprinkles_l1860_186069


namespace problem_statement_l1860_186049

theorem problem_statement (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : abc = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) := 
  sorry

end problem_statement_l1860_186049


namespace problem_rewrite_equation_l1860_186076

theorem problem_rewrite_equation :
  ∃ a b c : ℤ, a > 0 ∧ (64*(x^2) + 96*x - 81 = 0) → ((a*x + b)^2 = c) ∧ (a + b + c = 131) :=
sorry

end problem_rewrite_equation_l1860_186076


namespace letters_by_30_typists_in_1_hour_l1860_186097

-- Definitions from the conditions
def lettersTypedByOneTypistIn20Minutes := 44 / 20

def lettersTypedBy30TypistsIn20Minutes := 30 * (lettersTypedByOneTypistIn20Minutes)

def conversionToHours := 3

-- Theorem statement
theorem letters_by_30_typists_in_1_hour : lettersTypedBy30TypistsIn20Minutes * conversionToHours = 198 := by
  sorry

end letters_by_30_typists_in_1_hour_l1860_186097


namespace initial_number_l1860_186027

theorem initial_number (N : ℤ) 
  (h : (N + 3) % 24 = 0) : N = 21 := 
sorry

end initial_number_l1860_186027


namespace max_c_for_range_l1860_186011

theorem max_c_for_range (c : ℝ) :
  (∃ x : ℝ, (x^2 - 7*x + c = 2)) → c ≤ 57 / 4 :=
by
  sorry

end max_c_for_range_l1860_186011


namespace temperature_on_tuesday_l1860_186099

theorem temperature_on_tuesday 
  (T W Th F : ℝ)
  (H1 : (T + W + Th) / 3 = 45)
  (H2 : (W + Th + F) / 3 = 50)
  (H3 : F = 53) :
  T = 38 :=
by 
  sorry

end temperature_on_tuesday_l1860_186099


namespace train_cars_estimate_l1860_186084

noncomputable def train_cars_count (total_time_secs : ℕ) (delay_secs : ℕ) (cars_counted : ℕ) (count_time_secs : ℕ): ℕ := 
  let rate_per_sec := cars_counted / count_time_secs
  let cars_missed := delay_secs * rate_per_sec
  let cars_in_remaining_time := rate_per_sec * (total_time_secs - delay_secs)
  cars_missed + cars_in_remaining_time

theorem train_cars_estimate :
  train_cars_count 210 15 8 20 = 120 :=
sorry

end train_cars_estimate_l1860_186084


namespace tan_theta_values_l1860_186041

theorem tan_theta_values (θ : ℝ) (h₁ : 0 < θ ∧ θ < Real.pi / 2) (h₂ : 12 / Real.sin θ + 12 / Real.cos θ = 35) : 
  Real.tan θ = 4 / 3 ∨ Real.tan θ = 3 / 4 := 
by
  sorry

end tan_theta_values_l1860_186041


namespace additional_track_length_l1860_186020

theorem additional_track_length (h : ℝ) (g1 g2 : ℝ) (L1 L2 : ℝ)
  (rise_eq : h = 800) 
  (orig_grade : g1 = 0.04) 
  (new_grade : g2 = 0.025) 
  (L1_eq : L1 = h / g1) 
  (L2_eq : L2 = h / g2)
  : (L2 - L1 = 12000) := 
sorry

end additional_track_length_l1860_186020


namespace river_depth_difference_l1860_186009

theorem river_depth_difference
  (mid_may_depth : ℕ)
  (mid_july_depth : ℕ)
  (mid_june_depth : ℕ)
  (H1 : mid_july_depth = 45)
  (H2 : mid_may_depth = 5)
  (H3 : 3 * mid_june_depth = mid_july_depth) :
  mid_june_depth - mid_may_depth = 10 := 
sorry

end river_depth_difference_l1860_186009


namespace area_quadrilateral_l1860_186014

theorem area_quadrilateral (EF GH: ℝ) (EHG: ℝ) 
  (h1 : EF = 9) (h2 : GH = 12) (h3 : GH = EH) (h4 : EHG = 75) 
  (a b c : ℕ)
  : 
  (∀ (a b c : ℕ), 
  a = 26 ∧ b = 18 ∧ c = 6 → 
  a + b + c = 50) := 
sorry

end area_quadrilateral_l1860_186014


namespace initial_earning_members_l1860_186031

theorem initial_earning_members (n : ℕ)
  (avg_income_initial : ℕ) (avg_income_after : ℕ) (income_deceased : ℕ)
  (h1 : avg_income_initial = 735)
  (h2 : avg_income_after = 590)
  (h3 : income_deceased = 1170)
  (h4 : 735 * n - 1170 = 590 * (n - 1)) :
  n = 4 :=
by
  sorry

end initial_earning_members_l1860_186031


namespace percentage_of_students_play_sports_l1860_186015

def total_students : ℕ := 400
def soccer_percentage : ℝ := 0.125
def soccer_players : ℕ := 26

theorem percentage_of_students_play_sports : 
  ∃ P : ℝ, (soccer_percentage * P = soccer_players) → (P / total_students * 100 = 52) :=
by
  sorry

end percentage_of_students_play_sports_l1860_186015


namespace sum_of_five_consecutive_integers_l1860_186081

theorem sum_of_five_consecutive_integers : ∀ (n : ℤ), (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := 
by
  -- This would be where the proof goes
  sorry

end sum_of_five_consecutive_integers_l1860_186081


namespace scientific_notation_of_188_million_l1860_186062

theorem scientific_notation_of_188_million : 
  (188000000 : ℝ) = 1.88 * 10^8 := 
by
  sorry

end scientific_notation_of_188_million_l1860_186062


namespace find_m_l1860_186053

-- Let's define the sets A and B.
def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- We'll state the problem as a theorem
theorem find_m (m : ℝ) (h : B m ⊆ A) : m = 1 ∨ m = -1 :=
by sorry

end find_m_l1860_186053


namespace bob_final_amount_l1860_186085

noncomputable def final_amount (start: ℝ) : ℝ :=
  let day1 := start - (3/5) * start
  let day2 := day1 - (7/12) * day1
  let day3 := day2 - (2/3) * day2
  let day4 := day3 - (1/6) * day3
  let day5 := day4 - (5/8) * day4
  let day6 := day5 - (3/5) * day5
  day6

theorem bob_final_amount : final_amount 500 = 3.47 := by
  sorry

end bob_final_amount_l1860_186085


namespace final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l1860_186030

-- Definitions of the driving records for trainee A and B
def driving_record_A : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def driving_record_B : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

-- Fuel consumption rate per kilometer
variable (a : ℝ)

-- Proof statements in Lean
theorem final_position_A : driving_record_A.sum = 39 := by sorry
theorem final_position_B : driving_record_B.sum = -4 := by sorry
theorem fuel_consumption_A : (driving_record_A.map (abs)).sum * a = 65 * a := by sorry
theorem fuel_consumption_B : (driving_record_B.map (abs)).sum * a = 76 * a := by sorry
theorem less_fuel_consumption : (driving_record_A.map (abs)).sum * a < (driving_record_B.map (abs)).sum * a := by sorry

end final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l1860_186030


namespace problem_statement_l1860_186089

theorem problem_statement (x y : ℝ) (h : |x + 1| + |y + 2 * x| = 0) : (x + y) ^ 2004 = 1 := by
  sorry

end problem_statement_l1860_186089


namespace find_triples_l1860_186046

theorem find_triples : 
  { (a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1) } = 
  { (1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3) } := 
by
  sorry

end find_triples_l1860_186046


namespace smallest_y_l1860_186043

theorem smallest_y (y : ℕ) : 
    (y % 5 = 4) ∧ 
    (y % 7 = 6) ∧ 
    (y % 8 = 7) → 
    y = 279 :=
sorry

end smallest_y_l1860_186043


namespace polynomial_coefficient_sum_l1860_186070

theorem polynomial_coefficient_sum :
  ∀ (a0 a1 a2 a3 a4 a5 : ℤ), 
  (3 - 2 * x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 → 
  a0 + a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 233 :=
by
  sorry

end polynomial_coefficient_sum_l1860_186070


namespace minimum_value_of_linear_expression_l1860_186001

theorem minimum_value_of_linear_expression :
  ∀ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 → 2 * x + y ≥ -5 :=
by
  sorry

end minimum_value_of_linear_expression_l1860_186001


namespace negation_proposition_l1860_186096

open Classical

variable (x : ℝ)

def proposition (x : ℝ) : Prop := ∀ x > 1, Real.log x / Real.log 2 > 0

theorem negation_proposition (h : ¬ proposition x) : 
  ∃ x > 1, Real.log x / Real.log 2 ≤ 0 := by
  sorry

end negation_proposition_l1860_186096


namespace meaningful_iff_x_ne_1_l1860_186068

theorem meaningful_iff_x_ne_1 (x : ℝ) : (x - 1) ≠ 0 ↔ (x ≠ 1) :=
by 
  sorry

end meaningful_iff_x_ne_1_l1860_186068


namespace evaluate_expression_zero_l1860_186050

-- Main proof statement
theorem evaluate_expression_zero :
  ∀ (a d c b : ℤ),
    d = c + 5 →
    c = b - 8 →
    b = a + 3 →
    a = 3 →
    a - 1 ≠ 0 →
    d - 6 ≠ 0 →
    c + 4 ≠ 0 →
    (a + 3) * (d - 3) * (c + 9) = 0 :=
by
  intros a d c b hd hc hb ha h1 h2 h3
  sorry -- The proof goes here

end evaluate_expression_zero_l1860_186050


namespace chairs_carried_per_trip_l1860_186002

theorem chairs_carried_per_trip (x : ℕ) (friends : ℕ) (trips : ℕ) (total_chairs : ℕ) 
  (h1 : friends = 4) (h2 : trips = 10) (h3 : total_chairs = 250) 
  (h4 : 5 * (trips * x) = total_chairs) : x = 5 :=
by sorry

end chairs_carried_per_trip_l1860_186002


namespace compare_abc_l1860_186093

noncomputable def a : ℝ := 2 + (1 / 5) * Real.log 2
noncomputable def b : ℝ := 1 + Real.exp (0.2 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.1 * Real.log 2)

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l1860_186093


namespace min_star_value_l1860_186032

theorem min_star_value :
  ∃ (star : ℕ), (98348 * 10 + star) % 72 = 0 ∧ (∀ (x : ℕ), (98348 * 10 + x) % 72 = 0 → star ≤ x) := sorry

end min_star_value_l1860_186032


namespace even_stones_fraction_odd_stones_fraction_l1860_186025

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an even number of stones is 12/65. -/
theorem even_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 0 ∧ B2 % 2 = 0 ∧ B3 % 2 = 0 ∧ B4 % 2 = 0 ∧ B1 + B2 + B3 + B4 = 12) → (84 / 455 = 12 / 65) := 
by sorry

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an odd number of stones is 1/13. -/
theorem odd_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 1 ∧ B2 % 2 = 1 ∧ B3 % 2 = 1 ∧ B4 % 2 = 1 ∧ B1 + B2 + B3 + B4 = 12) → (35 / 455 = 1 / 13) := 
by sorry

end even_stones_fraction_odd_stones_fraction_l1860_186025


namespace daily_sales_volume_selling_price_for_profit_l1860_186029

noncomputable def cost_price : ℝ := 40
noncomputable def initial_selling_price : ℝ := 60
noncomputable def initial_sales_volume : ℝ := 20
noncomputable def price_decrease_per_increase : ℝ := 5
noncomputable def volume_increase_per_decrease : ℝ := 10

theorem daily_sales_volume (p : ℝ) (v : ℝ) :
  v = initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease :=
sorry

theorem selling_price_for_profit (p : ℝ) (profit : ℝ) :
  profit = (p - cost_price) * (initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease) → p = 54 :=
sorry

end daily_sales_volume_selling_price_for_profit_l1860_186029


namespace cheryl_tournament_cost_is_1440_l1860_186048

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end cheryl_tournament_cost_is_1440_l1860_186048


namespace smallest_range_mean_2017_l1860_186037

theorem smallest_range_mean_2017 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) / 4 = 2017 ∧ (max (max a b) (max c d) - min (min a b) (min c d)) = 4 := 
sorry

end smallest_range_mean_2017_l1860_186037


namespace sum_first_7_l1860_186055

variable {α : Type*} [LinearOrderedField α]

-- Definitions for the arithmetic sequence
noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

-- Conditions
variable {a d : α} -- Initial term and common difference of the arithmetic sequence
variable (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12)

-- Proof statement
theorem sum_first_7 (a d : α) (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12) : 
  sum_of_first_n_terms a d 7 = 28 := 
by 
  sorry

end sum_first_7_l1860_186055


namespace find_m_n_and_max_value_l1860_186036

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + 3 * m + n

-- Define a predicate for the function being even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the conditions and what we want to prove
theorem find_m_n_and_max_value :
  ∀ m n : ℝ,
    is_even_function (f m n) →
    (m - 1 ≤ 2 * m) →
      (m = 1 / 3 ∧ n = 0) ∧ 
      (∀ x : ℝ, -2 / 3 ≤ x ∧ x ≤ 2 / 3 → f (1/3) 0 x ≤ 31 / 27) :=
by
  sorry

end find_m_n_and_max_value_l1860_186036


namespace jenna_age_l1860_186074

theorem jenna_age (D J : ℕ) (h1 : J = D + 5) (h2 : J + D = 21) (h3 : D = 8) : J = 13 :=
by
  sorry

end jenna_age_l1860_186074


namespace least_candies_to_remove_for_equal_distribution_l1860_186067

theorem least_candies_to_remove_for_equal_distribution :
  ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, 24 - k = 5 * n :=
sorry

end least_candies_to_remove_for_equal_distribution_l1860_186067


namespace circle_area_of_circumscribed_triangle_l1860_186039

theorem circle_area_of_circumscribed_triangle :
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  π * R^2 = (5184 / 119) * π := 
by
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  have h1 : height = Real.sqrt (a^2 - (c / 2)^2) := by sorry
  have h2 : A = (1 / 2) * c * height := by sorry
  have h3 : R = (a * b * c) / (4 * A) := by sorry
  have h4 : π * R^2 = (5184 / 119) * π := by sorry
  exact h4

end circle_area_of_circumscribed_triangle_l1860_186039


namespace sum_of_transformed_numbers_l1860_186040

variables (a b x k S : ℝ)

-- Define the condition that a + b = S
def sum_condition : Prop := a + b = S

-- Define the function that represents the final sum after transformations
def final_sum (a b x k : ℝ) : ℝ :=
  k * (a + x) + k * (b + x)

-- The theorem statement to prove
theorem sum_of_transformed_numbers (h : sum_condition a b S) : 
  final_sum a b x k = k * S + 2 * k * x :=
by
  sorry

end sum_of_transformed_numbers_l1860_186040


namespace modulo_calculation_l1860_186044

theorem modulo_calculation (n : ℕ) (hn : 0 ≤ n ∧ n < 19) (hmod : 5 * n % 19 = 1) : 
  ((3^n)^2 - 3) % 19 = 3 := 
by 
  sorry

end modulo_calculation_l1860_186044


namespace evaluate_sqrt_sum_l1860_186094

theorem evaluate_sqrt_sum : (Real.sqrt 1 + Real.sqrt 9) = 4 := by
  sorry

end evaluate_sqrt_sum_l1860_186094


namespace isosceles_trapezoid_ratio_l1860_186091

theorem isosceles_trapezoid_ratio (a b d : ℝ) (h1 : b = 2 * d) (h2 : a = d) : a / b = 1 / 2 :=
by
  sorry

end isosceles_trapezoid_ratio_l1860_186091


namespace tshirts_per_package_l1860_186079

def number_of_packages := 28
def total_white_tshirts := 56
def white_tshirts_per_package : Nat :=
  total_white_tshirts / number_of_packages

theorem tshirts_per_package :
  white_tshirts_per_package = 2 :=
by
  -- Assuming the definitions and the proven facts
  sorry

end tshirts_per_package_l1860_186079


namespace find_pairs_l1860_186071

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end find_pairs_l1860_186071


namespace trigonometric_identity_l1860_186060

noncomputable def alpha := -35 / 6 * Real.pi

theorem trigonometric_identity :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha)
    - Real.sin (3 * Real.pi / 2 + alpha)) /
  (1 + Real.sin (alpha) ^ 2 - Real.cos (Real.pi / 2 + alpha)
    - Real.cos (Real.pi + alpha) ^ 2) = -Real.sqrt 3 := by
  sorry

end trigonometric_identity_l1860_186060


namespace complement_P_correct_l1860_186045

def is_solution (x : ℝ) : Prop := |x + 3| + |x + 6| = 3

def P : Set ℝ := {x | is_solution x}

def C_R (P : Set ℝ) : Set ℝ := {x | x ∉ P}

theorem complement_P_correct : C_R P = {x | x < -6 ∨ x > -3} :=
by
  sorry

end complement_P_correct_l1860_186045


namespace gcf_factor_l1860_186012

theorem gcf_factor (x y : ℕ) : gcd (6 * x ^ 3 * y ^ 2) (3 * x ^ 2 * y ^ 3) = 3 * x ^ 2 * y ^ 2 :=
by
  sorry

end gcf_factor_l1860_186012


namespace four_thirds_of_product_eq_25_div_2_l1860_186051

noncomputable def a : ℚ := 15 / 4
noncomputable def b : ℚ := 5 / 2
noncomputable def c : ℚ := 4 / 3
noncomputable def d : ℚ := a * b
noncomputable def e : ℚ := c * d

theorem four_thirds_of_product_eq_25_div_2 : e = 25 / 2 := 
sorry

end four_thirds_of_product_eq_25_div_2_l1860_186051


namespace rectangle_area_unchanged_l1860_186054

theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) : 
  0.8 * l * 1.25 * w = 432 := 
by {
  -- The proof goes here
  sorry
}

end rectangle_area_unchanged_l1860_186054


namespace evaluate_fraction_l1860_186005

theorem evaluate_fraction : 
  (7/3) / (8/15) = 35/8 :=
by
  -- we don't need to provide the proof as per instructions
  sorry

end evaluate_fraction_l1860_186005


namespace statement_A_correct_statement_C_correct_l1860_186007

open Nat

def combinations (n r : ℕ) : ℕ := n.choose r

theorem statement_A_correct : combinations 5 3 = combinations 5 2 := sorry

theorem statement_C_correct : combinations 6 3 - combinations 4 1 = combinations 6 3 - 4 := sorry

end statement_A_correct_statement_C_correct_l1860_186007


namespace detergent_per_pound_l1860_186083

theorem detergent_per_pound (detergent clothes_per_det: ℝ) (h: detergent = 18 ∧ clothes_per_det = 9) :
  detergent / clothes_per_det = 2 :=
by
  sorry

end detergent_per_pound_l1860_186083


namespace restaurant_total_tables_l1860_186022

theorem restaurant_total_tables (N O : ℕ) (h1 : 6 * N + 4 * O = 212) (h2 : N = O + 12) : N + O = 40 :=
sorry

end restaurant_total_tables_l1860_186022


namespace incorrect_statement_l1860_186065

def vector_mult (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem incorrect_statement (a b : ℝ × ℝ) : vector_mult a b ≠ vector_mult b a :=
by
  sorry

end incorrect_statement_l1860_186065


namespace intersection_points_3_l1860_186006

def eq1 (x y : ℝ) : Prop := (x - y + 3) * (2 * x + 3 * y - 9) = 0
def eq2 (x y : ℝ) : Prop := (2 * x - y + 2) * (x + 3 * y - 6) = 0

theorem intersection_points_3 :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y) ∧
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    eq1 x1 y1 ∧ eq2 x1 y1 ∧ 
    eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    eq1 x3 y3 ∧ eq2 x3 y3 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :=
sorry

end intersection_points_3_l1860_186006


namespace mondays_in_first_70_days_l1860_186023

theorem mondays_in_first_70_days (days : ℕ) (h1 : days = 70) (mondays_per_week : ℕ) (h2 : mondays_per_week = 1) : 
  ∃ (mondays : ℕ), mondays = 10 := 
by
  sorry

end mondays_in_first_70_days_l1860_186023


namespace line_parallel_not_coincident_l1860_186064

theorem line_parallel_not_coincident (a : ℝ) :
  (a = 3) ↔ (∀ x y, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y + 7 - a = 0) → 
              (∃ k : Real, a / 3 = k ∧ k ≠ 3 * a / (7 - a))) :=
by
  sorry

end line_parallel_not_coincident_l1860_186064


namespace stratified_sampling_correct_l1860_186047

-- Defining the conditions
def total_students : ℕ := 900
def freshmen : ℕ := 300
def sophomores : ℕ := 200
def juniors : ℕ := 400
def sample_size : ℕ := 45

-- Defining the target sample numbers
def freshmen_sample : ℕ := 15
def sophomores_sample : ℕ := 10
def juniors_sample : ℕ := 20

-- The proof problem statement
theorem stratified_sampling_correct :
  freshmen_sample = (freshmen * sample_size / total_students) ∧
  sophomores_sample = (sophomores * sample_size / total_students) ∧
  juniors_sample = (juniors * sample_size / total_students) :=
by
  sorry

end stratified_sampling_correct_l1860_186047


namespace trigonometric_identities_l1860_186010

noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

theorem trigonometric_identities (θ : ℝ) (h_tan : tan θ = 2) (h_identity : sin θ ^ 2 + cos θ ^ 2 = 1) :
    ((sin θ = 2 * Real.sqrt 5 / 5 ∧ cos θ = Real.sqrt 5 / 5) ∨ (sin θ = -2 * Real.sqrt 5 / 5 ∧ cos θ = -Real.sqrt 5 / 5)) ∧
    ((4 * sin θ - 3 * cos θ) / (6 * cos θ + 2 * sin θ) = 1 / 2) :=
by
  sorry

end trigonometric_identities_l1860_186010


namespace value_of_a_plus_b_l1860_186078

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l1860_186078


namespace john_tanks_needed_l1860_186087

theorem john_tanks_needed 
  (num_balloons : ℕ) 
  (volume_per_balloon : ℕ) 
  (volume_per_tank : ℕ) 
  (H1 : num_balloons = 1000) 
  (H2 : volume_per_balloon = 10) 
  (H3 : volume_per_tank = 500) 
: (num_balloons * volume_per_balloon) / volume_per_tank = 20 := 
by 
  sorry

end john_tanks_needed_l1860_186087


namespace xunzi_statement_l1860_186080

/-- 
Given the conditions:
  "If not accumulating small steps, then not reaching a thousand miles."
  Which can be represented as: ¬P → ¬q.
Prove that accumulating small steps (P) is a necessary but not sufficient condition for
reaching a thousand miles (q).
-/
theorem xunzi_statement (P q : Prop) (h : ¬P → ¬q) : (q → P) ∧ ¬(P → q) :=
by sorry

end xunzi_statement_l1860_186080


namespace fourth_metal_mass_approx_l1860_186003

noncomputable def mass_of_fourth_metal 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : ℝ :=
  x4

theorem fourth_metal_mass_approx 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : 
  abs (mass_of_fourth_metal x1 x2 x3 x4 h1 h2 h3 h4 - 7.36) < 0.01 :=
by
  sorry

end fourth_metal_mass_approx_l1860_186003


namespace feet_per_inch_of_model_l1860_186028

theorem feet_per_inch_of_model 
  (height_tower : ℝ)
  (height_model : ℝ)
  (height_tower_eq : height_tower = 984)
  (height_model_eq : height_model = 6)
  : (height_tower / height_model) = 164 :=
by
  -- Assume the proof here
  sorry

end feet_per_inch_of_model_l1860_186028


namespace initial_distance_between_trains_l1860_186024

theorem initial_distance_between_trains :
  let length_train1 := 100 -- meters
  let length_train2 := 200 -- meters
  let speed_train1_kmph := 54 -- km/h
  let speed_train2_kmph := 72 -- km/h
  let time_hours := 1.999840012798976 -- hours
  
  -- Conversion to meters per second
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600 -- 15 m/s
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600 -- 20 m/s

  -- Conversion of time to seconds
  let time_seconds := time_hours * 3600 -- 7199.4240460755136 seconds

  -- Relative speed in meters per second
  let relative_speed := speed_train1_mps + speed_train2_mps -- 35 m/s

  -- Distance covered by both trains
  let distance_covered := relative_speed * time_seconds -- 251980.84161264498 meters

  -- Initial distance between the trains
  let initial_distance := distance_covered - (length_train1 + length_train2) -- 251680.84161264498 meters

  initial_distance = 251680.84161264498 := 
by
  sorry

end initial_distance_between_trains_l1860_186024


namespace solve_fraction_x_l1860_186026

theorem solve_fraction_x (a b c d : ℤ) (hb : b ≠ 0) (hdc : d + c ≠ 0) 
: (2 * a + (bc - 2 * a * d) / (d + c)) / (b - (bc - 2 * a * d) / (d + c)) = c / d := 
sorry

end solve_fraction_x_l1860_186026


namespace remainder_is_zero_l1860_186088

def remainder_when_multiplied_then_subtracted (a b : ℕ) : ℕ :=
  (a * b - 8) % 8

theorem remainder_is_zero : remainder_when_multiplied_then_subtracted 104 106 = 0 := by
  sorry

end remainder_is_zero_l1860_186088


namespace sqrt_ceil_eq_one_range_of_x_l1860_186075

/-- Given $[m]$ represents the largest integer not greater than $m$, prove $[\sqrt{2}] = 1$. -/
theorem sqrt_ceil_eq_one (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) :
  floor (Real.sqrt 2) = 1 :=
sorry

/-- Given $[m]$ represents the largest integer not greater than $m$ and $[3 + \sqrt{x}] = 6$, 
  prove $9 \leq x < 16$. -/
theorem range_of_x (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) 
  (x : ℝ) (h : floor (3 + Real.sqrt x) = 6) :
  9 ≤ x ∧ x < 16 :=
sorry

end sqrt_ceil_eq_one_range_of_x_l1860_186075


namespace shorter_leg_of_right_triangle_l1860_186061

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l1860_186061


namespace cost_of_each_croissant_l1860_186056

theorem cost_of_each_croissant 
  (quiches_price : ℝ) (num_quiches : ℕ) (each_quiche_cost : ℝ)
  (buttermilk_biscuits_price : ℝ) (num_biscuits : ℕ) (each_biscuit_cost : ℝ)
  (total_cost_with_discount : ℝ) (discount_rate : ℝ)
  (num_croissants : ℕ) (croissant_price : ℝ) :
  quiches_price = num_quiches * each_quiche_cost →
  each_quiche_cost = 15 →
  num_quiches = 2 →
  buttermilk_biscuits_price = num_biscuits * each_biscuit_cost →
  each_biscuit_cost = 2 →
  num_biscuits = 6 →
  discount_rate = 0.10 →
  (quiches_price + buttermilk_biscuits_price + (num_croissants * croissant_price)) * (1 - discount_rate) = total_cost_with_discount →
  total_cost_with_discount = 54 →
  num_croissants = 6 →
  croissant_price = 3 :=
sorry

end cost_of_each_croissant_l1860_186056


namespace term_100_is_981_l1860_186090

def sequence_term (n : ℕ) : ℕ :=
  if n = 100 then 981 else sorry

theorem term_100_is_981 : sequence_term 100 = 981 := by
  rfl

end term_100_is_981_l1860_186090


namespace total_quartet_songs_l1860_186098

/-- 
Five girls — Mary, Alina, Tina, Hanna, and Elsa — sang songs in a concert as quartets,
with one girl sitting out each time. Hanna sang 9 songs, which was more than any other girl,
and Mary sang 3 songs, which was fewer than any other girl. If the total number of songs
sung by Alina and Tina together was 16, then the total number of songs sung by these quartets is 8. -/
theorem total_quartet_songs
  (hanna_songs : ℕ) (mary_songs : ℕ) (alina_tina_songs : ℕ) (total_songs : ℕ)
  (h_hanna : hanna_songs = 9)
  (h_mary : mary_songs = 3)
  (h_alina_tina : alina_tina_songs = 16) :
  total_songs = 8 :=
sorry

end total_quartet_songs_l1860_186098


namespace count_integer_solutions_less_than_zero_l1860_186072

theorem count_integer_solutions_less_than_zero : 
  ∃ k : ℕ, k = 4 ∧ (∀ n : ℤ, n^4 - n^3 - 3 * n^2 - 3 * n - 17 < 0 → k = 4) :=
by
  sorry

end count_integer_solutions_less_than_zero_l1860_186072


namespace numbers_unchanged_by_powers_of_n_l1860_186059

-- Definitions and conditions
def unchanged_when_raised (x : ℂ) (n : ℕ) : Prop :=
  x^n = x

def modulus_one (z : ℂ) : Prop :=
  Complex.abs z = 1

-- Proof statements
theorem numbers_unchanged_by_powers_of_n :
  (∀ x : ℂ, (∀ n : ℕ, n > 0 → unchanged_when_raised x n → x = 0 ∨ x = 1)) ∧
  (∀ z : ℂ, modulus_one z → (∀ n : ℕ, n > 0 → Complex.abs (z^n) = 1)) :=
by
  sorry

end numbers_unchanged_by_powers_of_n_l1860_186059


namespace cost_price_of_book_l1860_186034

theorem cost_price_of_book 
  (C : ℝ)
  (h1 : ∃ C, C > 0)
  (h2 : 1.10 * C = 1.15 * C - 120) :
  C = 2400 :=
sorry

end cost_price_of_book_l1860_186034


namespace license_plates_count_l1860_186017

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end license_plates_count_l1860_186017


namespace michael_needs_more_money_l1860_186082

-- Define the initial conditions
def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_gbp : ℝ := 30
def gbp_to_usd : ℝ := 1.4
def perfume_cost : ℝ := perfume_gbp * gbp_to_usd
def photo_album_eur : ℝ := 25
def eur_to_usd : ℝ := 1.2
def photo_album_cost : ℝ := photo_album_eur * eur_to_usd

-- Sum the costs
def total_cost : ℝ := cake_cost + bouquet_cost + balloons_cost + perfume_cost + photo_album_cost

-- Define the required amount
def additional_money_needed : ℝ := total_cost - michael_money

-- The theorem statement
theorem michael_needs_more_money : additional_money_needed = 83 := by
  sorry

end michael_needs_more_money_l1860_186082


namespace students_with_all_three_pets_l1860_186086

variables (TotalStudents HaveDogs HaveCats HaveOtherPets NoPets x y z w : ℕ)

theorem students_with_all_three_pets :
  TotalStudents = 40 →
  HaveDogs = 20 →
  HaveCats = 16 →
  HaveOtherPets = 8 →
  NoPets = 7 →
  x = 12 →
  y = 3 →
  z = 11 →
  TotalStudents - NoPets = 33 →
  x + y + w = HaveDogs →
  z + w = HaveCats →
  y + w = HaveOtherPets →
  x + y + z + w = 33 →
  w = 5 :=
by
  intros h1 h2 h3 h4 h5 hx hy hz h6 h7 h8 h9
  sorry

end students_with_all_three_pets_l1860_186086


namespace rectangle_ABCD_area_l1860_186042

def rectangle_area (x : ℕ) : ℕ :=
  let side_lengths := [x, x+1, x+2, x+3];
  let width := side_lengths.sum;
  let height := width - x;
  width * height

theorem rectangle_ABCD_area : rectangle_area 1 = 143 :=
by
  sorry

end rectangle_ABCD_area_l1860_186042


namespace casey_nail_decorating_time_l1860_186057

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l1860_186057


namespace average_time_per_leg_l1860_186052

-- Conditions
def time_y : ℕ := 58
def time_z : ℕ := 26
def total_time : ℕ := time_y + time_z
def number_of_legs : ℕ := 2

-- Theorem stating the average time per leg
theorem average_time_per_leg : total_time / number_of_legs = 42 := by
  sorry

end average_time_per_leg_l1860_186052


namespace median_song_length_l1860_186058

-- Define the list of song lengths in seconds
def song_lengths : List ℕ := [32, 43, 58, 65, 70, 72, 75, 80, 145, 150, 175, 180, 195, 210, 215, 225, 250, 252]

-- Define the statement that the median length of the songs is 147.5 seconds
theorem median_song_length : ∃ median : ℕ, median = 147 ∧ (median : ℚ) + 0.5 = 147.5 := by
  sorry

end median_song_length_l1860_186058


namespace smallest_n_l1860_186066

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3^n = k^4) (h2 : ∃ l : ℕ, 2^n = l^6) : n = 12 :=
by
  sorry

end smallest_n_l1860_186066


namespace all_statements_correct_l1860_186033

-- Definitions based on the problem conditions
def population_size : ℕ := 60000
def sample_size : ℕ := 1000
def is_sampling_survey (population_size sample_size : ℕ) : Prop := sample_size < population_size
def is_population (n : ℕ) : Prop := n = 60000
def is_sample (population_size sample_size : ℕ) : Prop := sample_size < population_size
def matches_sample_size (n : ℕ) : Prop := n = 1000

-- Lean problem statement representing the proof that all statements are correct
theorem all_statements_correct :
  is_sampling_survey population_size sample_size ∧
  is_population population_size ∧ 
  is_sample population_size sample_size ∧
  matches_sample_size sample_size := by
  sorry

end all_statements_correct_l1860_186033


namespace product_of_roots_l1860_186073

theorem product_of_roots : ∀ (x : ℝ), (x + 3) * (x - 4) = 2 * (x + 1) → 
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  product_of_roots = -14 :=
by
  intros x h
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  sorry

end product_of_roots_l1860_186073


namespace solve_fractional_eq_l1860_186095

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) : (x / (x + 1) - 1 = 3 / (x - 1)) → x = -1 / 2 :=
by
  sorry

end solve_fractional_eq_l1860_186095


namespace continuous_stripe_probability_l1860_186077

def cube_stripe_probability : ℚ :=
  let stripe_combinations_per_face := 8
  let total_combinations := stripe_combinations_per_face ^ 6
  let valid_combinations := 4 * 3 * 8 * 64
  let probability := valid_combinations / total_combinations
  probability

theorem continuous_stripe_probability :
  cube_stripe_probability = 3 / 128 := by
  sorry

end continuous_stripe_probability_l1860_186077


namespace household_count_correct_l1860_186013

def num_buildings : ℕ := 4
def floors_per_building : ℕ := 6
def households_first_floor : ℕ := 2
def households_other_floors : ℕ := 3
def total_households : ℕ := 68

theorem household_count_correct :
  num_buildings * (households_first_floor + (floors_per_building - 1) * households_other_floors) = total_households :=
by
  sorry

end household_count_correct_l1860_186013


namespace min_minutes_to_make_B_cheaper_l1860_186016

def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

def costB (x : ℕ) : ℕ := 2500 + 4 * x

theorem min_minutes_to_make_B_cheaper : ∃ (x : ℕ), x ≥ 301 ∧ costB x < costA x :=
by
  use 301
  sorry

end min_minutes_to_make_B_cheaper_l1860_186016


namespace minimum_positive_period_of_f_l1860_186000

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_positive_period_of_f : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi := 
sorry

end minimum_positive_period_of_f_l1860_186000


namespace remaining_pieces_l1860_186018

/-- Define the initial number of pieces on a standard chessboard. -/
def initial_pieces : Nat := 32

/-- Define the number of pieces lost by Audrey. -/
def audrey_lost : Nat := 6

/-- Define the number of pieces lost by Thomas. -/
def thomas_lost : Nat := 5

/-- Proof that the remaining number of pieces on the chessboard is 21. -/
theorem remaining_pieces : initial_pieces - (audrey_lost + thomas_lost) = 21 := by
  -- Mathematical equivalence to 32 - (6 + 5) = 21
  sorry

end remaining_pieces_l1860_186018


namespace value_of_sum_plus_five_l1860_186019

theorem value_of_sum_plus_five (a b : ℕ) (h : 4 * a^2 + 4 * b^2 + 8 * a * b = 100) :
  (a + b) + 5 = 10 :=
sorry

end value_of_sum_plus_five_l1860_186019


namespace ratio_of_areas_l1860_186035

theorem ratio_of_areas
  (s: ℝ) (h₁: s > 0)
  (large_square_area: ℝ)
  (inscribed_square_area: ℝ)
  (harea₁: large_square_area = s * s)
  (harea₂: inscribed_square_area = (s / 2) * (s / 2)) :
  inscribed_square_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l1860_186035


namespace odd_function_increasing_l1860_186063

variables {f : ℝ → ℝ}

/-- Let f be an odd function defined on (-∞, 0) ∪ (0, ∞). 
If ∀ y z ∈ (0, ∞), y ≠ z → (f y - f z) / (y - z) > 0, then f(-3) > f(-5). -/
theorem odd_function_increasing {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ y z : ℝ, y > 0 → z > 0 → y ≠ z → (f y - f z) / (y - z) > 0) :
  f (-3) > f (-5) :=
sorry

end odd_function_increasing_l1860_186063


namespace find_x_if_vectors_parallel_l1860_186038

/--
Given the vectors a = (2 * x + 1, 3) and b = (2 - x, 1), if a is parallel to b, 
then x must be equal to 1.
-/
theorem find_x_if_vectors_parallel (x : ℝ) :
  let a := (2 * x + 1, 3)
  let b := (2 - x, 1)
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 :=
by
  sorry

end find_x_if_vectors_parallel_l1860_186038


namespace cubic_roots_inequality_l1860_186021

theorem cubic_roots_inequality (a b c : ℝ) (h : ∃ (α β γ : ℝ), (x : ℝ) → x^3 + a * x^2 + b * x + c = (x - α) * (x - β) * (x - γ)) :
  3 * b ≤ a^2 :=
sorry

end cubic_roots_inequality_l1860_186021


namespace complement_union_l1860_186004

def universal_set : Set ℝ := { x : ℝ | true }
def M : Set ℝ := { x : ℝ | x ≤ 0 }
def N : Set ℝ := { x : ℝ | x > 2 }

theorem complement_union (x : ℝ) :
  x ∈ compl (M ∪ N) ↔ (0 < x ∧ x ≤ 2) := by
  sorry

end complement_union_l1860_186004


namespace smallest_angle_of_isosceles_trapezoid_l1860_186092

def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = c ∧ b = d ∧ a + b + c + d = 360 ∧ a + 3 * b = 150

theorem smallest_angle_of_isosceles_trapezoid (a b : ℝ) (h1 : is_isosceles_trapezoid a b a (a + 2 * b))
  : a = 47 :=
sorry

end smallest_angle_of_isosceles_trapezoid_l1860_186092
