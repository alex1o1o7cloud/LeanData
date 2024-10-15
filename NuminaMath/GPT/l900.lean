import Mathlib

namespace NUMINAMATH_GPT_order_of_f0_f1_f_2_l900_90026

noncomputable def f (m x : ℝ) := (m-1) * x^2 + 6 * m * x + 2

theorem order_of_f0_f1_f_2 (m : ℝ) (h_even : ∀ x : ℝ, f m x = f m (-x)) :
  m = 0 → f m (-2) < f m 1 ∧ f m 1 < f m 0 :=
by 
  sorry

end NUMINAMATH_GPT_order_of_f0_f1_f_2_l900_90026


namespace NUMINAMATH_GPT_fraction_value_l900_90006

theorem fraction_value : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l900_90006


namespace NUMINAMATH_GPT_selling_price_is_correct_l900_90023

noncomputable def purchase_price : ℝ := 36400
noncomputable def repair_costs : ℝ := 8000
noncomputable def profit_percent : ℝ := 54.054054054054056

noncomputable def total_cost := purchase_price + repair_costs
noncomputable def selling_price := total_cost * (1 + profit_percent / 100)

theorem selling_price_is_correct :
    selling_price = 68384 := by
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l900_90023


namespace NUMINAMATH_GPT_distance_equal_axes_l900_90087

theorem distance_equal_axes (m : ℝ) :
  (abs (3 * m + 1) = abs (2 * m - 5)) ↔ (m = -6 ∨ m = 4 / 5) :=
by 
  sorry

end NUMINAMATH_GPT_distance_equal_axes_l900_90087


namespace NUMINAMATH_GPT_tiles_cover_the_floor_l900_90095

theorem tiles_cover_the_floor
  (n : ℕ)
  (h : 2 * n - 1 = 101)
  : n ^ 2 = 2601 := sorry

end NUMINAMATH_GPT_tiles_cover_the_floor_l900_90095


namespace NUMINAMATH_GPT_fair_game_x_value_l900_90052

theorem fair_game_x_value (x : ℕ) (h : x + 2 * x + 2 * x = 15) : x = 3 := 
by sorry

end NUMINAMATH_GPT_fair_game_x_value_l900_90052


namespace NUMINAMATH_GPT_mart_income_more_than_tim_l900_90013

variable (J : ℝ) -- Let's denote Juan's income as J
def T : ℝ := J - 0.40 * J -- Tim's income is 40 percent less than Juan's income
def M : ℝ := 0.78 * J -- Mart's income is 78 percent of Juan's income

theorem mart_income_more_than_tim : (M - T) / T * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_mart_income_more_than_tim_l900_90013


namespace NUMINAMATH_GPT_problem_statement_period_property_symmetry_property_zero_property_l900_90029

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem problem_statement : ¬(∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + ε))
  → ∃ x : ℝ, f (x + Real.pi) = 0 :=
by
  intro h
  use Real.pi / 6
  sorry

theorem period_property : ∀ k : ℤ, f (x + 2 * k * Real.pi) = f x :=
by
  intro k
  sorry

theorem symmetry_property : ∀ y : ℝ, f (8 * Real.pi / 3 - y) = f (8 * Real.pi / 3 + y) :=
by
  intro y
  sorry

theorem zero_property : f (Real.pi / 6 + Real.pi) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_period_property_symmetry_property_zero_property_l900_90029


namespace NUMINAMATH_GPT_find_a_l900_90039

theorem find_a (x y a : ℝ) (h1 : x + 2 * y = 2) (h2 : 2 * x + y = a) (h3 : x + y = 5) : a = 13 := by
  sorry

end NUMINAMATH_GPT_find_a_l900_90039


namespace NUMINAMATH_GPT_fraction_of_total_cost_for_raisins_l900_90000

-- Define variables and constants
variable (R : ℝ) -- cost of a pound of raisins

-- Define the conditions as assumptions
variable (cost_of_nuts : ℝ := 4 * R)
variable (cost_of_dried_berries : ℝ := 2 * R)

variable (total_cost : ℝ := 3 * R + 4 * cost_of_nuts + 2 * cost_of_dried_berries)
variable (cost_of_raisins : ℝ := 3 * R)

-- Main statement that we want to prove
theorem fraction_of_total_cost_for_raisins :
  cost_of_raisins / total_cost = 3 / 23 := by
  sorry

end NUMINAMATH_GPT_fraction_of_total_cost_for_raisins_l900_90000


namespace NUMINAMATH_GPT_average_value_of_series_l900_90082

theorem average_value_of_series (z : ℤ) :
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sum_series / n = 21 * z^2 :=
by
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sorry

end NUMINAMATH_GPT_average_value_of_series_l900_90082


namespace NUMINAMATH_GPT_orange_juice_production_correct_l900_90066

noncomputable def orangeJuiceProduction (total_oranges : Float) (export_percent : Float) (juice_percent : Float) : Float :=
  let remaining_oranges := total_oranges * (1 - export_percent / 100)
  let juice_oranges := remaining_oranges * (juice_percent / 100)
  Float.round (juice_oranges * 10) / 10

theorem orange_juice_production_correct :
  orangeJuiceProduction 8.2 30 40 = 2.3 := by
  sorry

end NUMINAMATH_GPT_orange_juice_production_correct_l900_90066


namespace NUMINAMATH_GPT_solve_for_b_l900_90022

theorem solve_for_b (b : ℝ) : (∃ y x : ℝ, 4 * y - 2 * x - 6 = 0 ∧ 5 * y + b * x + 1 = 0) → b = 10 :=
by sorry

end NUMINAMATH_GPT_solve_for_b_l900_90022


namespace NUMINAMATH_GPT_unique_pair_exists_l900_90049

theorem unique_pair_exists (n : ℕ) (hn : n > 0) : 
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ 0 ≤ l ∧ l < k :=
sorry

end NUMINAMATH_GPT_unique_pair_exists_l900_90049


namespace NUMINAMATH_GPT_bricklayer_hours_l900_90086

theorem bricklayer_hours
  (B E : ℝ)
  (h1 : B + E = 90)
  (h2 : 12 * B + 16 * E = 1350) :
  B = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_bricklayer_hours_l900_90086


namespace NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg_30_l900_90056

theorem largest_multiple_of_7_less_than_neg_30 (m : ℤ) (h1 : m % 7 = 0) (h2 : m < -30) : m = -35 :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg_30_l900_90056


namespace NUMINAMATH_GPT_annual_increase_rate_l900_90003

theorem annual_increase_rate (r : ℝ) (h : 70400 * (1 + r)^2 = 89100) : r = 0.125 :=
sorry

end NUMINAMATH_GPT_annual_increase_rate_l900_90003


namespace NUMINAMATH_GPT_find_max_m_l900_90041

-- We define real numbers a, b, c that satisfy the given conditions
variable (a b c m : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 12)
variable (h_prod_sum : a * b + b * c + c * a = 30)
variable (m_def : m = min (a * b) (min (b * c) (c * a)))

-- We state the main theorem to be proved
theorem find_max_m : m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_max_m_l900_90041


namespace NUMINAMATH_GPT_man_owns_fraction_of_business_l900_90089

theorem man_owns_fraction_of_business
  (x : ℚ)
  (H1 : (3 / 4) * (x * 90000) = 45000)
  (H2 : x * 90000 = y) : 
  x = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_man_owns_fraction_of_business_l900_90089


namespace NUMINAMATH_GPT_hyperbola_properties_l900_90032

theorem hyperbola_properties :
  (∃ x y : Real,
    (x^2 / 4 - y^2 / 2 = 1) ∧
    (∃ a b c e : Real,
      2 * a = 4 ∧
      2 * b = 2 * Real.sqrt 2 ∧
      c = Real.sqrt (a^2 + b^2) ∧
      2 * c = 2 * Real.sqrt 6 ∧
      e = c / a)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l900_90032


namespace NUMINAMATH_GPT_sum_le_xyz_plus_two_l900_90040

theorem sum_le_xyz_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ xyz + 2 := 
sorry

end NUMINAMATH_GPT_sum_le_xyz_plus_two_l900_90040


namespace NUMINAMATH_GPT_value_range_of_f_l900_90044

noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

theorem value_range_of_f : Set.range (fun x => f x) ∩ Set.Icc 3 6 = Set.Icc 1 4 :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_l900_90044


namespace NUMINAMATH_GPT_smallest_integer_l900_90075

theorem smallest_integer (M : ℕ) :
  (M % 4 = 3) ∧ (M % 5 = 4) ∧ (M % 6 = 5) ∧ (M % 7 = 6) ∧
  (M % 8 = 7) ∧ (M % 9 = 8) → M = 2519 :=
by sorry

end NUMINAMATH_GPT_smallest_integer_l900_90075


namespace NUMINAMATH_GPT_santiago_stay_in_australia_l900_90063

/-- Santiago leaves his home country in the month of January,
    stays in Australia for a few months,
    and returns on the same date in the month of December.
    Prove that Santiago stayed in Australia for 11 months. -/
theorem santiago_stay_in_australia :
  ∃ (months : ℕ), months = 11 ∧
  (months = if (departure_month = 1) ∧ (return_month = 12) then 11 else 0) :=
by sorry

end NUMINAMATH_GPT_santiago_stay_in_australia_l900_90063


namespace NUMINAMATH_GPT_algebraic_expression_simplification_l900_90028

theorem algebraic_expression_simplification (x y : ℝ) (h : x + y = 1) : x^3 + y^3 + 3 * x * y = 1 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_simplification_l900_90028


namespace NUMINAMATH_GPT_two_p_in_S_l900_90035

def is_in_S (a b : ℤ) : Prop :=
  ∃ k : ℤ, k = a^2 + 5 * b^2 ∧ Int.gcd a b = 1

def S : Set ℤ := { x | ∃ a b : ℤ, is_in_S a b ∧ a^2 + 5 * b^2 = x }

theorem two_p_in_S (k p n : ℤ) (hp1 : p = 4 * n + 3) (hp2 : Nat.Prime (Int.natAbs p))
  (hk : 0 < k) (hkp : k * p ∈ S) : 2 * p ∈ S := 
sorry

end NUMINAMATH_GPT_two_p_in_S_l900_90035


namespace NUMINAMATH_GPT_a_greater_than_1_and_b_less_than_1_l900_90069

theorem a_greater_than_1_and_b_less_than_1
  (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∧ b < 1 :=
by
  sorry

end NUMINAMATH_GPT_a_greater_than_1_and_b_less_than_1_l900_90069


namespace NUMINAMATH_GPT_sum_red_equals_sum_blue_l900_90019

variable (r1 r2 r3 r4 b1 b2 b3 b4 w1 w2 w3 w4 : ℝ)

theorem sum_red_equals_sum_blue (h : (r1 + w1 / 2) + (r2 + w2 / 2) + (r3 + w3 / 2) + (r4 + w4 / 2) 
                                 = (b1 + w1 / 2) + (b2 + w2 / 2) + (b3 + w3 / 2) + (b4 + w4 / 2)) : 
  r1 + r2 + r3 + r4 = b1 + b2 + b3 + b4 :=
by sorry

end NUMINAMATH_GPT_sum_red_equals_sum_blue_l900_90019


namespace NUMINAMATH_GPT_minimum_filtrations_needed_l900_90047

theorem minimum_filtrations_needed (I₀ I_n : ℝ) (n : ℕ) (h1 : I₀ = 0.02) (h2 : I_n ≤ 0.001) (h3 : I_n = I₀ * 0.5 ^ n) :
  n = 8 := by
sorry

end NUMINAMATH_GPT_minimum_filtrations_needed_l900_90047


namespace NUMINAMATH_GPT_symmetric_point_to_origin_l900_90094

theorem symmetric_point_to_origin (a b : ℝ) :
  (∃ (a b : ℝ), (a / 2) - 2 * (b / 2) + 2 = 0 ∧ (b / a) * (1 / 2) = -1) →
  (a = -4 / 5 ∧ b = 8 / 5) :=
sorry

end NUMINAMATH_GPT_symmetric_point_to_origin_l900_90094


namespace NUMINAMATH_GPT_polygon_interior_angle_sum_l900_90078

theorem polygon_interior_angle_sum (n : ℕ) (h : (n-1) * 180 = 2400 + 120) : n = 16 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_angle_sum_l900_90078


namespace NUMINAMATH_GPT_problem_solution_l900_90083

theorem problem_solution {n : ℕ} :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l900_90083


namespace NUMINAMATH_GPT_number_of_days_l900_90014

variables (S Wx Wy : ℝ)

-- Given conditions
def condition1 : Prop := S = 36 * Wx
def condition2 : Prop := S = 45 * Wy

-- The lean statement to prove the number of days D = 20
theorem number_of_days (h1 : condition1 S Wx) (h2 : condition2 S Wy) : 
  S / (Wx + Wy) = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_l900_90014


namespace NUMINAMATH_GPT_more_girls_than_boys_l900_90060

theorem more_girls_than_boys (girls boys total_pupils : ℕ) (h1 : girls = 692) (h2 : total_pupils = 926) (h3 : boys = total_pupils - girls) : girls - boys = 458 :=
by
  sorry

end NUMINAMATH_GPT_more_girls_than_boys_l900_90060


namespace NUMINAMATH_GPT_trey_will_sell_bracelets_for_days_l900_90008

def cost : ℕ := 112
def price_per_bracelet : ℕ := 1
def bracelets_per_day : ℕ := 8

theorem trey_will_sell_bracelets_for_days :
  ∃ d : ℕ, d = cost / (price_per_bracelet * bracelets_per_day) ∧ d = 14 := by
  sorry

end NUMINAMATH_GPT_trey_will_sell_bracelets_for_days_l900_90008


namespace NUMINAMATH_GPT_solve_for_x_l900_90077

theorem solve_for_x (x : ℝ) (h : 10 - x = 15) : x = -5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l900_90077


namespace NUMINAMATH_GPT_cosine_of_angle_in_second_quadrant_l900_90038

theorem cosine_of_angle_in_second_quadrant
  (α : ℝ)
  (h1 : Real.sin α = 1 / 3)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_cosine_of_angle_in_second_quadrant_l900_90038


namespace NUMINAMATH_GPT_coordinates_of_P_l900_90051

structure Point (α : Type) [LinearOrderedField α] :=
  (x : α)
  (y : α)

def in_fourth_quadrant {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  P.x > 0 ∧ P.y < 0

def distance_to_axes_is_4 {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  abs P.x = 4 ∧ abs P.y = 4

theorem coordinates_of_P {α : Type} [LinearOrderedField α] (P : Point α) :
  in_fourth_quadrant P ∧ distance_to_axes_is_4 P → P = ⟨4, -4⟩ :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l900_90051


namespace NUMINAMATH_GPT_incorrect_games_less_than_three_fourths_l900_90059

/-- In a round-robin chess tournament, each participant plays against every other participant exactly once.
A win earns one point, a draw earns half a point, and a loss earns zero points.
We will call a game incorrect if the player who won the game ends up with fewer total points than the player who lost.

1. Prove that incorrect games make up less than 3/4 of the total number of games in the tournament.
2. Prove that in part (1), the number 3/4 cannot be replaced with a smaller number.
--/
theorem incorrect_games_less_than_three_fourths {n : ℕ} (h : n > 1) :
  ∃ m, (∃ (incorrect_games total_games : ℕ), m = incorrect_games ∧ total_games = (n * (n - 1)) / 2 
    ∧ (incorrect_games : ℚ) / total_games < 3 / 4) 
    ∧ (∀ m' : ℚ, m' ≥ 0 → m = incorrect_games ∧ (incorrect_games : ℚ) / total_games < m' → m' ≥ 3 / 4) :=
sorry

end NUMINAMATH_GPT_incorrect_games_less_than_three_fourths_l900_90059


namespace NUMINAMATH_GPT_greatest_possible_xy_value_l900_90045

-- Define the conditions
variables (a b c d x y : ℕ)
variables (h1 : a < b) (h2 : b < c) (h3 : c < d)
variables (sums : Finset ℕ) (hsums : sums = {189, 320, 287, 234, x, y})

-- Define the goal statement to prove
theorem greatest_possible_xy_value : x + y = 791 :=
sorry

end NUMINAMATH_GPT_greatest_possible_xy_value_l900_90045


namespace NUMINAMATH_GPT_exists_not_odd_l900_90091

variable (f : ℝ → ℝ)

-- Define the condition that f is not an odd function
def not_odd_function := ¬ (∀ x : ℝ, f (-x) = -f x)

-- Lean statement to prove the correct answer
theorem exists_not_odd (h : not_odd_function f) : ∃ x : ℝ, f (-x) ≠ -f x :=
sorry

end NUMINAMATH_GPT_exists_not_odd_l900_90091


namespace NUMINAMATH_GPT_total_number_of_questions_l900_90031

/-
  Given:
    1. There are 20 type A problems.
    2. Type A problems require twice as much time as type B problems.
    3. 32.73 minutes are spent on type A problems.
    4. Total examination time is 3 hours.

  Prove that the total number of questions is 199.
-/

theorem total_number_of_questions
  (type_A_problems : ℕ)
  (type_B_to_A_time_ratio : ℝ)
  (time_spent_on_type_A : ℝ)
  (total_exam_time_hours : ℝ)
  (total_number_of_questions : ℕ)
  (h_type_A_problems : type_A_problems = 20)
  (h_time_ratio : type_B_to_A_time_ratio = 2)
  (h_time_spent_on_type_A : time_spent_on_type_A = 32.73)
  (h_total_exam_time_hours : total_exam_time_hours = 3) :
  total_number_of_questions = 199 := 
sorry

end NUMINAMATH_GPT_total_number_of_questions_l900_90031


namespace NUMINAMATH_GPT_problem_solution_l900_90081

theorem problem_solution
  (x y : ℝ)
  (h : 5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0) :
  (x - y) ^ 2007 = -1 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l900_90081


namespace NUMINAMATH_GPT_final_quantity_of_milk_l900_90080

-- Initially, a vessel is filled with 45 litres of pure milk
def initial_milk : Nat := 45

-- First operation: removing 9 litres of milk and replacing with water
def first_operation_milk(initial_milk : Nat) : Nat := initial_milk - 9
def first_operation_water : Nat := 9

-- Second operation: removing 9 litres of the mixture and replacing with water
def milk_fraction_mixture(milk : Nat) (total : Nat) : Rat := milk / total
def water_fraction_mixture(water : Nat) (total : Nat) : Rat := water / total

def second_operation_milk(milk : Nat) (total : Nat) (removed : Nat) : Rat := 
  milk - (milk_fraction_mixture milk total) * removed
def second_operation_water(water : Nat) (total : Nat) (removed : Nat) : Rat := 
  water - (water_fraction_mixture water total) * removed + removed

-- Prove the final quantity of milk
theorem final_quantity_of_milk : second_operation_milk 36 45 9 = 28.8 := by
  sorry

end NUMINAMATH_GPT_final_quantity_of_milk_l900_90080


namespace NUMINAMATH_GPT_line_equation_l900_90055

theorem line_equation (P : ℝ × ℝ) (slope : ℝ) (hP : P = (-2, 0)) (hSlope : slope = 3) :
    ∃ (a b : ℝ), ∀ x y : ℝ, y = a * x + b ↔ P.1 = -2 ∧ P.2 = 0 ∧ slope = 3 ∧ y = 3 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l900_90055


namespace NUMINAMATH_GPT_base_seven_sum_l900_90020

def base_seven_sum_of_product (n m : ℕ) : ℕ :=
  let product := n * m
  let digits := product.digits 7
  digits.sum

theorem base_seven_sum (k l : ℕ) (hk : k = 5 * 7 + 3) (hl : l = 343) :
  base_seven_sum_of_product k l = 11 := by
  sorry

end NUMINAMATH_GPT_base_seven_sum_l900_90020


namespace NUMINAMATH_GPT_factorize_expression_l900_90058

theorem factorize_expression (x y : ℝ) : 25 * x - x * y ^ 2 = x * (5 + y) * (5 - y) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l900_90058


namespace NUMINAMATH_GPT_stormi_additional_money_needed_l900_90053

noncomputable def earnings_from_jobs : ℝ :=
  let washing_cars := 5 * 8.50
  let walking_dogs := 4 * 6.75
  let mowing_lawns := 3 * 12.25
  let gardening := 2 * 7.40
  washing_cars + walking_dogs + mowing_lawns + gardening

noncomputable def discounted_prices : ℝ :=
  let bicycle := 150.25 * (1 - 0.15)
  let helmet := 35.75 - 5.00
  let lock := 24.50
  bicycle + helmet + lock

noncomputable def total_cost_after_tax : ℝ :=
  let cost_before_tax := discounted_prices
  cost_before_tax * 1.05

noncomputable def amount_needed : ℝ :=
  total_cost_after_tax - earnings_from_jobs

theorem stormi_additional_money_needed : amount_needed = 71.06 := by
  sorry

end NUMINAMATH_GPT_stormi_additional_money_needed_l900_90053


namespace NUMINAMATH_GPT_balls_in_third_pile_l900_90027

theorem balls_in_third_pile (a b c x : ℕ) (h1 : a + b + c = 2012) (h2 : b - x = 17) (h3 : a - x = 2 * (c - x)) : c = 665 := by
  sorry

end NUMINAMATH_GPT_balls_in_third_pile_l900_90027


namespace NUMINAMATH_GPT_num_ordered_pairs_l900_90034

theorem num_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) :
  (M * N = 32) → ∃ (k : ℕ), k = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_l900_90034


namespace NUMINAMATH_GPT_point_P_position_l900_90093

variable {a b c d : ℝ}
variable (h1: a ≠ b) (h2: a ≠ c) (h3: a ≠ d) (h4: b ≠ c) (h5: b ≠ d) (h6: c ≠ d)

theorem point_P_position (P : ℝ) (hP: b < P ∧ P < c) (hRatio: (|a - P| / |P - d|) = (|b - P| / |P - c|)) : 
  P = (a * c - b * d) / (a - b + c - d) := 
by
  sorry

end NUMINAMATH_GPT_point_P_position_l900_90093


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l900_90064

theorem arithmetic_sequence_nth_term (a b c n : ℕ) (x: ℕ)
  (h1: a = 3*x - 4)
  (h2: b = 6*x - 17)
  (h3: c = 4*x + 5)
  (h4: b - a = c - b)
  (h5: a + (n - 1) * (b - a) = 4021) : 
  n = 502 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l900_90064


namespace NUMINAMATH_GPT_problem_statement_l900_90062

theorem problem_statement :
  ∃ p q r : ℤ,
    (∀ x : ℝ, (x^2 + 19*x + 88 = (x + p) * (x + q)) ∧ (x^2 - 23*x + 132 = (x - q) * (x - r))) →
      p + q + r = 31 :=
sorry

end NUMINAMATH_GPT_problem_statement_l900_90062


namespace NUMINAMATH_GPT_continuity_at_x_2_l900_90097

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem continuity_at_x_2 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_x_2_l900_90097


namespace NUMINAMATH_GPT_distance_between_city_and_village_l900_90002

variables (S x y : ℝ)

theorem distance_between_city_and_village (h1 : S / 2 - 2 = y * S / (2 * x))
    (h2 : 2 * S / 3 + 2 = x * S / (3 * y)) : S = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_city_and_village_l900_90002


namespace NUMINAMATH_GPT_number_of_girls_in_school_l900_90085

-- Variables representing the population and the sample.
variables (total_students sample_size boys_sample girls_sample : ℕ)

-- Initial conditions.
def initial_conditions := 
  total_students = 1600 ∧ 
  sample_size = 200 ∧
  girls_sample = 90 ∧
  boys_sample = 110 ∧
  (girls_sample + 20 = boys_sample)

-- Statement to prove.
theorem number_of_girls_in_school (x: ℕ) 
  (h : initial_conditions total_students sample_size boys_sample girls_sample) :
  x = 720 :=
by {
  -- Obligatory proof omitted.
  sorry
}

end NUMINAMATH_GPT_number_of_girls_in_school_l900_90085


namespace NUMINAMATH_GPT_binomial_60_3_eq_34220_l900_90001

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_GPT_binomial_60_3_eq_34220_l900_90001


namespace NUMINAMATH_GPT_sqrt_arithmetic_identity_l900_90042

theorem sqrt_arithmetic_identity : 4 * (Real.sqrt 2) * (Real.sqrt 3) - (Real.sqrt 12) / (Real.sqrt 2) + (Real.sqrt 24) = 5 * (Real.sqrt 6) := by
  sorry

end NUMINAMATH_GPT_sqrt_arithmetic_identity_l900_90042


namespace NUMINAMATH_GPT_simplify_expression_l900_90061

variable (x y : ℝ)

theorem simplify_expression : 2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l900_90061


namespace NUMINAMATH_GPT_find_deaf_students_l900_90018

-- Definitions based on conditions
variables (B D : ℕ)
axiom deaf_students_triple_blind_students : D = 3 * B
axiom total_students : D + B = 240

-- Proof statement
theorem find_deaf_students (h1 : D = 3 * B) (h2 : D + B = 240) : D = 180 :=
sorry

end NUMINAMATH_GPT_find_deaf_students_l900_90018


namespace NUMINAMATH_GPT_find_hourly_charge_l900_90050

variable {x : ℕ}

--Assumptions and conditions
def fixed_charge := 17
def total_paid := 80
def rental_hours := 9

-- Proof problem
theorem find_hourly_charge (h : fixed_charge + rental_hours * x = total_paid) : x = 7 :=
sorry

end NUMINAMATH_GPT_find_hourly_charge_l900_90050


namespace NUMINAMATH_GPT_intersection_point_and_distance_l900_90009

/-- Define the points A, B, C, D, and M based on the specified conditions. --/
def A := (0, 3)
def B := (6, 3)
def C := (6, 0)
def D := (0, 0)
def M := (3, 0)

/-- Define the equations of the circles. --/
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2.25
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 25

/-- The point P that is one of the intersection points of the two circles. --/
def P := (2, 1.5)

/-- Define the line AD as the y-axis. --/
def AD := 0

/-- Calculate the distance from point P to the y-axis (AD). --/
def distance_to_ad (x : ℝ) := |x|

theorem intersection_point_and_distance :
  circle1 (2 : ℝ) (1.5 : ℝ) ∧ circle2 (2 : ℝ) (1.5 : ℝ) ∧ distance_to_ad 2 = 2 :=
by
  unfold circle1 circle2 distance_to_ad
  norm_num
  sorry

end NUMINAMATH_GPT_intersection_point_and_distance_l900_90009


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l900_90021

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l900_90021


namespace NUMINAMATH_GPT_geom_sequence_a_n_l900_90084

variable {a : ℕ → ℝ}

-- Given conditions
def is_geom_seq (a : ℕ → ℝ) : Prop :=
  |a 1| = 1 ∧ a 5 = -8 * a 2 ∧ a 5 > a 2

-- Statement to prove
theorem geom_sequence_a_n (h : is_geom_seq a) : ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end NUMINAMATH_GPT_geom_sequence_a_n_l900_90084


namespace NUMINAMATH_GPT_find_P_l900_90054

noncomputable def P (x : ℝ) : ℝ :=
  4 * x^3 - 6 * x^2 - 12 * x

theorem find_P (a b c : ℝ) (h_root : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_roots : ∀ x, x^3 - 2 * x^2 - 4 * x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c)
  (h_Pa : P a = b + 2 * c)
  (h_Pb : P b = 2 * a + c)
  (h_Pc : P c = a + 2 * b)
  (h_Psum : P (a + b + c) = -20) :
  ∀ x, P x = 4 * x^3 - 6 * x^2 - 12 * x :=
by
  sorry

end NUMINAMATH_GPT_find_P_l900_90054


namespace NUMINAMATH_GPT_mnp_sum_correct_l900_90072

noncomputable def mnp_sum : ℕ :=
  let m := 1032
  let n := 40
  let p := 3
  m + n + p

theorem mnp_sum_correct : mnp_sum = 1075 := by
  -- Given the conditions, the established value for m, n, and p should sum to 1075
  sorry

end NUMINAMATH_GPT_mnp_sum_correct_l900_90072


namespace NUMINAMATH_GPT_repeated_process_pure_alcohol_l900_90037

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end NUMINAMATH_GPT_repeated_process_pure_alcohol_l900_90037


namespace NUMINAMATH_GPT_total_songs_l900_90046

theorem total_songs (h : ℕ) (m : ℕ) (a : ℕ) (t : ℕ) (P : ℕ)
  (Hh : h = 6) (Hm : m = 3) (Ha : a = 5) 
  (Htotal : P = (h + m + a + t) / 3) 
  (Hdiv : (h + m + a + t) % 3 = 0) : P = 6 := by
  sorry

end NUMINAMATH_GPT_total_songs_l900_90046


namespace NUMINAMATH_GPT_agatha_remaining_amount_l900_90005

theorem agatha_remaining_amount :
  let initial_amount := 60
  let frame_price := 15
  let frame_discount := 0.10 * frame_price
  let frame_final := frame_price - frame_discount
  let wheel_price := 25
  let wheel_discount := 0.05 * wheel_price
  let wheel_final := wheel_price - wheel_discount
  let seat_price := 8
  let seat_discount := 0.15 * seat_price
  let seat_final := seat_price - seat_discount
  let tape_price := 5
  let total_spent := frame_final + wheel_final + seat_final + tape_price
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 10.95 :=
by
  sorry

end NUMINAMATH_GPT_agatha_remaining_amount_l900_90005


namespace NUMINAMATH_GPT_five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l900_90015

theorem five_digit_numbers_greater_than_20314_and_formable_with_0_to_5 :
  (∃ (f : Fin 6 → Fin 5) (n : ℕ), 
    (n = 120 * 3 + 24 * 4 + 6 * 3 - 1) ∧
    (n = 473) ∧ 
    (∀ (x : Fin 6), f x = 0 ∨ f x = 1 ∨ f x = 2 ∨ f x = 3 ∨ f x = 4 ∨ f x = 5) ∧
    (∀ (i j : Fin 5), i ≠ j → f i ≠ f j)) :=
sorry

end NUMINAMATH_GPT_five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l900_90015


namespace NUMINAMATH_GPT_problem_statement_l900_90036

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f'' (x : ℝ) : ℝ := -Real.sin x - Real.cos x

theorem problem_statement (a : ℝ) (h : f'' a = 3 * f a) : 
  (Real.sin a)^2 - 3 / (Real.cos a)^2 + 1 = -14 / 9 := 
sorry

end NUMINAMATH_GPT_problem_statement_l900_90036


namespace NUMINAMATH_GPT_expected_value_coin_flip_l900_90090

-- Definitions based on conditions
def P_heads : ℚ := 2 / 3
def P_tails : ℚ := 1 / 3
def win_heads : ℚ := 4
def lose_tails : ℚ := -9

-- Expected value calculation
def expected_value : ℚ :=
  P_heads * win_heads + P_tails * lose_tails

-- Theorem statement to be proven
theorem expected_value_coin_flip : expected_value = -1 / 3 :=
by sorry

end NUMINAMATH_GPT_expected_value_coin_flip_l900_90090


namespace NUMINAMATH_GPT_sequence_sum_l900_90057

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end NUMINAMATH_GPT_sequence_sum_l900_90057


namespace NUMINAMATH_GPT_div_fractions_eq_l900_90067

theorem div_fractions_eq : (3/7) / (5/2) = 6/35 := 
by sorry

end NUMINAMATH_GPT_div_fractions_eq_l900_90067


namespace NUMINAMATH_GPT_sum_S16_over_S4_l900_90073

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a q : α) (n : ℕ) := a * q^n

def sum_of_first_n_terms (a q : α) (n : ℕ) : α :=
if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem sum_S16_over_S4
  (a q : α)
  (hq : q ≠ 1)
  (h8_over_4 : sum_of_first_n_terms a q 8 / sum_of_first_n_terms a q 4 = 3) :
  sum_of_first_n_terms a q 16 / sum_of_first_n_terms a q 4 = 15 :=
sorry

end NUMINAMATH_GPT_sum_S16_over_S4_l900_90073


namespace NUMINAMATH_GPT_books_left_after_sale_l900_90076

theorem books_left_after_sale (initial_books sold_books books_left : ℕ)
    (h1 : initial_books = 33)
    (h2 : sold_books = 26)
    (h3 : books_left = initial_books - sold_books) :
    books_left = 7 := by
  sorry

end NUMINAMATH_GPT_books_left_after_sale_l900_90076


namespace NUMINAMATH_GPT_triangle_hypotenuse_and_area_l900_90092

theorem triangle_hypotenuse_and_area 
  (A B C D : Type) 
  (CD : ℝ) 
  (angle_A : ℝ) 
  (hypotenuse_AC : ℝ) 
  (area_ABC : ℝ) 
  (h1 : CD = 1) 
  (h2 : angle_A = 45) : 
  hypotenuse_AC = Real.sqrt 2 
  ∧ 
  area_ABC = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_hypotenuse_and_area_l900_90092


namespace NUMINAMATH_GPT_gnollish_valid_sentences_count_is_50_l900_90033

def gnollish_words : List String := ["splargh", "glumph", "amr", "blort"]

def is_valid_sentence (sentence : List String) : Prop :=
  match sentence with
  | [_, "splargh", "glumph"] => False
  | ["splargh", "glumph", _] => False
  | [_, "blort", "amr"] => False
  | ["blort", "amr", _] => False
  | _ => True

def count_valid_sentences (n : Nat) : Nat :=
  (List.replicate n gnollish_words).mapM id |>.length

theorem gnollish_valid_sentences_count_is_50 : count_valid_sentences 3 = 50 :=
by 
  sorry

end NUMINAMATH_GPT_gnollish_valid_sentences_count_is_50_l900_90033


namespace NUMINAMATH_GPT_element_in_set_l900_90048

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def complement_U_M : Set ℕ := {1, 2}

-- The main statement to prove
theorem element_in_set (M : Set ℕ) (h1 : U = {1, 2, 3, 4, 5}) (h2 : U \ M = complement_U_M) : 3 ∈ M := 
sorry

end NUMINAMATH_GPT_element_in_set_l900_90048


namespace NUMINAMATH_GPT_simple_interest_rate_l900_90098

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (hSI : SI = 250) (hP : P = 1500) (hT : T = 5)
  (hSIFormula : SI = (P * R * T) / 100) :
  R = 3.33 := 
by 
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l900_90098


namespace NUMINAMATH_GPT_expression_evaluation_l900_90096

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l900_90096


namespace NUMINAMATH_GPT_total_amount_l900_90099

theorem total_amount (A N J : ℕ) (h1 : A = N - 5) (h2 : J = 4 * N) (h3 : J = 48) : A + N + J = 67 :=
by
  -- Proof will be constructed here
  sorry

end NUMINAMATH_GPT_total_amount_l900_90099


namespace NUMINAMATH_GPT_eddys_climbing_rate_l900_90088

def base_camp_ft := 5000
def departure_time := 6 -- in hours: 6:00 AM
def hillary_climbing_rate := 800 -- ft/hr
def stopping_distance_ft := 1000 -- ft short of summit
def hillary_descending_rate := 1000 -- ft/hr
def passing_time := 12 -- in hours: 12:00 PM

theorem eddys_climbing_rate :
  ∀ (base_ft departure hillary_rate stop_dist descend_rate pass_time : ℕ),
    base_ft = base_camp_ft →
    departure = departure_time →
    hillary_rate = hillary_climbing_rate →
    stop_dist = stopping_distance_ft →
    descend_rate = hillary_descending_rate →
    pass_time = passing_time →
    (pass_time - departure) * hillary_rate - descend_rate * (pass_time - (departure + (base_ft - stop_dist) / hillary_rate)) = 6 * 500 :=
by
  intros
  sorry

end NUMINAMATH_GPT_eddys_climbing_rate_l900_90088


namespace NUMINAMATH_GPT_initial_thickness_of_blanket_l900_90071

theorem initial_thickness_of_blanket (T : ℝ)
  (h : ∀ n, n = 4 → T * 2^n = 48) : T = 3 :=
by
  have h4 := h 4 rfl
  sorry

end NUMINAMATH_GPT_initial_thickness_of_blanket_l900_90071


namespace NUMINAMATH_GPT_first_player_win_condition_l900_90025

def player_one_wins (p q : ℕ) : Prop :=
  p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 4 ∨
  q % 5 = 0 ∨ q % 5 = 1 ∨ q % 5 = 4

theorem first_player_win_condition (p q : ℕ) :
  player_one_wins p q ↔
  (∃ (a b : ℕ), (a, b) = (p, q) ∧ (a % 5 = 0 ∨ a % 5 = 1 ∨ a % 5 = 4 ∨ 
                                     b % 5 = 0 ∨ b % 5 = 1 ∨ b % 5 = 4)) :=
sorry

end NUMINAMATH_GPT_first_player_win_condition_l900_90025


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l900_90004

theorem isosceles_right_triangle_area (h : ℝ) (area : ℝ) (hypotenuse_condition : h = 6 * Real.sqrt 2) : 
  area = 18 :=
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l900_90004


namespace NUMINAMATH_GPT_alicia_taxes_l900_90065

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end NUMINAMATH_GPT_alicia_taxes_l900_90065


namespace NUMINAMATH_GPT_exercise_l900_90079

theorem exercise (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) :=
sorry

end NUMINAMATH_GPT_exercise_l900_90079


namespace NUMINAMATH_GPT_sum_of_triangle_ops_l900_90068

def triangle_op (a b c : ℕ) : ℕ := 2 * a + b - c 

theorem sum_of_triangle_ops : 
  triangle_op 1 2 3 + triangle_op 4 6 5 + triangle_op 2 7 1 = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_triangle_ops_l900_90068


namespace NUMINAMATH_GPT_find_m_for_one_solution_l900_90024

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end NUMINAMATH_GPT_find_m_for_one_solution_l900_90024


namespace NUMINAMATH_GPT_sqrt_144_times_3_squared_l900_90010

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end NUMINAMATH_GPT_sqrt_144_times_3_squared_l900_90010


namespace NUMINAMATH_GPT_solve_for_x_l900_90011

-- Definition of the operation
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- The mathematical statement to be proved
theorem solve_for_x (x : ℝ) (h : otimes x (x - 1) = 3) : x = 2 ∨ x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l900_90011


namespace NUMINAMATH_GPT_number_of_common_tangents_l900_90074

noncomputable def circle1_center : ℝ × ℝ := (-3, 0)
noncomputable def circle1_radius : ℝ := 4

noncomputable def circle2_center : ℝ × ℝ := (0, 3)
noncomputable def circle2_radius : ℝ := 6

theorem number_of_common_tangents 
  (center1 center2 : ℝ × ℝ)
  (radius1 radius2 : ℝ)
  (h_center1: center1 = (-3, 0))
  (h_radius1: radius1 = 4)
  (h_center2: center2 = (0, 3))
  (h_radius2: radius2 = 6) :
  -- The sought number of common tangents between the two circles
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_common_tangents_l900_90074


namespace NUMINAMATH_GPT_sabrina_herbs_l900_90007

theorem sabrina_herbs (S V : ℕ) 
  (h1 : 2 * S = 12)
  (h2 : 12 + S + V = 29) :
  V - S = 5 := by
  sorry

end NUMINAMATH_GPT_sabrina_herbs_l900_90007


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l900_90030

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_123 : a 0 + a 1 + a 2 = -3)
  (h_456 : a 3 + a 4 + a 5 = 6) :
  ∀ n, S n = n * (-2) + n * (n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l900_90030


namespace NUMINAMATH_GPT_division_of_expressions_l900_90070

theorem division_of_expressions : 
  (2 * 3 + 4) / (2 + 3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_division_of_expressions_l900_90070


namespace NUMINAMATH_GPT_value_of_expression_l900_90043

theorem value_of_expression : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l900_90043


namespace NUMINAMATH_GPT_polynomial_factorization_l900_90017

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end NUMINAMATH_GPT_polynomial_factorization_l900_90017


namespace NUMINAMATH_GPT_power_function_value_l900_90016

theorem power_function_value (a : ℝ) (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 4) :
  f 9 = 81 :=
by
  sorry

end NUMINAMATH_GPT_power_function_value_l900_90016


namespace NUMINAMATH_GPT_correct_number_of_sequences_l900_90012

noncomputable def athlete_sequences : Nat :=
  let total_permutations := 24
  let A_first_leg := 6
  let B_fourth_leg := 6
  let A_first_and_B_fourth := 2
  total_permutations - (A_first_leg + B_fourth_leg - A_first_and_B_fourth)

theorem correct_number_of_sequences : athlete_sequences = 14 := by
  sorry

end NUMINAMATH_GPT_correct_number_of_sequences_l900_90012
