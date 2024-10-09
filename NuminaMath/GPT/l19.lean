import Mathlib

namespace rectangle_width_l19_1979

-- Define the conditions
def length := 6
def area_triangle := 60
def area_ratio := 2/5

-- The theorem: proving that the width of the rectangle is 4 cm
theorem rectangle_width (w : ℝ) (A_triangle : ℝ) (len : ℝ) 
  (ratio : ℝ) (h1 : A_triangle = 60) (h2 : len = 6) (h3 : ratio = 2 / 5) 
  (h4 : (len * w) / A_triangle = ratio) : 
  w = 4 := 
by 
  sorry

end rectangle_width_l19_1979


namespace trade_and_unification_effects_l19_1971

theorem trade_and_unification_effects :
  let country_A_corn := 8
  let country_B_eggplants := 18
  let country_B_corn := 12
  let country_A_eggplants := 10
  
  -- Part (a): Absolute and comparative advantages
  (country_B_corn > country_A_corn) ∧ (country_B_eggplants > country_A_eggplants) ∧
  let opportunity_cost_A_eggplants := country_A_corn / country_A_eggplants
  let opportunity_cost_A_corn := country_A_eggplants / country_A_corn
  let opportunity_cost_B_eggplants := country_B_corn / country_B_eggplants
  let opportunity_cost_B_corn := country_B_eggplants / country_B_corn
  (opportunity_cost_B_eggplants < opportunity_cost_A_eggplants) ∧ (opportunity_cost_A_corn < opportunity_cost_B_corn) ∧

  -- Part (b): Volumes produced and consumed with trade
  let price := 1
  let income_A := country_A_corn * price
  let income_B := country_B_eggplants * price
  let consumption_A_eggplants := income_A / price / 2
  let consumption_A_corn := country_A_corn / 2
  let consumption_B_corn := income_B / price / 2
  let consumption_B_eggplants := country_B_eggplants / 2
  (consumption_A_eggplants = 4) ∧ (consumption_A_corn = 4) ∧
  (consumption_B_corn = 9) ∧ (consumption_B_eggplants = 9) ∧

  -- Part (c): Volumes after unification without trade
  let unified_eggplants := 18 - (1.5 * 4)
  let unified_corn := 8 + 4
  let total_unified_eggplants := unified_eggplants
  let total_unified_corn := unified_corn
  (total_unified_eggplants = 12) ∧ (total_unified_corn = 12) ->
  
  total_unified_eggplants = 12 ∧ total_unified_corn = 12 ∧
  (total_unified_eggplants < (consumption_A_eggplants + consumption_B_eggplants)) ∧
  (total_unified_corn < (consumption_A_corn + consumption_B_corn))
:= by
  -- Proof omitted
  sorry

end trade_and_unification_effects_l19_1971


namespace inequalities_hold_l19_1920

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l19_1920


namespace bird_wings_l19_1973

theorem bird_wings (birds wings_per_bird : ℕ) (h1 : birds = 13) (h2 : wings_per_bird = 2) : birds * wings_per_bird = 26 := by
  sorry

end bird_wings_l19_1973


namespace son_work_time_l19_1982

theorem son_work_time :
  let M := (1 : ℚ) / 7
  let combined_rate := (1 : ℚ) / 3
  let S := combined_rate - M
  1 / S = 5.25 :=  
by
  sorry

end son_work_time_l19_1982


namespace sqrt_eq_cubrt_l19_1911

theorem sqrt_eq_cubrt (x : ℝ) (h : Real.sqrt x = x^(1/3)) : x = 0 ∨ x = 1 :=
by
  sorry

end sqrt_eq_cubrt_l19_1911


namespace norma_cards_lost_l19_1998

theorem norma_cards_lost (original_cards : ℕ) (current_cards : ℕ) (cards_lost : ℕ)
  (h1 : original_cards = 88) (h2 : current_cards = 18) :
  original_cards - current_cards = cards_lost →
  cards_lost = 70 := by
  sorry

end norma_cards_lost_l19_1998


namespace quadratic_equation_nonzero_coefficient_l19_1950

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end quadratic_equation_nonzero_coefficient_l19_1950


namespace cylinder_volume_increase_l19_1943

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l19_1943


namespace second_to_last_digit_of_n_squared_plus_2n_l19_1951
open Nat

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) (h : (n^2 + 2 * n) % 10 = 4) : ((n^2 + 2 * n) / 10) % 10 = 2 :=
  sorry

end second_to_last_digit_of_n_squared_plus_2n_l19_1951


namespace product_eq_one_l19_1959

theorem product_eq_one (a b c : ℝ) (h1 : a^2 + 2 = b^4) (h2 : b^2 + 2 = c^4) (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 :=
sorry

end product_eq_one_l19_1959


namespace divides_p_minus_one_l19_1931

theorem divides_p_minus_one {p a b : ℕ} {n : ℕ} 
  (hp : p ≥ 3) 
  (prime_p : Nat.Prime p )
  (gcd_ab : Nat.gcd a b = 1)
  (hdiv : p ∣ (a ^ (2 ^ n) + b ^ (2 ^ n))) : 
  2 ^ (n + 1) ∣ p - 1 := 
sorry

end divides_p_minus_one_l19_1931


namespace cubics_identity_l19_1993

variable (a b c x y z : ℝ)

theorem cubics_identity (X Y Z : ℝ)
  (h1 : X = a * x + b * y + c * z)
  (h2 : Y = a * y + b * z + c * x)
  (h3 : Z = a * z + b * x + c * y) :
  X^3 + Y^3 + Z^3 - 3 * X * Y * Z = 
  (x^3 + y^3 + z^3 - 3 * x * y * z) * (a^3 + b^3 + c^3 - 3 * a * b * c) :=
sorry

end cubics_identity_l19_1993


namespace Bran_remaining_payment_l19_1935

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end Bran_remaining_payment_l19_1935


namespace solve_picnic_problem_l19_1994

def picnic_problem : Prop :=
  ∃ (M W A C : ℕ), 
    M = W + 80 ∧ 
    A = C + 80 ∧ 
    M + W = A ∧ 
    A + C = 240 ∧ 
    M = 120

theorem solve_picnic_problem : picnic_problem :=
  sorry

end solve_picnic_problem_l19_1994


namespace interval_f_has_two_roots_l19_1988

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

theorem interval_f_has_two_roots (a : ℝ) : (∀ x : ℝ, f a x = 0 → ∃ u v : ℝ, u ≠ v ∧ f a u = 0 ∧ f a v = 0) ↔ 0 < a ∧ a < 1 / 8 := 
sorry

end interval_f_has_two_roots_l19_1988


namespace integer_values_abc_l19_1937

theorem integer_values_abc (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) →
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  sorry

end integer_values_abc_l19_1937


namespace aang_caught_7_fish_l19_1936

theorem aang_caught_7_fish (A : ℕ) (h_avg : (A + 5 + 12) / 3 = 8) : A = 7 :=
by
  sorry

end aang_caught_7_fish_l19_1936


namespace solve_x_l19_1919

theorem solve_x (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 8 * x ^ 2 + 16 * x * y = x ^ 3 + 3 * x ^ 2 * y) (h₄ : y = 2 * x) : x = 40 / 7 :=
by
  sorry

end solve_x_l19_1919


namespace intercepted_segments_length_l19_1987

theorem intercepted_segments_length {a b c x : ℝ} 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : x = a * b * c / (a * b + b * c + c * a)) : 
  x = a * b * c / (a * b + b * c + c * a) :=
by sorry

end intercepted_segments_length_l19_1987


namespace possible_galina_numbers_l19_1915

def is_divisible_by (m n : ℕ) : Prop := n % m = 0

def conditions_for_galina_number (n : ℕ) : Prop :=
  let C1 := is_divisible_by 7 n
  let C2 := is_divisible_by 11 n
  let C3 := n < 13
  let C4 := is_divisible_by 77 n
  (C1 ∧ ¬C2 ∧ C3 ∧ ¬C4) ∨ (¬C1 ∧ C2 ∧ C3 ∧ ¬C4)

theorem possible_galina_numbers (n : ℕ) :
  conditions_for_galina_number n ↔ (n = 7 ∨ n = 11) :=
by
  -- Proof to be filled in
  sorry

end possible_galina_numbers_l19_1915


namespace strawberry_rows_l19_1972

theorem strawberry_rows (yield_per_row total_harvest : ℕ) (h1 : yield_per_row = 268) (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := 
by 
  sorry

end strawberry_rows_l19_1972


namespace three_boys_in_shop_at_same_time_l19_1952

-- Definitions for the problem conditions
def boys : Type := Fin 7  -- Representing the 7 boys
def visits : Type := Fin 3  -- Each boy makes 3 visits

-- A structure representing a visit by a boy
structure Visit := (boy : boys) (visit_num : visits)

-- Meeting condition: Every pair of boys meets at the shop
def meets_at_shop (v1 v2 : Visit) : Prop :=
  v1.boy ≠ v2.boy  -- Ensure it's not the same boy (since we assume each pair meets)

-- The theorem to be proven
theorem three_boys_in_shop_at_same_time :
  ∃ (v1 v2 v3 : Visit), v1.boy ≠ v2.boy ∧ v2.boy ≠ v3.boy ∧ v1.boy ≠ v3.boy :=
sorry

end three_boys_in_shop_at_same_time_l19_1952


namespace units_digit_of_square_l19_1983

theorem units_digit_of_square (a b : ℕ) (h₁ : (10 * a + b) ^ 2 % 100 / 10 = 7) : b = 6 :=
sorry

end units_digit_of_square_l19_1983


namespace vector_BC_calculation_l19_1921

/--
If \(\overrightarrow{AB} = (3, 6)\) and \(\overrightarrow{AC} = (1, 2)\),
then \(\overrightarrow{BC} = (-2, -4)\).
-/
theorem vector_BC_calculation (AB AC BC : ℤ × ℤ) 
  (hAB : AB = (3, 6))
  (hAC : AC = (1, 2)) : 
  BC = (-2, -4) := 
by
  sorry

end vector_BC_calculation_l19_1921


namespace evaluate_polynomial_at_2_l19_1986

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + x^2 + 2 * x + 3

theorem evaluate_polynomial_at_2 : polynomial 2 = 67 := by
  sorry

end evaluate_polynomial_at_2_l19_1986


namespace fraction_of_girls_correct_l19_1992

-- Define the total number of students in each school
def total_greenwood : ℕ := 300
def total_maplewood : ℕ := 240

-- Define the ratios of boys to girls
def ratio_boys_girls_greenwood := (3, 2)
def ratio_boys_girls_maplewood := (3, 4)

-- Define the number of boys and girls at Greenwood Middle School
def boys_greenwood (x : ℕ) : ℕ := 3 * x
def girls_greenwood (x : ℕ) : ℕ := 2 * x

-- Define the number of boys and girls at Maplewood Middle School
def boys_maplewood (y : ℕ) : ℕ := 3 * y
def girls_maplewood (y : ℕ) : ℕ := 4 * y

-- Define the total fractions
def total_girls (x y : ℕ) : ℚ := (girls_greenwood x + girls_maplewood y)
def total_students : ℚ := (total_greenwood + total_maplewood)

-- Main theorem to prove the fraction of girls at the event
theorem fraction_of_girls_correct (x y : ℕ)
  (h1 : 5 * x = total_greenwood)
  (h2 : 7 * y = total_maplewood) :
  (total_girls x y) / total_students = 5 / 7 :=
by
  sorry

end fraction_of_girls_correct_l19_1992


namespace cost_of_apples_l19_1974

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l19_1974


namespace find_a_value_l19_1906

noncomputable def find_a (a : ℝ) : Prop :=
  (a > 0) ∧ (1 / 3 = 2 / a)

theorem find_a_value (a : ℝ) (h : find_a a) : a = 6 :=
sorry

end find_a_value_l19_1906


namespace bugs_initial_count_l19_1966

theorem bugs_initial_count (B : ℝ) 
  (h_spray : ∀ (b : ℝ), b * 0.8 = b * (4 / 5)) 
  (h_spiders : ∀ (s : ℝ), s * 7 = 12 * 7) 
  (h_initial_spray_spiders : ∀ (b : ℝ), b * 0.8 - (12 * 7) = 236) 
  (h_final_bugs : 320 / 0.8 = 400) : 
  B = 400 :=
sorry

end bugs_initial_count_l19_1966


namespace investment_C_l19_1961

-- Definitions of the given conditions
def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def total_profit : ℝ := 12700
def profit_A : ℝ := 3810

-- Defining the total investment, including C's investment
noncomputable def investment_total_including_C (C : ℝ) : ℝ := investment_A + investment_B + C

-- Proving the correct investment for C under the given conditions
theorem investment_C (C : ℝ) :
  (investment_A / investment_total_including_C C) = (profit_A / total_profit) → 
  C = 10500 :=
by
  -- Placeholder for the actual proof
  sorry

end investment_C_l19_1961


namespace planes_parallel_if_perpendicular_to_same_line_l19_1996

variables {Point : Type} {Line : Type} {Plane : Type} 

-- Definitions and conditions
noncomputable def is_parallel (α β : Plane) : Prop := sorry
noncomputable def is_perpendicular (l : Line) (α : Plane) : Prop := sorry

variables (l1 : Line) (α β : Plane)

theorem planes_parallel_if_perpendicular_to_same_line
  (h1 : is_perpendicular l1 α)
  (h2 : is_perpendicular l1 β) : is_parallel α β := 
sorry

end planes_parallel_if_perpendicular_to_same_line_l19_1996


namespace solve_for_a_l19_1905

theorem solve_for_a (a : ℚ) (h : a + a / 4 = 10 / 4) : a = 2 :=
sorry

end solve_for_a_l19_1905


namespace solve_for_f_8_l19_1922

noncomputable def f (x : ℝ) : ℝ := (Real.logb 2 x)

theorem solve_for_f_8 {x : ℝ} (h : f (x^3) = Real.logb 2 x) : f 8 = 1 :=
by
sorry

end solve_for_f_8_l19_1922


namespace maximum_value_of_d_l19_1977

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l19_1977


namespace train_stops_for_10_minutes_per_hour_l19_1995

-- Define the conditions
def speed_excluding_stoppages : ℕ := 48 -- in kmph
def speed_including_stoppages : ℕ := 40 -- in kmph

-- Define the question as proving the train stops for 10 minutes per hour
theorem train_stops_for_10_minutes_per_hour :
  (speed_excluding_stoppages - speed_including_stoppages) * 60 / speed_excluding_stoppages = 10 :=
by
  sorry

end train_stops_for_10_minutes_per_hour_l19_1995


namespace dime_probability_l19_1940

theorem dime_probability (dime_value quarter_value : ℝ) (dime_worth quarter_worth total_coins: ℕ) :
  dime_value = 0.10 ∧
  quarter_value = 0.25 ∧
  dime_worth = 10 ∧
  quarter_worth = 4 ∧
  total_coins = 14 →
  (dime_worth / total_coins : ℝ) = 5 / 7 :=
by
  sorry

end dime_probability_l19_1940


namespace parallelogram_angle_l19_1939

theorem parallelogram_angle (a b : ℝ) (h1 : a + b = 180) (h2 : a = b + 50) : b = 65 :=
by
  -- Proof would go here, but we're adding a placeholder
  sorry

end parallelogram_angle_l19_1939


namespace total_cost_is_18_l19_1934

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l19_1934


namespace divisible_by_6_l19_1900

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n^3 - n + 6) :=
by
  sorry

end divisible_by_6_l19_1900


namespace right_triangle_set_D_l19_1918

theorem right_triangle_set_D : (5^2 + 12^2 = 13^2) ∧ 
  ((3^2 + 3^2 ≠ 5^2) ∧ (6^2 + 8^2 ≠ 9^2) ∧ (4^2 + 5^2 ≠ 6^2)) :=
by
  sorry

end right_triangle_set_D_l19_1918


namespace sum_of_largest_and_smallest_l19_1904

theorem sum_of_largest_and_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  a + c = 22 :=
by
  sorry

end sum_of_largest_and_smallest_l19_1904


namespace solve_trig_eq_l19_1968

open Real -- Open real number structure

theorem solve_trig_eq (x : ℝ) :
  (sin x)^2 + (sin (2 * x))^2 + (sin (3 * x))^2 = 2 ↔ 
  (∃ n : ℤ, x = π / 4 + (π * n) / 2)
  ∨ (∃ n : ℤ, x = π / 2 + π * n)
  ∨ (∃ n : ℤ, x = π / 6 + π * n ∨ x = -π / 6 + π * n) := by sorry

end solve_trig_eq_l19_1968


namespace vertex_in_fourth_quadrant_l19_1970

theorem vertex_in_fourth_quadrant (m : ℝ) (h : m < 0) : 
  (0 < -m) ∧ (-1 < 0) :=
by
  sorry

end vertex_in_fourth_quadrant_l19_1970


namespace feathers_to_cars_ratio_l19_1958

theorem feathers_to_cars_ratio (initial_feathers : ℕ) (final_feathers : ℕ) (cars_dodged : ℕ)
  (h₁ : initial_feathers = 5263) (h₂ : final_feathers = 5217) (h₃ : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 :=
by
  sorry

end feathers_to_cars_ratio_l19_1958


namespace cubic_trinomial_degree_l19_1991

theorem cubic_trinomial_degree (n : ℕ) (P : ℕ → ℕ →  ℕ → Prop) : 
  (P n 5 4) → n = 3 := 
  sorry

end cubic_trinomial_degree_l19_1991


namespace temperature_difference_l19_1997

theorem temperature_difference (T_high T_low : ℤ) (h_high : T_high = 11) (h_low : T_low = -11) :
  T_high - T_low = 22 := by
  sorry

end temperature_difference_l19_1997


namespace negation_of_proposition_l19_1964

theorem negation_of_proposition:
  (∀ x : ℝ, x ≥ 0 → x - 2 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) := 
sorry

end negation_of_proposition_l19_1964


namespace unique_integer_sum_squares_l19_1949

theorem unique_integer_sum_squares (n : ℤ) (h : ∃ d1 d2 d3 d4 : ℕ, d1 * d2 * d3 * d4 = n ∧ n = d1*d1 + d2*d2 + d3*d3 + d4*d4) : n = 42 := 
sorry

end unique_integer_sum_squares_l19_1949


namespace westbound_speed_is_275_l19_1927

-- Define the conditions for the problem at hand.
def east_speed : ℕ := 325
def separation_time : ℝ := 3.5
def total_distance : ℕ := 2100

-- Compute the known east-bound distance.
def east_distance : ℝ := east_speed * separation_time

-- Define the speed of the west-bound plane as an unknown variable.
variable (v : ℕ)

-- Compute the west-bound distance.
def west_distance := v * separation_time

-- The assertion that the sum of two distances equals the total distance.
def distance_equation := east_distance + (v * separation_time) = total_distance

-- Prove that the west-bound speed is 275 mph.
theorem westbound_speed_is_275 : v = 275 :=
by
  sorry

end westbound_speed_is_275_l19_1927


namespace correct_equation_l19_1910

theorem correct_equation (x : ℤ) : 232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l19_1910


namespace current_babysitter_hourly_rate_l19_1941

-- Define variables
def new_babysitter_hourly_rate := 12
def extra_charge_per_scream := 3
def hours_hired := 6
def number_of_screams := 2
def cost_difference := 18

-- Define the total cost calculations
def new_babysitter_total_cost :=
  new_babysitter_hourly_rate * hours_hired + extra_charge_per_scream * number_of_screams

def current_babysitter_total_cost :=
  new_babysitter_total_cost + cost_difference

theorem current_babysitter_hourly_rate :
  current_babysitter_total_cost / hours_hired = 16 := by
  sorry

end current_babysitter_hourly_rate_l19_1941


namespace factorial_expression_evaluation_l19_1947

theorem factorial_expression_evaluation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := 
by 
  sorry

end factorial_expression_evaluation_l19_1947


namespace find_divisor_l19_1967

theorem find_divisor (X : ℕ) (h12 : 12 ∣ (1020 - 12)) (h24 : 24 ∣ (1020 - 12)) (h48 : 48 ∣ (1020 - 12)) (h56 : 56 ∣ (1020 - 12)) :
  X = 63 :=
sorry

end find_divisor_l19_1967


namespace margaret_time_is_10_minutes_l19_1957

variable (time_billy_first_5_laps : ℕ)
variable (time_billy_next_3_laps : ℕ)
variable (time_billy_next_lap : ℕ)
variable (time_billy_final_lap : ℕ)
variable (time_difference : ℕ)

def billy_total_time := time_billy_first_5_laps + time_billy_next_3_laps + time_billy_next_lap + time_billy_final_lap

def margaret_total_time := billy_total_time + time_difference

theorem margaret_time_is_10_minutes :
  time_billy_first_5_laps = 120 ∧
  time_billy_next_3_laps = 240 ∧
  time_billy_next_lap = 60 ∧
  time_billy_final_lap = 150 ∧
  time_difference = 30 →
  margaret_total_time = 600 :=
by 
  sorry

end margaret_time_is_10_minutes_l19_1957


namespace expr_is_irreducible_fraction_l19_1981

def a : ℚ := 3 / 2015
def b : ℚ := 11 / 2016

noncomputable def expr : ℚ := 
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a

theorem expr_is_irreducible_fraction : expr = 11 / 112 := by
  sorry

end expr_is_irreducible_fraction_l19_1981


namespace paper_clips_in_two_cases_l19_1985

-- Define the conditions
variables (c b : ℕ)

-- Define the theorem statement
theorem paper_clips_in_two_cases (c b : ℕ) : 
    2 * c * b * 400 = 2 * c * b * 400 :=
by
  sorry

end paper_clips_in_two_cases_l19_1985


namespace candy_difference_l19_1930

theorem candy_difference (frankie_candies : ℕ) (max_candies : ℕ) (h1 : frankie_candies = 74) (h2 : max_candies = 92) : max_candies - frankie_candies = 18 := by
  sorry

end candy_difference_l19_1930


namespace flower_shop_sold_bouquets_l19_1932

theorem flower_shop_sold_bouquets (roses_per_bouquet : ℕ) (daisies_per_bouquet : ℕ) 
  (rose_bouquets_sold : ℕ) (daisy_bouquets_sold : ℕ) (total_flowers_sold : ℕ)
  (h1 : roses_per_bouquet = 12) (h2 : rose_bouquets_sold = 10) 
  (h3 : daisy_bouquets_sold = 10) (h4 : total_flowers_sold = 190) : 
  (rose_bouquets_sold + daisy_bouquets_sold) = 20 :=
by sorry

end flower_shop_sold_bouquets_l19_1932


namespace toll_for_18_wheel_truck_l19_1984

-- Define the conditions
def wheels_per_axle : Nat := 2
def total_wheels : Nat := 18
def toll_formula (x : Nat) : ℝ := 1.5 + 0.5 * (x - 2)

-- Calculate number of axles from the number of wheels
def number_of_axles := total_wheels / wheels_per_axle

-- Target statement: The toll for the given truck
theorem toll_for_18_wheel_truck : toll_formula number_of_axles = 5.0 := by
  sorry

end toll_for_18_wheel_truck_l19_1984


namespace incircle_excircle_relation_l19_1978

variables {α : Type*} [LinearOrderedField α]

-- Defining the area expressions and radii
def area_inradius (a b c r : α) : α := (a + b + c) * r / 2
def area_exradius1 (a b c r1 : α) : α := (b + c - a) * r1 / 2
def area_exradius2 (a b c r2 : α) : α := (a + c - b) * r2 / 2
def area_exradius3 (a b c r3 : α) : α := (a + b - c) * r3 / 2

theorem incircle_excircle_relation (a b c r r1 r2 r3 Q : α) 
  (h₁ : Q = area_inradius a b c r)
  (h₂ : Q = area_exradius1 a b c r1)
  (h₃ : Q = area_exradius2 a b c r2)
  (h₄ : Q = area_exradius3 a b c r3) :
  1 / r = 1 / r1 + 1 / r2 + 1 / r3 :=
by 
  sorry

end incircle_excircle_relation_l19_1978


namespace triangle_height_l19_1999

theorem triangle_height (base : ℝ) (height : ℝ) (area : ℝ)
  (h_base : base = 8) (h_area : area = 16) (h_area_formula : area = (base * height) / 2) :
  height = 4 :=
by
  sorry

end triangle_height_l19_1999


namespace binary_to_base4_representation_l19_1942

def binary_to_base4 (n : ℕ) : ℕ :=
  -- Assuming implementation that converts binary number n to its base 4 representation 
  sorry

theorem binary_to_base4_representation :
  binary_to_base4 0b10110110010 = 23122 :=
by sorry

end binary_to_base4_representation_l19_1942


namespace price_difference_l19_1953

-- Definitions of conditions
def market_price : ℝ := 15400
def initial_sales_tax_rate : ℝ := 0.076
def new_sales_tax_rate : ℝ := 0.0667
def discount_rate : ℝ := 0.05
def handling_fee : ℝ := 200

-- Calculation of original sales tax
def original_sales_tax_amount : ℝ := market_price * initial_sales_tax_rate
-- Calculation of price after discount
def discount_amount : ℝ := market_price * discount_rate
def price_after_discount : ℝ := market_price - discount_amount
-- Calculation of new sales tax
def new_sales_tax_amount : ℝ := price_after_discount * new_sales_tax_rate
-- Calculation of total price with new sales tax and handling fee
def total_price_new : ℝ := price_after_discount + new_sales_tax_amount + handling_fee
-- Calculation of original total price with handling fee
def original_total_price : ℝ := market_price + original_sales_tax_amount + handling_fee

-- Expected difference in total cost
def expected_difference : ℝ := 964.60

-- Lean 4 statement to prove the difference
theorem price_difference :
  original_total_price - total_price_new = expected_difference :=
by
  sorry

end price_difference_l19_1953


namespace find_y_value_l19_1908

-- Define the angles in Lean
def angle1 (y : ℕ) : ℕ := 6 * y
def angle2 (y : ℕ) : ℕ := 7 * y
def angle3 (y : ℕ) : ℕ := 3 * y
def angle4 (y : ℕ) : ℕ := 2 * y

-- The condition that the sum of the angles is 360
def angles_sum_to_360 (y : ℕ) : Prop :=
  angle1 y + angle2 y + angle3 y + angle4 y = 360

-- The proof problem statement
theorem find_y_value (y : ℕ) (h : angles_sum_to_360 y) : y = 20 :=
sorry

end find_y_value_l19_1908


namespace number_of_problems_l19_1980

/-- Given the conditions of the problem, prove that the number of problems I did is exactly 140.-/
theorem number_of_problems (p t : ℕ) (h1 : p > 12) (h2 : p * t = (p + 6) * (t - 3)) : p * t = 140 :=
by
  sorry

end number_of_problems_l19_1980


namespace train_speed_l19_1923

theorem train_speed
  (length_m : ℝ)
  (time_s : ℝ)
  (h_length : length_m = 280.0224)
  (h_time : time_s = 25.2) :
  (length_m / 1000) / (time_s / 3600) = 40.0032 :=
by
  sorry

end train_speed_l19_1923


namespace amount_of_cocoa_powder_given_by_mayor_l19_1945

def total_cocoa_powder_needed : ℕ := 306
def cocoa_powder_still_needed : ℕ := 47

def cocoa_powder_given_by_mayor : ℕ :=
  total_cocoa_powder_needed - cocoa_powder_still_needed

theorem amount_of_cocoa_powder_given_by_mayor :
  cocoa_powder_given_by_mayor = 259 := by
  sorry

end amount_of_cocoa_powder_given_by_mayor_l19_1945


namespace range_of_a_l19_1955

theorem range_of_a {a : ℝ} : (∀ x : ℝ, (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0) → (x = 0 ∨ x = -4)) → (a = 1 ∨ a ≤ -1) := 
by {
  sorry
}

end range_of_a_l19_1955


namespace square_plot_area_l19_1962

theorem square_plot_area (cost_per_foot total_cost : ℕ) (hcost_per_foot : cost_per_foot = 60) (htotal_cost : total_cost = 4080) :
  ∃ (A : ℕ), A = 289 :=
by
  have h : 4 * 60 * 17 = 4080 := by rfl
  have s : 17 = 4080 / (4 * 60) := by sorry
  use 17 ^ 2
  have hsquare : 17 ^ 2 = 289 := by rfl
  exact hsquare

end square_plot_area_l19_1962


namespace spring_outing_students_l19_1907

variable (x y : ℕ)

theorem spring_outing_students (hx : x % 10 = 0) (hy : y % 10 = 0) (h1 : x + y = 1008) (h2 : y - x = 133) :
  x = 437 ∧ y = 570 :=
by
  sorry

end spring_outing_students_l19_1907


namespace find_y_l19_1916

theorem find_y (n x y : ℕ) 
    (h1 : (n + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + n + x + y) / 5 = 200) :
    y = 50 := 
by
  -- Placeholder for the proof
  sorry

end find_y_l19_1916


namespace train_length_is_150_meters_l19_1938

def train_speed_kmph : ℝ := 68
def man_speed_kmph : ℝ := 8
def passing_time_sec : ℝ := 8.999280057595392

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph - man_speed_kmph
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * passing_time_sec

theorem train_length_is_150_meters (train_speed_kmph man_speed_kmph passing_time_sec : ℝ) :
  train_speed_kmph = 68 → man_speed_kmph = 8 → passing_time_sec = 8.999280057595392 →
  length_of_train = 150 :=
by
  intros h1 h2 h3
  simp [length_of_train, h1, h2, h3]
  sorry

end train_length_is_150_meters_l19_1938


namespace percent_other_birds_is_31_l19_1990

noncomputable def initial_hawk_percentage : ℝ := 0.30
noncomputable def initial_paddyfield_warbler_percentage : ℝ := 0.25
noncomputable def initial_kingfisher_percentage : ℝ := 0.10
noncomputable def initial_hp_k_total : ℝ := initial_hawk_percentage + initial_paddyfield_warbler_percentage + initial_kingfisher_percentage

noncomputable def migrated_hawk_percentage : ℝ := 0.8 * initial_hawk_percentage
noncomputable def migrated_kingfisher_percentage : ℝ := 2 * initial_kingfisher_percentage
noncomputable def migrated_hp_k_total : ℝ := migrated_hawk_percentage + initial_paddyfield_warbler_percentage + migrated_kingfisher_percentage

noncomputable def other_birds_percentage : ℝ := 1 - migrated_hp_k_total

theorem percent_other_birds_is_31 : other_birds_percentage = 0.31 := sorry

end percent_other_birds_is_31_l19_1990


namespace peter_and_susan_dollars_l19_1960

theorem peter_and_susan_dollars :
  (2 / 5 : Real) + (1 / 4 : Real) = 0.65 := 
by
  sorry

end peter_and_susan_dollars_l19_1960


namespace reality_show_duration_l19_1946

variable (x : ℕ)

theorem reality_show_duration :
  (5 * x + 10 = 150) → (x = 28) :=
by
  intro h
  sorry

end reality_show_duration_l19_1946


namespace value_of_x_l19_1929

-- Define the conditions extracted from problem (a)
def condition1 (x : ℝ) : Prop := x^2 - 1 = 0
def condition2 (x : ℝ) : Prop := x - 1 ≠ 0

-- The statement to be proved
theorem value_of_x : ∀ x : ℝ, condition1 x → condition2 x → x = -1 :=
by
  intros x h1 h2
  sorry

end value_of_x_l19_1929


namespace race_problem_l19_1926

theorem race_problem (a_speed b_speed : ℕ) (A B : ℕ) (finish_dist : ℕ)
  (h1 : finish_dist = 3000)
  (h2 : A = finish_dist - 500)
  (h3 : B = finish_dist - 600)
  (h4 : A / a_speed = B / b_speed)
  (h5 : a_speed / b_speed = 25 / 24) :
  B - ((500 * b_speed) / a_speed) = 120 :=
by
  sorry

end race_problem_l19_1926


namespace inequality_proof_l19_1969

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
by
  sorry

end inequality_proof_l19_1969


namespace f_x1_plus_f_x2_always_greater_than_zero_l19_1965

theorem f_x1_plus_f_x2_always_greater_than_zero
  {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f (x + 2))
  (h2 : ∀ x > 1, ∀ y > 1, x < y → f y < f x)
  (h3 : ∃ x₁ x₂ : ℝ, 1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) :
  ∀ x₁ x₂ : ℝ, (1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) → f x₁ + f x₂ > 0 := by
  sorry

end f_x1_plus_f_x2_always_greater_than_zero_l19_1965


namespace initial_ratio_milk_water_l19_1989

theorem initial_ratio_milk_water (M W : ℕ) (h1 : M + W = 165) (h2 : ∀ W', W' = W + 66 → M * 4 = 3 * W') : M / gcd M W = 3 ∧ W / gcd M W = 2 :=
by
  -- Proof here
  sorry

end initial_ratio_milk_water_l19_1989


namespace percentage_increase_l19_1963

variables (a b x m : ℝ) (p : ℝ)
variables (h1 : a / b = 4 / 5)
variables (h2 : x = a + (p / 100) * a)
variables (h3 : m = b - 0.6 * b)
variables (h4 : m / x = 0.4)

theorem percentage_increase (a_pos : 0 < a) (b_pos : 0 < b) : p = 25 :=
by sorry

end percentage_increase_l19_1963


namespace t_shirts_per_package_l19_1956

theorem t_shirts_per_package (total_tshirts : ℕ) (packages : ℕ) (tshirts_per_package : ℕ) :
  total_tshirts = 70 → packages = 14 → tshirts_per_package = total_tshirts / packages → tshirts_per_package = 5 :=
by
  sorry

end t_shirts_per_package_l19_1956


namespace man_and_son_together_days_l19_1901

noncomputable def man_days : ℝ := 7
noncomputable def son_days : ℝ := 5.25
noncomputable def combined_days : ℝ := man_days * son_days / (man_days + son_days)

theorem man_and_son_together_days :
  combined_days = 7 / 5 :=
by
  sorry

end man_and_son_together_days_l19_1901


namespace greatest_divisor_l19_1917

theorem greatest_divisor (d : ℕ) :
  (6215 % d = 23 ∧ 7373 % d = 29 ∧ 8927 % d = 35) → d = 36 :=
by
  sorry

end greatest_divisor_l19_1917


namespace ones_digit_of_sum_is_0_l19_1944

-- Define the integer n
def n : ℕ := 2012

-- Define the ones digit function
def ones_digit (x : ℕ) : ℕ := x % 10

-- Define the power function mod 10
def power_mod_10 (d a : ℕ) : ℕ := (d^a) % 10

-- Define the sequence sum for ones digits
def seq_sum_mod_10 (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ k => power_mod_10 (k+1) n)

-- Define the final sum mod 10 considering the repeating cycle and sum
def total_ones_digit_sum (a b : ℕ) : ℕ :=
  let cycle_sum := Finset.sum (Finset.range 10) (λ k => power_mod_10 (k+1) n)
  let s := cycle_sum * (a / 10) + Finset.sum (Finset.range b) (λ k => power_mod_10 (k+1) n)
  s % 10

-- Prove that the ones digit of the sum is 0
theorem ones_digit_of_sum_is_0 : total_ones_digit_sum n (n % 10) = 0 :=
sorry

end ones_digit_of_sum_is_0_l19_1944


namespace winning_percentage_is_70_l19_1933

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end winning_percentage_is_70_l19_1933


namespace hyperbola_real_axis_length_l19_1975

theorem hyperbola_real_axis_length (x y : ℝ) :
  x^2 - y^2 / 9 = 1 → 2 = 2 :=
by
  sorry

end hyperbola_real_axis_length_l19_1975


namespace gcd_228_1995_l19_1913

theorem gcd_228_1995 :
  Nat.gcd 228 1995 = 21 :=
sorry

end gcd_228_1995_l19_1913


namespace population_meets_capacity_l19_1924

-- Define the initial conditions and parameters
def initial_year : ℕ := 1998
def initial_population : ℕ := 100
def population_growth_rate : ℕ := 4  -- quadruples every 20 years
def years_per_growth_period : ℕ := 20
def land_area_hectares : ℕ := 15000
def hectares_per_person : ℕ := 2
def maximum_capacity : ℕ := land_area_hectares / hectares_per_person

-- Define the statement
theorem population_meets_capacity :
  ∃ (years_from_initial : ℕ), years_from_initial = 60 ∧
  initial_population * population_growth_rate ^ (years_from_initial / years_per_growth_period) ≥ maximum_capacity :=
by
  sorry

end population_meets_capacity_l19_1924


namespace find_f_of_3_l19_1954

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x^2 - 2 * x) : f 3 = -1 :=
by 
  sorry

end find_f_of_3_l19_1954


namespace sum_first_60_natural_numbers_l19_1976

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l19_1976


namespace range_of_a_l19_1925

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ a → (x + y + 1 ≤ 2 * (x + 1) - 3 * (y + 1))) → a ≤ -2 :=
by 
  intros h
  sorry

end range_of_a_l19_1925


namespace binary_to_decimal_l19_1902

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 := by
  sorry

end binary_to_decimal_l19_1902


namespace tim_initial_books_l19_1928

def books_problem : Prop :=
  ∃ T : ℕ, 10 + T - 24 = 19 ∧ T = 33

theorem tim_initial_books : books_problem :=
  sorry

end tim_initial_books_l19_1928


namespace arithmetic_sequence_l19_1914

noncomputable def M (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.sum (Finset.range n) (λ i => a (i + 1))) / n

theorem arithmetic_sequence (a : ℕ → ℝ) (C : ℝ)
  (h : ∀ {i j k : ℕ}, i ≠ j → j ≠ k → k ≠ i →
    (i - j) * M a k + (j - k) * M a i + (k - i) * M a j = C) :
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a 1 + n * d :=
sorry

end arithmetic_sequence_l19_1914


namespace sum_of_numbers_l19_1909

theorem sum_of_numbers (x : ℕ) (first_num second_num third_num sum : ℕ) 
  (h1 : 5 * x = first_num) 
  (h2 : 3 * x = second_num)
  (h3 : 4 * x = third_num) 
  (h4 : second_num = 27)
  : first_num + second_num + third_num = 108 :=
by {
  sorry
}

end sum_of_numbers_l19_1909


namespace minimize_expression_l19_1912

open Real

theorem minimize_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
sorry

end minimize_expression_l19_1912


namespace mother_duck_multiple_of_first_two_groups_l19_1903

variables (num_ducklings : ℕ) (snails_first_batch : ℕ) (snails_second_batch : ℕ)
          (total_snails : ℕ) (mother_duck_snails : ℕ)

-- Given conditions
def conditions : Prop :=
  num_ducklings = 8 ∧ 
  snails_first_batch = 3 * 5 ∧ 
  snails_second_batch = 3 * 9 ∧ 
  total_snails = 294 ∧ 
  total_snails = snails_first_batch + snails_second_batch + 2 * mother_duck_snails ∧ 
  mother_duck_snails > 0

-- Our goal is to prove that the mother duck finds 3 times the snails the first two groups of ducklings find
theorem mother_duck_multiple_of_first_two_groups (h : conditions num_ducklings snails_first_batch snails_second_batch total_snails mother_duck_snails) : 
  mother_duck_snails / (snails_first_batch + snails_second_batch) = 3 :=
by 
  sorry

end mother_duck_multiple_of_first_two_groups_l19_1903


namespace coloring_count_in_3x3_grid_l19_1948

theorem coloring_count_in_3x3_grid (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : 
  ∃ count : ℕ, count = 15 ∧ ∀ (cells : Finset (Fin n × Fin m)),
  (cells.card = 3 ∧ ∀ (c1 c2 : Fin n × Fin m), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → 
  (c1.fst ≠ c2.fst ∧ c1.snd ≠ c2.snd)) → cells.card ∣ count :=
sorry

end coloring_count_in_3x3_grid_l19_1948
