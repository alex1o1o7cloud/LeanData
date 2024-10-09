import Mathlib

namespace proof_problem_l1090_109021

variable (γ θ α : ℝ)
variable (x y : ℝ)

def condition1 := x = γ * Real.sin ((θ - α) / 2)
def condition2 := y = γ * Real.sin ((θ + α) / 2)

theorem proof_problem
  (h1 : condition1 γ θ α x)
  (h2 : condition2 γ θ α y)
  : x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * (Real.sin α)^2 :=
by
  sorry

end proof_problem_l1090_109021


namespace half_angle_quadrant_l1090_109088

-- Define the given condition
def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define the result that needs to be proved
def is_angle_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * 180 < α / 2 ∧ α / 2 < k * 180 + 45) ∨ (k * 180 + 180 < α / 2 ∧ α / 2 < k * 180 + 225)

-- The main theorem statement
theorem half_angle_quadrant (α : ℝ) (h : is_angle_in_first_quadrant α) : is_angle_in_first_or_third_quadrant α :=
sorry

end half_angle_quadrant_l1090_109088


namespace interval_monotonicity_no_zeros_min_a_l1090_109058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem interval_monotonicity (a : ℝ) :
  a = 1 →
  (∀ x, 0 < x ∧ x ≤ 2 → f a x < f a (x+1)) ∧
  (∀ x, x ≥ 2 → f a x < f a (x-1)) :=
by
  sorry

theorem no_zeros_min_a : 
  (∀ x, x ∈ Set.Ioo 0 (1/2 : ℝ) → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by
  sorry

end interval_monotonicity_no_zeros_min_a_l1090_109058


namespace rachel_older_than_leah_l1090_109040

theorem rachel_older_than_leah (rachel_age leah_age : ℕ) (h1 : rachel_age = 19) (h2 : rachel_age + leah_age = 34) :
  rachel_age - leah_age = 4 :=
by sorry

end rachel_older_than_leah_l1090_109040


namespace quadratic_real_roots_l1090_109081

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) :=
by
  sorry

end quadratic_real_roots_l1090_109081


namespace odd_function_expression_l1090_109024

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_expression (x : ℝ) (h1 : x < 0 → f x = x^2 - x) (h2 : ∀ x, f (-x) = -f x) (h3 : 0 < x) :
  f x = -x^2 - x :=
sorry

end odd_function_expression_l1090_109024


namespace find_x_l1090_109051

open Real

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def problem_statement (x : ℝ) : Prop :=
  let m := vector 2 x
  let n := vector 4 (-2)
  let m_minus_n := vector (2 - 4) (x - (-2))
  perpendicular m m_minus_n → x = -1 + sqrt 5 ∨ x = -1 - sqrt 5

-- We assert the theorem based on the problem statement
theorem find_x (x : ℝ) : problem_statement x :=
  sorry

end find_x_l1090_109051


namespace equality_equiv_l1090_109018

-- Problem statement
theorem equality_equiv (a b c : ℝ) :
  (a + b + c ≠ 0 → ( (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0)) ∧
  (a + b + c = 0 → ∀ w x y z: ℝ, w * x + y * z = 0) :=
by
  sorry

end equality_equiv_l1090_109018


namespace car_fewer_minutes_than_bus_l1090_109039

-- Conditions translated into Lean definitions
def bus_time_to_beach : ℕ := 40
def car_round_trip_time : ℕ := 70

-- Derived condition
def car_one_way_time : ℕ := car_round_trip_time / 2

-- Theorem statement to be proven
theorem car_fewer_minutes_than_bus : car_one_way_time = bus_time_to_beach - 5 := by
  -- This is the placeholder for the proof
  sorry

end car_fewer_minutes_than_bus_l1090_109039


namespace larger_number_is_437_l1090_109015

-- Definitions from the conditions
def hcf : ℕ := 23
def factor1 : ℕ := 13
def factor2 : ℕ := 19

-- The larger number should be the product of H.C.F and the larger factor.
theorem larger_number_is_437 : hcf * factor2 = 437 := by
  sorry

end larger_number_is_437_l1090_109015


namespace karens_speed_l1090_109091

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l1090_109091


namespace max_value_of_expr_l1090_109084

noncomputable def max_value (t : ℕ) : ℝ := (3^t - 2*t)*t / 9^t

theorem max_value_of_expr :
  ∃ t : ℕ, max_value t = 1 / 8 :=
sorry

end max_value_of_expr_l1090_109084


namespace f_g_2_equals_169_l1090_109060

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + x + 3

-- The theorem statement
theorem f_g_2_equals_169 : f (g 2) = 169 :=
by
  sorry

end f_g_2_equals_169_l1090_109060


namespace repeating_decimal_as_fraction_l1090_109014

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l1090_109014


namespace simplify_expression_l1090_109000

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) *
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := 
by sorry

end simplify_expression_l1090_109000


namespace solution_set_of_inequality_l1090_109031

variable {a b x : ℝ}

theorem solution_set_of_inequality (h : ∃ y, y = 3*(-5) + a ∧ y = -2*(-5) + b) :
  (3*x + a < -2*x + b) ↔ (x < -5) :=
by sorry

end solution_set_of_inequality_l1090_109031


namespace probability_five_blue_marbles_is_correct_l1090_109078

noncomputable def probability_of_five_blue_marbles : ℝ :=
let p_blue := (9 : ℝ) / 15
let p_red := (6 : ℝ) / 15
let specific_sequence_prob := p_blue ^ 5 * p_red ^ 3
let number_of_ways := (Nat.choose 8 5 : ℝ)
(number_of_ways * specific_sequence_prob)

theorem probability_five_blue_marbles_is_correct :
  probability_of_five_blue_marbles = 0.279 := by
sorry

end probability_five_blue_marbles_is_correct_l1090_109078


namespace ratio_of_installing_to_downloading_l1090_109080

noncomputable def timeDownloading : ℕ := 10

noncomputable def ratioTimeSpent (installingTime : ℕ) : ℚ :=
  let tutorialTime := 3 * (timeDownloading + installingTime)
  let totalTime := timeDownloading + installingTime + tutorialTime
  if totalTime = 60 then
    (installingTime : ℚ) / (timeDownloading : ℚ)
  else 0

theorem ratio_of_installing_to_downloading : ratioTimeSpent 5 = 1 / 2 := by
  sorry

end ratio_of_installing_to_downloading_l1090_109080


namespace shorter_tree_height_l1090_109001

theorem shorter_tree_height
  (s : ℝ)
  (h₁ : ∀ s, s > 0 )
  (h₂ : s + (s + 20) = 240)
  (h₃ : s / (s + 20) = 5 / 7) :
  s = 110 :=
by
sorry

end shorter_tree_height_l1090_109001


namespace brett_total_miles_l1090_109063

def miles_per_hour : ℕ := 75
def hours_driven : ℕ := 12

theorem brett_total_miles : miles_per_hour * hours_driven = 900 := 
by 
  sorry

end brett_total_miles_l1090_109063


namespace weight_of_10_moles_approx_l1090_109086

def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def atomic_mass_O : ℝ := 16.00

def molar_mass_C6H8O6 : ℝ := 
  (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)

def moles : ℝ := 10
def given_total_weight : ℝ := 1760

theorem weight_of_10_moles_approx (ε : ℝ) (hε : ε > 0) :
  abs ((moles * molar_mass_C6H8O6) - given_total_weight) < ε := by
  -- proof will go here.
  sorry

end weight_of_10_moles_approx_l1090_109086


namespace calculate_expression_l1090_109005

theorem calculate_expression : (632^2 - 568^2 + 100) = 76900 :=
by sorry

end calculate_expression_l1090_109005


namespace correct_calculated_value_l1090_109095

theorem correct_calculated_value (n : ℕ) (h : n + 9 = 30) : n + 7 = 28 :=
by
  sorry

end correct_calculated_value_l1090_109095


namespace shop_sold_price_l1090_109054

noncomputable def clock_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.60 * C
  let maintenance_cost := 0.10 * buy_back_price
  let total_spent := buy_back_price + maintenance_cost
  let selling_price := 1.80 * total_spent
  selling_price

theorem shop_sold_price (C : ℝ) (h1 : C - 0.60 * C = 100) :
  clock_selling_price C = 297 := by
  sorry

end shop_sold_price_l1090_109054


namespace max_n_l1090_109059

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2 ^ a n

theorem max_n (n : ℕ) (h1 : a 2 = 2) (h2 : ∀ n, b n = 2 ^ a n)
  (h3 : b 4 = 4 * b 2) : n ≤ 9 :=
by 
  sorry

end max_n_l1090_109059


namespace optimal_pricing_l1090_109092

-- Define the conditions given in the problem
def cost_price : ℕ := 40
def selling_price : ℕ := 60
def weekly_sales : ℕ := 300

def sales_volume (price : ℕ) : ℕ := weekly_sales - 10 * (price - selling_price)
def profit (price : ℕ) : ℕ := (price - cost_price) * sales_volume price

-- Statement to prove
theorem optimal_pricing : ∃ (price : ℕ), price = 65 ∧ profit price = 6250 :=
by {
  sorry
}

end optimal_pricing_l1090_109092


namespace max_x1_sq_plus_x2_sq_l1090_109099

theorem max_x1_sq_plus_x2_sq (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k - 2) 
  (h2 : x1 * x2 = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) : 
  x1^2 + x2^2 ≤ 18 :=
by sorry

end max_x1_sq_plus_x2_sq_l1090_109099


namespace shop_discount_percentage_l1090_109025

-- Definitions based on conditions
def original_price := 800
def price_paid := 560
def discount_amount := original_price - price_paid
def percentage_discount := (discount_amount / original_price) * 100

-- Proposition to prove
theorem shop_discount_percentage : percentage_discount = 30 := by
  sorry

end shop_discount_percentage_l1090_109025


namespace calculate_probability_l1090_109052

-- Definitions
def total_coins : ℕ := 16  -- Total coins (3 pennies + 5 nickels + 8 dimes)
def draw_coins : ℕ := 8    -- Coins drawn
def successful_outcomes : ℕ := 321  -- Number of successful outcomes
def total_outcomes : ℕ := Nat.choose total_coins draw_coins  -- Total number of ways to choose draw_coins from total_coins

-- Question statement in Lean 4: Probability of drawing coins worth at least 75 cents
theorem calculate_probability : (successful_outcomes : ℝ) / (total_outcomes : ℝ) = 321 / 12870 := by
  sorry

end calculate_probability_l1090_109052


namespace remainder_when_divided_by_11_l1090_109073

theorem remainder_when_divided_by_11 (n : ℕ) 
  (h1 : 10 ≤ n ∧ n < 100) 
  (h2 : n % 9 = 1) 
  (h3 : n % 10 = 3) : 
  n % 11 = 7 := 
sorry

end remainder_when_divided_by_11_l1090_109073


namespace drinking_ratio_l1090_109066

variable (t_mala t_usha : ℝ) (d_usha : ℝ)

theorem drinking_ratio :
  (t_mala = t_usha) → 
  (d_usha = 2 / 10) →
  (1 - d_usha = 8 / 10) →
  (4 * d_usha = 8) :=
by
  intros h1 h2 h3
  sorry

end drinking_ratio_l1090_109066


namespace inequality_range_l1090_109004

theorem inequality_range (y : ℝ) (b : ℝ) (hb : 0 < b) : (|y-5| + 2 * |y-2| > b) ↔ (b < 3) := 
sorry

end inequality_range_l1090_109004


namespace gcd_228_1995_base3_to_base6_conversion_l1090_109042

-- Proof Problem 1: GCD of 228 and 1995 is 57
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

-- Proof Problem 2: Converting base-3 number 11102 to base-6
theorem base3_to_base6_conversion : Nat.ofDigits 6 [3, 1, 5] = Nat.ofDigits 10 [1, 1, 1, 0, 2] :=
by
  sorry

end gcd_228_1995_base3_to_base6_conversion_l1090_109042


namespace profit_distribution_l1090_109047

theorem profit_distribution (investment_LiWei investment_WangGang profit total_investment : ℝ)
  (h1 : investment_LiWei = 16000)
  (h2 : investment_WangGang = 12000)
  (h3 : profit = 14000)
  (h4 : total_investment = investment_LiWei + investment_WangGang) :
  (profit * (investment_LiWei / total_investment) = 8000) ∧ 
  (profit * (investment_WangGang / total_investment) = 6000) :=
by
  sorry

end profit_distribution_l1090_109047


namespace perimeter_of_square_C_l1090_109027

theorem perimeter_of_square_C (s_A s_B s_C : ℕ) (hpA : 4 * s_A = 16) (hpB : 4 * s_B = 32) (hC : s_C = s_A + s_B - 2) :
  4 * s_C = 40 := 
by
  sorry

end perimeter_of_square_C_l1090_109027


namespace Jerry_average_speed_l1090_109098

variable (J : ℝ) -- Jerry's average speed in miles per hour
variable (C : ℝ) -- Carla's average speed in miles per hour
variable (T_J : ℝ) -- Time Jerry has been driving in hours
variable (T_C : ℝ) -- Time Carla has been driving in hours
variable (D : ℝ) -- Distance covered in miles

-- Given conditions
axiom Carla_speed : C = 35
axiom Carla_time : T_C = 3
axiom Jerry_time : T_J = T_C + 0.5

-- Distance covered by Carla in T_C hours at speed C
axiom Carla_distance : D = C * T_C

-- Distance covered by Jerry in T_J hours at speed J
axiom Jerry_distance : D = J * T_J

-- The goal to prove
theorem Jerry_average_speed : J = 30 :=
by
  sorry

end Jerry_average_speed_l1090_109098


namespace value_of_x_l1090_109008

theorem value_of_x (x : ℝ) (h : x = 88 + 0.25 * 88) : x = 110 :=
sorry

end value_of_x_l1090_109008


namespace days_c_worked_l1090_109071

theorem days_c_worked (Da Db Dc : ℕ) (Wa Wb Wc : ℕ)
  (h1 : Da = 6) (h2 : Db = 9) (h3 : Wc = 100) (h4 : 3 * Wc = 5 * Wa)
  (h5 : 4 * Wc = 5 * Wb)
  (h6 : Wa * Da + Wb * Db + Wc * Dc = 1480) : Dc = 4 :=
by
  sorry

end days_c_worked_l1090_109071


namespace bug_travel_distance_half_l1090_109033

-- Define the conditions
def isHexagonalGrid (side_length : ℝ) : Prop :=
  side_length = 1

def shortest_path_length (path_length : ℝ) : Prop :=
  path_length = 100

-- Define a theorem that encapsulates the problem statement
theorem bug_travel_distance_half (side_length path_length : ℝ)
  (H1 : isHexagonalGrid side_length)
  (H2 : shortest_path_length path_length) :
  ∃ one_direction_distance : ℝ, one_direction_distance = path_length / 2 :=
sorry -- Proof to be provided.

end bug_travel_distance_half_l1090_109033


namespace fair_coin_heads_probability_l1090_109045

theorem fair_coin_heads_probability
  (fair_coin : ∀ n : ℕ, (∀ (heads tails : ℕ), heads + tails = n → (heads / n = 1 / 2) ∧ (tails / n = 1 / 2)))
  (n : ℕ)
  (heads : ℕ)
  (tails : ℕ)
  (h1 : n = 20)
  (h2 : heads = 8)
  (h3 : tails = 12)
  (h4 : heads + tails = n)
  : heads / n = 1 / 2 :=
by
  sorry

end fair_coin_heads_probability_l1090_109045


namespace find_xy_l1090_109074

-- Defining the initial conditions
variable (x y : ℕ)

-- Defining the rectangular prism dimensions and the volume equation
def prism_volume_original : ℕ := 15 * 5 * 4 -- Volume = 300
def remaining_volume : ℕ := 120

-- The main theorem statement to prove the conditions and their solution
theorem find_xy (h1 : prism_volume_original - 5 * y * x = remaining_volume)
    (h2 : x < 4) 
    (h3 : y < 15) : 
    x = 3 ∧ y = 12 := sorry

end find_xy_l1090_109074


namespace y_coord_equidistant_l1090_109087

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l1090_109087


namespace rectangle_perimeter_l1090_109007
-- Refined definitions and setup
variables (AB BC AE BE CF : ℝ)
-- Conditions provided in the problem
def conditions := AB = 2 * BC ∧ AE = 10 ∧ BE = 26 ∧ CF = 5
-- Perimeter calculation based on the conditions
def perimeter (AB BC : ℝ) : ℝ := 2 * (AB + BC)
-- Main theorem stating the conditions and required result
theorem rectangle_perimeter {m n : ℕ} (h: conditions AB BC AE BE CF) :
  m + n = 105 ∧ Int.gcd m n = 1 ∧ perimeter AB BC = m / n := sorry

end rectangle_perimeter_l1090_109007


namespace x_y_sum_l1090_109069

theorem x_y_sum (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end x_y_sum_l1090_109069


namespace time_increases_with_water_speed_increase_l1090_109049

variable (S : ℝ) -- Total distance
variable (V : ℝ) -- Speed of the ferry in still water
variable (V1 V2 : ℝ) -- Speed of the water flow before and after increase

-- Ensure realistic conditions
axiom V_pos : 0 < V
axiom V1_pos : 0 < V1
axiom V2_pos : 0 < V2
axiom V1_less_V : V1 < V
axiom V2_less_V : V2 < V
axiom V1_less_V2 : V1 < V2

theorem time_increases_with_water_speed_increase :
  (S / (V + V1) + S / (V - V1)) < (S / (V + V2) + S / (V - V2)) :=
sorry

end time_increases_with_water_speed_increase_l1090_109049


namespace find_a_value_l1090_109090

theorem find_a_value (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 + x2 = 15) 
  (h3 : ∀ x, x^2 - 2 * a * x - 8 * a^2 < 0) : a = 15 / 2 :=
  sorry

end find_a_value_l1090_109090


namespace aluminum_foil_thickness_l1090_109055

-- Define the variables and constants
variables (d l m w t : ℝ)

-- Define the conditions
def density_condition : Prop := d = m / (l * w * t)
def volume_formula : Prop := t = m / (d * l * w)

-- The theorem to prove
theorem aluminum_foil_thickness (h1 : density_condition d l m w t) : volume_formula d l m w t :=
sorry

end aluminum_foil_thickness_l1090_109055


namespace sufficient_but_not_necessary_condition_l1090_109046

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a < -1) → (|a| > 1) ∧ ¬((|a| > 1) → (a < -1)) :=
by
-- This statement represents the required proof.
sorry

end sufficient_but_not_necessary_condition_l1090_109046


namespace solve_system_l1090_109030

-- Define the conditions
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 8) ∧ (2 * x - y = 7)

-- Define the proof problem statement
theorem solve_system : 
  system_of_equations 5 3 :=
by
  -- Proof will be filled in here
  sorry

end solve_system_l1090_109030


namespace clerks_needed_eq_84_l1090_109017

def forms_processed_per_hour : ℕ := 25
def type_a_forms_count : ℕ := 3000
def type_b_forms_count : ℕ := 4000
def type_a_form_time_minutes : ℕ := 3
def type_b_form_time_minutes : ℕ := 4
def working_hours_per_day : ℕ := 5
def total_minutes_in_an_hour : ℕ := 60
def forms_time_needed (count : ℕ) (time_per_form : ℕ) : ℕ := count * time_per_form
def total_forms_time_needed : ℕ := forms_time_needed type_a_forms_count type_a_form_time_minutes +
                                    forms_time_needed type_b_forms_count type_b_form_time_minutes
def total_hours_needed : ℕ := total_forms_time_needed / total_minutes_in_an_hour
def clerk_hours_needed : ℕ := total_hours_needed / working_hours_per_day
def required_clerks : ℕ := Nat.ceil (clerk_hours_needed)

theorem clerks_needed_eq_84 :
  required_clerks = 84 :=
by
  sorry

end clerks_needed_eq_84_l1090_109017


namespace probability_toner_never_displayed_l1090_109026

theorem probability_toner_never_displayed:
  let total_votes := 129
  let toner_votes := 63
  let celery_votes := 66
  (toner_votes + celery_votes = total_votes) →
  let probability := (celery_votes - toner_votes) / (celery_votes + toner_votes)
  probability = 1 / 43 := 
by
  sorry

end probability_toner_never_displayed_l1090_109026


namespace find_constants_l1090_109035

theorem find_constants (a b : ℝ) (h₀ : ∀ x : ℝ, (x^3 + 3*a*x^2 + b*x + a^2 = 0 → x = -1)) :
    a = 2 ∧ b = 9 :=
by
  sorry

end find_constants_l1090_109035


namespace original_cost_of_luxury_bag_l1090_109056

theorem original_cost_of_luxury_bag (SP : ℝ) (profit_margin : ℝ) (original_cost : ℝ) 
  (h1 : SP = 3450) (h2 : profit_margin = 0.15) (h3 : SP = original_cost * (1 + profit_margin)) : 
  original_cost = 3000 :=
by
  sorry

end original_cost_of_luxury_bag_l1090_109056


namespace arctan_sum_pi_l1090_109083

open Real

theorem arctan_sum_pi : arctan (1 / 3) + arctan (3 / 8) + arctan (8 / 3) = π := 
sorry

end arctan_sum_pi_l1090_109083


namespace fib_subsequence_fib_l1090_109011

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fib_subsequence_fib (p : ℕ) (hp : p > 0) :
  ∀ n : ℕ, fibonacci ((n - 1) * p) + fibonacci (n * p) = fibonacci ((n + 1) * p) := 
by
  sorry

end fib_subsequence_fib_l1090_109011


namespace acute_triangle_altitude_inequality_l1090_109002

theorem acute_triangle_altitude_inequality (a b c d e f : ℝ) 
  (A B C : ℝ) 
  (acute_triangle : (d = b * Real.sin C) ∧ (d = c * Real.sin B) ∧
                    (e = a * Real.sin C) ∧ (f = a * Real.sin B))
  (projections : (de = b * Real.cos B) ∧ (df = c * Real.cos C))
  : (de + df ≤ a) := 
sorry

end acute_triangle_altitude_inequality_l1090_109002


namespace tom_and_mary_age_l1090_109023

-- Define Tom's and Mary's ages
variables (T M : ℕ)

-- Define the two given conditions
def condition1 : Prop := T^2 + M = 62
def condition2 : Prop := M^2 + T = 176

-- State the theorem
theorem tom_and_mary_age (h1 : condition1 T M) (h2 : condition2 T M) : T = 7 ∧ M = 13 :=
by {
  -- sorry acts as a placeholder for the proof
  sorry
}

end tom_and_mary_age_l1090_109023


namespace difference_q_r_share_l1090_109053

theorem difference_q_r_share (x : ℝ) (h1 : 7 * x - 3 * x = 2800) :
  12 * x - 7 * x = 3500 :=
by
  sorry

end difference_q_r_share_l1090_109053


namespace fraction_of_females_l1090_109082

variable (participants_last_year males_last_year females_last_year males_this_year females_this_year participants_this_year : ℕ)

-- The conditions
def conditions :=
  males_last_year = 20 ∧
  participants_this_year = (110 * (participants_last_year/100)) ∧
  males_this_year = (105 * males_last_year / 100) ∧
  females_this_year = (120 * females_last_year / 100) ∧
  participants_last_year = males_last_year + females_last_year ∧
  participants_this_year = males_this_year + females_this_year

-- The proof statement
theorem fraction_of_females (h : conditions males_last_year females_last_year males_this_year females_this_year participants_last_year participants_this_year) :
  (females_this_year : ℚ) / (participants_this_year : ℚ) = 4 / 11 :=
  sorry

end fraction_of_females_l1090_109082


namespace find_value_of_x8_plus_x4_plus_1_l1090_109036

theorem find_value_of_x8_plus_x4_plus_1 (x : ℂ) (hx : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 :=
sorry

end find_value_of_x8_plus_x4_plus_1_l1090_109036


namespace opposite_of_7_l1090_109075

-- Define the concept of an opposite number for real numbers
def is_opposite (x y : ℝ) : Prop := x = -y

-- Theorem statement
theorem opposite_of_7 :
  is_opposite 7 (-7) :=
by {
  sorry
}

end opposite_of_7_l1090_109075


namespace fraction_lost_down_sewer_l1090_109034

-- Definitions of the conditions derived from the problem
def initial_marbles := 100
def street_loss_percent := 60 / 100
def sewer_loss := 40 - 20
def remaining_marbles_after_street := initial_marbles - (initial_marbles * street_loss_percent)
def marbles_left := 20

-- The theorem statement proving the fraction of remaining marbles lost down the sewer
theorem fraction_lost_down_sewer :
  (sewer_loss / remaining_marbles_after_street) = 1 / 2 :=
by
  sorry

end fraction_lost_down_sewer_l1090_109034


namespace find_r_s_l1090_109037

noncomputable def r_s_proof_problem (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : Prop :=
(r, s) = (4, 5)

theorem find_r_s (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : r_s_proof_problem r s h1 h2 :=
sorry

end find_r_s_l1090_109037


namespace range_of_b_l1090_109022

-- Given a function f(x)
def f (b x : ℝ) : ℝ := x^3 - 3 * b * x + 3 * b

-- Derivative of the function f(x)
def f' (b x : ℝ) : ℝ := 3 * x^2 - 3 * b

-- The theorem to prove the range of b
theorem range_of_b (b : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f' b x = 0) → (0 < b ∧ b < 1) := by
  sorry

end range_of_b_l1090_109022


namespace fraction_of_cream_in_cup1_after_operations_l1090_109009

/-
We consider two cups of liquids with the following contents initially:
Cup 1 has 6 ounces of coffee.
Cup 2 has 2 ounces of coffee and 4 ounces of cream.
After pouring half of Cup 1's content into Cup 2, stirring, and then pouring half of Cup 2's new content back into Cup 1, we need to show that 
the fraction of the liquid in Cup 1 that is now cream is 4/15.
-/

theorem fraction_of_cream_in_cup1_after_operations :
  let cup1_initial_coffee := 6
  let cup2_initial_coffee := 2
  let cup2_initial_cream := 4
  let cup2_initial_liquid := cup2_initial_coffee + cup2_initial_cream
  let cup1_to_cup2_coffee := cup1_initial_coffee / 2
  let cup1_final_coffee := cup1_initial_coffee - cup1_to_cup2_coffee
  let cup2_final_coffee := cup2_initial_coffee + cup1_to_cup2_coffee
  let cup2_final_liquid := cup2_final_coffee + cup2_initial_cream
  let cup2_to_cup1_liquid := cup2_final_liquid / 2
  let cup2_coffee_fraction := cup2_final_coffee / cup2_final_liquid
  let cup2_cream_fraction := cup2_initial_cream / cup2_final_liquid
  let cup2_to_cup1_coffee := cup2_to_cup1_liquid * cup2_coffee_fraction
  let cup2_to_cup1_cream := cup2_to_cup1_liquid * cup2_cream_fraction
  let cup1_final_liquid_coffee := cup1_final_coffee + cup2_to_cup1_coffee
  let cup1_final_liquid_cream := cup2_to_cup1_cream
  let cup1_final_liquid := cup1_final_liquid_coffee + cup1_final_liquid_cream
  (cup1_final_liquid_cream / cup1_final_liquid) = 4 / 15 :=
by
  sorry

end fraction_of_cream_in_cup1_after_operations_l1090_109009


namespace sale_price_is_correct_l1090_109050

def initial_price : ℝ := 560
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.30
def discount3 : ℝ := 0.15
def tax_rate : ℝ := 0.12

noncomputable def final_price : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let price_after_third_discount := price_after_second_discount * (1 - discount3)
  let price_after_tax := price_after_third_discount * (1 + tax_rate)
  price_after_tax

theorem sale_price_is_correct :
  final_price = 298.55 :=
sorry

end sale_price_is_correct_l1090_109050


namespace ratio_of_candies_l1090_109057

theorem ratio_of_candies (candiesEmily candiesBob : ℕ) (candiesJennifer : ℕ) 
  (hEmily : candiesEmily = 6) 
  (hBob : candiesBob = 4)
  (hJennifer : candiesJennifer = 3 * candiesBob) : 
  (candiesJennifer / Nat.gcd candiesJennifer candiesEmily) = 2 ∧ (candiesEmily / Nat.gcd candiesJennifer candiesEmily) = 1 := 
by
  sorry

end ratio_of_candies_l1090_109057


namespace find_a_l1090_109093

open Set

theorem find_a (A : Set ℝ) (B : Set ℝ) (f : ℝ → ℝ) (a : ℝ)
  (hA : A = Ici 0) 
  (hB : B = univ)
  (hf : ∀ x ∈ A, f x = 2^x - 1) 
  (ha_in_A : a ∈ A) 
  (ha_f_eq_3 : f a = 3) :
  a = 2 := 
by
  sorry

end find_a_l1090_109093


namespace smallest_prime_8_less_than_square_l1090_109097

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l1090_109097


namespace cos_alpha_add_beta_over_2_l1090_109010

variable (α β : ℝ)

-- Conditions
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : Real.cos (π / 4 + α) = 1 / 3)
variables (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Result
theorem cos_alpha_add_beta_over_2 :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_2_l1090_109010


namespace half_abs_diff_squares_eq_40_l1090_109064

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l1090_109064


namespace max_value_fractions_l1090_109043

noncomputable def maxFractions (a b c : ℝ) : ℝ :=
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c)

theorem max_value_fractions (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
    (h_sum : a + b + c = 2) :
    maxFractions a b c ≤ 1 ∧ 
    (a = 2 / 3 ∧ b = 2 / 3 ∧ c = 2 / 3 → maxFractions a b c = 1) := 
  by
    sorry

end max_value_fractions_l1090_109043


namespace keats_library_percentage_increase_l1090_109019

theorem keats_library_percentage_increase :
  let total_books_A := 8000
  let total_books_B := 10000
  let total_books_C := 12000
  let initial_bio_A := 0.20 * total_books_A
  let initial_bio_B := 0.25 * total_books_B
  let initial_bio_C := 0.28 * total_books_C
  let total_initial_bio := initial_bio_A + initial_bio_B + initial_bio_C
  let final_bio_A := 0.32 * total_books_A
  let final_bio_B := 0.35 * total_books_B
  let final_bio_C := 0.40 * total_books_C
  --
  let total_final_bio := final_bio_A + final_bio_B + final_bio_C
  let increase_in_bio := total_final_bio - total_initial_bio
  let percentage_increase := (increase_in_bio / total_initial_bio) * 100
  --
  percentage_increase = 45.58 := 
by
  sorry

end keats_library_percentage_increase_l1090_109019


namespace value_of_linear_combination_l1090_109016

theorem value_of_linear_combination :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    2*x1 + x2 + x3 + x4 + x5 = 6 →
    x1 + 2*x2 + x3 + x4 + x5 = 12 →
    x1 + x2 + 2*x3 + x4 + x5 = 24 →
    x1 + x2 + x3 + 2*x4 + x5 = 48 →
    x1 + x2 + x3 + x4 + 2*x5 = 96 →
    3*x4 + 2*x5 = 181 :=
by
  intros x1 x2 x3 x4 x5 h1 h2 h3 h4 h5
  sorry

end value_of_linear_combination_l1090_109016


namespace find_number_l1090_109094

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l1090_109094


namespace find_h_l1090_109006

theorem find_h (h j k : ℤ) (y_intercept1 : 3 * h ^ 2 + j = 2013) 
  (y_intercept2 : 2 * h ^ 2 + k = 2014)
  (x_intercepts1 : ∃ (y : ℤ), j = -3 * y ^ 2)
  (x_intercepts2 : ∃ (x : ℤ), k = -2 * x ^ 2) :
  h = 36 :=
by sorry

end find_h_l1090_109006


namespace unique_solution_p_zero_l1090_109070

theorem unique_solution_p_zero :
  ∃! (x y p : ℝ), 
    (x^2 - y^2 = 0) ∧ 
    (x * y + p * x - p * y = p^2) ↔ 
    p = 0 :=
by sorry

end unique_solution_p_zero_l1090_109070


namespace minute_hand_angle_l1090_109048

theorem minute_hand_angle (minutes_slow : ℕ) (total_minutes : ℕ) (full_rotation : ℝ) (h1 : minutes_slow = 5) (h2 : total_minutes = 60) (h3 : full_rotation = 2 * Real.pi) : 
  (minutes_slow / total_minutes : ℝ) * full_rotation = Real.pi / 6 :=
by
  sorry

end minute_hand_angle_l1090_109048


namespace find_number_l1090_109079

theorem find_number (a : ℕ) (h : a = 105) : 
  a^3 / (49 * 45 * 25) = 21 :=
by
  sorry

end find_number_l1090_109079


namespace reggie_marbles_bet_l1090_109085

theorem reggie_marbles_bet 
  (initial_marbles : ℕ) (final_marbles : ℕ) (games_played : ℕ) (games_lost : ℕ) (bet_per_game : ℕ)
  (h_initial : initial_marbles = 100) 
  (h_final : final_marbles = 90) 
  (h_games : games_played = 9) 
  (h_losses : games_lost = 1) : 
  bet_per_game = 13 :=
by
  sorry

end reggie_marbles_bet_l1090_109085


namespace sub_frac_pow_eq_l1090_109089

theorem sub_frac_pow_eq :
  7 - (2 / 5)^3 = 867 / 125 := by
  sorry

end sub_frac_pow_eq_l1090_109089


namespace minerals_found_today_l1090_109068

noncomputable def yesterday_gemstones := 21
noncomputable def today_minerals := 48
noncomputable def today_gemstones := 21

theorem minerals_found_today :
  (today_minerals - (2 * yesterday_gemstones) = 6) :=
by
  sorry

end minerals_found_today_l1090_109068


namespace wall_paint_area_l1090_109032

theorem wall_paint_area
  (A₁ : ℕ) (A₂ : ℕ) (A₃ : ℕ) (A₄ : ℕ)
  (H₁ : A₁ = 32)
  (H₂ : A₂ = 48)
  (H₃ : A₃ = 32)
  (H₄ : A₄ = 48) :
  A₁ + A₂ + A₃ + A₄ = 160 :=
by
  sorry

end wall_paint_area_l1090_109032


namespace complement_set_l1090_109003

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = Real.log x / Real.log 2}

theorem complement_set :
  Set.compl M = {y : ℝ | y ≥ 0} :=
by
  sorry

end complement_set_l1090_109003


namespace sum_of_products_of_two_at_a_time_l1090_109076

theorem sum_of_products_of_two_at_a_time (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a + b + c = 21) : 
  a * b + b * c + a * c = 100 := 
  sorry

end sum_of_products_of_two_at_a_time_l1090_109076


namespace afternoon_sales_l1090_109041

theorem afternoon_sales (x : ℕ) (h : 3 * x = 510) : 2 * x = 340 :=
by sorry

end afternoon_sales_l1090_109041


namespace total_rowing_proof_l1090_109012

def morning_rowing := 13
def afternoon_rowing := 21
def total_rowing := 34

theorem total_rowing_proof :
  morning_rowing + afternoon_rowing = total_rowing :=
by
  sorry

end total_rowing_proof_l1090_109012


namespace total_toothpicks_grid_area_l1090_109020

open Nat

-- Definitions
def grid_length : Nat := 30
def grid_width : Nat := 50

-- Prove the total number of toothpicks
theorem total_toothpicks : (31 * grid_width + 51 * grid_length) = 3080 := by
  sorry

-- Prove the area enclosed by the grid
theorem grid_area : (grid_length * grid_width) = 1500 := by
  sorry

end total_toothpicks_grid_area_l1090_109020


namespace part1_part2_l1090_109067

def P : Set ℝ := {x | x ≥ 1 / 2 ∧ x ≤ 2}

def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 > 0}

def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 = 0}

theorem part1 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ Q a) → a > -1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ R a) → a ≥ -1 / 2 ∧ a ≤ 1 / 2 :=
by
  sorry

end part1_part2_l1090_109067


namespace hall_ratio_l1090_109013

open Real

theorem hall_ratio (w l : ℝ) (h_area : w * l = 288) (h_diff : l - w = 12) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l1090_109013


namespace total_dogs_in_kennel_l1090_109065

-- Definition of the given conditions
def T := 45       -- Number of dogs that wear tags
def C := 40       -- Number of dogs that wear flea collars
def B := 6        -- Number of dogs that wear both tags and collars
def D_neither := 1 -- Number of dogs that wear neither a collar nor tags

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + D_neither = 80 := 
by
  -- Proof omitted
  sorry

end total_dogs_in_kennel_l1090_109065


namespace max_soap_boxes_l1090_109062

theorem max_soap_boxes 
  (base_width base_length top_width top_length height soap_width soap_length soap_height max_weight soap_weight : ℝ)
  (h_base_dims : base_width = 25)
  (h_base_len : base_length = 42)
  (h_top_width : top_width = 20)
  (h_top_length : top_length = 35)
  (h_height : height = 60)
  (h_soap_width : soap_width = 7)
  (h_soap_length : soap_length = 6)
  (h_soap_height : soap_height = 10)
  (h_max_weight : max_weight = 150)
  (h_soap_weight : soap_weight = 3) :
  (50 = 
    min 
      (⌊top_width / soap_width⌋ * ⌊top_length / soap_length⌋ * ⌊height / soap_height⌋)
      (⌊max_weight / soap_weight⌋)) := by sorry

end max_soap_boxes_l1090_109062


namespace contest_score_order_l1090_109077

variables (E F G H : ℕ) -- nonnegative scores of Emily, Fran, Gina, and Harry respectively

-- Conditions
axiom cond1 : E - F = G + H + 10
axiom cond2 : G + E > F + H + 5
axiom cond3 : H = F + 8

-- Statement to prove
theorem contest_score_order : (H > E) ∧ (E > F) ∧ (F > G) :=
sorry

end contest_score_order_l1090_109077


namespace symmetry_y_axis_l1090_109028

theorem symmetry_y_axis (A B C D : ℝ → ℝ → Prop) 
  (A_eq : ∀ x y : ℝ, A x y ↔ (x^2 - x + y^2 = 1))
  (B_eq : ∀ x y : ℝ, B x y ↔ (x^2 * y + x * y^2 = 1))
  (C_eq : ∀ x y : ℝ, C x y ↔ (x^2 - y^2 = 1))
  (D_eq : ∀ x y : ℝ, D x y ↔ (x - y = 1)) : 
  (∀ x y : ℝ, C x y ↔ C (-x) y) ∧ 
  ¬(∀ x y : ℝ, A x y ↔ A (-x) y) ∧ 
  ¬(∀ x y : ℝ, B x y ↔ B (-x) y) ∧ 
  ¬(∀ x y : ℝ, D x y ↔ D (-x) y) :=
by
  -- Proof goes here
  sorry

end symmetry_y_axis_l1090_109028


namespace cube_inequality_l1090_109038

theorem cube_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 :=
by 
  sorry

end cube_inequality_l1090_109038


namespace domain_of_g_l1090_109029

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : {x : ℝ | x > 6^625} = {x : ℝ | ∃ y : ℝ, y = g x } := sorry

end domain_of_g_l1090_109029


namespace find_complement_intersection_find_union_complement_subset_implies_a_range_l1090_109072

-- Definitions for sets A and B
def A : Set ℝ := { x | 3 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 2 < x ∧ x < 9 }

-- Definitions for complements and subsets
def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S }
def intersection (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∧ x ∈ T }
def union (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∨ x ∈ T }

-- Definition for set C as a parameterized set by a
def C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Proof statements
theorem find_complement_intersection :
  complement (intersection A B) = { x | x < 3 ∨ x ≥ 6 } :=
by sorry

theorem find_union_complement :
  union (complement B) A = { x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by sorry

theorem subset_implies_a_range (a : ℝ) :
  C a ⊆ B → a ∈ {x | 2 ≤ x ∧ x ≤ 8} :=
by sorry

end find_complement_intersection_find_union_complement_subset_implies_a_range_l1090_109072


namespace cape_may_vs_daytona_shark_sightings_diff_l1090_109044

-- Definitions based on the conditions
def total_shark_sightings := 40
def cape_may_sightings : ℕ := 24
def daytona_beach_sightings : ℕ := total_shark_sightings - cape_may_sightings

-- The main theorem stating the problem in Lean
theorem cape_may_vs_daytona_shark_sightings_diff :
  (2 * daytona_beach_sightings - cape_may_sightings) = 8 := by
  sorry

end cape_may_vs_daytona_shark_sightings_diff_l1090_109044


namespace podcast_ratio_l1090_109061

theorem podcast_ratio
  (total_drive_time : ℕ)
  (first_podcast : ℕ)
  (third_podcast : ℕ)
  (fourth_podcast : ℕ)
  (next_podcast : ℕ)
  (second_podcast : ℕ) :
  total_drive_time = 360 →
  first_podcast = 45 →
  third_podcast = 105 →
  fourth_podcast = 60 →
  next_podcast = 60 →
  second_podcast = total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast) →
  second_podcast / first_podcast = 2 :=
by
  sorry

end podcast_ratio_l1090_109061


namespace find_f_2015_plus_f_2016_l1090_109096

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom functional_equation (x : ℝ) : f (3/2 - x) = f x
axiom value_at_minus2 : f (-2) = -3

theorem find_f_2015_plus_f_2016 : f 2015 + f 2016 = 3 := 
by {
  sorry
}

end find_f_2015_plus_f_2016_l1090_109096
