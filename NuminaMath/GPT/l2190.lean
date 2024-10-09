import Mathlib

namespace find_x_l2190_219089

theorem find_x
    (x : ℝ)
    (l : ℝ := 4 * x)
    (w : ℝ := x + 8)
    (area_eq_twice_perimeter : l * w = 2 * (2 * l + 2 * w)) :
    x = 2 :=
by
  sorry

end find_x_l2190_219089


namespace determine_n_for_square_l2190_219078

theorem determine_n_for_square (n : ℕ) : (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 :=
by
-- The proof will be included here, but for now, we just provide the structure
sorry

end determine_n_for_square_l2190_219078


namespace people_distribution_l2190_219016

theorem people_distribution
  (total_mentions : ℕ)
  (mentions_house : ℕ)
  (mentions_fountain : ℕ)
  (mentions_bench : ℕ)
  (mentions_tree : ℕ)
  (each_person_mentions : ℕ)
  (total_people : ℕ)
  (facing_house : ℕ)
  (facing_fountain : ℕ)
  (facing_bench : ℕ)
  (facing_tree : ℕ)
  (h_total_mentions : total_mentions = 27)
  (h_mentions_house : mentions_house = 5)
  (h_mentions_fountain : mentions_fountain = 6)
  (h_mentions_bench : mentions_bench = 7)
  (h_mentions_tree : mentions_tree = 9)
  (h_each_person_mentions : each_person_mentions = 3)
  (h_total_people : total_people = 9)
  (h_facing_house : facing_house = 5)
  (h_facing_fountain : facing_fountain = 4)
  (h_facing_bench : facing_bench = 2)
  (h_facing_tree : facing_tree = 9) :
  total_mentions / each_person_mentions = total_people ∧ 
  facing_house = mentions_house ∧
  facing_fountain = total_people - mentions_house ∧
  facing_bench = total_people - mentions_bench ∧
  facing_tree = total_people - mentions_tree :=
by
  sorry

end people_distribution_l2190_219016


namespace sqrt_diff_approx_l2190_219087

theorem sqrt_diff_approx : abs ((Real.sqrt 122) - (Real.sqrt 120) - 0.15) < 0.01 := 
sorry

end sqrt_diff_approx_l2190_219087


namespace who_had_second_value_card_in_first_game_l2190_219094

variable (A B C : ℕ)
variable (x y z : ℕ)
variable (points_A points_B points_C : ℕ)

-- Provided conditions
variable (h1 : x < y ∧ y < z)
variable (h2 : points_A = 20)
variable (h3 : points_B = 10)
variable (h4 : points_C = 9)
variable (number_of_games : ℕ)
variable (h5 : number_of_games = 3)
variable (h6 : A + B + C = 39)  -- This corresponds to points_A + points_B + points_C = 39.
variable (h7 : ∃ x y z, x + y + z = 13 ∧ x < y ∧ y < z)
variable (h8 : B = z)

-- Question/Proof to establish
theorem who_had_second_value_card_in_first_game :
  ∃ p : ℕ, p = C :=
sorry

end who_had_second_value_card_in_first_game_l2190_219094


namespace oranges_to_apples_ratio_l2190_219090

theorem oranges_to_apples_ratio :
  ∀ (total_fruits : ℕ) (weight_oranges : ℕ) (weight_apples : ℕ),
  total_fruits = 12 →
  weight_oranges = 10 →
  weight_apples = total_fruits - weight_oranges →
  weight_oranges / weight_apples = 5 :=
by
  intros total_fruits weight_oranges weight_apples h1 h2 h3
  sorry

end oranges_to_apples_ratio_l2190_219090


namespace find_d_values_l2190_219035

open Set

theorem find_d_values :
  ∀ {f : ℝ → ℝ}, ContinuousOn f (Icc 0 1) → (f 0 = f 1) →
  ∃ (d : ℝ), d ∈ Ioo 0 1 ∧ (∀ x₀, x₀ ∈ Icc 0 (1 - d) → (f x₀ = f (x₀ + d))) ↔
  ∃ k : ℕ, d = 1 / k :=
by
  sorry

end find_d_values_l2190_219035


namespace fixed_point_coordinates_l2190_219046

theorem fixed_point_coordinates (a b x y : ℝ) 
  (h1 : a + 2 * b = 1) 
  (h2 : (a * x + 3 * y + b) = 0) :
  x = 1 / 2 ∧ y = -1 / 6 := by
  sorry

end fixed_point_coordinates_l2190_219046


namespace snail_reaches_tree_l2190_219044

theorem snail_reaches_tree
  (l1 l2 s : ℝ) 
  (h_l1 : l1 = 4) 
  (h_l2 : l2 = 3) 
  (h_s : s = 40) : 
  ∃ n : ℕ, n = 37 ∧ s - n*(l1 - l2) ≤ l1 :=
  by
    sorry

end snail_reaches_tree_l2190_219044


namespace sin2theta_plus_cos2theta_l2190_219013

theorem sin2theta_plus_cos2theta (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_plus_cos2theta_l2190_219013


namespace problem_l2190_219050

noncomputable def h (p x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15

noncomputable def k (q r x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

theorem problem
  (p q r : ℝ)
  (h_has_distinct_roots: ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h p a = 0 ∧ h p b = 0 ∧ h p c = 0)
  (h_roots_are_k_roots: ∀ x, h p x = 0 → k q r x = 0) :
  k q r 1 = -3322.25 :=
sorry

end problem_l2190_219050


namespace daniel_earnings_l2190_219051

def fabric_monday := 20
def yarn_monday := 15

def fabric_tuesday := 2 * fabric_monday
def yarn_tuesday := yarn_monday + 10

def fabric_wednesday := fabric_tuesday / 4
def yarn_wednesday := yarn_tuesday / 2

def price_per_yard_fabric := 2
def price_per_yard_yarn := 3

def total_fabric := fabric_monday + fabric_tuesday + fabric_wednesday
def total_yarn := yarn_monday + yarn_tuesday + yarn_wednesday

def earnings_fabric := total_fabric * price_per_yard_fabric
def earnings_yarn := total_yarn * price_per_yard_yarn

def total_earnings := earnings_fabric + earnings_yarn

theorem daniel_earnings :
  total_earnings = 299 := by
  sorry

end daniel_earnings_l2190_219051


namespace Cody_initial_money_l2190_219095

-- Define the conditions
def initial_money (x : ℕ) : Prop :=
  x + 9 - 19 = 35

-- Define the theorem we need to prove
theorem Cody_initial_money : initial_money 45 :=
by
  -- Add a placeholder for the proof
  sorry

end Cody_initial_money_l2190_219095


namespace taxi_fare_distance_l2190_219072

variable (x : ℝ)

theorem taxi_fare_distance (h1 : 0 ≤ x - 2) (h2 : 3 + 1.2 * (x - 2) = 9) : x = 7 := by
  sorry

end taxi_fare_distance_l2190_219072


namespace true_discount_double_time_l2190_219065

theorem true_discount_double_time (PV FV1 FV2 I1 I2 TD1 TD2 : ℕ) 
  (h1 : FV1 = 110)
  (h2 : TD1 = 10)
  (h3 : FV1 - TD1 = PV)
  (h4 : I1 = FV1 - PV)
  (h5 : FV2 = PV + 2 * I1)
  (h6 : TD2 = FV2 - PV) :
  TD2 = 20 := by
  sorry

end true_discount_double_time_l2190_219065


namespace alice_bob_meeting_point_l2190_219080

def meet_same_point (turns : ℕ) : Prop :=
  ∃ n : ℕ, turns = 2 * n ∧ 18 ∣ (7 * n - (7 * n + n))

theorem alice_bob_meeting_point :
  meet_same_point 36 :=
by
  sorry

end alice_bob_meeting_point_l2190_219080


namespace henry_apple_weeks_l2190_219064

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end henry_apple_weeks_l2190_219064


namespace find_fg3_l2190_219047

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := (x + 2)^2 - 4 * x

theorem find_fg3 : f (g 3) = 49 :=
by
  sorry

end find_fg3_l2190_219047


namespace find_range_of_t_l2190_219007

variable {f : ℝ → ℝ}

-- Definitions for the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ ⦃x y⦄, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x > f y

-- Given the conditions, we need to prove the statement
theorem find_range_of_t (h_odd : is_odd_function f)
    (h_decreasing : is_decreasing_on f (-1) 1)
    (h_inequality : ∀ t : ℝ, -1 < t ∧ t < 1 → f (1 - t) + f (1 - t^2) < 0) :
  ∀ t, -1 < t ∧ t < 1 → 0 < t ∧ t < 1 :=
  by
  sorry

end find_range_of_t_l2190_219007


namespace bob_and_jim_total_skips_l2190_219066

-- Definitions based on conditions
def bob_skips_per_rock : Nat := 12
def jim_skips_per_rock : Nat := 15
def rocks_skipped_by_each : Nat := 10

-- Total skips calculation based on the given conditions
def bob_total_skips : Nat := bob_skips_per_rock * rocks_skipped_by_each
def jim_total_skips : Nat := jim_skips_per_rock * rocks_skipped_by_each
def total_skips : Nat := bob_total_skips + jim_total_skips

-- Theorem statement
theorem bob_and_jim_total_skips : total_skips = 270 := by
  sorry

end bob_and_jim_total_skips_l2190_219066


namespace cylinder_height_l2190_219034

theorem cylinder_height
  (r : ℝ) (SA : ℝ) (h : ℝ)
  (h_radius : r = 3)
  (h_surface_area_given : SA = 30 * Real.pi)
  (h_surface_area_formula : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  h = 2 :=
by
  -- Proof can be written here
  sorry

end cylinder_height_l2190_219034


namespace combined_tax_rate_l2190_219061

theorem combined_tax_rate
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h_john_income : john_income = 58000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_income : ingrid_income = 72000)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40) :
  ((john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)) = 0.3553846154 :=
by
  sorry

end combined_tax_rate_l2190_219061


namespace enrico_earnings_l2190_219036

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end enrico_earnings_l2190_219036


namespace final_expression_simplified_l2190_219073

variable (b : ℝ)

theorem final_expression_simplified :
  ((3 * b + 6 - 5 * b) / 3) = (-2 / 3) * b + 2 := by
  sorry

end final_expression_simplified_l2190_219073


namespace Tobias_monthly_allowance_l2190_219009

noncomputable def monthly_allowance (shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways : ℕ) : ℕ :=
  (shoes_cost + change - (num_lawns * lawn_charge + num_driveways * driveway_charge)) / monthly_saving_period

theorem Tobias_monthly_allowance :
  let shoes_cost := 95
  let monthly_saving_period := 3
  let lawn_charge := 15
  let driveway_charge := 7
  let change := 15
  let num_lawns := 4
  let num_driveways := 5
  monthly_allowance shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways = 5 :=
by
  sorry

end Tobias_monthly_allowance_l2190_219009


namespace each_person_paid_l2190_219011

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l2190_219011


namespace find_unique_pair_l2190_219043

theorem find_unique_pair (x y : ℝ) :
  (∀ (u v : ℝ), (u * x + v * y = u) ∧ (u * y + v * x = v)) ↔ (x = 1 ∧ y = 0) :=
by
  -- This is to ignore the proof part
  sorry

end find_unique_pair_l2190_219043


namespace mr_green_potato_yield_l2190_219052

theorem mr_green_potato_yield :
  let steps_to_feet := 2.5
  let length_steps := 18
  let width_steps := 25
  let yield_per_sqft := 0.75
  let length_feet := length_steps * steps_to_feet
  let width_feet := width_steps * steps_to_feet
  let area_sqft := length_feet * width_feet
  let expected_yield := area_sqft * yield_per_sqft
  expected_yield = 2109.375 := by sorry

end mr_green_potato_yield_l2190_219052


namespace necessary_but_not_sufficient_condition_l2190_219018

-- Definitions
variable (f : ℝ → ℝ)

-- Condition that we need to prove
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -g (-x)

-- Necessary and sufficient condition
theorem necessary_but_not_sufficient_condition : 
  (∀ x, |f x| = |f (-x)|) ↔ (∀ x, f x = -f (-x)) ∧ ¬(∀ x, |f x| = |f (-x)| → f x = -f (-x)) := by 
sorry

end necessary_but_not_sufficient_condition_l2190_219018


namespace total_collection_value_l2190_219075

theorem total_collection_value (total_stickers : ℕ) (partial_stickers : ℕ) (partial_value : ℕ)
  (same_value : ∀ (stickers : ℕ), stickers = total_stickers → stickers * partial_value / partial_stickers = stickers * (partial_value / partial_stickers)):
  partial_value = 24 ∧ partial_stickers = 6 ∧ total_stickers = 18 → total_stickers * (partial_value / partial_stickers) = 72 :=
by {
  sorry
}

end total_collection_value_l2190_219075


namespace necessary_but_not_sufficient_l2190_219039

def mutually_exclusive (A1 A2 : Prop) : Prop := (A1 ∧ A2) → False
def complementary (A1 A2 : Prop) : Prop := (A1 ∨ A2) ∧ ¬(A1 ∧ A2)

theorem necessary_but_not_sufficient {A1 A2 : Prop}: 
  mutually_exclusive A1 A2 → complementary A1 A2 → (¬(mutually_exclusive A1 A2 → complementary A1 A2) ∧ (complementary A1 A2 → mutually_exclusive A1 A2)) := 
  by
    sorry

end necessary_but_not_sufficient_l2190_219039


namespace sum_partition_ominous_years_l2190_219099

def is_ominous (n : ℕ) : Prop :=
  n = 1 ∨ Nat.Prime n

theorem sum_partition_ominous_years :
  ∀ n : ℕ, (¬ ∃ (A B : Finset ℕ), A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id ∧ A.card = B.card)) ↔ is_ominous n := 
sorry

end sum_partition_ominous_years_l2190_219099


namespace flowers_on_porch_l2190_219071

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l2190_219071


namespace square_diagonal_y_coordinate_l2190_219000

theorem square_diagonal_y_coordinate 
(point_vertex : ℝ × ℝ) 
(x_int : ℝ) 
(area_square : ℝ) 
(y_int : ℝ) :
(point_vertex = (-6, -4)) →
(x_int = 3) →
(area_square = 324) →
(y_int = 5) → 
y_int = 5 := 
by
  intros h1 h2 h3 h4
  exact h4

end square_diagonal_y_coordinate_l2190_219000


namespace equivalent_expression_l2190_219098

theorem equivalent_expression (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h1 : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by
  sorry

end equivalent_expression_l2190_219098


namespace journey_time_l2190_219054

noncomputable def journey_time_proof : Prop :=
  ∃ t1 t2 t3 : ℝ,
    25 * t1 - 25 * t2 + 25 * t3 = 100 ∧
    5 * t1 + 5 * t2 + 25 * t3 = 100 ∧
    25 * t1 + 5 * t2 + 5 * t3 = 100 ∧
    t1 + t2 + t3 = 8

theorem journey_time : journey_time_proof := by sorry

end journey_time_l2190_219054


namespace bench_allocation_l2190_219060

theorem bench_allocation (M : ℕ) : (∃ M, M > 0 ∧ 5 * M = 13 * M) → M = 5 :=
by
  sorry

end bench_allocation_l2190_219060


namespace min_initial_questionnaires_l2190_219024

theorem min_initial_questionnaires 
(N : ℕ) 
(h1 : 0.60 * (N:ℝ) + 0.60 * (N:ℝ) * 0.80 + 0.60 * (N:ℝ) * (0.80^2) ≥ 750) : 
  N ≥ 513 := sorry

end min_initial_questionnaires_l2190_219024


namespace product_not_zero_l2190_219021

theorem product_not_zero (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) : (x - 2) * (x - 5) ≠ 0 := 
by 
  sorry

end product_not_zero_l2190_219021


namespace annie_initial_money_l2190_219031

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end annie_initial_money_l2190_219031


namespace v3_value_at_2_l2190_219001

def f (x : ℝ) : ℝ :=
  x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

def v3 (x : ℝ) : ℝ :=
  ((x - 12) * x + 60) * x - 160

theorem v3_value_at_2 :
  v3 2 = -80 :=
by
  sorry

end v3_value_at_2_l2190_219001


namespace gcd_660_924_l2190_219081

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end gcd_660_924_l2190_219081


namespace value_of_I_l2190_219068

variables (T H I S : ℤ)

theorem value_of_I :
  H = 10 →
  T + H + I + S = 50 →
  H + I + T = 35 →
  S + I + T = 40 →
  I = 15 :=
  by
  sorry

end value_of_I_l2190_219068


namespace point_B_l2190_219019

-- Define constants for perimeter and speed factor
def perimeter : ℕ := 24
def speed_factor : ℕ := 2

-- Define the speeds of Jane and Hector
def hector_speed (s : ℕ) : ℕ := s
def jane_speed (s : ℕ) : ℕ := speed_factor * s

-- Define the times until they meet
def time_until_meeting (s : ℕ) : ℚ := perimeter / (hector_speed s + jane_speed s)

-- Distances walked by Hector and Jane upon meeting
noncomputable def hector_distance (s : ℕ) : ℚ := hector_speed s * time_until_meeting s
noncomputable def jane_distance (s : ℕ) : ℚ := jane_speed s * time_until_meeting s

-- Map the perimeter position to a point
def position_on_track (d : ℚ) : ℚ := d % perimeter

-- When they meet
theorem point_B (s : ℕ) (h₀ : 0 < s) : position_on_track (hector_distance s) = position_on_track (jane_distance s) → 
                          position_on_track (hector_distance s) = 8 := 
by 
  sorry

end point_B_l2190_219019


namespace number_of_speedster_convertibles_l2190_219026

def proof_problem (T : ℕ) :=
  let Speedsters := 2 * T / 3
  let NonSpeedsters := 50
  let TotalInventory := NonSpeedsters * 3
  let SpeedsterConvertibles := 4 * Speedsters / 5
  (Speedsters = 2 * TotalInventory / 3) ∧ (SpeedsterConvertibles = 4 * Speedsters / 5)

theorem number_of_speedster_convertibles : proof_problem 150 → ∃ (x : ℕ), x = 80 :=
by
  -- Provide the definition of Speedsters, NonSpeedsters, TotalInventory, and SpeedsterConvertibles
  sorry

end number_of_speedster_convertibles_l2190_219026


namespace minimal_value_of_function_l2190_219097

theorem minimal_value_of_function (x : ℝ) (hx : x > 1 / 2) :
  (x = 1 → (x^2 + 1) / x = 2) ∧
  (∀ y, (∀ z, z > 1 / 2 → y ≤ (z^2 + 1) / z) → y = 2) :=
by {
  sorry
}

end minimal_value_of_function_l2190_219097


namespace profit_percentage_correct_l2190_219041

noncomputable def overall_profit_percentage : ℚ :=
  let cost_radio := 225
  let overhead_radio := 15
  let price_radio := 300
  let cost_watch := 425
  let overhead_watch := 20
  let price_watch := 525
  let cost_mobile := 650
  let overhead_mobile := 30
  let price_mobile := 800
  
  let total_cost_price := (cost_radio + overhead_radio) + (cost_watch + overhead_watch) + (cost_mobile + overhead_mobile)
  let total_selling_price := price_radio + price_watch + price_mobile
  let total_profit := total_selling_price - total_cost_price
  (total_profit * 100 : ℚ) / total_cost_price
  
theorem profit_percentage_correct :
  overall_profit_percentage = 19.05 := by
  sorry

end profit_percentage_correct_l2190_219041


namespace expression_B_between_2_and_3_l2190_219092

variable (a b : ℝ)
variable (h : 3 * a = 5 * b)

theorem expression_B_between_2_and_3 : 2 < (|a + b| / b) ∧ (|a + b| / b) < 3 :=
by sorry

end expression_B_between_2_and_3_l2190_219092


namespace contractor_absent_days_l2190_219077

theorem contractor_absent_days
    (total_days : ℤ) (work_rate : ℤ) (fine_rate : ℤ) (total_amount : ℤ)
    (x y : ℤ)
    (h1 : total_days = 30)
    (h2 : work_rate = 25)
    (h3 : fine_rate = 75) -- fine_rate here is multiplied by 10 to avoid decimals
    (h4 : total_amount = 4250) -- total_amount multiplied by 10 for the same reason
    (h5 : x + y = total_days)
    (h6 : work_rate * x - fine_rate * y = total_amount) :
  y = 10 := 
by
  -- Here, we would provide the proof steps.
  sorry

end contractor_absent_days_l2190_219077


namespace find_positive_square_root_l2190_219083

theorem find_positive_square_root (x : ℝ) (h_pos : x > 0) (h_eq : x^2 = 625) : x = 25 :=
sorry

end find_positive_square_root_l2190_219083


namespace find_numbers_l2190_219049

theorem find_numbers : ∃ x y : ℕ, x + y = 2016 ∧ (∃ d : ℕ, d < 10 ∧ (x = 10 * y + d) ∧ x = 1833 ∧ y = 183) :=
by 
  sorry

end find_numbers_l2190_219049


namespace fraction_of_married_women_l2190_219030

theorem fraction_of_married_women (total_employees : ℕ) 
  (women_fraction : ℝ) (married_fraction : ℝ) (single_men_fraction : ℝ)
  (hwf : women_fraction = 0.64) (hmf : married_fraction = 0.60) 
  (hsf : single_men_fraction = 2/3) : 
  ∃ (married_women_fraction : ℝ), married_women_fraction = 3/4 := 
by
  sorry

end fraction_of_married_women_l2190_219030


namespace yanni_money_left_in_cents_l2190_219025

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end yanni_money_left_in_cents_l2190_219025


namespace top_face_not_rotated_by_90_l2190_219004

-- Define the cube and the conditions of rolling and returning
structure Cube :=
  (initial_top_face_orientation : ℕ) -- an integer representation of the orientation of the top face
  (position : ℤ × ℤ) -- (x, y) coordinates on a 2D plane

def rolls_over_edges (c : Cube) : Cube :=
  sorry -- placeholder for the actual rolling operation

def returns_to_original_position (c : Cube) (original : Cube) : Prop :=
  c.position = original.position ∧ c.initial_top_face_orientation = original.initial_top_face_orientation

-- The main theorem to prove
theorem top_face_not_rotated_by_90 {c : Cube} (original : Cube) :
  returns_to_original_position c original → c.initial_top_face_orientation ≠ (original.initial_top_face_orientation + 1) % 4 :=
sorry

end top_face_not_rotated_by_90_l2190_219004


namespace median_hypotenuse_right_triangle_l2190_219062

/-- Prove that in a right triangle with legs of lengths 5 and 12,
  the median on the hypotenuse can be either 6 or 6.5. -/
theorem median_hypotenuse_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) :
  ∃ c : ℝ, (c = 6 ∨ c = 6.5) :=
sorry

end median_hypotenuse_right_triangle_l2190_219062


namespace colleen_paid_more_l2190_219002

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l2190_219002


namespace max_non_intersecting_diagonals_l2190_219012

theorem max_non_intersecting_diagonals (n : ℕ) (h : n ≥ 3) :
  ∃ k, k ≤ n - 3 ∧ (∀ m, m > k → ¬(m ≤ n - 3)) :=
by
  sorry

end max_non_intersecting_diagonals_l2190_219012


namespace Tony_can_add_4_pairs_of_underwear_l2190_219008

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l2190_219008


namespace dice_multiple_3_prob_l2190_219028

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end dice_multiple_3_prob_l2190_219028


namespace distance_A_C_15_l2190_219082

noncomputable def distance_from_A_to_C : ℝ := 
  let AB := 6
  let AC := AB + (3 * AB) / 2
  AC

theorem distance_A_C_15 (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 24) (h5 : D - B = 3 * (B - A)) 
  (h6 : C = (B + D) / 2) :
  distance_from_A_to_C = 15 :=
by sorry

end distance_A_C_15_l2190_219082


namespace max_M_value_l2190_219063

noncomputable def M (x y z w : ℝ) : ℝ :=
  x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z

theorem max_M_value (x y z w : ℝ) (h : x + y + z + w = 1) :
  (M x y z w) ≤ 3 / 2 :=
sorry

end max_M_value_l2190_219063


namespace bookA_net_change_bookB_net_change_bookC_net_change_l2190_219088

-- Define the price adjustments for Book A
def bookA_initial_price := 100.0
def bookA_after_first_adjustment := bookA_initial_price * (1 - 0.5)
def bookA_after_second_adjustment := bookA_after_first_adjustment * (1 + 0.6)
def bookA_final_price := bookA_after_second_adjustment * (1 + 0.1)
def bookA_net_percentage_change := (bookA_final_price - bookA_initial_price) / bookA_initial_price * 100

-- Define the price adjustments for Book B
def bookB_initial_price := 100.0
def bookB_after_first_adjustment := bookB_initial_price * (1 + 0.2)
def bookB_after_second_adjustment := bookB_after_first_adjustment * (1 - 0.3)
def bookB_final_price := bookB_after_second_adjustment * (1 + 0.25)
def bookB_net_percentage_change := (bookB_final_price - bookB_initial_price) / bookB_initial_price * 100

-- Define the price adjustments for Book C
def bookC_initial_price := 100.0
def bookC_after_first_adjustment := bookC_initial_price * (1 + 0.4)
def bookC_after_second_adjustment := bookC_after_first_adjustment * (1 - 0.1)
def bookC_final_price := bookC_after_second_adjustment * (1 - 0.05)
def bookC_net_percentage_change := (bookC_final_price - bookC_initial_price) / bookC_initial_price * 100

-- Statements to prove the net percentage changes
theorem bookA_net_change : bookA_net_percentage_change = -12 := by
  sorry

theorem bookB_net_change : bookB_net_percentage_change = 5 := by
  sorry

theorem bookC_net_change : bookC_net_percentage_change = 19.7 := by
  sorry

end bookA_net_change_bookB_net_change_bookC_net_change_l2190_219088


namespace max_min_x2_minus_xy_plus_y2_l2190_219085

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2_l2190_219085


namespace initial_kids_count_l2190_219010

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l2190_219010


namespace measure_angle_C_and_area_l2190_219084

noncomputable def triangleProblem (a b c A B C : ℝ) :=
  (a + b = 5) ∧ (c = Real.sqrt 7) ∧ (4 * Real.sin ((A + B) / 2)^2 - Real.cos (2 * C) = 7 / 2)

theorem measure_angle_C_and_area (a b c A B C : ℝ) (h: triangleProblem a b c A B C) :
  C = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  sorry

end measure_angle_C_and_area_l2190_219084


namespace infinite_superset_of_infinite_subset_l2190_219070

theorem infinite_superset_of_infinite_subset {A B : Set ℕ} (h_subset : B ⊆ A) (h_infinite : Infinite B) : Infinite A := 
sorry

end infinite_superset_of_infinite_subset_l2190_219070


namespace bricks_for_wall_l2190_219074

theorem bricks_for_wall
  (wall_length : ℕ) (wall_height : ℕ) (wall_width : ℕ)
  (brick_length : ℕ) (brick_height : ℕ) (brick_width : ℕ)
  (L_eq : wall_length = 600) (H_eq : wall_height = 400) (W_eq : wall_width = 2050)
  (l_eq : brick_length = 30) (h_eq : brick_height = 12) (w_eq : brick_width = 10)
  : (wall_length * wall_height * wall_width) / (brick_length * brick_height * brick_width) = 136667 :=
by
  sorry

end bricks_for_wall_l2190_219074


namespace solve_for_n_l2190_219093

theorem solve_for_n (n : ℝ) : 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 → n = 62.5 :=
by
  intros h
  sorry

end solve_for_n_l2190_219093


namespace toothpicks_in_12th_stage_l2190_219059

def toothpicks_in_stage (n : ℕ) : ℕ :=
  3 * n

theorem toothpicks_in_12th_stage : toothpicks_in_stage 12 = 36 :=
by
  -- Proof steps would go here, including simplification and calculations, but are omitted with 'sorry'.
  sorry

end toothpicks_in_12th_stage_l2190_219059


namespace two_trains_crossing_time_l2190_219086

theorem two_trains_crossing_time
  (length_train: ℝ) (time_telegraph_post_first: ℝ) (time_telegraph_post_second: ℝ)
  (length_train_eq: length_train = 120) 
  (time_telegraph_post_first_eq: time_telegraph_post_first = 10) 
  (time_telegraph_post_second_eq: time_telegraph_post_second = 15) :
  (2 * length_train) / (length_train / time_telegraph_post_first + length_train / time_telegraph_post_second) = 12 :=
by
  sorry

end two_trains_crossing_time_l2190_219086


namespace rectangular_solid_depth_l2190_219040

def SurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth
  (l w A : ℝ)
  (hl : l = 10)
  (hw : w = 9)
  (hA : A = 408) :
  ∃ h : ℝ, SurfaceArea l w h = A ∧ h = 6 :=
by
  use 6
  sorry

end rectangular_solid_depth_l2190_219040


namespace verify_optionD_is_correct_l2190_219005

-- Define the equations as options
def optionA : Prop := -abs (-6) = 6
def optionB : Prop := -(-6) = -6
def optionC : Prop := abs (-6) = -6
def optionD : Prop := -(-6) = 6

-- The proof problem to verify option D is correct
theorem verify_optionD_is_correct : optionD :=
by
  sorry

end verify_optionD_is_correct_l2190_219005


namespace complement_union_A_B_in_U_l2190_219045

open Set Nat

def U : Set ℕ := { x | x < 6 ∧ x > 0 }
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_A_B_in_U : (U \ (A ∪ B)) = {2, 4} := by
  sorry

end complement_union_A_B_in_U_l2190_219045


namespace inequality_solution_reciprocal_inequality_l2190_219053

-- Proof Problem (1)
theorem inequality_solution (x : ℝ) : |x-1| + (1/2)*|x-3| < 2 ↔ (1 < x ∧ x < 3) :=
sorry

-- Proof Problem (2)
theorem reciprocal_inequality (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 2) : 
  (1/a) + (1/b) + (1/c) ≥ 9/2 :=
sorry

end inequality_solution_reciprocal_inequality_l2190_219053


namespace distinct_units_digits_of_cube_l2190_219022

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l2190_219022


namespace water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l2190_219015

-- Define the tiered water pricing function
def tiered_water_cost (m : ℕ) : ℝ :=
  if m ≤ 20 then
    1.6 * m
  else if m ≤ 30 then
    1.6 * 20 + 2.4 * (m - 20)
  else
    1.6 * 20 + 2.4 * 10 + 4.8 * (m - 30)

-- Problem 1
theorem water_cost_10_tons : tiered_water_cost 10 = 16 := 
sorry

-- Problem 2
theorem water_cost_27_tons : tiered_water_cost 27 = 48.8 := 
sorry

-- Problem 3
theorem water_cost_between_20_30 (m : ℕ) (h : 20 < m ∧ m < 30) : tiered_water_cost m = 2.4 * m - 16 := 
sorry

-- Problem 4
theorem water_cost_above_30 (m : ℕ) (h : m > 30) : tiered_water_cost m = 4.8 * m - 88 := 
sorry

end water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l2190_219015


namespace dividend_percentage_paid_by_company_l2190_219029

-- Define the parameters
def faceValue : ℝ := 50
def investmentReturnPercentage : ℝ := 25
def investmentPerShare : ℝ := 37

-- Define the theorem
theorem dividend_percentage_paid_by_company :
  (investmentReturnPercentage / 100 * investmentPerShare / faceValue * 100) = 18.5 :=
by
  -- The proof is omitted
  sorry

end dividend_percentage_paid_by_company_l2190_219029


namespace half_percent_of_160_l2190_219057

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l2190_219057


namespace multiple_of_first_number_l2190_219055

theorem multiple_of_first_number (F S M : ℕ) (hF : F = 15) (hS : S = 55) (h_relation : S = M * F + 10) : M = 3 :=
by
  -- We are given that F = 15, S = 55 and the relation S = M * F + 10
  -- We need to prove that M = 3
  sorry

end multiple_of_first_number_l2190_219055


namespace maisy_earnings_increase_l2190_219006

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l2190_219006


namespace necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l2190_219067

theorem necessary_ab_given_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 4) : 
  a + b ≥ 4 :=
sorry

theorem not_sufficient_ab_given_a_b : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4 :=
sorry

end necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l2190_219067


namespace total_sample_variance_l2190_219042

/-- In a survey of the heights (in cm) of high school students at Shuren High School:

 - 20 boys were selected with an average height of 174 cm and a variance of 12.
 - 30 girls were selected with an average height of 164 cm and a variance of 30.

We need to prove that the variance of the total sample is 46.8. -/
theorem total_sample_variance :
  let boys_count := 20
  let girls_count := 30
  let boys_avg := 174
  let girls_avg := 164
  let boys_var := 12
  let girls_var := 30
  let total_count := boys_count + girls_count
  let overall_avg := (boys_avg * boys_count + girls_avg * girls_count) / total_count
  let total_var := 
    (boys_count * (boys_var + (boys_avg - overall_avg)^2) / total_count)
    + (girls_count * (girls_var + (girls_avg - overall_avg)^2) / total_count)
  total_var = 46.8 := by
    sorry

end total_sample_variance_l2190_219042


namespace find_omega_increasing_intervals_l2190_219056

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (x : ℝ) : ℝ :=
  let ω := 3/2
  f ω (x - (Real.pi / 2))

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x : ℝ, f ω (x + 2*Real.pi / (2*ω)) = f ω x) :
  ω = 3/2 :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∃ a b, 
  a = (2/3 * k * Real.pi + Real.pi / 4) ∧ 
  b = (2/3 * k * Real.pi + 7 * Real.pi / 12) ∧
  ∀ x, a ≤ x ∧ x ≤ b → g x < g (x + 1) :=
  sorry

end find_omega_increasing_intervals_l2190_219056


namespace polynomial_identity_l2190_219014

theorem polynomial_identity (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end polynomial_identity_l2190_219014


namespace part_I_part_II_l2190_219096

noncomputable def a (n : Nat) : Nat := sorry

def is_odd (n : Nat) : Prop := n % 2 = 1

theorem part_I
  (h : a 1 = 19) :
  a 2014 = 98 := by
  sorry

theorem part_II
  (h1: ∀ n : Nat, is_odd (a n))
  (h2: ∀ n m : Nat, a n = a m) -- constant sequence
  (h3: ∀ n : Nat, a n > 1) :
  ∃ k : Nat, a k = 5 := by
  sorry


end part_I_part_II_l2190_219096


namespace group_population_l2190_219027

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l2190_219027


namespace quadratic_form_sum_const_l2190_219038

theorem quadratic_form_sum_const (a b c x : ℝ) (h : 4 * x^2 - 28 * x - 48 = a * (x + b)^2 + c) : 
  a + b + c = -96.5 :=
by
  sorry

end quadratic_form_sum_const_l2190_219038


namespace remainder_101_pow_37_mod_100_l2190_219023

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l2190_219023


namespace expected_groups_l2190_219037

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l2190_219037


namespace diego_apples_weight_l2190_219003

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l2190_219003


namespace surface_area_l2190_219069

theorem surface_area (r : ℝ) (π : ℝ) (V : ℝ) (S : ℝ) 
  (h1 : V = 48 * π) 
  (h2 : V = (4 / 3) * π * r^3) : 
  S = 4 * π * r^2 :=
  sorry

end surface_area_l2190_219069


namespace kickers_goals_in_first_period_l2190_219058

theorem kickers_goals_in_first_period (K : ℕ) 
  (h1 : ∀ n : ℕ, n = K) 
  (h2 : ∀ n : ℕ, n = 2 * K) 
  (h3 : ∀ n : ℕ, n = K / 2) 
  (h4 : ∀ n : ℕ, n = 4 * K) 
  (h5 : K + 2 * K + (K / 2) + 4 * K = 15) : 
  K = 2 := 
by
  sorry

end kickers_goals_in_first_period_l2190_219058


namespace find_p_l2190_219079

open Real

variable (A : ℝ × ℝ)
variable (p : ℝ) (hp : p > 0)

-- Conditions
def on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop := A.snd^2 = 2 * p * A.fst
def dist_focus (A : ℝ × ℝ) (p : ℝ) : Prop := sqrt ((A.fst - p / 2)^2 + A.snd^2) = 12
def dist_y_axis (A : ℝ × ℝ) : Prop := abs (A.fst) = 9

-- Theorem to prove
theorem find_p (h1 : on_parabola A p) (h2 : dist_focus A p) (h3 : dist_y_axis A) : p = 6 :=
sorry

end find_p_l2190_219079


namespace certain_number_l2190_219076

theorem certain_number (n q1 q2: ℕ) (h1 : 49 = n * q1 + 4) (h2 : 66 = n * q2 + 6): n = 15 :=
sorry

end certain_number_l2190_219076


namespace brian_stones_l2190_219091

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l2190_219091


namespace friends_payment_l2190_219017

theorem friends_payment
  (num_friends : ℕ) (num_bread : ℕ) (cost_bread : ℕ) 
  (num_hotteok : ℕ) (cost_hotteok : ℕ) (total_cost : ℕ)
  (cost_per_person : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_bread = 5)
  (h3 : cost_bread = 200)
  (h4 : num_hotteok = 7)
  (h5 : cost_hotteok = 800)
  (h6 : total_cost = num_bread * cost_bread + num_hotteok * cost_hotteok)
  (h7 : cost_per_person = total_cost / num_friends) :
  cost_per_person = 1650 := by
  sorry

end friends_payment_l2190_219017


namespace eval_expression_l2190_219032

theorem eval_expression : (503 * 503 - 502 * 504) = 1 :=
by
  sorry

end eval_expression_l2190_219032


namespace gcd_140_396_is_4_l2190_219048

def gcd_140_396 : ℕ := Nat.gcd 140 396

theorem gcd_140_396_is_4 : gcd_140_396 = 4 :=
by
  unfold gcd_140_396
  sorry

end gcd_140_396_is_4_l2190_219048


namespace income_is_10000_l2190_219033

-- Define the necessary variables: income, expenditure, and savings
variables (income expenditure : ℕ) (x : ℕ)

-- Define the conditions given in the problem
def ratio_condition : Prop := income = 10 * x ∧ expenditure = 7 * x
def savings_condition : Prop := income - expenditure = 3000

-- State the theorem that needs to be proved
theorem income_is_10000 (h_ratio : ratio_condition income expenditure x) (h_savings : savings_condition income expenditure) : income = 10000 :=
sorry

end income_is_10000_l2190_219033


namespace sum_of_three_numbers_l2190_219020

noncomputable def lcm_three_numbers (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_of_three_numbers 
  (a b c : ℕ)
  (x : ℕ)
  (h1 : lcm_three_numbers a b c = 180)
  (h2 : a = 2 * x)
  (h3 : b = 3 * x)
  (h4 : c = 5 * x) : a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l2190_219020
