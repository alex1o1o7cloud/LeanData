import Mathlib

namespace fraction_power_zero_l418_41821

variable (a b : ℤ)
variable (h_a : a ≠ 0) (h_b : b ≠ 0)

theorem fraction_power_zero : (a / b)^0 = 1 := by
  sorry

end fraction_power_zero_l418_41821


namespace sum_of_reciprocals_is_two_l418_41833

variable (x y : ℝ)
variable (h1 : x + y = 50)
variable (h2 : x * y = 25)

theorem sum_of_reciprocals_is_two (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1/x + 1/y) = 2 :=
by
  sorry

end sum_of_reciprocals_is_two_l418_41833


namespace john_total_fuel_usage_l418_41828

def city_fuel_rate := 6 -- liters per km for city traffic
def highway_fuel_rate := 4 -- liters per km for highway traffic

def trip1_city_distance := 50 -- km for Trip 1
def trip2_highway_distance := 35 -- km for Trip 2
def trip3_city_distance := 15 -- km for Trip 3 in city traffic
def trip3_highway_distance := 10 -- km for Trip 3 on highway

-- Define the total fuel consumption
def total_fuel_used : Nat :=
  (trip1_city_distance * city_fuel_rate) +
  (trip2_highway_distance * highway_fuel_rate) +
  (trip3_city_distance * city_fuel_rate) +
  (trip3_highway_distance * highway_fuel_rate)

theorem john_total_fuel_usage :
  total_fuel_used = 570 :=
by
  sorry

end john_total_fuel_usage_l418_41828


namespace sqrt_x_minus_2_domain_l418_41859

theorem sqrt_x_minus_2_domain {x : ℝ} : (∃y : ℝ, y = Real.sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_domain_l418_41859


namespace q_alone_time_24_days_l418_41886

theorem q_alone_time_24_days:
  ∃ (Wq : ℝ), (∀ (Wp Ws : ℝ), 
    Wp = Wq + 1 / 60 → 
    Wp + Wq = 1 / 10 → 
    Wp + 1 / 60 + 2 * Wq = 1 / 6 → 
    1 / Wq = 24) :=
by
  sorry

end q_alone_time_24_days_l418_41886


namespace range_of_y_l418_41827

theorem range_of_y (x : ℝ) : 
  - (Real.sqrt 3) / 3 ≤ (Real.sin x) / (2 - Real.cos x) ∧ (Real.sin x) / (2 - Real.cos x) ≤ (Real.sqrt 3) / 3 :=
sorry

end range_of_y_l418_41827


namespace sequence_is_odd_l418_41848

theorem sequence_is_odd (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 7) 
  (h3 : ∀ n ≥ 2, -1/2 < (a (n + 1)) - (a n) * (a n) / a (n-1) ∧
                (a (n + 1)) - (a n) * (a n) / a (n-1) ≤ 1/2) :
  ∀ n > 1, (a n) % 2 = 1 :=
by
  sorry

end sequence_is_odd_l418_41848


namespace total_flying_days_l418_41893

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l418_41893


namespace probability_even_distinct_digits_l418_41861

theorem probability_even_distinct_digits :
  let count_even_distinct := 1960
  let total_numbers := 8000
  count_even_distinct / total_numbers = 49 / 200 :=
by
  sorry

end probability_even_distinct_digits_l418_41861


namespace local_food_drive_correct_l418_41842

def local_food_drive_condition1 (R J x : ℕ) : Prop :=
  J = 2 * R + x

def local_food_drive_condition2 (J : ℕ) : Prop :=
  4 * J = 100

def local_food_drive_condition3 (R J : ℕ) : Prop :=
  R + J = 35

theorem local_food_drive_correct (R J x : ℕ)
  (h1 : local_food_drive_condition1 R J x)
  (h2 : local_food_drive_condition2 J)
  (h3 : local_food_drive_condition3 R J) :
  x = 5 :=
by
  sorry

end local_food_drive_correct_l418_41842


namespace minimum_value_of_expression_l418_41872

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  2 * a + b + c ≥ 4 := 
by 
  sorry

end minimum_value_of_expression_l418_41872


namespace swap_numbers_l418_41835

theorem swap_numbers (a b : ℕ) (hc: b = 17) (ha : a = 8) : 
  ∃ c, c = b ∧ b = a ∧ a = c := 
by
  sorry

end swap_numbers_l418_41835


namespace restaurant_total_cost_l418_41890

theorem restaurant_total_cost (burger_cost pizza_cost : ℕ)
    (h1 : burger_cost = 9)
    (h2 : pizza_cost = 2 * burger_cost) :
    pizza_cost + 3 * burger_cost = 45 := 
by
  sorry

end restaurant_total_cost_l418_41890


namespace retail_price_l418_41805

/-- A retailer bought a machine at a wholesale price of $99 and later sold it after a 10% discount of the retail price.
If the retailer made a profit equivalent to 20% of the wholesale price, then the retail price of the machine before the discount was $132. -/
theorem retail_price (wholesale_price : ℝ) (profit_percent discount_percent : ℝ) (P : ℝ) 
  (h₁ : wholesale_price = 99) 
  (h₂ : profit_percent = 0.20) 
  (h₃ : discount_percent = 0.10)
  (h₄ : (1 - discount_percent) * P = wholesale_price + profit_percent * wholesale_price) : 
  P = 132 := 
by
  sorry

end retail_price_l418_41805


namespace percentage_tax_proof_l418_41841

theorem percentage_tax_proof (total_worth tax_free cost taxable tax_rate tax_value percentage_sales_tax : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_free = 34.7)
  (h3 : tax_rate = 0.06)
  (h4 : total_worth = taxable + tax_rate * taxable + tax_free)
  (h5 : tax_value = tax_rate * taxable)
  (h6 : percentage_sales_tax = (tax_value / total_worth) * 100) :
  percentage_sales_tax = 0.75 :=
by
  sorry

end percentage_tax_proof_l418_41841


namespace average_income_QR_l418_41804

theorem average_income_QR (P Q R : ℝ) 
  (h1: (P + Q) / 2 = 5050) 
  (h2: (P + R) / 2 = 5200) 
  (hP: P = 4000) : 
  (Q + R) / 2 = 6250 := 
by 
  -- additional steps and proof to be provided here
  sorry

end average_income_QR_l418_41804


namespace remainder_21_pow_2051_mod_29_l418_41863

theorem remainder_21_pow_2051_mod_29 :
  ∀ (a : ℤ), (21^4 ≡ 1 [MOD 29]) -> (2051 = 4 * 512 + 3) -> (21^3 ≡ 15 [MOD 29]) -> (21^2051 ≡ 15 [MOD 29]) :=
by
  intros a h1 h2 h3
  sorry

end remainder_21_pow_2051_mod_29_l418_41863


namespace sarah_gave_away_16_apples_to_teachers_l418_41815

def initial_apples : Nat := 25
def apples_given_to_friends : Nat := 5
def apples_eaten : Nat := 1
def apples_left_after_journey : Nat := 3

theorem sarah_gave_away_16_apples_to_teachers :
  let apples_after_giving_to_friends := initial_apples - apples_given_to_friends
  let apples_after_eating := apples_after_giving_to_friends - apples_eaten
  apples_after_eating - apples_left_after_journey = 16 :=
by
  sorry

end sarah_gave_away_16_apples_to_teachers_l418_41815


namespace largest_among_given_numbers_l418_41832

theorem largest_among_given_numbers : 
    let a := 24680 + (1 / 1357)
    let b := 24680 - (1 / 1357)
    let c := 24680 * (1 / 1357)
    let d := 24680 / (1 / 1357)
    let e := 24680.1357
    d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_among_given_numbers_l418_41832


namespace cubic_roots_l418_41892

open Real

theorem cubic_roots (x1 x2 x3 : ℝ) (h1 : x1 * x2 = 1)
  (h2 : 3 * x1^3 + 2 * sqrt 3 * x1^2 - 21 * x1 + 6 * sqrt 3 = 0)
  (h3 : 3 * x2^3 + 2 * sqrt 3 * x2^2 - 21 * x2 + 6 * sqrt 3 = 0)
  (h4 : 3 * x3^3 + 2 * sqrt 3 * x3^2 - 21 * x3 + 6 * sqrt 3 = 0) :
  (x1 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x1 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) := 
sorry

end cubic_roots_l418_41892


namespace geometric_sequence_a5_l418_41830

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) 
  : a 5 = -8 :=
sorry

end geometric_sequence_a5_l418_41830


namespace sequence_term_four_l418_41875

theorem sequence_term_four (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 4 = 7 :=
sorry

end sequence_term_four_l418_41875


namespace solution_set_circle_l418_41816

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l418_41816


namespace grapes_average_seeds_l418_41814

def total_seeds_needed : ℕ := 60
def apple_seed_average : ℕ := 6
def pear_seed_average : ℕ := 2
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def extra_seeds_needed : ℕ := 3

-- Calculation of total seeds from apples and pears:
def seeds_from_apples : ℕ := apples_count * apple_seed_average
def seeds_from_pears : ℕ := pears_count * pear_seed_average

def total_seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculation of the remaining seeds needed from grapes:
def seeds_needed_from_grapes : ℕ := total_seeds_needed - total_seeds_from_apples_and_pears - extra_seeds_needed

-- Calculation of the average number of seeds per grape:
def grape_seed_average : ℕ := seeds_needed_from_grapes / grapes_count

-- Prove the correct average number of seeds per grape:
theorem grapes_average_seeds : grape_seed_average = 3 :=
by
  sorry

end grapes_average_seeds_l418_41814


namespace cherry_pies_count_correct_l418_41874

def total_pies : ℕ := 36

def ratio_ap_bb_ch : (ℕ × ℕ × ℕ) := (2, 3, 4)

def total_ratio_parts : ℕ := 2 + 3 + 4

def pies_per_part (total_pies : ℕ) (total_ratio_parts : ℕ) : ℕ := total_pies / total_ratio_parts

def num_parts_ch : ℕ := 4

def num_cherry_pies (total_pies : ℕ) (total_ratio_parts : ℕ) (num_parts_ch : ℕ) : ℕ :=
  pies_per_part total_pies total_ratio_parts * num_parts_ch

theorem cherry_pies_count_correct : num_cherry_pies total_pies total_ratio_parts num_parts_ch = 16 := by
  sorry

end cherry_pies_count_correct_l418_41874


namespace second_day_more_than_third_day_l418_41807

-- Define the conditions
def total_people (d1 d2 d3 : ℕ) := d1 + d2 + d3 = 246 
def first_day := 79
def third_day := 120

-- Define the statement to prove
theorem second_day_more_than_third_day : 
  ∃ d2 : ℕ, total_people first_day d2 third_day ∧ (d2 - third_day) = 47 :=
by
  sorry

end second_day_more_than_third_day_l418_41807


namespace parabola_circle_intercept_l418_41883

theorem parabola_circle_intercept (p : ℝ) (h_pos : p > 0) :
  (∃ (x y : ℝ), y^2 = 2 * p * x ∧ x^2 + y^2 + 2 * x - 3 = 0) ∧
  (∃ (y1 y2 : ℝ), (y1 - y2)^2 + (-(p / 2) + 1)^2 = 4^2) → p = 2 :=
by sorry

end parabola_circle_intercept_l418_41883


namespace smallest_sum_is_381_l418_41854

def is_valid_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits_once (n m : ℕ) : Prop :=
  (∀ d ∈ [1, 2, 3, 4, 5, 6], (d ∈ n.digits 10 ∨ d ∈ m.digits 10)) ∧
  (∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ m.digits 10 → d ∈ [1, 2, 3, 4, 5, 6])

theorem smallest_sum_is_381 :
  ∃ (n m : ℕ), is_valid_3_digit_number n ∧ is_valid_3_digit_number m ∧
  uses_digits_once n m ∧ n + m = 381 :=
sorry

end smallest_sum_is_381_l418_41854


namespace correct_statement_l418_41876

theorem correct_statement (a b : ℝ) (ha : a < b) (hb : b < 0) : |a| / |b| > 1 :=
sorry

end correct_statement_l418_41876


namespace factorize_expression_l418_41879

variables {a x y : ℝ}

theorem factorize_expression (a x y : ℝ) : 3 * a * x ^ 2 + 6 * a * x * y + 3 * a * y ^ 2 = 3 * a * (x + y) ^ 2 :=
by
  sorry

end factorize_expression_l418_41879


namespace cost_of_50_lavenders_l418_41831

noncomputable def cost_of_bouquet (lavenders : ℕ) : ℚ :=
  (25 / 15) * lavenders

theorem cost_of_50_lavenders :
  cost_of_bouquet 50 = 250 / 3 :=
sorry

end cost_of_50_lavenders_l418_41831


namespace vectorBC_computation_l418_41819

open Vector

def vectorAB : ℝ × ℝ := (2, 4)

def vectorAC : ℝ × ℝ := (1, 3)

theorem vectorBC_computation :
  (vectorAC.1 - vectorAB.1, vectorAC.2 - vectorAB.2) = (-1, -1) :=
sorry

end vectorBC_computation_l418_41819


namespace set_of_values_l418_41817

theorem set_of_values (a : ℝ) (h : 2 ∉ {x : ℝ | x - a < 0}) : a ≤ 2 := 
sorry

end set_of_values_l418_41817


namespace option_b_correct_l418_41846

variable (Line Plane : Type)

-- Definitions for perpendicularity and parallelism
variable (perp parallel : Line → Plane → Prop) (parallel_line : Line → Line → Prop)

-- Assumptions reflecting the conditions in the problem
axiom perp_alpha_1 {a : Line} {alpha : Plane} : perp a alpha
axiom perp_alpha_2 {b : Line} {alpha : Plane} : perp b alpha

-- The statement to prove
theorem option_b_correct (a b : Line) (alpha : Plane) :
  perp a alpha → perp b alpha → parallel_line a b :=
by
  intro h1 h2
  -- proof omitted
  sorry

end option_b_correct_l418_41846


namespace fraction_to_decimal_l418_41826

theorem fraction_to_decimal :
  (7 / 125 : ℚ) = 0.056 :=
sorry

end fraction_to_decimal_l418_41826


namespace bullet_speed_difference_l418_41840

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l418_41840


namespace valid_seating_arrangements_l418_41824

theorem valid_seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let restricted_arrangements := Nat.factorial 7 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 3507840 :=
by
  sorry

end valid_seating_arrangements_l418_41824


namespace other_books_new_releases_percentage_l418_41811

theorem other_books_new_releases_percentage
  (T : ℝ)
  (h1 : 0 < T)
  (hf_books : ℝ := 0.4 * T)
  (hf_new_releases : ℝ := 0.4 * hf_books)
  (other_books : ℝ := 0.6 * T)
  (total_new_releases : ℝ := hf_new_releases + (P * other_books))
  (fraction_hf_new : ℝ := hf_new_releases / total_new_releases)
  (fraction_value : fraction_hf_new = 0.27586206896551724)
  : P = 0.7 :=
sorry

end other_books_new_releases_percentage_l418_41811


namespace solution_comparison_l418_41803

variables (a a' b b' : ℝ)

theorem solution_comparison (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-(b / a) < -(b' / a')) ↔ (b' / a' < b / a) :=
by sorry

end solution_comparison_l418_41803


namespace six_digit_numbers_with_at_least_two_zeros_l418_41834

noncomputable def num_six_digit_numbers_with_at_least_two_zeros : ℕ :=
  73314

theorem six_digit_numbers_with_at_least_two_zeros :
  ∃ n : ℕ, n = num_six_digit_numbers_with_at_least_two_zeros := by
  use 73314
  sorry

end six_digit_numbers_with_at_least_two_zeros_l418_41834


namespace valid_conditions_x_y_z_l418_41855

theorem valid_conditions_x_y_z (x y z : ℤ) :
  x = y - 1 ∧ z = y + 1 ∨ x = y ∧ z = y + 1 ↔ x * (x - y) + y * (y - x) + z * (z - y) = 1 :=
sorry

end valid_conditions_x_y_z_l418_41855


namespace sum_is_eight_l418_41825

theorem sum_is_eight (a b c d : ℤ)
  (h1 : 2 * (a - b + c) = 10)
  (h2 : 2 * (b - c + d) = 12)
  (h3 : 2 * (c - d + a) = 6)
  (h4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 :=
by
  sorry

end sum_is_eight_l418_41825


namespace range_of_a_l418_41838

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 :=
by
  intro h
  sorry

end range_of_a_l418_41838


namespace final_price_percentage_of_original_l418_41844

theorem final_price_percentage_of_original (original_price sale_price final_price : ℝ)
  (h1 : sale_price = original_price * 0.5)
  (h2 : final_price = sale_price * 0.9) :
  final_price = original_price * 0.45 :=
by
  sorry

end final_price_percentage_of_original_l418_41844


namespace rational_solution_for_quadratic_l418_41888

theorem rational_solution_for_quadratic (k : ℕ) (h_pos : 0 < k) : 
  ∃ m : ℕ, (18^2 - 4 * k * (2 * k)) = m^2 ↔ k = 4 :=
by
  sorry

end rational_solution_for_quadratic_l418_41888


namespace cannot_take_value_l418_41818

theorem cannot_take_value (x y : ℝ) (h : |x| + |y| = 13) : 
  ∀ (v : ℝ), x^2 + 7*x - 3*y + y^2 = v → (0 ≤ v ∧ v ≤ 260) := 
by
  sorry

end cannot_take_value_l418_41818


namespace total_profit_proof_l418_41895
-- Import the necessary libraries

-- Define the investments and profits
def investment_tom : ℕ := 3000 * 12
def investment_jose : ℕ := 4500 * 10
def profit_jose : ℕ := 3500

-- Define the ratio and profit parts
def ratio_tom : ℕ := investment_tom / Nat.gcd investment_tom investment_jose
def ratio_jose : ℕ := investment_jose / Nat.gcd investment_tom investment_jose
def ratio_total : ℕ := ratio_tom + ratio_jose
def one_part_value : ℕ := profit_jose / ratio_jose
def profit_tom : ℕ := ratio_tom * one_part_value

-- The total profit
def total_profit : ℕ := profit_tom + profit_jose

-- The theorem to prove
theorem total_profit_proof : total_profit = 6300 := by
  sorry

end total_profit_proof_l418_41895


namespace equation_one_solution_equation_two_solution_l418_41867

theorem equation_one_solution (x : ℝ) : 4 * (x - 1)^2 - 9 = 0 ↔ (x = 5 / 2) ∨ (x = - 1 / 2) := 
by sorry

theorem equation_two_solution (x : ℝ) : x^2 - 6 * x - 7 = 0 ↔ (x = 7) ∨ (x = - 1) :=
by sorry

end equation_one_solution_equation_two_solution_l418_41867


namespace sum_of_largest_and_smallest_l418_41860

theorem sum_of_largest_and_smallest (n : ℕ) (h : 6 * n + 15 = 105) : (n + (n + 5) = 35) :=
by
  sorry

end sum_of_largest_and_smallest_l418_41860


namespace cost_split_difference_l418_41880

-- Definitions of amounts paid
def SarahPaid : ℕ := 150
def DerekPaid : ℕ := 210
def RitaPaid : ℕ := 240

-- Total paid by all three
def TotalPaid : ℕ := SarahPaid + DerekPaid + RitaPaid

-- Each should have paid:
def EachShouldHavePaid : ℕ := TotalPaid / 3

-- Amount Sarah owes Rita
def SarahOwesRita : ℕ := EachShouldHavePaid - SarahPaid

-- Amount Derek should receive back from Rita
def DerekShouldReceiveFromRita : ℕ := DerekPaid - EachShouldHavePaid

-- Difference between the amounts Sarah and Derek owe/should receive from Rita
theorem cost_split_difference : SarahOwesRita - DerekShouldReceiveFromRita = 60 := by
    sorry

end cost_split_difference_l418_41880


namespace find_k_and_a_range_l418_41889

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^2 + Real.exp x - k * Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem find_k_and_a_range (k a : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) :
  k = -1 ∧ 2 ≤ a := by
    sorry

end find_k_and_a_range_l418_41889


namespace axis_of_symmetry_l418_41856

variable (f : ℝ → ℝ)

theorem axis_of_symmetry (h : ∀ x, f x = f (5 - x)) :  ∀ x y, y = f x ↔ (x = 2.5 ∧ y = f 2.5) := 
sorry

end axis_of_symmetry_l418_41856


namespace solve_for_x_l418_41847

theorem solve_for_x (x : ℚ) (h : 5 * (x - 4) = 3 * (3 - 3 * x) + 6) : x = 5 / 2 :=
by {
  sorry
}

end solve_for_x_l418_41847


namespace point_on_right_branch_l418_41843

noncomputable def on_hyperbola_right_branch (a b m : ℝ) :=
  (∀ a b m : ℝ, (a - 2 * b > 0) → (a + 2 * b > 0) → (a ^ 2 - 4 * b ^ 2 = m) → (m ≠ 0) → a > 0)

theorem point_on_right_branch (a b m : ℝ) (h₁ : a - 2 * b > 0) (h₂ : a + 2 * b > 0) (h₃ : a ^ 2 - 4 * b ^ 2 = m) (h₄ : m ≠ 0) :
  a > 0 := 
by 
  sorry

end point_on_right_branch_l418_41843


namespace recess_breaks_l418_41870

theorem recess_breaks (total_outside_time : ℕ) (lunch_break : ℕ) (extra_recess : ℕ) (recess_duration : ℕ) 
  (h1 : total_outside_time = 80)
  (h2 : lunch_break = 30)
  (h3 : extra_recess = 20)
  (h4 : recess_duration = 15) : 
  (total_outside_time - (lunch_break + extra_recess)) / recess_duration = 2 := 
by {
  -- proof starts here
  sorry
}

end recess_breaks_l418_41870


namespace unit_digit_product_7858_1086_4582_9783_l418_41882

-- Define the unit digits of the given numbers
def unit_digit_7858 : ℕ := 8
def unit_digit_1086 : ℕ := 6
def unit_digit_4582 : ℕ := 2
def unit_digit_9783 : ℕ := 3

-- Define a function to calculate the unit digit of a product of two numbers based on their unit digits
def unit_digit_product (a b : ℕ) : ℕ :=
  (a * b) % 10

-- The theorem that states the unit digit of the product of the numbers is 4
theorem unit_digit_product_7858_1086_4582_9783 :
  unit_digit_product (unit_digit_product (unit_digit_product unit_digit_7858 unit_digit_1086) unit_digit_4582) unit_digit_9783 = 4 :=
  by
  sorry

end unit_digit_product_7858_1086_4582_9783_l418_41882


namespace triangle_area_l418_41869

theorem triangle_area (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 54 := by
  -- conditions provided
  sorry

end triangle_area_l418_41869


namespace numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l418_41891

theorem numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1 :
  (63 ∣ 2^48 - 1) ∧ (65 ∣ 2^48 - 1) := 
by
  sorry

end numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l418_41891


namespace find_possible_values_l418_41820

theorem find_possible_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 36 / 11 ∨ y = 468 / 23) :=
by
  sorry

end find_possible_values_l418_41820


namespace set_A_enumeration_l418_41857

-- Define the conditions of the problem.
def A : Set ℕ := { x | ∃ (n : ℕ), 6 = n * (6 - x) }

-- State the theorem to be proved.
theorem set_A_enumeration : A = {0, 2, 3, 4, 5} :=
by
  sorry

end set_A_enumeration_l418_41857


namespace necessary_and_sufficient_condition_holds_l418_41845

noncomputable def necessary_and_sufficient_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + m > 0

theorem necessary_and_sufficient_condition_holds (m : ℝ) :
  necessary_and_sufficient_condition m ↔ m > 1 :=
by
  sorry

end necessary_and_sufficient_condition_holds_l418_41845


namespace f_periodic_odd_condition_l418_41884

theorem f_periodic_odd_condition (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 4) = f x) (h_one : f 1 = 5) : f 2015 = -5 :=
by
  sorry

end f_periodic_odd_condition_l418_41884


namespace num_solutions_eq_three_l418_41866

theorem num_solutions_eq_three :
  (∃ n : Nat, (x : ℝ) → (x^2 - 4) * (x^2 - 1) = (x^2 + 3 * x + 2) * (x^2 - 8 * x + 7) → n = 3) :=
sorry

end num_solutions_eq_three_l418_41866


namespace radius_ratio_of_spheres_l418_41812

theorem radius_ratio_of_spheres
  (V_large : ℝ) (V_small : ℝ) (r_large r_small : ℝ)
  (h1 : V_large = 324 * π)
  (h2 : V_small = 0.25 * V_large)
  (h3 : (4/3) * π * r_large^3 = V_large)
  (h4 : (4/3) * π * r_small^3 = V_small) :
  (r_small / r_large) = (1/2) := 
sorry

end radius_ratio_of_spheres_l418_41812


namespace palm_meadows_total_beds_l418_41899

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l418_41899


namespace hyperbola_asymptote_l418_41873

def hyperbola_eqn (m x y : ℝ) := m * x^2 - y^2 = 1

def vertex_distance_condition (m : ℝ) := 2 * Real.sqrt (1 / m) = 4

theorem hyperbola_asymptote (m : ℝ) (h_eq : hyperbola_eqn m x y) (h_dist : vertex_distance_condition m) :
  ∃ k, y = k * x ∧ k = 1 / 2 ∨ k = -1 / 2 := by
  sorry

end hyperbola_asymptote_l418_41873


namespace solve_equation_l418_41836

theorem solve_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 :=
sorry

end solve_equation_l418_41836


namespace inequality_proof_l418_41858

variable {a b c d : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end inequality_proof_l418_41858


namespace solution_set_of_inequality_l418_41871

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) * (x + 3) > 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l418_41871


namespace minimum_planks_required_l418_41806

theorem minimum_planks_required (colors : Finset ℕ) (planks : List ℕ) :
  colors.card = 100 ∧
  ∀ i j, i ∈ colors → j ∈ colors → i ≠ j →
  ∃ k₁ k₂, k₁ < k₂ ∧ planks.get? k₁ = some i ∧ planks.get? k₂ = some j
  → planks.length = 199 := 
sorry

end minimum_planks_required_l418_41806


namespace depth_of_second_hole_l418_41850

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let total_man_hours1 := workers1 * hours1
  let rate_of_work := depth1 / total_man_hours1
  let workers2 := 45 + 45
  let hours2 := 6
  let total_man_hours2 := workers2 * hours2
  let depth2 := rate_of_work * total_man_hours2
  depth2 = 45 := by
    sorry

end depth_of_second_hole_l418_41850


namespace average_branches_per_foot_correct_l418_41801

def height_tree_1 : ℕ := 50
def branches_tree_1 : ℕ := 200
def height_tree_2 : ℕ := 40
def branches_tree_2 : ℕ := 180
def height_tree_3 : ℕ := 60
def branches_tree_3 : ℕ := 180
def height_tree_4 : ℕ := 34
def branches_tree_4 : ℕ := 153

def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4
def total_branches := branches_tree_1 + branches_tree_2 + branches_tree_3 + branches_tree_4
def average_branches_per_foot := total_branches / total_height

theorem average_branches_per_foot_correct : average_branches_per_foot = 713 / 184 := 
  by
    -- Proof omitted, directly state the result
    sorry

end average_branches_per_foot_correct_l418_41801


namespace player_current_average_l418_41852

theorem player_current_average (A : ℝ) 
  (h1 : 10 * A + 76 = (A + 4) * 11) : 
  A = 32 :=
sorry

end player_current_average_l418_41852


namespace smallest_possible_r_l418_41877

theorem smallest_possible_r (p q r : ℤ) (hpq: p < q) (hqr: q < r) 
  (hgeo: q^2 = p * r) (harith: 2 * q = p + r) : r = 4 :=
sorry

end smallest_possible_r_l418_41877


namespace arctan_tan_expr_is_75_degrees_l418_41808

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l418_41808


namespace thief_distance_l418_41853

variable (d : ℝ := 250)   -- initial distance in meters
variable (v_thief : ℝ := 12 * 1000 / 3600)  -- thief's speed in m/s (converted from km/hr)
variable (v_policeman : ℝ := 15 * 1000 / 3600)  -- policeman's speed in m/s (converted from km/hr)

noncomputable def distance_thief_runs : ℝ :=
  v_thief * (d / (v_policeman - v_thief))

theorem thief_distance :
  distance_thief_runs d v_thief v_policeman = 990.47 := sorry

end thief_distance_l418_41853


namespace division_theorem_l418_41823

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l418_41823


namespace dan_initial_money_l418_41849

def initial_amount (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ) : ℕ :=
  spent_candy + spent_chocolate + remaining

theorem dan_initial_money 
  (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ)
  (h_candy : spent_candy = 2)
  (h_chocolate : spent_chocolate = 3)
  (h_remaining : remaining = 2) :
  initial_amount spent_candy spent_chocolate remaining = 7 :=
by
  rw [h_candy, h_chocolate, h_remaining]
  unfold initial_amount
  rfl

end dan_initial_money_l418_41849


namespace ones_digit_expression_l418_41810

theorem ones_digit_expression :
  ((73 ^ 1253 * 44 ^ 987 + 47 ^ 123 / 39 ^ 654 * 86 ^ 1484 - 32 ^ 1987) % 10) = 2 := by
  sorry

end ones_digit_expression_l418_41810


namespace product_increase_by_13_l418_41862

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l418_41862


namespace total_time_for_process_l418_41881

-- Given conditions
def cat_resistance_time : ℕ := 20
def walking_distance : ℕ := 64
def walking_rate : ℕ := 8

-- Prove the total time
theorem total_time_for_process : cat_resistance_time + (walking_distance / walking_rate) = 28 := by
  sorry

end total_time_for_process_l418_41881


namespace simple_interest_difference_l418_41865

/-- The simple interest on a certain amount at a 4% rate for 5 years amounted to a certain amount less than the principal. The principal was Rs 2400. Prove that the difference between the principal and the simple interest is Rs 1920. 
-/
theorem simple_interest_difference :
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  P - SI = 1920 :=
by
  /- We introduce the let definitions for the conditions and then state the theorem
    with the conclusion that needs to be proved. -/
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  /- The final step where we would conclude our theorem. -/
  sorry

end simple_interest_difference_l418_41865


namespace tower_remainder_l418_41800

def num_towers : ℕ := 907200  -- the total number of different towers S for 9 cubes

theorem tower_remainder : num_towers % 1000 = 200 :=
by
  sorry

end tower_remainder_l418_41800


namespace dice_sum_probability_l418_41868

theorem dice_sum_probability :
  let total_outcomes := 36
  let sum_le_8_outcomes := 13
  (sum_le_8_outcomes : ℕ) / (total_outcomes : ℕ) = (13 / 18 : ℝ) :=
by
  sorry

end dice_sum_probability_l418_41868


namespace chord_length_3pi_4_chord_bisected_by_P0_l418_41822

open Real

-- Define conditions and the problem.
def Circle := {p : ℝ × ℝ // p.1^2 + p.2^2 = 8}
def P0 : ℝ × ℝ := (-1, 2)

-- Proving the first part (1)
theorem chord_length_3pi_4 (α : ℝ) (hα : α = 3 * π / 4) (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  dist A B = sqrt 30 := sorry

-- Proving the second part (2)
theorem chord_bisected_by_P0 (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  ∃ k : ℝ, (B.2 - A.2) = k * (B.1 - A.1) ∧ k = 1 / 2 ∧
  (k * (x - (-1))) = y - 2 := sorry

end chord_length_3pi_4_chord_bisected_by_P0_l418_41822


namespace remaining_to_original_ratio_l418_41839

-- Define the number of rows and production per row for corn and potatoes.
def rows_of_corn : ℕ := 10
def corn_per_row : ℕ := 9
def rows_of_potatoes : ℕ := 5
def potatoes_per_row : ℕ := 30

-- Define the remaining crops after pest destruction.
def remaining_crops : ℕ := 120

-- Calculate the original number of crops from corn and potato productions.
def original_crops : ℕ :=
  (rows_of_corn * corn_per_row) + (rows_of_potatoes * potatoes_per_row)

-- Define the ratio of remaining crops to original crops.
def crops_ratio : ℚ := remaining_crops / original_crops

theorem remaining_to_original_ratio : crops_ratio = 1 / 2 := 
by
  sorry

end remaining_to_original_ratio_l418_41839


namespace set_complement_intersection_l418_41878

open Set

variable (U M N : Set ℕ)

theorem set_complement_intersection :
  U = {1, 2, 3, 4, 5, 6, 7} →
  M = {3, 4, 5} →
  N = {1, 3, 6} →
  {2, 7} = (U \ M) ∩ (U \ N) :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end set_complement_intersection_l418_41878


namespace arithmetic_sequence_general_term_l418_41894

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n - 1) = 2) : ∀ n, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l418_41894


namespace roots_quadratic_inequality_l418_41809

theorem roots_quadratic_inequality (t x1 x2 : ℝ) (h_eqn : x1 ^ 2 - t * x1 + t = 0) 
  (h_eqn2 : x2 ^ 2 - t * x2 + t = 0) (h_real : x1 + x2 = t) (h_prod : x1 * x2 = t) :
  x1 ^ 2 + x2 ^ 2 ≥ 2 * (x1 + x2) := 
sorry

end roots_quadratic_inequality_l418_41809


namespace solve_for_x_l418_41813

theorem solve_for_x (x : ℝ) (h : 3 * x - 8 = 4 * x + 5) : x = -13 :=
by 
  sorry

end solve_for_x_l418_41813


namespace find_uv_l418_41864

open Real

def vec1 : ℝ × ℝ := (3, -2)
def vec2 : ℝ × ℝ := (-1, 2)
def vec3 : ℝ × ℝ := (1, -1)
def vec4 : ℝ × ℝ := (4, -7)
def vec5 : ℝ × ℝ := (-3, 5)

theorem find_uv (u v : ℝ) :
  vec1 + ⟨4 * u, -7 * u⟩ = vec2 + ⟨-3 * v, 5 * v⟩ + vec3 →
  u = 3 / 4 ∧ v = -9 / 4 :=
by
  sorry

end find_uv_l418_41864


namespace tv_cost_solution_l418_41851

theorem tv_cost_solution (M T : ℝ) 
  (h1 : 2 * M + T = 7000)
  (h2 : M + 2 * T = 9800) : 
  T = 4200 :=
by
  sorry

end tv_cost_solution_l418_41851


namespace possible_values_n_l418_41887

theorem possible_values_n (n : ℕ) (h_pos : 0 < n) (h1 : n > 9 / 4) (h2 : n < 14) :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ k ∈ S, k = n :=
by
  -- proof to be filled in
  sorry

end possible_values_n_l418_41887


namespace closest_to_10_l418_41885

theorem closest_to_10
  (A B C D : ℝ)
  (hA : A = 9.998)
  (hB : B = 10.1)
  (hC : C = 10.09)
  (hD : D = 10.001) :
  abs (10 - D) < abs (10 - A) ∧ abs (10 - D) < abs (10 - B) ∧ abs (10 - D) < abs (10 - C) :=
by
  sorry

end closest_to_10_l418_41885


namespace gcd_factorials_l418_41896

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := by
  sorry

end gcd_factorials_l418_41896


namespace ratio_of_a_b_l418_41837

-- Define the system of equations as given in the problem
variables (x y a b : ℝ)

-- Conditions: the system of equations and b ≠ 0
def system_of_equations (a b : ℝ) (x y : ℝ) := 
  4 * x - 3 * y = a ∧ 6 * y - 8 * x = b

-- The theorem we aim to prove
theorem ratio_of_a_b (h : system_of_equations a b x y) (h₀ : b ≠ 0) : a / b = -1 / 2 :=
sorry

end ratio_of_a_b_l418_41837


namespace min_sum_ab_l418_41898

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l418_41898


namespace smart_charging_piles_eq_l418_41802

theorem smart_charging_piles_eq (x : ℝ) :
  301 * (1 + x) ^ 2 = 500 :=
by sorry

end smart_charging_piles_eq_l418_41802


namespace min_birthday_employees_wednesday_l418_41829

theorem min_birthday_employees_wednesday :
  ∀ (employees : ℕ) (n : ℕ), 
  employees = 50 → 
  n ≥ 1 →
  ∃ (x : ℕ), 6 * x + (x + n) = employees ∧ x + n ≥ 8 :=
by
  sorry

end min_birthday_employees_wednesday_l418_41829


namespace min_sum_chessboard_labels_l418_41897

theorem min_sum_chessboard_labels :
  ∃ (r : Fin 9 → Fin 9), 
  (∀ (i j : Fin 9), i ≠ j → r i ≠ r j) ∧ 
  ((Finset.univ : Finset (Fin 9)).sum (λ i => 1 / (r i + i.val + 1)) = 1) :=
by
  sorry

end min_sum_chessboard_labels_l418_41897
