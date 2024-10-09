import Mathlib

namespace moles_of_C2H6_l1180_118061

-- Define the reactive coefficients
def ratio_C := 2
def ratio_H2 := 3
def ratio_C2H6 := 1

-- Given conditions
def moles_C := 6
def moles_H2 := 9

-- Function to calculate moles of C2H6 formed
def moles_C2H6_formed (m_C : ℕ) (m_H2 : ℕ) : ℕ :=
  min (m_C * ratio_C2H6 / ratio_C) (m_H2 * ratio_C2H6 / ratio_H2)

-- Theorem statement: the number of moles of C2H6 formed is 3
theorem moles_of_C2H6 : moles_C2H6_formed moles_C moles_H2 = 3 :=
by {
  -- Sorry is used since we are not providing the proof here
  sorry
}

end moles_of_C2H6_l1180_118061


namespace solution_set_of_inequality_l1180_118072

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 3*x + 4 > 0 } = { x : ℝ | -1 < x ∧ x < 4 } := 
sorry

end solution_set_of_inequality_l1180_118072


namespace time_left_to_room_l1180_118060

theorem time_left_to_room (total_time minutes_to_gate minutes_to_building : ℕ) 
  (h1 : total_time = 30) 
  (h2 : minutes_to_gate = 15) 
  (h3 : minutes_to_building = 6) : 
  total_time - (minutes_to_gate + minutes_to_building) = 9 :=
by 
  sorry

end time_left_to_room_l1180_118060


namespace find_fraction_l1180_118018

theorem find_fraction (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4)
  (h3 : b = 1) : a / b = (17 + Real.sqrt 269) / 10 :=
by
  sorry

end find_fraction_l1180_118018


namespace camera_lens_distance_l1180_118022

theorem camera_lens_distance (f u : ℝ) (h_fu : f ≠ u) (h_f : f ≠ 0) (h_u : u ≠ 0) :
  (∃ v : ℝ, (1 / f) = (1 / u) + (1 / v) ∧ v = (f * u) / (u - f)) :=
by {
  sorry
}

end camera_lens_distance_l1180_118022


namespace distribute_ways_l1180_118071

/-- There are 5 distinguishable balls and 4 distinguishable boxes.
The total number of ways to distribute these balls into the boxes is 1024. -/
theorem distribute_ways : (4 : ℕ) ^ (5 : ℕ) = 1024 := by
  sorry

end distribute_ways_l1180_118071


namespace relationship_between_m_and_n_l1180_118066

variable {X_1 X_2 k m n : ℝ}

-- Given conditions
def inverse_proportional_points (X_1 X_2 k : ℝ) (m n : ℝ) : Prop :=
  m = k / X_1 ∧ n = k / X_2 ∧ k > 0 ∧ X_1 < X_2

theorem relationship_between_m_and_n (h : inverse_proportional_points X_1 X_2 k m n) : m > n :=
by
  -- Insert proof here, skipping with sorry
  sorry

end relationship_between_m_and_n_l1180_118066


namespace negation_of_proposition_l1180_118042

theorem negation_of_proposition :
  (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) → (∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) :=
sorry

end negation_of_proposition_l1180_118042


namespace find_f_2_l1180_118038

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem find_f_2 (a b : ℝ) (hf_neg2 : f a b (-2) = 7) : f a b 2 = -13 :=
by
  sorry

end find_f_2_l1180_118038


namespace max_profit_at_800_l1180_118065

open Nat

def P (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 80
  else if h : 100 < x ∧ x ≤ 1000 then 82 - 0.02 * x
  else 0

def f (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 30 * x
  else if h : 100 < x ∧ x ≤ 1000 then 32 * x - 0.02 * x^2
  else 0

theorem max_profit_at_800 :
  ∀ x : ℕ, f x ≤ 12800 ∧ f 800 = 12800 :=
sorry

end max_profit_at_800_l1180_118065


namespace correct_sqrt_evaluation_l1180_118045

theorem correct_sqrt_evaluation:
  2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 :=
by 
  sorry

end correct_sqrt_evaluation_l1180_118045


namespace missing_number_l1180_118059

theorem missing_number (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 :=
by
sorry

end missing_number_l1180_118059


namespace proposition_false_n4_l1180_118085

variable {P : ℕ → Prop}

theorem proposition_false_n4
  (h_ind : ∀ (k : ℕ), k ≠ 0 → P k → P (k + 1))
  (h_false_5 : P 5 = False) :
  P 4 = False :=
sorry

end proposition_false_n4_l1180_118085


namespace intersection_proof_l1180_118043

-- Definitions of sets M and N
def M : Set ℝ := { x | x^2 < 4 }
def N : Set ℝ := { x | x < 1 }

-- The intersection of M and N
def intersection : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Proposition to prove
theorem intersection_proof : M ∩ N = intersection :=
by sorry

end intersection_proof_l1180_118043


namespace age_is_50_l1180_118039

-- Definitions only based on the conditions provided
def future_age (A: ℕ) := A + 5
def past_age (A: ℕ) := A - 5

theorem age_is_50 (A : ℕ) (h : 5 * future_age A - 5 * past_age A = A) : A = 50 := 
by 
  sorry  -- proof should be provided here

end age_is_50_l1180_118039


namespace solution_set_inequality_l1180_118004

theorem solution_set_inequality :
  {x : ℝ | (x^2 - 4) * (x - 6)^2 ≤ 0} = {x : ℝ | (-2 ≤ x ∧ x ≤ 2) ∨ x = 6} :=
  sorry

end solution_set_inequality_l1180_118004


namespace number_four_units_away_from_neg_five_l1180_118032

theorem number_four_units_away_from_neg_five (x : ℝ) : 
    abs (x + 5) = 4 ↔ x = -9 ∨ x = -1 :=
by 
  sorry

end number_four_units_away_from_neg_five_l1180_118032


namespace binomial_evaluation_l1180_118091

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l1180_118091


namespace third_root_of_polynomial_l1180_118073

variable (a b x : ℝ)
noncomputable def polynomial := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

theorem third_root_of_polynomial (h1 : polynomial a b (-3) = 0) (h2 : polynomial a b 4 = 0) :
  ∃ r : ℝ, r = -17 / 10 ∧ polynomial a b r = 0 :=
by
  sorry

end third_root_of_polynomial_l1180_118073


namespace total_purchase_cost_l1180_118074

-- Definitions for the quantities of the items
def quantity_chocolate_bars : ℕ := 10
def quantity_gummy_bears : ℕ := 10
def quantity_chocolate_chips : ℕ := 20

-- Definitions for the costs of the items
def cost_per_chocolate_bar : ℕ := 3
def cost_per_gummy_bear_pack : ℕ := 2
def cost_per_chocolate_chip_bag : ℕ := 5

-- Proof statement to be shown
theorem total_purchase_cost :
  (quantity_chocolate_bars * cost_per_chocolate_bar) + 
  (quantity_gummy_bears * cost_per_gummy_bear_pack) + 
  (quantity_chocolate_chips * cost_per_chocolate_chip_bag) = 150 :=
sorry

end total_purchase_cost_l1180_118074


namespace max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l1180_118006

theorem max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h₁ : a ≤ b) (h₂ : b ≤ c) (h₃ : c ≤ 2 * a) :
    b / a + c / b + a / c ≤ 7 / 2 := 
  sorry

end max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l1180_118006


namespace number_of_seeds_in_bucket_B_l1180_118099

theorem number_of_seeds_in_bucket_B :
  ∃ (x : ℕ), 
    ∃ (y : ℕ), 
    ∃ (z : ℕ), 
      y = x + 10 ∧ 
      z = 30 ∧ 
      x + y + z = 100 ∧
      x = 30 :=
by {
  -- the proof is omitted.
  sorry
}

end number_of_seeds_in_bucket_B_l1180_118099


namespace sara_red_balloons_l1180_118027

theorem sara_red_balloons (initial_red : ℕ) (given_red : ℕ) 
  (h_initial : initial_red = 31) (h_given : given_red = 24) : 
  initial_red - given_red = 7 :=
by {
  sorry
}

end sara_red_balloons_l1180_118027


namespace flight_duration_problem_l1180_118020

def problem_conditions : Prop :=
  let la_departure_pst := (7, 15) -- 7:15 AM PST
  let ny_arrival_est := (17, 40) -- 5:40 PM EST (17:40 in 24-hour format)
  let time_difference := 3 -- Hours difference (EST is 3 hours ahead of PST)
  let dst_adjustment := 1 -- Daylight saving time adjustment in hours
  ∃ (h m : ℕ), (0 < m ∧ m < 60) ∧ ((h = 7 ∧ m = 25) ∧ (h + m = 32))

theorem flight_duration_problem :
  problem_conditions :=
by
  -- Placeholder for the proof that shows the conditions established above imply h + m = 32
  sorry

end flight_duration_problem_l1180_118020


namespace intersection_point_of_lines_l1180_118056

theorem intersection_point_of_lines (x y : ℝ) :
  (2 * x - 3 * y = 3) ∧ (4 * x + 2 * y = 2) ↔ (x = 3/4) ∧ (y = -1/2) :=
by
  sorry

end intersection_point_of_lines_l1180_118056


namespace base_conversion_subtraction_l1180_118030

namespace BaseConversion

def base9_to_base10 (n : ℕ) : ℕ :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def base6_to_base10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 6 * 6^0

theorem base_conversion_subtraction : (base9_to_base10 324) - (base6_to_base10 156) = 193 := by
  sorry

end BaseConversion

end base_conversion_subtraction_l1180_118030


namespace correct_exponentiation_l1180_118031

theorem correct_exponentiation (a : ℝ) :
  (a^2 * a^3 = a^5) ∧
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 + a^2 ≠ a^4) ∧
  (3 * a^3 - a^2 ≠ 2 * a) :=
by
  sorry

end correct_exponentiation_l1180_118031


namespace part1_part2_l1180_118002

section

variable (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_seq : ∀ n, a_seq (n + 1) = (5 * a_seq n - 8) / (a_seq n - 1))
variable (h_initial : a_seq 1 = a)

-- Part 1:
theorem part1 (h_a : a = 3) : 
  ∃ r : ℝ, ∀ n, (a_seq n - 2) / (a_seq n - 4) = r ^ n ∧ a_seq n = (4 * 3 ^ (n - 1) + 2) / (3 ^ (n - 1) + 1) := 
sorry

-- Part 2:
theorem part2 (h_pos : ∀ n, a_seq n > 3) : 3 < a := 
sorry

end

end part1_part2_l1180_118002


namespace abs_inequality_k_ge_neg3_l1180_118026

theorem abs_inequality_k_ge_neg3 (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 :=
sorry

end abs_inequality_k_ge_neg3_l1180_118026


namespace fewest_people_to_join_CBL_l1180_118089

theorem fewest_people_to_join_CBL (initial_people teamsize : ℕ) (even_teams : ℕ → Prop)
  (initial_people_eq : initial_people = 38)
  (teamsize_eq : teamsize = 9)
  (even_teams_def : ∀ n, even_teams n ↔ n % 2 = 0) :
  ∃(p : ℕ), (initial_people + p) % teamsize = 0 ∧ even_teams ((initial_people + p) / teamsize) ∧ p = 16 := by
  sorry

end fewest_people_to_join_CBL_l1180_118089


namespace exists_x_geq_zero_l1180_118069

theorem exists_x_geq_zero (h : ∀ x : ℝ, x^2 + x - 1 < 0) : ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
sorry

end exists_x_geq_zero_l1180_118069


namespace value_of_p_l1180_118041

theorem value_of_p (m n p : ℝ) (h1 : m = 6 * n + 5) (h2 : m + 2 = 6 * (n + p) + 5) : p = 1 / 3 :=
by
  sorry

end value_of_p_l1180_118041


namespace grapes_difference_l1180_118097

theorem grapes_difference (R A_i A_l : ℕ) 
  (hR : R = 25) 
  (hAi : A_i = R + 2) 
  (hTotal : R + A_i + A_l = 83) : 
  A_l - A_i = 4 := 
by
  sorry

end grapes_difference_l1180_118097


namespace right_triangle_conditions_l1180_118025

-- Definitions
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

-- Conditions
def cond1 (A B C : ℝ) : Prop := A + B = C
def cond2 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def cond3 (A B C : ℝ) : Prop := A = B ∧ B = 2 * C
def cond4 (A B C : ℝ) : Prop := A = 2 * B ∧ B = 3 * C

-- Problem statement
theorem right_triangle_conditions (A B C : ℝ) :
  (cond1 A B C → is_right_triangle A B C) ∧
  (cond2 A B C → is_right_triangle A B C) ∧
  ¬(cond3 A B C → is_right_triangle A B C) ∧
  ¬(cond4 A B C → is_right_triangle A B C) :=
by
  sorry

end right_triangle_conditions_l1180_118025


namespace vectors_coplanar_l1180_118011

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (1, -3, -7)
def vector_c : ℝ × ℝ × ℝ := (1, 2, 3)

def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product vector_a vector_b vector_c = 0 := 
by
  sorry

end vectors_coplanar_l1180_118011


namespace diophantine_3x_5y_diophantine_3x_5y_indefinite_l1180_118015

theorem diophantine_3x_5y (n : ℕ) (h_n_pos : n > 0) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n) ↔ 
    (∃ k : ℕ, (n = 3 * k ∧ n ≥ 15) ∨ 
              (n = 3 * k + 1 ∧ n ≥ 13) ∨ 
              (n = 3 * k + 2 ∧ n ≥ 11) ∨ 
              (n = 8)) :=
sorry

theorem diophantine_3x_5y_indefinite (n m : ℕ) (h_n_large : n > 40 * m):
  ∃ (N : ℕ), ∀ k ≤ N, ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n + k :=
sorry

end diophantine_3x_5y_diophantine_3x_5y_indefinite_l1180_118015


namespace total_spent_l1180_118098

def cost_sandwich : ℕ := 2
def cost_hamburger : ℕ := 2
def cost_hotdog : ℕ := 1
def cost_fruit_juice : ℕ := 2

def selene_sandwiches : ℕ := 3
def selene_fruit_juice : ℕ := 1
def tanya_hamburgers : ℕ := 2
def tanya_fruit_juice : ℕ := 2

def total_selene_spent : ℕ := (selene_sandwiches * cost_sandwich) + (selene_fruit_juice * cost_fruit_juice)
def total_tanya_spent : ℕ := (tanya_hamburgers * cost_hamburger) + (tanya_fruit_juice * cost_fruit_juice)

theorem total_spent : total_selene_spent + total_tanya_spent = 16 := by
  sorry

end total_spent_l1180_118098


namespace employee_pays_204_l1180_118009

-- Definitions based on conditions
def wholesale_cost : ℝ := 200
def markup_percent : ℝ := 0.20
def discount_percent : ℝ := 0.15

def retail_price := wholesale_cost * (1 + markup_percent)
def employee_payment := retail_price * (1 - discount_percent)

-- Theorem with the expected result
theorem employee_pays_204 : employee_payment = 204 := by
  -- Proof not required, we add sorry to avoid the proof details
  sorry

end employee_pays_204_l1180_118009


namespace exactly_one_even_needs_assumption_l1180_118095

open Nat

theorem exactly_one_even_needs_assumption 
  {a b c : ℕ} 
  (h : (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ (a % 2 = 0 → b % 2 = 1) ∧ (a % 2 = 0 → c % 2 = 1) ∧ (b % 2 = 0 → c % 2 = 1)) :
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) → (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) → (¬(a % 2 = 0 ∧ b % 2 = 0) ∧ ¬(b % 2 = 0 ∧ c % 2 = 0) ∧ ¬(a % 2 = 0 ∧ c % 2 = 0)) := 
by
  sorry

end exactly_one_even_needs_assumption_l1180_118095


namespace rohan_house_rent_percentage_l1180_118078

noncomputable def house_rent_percentage (food_percentage entertainment_percentage conveyance_percentage salary savings: ℝ) : ℝ :=
  100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

-- Conditions
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def salary : ℝ := 10000
def savings : ℝ := 2000

-- Theorem
theorem rohan_house_rent_percentage :
  house_rent_percentage food_percentage entertainment_percentage conveyance_percentage salary savings = 20 := 
sorry

end rohan_house_rent_percentage_l1180_118078


namespace arccos_of_one_over_sqrt_two_l1180_118005

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l1180_118005


namespace sin_alpha_plus_beta_eq_33_by_65_l1180_118000

theorem sin_alpha_plus_beta_eq_33_by_65 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hcosα : Real.cos α = 12 / 13) 
  (hcos_2α_β : Real.cos (2 * α + β) = 3 / 5) :
  Real.sin (α + β) = 33 / 65 := 
by 
  sorry

end sin_alpha_plus_beta_eq_33_by_65_l1180_118000


namespace selection_methods_count_l1180_118067

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem selection_methods_count :
  let females := 8
  let males := 4
  (binomial females 2 * binomial males 1) + (binomial females 1 * binomial males 2) = 112 :=
by
  sorry

end selection_methods_count_l1180_118067


namespace miles_flown_on_thursday_l1180_118049
-- Importing the necessary library

-- Defining the problem conditions and the proof goal
theorem miles_flown_on_thursday (x : ℕ) : 
  (∀ y, (3 * (1134 + y) = 7827) → y = x) → x = 1475 :=
by
  intro h
  specialize h 1475
  sorry

end miles_flown_on_thursday_l1180_118049


namespace cost_per_minute_of_each_call_l1180_118090

theorem cost_per_minute_of_each_call :
  let calls_per_week := 50
  let hours_per_call := 1
  let weeks_per_month := 4
  let total_hours_in_month := calls_per_week * hours_per_call * weeks_per_month
  let total_cost := 600
  let cost_per_hour := total_cost / total_hours_in_month
  let minutes_per_hour := 60
  let cost_per_minute := cost_per_hour / minutes_per_hour
  cost_per_minute = 0.05 := 
by
  sorry

end cost_per_minute_of_each_call_l1180_118090


namespace div_pow_eq_l1180_118062

theorem div_pow_eq {a : ℝ} (h : a ≠ 0) : a^3 / a^2 = a :=
sorry

end div_pow_eq_l1180_118062


namespace expression_evaluation_l1180_118013

theorem expression_evaluation : 2^2 - Real.tan (Real.pi / 3) + abs (Real.sqrt 3 - 1) - (3 - Real.pi)^0 = 2 :=
by
  sorry

end expression_evaluation_l1180_118013


namespace strictly_increasing_interval_l1180_118088

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb (1/3) (x^2 - 4 * x + 3)

theorem strictly_increasing_interval : ∀ x y : ℝ, x < 1 → y < 1 → x < y → f x < f y :=
by
  sorry

end strictly_increasing_interval_l1180_118088


namespace no_pairs_xy_perfect_square_l1180_118008

theorem no_pairs_xy_perfect_square :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ ∃ k : ℕ, (xy + 1) * (xy + x + 2) = k^2 := 
by {
  sorry
}

end no_pairs_xy_perfect_square_l1180_118008


namespace speed_of_B_l1180_118019

theorem speed_of_B 
    (initial_distance : ℕ)
    (speed_of_A : ℕ)
    (time : ℕ)
    (distance_covered_by_A : ℕ)
    (distance_covered_by_B : ℕ)
    : initial_distance = 24 → speed_of_A = 5 → time = 2 → distance_covered_by_A = speed_of_A * time → distance_covered_by_B = initial_distance - distance_covered_by_A → distance_covered_by_B / time = 7 :=
by
  sorry

end speed_of_B_l1180_118019


namespace line_equation_k_value_l1180_118024

theorem line_equation_k_value (m n k : ℝ) 
    (h1 : m = 2 * n + 5) 
    (h2 : m + 5 = 2 * (n + k) + 5) : 
    k = 2.5 :=
by sorry

end line_equation_k_value_l1180_118024


namespace cynthia_more_miles_l1180_118047

open Real

noncomputable def david_speed : ℝ := 55 / 5
noncomputable def cynthia_speed : ℝ := david_speed + 3

theorem cynthia_more_miles (t : ℝ) (ht : t = 5) :
  (cynthia_speed * t) - (david_speed * t) = 15 :=
by
  sorry

end cynthia_more_miles_l1180_118047


namespace kelly_sony_games_solution_l1180_118081

def kelly_sony_games_left (n g : Nat) : Nat :=
  n - g

theorem kelly_sony_games_solution (initial : Nat) (given_away : Nat) 
  (h_initial : initial = 132)
  (h_given_away : given_away = 101) :
  kelly_sony_games_left initial given_away = 31 :=
by
  rw [h_initial, h_given_away]
  unfold kelly_sony_games_left
  norm_num

end kelly_sony_games_solution_l1180_118081


namespace integer_count_of_sqrt_x_l1180_118051

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l1180_118051


namespace amount_after_two_years_l1180_118083

noncomputable def annual_increase (initial_amount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ years

theorem amount_after_two_years :
  annual_increase 32000 (1/8) 2 = 40500 :=
by
  sorry

end amount_after_two_years_l1180_118083


namespace total_potatoes_brought_home_l1180_118077

def number_of_potatoes_each : ℕ := 8

theorem total_potatoes_brought_home (jane_potatoes mom_potatoes dad_potatoes : ℕ) :
  jane_potatoes = number_of_potatoes_each →
  mom_potatoes = number_of_potatoes_each →
  dad_potatoes = number_of_potatoes_each →
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end total_potatoes_brought_home_l1180_118077


namespace log_product_max_l1180_118096

open Real

theorem log_product_max (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : log x + log y = 4) : log x * log y ≤ 4 := 
by
  sorry

end log_product_max_l1180_118096


namespace complement_U_A_l1180_118086

open Set

def U : Set ℤ := univ
def A : Set ℤ := { x | x^2 - x - 2 ≥ 0 }

theorem complement_U_A :
  (U \ A) = { 0, 1 } := by
  sorry

end complement_U_A_l1180_118086


namespace solve_eq1_solve_eq2_solve_eq3_l1180_118028

def equation1 (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0
def solution1 (x : ℝ) : Prop := x = 5 ∨ x = 1

theorem solve_eq1 : ∀ x : ℝ, equation1 x ↔ solution1 x := sorry

def equation2 (x : ℝ) : Prop := 3 * x * (2 * x - 1) = 4 * x - 2
def solution2 (x : ℝ) : Prop := x = 1/2 ∨ x = 2/3

theorem solve_eq2 : ∀ x : ℝ, equation2 x ↔ solution2 x := sorry

def equation3 (x : ℝ) : Prop := x^2 - 2 * Real.sqrt 2 * x - 2 = 0
def solution3 (x : ℝ) : Prop := x = Real.sqrt 2 + 2 ∨ x = Real.sqrt 2 - 2

theorem solve_eq3 : ∀ x : ℝ, equation3 x ↔ solution3 x := sorry

end solve_eq1_solve_eq2_solve_eq3_l1180_118028


namespace book_organizing_activity_l1180_118048

theorem book_organizing_activity (x : ℕ) (h₁ : x > 0):
  (80 : ℝ) / (x + 5 : ℝ) = (70 : ℝ) / (x : ℝ) :=
sorry

end book_organizing_activity_l1180_118048


namespace time_taken_by_x_alone_l1180_118092

theorem time_taken_by_x_alone 
  (W : ℝ)
  (Rx Ry Rz : ℝ)
  (h1 : Ry = W / 24)
  (h2 : Ry + Rz = W / 6)
  (h3 : Rx + Rz = W / 4) :
  (W / Rx) = 16 :=
by
  sorry

end time_taken_by_x_alone_l1180_118092


namespace arithmetic_sequence_sum_l1180_118054

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + 3)
  (h_a1_a2 : a 1 + a 2 = 7)
  (h_a3 : a 3 = 8)
  (h_bn : ∀ n, b n = 1 / (a n * a (n+1)))
  :
  (∀ n, a n = 3 * n - 1) ∧ (T n = n / (2 * (3 * n + 2))) :=
by 
  sorry

end arithmetic_sequence_sum_l1180_118054


namespace rectangle_perimeter_from_square_l1180_118023

theorem rectangle_perimeter_from_square (d : ℝ)
  (h : d = 6) :
  ∃ (p : ℝ), p = 12 :=
by
  sorry

end rectangle_perimeter_from_square_l1180_118023


namespace ratio_is_one_third_l1180_118055

-- Definitions based on given conditions
def total_students : ℕ := 90
def initial_cafeteria_students : ℕ := (2 * total_students) / 3
def initial_outside_students : ℕ := total_students - initial_cafeteria_students
def moved_cafeteria_to_outside : ℕ := 3
def final_cafeteria_students : ℕ := 67
def students_ran_inside : ℕ := final_cafeteria_students - (initial_cafeteria_students - moved_cafeteria_to_outside)

-- Ratio calculation as a proof statement
def ratio_ran_inside_to_outside : ℚ := students_ran_inside / initial_outside_students

-- Proof that the ratio is 1/3
theorem ratio_is_one_third : ratio_ran_inside_to_outside = 1 / 3 :=
by sorry -- Proof omitted

end ratio_is_one_third_l1180_118055


namespace arithmetic_sequence_general_term_l1180_118033

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ)
  (h1 : ∀ n m, a (n+1) - a n = a (m+1) - a m)
  (h2 : (a 2 + a 6) / 2 = 5)
  (h3 : (a 3 + a 7) / 2 = 7) :
  ∀ n, a n = 2 * n - 3 :=
by 
  sorry

end arithmetic_sequence_general_term_l1180_118033


namespace fraction_of_top10_lists_l1180_118001

theorem fraction_of_top10_lists (total_members : ℕ) (min_lists : ℝ) (H1 : total_members = 795) (H2 : min_lists = 198.75) :
  (min_lists / total_members) = 1 / 4 :=
by
  -- The proof is omitted as requested
  sorry

end fraction_of_top10_lists_l1180_118001


namespace correct_diagram_is_B_l1180_118079

-- Define the diagrams and their respected angles
def sector_angle_A : ℝ := 90
def sector_angle_B : ℝ := 135
def sector_angle_C : ℝ := 180

-- Define the target central angle for one third of the circle
def target_angle : ℝ := 120

-- The proof statement that Diagram B is the correct diagram with the sector angle closest to one third of the circle (120 degrees)
theorem correct_diagram_is_B (A B C : Prop) :
  (B = (sector_angle_A < target_angle ∧ target_angle < sector_angle_B)) := 
sorry

end correct_diagram_is_B_l1180_118079


namespace no_nat_solutions_l1180_118029

theorem no_nat_solutions (x y z : ℕ) : x^2 + y^2 + z^2 ≠ 2 * x * y * z :=
sorry

end no_nat_solutions_l1180_118029


namespace cos_180_eq_neg1_l1180_118046

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l1180_118046


namespace max_jogs_l1180_118082

theorem max_jogs (x y z : ℕ) (h1 : 3 * x + 2 * y + 8 * z = 60) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  z ≤ 6 := 
sorry

end max_jogs_l1180_118082


namespace ellipse_major_axis_length_l1180_118044

theorem ellipse_major_axis_length : 
  ∀ (x y : ℝ), x^2 + 2 * y^2 = 2 → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_major_axis_length_l1180_118044


namespace volume_of_tetrahedron_PQRS_l1180_118076

-- Definitions of the given conditions for the tetrahedron
def PQ := 6
def PR := 4
def PS := 5
def QR := 5
def QS := 6
def RS := 15 / 2  -- RS is (15 / 2), i.e., 7.5
def area_PQR := 12

noncomputable def volume_tetrahedron (PQ PR PS QR QS RS area_PQR : ℝ) : ℝ := 1 / 3 * area_PQR * 4

theorem volume_of_tetrahedron_PQRS :
  volume_tetrahedron PQ PR PS QR QS RS area_PQR = 16 :=
by sorry

end volume_of_tetrahedron_PQRS_l1180_118076


namespace number_of_terms_geometric_seq_l1180_118080

-- Given conditions
variables (a1 q : ℝ)  -- First term and common ratio of the sequence
variable  (n : ℕ)     -- Number of terms in the sequence

-- The product of the first three terms
axiom condition1 : a1^3 * q^3 = 3

-- The product of the last three terms
axiom condition2 : a1^3 * q^(3 * n - 6) = 9

-- The product of all terms
axiom condition3 : a1^n * q^(n * (n - 1) / 2) = 729

-- Proving the number of terms in the sequence
theorem number_of_terms_geometric_seq : n = 12 := by
  sorry

end number_of_terms_geometric_seq_l1180_118080


namespace two_color_K6_contains_monochromatic_triangle_l1180_118016

theorem two_color_K6_contains_monochromatic_triangle (V : Type) [Fintype V] [DecidableEq V]
  (hV : Fintype.card V = 6)
  (color : V → V → Fin 2) :
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (color a b = color b c ∧ color b c = color c a) := by
  sorry

end two_color_K6_contains_monochromatic_triangle_l1180_118016


namespace bisection_second_iteration_value_l1180_118063

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_second_iteration_value :
  f 0.25 = -0.234375 :=
by
  -- The proof steps would go here
  sorry

end bisection_second_iteration_value_l1180_118063


namespace siblings_count_l1180_118007

noncomputable def Masud_siblings (M : ℕ) : Prop :=
  (4 * M - 60 = (3 * M) / 4 + 135) → M = 60

theorem siblings_count (M : ℕ) : Masud_siblings M :=
  by
  sorry

end siblings_count_l1180_118007


namespace apples_eq_pears_l1180_118050

-- Define the conditions
def apples_eq_oranges (a o : ℕ) : Prop := 4 * a = 6 * o
def oranges_eq_pears (o p : ℕ) : Prop := 5 * o = 3 * p

-- The main problem statement
theorem apples_eq_pears (a o p : ℕ) (h1 : apples_eq_oranges a o) (h2 : oranges_eq_pears o p) :
  24 * a = 21 * p :=
sorry

end apples_eq_pears_l1180_118050


namespace positive_real_as_sum_l1180_118084

theorem positive_real_as_sum (k : ℝ) (hk : k > 0) : 
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∑' n, 1 / 10 ^ a n = k) :=
sorry

end positive_real_as_sum_l1180_118084


namespace prairie_total_area_l1180_118087

theorem prairie_total_area (acres_dust_storm : ℕ) (acres_untouched : ℕ) (h₁ : acres_dust_storm = 64535) (h₂ : acres_untouched = 522) : acres_dust_storm + acres_untouched = 65057 :=
by
  sorry

end prairie_total_area_l1180_118087


namespace Mason_tables_needed_l1180_118052

theorem Mason_tables_needed
  (w_silverware_piece : ℕ := 4) 
  (n_silverware_piece_per_setting : ℕ := 3) 
  (w_plate : ℕ := 12) 
  (n_plates_per_setting : ℕ := 2) 
  (n_settings_per_table : ℕ := 8) 
  (n_backup_settings : ℕ := 20) 
  (total_weight : ℕ := 5040) : 
  ∃ (n_tables : ℕ), n_tables = 15 :=
by
  sorry

end Mason_tables_needed_l1180_118052


namespace rectangle_inscribed_area_l1180_118070

variables (b h x : ℝ) 

theorem rectangle_inscribed_area (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hx_lt_h : x < h) :
  ∃ A, A = (b * x * (h - x)) / h :=
sorry

end rectangle_inscribed_area_l1180_118070


namespace proportion_of_line_segments_l1180_118003

theorem proportion_of_line_segments (a b c d : ℕ)
  (h_proportion : a * d = b * c)
  (h_a : a = 2)
  (h_b : b = 4)
  (h_c : c = 3) :
  d = 6 :=
by
  sorry

end proportion_of_line_segments_l1180_118003


namespace larger_of_two_numbers_l1180_118064

theorem larger_of_two_numbers (A B : ℕ) (hcf : A.gcd B = 47) (lcm_factors : A.lcm B = 47 * 49 * 11 * 13 * 4913) : max A B = 123800939 :=
sorry

end larger_of_two_numbers_l1180_118064


namespace combined_weight_of_student_and_sister_l1180_118014

theorem combined_weight_of_student_and_sister
  (S : ℝ) (R : ℝ)
  (h1 : S = 90)
  (h2 : S - 6 = 2 * R) :
  S + R = 132 :=
by
  sorry

end combined_weight_of_student_and_sister_l1180_118014


namespace total_children_l1180_118040

variable (S C B T : ℕ)

theorem total_children (h1 : T < 19) 
                       (h2 : S = 3 * C) 
                       (h3 : B = S / 2) 
                       (h4 : T = B + S + 1) : 
                       T = 10 := 
  sorry

end total_children_l1180_118040


namespace janessa_kept_20_cards_l1180_118037

-- Definitions based on conditions
def initial_cards : Nat := 4
def father_cards : Nat := 13
def ebay_cards : Nat := 36
def bad_shape_cards : Nat := 4
def cards_given_to_dexter : Nat := 29

-- Prove that Janessa kept 20 cards for herself
theorem janessa_kept_20_cards :
  (initial_cards + father_cards  + ebay_cards - bad_shape_cards) - cards_given_to_dexter = 20 :=
by
  sorry

end janessa_kept_20_cards_l1180_118037


namespace sum_of_squares_is_42_l1180_118034

variables (D T H : ℕ)

theorem sum_of_squares_is_42
  (h1 : 3 * D + T = 2 * H)
  (h2 : 2 * H^3 = 3 * D^3 + T^3)
  (coprime : Nat.gcd (Nat.gcd D T) H = 1) :
  (T^2 + D^2 + H^2 = 42) :=
sorry

end sum_of_squares_is_42_l1180_118034


namespace laps_needed_l1180_118010

theorem laps_needed (r1 r2 : ℕ) (laps1 : ℕ) (h1 : r1 = 30) (h2 : r2 = 10) (h3 : laps1 = 40) : 
  (r1 * laps1) / r2 = 120 := by
  sorry

end laps_needed_l1180_118010


namespace number_of_boys_l1180_118093

-- We define the conditions provided in the problem
def child_1_has_3_brothers : Prop := ∃ B G : ℕ, B - 1 = 3 ∧ G = 6
def child_2_has_4_brothers : Prop := ∃ B G : ℕ, B - 1 = 4 ∧ G = 5

theorem number_of_boys (B G : ℕ) (h1 : child_1_has_3_brothers) (h2 : child_2_has_4_brothers) : B = 4 :=
by
  sorry

end number_of_boys_l1180_118093


namespace determine_d_l1180_118057

theorem determine_d (m n d : ℝ) (p : ℝ) (hp : p = 0.6666666666666666) 
  (h1 : m = 3 * n + 5) (h2 : m + d = 3 * (n + p) + 5) : d = 2 :=
by {
  sorry
}

end determine_d_l1180_118057


namespace total_points_scored_l1180_118068

theorem total_points_scored (points_per_round : ℕ) (rounds : ℕ) (h1 : points_per_round = 42) (h2 : rounds = 2) : 
  points_per_round * rounds = 84 :=
by
  sorry

end total_points_scored_l1180_118068


namespace integer_values_of_a_l1180_118058

variable (a b c x : ℤ)

theorem integer_values_of_a (h : (x - a) * (x - 12) + 4 = (x + b) * (x + c)) : a = 7 ∨ a = 17 := by
  sorry

end integer_values_of_a_l1180_118058


namespace abs_div_nonzero_l1180_118012

theorem abs_div_nonzero (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  ¬ (|a| / a + |b| / b = 1) :=
by
  sorry

end abs_div_nonzero_l1180_118012


namespace rectangle_breadth_l1180_118017

theorem rectangle_breadth (length radius side breadth: ℝ)
  (h1: length = (2/5) * radius)
  (h2: radius = side)
  (h3: side ^ 2 = 1600)
  (h4: length * breadth = 160) :
  breadth = 10 := 
by
  sorry

end rectangle_breadth_l1180_118017


namespace average_growth_rate_le_max_growth_rate_l1180_118053

variable (P : ℝ) (a : ℝ) (b : ℝ) (x : ℝ)

theorem average_growth_rate_le_max_growth_rate (h : (1 + x)^2 = (1 + a) * (1 + b)) :
  x ≤ max a b := 
sorry

end average_growth_rate_le_max_growth_rate_l1180_118053


namespace find_side_a_in_triangle_l1180_118021

noncomputable def triangle_side_a (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  cosA = 4/5 ∧ b = 2 ∧ S = 3 → a = Real.sqrt 13

-- Theorem statement with explicit conditions and proof goal
theorem find_side_a_in_triangle
  (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) :
  cosA = 4 / 5 → b = 2 → S = 3 → a = Real.sqrt 13 :=
by 
  intros 
  sorry

end find_side_a_in_triangle_l1180_118021


namespace find_m_value_l1180_118035

theorem find_m_value (m : Real) (h : (3 * m + 8) * (m - 3) = 72) : m = (1 + Real.sqrt 1153) / 6 :=
by
  sorry

end find_m_value_l1180_118035


namespace exists_similarity_point_l1180_118036

variable {Point : Type} [MetricSpace Point]

noncomputable def similar_triangles (A B A' B' : Point) (O : Point) : Prop :=
  dist A O / dist A' O = dist A B / dist A' B' ∧ dist B O / dist B' O = dist A B / dist A' B'

theorem exists_similarity_point (A B A' B' : Point) (h1 : dist A B ≠ 0) (h2: dist A' B' ≠ 0) :
  ∃ O : Point, similar_triangles A B A' B' O :=
  sorry

end exists_similarity_point_l1180_118036


namespace john_total_jury_duty_days_l1180_118094

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l1180_118094


namespace sin_range_l1180_118075

theorem sin_range (p : Prop) (q : Prop) :
  (¬ ∃ x : ℝ, Real.sin x = 3/2) → (∀ x : ℝ, x^2 - 4 * x + 5 > 0) → (¬p ∧ q) :=
by
  sorry

end sin_range_l1180_118075
