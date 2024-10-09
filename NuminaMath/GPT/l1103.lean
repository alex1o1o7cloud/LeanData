import Mathlib

namespace rectangular_solid_surface_area_l1103_110337

theorem rectangular_solid_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 14) 
  (h2 : a^2 + b^2 + c^2 = 121) : 
  2 * (a * b + b * c + a * c) = 75 := 
by
  sorry

end rectangular_solid_surface_area_l1103_110337


namespace domain_of_function_l1103_110384

theorem domain_of_function :
  {x : ℝ | (x^2 - 9*x + 20 ≥ 0) ∧ (|x - 5| + |x + 2| ≠ 0)} = {x : ℝ | x ≤ 4 ∨ x ≥ 5} :=
by
  sorry

end domain_of_function_l1103_110384


namespace purple_candy_minimum_cost_l1103_110371

theorem purple_candy_minimum_cost (r g b n : ℕ) (h : 10 * r = 15 * g) (h1 : 15 * g = 18 * b) (h2 : 18 * b = 24 * n) : 
  ∃ k, k = n ∧ k ≥ 1 ∧ ∀ m, (24 * m = 360) → (m ≥ k) :=
by
  sorry

end purple_candy_minimum_cost_l1103_110371


namespace game_score_correct_answers_l1103_110358

theorem game_score_correct_answers :
  ∃ x : ℕ, (∃ y : ℕ, x + y = 30 ∧ 7 * x - 12 * y = 77) ∧ x = 23 :=
by
  use 23
  sorry

end game_score_correct_answers_l1103_110358


namespace power_calc_l1103_110342

noncomputable def n := 2 ^ 0.3
noncomputable def b := 13.333333333333332

theorem power_calc : n ^ b = 16 := by
  sorry

end power_calc_l1103_110342


namespace nested_sqrt_expr_l1103_110379

theorem nested_sqrt_expr (M : ℝ) (h : M > 1) : (↑(M) ^ (1 / 4) ^ (1 / 4) ^ (1 / 4)) = M ^ (21 / 64) :=
by
  sorry

end nested_sqrt_expr_l1103_110379


namespace jake_earnings_per_hour_l1103_110338

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end jake_earnings_per_hour_l1103_110338


namespace convert_4512_base8_to_base10_l1103_110332

-- Definitions based on conditions
def base8_to_base10 (n : Nat) : Nat :=
  let d3 := 4 * 8^3
  let d2 := 5 * 8^2
  let d1 := 1 * 8^1
  let d0 := 2 * 8^0
  d3 + d2 + d1 + d0

-- The proof statement
theorem convert_4512_base8_to_base10 :
  base8_to_base10 4512 = 2378 :=
by
  -- proof goes here
  sorry

end convert_4512_base8_to_base10_l1103_110332


namespace sqrt_product_is_four_l1103_110315

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end sqrt_product_is_four_l1103_110315


namespace total_items_to_buy_l1103_110391

theorem total_items_to_buy (total_money : ℝ) (cost_sandwich : ℝ) (cost_drink : ℝ) (num_items : ℕ) :
  total_money = 30 → cost_sandwich = 4.5 → cost_drink = 1 → num_items = 9 :=
by
  sorry

end total_items_to_buy_l1103_110391


namespace first_pair_weight_l1103_110350

variable (total_weight : ℕ) (second_pair_weight : ℕ) (third_pair_weight : ℕ)

theorem first_pair_weight (h : total_weight = 32) (h_second : second_pair_weight = 5) (h_third : third_pair_weight = 8) : 
    total_weight - 2 * (second_pair_weight + third_pair_weight) = 6 :=
by
  sorry

end first_pair_weight_l1103_110350


namespace expression_value_l1103_110318

theorem expression_value : 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end expression_value_l1103_110318


namespace min_value_expression_l1103_110378

variable (p q r : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)

theorem min_value_expression :
  (9 * r / (3 * p + 2 * q) + 9 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ≥ 2 :=
sorry

end min_value_expression_l1103_110378


namespace min_value_fraction_l1103_110393

theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2: b > 0) (h3 : a + b = 1) : 
  ∃ c : ℝ, c = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → x + 2 * y ≥ c) :=
by
  sorry

end min_value_fraction_l1103_110393


namespace irreducible_fraction_l1103_110376

theorem irreducible_fraction (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := 
sorry

end irreducible_fraction_l1103_110376


namespace find_x_l1103_110336

-- Define the conditions
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def molecular_weight : ℝ := 152

-- State the theorem
theorem find_x : ∃ x : ℕ, molecular_weight = atomic_weight_C + atomic_weight_Cl * x ∧ x = 4 := by
  sorry

end find_x_l1103_110336


namespace part1_solution_set_part2_range_m_l1103_110353
open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := - abs (x + 4) + m

-- Part I: Solution set for f(x) > x + 1 is (-∞, 0)
theorem part1_solution_set : { x : ℝ | f x > x + 1 } = { x : ℝ | x < 0 } :=
sorry

-- Part II: Range of m when the graphs of y = f(x) and y = g(x) have common points
theorem part2_range_m (m : ℝ) : (∃ x : ℝ, f x = g x m) → m ≥ 5 :=
sorry

end part1_solution_set_part2_range_m_l1103_110353


namespace magic_card_profit_l1103_110363

theorem magic_card_profit (purchase_price : ℝ) (multiplier : ℝ) (selling_price : ℝ) (profit : ℝ) 
                          (h1 : purchase_price = 100) 
                          (h2 : multiplier = 3) 
                          (h3 : selling_price = purchase_price * multiplier) 
                          (h4 : profit = selling_price - purchase_price) : 
                          profit = 200 :=
by 
  -- Here, you can introduce intermediate steps if needed.
  sorry

end magic_card_profit_l1103_110363


namespace choose_three_of_nine_l1103_110334

def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_three_of_nine : combination 9 3 = 84 :=
by 
  sorry

end choose_three_of_nine_l1103_110334


namespace boys_of_other_communities_l1103_110386

theorem boys_of_other_communities (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℝ) 
  (h_tm : total_boys = 1500)
  (h_pm : percentage_muslims = 37.5)
  (h_ph : percentage_hindus = 25.6)
  (h_ps : percentage_sikhs = 8.4) : 
  ∃ (boys_other_communities : ℕ), boys_other_communities = 428 :=
by
  sorry

end boys_of_other_communities_l1103_110386


namespace rocco_total_usd_l1103_110333

def us_quarters := 4 * 8 * 0.25
def canadian_dimes := 6 * 12 * 0.10 * 0.8
def us_nickels := 9 * 10 * 0.05
def euro_cents := 5 * 15 * 0.01 * 1.18
def british_pence := 3 * 20 * 0.01 * 1.4
def japanese_yen := 2 * 10 * 1 * 0.0091
def mexican_pesos := 4 * 5 * 1 * 0.05

def total_usd := us_quarters + canadian_dimes + us_nickels + euro_cents + british_pence + japanese_yen + mexican_pesos

theorem rocco_total_usd : total_usd = 21.167 := by
  simp [us_quarters, canadian_dimes, us_nickels, euro_cents, british_pence, japanese_yen, mexican_pesos]
  sorry

end rocco_total_usd_l1103_110333


namespace baker_cakes_total_l1103_110316

-- Define the variables corresponding to the conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- State the theorem to prove that the total number of cakes made is 217
theorem baker_cakes_total : cakes_sold + cakes_left = 217 := 
by 
-- The proof is omitted according to the instructions
sorry

end baker_cakes_total_l1103_110316


namespace roberta_has_11_3_left_l1103_110335

noncomputable def roberta_leftover_money (initial: ℝ) (shoes: ℝ) (bag: ℝ) (lunch: ℝ) (dress: ℝ) (accessory: ℝ) : ℝ :=
  initial - (shoes + bag + lunch + dress + accessory)

theorem roberta_has_11_3_left :
  roberta_leftover_money 158 45 28 (28 / 4) (62 - 0.15 * 62) (2 * (28 / 4)) = 11.3 :=
by
  sorry

end roberta_has_11_3_left_l1103_110335


namespace find_2x_plus_y_l1103_110385

theorem find_2x_plus_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 5 / 3) 
  (h2 : x + 2*y = 8) : 
  2*x + y = 7 :=
sorry

end find_2x_plus_y_l1103_110385


namespace abs_inequality_solution_set_l1103_110370

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 1| + |x + 2| < 5 ↔ -3 < x ∧ x < 2 :=
by {
  sorry
}

end abs_inequality_solution_set_l1103_110370


namespace mickey_horses_per_week_l1103_110381

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l1103_110381


namespace value_of_expression_l1103_110345

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : x^2 + 3*x - 2 = 0 := 
by {
  -- proof logic will be here
  sorry
}

end value_of_expression_l1103_110345


namespace solution_is_unique_zero_l1103_110375

theorem solution_is_unique_zero : ∀ (x y z : ℤ), x^3 + 2 * y^3 = 4 * z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros x y z h
  sorry

end solution_is_unique_zero_l1103_110375


namespace alice_minimum_speed_l1103_110395

-- Conditions
def distance : ℝ := 60 -- The distance from City A to City B in miles
def bob_speed : ℝ := 40 -- Bob's constant speed in miles per hour
def alice_delay : ℝ := 0.5 -- Alice's delay in hours before she starts

-- Question as a proof statement
theorem alice_minimum_speed : ∀ (alice_speed : ℝ), alice_speed > 60 → 
  (alice_speed * (1.5 - alice_delay) < distance) → true :=
by
  sorry

end alice_minimum_speed_l1103_110395


namespace coaches_together_next_l1103_110310

theorem coaches_together_next (a b c d : ℕ) (h_a : a = 5) (h_b : b = 9) (h_c : c = 8) (h_d : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 3960 :=
by 
  rw [h_a, h_b, h_c, h_d]
  sorry

end coaches_together_next_l1103_110310


namespace find_number_l1103_110340

theorem find_number (Number : ℝ) (h : Number / 5 = 30 / 600) : Number = 1 / 4 :=
by sorry

end find_number_l1103_110340


namespace find_x_plus_2y_sq_l1103_110389

theorem find_x_plus_2y_sq (x y : ℝ) 
  (h : 8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1) : 
  x + 2 * y^2 = 1 / 2 :=
sorry

end find_x_plus_2y_sq_l1103_110389


namespace find_n_sequence_l1103_110355

theorem find_n_sequence (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 45) (h1 : b 1 = 80) (hn : b n = 0)
  (hrec : ∀ k, 1 ≤ k ∧ k ≤ n-1 → b (k+1) = b (k-1) - 4 / b k) :
  n = 901 :=
sorry

end find_n_sequence_l1103_110355


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l1103_110364

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l1103_110364


namespace complete_square_l1103_110348

theorem complete_square (x m : ℝ) : x^2 + 2 * x - 2 = 0 → (x + m)^2 = 3 → m = 1 := sorry

end complete_square_l1103_110348


namespace instantaneous_velocity_at_3_l1103_110317

noncomputable def s (t : ℝ) : ℝ := t^2 + 10

theorem instantaneous_velocity_at_3 :
  deriv s 3 = 6 :=
by {
  -- proof goes here
  sorry
}

end instantaneous_velocity_at_3_l1103_110317


namespace line_of_intersection_l1103_110392

theorem line_of_intersection (x y z : ℝ) :
  (2 * x + 3 * y + 3 * z - 9 = 0) ∧ (4 * x + 2 * y + z - 8 = 0) →
  ((x / 4.5 + y / 3 + z / 3 = 1) ∧ (x / 2 + y / 4 + z / 8 = 1)) :=
by
  sorry

end line_of_intersection_l1103_110392


namespace exactly_two_toads_l1103_110331

universe u

structure Amphibian where
  brian : Bool
  julia : Bool
  sean : Bool
  victor : Bool

def are_same_species (x y : Bool) : Bool := x = y

-- Definitions of statements by each amphibian
def Brian_statement (a : Amphibian) : Bool :=
  are_same_species a.brian a.sean

def Julia_statement (a : Amphibian) : Bool :=
  a.victor

def Sean_statement (a : Amphibian) : Bool :=
  ¬ a.julia

def Victor_statement (a : Amphibian) : Bool :=
  (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2

-- Conditions translated to Lean definition
def valid_statements (a : Amphibian) : Prop :=
  (a.brian → Brian_statement a) ∧
  (¬ a.brian → ¬ Brian_statement a) ∧
  (a.julia → Julia_statement a) ∧
  (¬ a.julia → ¬ Julia_statement a) ∧
  (a.sean → Sean_statement a) ∧
  (¬ a.sean → ¬ Sean_statement a) ∧
  (a.victor → Victor_statement a) ∧
  (¬ a.victor → ¬ Victor_statement a)

theorem exactly_two_toads (a : Amphibian) (h : valid_statements a) : 
( (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2 ) :=
sorry

end exactly_two_toads_l1103_110331


namespace problem_statement_l1103_110356

-- Definition of the function f with the given condition
def satisfies_condition (f : ℝ → ℝ) := ∀ (α β : ℝ), f (α + β) - (f α + f β) = 2008

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ (x : ℝ), f (-x) = -f x

-- Main statement to prove in Lean
theorem problem_statement (f : ℝ → ℝ) (h : satisfies_condition f) : is_odd (fun x => f x + 2008) :=
sorry

end problem_statement_l1103_110356


namespace fraction_zero_iff_numerator_zero_l1103_110341

-- Define the conditions and the result in Lean 4.
theorem fraction_zero_iff_numerator_zero (x : ℝ) (h : x ≠ 0) : (x - 3) / x = 0 ↔ x = 3 :=
by
  sorry

end fraction_zero_iff_numerator_zero_l1103_110341


namespace geometric_sequence_sum_10_l1103_110390

theorem geometric_sequence_sum_10 (a : ℕ) (r : ℕ) (h : r = 2) (sum5 : a + r * a + r^2 * a + r^3 * a + r^4 * a = 1) : 
    a * (1 - r^10) / (1 - r) = 33 := 
by 
    sorry

end geometric_sequence_sum_10_l1103_110390


namespace smallest_num_rectangles_to_cover_square_l1103_110366

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l1103_110366


namespace original_number_is_16_l1103_110311

theorem original_number_is_16 (x : ℕ) : 213 * x = 3408 → x = 16 :=
by
  sorry

end original_number_is_16_l1103_110311


namespace taxi_fare_calculation_l1103_110301

def fare_per_km : ℝ := 1.8
def starting_fare : ℝ := 8
def starting_distance : ℝ := 2
def total_distance : ℝ := 12

theorem taxi_fare_calculation : 
  (if total_distance <= starting_distance then starting_fare
   else starting_fare + (total_distance - starting_distance) * fare_per_km) = 26 := by
  sorry

end taxi_fare_calculation_l1103_110301


namespace rectangle_side_excess_percentage_l1103_110360

theorem rectangle_side_excess_percentage (A B : ℝ) (x : ℝ) (h : A * (1 + x) * B * (1 - 0.04) = A * B * 1.008) : x = 0.05 :=
by
  sorry

end rectangle_side_excess_percentage_l1103_110360


namespace sum_of_common_divisors_is_10_l1103_110367

-- Define the list of numbers
def numbers : List ℤ := [42, 84, -14, 126, 210]

-- Define the common divisors
def common_divisors : List ℕ := [1, 2, 7]

-- Define the function that checks if a number is a common divisor of all numbers in the list
def is_common_divisor (d : ℕ) : Prop :=
  ∀ n ∈ numbers, (d : ℤ) ∣ n

-- Specify the sum of the common divisors
def sum_common_divisors : ℕ := common_divisors.sum

-- State the theorem to be proved
theorem sum_of_common_divisors_is_10 : 
  (∀ d ∈ common_divisors, is_common_divisor d) → 
  sum_common_divisors = 10 := 
by
  sorry

end sum_of_common_divisors_is_10_l1103_110367


namespace arithmetic_sequence_angle_l1103_110349

-- Define the conditions
variables (A B C a b c : ℝ)
-- The statement assumes that A, B, C form an arithmetic sequence
-- which implies 2B = A + C
-- We need to show that 1/(a + b) + 1/(b + c) = 3/(a + b + c)

theorem arithmetic_sequence_angle
  (h : 2 * B = A + C)
  (cos_rule : b^2 = c^2 + a^2 - 2 * c * a * Real.cos B):
    1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := sorry

end arithmetic_sequence_angle_l1103_110349


namespace number_of_solutions_l1103_110324

theorem number_of_solutions (p : ℕ) (hp : Nat.Prime p) : (∃ n : ℕ, 
  (p % 4 = 1 → n = 11) ∧
  (p = 2 → n = 5) ∧
  (p % 4 = 3 → n = 3)) :=
sorry

end number_of_solutions_l1103_110324


namespace sum_of_interior_angles_l1103_110307

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1980) :
    180 * ((n + 3) - 2) = 2520 :=
by
  sorry

end sum_of_interior_angles_l1103_110307


namespace paul_packed_total_toys_l1103_110306

def toys_in_box : ℕ := 8
def number_of_boxes : ℕ := 4
def total_toys_packed (toys_in_box number_of_boxes : ℕ) : ℕ := toys_in_box * number_of_boxes

theorem paul_packed_total_toys :
  total_toys_packed toys_in_box number_of_boxes = 32 :=
by
  sorry

end paul_packed_total_toys_l1103_110306


namespace find_B_in_product_l1103_110347

theorem find_B_in_product (B : ℕ) (hB : B < 10) (h : (B * 100 + 2) * (900 + B) = 8016) : B = 8 := by
  sorry

end find_B_in_product_l1103_110347


namespace rhombus_area_three_times_diagonals_l1103_110329

theorem rhombus_area_three_times_diagonals :
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  (new_d1 * new_d2) / 2 = 108 :=
by
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  have h : (new_d1 * new_d2) / 2 = 108 := sorry
  exact h

end rhombus_area_three_times_diagonals_l1103_110329


namespace solve_equation_1_solve_quadratic_equation_2_l1103_110343

theorem solve_equation_1 (x : ℝ) : 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2 := sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  4 * x^2 - 2 * (Real.sqrt 3) * x - 1 = 0 ↔
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4 := sorry

end solve_equation_1_solve_quadratic_equation_2_l1103_110343


namespace parallelogram_area_l1103_110368

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l1103_110368


namespace Paul_sold_350_pencils_l1103_110308

-- Variables representing conditions
def pencils_per_day : ℕ := 100
def days_in_week : ℕ := 5
def starting_stock : ℕ := 80
def ending_stock : ℕ := 230

-- The total pencils Paul made in a week
def total_pencils_made : ℕ := pencils_per_day * days_in_week

-- The total pencils before selling any
def total_pencils_before_selling : ℕ := total_pencils_made + starting_stock

-- The number of pencils sold is the difference between total pencils before selling and ending stock
def pencils_sold : ℕ := total_pencils_before_selling - ending_stock

theorem Paul_sold_350_pencils :
  pencils_sold = 350 :=
by {
  -- The proof body is replaced with sorry to indicate a placeholder for the proof.
  sorry
}

end Paul_sold_350_pencils_l1103_110308


namespace min_x_y_l1103_110398

theorem min_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_x_y_l1103_110398


namespace percentage_difference_correct_l1103_110312

noncomputable def percentage_difference (initial_price : ℝ) (increase_2012_percent : ℝ) (decrease_2013_percent : ℝ) : ℝ :=
  let price_end_2012 := initial_price * (1 + increase_2012_percent / 100)
  let price_end_2013 := price_end_2012 * (1 - decrease_2013_percent / 100)
  ((price_end_2013 - initial_price) / initial_price) * 100

theorem percentage_difference_correct :
  ∀ (initial_price : ℝ),
  percentage_difference initial_price 25 12 = 10 := 
by
  intros
  sorry

end percentage_difference_correct_l1103_110312


namespace at_least_one_composite_l1103_110314

theorem at_least_one_composite (a b c : ℕ) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) 
    (h_not_perfect_square : ∀ m : ℕ, m * m ≠ a) : 
    a ^ 2 + a + 1 = 3 * (b ^ 2 + b + 1) * (c ^ 2 + c + 1) →
    (∃ p, p > 1 ∧ p ∣ (b ^ 2 + b + 1)) ∨ (∃ q, q > 1 ∧ q ∣ (c ^ 2 + c + 1)) :=
by sorry

end at_least_one_composite_l1103_110314


namespace Mike_monthly_time_is_200_l1103_110325

def tv_time (days : Nat) (hours_per_day : Nat) : Nat := days * hours_per_day

def video_game_time (total_tv_time_per_week : Nat) (num_days_playing : Nat) : Nat :=
  (total_tv_time_per_week / 7 / 2) * num_days_playing

def piano_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours * 5 + weekend_hours * 2

def weekly_time (tv_time : Nat) (video_game_time : Nat) (piano_time : Nat) : Nat :=
  tv_time + video_game_time + piano_time

def monthly_time (weekly_time : Nat) (weeks : Nat) : Nat :=
  weekly_time * weeks

theorem Mike_monthly_time_is_200 : monthly_time
  (weekly_time 
     (tv_time 3 4 + tv_time 2 3 + tv_time 2 5) 
     (video_game_time 28 3) 
     (piano_time 2 3))
  4 = 200 :=
  by
  sorry

end Mike_monthly_time_is_200_l1103_110325


namespace max_min_value_of_a_l1103_110382

theorem max_min_value_of_a 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end max_min_value_of_a_l1103_110382


namespace greatest_value_of_sum_l1103_110344

variable (a b c : ℕ)

theorem greatest_value_of_sum
    (h1 : 2022 < a)
    (h2 : 2022 < b)
    (h3 : 2022 < c)
    (h4 : ∃ k1 : ℕ, a + b = k1 * (c - 2022))
    (h5 : ∃ k2 : ℕ, a + c = k2 * (b - 2022))
    (h6 : ∃ k3 : ℕ, b + c = k3 * (a - 2022)) :
    a + b + c = 2022 * 85 := 
  sorry

end greatest_value_of_sum_l1103_110344


namespace jasmine_gives_lola_marbles_l1103_110328

theorem jasmine_gives_lola_marbles :
  ∃ (y : ℕ), ∀ (j l : ℕ), 
    j = 120 ∧ l = 15 ∧ 120 - y = 3 * (15 + y) → y = 19 := 
sorry

end jasmine_gives_lola_marbles_l1103_110328


namespace calculate_principal_l1103_110339

theorem calculate_principal
  (I : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hI : I = 8625)
  (hR : R = 50 / 3)
  (hT : T = 3 / 4)
  (hInterest : I = (P * (R / 100) * T)) :
  P = 6900000 := by
  sorry

end calculate_principal_l1103_110339


namespace max_value_of_sum_l1103_110300

theorem max_value_of_sum (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) :
  a + b + c + d ≤ -5 := 
sorry

end max_value_of_sum_l1103_110300


namespace profit_function_profit_for_240_barrels_barrels_for_760_profit_l1103_110373

-- Define fixed costs, cost price per barrel, and selling price per barrel as constants
def fixed_costs : ℝ := 200
def cost_price_per_barrel : ℝ := 5
def selling_price_per_barrel : ℝ := 8

-- Definitions for daily sales quantity (x) and daily profit (y)
def daily_sales_quantity (x : ℝ) : ℝ := x
def daily_profit (x : ℝ) : ℝ := (selling_price_per_barrel * x) - (cost_price_per_barrel * x) - fixed_costs

-- Prove the functional relationship y = 3x - 200
theorem profit_function (x : ℝ) : daily_profit x = 3 * x - fixed_costs :=
by sorry

-- Given sales quantity is 240 barrels, prove profit is 520 yuan
theorem profit_for_240_barrels : daily_profit 240 = 520 :=
by sorry

-- Given profit is 760 yuan, prove sales quantity is 320 barrels
theorem barrels_for_760_profit : ∃ (x : ℝ), daily_profit x = 760 ∧ x = 320 :=
by sorry

end profit_function_profit_for_240_barrels_barrels_for_760_profit_l1103_110373


namespace parallel_lines_slope_l1103_110330

theorem parallel_lines_slope (n : ℝ) :
  (∀ x y : ℝ, 2 * x + 2 * y - 5 = 0 → 4 * x + n * y + 1 = 0 → -1 = - (4 / n)) →
  n = 4 :=
by sorry

end parallel_lines_slope_l1103_110330


namespace range_of_phi_l1103_110313

theorem range_of_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : |φ| < (Real.pi / 2))
  (h3 : ∀ x, f x = Real.sin (ω * x + φ))
  (h4 : ∀ x, f (x + (Real.pi / ω)) = f x)
  (h5 : ∀ x y, (x ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) ∧
                  (y ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) → 
                  (x < y → f x ≤ f y)) :
  (φ ∈ Set.Icc (- Real.pi / 6) (- Real.pi / 10)) :=
by
  sorry

end range_of_phi_l1103_110313


namespace mushroom_ratio_l1103_110380

theorem mushroom_ratio (total_mushrooms safe_mushrooms uncertain_mushrooms : ℕ)
  (h_total : total_mushrooms = 32)
  (h_safe : safe_mushrooms = 9)
  (h_uncertain : uncertain_mushrooms = 5) :
  (total_mushrooms - safe_mushrooms - uncertain_mushrooms) / safe_mushrooms = 2 :=
by sorry

end mushroom_ratio_l1103_110380


namespace sum_of_ages_l1103_110327

theorem sum_of_ages (a b : ℕ) :
  let c1 := a
  let c2 := a + 2
  let c3 := a + 4
  let c4 := a + 6
  let coach1 := b
  let coach2 := b + 2
  c1^2 + c2^2 + c3^2 + c4^2 + coach1^2 + coach2^2 = 2796 →
  c1 + c2 + c3 + c4 + coach1 + coach2 = 106 :=
by
  intro h
  sorry

end sum_of_ages_l1103_110327


namespace range_of_t_l1103_110305

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - exp 1 * x + exp 1 ^ 2)

theorem range_of_t :
  (∀ x > 1, ∀ t > 0, (t + 1) * g x ≤ t * f x)
  ↔ (∀ t > 0, t ≥ 1 / (exp 1 ^ 2 - 1)) :=
by
  sorry

end range_of_t_l1103_110305


namespace wendy_initial_flowers_l1103_110394

theorem wendy_initial_flowers (wilted: ℕ) (bouquets_made: ℕ) (flowers_per_bouquet: ℕ) (flowers_initially_picked: ℕ):
  wilted = 35 →
  bouquets_made = 2 →
  flowers_per_bouquet = 5 →
  flowers_initially_picked = wilted + bouquets_made * flowers_per_bouquet →
  flowers_initially_picked = 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_initial_flowers_l1103_110394


namespace inequality_holds_l1103_110361

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := sorry

end inequality_holds_l1103_110361


namespace factorize_expression_l1103_110399

theorem factorize_expression (x : ℝ) : 
  (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 := 
by sorry

end factorize_expression_l1103_110399


namespace non_zero_real_m_value_l1103_110387

theorem non_zero_real_m_value (m : ℝ) (h1 : 3 - m ∈ ({1, 2, 3} : Set ℝ)) (h2 : m ≠ 0) : m = 2 := 
sorry

end non_zero_real_m_value_l1103_110387


namespace ants_square_paths_l1103_110321

theorem ants_square_paths (a : ℝ) :
  (∃ a, a = 4 ∧ a + 2 = 6 ∧ a + 4 = 8) →
  (∀ (Mu Ra Vey : ℝ), 
    (Mu = (a + 4) / 2) ∧ 
    (Ra = (a + 2) / 2 + 1) ∧ 
    (Vey = 6) →
    (Mu + Ra + Vey = 2 * (a + 4) + 2)) :=
sorry

end ants_square_paths_l1103_110321


namespace solve_for_b_l1103_110357

theorem solve_for_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 :=
sorry

end solve_for_b_l1103_110357


namespace solve_inequality_l1103_110352

theorem solve_inequality (y : ℚ) :
  (3 / 40 : ℚ) + |y - (17 / 80 : ℚ)| < (1 / 8 : ℚ) ↔ (13 / 80 : ℚ) < y ∧ y < (21 / 80 : ℚ) := 
by
  sorry

end solve_inequality_l1103_110352


namespace corey_candies_l1103_110374

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l1103_110374


namespace crazy_silly_school_diff_books_movies_l1103_110323

theorem crazy_silly_school_diff_books_movies 
    (total_books : ℕ) (total_movies : ℕ)
    (hb : total_books = 36)
    (hm : total_movies = 25) :
    total_books - total_movies = 11 :=
by {
  sorry
}

end crazy_silly_school_diff_books_movies_l1103_110323


namespace number_of_monsters_l1103_110309

theorem number_of_monsters
    (M S : ℕ)
    (h1 : 4 * M + 3 = S)
    (h2 : 5 * M = S - 6) :
  M = 9 :=
sorry

end number_of_monsters_l1103_110309


namespace correct_region_l1103_110372

-- Define the condition for x > 1
def condition_x_gt_1 (x : ℝ) (y : ℝ) : Prop :=
  x > 1 → y^2 > x

-- Define the condition for 0 < x < 1
def condition_0_lt_x_lt_1 (x : ℝ) (y : ℝ) : Prop :=
  0 < x ∧ x < 1 → 0 < y^2 ∧ y^2 < x

-- Formal statement to check the correct region
theorem correct_region (x y : ℝ) : 
  (condition_x_gt_1 x y ∨ condition_0_lt_x_lt_1 x y) →
  y^2 > x ∨ (0 < y^2 ∧ y^2 < x) :=
sorry

end correct_region_l1103_110372


namespace problem_solution_l1103_110369

def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }
def complement_N : Set ℝ := { x | x ≤ 0 ∨ x ≥ 1 }

theorem problem_solution : M ∪ complement_N = Set.univ := 
sorry

end problem_solution_l1103_110369


namespace last_digit_of_product_l1103_110397

theorem last_digit_of_product :
    (3 ^ 65 * 6 ^ 59 * 7 ^ 71) % 10 = 4 := 
  by sorry

end last_digit_of_product_l1103_110397


namespace jenny_collects_20_cans_l1103_110319

theorem jenny_collects_20_cans (b c : ℕ) (h1 : 6 * b + 2 * c = 100) (h2 : 10 * b + 3 * c = 160) : c = 20 := 
by sorry

end jenny_collects_20_cans_l1103_110319


namespace fencing_required_l1103_110320

def width : ℝ := 25
def area : ℝ := 260
def height_difference : ℝ := 15
def extra_fencing_per_5ft_height : ℝ := 2

noncomputable def length : ℝ := area / width

noncomputable def expected_fencing : ℝ := 2 * length + width + (height_difference / 5) * extra_fencing_per_5ft_height

-- Theorem stating the problem's conclusion
theorem fencing_required : expected_fencing = 51.8 := by
  sorry -- Proof will go here

end fencing_required_l1103_110320


namespace factor_expression_l1103_110365

-- Define the expressions E1 and E2
def E1 (y : ℝ) : ℝ := 12 * y^6 + 35 * y^4 - 5
def E2 (y : ℝ) : ℝ := 2 * y^6 - 4 * y^4 + 5

-- Define the target expression E
def E (y : ℝ) : ℝ := E1 y - E2 y

-- The main theorem to prove
theorem factor_expression (y : ℝ) : E y = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end factor_expression_l1103_110365


namespace megan_savings_days_l1103_110362

theorem megan_savings_days :
  let josiah_saving_rate : ℝ := 0.25
  let josiah_days : ℕ := 24
  let josiah_total := josiah_saving_rate * josiah_days

  let leah_saving_rate : ℝ := 0.5
  let leah_days : ℕ := 20
  let leah_total := leah_saving_rate * leah_days

  let total_savings : ℝ := 28.0
  let josiah_leah_total := josiah_total + leah_total
  let megan_total := total_savings - josiah_leah_total

  let megan_saving_rate := 2 * leah_saving_rate
  let megan_days := megan_total / megan_saving_rate
  
  megan_days = 12 :=
by
  sorry

end megan_savings_days_l1103_110362


namespace xyz_value_l1103_110346

theorem xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (hx : x * (y + z) = 162)
    (hy : y * (z + x) = 180)
    (hz : z * (x + y) = 198)
    (h_sum : x + y + z = 26) :
    x * y * z = 2294.67 :=
by
  sorry

end xyz_value_l1103_110346


namespace volume_of_regular_triangular_pyramid_l1103_110304

noncomputable def regular_triangular_pyramid_volume (h : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3) / 2

theorem volume_of_regular_triangular_pyramid (h : ℝ) :
  regular_triangular_pyramid_volume h = (h^3 * Real.sqrt 3) / 2 :=
by
  sorry

end volume_of_regular_triangular_pyramid_l1103_110304


namespace gcd_84_108_132_156_l1103_110354

theorem gcd_84_108_132_156 : Nat.gcd (Nat.gcd 84 108) (Nat.gcd 132 156) = 12 := 
by
  sorry

end gcd_84_108_132_156_l1103_110354


namespace boys_to_girls_ratio_l1103_110388

theorem boys_to_girls_ratio (S G B : ℕ) (h : (1/2 : ℚ) * G = (1/3 : ℚ) * S) :
  B / G = 1 / 2 :=
by sorry

end boys_to_girls_ratio_l1103_110388


namespace largest_obtuse_prime_angle_l1103_110351

theorem largest_obtuse_prime_angle (alpha beta gamma : ℕ) 
    (h_triangle_sum : alpha + beta + gamma = 180) 
    (h_alpha_gt_beta : alpha > beta) 
    (h_beta_gt_gamma : beta > gamma)
    (h_obtuse_alpha : alpha > 90) 
    (h_alpha_prime : Prime alpha) 
    (h_beta_prime : Prime beta) : 
    alpha = 173 := 
sorry

end largest_obtuse_prime_angle_l1103_110351


namespace tammy_investment_change_l1103_110326

theorem tammy_investment_change :
  ∀ (initial_investment : ℝ) (loss_percent : ℝ) (gain_percent : ℝ),
    initial_investment = 200 → 
    loss_percent = 0.2 → 
    gain_percent = 0.25 →
    ((initial_investment * (1 - loss_percent)) * (1 + gain_percent)) = initial_investment :=
by
  intros initial_investment loss_percent gain_percent
  sorry

end tammy_investment_change_l1103_110326


namespace scalene_triangle_not_unique_by_two_non_opposite_angles_l1103_110383

theorem scalene_triangle_not_unique_by_two_non_opposite_angles
  (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α + β < π) :
  ∃ (γ δ : ℝ), γ ≠ δ ∧ γ + α + β = δ + α + β :=
sorry

end scalene_triangle_not_unique_by_two_non_opposite_angles_l1103_110383


namespace triangle_third_side_length_l1103_110303

theorem triangle_third_side_length
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b = 10)
  (h2 : c = 7)
  (h3 : A = 2 * B) :
  a = (50 + 5 * Real.sqrt 2) / 7 ∨ a = (50 - 5 * Real.sqrt 2) / 7 :=
sorry

end triangle_third_side_length_l1103_110303


namespace Kath_takes_3_friends_l1103_110302

theorem Kath_takes_3_friends
  (total_paid: Int)
  (price_before_6: Int)
  (price_reduction: Int)
  (num_family_members: Int)
  (start_time: Int)
  (start_time_condition: start_time < 18)
  (total_payment_condition: total_paid = 30)
  (admission_cost_before_6: price_before_6 = 8 - price_reduction)
  (num_family_members_condition: num_family_members = 3):
  (total_paid / price_before_6 - num_family_members = 3) := 
by
  -- Since no proof is required, simply add sorry to skip the proof
  sorry

end Kath_takes_3_friends_l1103_110302


namespace problem1_problem2_l1103_110377

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions:
axiom condition1 : sin (α + π / 6) = sqrt 10 / 10
axiom condition2 : cos (α + π / 6) = 3 * sqrt 10 / 10
axiom condition3 : tan (α + β) = 2 / 5

-- Prove:
theorem problem1 : sin (2 * α + π / 6) = (3 * sqrt 3 - 4) / 10 :=
by sorry

theorem problem2 : tan (2 * β - π / 3) = 17 / 144 :=
by sorry

end problem1_problem2_l1103_110377


namespace interval_contains_zeros_l1103_110396

-- Define the conditions and the function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c 

theorem interval_contains_zeros (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) : 
  ∃ x ∈ Set.Ioc (-2 : ℝ) 0, quadratic a b c x = 0 :=
by
  -- Problem Statement: given conditions, interval (-2, 0) contains a zero
  sorry

end interval_contains_zeros_l1103_110396


namespace smallest_number_of_2_by_3_rectangles_l1103_110359

def area_2_by_3_rectangle : Int := 2 * 3

def smallest_square_area_multiple_of_6 : Int :=
  let side_length := 6
  side_length * side_length

def number_of_rectangles_to_cover_square (square_area : Int) (rectangle_area : Int) : Int :=
  square_area / rectangle_area

theorem smallest_number_of_2_by_3_rectangles :
  number_of_rectangles_to_cover_square smallest_square_area_multiple_of_6 area_2_by_3_rectangle = 6 := by
  sorry

end smallest_number_of_2_by_3_rectangles_l1103_110359


namespace chessboard_ratio_sum_l1103_110322

theorem chessboard_ratio_sum :
  let m := 19
  let n := 135
  m + n = 154 :=
by
  sorry

end chessboard_ratio_sum_l1103_110322
