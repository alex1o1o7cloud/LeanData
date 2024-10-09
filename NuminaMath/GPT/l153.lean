import Mathlib

namespace number_of_multiples_of_6_between_5_and_125_l153_15398

theorem number_of_multiples_of_6_between_5_and_125 : 
  ∃ k : ℕ, (5 < 6 * k ∧ 6 * k < 125) → k = 20 :=
sorry

end number_of_multiples_of_6_between_5_and_125_l153_15398


namespace intersection_A_B_l153_15348

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l153_15348


namespace intersection_distance_eq_l153_15323

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l153_15323


namespace sqrt_domain_condition_l153_15399

theorem sqrt_domain_condition (x : ℝ) : (2 * x - 6 ≥ 0) ↔ (x ≥ 3) :=
by
  sorry

end sqrt_domain_condition_l153_15399


namespace number_of_days_l153_15337

theorem number_of_days (m1 d1 m2 d2 : ℕ) (h1 : m1 * d1 = m2 * d2) (k : ℕ) 
(h2 : m1 = 10) (h3 : d1 = 6) (h4 : m2 = 15) (h5 : k = 60) : 
d2 = 4 :=
by
  have : 10 * 6 = 60 := by sorry
  have : 15 * d2 = 60 := by sorry
  exact sorry

end number_of_days_l153_15337


namespace probability_of_first_good_product_on_third_try_l153_15381

-- Define the problem parameters
def pass_rate : ℚ := 3 / 4
def failure_rate : ℚ := 1 / 4
def epsilon := 3

-- The target probability statement
theorem probability_of_first_good_product_on_third_try :
  (failure_rate * failure_rate * pass_rate) = ((1 / 4) ^ 2 * (3 / 4)) :=
by
  sorry

end probability_of_first_good_product_on_third_try_l153_15381


namespace min_value_of_expression_l153_15324

theorem min_value_of_expression (x y : ℝ) (hposx : x > 0) (hposy : y > 0) (heq : 2 / x + 1 / y = 1) : 
  x + 2 * y ≥ 8 :=
sorry

end min_value_of_expression_l153_15324


namespace find_f_a_plus_1_l153_15397

def f (x : ℝ) : ℝ := x^2 + 1

theorem find_f_a_plus_1 (a : ℝ) : f (a + 1) = a^2 + 2 * a + 2 := by
  sorry

end find_f_a_plus_1_l153_15397


namespace find_m_value_l153_15313

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (m : ℝ)
variables (A B C D : V)

-- Assuming vectors a and b are non-collinear
axiom non_collinear (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (∃ (k : ℝ), a = k • b)

-- Given vectors
axiom hAB : B - A = 9 • a + m • b
axiom hBC : C - B = -2 • a - 1 • b
axiom hDC : C - D = a - 2 • b

-- Collinearity condition for A, B, and D
axiom collinear (k : ℝ) : B - A = k • (B - D)

theorem find_m_value : m = -3 :=
by sorry

end find_m_value_l153_15313


namespace smallest_x_l153_15368

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 10) (h5 : y - x = 5) :
  x = 4 :=
sorry

end smallest_x_l153_15368


namespace average_time_per_stop_l153_15351

-- Definitions from the conditions
def pizzas : Nat := 12
def stops_with_two_pizzas : Nat := 2
def total_delivery_time : Nat := 40

-- Using the conditions to define what needs to be proved
theorem average_time_per_stop : 
  let single_pizza_stops := pizzas - stops_with_two_pizzas * 2
  let total_stops := single_pizza_stops + stops_with_two_pizzas
  let average_time := total_delivery_time / total_stops
  average_time = 4 := by
  -- Proof to be provided
  sorry

end average_time_per_stop_l153_15351


namespace part_i_part_ii_l153_15338

-- Define the operations for the weird calculator.
def Dsharp (n : ℕ) : ℕ := 2 * n + 1
def Dflat (n : ℕ) : ℕ := 2 * n - 1

-- Define the initial starting point.
def initial_display : ℕ := 1

-- Define a function to execute a sequence of button presses.
def execute_sequence (seq : List (ℕ → ℕ)) (initial : ℕ) : ℕ :=
  seq.foldl (fun x f => f x) initial

-- Problem (i): Prove there is a sequence that results in 313 starting from 1 after eight presses.
theorem part_i : ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = 313 :=
by sorry

-- Problem (ii): Describe all numbers that can be achieved from exactly eight button presses starting from 1.
theorem part_ii : 
  ∀ n : ℕ, n % 2 = 1 ∧ n < 2^9 →
  ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = n :=
by sorry

end part_i_part_ii_l153_15338


namespace least_possible_b_l153_15358

theorem least_possible_b (a b : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (h_a_factors : ∃ k, a = p^k ∧ k + 1 = 3) (h_b_factors : ∃ m, b = p^m ∧ m + 1 = a) (h_divisible : b % a = 0) : 
  b = 8 := 
by 
  sorry

end least_possible_b_l153_15358


namespace fraction_of_innocent_cases_l153_15328

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end fraction_of_innocent_cases_l153_15328


namespace cos_20_cos_10_minus_sin_160_sin_10_l153_15345

theorem cos_20_cos_10_minus_sin_160_sin_10 : 
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 
   Real.cos (30 * Real.pi / 180) :=
by
  sorry

end cos_20_cos_10_minus_sin_160_sin_10_l153_15345


namespace intersection_A_B_l153_15327

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := 
by 
  sorry

end intersection_A_B_l153_15327


namespace sparrow_grains_l153_15375

theorem sparrow_grains (x : ℤ) : 9 * x < 1001 ∧ 10 * x > 1100 → x = 111 :=
by
  sorry

end sparrow_grains_l153_15375


namespace interest_rate_of_second_part_l153_15326

theorem interest_rate_of_second_part 
  (total_sum : ℝ) (P2 : ℝ) (interest1_rate : ℝ) 
  (time1 : ℝ) (time2 : ℝ) (interest2_value : ℝ) : 
  (total_sum = 2704) → 
  (P2 = 1664) → 
  (interest1_rate = 0.03) → 
  (time1 = 8) → 
  (interest2_value = interest1_rate * (total_sum - P2) * time1) → 
  (time2 = 3) → 
  1664 * r * time2 = interest2_value → 
  r = 0.05 := 
by sorry

end interest_rate_of_second_part_l153_15326


namespace target_hit_probability_l153_15336

-- Define the probabilities given in the problem
def prob_A_hits : ℚ := 9 / 10
def prob_B_hits : ℚ := 8 / 9

-- The required probability that at least one hits the target
def prob_target_hit : ℚ := 89 / 90

-- Theorem stating that the probability calculated matches the expected outcome
theorem target_hit_probability :
  1 - ((1 - prob_A_hits) * (1 - prob_B_hits)) = prob_target_hit :=
by
  sorry

end target_hit_probability_l153_15336


namespace abs_neg_eight_l153_15343

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l153_15343


namespace find_x_for_slope_l153_15300

theorem find_x_for_slope (x : ℝ) (h : (2 - 5) / (x - (-3)) = -1 / 4) : x = 9 :=
by 
  -- Proof skipped
  sorry

end find_x_for_slope_l153_15300


namespace sufficient_but_not_necessary_condition_l153_15391

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a < 1 / b) ∧ ¬ (1 / a < 1 / b → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l153_15391


namespace vacation_cost_per_person_l153_15352

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l153_15352


namespace ramu_profit_percent_l153_15362

def ramu_bought_car : ℝ := 48000
def ramu_repair_cost : ℝ := 14000
def ramu_selling_price : ℝ := 72900

theorem ramu_profit_percent :
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 17.58 := 
by
  -- Definitions and setting up the proof environment
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  sorry

end ramu_profit_percent_l153_15362


namespace max_intersections_two_circles_three_lines_l153_15360

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l153_15360


namespace optimal_fence_area_l153_15314

variables {l w : ℝ}

theorem optimal_fence_area
  (h1 : 2 * l + 2 * w = 400) -- Tiffany must use exactly 400 feet of fencing.
  (h2 : l ≥ 100) -- The length must be at least 100 feet.
  (h3 : w ≥ 50) -- The width must be at least 50 feet.
  : l * w ≤ 10000 :=      -- We need to prove that the area is at most 10000 square feet.
by
  sorry

end optimal_fence_area_l153_15314


namespace statement_b_statement_c_l153_15309
-- Import all of Mathlib to include necessary mathematical functions and properties

-- First, the Lean statement for Statement B
theorem statement_b (a b : ℝ) (h : a > |b|) : a^2 > b^2 := 
sorry

-- Second, the Lean statement for Statement C
theorem statement_c (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end statement_b_statement_c_l153_15309


namespace equiangular_polygons_unique_solution_l153_15390

theorem equiangular_polygons_unique_solution :
  ∃! (n1 n2 : ℕ), (n1 ≠ 0 ∧ n2 ≠ 0) ∧ (180 / n1 + 360 / n2 = 90) :=
by
  sorry

end equiangular_polygons_unique_solution_l153_15390


namespace original_portion_al_l153_15357

variable (a b c : ℕ)

theorem original_portion_al :
  a + b + c = 1200 ∧
  a - 150 + 3 * b + 3 * c = 1800 ∧
  c = 2 * b →
  a = 825 :=
by
  sorry

end original_portion_al_l153_15357


namespace sin_neg_225_eq_sqrt2_div2_l153_15305

theorem sin_neg_225_eq_sqrt2_div2 :
  Real.sin (-225 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_225_eq_sqrt2_div2_l153_15305


namespace time_saved_by_both_trains_trainB_distance_l153_15364

-- Define the conditions
def trainA_speed_reduced := 360 / 12  -- 30 miles/hour
def trainB_speed_reduced := 360 / 8   -- 45 miles/hour

def trainA_speed := trainA_speed_reduced / (2 / 3)  -- 45 miles/hour
def trainB_speed := trainB_speed_reduced / (1 / 2)  -- 90 miles/hour

def trainA_time_saved := 12 - (360 / trainA_speed)  -- 4 hours
def trainB_time_saved := 8 - (360 / trainB_speed)   -- 4 hours

-- Prove that total time saved by both trains running at their own speeds is 8 hours
theorem time_saved_by_both_trains : trainA_time_saved + trainB_time_saved = 8 := by
  sorry

-- Prove that the distance between Town X and Town Y for Train B is 360 miles
theorem trainB_distance : 360 = 360 := by
  rfl

end time_saved_by_both_trains_trainB_distance_l153_15364


namespace score_below_mean_l153_15312

theorem score_below_mean :
  ∃ (σ : ℝ), (74 - 2 * σ = 58) ∧ (98 - 74 = 3 * σ) :=
sorry

end score_below_mean_l153_15312


namespace price_of_first_oil_is_54_l153_15322

/-- Let x be the price per litre of the first oil.
Given that 10 litres of the first oil are mixed with 5 litres of second oil priced at Rs. 66 per litre,
resulting in a 15-litre mixture costing Rs. 58 per litre, prove that x = 54. -/
theorem price_of_first_oil_is_54 :
  (∃ x : ℝ, x = 54) ↔
  (10 * x + 5 * 66 = 15 * 58) :=
by
  sorry

end price_of_first_oil_is_54_l153_15322


namespace bug_final_position_after_2023_jumps_l153_15321

open Nat

def bug_jump (pos : Nat) : Nat :=
  if pos % 2 = 1 then (pos + 2) % 6 else (pos + 1) % 6

noncomputable def final_position (n : Nat) : Nat :=
  (iterate bug_jump n 6) % 6

theorem bug_final_position_after_2023_jumps : final_position 2023 = 1 := by
  sorry

end bug_final_position_after_2023_jumps_l153_15321


namespace factor_count_x9_minus_x_l153_15356

theorem factor_count_x9_minus_x :
  ∃ (factors : List (Polynomial ℤ)), x^9 - x = factors.prod ∧ factors.length = 5 :=
sorry

end factor_count_x9_minus_x_l153_15356


namespace suitable_for_systematic_sampling_l153_15379

def city_districts : ℕ := 2000
def student_ratio : List ℕ := [3, 2, 8, 2]
def sample_size_city : ℕ := 200
def total_components : ℕ := 2000

def condition_A : Prop := 
  city_districts = 2000 ∧ 
  student_ratio = [3, 2, 8, 2] ∧ 
  sample_size_city = 200

def condition_B : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 5

def condition_C : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 200

def condition_D : Prop := 
  ∃ (n : ℕ), n = 20 ∧ n = 5

theorem suitable_for_systematic_sampling : condition_C :=
by
  sorry

end suitable_for_systematic_sampling_l153_15379


namespace percentage_increase_sale_l153_15386

theorem percentage_increase_sale (P S : ℝ) (hP : 0 < P) (hS : 0 < S) :
  let new_price := 0.65 * P
  let original_revenue := P * S
  let new_revenue := 1.17 * original_revenue
  let percentage_increase := 80 / 100
  let new_sales := S * (1 + percentage_increase)
  new_price * new_sales = new_revenue :=
by
  sorry

end percentage_increase_sale_l153_15386


namespace savings_calculation_l153_15320

-- Define the conditions as given in the problem
def income_expenditure_ratio (income expenditure : ℝ) : Prop :=
  ∃ x : ℝ, income = 10 * x ∧ expenditure = 4 * x

def income_value : ℝ := 19000

-- The final statement for the savings, where we will prove the above question == answer
theorem savings_calculation (income expenditure savings : ℝ)
  (h_ratio : income_expenditure_ratio income expenditure)
  (h_income : income = income_value) : savings = 11400 :=
by
  sorry

end savings_calculation_l153_15320


namespace differences_occur_10_times_l153_15310

variable (a : Fin 45 → Nat)

theorem differences_occur_10_times 
    (h : ∀ i j : Fin 44, i < j → a i < a j)
    (h_lt_125 : ∀ i : Fin 44, a i < 125) :
    ∃ i : Fin 43, ∃ j : Fin 43, i ≠ j ∧ (a (i + 1) - a i) = (a (j + 1) - a j) ∧ 
    (∃ k : Nat, k ≥ 10 ∧ (a (j + 1) - a j) = (a (k + 1) - a k)) :=
sorry

end differences_occur_10_times_l153_15310


namespace first_car_speed_l153_15367

theorem first_car_speed
  (highway_length : ℝ)
  (second_car_speed : ℝ)
  (meeting_time : ℝ)
  (D1 D2 : ℝ) :
  highway_length = 45 → second_car_speed = 16 → meeting_time = 1.5 → D2 = second_car_speed * meeting_time → D1 + D2 = highway_length → D1 = 14 * meeting_time :=
by
  intros h_highway h_speed h_time h_D2 h_sum
  sorry

end first_car_speed_l153_15367


namespace composite_has_at_least_three_factors_l153_15325

-- Definition of composite number in terms of its factors
def is_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Theorem stating that a composite number has at least 3 factors
theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : 
  (∃ f1 f2 f3, f1 ∣ n ∧ f2 ∣ n ∧ f3 ∣ n ∧ f1 ≠ 1 ∧ f1 ≠ n ∧ f2 ≠ 1 ∧ f2 ≠ n ∧ f3 ≠ 1 ∧ f3 ≠ n ∧ f1 ≠ f2 ∧ f2 ≠ f3) := 
sorry

end composite_has_at_least_three_factors_l153_15325


namespace problem_solution_l153_15342

theorem problem_solution :
  0.45 * 0.65 + 0.1 * 0.2 = 0.3125 :=
by
  sorry

end problem_solution_l153_15342


namespace children_attended_l153_15347

theorem children_attended (A C : ℕ) (h1 : C = 2 * A) (h2 : A + C = 42) : C = 28 :=
by
  sorry

end children_attended_l153_15347


namespace total_pears_sold_l153_15341

theorem total_pears_sold (sold_morning : ℕ) (sold_afternoon : ℕ) (h_morning : sold_morning = 120) (h_afternoon : sold_afternoon = 240) :
  sold_morning + sold_afternoon = 360 :=
by
  sorry

end total_pears_sold_l153_15341


namespace last_term_of_sequence_l153_15353

theorem last_term_of_sequence (u₀ : ℤ) (diffs : List ℤ) (sum_diffs : ℤ) :
  u₀ = 0 → diffs = [2, 4, -1, 0, -5, -3, 3] → sum_diffs = diffs.sum → 
  u₀ + sum_diffs = 0 := by
  sorry

end last_term_of_sequence_l153_15353


namespace range_of_a_l153_15384

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a ∈ [-1, 3]) := 
by
  sorry

end range_of_a_l153_15384


namespace Jenny_total_wins_l153_15315

theorem Jenny_total_wins :
  let games_against_mark := 10
  let mark_wins := 1
  let mark_losses := games_against_mark - mark_wins
  let games_against_jill := 2 * games_against_mark
  let jill_wins := (75 / 100) * games_against_jill
  let jenny_wins_against_jill := games_against_jill - jill_wins
  mark_losses + jenny_wins_against_jill = 14 :=
by
  sorry

end Jenny_total_wins_l153_15315


namespace find_b_l153_15319

theorem find_b (b : ℤ) (h_quad : ∃ m : ℤ, (x + m)^2 + 20 = x^2 + b * x + 56) (h_pos : b > 0) : b = 12 :=
sorry

end find_b_l153_15319


namespace circumcircle_radius_min_cosA_l153_15388

noncomputable def circumcircle_radius (a b c : ℝ) (A B C : ℝ) :=
  a / (2 * (Real.sin A))

theorem circumcircle_radius_min_cosA
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : Real.sin C + Real.sin B = 4 * Real.sin A)
  (h3 : a^2 + b^2 - 2 * a * b * (Real.cos A) = c^2)
  (h4 : a^2 + c^2 - 2 * a * c * (Real.cos B) = b^2)
  (h5 : b^2 + c^2 - 2 * b * c * (Real.cos C) = a^2) :
  circumcircle_radius a b c A B C = 8 * Real.sqrt 15 / 15 :=
sorry

end circumcircle_radius_min_cosA_l153_15388


namespace find_a_value_l153_15340

theorem find_a_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (max (a^1) (a^2) + min (a^1) (a^2)) = 12) : a = 3 :=
by
  sorry

end find_a_value_l153_15340


namespace min_area_is_fifteen_l153_15377

variable (L W : ℕ)

def minimum_possible_area (L W : ℕ) : ℕ :=
  if L = 3 ∧ W = 5 then 3 * 5 else 0

theorem min_area_is_fifteen (hL : 3 ≤ L ∧ L ≤ 5) (hW : 5 ≤ W ∧ W ≤ 7) : 
  minimum_possible_area 3 5 = 15 := 
by
  sorry

end min_area_is_fifteen_l153_15377


namespace prices_of_books_book_purchasing_plans_l153_15302

-- Define the conditions
def cost_eq1 (x y : ℕ): Prop := 20 * x + 40 * y = 1520
def cost_eq2 (x y : ℕ): Prop := 20 * x - 20 * y = 440
def plan_conditions (x y : ℕ): Prop := (20 + y - x = 20) ∧ (x + y + 20 ≥ 72) ∧ (40 * x + 18 * (y + 20) ≤ 2000)

-- Prove price of each book
theorem prices_of_books : 
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧ x = 40 ∧ y = 18 :=
by {
  sorry
}

-- Prove possible book purchasing plans
theorem book_purchasing_plans : 
  ∃ (x : ℕ), plan_conditions x (x + 20) ∧ 
  (x = 26 ∧ x + 20 = 46 ∨ 
   x = 27 ∧ x + 20 = 47 ∨ 
   x = 28 ∧ x + 20 = 48) :=
by {
  sorry
}

end prices_of_books_book_purchasing_plans_l153_15302


namespace nora_muffin_price_l153_15308

theorem nora_muffin_price
  (cases : ℕ)
  (packs_per_case : ℕ)
  (muffins_per_pack : ℕ)
  (total_money : ℕ)
  (total_cases : ℕ)
  (h1 : total_money = 120)
  (h2 : packs_per_case = 3)
  (h3 : muffins_per_pack = 4)
  (h4 : total_cases = 5) :
  (total_money / (total_cases * packs_per_case * muffins_per_pack) = 2) :=
by
  sorry

end nora_muffin_price_l153_15308


namespace xyz_value_l153_15370

variables {x y z : ℂ}

theorem xyz_value (h1 : x * y + 2 * y = -8)
                  (h2 : y * z + 2 * z = -8)
                  (h3 : z * x + 2 * x = -8) :
  x * y * z = 32 :=
by
  sorry

end xyz_value_l153_15370


namespace exists_pairs_of_stops_l153_15389

def problem := ∃ (A1 B1 A2 B2 : Fin 6) (h1 : A1 < B1) (h2 : A2 < B2),
  (A1 ≠ A2 ∧ A1 ≠ B2 ∧ B1 ≠ A2 ∧ B1 ≠ B2) ∧
  ¬(∃ (a b : Fin 6), A1 = a ∧ B1 = b ∧ A2 = a ∧ B2 = b) -- such that no passenger boards at A1 and alights at B1
                                                              -- and no passenger boards at A2 and alights at B2.

theorem exists_pairs_of_stops (n : ℕ) (stops : Fin n) (max_passengers : ℕ) 
  (h : n = 6 ∧ max_passengers = 5 ∧ 
  ∀ (a b : Fin n), a < b → a < stops ∧ b < stops) : problem :=
sorry

end exists_pairs_of_stops_l153_15389


namespace num_pens_l153_15366

theorem num_pens (pencils : ℕ) (students : ℕ) (pens : ℕ)
  (h_pencils : pencils = 520)
  (h_students : students = 40)
  (h_div : pencils % students = 0)
  (h_pens_per_student : pens = (pencils / students) * students) :
  pens = 520 := by
  sorry

end num_pens_l153_15366


namespace find_k_l153_15350

theorem find_k 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 4 * x + 2)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 8)
  (intersect : ∃ x y : ℝ, x = -2 ∧ y = -6 ∧ 4 * x + 2 = y ∧ k * x - 8 = y) :
  k = -1 := 
sorry

end find_k_l153_15350


namespace correct_polynomials_are_l153_15331

noncomputable def polynomial_solution (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p.eval (x^2) = (p.eval x) * (p.eval (x - 1))

theorem correct_polynomials_are (p : Polynomial ℝ) :
  polynomial_solution p ↔ ∃ n : ℕ, p = (Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ)) ^ n :=
by
  sorry

end correct_polynomials_are_l153_15331


namespace n_fraction_sum_l153_15376

theorem n_fraction_sum {n : ℝ} {lst : List ℝ} (h_len : lst.length = 21) 
(h_mem : n ∈ lst) 
(h_avg : n = 4 * (lst.erase n).sum / 20) :
  n = (lst.sum) / 6 :=
by
  sorry

end n_fraction_sum_l153_15376


namespace find_number_of_As_l153_15334

variables (M L S : ℕ)

def number_of_As (M L S : ℕ) : Prop :=
  M + L = 23 ∧ S + M = 18 ∧ S + L = 15

theorem find_number_of_As (M L S : ℕ) (h : number_of_As M L S) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end find_number_of_As_l153_15334


namespace gcd_le_sqrt_sum_l153_15307

theorem gcd_le_sqrt_sum {a b : ℕ} (h : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  ↑(Nat.gcd a b) ≤ Real.sqrt (a + b) := sorry

end gcd_le_sqrt_sum_l153_15307


namespace parabola_symmetry_l153_15344

theorem parabola_symmetry (a h m : ℝ) (A_on_parabola : 4 = a * (-1 - 3)^2 + h) (B_on_parabola : 4 = a * (m - 3)^2 + h) : 
  m = 7 :=
by 
  sorry

end parabola_symmetry_l153_15344


namespace medical_team_selection_l153_15359

theorem medical_team_selection : 
  let male_doctors := 6
  let female_doctors := 5
  let choose_male := Nat.choose male_doctors 2
  let choose_female := Nat.choose female_doctors 1
  choose_male * choose_female = 75 := 
by 
  sorry

end medical_team_selection_l153_15359


namespace smallest_M_conditions_l153_15374

theorem smallest_M_conditions :
  ∃ M : ℕ, M > 0 ∧
  ((∃ k₁, M = 8 * k₁) ∨ (∃ k₂, M + 2 = 8 * k₂) ∨ (∃ k₃, M + 4 = 8 * k₃)) ∧
  ((∃ k₄, M = 9 * k₄) ∨ (∃ k₅, M + 2 = 9 * k₅) ∨ (∃ k₆, M + 4 = 9 * k₆)) ∧
  ((∃ k₇, M = 25 * k₇) ∨ (∃ k₈, M + 2 = 25 * k₈) ∨ (∃ k₉, M + 4 = 25 * k₉)) ∧
  M = 100 :=
sorry

end smallest_M_conditions_l153_15374


namespace minimum_value_l153_15355

theorem minimum_value {a b c : ℝ} (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * b * c = 1 / 2) :
  ∃ x, x = a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2 ∧ x = 13.5 :=
sorry

end minimum_value_l153_15355


namespace foreign_students_next_semester_l153_15335

theorem foreign_students_next_semester (total_students : ℕ) (percent_foreign : ℝ) (new_foreign_students : ℕ) 
  (h_total : total_students = 1800) (h_percent : percent_foreign = 0.30) (h_new : new_foreign_students = 200) : 
  (0.30 * 1800 + 200 : ℝ) = 740 := by
  sorry

end foreign_students_next_semester_l153_15335


namespace total_books_bought_l153_15332

-- Let x be the number of math books and y be the number of history books
variables (x y : ℕ)

-- Conditions
def math_book_cost := 4
def history_book_cost := 5
def total_price := 368
def num_math_books := 32

-- The total number of books bought is the sum of the number of math books and history books, which should result in 80
theorem total_books_bought : 
  y * history_book_cost + num_math_books * math_book_cost = total_price → 
  x = num_math_books → 
  x + y = 80 :=
by
  sorry

end total_books_bought_l153_15332


namespace sum_of_cubes_l153_15354

variable (a b c : ℝ)

theorem sum_of_cubes (h1 : a^2 + 3 * b = 2) (h2 : b^2 + 5 * c = 3) (h3 : c^2 + 7 * a = 6) :
  a^3 + b^3 + c^3 = -0.875 :=
by
  sorry

end sum_of_cubes_l153_15354


namespace product_ends_in_36_l153_15373

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  ((10 * a + 6) * (10 * b + 6)) % 100 = 36 ↔ (a + b = 0 ∨ a + b = 5 ∨ a + b = 10 ∨ a + b = 15) :=
by
  sorry

end product_ends_in_36_l153_15373


namespace square_plot_area_l153_15349

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (s : ℝ) (A : ℝ)
  (h1 : price_per_foot = 58)
  (h2 : total_cost = 1160)
  (h3 : total_cost = 4 * s * price_per_foot)
  (h4 : A = s * s) :
  A = 25 := by
  sorry

end square_plot_area_l153_15349


namespace solve_for_x_l153_15301

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 3029) : x = 200.4 :=
by
  sorry

end solve_for_x_l153_15301


namespace real_solutions_of_equation_l153_15396

theorem real_solutions_of_equation : 
  ∃! x₁ x₂ : ℝ, (3 * x₁^2 - 10 * x₁ + 7 = 0) ∧ (3 * x₂^2 - 10 * x₂ + 7 = 0) ∧ x₁ ≠ x₂ :=
sorry

end real_solutions_of_equation_l153_15396


namespace solve_equation1_solve_equation2_l153_15304

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  2 * x^2 = 3 * (2 * x + 1)

-- Define the solution set for the first equation
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2

-- Prove that the solutions for the first equation are correct
theorem solve_equation1 (x : ℝ) : equation1 x ↔ solution1 x :=
by
  sorry

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  3 * x * (x + 2) = 4 * x + 8

-- Define the solution set for the second equation
def solution2 (x : ℝ) : Prop :=
  x = -2 ∨ x = 4 / 3

-- Prove that the solutions for the second equation are correct
theorem solve_equation2 (x : ℝ) : equation2 x ↔ solution2 x :=
by
  sorry

end solve_equation1_solve_equation2_l153_15304


namespace scientific_notation_123000_l153_15380

theorem scientific_notation_123000 : (123000 : ℝ) = 1.23 * 10^5 := by
  sorry

end scientific_notation_123000_l153_15380


namespace distinct_c_values_l153_15339

theorem distinct_c_values (c r s t : ℂ) 
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_unity : ∃ ω : ℂ, ω^3 = 1 ∧ r = 1 ∧ s = ω ∧ t = ω^2)
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)) :
  ∃ (c_vals : Finset ℂ), c_vals.card = 3 ∧ ∀ (c' : ℂ), c' ∈ c_vals → c'^3 = 1 :=
by
  sorry

end distinct_c_values_l153_15339


namespace inequality_abc_l153_15361

theorem inequality_abc (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b * c = 8) :
  (a^2 / Real.sqrt ((1 + a^3) * (1 + b^3))) + (b^2 / Real.sqrt ((1 + b^3) * (1 + c^3))) +
  (c^2 / Real.sqrt ((1 + c^3) * (1 + a^3))) ≥ 4 / 3 :=
sorry

end inequality_abc_l153_15361


namespace cube_volume_l153_15329

theorem cube_volume (SA : ℕ) (h : SA = 294) : 
  ∃ V : ℕ, V = 343 := 
by
  sorry

end cube_volume_l153_15329


namespace swimming_pool_min_cost_l153_15316

theorem swimming_pool_min_cost (a : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x : ℝ), x > 0 → y = 2400 * a + 6 * (x + 1600 / x) * a) →
  (∃ (x : ℝ), x > 0 ∧ y = 2880 * a) :=
by
  sorry

end swimming_pool_min_cost_l153_15316


namespace max_vertices_of_divided_triangle_l153_15383

theorem max_vertices_of_divided_triangle (n : ℕ) (h : n ≥ 1) : 
  (∀ t : ℕ, t = 1000 → exists T : ℕ, T = (n + 2)) :=
by sorry

end max_vertices_of_divided_triangle_l153_15383


namespace age_of_cat_l153_15382

variables (cat_age rabbit_age dog_age : ℕ)

-- Conditions
def condition1 : Prop := rabbit_age = cat_age / 2
def condition2 : Prop := dog_age = 3 * rabbit_age
def condition3 : Prop := dog_age = 12

-- Question
def question (cat_age : ℕ) : Prop := cat_age = 8

theorem age_of_cat (h1 : condition1 cat_age rabbit_age) (h2 : condition2 rabbit_age dog_age) (h3 : condition3 dog_age) : question cat_age :=
by
  sorry

end age_of_cat_l153_15382


namespace find_a10_l153_15372

-- Conditions
variables (S : ℕ → ℕ) (a : ℕ → ℕ)
variables (hS9 : S 9 = 81) (ha2 : a 2 = 3)

-- Arithmetic sequence sum definition
def arithmetic_sequence_sum (n : ℕ) (a1 : ℕ) (d : ℕ) :=
  n * (2 * a1 + (n - 1) * d) / 2

-- a_n formula definition
def a_n (n a1 d : ℕ) := a1 + (n - 1) * d

-- Proof statement
theorem find_a10 (a1 d : ℕ) (hS9' : 9 * (2 * a1 + 8 * d) / 2 = 81) (ha2' : a1 + d = 3) :
  a 10 = a1 + 9 * d :=
sorry

end find_a10_l153_15372


namespace value_of_x_squared_plus_y_squared_l153_15371

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l153_15371


namespace problem_statement_l153_15378

-- Define a set S
variable {S : Type*}

-- Define the binary operation on S
variable (mul : S → S → S)

-- Assume the given condition: (a * b) * a = b for all a, b in S
axiom given_condition : ∀ (a b : S), (mul (mul a b) a) = b

-- Prove that a * (b * a) = b for all a, b in S
theorem problem_statement : ∀ (a b : S), mul a (mul b a) = b :=
by
  sorry

end problem_statement_l153_15378


namespace widow_share_l153_15363

theorem widow_share (w d s : ℝ) (h_sum : w + 5 * s + 4 * d = 8000)
  (h1 : d = 2 * w)
  (h2 : s = 3 * d) :
  w = 8000 / 39 := by
sorry

end widow_share_l153_15363


namespace jeremy_oranges_l153_15311

theorem jeremy_oranges (M : ℕ) (h : M + 3 * M + 70 = 470) : M = 100 := 
by
  sorry

end jeremy_oranges_l153_15311


namespace circles_intersect_and_common_chord_l153_15330

open Real

def circle1 (x y : ℝ) := x^2 + y^2 - 6 * x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6 = 0

theorem circles_intersect_and_common_chord :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) ∧ (∀ x y : ℝ, circle1 x y → circle2 x y → 3 * x - 2 * y = 0) :=
by
  sorry

end circles_intersect_and_common_chord_l153_15330


namespace exists_k_l153_15365

theorem exists_k (m n : ℕ) : ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end exists_k_l153_15365


namespace carol_mike_equal_savings_weeks_l153_15385

theorem carol_mike_equal_savings_weeks :
  ∃ x : ℕ, (60 + 9 * x = 90 + 3 * x) ↔ x = 5 := 
by
  sorry

end carol_mike_equal_savings_weeks_l153_15385


namespace Sue_necklace_total_beads_l153_15303

theorem Sue_necklace_total_beads :
  ∃ (purple blue green red total : ℕ),
  purple = 7 ∧
  blue = 2 * purple ∧
  green = blue + 11 ∧
  (red : ℕ) = green / 2 ∧
  total = purple + blue + green + red ∧
  total % 2 = 0 ∧
  total = 58 := by
    sorry

end Sue_necklace_total_beads_l153_15303


namespace find_x_plus_y_l153_15306

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2011 + Real.pi :=
sorry

end find_x_plus_y_l153_15306


namespace main_theorem_l153_15317

-- Definitions based on conditions
variables (A P H M E C : ℕ) 
-- Thickness of an algebra book
def x := 1
-- Thickness of a history book (twice that of algebra)
def history_thickness := 2 * x
-- Length of shelf filled by books
def z := A * x

-- Condition equations based on shelf length equivalences
def equation1 := A = P
def equation2 := 2 * H * x = M * x
def equation3 := E * x + C * history_thickness = z

-- Prove the relationship
theorem main_theorem : C = (M * (A - E)) / (2 * A * H) :=
by
  sorry

end main_theorem_l153_15317


namespace inv_matrix_eq_l153_15369

variable (a : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 1, a])
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![a, -3; -1, a])

theorem inv_matrix_eq : (A⁻¹ = A_inv) → (a = 2) := 
by 
  sorry

end inv_matrix_eq_l153_15369


namespace count_whole_numbers_between_4_and_18_l153_15395

theorem count_whole_numbers_between_4_and_18 :
  ∀ (x : ℕ), 4 < x ∧ x < 18 ↔ ∃ n : ℕ, n = 13 :=
by sorry

end count_whole_numbers_between_4_and_18_l153_15395


namespace a_is_5_if_extreme_at_neg3_l153_15333

-- Define the function f with parameter a
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

-- Define the given condition that f reaches an extreme value at x = -3
def reaches_extreme_at (a : ℝ) : Prop := f_prime a (-3) = 0

-- Prove that a = 5 if f reaches an extreme value at x = -3
theorem a_is_5_if_extreme_at_neg3 : ∀ a : ℝ, reaches_extreme_at a → a = 5 :=
by
  intros a h
  -- Proof omitted
  sorry

end a_is_5_if_extreme_at_neg3_l153_15333


namespace cards_probability_comparison_l153_15346

noncomputable def probability_case_a : ℚ :=
  (Nat.choose 13 10) * (Nat.choose 39 3) / Nat.choose 52 13

noncomputable def probability_case_b : ℚ :=
  4 ^ 13 / Nat.choose 52 13

theorem cards_probability_comparison :
  probability_case_b > probability_case_a :=
  sorry

end cards_probability_comparison_l153_15346


namespace largest_angle_in_triangle_l153_15392

theorem largest_angle_in_triangle (a b c : ℝ)
  (h1 : a + b = (4 / 3) * 90)
  (h2 : b = a + 36)
  (h3 : a + b + c = 180) :
  max a (max b c) = 78 :=
sorry

end largest_angle_in_triangle_l153_15392


namespace correct_proposition_four_l153_15393

universe u

-- Definitions
variable {Point : Type u} (A B : Point) (a α : Set Point)
variable (h5 : A ∉ α)
variable (h6 : a ⊂ α)

-- The statement to be proved
theorem correct_proposition_four : A ∉ a :=
sorry

end correct_proposition_four_l153_15393


namespace find_first_dimension_l153_15394

variable (w h cost_per_sqft total_cost : ℕ)

def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

def insulation_cost (A cost_per_sqft : ℕ) : ℕ := A * cost_per_sqft

theorem find_first_dimension 
  (w := 7) (h := 2) (cost_per_sqft := 20) (total_cost := 1640) : 
  (∃ l : ℕ, insulation_cost (surface_area l w h) cost_per_sqft = total_cost) → 
  l = 3 := 
sorry

end find_first_dimension_l153_15394


namespace jade_transactions_l153_15387

theorem jade_transactions 
    (mabel_transactions : ℕ)
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = cal_transactions + 14) : 
    jade_transactions = 80 :=
sorry

end jade_transactions_l153_15387


namespace oliver_cycling_distance_l153_15318

/-- Oliver has a training loop for his weekend cycling. He starts by cycling due north for 3 miles. 
  Then he cycles northeast, making a 30° angle with the north for 2 miles, followed by cycling 
  southeast, making a 60° angle with the south for 2 miles. He completes his loop by cycling 
  directly back to the starting point. Prove that the distance of this final segment of his ride 
  is √(11 + 6√3) miles. -/
theorem oliver_cycling_distance :
  let north_displacement : ℝ := 3
  let northeast_displacement : ℝ := 2
  let northeast_angle : ℝ := 30
  let southeast_displacement : ℝ := 2
  let southeast_angle : ℝ := 60
  let north_northeast : ℝ := northeast_displacement * Real.cos (northeast_angle * Real.pi / 180)
  let east_northeast : ℝ := northeast_displacement * Real.sin (northeast_angle * Real.pi / 180)
  let south_southeast : ℝ := southeast_displacement * Real.cos (southeast_angle * Real.pi / 180)
  let east_southeast : ℝ := southeast_displacement * Real.sin (southeast_angle * Real.pi / 180)
  let total_north : ℝ := north_displacement + north_northeast - south_southeast
  let total_east : ℝ := east_northeast + east_southeast
  total_north = 2 + Real.sqrt 3 ∧ total_east = 1 + Real.sqrt 3
  → Real.sqrt (total_north^2 + total_east^2) = Real.sqrt (11 + 6 * Real.sqrt 3) :=
by
  sorry

end oliver_cycling_distance_l153_15318
