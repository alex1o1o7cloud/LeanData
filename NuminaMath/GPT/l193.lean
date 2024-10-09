import Mathlib

namespace first_player_has_winning_strategy_l193_19385

-- Define the initial heap sizes and rules of the game.
def initial_heaps : List Nat := [38, 45, 61, 70]

-- Define a function that checks using the rules whether the first player has a winning strategy given the initial heap sizes.
def first_player_wins : Bool :=
  -- placeholder for the actual winning strategy check logic
  sorry

-- Theorem statement referring to the equivalency proof problem where player one is established to have the winning strategy.
theorem first_player_has_winning_strategy : first_player_wins = true :=
  sorry

end first_player_has_winning_strategy_l193_19385


namespace cost_500_pencils_is_25_dollars_l193_19360

def cost_of_500_pencils (cost_per_pencil : ℕ) (pencils : ℕ) (cents_per_dollar : ℕ) : ℕ :=
  (cost_per_pencil * pencils) / cents_per_dollar

theorem cost_500_pencils_is_25_dollars : cost_of_500_pencils 5 500 100 = 25 := by
  sorry

end cost_500_pencils_is_25_dollars_l193_19360


namespace pos_int_divides_l193_19336

theorem pos_int_divides (n : ℕ) (h₀ : 0 < n) (h₁ : (n - 1) ∣ (n^3 + 4)) : n = 2 ∨ n = 6 :=
by sorry

end pos_int_divides_l193_19336


namespace ratio_minutes_l193_19308

theorem ratio_minutes (x : ℝ) : 
  (12 / 8) = (6 / (x * 60)) → x = 1 / 15 :=
by
  sorry

end ratio_minutes_l193_19308


namespace seeking_the_cause_from_the_result_means_sufficient_condition_l193_19377

-- Define the necessary entities for the conditions
inductive Condition
| Necessary
| Sufficient
| NecessaryAndSufficient
| NecessaryOrSufficient

-- Define the statement of the proof problem
theorem seeking_the_cause_from_the_result_means_sufficient_condition :
  (seeking_the_cause_from_the_result : Condition) = Condition.Sufficient :=
sorry

end seeking_the_cause_from_the_result_means_sufficient_condition_l193_19377


namespace average_alligators_l193_19332

theorem average_alligators (t s n : ℕ) (h1 : t = 50) (h2 : s = 20) (h3 : n = 3) :
  (t - s) / n = 10 :=
by 
  sorry

end average_alligators_l193_19332


namespace smallest_n_satisfying_conditions_l193_19391

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ x : ℕ, 3 * n = x^4) ∧ (∃ y : ℕ, 2 * n = y^5) ∧ n = 432 :=
by
  sorry

end smallest_n_satisfying_conditions_l193_19391


namespace percentage_of_green_ducks_l193_19309

def total_ducks := 100
def green_ducks_smaller_pond := 9
def green_ducks_larger_pond := 22
def total_green_ducks := green_ducks_smaller_pond + green_ducks_larger_pond

theorem percentage_of_green_ducks :
  (total_green_ducks / total_ducks) * 100 = 31 :=
by
  sorry

end percentage_of_green_ducks_l193_19309


namespace negative_number_from_operations_l193_19357

theorem negative_number_from_operations :
  (∀ (a b : Int), a + b < 0 → a = -1 ∧ b = -3) ∧
  (∀ (a b : Int), a - b < 0 → a = 1 ∧ b = 4) ∧
  (∀ (a b : Int), a * b > 0 → a = 3 ∧ b = -2) ∧
  (∀ (a b : Int), a / b = 0 → a = 0 ∧ b = -7) :=
by
  sorry

end negative_number_from_operations_l193_19357


namespace joe_money_left_l193_19310

theorem joe_money_left
  (joe_savings : ℕ := 6000)
  (flight_cost : ℕ := 1200)
  (hotel_cost : ℕ := 800)
  (food_cost : ℕ := 3000) :
  joe_savings - (flight_cost + hotel_cost + food_cost) = 1000 :=
by
  sorry

end joe_money_left_l193_19310


namespace r_values_if_polynomial_divisible_l193_19334

noncomputable
def find_r_iff_divisible (r : ℝ) : Prop :=
  (10 * (r^2 * (1 - 2*r))) = -6 ∧ 
  (2 * r + (1 - 2*r)) = 1 ∧ 
  (r^2 + 2 * r * (1 - 2*r)) = -5.2

theorem r_values_if_polynomial_divisible (r : ℝ) :
  (find_r_iff_divisible r) ↔ 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := 
by
  sorry

end r_values_if_polynomial_divisible_l193_19334


namespace range_of_a_l193_19306

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l193_19306


namespace find_constants_l193_19324

theorem find_constants :
  ∃ A B C D : ℚ,
    (∀ x : ℚ,
      x ≠ 2 → x ≠ 3 → x ≠ 5 → x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1)) ∧
  A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 :=
by
  sorry

end find_constants_l193_19324


namespace perfect_apples_count_l193_19378

-- Definitions (conditions)
def total_apples := 30
def too_small_fraction := (1 : ℚ) / 6
def not_ripe_fraction := (1 : ℚ) / 3
def too_small_apples := (too_small_fraction * total_apples : ℚ)
def not_ripe_apples := (not_ripe_fraction * total_apples : ℚ)

-- Statement of the theorem (proof problem)
theorem perfect_apples_count : total_apples - too_small_apples - not_ripe_apples = 15 := by
  sorry

end perfect_apples_count_l193_19378


namespace sum_of_roots_of_quadratic_eqn_l193_19351

theorem sum_of_roots_of_quadratic_eqn (A B : ℝ) 
  (h₁ : 3 * A ^ 2 - 9 * A + 6 = 0)
  (h₂ : 3 * B ^ 2 - 9 * B + 6 = 0)
  (h_distinct : A ≠ B):
  A + B = 3 := by
  sorry

end sum_of_roots_of_quadratic_eqn_l193_19351


namespace myrtle_eggs_count_l193_19367

-- Definition for daily egg production
def daily_eggs : ℕ := 3 * 3

-- Definition for the number of days Myrtle is gone
def days_gone : ℕ := 7

-- Definition for total eggs laid
def total_eggs : ℕ := daily_eggs * days_gone

-- Definition for eggs taken by neighbor
def eggs_taken_by_neighbor : ℕ := 12

-- Definition for eggs remaining after neighbor takes some
def eggs_after_neighbor : ℕ := total_eggs - eggs_taken_by_neighbor

-- Definition for eggs dropped by Myrtle
def eggs_dropped_by_myrtle : ℕ := 5

-- Definition for total remaining eggs Myrtle has
def eggs_remaining : ℕ := eggs_after_neighbor - eggs_dropped_by_myrtle

-- Theorem statement
theorem myrtle_eggs_count : eggs_remaining = 46 := by
  sorry

end myrtle_eggs_count_l193_19367


namespace total_amount_spent_l193_19327

variable (your_spending : ℝ) (friend_spending : ℝ)
variable (h1 : friend_spending = your_spending + 3) (h2 : friend_spending = 10)

theorem total_amount_spent : your_spending + friend_spending = 17 :=
by sorry

end total_amount_spent_l193_19327


namespace possible_rectangle_configurations_l193_19386

-- Define the conditions as variables
variables (m n : ℕ)
-- Define the number of segments
def segments (m n : ℕ) : ℕ := 2 * m * n + m + n

theorem possible_rectangle_configurations : 
  (segments m n = 1997) → (m = 2 ∧ n = 399) ∨ (m = 8 ∧ n = 117) ∨ (m = 23 ∧ n = 42) :=
by
  sorry

end possible_rectangle_configurations_l193_19386


namespace repair_cost_total_l193_19322

-- Define the inputs
def labor_cost_rate : ℤ := 75
def labor_hours : ℤ := 16
def part_cost : ℤ := 1200

-- Define the required computation and proof statement
def total_repair_cost : ℤ :=
  let labor_cost := labor_cost_rate * labor_hours
  labor_cost + part_cost

theorem repair_cost_total : total_repair_cost = 2400 := by
  -- Proof would go here
  sorry

end repair_cost_total_l193_19322


namespace derivative_of_curve_tangent_line_at_one_l193_19317

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^3 + 5 * x^2 + 3 * x

-- Part 1: Prove the derivative of the curve
theorem derivative_of_curve (x : ℝ) :
  deriv curve x = 3 * x^2 + 10 * x + 3 :=
sorry

-- Part 2: Prove the equation of the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a = 16 ∧ b = -1 ∧ c = -7 ∧
  ∀ (x y : ℝ), curve 1 = 9 → y - 9 = 16 * (x - 1) → a * x + b * y + c = 0 :=
sorry

end derivative_of_curve_tangent_line_at_one_l193_19317


namespace arithmetic_seq_a10_l193_19395

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a10 (h_arith : arithmetic_sequence a) (h2 : a 3 = 5) (h5 : a 6 = 11) : a 10 = 19 := by
  sorry

end arithmetic_seq_a10_l193_19395


namespace unique_n_for_solutions_l193_19331

theorem unique_n_for_solutions :
  ∃! (n : ℕ), (∀ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (3 * x + 3 * y + 2 * z = n)) → 
  ((∃ (s : ℕ), s = 10) ∧ (n = 17)) :=
sorry

end unique_n_for_solutions_l193_19331


namespace increase_by_percentage_l193_19321

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l193_19321


namespace find_number_l193_19325

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l193_19325


namespace upper_bound_y_l193_19315

theorem upper_bound_y 
  (U : ℤ) 
  (x y : ℤ)
  (h1 : 3 < x ∧ x < 6) 
  (h2 : 6 < y ∧ y < U) 
  (h3 : y - x = 4) : 
  U = 10 := 
sorry

end upper_bound_y_l193_19315


namespace focal_chord_length_perpendicular_l193_19365

theorem focal_chord_length_perpendicular (x1 y1 x2 y2 : ℝ)
  (h_parabola : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_perpendicular : x1 = x2) :
  abs (y1 - y2) = 4 :=
by sorry

end focal_chord_length_perpendicular_l193_19365


namespace city_G_has_highest_percentage_increase_l193_19390

-- Define the population data as constants.
def population_1990_F : ℕ := 50
def population_2000_F : ℕ := 60
def population_1990_G : ℕ := 60
def population_2000_G : ℕ := 80
def population_1990_H : ℕ := 90
def population_2000_H : ℕ := 110
def population_1990_I : ℕ := 120
def population_2000_I : ℕ := 150
def population_1990_J : ℕ := 150
def population_2000_J : ℕ := 190

-- Define the function that calculates the percentage increase.
def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ :=
  (pop_2000 : ℚ) / (pop_1990 : ℚ)

-- Calculate the percentage increases for each city.
def percentage_increase_F := percentage_increase population_1990_F population_2000_F
def percentage_increase_G := percentage_increase population_1990_G population_2000_G
def percentage_increase_H := percentage_increase population_1990_H population_2000_H
def percentage_increase_I := percentage_increase population_1990_I population_2000_I
def percentage_increase_J := percentage_increase population_1990_J population_2000_J

-- Prove that City G has the greatest percentage increase.
theorem city_G_has_highest_percentage_increase :
  percentage_increase_G > percentage_increase_F ∧ 
  percentage_increase_G > percentage_increase_H ∧
  percentage_increase_G > percentage_increase_I ∧
  percentage_increase_G > percentage_increase_J :=
by sorry

end city_G_has_highest_percentage_increase_l193_19390


namespace girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l193_19338

namespace PhotoArrangement

/-- There are 4 boys and 3 girls. -/
def boys : ℕ := 4
def girls : ℕ := 3

/-- Number of ways to arrange given conditions -/
def arrangementsWithGirlsAtEnds : ℕ := 720
def arrangementsWithNoGirlsNextToEachOther : ℕ := 1440
def arrangementsWithGirlAtoRightOfGirlB : ℕ := 2520

-- Problem 1: If there are girls at both ends
theorem girls_at_ends (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlsAtEnds := by
  sorry

-- Problem 2: If no two girls are standing next to each other
theorem no_girls_next_to_each_other (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithNoGirlsNextToEachOther := by
  sorry

-- Problem 3: If girl A must be to the right of girl B
theorem girl_A_right_of_girl_B (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlAtoRightOfGirlB := by
  sorry

end PhotoArrangement

end girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l193_19338


namespace cubic_function_decreasing_l193_19370

theorem cubic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → (a ≤ 0) := 
by 
  sorry

end cubic_function_decreasing_l193_19370


namespace johns_shirt_percentage_increase_l193_19374

variable (P S : ℕ)

theorem johns_shirt_percentage_increase :
  P = 50 →
  S + P = 130 →
  ((S - P) * 100 / P) = 60 := by
  sorry

end johns_shirt_percentage_increase_l193_19374


namespace evaluate_expression_l193_19384

noncomputable def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem evaluate_expression (A B C D : ℝ) (h1 : g A B C D 2 = 5) (h2 : g A B C D (-1) = -8) (h3 : g A B C D 0 = 2) :
  -12 * A + 6 * B - 3 * C + D = 27.5 :=
by
  sorry

end evaluate_expression_l193_19384


namespace mean_cost_of_diesel_l193_19379

-- Define the diesel rates and the number of years.
def dieselRates : List ℝ := [1.2, 1.3, 1.8, 2.1]
def years : ℕ := 4

-- Define the mean calculation and the proof requirement.
theorem mean_cost_of_diesel (h₁ : dieselRates = [1.2, 1.3, 1.8, 2.1]) 
                               (h₂ : years = 4) : 
  (dieselRates.sum / years) = 1.6 :=
by
  sorry

end mean_cost_of_diesel_l193_19379


namespace compare_logs_l193_19392

open Real

noncomputable def a := log 6 / log 3
noncomputable def b := 1 / log 5
noncomputable def c := log 14 / log 7

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l193_19392


namespace second_sweet_red_probability_l193_19388

theorem second_sweet_red_probability (x y : ℕ) : 
  (y / (x + y : ℝ)) = y / (x + y + 10) * x / (x + y) + (y + 10) / (x + y + 10) * y / (x + y) :=
by
  sorry

end second_sweet_red_probability_l193_19388


namespace area_of_rhombus_l193_19397

noncomputable def diagonal_length_1 : ℕ := 30
noncomputable def diagonal_length_2 : ℕ := 14

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = diagonal_length_1) (h2 : d2 = diagonal_length_2) : 
  (d1 * d2) / 2 = 210 :=
by 
  rw [h1, h2]
  sorry

end area_of_rhombus_l193_19397


namespace A_and_B_finish_work_together_in_12_days_l193_19345

theorem A_and_B_finish_work_together_in_12_days 
  (T_B : ℕ) 
  (T_A : ℕ)
  (h1 : T_B = 18) 
  (h2 : T_A = 2 * T_B) : 
  1 / (1 / T_A + 1 / T_B) = 12 := 
by 
  sorry

end A_and_B_finish_work_together_in_12_days_l193_19345


namespace area_ratio_PQR_to_STU_l193_19352

-- Given Conditions
def triangle_PQR_sides (a b c : Nat) : Prop :=
  a = 9 ∧ b = 40 ∧ c = 41

def triangle_STU_sides (x y z : Nat) : Prop :=
  x = 7 ∧ y = 24 ∧ z = 25

-- Theorem Statement (math proof problem)
theorem area_ratio_PQR_to_STU :
  (∃ (a b c x y z : Nat), triangle_PQR_sides a b c ∧ triangle_STU_sides x y z) →
  9 * 40 / (7 * 24) = 15 / 7 :=
by
  intro h
  sorry

end area_ratio_PQR_to_STU_l193_19352


namespace smallest_n_satisfying_equation_l193_19323

theorem smallest_n_satisfying_equation : ∃ (k : ℤ), (∃ (n : ℤ), n > 0 ∧ n % 2 = 1 ∧ (n ^ 3 + 2 * n ^ 2 = k ^ 2) ∧ ∀ m : ℤ, (m > 0 ∧ m < n ∧ m % 2 = 1) → ¬ (∃ j : ℤ, m ^ 3 + 2 * m ^ 2 = j ^ 2)) ∧ k % 2 = 1 :=
sorry

end smallest_n_satisfying_equation_l193_19323


namespace larger_angle_at_3_30_l193_19375

def hour_hand_angle_3_30 : ℝ := 105.0
def minute_hand_angle_3_30 : ℝ := 180.0
def smaller_angle_between_hands : ℝ := abs (minute_hand_angle_3_30 - hour_hand_angle_3_30)
def larger_angle_between_hands : ℝ := 360.0 - smaller_angle_between_hands

theorem larger_angle_at_3_30 :
  larger_angle_between_hands = 285.0 := 
  sorry

end larger_angle_at_3_30_l193_19375


namespace calculate_weight_l193_19373

theorem calculate_weight (W : ℝ) (h : 0.75 * W + 2 = 62) : W = 80 :=
by
  sorry

end calculate_weight_l193_19373


namespace emily_strawberry_harvest_l193_19312

-- Define the dimensions of the garden
def garden_length : ℕ := 10
def garden_width : ℕ := 7

-- Define the planting density
def plants_per_sqft : ℕ := 3

-- Define the yield per plant
def strawberries_per_plant : ℕ := 12

-- Define the expected number of strawberries
def expected_strawberries : ℕ := 2520

-- Theorem statement to prove the total number of strawberries
theorem emily_strawberry_harvest :
  garden_length * garden_width * plants_per_sqft * strawberries_per_plant = expected_strawberries :=
by
  -- Proof goes here (for now, we use sorry to indicate the proof is omitted)
  sorry

end emily_strawberry_harvest_l193_19312


namespace angle_sum_triangle_l193_19326

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l193_19326


namespace fraction_value_l193_19354

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value_l193_19354


namespace homer_second_try_points_l193_19311

theorem homer_second_try_points (x : ℕ) :
  400 + x + 2 * x = 1390 → x = 330 :=
by
  sorry

end homer_second_try_points_l193_19311


namespace book_pairs_count_l193_19301

theorem book_pairs_count :
  let mystery_books := 4
  let science_fiction_books := 4
  let historical_books := 4
  (mystery_books + science_fiction_books + historical_books) = 12 ∧ 
  (mystery_books = 4 ∧ science_fiction_books = 4 ∧ historical_books = 4) →
  let genres := 3
  ∃ pairs, pairs = 48 :=
by
  sorry

end book_pairs_count_l193_19301


namespace greatest_consecutive_integers_sum_36_l193_19307

-- Definition of the sum of N consecutive integers starting from a
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Problem statement in Lean 4
theorem greatest_consecutive_integers_sum_36 (N : ℤ) (h : sum_consecutive_integers (-35) 72 = 36) : N = 72 := by
  sorry

end greatest_consecutive_integers_sum_36_l193_19307


namespace math_problem_l193_19344

theorem math_problem 
  (a1 : (10^4 + 500) = 100500)
  (a2 : (25^4 + 500) = 390625500)
  (a3 : (40^4 + 500) = 256000500)
  (a4 : (55^4 + 500) = 915062500)
  (a5 : (70^4 + 500) = 24010062500)
  (b1 : (5^4 + 500) = 625+500)
  (b2 : (20^4 + 500) = 160000500)
  (b3 : (35^4 + 500) = 150062500)
  (b4 : (50^4 + 500) = 625000500)
  (b5 : (65^4 + 500) = 1785062500) :
  ( (100500 * 390625500 * 256000500 * 915062500 * 24010062500) / (625+500 * 160000500 * 150062500 * 625000500 * 1785062500) = 240) :=
by
  sorry

end math_problem_l193_19344


namespace gcd_of_35_and_number_between_70_and_90_is_7_l193_19396

def number_between_70_and_90 (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 90

def gcd_is_7 (a b : ℕ) : Prop :=
  Nat.gcd a b = 7

theorem gcd_of_35_and_number_between_70_and_90_is_7 : 
  ∃ (n : ℕ), number_between_70_and_90 n ∧ gcd_is_7 35 n ∧ (n = 77 ∨ n = 84) :=
by
  sorry

end gcd_of_35_and_number_between_70_and_90_is_7_l193_19396


namespace sin_two_alpha_sub_pi_eq_24_div_25_l193_19340

noncomputable def pi_div_2 : ℝ := Real.pi / 2

theorem sin_two_alpha_sub_pi_eq_24_div_25
  (α : ℝ) 
  (h1 : pi_div_2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan (α + Real.pi / 4) = -1 / 7) : 
  Real.sin (2 * α - Real.pi) = 24 / 25 := 
sorry

end sin_two_alpha_sub_pi_eq_24_div_25_l193_19340


namespace not_possible_acquaintance_arrangement_l193_19303

-- Definitions and conditions for the problem
def num_people : ℕ := 40
def even_people_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 0 → A ≠ B → true -- A and B have a mutual acquaintance if an even number of people sit between them

def odd_people_not_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 1 → A ≠ B → true -- A and B do not have a mutual acquaintance if an odd number of people sit between them

theorem not_possible_acquaintance_arrangement : ¬ (∀ A B : ℕ, A ≠ B →
  (∀ num_between : ℕ, (num_between % 2 = 0 → even_people_acquainted A B num_between) ∧
  (num_between % 2 = 1 → odd_people_not_acquainted A B num_between))) :=
sorry

end not_possible_acquaintance_arrangement_l193_19303


namespace ellie_total_distance_after_six_steps_l193_19366

-- Define the initial conditions and parameters
def initial_position : ℚ := 0
def target_distance : ℚ := 5
def step_fraction : ℚ := 1 / 4
def steps : ℕ := 6

-- Define the function that calculates the sum of the distances walked
def distance_walked (n : ℕ) : ℚ :=
  let first_term := target_distance * step_fraction
  let common_ratio := 3 / 4
  first_term * (1 - common_ratio^n) / (1 - common_ratio)

-- Define the theorem we want to prove
theorem ellie_total_distance_after_six_steps :
  distance_walked steps = 16835 / 4096 :=
by 
  sorry

end ellie_total_distance_after_six_steps_l193_19366


namespace find_a2_l193_19304

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2_l193_19304


namespace sand_art_l193_19342

theorem sand_art (len_blue_rect : ℕ) (area_blue_rect : ℕ) (side_red_square : ℕ) (sand_per_sq_inch : ℕ) (h1 : len_blue_rect = 7) (h2 : area_blue_rect = 42) (h3 : side_red_square = 5) (h4 : sand_per_sq_inch = 3) :
  (area_blue_rect * sand_per_sq_inch) + (side_red_square * side_red_square * sand_per_sq_inch) = 201 :=
by
  sorry

end sand_art_l193_19342


namespace negation_of_P_is_exists_ge_1_l193_19329

theorem negation_of_P_is_exists_ge_1 :
  let P := ∀ x : ℤ, x < 1
  ¬P ↔ ∃ x : ℤ, x ≥ 1 := by
  sorry

end negation_of_P_is_exists_ge_1_l193_19329


namespace number_of_boxes_sold_on_saturday_l193_19369

theorem number_of_boxes_sold_on_saturday (S : ℝ) 
  (h : S + 1.5 * S + 1.95 * S + 2.34 * S + 2.574 * S = 720) : 
  S = 77 := 
sorry

end number_of_boxes_sold_on_saturday_l193_19369


namespace hotel_cost_l193_19318

theorem hotel_cost (x y : ℕ) (h1 : 3 * x + 6 * y = 1020) (h2 : x + 5 * y = 700) :
  5 * (x + y) = 1100 :=
sorry

end hotel_cost_l193_19318


namespace abs_eq_neg_l193_19364

theorem abs_eq_neg (x : ℝ) (h : |x + 6| = -(x + 6)) : x ≤ -6 :=
by 
  sorry

end abs_eq_neg_l193_19364


namespace merchant_spent_for_belle_l193_19387

def dress_cost (S : ℤ) (H : ℤ) : ℤ := 6 * S + 3 * H
def hat_cost (S : ℤ) (H : ℤ) : ℤ := 3 * S + 5 * H
def belle_cost (S : ℤ) (H : ℤ) : ℤ := S + 2 * H

theorem merchant_spent_for_belle :
  ∃ (S H : ℤ), dress_cost S H = 105 ∧ hat_cost S H = 70 ∧ belle_cost S H = 25 :=
by
  sorry

end merchant_spent_for_belle_l193_19387


namespace beads_problem_l193_19381

noncomputable def number_of_blue_beads (total_beads : ℕ) (beads_with_blue_neighbor : ℕ) (beads_with_green_neighbor : ℕ) : ℕ :=
  let beads_with_both_neighbors := beads_with_blue_neighbor + beads_with_green_neighbor - total_beads
  let beads_with_only_blue_neighbor := beads_with_blue_neighbor - beads_with_both_neighbors
  (2 * beads_with_only_blue_neighbor + beads_with_both_neighbors) / 2

theorem beads_problem : number_of_blue_beads 30 26 20 = 18 := by 
  -- ...
  sorry

end beads_problem_l193_19381


namespace candy_distribution_l193_19313

-- Definition of the problem
def emily_candies : ℕ := 30
def friends : ℕ := 4

-- Lean statement to prove
theorem candy_distribution : emily_candies % friends = 2 :=
by sorry

end candy_distribution_l193_19313


namespace sufficient_but_not_necessary_condition_l193_19376

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x + 1) * (x - 3) < 0 → x > -1 ∧ ((x > -1) → (x + 1) * (x - 3) < 0) = false :=
sorry

end sufficient_but_not_necessary_condition_l193_19376


namespace spring_work_compression_l193_19337

theorem spring_work_compression :
  ∀ (k : ℝ) (F : ℝ) (x : ℝ), 
  (F = 10) → (x = 1 / 100) → (k = F / x) → (W = 5) :=
by
sorry

end spring_work_compression_l193_19337


namespace largest_neg_integer_solution_l193_19383

theorem largest_neg_integer_solution 
  (x : ℤ) 
  (h : 34 * x + 6 ≡ 2 [ZMOD 20]) : 
  x = -6 := 
sorry

end largest_neg_integer_solution_l193_19383


namespace c_impossible_value_l193_19353

theorem c_impossible_value (a b c : ℤ) (h : (∀ x : ℤ, (x + a) * (x + b) = x^2 + c * x - 8)) : c ≠ 4 :=
by
  sorry

end c_impossible_value_l193_19353


namespace line_y2_not_pass_second_quadrant_l193_19399

theorem line_y2_not_pass_second_quadrant {a b : ℝ} (h1 : a < 0) (h2 : b > 0) :
  ¬∃ x : ℝ, x < 0 ∧ bx + a > 0 :=
by
  sorry

end line_y2_not_pass_second_quadrant_l193_19399


namespace sector_area_l193_19393

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) (area : ℝ) 
  (h1 : arc_length = 6) 
  (h2 : central_angle = 2) 
  (h3 : radius = arc_length / central_angle): 
  area = (1 / 2) * arc_length * radius := 
  sorry

end sector_area_l193_19393


namespace component_unqualified_l193_19339

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l193_19339


namespace total_weight_proof_l193_19358
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l193_19358


namespace sum_of_consecutive_even_negative_integers_l193_19302

theorem sum_of_consecutive_even_negative_integers (n m : ℤ) 
  (h1 : n % 2 = 0)
  (h2 : m % 2 = 0)
  (h3 : n < 0)
  (h4 : m < 0)
  (h5 : m = n + 2)
  (h6 : n * m = 2496) : n + m = -102 := 
sorry

end sum_of_consecutive_even_negative_integers_l193_19302


namespace martha_initial_apples_l193_19348

theorem martha_initial_apples :
  ∀ (jane_apples james_apples keep_apples more_to_give initial_apples : ℕ),
    jane_apples = 5 →
    james_apples = jane_apples + 2 →
    keep_apples = 4 →
    more_to_give = 4 →
    initial_apples = jane_apples + james_apples + keep_apples + more_to_give →
    initial_apples = 20 :=
by
  intros jane_apples james_apples keep_apples more_to_give initial_apples
  intro h_jane
  intro h_james
  intro h_keep
  intro h_more
  intro h_initial
  exact sorry

end martha_initial_apples_l193_19348


namespace correct_option_is_A_l193_19335

variable (a b : ℤ)

-- Option A condition
def optionA : Prop := 3 * a^2 * b / b = 3 * a^2

-- Option B condition
def optionB : Prop := a^12 / a^3 = a^4

-- Option C condition
def optionC : Prop := (a + b)^2 = a^2 + b^2

-- Option D condition
def optionD : Prop := (-2 * a^2)^3 = 8 * a^6

theorem correct_option_is_A : 
  optionA a b ∧ ¬optionB a ∧ ¬optionC a b ∧ ¬optionD a :=
by
  sorry

end correct_option_is_A_l193_19335


namespace initial_price_of_phone_l193_19305

theorem initial_price_of_phone
  (initial_price_TV : ℕ)
  (increase_TV_fraction : ℚ)
  (initial_price_phone : ℚ)
  (increase_phone_percentage : ℚ)
  (total_amount : ℚ)
  (h1 : initial_price_TV = 500)
  (h2 : increase_TV_fraction = 2/5)
  (h3 : increase_phone_percentage = 0.40)
  (h4 : total_amount = 1260) :
  initial_price_phone = 400 := by
  sorry

end initial_price_of_phone_l193_19305


namespace min_a_plus_b_l193_19316

-- Given conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Equation of line L passing through point (4,1) with intercepts a and b
def line_eq (a b : ℝ) : Prop := (4 / a) + (1 / b) = 1

-- Proof statement
theorem min_a_plus_b (h : line_eq a b) : a + b ≥ 9 :=
sorry

end min_a_plus_b_l193_19316


namespace sides_of_original_polygon_l193_19398

-- Define the sum of interior angles formula for a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the total sum of angles for the resulting polygon
def sum_of_new_polygon_angles : ℝ := 1980

-- The lean theorem statement to prove
theorem sides_of_original_polygon (n : ℕ) :
    sum_interior_angles n = sum_of_new_polygon_angles →
    n = 13 →
    12 ≤ n+1 ∧ n+1 ≤ 14 :=
by
  intro h1 h2
  sorry

end sides_of_original_polygon_l193_19398


namespace value_of_c_over_b_l193_19349

def is_median (a b c : ℤ) (m : ℤ) : Prop :=
a < b ∧ b < c ∧ m = b

def in_geometric_progression (p q r : ℤ) : Prop :=
∃ k : ℤ, k ≠ 0 ∧ q = p * k ∧ r = q * k

theorem value_of_c_over_b (a b c p q r : ℤ) 
  (h1 : (a + b + c) / 3 = (b / 2))
  (h2 : a * b * c = 0)
  (h3 : a < b ∧ b < c ∧ a = 0)
  (h4 : p < q ∧ q < r ∧ r ≠ 0)
  (h5 : in_geometric_progression p q r)
  (h6 : a^2 + b^2 + c^2 = (p + q + r)^2) : 
  c / b = 2 := 
sorry

end value_of_c_over_b_l193_19349


namespace largest_n_satisfies_l193_19320

noncomputable def sin_plus_cos_bound (n : ℕ) (x : ℝ) : Prop :=
  (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt n)

theorem largest_n_satisfies :
  ∃ (n : ℕ), (∀ x : ℝ, sin_plus_cos_bound n x) ∧
  ∀ m : ℕ, (∀ x : ℝ, sin_plus_cos_bound m x) → m ≤ 2 := 
sorry

end largest_n_satisfies_l193_19320


namespace find_balanced_grid_pairs_l193_19343

-- Define a balanced grid condition
def is_balanced_grid (m n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < m → j < n →
    (∀ k, k < m → grid i k = grid i j) ∧ (∀ l, l < n → grid l j = grid i j)

-- Main theorem statement
theorem find_balanced_grid_pairs (m n : ℕ) :
  (∃ grid, is_balanced_grid m n grid) ↔ (m = n ∨ m = n / 2 ∨ n = 2 * m) :=
by
  sorry

end find_balanced_grid_pairs_l193_19343


namespace determine_common_difference_l193_19394

variables {a : ℕ → ℤ} {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 + n * d

-- The given condition in the problem
def given_condition (a : ℕ → ℤ) (d : ℤ) : Prop :=
  3 * a 6 = a 3 + a 4 + a 5 + 6

-- The theorem to prove
theorem determine_common_difference
  (h_seq : arithmetic_seq a d)
  (h_cond : given_condition a d) :
  d = 1 :=
sorry

end determine_common_difference_l193_19394


namespace no_valid_x_for_given_circle_conditions_l193_19363

theorem no_valid_x_for_given_circle_conditions :
  ∀ x : ℝ,
    ¬ ((x - 15)^2 + 18^2 = 225 ∧ (x - 15)^2 + (-18)^2 = 225) :=
by
  sorry

end no_valid_x_for_given_circle_conditions_l193_19363


namespace necessary_and_sufficient_condition_for_geometric_sequence_l193_19362

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {c : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n+1) = r * a_n n

theorem necessary_and_sufficient_condition_for_geometric_sequence :
  (∀ n : ℕ, S_n n = 2^n + c) →
  (∀ n : ℕ, a_n n = S_n n - S_n (n-1)) →
  is_geometric_sequence a_n ↔ c = -1 :=
by
  sorry

end necessary_and_sufficient_condition_for_geometric_sequence_l193_19362


namespace box_volume_possible_l193_19380

theorem box_volume_possible (x : ℕ) (V : ℕ) (H1 : V = 40 * x^3) (H2 : (2 * x) * (4 * x) * (5 * x) = V) : 
  V = 320 :=
by 
  have x_possible_values := x
  -- checking if V = 320 and x = 2 satisfies the given conditions
  sorry

end box_volume_possible_l193_19380


namespace sequence_a6_value_l193_19341

theorem sequence_a6_value 
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n : ℕ, n ≥ 1 → (1 / a n) + (1 / a (n + 2)) = 2 / a (n + 1)) :
  a 6 = 1 / 3 :=
by
  sorry

end sequence_a6_value_l193_19341


namespace max_sum_of_solutions_l193_19346

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l193_19346


namespace part1_l193_19356

theorem part1 (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^2005 + z^2006 + z^2008 + z^2009 = -2 :=
  sorry

end part1_l193_19356


namespace unique_solution_of_equation_l193_19319

theorem unique_solution_of_equation :
  ∃! (x : Fin 8 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + 
                                  (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
                                  (x 4 - x 5)^2 + (x 5 - x 6)^2 + 
                                  (x 6 - x 7)^2 + (x 7)^2 = 1 / 9 :=
sorry

end unique_solution_of_equation_l193_19319


namespace simplify_expression_l193_19361

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : (x^2 - x * y) / (x - y)^2 = x / (x - y) :=
by sorry

end simplify_expression_l193_19361


namespace Mona_joined_groups_l193_19359

theorem Mona_joined_groups (G : ℕ) (h : G * 4 - 3 = 33) : G = 9 :=
by
  sorry

end Mona_joined_groups_l193_19359


namespace double_rooms_percentage_l193_19372

theorem double_rooms_percentage (S : ℝ) (h1 : 0 < S)
  (h2 : ∃ Sd : ℝ, Sd = 0.75 * S)
  (h3 : ∃ Ss : ℝ, Ss = 0.25 * S):
  (0.375 * S) / (0.625 * S) * 100 = 60 := 
by 
  sorry

end double_rooms_percentage_l193_19372


namespace count_valid_m_l193_19333

theorem count_valid_m (h : 1260 > 0) :
  ∃! (n : ℕ), n = 3 := by
  sorry

end count_valid_m_l193_19333


namespace problem_statement_l193_19330

noncomputable def C : ℝ := 49
noncomputable def D : ℝ := 3.75

theorem problem_statement : C + D = 52.75 := by
  sorry

end problem_statement_l193_19330


namespace triangle_area_given_conditions_l193_19368

theorem triangle_area_given_conditions (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6) (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_given_conditions_l193_19368


namespace minimum_value_l193_19371

-- Define the geometric sequence and its conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (positive : ∀ n, 0 < a n)
variable (geometric_seq : ∀ n, a (n+1) = q * a n)
variable (condition1 : a 6 = a 5 + 2 * a 4)
variable (m n : ℕ)
variable (condition2 : ∀ m n, sqrt (a m * a n) = 2 * a 1 → a m = a n)

-- Prove that the minimum value of 1/m + 9/n is 4
theorem minimum_value : m + n = 4 → (∀ x y : ℝ, (0 < x ∧ 0 < y) → (1 / x + 9 / y) ≥ 4) :=
sorry

end minimum_value_l193_19371


namespace set_intersection_l193_19328

def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | 1 < x ∧ x ≤ 3}
def complement_M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def intersection := N ∩ complement_M

theorem set_intersection : intersection = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end set_intersection_l193_19328


namespace inequality_proof_l193_19300

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
    (a * b + b * c + c * a) * (1 / (a + b)^2 + 1 / (b + c)^2 + 1 / (c + a)^2) ≥ 9 / 4 := 
by
  sorry

end inequality_proof_l193_19300


namespace janice_remaining_time_l193_19389

theorem janice_remaining_time
  (homework_time : ℕ := 30)
  (clean_room_time : ℕ := homework_time / 2)
  (walk_dog_time : ℕ := homework_time + 5)
  (take_out_trash_time : ℕ := homework_time / 6)
  (total_time_before_movie : ℕ := 120) :
  (total_time_before_movie - (homework_time + clean_room_time + walk_dog_time + take_out_trash_time)) = 35 :=
by
  sorry

end janice_remaining_time_l193_19389


namespace canonical_form_lines_l193_19314

theorem canonical_form_lines (x y z : ℝ) :
  (2 * x - y + 3 * z - 1 = 0) →
  (5 * x + 4 * y - z - 7 = 0) →
  (∃ (k : ℝ), x = -11 * k ∧ y = 17 * k + 2 ∧ z = 13 * k + 1) :=
by
  intros h1 h2
  sorry

end canonical_form_lines_l193_19314


namespace sum_of_digits_of_special_two_digit_number_l193_19347

theorem sum_of_digits_of_special_two_digit_number (x : ℕ) (h1 : 1 ≤ x ∧ x < 10) 
  (h2 : ∃ (n : ℕ), n = 11 * x + 30) 
  (h3 : ∃ (sum_digits : ℕ), sum_digits = (x + 3) + x) 
  (h4 : (11 * x + 30) % ((x + 3) + x) = 3)
  (h5 : (11 * x + 30) / ((x + 3) + x) = 7) :
  (x + 3) + x = 7 := 
by 
  sorry

end sum_of_digits_of_special_two_digit_number_l193_19347


namespace range_of_a_l193_19382

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 := sorry

end range_of_a_l193_19382


namespace transport_cost_l193_19350

theorem transport_cost (cost_per_kg : ℝ) (weight_g : ℝ) : 
  (cost_per_kg = 30000) → (weight_g = 400) → 
  ((weight_g / 1000) * cost_per_kg = 12000) :=
by
  intros h1 h2
  sorry

end transport_cost_l193_19350


namespace problem1_problem2_l193_19355

-- For problem 1: Prove the quotient is 5.
def f (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c + a * b + b * c + c * a + a * b * c

theorem problem1 : (625 / f 625) = 5 :=
by
  sorry

-- For problem 2: Prove the set of numbers.
def three_digit_numbers_satisfying_quotient : Finset ℕ :=
  {199, 299, 399, 499, 599, 699, 799, 899, 999}

theorem problem2 (n : ℕ) : (100 ≤ n ∧ n < 1000) ∧ n / f n = 1 ↔ n ∈ three_digit_numbers_satisfying_quotient :=
by
  sorry

end problem1_problem2_l193_19355
