import Mathlib

namespace number_of_males_is_one_part_l183_18346

-- Define the total population
def population : ℕ := 480

-- Define the number of divided parts
def parts : ℕ := 3

-- Define the population part represented by one square.
def part_population (total_population : ℕ) (n_parts : ℕ) : ℕ :=
  total_population / n_parts

-- The Lean statement for the problem
theorem number_of_males_is_one_part : part_population population parts = 160 :=
by
  -- Proof omitted
  sorry

end number_of_males_is_one_part_l183_18346


namespace find_a_of_odd_function_l183_18378

theorem find_a_of_odd_function (a : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x)
  (h_pos_value : f 2 = 6) : a = 5 := by
  sorry

end find_a_of_odd_function_l183_18378


namespace modulus_of_z_l183_18353

open Complex

theorem modulus_of_z (z : ℂ) (hz : (1 + I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l183_18353


namespace simplify_expression_l183_18382

theorem simplify_expression :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 :=
by
  sorry

end simplify_expression_l183_18382


namespace evan_books_two_years_ago_l183_18338

theorem evan_books_two_years_ago (B B2 : ℕ) 
  (h1 : 860 = 5 * B + 60) 
  (h2 : B2 = B + 40) : 
  B2 = 200 := 
by 
  sorry

end evan_books_two_years_ago_l183_18338


namespace math_problem_l183_18384

-- Define the main variables a and b
def a : ℕ := 312
def b : ℕ := 288

-- State the main theorem to be proved
theorem math_problem : (a^2 - b^2) / 24 + 50 = 650 := 
by 
  sorry

end math_problem_l183_18384


namespace cafeteria_total_cost_l183_18317

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l183_18317


namespace residue_neg_1234_mod_32_l183_18306

theorem residue_neg_1234_mod_32 : -1234 % 32 = 14 := 
by sorry

end residue_neg_1234_mod_32_l183_18306


namespace probability_of_one_or_two_l183_18380

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l183_18380


namespace regular_polygon_perimeter_l183_18312

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l183_18312


namespace ratio_of_discount_l183_18356

theorem ratio_of_discount (price_pair1 price_pair2 : ℕ) (total_paid : ℕ) (discount_percent : ℕ) (h1 : price_pair1 = 40)
    (h2 : price_pair2 = 60) (h3 : total_paid = 60) (h4 : discount_percent = 50) :
    (price_pair1 * discount_percent / 100) / (price_pair1 + (price_pair2 - price_pair1 * discount_percent / 100)) = 1 / 4 :=
by
  sorry

end ratio_of_discount_l183_18356


namespace find_x_value_l183_18370

def my_operation (a b : ℝ) : ℝ := 2 * a * b + 3 * b - 2 * a

theorem find_x_value (x : ℝ) (h : my_operation 3 x = 60) : x = 7.33 := 
by 
  sorry

end find_x_value_l183_18370


namespace win_sector_area_l183_18348

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l183_18348


namespace supermarket_profit_and_discount_l183_18302

theorem supermarket_profit_and_discount :
  ∃ (x : ℕ) (nB1 nB2 : ℕ) (discount_rate : ℝ),
    22*x + 30*(nB1) = 6000 ∧
    nB1 = (1 / 2 : ℝ) * x + 15 ∧
    150 * (29 - 22) + 90 * (40 - 30) = 1950 ∧
    nB2 = 3 * nB1 ∧
    150 * (29 - 22) + 270 * (40 * (1 - discount_rate / 100) - 30) = 2130 ∧
    discount_rate = 8.5 := sorry

end supermarket_profit_and_discount_l183_18302


namespace abs_distance_equation_1_abs_distance_equation_2_l183_18351

theorem abs_distance_equation_1 (x : ℚ) : |x - (3 : ℚ)| = 5 ↔ x = 8 ∨ x = -2 := 
sorry

theorem abs_distance_equation_2 (x : ℚ) : |x - (3 : ℚ)| = |x + (1 : ℚ)| ↔ x = 1 :=
sorry

end abs_distance_equation_1_abs_distance_equation_2_l183_18351


namespace degree_of_d_l183_18392

theorem degree_of_d (f d q r : Polynomial ℝ) (f_deg : f.degree = 17)
  (q_deg : q.degree = 10) (r_deg : r.degree = 4) 
  (remainder : r = Polynomial.C 5 * X^4 - Polynomial.C 3 * X^3 + Polynomial.C 2 * X^2 - X + 15)
  (div_relation : f = d * q + r) (r_deg_lt_d_deg : r.degree < d.degree) :
  d.degree = 7 :=
sorry

end degree_of_d_l183_18392


namespace mutual_fund_share_increase_l183_18379

theorem mutual_fund_share_increase (P : ℝ) (h1 : (P * 1.20) = 1.20 * P) (h2 : (1.20 * P) * (1 / 3) = 0.40 * P) :
  ((1.60 * P) = (P * 1.60)) :=
by
  sorry

end mutual_fund_share_increase_l183_18379


namespace cubic_expansion_solution_l183_18324

theorem cubic_expansion_solution (x y : ℕ) (h_x : x = 27) (h_y : y = 9) : 
  x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3 = 46656 :=
by
  sorry

end cubic_expansion_solution_l183_18324


namespace third_place_books_max_l183_18363

theorem third_place_books_max (x y z : ℕ) (hx : 100 ∣ x) (hxpos : 0 < x) (hy : 100 ∣ y) (hz : 100 ∣ z)
  (h_sum : 2 * x + 100 + x + 100 + x + y + z ≤ 10000)
  (h_first_eq : 2 * x + 100 = x + 100 + x)
  (h_second_eq : x + 100 = y + z) 
  : x ≤ 1900 := sorry

end third_place_books_max_l183_18363


namespace relationship_between_a_and_b_l183_18361

noncomputable section
open Classical

theorem relationship_between_a_and_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end relationship_between_a_and_b_l183_18361


namespace green_ball_removal_l183_18337

variable (total_balls : ℕ)
variable (initial_green_balls : ℕ)
variable (initial_yellow_balls : ℕ)
variable (desired_green_percentage : ℚ)
variable (removals : ℕ)

theorem green_ball_removal :
  initial_green_balls = 420 → 
  total_balls = 600 → 
  desired_green_percentage = 3 / 5 →
  (420 - removals) / (600 - removals) = desired_green_percentage → 
  removals = 150 :=
sorry

end green_ball_removal_l183_18337


namespace positive_difference_sums_even_odd_l183_18391

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l183_18391


namespace algebra_problem_l183_18395

theorem algebra_problem
  (x : ℝ)
  (h : 59 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = Real.sqrt 61 :=
sorry

end algebra_problem_l183_18395


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l183_18301

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l183_18301


namespace positive_number_solution_exists_l183_18366

theorem positive_number_solution_exists (x : ℝ) (h₁ : 0 < x) (h₂ : (2 / 3) * x = (64 / 216) * (1 / x)) : x = 2 / 3 :=
by sorry

end positive_number_solution_exists_l183_18366


namespace non_congruent_triangles_l183_18371

-- Definition of points and isosceles property
variable (A B C P Q R : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

-- Conditions of the problem
def is_isosceles (A B C : Type) : Prop := (A = B) ∧ (A = C)
def is_midpoint (P Q R : Type) (A B C : Type) : Prop := sorry -- precise formal definition omitted for brevity

-- Theorem stating the final result
theorem non_congruent_triangles (A B C P Q R : Type)
  (h_iso : is_isosceles A B C)
  (h_midpoints : is_midpoint P Q R A B C) :
  ∃ (n : ℕ), n = 4 := 
  by 
    -- proof abbreviated
    sorry

end non_congruent_triangles_l183_18371


namespace expression_X_l183_18325

variable {a b X : ℝ}

theorem expression_X (h1 : a / b = 4 / 3) (h2 : (3 * a + 2 * b) / X = 3) : X = 2 * b := 
sorry

end expression_X_l183_18325


namespace anja_equal_integers_l183_18304

theorem anja_equal_integers (S : Finset ℤ) (h_card : S.card = 2014)
  (h_mean : ∀ (x y z : ℤ), x ∈ S → y ∈ S → z ∈ S → (x + y + z) / 3 ∈ S) :
  ∃ k, ∀ x ∈ S, x = k :=
sorry

end anja_equal_integers_l183_18304


namespace num_proper_subsets_of_A_l183_18316

open Set

def A : Finset ℕ := {2, 3}

theorem num_proper_subsets_of_A : (A.powerset \ {A, ∅}).card = 3 := by
  sorry

end num_proper_subsets_of_A_l183_18316


namespace max_value_of_8a_5b_15c_l183_18314

theorem max_value_of_8a_5b_15c (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  8*a + 5*b + 15*c ≤ (Real.sqrt 115) / 2 :=
by
  sorry

end max_value_of_8a_5b_15c_l183_18314


namespace games_against_other_division_l183_18367

theorem games_against_other_division
  (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5)
  (total_games : N * 4 + 5 * M = 82) :
  5 * M = 30 :=
by
  sorry

end games_against_other_division_l183_18367


namespace power_modulus_difference_l183_18343

theorem power_modulus_difference (m : ℤ) :
  (51 % 6 = 3) → (9 % 6 = 3) → ((51 : ℤ)^1723 - (9 : ℤ)^1723) % 6 = 0 :=
by 
  intros h1 h2
  sorry

end power_modulus_difference_l183_18343


namespace find_larger_number_l183_18341

theorem find_larger_number (x y : ℤ) (h1 : 5 * y = 6 * x) (h2 : y - x = 12) : y = 72 :=
sorry

end find_larger_number_l183_18341


namespace angle_QRS_determination_l183_18394

theorem angle_QRS_determination (PQ_parallel_RS : ∀ (P Q R S T : Type) 
  (angle_PTQ : ℝ) (angle_SRT : ℝ), 
  PQ_parallel_RS → (angle_PTQ = angle_SRT) → (angle_PTQ = 4 * angle_SRT - 120)) 
  (angle_SRT : ℝ) (angle_QRS : ℝ) 
  (h : angle_SRT = 4 * angle_SRT - 120) : angle_QRS = 40 :=
by 
  sorry

end angle_QRS_determination_l183_18394


namespace sequence_sum_formula_l183_18326

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_S : ∀ n, S n = (1 / 6) * (a n ^ 2 + 3 * a n - 4)) : 
  ∀ n, S n = (3 / 2) * n ^ 2 + (5 / 2) * n :=
by
  sorry

end sequence_sum_formula_l183_18326


namespace sequence_term_number_l183_18321

theorem sequence_term_number (n : ℕ) : (n ≥ 1) → (n + 3 = 17 ∧ n + 1 = 15) → n = 14 := 
by
  intro h1 h2
  sorry

end sequence_term_number_l183_18321


namespace jordan_novels_read_l183_18360

variable (J A : ℕ)

theorem jordan_novels_read (h1 : A = (1 / 10) * J)
                          (h2 : J = A + 108) :
                          J = 120 := 
by
  sorry

end jordan_novels_read_l183_18360


namespace contractor_days_l183_18385

def days_engaged (days_worked days_absent : ℕ) (earnings_per_day : ℝ) (fine_per_absent_day : ℝ) : ℝ :=
  earnings_per_day * days_worked - fine_per_absent_day * days_absent

theorem contractor_days
  (days_absent : ℕ)
  (earnings_per_day : ℝ)
  (fine_per_absent_day : ℝ)
  (total_amount : ℝ)
  (days_worked : ℕ)
  (h1 : days_absent = 12)
  (h2 : earnings_per_day = 25)
  (h3 : fine_per_absent_day = 7.50)
  (h4 : total_amount = 360)
  (h5 : days_engaged days_worked days_absent earnings_per_day fine_per_absent_day = total_amount) :
  days_worked = 18 :=
by sorry

end contractor_days_l183_18385


namespace quadratic_inequalities_l183_18345

variable (c x₁ y₁ y₂ y₃ : ℝ)
noncomputable def quadratic_function := -x₁^2 + 2*x₁ + c

theorem quadratic_inequalities
  (h_c : c < 0)
  (h_y₁ : quadratic_function c x₁ > 0)
  (h_y₂ : y₂ = quadratic_function c (x₁ - 2))
  (h_y₃ : y₃ = quadratic_function c (x₁ + 2)) :
  y₂ < 0 ∧ y₃ < 0 :=
by sorry

end quadratic_inequalities_l183_18345


namespace depth_of_box_l183_18344

theorem depth_of_box (length width depth : ℕ) (side_length : ℕ)
  (h_length : length = 30)
  (h_width : width = 48)
  (h_side_length : Nat.gcd length width = side_length)
  (h_cubes : side_length ^ 3 = 216)
  (h_volume : 80 * (side_length ^ 3) = length * width * depth) :
  depth = 12 :=
by
  sorry

end depth_of_box_l183_18344


namespace product_of_four_consecutive_integers_is_not_square_l183_18331

theorem product_of_four_consecutive_integers_is_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = (n-1)*n*(n+1)*(n+2) :=
sorry

end product_of_four_consecutive_integers_is_not_square_l183_18331


namespace Tom_money_made_l183_18310

theorem Tom_money_made (money_last_week money_now : ℕ) (h1 : money_last_week = 74) (h2 : money_now = 160) : 
  (money_now - money_last_week = 86) :=
by 
  sorry

end Tom_money_made_l183_18310


namespace cube_mod_35_divisors_l183_18374

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors_l183_18374


namespace ab_value_l183_18362

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l183_18362


namespace molecular_weight_of_benzene_l183_18311

def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def number_of_C_atoms : ℕ := 6
def number_of_H_atoms : ℕ := 6

theorem molecular_weight_of_benzene : 
  (number_of_C_atoms * molecular_weight_C + number_of_H_atoms * molecular_weight_H) = 78.108 :=
by
  sorry

end molecular_weight_of_benzene_l183_18311


namespace Frank_worked_days_l183_18350

theorem Frank_worked_days
  (h_per_day : ℕ) (total_hours : ℕ) (d : ℕ) 
  (h_day_def : h_per_day = 8) 
  (total_hours_def : total_hours = 32) 
  (d_def : d = total_hours / h_per_day) : 
  d = 4 :=
by 
  rw [total_hours_def, h_day_def] at d_def
  exact d_def

end Frank_worked_days_l183_18350


namespace cmp_c_b_a_l183_18365

noncomputable def a : ℝ := 17 / 18
noncomputable def b : ℝ := Real.cos (1 / 3)
noncomputable def c : ℝ := 3 * Real.sin (1 / 3)

theorem cmp_c_b_a:
  c > b ∧ b > a := by
  sorry

end cmp_c_b_a_l183_18365


namespace total_baseball_cards_is_100_l183_18319

-- Define the initial number of baseball cards Mike has
def initial_baseball_cards : ℕ := 87

-- Define the number of baseball cards Sam gave to Mike
def given_baseball_cards : ℕ := 13

-- Define the total number of baseball cards Mike has now
def total_baseball_cards : ℕ := initial_baseball_cards + given_baseball_cards

-- State the theorem that the total number of baseball cards is 100
theorem total_baseball_cards_is_100 : total_baseball_cards = 100 := by
  sorry

end total_baseball_cards_is_100_l183_18319


namespace compatibility_condition_l183_18329

theorem compatibility_condition (a b c d x : ℝ) 
  (h1 : a * x + b = 0) (h2 : c * x + d = 0) : a * d - b * c = 0 :=
sorry

end compatibility_condition_l183_18329


namespace train_length_l183_18313

theorem train_length (V L : ℝ) (h1 : L = V * 18) (h2 : L + 550 = V * 51) : L = 300 := sorry

end train_length_l183_18313


namespace max_consecutive_sum_l183_18309

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l183_18309


namespace revenue_after_decrease_l183_18322

theorem revenue_after_decrease (original_revenue : ℝ) (percentage_decrease : ℝ) (final_revenue : ℝ) 
  (h1 : original_revenue = 69.0) 
  (h2 : percentage_decrease = 24.637681159420293) 
  (h3 : final_revenue = original_revenue - (original_revenue * (percentage_decrease / 100))) 
  : final_revenue = 52.0 :=
by
  sorry

end revenue_after_decrease_l183_18322


namespace yards_dyed_green_calc_l183_18357

-- Given conditions: total yards dyed and yards dyed pink
def total_yards_dyed : ℕ := 111421
def yards_dyed_pink : ℕ := 49500

-- Goal: Prove the number of yards dyed green
theorem yards_dyed_green_calc : total_yards_dyed - yards_dyed_pink = 61921 :=
by 
-- sorry means that the proof is skipped.
sorry

end yards_dyed_green_calc_l183_18357


namespace cost_backpack_is_100_l183_18336

-- Definitions based on the conditions
def cost_wallet : ℕ := 50
def cost_sneakers_per_pair : ℕ := 100
def num_sneakers_pairs : ℕ := 2
def cost_jeans_per_pair : ℕ := 50
def num_jeans_pairs : ℕ := 2
def total_spent : ℕ := 450

-- The problem statement
theorem cost_backpack_is_100 (x : ℕ) 
  (leonard_total : ℕ := cost_wallet + num_sneakers_pairs * cost_sneakers_per_pair) 
  (michael_non_backpack_total : ℕ := num_jeans_pairs * cost_jeans_per_pair) :
  total_spent = leonard_total + michael_non_backpack_total + x → x = 100 := 
by
  unfold cost_wallet cost_sneakers_per_pair num_sneakers_pairs total_spent cost_jeans_per_pair num_jeans_pairs
  intro h
  sorry

end cost_backpack_is_100_l183_18336


namespace number_of_rectangles_in_5x5_grid_l183_18342

-- Number of ways to choose k elements from a set of n elements
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def points_in_each_direction : ℕ := 5
def number_of_rectangles : ℕ :=
  binomial points_in_each_direction 2 * binomial points_in_each_direction 2

-- Lean statement to prove the problem
theorem number_of_rectangles_in_5x5_grid :
  number_of_rectangles = 100 :=
by
  -- begin Lean proof
  sorry

end number_of_rectangles_in_5x5_grid_l183_18342


namespace total_passengers_landed_l183_18354

theorem total_passengers_landed (on_time late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) : 
    on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l183_18354


namespace boy_usual_time_l183_18359

noncomputable def usual_rate (R : ℝ) := R
noncomputable def usual_time (T : ℝ) := T
noncomputable def faster_rate (R : ℝ) := (7 / 6) * R
noncomputable def faster_time (T : ℝ) := T - 5

theorem boy_usual_time
  (R : ℝ) (T : ℝ) 
  (h1 : usual_rate R * usual_time T = faster_rate R * faster_time T) :
  T = 35 :=
by 
  unfold usual_rate usual_time faster_rate faster_time at h1
  sorry

end boy_usual_time_l183_18359


namespace shorter_piece_length_l183_18305

-- Definitions for the conditions
def total_length : ℕ := 70
def ratio (short long : ℕ) : Prop := long = (5 * short) / 2

-- The proof problem statement
theorem shorter_piece_length (x : ℕ) (h1 : total_length = x + (5 * x) / 2) : x = 20 :=
sorry

end shorter_piece_length_l183_18305


namespace find_other_number_l183_18398

-- Define the conditions and the theorem
theorem find_other_number (hcf lcm a b : ℕ) (hcf_def : hcf = 20) (lcm_def : lcm = 396) (a_def : a = 36) (rel : hcf * lcm = a * b) : b = 220 :=
by 
  sorry -- Proof to be provided

end find_other_number_l183_18398


namespace AM_GM_Ineq_l183_18320

theorem AM_GM_Ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_Ineq_l183_18320


namespace product_mod_25_l183_18393

def remainder_when_divided_by_25 (n : ℕ) : ℕ := n % 25

theorem product_mod_25 (a b c d : ℕ) 
  (h1 : a = 1523) (h2 : b = 1857) (h3 : c = 1919) (h4 : d = 2012) :
  remainder_when_divided_by_25 (a * b * c * d) = 8 :=
by
  sorry

end product_mod_25_l183_18393


namespace uniquely_identify_figure_l183_18330

structure Figure where
  is_curve : Bool
  has_axis_of_symmetry : Bool
  has_center_of_symmetry : Bool

def Circle : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Ellipse : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := false }
def Triangle : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }
def Square : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Rectangle : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Parallelogram : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := true }
def Trapezoid : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }

theorem uniquely_identify_figure (figures : List Figure) (q1 q2 q3 : Figure → Bool) :
  ∀ (f : Figure), ∃! (f' : Figure), 
    q1 f' = q1 f ∧ q2 f' = q2 f ∧ q3 f' = q3 f :=
by
  sorry

end uniquely_identify_figure_l183_18330


namespace range_of_c_l183_18386

-- Definitions of the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∃ x : ℝ, x^2 - c^2 ≤ - (1 / 16)

-- Main theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : p c) (h3 : q c) : c ≥ 1 / 4 ∧ c < 1 :=
  sorry

end range_of_c_l183_18386


namespace cos_double_angle_of_tan_l183_18307

theorem cos_double_angle_of_tan (θ : ℝ) (h : Real.tan θ = -1 / 3) : Real.cos (2 * θ) = 4 / 5 :=
sorry

end cos_double_angle_of_tan_l183_18307


namespace sum_of_two_integers_is_22_l183_18300

noncomputable def a_and_b_sum_to_S : Prop :=
  ∃ (a b S : ℕ), 
    a + b = S ∧ 
    a^2 - b^2 = 44 ∧ 
    a * b = 120 ∧ 
    S = 22

theorem sum_of_two_integers_is_22 : a_and_b_sum_to_S :=
by {
  sorry
}

end sum_of_two_integers_is_22_l183_18300


namespace find_a_of_inequality_solution_set_l183_18328

theorem find_a_of_inequality_solution_set
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) :
  a = -4 :=
sorry

end find_a_of_inequality_solution_set_l183_18328


namespace well_filled_ways_1_5_l183_18373

-- Define a structure for representing the conditions of the figure filled with integers
structure WellFilledFigure where
  top_circle : ℕ
  shaded_circle_possibilities : Finset ℕ
  sub_diagram_possibilities : ℕ

-- Define an example of this structure corresponding to our problem
def figure1_5 : WellFilledFigure :=
  { top_circle := 5,
    shaded_circle_possibilities := {1, 2, 3, 4},
    sub_diagram_possibilities := 2 }

-- Define the theorem statement
theorem well_filled_ways_1_5 (f : WellFilledFigure) : (f.top_circle = 5) → 
  (f.shaded_circle_possibilities.card = 4) → 
  (f.sub_diagram_possibilities = 2) → 
  (4 * 2 = 8) := by
  sorry

end well_filled_ways_1_5_l183_18373


namespace beth_longer_distance_by_5_miles_l183_18335

noncomputable def average_speed_john : ℝ := 40
noncomputable def time_john_hours : ℝ := 30 / 60
noncomputable def distance_john : ℝ := average_speed_john * time_john_hours

noncomputable def average_speed_beth : ℝ := 30
noncomputable def time_beth_hours : ℝ := (30 + 20) / 60
noncomputable def distance_beth : ℝ := average_speed_beth * time_beth_hours

theorem beth_longer_distance_by_5_miles : distance_beth - distance_john = 5 := by 
  sorry

end beth_longer_distance_by_5_miles_l183_18335


namespace boys_without_calculators_l183_18339

theorem boys_without_calculators :
    ∀ (total_boys students_with_calculators girls_with_calculators : ℕ),
    total_boys = 16 →
    students_with_calculators = 22 →
    girls_with_calculators = 13 →
    total_boys - (students_with_calculators - girls_with_calculators) = 7 :=
by
  intros
  sorry

end boys_without_calculators_l183_18339


namespace alyssa_total_games_l183_18334

def calc_total_games (games_this_year games_last_year games_next_year : ℕ) : ℕ :=
  games_this_year + games_last_year + games_next_year

theorem alyssa_total_games :
  calc_total_games 11 13 15 = 39 :=
by
  -- Proof goes here
  sorry

end alyssa_total_games_l183_18334


namespace machine_subtract_l183_18396

theorem machine_subtract (x : ℤ) (h1 : 26 + 15 - x = 35) : x = 6 :=
by
  sorry

end machine_subtract_l183_18396


namespace smallest_value_of_c_l183_18383

def bound_a (a b : ℝ) : Prop := 1 + a ≤ b
def bound_inv (a b c : ℝ) : Prop := (1 / a) + (1 / b) ≤ (1 / c)

theorem smallest_value_of_c (a b c : ℝ) (ha : 1 < a) (hb : a < b) 
  (hc : b < c) (h_ab : bound_a a b) (h_inv : bound_inv a b c) : 
  c ≥ (3 + Real.sqrt 5) / 2 := 
sorry

end smallest_value_of_c_l183_18383


namespace matrix_cube_computation_l183_18397

-- Define the original matrix
def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2], ![2, 0]]

-- Define the expected result matrix
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-8, 0], ![0, -8]]

-- State the theorem to be proved
theorem matrix_cube_computation : matrix1 ^ 3 = expected_matrix :=
  by sorry

end matrix_cube_computation_l183_18397


namespace foldable_positions_are_7_l183_18327

-- Define the initial polygon with 6 congruent squares forming a cross shape
def initial_polygon : Prop :=
  -- placeholder definition, in practice, this would be a more detailed geometrical model
  sorry

-- Define the positions where an additional square can be attached (11 positions in total)
def position (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 11

-- Define the resulting polygon when an additional square is attached at position n
def resulting_polygon (n : ℕ) : Prop :=
  position n ∧ initial_polygon

-- Define the condition that a polygon can be folded into a cube with one face missing
def can_fold_to_cube_with_missing_face (p : Prop) : Prop := sorry

-- The theorem that needs to be proved
theorem foldable_positions_are_7 : 
  ∃ (positions : Finset ℕ), 
    positions.card = 7 ∧ 
    ∀ n ∈ positions, can_fold_to_cube_with_missing_face (resulting_polygon n) :=
  sorry

end foldable_positions_are_7_l183_18327


namespace identity_x_squared_minus_y_squared_l183_18390

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l183_18390


namespace amount_due_years_l183_18315

noncomputable def years_due (PV FV : ℝ) (r : ℝ) : ℝ :=
  (Real.log (FV / PV)) / (Real.log (1 + r))

theorem amount_due_years : 
  years_due 200 242 0.10 = 2 :=
by
  sorry

end amount_due_years_l183_18315


namespace area_of_region_l183_18372

theorem area_of_region :
  (∃ (x y: ℝ), x^2 + y^2 = 5 * |x - y| + 2 * |x + y|) → 
  (∃ (A : ℝ), A = 14.5 * Real.pi) :=
sorry

end area_of_region_l183_18372


namespace mass_percentage_of_Cl_in_bleach_l183_18323

-- Definitions based on conditions
def Na_molar_mass : Float := 22.99
def Cl_molar_mass : Float := 35.45
def O_molar_mass : Float := 16.00

def NaClO_molar_mass : Float := Na_molar_mass + Cl_molar_mass + O_molar_mass

def mass_NaClO (mass_na: Float) (mass_cl: Float) (mass_o: Float) : Float :=
  mass_na + mass_cl + mass_o

def mass_of_NaClO : Float := 100.0

def mass_of_Cl_in_NaClO (mass_of_NaClO: Float) : Float :=
  (Cl_molar_mass / NaClO_molar_mass) * mass_of_NaClO

-- Statement to prove
theorem mass_percentage_of_Cl_in_bleach :
  let mass_Cl := mass_of_Cl_in_NaClO mass_of_NaClO
  (mass_Cl / mass_of_NaClO) * 100 = 47.61 :=
by 
  -- Skip the proof
  sorry

end mass_percentage_of_Cl_in_bleach_l183_18323


namespace least_integer_value_l183_18303

theorem least_integer_value (x : ℤ) :
  (|3 * x + 4| ≤ 25) → ∃ y : ℤ, x = y ∧ y = -9 :=
by
  sorry

end least_integer_value_l183_18303


namespace reggie_father_money_l183_18318

theorem reggie_father_money :
  let books := 5
  let cost_per_book := 2
  let amount_left := 38
  books * cost_per_book + amount_left = 48 :=
by
  sorry

end reggie_father_money_l183_18318


namespace find_prime_pair_l183_18381

-- Definition of the problem
def is_integral_expression (p q : ℕ) : Prop :=
  (p + q)^(p + q) * (p - q)^(p - q) - 1 ≠ 0 ∧
  (p + q)^(p - q) * (p - q)^(p + q) - 1 ≠ 0 ∧
  ((p + q)^(p + q) * (p - q)^(p - q) - 1) % ((p + q)^(p - q) * (p - q)^(p + q) - 1) = 0

-- Mathematical theorem to be proved
theorem find_prime_pair (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (h : p > q) :
  is_integral_expression p q → (p, q) = (3, 2) :=
by 
  sorry

end find_prime_pair_l183_18381


namespace total_amount_paid_l183_18355

theorem total_amount_paid (monthly_payment_1 monthly_payment_2 : ℕ) (years_1 years_2 : ℕ)
  (monthly_payment_1_eq : monthly_payment_1 = 300)
  (monthly_payment_2_eq : monthly_payment_2 = 350)
  (years_1_eq : years_1 = 3)
  (years_2_eq : years_2 = 2) :
  let annual_payment_1 := monthly_payment_1 * 12
  let annual_payment_2 := monthly_payment_2 * 12
  let total_1 := annual_payment_1 * years_1
  let total_2 := annual_payment_2 * years_2
  total_1 + total_2 = 19200 :=
by
  sorry

end total_amount_paid_l183_18355


namespace income_ratio_l183_18377

theorem income_ratio (I1 I2 E1 E2 : ℕ)
  (hI1 : I1 = 3500)
  (hE_ratio : (E1:ℚ) / E2 = 3 / 2)
  (hSavings : ∀ (x y : ℕ), x - E1 = 1400 ∧ y - E2 = 1400 → x = I1 ∧ y = I2) :
  I1 / I2 = 5 / 4 :=
by
  -- The proof steps would go here
  sorry

end income_ratio_l183_18377


namespace find_n_eq_6_l183_18347

theorem find_n_eq_6 (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) : 2^n + n^2 + 25 = p^3 → n = 6 := by
  sorry

end find_n_eq_6_l183_18347


namespace who_drank_most_l183_18340

theorem who_drank_most (eunji yujeong yuna : ℝ) 
    (h1 : eunji = 0.5) 
    (h2 : yujeong = 7 / 10) 
    (h3 : yuna = 6 / 10) :
    max (max eunji yujeong) yuna = yujeong :=
by {
    sorry
}

end who_drank_most_l183_18340


namespace water_bottles_needed_l183_18376

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l183_18376


namespace abcd_inequality_l183_18349

theorem abcd_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_eq : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) + (d^2 / (1 + d^2)) = 1) :
  a * b * c * d ≤ 1 / 9 :=
sorry

end abcd_inequality_l183_18349


namespace isosceles_triangle_sides_l183_18364

theorem isosceles_triangle_sides (P : ℝ) (a b c : ℝ) (h₀ : P = 26) (h₁ : a = 11) (h₂ : a = b ∨ a = c)
  (h₃ : a + b + c = P) : 
  (b = 11 ∧ c = 4) ∨ (b = 7.5 ∧ c = 7.5) :=
by
  sorry

end isosceles_triangle_sides_l183_18364


namespace inequality_lt_l183_18368

theorem inequality_lt (x y : ℝ) (h1 : x > y) (h2 : y > 0) (n k : ℕ) (h3 : n > k) :
  (x^k - y^k) ^ n < (x^n - y^n) ^ k := 
  sorry

end inequality_lt_l183_18368


namespace simple_interest_correct_l183_18387

-- Define the parameters
def principal : ℝ := 10000
def rate_decimal : ℝ := 0.04
def time_years : ℝ := 1

-- Define the simple interest calculation function
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Prove that the simple interest is equal to $400
theorem simple_interest_correct : simple_interest principal rate_decimal time_years = 400 :=
by
  -- Placeholder for the proof
  sorry

end simple_interest_correct_l183_18387


namespace jackson_sandwiches_l183_18332

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l183_18332


namespace pie_chart_shows_percentage_l183_18308

-- Define the different types of graphs
inductive GraphType
| PieChart
| BarGraph
| LineGraph
| Histogram

-- Define conditions of the problem
def shows_percentage_of_whole (g : GraphType) : Prop :=
  g = GraphType.PieChart

def displays_with_rectangular_bars (g : GraphType) : Prop :=
  g = GraphType.BarGraph

def shows_trends (g : GraphType) : Prop :=
  g = GraphType.LineGraph

def shows_frequency_distribution (g : GraphType) : Prop :=
  g = GraphType.Histogram

-- We need to prove that a pie chart satisfies the condition of showing percentages of parts in a whole
theorem pie_chart_shows_percentage : shows_percentage_of_whole GraphType.PieChart :=
  by
    -- Proof is skipped
    sorry

end pie_chart_shows_percentage_l183_18308


namespace expression_equals_two_l183_18375

noncomputable def math_expression : ℝ :=
  27^(1/3) + Real.log 4 + 2 * Real.log 5 - Real.exp (Real.log 3)

theorem expression_equals_two : math_expression = 2 := by
  sorry

end expression_equals_two_l183_18375


namespace problem_a4_inv_a4_l183_18389

theorem problem_a4_inv_a4 (a : ℝ) (h : (a + 1/a)^4 = 16) : (a^4 + 1/a^4) = 2 := 
by 
  sorry

end problem_a4_inv_a4_l183_18389


namespace largest_polygon_is_E_l183_18369

def area (num_unit_squares num_right_triangles num_half_squares: ℕ) : ℚ :=
  num_unit_squares + num_right_triangles * 0.5 + num_half_squares * 0.25

def polygon_A_area := area 3 2 0
def polygon_B_area := area 4 1 0
def polygon_C_area := area 2 4 2
def polygon_D_area := area 5 0 0
def polygon_E_area := area 3 3 4

theorem largest_polygon_is_E :
  polygon_E_area > polygon_A_area ∧ 
  polygon_E_area > polygon_B_area ∧ 
  polygon_E_area > polygon_C_area ∧ 
  polygon_E_area > polygon_D_area :=
by
  sorry

end largest_polygon_is_E_l183_18369


namespace exact_sunny_days_probability_l183_18333

noncomputable def choose (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

def rain_prob : ℚ := 3 / 4
def sun_prob : ℚ := 1 / 4
def days : ℕ := 5

theorem exact_sunny_days_probability : (choose days 2 * (sun_prob^2 * rain_prob^3) = 135 / 512) :=
by
  sorry

end exact_sunny_days_probability_l183_18333


namespace solve_for_x_l183_18388

theorem solve_for_x (x : ℤ) (h : 45 - (5 * 3) = x + 7) : x = 23 := 
by
  sorry

end solve_for_x_l183_18388


namespace smallest_6_digit_divisible_by_111_l183_18352

theorem smallest_6_digit_divisible_by_111 :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 111 = 0 ∧ x = 100011 :=
  by
    sorry

end smallest_6_digit_divisible_by_111_l183_18352


namespace intersection_points_l183_18358

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def parabola2 (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, parabola1 x) ∧ parabola1 x = parabola2 x)} =
  { 
    ( (3 + Real.sqrt 13) / 4, (74 + 14 * Real.sqrt 13) / 16 ),
    ( (3 - Real.sqrt 13) / 4, (74 - 14 * Real.sqrt 13) / 16 )
  } := sorry

end intersection_points_l183_18358


namespace train_length_l183_18399

theorem train_length (speed_kmph : ℕ) (time_s : ℕ) (platform_length_m : ℕ) (h1 : speed_kmph = 72) (h2 : time_s = 26) (h3 : platform_length_m = 260) :
  ∃ train_length_m : ℕ, train_length_m = 260 := by
  sorry

end train_length_l183_18399
