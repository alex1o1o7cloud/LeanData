import Mathlib

namespace NUMINAMATH_GPT_jeans_price_increase_l850_85088

theorem jeans_price_increase 
  (C : ℝ) 
  (R : ℝ) 
  (F : ℝ) 
  (H1 : R = 1.40 * C)
  (H2 : F = 1.82 * C) 
  : (F - C) / C * 100 = 82 := 
sorry

end NUMINAMATH_GPT_jeans_price_increase_l850_85088


namespace NUMINAMATH_GPT_combine_like_terms_substitute_expression_complex_expression_l850_85053

-- Part 1
theorem combine_like_terms (a b : ℝ) : 
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 :=
by
  sorry

-- Part 2
theorem substitute_expression (x y : ℝ) (h1 : x^2 - 2 * y = -5) : 
  4 * x^2 - 8 * y + 24 = 4 :=
by
  sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2 * b = 1009.5) 
  (h2 : 2 * b - c = -2024.6666)
  (h3 : c - d = 1013.1666) : 
  (a - c) + (2 * b - d) - (2 * b - c) = -2 :=
by
  sorry

end NUMINAMATH_GPT_combine_like_terms_substitute_expression_complex_expression_l850_85053


namespace NUMINAMATH_GPT_factorize_expr_l850_85050

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end NUMINAMATH_GPT_factorize_expr_l850_85050


namespace NUMINAMATH_GPT_simplified_expression_l850_85079

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplified_expression_l850_85079


namespace NUMINAMATH_GPT_pablo_days_to_complete_puzzles_l850_85043

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end NUMINAMATH_GPT_pablo_days_to_complete_puzzles_l850_85043


namespace NUMINAMATH_GPT_M_intersect_P_l850_85003

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }
noncomputable def P : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

theorem M_intersect_P :
  M ∩ P = { y | y ≥ 1 } :=
sorry

end NUMINAMATH_GPT_M_intersect_P_l850_85003


namespace NUMINAMATH_GPT_sum_last_two_digits_l850_85016

theorem sum_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) :
  (a ^ 30 + b ^ 30) % 100 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_l850_85016


namespace NUMINAMATH_GPT_find_value_of_m_l850_85094

theorem find_value_of_m (m : ℤ) (x : ℤ) (h : (x - 3 ≠ 0) ∧ (x = 3)) : 
  ((x - 1) / (x - 3) = m / (x - 3)) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_m_l850_85094


namespace NUMINAMATH_GPT_Ivan_walk_time_l850_85005

variables (u v : ℝ) (T t : ℝ)

-- Define the conditions
def condition1 : Prop := T = 10 * v / u
def condition2 : Prop := T + 70 = t
def condition3 : Prop := v * t = u * T + v * (t - T + 70)

-- Problem statement: Given the conditions, prove T = 80
theorem Ivan_walk_time (h1 : condition1 u v T) (h2 : condition2 T t) (h3 : condition3 u v T t) : 
  T = 80 := by
  sorry

end NUMINAMATH_GPT_Ivan_walk_time_l850_85005


namespace NUMINAMATH_GPT_N_divisible_by_9_l850_85085

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem N_divisible_by_9 (N : ℕ) (h : sum_of_digits N = sum_of_digits (5 * N)) : N % 9 = 0 := 
sorry

end NUMINAMATH_GPT_N_divisible_by_9_l850_85085


namespace NUMINAMATH_GPT_eccentricity_condition_l850_85077

theorem eccentricity_condition (m : ℝ) (h : 0 < m) : 
  (m < (4 / 3) ∨ m > (3 / 4)) ↔ ((1 - m) > (1 / 4) ∨ ((m - 1) / m) > (1 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_condition_l850_85077


namespace NUMINAMATH_GPT_smallest_divisible_by_15_16_18_l850_85070

def factors_of_15 : Prop := 15 = 3 * 5
def factors_of_16 : Prop := 16 = 2^4
def factors_of_18 : Prop := 18 = 2 * 3^2

theorem smallest_divisible_by_15_16_18 (h1: factors_of_15) (h2: factors_of_16) (h3: factors_of_18) : 
  ∃ n, n > 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ n % 18 = 0 ∧ n = 720 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisible_by_15_16_18_l850_85070


namespace NUMINAMATH_GPT_range_of_b_l850_85074

theorem range_of_b (x b : ℝ) (hb : b > 0) : 
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l850_85074


namespace NUMINAMATH_GPT_cos_seven_pi_over_four_proof_l850_85010

def cos_seven_pi_over_four : Prop := (Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2)

theorem cos_seven_pi_over_four_proof : cos_seven_pi_over_four :=
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_four_proof_l850_85010


namespace NUMINAMATH_GPT_leo_trousers_count_l850_85030

theorem leo_trousers_count (S T : ℕ) (h1 : 5 * S + 9 * T = 140) (h2 : S = 10) : T = 10 :=
by
  sorry

end NUMINAMATH_GPT_leo_trousers_count_l850_85030


namespace NUMINAMATH_GPT_gcd_288_123_l850_85081

-- Define the conditions
def cond1 : 288 = 2 * 123 + 42 := by sorry
def cond2 : 123 = 2 * 42 + 39 := by sorry
def cond3 : 42 = 39 + 3 := by sorry
def cond4 : 39 = 13 * 3 := by sorry

-- Prove that GCD of 288 and 123 is 3
theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_288_123_l850_85081


namespace NUMINAMATH_GPT_Frank_days_to_finish_book_l850_85012

theorem Frank_days_to_finish_book (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 22) (h2 : total_pages = 12518) : total_pages / pages_per_day = 569 := by
  sorry

end NUMINAMATH_GPT_Frank_days_to_finish_book_l850_85012


namespace NUMINAMATH_GPT_range_of_k_l850_85092

theorem range_of_k (f : ℝ → ℝ) (a : ℝ) (k : ℝ) 
  (h₀ : ∀ x > 0, f x = 2 - 1 / (a - x)^2) 
  (h₁ : ∀ x > 0, k^2 * x + f (1 / 4 * x + 1) > 0) : 
  k ≠ 0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_range_of_k_l850_85092


namespace NUMINAMATH_GPT_workshop_workers_l850_85025

theorem workshop_workers (W N: ℕ) 
  (h1: 8000 * W = 70000 + 6000 * N) 
  (h2: W = 7 + N) : 
  W = 14 := 
  by 
    sorry

end NUMINAMATH_GPT_workshop_workers_l850_85025


namespace NUMINAMATH_GPT_carter_stretching_legs_frequency_l850_85061

-- Given conditions
def tripDuration : ℤ := 14 * 60 -- in minutes
def foodStops : ℤ := 2
def gasStops : ℤ := 3
def pitStopDuration : ℤ := 20 -- in minutes
def totalTripDuration : ℤ := 18 * 60 -- in minutes

-- Prove that Carter stops to stretch his legs every 2 hours
theorem carter_stretching_legs_frequency :
  ∃ (stretchingStops : ℤ), (totalTripDuration - tripDuration = (foodStops + gasStops + stretchingStops) * pitStopDuration) ∧
    (stretchingStops * pitStopDuration = totalTripDuration - (tripDuration + (foodStops + gasStops) * pitStopDuration)) ∧
    (14 / stretchingStops = 2) :=
by sorry

end NUMINAMATH_GPT_carter_stretching_legs_frequency_l850_85061


namespace NUMINAMATH_GPT_expectation_of_X_l850_85020

-- Conditions:
-- Defect rate of the batch of products is 0.05
def defect_rate : ℚ := 0.05

-- 5 items are randomly selected for quality inspection
def n : ℕ := 5

-- The probability of obtaining a qualified product in each trial
def P : ℚ := 1 - defect_rate

-- Question:
-- The random variable X, representing the number of qualified products, follows a binomial distribution.
-- Expectation of X
def expectation_X : ℚ := n * P

-- Prove that the mathematical expectation E(X) is equal to 4.75
theorem expectation_of_X :
  expectation_X = 4.75 := 
sorry

end NUMINAMATH_GPT_expectation_of_X_l850_85020


namespace NUMINAMATH_GPT_chris_current_age_l850_85009

def praveens_age_after_10_years (P : ℝ) : ℝ := P + 10
def praveens_age_3_years_back (P : ℝ) : ℝ := P - 3

def praveens_age_condition (P : ℝ) : Prop :=
  praveens_age_after_10_years P = 3 * praveens_age_3_years_back P

def chris_age (P : ℝ) : ℝ := (P - 4) - 2

theorem chris_current_age (P : ℝ) (h₁ : praveens_age_condition P) :
  chris_age P = 3.5 :=
sorry

end NUMINAMATH_GPT_chris_current_age_l850_85009


namespace NUMINAMATH_GPT_arrangement_is_correct_l850_85008

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end NUMINAMATH_GPT_arrangement_is_correct_l850_85008


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l850_85067

theorem isosceles_triangle_perimeter (a b c : ℝ) 
  (h1 : a = 4 ∨ b = 4 ∨ c = 4) 
  (h2 : a = 8 ∨ b = 8 ∨ c = 8) 
  (isosceles : a = b ∨ b = c ∨ a = c) : 
  a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l850_85067


namespace NUMINAMATH_GPT_f_2021_value_l850_85082

def A : Set ℚ := {x | x ≠ -1 ∧ x ≠ 0}

def f (x : ℚ) : ℝ := sorry -- Placeholder for function definition with its properties

axiom f_property : ∀ x ∈ A, f x + f (1 + 1 / x) = 1 / 2 * Real.log (|x|)

theorem f_2021_value : f 2021 = 1 / 2 * Real.log 2021 :=
by
  sorry

end NUMINAMATH_GPT_f_2021_value_l850_85082


namespace NUMINAMATH_GPT_tom_initial_money_l850_85018

-- Defining the given values
def super_nintendo_value : ℝ := 150
def store_percentage : ℝ := 0.80
def nes_price : ℝ := 160
def game_value : ℝ := 30
def change_received : ℝ := 10

-- Calculate the credit received for the Super Nintendo
def credit_received := store_percentage * super_nintendo_value

-- Calculate the remaining amount Tom needs to pay for the NES after using the credit
def remaining_amount := nes_price - credit_received

-- Calculate the total amount Tom needs to pay, including the game value
def total_amount_needed := remaining_amount + game_value

-- Proving that the initial money Tom gave is $80
theorem tom_initial_money : total_amount_needed + change_received = 80 :=
by
    sorry

end NUMINAMATH_GPT_tom_initial_money_l850_85018


namespace NUMINAMATH_GPT_machine_A_sprockets_per_hour_l850_85099

theorem machine_A_sprockets_per_hour :
  ∃ (A : ℝ), 
    (∃ (G : ℝ), 
      (G = 1.10 * A) ∧ 
      (∃ (T : ℝ), 
        (660 = A * (T + 10)) ∧ 
        (660 = G * T) 
      )
    ) ∧ 
    (A = 6) :=
by
  -- Conditions and variables will be introduced here...
  -- Proof can be implemented here
  sorry

end NUMINAMATH_GPT_machine_A_sprockets_per_hour_l850_85099


namespace NUMINAMATH_GPT_fraction_equality_l850_85048

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l850_85048


namespace NUMINAMATH_GPT_polynomial_identity_solution_l850_85000

theorem polynomial_identity_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) ↔ (∃ a : ℝ, P = Polynomial.C a * (Polynomial.X ^ 2 - Polynomial.X)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_solution_l850_85000


namespace NUMINAMATH_GPT_debate_organizing_committees_count_l850_85031

theorem debate_organizing_committees_count :
    ∃ (n : ℕ), n = 5 * (Nat.choose 8 4) * (Nat.choose 8 3)^4 ∧ n = 3442073600 :=
by
  sorry

end NUMINAMATH_GPT_debate_organizing_committees_count_l850_85031


namespace NUMINAMATH_GPT_infinite_positive_integer_solutions_l850_85069

theorem infinite_positive_integer_solutions : ∃ (a b c : ℕ), (∃ k : ℕ, k > 0 ∧ a = k * (k^3 + 1990) ∧ b = (k^3 + 1990) ∧ c = (k^3 + 1990)) ∧ (a^3 + 1990 * b^3) = c^4 :=
sorry

end NUMINAMATH_GPT_infinite_positive_integer_solutions_l850_85069


namespace NUMINAMATH_GPT_not_geometric_sequence_of_transformed_l850_85039

theorem not_geometric_sequence_of_transformed (a b c : ℝ) (q : ℝ) (hq : q ≠ 1) 
  (h_geometric : b = a * q ∧ c = b * q) :
  ¬ (∃ q' : ℝ, 1 - b = (1 - a) * q' ∧ 1 - c = (1 - b) * q') :=
by
  sorry

end NUMINAMATH_GPT_not_geometric_sequence_of_transformed_l850_85039


namespace NUMINAMATH_GPT_expression_indeterminate_l850_85057

-- Given variables a, b, c, d which are real numbers
variables {a b c d : ℝ}

-- Statement asserting that the expression is indeterminate under given conditions
theorem expression_indeterminate
  (h : true) :
  ¬∃ k, (a^2 + b^2 - c^2 - 2 * b * d)/(a^2 + c^2 - b^2 - 2 * c * d) = k :=
sorry

end NUMINAMATH_GPT_expression_indeterminate_l850_85057


namespace NUMINAMATH_GPT_henry_books_l850_85045

def initial_books := 99
def boxes := 3
def books_per_box := 15
def room_books := 21
def coffee_table_books := 4
def kitchen_books := 18
def picked_books := 12

theorem henry_books :
  (initial_books - (boxes * books_per_box + room_books + coffee_table_books + kitchen_books) + picked_books) = 23 :=
by
  sorry

end NUMINAMATH_GPT_henry_books_l850_85045


namespace NUMINAMATH_GPT_triangle_perimeter_l850_85002

theorem triangle_perimeter
  (a b : ℕ) (c : ℕ) 
  (h_side1 : a = 3)
  (h_side2 : b = 4)
  (h_third_side : c^2 - 13 * c + 40 = 0)
  (h_valid_triangle : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
  a + b + c = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_perimeter_l850_85002


namespace NUMINAMATH_GPT_find_k_l850_85066

theorem find_k (k : ℝ) (h : (3, 1) ∈ {(x, y) | y = k * x - 2} ∧ k ≠ 0) : k = 1 :=
by sorry

end NUMINAMATH_GPT_find_k_l850_85066


namespace NUMINAMATH_GPT_thursday_loaves_baked_l850_85019

theorem thursday_loaves_baked (wednesday friday saturday sunday monday : ℕ) (p1 : wednesday = 5) (p2 : friday = 10) (p3 : saturday = 14) (p4 : sunday = 19) (p5 : monday = 25) : 
  ∃ thursday : ℕ, thursday = 11 := 
by 
  sorry

end NUMINAMATH_GPT_thursday_loaves_baked_l850_85019


namespace NUMINAMATH_GPT_abc_inequality_l850_85063

open Real

noncomputable def posReal (x : ℝ) : Prop := x > 0

theorem abc_inequality (a b c : ℝ) 
  (hCond1 : posReal a) 
  (hCond2 : posReal b) 
  (hCond3 : posReal c) 
  (hCond4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_abc_inequality_l850_85063


namespace NUMINAMATH_GPT_unique_solution_exists_l850_85064

theorem unique_solution_exists :
  ∃ (a b c d e : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a + b = 1/7 * (c + d + e) ∧
  a + c = 1/5 * (b + d + e) ∧
  (a, b, c, d, e) = (1, 2, 3, 9, 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_exists_l850_85064


namespace NUMINAMATH_GPT_find_x_value_l850_85086

def solve_for_x (a b x : ℝ) (rectangle_perimeter triangle_height equated_areas : Prop) :=
  rectangle_perimeter -> triangle_height -> equated_areas -> x = 20 / 3

-- Definitions of the conditions
def rectangle_perimeter (a b : ℝ) : Prop := 2 * (a + b) = 60
def triangle_height : Prop := 60 > 0
def equated_areas (a b x : ℝ) : Prop := a * b = 30 * x

theorem find_x_value :
  ∃ a b x : ℝ, solve_for_x a b x (rectangle_perimeter a b) triangle_height (equated_areas a b x) :=
  sorry

end NUMINAMATH_GPT_find_x_value_l850_85086


namespace NUMINAMATH_GPT_calc_expr_eq_l850_85021

-- Define the polynomial and expression
def expr (x : ℝ) : ℝ := x * (x * (x * (3 - 2 * x) - 4) + 8) + 3 * x^2

theorem calc_expr_eq (x : ℝ) : expr x = -2 * x^4 + 3 * x^3 - x^2 + 8 * x := 
by
  sorry

end NUMINAMATH_GPT_calc_expr_eq_l850_85021


namespace NUMINAMATH_GPT_census_entirety_is_population_l850_85072

-- Define the options as a type
inductive CensusOptions
| Part
| Whole
| Individual
| Population

-- Define the condition: the entire object under investigation in a census
def entirety_of_objects_under_investigation : CensusOptions := CensusOptions.Population

-- Prove that the entirety of objects under investigation in a census is called Population
theorem census_entirety_is_population :
  entirety_of_objects_under_investigation = CensusOptions.Population :=
sorry

end NUMINAMATH_GPT_census_entirety_is_population_l850_85072


namespace NUMINAMATH_GPT_num_k_vals_l850_85046

-- Definitions of the conditions
def div_by_7 (n k : ℕ) : Prop :=
  (2 * 3^(6*n) + k * 2^(3*n + 1) - 1) % 7 = 0

-- Main theorem statement
theorem num_k_vals : 
  ∃ (S : Finset ℕ), (∀ k ∈ S, k < 100 ∧ ∀ n, div_by_7 n k) ∧ S.card = 14 := 
by
  sorry

end NUMINAMATH_GPT_num_k_vals_l850_85046


namespace NUMINAMATH_GPT_determinant_zero_l850_85058

def matrix_determinant (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1, x, y + z],
    ![1, x + y, z],
    ![1, x + z, y]
  ]

theorem determinant_zero (x y z : ℝ) : matrix_determinant x y z = 0 := 
by
  sorry

end NUMINAMATH_GPT_determinant_zero_l850_85058


namespace NUMINAMATH_GPT_books_left_over_after_repacking_l850_85089

theorem books_left_over_after_repacking :
  ((1335 * 39) % 40) = 25 :=
sorry

end NUMINAMATH_GPT_books_left_over_after_repacking_l850_85089


namespace NUMINAMATH_GPT_sum_of_factors_of_120_is_37_l850_85023

theorem sum_of_factors_of_120_is_37 :
  ∃ a b c d e : ℤ, (a * b = 120) ∧ (b = a + 1) ∧ (c * d * e = 120) ∧ (d = c + 1) ∧ (e = d + 1) ∧ (a + b + c + d + e = 37) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_factors_of_120_is_37_l850_85023


namespace NUMINAMATH_GPT_volume_ratio_of_cubes_l850_85038

def cube_volume (a : ℝ) : ℝ := a ^ 3

theorem volume_ratio_of_cubes :
  cube_volume 3 / cube_volume 18 = 1 / 216 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cubes_l850_85038


namespace NUMINAMATH_GPT_rectangle_length_35_l850_85062

theorem rectangle_length_35
  (n_rectangles : ℕ) (area_abcd : ℝ) (rect_length_multiple : ℕ) (rect_width_multiple : ℕ) 
  (n_rectangles_eq : n_rectangles = 6)
  (area_abcd_eq : area_abcd = 4800)
  (rect_length_multiple_eq : rect_length_multiple = 3)
  (rect_width_multiple_eq : rect_width_multiple = 2) :
  ∃ y : ℝ, round y = 35 ∧ y^2 * (4/3) = area_abcd :=
by
  sorry


end NUMINAMATH_GPT_rectangle_length_35_l850_85062


namespace NUMINAMATH_GPT_percentage_decrease_is_14_percent_l850_85027

-- Definitions based on conditions
def original_price_per_pack : ℚ := 7 / 3
def new_price_per_pack : ℚ := 8 / 4

-- Statement to prove that percentage decrease is 14%
theorem percentage_decrease_is_14_percent :
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100 = 14 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_14_percent_l850_85027


namespace NUMINAMATH_GPT_root_reciprocals_identity_l850_85011

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  (a + b + c = 12) ∧ (a * b + b * c + c * a = 20) ∧ (a * b * c = -5)

theorem root_reciprocals_identity (a b c : ℝ) (h : cubic_roots a b c) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 20.8 :=
by
  sorry

end NUMINAMATH_GPT_root_reciprocals_identity_l850_85011


namespace NUMINAMATH_GPT_lice_checks_time_in_hours_l850_85035

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end NUMINAMATH_GPT_lice_checks_time_in_hours_l850_85035


namespace NUMINAMATH_GPT_problem1_problem2_l850_85073

-- Definition of the function
def f (a x : ℝ) := x^2 + a * x + 3

-- Problem statement 1: Prove that if f(x) ≥ a for all x ∈ ℜ, then a ≤ 3.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x ≥ a) → a ≤ 3 := sorry

-- Problem statement 2: Prove that if f(x) ≥ a for all x ∈ [-2, 2], then -6 ≤ a ≤ 2.
theorem problem2 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≥ a) → -6 ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l850_85073


namespace NUMINAMATH_GPT_range_of_a_l850_85036

-- Definitions of propositions p and q

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem stating the range of values for a given p ∧ q is true

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≤ -2 ∨ a = 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l850_85036


namespace NUMINAMATH_GPT_max_value_l850_85059

theorem max_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2 * b + 3 * c ≤ 30.333 :=
by
  sorry

end NUMINAMATH_GPT_max_value_l850_85059


namespace NUMINAMATH_GPT_total_miles_traveled_l850_85078

noncomputable def initial_fee : ℝ := 2.0
noncomputable def charge_per_2_5_mile : ℝ := 0.35
noncomputable def total_charge : ℝ := 5.15

theorem total_miles_traveled :
  ∃ (miles : ℝ), total_charge = initial_fee + (charge_per_2_5_mile * miles * (5 / 2)) ∧ miles = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_total_miles_traveled_l850_85078


namespace NUMINAMATH_GPT_find_missing_number_l850_85091

theorem find_missing_number:
  ∃ x : ℕ, (306 / 34) * 15 + x = 405 := sorry

end NUMINAMATH_GPT_find_missing_number_l850_85091


namespace NUMINAMATH_GPT_positional_relationship_l850_85068

-- Defining the concepts of parallelism, containment, and positional relationships
structure Line -- subtype for a Line
structure Plane -- subtype for a Plane

-- Definitions and Conditions
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry  -- A line being parallel to a plane
def is_contained_in (l : Line) (p : Plane) : Prop := sorry  -- A line being contained within a plane
def are_skew (l₁ l₂ : Line) : Prop := sorry  -- Two lines being skew
def are_parallel (l₁ l₂ : Line) : Prop := sorry  -- Two lines being parallel

-- Given conditions
variables (a b : Line) (α : Plane)
axiom Ha : is_parallel_to a α
axiom Hb : is_contained_in b α

-- The theorem to be proved
theorem positional_relationship (a b : Line) (α : Plane) 
  (Ha : is_parallel_to a α) 
  (Hb : is_contained_in b α) : 
  (are_skew a b ∨ are_parallel a b) :=
sorry

end NUMINAMATH_GPT_positional_relationship_l850_85068


namespace NUMINAMATH_GPT_Tim_scores_expected_value_l850_85024

theorem Tim_scores_expected_value :
  let LAIMO := 15
  let FARML := 10
  let DOMO := 50
  let p := 1 / 3
  let expected_LAIMO := LAIMO * p
  let expected_FARML := FARML * p
  let expected_DOMO := DOMO * p
  expected_LAIMO + expected_FARML + expected_DOMO = 25 :=
by
  -- The Lean proof would go here
  sorry

end NUMINAMATH_GPT_Tim_scores_expected_value_l850_85024


namespace NUMINAMATH_GPT_period_tan_2x_3_l850_85037

noncomputable def period_of_tan_transformed : Real :=
  let period_tan := Real.pi
  let coeff := 2/3
  (period_tan / coeff : Real)

theorem period_tan_2x_3 : period_of_tan_transformed = 3 * Real.pi / 2 :=
  sorry

end NUMINAMATH_GPT_period_tan_2x_3_l850_85037


namespace NUMINAMATH_GPT_algebraic_expression_value_l850_85097

-- Define the premises as a Lean statement
theorem algebraic_expression_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a * (b + c) + b * (a + c) + c * (a + b) = -1 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l850_85097


namespace NUMINAMATH_GPT_average_viewing_times_correct_l850_85022

-- Define the viewing times for each family member per week
def Evelyn_week1 : ℕ := 10
def Evelyn_week2 : ℕ := 8
def Evelyn_week3 : ℕ := 6

def Eric_week1 : ℕ := 8
def Eric_week2 : ℕ := 6
def Eric_week3 : ℕ := 5

def Kate_week2_episodes : ℕ := 12
def minutes_per_episode : ℕ := 40
def Kate_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def Kate_week3 : ℕ := 4

def John_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def John_week3 : ℕ := 8

-- Calculate the averages
def average (total : ℚ) (weeks : ℚ) : ℚ := total / weeks

-- Define the total viewing time for each family member
def Evelyn_total : ℕ := Evelyn_week1 + Evelyn_week2 + Evelyn_week3
def Eric_total : ℕ := Eric_week1 + Eric_week2 + Eric_week3
def Kate_total : ℕ := 0 + Kate_week2 + Kate_week3
def John_total : ℕ := 0 + John_week2 + John_week3

-- Define the expected averages
def Evelyn_expected_avg : ℚ := 8
def Eric_expected_avg : ℚ := 19 / 3
def Kate_expected_avg : ℚ := 4
def John_expected_avg : ℚ := 16 / 3

-- The theorem to prove that the calculated averages are correct
theorem average_viewing_times_correct :
  average Evelyn_total 3 = Evelyn_expected_avg ∧
  average Eric_total 3 = Eric_expected_avg ∧
  average Kate_total 3 = Kate_expected_avg ∧
  average John_total 3 = John_expected_avg :=
by sorry

end NUMINAMATH_GPT_average_viewing_times_correct_l850_85022


namespace NUMINAMATH_GPT_find_p_l850_85007

theorem find_p (p : ℝ) (h1 : (1/2) * 15 * (3 + 15) - ((1/2) * 3 * (15 - p) + (1/2) * 15 * p) = 40) : 
  p = 12.0833 :=
by sorry

end NUMINAMATH_GPT_find_p_l850_85007


namespace NUMINAMATH_GPT_taxi_cost_per_mile_l850_85051

variable (x : ℝ)

-- Mike's total cost
def Mike_total_cost := 2.50 + 36 * x

-- Annie's total cost
def Annie_total_cost := 2.50 + 5.00 + 16 * x

-- The primary theorem to prove
theorem taxi_cost_per_mile : Mike_total_cost x = Annie_total_cost x → x = 0.25 := by
  sorry

end NUMINAMATH_GPT_taxi_cost_per_mile_l850_85051


namespace NUMINAMATH_GPT_notebook_cost_l850_85047

theorem notebook_cost (n p : ℝ) (h1 : n + p = 2.40) (h2 : n = 2 + p) : n = 2.20 := by
  sorry

end NUMINAMATH_GPT_notebook_cost_l850_85047


namespace NUMINAMATH_GPT_part_I_min_value_part_II_a_range_l850_85080

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) - abs (x + 3)

theorem part_I_min_value (x : ℝ) : f x 1 ≥ -7 / 2 :=
by sorry 

theorem part_II_a_range (x a : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 3) (hf : f x a ≤ 4) : -4 ≤ a ∧ a ≤ 7 :=
by sorry

end NUMINAMATH_GPT_part_I_min_value_part_II_a_range_l850_85080


namespace NUMINAMATH_GPT_cost_fly_D_to_E_l850_85041

-- Definitions for the given conditions
def distance_DE : ℕ := 4750
def cost_per_km_plane : ℝ := 0.12
def booking_fee_plane : ℝ := 150

-- The proof statement about the total cost
theorem cost_fly_D_to_E : (distance_DE * cost_per_km_plane + booking_fee_plane = 720) :=
by sorry

end NUMINAMATH_GPT_cost_fly_D_to_E_l850_85041


namespace NUMINAMATH_GPT_tony_bread_slices_left_l850_85075

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end NUMINAMATH_GPT_tony_bread_slices_left_l850_85075


namespace NUMINAMATH_GPT_mandy_gets_15_pieces_l850_85006

def initial_pieces : ℕ := 75
def michael_takes (pieces : ℕ) : ℕ := pieces / 3
def paige_takes (pieces : ℕ) : ℕ := (pieces - michael_takes pieces) / 2
def ben_takes (pieces : ℕ) : ℕ := 2 * (pieces - michael_takes pieces - paige_takes pieces) / 5
def mandy_takes (pieces : ℕ) : ℕ := pieces - michael_takes pieces - paige_takes pieces - ben_takes pieces

theorem mandy_gets_15_pieces :
  mandy_takes initial_pieces = 15 :=
by
  sorry

end NUMINAMATH_GPT_mandy_gets_15_pieces_l850_85006


namespace NUMINAMATH_GPT_wallet_amount_l850_85032

-- Definitions of given conditions
def num_toys := 28
def cost_per_toy := 10
def num_teddy_bears := 20
def cost_per_teddy_bear := 15

-- Calculation of total costs
def total_cost_of_toys := num_toys * cost_per_toy
def total_cost_of_teddy_bears := num_teddy_bears * cost_per_teddy_bear

-- Total amount of money in Louise's wallet
def total_cost := total_cost_of_toys + total_cost_of_teddy_bears

-- Proof that the total cost is $580
theorem wallet_amount : total_cost = 580 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_wallet_amount_l850_85032


namespace NUMINAMATH_GPT_solve_system_of_equations_l850_85028

theorem solve_system_of_equations (x y_1 y_2 y_3: ℝ) (n : ℤ) (h1 : -3 ≤ n) (h2 : n ≤ 3)
  (h_eq1 : (1 - x^2) * y_1 = 2 * x)
  (h_eq2 : (1 - y_1^2) * y_2 = 2 * y_1)
  (h_eq3 : (1 - y_2^2) * y_3 = 2 * y_2)
  (h_eq4 : y_3 = x) :
  y_1 = Real.tan (2 * n * Real.pi / 7) ∧
  y_2 = Real.tan (4 * n * Real.pi / 7) ∧
  y_3 = Real.tan (n * Real.pi / 7) ∧
  x = Real.tan (n * Real.pi / 7) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l850_85028


namespace NUMINAMATH_GPT_inequality_abc_lt_l850_85026

variable (a b c : ℝ)

theorem inequality_abc_lt:
  c > b → b > a → a^2 * b + b^2 * c + c^2 * a < a * b^2 + b * c^2 + c * a^2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_inequality_abc_lt_l850_85026


namespace NUMINAMATH_GPT_intersection_M_N_l850_85015

noncomputable def set_M : Set ℚ := {α | ∃ k : ℤ, α = k * 90 - 36}
noncomputable def set_N : Set ℚ := {α | -180 < α ∧ α < 180}

theorem intersection_M_N : set_M ∩ set_N = {-36, 54, 144, -126} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l850_85015


namespace NUMINAMATH_GPT_cube_expression_l850_85014

theorem cube_expression (a : ℝ) (h : (a + 1/a)^2 = 5) : a^3 + 1/a^3 = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_cube_expression_l850_85014


namespace NUMINAMATH_GPT_find_cos_alpha_l850_85013

theorem find_cos_alpha (α : ℝ) (h : (1 - Real.cos α) / Real.sin α = 3) : Real.cos α = -4/5 :=
by
  sorry

end NUMINAMATH_GPT_find_cos_alpha_l850_85013


namespace NUMINAMATH_GPT_middle_card_number_is_6_l850_85044

noncomputable def middle_card_number : ℕ :=
  6

theorem middle_card_number_is_6 (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 17)
  (casey_cannot_determine : ∀ (x : ℕ), (a = x) → ∃ (y z : ℕ), y ≠ z ∧ a + y + z = 17 ∧ a < y ∧ y < z)
  (tracy_cannot_determine : ∀ (x : ℕ), (c = x) → ∃ (y z : ℕ), y ≠ z ∧ y + z + c = 17 ∧ y < z ∧ z < c)
  (stacy_cannot_determine : ∀ (x : ℕ), (b = x) → ∃ (y z : ℕ), y ≠ z ∧ y + b + z = 17 ∧ y < b ∧ b < z) : 
  b = middle_card_number :=
sorry

end NUMINAMATH_GPT_middle_card_number_is_6_l850_85044


namespace NUMINAMATH_GPT_quadratic_m_value_l850_85017

theorem quadratic_m_value (m : ℕ) :
  (∃ x : ℝ, x^(m + 1) - (m + 1) * x - 2 = 0) →
  m + 1 = 2 →
  m = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_m_value_l850_85017


namespace NUMINAMATH_GPT_erasers_in_each_box_l850_85076

theorem erasers_in_each_box (boxes : ℕ) (price_per_eraser : ℚ) (total_money_made : ℚ) (total_erasers_sold : ℕ) (erasers_per_box : ℕ) :
  boxes = 48 → price_per_eraser = 0.75 → total_money_made = 864 → total_erasers_sold = 1152 → total_erasers_sold / boxes = erasers_per_box → erasers_per_box = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_erasers_in_each_box_l850_85076


namespace NUMINAMATH_GPT_no_solution_for_99_l850_85060

theorem no_solution_for_99 :
  ∃ n : ℕ, (¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = n) ∧
  (∀ m : ℕ, n < m → ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = m) ∧
  n = 99 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_99_l850_85060


namespace NUMINAMATH_GPT_seashells_initial_count_l850_85095

theorem seashells_initial_count (S : ℕ)
  (h1 : S - 70 = 2 * 55) : S = 180 :=
by
  sorry

end NUMINAMATH_GPT_seashells_initial_count_l850_85095


namespace NUMINAMATH_GPT_arith_seq_ratio_l850_85049

variables {a₁ d : ℝ} (h₁ : d ≠ 0) (h₂ : (a₁ + 2*d)^2 ≠ a₁ * (a₁ + 8*d))

theorem arith_seq_ratio:
  (a₁ + 2*d) / (a₁ + 5*d) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_arith_seq_ratio_l850_85049


namespace NUMINAMATH_GPT_side_length_of_square_l850_85004

theorem side_length_of_square (s : ℝ) (h : s^2 = 100) : s = 10 := 
sorry

end NUMINAMATH_GPT_side_length_of_square_l850_85004


namespace NUMINAMATH_GPT_sum_and_round_to_nearest_ten_l850_85083

/-- A function to round a number to the nearest ten -/
def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + 10 - n % 10

/-- The sum of 54 and 29 rounded to the nearest ten is 80 -/
theorem sum_and_round_to_nearest_ten : round_to_nearest_ten (54 + 29) = 80 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_round_to_nearest_ten_l850_85083


namespace NUMINAMATH_GPT_initial_number_of_eggs_l850_85001

theorem initial_number_of_eggs (eggs_taken harry_eggs eggs_left initial_eggs : ℕ)
    (h1 : harry_eggs = 5)
    (h2 : eggs_left = 42)
    (h3 : initial_eggs = eggs_left + harry_eggs) : 
    initial_eggs = 47 := by
  sorry

end NUMINAMATH_GPT_initial_number_of_eggs_l850_85001


namespace NUMINAMATH_GPT_sequence_condition_satisfies_l850_85093

def seq_prove_abs_lt_1 (a : ℕ → ℝ) : Prop :=
  (∃ i : ℕ, |a i| < 1)

theorem sequence_condition_satisfies (a : ℕ → ℝ)
  (h1 : a 1 * a 2 < 0)
  (h2 : ∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧ (∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)) :
  seq_prove_abs_lt_1 a :=
by
  sorry

end NUMINAMATH_GPT_sequence_condition_satisfies_l850_85093


namespace NUMINAMATH_GPT_necessary_condition_l850_85052

variables (a b : ℝ)

theorem necessary_condition (h : a > b) : a > b - 1 :=
sorry

end NUMINAMATH_GPT_necessary_condition_l850_85052


namespace NUMINAMATH_GPT_martin_crayons_l850_85029

theorem martin_crayons : (8 * 7 = 56) := by
  sorry

end NUMINAMATH_GPT_martin_crayons_l850_85029


namespace NUMINAMATH_GPT_positive_integer_solutions_count_l850_85087

theorem positive_integer_solutions_count :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ x + y + z = 2010) → (336847 = 336847) :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integer_solutions_count_l850_85087


namespace NUMINAMATH_GPT_power_of_power_l850_85055

theorem power_of_power {a : ℝ} : (a^2)^3 = a^6 := 
by
  sorry

end NUMINAMATH_GPT_power_of_power_l850_85055


namespace NUMINAMATH_GPT_area_of_circumscribed_circle_l850_85042

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end NUMINAMATH_GPT_area_of_circumscribed_circle_l850_85042


namespace NUMINAMATH_GPT_find_u_plus_v_l850_85090

theorem find_u_plus_v (u v : ℤ) (huv : 0 < v ∧ v < u) (h_area : u * u + 3 * u * v = 451) : u + v = 21 := 
sorry

end NUMINAMATH_GPT_find_u_plus_v_l850_85090


namespace NUMINAMATH_GPT_comb_identity_a_l850_85071

theorem comb_identity_a (r m k : ℕ) (h : 0 ≤ k ∧ k ≤ m ∧ m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end NUMINAMATH_GPT_comb_identity_a_l850_85071


namespace NUMINAMATH_GPT_triangle_sides_l850_85065

theorem triangle_sides (a : ℕ) (h : a > 0) : 
  (a + 1) + (a + 2) > (a + 3) ∧ (a + 1) + (a + 3) > (a + 2) ∧ (a + 2) + (a + 3) > (a + 1) := 
by 
  sorry

end NUMINAMATH_GPT_triangle_sides_l850_85065


namespace NUMINAMATH_GPT_sequence_term_formula_l850_85096

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 1/2 - 1/2 * a n

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n ≥ 2, a n = r * a (n - 1)

theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 1, S n = 1/2 - 1/2 * a n) →
  (S 1 = 1/2 - 1/2 * a 1) →
  a 1 = 1/3 →
  (∀ n ≥ 2, S n = 1/2 - 1/2 * (a n) → S (n - 1) = 1/2 - 1/2 * (a (n - 1)) → a n = 1/3 * a (n-1)) →
  ∀ n, a n = (1/3)^n :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sequence_term_formula_l850_85096


namespace NUMINAMATH_GPT_find_ABC_l850_85098

theorem find_ABC :
    ∃ (A B C : ℚ), 
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 → 
        (x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) 
    ∧ A = 5 / 3 ∧ B = -7 / 2 ∧ C = 8 / 3 := 
sorry

end NUMINAMATH_GPT_find_ABC_l850_85098


namespace NUMINAMATH_GPT_train_a_distance_at_meeting_l850_85034

-- Define the problem conditions as constants
def distance := 75 -- distance between start points of Train A and B
def timeA := 3 -- time taken by Train A to complete the trip in hours
def timeB := 2 -- time taken by Train B to complete the trip in hours

-- Calculate the speeds
def speedA := distance / timeA -- speed of Train A in miles per hour
def speedB := distance / timeB -- speed of Train B in miles per hour

-- Calculate the combined speed and time to meet
def combinedSpeed := speedA + speedB
def timeToMeet := distance / combinedSpeed

-- Define the distance traveled by Train A at the time of meeting
def distanceTraveledByTrainA := speedA * timeToMeet

-- Theorem stating Train A has traveled 30 miles when it met Train B
theorem train_a_distance_at_meeting : distanceTraveledByTrainA = 30 := by
  sorry

end NUMINAMATH_GPT_train_a_distance_at_meeting_l850_85034


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l850_85054

theorem cost_of_one_dozen_pens
  (p q : ℕ)
  (h1 : 3 * p + 5 * q = 240)
  (h2 : p = 5 * q) :
  12 * p = 720 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l850_85054


namespace NUMINAMATH_GPT_find_c_l850_85056

noncomputable def parabola_equation (a b c y : ℝ) : ℝ :=
  a * y^2 + b * y + c

theorem find_c (a b c : ℝ) (h_vertex : (-4, 2) = (-4, 2)) (h_point : (-2, 4) = (-2, 4)) :
  ∃ c : ℝ, parabola_equation a b c 0 = -2 :=
  by {
    use -2,
    sorry
  }

end NUMINAMATH_GPT_find_c_l850_85056


namespace NUMINAMATH_GPT_kendra_bought_3_hats_l850_85040

-- Define the price of a wooden toy
def price_of_toy : ℕ := 20

-- Define the price of a hat
def price_of_hat : ℕ := 10

-- Define the amount Kendra went to the shop with
def initial_amount : ℕ := 100

-- Define the number of wooden toys Kendra bought
def number_of_toys : ℕ := 2

-- Define the amount of change Kendra received
def change_received : ℕ := 30

-- Prove that Kendra bought 3 hats
theorem kendra_bought_3_hats : 
  initial_amount - change_received - (number_of_toys * price_of_toy) = 3 * price_of_hat := by
  sorry

end NUMINAMATH_GPT_kendra_bought_3_hats_l850_85040


namespace NUMINAMATH_GPT_Gage_skating_time_l850_85084

theorem Gage_skating_time :
  let min_per_hr := 60
  let skating_6_days := 6 * (1 * min_per_hr + 20)
  let skating_4_days := 4 * (1 * min_per_hr + 35)
  let needed_total := 11 * 90
  let skating_10_days := skating_6_days + skating_4_days
  let minutes_on_eleventh_day := needed_total - skating_10_days
  minutes_on_eleventh_day = 130 :=
by
  sorry

end NUMINAMATH_GPT_Gage_skating_time_l850_85084


namespace NUMINAMATH_GPT_reduced_price_l850_85033

variable (P R : ℝ)
variable (price_reduction : R = 0.75 * P)
variable (buy_more_oil : 700 / R = 700 / P + 5)

theorem reduced_price (non_zero_P : P ≠ 0) (non_zero_R : R ≠ 0) : R = 35 := 
by
  sorry

end NUMINAMATH_GPT_reduced_price_l850_85033
