import Mathlib

namespace NUMINAMATH_GPT_concert_tickets_l204_20454

theorem concert_tickets (A C : ℕ) (h1 : C = 3 * A) (h2 : 7 * A + 3 * C = 6000) : A + C = 1500 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_concert_tickets_l204_20454


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_3_l204_20487

def p (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7

theorem remainder_when_divided_by_x_minus_3 : p 3 = 52 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_3_l204_20487


namespace NUMINAMATH_GPT_set_intersection_complement_eq_l204_20434

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem set_intersection_complement_eq {U : Set ℕ} {M : Set ℕ} {N : Set ℕ}
    (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {3, 4, 5}) :
    (U \ M) ∩ N = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_eq_l204_20434


namespace NUMINAMATH_GPT_scientific_notation_of_75500000_l204_20439

theorem scientific_notation_of_75500000 :
  ∃ (a : ℝ) (n : ℤ), 75500000 = a * 10 ^ n ∧ a = 7.55 ∧ n = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_75500000_l204_20439


namespace NUMINAMATH_GPT_evaluate_g_at_3_l204_20489

def g (x : ℝ) := 3 * x ^ 4 - 5 * x ^ 3 + 4 * x ^ 2 - 7 * x + 2

theorem evaluate_g_at_3 : g 3 = 125 :=
by
  -- Proof omitted for this exercise.
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l204_20489


namespace NUMINAMATH_GPT_relation_between_A_and_B_l204_20476

-- Define the sets A and B
def A : Set ℤ := { x | ∃ k : ℕ, x = 7 * k + 3 }
def B : Set ℤ := { x | ∃ k : ℤ, x = 7 * k - 4 }

-- Prove the relationship between A and B
theorem relation_between_A_and_B : A ⊆ B :=
by
  sorry

end NUMINAMATH_GPT_relation_between_A_and_B_l204_20476


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l204_20459

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l204_20459


namespace NUMINAMATH_GPT_no_possible_salary_distribution_l204_20475

theorem no_possible_salary_distribution (x y z : ℕ) (h1 : x + y + z = 13) (h2 : x + 3 * y + 5 * z = 200) : false :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_no_possible_salary_distribution_l204_20475


namespace NUMINAMATH_GPT_part_a_part_b_l204_20414

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem part_a :
  ¬∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^3) :=
sorry

theorem part_b :
  ∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^4) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l204_20414


namespace NUMINAMATH_GPT_multiples_of_2_correct_multiples_of_3_correct_l204_20426

def numbers : Set ℕ := {28, 35, 40, 45, 53, 10, 78}

def multiples_of_2_in_numbers : Set ℕ := {n ∈ numbers | n % 2 = 0}
def multiples_of_3_in_numbers : Set ℕ := {n ∈ numbers | n % 3 = 0}

theorem multiples_of_2_correct :
  multiples_of_2_in_numbers = {28, 40, 10, 78} :=
sorry

theorem multiples_of_3_correct :
  multiples_of_3_in_numbers = {45, 78} :=
sorry

end NUMINAMATH_GPT_multiples_of_2_correct_multiples_of_3_correct_l204_20426


namespace NUMINAMATH_GPT_valid_number_count_l204_20416

def is_valid_digit (d: Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def are_adjacent (d1 d2: Nat) : Bool :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 1) ∨
  (d1 = 5 ∧ (d2 = 1 ∨ d2 = 2)) ∨ 
  (d2 = 5 ∧ (d1 = 1 ∨ d1 = 2))

def count_valid_numbers : Nat :=
  sorry -- expression to count numbers according to given conditions.

theorem valid_number_count : count_valid_numbers = 36 :=
  sorry

end NUMINAMATH_GPT_valid_number_count_l204_20416


namespace NUMINAMATH_GPT_breakEvenBooks_l204_20417

theorem breakEvenBooks (FC VC_per_book SP : ℝ) (hFC : FC = 56430) (hVC : VC_per_book = 8.25) (hSP : SP = 21.75) :
  ∃ x : ℕ, FC + (VC_per_book * x) = SP * x ∧ x = 4180 :=
by {
  sorry
}

end NUMINAMATH_GPT_breakEvenBooks_l204_20417


namespace NUMINAMATH_GPT_tile_C_is_TileIV_l204_20430

-- Define the tiles with their respective sides
structure Tile :=
(top right bottom left : ℕ)

def TileI : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileII : Tile := { top := 6, right := 3, bottom := 1, left := 5 }
def TileIII : Tile := { top := 5, right := 7, bottom := 2, left := 3 }
def TileIV : Tile := { top := 3, right := 5, bottom := 7, left := 2 }

-- Define Rectangles for reasoning
inductive Rectangle
| A
| B
| C
| D

open Rectangle

-- Define the mathematical statement to prove
theorem tile_C_is_TileIV : ∃ tile, tile = TileIV :=
  sorry

end NUMINAMATH_GPT_tile_C_is_TileIV_l204_20430


namespace NUMINAMATH_GPT_initial_fraction_of_larger_jar_l204_20494

theorem initial_fraction_of_larger_jar (S L W : ℝ) 
  (h1 : W = 1/6 * S) 
  (h2 : W = 1/3 * L) : 
  W / L = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_initial_fraction_of_larger_jar_l204_20494


namespace NUMINAMATH_GPT_num_perfect_square_factors_of_360_l204_20448

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_num_perfect_square_factors_of_360_l204_20448


namespace NUMINAMATH_GPT_find_sum_of_squares_l204_20403

-- Definitions for the conditions: a, b, and c are different prime numbers,
-- and their product equals five times their sum.

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def condition (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a * b * c = 5 * (a + b + c)

-- Statement of the proof problem.
theorem find_sum_of_squares (a b c : ℕ) (h : condition a b c) : a^2 + b^2 + c^2 = 78 :=
sorry

end NUMINAMATH_GPT_find_sum_of_squares_l204_20403


namespace NUMINAMATH_GPT_purely_imaginary_condition_l204_20444

theorem purely_imaginary_condition (x : ℝ) :
  (z : ℂ) → (z = (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I) →
  (x = 1 ↔ (∃ y : ℂ, z = y * Complex.I)) :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_condition_l204_20444


namespace NUMINAMATH_GPT_average_episodes_per_year_l204_20469

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end NUMINAMATH_GPT_average_episodes_per_year_l204_20469


namespace NUMINAMATH_GPT_find_a1_l204_20452

theorem find_a1 (a_1 : ℕ) (S : ℕ → ℕ) (S_formula : ∀ n : ℕ, S n = (a_1 * (3^n - 1)) / 2)
  (a_4_eq : (S 4) - (S 3) = 54) : a_1 = 2 :=
  sorry

end NUMINAMATH_GPT_find_a1_l204_20452


namespace NUMINAMATH_GPT_fraction_problem_l204_20492

-- Define the fractions involved in the problem
def frac1 := 18 / 45
def frac2 := 3 / 8
def frac3 := 1 / 9

-- Define the expected result
def expected_result := 49 / 360

-- The proof statement
theorem fraction_problem : frac1 - frac2 + frac3 = expected_result := by
  sorry

end NUMINAMATH_GPT_fraction_problem_l204_20492


namespace NUMINAMATH_GPT_toy_train_produces_5_consecutive_same_tune_l204_20412

noncomputable def probability_same_tune (plays : ℕ) (p : ℚ) (tunes : ℕ) : ℚ :=
  p ^ plays

theorem toy_train_produces_5_consecutive_same_tune :
  probability_same_tune 5 (1/3) 3 = 1/243 :=
by
  sorry

end NUMINAMATH_GPT_toy_train_produces_5_consecutive_same_tune_l204_20412


namespace NUMINAMATH_GPT_tom_books_problem_l204_20437

theorem tom_books_problem 
  (original_books : ℕ)
  (books_sold : ℕ)
  (books_bought : ℕ)
  (h1 : original_books = 5)
  (h2 : books_sold = 4)
  (h3 : books_bought = 38) : 
  original_books - books_sold + books_bought = 39 :=
by
  sorry

end NUMINAMATH_GPT_tom_books_problem_l204_20437


namespace NUMINAMATH_GPT_regular_polygon_sides_l204_20472

theorem regular_polygon_sides (N : ℕ) (h : ∀ θ, θ = 140 → N * (180 -θ) = 360) : N = 9 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l204_20472


namespace NUMINAMATH_GPT_women_in_first_class_equals_22_l204_20400

def number_of_women (total_passengers : Nat) : Nat :=
  total_passengers * 50 / 100

def number_of_women_in_first_class (number_of_women : Nat) : Nat :=
  number_of_women * 15 / 100

theorem women_in_first_class_equals_22 (total_passengers : Nat) (h1 : total_passengers = 300) : 
  number_of_women_in_first_class (number_of_women total_passengers) = 22 :=
by
  sorry

end NUMINAMATH_GPT_women_in_first_class_equals_22_l204_20400


namespace NUMINAMATH_GPT_Alfonso_daily_earnings_l204_20467

-- Define the conditions given in the problem
def helmet_cost : ℕ := 340
def current_savings : ℕ := 40
def days_per_week : ℕ := 5
def weeks_to_work : ℕ := 10

-- Define the question as a property to prove
def daily_earnings : ℕ := 6

-- Prove that the daily earnings are $6 given the conditions
theorem Alfonso_daily_earnings :
  (helmet_cost - current_savings) / (days_per_week * weeks_to_work) = daily_earnings :=
by
  sorry

end NUMINAMATH_GPT_Alfonso_daily_earnings_l204_20467


namespace NUMINAMATH_GPT_max_notebooks_lucy_can_buy_l204_20483

-- Definitions given in the conditions
def lucyMoney : ℕ := 2145
def notebookCost : ℕ := 230

-- Theorem to prove the number of notebooks Lucy can buy
theorem max_notebooks_lucy_can_buy : lucyMoney / notebookCost = 9 := 
by
  sorry

end NUMINAMATH_GPT_max_notebooks_lucy_can_buy_l204_20483


namespace NUMINAMATH_GPT_three_hundredth_term_without_squares_l204_20427

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end NUMINAMATH_GPT_three_hundredth_term_without_squares_l204_20427


namespace NUMINAMATH_GPT_initial_people_per_column_l204_20424

theorem initial_people_per_column (P x : ℕ) (h1 : P = 16 * x) (h2 : P = 48 * 10) : x = 30 :=
by 
  sorry

end NUMINAMATH_GPT_initial_people_per_column_l204_20424


namespace NUMINAMATH_GPT_summation_eq_16_implies_x_eq_3_over_4_l204_20466

theorem summation_eq_16_implies_x_eq_3_over_4 (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x^n = 16) : x = 3 / 4 :=
sorry

end NUMINAMATH_GPT_summation_eq_16_implies_x_eq_3_over_4_l204_20466


namespace NUMINAMATH_GPT_minimum_steps_to_catch_thief_l204_20456

-- Definitions of positions A, B, C, D, etc., along the board
-- Assuming the positions and movement rules are predefined somewhere in the environment.
-- For a simple abstract model, we assume the following:
-- The positions are nodes in a graph, and each move is one step along the edges of this graph.

def Position : Type := String -- This can be refined to reflect the actual chessboard structure.
def neighbor (p1 p2 : Position) : Prop := sorry -- Predicate defining that p1 and p2 are neighbors.

-- Positions are predefined for simplicity.
def A : Position := "A"
def B : Position := "B"
def C : Position := "C"
def D : Position := "D"
def F : Position := "F"

-- Condition: policeman and thief take turns moving, starting with the policeman.
-- Initial positions of the policeman and the thief.
def policemanStart : Position := A
def thiefStart : Position := B

-- Statement: Prove that the policeman can catch the thief in a minimum of 4 moves.
theorem minimum_steps_to_catch_thief (policeman thief : Position) (turns : ℕ) :
  policeman = policemanStart →
  thief = thiefStart →
  (∀ t < turns, (neighbor policeman thief)) →
  (turns = 4) :=
sorry

end NUMINAMATH_GPT_minimum_steps_to_catch_thief_l204_20456


namespace NUMINAMATH_GPT_calculation_result_l204_20402

theorem calculation_result :
  3 * 3^3 + 4^7 / 4^5 = 97 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l204_20402


namespace NUMINAMATH_GPT_average_speed_round_trip_l204_20493

theorem average_speed_round_trip (D T : ℝ) (h1 : D = 51 * T) : (2 * D) / (3 * T) = 34 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l204_20493


namespace NUMINAMATH_GPT_max_marks_eq_300_l204_20495

-- Problem Statement in Lean 4

theorem max_marks_eq_300 (m_score p_score c_score : ℝ) 
    (m_percent p_percent c_percent : ℝ)
    (h1 : m_score = 285) (h2 : m_percent = 95) 
    (h3 : p_score = 270) (h4 : p_percent = 90) 
    (h5 : c_score = 255) (h6 : c_percent = 85) :
    (m_score / (m_percent / 100) = 300) ∧ 
    (p_score / (p_percent / 100) = 300) ∧ 
    (c_score / (c_percent / 100) = 300) :=
by
  sorry

end NUMINAMATH_GPT_max_marks_eq_300_l204_20495


namespace NUMINAMATH_GPT_find_constant_c_l204_20463

def f: ℝ → ℝ := sorry

noncomputable def constant_c := 8

theorem find_constant_c (h : ∀ x : ℝ, f x + 3 * f (constant_c - x) = x) (h2 : f 2 = 2) : 
  constant_c = 8 :=
sorry

end NUMINAMATH_GPT_find_constant_c_l204_20463


namespace NUMINAMATH_GPT_car_speed_l204_20468

theorem car_speed (distance time : ℝ) (h_distance : distance = 275) (h_time : time = 5) : (distance / time = 55) :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l204_20468


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_geometric_sequence_ratio_l204_20446

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :=
  ∀ n, a (n + 1) = a n * q
  
-- Prove the sum of the first n terms for an arithmetic sequence
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧ (∀ n, S n = (n * (3 + a (n + 1) - 1)) / 2) ∧ is_arithmetic_sequence a 4 → 
  S n = 2 * n^2 + n :=
sorry

-- Prove the range of the common ratio for a geometric sequence
theorem geometric_sequence_ratio (a : ℕ → ℕ) (S : ℕ → ℚ) (q : ℚ) :
  a 1 = 3 ∧ is_geometric_sequence a q ∧ ∃ lim : ℚ, (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) ∧ lim < 12 → 
  -1 < q ∧ q < 1 ∧ q ≠ 0 ∧ q < 3/4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_geometric_sequence_ratio_l204_20446


namespace NUMINAMATH_GPT_triplet_divisibility_cond_l204_20433

theorem triplet_divisibility_cond (a b c : ℤ) (hac : a ≥ 2) (hbc : b ≥ 2) (hcc : c ≥ 2) :
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ 
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 3 ∧ b = 15 ∧ c = 5) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 2 ∧ b = 8 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_GPT_triplet_divisibility_cond_l204_20433


namespace NUMINAMATH_GPT_probability_y_greater_than_x_equals_3_4_l204_20481

noncomputable def probability_y_greater_than_x : Real :=
  let total_area : Real := 1000 * 4034
  let triangle_area : Real := 0.5 * 1000 * (4034 - 1000)
  let rectangle_area : Real := 3034 * 4034
  let area_y_greater_than_x : Real := triangle_area + rectangle_area
  area_y_greater_than_x / total_area

theorem probability_y_greater_than_x_equals_3_4 :
  probability_y_greater_than_x = 3 / 4 :=
sorry

end NUMINAMATH_GPT_probability_y_greater_than_x_equals_3_4_l204_20481


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l204_20464

theorem sum_of_fourth_powers (n : ℤ) 
  (h : n * (n + 1) * (n + 2) = 12 * (n + (n + 1) + (n + 2))) : 
  (n^4 + (n + 1)^4 + (n + 2)^4) = 7793 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l204_20464


namespace NUMINAMATH_GPT_geometric_sequence_condition_l204_20458

theorem geometric_sequence_condition (A B q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn_def : ∀ n, S n = A * q^n + B) (hq_ne_zero : q ≠ 0) :
  (∀ n, a n = S n - S (n-1)) → (A = -B) ↔ (∀ n, a n = A * (q - 1) * q^(n-1)) := 
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l204_20458


namespace NUMINAMATH_GPT_base_prime_rep_360_l204_20411

-- Define the value 360 as n
def n : ℕ := 360

-- Function to compute the base prime representation.
noncomputable def base_prime_representation (n : ℕ) : ℕ :=
  -- Normally you'd implement the actual function to convert n to its base prime representation here
  sorry

-- The theorem statement claiming that the base prime representation of 360 is 213
theorem base_prime_rep_360 : base_prime_representation n = 213 := 
  sorry

end NUMINAMATH_GPT_base_prime_rep_360_l204_20411


namespace NUMINAMATH_GPT_amount_each_girl_gets_l204_20445

theorem amount_each_girl_gets
  (B G : ℕ) 
  (total_sum : ℝ)
  (amount_each_boy : ℝ)
  (sum_boys_girls : B + G = 100)
  (total_sum_distributed : total_sum = 312)
  (amount_boy : amount_each_boy = 3.60)
  (B_approx : B = 60) :
  (total_sum - amount_each_boy * B) / G = 2.40 := 
by 
  sorry

end NUMINAMATH_GPT_amount_each_girl_gets_l204_20445


namespace NUMINAMATH_GPT_no_natural_numbers_condition_l204_20438

theorem no_natural_numbers_condition :
  ¬ ∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018,
    ∃ k : ℕ, (a i) ^ 2018 + a ((i + 1) % 2018) = 5 ^ k :=
by sorry

end NUMINAMATH_GPT_no_natural_numbers_condition_l204_20438


namespace NUMINAMATH_GPT_average_weight_of_children_l204_20401

theorem average_weight_of_children :
  let ages := [3, 4, 5, 6, 7]
  let regression_equation (x : ℕ) := 3 * x + 5
  let average l := (l.foldr (· + ·) 0) / l.length
  average (List.map regression_equation ages) = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_children_l204_20401


namespace NUMINAMATH_GPT_complex_number_solution_l204_20486

open Complex

theorem complex_number_solution (z : ℂ) (h : (2 * z - I) * (2 - I) = 5) : 
  z = 1 + I :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l204_20486


namespace NUMINAMATH_GPT_inequality_l204_20404

noncomputable def x : ℝ := Real.sqrt 3
noncomputable def y : ℝ := Real.log 2 / Real.log 3
noncomputable def z : ℝ := Real.cos 2

theorem inequality : z < y ∧ y < x := by
  sorry

end NUMINAMATH_GPT_inequality_l204_20404


namespace NUMINAMATH_GPT_find_n_in_geometric_sequence_l204_20465

def geometric_sequence (an : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ q : ℝ, ∀ k : ℕ, an (k + 1) = an k * q

theorem find_n_in_geometric_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h3 : ∀ q : ℝ, a n = a 1 * a 2 * a 3 * a 4 * a 5) :
  n = 11 :=
sorry

end NUMINAMATH_GPT_find_n_in_geometric_sequence_l204_20465


namespace NUMINAMATH_GPT_joan_seashells_count_l204_20406

variable (total_seashells_given_to_sam : ℕ) (seashells_left_with_joan : ℕ)

theorem joan_seashells_count
  (h_given : total_seashells_given_to_sam = 43)
  (h_left : seashells_left_with_joan = 27) :
  total_seashells_given_to_sam + seashells_left_with_joan = 70 :=
sorry

end NUMINAMATH_GPT_joan_seashells_count_l204_20406


namespace NUMINAMATH_GPT_bandage_overlap_l204_20490

theorem bandage_overlap
  (n : ℕ) (l : ℝ) (total_length : ℝ) (required_length : ℝ)
  (h_n : n = 20) (h_l : l = 15.25) (h_required_length : required_length = 248) :
  (required_length = l * n - (n - 1) * 3) :=
by
  sorry

end NUMINAMATH_GPT_bandage_overlap_l204_20490


namespace NUMINAMATH_GPT_mean_score_of_all_students_l204_20491

-- Define the conditions as given in the problem
variables (M A : ℝ) (m a : ℝ)
  (hM : M = 90)
  (hA : A = 75)
  (hRatio : m / a = 2 / 5)

-- State the theorem which proves that the mean score of all students is 79
theorem mean_score_of_all_students (hM : M = 90) (hA : A = 75) (hRatio : m / a = 2 / 5) : 
  (36 * a + 75 * a) / ((2 / 5) * a + a) = 79 := 
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_mean_score_of_all_students_l204_20491


namespace NUMINAMATH_GPT_find_value_of_square_sums_l204_20477

variable (x y z : ℝ)

-- Define the conditions
def weighted_arithmetic_mean := (2 * x + 2 * y + 3 * z) / 8 = 9
def weighted_geometric_mean := Real.rpow (x^2 * y^2 * z^3) (1 / 7) = 6
def weighted_harmonic_mean := 7 / ((2 / x) + (2 / y) + (3 / z)) = 4

-- State the theorem to be proved
theorem find_value_of_square_sums
  (h1 : weighted_arithmetic_mean x y z)
  (h2 : weighted_geometric_mean x y z)
  (h3 : weighted_harmonic_mean x y z) :
  x^2 + y^2 + z^2 = 351 :=
by sorry

end NUMINAMATH_GPT_find_value_of_square_sums_l204_20477


namespace NUMINAMATH_GPT_correct_statements_l204_20408

-- Define the conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := 2 < a ∧ a < 3
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := a > 1

-- Prove the statements
theorem correct_statements (a : ℝ) : (p a → q a) ∧ (r a → q a) :=
by {
    sorry
}

end NUMINAMATH_GPT_correct_statements_l204_20408


namespace NUMINAMATH_GPT_year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l204_20418

-- Define what it means to be a leap year based on the given conditions.
def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ (y % 4 = 0 ∧ y % 100 ≠ 0)

-- Define the specific years we are examining.
def year_1800 := 1800
def year_1992 := 1992
def year_1994 := 1994
def year_2040 := 2040

-- Assertions about whether each year is a leap year or a common year
theorem year_1800_is_common : ¬ is_leap_year year_1800 :=
  by sorry

theorem year_1992_is_leap : is_leap_year year_1992 :=
  by sorry

theorem year_1994_is_common : ¬ is_leap_year year_1994 :=
  by sorry

theorem year_2040_is_leap : is_leap_year year_2040 :=
  by sorry

end NUMINAMATH_GPT_year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l204_20418


namespace NUMINAMATH_GPT_outdoor_section_area_l204_20442

theorem outdoor_section_area :
  ∀ (width length : ℕ), width = 4 → length = 6 → (width * length = 24) :=
by
  sorry

end NUMINAMATH_GPT_outdoor_section_area_l204_20442


namespace NUMINAMATH_GPT_max_profit_l204_20429

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 
    -0.5 * x^2 + 3.5 * x - 0.5 
  else if x > 5 then 
    17 - 2.5 * x 
  else 
    0

theorem max_profit :
  ∀ x : ℝ, (annual_profit 3.5 = 5.625) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_max_profit_l204_20429


namespace NUMINAMATH_GPT_calculate_value_l204_20420

theorem calculate_value : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l204_20420


namespace NUMINAMATH_GPT_cubes_with_even_red_faces_l204_20428

theorem cubes_with_even_red_faces :
  let block_dimensions := (5, 5, 1)
  let painted_sides := 6
  let total_cubes := 25
  let cubes_with_2_red_faces := 16
  cubes_with_2_red_faces = 16 := by
  sorry

end NUMINAMATH_GPT_cubes_with_even_red_faces_l204_20428


namespace NUMINAMATH_GPT_sum_of_numbers_mod_11_l204_20409

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_mod_11_l204_20409


namespace NUMINAMATH_GPT_verify_extending_points_l204_20453

noncomputable def verify_P_and_Q (A B P Q : ℝ → ℝ → ℝ) : Prop := 
  let vector_relation_P := P = - (2/5) • A + (7/5) • B
  let vector_relation_Q := Q = - (1/4) • A + (5/4) • B 
  vector_relation_P ∧ vector_relation_Q

theorem verify_extending_points 
  (A B P Q : ℝ → ℝ → ℝ)
  (h1 : 7 • (P - A) = 2 • (B - P))
  (h2 : 5 • (Q - A) = 1 • (Q - B)) :
  verify_P_and_Q A B P Q := 
by
  sorry  

end NUMINAMATH_GPT_verify_extending_points_l204_20453


namespace NUMINAMATH_GPT_algebraic_expression_l204_20421

theorem algebraic_expression (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l204_20421


namespace NUMINAMATH_GPT_tank_holds_gallons_l204_20488

noncomputable def tank_initial_fraction := (7 : ℚ) / 8
noncomputable def tank_partial_fraction := (2 : ℚ) / 3
def gallons_used := 15

theorem tank_holds_gallons
  (x : ℚ) -- number of gallons the tank holds when full
  (h_initial : tank_initial_fraction * x - gallons_used = tank_partial_fraction * x) :
  x = 72 := 
sorry

end NUMINAMATH_GPT_tank_holds_gallons_l204_20488


namespace NUMINAMATH_GPT_num_birds_is_six_l204_20499

-- Define the number of nests
def N : ℕ := 3

-- Define the difference between the number of birds and nests
def diff : ℕ := 3

-- Prove that the number of birds is 6
theorem num_birds_is_six (B : ℕ) (h1 : N = 3) (h2 : B - N = diff) : B = 6 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_num_birds_is_six_l204_20499


namespace NUMINAMATH_GPT_quadrilateral_is_rhombus_l204_20449

theorem quadrilateral_is_rhombus (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + ad) : a = b ∧ b = c ∧ c = d :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_is_rhombus_l204_20449


namespace NUMINAMATH_GPT_missing_score_find_missing_score_l204_20473

theorem missing_score
  (score1 score2 score3 score4 mean total : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89)
  (hTotal : total = 445) :
  score1 + score2 + score3 + score4 + x = total :=
by
  sorry

theorem find_missing_score
  (score1 score2 score3 score4 mean : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89) :
  (score1 + score2 + score3 + score4 + x) / 5 = mean
  → x = 90 :=
by
  sorry

end NUMINAMATH_GPT_missing_score_find_missing_score_l204_20473


namespace NUMINAMATH_GPT_exists_such_h_l204_20496

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end NUMINAMATH_GPT_exists_such_h_l204_20496


namespace NUMINAMATH_GPT_sum_of_fractions_l204_20410

theorem sum_of_fractions :
  (3 / 9) + (7 / 12) = (11 / 12) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l204_20410


namespace NUMINAMATH_GPT_trumpet_cost_l204_20485

def cost_of_song_book : Real := 5.84
def total_spent : Real := 151
def cost_of_trumpet : Real := total_spent - cost_of_song_book

theorem trumpet_cost : cost_of_trumpet = 145.16 :=
by
  sorry

end NUMINAMATH_GPT_trumpet_cost_l204_20485


namespace NUMINAMATH_GPT_num_statements_imply_impl_l204_20419

variable (p q r : Prop)

def cond1 := p ∧ q ∧ ¬r
def cond2 := ¬p ∧ q ∧ r
def cond3 := p ∧ q ∧ r
def cond4 := ¬p ∧ ¬q ∧ ¬r

def impl := ((p → ¬q) → ¬r)

theorem num_statements_imply_impl : 
  (cond1 p q r → impl p q r) ∧ 
  (cond3 p q r → impl p q r) ∧ 
  (cond4 p q r → impl p q r) ∧ 
  ¬(cond2 p q r → impl p q r) :=
by {
  sorry
}

end NUMINAMATH_GPT_num_statements_imply_impl_l204_20419


namespace NUMINAMATH_GPT_bsnt_value_l204_20435

theorem bsnt_value (B S N T : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hT : 0 < T)
    (h1 : Real.log (B * S) / Real.log 10 + Real.log (B * N) / Real.log 10 = 3)
    (h2 : Real.log (N * T) / Real.log 10 + Real.log (N * S) / Real.log 10 = 4)
    (h3 : Real.log (S * T) / Real.log 10 + Real.log (S * B) / Real.log 10 = 5) :
    B * S * N * T = 10000 :=
sorry

end NUMINAMATH_GPT_bsnt_value_l204_20435


namespace NUMINAMATH_GPT_intersection_M_N_l204_20461

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l204_20461


namespace NUMINAMATH_GPT_mushrooms_on_log_l204_20455

theorem mushrooms_on_log :
  ∃ (G : ℕ), ∃ (S : ℕ), S = 9 * G ∧ G + S = 30 ∧ G = 3 :=
by
  sorry

end NUMINAMATH_GPT_mushrooms_on_log_l204_20455


namespace NUMINAMATH_GPT_trevor_comic_first_issue_pages_l204_20497

theorem trevor_comic_first_issue_pages
  (x : ℕ) 
  (h1 : 3 * x + 4 = 220) :
  x = 72 := 
by
  sorry

end NUMINAMATH_GPT_trevor_comic_first_issue_pages_l204_20497


namespace NUMINAMATH_GPT_ball_speed_is_20_l204_20471

def ball_flight_time : ℝ := 8
def collie_speed : ℝ := 5
def collie_catch_time : ℝ := 32

noncomputable def collie_distance : ℝ := collie_speed * collie_catch_time

theorem ball_speed_is_20 :
  collie_distance = ball_flight_time * 20 :=
by
  sorry

end NUMINAMATH_GPT_ball_speed_is_20_l204_20471


namespace NUMINAMATH_GPT_amount_paid_l204_20478

-- Defining the conditions as constants
def cost_of_apple : ℝ := 0.75
def change_received : ℝ := 4.25

-- Stating the theorem that needs to be proved
theorem amount_paid (a : ℝ) : a = cost_of_apple + change_received :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_l204_20478


namespace NUMINAMATH_GPT_triangle_angle_A_eq_pi_div_3_triangle_area_l204_20479

variable (A B C a b c : ℝ)
variable (S : ℝ)

-- First part: Proving A = π / 3
theorem triangle_angle_A_eq_pi_div_3 (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                                      (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : A > 0) (h6 : A < Real.pi) :
  A = Real.pi / 3 :=
sorry

-- Second part: Finding the area of the triangle
theorem triangle_area (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                      (h2 : b + c = Real.sqrt 10) (h3 : a = 2) (h4 : A = Real.pi / 3) :
  S = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_angle_A_eq_pi_div_3_triangle_area_l204_20479


namespace NUMINAMATH_GPT_tan_double_angle_l204_20425

theorem tan_double_angle (x : ℝ) (h : (Real.sqrt 3) * Real.cos x - Real.sin x = 0) : Real.tan (2 * x) = - (Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l204_20425


namespace NUMINAMATH_GPT_basketball_scores_l204_20413

theorem basketball_scores : ∃ (scores : Finset ℕ), 
  scores = { x | ∃ a b : ℕ, a + b = 7 ∧ x = 2 * a + 3 * b } ∧ scores.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_basketball_scores_l204_20413


namespace NUMINAMATH_GPT_domino_covering_impossible_odd_squares_l204_20462

theorem domino_covering_impossible_odd_squares
  (board1 : ℕ) -- 24 squares
  (board2 : ℕ) -- 21 squares
  (board3 : ℕ) -- 23 squares
  (board4 : ℕ) -- 35 squares
  (board5 : ℕ) -- 63 squares
  (h1 : board1 = 24)
  (h2 : board2 = 21)
  (h3 : board3 = 23)
  (h4 : board4 = 35)
  (h5 : board5 = 63) :
  (board2 % 2 = 1) ∧ (board3 % 2 = 1) ∧ (board4 % 2 = 1) ∧ (board5 % 2 = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_domino_covering_impossible_odd_squares_l204_20462


namespace NUMINAMATH_GPT_angle_B_in_geometric_progression_l204_20482

theorem angle_B_in_geometric_progression 
  {A B C a b c : ℝ} 
  (hSum : A + B + C = Real.pi)
  (hGeo : A = B / 2)
  (hGeo2 : C = 2 * B)
  (hSide : b^2 - a^2 = a * c)
  : B = 2 * Real.pi / 7 := 
by
  sorry

end NUMINAMATH_GPT_angle_B_in_geometric_progression_l204_20482


namespace NUMINAMATH_GPT_sandy_younger_than_molly_l204_20460

variable (s m : ℕ)
variable (h_ratio : 7 * m = 9 * s)
variable (h_sandy : s = 56)

theorem sandy_younger_than_molly : 
  m - s = 16 := 
by
  sorry

end NUMINAMATH_GPT_sandy_younger_than_molly_l204_20460


namespace NUMINAMATH_GPT_kangaroo_chase_l204_20432

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end NUMINAMATH_GPT_kangaroo_chase_l204_20432


namespace NUMINAMATH_GPT_matrix_B_cannot_be_obtained_from_matrix_A_l204_20443

def A : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, -1, -1, 1],
  ![1, 1, -1, 1, -1]
]

def B : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, 1, -1, 1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, 1, -1, 1]
]

theorem matrix_B_cannot_be_obtained_from_matrix_A :
  A.det ≠ B.det := by
  sorry

end NUMINAMATH_GPT_matrix_B_cannot_be_obtained_from_matrix_A_l204_20443


namespace NUMINAMATH_GPT_darius_drive_miles_l204_20407

theorem darius_drive_miles (total_miles : ℕ) (julia_miles : ℕ) (darius_miles : ℕ) 
  (h1 : total_miles = 1677) (h2 : julia_miles = 998) (h3 : total_miles = darius_miles + julia_miles) : 
  darius_miles = 679 :=
by
  sorry

end NUMINAMATH_GPT_darius_drive_miles_l204_20407


namespace NUMINAMATH_GPT_li_ming_estimated_weight_is_correct_l204_20450

-- Define the regression equation as a function
def regression_equation (x : ℝ) : ℝ := 0.7 * x - 52

-- Define the height of Li Ming
def li_ming_height : ℝ := 180

-- The estimated weight according to the regression equation
def estimated_weight : ℝ := regression_equation li_ming_height

-- Theorem statement: Given the height, the weight should be 74
theorem li_ming_estimated_weight_is_correct : estimated_weight = 74 :=
by
  sorry

end NUMINAMATH_GPT_li_ming_estimated_weight_is_correct_l204_20450


namespace NUMINAMATH_GPT_odds_against_C_winning_l204_20440

theorem odds_against_C_winning :
  let P_A := 2 / 7
  let P_B := 1 / 5
  let P_C := 1 - (P_A + P_B)
  (1 - P_C) / P_C = 17 / 18 :=
by
  sorry

end NUMINAMATH_GPT_odds_against_C_winning_l204_20440


namespace NUMINAMATH_GPT_graph_passes_through_quadrants_l204_20431

-- Definitions based on the conditions
def linear_function (x : ℝ) : ℝ := -2 * x + 1

-- The property to be proven
theorem graph_passes_through_quadrants :
  (∃ x > 0, linear_function x > 0) ∧  -- Quadrant I
  (∃ x < 0, linear_function x > 0) ∧  -- Quadrant II
  (∃ x > 0, linear_function x < 0) := -- Quadrant IV
sorry

end NUMINAMATH_GPT_graph_passes_through_quadrants_l204_20431


namespace NUMINAMATH_GPT_initially_had_8_l204_20474

-- Define the number of puppies given away
def given_away : ℕ := 4

-- Define the number of puppies still with Sandy
def still_has : ℕ := 4

-- Define the total number of puppies initially
def initially_had (x y : ℕ) : ℕ := x + y

-- Prove that the number of puppies Sandy's dog had initially equals 8
theorem initially_had_8 : initially_had given_away still_has = 8 :=
by sorry

end NUMINAMATH_GPT_initially_had_8_l204_20474


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l204_20415

/-
A student drives 150 miles to university in a sedan that averages 25 miles per gallon.
The same student drives 150 miles back home in a minivan that averages 15 miles per gallon.
Calculate the average gas mileage for the entire round trip.
-/
theorem average_gas_mileage_round_trip (d1 d2 m1 m2 : ℝ) (h1 : d1 = 150) (h2 : m1 = 25) 
  (h3 : d2 = 150) (h4 : m2 = 15) : 
  (2 * d1) / ((d1/m1) + (d2/m2)) = 18.75 := by
  sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l204_20415


namespace NUMINAMATH_GPT_towels_folded_in_one_hour_l204_20480

theorem towels_folded_in_one_hour :
  let jane_rate := 12 * 5 -- Jane's rate in towels/hour
  let kyla_rate := 6 * 9  -- Kyla's rate in towels/hour
  let anthony_rate := 3 * 14 -- Anthony's rate in towels/hour
  let david_rate := 4 * 6 -- David's rate in towels/hour
  jane_rate + kyla_rate + anthony_rate + david_rate = 180 := 
by
  let jane_rate := 12 * 5
  let kyla_rate := 6 * 9
  let anthony_rate := 3 * 14
  let david_rate := 4 * 6
  show jane_rate + kyla_rate + anthony_rate + david_rate = 180
  sorry

end NUMINAMATH_GPT_towels_folded_in_one_hour_l204_20480


namespace NUMINAMATH_GPT_surface_area_three_dimensional_shape_l204_20422

-- Define the edge length of the largest cube
def edge_length_large : ℕ := 5

-- Define the condition for dividing the edge of the attachment face of the large cube into five equal parts
def divided_into_parts (edge_length : ℕ) (parts : ℕ) : Prop :=
  parts = 5

-- Define the condition that the edge lengths of all three blocks are different
def edge_lengths_different (e1 e2 e3 : ℕ) : Prop :=
  e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3

-- Define the surface area formula for a cube
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- State the problem as a theorem
theorem surface_area_three_dimensional_shape (e1 e2 e3 : ℕ) (h1 : e1 = edge_length_large)
    (h2 : divided_into_parts e1 5) (h3 : edge_lengths_different e1 e2 e3) : 
    surface_area e1 + (surface_area e2 + surface_area e3 - 4 * (e2 * e3)) = 270 :=
sorry

end NUMINAMATH_GPT_surface_area_three_dimensional_shape_l204_20422


namespace NUMINAMATH_GPT_set_inter_and_complement_l204_20451

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {4, 6}

theorem set_inter_and_complement :
  A ∩ (U \ B) = {5, 7} := by
  sorry

end NUMINAMATH_GPT_set_inter_and_complement_l204_20451


namespace NUMINAMATH_GPT_max_value_x_minus_y_proof_l204_20498

noncomputable def max_value_x_minus_y (θ : ℝ) : ℝ :=
  sorry

theorem max_value_x_minus_y_proof (θ : ℝ) (h1 : x = Real.sin θ) (h2 : y = Real.cos θ)
(h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) (h4 : (x^2 + y^2)^2 = x + y) : 
  max_value_x_minus_y θ = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_x_minus_y_proof_l204_20498


namespace NUMINAMATH_GPT_cos_300_eq_half_l204_20470

theorem cos_300_eq_half : Real.cos (2 * π * (300 / 360)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_300_eq_half_l204_20470


namespace NUMINAMATH_GPT_geometric_seq_sum_l204_20436

-- Definitions of the conditions
def a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | _ => (-3)^(n - 1)

theorem geometric_seq_sum : 
  a 0 + |a 1| + a 2 + |a 3| + a 4 = 121 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_l204_20436


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_eq_53_l204_20457

theorem arithmetic_sequence_15th_term_eq_53 (a1 : ℤ) (d : ℤ) (n : ℕ) (a_15 : ℤ) 
    (h1 : a1 = -3)
    (h2 : d = 4)
    (h3 : n = 15)
    (h4 : a_15 = a1 + (n - 1) * d) : 
    a_15 = 53 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_eq_53_l204_20457


namespace NUMINAMATH_GPT_ratio_of_radii_l204_20423

theorem ratio_of_radii 
  (a b : ℝ)
  (h1 : ∀ (a b : ℝ), π * b^2 - π * a^2 = 4 * π * a^2) : 
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_l204_20423


namespace NUMINAMATH_GPT_books_given_away_l204_20484

theorem books_given_away (original_books : ℝ) (books_left : ℝ) (books_given : ℝ) 
    (h1 : original_books = 54.0) 
    (h2 : books_left = 31) : 
    books_given = original_books - books_left → books_given = 23 :=
by
  sorry

end NUMINAMATH_GPT_books_given_away_l204_20484


namespace NUMINAMATH_GPT_leak_empties_tank_in_12_hours_l204_20405

theorem leak_empties_tank_in_12_hours 
  (capacity : ℕ) (inlet_rate : ℕ) (net_emptying_time : ℕ) (leak_rate : ℤ) (leak_emptying_time : ℕ) :
  capacity = 5760 →
  inlet_rate = 4 →
  net_emptying_time = 8 →
  (inlet_rate - leak_rate : ℤ) = (capacity / (net_emptying_time * 60)) →
  leak_emptying_time = (capacity / leak_rate) →
  leak_emptying_time = 12 * 60 / 60 :=
by sorry

end NUMINAMATH_GPT_leak_empties_tank_in_12_hours_l204_20405


namespace NUMINAMATH_GPT_average_percent_score_l204_20447

theorem average_percent_score :
    let students := 120
    let score_95 := 95 * 12
    let score_85 := 85 * 24
    let score_75 := 75 * 30
    let score_65 := 65 * 20
    let score_55 := 55 * 18
    let score_45 := 45 * 10
    let score_35 := 35 * 6
    let total_score := score_95 + score_85 + score_75 + score_65 + score_55 + score_45 + score_35
    (total_score.toFloat / students.toFloat) = 69.8333 :=
by
  sorry

end NUMINAMATH_GPT_average_percent_score_l204_20447


namespace NUMINAMATH_GPT_largest_common_term_in_range_l204_20441

def seq1 (n : ℕ) : ℕ := 5 + 9 * n
def seq2 (m : ℕ) : ℕ := 3 + 8 * m

theorem largest_common_term_in_range :
  ∃ (a : ℕ) (n m : ℕ), seq1 n = a ∧ seq2 m = a ∧ 1 ≤ a ∧ a ≤ 200 ∧ (∀ b, (∃ nf mf, seq1 nf = b ∧ seq2 mf = b ∧ 1 ≤ b ∧ b ≤ 200) → b ≤ a) :=
sorry

end NUMINAMATH_GPT_largest_common_term_in_range_l204_20441
