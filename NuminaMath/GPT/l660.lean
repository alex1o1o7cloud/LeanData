import Mathlib

namespace NUMINAMATH_GPT_num_colors_l660_66082

def total_balls := 350
def balls_per_color := 35

theorem num_colors :
  total_balls / balls_per_color = 10 := 
by
  sorry

end NUMINAMATH_GPT_num_colors_l660_66082


namespace NUMINAMATH_GPT_green_balls_more_than_red_l660_66079

theorem green_balls_more_than_red
  (total_balls : ℕ) (red_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 66)
  (h2 : red_balls = 30)
  (h3 : green_balls = total_balls - red_balls) : green_balls - red_balls = 6 :=
by
  sorry

end NUMINAMATH_GPT_green_balls_more_than_red_l660_66079


namespace NUMINAMATH_GPT_claire_photos_l660_66001

theorem claire_photos (L R C : ℕ) (h1 : L = R) (h2 : L = 3 * C) (h3 : R = C + 28) : C = 14 := by
  sorry

end NUMINAMATH_GPT_claire_photos_l660_66001


namespace NUMINAMATH_GPT_proof_of_min_value_l660_66040

def constraints_on_powers (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

noncomputable def minimum_third_power_sum (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem proof_of_min_value : 
  ∃ a b c d : ℝ, constraints_on_powers a b c d → ∃ min_val : ℝ, min_val = minimum_third_power_sum a b c d :=
sorry -- Further method to rigorously find the minimum value.

end NUMINAMATH_GPT_proof_of_min_value_l660_66040


namespace NUMINAMATH_GPT_am_gm_inequality_l660_66067

theorem am_gm_inequality {a1 a2 a3 : ℝ} (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) :
  (a1 * a2 / a3) + (a2 * a3 / a1) + (a3 * a1 / a2) ≥ a1 + a2 + a3 := 
by 
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l660_66067


namespace NUMINAMATH_GPT_min_digs_is_three_l660_66066

/-- Represents an 8x8 board --/
structure Board :=
(dim : ℕ := 8)

/-- Each cell either contains the treasure or a plaque indicating minimum steps --/
structure Cell :=
(content : CellContent)

/-- Possible content of a cell --/
inductive CellContent
| Treasure
| Plaque (steps : ℕ)

/-- Function that returns the minimum number of cells to dig to find the treasure --/
def min_digs_to_find_treasure (board : Board) : ℕ := 3

/-- The main theorem stating the minimum number of cells needed to find the treasure on an 8x8 board --/
theorem min_digs_is_three : 
  ∀ board : Board, min_digs_to_find_treasure board = 3 := 
by 
  intro board
  sorry

end NUMINAMATH_GPT_min_digs_is_three_l660_66066


namespace NUMINAMATH_GPT_candy_box_original_price_l660_66081

theorem candy_box_original_price (P : ℝ) (h1 : 1.25 * P = 20) : P = 16 :=
sorry

end NUMINAMATH_GPT_candy_box_original_price_l660_66081


namespace NUMINAMATH_GPT_difference_in_earnings_in_currency_B_l660_66011

-- Definitions based on conditions
def num_red_stamps : Nat := 30
def num_white_stamps : Nat := 80
def price_per_red_stamp_currency_A : Nat := 5
def price_per_white_stamp_currency_B : Nat := 50
def exchange_rate_A_to_B : Nat := 2

-- Theorem based on the question and correct answer
theorem difference_in_earnings_in_currency_B : 
  num_white_stamps * price_per_white_stamp_currency_B - 
  (num_red_stamps * price_per_red_stamp_currency_A * exchange_rate_A_to_B) = 3700 := 
  by
  sorry

end NUMINAMATH_GPT_difference_in_earnings_in_currency_B_l660_66011


namespace NUMINAMATH_GPT_smallest_base_for_100_l660_66073

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end NUMINAMATH_GPT_smallest_base_for_100_l660_66073


namespace NUMINAMATH_GPT_solve_cubic_equation_l660_66087

theorem solve_cubic_equation (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by sorry

end NUMINAMATH_GPT_solve_cubic_equation_l660_66087


namespace NUMINAMATH_GPT_S_3n_plus_1_l660_66037

noncomputable def S : ℕ → ℝ := sorry  -- S_n is the sum of the first n terms of the sequence {a_n}
noncomputable def a : ℕ → ℝ := sorry  -- Sequence {a_n}

-- Given conditions
axiom S3 : S 3 = 1
axiom S4 : S 4 = 11
axiom a_recurrence (n : ℕ) : a (n + 3) = 2 * a n

-- Define S_{3n+1} in terms of n
theorem S_3n_plus_1 (n : ℕ) : S (3 * n + 1) = 3 * 2^(n+1) - 1 :=
sorry

end NUMINAMATH_GPT_S_3n_plus_1_l660_66037


namespace NUMINAMATH_GPT_square_carpet_side_length_l660_66008

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ s : ℝ, s * s = area ∧ 3 < s ∧ s < 4 :=
by
  sorry

end NUMINAMATH_GPT_square_carpet_side_length_l660_66008


namespace NUMINAMATH_GPT_angle_ABC_30_degrees_l660_66023

theorem angle_ABC_30_degrees 
    (angle_CBD : ℝ)
    (angle_ABD : ℝ)
    (angle_ABC : ℝ)
    (h1 : angle_CBD = 90)
    (h2 : angle_ABC + angle_ABD + angle_CBD = 180)
    (h3 : angle_ABD = 60) :
    angle_ABC = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_30_degrees_l660_66023


namespace NUMINAMATH_GPT_jackie_first_tree_height_l660_66013

theorem jackie_first_tree_height
  (h : ℝ)
  (avg_height : (h + 2 * (h / 2) + (h + 200)) / 4 = 800) :
  h = 1000 :=
by
  sorry

end NUMINAMATH_GPT_jackie_first_tree_height_l660_66013


namespace NUMINAMATH_GPT_problem_rational_sum_of_powers_l660_66096

theorem problem_rational_sum_of_powers :
  ∃ (a b : ℚ), (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
by
  sorry

end NUMINAMATH_GPT_problem_rational_sum_of_powers_l660_66096


namespace NUMINAMATH_GPT_train_length_l660_66020

-- Definitions of speeds and times
def speed_person_A := 5 / 3.6 -- in meters per second
def speed_person_B := 15 / 3.6 -- in meters per second
def time_to_overtake_A := 36 -- in seconds
def time_to_overtake_B := 45 -- in seconds

-- The length of the train
theorem train_length :
  ∃ x : ℝ, x = 500 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l660_66020


namespace NUMINAMATH_GPT_remainder_of_power_mod_l660_66032

theorem remainder_of_power_mod 
  (n : ℕ)
  (h₁ : 7 ≡ 1 [MOD 6]) : 7^51 ≡ 1 [MOD 6] := 
sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l660_66032


namespace NUMINAMATH_GPT_converse_example_l660_66076

theorem converse_example (x : ℝ) (h : x^2 = 1) : x = 1 :=
sorry

end NUMINAMATH_GPT_converse_example_l660_66076


namespace NUMINAMATH_GPT_sequence_general_term_l660_66098

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a n = n * (a (n + 1) - a n)) : 
  ∀ n : ℕ, n ≥ 1 → a n = n := 
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l660_66098


namespace NUMINAMATH_GPT_pages_in_each_book_l660_66092

variable (BooksRead DaysPerBook TotalDays : ℕ)

theorem pages_in_each_book (h1 : BooksRead = 41) (h2 : DaysPerBook = 12) (h3 : TotalDays = 492) : (TotalDays / DaysPerBook) * DaysPerBook = 492 :=
by
  sorry

end NUMINAMATH_GPT_pages_in_each_book_l660_66092


namespace NUMINAMATH_GPT_product_of_two_numbers_l660_66025

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 40) (h₂ : x - y = 16) : x * y = 336 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l660_66025


namespace NUMINAMATH_GPT_how_many_one_halves_in_two_sevenths_l660_66086

theorem how_many_one_halves_in_two_sevenths : (2 / 7) / (1 / 2) = 4 / 7 := by 
  sorry

end NUMINAMATH_GPT_how_many_one_halves_in_two_sevenths_l660_66086


namespace NUMINAMATH_GPT_initial_number_of_persons_l660_66002

theorem initial_number_of_persons (n : ℕ) 
  (w_increase : ∀ (k : ℕ), k = 4) 
  (old_weight new_weight : ℕ) 
  (h_old : old_weight = 58) 
  (h_new : new_weight = 106) 
  (h_difference : new_weight - old_weight = 48) 
  : n = 12 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_of_persons_l660_66002


namespace NUMINAMATH_GPT_interval_contains_n_l660_66017

theorem interval_contains_n (n : ℕ) (h1 : n < 1000) (h2 : n ∣ 999) (h3 : n + 6 ∣ 99) : 1 ≤ n ∧ n ≤ 250 := 
sorry

end NUMINAMATH_GPT_interval_contains_n_l660_66017


namespace NUMINAMATH_GPT_all_numbers_even_l660_66093

theorem all_numbers_even
  (A B C D E : ℤ)
  (h1 : (A + B + C) % 2 = 0)
  (h2 : (A + B + D) % 2 = 0)
  (h3 : (A + B + E) % 2 = 0)
  (h4 : (A + C + D) % 2 = 0)
  (h5 : (A + C + E) % 2 = 0)
  (h6 : (A + D + E) % 2 = 0)
  (h7 : (B + C + D) % 2 = 0)
  (h8 : (B + C + E) % 2 = 0)
  (h9 : (B + D + E) % 2 = 0)
  (h10 : (C + D + E) % 2 = 0) :
  (A % 2 = 0) ∧ (B % 2 = 0) ∧ (C % 2 = 0) ∧ (D % 2 = 0) ∧ (E % 2 = 0) :=
sorry

end NUMINAMATH_GPT_all_numbers_even_l660_66093


namespace NUMINAMATH_GPT_parametric_to_standard_line_parametric_to_standard_ellipse_l660_66041

theorem parametric_to_standard_line (t : ℝ) (x y : ℝ) 
  (h₁ : x = 1 - 3 * t)
  (h₂ : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := by
sorry

theorem parametric_to_standard_ellipse (θ x y : ℝ) 
  (h₁ : x = 5 * Real.cos θ)
  (h₂ : y = 4 * Real.sin θ) :
  (x^2 / 25) + (y^2 / 16) = 1 := by
sorry

end NUMINAMATH_GPT_parametric_to_standard_line_parametric_to_standard_ellipse_l660_66041


namespace NUMINAMATH_GPT_find_hidden_data_points_l660_66065

-- Given conditions and data
def student_A_score := 81
def student_B_score := 76
def student_D_score := 80
def student_E_score := 83
def number_of_students := 5
def average_score := 80

-- The total score from the average and number of students
def total_score := average_score * number_of_students

theorem find_hidden_data_points (student_C_score mode_score : ℕ) :
  (student_A_score + student_B_score + student_C_score + student_D_score + student_E_score = total_score) ∧
  (mode_score = 80) :=
by
  sorry

end NUMINAMATH_GPT_find_hidden_data_points_l660_66065


namespace NUMINAMATH_GPT_thomas_payment_weeks_l660_66016

theorem thomas_payment_weeks 
    (weekly_rate : ℕ) 
    (total_amount_paid : ℕ) 
    (h1 : weekly_rate = 4550) 
    (h2 : total_amount_paid = 19500) :
    (19500 / 4550 : ℕ) = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_thomas_payment_weeks_l660_66016


namespace NUMINAMATH_GPT_find_z_l660_66019

theorem find_z (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : 1/x + 1/y = k) :
  ∃ z : ℝ, 1/z = k ∧ z = xy/(x + y) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_z_l660_66019


namespace NUMINAMATH_GPT_five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l660_66028

/-
Problem (a): Given five points on a plane, where the distance between any two points is greater than 2. 
             Prove that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_plane_distance_gt3 (P : Fin 5 → ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

/-
Problem (b): Given five points in space, where the distance between any two points is greater than 2. 
             Prove that it is not necessarily true that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_space_not_necessarily_gt3 (P : Fin 5 → ℝ × ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ¬ ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

end NUMINAMATH_GPT_five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l660_66028


namespace NUMINAMATH_GPT_find_number_ge_40_l660_66097

theorem find_number_ge_40 (x : ℝ) : 0.90 * x > 0.80 * 30 + 12 → x > 40 :=
by sorry

end NUMINAMATH_GPT_find_number_ge_40_l660_66097


namespace NUMINAMATH_GPT_count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l660_66060

-- Define the weight 's'.
variable (s : ℕ)

-- Define the function that counts the number of Young diagrams for a given weight.
def countYoungDiagrams (s : ℕ) : ℕ :=
  -- Placeholder for actual implementation of counting Young diagrams.
  sorry

-- Prove that the count of Young diagrams for s = 4 is 5
theorem count_young_diagrams_4 : countYoungDiagrams 4 = 5 :=
by sorry

-- Prove that the count of Young diagrams for s = 5 is 7
theorem count_young_diagrams_5 : countYoungDiagrams 5 = 7 :=
by sorry

-- Prove that the count of Young diagrams for s = 6 is 11
theorem count_young_diagrams_6 : countYoungDiagrams 6 = 11 :=
by sorry

-- Prove that the count of Young diagrams for s = 7 is 15
theorem count_young_diagrams_7 : countYoungDiagrams 7 = 15 :=
by sorry

end NUMINAMATH_GPT_count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l660_66060


namespace NUMINAMATH_GPT_rounds_played_l660_66000

-- Define the given conditions as Lean constants
def totalPoints : ℝ := 378.5
def pointsPerRound : ℝ := 83.25

-- Define the goal as a Lean theorem
theorem rounds_played :
  Int.ceil (totalPoints / pointsPerRound) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_rounds_played_l660_66000


namespace NUMINAMATH_GPT_problem1_problem2_l660_66062

-- Definitions of the sets
def U : Set ℕ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | 3 ≤ x ∧ x ≤ 7 }

-- Problems to prove (statements only, no proofs provided)
theorem problem1 : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
by
  sorry

theorem problem2 : U \ A ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)} :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l660_66062


namespace NUMINAMATH_GPT_mean_of_data_is_5_l660_66061

theorem mean_of_data_is_5 (h : s^2 = (1 / 4) * ((3.2 - x)^2 + (5.7 - x)^2 + (4.3 - x)^2 + (6.8 - x)^2))
  : x = 5 := 
sorry

end NUMINAMATH_GPT_mean_of_data_is_5_l660_66061


namespace NUMINAMATH_GPT_polynomial_division_l660_66057

variable (a p x : ℝ)

theorem polynomial_division :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end NUMINAMATH_GPT_polynomial_division_l660_66057


namespace NUMINAMATH_GPT_sqrt_exp_sum_eq_eight_sqrt_two_l660_66004

theorem sqrt_exp_sum_eq_eight_sqrt_two : 
  (Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_exp_sum_eq_eight_sqrt_two_l660_66004


namespace NUMINAMATH_GPT_greatest_value_a4_b4_l660_66053

theorem greatest_value_a4_b4
    (a b : Nat → ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + a 1)
    (h_geom_seq : ∀ n, b (n + 1) = b n * b 1)
    (h_a1b1 : a 1 * b 1 = 20)
    (h_a2b2 : a 2 * b 2 = 19)
    (h_a3b3 : a 3 * b 3 = 14) :
    ∃ m : ℝ, a 4 * b 4 = 8 ∧ ∀ x, a 4 * b 4 ≤ x -> x = 8 := by
  sorry

end NUMINAMATH_GPT_greatest_value_a4_b4_l660_66053


namespace NUMINAMATH_GPT_length_AB_given_conditions_l660_66024

variable {A B P Q : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField P] [LinearOrderedField Q]

def length_of_AB (x y : A) : A := x + y

theorem length_AB_given_conditions (x y u v : A) (hx : y = 4 * x) (hv : 5 * u = 2 * v) (hu : u = x + 3) (hv' : v = y - 3) (hPQ : PQ = 3) : length_of_AB x y = 35 :=
by
  sorry

end NUMINAMATH_GPT_length_AB_given_conditions_l660_66024


namespace NUMINAMATH_GPT_find_original_acid_amount_l660_66095

noncomputable def original_amount_of_acid (a w : ℝ) : Prop :=
  3 * a = w + 2 ∧ 5 * a = 3 * w - 10

theorem find_original_acid_amount (a w : ℝ) (h : original_amount_of_acid a w) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_original_acid_amount_l660_66095


namespace NUMINAMATH_GPT_cyclist_first_part_distance_l660_66044

theorem cyclist_first_part_distance
  (T₁ T₂ T₃ : ℝ)
  (D : ℝ)
  (h1 : D = 9 * T₁)
  (h2 : T₂ = 12 / 10)
  (h3 : T₃ = (D + 12) / 7.5)
  (h4 : T₁ + T₂ + T₃ = 7.2) : D = 18 := by
  sorry

end NUMINAMATH_GPT_cyclist_first_part_distance_l660_66044


namespace NUMINAMATH_GPT_shaded_squares_percentage_l660_66003

theorem shaded_squares_percentage : 
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2
  (shaded_squares / total_squares) * 100 = 50 :=
by
  /- Definitions and conditions -/
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2

  /- Required proof statement -/
  have percentage_shaded : (shaded_squares / total_squares) * 100 = 50 := sorry

  /- Return the proof -/
  exact percentage_shaded

end NUMINAMATH_GPT_shaded_squares_percentage_l660_66003


namespace NUMINAMATH_GPT_total_weight_of_13_gold_bars_l660_66034

theorem total_weight_of_13_gold_bars
    (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
    (w12 w13 w23 w45 w67 w89 w1011 w1213 : ℝ)
    (h1 : w12 = C1 + C2)
    (h2 : w13 = C1 + C3)
    (h3 : w23 = C2 + C3)
    (h4 : w45 = C4 + C5)
    (h5 : w67 = C6 + C7)
    (h6 : w89 = C8 + C9)
    (h7 : w1011 = C10 + C11)
    (h8 : w1213 = C12 + C13) :
    C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13 = 
    (C1 + C2 + C3) + (C4 + C5) + (C6 + C7) + (C8 + C9) + (C10 + C11) + (C12 + C13) := 
  by
  sorry

end NUMINAMATH_GPT_total_weight_of_13_gold_bars_l660_66034


namespace NUMINAMATH_GPT_NumFriendsNextToCaraOnRight_l660_66018

open Nat

def total_people : ℕ := 8
def freds_next_to_Cara : ℕ := 7

theorem NumFriendsNextToCaraOnRight (h : total_people = 8) : freds_next_to_Cara = 7 :=
by
  sorry

end NUMINAMATH_GPT_NumFriendsNextToCaraOnRight_l660_66018


namespace NUMINAMATH_GPT_factor_x10_minus_1024_l660_66015

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end NUMINAMATH_GPT_factor_x10_minus_1024_l660_66015


namespace NUMINAMATH_GPT_pie_not_crust_percentage_l660_66083

theorem pie_not_crust_percentage (total_weight crust_weight : ℝ) 
  (h1 : total_weight = 200) (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_pie_not_crust_percentage_l660_66083


namespace NUMINAMATH_GPT_scientific_notation_of_sesame_mass_l660_66030

theorem scientific_notation_of_sesame_mass :
  0.00000201 = 2.01 * 10^(-6) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_sesame_mass_l660_66030


namespace NUMINAMATH_GPT_tangent_line_at_zero_range_of_a_l660_66068

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (h : ∀ x, f 1 x = Real.exp x - Real.sin x - 1) :
  ∀ x, Real.exp x - Real.sin x - 1 = f 1 x :=
by
  sorry

theorem range_of_a (h : ∀ x, f a x ≥ 0) : a ∈ Set.Iic 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_zero_range_of_a_l660_66068


namespace NUMINAMATH_GPT_tenth_term_arithmetic_seq_l660_66089

theorem tenth_term_arithmetic_seq : 
  ∀ (first_term common_diff : ℤ) (n : ℕ), 
    first_term = 10 → common_diff = -2 → n = 10 → 
    (first_term + (n - 1) * common_diff) = -8 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_seq_l660_66089


namespace NUMINAMATH_GPT_find_y_l660_66009

variable {x y : ℤ}
variables (h1 : y = 2 * x - 3) (h2 : x + y = 57)

theorem find_y : y = 37 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_y_l660_66009


namespace NUMINAMATH_GPT_tangent_lines_inequality_l660_66050

theorem tangent_lines_inequality (k k1 k2 b b1 b2 : ℝ)
  (h1 : k = - (b * b) / 4)
  (h2 : k1 = - (b1 * b1) / 4)
  (h3 : k2 = - (b2 * b2) / 4)
  (h4 : b = b1 + b2) :
  k ≥ 2 * (k1 + k2) := sorry

end NUMINAMATH_GPT_tangent_lines_inequality_l660_66050


namespace NUMINAMATH_GPT_minimum_n_value_l660_66031

theorem minimum_n_value : ∃ n : ℕ, n > 0 ∧ ∀ r : ℕ, (2 * n = 5 * r) → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_n_value_l660_66031


namespace NUMINAMATH_GPT_height_of_spherical_caps_l660_66046

theorem height_of_spherical_caps
  (r q : ℝ)
  (m₁ m₂ m₃ m₄ : ℝ)
  (h1 : m₂ = m₁ * q)
  (h2 : m₃ = m₁ * q^2)
  (h3 : m₄ = m₁ * q^3)
  (h4 : m₁ + m₂ + m₃ + m₄ = 2 * r) :
  m₁ = 2 * r * (q - 1) / (q^4 - 1) := 
sorry

end NUMINAMATH_GPT_height_of_spherical_caps_l660_66046


namespace NUMINAMATH_GPT_find_n_l660_66022

-- Define the hyperbola and its properties
def hyperbola (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ 2 = (m / (m / 2)) ∧ ∃ f : ℝ × ℝ, f = (m, 0)

-- Define the parabola and its properties
def parabola_focus (m : ℝ) : Prop :=
  (m, 0) = (m, 0)

-- The statement we want to prove
theorem find_n (m : ℝ) (n : ℝ) (H_hyperbola : hyperbola m n) (H_parabola : parabola_focus m) : n = 12 :=
sorry

end NUMINAMATH_GPT_find_n_l660_66022


namespace NUMINAMATH_GPT_polynomial_division_l660_66038

noncomputable def poly1 : Polynomial ℤ := Polynomial.X ^ 13 - Polynomial.X + 100
noncomputable def poly2 : Polynomial ℤ := Polynomial.X ^ 2 + Polynomial.X + 2

theorem polynomial_division : ∃ q : Polynomial ℤ, poly1 = poly2 * q :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_division_l660_66038


namespace NUMINAMATH_GPT_stuffed_animals_mom_gift_l660_66056

theorem stuffed_animals_mom_gift (x : ℕ) :
  (10 + x) + 3 * (10 + x) = 48 → x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_stuffed_animals_mom_gift_l660_66056


namespace NUMINAMATH_GPT_chemistry_marks_more_than_physics_l660_66021

theorem chemistry_marks_more_than_physics (M P C x : ℕ) 
  (h1 : M + P = 32) 
  (h2 : (M + C) / 2 = 26) 
  (h3 : C = P + x) : 
  x = 20 := 
by
  sorry

end NUMINAMATH_GPT_chemistry_marks_more_than_physics_l660_66021


namespace NUMINAMATH_GPT_area_Q1RQ3Q5_of_regular_hexagon_l660_66077

noncomputable def area_quadrilateral (s : ℝ) (θ : ℝ) : ℝ := s^2 * Real.sin θ / 2

theorem area_Q1RQ3Q5_of_regular_hexagon :
  let apothem := 3
  let side_length := 6 * Real.sqrt 3
  let θ := Real.pi / 3  -- 60 degrees in radians
  area_quadrilateral (3 * Real.sqrt 3) θ = 27 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_Q1RQ3Q5_of_regular_hexagon_l660_66077


namespace NUMINAMATH_GPT_volume_of_prism_l660_66099

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 54) : 
  a * b * c = 270 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l660_66099


namespace NUMINAMATH_GPT_average_salary_is_8000_l660_66006

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

def average_salary : ℕ := total_salary / num_people

theorem average_salary_is_8000 : average_salary = 8000 := by
  sorry

end NUMINAMATH_GPT_average_salary_is_8000_l660_66006


namespace NUMINAMATH_GPT_minimum_chocolates_l660_66054

theorem minimum_chocolates (x : ℤ) (h1 : x ≥ 150) (h2 : x % 15 = 7) : x = 157 :=
sorry

end NUMINAMATH_GPT_minimum_chocolates_l660_66054


namespace NUMINAMATH_GPT_angle_between_slant_height_and_base_l660_66072

theorem angle_between_slant_height_and_base (R : ℝ) (diam_base_upper diam_base_lower : ℝ) 
(h1 : diam_base_upper + diam_base_lower = 5 * R)
: ∃ θ : ℝ, θ = Real.arcsin (4 / 5) := 
sorry

end NUMINAMATH_GPT_angle_between_slant_height_and_base_l660_66072


namespace NUMINAMATH_GPT_rectangle_long_side_eq_12_l660_66090

theorem rectangle_long_side_eq_12 (s : ℕ) (a b : ℕ) (congruent_triangles : true) (h : a + b = s) (short_side_is_8 : s = 8) : a + b + 4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_long_side_eq_12_l660_66090


namespace NUMINAMATH_GPT_max_value_4x_plus_y_l660_66051

theorem max_value_4x_plus_y (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (u : ℝ), (∃ (x y : ℝ), 16 * x^2 + y^2 + 4 * x * y = 3 ∧ u = 4 * x + y) → u ≤ M :=
by
  use 2
  sorry

end NUMINAMATH_GPT_max_value_4x_plus_y_l660_66051


namespace NUMINAMATH_GPT_find_a_l660_66055

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, x - 2 * a * y - 3 = 0 ∧ x^2 + y^2 - 2 * x + 2 * y - 3 = 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l660_66055


namespace NUMINAMATH_GPT_min_value_l660_66039

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end NUMINAMATH_GPT_min_value_l660_66039


namespace NUMINAMATH_GPT_find_f_a_l660_66052

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 4 * Real.logb 2 (-x) else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h1 : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
sorry

end NUMINAMATH_GPT_find_f_a_l660_66052


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l660_66010

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition_for_increasing_geometric_sequence
  (a : ℕ → ℝ)
  (h0 : a 0 > 0)
  (h_geom : is_geometric_sequence a) :
  (a 0^2 < a 1^2) ↔ (is_increasing_sequence a) ∧ ¬ (∀ n, a n > 0 → a (n + 1) > 0) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l660_66010


namespace NUMINAMATH_GPT_rons_baseball_team_l660_66033

/-- Ron's baseball team scored 270 points in the year. 
    5 players averaged 50 points each, 
    and the remaining players averaged 5 points each.
    Prove that the number of players on the team is 9. -/
theorem rons_baseball_team : (∃ n m : ℕ, 5 * 50 + m * 5 = 270 ∧ n = 5 + m ∧ 5 = 50 ∧ m = 4) :=
sorry

end NUMINAMATH_GPT_rons_baseball_team_l660_66033


namespace NUMINAMATH_GPT_flowmaster_pump_output_l660_66007

theorem flowmaster_pump_output (hourly_rate : ℕ) (time_minutes : ℕ) (output_gallons : ℕ) 
  (h1 : hourly_rate = 600) 
  (h2 : time_minutes = 30) 
  (h3 : output_gallons = (hourly_rate * time_minutes) / 60) : 
  output_gallons = 300 :=
by sorry

end NUMINAMATH_GPT_flowmaster_pump_output_l660_66007


namespace NUMINAMATH_GPT_minimum_percentage_increase_in_mean_replacing_with_primes_l660_66048

def mean (S : List ℤ) : ℚ :=
  (S.sum : ℚ) / S.length

noncomputable def percentage_increase (original new : ℚ) : ℚ :=
  ((new - original) / original) * 100

theorem minimum_percentage_increase_in_mean_replacing_with_primes :
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  percentage_increase (mean F) (mean G) = 100 :=
by {
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  sorry 
}

end NUMINAMATH_GPT_minimum_percentage_increase_in_mean_replacing_with_primes_l660_66048


namespace NUMINAMATH_GPT_analytical_expression_of_f_l660_66078

theorem analytical_expression_of_f (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x) = x^2 + 1 / x^2) →
  (∀ y : ℝ, (y ≥ 2 ∨ y ≤ -2) → f y = y^2 - 2) :=
by
  intro h1 y hy
  sorry

end NUMINAMATH_GPT_analytical_expression_of_f_l660_66078


namespace NUMINAMATH_GPT_sqrt_9_minus_2_pow_0_plus_abs_neg1_l660_66005

theorem sqrt_9_minus_2_pow_0_plus_abs_neg1 :
  (Real.sqrt 9 - 2^0 + abs (-1) = 3) :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_sqrt_9_minus_2_pow_0_plus_abs_neg1_l660_66005


namespace NUMINAMATH_GPT_maximum_value_f_zeros_l660_66014

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if 1 < x then k * x + 1
  else 0

theorem maximum_value_f_zeros (k : ℝ) (x1 x2 : ℝ) :
  0 < k ∧ ∀ x, f x k = 0 ↔ x = x1 ∨ x = x2 → x1 ≠ x2 →
  x1 > 0 → x2 > 0 → -1 < k ∧ k < 0 →
  (x1 = -1 / k) ∧ (x2 = 1 / (1 + Real.sqrt (1 + k))) →
  ∃ y, (1 / x1) + (1 / x2) = y ∧ y = 9 / 4 := sorry

end NUMINAMATH_GPT_maximum_value_f_zeros_l660_66014


namespace NUMINAMATH_GPT_keiths_total_spending_l660_66071

theorem keiths_total_spending :
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  total_cost = 77.05 :=
by
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  have h : total_cost = 77.05 := sorry
  exact h

end NUMINAMATH_GPT_keiths_total_spending_l660_66071


namespace NUMINAMATH_GPT_find_x_l660_66091

theorem find_x 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (dot_product : ℝ)
  (ha : a = (1, 2)) 
  (hb : b = (x, 3)) 
  (hdot : a.1 * b.1 + a.2 * b.2 = dot_product) 
  (hdot_val : dot_product = 4) : 
  x = -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l660_66091


namespace NUMINAMATH_GPT_solution_set_for_inequality_l660_66075

theorem solution_set_for_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decreasing : ∀ ⦃x y⦄, 0 < x → x < y → f y < f x)
  (h_f_neg3 : f (-3) = 1) :
  { x | f x < 1 } = { x | x < -3 ∨ 3 < x } := 
by
  -- TODO: Prove this theorem
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l660_66075


namespace NUMINAMATH_GPT_total_stones_l660_66070

theorem total_stones (sent_away kept total : ℕ) (h1 : sent_away = 63) (h2 : kept = 15) (h3 : total = sent_away + kept) : total = 78 :=
by
  sorry

end NUMINAMATH_GPT_total_stones_l660_66070


namespace NUMINAMATH_GPT_smallest_sum_squares_edges_is_cube_l660_66069

theorem smallest_sum_squares_edges_is_cube (V : ℝ) (a b c : ℝ)
  (h_vol : a * b * c = V) :
  a^2 + b^2 + c^2 ≥ 3 * (V^(2/3)) := 
sorry

end NUMINAMATH_GPT_smallest_sum_squares_edges_is_cube_l660_66069


namespace NUMINAMATH_GPT_train_crosses_pole_time_l660_66088

theorem train_crosses_pole_time
  (l : ℕ) (v_kmh : ℕ) (v_ms : ℚ) (t : ℕ)
  (h_l : l = 100)
  (h_v_kmh : v_kmh = 180)
  (h_v_ms_conversion : v_ms = v_kmh * 1000 / 3600)
  (h_v_ms : v_ms = 50) :
  t = l / v_ms := by
  sorry

end NUMINAMATH_GPT_train_crosses_pole_time_l660_66088


namespace NUMINAMATH_GPT_sunflower_count_l660_66063

theorem sunflower_count (r l d : ℕ) (t : ℕ) (h1 : r + l + d = 40) (h2 : t = 160) : 
  t - (r + l + d) = 120 := by
  sorry

end NUMINAMATH_GPT_sunflower_count_l660_66063


namespace NUMINAMATH_GPT_determine_B_l660_66049

-- Declare the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 1}

-- The conditions given in the problem
axiom h1 : A ∩ B = {1}
axiom h2 : A ∪ B = {0, 1, 2}

-- The theorem we want to prove
theorem determine_B : B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_determine_B_l660_66049


namespace NUMINAMATH_GPT_zander_stickers_l660_66058

theorem zander_stickers (S : ℕ) (h1 : 44 = (11 / 25) * S) : S = 100 :=
by
  sorry

end NUMINAMATH_GPT_zander_stickers_l660_66058


namespace NUMINAMATH_GPT_compute_xy_l660_66080

variable (x y : ℝ)

-- Conditions from the problem
def condition1 : Prop := x + y = 10
def condition2 : Prop := x^3 + y^3 = 172

-- Theorem statement to prove the answer
theorem compute_xy (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 41.4 :=
sorry

end NUMINAMATH_GPT_compute_xy_l660_66080


namespace NUMINAMATH_GPT_sqrt_expression_equality_l660_66085

theorem sqrt_expression_equality :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * 5^(3/4) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_equality_l660_66085


namespace NUMINAMATH_GPT_part1_part2_l660_66029

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1 (x : ℝ) : (∃ a, a = 1) → f x 1 > 1 ↔ -2 < x ∧ x < -(2/3) := by
  sorry

theorem part2 (a : ℝ) : (∀ x, 2 ≤ x → x ≤ 3 → f x a > 0) ↔ (-5/2) < a ∧ a < -2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l660_66029


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l660_66027

variables (A B C : Prop)

theorem sufficient_but_not_necessary_condition (h1 : B → A) (h2 : C → B) (h3 : ¬(B → C)) : (C → A) ∧ ¬(A → C) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l660_66027


namespace NUMINAMATH_GPT_stratified_sampling_l660_66064

theorem stratified_sampling (lathe_A lathe_B total_samples : ℕ) (hA : lathe_A = 56) (hB : lathe_B = 42) (hTotal : total_samples = 14) :
  ∃ (sample_A sample_B : ℕ), sample_A = 8 ∧ sample_B = 6 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l660_66064


namespace NUMINAMATH_GPT_evaluate_expression_l660_66026

theorem evaluate_expression:
  let a := 3
  let b := 2
  (a^b)^a - (b^a)^b = 665 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l660_66026


namespace NUMINAMATH_GPT_least_number_of_square_tiles_l660_66035

-- Definitions based on conditions
def room_length_cm : ℕ := 672
def room_width_cm : ℕ := 432

-- Correct Answer is 126 tiles

-- Lean Statement for the proof problem
theorem least_number_of_square_tiles : 
  ∃ tile_size tiles_needed, 
    (tile_size = Int.gcd room_length_cm room_width_cm) ∧
    (tiles_needed = (room_length_cm / tile_size) * (room_width_cm / tile_size)) ∧
    tiles_needed = 126 := 
by
  sorry

end NUMINAMATH_GPT_least_number_of_square_tiles_l660_66035


namespace NUMINAMATH_GPT_unique_solution_triplet_l660_66012

theorem unique_solution_triplet :
  ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x^y + y^x = z^y ∧ x^y + 2012 = y^(z+1)) ∧ (x = 6 ∧ y = 2 ∧ z = 10) := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_triplet_l660_66012


namespace NUMINAMATH_GPT_gino_popsicle_sticks_left_l660_66084

-- Define the initial number of popsicle sticks Gino has
def initial_popsicle_sticks : ℝ := 63.0

-- Define the number of popsicle sticks Gino gives away
def given_away_popsicle_sticks : ℝ := 50.0

-- Expected number of popsicle sticks Gino has left
def expected_remaining_popsicle_sticks : ℝ := 13.0

-- Main theorem to be proven
theorem gino_popsicle_sticks_left :
  initial_popsicle_sticks - given_away_popsicle_sticks = expected_remaining_popsicle_sticks := 
by
  -- This is where the proof would go, but we leave it as 'sorry' for now
  sorry

end NUMINAMATH_GPT_gino_popsicle_sticks_left_l660_66084


namespace NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l660_66059

theorem line_does_not_pass_second_quadrant 
  (A B C x y : ℝ) 
  (h1 : A * C < 0) 
  (h2 : B * C > 0) 
  (h3 : A * x + B * y + C = 0) :
  ¬ (x < 0 ∧ y > 0) := 
sorry

end NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l660_66059


namespace NUMINAMATH_GPT_luke_money_at_end_of_june_l660_66042

noncomputable def initial_money : ℝ := 48
noncomputable def february_money : ℝ := initial_money - 0.30 * initial_money
noncomputable def march_money : ℝ := february_money - 11 + 21 + 50 * 1.20

noncomputable def april_savings : ℝ := 0.10 * march_money
noncomputable def april_money : ℝ := (march_money - april_savings) - 10 * 1.18 + 0.05 * (march_money - april_savings)

noncomputable def may_savings : ℝ := 0.15 * april_money
noncomputable def may_money : ℝ := (april_money - may_savings) + 100 * 1.22 - 0.25 * ((april_money - may_savings) + 100 * 1.22)

noncomputable def june_savings : ℝ := 0.10 * may_money
noncomputable def june_money : ℝ := (may_money - june_savings) - 0.08 * (may_money - june_savings)
noncomputable def final_money : ℝ := june_money + 0.06 * (may_money - june_savings)

theorem luke_money_at_end_of_june : final_money = 128.15 := sorry

end NUMINAMATH_GPT_luke_money_at_end_of_june_l660_66042


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l660_66043

theorem arithmetic_sequence_sum (a b d : ℕ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h1 : a₁ + a₂ + a₃ = 39)
  (h2 : a₄ + a₅ + a₆ = 27)
  (h3 : a₄ = a₁ + 3 * d)
  (h4 : a₅ = a₂ + 3 * d)
  (h5 : a₆ = a₃ + 3 * d)
  (h6 : a₇ = a₄ + 3 * d)
  (h7 : a₈ = a₅ + 3 * d)
  (h8 : a₉ = a₆ + 3 * d) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 81 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l660_66043


namespace NUMINAMATH_GPT_quadratic_has_real_solutions_l660_66094

theorem quadratic_has_real_solutions (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_solutions_l660_66094


namespace NUMINAMATH_GPT_set_union_is_all_real_l660_66074

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the set M as {x | x > 0}
def M : Set ℝ := {x | x > 0}

-- Define the set N as {x | x^2 ≥ x}
def N : Set ℝ := {x | x^2 ≥ x}

-- Prove the relationship M ∪ N = ℝ
theorem set_union_is_all_real : M ∪ N = U := by
  sorry

end NUMINAMATH_GPT_set_union_is_all_real_l660_66074


namespace NUMINAMATH_GPT_intersection_complements_l660_66036

open Set

variable (U : Set (ℝ × ℝ))
variable (M : Set (ℝ × ℝ))
variable (N : Set (ℝ × ℝ))

noncomputable def complementU (A : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ A

theorem intersection_complements :
  let U := {p : ℝ × ℝ | True}
  let M := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y + 2 = x - 2 ∧ x ≠ 2)}
  let N := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y ≠ x - 4)}
  ((complementU U M) ∩ (complementU U N)) = {(2, -2)} :=
by
  let U := {(x, y) : ℝ × ℝ | True}
  let M := {(x, y) : ℝ × ℝ | (y + 2) = (x - 2) ∧ x ≠ 2}
  let N := {(x, y) : ℝ × ℝ | y ≠ (x - 4)}
  have complement_M := U \ M
  have complement_N := U \ N
  sorry

end NUMINAMATH_GPT_intersection_complements_l660_66036


namespace NUMINAMATH_GPT_find_number_l660_66047

theorem find_number (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 := sorry

end NUMINAMATH_GPT_find_number_l660_66047


namespace NUMINAMATH_GPT_evaluate_fraction_l660_66045

theorem evaluate_fraction :
  1 + (2 / (3 + (6 / (7 + (8 / 9))))) = 409 / 267 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l660_66045
