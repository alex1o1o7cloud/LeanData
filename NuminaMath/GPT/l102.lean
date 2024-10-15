import Mathlib

namespace NUMINAMATH_GPT_function_in_second_quadrant_l102_10217

theorem function_in_second_quadrant (k : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → (k / x₁ < k / x₂)) → (∀ x : ℝ, x < 0 → (k > 0)) :=
sorry

end NUMINAMATH_GPT_function_in_second_quadrant_l102_10217


namespace NUMINAMATH_GPT_smallest_number_among_four_l102_10280

theorem smallest_number_among_four (a b c d : ℤ) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) : 
  min (min a (min b c)) d = -2023 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_smallest_number_among_four_l102_10280


namespace NUMINAMATH_GPT_eval_expression_l102_10231

theorem eval_expression : (20 - 16) * (12 + 8) / 4 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l102_10231


namespace NUMINAMATH_GPT_car_speed_conversion_l102_10230

theorem car_speed_conversion :
  let speed_mps := 10 -- speed of the car in meters per second
  let conversion_factor := 3.6 -- conversion factor from m/s to km/h
  let speed_kmph := speed_mps * conversion_factor -- speed of the car in kilometers per hour
  speed_kmph = 36 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_conversion_l102_10230


namespace NUMINAMATH_GPT_solve_symbols_values_l102_10294

def square_value : Nat := 423 / 47

def boxminus_and_boxtimes_relation (boxminus boxtimes : Nat) : Prop :=
  1448 = 282 * boxminus + 9 * boxtimes

def boxtimes_value : Nat := 38 / 9

def boxplus_value : Nat := 846 / 423

theorem solve_symbols_values :
  ∃ (square boxplus boxtimes boxminus : Nat),
    square = 9 ∧
    boxplus = 2 ∧
    boxtimes = 8 ∧
    boxminus = 5 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + 9 * boxtimes ∧
    9 * boxtimes = 38 ∧
    423 * boxplus / 3 = 282 := by
  sorry

end NUMINAMATH_GPT_solve_symbols_values_l102_10294


namespace NUMINAMATH_GPT_at_least_one_not_less_than_2_l102_10257

theorem at_least_one_not_less_than_2 (x y z : ℝ) (hp : 0 < x ∧ 0 < y ∧ 0 < z) :
  let a := x + 1/y
  let b := y + 1/z
  let c := z + 1/x
  (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) := by
    sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_2_l102_10257


namespace NUMINAMATH_GPT_tan_square_of_cos_double_angle_l102_10262

theorem tan_square_of_cos_double_angle (α : ℝ) (h : Real.cos (2 * α) = -1/9) : Real.tan (α)^2 = 5/4 :=
by
  sorry

end NUMINAMATH_GPT_tan_square_of_cos_double_angle_l102_10262


namespace NUMINAMATH_GPT_chord_length_squared_l102_10284

theorem chord_length_squared
  (r5 r10 r15 : ℝ) 
  (externally_tangent : r5 = 5 ∧ r10 = 10)
  (internally_tangent : r15 = 15)
  (common_external_tangent : r15 - r10 - r5 = 0) :
  ∃ PQ_squared : ℝ, PQ_squared = 622.44 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_squared_l102_10284


namespace NUMINAMATH_GPT_rope_length_l102_10247

-- Definitions and assumptions directly derived from conditions
variable (total_length : ℕ)
variable (part_length : ℕ)
variable (sub_part_length : ℕ)

-- Conditions
def condition1 : Prop := total_length / 4 = part_length
def condition2 : Prop := (part_length / 2) * 2 = part_length
def condition3 : Prop := part_length / 2 = sub_part_length
def condition4 : Prop := sub_part_length = 25

-- Proof problem statement
theorem rope_length (h1 : condition1 total_length part_length)
                    (h2 : condition2 part_length)
                    (h3 : condition3 part_length sub_part_length)
                    (h4 : condition4 sub_part_length) :
                    total_length = 100 := 
sorry

end NUMINAMATH_GPT_rope_length_l102_10247


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l102_10285

variable {α : Type}
variable [LinearOrderedField α]

def a1 (a_1 : α) : Prop := a_1 ≠ 0 
def a2_eq_3a1 (a_1 a_2 : α) : Prop := a_2 = 3 * a_1 

noncomputable def common_difference (a_1 a_2 : α) : α :=
  a_2 - a_1

noncomputable def S (n : ℕ) (a_1 d : α) : α :=
  n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio
  (a_1 a_2 : α)
  (h₀ : a1 a_1)
  (h₁ : a2_eq_3a1 a_1 a_2) :
  (S 10 a_1 (common_difference a_1 a_2)) / (S 5 a_1 (common_difference a_1 a_2)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l102_10285


namespace NUMINAMATH_GPT_total_fuel_l102_10282

theorem total_fuel (fuel_this_week : ℝ) (reduction_percent : ℝ) :
  fuel_this_week = 15 → reduction_percent = 0.20 → 
  (fuel_this_week + (fuel_this_week * (1 - reduction_percent))) = 27 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_total_fuel_l102_10282


namespace NUMINAMATH_GPT_test_questions_l102_10239

theorem test_questions (x : ℕ) (h1 : x % 5 = 0) (h2 : 70 < 32 * 100 / x) (h3 : 32 * 100 / x < 77) : x = 45 := 
by sorry

end NUMINAMATH_GPT_test_questions_l102_10239


namespace NUMINAMATH_GPT_combinatorial_problem_correct_l102_10299

def combinatorial_problem : Prop :=
  let boys := 4
  let girls := 3
  let chosen_boys := 3
  let chosen_girls := 2
  let num_ways_select := Nat.choose boys chosen_boys * Nat.choose girls chosen_girls
  let arrangements_no_consecutive_girls := 6 * Nat.factorial 4 / Nat.factorial 2
  num_ways_select * arrangements_no_consecutive_girls = 864

theorem combinatorial_problem_correct : combinatorial_problem := 
  by 
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_combinatorial_problem_correct_l102_10299


namespace NUMINAMATH_GPT_polynomial_value_at_minus_1_l102_10241

-- Definitions for the problem conditions
def polynomial_1 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x + 1
def polynomial_2 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x - 2

theorem polynomial_value_at_minus_1 :
  ∀ (a b : ℤ), (a + b = 2022) → polynomial_2 a b (-1) = -2024 :=
by
  intro a b h
  sorry

end NUMINAMATH_GPT_polynomial_value_at_minus_1_l102_10241


namespace NUMINAMATH_GPT_x_y_divisible_by_3_l102_10263

theorem x_y_divisible_by_3
    (x y z t : ℤ)
    (h : x^3 + y^3 = 3 * (z^3 + t^3)) :
    (3 ∣ x) ∧ (3 ∣ y) :=
by sorry

end NUMINAMATH_GPT_x_y_divisible_by_3_l102_10263


namespace NUMINAMATH_GPT__l102_10251

noncomputable def gear_speeds_relationship (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ) 
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : Prop :=
  ω₁ = (2 * z / x) * ω₃ ∧ ω₂ = (4 * z / (3 * y)) * ω₃

-- Example theorem statement
example (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ)
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : gear_speeds_relationship x y z ω₁ ω₂ ω₃ h1 h2 :=
by sorry

end NUMINAMATH_GPT__l102_10251


namespace NUMINAMATH_GPT_exist_sequences_l102_10254

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n ≥ 2, a n = 6 * a (n - 1) - a (n - 2)

theorem exist_sequences (a : ℕ → ℤ) (x y : ℕ → ℤ) :
  sequence_a a → (∀ n, 0 < x n ∧ 0 < y n) →
  (∀ n, a n = (y n ^ 2 + 7) / (x n - y n)) :=
by
  intro h_seq_a h_pos
  sorry

end NUMINAMATH_GPT_exist_sequences_l102_10254


namespace NUMINAMATH_GPT_team_won_five_games_l102_10223
-- Import the entire Mathlib library

-- Number of games played (given as a constant)
def numberOfGamesPlayed : ℕ := 10

-- Number of losses definition based on the ratio condition
def numberOfLosses : ℕ := numberOfGamesPlayed / 2

-- The number of wins is defined as the total games played minus the number of losses
def numberOfWins : ℕ := numberOfGamesPlayed - numberOfLosses

-- Proof statement: The number of wins is 5
theorem team_won_five_games :
  numberOfWins = 5 := by
  sorry

end NUMINAMATH_GPT_team_won_five_games_l102_10223


namespace NUMINAMATH_GPT_prime_sum_divisible_l102_10226

theorem prime_sum_divisible (p : Fin 2021 → ℕ) (prime : ∀ i, Nat.Prime (p i))
  (h : 6060 ∣ Finset.univ.sum (fun i => (p i)^4)) : 4 ≤ Finset.card (Finset.univ.filter (fun i => p i < 2021)) :=
sorry

end NUMINAMATH_GPT_prime_sum_divisible_l102_10226


namespace NUMINAMATH_GPT_max_length_PC_l102_10219

-- Define the circle C and its properties
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- The equilateral triangle condition and what we need to prove
theorem max_length_PC :
  (∃ (P A B : ℝ × ℝ), 
    (Circle A.1 A.2) ∧
    (Circle B.1 B.2) ∧
    (Circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) ∧
    (A ≠ B) ∧
    (∃ r : ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
               (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧ 
               (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2)) → 
  (∀ (P : ℝ × ℝ), 
     ∃ (max_val : ℝ), max_val = 4 ∧
     (¬(∃ (Q : ℝ × ℝ), (Circle P.1 P.2) ∧ ((Q.1 - 0)^2 + (Q.2 - 1)^2 > max_val^2))))
:= 
sorry

end NUMINAMATH_GPT_max_length_PC_l102_10219


namespace NUMINAMATH_GPT_problem_statement_l102_10218

def are_collinear (A B C : Point) : Prop := sorry -- Definition for collinearity should be expanded.
def area (A B C : Point) : ℝ := sorry -- Definition for area must be provided.

theorem problem_statement :
  ∀ n : ℕ, (n > 3) →
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ are_collinear (A i) (A j) (A k)) ∧
    (∀ i j k : Fin n, area (A i) (A j) (A k) = r i + r j + r k)) →
  n = 4 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l102_10218


namespace NUMINAMATH_GPT_monotone_f_range_a_l102_10292

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

theorem monotone_f_range_a (a : ℝ) :
  (∀ (x y : ℝ), x <= y → f a x >= f a y) →
  1 / 2 <= a ∧ a <= 5 / 8 :=
sorry

end NUMINAMATH_GPT_monotone_f_range_a_l102_10292


namespace NUMINAMATH_GPT_fg_minus_gf_eq_zero_l102_10298

noncomputable def f (x : ℝ) : ℝ := 4 * x + 6

noncomputable def g (x : ℝ) : ℝ := x / 2 - 1

theorem fg_minus_gf_eq_zero (x : ℝ) : (f (g x)) - (g (f x)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fg_minus_gf_eq_zero_l102_10298


namespace NUMINAMATH_GPT_share_of_c_l102_10228

variable (a b c : ℝ)

theorem share_of_c (h1 : a + b + c = 427) (h2 : 3 * a = 7 * c) (h3 : 4 * b = 7 * c) : c = 84 :=
  by
  sorry

end NUMINAMATH_GPT_share_of_c_l102_10228


namespace NUMINAMATH_GPT_number_of_bottom_row_bricks_l102_10213

theorem number_of_bottom_row_bricks :
  ∃ (x : ℕ), (x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) ∧ x = 22 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_bottom_row_bricks_l102_10213


namespace NUMINAMATH_GPT_minimum_value_of_fm_plus_fp_l102_10225

def f (x a : ℝ) : ℝ := -x^3 + a * x^2 - 4

def f_prime (x a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_fm_plus_fp (a : ℝ) (h_extremum : f_prime 2 a = 0) (m n : ℝ) 
  (hm : -1 ≤ m ∧ m ≤ 1) (hn : -1 ≤ n ∧ n ≤ 1) : 
  f m a + f_prime n a = -13 := 
by
  -- steps of the proof would go here
  sorry

end NUMINAMATH_GPT_minimum_value_of_fm_plus_fp_l102_10225


namespace NUMINAMATH_GPT_max_value_of_function_l102_10268

theorem max_value_of_function : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y == (2*x^2 - 2*x + 3) / (x^2 - x + 1)) → y ≤ 10/3) ∧
  (∃ x : ℝ, (2*x^2 - 2*x + 3) / (x^2 - x + 1) = 10/3) := 
sorry

end NUMINAMATH_GPT_max_value_of_function_l102_10268


namespace NUMINAMATH_GPT_line_l_statements_correct_l102_10224

theorem line_l_statements_correct
  (A B C : ℝ)
  (hAB : ¬(A = 0 ∧ B = 0)) :
  ( (2 * A + B + C = 0 → ∀ x y, A * (x - 2) + B * (y - 1) = 0 ↔ A * x + B * y + C = 0 ) ∧
    ((A ≠ 0 ∧ B ≠ 0) → ∃ x, A * x + C = 0 ∧ ∃ y, B * y + C = 0) ∧
    (A = 0 ∧ B ≠ 0 ∧ C ≠ 0 → ∀ y, B * y + C = 0 ↔ y = -C / B) ∧
    (A ≠ 0 ∧ B^2 + C^2 = 0 → ∀ x, A * x = 0 ↔ x = 0) ) :=
by
  sorry

end NUMINAMATH_GPT_line_l_statements_correct_l102_10224


namespace NUMINAMATH_GPT_measure_of_y_l102_10243

variables (A B C D : Point) (y : ℝ)
-- Given conditions
def angle_ABC := 120
def angle_BAD := 30
def angle_BDA := 21
def angle_ABD := 180 - angle_ABC

-- Theorem to prove
theorem measure_of_y :
  angle_BAD + angle_ABD + angle_BDA + y = 180 → y = 69 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_y_l102_10243


namespace NUMINAMATH_GPT_Patricia_read_21_books_l102_10297

theorem Patricia_read_21_books
  (Candice_books Amanda_books Kara_books Patricia_books : ℕ)
  (h1 : Candice_books = 18)
  (h2 : Candice_books = 3 * Amanda_books)
  (h3 : Kara_books = Amanda_books / 2)
  (h4 : Patricia_books = 7 * Kara_books) :
  Patricia_books = 21 :=
by
  sorry

end NUMINAMATH_GPT_Patricia_read_21_books_l102_10297


namespace NUMINAMATH_GPT_exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l102_10238

-- Define the conditions
def num_mathematicians (n : ℕ) : ℕ := 6 * n + 4
def num_meetings (n : ℕ) : ℕ := 2 * n + 1
def num_4_person_tables (n : ℕ) : ℕ := 1
def num_6_person_tables (n : ℕ) : ℕ := n

-- Define the constraint on arrangements
def valid_arrangement (n : ℕ) : Prop :=
  -- A placeholder for the actual arrangement checking logic.
  -- This should ensure no two people sit next to or opposite each other more than once.
  sorry

-- Proof of existence of a valid arrangement when n = 1
theorem exists_valid_arrangement_n_1 : valid_arrangement 1 :=
sorry

-- Proof of existence of a valid arrangement when n > 1
theorem exists_valid_arrangement_n_gt_1 (n : ℕ) (h : n > 1) : valid_arrangement n :=
sorry

end NUMINAMATH_GPT_exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l102_10238


namespace NUMINAMATH_GPT_max_value_a_l102_10205

theorem max_value_a (a b c d : ℝ) 
  (h1 : a ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : b ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : c ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h4 : d ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h5 : Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1)
  (h6 : Real.cos (2 * a) + Real.cos (2 * b) + Real.cos (2 * c) + Real.cos (2 * d) ≥ 10 / 3) : 
  a ≤ Real.arcsin (1 / 2) := 
sorry

end NUMINAMATH_GPT_max_value_a_l102_10205


namespace NUMINAMATH_GPT_jogging_track_circumference_l102_10276

noncomputable def Deepak_speed : ℝ := 4.5 -- km/hr
noncomputable def Wife_speed : ℝ := 3.75 -- km/hr
noncomputable def time_meet : ℝ := 4.8 / 60 -- hours

noncomputable def Distance_Deepak : ℝ := Deepak_speed * time_meet
noncomputable def Distance_Wife : ℝ := Wife_speed * time_meet

theorem jogging_track_circumference : 2 * (Distance_Deepak + Distance_Wife) = 1.32 := by
  sorry

end NUMINAMATH_GPT_jogging_track_circumference_l102_10276


namespace NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l102_10207

open Set

def setA : Set ℝ := {x | -4 < x ∧ x < 2}
def setB : Set ℝ := {x | x < -5 ∨ x > 1}
def setComplementB : Set ℝ := {x | -5 ≤ x ∧ x ≤ 1}

theorem union_of_A_and_B : setA ∪ setB = {x | x < -5 ∨ x > -4} := by
  sorry

theorem intersection_of_A_and_complementB : setA ∩ setComplementB = {x | -4 < x ∧ x ≤ 1} := by
  sorry

noncomputable def setC (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) (h : setB ∩ (setC m) = ∅) : -4 ≤ m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l102_10207


namespace NUMINAMATH_GPT_polynomial_divisible_by_x_sub_a_squared_l102_10291

theorem polynomial_divisible_by_x_sub_a_squared (a x : ℕ) (n : ℕ) 
    (h : a ≠ 0) : ∃ q : ℕ → ℕ, x ^ n - n * a ^ (n - 1) * x + (n - 1) * a ^ n = (x - a) ^ 2 * q x := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_divisible_by_x_sub_a_squared_l102_10291


namespace NUMINAMATH_GPT_pq_plus_sum_eq_20_l102_10295

theorem pq_plus_sum_eq_20 
  (p q : ℕ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hpl : p < 30) 
  (hql : q < 30) 
  (heq : p + q + p * q = 119) : 
  p + q = 20 :=
sorry

end NUMINAMATH_GPT_pq_plus_sum_eq_20_l102_10295


namespace NUMINAMATH_GPT_complement_intersection_l102_10232

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

-- Define the complements of A and B with respect to U
def complement_U (s : Set ℕ) := { x ∈ U | x ∉ s }

-- The theorem to prove the intersection of the complements
theorem complement_intersection :
  (complement_U A) ∩ (complement_U B) = {7, 9} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l102_10232


namespace NUMINAMATH_GPT_combined_weight_after_removal_l102_10293

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end NUMINAMATH_GPT_combined_weight_after_removal_l102_10293


namespace NUMINAMATH_GPT_probability_300_feet_or_less_l102_10236

noncomputable def calculate_probability : ℚ :=
  let gates := 16
  let distance := 75
  let max_distance := 300
  let initial_choices := gates
  let final_choices := gates - 1 -- because the final choice cannot be the same as the initial one
  let total_choices := initial_choices * final_choices
  let valid_choices :=
    (2 * 4 + 2 * 5 + 2 * 6 + 2 * 7 + 8 * 8) -- the total valid assignments as calculated in the solution
  (valid_choices : ℚ) / total_choices

theorem probability_300_feet_or_less : calculate_probability = 9 / 20 := 
by 
  sorry

end NUMINAMATH_GPT_probability_300_feet_or_less_l102_10236


namespace NUMINAMATH_GPT_second_train_length_l102_10240

noncomputable def length_of_second_train (speed1_kmph speed2_kmph : ℝ) (time_seconds : ℝ) (length1_meters : ℝ) : ℝ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed_mps := speed1_mps + speed2_mps
  let distance := relative_speed_mps * time_seconds
  distance - length1_meters

theorem second_train_length :
  length_of_second_train 72 18 17.998560115190784 200 = 250 :=
by
  sorry

end NUMINAMATH_GPT_second_train_length_l102_10240


namespace NUMINAMATH_GPT_potatoes_leftover_l102_10220

-- Define the necessary conditions
def fries_per_potato : ℕ := 25
def total_potatoes : ℕ := 15
def fries_needed : ℕ := 200

-- Prove the goal
theorem potatoes_leftover : total_potatoes - (fries_needed / fries_per_potato) = 7 :=
sorry

end NUMINAMATH_GPT_potatoes_leftover_l102_10220


namespace NUMINAMATH_GPT_number_of_female_officers_l102_10267

theorem number_of_female_officers (h1 : 0.19 * T = 76) (h2 : T = 152 / 2) : T = 400 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_officers_l102_10267


namespace NUMINAMATH_GPT_inequality_holds_equality_condition_l102_10264

theorem inequality_holds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) ≥ 1 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) = 1 / 2 ↔ 
  ((a = 0 ∧ b = 0 ∧ 0 < c) ∨ (a = 0 ∧ c = 0 ∧ 0 < b) ∨ (b = 0 ∧ c = 0 ∧ 0 < a)) :=
sorry

end NUMINAMATH_GPT_inequality_holds_equality_condition_l102_10264


namespace NUMINAMATH_GPT_divisible_2n_minus_3_l102_10278

theorem divisible_2n_minus_3 (n : ℕ) : (2^n - 1)^n - 3 ≡ 0 [MOD 2^n - 3] :=
by
  sorry

end NUMINAMATH_GPT_divisible_2n_minus_3_l102_10278


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l102_10234

variable (A B : Set ℝ)
def C_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem problem_part1 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  C_R (A ∩ B) = { x : ℝ | x < 3 ∨ x ≥ 6 } :=
by
  intros hA hB
  sorry

theorem problem_part2 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  (C_R B) ∪ A = { x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by
  intros hA hB
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l102_10234


namespace NUMINAMATH_GPT_angle_C_obtuse_l102_10255

theorem angle_C_obtuse (a b c C : ℝ) (h1 : a^2 + b^2 < c^2) (h2 : Real.sin C = Real.sqrt 3 / 2) : C = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_obtuse_l102_10255


namespace NUMINAMATH_GPT_susan_remaining_spaces_to_win_l102_10248

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end NUMINAMATH_GPT_susan_remaining_spaces_to_win_l102_10248


namespace NUMINAMATH_GPT_masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l102_10201

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end NUMINAMATH_GPT_masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l102_10201


namespace NUMINAMATH_GPT_tan_identity_l102_10229

variable (α β : ℝ)

theorem tan_identity (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : Real.sin (2 * α) = 2 * Real.sin (2 * β)) : 
  Real.tan (α + β) = 3 * Real.tan (α - β) := 
by 
  sorry

end NUMINAMATH_GPT_tan_identity_l102_10229


namespace NUMINAMATH_GPT_remainder_of_f_div_r_minus_2_l102_10277

def f (r : ℝ) : ℝ := r^15 - 3

theorem remainder_of_f_div_r_minus_2 : f 2 = 32765 := by
  sorry

end NUMINAMATH_GPT_remainder_of_f_div_r_minus_2_l102_10277


namespace NUMINAMATH_GPT_sum_of_x_coordinates_where_g_eq_2_5_l102_10214

def g1 (x : ℝ) : ℝ := 3 * x + 6
def g2 (x : ℝ) : ℝ := -x + 2
def g3 (x : ℝ) : ℝ := 2 * x - 2
def g4 (x : ℝ) : ℝ := -2 * x + 8

def is_within (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

theorem sum_of_x_coordinates_where_g_eq_2_5 :
     (∀ x, g1 x = 2.5 → (is_within x (-4) (-2) → false)) ∧
     (∀ x, g2 x = 2.5 → (is_within x (-2) (0) → x = -0.5)) ∧
     (∀ x, g3 x = 2.5 → (is_within x 0 3 → x = 2.25)) ∧
     (∀ x, g4 x = 2.5 → (is_within x 3 5 → x = 2.75)) →
     (-0.5 + 2.25 + 2.75 = 4.5) :=
by { sorry }

end NUMINAMATH_GPT_sum_of_x_coordinates_where_g_eq_2_5_l102_10214


namespace NUMINAMATH_GPT_simplify_expression_l102_10265

variable (x : ℝ)

theorem simplify_expression : (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l102_10265


namespace NUMINAMATH_GPT_fish_remaining_when_discovered_l102_10200

def start_fish := 60
def fish_eaten_per_day := 2
def days_two_weeks := 2 * 7
def fish_added_after_two_weeks := 8
def days_one_week := 7

def fish_after_two_weeks (start: ℕ) (eaten_per_day: ℕ) (days: ℕ) (added: ℕ): ℕ :=
  start - eaten_per_day * days + added

def fish_after_three_weeks (fish_after_two_weeks: ℕ) (eaten_per_day: ℕ) (days: ℕ): ℕ :=
  fish_after_two_weeks - eaten_per_day * days

theorem fish_remaining_when_discovered :
  (fish_after_three_weeks (fish_after_two_weeks start_fish fish_eaten_per_day days_two_weeks fish_added_after_two_weeks) fish_eaten_per_day days_one_week) = 26 := 
by {
  sorry
}

end NUMINAMATH_GPT_fish_remaining_when_discovered_l102_10200


namespace NUMINAMATH_GPT_mixed_fractions_calculation_l102_10237

theorem mixed_fractions_calculation :
  2017 + (2016 / 2017) / (2019 + (1 / 2016)) + (1 / 2017) = 1 :=
by
  sorry

end NUMINAMATH_GPT_mixed_fractions_calculation_l102_10237


namespace NUMINAMATH_GPT_convert_to_base7_l102_10211

theorem convert_to_base7 : 3589 = 1 * 7^4 + 3 * 7^3 + 3 * 7^2 + 1 * 7^1 + 5 * 7^0 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_base7_l102_10211


namespace NUMINAMATH_GPT_solve_sqrt_eq_l102_10245

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 3) ↔ (x = 2 ∨ x = -2) := 
by sorry

end NUMINAMATH_GPT_solve_sqrt_eq_l102_10245


namespace NUMINAMATH_GPT_min_value_exp_l102_10212

theorem min_value_exp (x y : ℝ) (h : x + 2 * y = 4) : ∃ z : ℝ, (2^x + 4^y = z) ∧ (∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ z) :=
sorry

end NUMINAMATH_GPT_min_value_exp_l102_10212


namespace NUMINAMATH_GPT_original_number_of_students_l102_10235

theorem original_number_of_students (x : ℕ)
  (h1: 40 * x / x = 40)
  (h2: 12 * 34 = 408)
  (h3: (40 * x + 408) / (x + 12) = 36) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_students_l102_10235


namespace NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l102_10209

theorem range_of_a_for_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, ¬ (|x - 3| + |x - 4| < a)) ↔ a ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l102_10209


namespace NUMINAMATH_GPT_bcdeq65_l102_10287

theorem bcdeq65 (a b c d e f : ℝ)
  (h₁ : a * b * c = 130)
  (h₂ : c * d * e = 500)
  (h₃ : d * e * f = 250)
  (h₄ : (a * f) / (c * d) = 1) :
  b * c * d = 65 :=
sorry

end NUMINAMATH_GPT_bcdeq65_l102_10287


namespace NUMINAMATH_GPT_factorization_correct_l102_10286

theorem factorization_correct : ∃ a b : ℤ, (5*y + a)*(y + b) = 5*y^2 + 17*y + 6 ∧ a - b = -1 := by
  sorry

end NUMINAMATH_GPT_factorization_correct_l102_10286


namespace NUMINAMATH_GPT_total_distance_covered_l102_10290

theorem total_distance_covered :
  let speed_upstream := 12 -- km/h
  let time_upstream := 2 -- hours
  let speed_downstream := 38 -- km/h
  let time_downstream := 1 -- hour
  let distance_upstream := speed_upstream * time_upstream
  let distance_downstream := speed_downstream * time_downstream
  distance_upstream + distance_downstream = 62 := by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l102_10290


namespace NUMINAMATH_GPT_surface_area_ratio_l102_10274

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r ^ 2

theorem surface_area_ratio (k : ℝ) :
  let r1 := k
  let r2 := 2 * k
  let r3 := 3 * k
  let A1 := surface_area r1
  let A2 := surface_area r2
  let A3 := surface_area r3
  A3 / (A1 + A2) = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_ratio_l102_10274


namespace NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_l102_10215

theorem largest_divisor_of_n4_minus_n2 :
  ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_l102_10215


namespace NUMINAMATH_GPT_cos_product_identity_l102_10246

theorem cos_product_identity :
  3.422 * (Real.cos (π / 15)) * (Real.cos (2 * π / 15)) * (Real.cos (3 * π / 15)) *
  (Real.cos (4 * π / 15)) * (Real.cos (5 * π / 15)) * (Real.cos (6 * π / 15)) * (Real.cos (7 * π / 15)) =
  (1 / 2^7) :=
sorry

end NUMINAMATH_GPT_cos_product_identity_l102_10246


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l102_10227

open Set

theorem intersection_of_A_and_B (A B : Set ℕ) (hA : A = {1, 2, 4}) (hB : B = {2, 4, 6}) : A ∩ B = {2, 4} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l102_10227


namespace NUMINAMATH_GPT_expression_equals_two_l102_10272

noncomputable def expression (a b c : ℝ) : ℝ :=
  (1 + a) / (1 + a + a * b) + (1 + b) / (1 + b + b * c) + (1 + c) / (1 + c + c * a)

theorem expression_equals_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  expression a b c = 2 := by
  sorry

end NUMINAMATH_GPT_expression_equals_two_l102_10272


namespace NUMINAMATH_GPT_density_function_Y_l102_10288

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-x^2 / 2)

theorem density_function_Y (y : ℝ) (hy : 0 < y) : 
  (∃ (g : ℝ → ℝ), (∀ y, g y = (1 / Real.sqrt (2 * Real.pi * y)) * Real.exp (- y / 2))) :=
sorry

end NUMINAMATH_GPT_density_function_Y_l102_10288


namespace NUMINAMATH_GPT_rational_expr_evaluation_l102_10221

theorem rational_expr_evaluation (a b c : ℚ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a + b + c = a * b * c) :
  (a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a) = -3 :=
by
  sorry

end NUMINAMATH_GPT_rational_expr_evaluation_l102_10221


namespace NUMINAMATH_GPT_trip_duration_exactly_six_hours_l102_10266

theorem trip_duration_exactly_six_hours : 
  ∀ start_time end_time : ℕ,
  (start_time = (8 * 60 + 43 * 60 / 11)) ∧ 
  (end_time = (14 * 60 + 43 * 60 / 11)) → 
  (end_time - start_time) = 6 * 60 :=
by
  sorry

end NUMINAMATH_GPT_trip_duration_exactly_six_hours_l102_10266


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_converse_implies_l102_10204

theorem necessary_but_not_sufficient (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * (Real.log x) ^ 2 < 1) → (x * Real.log x < 1) :=
sorry

theorem converse_implies (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * Real.log x < 1) → (x * (Real.log x) ^ 2 < 1) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_converse_implies_l102_10204


namespace NUMINAMATH_GPT_total_water_in_containers_l102_10203

/-
We have four containers. The first three contain water, while the fourth is empty. 
The second container holds twice as much water as the first, and the third holds twice as much water as the second. 
We transfer half of the water from the first container, one-third of the water from the second container, 
and one-quarter of the water from the third container into the fourth container. 
Now, there are 26 liters of water in the fourth container. Prove that initially, 
there were 84 liters of water in total in the first three containers.
-/

theorem total_water_in_containers (x : ℕ) (h1 : x / 2 + 2 * x / 3 + x = 26) : x + 2 * x + 4 * x = 84 := 
sorry

end NUMINAMATH_GPT_total_water_in_containers_l102_10203


namespace NUMINAMATH_GPT_stratified_sampling_employees_over_50_l102_10210

theorem stratified_sampling_employees_over_50 :
  let total_employees := 500
  let employees_under_35 := 125
  let employees_35_to_50 := 280
  let employees_over_50 := 95
  let total_samples := 100
  (employees_over_50 / total_employees * total_samples) = 19 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_employees_over_50_l102_10210


namespace NUMINAMATH_GPT_find_y_value_l102_10269

theorem find_y_value (x y : ℝ) (h1 : x^2 + y^2 - 4 = 0) (h2 : x^2 - y + 2 = 0) : y = 2 :=
by sorry

end NUMINAMATH_GPT_find_y_value_l102_10269


namespace NUMINAMATH_GPT_find_friends_l102_10222

-- Definitions
def shells_Jillian : Nat := 29
def shells_Savannah : Nat := 17
def shells_Clayton : Nat := 8
def shells_per_friend : Nat := 27

-- Main statement
theorem find_friends :
  (shells_Jillian + shells_Savannah + shells_Clayton) / shells_per_friend = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_friends_l102_10222


namespace NUMINAMATH_GPT_julia_bill_ratio_l102_10270

-- Definitions
def saturday_miles_b (s_b : ℕ) (s_su : ℕ) := s_su = s_b + 4
def sunday_miles_j (s_su : ℕ) (t : ℕ) (s_j : ℕ) := s_j = t * s_su
def total_weekend_miles (s_b : ℕ) (s_su : ℕ) (s_j : ℕ) := s_b + s_su + s_j = 36

-- Proof statement
theorem julia_bill_ratio (s_b s_su s_j : ℕ) (h1 : saturday_miles_b s_b s_su) (h3 : total_weekend_miles s_b s_su s_j) (h_su : s_su = 10) : (2 * s_su = s_j) :=
by
  sorry  -- proof

end NUMINAMATH_GPT_julia_bill_ratio_l102_10270


namespace NUMINAMATH_GPT_shirts_count_l102_10233

theorem shirts_count (S : ℕ) (hours_per_shirt hours_per_pant cost_per_hour total_pants total_cost : ℝ) :
  hours_per_shirt = 1.5 →
  hours_per_pant = 3 →
  cost_per_hour = 30 →
  total_pants = 12 →
  total_cost = 1530 →
  45 * S + 1080 = total_cost →
  S = 10 :=
by
  intros hps hpp cph tp tc cost_eq
  sorry

end NUMINAMATH_GPT_shirts_count_l102_10233


namespace NUMINAMATH_GPT_sphere_surface_area_ratio_l102_10259

theorem sphere_surface_area_ratio (V1 V2 r1 r2 A1 A2 : ℝ)
    (h_volume_ratio : V1 / V2 = 8 / 27)
    (h_volume_formula1 : V1 = (4/3) * Real.pi * r1^3)
    (h_volume_formula2 : V2 = (4/3) * Real.pi * r2^3)
    (h_surface_area_formula1 : A1 = 4 * Real.pi * r1^2)
    (h_surface_area_formula2 : A2 = 4 * Real.pi * r2^2)
    (h_radius_ratio : r1 / r2 = 2 / 3) :
  A1 / A2 = 4 / 9 :=
sorry

end NUMINAMATH_GPT_sphere_surface_area_ratio_l102_10259


namespace NUMINAMATH_GPT_solve_equation_l102_10250

theorem solve_equation (x : ℝ) (h : x = 5) :
  (3 * x - 5) / (x^2 - 7 * x + 12) + (5 * x - 1) / (x^2 - 5 * x + 6) = (8 * x - 13) / (x^2 - 6 * x + 8) := 
  by 
  rw [h]
  sorry

end NUMINAMATH_GPT_solve_equation_l102_10250


namespace NUMINAMATH_GPT_polynomial_divisible_by_x_minus_4_l102_10271

theorem polynomial_divisible_by_x_minus_4 (m : ℤ) :
  (∀ x, 6 * x ^ 3 - 12 * x ^ 2 + m * x - 24 = 0 → x = 4) ↔ m = -42 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_by_x_minus_4_l102_10271


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l102_10249

-- Define the asymptotes as lines
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 7

-- Define the condition that the hyperbola passes through the point (4, 5)
def passes_through (x y : ℝ) : Prop := (x, y) = (4, 5)

-- Statement to prove
theorem distance_between_foci_of_hyperbola : 
  (asymptote1 4 = 5) ∧ (asymptote2 4 = 5) ∧ passes_through 4 5 → 
  (∀ a b c : ℝ, a^2 = 9 ∧ b^2 = 9/4 ∧ c^2 = a^2 + b^2 → 2 * c = 3 * Real.sqrt 5) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l102_10249


namespace NUMINAMATH_GPT_haley_money_difference_l102_10206

def initial_amount : ℕ := 2
def chores : ℕ := 5
def birthday : ℕ := 10
def neighbor : ℕ := 7
def candy : ℕ := 3
def lost : ℕ := 2

theorem haley_money_difference : (initial_amount + chores + birthday + neighbor - candy - lost) - initial_amount = 17 := by
  sorry

end NUMINAMATH_GPT_haley_money_difference_l102_10206


namespace NUMINAMATH_GPT_triangle_inequality_l102_10260

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l102_10260


namespace NUMINAMATH_GPT_syntheticMethod_correct_l102_10258

-- Definition: The synthetic method leads from cause to effect.
def syntheticMethod (s : String) : Prop :=
  s = "The synthetic method leads from cause to effect, gradually searching for the necessary conditions that are known."

-- Question: Is the statement correct?
def question : String :=
  "The thought process of the synthetic method is to lead from cause to effect, gradually searching for the necessary conditions that are known."

-- Options given
def options : List String := ["Correct", "Incorrect", "", ""]

-- Correct answer is Option A - "Correct"
def correctAnswer : String := "Correct"

theorem syntheticMethod_correct :
  syntheticMethod question → options.head? = some correctAnswer :=
sorry

end NUMINAMATH_GPT_syntheticMethod_correct_l102_10258


namespace NUMINAMATH_GPT_min_bailing_rate_l102_10283

noncomputable def slowest_bailing_rate (distance : ℝ) (rowing_speed : ℝ) (leak_rate : ℝ) (max_capacity : ℝ) : ℝ :=
  let time_to_shore := distance / rowing_speed
  let time_to_shore_in_minutes := time_to_shore * 60
  let total_water_intake := leak_rate * time_to_shore_in_minutes
  let excess_water := total_water_intake - max_capacity
  excess_water / time_to_shore_in_minutes

theorem min_bailing_rate : slowest_bailing_rate 3 3 14 40 = 13.3 :=
by
  sorry

end NUMINAMATH_GPT_min_bailing_rate_l102_10283


namespace NUMINAMATH_GPT_fixed_point_of_line_l102_10281

theorem fixed_point_of_line :
  ∀ m : ℝ, ∀ x y : ℝ, (y - 2 = m * (x + 1)) → (x = -1 ∧ y = 2) :=
by sorry

end NUMINAMATH_GPT_fixed_point_of_line_l102_10281


namespace NUMINAMATH_GPT_contractor_realized_after_20_days_l102_10261

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end NUMINAMATH_GPT_contractor_realized_after_20_days_l102_10261


namespace NUMINAMATH_GPT_math_problems_l102_10242

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_math_problems_l102_10242


namespace NUMINAMATH_GPT_best_scrap_year_limit_l102_10289

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end NUMINAMATH_GPT_best_scrap_year_limit_l102_10289


namespace NUMINAMATH_GPT_original_height_in_feet_l102_10208

-- Define the current height in inches
def current_height_in_inches : ℚ := 180

-- Define the percentage increase in height
def percentage_increase : ℚ := 0.5

-- Define the conversion factor from inches to feet
def inches_to_feet : ℚ := 12

-- Define the initial height in inches
def initial_height_in_inches : ℚ := current_height_in_inches / (1 + percentage_increase)

-- Prove that the original height in feet was 10 feet
theorem original_height_in_feet : initial_height_in_inches / inches_to_feet = 10 :=
by
  -- Placeholder for the full proof
  sorry

end NUMINAMATH_GPT_original_height_in_feet_l102_10208


namespace NUMINAMATH_GPT_work_duration_l102_10273

/-- Definition of the work problem, showing that the work lasts for 5 days. -/
theorem work_duration (work_rate_p work_rate_q : ℝ) (total_work time_p time_q : ℝ) 
  (p_work_days q_work_days : ℝ) 
  (H1 : p_work_days = 10)
  (H2 : q_work_days = 6)
  (H3 : work_rate_p = total_work / 10)
  (H4 : work_rate_q = total_work / 6)
  (H5 : time_p = 2)
  (H6 : time_q = 4 * total_work / 5 / (total_work / 2 / 3) )
  : (time_p + time_q = 5) := 
by 
  sorry

end NUMINAMATH_GPT_work_duration_l102_10273


namespace NUMINAMATH_GPT_ted_age_l102_10202

theorem ted_age (t s : ℝ) 
  (h1 : t = 3 * s - 20) 
  (h2: t + s = 70) : 
  t = 47.5 := 
by
  sorry

end NUMINAMATH_GPT_ted_age_l102_10202


namespace NUMINAMATH_GPT_evaluate_expression_l102_10253

theorem evaluate_expression (x : ℝ) (h : x = -3) : (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l102_10253


namespace NUMINAMATH_GPT_cos_difference_simplification_l102_10252

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  (y = 2 * x^2 - 1) →
  (x = 1 - 2 * y^2) →
  x - y = 1 / 2 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_cos_difference_simplification_l102_10252


namespace NUMINAMATH_GPT_find_number_satisfying_9y_eq_number12_l102_10256

noncomputable def power_9_y (y : ℝ) := (9 : ℝ) ^ y
noncomputable def root_12 (x : ℝ) := x ^ (1 / 12 : ℝ)

theorem find_number_satisfying_9y_eq_number12 :
  ∃ number : ℝ, power_9_y 6 = number ^ 12 ∧ abs (number - 3) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_find_number_satisfying_9y_eq_number12_l102_10256


namespace NUMINAMATH_GPT_solve_for_x_l102_10279

theorem solve_for_x (x : ℤ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l102_10279


namespace NUMINAMATH_GPT_sum_of_first_fifteen_terms_l102_10244

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_first_fifteen_terms_l102_10244


namespace NUMINAMATH_GPT_numCounterexamplesCorrect_l102_10216

-- Define a function to calculate the sum of digits of a number
def digitSum (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Predicate to check if a number is prime
def isPrime (n : Nat) : Prop := 
  Nat.Prime n

-- Set definition where the sum of digits must be 5 and all digits are non-zero
def validSet (n : Nat) : Prop :=
  digitSum n = 5 ∧ ∀ d ∈ n.digits 10, d ≠ 0

-- Define the number of counterexamples
def numCounterexamples : Nat := 6

-- The final theorem stating the number of counterexamples
theorem numCounterexamplesCorrect :
  (∃ ns : Finset Nat, 
    (∀ n ∈ ns, validSet n) ∧ 
    (∀ n ∈ ns, ¬ isPrime n) ∧ 
    ns.card = numCounterexamples) :=
sorry

end NUMINAMATH_GPT_numCounterexamplesCorrect_l102_10216


namespace NUMINAMATH_GPT_smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l102_10296

theorem smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62
  (n: ℕ) (h1: n - 8 = 44) 
  (h2: (n - 8) % 9 = 0)
  (h3: (n - 8) % 6 = 0)
  (h4: (n - 8) % 18 = 0) : 
  n = 62 :=
sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l102_10296


namespace NUMINAMATH_GPT_cistern_wet_surface_area_l102_10275

noncomputable def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  let bottom_surface_area := length * width
  let longer_side_area := 2 * (depth * length)
  let shorter_side_area := 2 * (depth * width)
  bottom_surface_area + longer_side_area + shorter_side_area

theorem cistern_wet_surface_area :
  total_wet_surface_area 9 4 1.25 = 68.5 :=
by
  sorry

end NUMINAMATH_GPT_cistern_wet_surface_area_l102_10275
