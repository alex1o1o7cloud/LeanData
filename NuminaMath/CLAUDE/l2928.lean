import Mathlib

namespace NUMINAMATH_CALUDE_product_closure_l2928_292827

def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem product_closure (x₁ x₂ : ℤ) (h₁ : x₁ ∈ M) (h₂ : x₂ ∈ M) : x₁ * x₂ ∈ M := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l2928_292827


namespace NUMINAMATH_CALUDE_white_blue_line_difference_l2928_292899

/-- The length difference between two lines -/
def length_difference (white_line blue_line : ℝ) : ℝ :=
  white_line - blue_line

/-- Theorem stating the length difference between the white and blue lines -/
theorem white_blue_line_difference :
  let white_line : ℝ := 7.666666666666667
  let blue_line : ℝ := 3.3333333333333335
  length_difference white_line blue_line = 4.333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_white_blue_line_difference_l2928_292899


namespace NUMINAMATH_CALUDE_evaluate_expression_l2928_292820

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12))^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2928_292820


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2928_292802

theorem quadratic_inequality_solution_sets (a : ℝ) :
  (∀ x, 6 * x^2 + a * x - a^2 < 0 ↔ 
    (a > 0 ∧ -a/2 < x ∧ x < a/3) ∨
    (a < 0 ∧ a/3 < x ∧ x < -a/2)) ∧
  (a = 0 → ∀ x, ¬(6 * x^2 + a * x - a^2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2928_292802


namespace NUMINAMATH_CALUDE_parents_present_l2928_292856

theorem parents_present (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 676) (h2 : pupils = 654) :
  total_people - pupils = 22 := by
  sorry

end NUMINAMATH_CALUDE_parents_present_l2928_292856


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2928_292859

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2928_292859


namespace NUMINAMATH_CALUDE_rectangle_diagonal_problem_l2928_292887

theorem rectangle_diagonal_problem (w l : ℝ) 
  (h1 : w^2 + l^2 = 400) 
  (h2 : 4*w^2 + l^2 = 484) : 
  w^2 = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_problem_l2928_292887


namespace NUMINAMATH_CALUDE_new_average_after_adding_l2928_292834

theorem new_average_after_adding (n : ℕ) (original_avg : ℚ) (add_value : ℚ) : 
  n > 0 → 
  let original_sum := n * original_avg
  let new_sum := original_sum + n * add_value
  let new_avg := new_sum / n
  n = 15 ∧ original_avg = 40 ∧ add_value = 10 → new_avg = 50 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_adding_l2928_292834


namespace NUMINAMATH_CALUDE_bus_stop_theorem_l2928_292801

/-- Represents the number of passengers boarding at stop i and alighting at stop j -/
def passenger_count (i j : Fin 6) : ℕ := sorry

/-- The total number of passengers on the bus between stops i and j -/
def bus_load (i j : Fin 6) : ℕ := sorry

theorem bus_stop_theorem :
  ∀ (passenger_count : Fin 6 → Fin 6 → ℕ),
  (∀ (i j : Fin 6), i < j → bus_load i j ≤ 5) →
  ∃ (A₁ B₁ A₂ B₂ : Fin 6),
    A₁ < B₁ ∧ A₂ < B₂ ∧ A₁ ≠ A₂ ∧ B₁ ≠ B₂ ∧
    passenger_count A₁ B₁ = 0 ∧ passenger_count A₂ B₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_theorem_l2928_292801


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2928_292896

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b + c = 15 :=        -- The perimeter is 15
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2928_292896


namespace NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_real_iff_l2928_292860

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 3 < x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- First part of the theorem
theorem union_when_a_is_3 :
  A 3 ∪ B = {x | x < -1 ∨ x > 0} :=
sorry

-- Second part of the theorem
theorem union_equals_real_iff :
  ∀ a : ℝ, A a ∪ B = Set.univ ↔ 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_real_iff_l2928_292860


namespace NUMINAMATH_CALUDE_fraction_simplification_l2928_292837

theorem fraction_simplification (a b : ℝ) (h1 : b ≠ 0) (h2 : a ≠ b) :
  (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2928_292837


namespace NUMINAMATH_CALUDE_coin_game_probability_l2928_292854

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The initial number of coins each player has -/
def initialCoins : Nat := 5

/-- The number of coins transferred when green and red balls are drawn -/
def coinTransfer : Nat := 2

/-- The total number of balls in the urn -/
def totalBalls : Nat := 5

/-- The number of green balls in the urn -/
def greenBalls : Nat := 1

/-- The number of red balls in the urn -/
def redBalls : Nat := 1

/-- The number of blue balls in the urn -/
def blueBalls : Nat := 3

/-- Represents the state of the game after each round -/
structure GameState :=
  (coins : Player → Nat)

/-- The probability of a specific pair (green/red) occurring in one round -/
def pairProbability : ℚ := 1 / 20

/-- 
Theorem: The probability that each player has exactly 5 coins after 5 rounds is 1/3,200,000
-/
theorem coin_game_probability : 
  ∀ (finalState : GameState),
    (∀ p : Player, finalState.coins p = initialCoins) →
    (pairProbability ^ numRounds : ℚ) = 1 / 3200000 := by
  sorry


end NUMINAMATH_CALUDE_coin_game_probability_l2928_292854


namespace NUMINAMATH_CALUDE_find_m_l2928_292816

def U : Set Nat := {1, 2, 3}

def A (m : Nat) : Set Nat := {1, m}

def complement_A : Set Nat := {2}

theorem find_m :
  ∃ (m : Nat), m ∈ U ∧ A m ∪ complement_A = U ∧ A m ∩ complement_A = ∅ ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2928_292816


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l2928_292883

theorem right_triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A + B = C) : C = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l2928_292883


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2928_292818

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 260)
  (h2 : rate = 7.142857142857143)
  (h3 : time = 4) :
  ∃ (principal : ℝ), principal = 910 ∧ interest = principal * rate * time / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2928_292818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2928_292885

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 8 + a 15 = Real.pi →
  Real.cos (a 4 + a 12) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2928_292885


namespace NUMINAMATH_CALUDE_amc_length_sum_amc_length_sum_value_l2928_292864

/-- The sum of lengths of line segments forming AMC on a unit-spaced grid --/
theorem amc_length_sum : ℝ := by
  -- Define the grid spacing
  let grid_spacing : ℝ := 1

  -- Define the lengths of different segments
  let a_diagonal : ℝ := Real.sqrt 2
  let a_horizontal : ℝ := 2
  let m_vertical : ℝ := 3
  let m_diagonal : ℝ := Real.sqrt 2
  let c_horizontal_long : ℝ := 3
  let c_horizontal_short : ℝ := 2

  -- Calculate the total length
  let total_length : ℝ := 
    2 * a_diagonal + a_horizontal + 
    2 * m_vertical + 2 * m_diagonal + 
    2 * c_horizontal_long + c_horizontal_short

  -- Prove that the total length equals 13 + 4√2
  sorry

/-- The result of amc_length_sum is equal to 13 + 4√2 --/
theorem amc_length_sum_value : amc_length_sum = 13 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_amc_length_sum_amc_length_sum_value_l2928_292864


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_for_x_squared_lt_4_l2928_292886

theorem necessary_sufficient_condition_for_x_squared_lt_4 :
  ∀ x : ℝ, x^2 < 4 ↔ -2 ≤ x ∧ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_for_x_squared_lt_4_l2928_292886


namespace NUMINAMATH_CALUDE_unique_number_pair_l2928_292813

theorem unique_number_pair : ∃! (x y : ℕ), 
  100 ≤ x ∧ x < 1000 ∧ 
  1000 ≤ y ∧ y < 10000 ∧ 
  10000 * x + y = 12 * x * y ∧
  x + y = 1083 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_pair_l2928_292813


namespace NUMINAMATH_CALUDE_square_side_length_l2928_292872

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 8 →
  rectangle_width = 10 →
  4 * square_side = 2 * (rectangle_length + rectangle_width) →
  square_side = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2928_292872


namespace NUMINAMATH_CALUDE_true_false_questions_count_l2928_292858

def number_of_multiple_choice_questions : ℕ := 2
def choices_per_multiple_choice_question : ℕ := 4
def total_answer_key_combinations : ℕ := 480

def valid_true_false_combinations (n : ℕ) : ℕ := 2^n - 2

theorem true_false_questions_count :
  ∃ n : ℕ, 
    n > 0 ∧
    valid_true_false_combinations n * 
    choices_per_multiple_choice_question ^ number_of_multiple_choice_questions = 
    total_answer_key_combinations ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_true_false_questions_count_l2928_292858


namespace NUMINAMATH_CALUDE_a_2006_mod_7_l2928_292823

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a n + (a (n + 1))^2

theorem a_2006_mod_7 : a 2006 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_2006_mod_7_l2928_292823


namespace NUMINAMATH_CALUDE_student_weight_l2928_292805

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 132)
  (h2 : student_weight - 6 = 2 * sister_weight) : 
  student_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_l2928_292805


namespace NUMINAMATH_CALUDE_scientific_notation_equality_coefficient_range_l2928_292892

-- Define the number we want to express in scientific notation
def number : ℕ := 18480000

-- Define the components of the scientific notation
def coefficient : ℝ := 1.848
def exponent : ℕ := 7

-- Theorem to prove
theorem scientific_notation_equality :
  (coefficient * (10 : ℝ) ^ exponent : ℝ) = number := by
  sorry

-- Verify that the coefficient is between 1 and 10
theorem coefficient_range :
  1 < coefficient ∧ coefficient < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_coefficient_range_l2928_292892


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l2928_292894

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem stating that the expected value of rolling the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l2928_292894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2928_292873

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (h₁ : a₁ = 150) (h₂ : aₙ = 42) (h₃ : d = -4) :
  (a₁ - aₙ) / d + 1 = 54 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2928_292873


namespace NUMINAMATH_CALUDE_most_likely_final_number_is_54_l2928_292825

/-- The initial number on the blackboard -/
def initial_number : ℕ := 15

/-- The lower bound of the random number added in each move -/
def lower_bound : ℕ := 1

/-- The upper bound of the random number added in each move -/
def upper_bound : ℕ := 5

/-- The threshold number for ending the game -/
def threshold : ℕ := 51

/-- The expected value of the random number added in each move -/
def expected_value : ℚ := (lower_bound + upper_bound) / 2

/-- The most likely final number on the blackboard -/
def most_likely_final_number : ℕ := 54

/-- Theorem stating that the most likely final number is 54 -/
theorem most_likely_final_number_is_54 :
  ∃ (n : ℕ), initial_number + n * expected_value > threshold ∧
             initial_number + (n - 1) * expected_value ≤ threshold ∧
             most_likely_final_number = initial_number + n * ⌊expected_value⌋ := by
  sorry

end NUMINAMATH_CALUDE_most_likely_final_number_is_54_l2928_292825


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2928_292882

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 2) > 1 ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2928_292882


namespace NUMINAMATH_CALUDE_max_self_intersection_points_seven_segments_l2928_292884

/-- A closed polyline is a sequence of connected line segments that form a closed loop. -/
def ClosedPolyline (n : ℕ) := Fin n → ℝ × ℝ

/-- The number of self-intersection points in a closed polyline. -/
def selfIntersectionPoints (p : ClosedPolyline 7) : ℕ := sorry

/-- The maximum number of self-intersection points in any closed polyline with 7 segments. -/
def maxSelfIntersectionPoints : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_self_intersection_points_seven_segments :
  maxSelfIntersectionPoints = 14 := by sorry

end NUMINAMATH_CALUDE_max_self_intersection_points_seven_segments_l2928_292884


namespace NUMINAMATH_CALUDE_candy_distribution_l2928_292835

theorem candy_distribution (marta_candies carmem_candies : ℕ) : 
  (marta_candies + carmem_candies = 200) →
  (marta_candies < 100) →
  (marta_candies > (4 * carmem_candies) / 5) →
  (∃ k : ℕ, marta_candies = 8 * k) →
  (∃ l : ℕ, carmem_candies = 8 * l) →
  (marta_candies = 96 ∧ carmem_candies = 104) := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2928_292835


namespace NUMINAMATH_CALUDE_max_actors_is_five_l2928_292865

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  -- The number of actors in the tournament
  num_actors : ℕ
  -- The results of games between actors
  results : Fin num_actors → Fin num_actors → ℝ
  -- Each actor plays every other actor exactly once
  played_once : ∀ i j : Fin num_actors, i ≠ j → results i j + results j i = 1
  -- Scores are either 0, 0.5, or 1
  valid_scores : ∀ i j : Fin num_actors, results i j ∈ ({0, 0.5, 1} : Set ℝ)
  -- For any three actors, one earned exactly 1.5 against the other two
  trio_condition : ∀ i j k : Fin num_actors, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (results i j + results i k = 1.5) ∨
    (results j i + results j k = 1.5) ∨
    (results k i + results k j = 1.5)

/-- The maximum number of actors in a valid chess tournament is 5 -/
theorem max_actors_is_five :
  (∃ t : ChessTournament, t.num_actors = 5) ∧
  (∀ t : ChessTournament, t.num_actors ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_actors_is_five_l2928_292865


namespace NUMINAMATH_CALUDE_integral_tan_over_trig_expression_l2928_292807

theorem integral_tan_over_trig_expression :
  let f := fun x : ℝ => (Real.tan x) / (Real.sin x ^ 2 - 5 * Real.cos x ^ 2 + 4)
  let a := Real.pi / 4
  let b := Real.arccos (1 / Real.sqrt 3)
  ∫ x in a..b, f x = (1 / 10) * Real.log (9 / 4) :=
by sorry

end NUMINAMATH_CALUDE_integral_tan_over_trig_expression_l2928_292807


namespace NUMINAMATH_CALUDE_no_integer_y_prime_abs_quadratic_l2928_292819

theorem no_integer_y_prime_abs_quadratic : ¬ ∃ y : ℤ, Nat.Prime (Int.natAbs (8*y^2 - 55*y + 21)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_y_prime_abs_quadratic_l2928_292819


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_snack_bag_l2928_292871

theorem min_blue_eyes_and_snack_bag 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (snack_bag : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 14) 
  (h3 : snack_bag = 22) 
  (h4 : blue_eyes ≤ total_students) 
  (h5 : snack_bag ≤ total_students) : 
  ∃ (both : ℕ), both ≥ 1 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ snack_bag ∧ 
    (blue_eyes - both) + (snack_bag - both) ≤ total_students := by
  sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_snack_bag_l2928_292871


namespace NUMINAMATH_CALUDE_remainder_of_g_x12_div_g_x_l2928_292851

/-- The polynomial g(x) = x^5 + x^4 + x^3 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The theorem stating that the remainder of g(x^12) divided by g(x) is 6 -/
theorem remainder_of_g_x12_div_g_x :
  ∃ (q : ℝ → ℝ), g (x^12) = g x * q x + 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_g_x12_div_g_x_l2928_292851


namespace NUMINAMATH_CALUDE_college_enrollment_l2928_292810

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 30 / 100

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 616

/-- Theorem stating the relationship between the total number of students,
    the percentage enrolled in biology, and the number not enrolled in biology -/
theorem college_enrollment :
  total_students = non_biology_students / (1 - biology_enrollment_percentage) := by
  sorry

end NUMINAMATH_CALUDE_college_enrollment_l2928_292810


namespace NUMINAMATH_CALUDE_composite_10201_composite_10101_l2928_292881

-- Definition for composite numbers
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Theorem 1: 10201 is composite in any base > 2
theorem composite_10201 (x : ℕ) (h : x > 2) : IsComposite (x^4 + 2*x^2 + 1) := by
  sorry

-- Theorem 2: 10101 is composite in any base ≥ 2
theorem composite_10101 (x : ℕ) (h : x ≥ 2) : IsComposite (x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_10201_composite_10101_l2928_292881


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2928_292891

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (first_part_wins : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  (first_part_wins : ℚ) / (first_part_games : ℚ) > target_percentage →
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ remaining_games ∧ 
    ((first_part_wins + remaining_wins : ℚ) / (total_games : ℚ) ≥ target_percentage) ∧
    (∀ (x : ℕ), x < remaining_wins → 
      (first_part_wins + x : ℚ) / (total_games : ℚ) < target_percentage) :=
by
  sorry

-- Example usage
example : 
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ 35 ∧ 
    ((45 + remaining_wins : ℚ) / 90 ≥ 3/4) ∧
    (∀ (x : ℕ), x < remaining_wins → (45 + x : ℚ) / 90 < 3/4) :=
basketball_win_rate 90 55 45 35 (3/4)
  (by norm_num)
  (by norm_num)

end NUMINAMATH_CALUDE_basketball_win_rate_l2928_292891


namespace NUMINAMATH_CALUDE_total_marble_weight_l2928_292877

def marble_weights : List Float := [
  0.3333333333333333,
  0.3333333333333333,
  0.08333333333333333,
  0.21666666666666667,
  0.4583333333333333,
  0.12777777777777778
]

theorem total_marble_weight :
  marble_weights.sum = 1.5527777777777777 := by sorry

end NUMINAMATH_CALUDE_total_marble_weight_l2928_292877


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2928_292847

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := 3 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (-3, 3) := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2928_292847


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l2928_292866

/-- Given a quadratic expression x^2 - 20x + 100 that can be written in the form (x + b)^2 + c,
    prove that b + c = -10 -/
theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 100 = (x + b)^2 + c) → b + c = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l2928_292866


namespace NUMINAMATH_CALUDE_music_festival_children_avg_age_l2928_292806

/-- Represents the demographics and age statistics of a music festival. -/
structure MusicFestival where
  total_participants : ℕ
  num_women : ℕ
  num_men : ℕ
  num_children : ℕ
  overall_avg_age : ℚ
  women_avg_age : ℚ
  men_avg_age : ℚ

/-- Calculates the average age of children in the music festival. -/
def children_avg_age (festival : MusicFestival) : ℚ :=
  (festival.total_participants * festival.overall_avg_age
   - festival.num_women * festival.women_avg_age
   - festival.num_men * festival.men_avg_age) / festival.num_children

/-- Theorem stating that for the given music festival data, the average age of children is 13. -/
theorem music_festival_children_avg_age :
  let festival : MusicFestival := {
    total_participants := 50,
    num_women := 22,
    num_men := 18,
    num_children := 10,
    overall_avg_age := 20,
    women_avg_age := 24,
    men_avg_age := 19
  }
  children_avg_age festival = 13 := by sorry

end NUMINAMATH_CALUDE_music_festival_children_avg_age_l2928_292806


namespace NUMINAMATH_CALUDE_average_speed_swim_run_l2928_292874

/-- 
Given a swimmer who swims at 1 mile per hour and runs at 11 miles per hour,
their average speed for these two events (assuming equal distances for both)
is 11/6 miles per hour.
-/
theorem average_speed_swim_run :
  let swim_speed : ℝ := 1
  let run_speed : ℝ := 11
  let total_distance : ℝ := 2 -- Assuming 1 mile each for swimming and running
  let swim_time : ℝ := 1 -- Time to swim 1 mile at 1 mph
  let run_time : ℝ := 1 / 11 -- Time to run 1 mile at 11 mph
  let total_time : ℝ := swim_time + run_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 11 / 6 := by sorry

end NUMINAMATH_CALUDE_average_speed_swim_run_l2928_292874


namespace NUMINAMATH_CALUDE_no_integer_solution_l2928_292869

theorem no_integer_solution : ¬∃ (a b : ℤ), a^2 + 1998 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2928_292869


namespace NUMINAMATH_CALUDE_probability_is_two_thirds_l2928_292800

/-- Given four evenly spaced points A, B, C, D on a number line with an interval of 1,
    this function calculates the probability that a randomly chosen point E on AD
    has a sum of distances to B and C less than 2. -/
def probability_sum_distances_less_than_two (A B C D : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the probability is 2/3 -/
theorem probability_is_two_thirds (A B C D : ℝ) 
  (h1 : B - A = 1) 
  (h2 : C - B = 1) 
  (h3 : D - C = 1) : 
  probability_sum_distances_less_than_two A B C D = 2/3 :=
sorry

end NUMINAMATH_CALUDE_probability_is_two_thirds_l2928_292800


namespace NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l2928_292888

-- Define the property of a function satisfying the given condition
def SatisfiesCondition (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

-- Define the set X_v
def X_v (f : ℤ → ℤ) (v : ℤ) : Set ℤ :=
  {x : ℤ | f x = v}

-- Define what it means for an integer to be rare under f
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  (X_v f v).Nonempty ∧ (X_v f v).Finite

-- Theorem statement
theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, SatisfiesCondition f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, SatisfiesCondition f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
sorry

end NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l2928_292888


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l2928_292811

/-- Represents a square-based pyramid made of unit cubes -/
structure CubePyramid where
  total_cubes : ℕ
  base_side_length : ℕ

/-- Calculates the total surface area of a cube pyramid -/
def total_surface_area (p : CubePyramid) : ℕ :=
  let base_area := p.base_side_length * p.base_side_length
  let top_area := base_area
  let vertical_area := 4 * (p.base_side_length * (p.base_side_length + 1) / 2)
  base_area + top_area + vertical_area

/-- Theorem stating that a pyramid of 30 cubes has a total surface area of 72 square units -/
theorem pyramid_surface_area :
  ∃ (p : CubePyramid), p.total_cubes = 30 ∧ total_surface_area p = 72 :=
by
  sorry


end NUMINAMATH_CALUDE_pyramid_surface_area_l2928_292811


namespace NUMINAMATH_CALUDE_subsets_of_size_two_l2928_292826

/-- Given a finite set S, returns the number of subsets of S with exactly k elements -/
def numSubsetsOfSize (n k : ℕ) : ℕ := Nat.choose n k

theorem subsets_of_size_two (S : Type) [Fintype S] :
  (numSubsetsOfSize (Fintype.card S) 7 = 36) →
  (numSubsetsOfSize (Fintype.card S) 2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_subsets_of_size_two_l2928_292826


namespace NUMINAMATH_CALUDE_log_inequality_l2928_292824

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2928_292824


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l2928_292844

/-- The least positive base ten number that requires seven digits for its binary representation -/
def leastSevenDigitBinary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary :
  (∀ m : ℕ, m < leastSevenDigitBinary → binaryDigits m < 7) ∧
  binaryDigits leastSevenDigitBinary = 7 := by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l2928_292844


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2928_292839

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  x₁ = -3/4 ∧ x₂ = -4/3 ∧ x₃ = 5/2 ∧
  x₁ * x₂ = 1 ∧
  24 * x₁^3 - 10 * x₁^2 - 101 * x₁ - 60 = 0 ∧
  24 * x₂^3 - 10 * x₂^2 - 101 * x₂ - 60 = 0 ∧
  24 * x₃^3 - 10 * x₃^2 - 101 * x₃ - 60 = 0 :=
by
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_cubic_equation_roots_l2928_292839


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2928_292808

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  2 = Nat.minFac (5^23 + 7^17) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2928_292808


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2928_292846

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 20 →  -- Mean is 20 more than the least
  (a + b + c) / 3 = c - 10 →  -- Mean is 10 less than the greatest
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2928_292846


namespace NUMINAMATH_CALUDE_ball_purchase_equation_l2928_292868

/-- Represents the price difference between a basketball and a soccer ball -/
def price_difference : ℝ := 20

/-- Represents the budget for basketballs -/
def basketball_budget : ℝ := 1500

/-- Represents the budget for soccer balls -/
def soccer_ball_budget : ℝ := 800

/-- Represents the quantity difference between basketballs and soccer balls purchased -/
def quantity_difference : ℝ := 5

/-- Theorem stating the equation that represents the relationship between
    the price of soccer balls and the quantities of basketballs and soccer balls purchased -/
theorem ball_purchase_equation (x : ℝ) :
  x > 0 →
  (basketball_budget / (x + price_difference) - soccer_ball_budget / x = quantity_difference) ↔
  (1500 / (x + 20) - 800 / x = 5) :=
by sorry

end NUMINAMATH_CALUDE_ball_purchase_equation_l2928_292868


namespace NUMINAMATH_CALUDE_ratio_equality_l2928_292843

theorem ratio_equality {a b c d : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a / b = c / d) : a / c = b / d := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2928_292843


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2928_292897

/-- Given two circles X and Y, if an arc of 60° on circle X has the same length as an arc of 40° on circle Y, 
    then the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (X Y : ℝ → ℝ → Prop) (R_X R_Y : ℝ) :
  (∃ L : ℝ, L = (60 / 360) * (2 * Real.pi * R_X) ∧ L = (40 / 360) * (2 * Real.pi * R_Y)) →
  (R_X > 0 ∧ R_Y > 0) →
  (X = λ x y => (x - 0)^2 + (y - 0)^2 = R_X^2) →
  (Y = λ x y => (x - 0)^2 + (y - 0)^2 = R_Y^2) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2928_292897


namespace NUMINAMATH_CALUDE_equal_expressions_l2928_292817

theorem equal_expressions : 
  (¬ (3^2 / 4 = (3/4)^2)) ∧ 
  (-1^2013 = (-1)^2025) ∧ 
  (¬ (-3^2 = (-3)^2)) ∧ 
  (¬ (-(2^2) / 3 = (-2)^2 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l2928_292817


namespace NUMINAMATH_CALUDE_amount_paid_is_correct_l2928_292853

/-- Calculates the amount paid to Jerry after discount --/
def amount_paid_after_discount (
  painting_hours : ℕ)
  (painting_rate : ℚ)
  (mowing_hours : ℕ)
  (mowing_rate : ℚ)
  (plumbing_hours : ℕ)
  (plumbing_rate : ℚ)
  (counter_time_multiplier : ℕ)
  (discount_rate : ℚ) : ℚ :=
  let painting_cost := painting_hours * painting_rate
  let mowing_cost := mowing_hours * mowing_rate
  let plumbing_cost := plumbing_hours * plumbing_rate
  let total_cost := painting_cost + mowing_cost + plumbing_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating that Miss Stevie paid $226.80 after the discount --/
theorem amount_paid_is_correct : 
  amount_paid_after_discount 8 15 6 10 4 18 3 (1/10) = 226.8 := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_is_correct_l2928_292853


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2928_292895

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x : ℕ | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2928_292895


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l2928_292840

/-- Given unit vectors e₁ and e₂ with an angle of 120° between them, 
    and x, y ∈ ℝ such that |x*e₁ + y*e₂| = √3, 
    prove that 1 ≤ |x*e₁ - y*e₂| ≤ 3 -/
theorem vector_magnitude_range (e₁ e₂ : ℝ × ℝ) (x y : ℝ) :
  (e₁.1^2 + e₁.2^2 = 1) →  -- e₁ is a unit vector
  (e₂.1^2 + e₂.2^2 = 1) →  -- e₂ is a unit vector
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = -1/2) →  -- angle between e₁ and e₂ is 120°
  ((x*e₁.1 + y*e₂.1)^2 + (x*e₁.2 + y*e₂.2)^2 = 3) →  -- |x*e₁ + y*e₂| = √3
  1 ≤ ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ∧ 
  ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l2928_292840


namespace NUMINAMATH_CALUDE_min_value_tangent_l2928_292893

/-- Given a function f(x) = 2cos(x) - 3sin(x) that reaches its minimum value when x = θ,
    prove that tan(θ) = -3/2 --/
theorem min_value_tangent (θ : ℝ) (h : ∀ x, 2 * Real.cos x - 3 * Real.sin x ≥ 2 * Real.cos θ - 3 * Real.sin θ) :
  Real.tan θ = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_tangent_l2928_292893


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l2928_292867

theorem contractor_engagement_days
  (daily_wage : ℕ)
  (daily_fine : ℚ)
  (total_pay : ℕ)
  (absent_days : ℕ)
  (h_daily_wage : daily_wage = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_pay : total_pay = 425)
  (h_absent_days : absent_days = 10) :
  ∃ (work_days : ℕ),
    (daily_wage : ℚ) * work_days - daily_fine * absent_days = total_pay ∧
    work_days + absent_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l2928_292867


namespace NUMINAMATH_CALUDE_proportion_problem_l2928_292848

/-- Given that a, d, b, c are in proportion, where a = 3 cm, b = 4 cm, and c = 6 cm, prove that d = 9/2 cm. -/
theorem proportion_problem (a d b c : ℚ) : 
  a = 3 → b = 4 → c = 6 → (a / d = b / c) → d = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2928_292848


namespace NUMINAMATH_CALUDE_y_derivative_l2928_292879

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.sin x

theorem y_derivative (x : ℝ) (h : Real.sin x ≠ 0) : 
  deriv y x = (-2*x*Real.sin x - (1 - x^2)*Real.cos x) / (Real.sin x)^2 :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2928_292879


namespace NUMINAMATH_CALUDE_drink_packing_l2928_292830

/-- The number of liters of Maaza -/
def maaza : ℕ := 215

/-- The number of liters of Pepsi -/
def pepsi : ℕ := 547

/-- The number of liters of Sprite -/
def sprite : ℕ := 991

/-- The least number of cans required to pack all drinks -/
def least_cans : ℕ := maaza + pepsi + sprite

theorem drink_packing :
  (∃ (can_size : ℕ), can_size > 0 ∧
    maaza % can_size = 0 ∧
    pepsi % can_size = 0 ∧
    sprite % can_size = 0) →
  least_cans = 1753 :=
sorry

end NUMINAMATH_CALUDE_drink_packing_l2928_292830


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2928_292809

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2928_292809


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2928_292870

theorem divisibility_by_five (x y : ℕ+) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 
  5 ∣ x.val := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2928_292870


namespace NUMINAMATH_CALUDE_taxi_fare_equation_l2928_292890

/-- Taxi fare calculation -/
theorem taxi_fare_equation 
  (x : ℝ) 
  (h_distance : x > 3) 
  (starting_price : ℝ := 6) 
  (price_per_km : ℝ := 2.4) 
  (total_fare : ℝ := 13.2) :
  starting_price + price_per_km * (x - 3) = total_fare := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equation_l2928_292890


namespace NUMINAMATH_CALUDE_dot_product_range_l2928_292875

/-- The locus M is defined as the set of points (x, y) satisfying x²/3 + y² = 1 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- F is the point (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- Given two points on M, compute their dot product with respect to F -/
def dot_product_with_F (C D : ℝ × ℝ) : ℝ :=
  let FC := (C.1 - F.1, C.2 - F.2)
  let FD := (D.1 - F.1, D.2 - F.2)
  FC.1 * FD.1 + FC.2 * FD.2

/-- The main theorem stating the range of the dot product -/
theorem dot_product_range (C D : ℝ × ℝ) (hC : C ∈ M) (hD : D ∈ M) 
  (h_line : ∃ (k : ℝ), C.2 = k * (C.1 - 2) ∧ D.2 = k * (D.1 - 2)) :
  1/3 < dot_product_with_F C D ∧ dot_product_with_F C D ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_dot_product_range_l2928_292875


namespace NUMINAMATH_CALUDE_square_card_arrangement_l2928_292841

theorem square_card_arrangement (perimeter_cards : ℕ) (h : perimeter_cards = 240) : 
  ∃ (side_length : ℕ), 
    4 * side_length - 4 = perimeter_cards ∧ 
    side_length * side_length = 3721 := by
  sorry

end NUMINAMATH_CALUDE_square_card_arrangement_l2928_292841


namespace NUMINAMATH_CALUDE_equation_proof_l2928_292861

theorem equation_proof (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2928_292861


namespace NUMINAMATH_CALUDE_sara_apples_l2928_292833

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples * (ali_ratio + 1) →
  sara_apples = 16 := by
sorry

end NUMINAMATH_CALUDE_sara_apples_l2928_292833


namespace NUMINAMATH_CALUDE_triangle_properties_l2928_292880

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.b - t.c) / t.a = (sin t.A - sin t.C) / (sin t.B + sin t.C))
  (h2 : t.a + t.c = 5)
  (h3 : 1/2 * t.a * t.c * sin t.B = 3 * sqrt 3 / 2) :
  t.B = π/3 ∧ t.b = sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2928_292880


namespace NUMINAMATH_CALUDE_unequal_outcome_probability_l2928_292822

theorem unequal_outcome_probability : 
  let n : ℕ := 12  -- number of grandchildren
  let p : ℝ := 1/2 -- probability of each gender
  let total_outcomes : ℕ := 2^n -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2) -- number of combinations with equal boys and girls
  
  (total_outcomes - equal_outcomes : ℝ) / total_outcomes = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_outcome_probability_l2928_292822


namespace NUMINAMATH_CALUDE_munchausen_polygon_theorem_l2928_292814

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop := sorry

/-- Counts the number of regions a line divides a polygon into -/
def count_regions (l : Line) (poly : Polygon) : ℕ := sorry

/-- Theorem: There exists a polygon and a point inside it such that 
    any line passing through this point divides the polygon into 
    exactly three smaller polygons -/
theorem munchausen_polygon_theorem : 
  ∃ (poly : Polygon) (p : Point), 
    is_inside p poly ∧ 
    ∀ (l : Line), l.point1 = p ∨ l.point2 = p → count_regions l poly = 3 := by
  sorry

end NUMINAMATH_CALUDE_munchausen_polygon_theorem_l2928_292814


namespace NUMINAMATH_CALUDE_log_xy_l2928_292857

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xy (x y : ℝ) (h1 : log (x * y^5) = 2) (h2 : log (x^3 * y) = 2) :
  log (x * y) = 6/7 := by sorry

end NUMINAMATH_CALUDE_log_xy_l2928_292857


namespace NUMINAMATH_CALUDE_angle_measure_when_complement_and_supplement_are_complementary_l2928_292815

theorem angle_measure_when_complement_and_supplement_are_complementary :
  ∀ x : ℝ,
  (90 - x) + (180 - x) = 90 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_when_complement_and_supplement_are_complementary_l2928_292815


namespace NUMINAMATH_CALUDE_dan_baseball_cards_l2928_292898

theorem dan_baseball_cards (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 97) 
  (h2 : remaining_cards = 82) : 
  initial_cards - remaining_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_dan_baseball_cards_l2928_292898


namespace NUMINAMATH_CALUDE_soda_production_in_8_hours_l2928_292845

/-- Represents the production rate of a soda machine -/
structure SodaMachine where
  cans_per_interval : ℕ
  interval_minutes : ℕ

/-- Calculates the number of cans produced in a given number of hours -/
def cans_produced (machine : SodaMachine) (hours : ℕ) : ℕ :=
  let intervals_per_hour : ℕ := 60 / machine.interval_minutes
  let total_intervals : ℕ := hours * intervals_per_hour
  machine.cans_per_interval * total_intervals

theorem soda_production_in_8_hours (machine : SodaMachine)
    (h1 : machine.cans_per_interval = 30)
    (h2 : machine.interval_minutes = 30) :
    cans_produced machine 8 = 480 := by
  sorry

end NUMINAMATH_CALUDE_soda_production_in_8_hours_l2928_292845


namespace NUMINAMATH_CALUDE_borrowed_amount_is_6800_l2928_292842

/-- Calculates the amount borrowed given interest rates and total interest paid -/
def calculate_borrowed_amount (total_interest : ℚ) (rate1 rate2 rate3 : ℚ) 
  (period1 period2 period3 : ℚ) : ℚ :=
  total_interest / (rate1 * period1 + rate2 * period2 + rate3 * period3)

/-- Proves that the amount borrowed is 6800, given the specified conditions -/
theorem borrowed_amount_is_6800 : 
  let total_interest : ℚ := 8160
  let rate1 : ℚ := 12 / 100
  let rate2 : ℚ := 9 / 100
  let rate3 : ℚ := 13 / 100
  let period1 : ℚ := 3
  let period2 : ℚ := 5
  let period3 : ℚ := 3
  calculate_borrowed_amount total_interest rate1 rate2 rate3 period1 period2 period3 = 6800 := by
  sorry

#eval calculate_borrowed_amount 8160 (12/100) (9/100) (13/100) 3 5 3

end NUMINAMATH_CALUDE_borrowed_amount_is_6800_l2928_292842


namespace NUMINAMATH_CALUDE_one_integral_root_l2928_292821

theorem one_integral_root :
  ∃! (x : ℤ), x - 9 / (x + 4 : ℚ) = 2 - 9 / (x + 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_one_integral_root_l2928_292821


namespace NUMINAMATH_CALUDE_landscape_breadth_l2928_292850

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 6 * length →
  playground_area = 4200 →
  length * width = 7 * playground_area →
  width = 420 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l2928_292850


namespace NUMINAMATH_CALUDE_order_powers_l2928_292878

theorem order_powers : 2^300 < 3^200 ∧ 3^200 < 10^100 := by
  sorry

end NUMINAMATH_CALUDE_order_powers_l2928_292878


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2928_292889

theorem quadratic_no_real_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m ≠ 0) → m > 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2928_292889


namespace NUMINAMATH_CALUDE_two_carp_heavier_than_three_bream_l2928_292838

/-- Represents the weight of a fish species -/
structure FishWeight where
  weight : ℝ
  weight_pos : weight > 0

/-- Given that 6 crucian carps are lighter than 5 perches and 6 crucian carps are heavier than 10 breams,
    prove that 2 crucian carp are heavier than 3 breams. -/
theorem two_carp_heavier_than_three_bream 
  (carp perch bream : FishWeight)
  (h1 : 6 * carp.weight < 5 * perch.weight)
  (h2 : 6 * carp.weight > 10 * bream.weight) :
  2 * carp.weight > 3 * bream.weight := by
sorry

end NUMINAMATH_CALUDE_two_carp_heavier_than_three_bream_l2928_292838


namespace NUMINAMATH_CALUDE_children_share_distribution_l2928_292855

theorem children_share_distribution (total : ℝ) (share_ac : ℝ) 
  (h1 : total = 15800)
  (h2 : share_ac = 7022.222222222222) :
  total - share_ac = 8777.777777777778 := by
  sorry

end NUMINAMATH_CALUDE_children_share_distribution_l2928_292855


namespace NUMINAMATH_CALUDE_smallest_a_value_l2928_292836

def rectangle_vertices : List (ℝ × ℝ) := [(34, 0), (41, 0), (34, 9), (41, 9)]

def line_equation (a : ℝ) (x : ℝ) : ℝ := a * x

def divides_rectangle (a : ℝ) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = 2 * area2 ∧
  area1 + area2 = 63 ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    ((x1, y1) ∈ rectangle_vertices ∨ (x1 ∈ Set.Icc 34 41 ∧ y1 = line_equation a x1)) ∧
    ((x2, y2) ∈ rectangle_vertices ∨ (x2 ∈ Set.Icc 34 41 ∧ y2 = line_equation a x2)) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2))

theorem smallest_a_value :
  ∀ ε > 0, divides_rectangle (0.08 + ε) → divides_rectangle 0.08 ∧ ¬divides_rectangle (0.08 - ε) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2928_292836


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l2928_292852

/-- A triangle with semiperimeter, inradius, and circumradius -/
structure Triangle where
  semiperimeter : ℝ
  inradius : ℝ
  circumradius : ℝ
  semiperimeter_pos : 0 < semiperimeter
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- An inscribed triangle with semiperimeter -/
structure InscribedTriangle (T : Triangle) where
  semiperimeter : ℝ
  semiperimeter_pos : 0 < semiperimeter
  semiperimeter_le : semiperimeter ≤ T.semiperimeter

/-- The theorem stating the inequality for inscribed triangles -/
theorem inscribed_triangle_inequality (T : Triangle) (IT : InscribedTriangle T) :
  T.inradius / T.circumradius ≤ IT.semiperimeter / T.semiperimeter ∧ 
  IT.semiperimeter / T.semiperimeter ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l2928_292852


namespace NUMINAMATH_CALUDE_sample_size_is_fifteen_l2928_292803

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  total_employees : ℕ
  young_employees : ℕ
  young_in_sample : ℕ

/-- Calculates the sample size for a given stratified sampling scenario -/
def sample_size (s : StratifiedSampling) : ℕ :=
  s.total_employees / (s.young_employees / s.young_in_sample)

/-- Theorem stating that the sample size is 15 for the given scenario -/
theorem sample_size_is_fifteen :
  let s : StratifiedSampling := {
    total_employees := 75,
    young_employees := 35,
    young_in_sample := 7
  }
  sample_size s = 15 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_fifteen_l2928_292803


namespace NUMINAMATH_CALUDE_tens_digit_of_power_five_l2928_292831

theorem tens_digit_of_power_five : ∃ (n : ℕ), 5^(5^5) ≡ 25 [MOD 100] ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_power_five_l2928_292831


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l2928_292849

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Michael has -/
def michael_apples : ℕ := 5

/-- Theorem: Adam has 3 more apples than the combined total of Jackie's and Michael's apples -/
theorem adam_has_more_apples : adam_apples - (jackie_apples + michael_apples) = 3 := by
  sorry


end NUMINAMATH_CALUDE_adam_has_more_apples_l2928_292849


namespace NUMINAMATH_CALUDE_smallest_n_congruence_fourteen_satisfies_congruence_fourteen_is_smallest_l2928_292829

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 5 * n ≡ 850 [ZMOD 26]) → n ≥ 14 :=
by sorry

theorem fourteen_satisfies_congruence : 
  5 * 14 ≡ 850 [ZMOD 26] :=
by sorry

theorem fourteen_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 14 → ¬(5 * m ≡ 850 [ZMOD 26]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_fourteen_satisfies_congruence_fourteen_is_smallest_l2928_292829


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_15_l2928_292832

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 5
  sum_2_5 : a 2 + a 5 = 12
  nth_term : ∃ n, a n = 29

/-- The theorem stating that n = 15 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_15 (seq : ArithmeticSequence) : 
  ∃ n, seq.a n = 29 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_15_l2928_292832


namespace NUMINAMATH_CALUDE_systematic_sampling_methods_systematic_sampling_characterization_l2928_292804

/-- Represents a sampling method -/
inductive SamplingMethod
| Method1
| Method2
| Method3
| Method4

/-- Predicate to determine if a sampling method is systematic -/
def is_systematic (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.Method1 => true
  | SamplingMethod.Method2 => true
  | SamplingMethod.Method3 => false
  | SamplingMethod.Method4 => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic SamplingMethod.Method1) ∧
  (is_systematic SamplingMethod.Method2) ∧
  (¬is_systematic SamplingMethod.Method3) ∧
  (is_systematic SamplingMethod.Method4) :=
by sorry

/-- Characterization of systematic sampling -/
theorem systematic_sampling_characterization (method : SamplingMethod) :
  is_systematic method ↔ 
    (∃ (rule : Prop), 
      (rule ↔ method = SamplingMethod.Method1 ∨ 
               method = SamplingMethod.Method2 ∨ 
               method = SamplingMethod.Method4) ∧
      (rule → ∃ (interval : Nat), interval > 0)) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_methods_systematic_sampling_characterization_l2928_292804


namespace NUMINAMATH_CALUDE_min_y_coordinate_polar_graph_l2928_292828

/-- The minimum y-coordinate of a point on the graph of r = cos(2θ) is -√6/3 -/
theorem min_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ y_min : ℝ, y_min = -Real.sqrt 6 / 3 ∧ ∀ θ : ℝ, y θ ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_coordinate_polar_graph_l2928_292828


namespace NUMINAMATH_CALUDE_reflection_line_l2928_292863

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by y = k (horizontal line) -/
structure HorizontalLine where
  k : ℝ

/-- Reflection of a point about a horizontal line -/
def reflect (p : Point) (l : HorizontalLine) : Point :=
  ⟨p.x, 2 * l.k - p.y⟩

theorem reflection_line (p q r p' q' r' : Point) (l : HorizontalLine) :
  p = Point.mk (-3) 1 ∧
  q = Point.mk 5 (-2) ∧
  r = Point.mk 2 7 ∧
  p' = Point.mk (-3) (-9) ∧
  q' = Point.mk 5 (-8) ∧
  r' = Point.mk 2 (-3) ∧
  reflect p l = p' ∧
  reflect q l = q' ∧
  reflect r l = r' →
  l = HorizontalLine.mk (-4) := by
sorry

end NUMINAMATH_CALUDE_reflection_line_l2928_292863


namespace NUMINAMATH_CALUDE_second_order_arithmetic_sequence_property_l2928_292812

/-- Second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ x y z : ℝ, ∀ n : ℕ, a n = x * n^2 + y * n + z

/-- First-order difference sequence -/
def FirstOrderDifference (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a (n + 1) - a n

/-- Second-order difference sequence -/
def SecondOrderDifference (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = b (n + 1) - b n

theorem second_order_arithmetic_sequence_property
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (c : ℕ → ℝ)
  (h1 : SecondOrderArithmeticSequence a)
  (h2 : FirstOrderDifference a b)
  (h3 : SecondOrderDifference b c)
  (h4 : ∀ n : ℕ, c n = 20)
  (h5 : a 10 = 23)
  (h6 : a 20 = 23) :
  a 30 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_second_order_arithmetic_sequence_property_l2928_292812


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l2928_292862

/-- The set of points satisfying the original equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the first line -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the second line -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating the equivalence of the sets -/
theorem equation_equivalent_to_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l2928_292862


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2928_292876

/-- Given a quadratic equation x^2 = 5x - 1, prove that its coefficients are 1, -5, and 1 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 = 5*x - 1 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -5 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2928_292876
