import Mathlib

namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l960_96080

theorem binomial_square_coefficient (x : ℝ) : ∃ b : ℝ, ∃ t u : ℝ, 
  b * x^2 + 20 * x + 1 = (t * x + u)^2 ∧ b = 100 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l960_96080


namespace NUMINAMATH_CALUDE_three_card_sequence_count_l960_96039

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of suits in the deck -/
def num_suits : ℕ := 5

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 12

/-- The number of cards to pick -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards in sequence from the deck -/
def ways_to_pick : ℕ := deck_size * (deck_size - 1) * (deck_size - 2)

theorem three_card_sequence_count :
  deck_size = num_suits * cards_per_suit →
  ways_to_pick = 205320 := by
  sorry

end NUMINAMATH_CALUDE_three_card_sequence_count_l960_96039


namespace NUMINAMATH_CALUDE_range_of_a_l960_96000

theorem range_of_a (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) 
  (h3 : ∃! (s : Finset ℤ), s.card = n ∧ ∀ x ∈ s, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l960_96000


namespace NUMINAMATH_CALUDE_fraction_comparison_l960_96052

theorem fraction_comparison : -8/21 > -3/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l960_96052


namespace NUMINAMATH_CALUDE_sum_equals_five_l960_96027

def star (a b : ℕ) : ℕ := a^b + a*b

theorem sum_equals_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 15) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_five_l960_96027


namespace NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l960_96014

/-- Given complex numbers a, b, c, z and a real number k satisfying certain conditions,
    the maximum value of |z| is (k^3 + √(k^6 + 4k^3)) / 2. -/
theorem max_abs_z_quadratic_equation (a b c z d : ℂ) (k : ℝ) 
    (h1 : Complex.abs a = Complex.abs d)
    (h2 : Complex.abs d > 0)
    (h3 : b = k • d)
    (h4 : c = k^2 • d)
    (h5 : a * z^2 + b * z + c = 0) :
    Complex.abs z ≤ (k^3 + Real.sqrt (k^6 + 4 * k^3)) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l960_96014


namespace NUMINAMATH_CALUDE_original_number_proof_l960_96095

theorem original_number_proof (y : ℚ) : 1 + 1 / y = 8 / 3 → y = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l960_96095


namespace NUMINAMATH_CALUDE_complement_P_subset_Q_l960_96045

open Set Real

theorem complement_P_subset_Q : 
  let P : Set ℝ := {x | x < 1}
  let Q : Set ℝ := {x | x > -1}
  (compl P : Set ℝ) ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_complement_P_subset_Q_l960_96045


namespace NUMINAMATH_CALUDE_regression_line_equation_l960_96057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation y = mx + b -/
structure LinearEquation where
  m : ℝ
  b : ℝ

/-- Check if a point lies on a line given by a linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.m * p.x + eq.b

theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.m = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l960_96057


namespace NUMINAMATH_CALUDE_total_animals_theorem_l960_96060

/-- Calculates the total number of animals seen given initial counts and changes --/
def total_animals_seen (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 4 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 20
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that given the specific initial counts and changes, the total animals seen is 410 --/
theorem total_animals_theorem : total_animals_seen 50 90 = 410 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_theorem_l960_96060


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l960_96091

/-- The quadratic function f(x) = -(x-3)^2 - 4 -/
def f (x : ℝ) : ℝ := -(x - 3)^2 - 4

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship :
  let y₁ := f (-1/2)
  let y₂ := f 1
  let y₃ := f 4
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l960_96091


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l960_96025

/-- Two 3x3 matrices that are inverses of each other -/
def A (x y z w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, 2, y],
    ![3, 3, 4],
    ![z, 6, w]]

def B (j k l m : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-6, j, -12],
    ![k, -14, l],
    ![3, m, 5]]

/-- The theorem stating that the sum of all variables in the inverse matrices equals 52 -/
theorem inverse_matrices_sum (x y z w j k l m : ℝ) :
  (A x y z w) * (B j k l m) = 1 →
  x + y + z + w + j + k + l + m = 52 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l960_96025


namespace NUMINAMATH_CALUDE_zero_in_set_implies_m_equals_two_l960_96075

theorem zero_in_set_implies_m_equals_two (m : ℝ) :
  0 ∈ ({m, m^2 - 2*m} : Set ℝ) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_set_implies_m_equals_two_l960_96075


namespace NUMINAMATH_CALUDE_tan_and_expression_values_l960_96061

theorem tan_and_expression_values (α : Real) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_tan : Real.tan (π / 4 + α) = 2) :
  Real.tan α = 1 / 3 ∧ 
  (Real.sqrt 2 * Real.sin (2 * α + π / 4) * Real.cos α - Real.sin α) / Real.cos (2 * α) = (2 / 5) * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_expression_values_l960_96061


namespace NUMINAMATH_CALUDE_paul_age_in_12_years_l960_96068

/-- Represents the ages of people in the problem -/
structure Ages where
  brian : ℝ
  christian : ℝ
  margaret : ℝ
  paul : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.christian = 3.5 * ages.brian ∧
  ages.brian + 12 = 45 ∧
  ages.margaret = 2 * ages.brian ∧
  ages.christian = ages.margaret + 15 ∧
  ages.paul = (ages.margaret + ages.christian) / 2

/-- The theorem to be proved -/
theorem paul_age_in_12_years (ages : Ages) :
  problem_conditions ages → ages.paul + 12 = 102.75 := by
  sorry

end NUMINAMATH_CALUDE_paul_age_in_12_years_l960_96068


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l960_96006

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l960_96006


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l960_96081

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℚ, (2 / (x + 3) - (x - 3) / (2*x + 6) = 1) ∧ x = 1/3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℚ, (2*x - 1 > 3*(x - 1) ∧ (5 - x)/2 < x + 4) ↔ (-1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l960_96081


namespace NUMINAMATH_CALUDE_angle_measure_l960_96069

theorem angle_measure (A B : ℝ) (h1 : A + B = 180) (h2 : A = 7 * B) : A = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l960_96069


namespace NUMINAMATH_CALUDE_sum_f_negative_l960_96070

/-- The function f(x) = -x^3 - x -/
def f (x : ℝ) : ℝ := -x^3 - x

/-- Theorem: For a, b, c ∈ ℝ satisfying a + b > 0, b + c > 0, and c + a > 0,
    it follows that f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l960_96070


namespace NUMINAMATH_CALUDE_one_integer_is_seventeen_l960_96018

theorem one_integer_is_seventeen (a b c d : ℕ+) 
  (eq1 : (b.val + c.val + d.val) / 3 + 2 * a.val = 54)
  (eq2 : (a.val + c.val + d.val) / 3 + 2 * b.val = 50)
  (eq3 : (a.val + b.val + d.val) / 3 + 2 * c.val = 42)
  (eq4 : (a.val + b.val + c.val) / 3 + 2 * d.val = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 := by
sorry

end NUMINAMATH_CALUDE_one_integer_is_seventeen_l960_96018


namespace NUMINAMATH_CALUDE_ab9_equals_459_implies_a_equals_4_l960_96090

/-- Represents a three-digit number with 9 as the last digit -/
structure ThreeDigitNumber9 where
  hundreds : Nat
  tens : Nat
  inv_hundreds : hundreds < 10
  inv_tens : tens < 10

/-- Converts a ThreeDigitNumber9 to its numerical value -/
def ThreeDigitNumber9.toNat (n : ThreeDigitNumber9) : Nat :=
  100 * n.hundreds + 10 * n.tens + 9

theorem ab9_equals_459_implies_a_equals_4 (ab9 : ThreeDigitNumber9) 
  (h : ab9.toNat = 459) : ab9.hundreds = 4 := by
  sorry

end NUMINAMATH_CALUDE_ab9_equals_459_implies_a_equals_4_l960_96090


namespace NUMINAMATH_CALUDE_milk_container_problem_l960_96094

theorem milk_container_problem (A : ℝ) 
  (hB : ℝ) (hC : ℝ) 
  (hB_initial : hB = 0.375 * A) 
  (hC_initial : hC = A - hB) 
  (h_equal_after_transfer : hB + 150 = hC - 150) : A = 1200 :=
by sorry

end NUMINAMATH_CALUDE_milk_container_problem_l960_96094


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l960_96037

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l960_96037


namespace NUMINAMATH_CALUDE_monday_average_is_7_l960_96077

/-- The average number of birds Kendra saw at each site on Monday -/
def monday_average : ℝ := sorry

/-- The number of sites visited on Monday -/
def monday_sites : ℕ := 5

/-- The number of sites visited on Tuesday -/
def tuesday_sites : ℕ := 5

/-- The number of sites visited on Wednesday -/
def wednesday_sites : ℕ := 10

/-- The average number of birds seen at each site on Tuesday -/
def tuesday_average : ℝ := 5

/-- The average number of birds seen at each site on Wednesday -/
def wednesday_average : ℝ := 8

/-- The overall average number of birds seen at each site across all three days -/
def overall_average : ℝ := 7

theorem monday_average_is_7 :
  monday_average = 7 :=
by sorry

end NUMINAMATH_CALUDE_monday_average_is_7_l960_96077


namespace NUMINAMATH_CALUDE_total_legs_in_group_l960_96048

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of humans in the group -/
def num_humans : ℕ := 2

/-- The number of dogs in the group -/
def num_dogs : ℕ := 2

/-- Theorem stating that the total number of legs in the group is 12 -/
theorem total_legs_in_group : 
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_group_l960_96048


namespace NUMINAMATH_CALUDE_snake_head_fraction_l960_96063

theorem snake_head_fraction (total_length body_length : ℝ) 
  (h1 : total_length = 10)
  (h2 : body_length = 9)
  (h3 : body_length < total_length) :
  (total_length - body_length) / total_length = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_snake_head_fraction_l960_96063


namespace NUMINAMATH_CALUDE_sculpture_base_height_l960_96032

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  12 * feet + inches

/-- Calculates the total height of a sculpture and its base -/
theorem sculpture_base_height 
  (sculpture_feet : ℕ) 
  (sculpture_inches : ℕ) 
  (base_inches : ℕ) : 
  feet_inches_to_inches sculpture_feet sculpture_inches + base_inches = 38 :=
by
  sorry

#check sculpture_base_height 2 10 4

end NUMINAMATH_CALUDE_sculpture_base_height_l960_96032


namespace NUMINAMATH_CALUDE_second_reduction_percentage_l960_96051

theorem second_reduction_percentage (P : ℝ) (R : ℝ) (h1 : P > 0) :
  (1 - R / 100) * (0.75 * P) = 0.375 * P →
  R = 50 := by
sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_l960_96051


namespace NUMINAMATH_CALUDE_pattern_repeats_proof_l960_96044

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := 14

/-- The number of beads in one bracelet -/
def beads_per_bracelet : ℕ := 42

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of times the pattern repeats per necklace -/
def pattern_repeats_per_necklace : ℕ := 5

/-- Theorem stating that the pattern repeats 5 times per necklace -/
theorem pattern_repeats_proof : 
  beads_per_bracelet + 10 * pattern_repeats_per_necklace * beads_per_pattern = total_beads :=
by sorry

end NUMINAMATH_CALUDE_pattern_repeats_proof_l960_96044


namespace NUMINAMATH_CALUDE_chemistry_mean_marks_l960_96009

/-- Proves that the mean mark in the second section is 60 given the conditions of the problem -/
theorem chemistry_mean_marks (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₃ m₄ : ℚ) (overall_avg : ℚ) :
  n₁ = 60 →
  n₂ = 35 →
  n₃ = 45 →
  n₄ = 42 →
  m₁ = 50 →
  m₃ = 55 →
  m₄ = 45 →
  overall_avg = 52005494505494504/1000000000000000 →
  ∃ m₂ : ℚ, m₂ = 60 ∧ 
    overall_avg * (n₁ + n₂ + n₃ + n₄ : ℚ) = n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄ :=
by sorry


end NUMINAMATH_CALUDE_chemistry_mean_marks_l960_96009


namespace NUMINAMATH_CALUDE_apartment_complex_households_l960_96015

/-- Calculates the total number of households in an apartment complex. -/
def total_households (num_buildings : ℕ) (num_floors : ℕ) 
  (households_first_floor : ℕ) (households_other_floors : ℕ) : ℕ :=
  num_buildings * (households_first_floor + (num_floors - 1) * households_other_floors)

/-- Theorem stating that the total number of households in the given apartment complex is 68. -/
theorem apartment_complex_households : 
  total_households 4 6 2 3 = 68 := by
  sorry

#eval total_households 4 6 2 3

end NUMINAMATH_CALUDE_apartment_complex_households_l960_96015


namespace NUMINAMATH_CALUDE_sum_of_fractions_l960_96073

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l960_96073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l960_96093

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_5th : a 5 = 25)
  (h_8th : a 8 = 43) :
  a 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l960_96093


namespace NUMINAMATH_CALUDE_expected_checks_on_4x4_board_l960_96092

/-- Represents a 4x4 chessboard -/
def Board := Fin 4 × Fin 4

/-- Calculates the number of ways a knight can check a king on a 4x4 board -/
def knight_check_positions (board : Board) : ℕ :=
  match board with
  | (0, 0) | (0, 3) | (3, 0) | (3, 3) => 2  -- corners
  | (0, 1) | (0, 2) | (1, 0) | (1, 3) | (2, 0) | (2, 3) | (3, 1) | (3, 2) => 3  -- edges
  | _ => 4  -- central squares

/-- The total number of possible knight-king pairs -/
def total_pairs : ℕ := 3 * 3

/-- The total number of ways to place a knight and a king on distinct squares -/
def total_placements : ℕ := 16 * 15

/-- The expected number of checks for a single knight-king pair -/
def expected_checks_per_pair : ℚ := 1 / 5

theorem expected_checks_on_4x4_board :
  (total_pairs : ℚ) * expected_checks_per_pair = 9 / 5 := by sorry

#check expected_checks_on_4x4_board

end NUMINAMATH_CALUDE_expected_checks_on_4x4_board_l960_96092


namespace NUMINAMATH_CALUDE_light_reflection_l960_96096

/-- Given a ray of light traveling along y = -3x + b, reflecting off x + y = 0,
    and then traveling along y = -ax + 3, prove that a = 1/3 and b = -9
    satisfy the relationship between a and b. -/
theorem light_reflection (x y a b : ℝ) : 
  (y = -3 * x + b) →  -- Initial ray
  (x + y = 0) →       -- Reflection line
  (y = -a * x + 3) →  -- Final ray
  (a = 1/3 ∧ b = -9) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l960_96096


namespace NUMINAMATH_CALUDE_border_tile_difference_l960_96056

/-- Represents an octagonal figure made of tiles -/
structure OctagonalFigure where
  white_tiles : ℕ
  black_tiles : ℕ

/-- Creates a new figure by adding a border of black tiles -/
def add_border (figure : OctagonalFigure) : OctagonalFigure :=
  { white_tiles := figure.white_tiles,
    black_tiles := figure.black_tiles + 8 }

/-- The difference between black and white tiles in a figure -/
def tile_difference (figure : OctagonalFigure) : ℤ :=
  figure.black_tiles - figure.white_tiles

theorem border_tile_difference (original : OctagonalFigure) 
  (h1 : original.white_tiles = 16)
  (h2 : original.black_tiles = 9) :
  tile_difference (add_border original) = 1 := by
  sorry

end NUMINAMATH_CALUDE_border_tile_difference_l960_96056


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l960_96079

theorem sine_cosine_inequality (x y : Real) (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ Real.pi / 2) :
  (Real.sin (x / 2))^2 * Real.cos y ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l960_96079


namespace NUMINAMATH_CALUDE_boat_round_trip_time_specific_boat_round_trip_time_l960_96038

/-- Calculate the total time for a round trip by boat -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  total_time

/-- The total time taken for the specific round trip is approximately 947.6923 hours -/
theorem specific_boat_round_trip_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |boat_round_trip_time 22 4 10080 - 947.6923| < ε :=
sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_specific_boat_round_trip_time_l960_96038


namespace NUMINAMATH_CALUDE_no_positive_rational_solution_l960_96023

theorem no_positive_rational_solution (n : ℕ+) :
  ¬∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end NUMINAMATH_CALUDE_no_positive_rational_solution_l960_96023


namespace NUMINAMATH_CALUDE_power_relation_l960_96019

theorem power_relation (x m n : ℝ) (h1 : x^m = 6) (h2 : x^n = 9) : x^(2*m - n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l960_96019


namespace NUMINAMATH_CALUDE_james_beat_record_by_116_l960_96013

/-- Represents James's scoring statistics for the football season -/
structure JamesStats where
  touchdownsPerGame : ℕ
  gamesInSeason : ℕ
  twoPointConversions : ℕ
  fieldGoals : ℕ
  extraPointAttempts : ℕ

/-- Calculates the total points scored by James -/
def totalPoints (stats : JamesStats) : ℕ :=
  stats.touchdownsPerGame * 6 * stats.gamesInSeason +
  stats.twoPointConversions * 2 +
  stats.fieldGoals * 3 +
  stats.extraPointAttempts

/-- The old record for points scored in a season -/
def oldRecord : ℕ := 300

/-- Theorem stating that James beat the old record by 116 points -/
theorem james_beat_record_by_116 (stats : JamesStats)
  (h1 : stats.touchdownsPerGame = 4)
  (h2 : stats.gamesInSeason = 15)
  (h3 : stats.twoPointConversions = 6)
  (h4 : stats.fieldGoals = 8)
  (h5 : stats.extraPointAttempts = 20) :
  totalPoints stats - oldRecord = 116 := by
  sorry

#eval totalPoints { touchdownsPerGame := 4, gamesInSeason := 15, twoPointConversions := 6, fieldGoals := 8, extraPointAttempts := 20 } - oldRecord

end NUMINAMATH_CALUDE_james_beat_record_by_116_l960_96013


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l960_96062

/-- Given two quadratic functions f and g, prove that A + B = 0 under certain conditions -/
theorem quadratic_function_sum (A B : ℝ) (f g : ℝ → ℝ) : 
  A ≠ B →
  (∀ x, f x = A * x^2 + B) →
  (∀ x, g x = B * x^2 + A) →
  (∀ x, f (g x) - g (f x) = -A^2 + B^2) →
  A + B = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_sum_l960_96062


namespace NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l960_96097

/-- The distance from a vertex of a cube to a plane, given the heights of adjacent vertices -/
theorem cube_vertex_distance_to_plane (h₁ h₂ h₃ : ℝ) :
  h₁ = 18 →
  h₂ = 20 →
  h₃ = 22 →
  ∃ (a b c d : ℝ),
    a^2 + b^2 + c^2 = 1 ∧
    15 * a + d = h₁ ∧
    15 * b + d = h₂ ∧
    15 * c + d = h₃ ∧
    d = (57 - Real.sqrt 597) / 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l960_96097


namespace NUMINAMATH_CALUDE_average_annual_reduction_l960_96089

theorem average_annual_reduction (total_reduction : ℝ) (years : ℕ) (average_reduction : ℝ) : 
  total_reduction = 0.19 → years = 2 → (1 - average_reduction) ^ years = 1 - total_reduction → average_reduction = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_average_annual_reduction_l960_96089


namespace NUMINAMATH_CALUDE_max_value_theorem_l960_96074

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*(Real.sqrt 2) ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l960_96074


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l960_96085

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l960_96085


namespace NUMINAMATH_CALUDE_order_parts_count_l960_96071

-- Define the master's productivity per hour
def master_productivity : ℕ → Prop :=
  λ y => y > 5

-- Define the apprentice's productivity relative to the master
def apprentice_productivity (y : ℕ) : ℕ := y - 2

-- Define the total number of parts in the order
def total_parts (y : ℕ) : ℕ := 2 * y * (y - 2) / (y - 4)

-- Theorem statement
theorem order_parts_count :
  ∀ y : ℕ,
    master_productivity y →
    (∃ t : ℕ, t * y = total_parts y) →
    2 * (apprentice_productivity y) * (t - 1) = total_parts y →
    total_parts y = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_order_parts_count_l960_96071


namespace NUMINAMATH_CALUDE_not_fascinating_l960_96028

theorem not_fascinating (F : ℤ → ℤ) : 
  (∀ c : ℤ, ∃ x : ℤ, F x ≠ c) →
  (∀ x : ℤ, F x = F (414 - x)) →
  (∀ x : ℤ, F x = F (412 - x)) →
  (∀ x : ℤ, F x = F (451 - x)) →
  False :=
sorry

end NUMINAMATH_CALUDE_not_fascinating_l960_96028


namespace NUMINAMATH_CALUDE_kerosene_cost_in_cents_l960_96012

-- Define the cost of a pound of rice in dollars
def rice_cost : ℚ := 33/100

-- Define the relationship between eggs and rice
def dozen_eggs_cost (rc : ℚ) : ℚ := rc

-- Define the relationship between kerosene and eggs
def half_liter_kerosene_cost (ec : ℚ) : ℚ := (8/12) * ec

-- Define the conversion from dollars to cents
def dollars_to_cents (d : ℚ) : ℚ := 100 * d

-- State the theorem
theorem kerosene_cost_in_cents : 
  dollars_to_cents (2 * half_liter_kerosene_cost (dozen_eggs_cost rice_cost)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_cost_in_cents_l960_96012


namespace NUMINAMATH_CALUDE_distance_to_reflection_l960_96005

/-- The distance between a point (2, -4) and its reflection over the y-axis is 4. -/
theorem distance_to_reflection : Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_l960_96005


namespace NUMINAMATH_CALUDE_f_range_characterization_l960_96001

def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem f_range_characterization :
  (∀ x : ℝ, f x > 2 ↔ x < (1/2) ∨ x > (5/2)) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ |a| * f x) ↔ (1/2) ≤ x ∧ x ≤ (5/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_range_characterization_l960_96001


namespace NUMINAMATH_CALUDE_jinsu_work_rate_l960_96067

/-- Given that Jinsu completes a task in 4 hours, prove that the amount of work he can do in one hour is 1/4 of the task. -/
theorem jinsu_work_rate (total_time : ℝ) (total_work : ℝ) (h : total_time = 4) :
  total_work / total_time = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_jinsu_work_rate_l960_96067


namespace NUMINAMATH_CALUDE_abc_zero_necessary_not_sufficient_for_a_zero_l960_96017

theorem abc_zero_necessary_not_sufficient_for_a_zero (a b c : ℝ) :
  (∀ a b c, a = 0 → a * b * c = 0) ∧
  (∃ a b c, a * b * c = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_abc_zero_necessary_not_sufficient_for_a_zero_l960_96017


namespace NUMINAMATH_CALUDE_acute_angle_relationship_l960_96064

theorem acute_angle_relationship (α β : Real) : 
  0 < α ∧ α < π / 2 →
  0 < β ∧ β < π / 2 →
  2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β →
  α < β := by
sorry

end NUMINAMATH_CALUDE_acute_angle_relationship_l960_96064


namespace NUMINAMATH_CALUDE_irrational_sqrt_N_l960_96008

def N (n : ℕ) : ℚ :=
  (10^n - 1) / 9 * 10^(2*n) + 4 * (10^(2*n) - 1) / 9

theorem irrational_sqrt_N (n : ℕ) (h : n > 1) :
  Irrational (Real.sqrt (N n)) :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_N_l960_96008


namespace NUMINAMATH_CALUDE_item_prices_l960_96002

theorem item_prices (x y z : ℝ) 
  (eq1 : 3 * x + 5 * y + z = 32) 
  (eq2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 := by
  sorry

end NUMINAMATH_CALUDE_item_prices_l960_96002


namespace NUMINAMATH_CALUDE_percentage_of_indian_women_l960_96047

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℝ) (percent_indian_children : ℝ) (percent_not_indian : ℝ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 →
  percent_indian_children = 70 →
  percent_not_indian = 55.38461538461539 →
  ∃ (percent_indian_women : ℝ),
    percent_indian_women = 60 ∧
    (percent_indian_men / 100 * total_men + percent_indian_women / 100 * total_women + percent_indian_children / 100 * total_children) /
    (total_men + total_women + total_children : ℝ) = 1 - percent_not_indian / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_women_l960_96047


namespace NUMINAMATH_CALUDE_system_solution_existence_l960_96043

theorem system_solution_existence (b : ℝ) : 
  (∃ a : ℝ, ∃ x y : ℝ, x = |y - b| + 3/b ∧ x^2 + y^2 + 32 = a*(2*y - a) + 12*x) ↔ 
  (b < 0 ∨ b ≥ 3/8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l960_96043


namespace NUMINAMATH_CALUDE_two_red_cards_probability_l960_96053

/-- The probability of drawing two red cards in succession from a deck of 100 cards
    containing 50 red cards and 50 black cards, without replacement. -/
theorem two_red_cards_probability (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) 
    (h1 : total_cards = 100)
    (h2 : red_cards = 50)
    (h3 : black_cards = 50)
    (h4 : total_cards = red_cards + black_cards) :
    (red_cards : ℚ) / total_cards * ((red_cards - 1) : ℚ) / (total_cards - 1) = 49 / 198 := by
  sorry

end NUMINAMATH_CALUDE_two_red_cards_probability_l960_96053


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l960_96042

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2*q - 1 = 0) :
  ∃ (x y : ℝ), x = 1/2 ∧ y = -1/6 ∧ p*x + 3*y + q = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l960_96042


namespace NUMINAMATH_CALUDE_set_product_theorem_l960_96087

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | 0 ≤ y}

-- Define the operation ×
def setProduct (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Theorem statement
theorem set_product_theorem :
  setProduct A B = {x | -1 ≤ x ∧ x < 0 ∨ 1 < x} :=
by sorry

end NUMINAMATH_CALUDE_set_product_theorem_l960_96087


namespace NUMINAMATH_CALUDE_root_in_interval_l960_96083

def f (x : ℝ) := x^3 + x - 1

theorem root_in_interval :
  (f 0.5 < 0) → (f 0.75 > 0) →
  ∃ x₀ ∈ Set.Ioo 0.5 0.75, f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l960_96083


namespace NUMINAMATH_CALUDE_multiple_of_nine_between_12_and_30_l960_96030

theorem multiple_of_nine_between_12_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_between_12_and_30_l960_96030


namespace NUMINAMATH_CALUDE_b_range_l960_96086

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (x + 1) / x^2 else Real.log (x + 2)

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x - 4

-- State the theorem
theorem b_range (b : ℝ) :
  (∃ a : ℝ, f a + g b = 1) → b ∈ Set.Icc (-3/2) (7/2) :=
by sorry

end NUMINAMATH_CALUDE_b_range_l960_96086


namespace NUMINAMATH_CALUDE_count_decimals_near_three_elevenths_l960_96050

theorem count_decimals_near_three_elevenths :
  let lower_bound : ℚ := 2614 / 10000
  let upper_bound : ℚ := 2792 / 10000
  let count := (upper_bound * 10000).floor.toNat - (lower_bound * 10000).ceil.toNat + 1
  (∀ s : ℚ, lower_bound ≤ s → s ≤ upper_bound →
    (∃ w x y z : ℕ, w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      s = (w * 1000 + x * 100 + y * 10 + z) / 10000) →
    (∀ n d : ℕ, n ≤ 3 → 0 < d → |s - n / d| ≥ |s - 3 / 11|)) →
  count = 179 := by
sorry

end NUMINAMATH_CALUDE_count_decimals_near_three_elevenths_l960_96050


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_l960_96024

/-- The derivative of x^2 * cos(x) is 2x * cos(x) - x^2 * sin(x) -/
theorem derivative_x_squared_cos (x : ℝ) :
  deriv (λ x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_l960_96024


namespace NUMINAMATH_CALUDE_log_function_not_in_fourth_quadrant_l960_96035

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function y = log_a(x+b)
noncomputable def f (a b x : ℝ) : ℝ := log_base a (x + b)

-- Theorem statement
theorem log_function_not_in_fourth_quadrant (a b : ℝ) 
  (ha : a > 1) (hb : b < -1) :
  ∀ x y : ℝ, f a b x = y → ¬(x > 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_log_function_not_in_fourth_quadrant_l960_96035


namespace NUMINAMATH_CALUDE_harris_dog_vegetable_cost_l960_96022

/-- Represents the cost and quantity of a vegetable in a 1-pound bag -/
structure VegetableInfo where
  quantity : ℕ
  cost : ℚ

/-- Calculates the annual cost of vegetables for Harris's dog -/
def annual_vegetable_cost (carrot_info celery_info pepper_info : VegetableInfo) 
  (daily_carrot daily_celery daily_pepper : ℕ) : ℚ :=
  let daily_cost := 
    daily_carrot * (carrot_info.cost / carrot_info.quantity) +
    daily_celery * (celery_info.cost / celery_info.quantity) +
    daily_pepper * (pepper_info.cost / pepper_info.quantity)
  daily_cost * 365

/-- Theorem stating the annual cost of vegetables for Harris's dog -/
theorem harris_dog_vegetable_cost :
  let carrot_info : VegetableInfo := ⟨5, 2⟩
  let celery_info : VegetableInfo := ⟨10, 3/2⟩
  let pepper_info : VegetableInfo := ⟨3, 5/2⟩
  annual_vegetable_cost carrot_info celery_info pepper_info 1 2 1 = 11169/20 := by
  sorry

end NUMINAMATH_CALUDE_harris_dog_vegetable_cost_l960_96022


namespace NUMINAMATH_CALUDE_max_squirrel_attacks_l960_96054

theorem max_squirrel_attacks (N : ℕ+) (a b c : ℤ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a - c = N) : 
  (∃ k : ℕ, k ≤ N ∧ 
    (∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m) ∧
    (∃ a' b' c' : ℤ, a' = b' ∧ b' ≥ c' ∧ a' - c' ≤ N - k)) ∧
  (∀ k : ℕ, k > N → 
    ¬(∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m)) :=
by sorry

end NUMINAMATH_CALUDE_max_squirrel_attacks_l960_96054


namespace NUMINAMATH_CALUDE_chips_purchased_l960_96011

/-- Given that P packets of chips can be purchased for R dimes,
    and 1 dollar is worth 10 dimes, the number of packets that
    can be purchased for M dollars is 10MP/R. -/
theorem chips_purchased (P R M : ℚ) (h1 : P > 0) (h2 : R > 0) (h3 : M > 0) :
  (P / R) * (M * 10) = 10 * M * P / R :=
by sorry

end NUMINAMATH_CALUDE_chips_purchased_l960_96011


namespace NUMINAMATH_CALUDE_symmetric_sequence_theorem_l960_96076

/-- A symmetric sequence of 7 terms -/
def SymmetricSequence (b : Fin 7 → ℝ) : Prop :=
  ∀ k, k < 7 → b k = b (6 - k)

/-- The first 4 terms form an arithmetic sequence -/
def ArithmeticSequence (b : Fin 7 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ k, k < 3 → b (k + 1) - b k = d

/-- The theorem statement -/
theorem symmetric_sequence_theorem (b : Fin 7 → ℝ) 
  (h_symmetric : SymmetricSequence b)
  (h_arithmetic : ArithmeticSequence b)
  (h_b1 : b 0 = 2)
  (h_sum : b 1 + b 3 = 16) :
  b = ![2, 5, 8, 11, 8, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_theorem_l960_96076


namespace NUMINAMATH_CALUDE_trees_on_road_l960_96021

theorem trees_on_road (road_length : ℕ) (interval : ℕ) (trees : ℕ) : 
  road_length = 156 ∧ 
  interval = 6 ∧ 
  trees = road_length / interval + 1 →
  trees = 27 :=
by sorry

end NUMINAMATH_CALUDE_trees_on_road_l960_96021


namespace NUMINAMATH_CALUDE_pool_filling_time_l960_96058

theorem pool_filling_time (t1 t2 t_combined : ℝ) : 
  t1 = 8 → t_combined = 4.8 → 1/t1 + 1/t2 = 1/t_combined → t2 = 12 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l960_96058


namespace NUMINAMATH_CALUDE_initial_workers_l960_96026

theorem initial_workers (total : ℕ) (increase_rate : ℚ) (initial : ℕ) : 
  total = 1065 →
  increase_rate = 25 / 100 →
  (1 + increase_rate) * initial = total →
  initial = 852 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_l960_96026


namespace NUMINAMATH_CALUDE_equation_two_solutions_l960_96033

theorem equation_two_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, Real.sqrt (9 - x) = x * Real.sqrt (9 - x)) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l960_96033


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l960_96072

/-- Given that (m-1)x^(m^2+1) - x - 2 = 0 is a quadratic equation, prove that m = -1 -/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 1) * x^(m^2 + 1) - x - 2 = a * x^2 + b * x + c) → 
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l960_96072


namespace NUMINAMATH_CALUDE_expected_consecutive_reds_l960_96049

/-- A bag containing one red, one yellow, and one blue ball -/
inductive Ball : Type
| Red : Ball
| Yellow : Ball
| Blue : Ball

/-- The process of drawing balls with replacement -/
def DrawProcess : Type := ℕ → Ball

/-- The probability of drawing each color is equal -/
axiom equal_probability (b : Ball) : ℝ

/-- The sum of probabilities is 1 -/
axiom prob_sum : equal_probability Ball.Red + equal_probability Ball.Yellow + equal_probability Ball.Blue = 1

/-- ξ is the number of draws until two consecutive red balls are drawn -/
def ξ (process : DrawProcess) : ℕ := sorry

/-- The expected value of ξ -/
def expected_ξ : ℝ := sorry

/-- Theorem: The expected value of ξ is 12 -/
theorem expected_consecutive_reds : expected_ξ = 12 := by sorry

end NUMINAMATH_CALUDE_expected_consecutive_reds_l960_96049


namespace NUMINAMATH_CALUDE_total_street_lights_l960_96066

theorem total_street_lights (neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : 
  neighborhoods = 10 → roads_per_neighborhood = 4 → lights_per_side = 250 →
  neighborhoods * roads_per_neighborhood * lights_per_side * 2 = 20000 := by
sorry

end NUMINAMATH_CALUDE_total_street_lights_l960_96066


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l960_96020

theorem incorrect_number_calculation (n : ℕ) (incorrect_avg correct_avg correct_num : ℝ) (X : ℝ) :
  n = 10 →
  incorrect_avg = 18 →
  correct_avg = 22 →
  correct_num = 66 →
  n * incorrect_avg = (n - 1) * correct_avg + X →
  n * correct_avg = (n - 1) * correct_avg + correct_num →
  X = 26 := by
    sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l960_96020


namespace NUMINAMATH_CALUDE_adams_farm_animals_l960_96031

theorem adams_farm_animals (cows : ℕ) (sheep : ℕ) (pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 := by
sorry

end NUMINAMATH_CALUDE_adams_farm_animals_l960_96031


namespace NUMINAMATH_CALUDE_inequality_proof_l960_96082

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l960_96082


namespace NUMINAMATH_CALUDE_function_non_negative_iff_k_geq_neg_one_l960_96055

/-- The function f(x) = |x^2 - 1| + x^2 + kx is non-negative on (0, +∞) if and only if k ≥ -1 -/
theorem function_non_negative_iff_k_geq_neg_one (k : ℝ) :
  (∀ x > 0, |x^2 - 1| + x^2 + k*x ≥ 0) ↔ k ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_non_negative_iff_k_geq_neg_one_l960_96055


namespace NUMINAMATH_CALUDE_julys_husband_age_l960_96098

/-- Given information about Hannah and July's ages, and July's husband's age relative to July,
    prove that July's husband is 25 years old. -/
theorem julys_husband_age :
  ∀ (hannah_initial_age : ℕ) 
    (july_initial_age : ℕ) 
    (years_passed : ℕ) 
    (age_difference_husband : ℕ),
  hannah_initial_age = 6 →
  hannah_initial_age = 2 * july_initial_age →
  years_passed = 20 →
  age_difference_husband = 2 →
  july_initial_age + years_passed + age_difference_husband = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_julys_husband_age_l960_96098


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l960_96046

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 5 → (x - 5) * (x + 1) < 0) ∧
  (∃ x, (x - 5) * (x + 1) < 0 ∧ (x < -1 ∨ x > 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l960_96046


namespace NUMINAMATH_CALUDE_multiple_inequalities_l960_96036

theorem multiple_inequalities :
  (∃ a b : ℝ, a + b < 2 * Real.sqrt (a * b)) ∧
  (∃ a : ℝ, a + 1 / a ≤ 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → b / a + a / b ≥ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 2 / x + 1 / y ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_multiple_inequalities_l960_96036


namespace NUMINAMATH_CALUDE_f_leq_f_f_eq_abs_f_l960_96003

/-- The function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem f_leq_f'_iff_a_geq_three_halves (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) ↔ a ≥ 3/2 := by sorry

theorem f_eq_abs_f'_solutions (a : ℝ) :
  (∀ x : ℝ, f a x = |f' a x|) ↔
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) := by sorry

end NUMINAMATH_CALUDE_f_leq_f_f_eq_abs_f_l960_96003


namespace NUMINAMATH_CALUDE_circle_symmetry_l960_96099

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the symmetrical circle
def symmetrical_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define symmetry with respect to the origin
def symmetrical_wrt_origin (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g (-x) (-y)

-- Theorem statement
theorem circle_symmetry :
  symmetrical_wrt_origin original_circle symmetrical_circle :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l960_96099


namespace NUMINAMATH_CALUDE_logarithm_properties_l960_96007

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (a b x : ℝ) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) (hx : x > 0) :
  (log a x = (log b x) / (log b a)) ∧ (log a b = 1 / (log b a)) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_properties_l960_96007


namespace NUMINAMATH_CALUDE_function_identity_l960_96004

def NatPos := {n : ℕ // n > 0}

theorem function_identity (f : NatPos → NatPos) 
  (h : ∀ m n : NatPos, (m.val ^ 2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ n : NatPos, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l960_96004


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l960_96041

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l960_96041


namespace NUMINAMATH_CALUDE_school_play_boys_count_school_play_problem_l960_96078

/-- Given a school play with girls and boys, prove the number of boys. -/
theorem school_play_boys_count (girls : ℕ) (total_parents : ℕ) : ℕ :=
  let boys := (total_parents - 2 * girls) / 2
  by
    -- Proof goes here
    sorry

/-- The actual problem statement -/
theorem school_play_problem : school_play_boys_count 6 28 = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_school_play_boys_count_school_play_problem_l960_96078


namespace NUMINAMATH_CALUDE_equilateral_triangle_line_count_l960_96040

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  /-- The number of vertices in the triangle -/
  vertices : ℕ
  /-- The number of sides in the triangle -/
  sides : ℕ
  /-- Assertion that the triangle has 3 vertices -/
  vertex_count : vertices = 3
  /-- Assertion that the triangle has 3 sides -/
  side_count : sides = 3

/-- Represents a line in the equilateral triangle -/
structure TriangleLine where
  /-- The line is an altitude -/
  is_altitude : Bool
  /-- The line is a median -/
  is_median : Bool
  /-- The line is an angle bisector -/
  is_angle_bisector : Bool

/-- 
Theorem: In an equilateral triangle, where each vertex is connected to the opposite side 
by a line that simultaneously acts as an altitude, a median, and an angle bisector, 
the total number of distinct lines is equal to the number of vertices.
-/
theorem equilateral_triangle_line_count (t : EquilateralTriangle) : 
  ∃ (lines : Finset TriangleLine), 
    (∀ l ∈ lines, l.is_altitude ∧ l.is_median ∧ l.is_angle_bisector) ∧ 
    lines.card = t.vertices := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_line_count_l960_96040


namespace NUMINAMATH_CALUDE_divisibility_property_l960_96084

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l960_96084


namespace NUMINAMATH_CALUDE_average_score_is_94_l960_96065

/-- The average math test score of Clyde's four children -/
def average_score (june_score patty_score josh_score henry_score : ℕ) : ℚ :=
  (june_score + patty_score + josh_score + henry_score : ℚ) / 4

/-- Theorem stating that the average math test score of Clyde's four children is 94 -/
theorem average_score_is_94 :
  average_score 97 85 100 94 = 94 := by sorry

end NUMINAMATH_CALUDE_average_score_is_94_l960_96065


namespace NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l960_96016

theorem sqrt_x_plus_sqrt_x_equals_y (m : ℕ) :
  ∃ (x y : ℚ), (x + Real.sqrt x).sqrt = y ∧
    y = Real.sqrt (m * (m + 1)) ∧
    x = (2 * y^2 + 1 + Real.sqrt (4 * y^2 + 1)) / 2 :=
by sorry

theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ (S : Finset (ℚ × ℚ)), S.card = n ∧
    ∀ (x y : ℚ), (x, y) ∈ S → (x + Real.sqrt x).sqrt = y :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l960_96016


namespace NUMINAMATH_CALUDE_probability_two_of_each_color_l960_96034

theorem probability_two_of_each_color (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = black_balls + white_balls) (h2 : total_balls = 17) 
  (h3 : black_balls = 9) (h4 : white_balls = 8) (h5 : drawn_balls = 4) : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 2) / Nat.choose total_balls drawn_balls = 168 / 397 :=
sorry

end NUMINAMATH_CALUDE_probability_two_of_each_color_l960_96034


namespace NUMINAMATH_CALUDE_inequality_proof_l960_96010

theorem inequality_proof (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ x₅) (h5 : x₅ ≥ 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 25/2 * (x₄^2 + x₅^2) ∧
  ((x₁ + x₂ + x₃ + x₄ + x₅)^2 = 25/2 * (x₄^2 + x₅^2) ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l960_96010


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l960_96029

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l960_96029


namespace NUMINAMATH_CALUDE_complex_modulus_3_plus_2i_l960_96088

theorem complex_modulus_3_plus_2i : 
  Complex.abs (3 + 2 * Complex.I) = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_3_plus_2i_l960_96088


namespace NUMINAMATH_CALUDE_egg_difference_l960_96059

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between the eggs in perfect condition and those that are cracked is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - broken - cracked - cracked = 9 :=
by sorry

end NUMINAMATH_CALUDE_egg_difference_l960_96059
