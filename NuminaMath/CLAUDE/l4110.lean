import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_points_distance_l4110_411095

/-- Given 5 consecutive points on a straight line, prove that ae = 22 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 2 * (d - c)) →   -- bc = 2 cd
  (e - d = 8) →             -- de = 8
  (b - a = 5) →             -- ab = 5
  (c - a = 11) →            -- ac = 11
  (e - a = 22) :=           -- ae = 22
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l4110_411095


namespace NUMINAMATH_CALUDE_proposition_evaluation_l4110_411074

theorem proposition_evaluation : 
  let p : Prop := (2 + 4 = 7)
  let q : Prop := (∀ x : ℝ, x = 1 → x^2 ≠ 1)
  ¬(p ∧ q) ∧ (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l4110_411074


namespace NUMINAMATH_CALUDE_inequalities_comparison_l4110_411045

theorem inequalities_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  ((1/2 : ℝ)^a > (1/2 : ℝ)^b) ∧
  (1/a > 1/b) ∧
  (b^2 > a^2) ∧
  (¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l4110_411045


namespace NUMINAMATH_CALUDE_count_integer_solutions_l4110_411041

/-- The number of integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
def integerSolutionCount : ℕ := 6

/-- The quadratic equation in question -/
def hasIntegerSolution (a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 + a*x + 9*a = 0

/-- The theorem stating that there are exactly 6 integer values of a for which
    the equation x^2 + ax + 9a = 0 has integer solutions for x -/
theorem count_integer_solutions :
  (∃! (s : Finset ℤ), s.card = integerSolutionCount ∧ ∀ a : ℤ, a ∈ s ↔ hasIntegerSolution a) :=
sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l4110_411041


namespace NUMINAMATH_CALUDE_calculation_proof_l4110_411059

theorem calculation_proof :
  (Real.sqrt 8 - Real.sqrt 2 - Real.sqrt (1/3) * Real.sqrt 6 = 0) ∧
  (Real.sqrt 15 / Real.sqrt 3 + (Real.sqrt 5 - 1)^2 = 6 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l4110_411059


namespace NUMINAMATH_CALUDE_b_value_l4110_411037

theorem b_value (b : ℚ) (h : b + b/4 - 1 = 3/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l4110_411037


namespace NUMINAMATH_CALUDE_committee_combinations_l4110_411084

theorem committee_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_committee_combinations_l4110_411084


namespace NUMINAMATH_CALUDE_point_in_plane_region_l4110_411088

def plane_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

theorem point_in_plane_region :
  plane_region 0 1 ∧
  ¬ plane_region 5 0 ∧
  ¬ plane_region 0 7 ∧
  ¬ plane_region 2 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l4110_411088


namespace NUMINAMATH_CALUDE_area_code_digits_l4110_411047

/-- Represents the set of allowed digits -/
def allowed_digits : Finset ℕ := {2, 3, 4}

/-- Calculates the number of valid area codes for a given number of digits -/
def valid_codes (n : ℕ) : ℕ := 3^n - 1

/-- The actual number of valid codes as per the problem statement -/
def actual_valid_codes : ℕ := 26

/-- The theorem stating that the number of digits in each area code is 3 -/
theorem area_code_digits :
  ∃ (n : ℕ), n > 0 ∧ valid_codes n = actual_valid_codes ∧ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_area_code_digits_l4110_411047


namespace NUMINAMATH_CALUDE_sticker_difference_l4110_411056

/-- Given two people with the same initial number of stickers, if one person uses 15 stickers
    and the other buys 18 stickers, the difference in their final number of stickers is 33. -/
theorem sticker_difference (initial_stickers : ℕ) : 
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l4110_411056


namespace NUMINAMATH_CALUDE_lemonade_proportion_l4110_411001

/-- Given that 40 lemons make 50 gallons of lemonade, prove that 12 lemons make 15 gallons -/
theorem lemonade_proportion :
  let lemons_for_50 : ℚ := 40
  let gallons_50 : ℚ := 50
  let gallons_15 : ℚ := 15
  let lemons_for_15 : ℚ := 12
  (lemons_for_50 / gallons_50 = lemons_for_15 / gallons_15) := by sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l4110_411001


namespace NUMINAMATH_CALUDE_range_of_f_l4110_411092

def f (x : ℝ) := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 2 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l4110_411092


namespace NUMINAMATH_CALUDE_nancy_total_games_l4110_411043

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Proof that Nancy will attend 24 games in total -/
theorem nancy_total_games :
  total_games 9 8 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l4110_411043


namespace NUMINAMATH_CALUDE_substitution_result_l4110_411039

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l4110_411039


namespace NUMINAMATH_CALUDE_train_crossing_time_l4110_411004

/-- Calculates the time it takes for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) : 
  train_speed_kmph = 72 →
  man_crossing_time = 18 →
  platform_length = 280 →
  (platform_length + train_speed_kmph * man_crossing_time * (5/18)) / (train_speed_kmph * (5/18)) = 32 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l4110_411004


namespace NUMINAMATH_CALUDE_popton_bus_toes_count_l4110_411003

/-- Represents the three races on planet Popton -/
inductive Race
  | Hoopit
  | Neglart
  | Zentorian

/-- Returns the number of toes per hand for a given race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2
  | Race.Zentorian => 4

/-- Returns the number of hands for a given race -/
def handsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5
  | Race.Zentorian => 6

/-- Returns the number of students of a given race on the bus -/
def studentsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8
  | Race.Zentorian => 5

/-- Calculates the total number of toes for a given race on the bus -/
def totalToesForRace (r : Race) : ℕ :=
  toesPerHand r * handsCount r * studentsCount r

/-- Theorem: The total number of toes on the Popton school bus is 284 -/
theorem popton_bus_toes_count :
  (totalToesForRace Race.Hoopit) + (totalToesForRace Race.Neglart) + (totalToesForRace Race.Zentorian) = 284 := by
  sorry

end NUMINAMATH_CALUDE_popton_bus_toes_count_l4110_411003


namespace NUMINAMATH_CALUDE_impossible_filling_l4110_411080

/-- Represents a 7 × 3 table filled with 0s and 1s -/
def Table := Fin 7 → Fin 3 → Bool

/-- Checks if a 2 × 2 submatrix in the table has all the same values -/
def has_same_2x2_submatrix (t : Table) : Prop :=
  ∃ (i j : Fin 7) (k l : Fin 3), i < j ∧ k < l ∧
    t i k = t i l ∧ t i k = t j k ∧ t i k = t j l

/-- Theorem stating that any 7 × 3 table filled with 0s and 1s
    always has a 2 × 2 submatrix with all the same values -/
theorem impossible_filling :
  ∀ (t : Table), has_same_2x2_submatrix t :=
sorry

end NUMINAMATH_CALUDE_impossible_filling_l4110_411080


namespace NUMINAMATH_CALUDE_equal_remainders_theorem_l4110_411021

theorem equal_remainders_theorem (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) (h_pos : x > 0) :
  (∃ r : ℕ, x % p = r ∧ p^2 % x = r) →
  ((x = p ∧ p % x = 0) ∨ (x = p^2 ∧ p^2 % x = 0) ∨ (x = p + 1 ∧ p^2 % x = 1)) :=
sorry

end NUMINAMATH_CALUDE_equal_remainders_theorem_l4110_411021


namespace NUMINAMATH_CALUDE_not_all_even_numbers_representable_l4110_411058

theorem not_all_even_numbers_representable :
  ∃ k : ℕ, k > 1000 ∧ k % 2 = 0 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) - m * (m + 1) :=
by sorry

end NUMINAMATH_CALUDE_not_all_even_numbers_representable_l4110_411058


namespace NUMINAMATH_CALUDE_rectangle_triangle_length_l4110_411063

/-- Given a rectangle ABCD with side lengths and a triangle DEF inside it, 
    proves that EF has a specific length when certain conditions are met. -/
theorem rectangle_triangle_length (AB BC DE DF EF : ℝ) : 
  AB = 8 → 
  BC = 10 → 
  DE = DF → 
  (1/2 * DE * DF) = (1/3 * AB * BC) → 
  EF = (16 * Real.sqrt 15) / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_length_l4110_411063


namespace NUMINAMATH_CALUDE_ln_inequality_l4110_411048

theorem ln_inequality (x : ℝ) (h : x > 1) : 2 * Real.log x < x - 1 / x := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l4110_411048


namespace NUMINAMATH_CALUDE_slope_of_line_l4110_411075

theorem slope_of_line (x y : ℝ) :
  x + 2 * y - 4 = 0 → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l4110_411075


namespace NUMINAMATH_CALUDE_screen_width_calculation_l4110_411005

theorem screen_width_calculation (height width diagonal : ℝ) : 
  height / width = 3 / 4 →
  height^2 + width^2 = diagonal^2 →
  diagonal = 36 →
  width = 28.8 :=
by sorry

end NUMINAMATH_CALUDE_screen_width_calculation_l4110_411005


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l4110_411064

theorem reciprocal_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  1 / (b - c) > 1 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l4110_411064


namespace NUMINAMATH_CALUDE_max_garden_area_l4110_411042

/-- Given 420 feet of fencing to enclose a rectangular garden on three sides
    (with the fourth side against a wall), the maximum area that can be achieved
    is 22050 square feet. -/
theorem max_garden_area (fencing : ℝ) (h : fencing = 420) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * l + w = fencing ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * l' + w' = fencing →
  l * w ≥ l' * w' ∧ l * w = 22050 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l4110_411042


namespace NUMINAMATH_CALUDE_downstream_distance_l4110_411081

theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 3) : 
  boat_speed + stream_speed * travel_time = 84 := by
sorry

end NUMINAMATH_CALUDE_downstream_distance_l4110_411081


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l4110_411050

theorem polynomial_equality_implies_sum (m n : ℝ) : 
  (∀ x : ℝ, (x + 8) * (x - 1) = x^2 + m*x + n) → m + n = -1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l4110_411050


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l4110_411013

theorem triangle_angle_theorem (A B C : ℝ) : 
  A = 32 →
  B = 3 * A →
  C = 2 * A - 12 →
  A + B + C = 180 →
  C = 52 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l4110_411013


namespace NUMINAMATH_CALUDE_enclosed_area_circular_arcs_octagon_l4110_411030

/-- The area enclosed by a curve formed by circular arcs centered on a regular octagon -/
theorem enclosed_area_circular_arcs_octagon (n : ℕ) (arc_length : ℝ) (side_length : ℝ) : 
  n = 12 → 
  arc_length = 3 * π / 4 → 
  side_length = 3 → 
  ∃ (area : ℝ), area = 54 + 18 * Real.sqrt 2 + 81 * π / 64 - 54 * π / 64 - 18 * π * Real.sqrt 2 / 64 :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_circular_arcs_octagon_l4110_411030


namespace NUMINAMATH_CALUDE_unique_solution_l4110_411049

/-- The sequence x_n defined by x_n = n / (n + 2016) -/
def x (n : ℕ) : ℚ := n / (n + 2016)

/-- Theorem stating the unique solution for m and n -/
theorem unique_solution :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ x 2016 = x m * x n ∧ m = 6048 ∧ n = 4032 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4110_411049


namespace NUMINAMATH_CALUDE_line_mb_value_l4110_411025

/-- Given a line passing through points (0, -1) and (1, 1) with equation y = mx + b, prove that mb = -2 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -1)
  (1 : ℝ) = m * 1 + b → -- The line passes through (1, 1)
  m * b = -2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_value_l4110_411025


namespace NUMINAMATH_CALUDE_quiz_probabilities_l4110_411070

/-- Represents the total number of questions -/
def total_questions : ℕ := 5

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 3

/-- Represents the number of true or false questions -/
def true_false_questions : ℕ := 2

/-- The probability that A draws a multiple-choice question while B draws a true or false question -/
def prob_A_multiple_B_true_false : ℚ := 3/10

/-- The probability that at least one of A and B draws a multiple-choice question -/
def prob_at_least_one_multiple : ℚ := 9/10

theorem quiz_probabilities :
  (prob_A_multiple_B_true_false = multiple_choice_questions * true_false_questions / (total_questions * (total_questions - 1))) ∧
  (prob_at_least_one_multiple = 1 - (true_false_questions * (true_false_questions - 1) / (total_questions * (total_questions - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_quiz_probabilities_l4110_411070


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l4110_411038

/-- A geometric sequence with common ratio q where the first, third, and second terms form an arithmetic sequence has q = 1 or q = -1 -/
theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (a 3 - a 2 = a 2 - a 1) →    -- arithmetic sequence condition
  (q = 1 ∨ q = -1) := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l4110_411038


namespace NUMINAMATH_CALUDE_consecutive_prime_product_l4110_411098

-- Define the first four consecutive prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define the product of these primes
def product_of_primes : Nat := first_four_primes.prod

theorem consecutive_prime_product :
  (product_of_primes = 210) ∧
  (product_of_primes % 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_prime_product_l4110_411098


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l4110_411016

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : ℕ
  intersections : ℕ

/-- Predicate to check if a configuration is valid -/
def IsValidConfiguration (config : LineConfiguration) : Prop :=
  config.lines = 100 ∧ (config.intersections = 100 ∨ config.intersections = 99)

theorem intersection_points_theorem :
  ∃ (config1 config2 : LineConfiguration),
    IsValidConfiguration config1 ∧
    IsValidConfiguration config2 ∧
    config1.intersections = 100 ∧
    config2.intersections = 99 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l4110_411016


namespace NUMINAMATH_CALUDE_two_digit_number_ending_with_zero_l4110_411073

/-- A two-digit number -/
structure TwoDigitNumber where
  value : ℕ
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : ℕ :=
  (n.value % 10) * 10 + (n.value / 10)

/-- Check if a natural number is a perfect fourth power -/
def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

theorem two_digit_number_ending_with_zero (N : TwoDigitNumber) :
  (N.value - reverse_digits N > 0) →
  is_perfect_fourth_power (N.value - reverse_digits N) →
  N.value % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_ending_with_zero_l4110_411073


namespace NUMINAMATH_CALUDE_largest_five_digit_product_120_sum_18_l4110_411023

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d5 : Nat
  is_valid : d1 ≥ 1 ∧ d1 ≤ 9 ∧ 
             d2 ≥ 0 ∧ d2 ≤ 9 ∧ 
             d3 ≥ 0 ∧ d3 ≤ 9 ∧ 
             d4 ≥ 0 ∧ d4 ≤ 9 ∧ 
             d5 ≥ 0 ∧ d5 ≤ 9

/-- The value of a five-digit number -/
def value (n : FiveDigitNumber) : Nat :=
  10000 * n.d1 + 1000 * n.d2 + 100 * n.d3 + 10 * n.d4 + n.d5

/-- The product of the digits of a five-digit number -/
def digit_product (n : FiveDigitNumber) : Nat :=
  n.d1 * n.d2 * n.d3 * n.d4 * n.d5

/-- The sum of the digits of a five-digit number -/
def digit_sum (n : FiveDigitNumber) : Nat :=
  n.d1 + n.d2 + n.d3 + n.d4 + n.d5

/-- Theorem: The sum of digits of the largest five-digit number 
    whose digits' product is 120 is 18 -/
theorem largest_five_digit_product_120_sum_18 :
  ∃ (N : FiveDigitNumber), 
    (∀ (M : FiveDigitNumber), digit_product M = 120 → value M ≤ value N) ∧ 
    digit_product N = 120 ∧ 
    digit_sum N = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_product_120_sum_18_l4110_411023


namespace NUMINAMATH_CALUDE_fraction_equality_l4110_411051

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4110_411051


namespace NUMINAMATH_CALUDE_magnitude_comparison_l4110_411017

theorem magnitude_comparison (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1/2) 
  (A : ℝ) (hA : A = 1 - a^2)
  (B : ℝ) (hB : B = 1 + a^2)
  (C : ℝ) (hC : C = 1 / (1 - a))
  (D : ℝ) (hD : D = 1 / (1 + a)) :
  (1 - a > a^2) ∧ (D < A ∧ A < B ∧ B < C) := by
sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l4110_411017


namespace NUMINAMATH_CALUDE_parallelogram_reconstruction_l4110_411029

/-- Given a parallelogram ABCD with E as the midpoint of BC and F as the midpoint of CD,
    prove that the coordinates of C can be determined from the coordinates of A, E, and F. -/
theorem parallelogram_reconstruction (A E F : ℝ × ℝ) :
  let K : ℝ × ℝ := ((E.1 + F.1) / 2, (E.2 + F.2) / 2)
  let C : ℝ × ℝ := (A.1 / 2, A.2 / 2)
  (∃ (B D : ℝ × ℝ), 
    -- ABCD is a parallelogram
    (A.1 - B.1 = D.1 - C.1 ∧ A.2 - B.2 = D.2 - C.2) ∧
    (A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2) ∧
    -- E is the midpoint of BC
    (E.1 = (B.1 + C.1) / 2 ∧ E.2 = (B.2 + C.2) / 2) ∧
    -- F is the midpoint of CD
    (F.1 = (C.1 + D.1) / 2 ∧ F.2 = (C.2 + D.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_reconstruction_l4110_411029


namespace NUMINAMATH_CALUDE_room_entry_exit_ways_l4110_411065

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of times the person enters the room -/
def num_entries : ℕ := 1

/-- The number of times the person exits the room -/
def num_exits : ℕ := 1

/-- The total number of ways to enter and exit the room -/
def total_ways : ℕ := num_doors ^ (num_entries + num_exits)

theorem room_entry_exit_ways :
  total_ways = 16 := by sorry

end NUMINAMATH_CALUDE_room_entry_exit_ways_l4110_411065


namespace NUMINAMATH_CALUDE_water_pouring_proof_l4110_411019

/-- Calculates the fraction of water remaining after n rounds -/
def water_remaining (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 1/2
  | 2 => 1/3
  | k + 3 => water_remaining (k + 2) * (2 * (k + 3)) / (2 * (k + 3) + 1)

/-- The number of rounds needed to reach exactly 1/5 of the original water -/
def rounds_to_one_fifth : ℕ := 6

theorem water_pouring_proof :
  water_remaining rounds_to_one_fifth = 1/5 :=
sorry

end NUMINAMATH_CALUDE_water_pouring_proof_l4110_411019


namespace NUMINAMATH_CALUDE_oplus_equation_solution_l4110_411046

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 4 * a + 2 * b

-- Theorem statement
theorem oplus_equation_solution :
  ∃ y : ℝ, oplus 3 (oplus 4 y) = -14 ∧ y = -14.5 := by
sorry

end NUMINAMATH_CALUDE_oplus_equation_solution_l4110_411046


namespace NUMINAMATH_CALUDE_popsicle_stick_ratio_l4110_411053

theorem popsicle_stick_ratio : 
  ∀ (steve sid sam : ℕ),
  steve = 12 →
  sid = 2 * steve →
  sam + sid + steve = 108 →
  sam / sid = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_ratio_l4110_411053


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l4110_411007

theorem quadratic_root_arithmetic_sequence (p q r : ℝ) : 
  p ≥ q → q ≥ r → r ≥ 0 →  -- Conditions on p, q, r
  (∃ d : ℝ, q = p - d ∧ r = p - 2*d) →  -- Arithmetic sequence condition
  (∃! x : ℝ, p*x^2 + q*x + r = 0) →  -- Exactly one root condition
  (∃ x : ℝ, p*x^2 + q*x + r = 0 ∧ x = -2 + Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l4110_411007


namespace NUMINAMATH_CALUDE_vanessa_saves_three_weeks_l4110_411010

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let net_weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + net_weekly_savings - 1) / net_weekly_savings

/-- Proves that Vanessa needs 3 weeks to save for the dress -/
theorem vanessa_saves_three_weeks :
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_saves_three_weeks_l4110_411010


namespace NUMINAMATH_CALUDE_total_revenue_proof_l4110_411031

def planned_daily_sales : ℕ := 100

def sales_data : List ℤ := [7, -5, -3, 13, -6, 12, 5]

def selling_price : ℚ := 5.5

def shipping_cost : ℚ := 2

def net_income_per_kg : ℚ := selling_price - shipping_cost

def total_planned_sales : ℕ := planned_daily_sales * 7

def actual_sales : ℤ := total_planned_sales + (sales_data.sum)

theorem total_revenue_proof :
  (actual_sales : ℚ) * net_income_per_kg = 2530.5 := by sorry

end NUMINAMATH_CALUDE_total_revenue_proof_l4110_411031


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l4110_411068

/-- Parabola equation: y = x^2 - x - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - x - 2

/-- Point P has y-coordinate 10 -/
def point_P : Set ℝ := {x : ℝ | parabola x = 10}

/-- Point Q has y-coordinate 0 -/
def point_Q : Set ℝ := {x : ℝ | parabola x = 0}

/-- The horizontal distance between two x-coordinates -/
def horizontal_distance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ point_P ∧ q ∈ point_Q ∧
  ∀ (p' q' : ℝ), p' ∈ point_P → q' ∈ point_Q →
  horizontal_distance p q ≤ horizontal_distance p' q' ∧
  horizontal_distance p q = 2 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l4110_411068


namespace NUMINAMATH_CALUDE_compound_composition_l4110_411006

/-- Atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of phosphorus in g/mol -/
def atomic_weight_P : ℝ := 30.97

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 122

/-- The number of oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 4

theorem compound_composition :
  ∀ x : ℕ, 
    atomic_weight_Al + atomic_weight_P + x * atomic_weight_O = compound_weight 
    ↔ 
    x = num_oxygen_atoms :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l4110_411006


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sqrt2_sum_l4110_411018

theorem sqrt_sum_squares_ge_sqrt2_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sqrt2_sum_l4110_411018


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l4110_411035

theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1020)
  (h2 : loss_percentage = 15) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l4110_411035


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_3_and_5_l4110_411076

theorem largest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, n = 990 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, m < 1000 → m % 3 = 0 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_3_and_5_l4110_411076


namespace NUMINAMATH_CALUDE_point_on_x_axis_l4110_411097

def on_x_axis (p : ℝ × ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.2.2 = 0

theorem point_on_x_axis : on_x_axis (5, 0, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l4110_411097


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4110_411009

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  2 - x + 3*y + 8*x - 5*y - 6 = 7*x - 2*y - 4 := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) :
  15*a^2*b - 12*a*b^2 + 12 - 4*a^2*b - 18 + 8*a*b^2 = 11*a^2*b - 4*a*b^2 - 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4110_411009


namespace NUMINAMATH_CALUDE_notebook_pen_combinations_l4110_411022

theorem notebook_pen_combinations (notebooks : Finset α) (pens : Finset β) 
  (h1 : notebooks.card = 4) (h2 : pens.card = 5) :
  (notebooks.product pens).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_combinations_l4110_411022


namespace NUMINAMATH_CALUDE_weight_difference_l4110_411090

/-- Given that Antoinette and Rupert have a combined weight of 98 kg,
    and Antoinette weighs 63 kg, prove that Antoinette weighs 7 kg less
    than twice Rupert's weight. -/
theorem weight_difference (antoinette_weight rupert_weight : ℝ) : 
  antoinette_weight = 63 →
  antoinette_weight + rupert_weight = 98 →
  2 * rupert_weight - antoinette_weight = 7 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_l4110_411090


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_origin_l4110_411082

/-- Given points F₁ and F₂ on the x-axis, and a point P satisfying the hyperbola equation,
    prove that the distance from P to the origin is √6/2 when P's y-coordinate is 1/2. -/
theorem hyperbola_point_distance_to_origin :
  ∀ (P : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 2, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  P.2 = 1/2 →
  dist P F₂ - dist P F₁ = 2 →
  dist P (0, 0) = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_to_origin_l4110_411082


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l4110_411085

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 2*k + 1

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0 :=
sorry

-- Theorem 2: If one root is greater than 3, then k > 1
theorem root_greater_than_three_implies_k_greater_than_one (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧ x > 3) → k > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l4110_411085


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l4110_411072

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l4110_411072


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l4110_411054

theorem triangle_side_length_range (a : ℝ) : 
  (∃ (s₁ s₂ s₃ : ℝ), s₁ = 3*a - 1 ∧ s₂ = 4*a + 1 ∧ s₃ = 12 - a ∧ 
    s₁ + s₂ > s₃ ∧ s₁ + s₃ > s₂ ∧ s₂ + s₃ > s₁) ↔ 
  (3/2 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l4110_411054


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l4110_411036

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (final_people : ℕ) (final_avg_weight : ℝ) :
  initial_people = 6 ∧ 
  initial_avg_weight = 160 ∧ 
  final_people = 7 ∧ 
  final_avg_weight = 151 →
  (final_people * final_avg_weight) - (initial_people * initial_avg_weight) = 97 := by
sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l4110_411036


namespace NUMINAMATH_CALUDE_prime_sum_product_l4110_411094

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → p + q = 10 → p * q = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l4110_411094


namespace NUMINAMATH_CALUDE_algae_growth_l4110_411066

/-- Represents the number of cells in an algae colony after a given number of days -/
def algaeCells (initialCells : ℕ) (divisionPeriod : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (2 ^ (totalDays / divisionPeriod))

/-- Theorem stating that an algae colony starting with 5 cells, doubling every 3 days,
    will have 20 cells after 9 days -/
theorem algae_growth : algaeCells 5 3 9 = 20 := by
  sorry


end NUMINAMATH_CALUDE_algae_growth_l4110_411066


namespace NUMINAMATH_CALUDE_process_time_600_parts_l4110_411093

/-- Linear regression equation for processing time -/
def process_time (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem: The time required to process 600 parts is 6.5 hours -/
theorem process_time_600_parts : process_time 600 = 6.5 := by
  sorry

#check process_time_600_parts

end NUMINAMATH_CALUDE_process_time_600_parts_l4110_411093


namespace NUMINAMATH_CALUDE_claires_calculation_l4110_411071

theorem claires_calculation (a b c d f : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  (a + b - c + d - f = a + (b - (c * (d - f)))) →
  f = 21/5 := by sorry

end NUMINAMATH_CALUDE_claires_calculation_l4110_411071


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l4110_411062

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 60 * π / 180)  -- A = 60°
  (h2 : t.a = 3)             -- a = 3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)  -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)  -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l4110_411062


namespace NUMINAMATH_CALUDE_points_per_bag_l4110_411034

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) :
  total_bags = 17 →
  unrecycled_bags = 8 →
  total_points = 45 →
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_bag_l4110_411034


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l4110_411061

theorem number_exceeds_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l4110_411061


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l4110_411077

def f (x : ℝ) := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l4110_411077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l4110_411008

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_11 (seq : ArithmeticSequence) :
  sum_n seq 15 = 75 ∧ seq.a 3 + seq.a 4 + seq.a 5 = 12 → sum_n seq 11 = 99 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l4110_411008


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l4110_411040

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def IsReducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (3 * n + 4) > 1

/-- The fraction (n-17)/(3n+4) is non-zero for positive n -/
def IsNonZero (n : ℕ) : Prop :=
  n > 0 ∧ n ≠ 17

theorem least_reducible_fraction :
  IsReducible 22 ∧ IsNonZero 22 ∧ ∀ n < 22, ¬(IsReducible n ∧ IsNonZero n) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l4110_411040


namespace NUMINAMATH_CALUDE_pen_pencil_cost_l4110_411028

/-- Given a pen and pencil where the pen costs twice as much as the pencil and the pen costs $4,
    prove that the total cost of the pen and pencil is $6. -/
theorem pen_pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = 4 → pen_cost = 2 * pencil_cost → pen_cost + pencil_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_l4110_411028


namespace NUMINAMATH_CALUDE_piece_exits_at_A2_l4110_411052

/-- Represents the directions a piece can move on the grid -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the grid -/
structure GridState :=
  (currentCell : Cell)
  (arrows : Cell → Direction)

/-- Defines a single move on the grid -/
def move (state : GridState) : GridState :=
  sorry

/-- Checks if a cell is on the boundary of the grid -/
def isOnBoundary (cell : Cell) : Bool :=
  sorry

/-- Simulates the movement of the piece until it reaches the boundary -/
def simulateUntilExit (initialState : GridState) : Cell :=
  sorry

/-- The main theorem to prove -/
theorem piece_exits_at_A2 :
  let initialState : GridState := {
    currentCell := { row := 2, col := 1 },  -- C2 in 0-indexed
    arrows := sorry  -- Initial arrow configuration
  }
  let exitCell := simulateUntilExit initialState
  exitCell = { row := 0, col := 1 }  -- A2 in 0-indexed
  :=
sorry

end NUMINAMATH_CALUDE_piece_exits_at_A2_l4110_411052


namespace NUMINAMATH_CALUDE_correct_proposition_l4110_411087

-- Define the parallel relation
def parallel (x y : Type) : Prop := sorry

-- Define the intersection of two planes
def intersection (α β : Type) : Type := sorry

-- Define proposition p
def p : Prop :=
  ∀ (a α β : Type), parallel a β ∧ parallel a α → parallel a β

-- Define proposition q
def q : Prop :=
  ∀ (a α β b : Type), parallel a α ∧ parallel a β ∧ intersection α β = b → parallel a b

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l4110_411087


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l4110_411024

theorem company_kw_price_percentage (price_kw assets_a assets_b : ℝ) : 
  price_kw = 2 * assets_b →
  price_kw = 0.7878787878787878 * (assets_a + assets_b) →
  (price_kw - assets_a) / assets_a = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l4110_411024


namespace NUMINAMATH_CALUDE_largest_choir_size_l4110_411002

theorem largest_choir_size :
  ∃ (x r m : ℕ),
    (r * x + 3 = m) ∧
    ((r - 3) * (x + 2) = m) ∧
    (m < 150) ∧
    (∀ (x' r' m' : ℕ),
      (r' * x' + 3 = m') ∧
      ((r' - 3) * (x' + 2) = m') ∧
      (m' < 150) →
      m' ≤ m) ∧
    m = 759 :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l4110_411002


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4110_411060

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4110_411060


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l4110_411012

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l4110_411012


namespace NUMINAMATH_CALUDE_soccer_team_size_l4110_411057

/-- The number of players prepared for a soccer game -/
def players_prepared (starting_players : ℕ) (first_half_subs : ℕ) (second_half_subs : ℕ) (non_playing_players : ℕ) : ℕ :=
  starting_players + first_half_subs + non_playing_players

theorem soccer_team_size :
  let starting_players : ℕ := 11
  let first_half_subs : ℕ := 2
  let second_half_subs : ℕ := 2 * first_half_subs
  let non_playing_players : ℕ := 7
  players_prepared starting_players first_half_subs second_half_subs non_playing_players = 20 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_size_l4110_411057


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4110_411026

-- Define sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4110_411026


namespace NUMINAMATH_CALUDE_hexagon_area_l4110_411067

/-- Given a square with area 16 and a regular hexagon with perimeter 3/4 of the square's perimeter,
    the area of the hexagon is 32√3/27. -/
theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 → 
  4 * s = 18 * t → 
  (3 * t^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 27 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l4110_411067


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4110_411011

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ (1 ≤ a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4110_411011


namespace NUMINAMATH_CALUDE_investment_growth_l4110_411015

theorem investment_growth (x : ℝ) : 
  (1 + x / 100) * (1 - 30 / 100) = 1 + 11.99999999999999 / 100 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l4110_411015


namespace NUMINAMATH_CALUDE_f_neg_two_value_l4110_411020

-- Define f as a function from R to R
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (2 * x) + x^2

-- State the theorem
theorem f_neg_two_value (h1 : f 2 = 2) (h2 : ∀ x, g x = -g (-x)) : f (-2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l4110_411020


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l4110_411078

/-- Given three temperatures and a four-day average, calculate the fourth temperature --/
theorem fourth_day_temperature 
  (temp1 temp2 temp3 : ℤ) 
  (average : ℚ) 
  (h1 : temp1 = -36)
  (h2 : temp2 = -15)
  (h3 : temp3 = -10)
  (h4 : average = -12)
  : (4 : ℚ) * average - (temp1 + temp2 + temp3 : ℚ) = 13 := by
  sorry

#check fourth_day_temperature

end NUMINAMATH_CALUDE_fourth_day_temperature_l4110_411078


namespace NUMINAMATH_CALUDE_area_traced_by_rolling_triangle_l4110_411089

/-- The area traced out by rolling an equilateral triangle -/
theorem area_traced_by_rolling_triangle (side_length : ℝ) (h : side_length = 6) :
  let triangle_height : ℝ := side_length * Real.sqrt 3 / 2
  let arc_length : ℝ := π * side_length / 3
  let rectangle_area : ℝ := side_length * arc_length
  let quarter_circle_area : ℝ := π * side_length^2 / 4
  rectangle_area + quarter_circle_area = 21 * π := by
  sorry

#check area_traced_by_rolling_triangle

end NUMINAMATH_CALUDE_area_traced_by_rolling_triangle_l4110_411089


namespace NUMINAMATH_CALUDE_total_sides_is_75_l4110_411079

/-- Represents the number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "hexagon" => 6
  | "octagon" => 8
  | "circle" => 0
  | "pentagon" => 5
  | _ => 0

/-- Calculates the total number of sides for a given shape and quantity --/
def total_sides (shape : String) (quantity : ℕ) : ℕ :=
  (sides_of_shape shape) * quantity

/-- Represents the cookie cutter drawer --/
structure CookieCutterDrawer :=
  (top_layer : ℕ)
  (middle_layer_squares : ℕ)
  (middle_layer_hexagons : ℕ)
  (bottom_layer_octagons : ℕ)
  (bottom_layer_circles : ℕ)
  (bottom_layer_pentagons : ℕ)

/-- Calculates the total number of sides for all cookie cutters in the drawer --/
def total_sides_in_drawer (drawer : CookieCutterDrawer) : ℕ :=
  total_sides "triangle" drawer.top_layer +
  total_sides "square" drawer.middle_layer_squares +
  total_sides "hexagon" drawer.middle_layer_hexagons +
  total_sides "octagon" drawer.bottom_layer_octagons +
  total_sides "circle" drawer.bottom_layer_circles +
  total_sides "pentagon" drawer.bottom_layer_pentagons

/-- The cookie cutter drawer described in the problem --/
def emery_drawer : CookieCutterDrawer :=
  { top_layer := 6,
    middle_layer_squares := 4,
    middle_layer_hexagons := 2,
    bottom_layer_octagons := 3,
    bottom_layer_circles := 5,
    bottom_layer_pentagons := 1 }

theorem total_sides_is_75 :
  total_sides_in_drawer emery_drawer = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_is_75_l4110_411079


namespace NUMINAMATH_CALUDE_solution_equivalence_l4110_411014

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) := {p | |p.1| + |p.2| = p.1^2}

-- Define the set of points as described in the solution
def T : Set (ℝ × ℝ) := 
  {(0, 0)} ∪ 
  {p | p.1 ≥ 1 ∧ (p.2 = p.1^2 - p.1 ∨ p.2 = -(p.1^2 - p.1))} ∪
  {p | p.1 ≤ -1 ∧ (p.2 = p.1^2 + p.1 ∨ p.2 = -(p.1^2 + p.1))}

-- Theorem statement
theorem solution_equivalence : S = T := by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l4110_411014


namespace NUMINAMATH_CALUDE_midpoint_sum_zero_l4110_411091

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 10) and (-4, -14) is 0. -/
theorem midpoint_sum_zero : 
  let x1 : ℝ := 8
  let y1 : ℝ := 10
  let x2 : ℝ := -4
  let y2 : ℝ := -14
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_zero_l4110_411091


namespace NUMINAMATH_CALUDE_problem_statement_l4110_411069

def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x, f a x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2) →
  a = 2 ∧ 
  (∀ x, f a x + |x - 1| ≥ 1 → a ∈ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4110_411069


namespace NUMINAMATH_CALUDE_geom_seq_306th_term_l4110_411044

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem geom_seq_306th_term (a₁ a₂ : ℝ) (h1 : a₁ = 7) (h2 : a₂ = -7) :
  geometric_sequence a₁ (a₂ / a₁) 306 = -7 :=
by sorry

end NUMINAMATH_CALUDE_geom_seq_306th_term_l4110_411044


namespace NUMINAMATH_CALUDE_teachers_survey_l4110_411083

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 80)
  (h_heart_trouble : heart_trouble = 50)
  (h_both : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 100/3 := by
sorry

end NUMINAMATH_CALUDE_teachers_survey_l4110_411083


namespace NUMINAMATH_CALUDE_set_union_problem_l4110_411086

theorem set_union_problem (M N : Set ℕ) (a : ℕ) :
  M = {a, 0} ∧ N = {1, 2} ∧ M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l4110_411086


namespace NUMINAMATH_CALUDE_five_integer_solutions_l4110_411096

theorem five_integer_solutions (x : ℤ) : 
  (∃ (S : Finset ℤ), (∀ y ∈ S, 5*y^2 + 19*y + 16 ≤ 20) ∧ 
                     (∀ z : ℤ, 5*z^2 + 19*z + 16 ≤ 20 → z ∈ S) ∧
                     S.card = 5) := by
  sorry

end NUMINAMATH_CALUDE_five_integer_solutions_l4110_411096


namespace NUMINAMATH_CALUDE_sum_first_8_even_numbers_l4110_411055

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_8_even_numbers :
  (first_n_even_numbers 8).sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_8_even_numbers_l4110_411055


namespace NUMINAMATH_CALUDE_circle_area_l4110_411027

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area (θ : Real) (r : Real → Real) :
  (r = fun θ ↦ 3 * Real.cos θ - 4 * Real.sin θ) →
  (∀ θ, ∃ x y : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (∃ c : Real × Real, ∃ radius : Real, ∀ x y : Real,
    (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ ∃ θ : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (π * (5/2)^2 : Real) = 25*π/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_l4110_411027


namespace NUMINAMATH_CALUDE_min_value_implies_a_value_l4110_411032

/-- The function f(x) = x^2 + ax - 1 has a minimum value of -2 on the interval [0, 3] -/
def has_min_value_neg_two (a : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, x^2 + a*x - 1 ≥ x₀^2 + a*x₀ - 1 ∧ x₀^2 + a*x₀ - 1 = -2

/-- If f(x) = x^2 + ax - 1 has a minimum value of -2 on [0, 3], then a = -10/3 -/
theorem min_value_implies_a_value (a : ℝ) :
  has_min_value_neg_two a → a = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_value_l4110_411032


namespace NUMINAMATH_CALUDE_lions_scored_18_l4110_411000

-- Define the total score and winning margin
def total_score : ℕ := 52
def winning_margin : ℕ := 16

-- Define the Lions' score as a function of the total score and winning margin
def lions_score (total : ℕ) (margin : ℕ) : ℕ :=
  (total - margin) / 2

-- Theorem statement
theorem lions_scored_18 :
  lions_score total_score winning_margin = 18 := by
  sorry

end NUMINAMATH_CALUDE_lions_scored_18_l4110_411000


namespace NUMINAMATH_CALUDE_angle_through_point_neg_pi_fourth_l4110_411099

/-- If the terminal side of angle α passes through the point (1, -1), 
    then α = -π/4 + 2kπ for some k ∈ ℤ, and specifically α = -π/4 when k = 0. -/
theorem angle_through_point_neg_pi_fourth (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -1) →
  (∃ (k : ℤ), α = -π/4 + 2 * k * π) ∧ 
  (α = -π/4 ∨ α = -π/4 + 2 * π ∨ α = -π/4 - 2 * π) :=
sorry

end NUMINAMATH_CALUDE_angle_through_point_neg_pi_fourth_l4110_411099


namespace NUMINAMATH_CALUDE_system_solution_implies_a_equals_five_l4110_411033

theorem system_solution_implies_a_equals_five 
  (x y a : ℝ) 
  (eq1 : 2 * x - y = 1) 
  (eq2 : 3 * x + y = 2 * a - 1) 
  (eq3 : 2 * y - x = 4) : 
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_a_equals_five_l4110_411033
