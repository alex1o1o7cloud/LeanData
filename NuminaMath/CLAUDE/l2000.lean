import Mathlib

namespace NUMINAMATH_CALUDE_grocery_store_buyers_difference_l2000_200062

/-- Given information about buyers in a grocery store over three days, 
    prove the difference in buyers between today and yesterday --/
theorem grocery_store_buyers_difference 
  (buyers_day_before_yesterday : ℕ) 
  (buyers_yesterday : ℕ) 
  (buyers_today : ℕ) 
  (total_buyers : ℕ) 
  (h1 : buyers_day_before_yesterday = 50)
  (h2 : buyers_yesterday = buyers_day_before_yesterday / 2)
  (h3 : total_buyers = buyers_day_before_yesterday + buyers_yesterday + buyers_today)
  (h4 : total_buyers = 140) :
  buyers_today - buyers_yesterday = 40 := by
sorry


end NUMINAMATH_CALUDE_grocery_store_buyers_difference_l2000_200062


namespace NUMINAMATH_CALUDE_rectangle_length_l2000_200055

theorem rectangle_length (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 4 * width →
  area = length * width →
  area = 100 →
  length = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l2000_200055


namespace NUMINAMATH_CALUDE_max_value_fourth_root_sum_l2000_200030

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (abcd : ℝ) ^ (1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ) ^ (1/4) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fourth_root_sum_l2000_200030


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2000_200027

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2000_200027


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2000_200000

theorem inequality_solution_set (x : ℝ) : 
  x^2 - 2*|x| - 15 > 0 ↔ x < -5 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2000_200000


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_199_l2000_200058

theorem inverse_of_3_mod_199 : ∃ x : ℕ, x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_199_l2000_200058


namespace NUMINAMATH_CALUDE_courtyard_width_l2000_200001

/-- The width of a courtyard given its length, brick dimensions, and total number of bricks --/
theorem courtyard_width (length : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℝ) :
  length = 28 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  total_bricks = 13787.878787878788 →
  ∃ width : ℝ, abs (width - 13.012) < 0.001 ∧ 
    length * width * 100 * 100 = total_bricks * brick_length * brick_width * 100 * 100 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l2000_200001


namespace NUMINAMATH_CALUDE_cos_product_special_angles_l2000_200046

theorem cos_product_special_angles : 
  Real.cos ((2 * π) / 5) * Real.cos ((6 * π) / 5) = -(1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_product_special_angles_l2000_200046


namespace NUMINAMATH_CALUDE_doubleBracket_two_l2000_200085

-- Define the double bracket notation
def doubleBracket (x : ℝ) : ℝ := x^2 + 2*x + 4

-- State the theorem
theorem doubleBracket_two : doubleBracket 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_doubleBracket_two_l2000_200085


namespace NUMINAMATH_CALUDE_multiples_of_five_relation_l2000_200003

theorem multiples_of_five_relation (a b c : ℤ) : 
  (∃ k l m : ℤ, a = 5 * k ∧ b = 5 * l ∧ c = 5 * m) →  -- a, b, c are multiples of 5
  a < b →                                            -- a < b
  b < c →                                            -- b < c
  c = a + 10 →                                       -- c = a + 10
  (a - b) * (a - c) / (b - c) = -10 := by             -- Prove that the expression equals -10
sorry


end NUMINAMATH_CALUDE_multiples_of_five_relation_l2000_200003


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2000_200077

theorem cubic_polynomial_root (d e : ℚ) :
  (3 - Real.sqrt 5 : ℂ) ^ 3 + d * (3 - Real.sqrt 5 : ℂ) + e = 0 →
  (-6 : ℂ) ^ 3 + d * (-6 : ℂ) + e = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2000_200077


namespace NUMINAMATH_CALUDE_angle_conversion_l2000_200015

def angle : Real := 54.12

theorem angle_conversion (ε : Real) (h : ε > 0) :
  ∃ (d : ℕ) (m : ℕ) (s : ℕ),
    d = 54 ∧ m = 7 ∧ s = 12 ∧ 
    abs (angle - (d : Real) - (m : Real) / 60 - (s : Real) / 3600) < ε :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l2000_200015


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2000_200068

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_division_simplification :
  (2 + i) / i = 1 - 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2000_200068


namespace NUMINAMATH_CALUDE_quadratic_root_n_value_l2000_200032

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

-- Define the root form
def root_form (x m n p : ℝ) : Prop := 
  (x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧ 
  m > 0 ∧ n > 0 ∧ p > 0 ∧ Int.gcd ⌊m⌋ (Int.gcd ⌊n⌋ ⌊p⌋) = 1

-- Theorem statement
theorem quadratic_root_n_value : 
  ∃ (x m p : ℝ), quadratic_equation x ∧ root_form x m 31 p := by sorry

end NUMINAMATH_CALUDE_quadratic_root_n_value_l2000_200032


namespace NUMINAMATH_CALUDE_bus_journey_l2000_200098

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.5)
  (h5 : ∀ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time → x = 160) :
  ∃ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time ∧ x = 160 := by
sorry

end NUMINAMATH_CALUDE_bus_journey_l2000_200098


namespace NUMINAMATH_CALUDE_function_derivative_inequality_l2000_200029

theorem function_derivative_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x < 1) :
  ∀ m : ℝ, f (1 - m) - f m > 1 - 2*m → m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_inequality_l2000_200029


namespace NUMINAMATH_CALUDE_banana_count_l2000_200023

theorem banana_count (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                     (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) :
  bunches_of_eight = 6 →
  bananas_per_bunch_eight = 8 →
  bunches_of_seven = 5 →
  bananas_per_bunch_seven = 7 →
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven = 83 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l2000_200023


namespace NUMINAMATH_CALUDE_f_values_l2000_200002

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

-- Theorem stating that f(2) = 14 and f(-2) = 2
theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l2000_200002


namespace NUMINAMATH_CALUDE_midpoint_after_translation_l2000_200018

-- Define the points B and G
def B : ℝ × ℝ := (2, 3)
def G : ℝ × ℝ := (6, 3)

-- Define the translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 7, p.2 - 3)

-- Theorem statement
theorem midpoint_after_translation :
  let B' := translate B
  let G' := translate G
  (B'.1 + G'.1) / 2 = -3 ∧ (B'.2 + G'.2) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l2000_200018


namespace NUMINAMATH_CALUDE_multiple_of_reciprocal_l2000_200078

theorem multiple_of_reciprocal (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 3) (h3 : x + 17 = k * (1 / x)) : k = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_reciprocal_l2000_200078


namespace NUMINAMATH_CALUDE_consecutive_ones_count_is_3719_l2000_200011

/-- Fibonacci-like sequence for numbers without consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => F (n + 1) + F n

/-- The number of 12-digit integers with digits 1 or 2 and two consecutive 1's -/
def consecutive_ones_count : ℕ := 2^12 - F 11

theorem consecutive_ones_count_is_3719 : consecutive_ones_count = 3719 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_ones_count_is_3719_l2000_200011


namespace NUMINAMATH_CALUDE_greatest_x_value_l2000_200080

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ (2.134 : ℝ) * (10 : ℝ) ^ (5 : ℝ) < 220000 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2000_200080


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2000_200087

/-- The perimeter of a hexagon with side lengths in arithmetic sequence -/
theorem hexagon_perimeter (a b c d e f : ℕ) (h1 : b = a + 2) (h2 : c = b + 2) (h3 : d = c + 2) 
  (h4 : e = d + 2) (h5 : f = e + 2) (h6 : a = 10) : a + b + c + d + e + f = 90 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2000_200087


namespace NUMINAMATH_CALUDE_adjacent_semicircles_perimeter_l2000_200059

/-- The perimeter of a shape formed by two adjacent semicircles with radius 1 --/
theorem adjacent_semicircles_perimeter :
  ∀ (r : ℝ), r = 1 →
  ∃ (perimeter : ℝ), perimeter = 3 * r :=
by sorry

end NUMINAMATH_CALUDE_adjacent_semicircles_perimeter_l2000_200059


namespace NUMINAMATH_CALUDE_product_of_terms_l2000_200051

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_terms_l2000_200051


namespace NUMINAMATH_CALUDE_little_krish_sweet_expense_l2000_200019

theorem little_krish_sweet_expense (initial_amount : ℚ) (friend_gift : ℚ) (amount_left : ℚ) :
  initial_amount = 200.50 →
  friend_gift = 25.20 →
  amount_left = 114.85 →
  initial_amount - 2 * friend_gift - amount_left = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_little_krish_sweet_expense_l2000_200019


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2000_200075

theorem arithmetic_evaluation : 8 + 15 / 3 * 2 - 5 + 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2000_200075


namespace NUMINAMATH_CALUDE_balloon_permutations_l2000_200073

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "balloon" -/
def total_letters : ℕ := 7

/-- The frequency of each letter in "balloon" -/
def letter_frequency : List ℕ := [1, 1, 2, 2, 1]

theorem balloon_permutations :
  balloon_arrangements = Nat.factorial total_letters / (List.prod letter_frequency) := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l2000_200073


namespace NUMINAMATH_CALUDE_equation_solution_l2000_200076

theorem equation_solution : ∃ n : ℚ, (6 / n) - (6 - 3) / 6 = 1 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2000_200076


namespace NUMINAMATH_CALUDE_first_day_over_100_l2000_200057

def paperclips (day : ℕ) : ℕ :=
  if day = 0 then 5
  else if day = 1 then 7
  else 7 + 3 * (day - 1)

def dayOfWeek (day : ℕ) : String :=
  match day % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_over_100 :
  (∀ d : ℕ, d < 33 → paperclips d ≤ 100) ∧
  paperclips 33 > 100 ∧
  dayOfWeek 33 = "Friday" := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_100_l2000_200057


namespace NUMINAMATH_CALUDE_anya_original_position_l2000_200063

def Friend := Fin 5

structure Seating :=
  (positions : Friend → Fin 5)
  (bijective : Function.Bijective positions)

def sum_positions (s : Seating) : Nat :=
  (List.range 5).sum

-- Define the movements
def move_right (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def move_left (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def swap (s : Seating) (f1 f2 : Friend) : Seating := sorry
def move_to_end (s : Seating) (f : Friend) : Seating := sorry

theorem anya_original_position 
  (initial : Seating) 
  (anya varya galya diana ella : Friend) 
  (h_distinct : anya ≠ varya ∧ anya ≠ galya ∧ anya ≠ diana ∧ anya ≠ ella ∧ 
                varya ≠ galya ∧ varya ≠ diana ∧ varya ≠ ella ∧ 
                galya ≠ diana ∧ galya ≠ ella ∧ 
                diana ≠ ella) 
  (final : Seating) 
  (h_movements : final = move_to_end (swap (move_left (move_right initial varya 3) galya 1) diana ella) anya) 
  (h_sum_equal : sum_positions initial = sum_positions final) :
  initial.positions anya = 3 := by sorry

end NUMINAMATH_CALUDE_anya_original_position_l2000_200063


namespace NUMINAMATH_CALUDE_complex_modulus_l2000_200021

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2000_200021


namespace NUMINAMATH_CALUDE_surfers_ratio_l2000_200071

def surfers_problem (first_day : ℕ) (second_day_increase : ℕ) (average : ℕ) : Prop :=
  let second_day := first_day + second_day_increase
  let total := average * 3
  let third_day := total - first_day - second_day
  (third_day : ℚ) / first_day = 2 / 5

theorem surfers_ratio : 
  surfers_problem 1500 600 1400 := by sorry

end NUMINAMATH_CALUDE_surfers_ratio_l2000_200071


namespace NUMINAMATH_CALUDE_quoted_poetry_mismatch_l2000_200097

-- Define a type for poetry quotes
inductive PoetryQuote
| A
| B
| C
| D

-- Define a function to check if a quote matches its context
def matchesContext (quote : PoetryQuote) : Prop :=
  match quote with
  | PoetryQuote.A => True
  | PoetryQuote.B => True
  | PoetryQuote.C => True
  | PoetryQuote.D => False

-- Theorem statement
theorem quoted_poetry_mismatch :
  ∃ (q : PoetryQuote), ¬(matchesContext q) ∧ ∀ (p : PoetryQuote), p ≠ q → matchesContext p :=
by
  sorry

end NUMINAMATH_CALUDE_quoted_poetry_mismatch_l2000_200097


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2000_200043

theorem arithmetic_sequence_common_difference
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : ∃ (d : ℝ), ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  ∃ (d : ℝ), d = 0 ∧ ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2000_200043


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l2000_200054

theorem cube_root_of_eight (x y : ℝ) : x^(3*y) = 8 ∧ x = 8 → y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l2000_200054


namespace NUMINAMATH_CALUDE_actual_speed_calculation_l2000_200074

/-- 
Given a person who travels a certain distance at an unknown speed, 
this theorem proves that if walking at 10 km/hr would allow them 
to travel 20 km more in the same time, and the actual distance 
traveled is 20 km, then their actual speed is 5 km/hr.
-/
theorem actual_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 20) 
  (h2 : faster_speed = 10) 
  (h3 : additional_distance = 20) 
  (h4 : actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) :
  actual_speed = 5 :=
sorry

#check actual_speed_calculation

end NUMINAMATH_CALUDE_actual_speed_calculation_l2000_200074


namespace NUMINAMATH_CALUDE_segment_length_product_l2000_200042

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3 * a - 5)^2 + (2 * a - 5)^2 = 125 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -749/676) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l2000_200042


namespace NUMINAMATH_CALUDE_polynomial_roots_b_value_l2000_200047

theorem polynomial_roots_b_value (A B C D : ℤ) : 
  (∀ z : ℤ, z > 0 → (z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 0) → 
   (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℤ, 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧
      z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 
      (z - x₁) * (z - x₂) * (z - x₃) * (z - x₄) * (z - x₅) * (z - x₆))) →
  B = -88 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_b_value_l2000_200047


namespace NUMINAMATH_CALUDE_board_operations_finite_and_invariant_l2000_200017

/-- Represents the state of the board with n natural numbers -/
def Board := List Nat

/-- Performs one operation on the board, replacing two numbers with their GCD and LCM -/
def performOperation (board : Board) (i j : Nat) : Board :=
  sorry

/-- Checks if the board is in its final state (all pairs are proper) -/
def isFinalState (board : Board) : Bool :=
  sorry

theorem board_operations_finite_and_invariant (initial_board : Board) :
  ∃ (final_board : Board),
    (∀ (sequence : List (Nat × Nat)), 
      isFinalState (sequence.foldl (λ b (i, j) => performOperation b i j) initial_board)) ∧
    (∀ (sequence1 sequence2 : List (Nat × Nat)),
      isFinalState (sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board) ∧
      isFinalState (sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) →
      sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board =
      sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) :=
by
  sorry

end NUMINAMATH_CALUDE_board_operations_finite_and_invariant_l2000_200017


namespace NUMINAMATH_CALUDE_problem_statement_l2000_200086

theorem problem_statement (a b : ℤ) : 
  ({1, a, b / a} : Set ℤ) = {0, a^2, a + b} → a^2017 + b^2017 = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2000_200086


namespace NUMINAMATH_CALUDE_original_number_proof_l2000_200060

theorem original_number_proof (y : ℝ) (h : 1 - 1/y = 1/5) : y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2000_200060


namespace NUMINAMATH_CALUDE_second_year_decrease_is_twenty_percent_l2000_200053

/-- Represents the population change over two years --/
structure PopulationChange where
  initial_population : ℕ
  first_year_increase : ℚ
  final_population : ℕ

/-- Calculates the percentage decrease in the second year --/
def second_year_decrease (pc : PopulationChange) : ℚ :=
  let first_year_population := pc.initial_population * (1 + pc.first_year_increase)
  1 - (pc.final_population : ℚ) / first_year_population

/-- Theorem: Given the specified population change, the second year decrease is 20% --/
theorem second_year_decrease_is_twenty_percent 
  (pc : PopulationChange) 
  (h1 : pc.initial_population = 10000)
  (h2 : pc.first_year_increase = 1/5)
  (h3 : pc.final_population = 9600) : 
  second_year_decrease pc = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_second_year_decrease_is_twenty_percent_l2000_200053


namespace NUMINAMATH_CALUDE_response_rate_percentage_l2000_200010

/-- Given that 300 responses are needed and the minimum number of questionnaires
    to be mailed is 375, prove that the response rate percentage is 80%. -/
theorem response_rate_percentage
  (responses_needed : ℕ)
  (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 300)
  (h2 : questionnaires_mailed = 375) :
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l2000_200010


namespace NUMINAMATH_CALUDE_carrots_removed_count_l2000_200009

-- Define the given constants
def total_carrots : ℕ := 30
def remaining_carrots : ℕ := 27
def total_weight : ℚ := 5.94
def avg_weight_remaining : ℚ := 0.2
def avg_weight_removed : ℚ := 0.18

-- Define the number of removed carrots
def removed_carrots : ℕ := total_carrots - remaining_carrots

-- Theorem statement
theorem carrots_removed_count :
  removed_carrots = 3 := by sorry

end NUMINAMATH_CALUDE_carrots_removed_count_l2000_200009


namespace NUMINAMATH_CALUDE_garage_wheels_l2000_200052

/-- The number of bikes that can be assembled -/
def bikes_assembled : ℕ := 7

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of bike wheels in the garage is 14 -/
theorem garage_wheels : bikes_assembled * wheels_per_bike = 14 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_l2000_200052


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2000_200014

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 12*x + 20) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 20) / (x^2 - 4*x + 8) < 2 ↔ 
  x > 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2000_200014


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l2000_200037

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 120 →
  ratio = 7 / 13 →
  ∃ (shorter_piece longer_piece : ℝ),
    shorter_piece + longer_piece = total_length ∧
    longer_piece = ratio * shorter_piece ∧
    shorter_piece = 78 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l2000_200037


namespace NUMINAMATH_CALUDE_inequality_solutions_l2000_200066

theorem inequality_solutions :
  (∀ x : ℝ, 2*x - 1 > x - 3 ↔ x > -2) ∧
  (∀ x : ℝ, x - 3*(x - 2) ≥ 4 ∧ (x - 1)/5 < (x + 1)/2 ↔ -7/3 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2000_200066


namespace NUMINAMATH_CALUDE_equality_of_powers_l2000_200007

theorem equality_of_powers (a b c d : ℕ) :
  a^a * b^(a + b) = c^c * d^(c + d) →
  Nat.gcd a b = 1 →
  Nat.gcd c d = 1 →
  a = c ∧ b = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_powers_l2000_200007


namespace NUMINAMATH_CALUDE_order_of_expressions_l2000_200031

theorem order_of_expressions : 
  let a : ℝ := (3/5)^(2/5)
  let b : ℝ := (2/5)^(3/5)
  let c : ℝ := (2/5)^(2/5)
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2000_200031


namespace NUMINAMATH_CALUDE_no_separable_representation_l2000_200094

theorem no_separable_representation : ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end NUMINAMATH_CALUDE_no_separable_representation_l2000_200094


namespace NUMINAMATH_CALUDE_unique_function_property_l2000_200026

theorem unique_function_property (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) = 2 * n) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l2000_200026


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l2000_200064

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h,k) -/
theorem parabola_vertex (a : ℝ) (h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The vertex of the parabola y = 1/3 * (x-7)^2 + 5 is (7,5) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ (1/3) * (x - 7)^2 + 5
  (7, 5) = (7, f 7) ∧ ∀ x, f x ≥ f 7 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l2000_200064


namespace NUMINAMATH_CALUDE_time_taken_BC_l2000_200034

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 40
def work_rate_B : ℚ := 1 / 60
def work_rate_C : ℚ := 1 / 80

-- Define the work done by A and B
def work_done_A : ℚ := 10 * work_rate_A
def work_done_B : ℚ := 5 * work_rate_B

-- Define the remaining work
def remaining_work : ℚ := 1 - (work_done_A + work_done_B)

-- Define the combined work rate of B and C
def combined_rate_BC : ℚ := work_rate_B + work_rate_C

-- Theorem stating the time taken by B and C to finish the remaining work
theorem time_taken_BC : (remaining_work / combined_rate_BC) = 160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_time_taken_BC_l2000_200034


namespace NUMINAMATH_CALUDE_excluded_age_is_nine_l2000_200079

/-- A 5-digit number with distinct, consecutive digits in increasing order -/
def ConsecutiveDigitNumber := { n : ℕ | 
  12345 ≤ n ∧ n ≤ 98765 ∧ 
  ∃ (a b c d e : ℕ), n = 10000*a + 1000*b + 100*c + 10*d + e ∧
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e }

/-- The set of ages of Mrs. Smith's children -/
def ChildrenAges := { n : ℕ | 5 ≤ n ∧ n ≤ 13 }

theorem excluded_age_is_nine :
  ∃ (n : ConsecutiveDigitNumber),
    ∀ (k : ℕ), k ∈ ChildrenAges → k ≠ 9 → n % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_excluded_age_is_nine_l2000_200079


namespace NUMINAMATH_CALUDE_remainder_of_3_to_20_mod_7_l2000_200095

theorem remainder_of_3_to_20_mod_7 : 3^20 % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_to_20_mod_7_l2000_200095


namespace NUMINAMATH_CALUDE_trains_crossing_time_l2000_200065

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) 
  (h1 : length_A = 200)
  (h2 : length_B = 250)
  (h3 : speed_A = 72)
  (h4 : speed_B = 18) : 
  (length_A + length_B) / ((speed_A + speed_B) * (1000 / 3600)) = 18 := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l2000_200065


namespace NUMINAMATH_CALUDE_tram_speed_l2000_200039

/-- The speed of a tram given observation times and tunnel length -/
theorem tram_speed (t_pass : ℝ) (t_tunnel : ℝ) (tunnel_length : ℝ) 
  (h_pass : t_pass = 3)
  (h_tunnel : t_tunnel = 13)
  (h_length : tunnel_length = 100)
  (h_positive : t_pass > 0 ∧ t_tunnel > 0 ∧ tunnel_length > 0) :
  tunnel_length / (t_tunnel - t_pass) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tram_speed_l2000_200039


namespace NUMINAMATH_CALUDE_trajectory_of_moving_point_l2000_200044

theorem trajectory_of_moving_point (x y : ℝ) :
  let segment_length : ℝ := 3
  let point_A : ℝ × ℝ := (3 * x, 0)
  let point_B : ℝ × ℝ := (0, 3 * y / 2)
  let point_C : ℝ × ℝ := (x, y)
  (point_A.1 - point_C.1)^2 + (point_A.2 - point_C.2)^2 = 4 * ((point_C.1 - point_B.1)^2 + (point_C.2 - point_B.2)^2) →
  x^2 + y^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_point_l2000_200044


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2000_200088

/-- The sum of the areas of six rectangles with specified dimensions -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2000_200088


namespace NUMINAMATH_CALUDE_coat_discount_proof_l2000_200056

theorem coat_discount_proof :
  ∃ (p q : ℕ), 
    p < 10 ∧ q < 10 ∧
    21250 * (1 - p / 100) * (1 - q / 100) = 19176 ∧
    ((p = 4 ∧ q = 6) ∨ (p = 6 ∧ q = 4)) := by
  sorry

end NUMINAMATH_CALUDE_coat_discount_proof_l2000_200056


namespace NUMINAMATH_CALUDE_board_game_cost_l2000_200025

theorem board_game_cost (jump_rope_cost playground_ball_cost dalton_savings uncle_gift additional_needed : ℕ) 
  (h1 : jump_rope_cost = 7)
  (h2 : playground_ball_cost = 4)
  (h3 : dalton_savings = 6)
  (h4 : uncle_gift = 13)
  (h5 : additional_needed = 4) :
  jump_rope_cost + playground_ball_cost + (dalton_savings + uncle_gift + additional_needed) - (dalton_savings + uncle_gift) = 12 := by
  sorry

end NUMINAMATH_CALUDE_board_game_cost_l2000_200025


namespace NUMINAMATH_CALUDE_least_four_digit_special_number_l2000_200070

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop := sorry

theorem least_four_digit_special_number :
  ∀ n : ℕ,
  1000 ≤ n →
  n < 10000 →
  has_different_digits n →
  divisible_by_digits n →
  n % 5 = 0 →
  1425 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_special_number_l2000_200070


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l2000_200061

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 84) : 
  ∃ (p : ℝ), p = 68 ∧ 
  p = dist.percent_less_than_mean_plus_std_dev - (100 - dist.percent_less_than_mean_plus_std_dev) := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l2000_200061


namespace NUMINAMATH_CALUDE_last_painted_cell_l2000_200005

/-- Represents a cell in a rectangular grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- Represents a rectangular grid --/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Defines the spiral painting process --/
def spiralPaint (rect : Rectangle) : Cell :=
  sorry

/-- Theorem statement for the last painted cell in a 333 × 444 rectangle --/
theorem last_painted_cell :
  let rect : Rectangle := { rows := 333, cols := 444 }
  spiralPaint rect = { row := 167, col := 278 } :=
sorry

end NUMINAMATH_CALUDE_last_painted_cell_l2000_200005


namespace NUMINAMATH_CALUDE_only_sperm_has_one_set_l2000_200090

-- Define the types of cells
inductive CellType
  | Zygote
  | SomaticCell
  | Spermatogonium
  | Sperm

-- Define a function to represent the number of chromosome sets in a cell
def chromosomeSets : CellType → ℕ
  | CellType.Zygote => 2
  | CellType.SomaticCell => 2
  | CellType.Spermatogonium => 2
  | CellType.Sperm => 1

-- Define that spermatogonium is a type of somatic cell
axiom spermatogonium_is_somatic : chromosomeSets CellType.Spermatogonium = chromosomeSets CellType.SomaticCell

-- Define that sperm is formed through meiosis (implicitly resulting in one set of chromosomes)
axiom sperm_meiosis : chromosomeSets CellType.Sperm = 1

-- Theorem: Only sperm contains one set of chromosomes
theorem only_sperm_has_one_set :
  ∀ (cell : CellType), chromosomeSets cell = 1 ↔ cell = CellType.Sperm :=
by sorry


end NUMINAMATH_CALUDE_only_sperm_has_one_set_l2000_200090


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2000_200091

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2000_200091


namespace NUMINAMATH_CALUDE_jiAnWinningCases_l2000_200035

/-- Represents the possible moves in rock-paper-scissors -/
inductive Move
  | Rock
  | Paper
  | Scissors

/-- Determines if the first move wins against the second move -/
def wins (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Paper, Move.Rock => true
  | Move.Scissors, Move.Paper => true
  | _, _ => false

/-- Counts the number of winning cases for the first player -/
def countWinningCases : Nat :=
  List.length (List.filter
    (fun (m1, m2) => wins m1 m2)
    [(Move.Rock, Move.Paper), (Move.Rock, Move.Scissors), (Move.Rock, Move.Rock),
     (Move.Paper, Move.Rock), (Move.Paper, Move.Scissors), (Move.Paper, Move.Paper),
     (Move.Scissors, Move.Rock), (Move.Scissors, Move.Paper), (Move.Scissors, Move.Scissors)])

theorem jiAnWinningCases :
  countWinningCases = 3 := by sorry

end NUMINAMATH_CALUDE_jiAnWinningCases_l2000_200035


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l2000_200092

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

theorem intersection_complement_equals_set (h : Set ℕ) : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l2000_200092


namespace NUMINAMATH_CALUDE_probability_at_most_one_correct_l2000_200012

theorem probability_at_most_one_correct (pA pB : ℚ) : 
  pA = 3/5 → pB = 2/3 → 
  let p_at_most_one := 
    (1 - pA) * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * (1 - pA) * (1 - pA) * pB * (1 - pB)
  p_at_most_one = 32/225 := by
sorry

end NUMINAMATH_CALUDE_probability_at_most_one_correct_l2000_200012


namespace NUMINAMATH_CALUDE_number_ordering_l2000_200033

theorem number_ordering : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2000_200033


namespace NUMINAMATH_CALUDE_distance_between_locations_l2000_200099

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  carA : Car
  carB : Car
  meetingTime : ℝ
  additionalTime : ℝ
  finalDistanceA : ℝ
  finalDistanceB : ℝ

/-- The theorem stating the distance between locations A and B -/
theorem distance_between_locations (setup : ProblemSetup)
  (h1 : setup.meetingTime = 5)
  (h2 : setup.additionalTime = 3)
  (h3 : setup.finalDistanceA = 130)
  (h4 : setup.finalDistanceB = 160) :
  setup.carA.speed * setup.meetingTime + setup.carB.speed * setup.meetingTime = 290 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l2000_200099


namespace NUMINAMATH_CALUDE_scissors_in_drawer_final_scissors_count_l2000_200072

theorem scissors_in_drawer (initial : ℕ) (added : ℕ) (removed : ℕ) : ℕ :=
  initial + added - removed

theorem final_scissors_count : scissors_in_drawer 54 22 15 = 61 := by
  sorry

end NUMINAMATH_CALUDE_scissors_in_drawer_final_scissors_count_l2000_200072


namespace NUMINAMATH_CALUDE_number_of_possible_sums_l2000_200020

def bag_A : Finset ℕ := {1, 3, 5}
def bag_B : Finset ℕ := {2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : possible_sums.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sums_l2000_200020


namespace NUMINAMATH_CALUDE_sarah_DC_probability_l2000_200083

def probability_DC : ℚ := 2/5

theorem sarah_DC_probability :
  let p : ℚ → ℚ := λ x => 1/3 + 1/6 * x
  ∃! x : ℚ, x = p x ∧ x = probability_DC :=
by sorry

end NUMINAMATH_CALUDE_sarah_DC_probability_l2000_200083


namespace NUMINAMATH_CALUDE_water_moles_in_reaction_l2000_200040

-- Define the chemical reaction
structure ChemicalReaction where
  lithium_nitride : ℚ
  water : ℚ
  lithium_hydroxide : ℚ
  ammonia : ℚ

-- Define the balanced equation
def balanced_equation (r : ChemicalReaction) : Prop :=
  r.lithium_nitride = r.water / 3 ∧ 
  r.lithium_hydroxide = r.water ∧ 
  r.ammonia = r.water / 3

-- Theorem statement
theorem water_moles_in_reaction 
  (r : ChemicalReaction) 
  (h1 : r.lithium_nitride = 1) 
  (h2 : r.lithium_hydroxide = 3) 
  (h3 : balanced_equation r) : 
  r.water = 3 := by sorry

end NUMINAMATH_CALUDE_water_moles_in_reaction_l2000_200040


namespace NUMINAMATH_CALUDE_combined_distance_theorem_l2000_200050

def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

def train_A_speed : ℝ := 150
def train_A_time : ℝ := 8

def train_B_speed : ℝ := 180
def train_B_time : ℝ := 6

def train_C_speed : ℝ := 120
def train_C_time : ℝ := 10

theorem combined_distance_theorem :
  train_distance train_A_speed train_A_time +
  train_distance train_B_speed train_B_time +
  train_distance train_C_speed train_C_time = 3480 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_theorem_l2000_200050


namespace NUMINAMATH_CALUDE_passengers_remaining_approx_40_l2000_200036

/-- Calculates the number of passengers remaining after three stops -/
def passengers_after_stops (initial : ℕ) : ℚ :=
  let after_first := initial - (initial / 3)
  let after_second := after_first - (after_first / 4)
  let after_third := after_second - (after_second / 5)
  after_third

/-- Theorem: Given 100 initial passengers and three stops with specified fractions of passengers getting off, 
    the number of remaining passengers is approximately 40 -/
theorem passengers_remaining_approx_40 :
  ∃ ε > 0, ε < 1 ∧ |passengers_after_stops 100 - 40| < ε :=
sorry

end NUMINAMATH_CALUDE_passengers_remaining_approx_40_l2000_200036


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2000_200045

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x₀ : ℝ, x₀^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2000_200045


namespace NUMINAMATH_CALUDE_minimum_values_l2000_200082

theorem minimum_values :
  (∀ x > 1, (x + 4 / (x - 1) ≥ 5) ∧ (x + 4 / (x - 1) = 5 ↔ x = 3)) ∧
  (∀ x, 0 < x → x < 1 → (4 / x + 1 / (1 - x) ≥ 9) ∧ (4 / x + 1 / (1 - x) = 9 ↔ x = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_values_l2000_200082


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_equals_i_l2000_200024

theorem complex_exp_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_equals_i_l2000_200024


namespace NUMINAMATH_CALUDE_unique_k_solution_l2000_200081

theorem unique_k_solution : 
  ∃! (k : ℕ), k ≥ 1 ∧ (∃ (n m : ℤ), 9 * n^6 = 2^k + 5 * m^2 + 2) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l2000_200081


namespace NUMINAMATH_CALUDE_union_equals_A_implies_p_le_3_l2000_200049

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (p : ℝ) : Set ℝ := {x : ℝ | p + 1 < x ∧ x < 2*p - 1}

-- State the theorem
theorem union_equals_A_implies_p_le_3 (p : ℝ) :
  A ∪ B p = A → p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_implies_p_le_3_l2000_200049


namespace NUMINAMATH_CALUDE_old_workers_in_sample_l2000_200004

/-- Represents the composition of workers in a unit -/
structure WorkerComposition where
  total : ℕ
  young : ℕ
  old : ℕ
  middleAged : ℕ
  young_count : young ≤ total
  middleAged_relation : middleAged = 2 * old
  total_sum : total = young + old + middleAged

/-- Represents a stratified sample of workers -/
structure StratifiedSample where
  composition : WorkerComposition
  young_sample : ℕ
  young_sample_valid : young_sample ≤ composition.young

/-- Theorem stating the number of old workers in the stratified sample -/
theorem old_workers_in_sample (unit : WorkerComposition) (sample : StratifiedSample)
    (h_unit : unit.total = 430 ∧ unit.young = 160)
    (h_sample : sample.composition = unit ∧ sample.young_sample = 32) :
    (sample.young_sample : ℚ) / unit.young * unit.old = 18 := by
  sorry

end NUMINAMATH_CALUDE_old_workers_in_sample_l2000_200004


namespace NUMINAMATH_CALUDE_theme_park_triplets_l2000_200041

theorem theme_park_triplets (total_cost mother_charge child_charge_per_year : ℚ)
  (h_total_cost : total_cost = 15.25)
  (h_mother_charge : mother_charge = 6.95)
  (h_child_charge : child_charge_per_year = 0.55)
  : ∃ (triplet_age youngest_age : ℕ),
    triplet_age > youngest_age ∧
    youngest_age = 3 ∧
    total_cost = mother_charge + child_charge_per_year * (3 * triplet_age + youngest_age) :=
by sorry

end NUMINAMATH_CALUDE_theme_park_triplets_l2000_200041


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2000_200084

theorem fraction_equation_solution :
  ∃ (x : ℚ), (x + 7) / (x - 4) = (x - 3) / (x + 6) ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2000_200084


namespace NUMINAMATH_CALUDE_union_equals_B_intersection_equals_B_l2000_200089

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the set C
def C : Set ℝ := {a : ℝ | a ≤ -1 ∨ a = 1}

-- Theorem 1
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem 2
theorem intersection_equals_B : A ∩ B a = B a → a ∈ C := by sorry

end NUMINAMATH_CALUDE_union_equals_B_intersection_equals_B_l2000_200089


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l2000_200096

-- Define the power function
def f (m : ℕ+) (x : ℝ) : ℝ := x^(m.val - 2)

-- Theorem statement
theorem power_function_not_through_origin (m : ℕ+) :
  (∀ x ≠ 0, f m x ≠ 0) → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l2000_200096


namespace NUMINAMATH_CALUDE_derivative_at_one_l2000_200028

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- State the theorem
theorem derivative_at_one : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2000_200028


namespace NUMINAMATH_CALUDE_factorial_simplification_l2000_200093

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_simplification :
  (factorial 12) / ((factorial 10) + 3 * (factorial 9)) = 132 / 13 :=
by sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2000_200093


namespace NUMINAMATH_CALUDE_class_gender_ratio_l2000_200022

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls + boys = 28 →
  girls = boys + 4 →
  (girls : ℚ) / (boys : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l2000_200022


namespace NUMINAMATH_CALUDE_division_simplification_l2000_200008

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  -6 * a^3 * b / (3 * a * b) = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2000_200008


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l2000_200067

/-- Properties of an acute triangle ABC with specific conditions -/
theorem acute_triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  -- Sum of angles in a triangle is π
  A + B + C = π ∧
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Given condition for b
  b = 2 * Real.sqrt 6 ∧
  -- Sine rule
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  -- Angle bisector theorem
  (Real.sqrt 3 * a * c) / (a + c) = b * Real.sin (B / 2) / Real.sin B →
  -- Conclusions to prove
  π / 6 < C ∧ C < π / 2 ∧
  2 * Real.sqrt 2 < c ∧ c < 4 * Real.sqrt 2 ∧
  16 < a * c ∧ a * c ≤ 24 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l2000_200067


namespace NUMINAMATH_CALUDE_recipe_total_is_24_l2000_200016

/-- The total cups of ingredients required for Mary's cake recipe -/
def total_ingredients (sugar flour cocoa : ℕ) : ℕ :=
  sugar + flour + cocoa

/-- Theorem stating that the total ingredients for the recipe is 24 cups -/
theorem recipe_total_is_24 :
  total_ingredients 11 8 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_is_24_l2000_200016


namespace NUMINAMATH_CALUDE_well_diameter_l2000_200013

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (π : ℝ) :
  depth = 10 →
  volume = 31.41592653589793 →
  π = 3.141592653589793 →
  (2 * (volume / (π * depth)))^(1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_well_diameter_l2000_200013


namespace NUMINAMATH_CALUDE_even_factors_count_l2000_200038

/-- The number of even positive factors of 2^4 * 3^2 * 5 * 7 -/
def num_even_factors (n : ℕ) : ℕ :=
  if n = 2^4 * 3^2 * 5 * 7 then
    (4 * 3 * 2 * 2)  -- 4 choices for 2's exponent (1 to 4), 3 for 3's (0 to 2), 2 for 5's (0 to 1), 2 for 7's (0 to 1)
  else
    0  -- Return 0 if n is not equal to 2^4 * 3^2 * 5 * 7

theorem even_factors_count (n : ℕ) :
  n = 2^4 * 3^2 * 5 * 7 → num_even_factors n = 48 := by
  sorry

end NUMINAMATH_CALUDE_even_factors_count_l2000_200038


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l2000_200048

theorem unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n.val = 24 * k) ∧ 
    (7.9 : ℝ) < (n.val : ℝ)^(1/3) ∧ 
    (n.val : ℝ)^(1/3) < 8 ∧
    n.val = 504 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l2000_200048


namespace NUMINAMATH_CALUDE_dryer_sheet_box_cost_l2000_200006

/-- The cost of a box of dryer sheets -/
def box_cost (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (yearly_savings : ℚ) : ℚ :=
  yearly_savings / (loads_per_week * 52 / sheets_per_box)

/-- Theorem stating the cost of a box of dryer sheets -/
theorem dryer_sheet_box_cost :
  box_cost 4 1 104 11 = (11/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_dryer_sheet_box_cost_l2000_200006


namespace NUMINAMATH_CALUDE_greatest_base_seven_digit_sum_proof_l2000_200069

/-- The greatest possible sum of the digits in the base-seven representation of a positive integer less than 2019 -/
def greatest_base_seven_digit_sum : ℕ := 22

/-- A function that converts a natural number to its base-seven representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- A function that calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

theorem greatest_base_seven_digit_sum_proof :
  ∀ n : ℕ, 0 < n → n < 2019 →
  digit_sum (to_base_seven n) ≤ greatest_base_seven_digit_sum ∧
  ∃ m : ℕ, 0 < m ∧ m < 2019 ∧ digit_sum (to_base_seven m) = greatest_base_seven_digit_sum :=
sorry

end NUMINAMATH_CALUDE_greatest_base_seven_digit_sum_proof_l2000_200069
