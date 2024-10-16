import Mathlib

namespace NUMINAMATH_CALUDE_matrix_vector_computation_l85_8560

variable {m n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![2, -3])
  (hb : N.mulVec b = ![5, 4]) :
  N.mulVec (3 • a - 2 • b) = ![-4, -17] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l85_8560


namespace NUMINAMATH_CALUDE_karlson_max_candies_l85_8530

/-- The number of vertices in the complete graph -/
def n : ℕ := 29

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := 406

/-- Theorem stating the maximum number of candies Karlson could eat -/
theorem karlson_max_candies :
  (n * (n - 1)) / 2 = max_candies := by
  sorry

end NUMINAMATH_CALUDE_karlson_max_candies_l85_8530


namespace NUMINAMATH_CALUDE_fibonacci_mod_4_2022_l85_8570

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 4

theorem fibonacci_mod_4_2022 : b 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_mod_4_2022_l85_8570


namespace NUMINAMATH_CALUDE_smallest_sum_five_consecutive_primes_l85_8550

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if five consecutive natural numbers are all prime -/
def fiveConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3) ∧ isPrime (n + 4)

/-- The sum of five consecutive natural numbers starting from n -/
def sumFiveConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3) + (n + 4)

/-- The main theorem: 119 is the smallest sum of five consecutive primes divisible by 5 -/
theorem smallest_sum_five_consecutive_primes :
  ∃ n : ℕ, fiveConsecutivePrimes n ∧ 
           sumFiveConsecutive n = 119 ∧
           119 % 5 = 0 ∧
           (∀ m : ℕ, m < n → fiveConsecutivePrimes m → sumFiveConsecutive m % 5 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_five_consecutive_primes_l85_8550


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l85_8507

theorem product_of_four_consecutive_integers (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l85_8507


namespace NUMINAMATH_CALUDE_money_ratio_to_anna_l85_8509

def total_money : ℕ := 2000
def furniture_cost : ℕ := 400
def money_left : ℕ := 400

def money_after_furniture : ℕ := total_money - furniture_cost
def money_given_to_anna : ℕ := money_after_furniture - money_left

theorem money_ratio_to_anna : 
  (money_given_to_anna : ℚ) / (money_left : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_money_ratio_to_anna_l85_8509


namespace NUMINAMATH_CALUDE_angle_B_is_60_l85_8587

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : Real
  B : Real
  C : Real
  scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A
  sum_180 : A + B + C = 180

-- Define the specific triangle with given angle relationships
def SpecificTriangle (t : ScaleneTriangle) : Prop :=
  t.C = 3 * t.A ∧ t.B = 2 * t.A

-- Theorem statement
theorem angle_B_is_60 (t : ScaleneTriangle) (h : SpecificTriangle t) : t.B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_l85_8587


namespace NUMINAMATH_CALUDE_igneous_sedimentary_ratio_l85_8547

/-- Represents Cliff's rock collection --/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection --/
def isValidCollection (c : RockCollection) : Prop :=
  c.shinyIgneous = (2 * c.igneous) / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 40 ∧
  c.igneous + c.sedimentary = 180

/-- The theorem stating the ratio of igneous to sedimentary rocks --/
theorem igneous_sedimentary_ratio (c : RockCollection) 
  (h : isValidCollection c) : c.igneous * 2 = c.sedimentary := by
  sorry


end NUMINAMATH_CALUDE_igneous_sedimentary_ratio_l85_8547


namespace NUMINAMATH_CALUDE_only_origin_satisfies_l85_8583

def satisfies_inequality (x y : ℝ) : Prop := x + y - 1 < 0

theorem only_origin_satisfies : 
  satisfies_inequality 0 0 ∧ 
  ¬satisfies_inequality 2 4 ∧ 
  ¬satisfies_inequality (-1) 4 ∧ 
  ¬satisfies_inequality 1 8 :=
by sorry

end NUMINAMATH_CALUDE_only_origin_satisfies_l85_8583


namespace NUMINAMATH_CALUDE_expected_pine_saplings_l85_8526

theorem expected_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) : 
  ℕ :=
  20

#check expected_pine_saplings

end NUMINAMATH_CALUDE_expected_pine_saplings_l85_8526


namespace NUMINAMATH_CALUDE_win_sector_area_l85_8537

/-- Given a circular spinner with radius 10 cm and a probability of winning 2/5,
    the area of the WIN sector is 40π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 10 →
  p = 2 / 5 →
  A_win = p * π * r^2 →
  A_win = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_win_sector_area_l85_8537


namespace NUMINAMATH_CALUDE_problem_solution_l85_8569

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, (f 2 x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) →
  -- Part 1: Prove that a = 2
  (∃! a : ℝ, ∀ x : ℝ, (f a x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) ∧
  -- Part 2: Prove that the minimum value of f(x) + f(x+5) is 5
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) ∧
  (∃ x : ℝ, f 2 x + f 2 (x + 5) = 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l85_8569


namespace NUMINAMATH_CALUDE_odd_sum_count_l85_8505

def card_set : Finset ℕ := {1, 2, 3, 4}

def is_sum_odd (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_sum_odd pair)

theorem odd_sum_count : odd_sum_pairs.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_count_l85_8505


namespace NUMINAMATH_CALUDE_highlighters_count_l85_8528

/-- The number of pink highlighters -/
def pink_highlighters : Nat := 47

/-- The number of yellow highlighters -/
def yellow_highlighters : Nat := 36

/-- The number of blue highlighters -/
def blue_highlighters : Nat := 21

/-- The number of orange highlighters -/
def orange_highlighters : Nat := 15

/-- The number of green highlighters -/
def green_highlighters : Nat := 27

/-- The total number of highlighters -/
def total_highlighters : Nat :=
  pink_highlighters + yellow_highlighters + blue_highlighters + orange_highlighters + green_highlighters

theorem highlighters_count : total_highlighters = 146 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l85_8528


namespace NUMINAMATH_CALUDE_right_triangle_legs_sum_l85_8501

theorem right_triangle_legs_sum (a b c : ℕ) : 
  a + 1 = b →                -- legs are consecutive integers
  a^2 + b^2 = 41^2 →         -- Pythagorean theorem with hypotenuse 41
  a + b = 59 :=              -- sum of legs is 59
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_sum_l85_8501


namespace NUMINAMATH_CALUDE_march_greatest_drop_l85_8580

/-- Represents the months in the first half of 2021 -/
inductive Month
| january
| february
| march
| april
| may
| june

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -3.00
  | Month.february => 1.50
  | Month.march    => -4.50
  | Month.april    => 2.00
  | Month.may      => -1.00
  | Month.june     => 0.50

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.march

theorem march_greatest_drop :
  ∀ m : Month, price_change greatest_drop ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l85_8580


namespace NUMINAMATH_CALUDE_consecutive_integers_product_272_sum_33_l85_8557

theorem consecutive_integers_product_272_sum_33 :
  ∀ x y : ℕ,
  x > 0 →
  y = x + 1 →
  x * y = 272 →
  x + y = 33 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_272_sum_33_l85_8557


namespace NUMINAMATH_CALUDE_alternating_sequence_solution_l85_8595

theorem alternating_sequence_solution (n : ℕ) (h : n ≥ 4) :
  ∃! (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ 2*n → a i > 0) ∧
    (∀ k, 0 ≤ k ∧ k < n →
      a (2*k+1) = 1/(a (2*n)) + 1/(a (2*k+2)) ∧
      a (2*k+2) = a (2*k+1) + a (2*k+3)) ∧
    (a (2*n) = a (2*n-1) + a 1) →
  ∀ k, 0 ≤ k ∧ k < n → a (2*k+1) = 1 ∧ a (2*k+2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_alternating_sequence_solution_l85_8595


namespace NUMINAMATH_CALUDE_tiles_difference_l85_8527

/-- Represents the number of tiles in the nth square of the progression -/
def tiles (n : ℕ) : ℕ := n^2

/-- The difference in the number of tiles between the 8th and 6th squares is 28 -/
theorem tiles_difference : tiles 8 - tiles 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_l85_8527


namespace NUMINAMATH_CALUDE_remainder_2685976_div_8_l85_8582

theorem remainder_2685976_div_8 : 2685976 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2685976_div_8_l85_8582


namespace NUMINAMATH_CALUDE_smallest_three_digit_ending_l85_8598

def ends_same_three_digits (x : ℕ) : Prop :=
  x^2 % 1000 = x % 1000

theorem smallest_three_digit_ending : ∀ y > 1, ends_same_three_digits y → y ≥ 376 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_ending_l85_8598


namespace NUMINAMATH_CALUDE_diagonal_segments_100x101_l85_8511

/-- The number of segments in the diagonal of a rectangle divided by grid lines -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - 1

/-- The width of the rectangle -/
def rectangle_width : ℕ := 100

/-- The height of the rectangle -/
def rectangle_height : ℕ := 101

theorem diagonal_segments_100x101 :
  diagonal_segments rectangle_width rectangle_height = 200 := by
  sorry

#eval diagonal_segments rectangle_width rectangle_height

end NUMINAMATH_CALUDE_diagonal_segments_100x101_l85_8511


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l85_8551

theorem inverse_variation_problem (k : ℝ) (h1 : k > 0) :
  (∀ x y : ℝ, x ≠ 0 → y * x^2 = k) →
  (2 * 3^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k ∧ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l85_8551


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l85_8536

theorem min_stamps_for_50_cents : ∃ (p q : ℕ), 
  5 * p + 4 * q = 50 ∧ 
  p + q = 11 ∧ 
  ∀ (x y : ℕ), 5 * x + 4 * y = 50 → x + y ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l85_8536


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l85_8545

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3/2) * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ k : Int, k * Real.pi + Real.pi/2 < α ∧ α < k * Real.pi + Real.pi) ∨
  (∃ k : Int, k * Real.pi + (3/2) * Real.pi < α ∧ α < (k + 1) * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α/2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l85_8545


namespace NUMINAMATH_CALUDE_calendar_box_sum_divisible_by_four_l85_8542

/-- Represents a box of four numbers in a 7-column calendar --/
structure CalendarBox where
  top_right : ℕ
  top_left : ℕ
  bottom_left : ℕ
  bottom_right : ℕ

/-- Creates a calendar box given the top right number --/
def make_calendar_box (a : ℕ) : CalendarBox :=
  { top_right := a
  , top_left := a - 1
  , bottom_left := a + 6
  , bottom_right := a + 7 }

/-- The sum of numbers in a calendar box --/
def box_sum (box : CalendarBox) : ℕ :=
  box.top_right + box.top_left + box.bottom_left + box.bottom_right

/-- Theorem: The sum of numbers in any calendar box is divisible by 4 --/
theorem calendar_box_sum_divisible_by_four (a : ℕ) :
  4 ∣ box_sum (make_calendar_box a) := by
  sorry

end NUMINAMATH_CALUDE_calendar_box_sum_divisible_by_four_l85_8542


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l85_8568

def a (n : ℕ) : ℚ := (1 + 3 * n) / (6 - n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l85_8568


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l85_8577

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (12 * x₁^2 + 16 * x₁ - 21 = 0) → 
  (12 * x₂^2 + 16 * x₂ - 21 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 95/18) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l85_8577


namespace NUMINAMATH_CALUDE_complex_on_negative_y_axis_l85_8539

def complex_operation : ℂ := (5 - 6*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I)

theorem complex_on_negative_y_axis : 
  complex_operation.re = 0 ∧ complex_operation.im < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_on_negative_y_axis_l85_8539


namespace NUMINAMATH_CALUDE_expression_simplification_l85_8596

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (18 * x^3) * (8 * y) * (1 / (6 * x * y)^2) = 4 * x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l85_8596


namespace NUMINAMATH_CALUDE_root_product_value_l85_8516

theorem root_product_value (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → 
  n^2 - 3*n - 2 = 0 → 
  (7*m^2 - 21*m - 3)*(3*n^2 - 9*n + 5) = 121 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l85_8516


namespace NUMINAMATH_CALUDE_relationship_abc_l85_8518

theorem relationship_abc : 
  let a := (3/4)^(2/3)
  let b := (2/3)^(3/4)
  let c := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l85_8518


namespace NUMINAMATH_CALUDE_population_difference_is_167_l85_8566

/-- Represents a tribe with male and female populations -/
structure Tribe where
  males : Nat
  females : Nat

/-- The Gaga tribe -/
def gaga : Tribe := ⟨204, 468⟩

/-- The Nana tribe -/
def nana : Tribe := ⟨334, 516⟩

/-- The Dada tribe -/
def dada : Tribe := ⟨427, 458⟩

/-- The Lala tribe -/
def lala : Tribe := ⟨549, 239⟩

/-- The list of all tribes on the couple continent -/
def tribes : List Tribe := [gaga, nana, dada, lala]

/-- The total number of males on the couple continent -/
def totalMales : Nat := (tribes.map (·.males)).sum

/-- The total number of females on the couple continent -/
def totalFemales : Nat := (tribes.map (·.females)).sum

/-- The difference between females and males on the couple continent -/
def populationDifference : Nat := totalFemales - totalMales

theorem population_difference_is_167 : populationDifference = 167 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_is_167_l85_8566


namespace NUMINAMATH_CALUDE_monotone_increasing_sequence_condition_l85_8531

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

-- State the theorem
theorem monotone_increasing_sequence_condition (b : ℝ) :
  (∀ n : ℕ, a (n + 1) b > a n b) → b > -3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_sequence_condition_l85_8531


namespace NUMINAMATH_CALUDE_base_ten_to_seven_l85_8578

theorem base_ten_to_seven : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_seven_l85_8578


namespace NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_l85_8503

variable (p q : Prop)

theorem either_false_sufficient_not_necessary :
  (((¬p ∨ ¬q) → ¬p) ∧ ¬(¬p → (¬p ∨ ¬q))) := by sorry

end NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_l85_8503


namespace NUMINAMATH_CALUDE_circle_rolling_in_triangle_l85_8502

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (h_radius : r = 2) : 
  (a - 2*r) + (b - 2*r) + (c - 2*r) = 24 := by
sorry


end NUMINAMATH_CALUDE_circle_rolling_in_triangle_l85_8502


namespace NUMINAMATH_CALUDE_expression_simplification_l85_8555

theorem expression_simplification (a b : ℝ) 
  (ha : a = 2 + Real.sqrt 3) 
  (hb : b = 2 - Real.sqrt 3) : 
  (a^2 - b^2) / a / (a - (2*a*b - b^2) / a) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l85_8555


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l85_8521

theorem consecutive_integers_sum_of_powers (n : ℕ) : 
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 9458) →
  ((n - 1)^4 + n^4 + (n + 1)^4 = 30212622) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l85_8521


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l85_8591

theorem quadratic_one_solution (k : ℝ) :
  (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ (k = 8 ∨ k = -8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l85_8591


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l85_8529

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l85_8529


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l85_8553

/-- For a circle where the product of three inches and its circumference (in inches) 
    equals its area, the radius of the circle is 6 inches. -/
theorem circle_radius_is_six (r : ℝ) (h : 3 * (2 * Real.pi * r) = Real.pi * r^2) : r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l85_8553


namespace NUMINAMATH_CALUDE_johns_croissants_l85_8546

theorem johns_croissants :
  ∀ (c : ℕ) (k : ℕ),
  c + k = 5 →
  (88 * c + 44 * k) % 100 = 0 →
  c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_croissants_l85_8546


namespace NUMINAMATH_CALUDE_range_of_a_l85_8525

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l85_8525


namespace NUMINAMATH_CALUDE_no_intersection_l85_8599

/-- The number of distinct points of intersection between two ellipses -/
def intersectionPoints (f g : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First ellipse: 3x^2 + 2y^2 = 4 -/
def ellipse1 (x y : ℝ) : Prop :=
  3 * x^2 + 2 * y^2 = 4

/-- Second ellipse: 6x^2 + 3y^2 = 9 -/
def ellipse2 (x y : ℝ) : Prop :=
  6 * x^2 + 3 * y^2 = 9

/-- Theorem: The number of distinct points of intersection between the two given ellipses is 0 -/
theorem no_intersection : intersectionPoints ellipse1 ellipse2 = 0 :=
  sorry

end NUMINAMATH_CALUDE_no_intersection_l85_8599


namespace NUMINAMATH_CALUDE_pet_store_dogs_l85_8514

/-- The number of dogs in a pet store with dogs and parakeets -/
def num_dogs : ℕ := 6

/-- The number of parakeets in the pet store -/
def num_parakeets : ℕ := 15 - num_dogs

/-- The total number of heads in the pet store -/
def total_heads : ℕ := 15

/-- The total number of feet in the pet store -/
def total_feet : ℕ := 42

theorem pet_store_dogs :
  num_dogs + num_parakeets = total_heads ∧
  4 * num_dogs + 2 * num_parakeets = total_feet :=
by sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l85_8514


namespace NUMINAMATH_CALUDE_factorial_ratio_l85_8500

-- Define the factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio : (factorial 50) / (factorial 48) = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l85_8500


namespace NUMINAMATH_CALUDE_four_digit_numbers_extrema_l85_8544

theorem four_digit_numbers_extrema :
  let sum_of_numbers : ℕ := 106656
  let is_valid_number : (ℕ → Bool) :=
    λ n => n ≥ 1000 ∧ n ≤ 9999 ∧ 
           (let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
            digits.all (· ≠ 0) ∧ digits.Nodup)
  let valid_numbers := (List.range 10000).filter is_valid_number
  sum_of_numbers = valid_numbers.sum →
  (∀ n ∈ valid_numbers, n ≤ 9421 ∧ n ≥ 1249) ∧
  9421 ∈ valid_numbers ∧ 1249 ∈ valid_numbers :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_extrema_l85_8544


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l85_8581

/-- Given two quadratic equations where the roots of one are three times those of the other,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -p ∧ s₁ * s₂ = m) ∧
               (3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n)) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l85_8581


namespace NUMINAMATH_CALUDE_triangle_area_l85_8512

/-- Given a triangle ABC with sides a, b, and c, prove that its area is 3√2/4
    when sinA = √3/3 and b² + c² - a² = 6 -/
theorem triangle_area (a b c : ℝ) (h1 : Real.sin A = Real.sqrt 3 / 3) 
  (h2 : b^2 + c^2 - a^2 = 6) : 
  (1/2 : ℝ) * b * c * Real.sin A = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l85_8512


namespace NUMINAMATH_CALUDE_max_number_after_two_moves_l85_8540

def initial_number : ℕ := 4597

def swap_adjacent_digits (n : ℕ) (i : ℕ) : ℕ := 
  sorry

def subtract_100 (n : ℕ) : ℕ := 
  sorry

def make_move (n : ℕ) (i : ℕ) : ℕ := 
  subtract_100 (swap_adjacent_digits n i)

def max_after_moves (n : ℕ) (moves : ℕ) : ℕ := 
  sorry

theorem max_number_after_two_moves : 
  max_after_moves initial_number 2 = 4659 := by
  sorry

end NUMINAMATH_CALUDE_max_number_after_two_moves_l85_8540


namespace NUMINAMATH_CALUDE_exactly_two_primes_in_ten_consecutive_l85_8556

/-- A function that determines if a number is prime --/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of primes in a list of natural numbers --/
def countPrimes (list : List ℕ) : ℕ := sorry

/-- A function that generates a list of 10 consecutive numbers starting from a given number --/
def consecutiveNumbers (start : ℕ) : List ℕ := sorry

/-- The theorem to be proved --/
theorem exactly_two_primes_in_ten_consecutive : 
  (Finset.filter (fun k => countPrimes (consecutiveNumbers k) = 2) (Finset.range 21)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_primes_in_ten_consecutive_l85_8556


namespace NUMINAMATH_CALUDE_binomial_expansion_unique_m_l85_8524

/-- Given constants b and y, and a natural number m, such that the second, third, and fourth terms
    in the expansion of (b + y)^m are 6, 24, and 60 respectively, prove that m = 11. -/
theorem binomial_expansion_unique_m (b y : ℝ) (m : ℕ) 
  (h1 : (m.choose 1) * b^(m-1) * y = 6)
  (h2 : (m.choose 2) * b^(m-2) * y^2 = 24)
  (h3 : (m.choose 3) * b^(m-3) * y^3 = 60) :
  m = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_unique_m_l85_8524


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l85_8510

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 ≤ 2 * ((7 * n + 21) / 7)) → n ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l85_8510


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l85_8563

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process described in the problem -/
def ballPlacementProcess (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of balls after n steps is equal to
    the sum of digits in the base 6 representation of n -/
theorem ball_placement_theorem (n : ℕ) :
  ballPlacementProcess n = sumDigits (toBase6 n) :=
  sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l85_8563


namespace NUMINAMATH_CALUDE_smallest_n_with_constant_term_l85_8517

theorem smallest_n_with_constant_term :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n → ¬∃ (r : ℕ), 2*k = 5*r) ∧
  (∃ (r : ℕ), 2*n = 5*r) ∧
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_constant_term_l85_8517


namespace NUMINAMATH_CALUDE_book_pages_sum_l85_8567

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 13 pages in the first chapter and 68 pages in the second chapter has 81 pages in total -/
theorem book_pages_sum : 
  ∀ (b : Book), b.chapter1_pages = 13 ∧ b.chapter2_pages = 68 → total_pages b = 81 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_sum_l85_8567


namespace NUMINAMATH_CALUDE_divisible_by_41_l85_8504

theorem divisible_by_41 (n : ℕ) : ∃ k : ℤ, 5 * 7^(2*(n+1)) + 2^(3*n) = 41 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_41_l85_8504


namespace NUMINAMATH_CALUDE_mass_percentage_not_sufficient_for_unique_compound_l85_8584

/-- Represents a chemical compound -/
structure Compound where
  name : String
  mass_percentage_O : Float

/-- The mass percentage of O in the compound -/
def given_mass_percentage : Float := 36.36

/-- Theorem stating that the given mass percentage of O is not sufficient to uniquely determine a compound -/
theorem mass_percentage_not_sufficient_for_unique_compound :
  ∃ (c1 c2 : Compound), c1.mass_percentage_O = given_mass_percentage ∧ 
                        c2.mass_percentage_O = given_mass_percentage ∧ 
                        c1.name ≠ c2.name :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_not_sufficient_for_unique_compound_l85_8584


namespace NUMINAMATH_CALUDE_polygon_side_length_theorem_l85_8576

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon

/-- Represents a way to divide a polygon into equilateral triangles and squares. -/
structure Division where
  -- Add necessary fields for a division

/-- Counts the number of ways to divide a polygon into equilateral triangles and squares. -/
def countDivisions (M : ConvexPolygon) : ℕ :=
  sorry

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop :=
  sorry

/-- Gets the length of a side of a polygon. -/
def sideLength (M : ConvexPolygon) (side : ℕ) : ℕ :=
  sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) :
  isPrime p → countDivisions M = p → ∃ side, sideLength M side = p - 1 :=
by sorry

end NUMINAMATH_CALUDE_polygon_side_length_theorem_l85_8576


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l85_8590

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(21 ∣ (105829 - y))) ∧ 
  (21 ∣ (105829 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l85_8590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_theorem_l85_8513

/-- The sum of an arithmetic sequence with first term 3, common difference 4, and last term not exceeding 47 -/
def arithmetic_sequence_sum : ℕ → ℕ := λ n => n * (3 + (4 * n - 1)) / 2

/-- The number of terms in the sequence -/
def n : ℕ := 12

theorem arithmetic_sequence_sum_theorem :
  (∀ k : ℕ, k ≤ n → 3 + 4 * (k - 1) ≤ 47) ∧ 
  3 + 4 * (n - 1) = 47 ∧
  arithmetic_sequence_sum n = 300 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_theorem_l85_8513


namespace NUMINAMATH_CALUDE_min_floor_sum_l85_8515

theorem min_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋ ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l85_8515


namespace NUMINAMATH_CALUDE_nedy_crackers_total_l85_8574

/-- The number of packs of crackers Nedy eats per day from Monday to Thursday -/
def daily_crackers : ℕ := 8

/-- The number of days from Monday to Thursday -/
def weekdays : ℕ := 4

/-- The factor by which Nedy increases his cracker consumption on Friday -/
def friday_factor : ℕ := 2

/-- Theorem: Given Nedy eats 8 packs of crackers per day from Monday to Thursday
    and twice that amount on Friday, the total number of crackers Nedy eats
    from Monday to Friday is 48 packs. -/
theorem nedy_crackers_total :
  daily_crackers * weekdays + daily_crackers * friday_factor = 48 := by
  sorry

end NUMINAMATH_CALUDE_nedy_crackers_total_l85_8574


namespace NUMINAMATH_CALUDE_christine_siri_money_difference_l85_8548

theorem christine_siri_money_difference :
  ∀ (christine_amount siri_amount : ℝ),
    christine_amount + siri_amount = 21 →
    christine_amount = 20.5 →
    christine_amount - siri_amount = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_christine_siri_money_difference_l85_8548


namespace NUMINAMATH_CALUDE_paper_covers_cube_l85_8522

theorem paper_covers_cube (cube_edge : ℝ) (paper_side : ℝ) 
  (h1 : cube_edge = 1) (h2 : paper_side = 2.5) : 
  paper_side ^ 2 ≥ 6 * cube_edge ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_paper_covers_cube_l85_8522


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l85_8538

theorem gas_station_candy_boxes : 3 + 5 + 2 + 4 + 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l85_8538


namespace NUMINAMATH_CALUDE_equation_satisfaction_l85_8585

theorem equation_satisfaction (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  ((10 * a + b) * (10 * b + a) = 100 * a^2 + a * b + 100 * b^2) ↔ (a = b) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l85_8585


namespace NUMINAMATH_CALUDE_no_simultaneous_doughnut_and_syrup_l85_8554

theorem no_simultaneous_doughnut_and_syrup :
  ¬∃ (x : ℝ), (x^2 - 9*x + 13 < 0) ∧ (x^2 + x - 5 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_doughnut_and_syrup_l85_8554


namespace NUMINAMATH_CALUDE_parabola_translation_l85_8533

/-- The translation of a parabola y = x^2 upwards by 3 units and to the left by 1 unit -/
theorem parabola_translation (x y : ℝ) :
  (y = x^2) →  -- Original parabola
  (y = (x + 1)^2 + 3) →  -- Resulting parabola after translation
  (∀ (x' y' : ℝ), y' = x'^2 → y' + 3 = ((x' + 1)^2 + 3)) -- Equivalence of the translation
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l85_8533


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l85_8532

/-- The sum of the infinite series Σ(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let series_term (k : ℕ) := (12 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1))
  ∑' k, series_term k = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l85_8532


namespace NUMINAMATH_CALUDE_managers_salary_l85_8552

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 24 →
  avg_salary = 2400 →
  avg_increase = 100 →
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + avg_increase →
  manager_salary = 4900 :=
by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l85_8552


namespace NUMINAMATH_CALUDE_eighth_term_value_l85_8534

theorem eighth_term_value (S : ℕ → ℕ) (h : ∀ n : ℕ, S n = n^2) :
  S 8 - S 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l85_8534


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l85_8523

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l85_8523


namespace NUMINAMATH_CALUDE_num_category_B_prob_both_categories_l85_8571

/- Define the number of category A housekeepers -/
def category_A : ℕ := 12

/- Define the total number of housekeepers selected for training -/
def selected_total : ℕ := 20

/- Define the number of category B housekeepers selected for training -/
def selected_B : ℕ := 16

/- Define the number of category A housekeepers available for hiring -/
def available_A : ℕ := 3

/- Define the number of category B housekeepers available for hiring -/
def available_B : ℕ := 2

/- Theorem for the number of category B housekeepers -/
theorem num_category_B : ∃ x : ℕ, 
  (category_A * selected_B) / (selected_total - selected_B) = x :=
sorry

/- Theorem for the probability of hiring from both categories -/
theorem prob_both_categories : 
  (available_A * available_B) / ((available_A + available_B) * (available_A + available_B - 1) / 2) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_num_category_B_prob_both_categories_l85_8571


namespace NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_720_l85_8543

def S : Finset Int := {-10, -7, -3, 0, 2, 4, 8, 9}

theorem min_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≥ x * y * z :=
by
  sorry

theorem min_product_is_neg_720 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a * b * c = -720 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≥ -720) :=
by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_720_l85_8543


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l85_8594

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : (Circle × Circle × Circle) :=
  let small_circle : Circle := { center := (0, 0), radius := 2 }
  let large_circle1 : Circle := { center := (-2, 0), radius := 3 }
  let large_circle2 : Circle := { center := (2, 0), radius := 3 }
  (small_circle, large_circle1, large_circle2)

-- Define the shaded area function
noncomputable def shaded_area (setup : Circle × Circle × Circle) : ℝ :=
  2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area problem_setup = 2 * Real.pi - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l85_8594


namespace NUMINAMATH_CALUDE_sum_of_digits_equation_l85_8558

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_equation : 
  ∃ (n : ℕ), n + sum_of_digits n = 2018 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_equation_l85_8558


namespace NUMINAMATH_CALUDE_watermelon_cost_proof_l85_8562

/-- Represents the number of fruits a container can hold -/
def ContainerCapacity : ℕ := 150

/-- Represents the total value of fruits in rubles -/
def TotalValue : ℕ := 24000

/-- Represents the capacity of the container in terms of melons -/
def MelonCapacity : ℕ := 120

/-- Represents the capacity of the container in terms of watermelons -/
def WatermelonCapacity : ℕ := 160

/-- Represents the cost of a single watermelon in rubles -/
def WatermelonCost : ℕ := 100

theorem watermelon_cost_proof :
  ∃ (num_watermelons num_melons : ℕ),
    num_watermelons + num_melons = ContainerCapacity ∧
    num_watermelons * WatermelonCost = num_melons * (TotalValue / num_melons) ∧
    num_watermelons * WatermelonCost + num_melons * (TotalValue / num_melons) = TotalValue ∧
    num_watermelons * (1 / WatermelonCapacity) + num_melons * (1 / MelonCapacity) = 1 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_cost_proof_l85_8562


namespace NUMINAMATH_CALUDE_manuscript_revisions_l85_8535

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_all_first_typing := total_pages * cost_first_typing
  let cost_revisions_once := pages_revised_once * cost_revision
  let remaining_cost := total_cost - cost_all_first_typing - cost_revisions_once
  remaining_cost / (2 * cost_revision)

theorem manuscript_revisions (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_revised_once = 30)
  (h3 : cost_first_typing = 10)
  (h4 : cost_revision = 5)
  (h5 : total_cost = 1350) :
  pages_revised_twice total_pages pages_revised_once cost_first_typing cost_revision total_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revisions_l85_8535


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_tenth_polygon_l85_8592

/-- The number of sides of the nth polygon in the sequence -/
def sides (n : ℕ) : ℕ := n + 2

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The 10th polygon in the sequence -/
def tenth_polygon : ℕ := 10

theorem sum_of_interior_angles_tenth_polygon :
  interior_angle_sum (sides tenth_polygon) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_tenth_polygon_l85_8592


namespace NUMINAMATH_CALUDE_line_equation_l85_8561

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point being on a line
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the midpoint of two points
def is_midpoint (p m q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

-- Theorem statement
theorem line_equation : 
  ∀ (l : Line) (p a b : Point),
    on_line p l →
    p.x = 4 ∧ p.y = 1 →
    hyperbola a.x a.y →
    hyperbola b.x b.y →
    is_midpoint a p b →
    l.a = 1 ∧ l.b = -1 ∧ l.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l85_8561


namespace NUMINAMATH_CALUDE_probability_divisible_by_8_l85_8575

def is_valid_digit (d : ℕ) : Prop := d ∈ ({3, 58} : Set ℕ)

def form_number (x y : ℕ) : ℕ := 460000 + x * 1000 + y * 100 + 12

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem probability_divisible_by_8 :
  ∀ x y : ℕ, is_valid_digit x → is_valid_digit y →
  (∃! y', is_valid_digit y' ∧ is_divisible_by_8 (form_number x y')) :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_8_l85_8575


namespace NUMINAMATH_CALUDE_function_properties_l85_8506

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ sorry

-- State the main theorem
theorem function_properties (h : ∀ x : ℝ, 3 * f (2 - x) - 2 * f x = x^2 - 2*x) :
  (∀ x : ℝ, f x = x^2 - 2*x) ∧
  (∀ a : ℝ, a > 1 → ∀ x : ℝ, f x + a > 0) ∧
  (∀ x : ℝ, f x + 1 > 0 ↔ x ≠ 1) ∧
  (∀ a : ℝ, a < 1 → ∀ x : ℝ, f x + a > 0 ↔ x > 1 + Real.sqrt (1 - a) ∨ x < 1 - Real.sqrt (1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l85_8506


namespace NUMINAMATH_CALUDE_line_MN_equation_l85_8589

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 48

-- Define points A, B, and C
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (2, 0)

-- Define P and Q as points on the ellipse
def P_on_ellipse (P : ℝ × ℝ) : Prop := is_on_ellipse P.1 P.2
def Q_on_ellipse (Q : ℝ × ℝ) : Prop := is_on_ellipse Q.1 Q.2

-- PQ passes through C but not origin
def PQ_through_C (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ C = (t • P.1 + (1 - t) • Q.1, t • P.2 + (1 - t) • Q.2)

-- Define M as intersection of AP and QB
def M_is_intersection (P Q M : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, M = (t₁ • A.1 + (1 - t₁) • P.1, t₁ • A.2 + (1 - t₁) • P.2) ∧
              M = (t₂ • Q.1 + (1 - t₂) • B.1, t₂ • Q.2 + (1 - t₂) • B.2)

-- Define N as intersection of PB and AQ
def N_is_intersection (P Q N : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, N = (t₁ • P.1 + (1 - t₁) • B.1, t₁ • P.2 + (1 - t₁) • B.2) ∧
              N = (t₂ • A.1 + (1 - t₂) • Q.1, t₂ • A.2 + (1 - t₂) • Q.2)

-- The main theorem
theorem line_MN_equation (P Q M N : ℝ × ℝ) :
  P_on_ellipse P → Q_on_ellipse Q → PQ_through_C P Q →
  M_is_intersection P Q M → N_is_intersection P Q N →
  M.1 = 8 ∧ N.1 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_MN_equation_l85_8589


namespace NUMINAMATH_CALUDE_age_ratio_proof_l85_8565

/-- Given three people a, b, and c with ages satisfying certain conditions,
    prove that the ratio of b's age to c's age is 2:1. -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →                  -- a is two years older than b
  b = 18 →                     -- b is 18 years old
  a + b + c = 47 →             -- total of ages is 47
  ∃ (k : ℕ), b = k * c →       -- b is some times as old as c
  b = 2 * c                    -- ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l85_8565


namespace NUMINAMATH_CALUDE_fill_time_both_pipes_l85_8559

def pipe1_time : ℝ := 8
def pipe2_time : ℝ := 12

theorem fill_time_both_pipes :
  let rate1 := 1 / pipe1_time
  let rate2 := 1 / pipe2_time
  let combined_rate := rate1 + rate2
  (1 / combined_rate) = 4.8 := by sorry

end NUMINAMATH_CALUDE_fill_time_both_pipes_l85_8559


namespace NUMINAMATH_CALUDE_cricket_team_handedness_l85_8508

theorem cricket_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_handedness_l85_8508


namespace NUMINAMATH_CALUDE_p_recurrence_l85_8519

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l85_8519


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l85_8520

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4) / (x^2 - 4*x + 4) + (x / (x^2 - x)) / ((x - 2) / (x - 1))
  f (-1) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l85_8520


namespace NUMINAMATH_CALUDE_difference_of_squares_l85_8573

theorem difference_of_squares (x y : ℝ) : 
  x + y = 15 → x - y = 10 → x^2 - y^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l85_8573


namespace NUMINAMATH_CALUDE_impossibleToMakeAllMultiplesOfTen_l85_8541

/-- Represents an 8x8 grid of integers -/
def Grid := Fin 8 → Fin 8 → ℤ

/-- Represents an operation on the grid -/
inductive Operation
| threeByThree (i j : Fin 8) : Operation
| fourByFour (i j : Fin 8) : Operation

/-- Apply an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Check if all numbers in the grid are multiples of 10 -/
def allMultiplesOfTen (g : Grid) : Prop :=
  ∀ i j, ∃ k, g i j = 10 * k

/-- The main theorem -/
theorem impossibleToMakeAllMultiplesOfTen :
  ∃ (g : Grid),
    (∀ i j, g i j ≥ 0) ∧
    ¬∃ (ops : List Operation), allMultiplesOfTen (ops.foldl applyOperation g) :=
  sorry

end NUMINAMATH_CALUDE_impossibleToMakeAllMultiplesOfTen_l85_8541


namespace NUMINAMATH_CALUDE_impossible_tiling_l85_8588

/-- Represents a rectangle with shaded cells -/
structure ShadedRectangle where
  rows : Nat
  cols : Nat
  shaded_cells : Nat

/-- Represents a tiling strip -/
structure TilingStrip where
  width : Nat
  height : Nat

/-- Checks if a rectangle can be tiled with given strips -/
def canBeTiled (rect : ShadedRectangle) (strip : TilingStrip) : Prop :=
  rect.rows * rect.cols % (strip.width * strip.height) = 0 ∧
  rect.shaded_cells % strip.width = 0 ∧
  rect.shaded_cells / strip.width = rect.rows * rect.cols / (strip.width * strip.height)

theorem impossible_tiling (rect : ShadedRectangle) (strip : TilingStrip) :
  rect.rows = 4 ∧ rect.cols = 9 ∧ rect.shaded_cells = 15 ∧
  strip.width = 3 ∧ strip.height = 1 →
  ¬ canBeTiled rect strip := by
  sorry

#check impossible_tiling

end NUMINAMATH_CALUDE_impossible_tiling_l85_8588


namespace NUMINAMATH_CALUDE_donut_selection_count_donut_problem_l85_8564

theorem donut_selection_count : Nat → Nat → Nat
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem donut_problem : donut_selection_count 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_donut_problem_l85_8564


namespace NUMINAMATH_CALUDE_third_shot_scores_l85_8597

/-- Represents a shooter's scores across 5 shots -/
structure ShooterScores where
  scores : Fin 5 → ℕ

/-- The problem setup -/
def ShootingProblem (shooter1 shooter2 : ShooterScores) : Prop :=
  -- The first three shots resulted in the same number of points
  (shooter1.scores 0 + shooter1.scores 1 + shooter1.scores 2 =
   shooter2.scores 0 + shooter2.scores 1 + shooter2.scores 2) ∧
  -- In the last three shots, the first shooter scored three times as many points as the second shooter
  (shooter1.scores 2 + shooter1.scores 3 + shooter1.scores 4 =
   3 * (shooter2.scores 2 + shooter2.scores 3 + shooter2.scores 4))

/-- The theorem to prove -/
theorem third_shot_scores (shooter1 shooter2 : ShooterScores)
    (h : ShootingProblem shooter1 shooter2) :
    shooter1.scores 2 = 10 ∧ shooter2.scores 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_third_shot_scores_l85_8597


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l85_8549

theorem quadratic_inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l85_8549


namespace NUMINAMATH_CALUDE_remainder_problem_l85_8572

theorem remainder_problem (greatest_divisor remainder_4521 : ℕ) 
  (h1 : greatest_divisor = 88)
  (h2 : remainder_4521 = 33)
  (h3 : ∃ q1 : ℕ, 3815 = greatest_divisor * q1 + (3815 % greatest_divisor))
  (h4 : ∃ q2 : ℕ, 4521 = greatest_divisor * q2 + remainder_4521) :
  3815 % greatest_divisor = 31 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l85_8572


namespace NUMINAMATH_CALUDE_smaller_field_area_l85_8593

theorem smaller_field_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 10 →
  smaller_area = 225 := by
sorry

end NUMINAMATH_CALUDE_smaller_field_area_l85_8593


namespace NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l85_8586

/-- The number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := n^2

/-- The total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l85_8586


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_7_range_of_m_for_solution_exists_l85_8579

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |2*x - 3|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_greater_than_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x < -3/2 ∨ x > 2} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_solution_exists :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_7_range_of_m_for_solution_exists_l85_8579
