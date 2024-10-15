import Mathlib

namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2166_216699

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ k : ℕ, k > 12 ∧ (∀ m : ℕ, m > 0 → k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) →
  False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2166_216699


namespace NUMINAMATH_CALUDE_book_pages_count_l2166_216696

/-- Represents the number of pages read in a day period --/
structure ReadingPeriod where
  days : ℕ
  pagesPerDay : ℕ

/-- Calculates the total pages read in a period --/
def totalPages (period : ReadingPeriod) : ℕ :=
  period.days * period.pagesPerDay

/-- Represents Robert's reading schedule --/
def robertReading : List ReadingPeriod :=
  [{ days := 3, pagesPerDay := 28 },
   { days := 3, pagesPerDay := 35 },
   { days := 3, pagesPerDay := 42 }]

/-- The number of pages Robert read on the last day --/
def lastDayPages : ℕ := 15

/-- Theorem stating the total number of pages in the book --/
theorem book_pages_count :
  (robertReading.map totalPages).sum + lastDayPages = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l2166_216696


namespace NUMINAMATH_CALUDE_triangle_construction_l2166_216689

-- Define the necessary structures and properties
structure Point where
  x : ℝ
  y : ℝ

def nonCollinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def isOrthocenter (P A B C : Point) : Prop :=
  ((B.y - A.y) * (P.x - A.x) + (A.x - B.x) * (P.y - A.y) = 0) ∧
  ((C.y - B.y) * (P.x - B.x) + (B.x - C.x) * (P.y - B.y) = 0) ∧
  ((A.y - C.y) * (P.x - C.x) + (C.x - A.x) * (P.y - C.y) = 0)

-- State the theorem
theorem triangle_construction (M N P : Point) (h : nonCollinear M N P) :
  ∃ (A B C : Point),
    (isMidpoint M A B ∨ isMidpoint M B C ∨ isMidpoint M A C) ∧
    (isMidpoint N A B ∨ isMidpoint N B C ∨ isMidpoint N A C) ∧
    (isMidpoint M A B → isMidpoint N A C ∨ isMidpoint N B C) ∧
    (isMidpoint M B C → isMidpoint N A B ∨ isMidpoint N A C) ∧
    (isMidpoint M A C → isMidpoint N A B ∨ isMidpoint N B C) ∧
    isOrthocenter P A B C :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_l2166_216689


namespace NUMINAMATH_CALUDE_democrat_ratio_l2166_216637

/-- Proves that the ratio of democrats to total participants is 1:3 given the specified conditions -/
theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 990 →
  female_democrats = 165 →
  (∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    2 * female_democrats = female_participants ∧
    4 * female_democrats = male_participants) →
  (3 : ℚ) * (female_democrats + female_democrats) = total_participants := by
  sorry


end NUMINAMATH_CALUDE_democrat_ratio_l2166_216637


namespace NUMINAMATH_CALUDE_recurrence_is_geometric_iff_first_two_equal_l2166_216646

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  (∀ n, b n > 0) ∧ (∀ n, b (n + 2) = 3 * b n * b (n + 1))

/-- A geometric progression -/
def IsGeometricProgression (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem recurrence_is_geometric_iff_first_two_equal
    (b : ℕ → ℝ) (h : RecurrenceSequence b) :
    IsGeometricProgression b ↔ b 1 = b 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_is_geometric_iff_first_two_equal_l2166_216646


namespace NUMINAMATH_CALUDE_intersection_M_N_l2166_216685

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

theorem intersection_M_N : M ∩ N = {(0, 1), (1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2166_216685


namespace NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_maximum_value_l2166_216643

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 1 = f' 1 * (x - 1) := by sorry

-- Theorem for the minimum value
theorem minimum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -14 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≥ f x := by sorry

-- Theorem for the maximum value
theorem maximum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 18 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≤ f x := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_maximum_value_l2166_216643


namespace NUMINAMATH_CALUDE_A_three_times_faster_than_B_l2166_216612

/-- The work rate of A -/
def work_rate_A : ℚ := 1 / 16

/-- The work rate of B -/
def work_rate_B : ℚ := 1 / 12 - 1 / 16

/-- The theorem stating that A is 3 times faster than B -/
theorem A_three_times_faster_than_B : work_rate_A / work_rate_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_A_three_times_faster_than_B_l2166_216612


namespace NUMINAMATH_CALUDE_point_coordinates_l2166_216694

/-- Given a point P in the Cartesian coordinate system, prove its coordinates. -/
theorem point_coordinates :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (-m - 1, 2 * m + 1)
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  P.2 = 5 →              -- Distance from M to x-axis is 5
  P = (-3, 5) :=         -- Coordinates of P are (-3, 5)
by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l2166_216694


namespace NUMINAMATH_CALUDE_repeated_two_digit_divisible_by_101_l2166_216653

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Constructs a four-digit number by repeating a two-digit number -/
def repeat_two_digit (n : TwoDigitNumber) : ℕ :=
  100 * n.val + n.val

theorem repeated_two_digit_divisible_by_101 (n : TwoDigitNumber) :
  (repeat_two_digit n) % 101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_repeated_two_digit_divisible_by_101_l2166_216653


namespace NUMINAMATH_CALUDE_intersection_A_B_l2166_216620

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ t : ℝ, x = t^2 + 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x * (x - 1) = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2166_216620


namespace NUMINAMATH_CALUDE_f_properties_l2166_216661

def f (x : ℝ) := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) ∧
  (∀ x, x > Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2166_216661


namespace NUMINAMATH_CALUDE_infinite_representable_elements_l2166_216698

def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) > a i

theorem infinite_representable_elements 
  (a : ℕ → ℕ) 
  (h_increasing : is_increasing_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧
    ∃ (x y h k : ℕ), 
      0 < h ∧ h < k ∧ k < m ∧
      a m = x * a h + y * a k :=
sorry

end NUMINAMATH_CALUDE_infinite_representable_elements_l2166_216698


namespace NUMINAMATH_CALUDE_strawberry_pies_l2166_216644

def christine_strawberries : ℕ := 10
def rachel_strawberries : ℕ := 2 * christine_strawberries
def strawberries_per_pie : ℕ := 3

theorem strawberry_pies : 
  (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_pies_l2166_216644


namespace NUMINAMATH_CALUDE_other_coin_denomination_l2166_216614

theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h1 : total_coins = 342)
  (h2 : total_value = 7100)  -- 71 Rs in paise
  (h3 : twenty_paise_coins = 290) :
  (total_value - 20 * twenty_paise_coins) / (total_coins - twenty_paise_coins) = 25 := by
sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l2166_216614


namespace NUMINAMATH_CALUDE_polynomial_equation_solutions_l2166_216678

-- Define the polynomials p and q
def p (x : ℂ) : ℂ := x^5 + x
def q (x : ℂ) : ℂ := x^5 + x^2

-- Define a primitive third root of unity
noncomputable def ε : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

-- Define the set of solution pairs
def solution_pairs : Set (ℂ × ℂ) :=
  {(ε, 1 - ε), (ε^2, 1 - ε^2), 
   ((1 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2)}

-- State the theorem
theorem polynomial_equation_solutions :
  ∀ w z : ℂ, w ≠ z → (p w = p z ∧ q w = q z) ↔ (w, z) ∈ solution_pairs :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solutions_l2166_216678


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l2166_216688

theorem largest_coefficient_binomial_expansion :
  ∃ (k : ℕ) (c : ℚ), 
    (k = 3 ∧ c = 160) ∧
    ∀ (j : ℕ) (d : ℚ), 
      (Nat.choose 6 j * (2 ^ j)) ≤ (Nat.choose 6 k * (2 ^ k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l2166_216688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2166_216654

theorem arithmetic_sequence_count (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 2.5 ∧ aₙ = 68.5 ∧ d = 6 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2166_216654


namespace NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2166_216639

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                 (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_weight : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 26 (50.25 : ℚ) 8 (45.15 : ℚ) = (49.05 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2166_216639


namespace NUMINAMATH_CALUDE_inequality_equivalence_f_less_than_one_l2166_216677

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I: Equivalence of the inequality
theorem inequality_equivalence (x : ℝ) : f x < x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

-- Part II: Prove f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_f_less_than_one_l2166_216677


namespace NUMINAMATH_CALUDE_correct_average_l2166_216640

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) 
  (misread1 misread2 misread3 : ℚ) 
  (correct1 correct2 correct3 : ℚ) :
  n = 15 ∧ 
  incorrect_avg = 62 ∧
  misread1 = 30 ∧ correct1 = 90 ∧
  misread2 = 60 ∧ correct2 = 120 ∧
  misread3 = 25 ∧ correct3 = 75 →
  (n : ℚ) * incorrect_avg + (correct1 - misread1) + (correct2 - misread2) + (correct3 - misread3) = n * (73 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2166_216640


namespace NUMINAMATH_CALUDE_exponent_division_l2166_216656

theorem exponent_division (a : ℝ) : 2 * a^3 / a = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2166_216656


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2166_216627

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_incr : increasing_sequence a)
  (h_first : a 1 = -2)
  (h_relation : ∀ n : ℕ, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q : ℝ, q = 1/3 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2166_216627


namespace NUMINAMATH_CALUDE_point_a_in_second_quadrant_l2166_216691

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The specific point we're considering -/
def point_a : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that point_a is in the second quadrant -/
theorem point_a_in_second_quadrant : in_second_quadrant point_a := by
  sorry

end NUMINAMATH_CALUDE_point_a_in_second_quadrant_l2166_216691


namespace NUMINAMATH_CALUDE_unique_triple_exists_l2166_216659

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem unique_triple_exists :
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 90 ∧
    Nat.lcm x z = 720 ∧
    Nat.lcm y z = 1000 ∧
    x < y ∧ y < z ∧
    (is_square x ∨ is_square y ∨ is_square z) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_exists_l2166_216659


namespace NUMINAMATH_CALUDE_chocolate_bar_weight_l2166_216668

/-- Proves that given a 2-kilogram box containing 16 chocolate bars, 
    each chocolate bar weighs 125 grams. -/
theorem chocolate_bar_weight :
  let box_weight_kg : ℕ := 2
  let bars_per_box : ℕ := 16
  let grams_per_kg : ℕ := 1000
  let box_weight_g : ℕ := box_weight_kg * grams_per_kg
  let bar_weight_g : ℕ := box_weight_g / bars_per_box
  bar_weight_g = 125 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_weight_l2166_216668


namespace NUMINAMATH_CALUDE_point_P_quadrants_l2166_216634

def is_root (x : ℝ) : Prop := (2 * x - 1) * (x + 1) = 0

def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_P_quadrants :
  ∃ (x y : ℝ), (is_root x ∧ is_root y) →
    (in_second_quadrant x y ∨ in_fourth_quadrant x y) ∧
    ¬(in_second_quadrant x y ∧ in_fourth_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_point_P_quadrants_l2166_216634


namespace NUMINAMATH_CALUDE_knight_probability_after_2023_moves_l2166_216652

/-- Knight's move on an infinite chessboard -/
def KnightMove (a b : ℤ) : Set (ℤ × ℤ) :=
  {(a+1, b+2), (a+1, b-2), (a-1, b+2), (a-1, b-2),
   (a+2, b+1), (a+2, b-1), (a-2, b+1), (a-2, b-1)}

/-- Probability space for knight's moves -/
def KnightProbSpace : Type := ℤ × ℤ

/-- Probability measure for knight's moves -/
noncomputable def KnightProb : KnightProbSpace → ℝ := sorry

/-- The set of positions (a, b) where a ≡ 4 (mod 8) and b ≡ 5 (mod 8) -/
def TargetPositions : Set (ℤ × ℤ) :=
  {(a, b) | a % 8 = 4 ∧ b % 8 = 5}

/-- The probability of the knight being at a target position after n moves -/
noncomputable def ProbAtTargetAfterMoves (n : ℕ) : ℝ := sorry

theorem knight_probability_after_2023_moves :
  ProbAtTargetAfterMoves 2023 = 1/32 - 1/2^2027 := by sorry

end NUMINAMATH_CALUDE_knight_probability_after_2023_moves_l2166_216652


namespace NUMINAMATH_CALUDE_smallest_even_with_repeated_seven_l2166_216645

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_repeated_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ p ^ k ∣ n

theorem smallest_even_with_repeated_seven :
  ∀ n : ℕ, 
    is_even n ∧ 
    has_repeated_prime_factor n 7 → 
    n ≥ 98 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_even_with_repeated_seven_l2166_216645


namespace NUMINAMATH_CALUDE_distance_between_points_l2166_216674

theorem distance_between_points : Real.sqrt ((24 - 0)^2 + (0 - 10)^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2166_216674


namespace NUMINAMATH_CALUDE_smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l2166_216630

/-- Predicate defining whether an integer m is in Row n -/
def in_row (n : ℕ) (m : ℕ) : Prop :=
  m % n = 0 ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row k m

theorem smallest_in_row_10 :
  ∀ m, in_row 10 m → m ≥ 10 :=
sorry

theorem n_squared_minus_n_and_2n_in_row (n : ℕ) (h : n ≥ 3) :
  in_row n (n^2 - n) ∧ in_row n (n^2 - 2*n) :=
sorry

theorem largest_n_not_including_n_squared_minus_10n :
  ∀ n > 9, in_row n (n^2 - 10*n) :=
sorry

end NUMINAMATH_CALUDE_smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l2166_216630


namespace NUMINAMATH_CALUDE_marts_income_percentage_l2166_216658

theorem marts_income_percentage (juan tim mart : ℝ) : 
  tim = 0.6 * juan →
  mart = 0.9599999999999999 * juan →
  (mart - tim) / tim * 100 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l2166_216658


namespace NUMINAMATH_CALUDE_asterisk_value_l2166_216617

theorem asterisk_value : ∃ x : ℚ, (63 / 21) * (x / 189) = 1 ∧ x = 63 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_value_l2166_216617


namespace NUMINAMATH_CALUDE_min_distinct_values_l2166_216613

/-- Given a list of 2023 positive integers with a unique mode occurring 15 times,
    the minimum number of distinct values is 146 -/
theorem min_distinct_values (l : List ℕ+) : 
  l.length = 2023 →
  ∃! m : ℕ+, (l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) →
  (∀ k : ℕ+, l.count k = 15 → k = m) →
  (Finset.card l.toFinset : ℕ) ≥ 146 ∧ 
  ∃ l' : List ℕ+, l'.length = 2023 ∧ 
    (∃! m' : ℕ+, (l'.count m' = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15)) ∧
    (Finset.card l'.toFinset : ℕ) = 146 :=
by
  sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2166_216613


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2166_216647

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2166_216647


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2166_216629

theorem quadratic_minimum : 
  (∃ (x : ℝ), x^2 + 12*x + 9 = -27) ∧ 
  (∀ (x : ℝ), x^2 + 12*x + 9 ≥ -27) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2166_216629


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l2166_216604

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2 ^ n) * 10^n % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l2166_216604


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2166_216651

theorem sqrt_product_equality : Real.sqrt 121 * Real.sqrt 49 * Real.sqrt 11 = 77 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2166_216651


namespace NUMINAMATH_CALUDE_lcm_of_231_and_300_l2166_216679

theorem lcm_of_231_and_300 (lcm hcf : ℕ) (a b : ℕ) : 
  hcf = 30 → a = 231 → b = 300 → lcm * hcf = a * b → lcm = 2310 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_231_and_300_l2166_216679


namespace NUMINAMATH_CALUDE_tangent_circles_m_values_l2166_216642

/-- Definition of circle C1 -/
def C1 (m x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9

/-- Definition of circle C2 -/
def C2 (m x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

/-- C1 is tangent to C2 from the inside -/
def is_tangent_inside (m : ℝ) : Prop :=
  ∃ x y : ℝ, C1 m x y ∧ C2 m x y ∧
  ∀ x' y' : ℝ, C1 m x' y' → C2 m x' y' → (x = x' ∧ y = y')

/-- The theorem to be proved -/
theorem tangent_circles_m_values :
  ∀ m : ℝ, is_tangent_inside m ↔ (m = -2 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_values_l2166_216642


namespace NUMINAMATH_CALUDE_special_parallelogram_perimeter_l2166_216671

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  /-- The length of the perpendicular from one vertex to the opposite side -/
  perpendicular : ℝ
  /-- The length of one diagonal -/
  diagonal : ℝ

/-- Theorem: The perimeter of a special parallelogram is 36 -/
theorem special_parallelogram_perimeter 
  (P : SpecialParallelogram) 
  (h1 : P.perpendicular = 12) 
  (h2 : P.diagonal = 15) : 
  Real.sqrt ((P.diagonal ^ 2 - P.perpendicular ^ 2) / 4) * 4 = 36 := by
  sorry

#check special_parallelogram_perimeter

end NUMINAMATH_CALUDE_special_parallelogram_perimeter_l2166_216671


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l2166_216690

noncomputable def f (x : ℝ) : ℝ := 16 * x^4 - 8 * x^3 + 9 * x^2 - 3 * x + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = 0.5 ∧ f r = 0 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l2166_216690


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2166_216609

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) :
  |x - y| = 2 * Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2166_216609


namespace NUMINAMATH_CALUDE_max_value_sqrt_quadratic_l2166_216655

theorem max_value_sqrt_quadratic :
  ∃ (max : ℝ), max = 9/2 ∧
  ∀ a : ℝ, -6 ≤ a ∧ a ≤ 3 →
    Real.sqrt ((3 - a) * (a + 6)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_quadratic_l2166_216655


namespace NUMINAMATH_CALUDE_complex_sum_equals_z_l2166_216623

theorem complex_sum_equals_z (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 = z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_z_l2166_216623


namespace NUMINAMATH_CALUDE_johnson_family_seating_theorem_l2166_216665

def johnson_family_seating (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) - (Nat.factorial n * Nat.factorial m)

theorem johnson_family_seating_theorem :
  johnson_family_seating 5 4 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_theorem_l2166_216665


namespace NUMINAMATH_CALUDE_sam_digits_of_pi_l2166_216692

theorem sam_digits_of_pi (carlos mina sam : ℕ) : 
  sam = carlos + 6 →
  mina = 6 * carlos →
  mina = 24 →
  sam = 10 := by sorry

end NUMINAMATH_CALUDE_sam_digits_of_pi_l2166_216692


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2166_216682

theorem ufo_convention_attendees :
  ∀ (total male female : ℕ),
    total = 120 →
    male = female + 4 →
    total = male + female →
    male = 62 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2166_216682


namespace NUMINAMATH_CALUDE_five_pq_is_odd_l2166_216603

theorem five_pq_is_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5 * p * q) := by
  sorry

end NUMINAMATH_CALUDE_five_pq_is_odd_l2166_216603


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l2166_216649

/-- Represents the percentage of salary saved -/
def P : ℝ := by sorry

theorem salary_savings_percentage :
  let S : ℝ := 20000  -- Monthly salary in Rs.
  let increase_factor : ℝ := 1.1  -- 10% increase in expenses
  let new_savings : ℝ := 200  -- New monthly savings in Rs.
  S - increase_factor * (S - P / 100 * S) = new_savings →
  P = 10 := by sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l2166_216649


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l2166_216632

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l2166_216632


namespace NUMINAMATH_CALUDE_circle_tangency_l2166_216666

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (m, 0) (0, 2) (|m|) 1 → m = 3/2 ∨ m = -3/2 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l2166_216666


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l2166_216635

theorem correct_mark_calculation (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 60 →
  wrong_mark = 90 →
  final_avg = 57.5 →
  (n : ℚ) * initial_avg - wrong_mark + ((n : ℚ) * final_avg - (n : ℚ) * initial_avg + wrong_mark) = 15 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l2166_216635


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2166_216648

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) : 
  total = 120 ∧ 
  red_freq = 15/100 ∧ 
  black_freq = 45/100 → 
  total - (red_freq * total).floor - (black_freq * total).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l2166_216648


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2166_216601

theorem largest_digit_divisible_by_six :
  ∀ M : ℕ, M ≤ 9 →
    (54320 + M).mod 6 = 0 →
    ∀ N : ℕ, N ≤ 9 → N > M →
      (54320 + N).mod 6 ≠ 0 →
    M = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2166_216601


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l2166_216626

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) : ℕ :=
  let cansUsedForChildren := (childrenFed + canCapacity.children - 1) / canCapacity.children
  let remainingCans := totalCans - cansUsedForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given 10 cans of soup, where each can feeds 4 adults or 6 children,
    if 30 children are fed, the remaining soup can feed 20 adults -/
theorem soup_feeding_theorem (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 6 →
  childrenFed = 30 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end NUMINAMATH_CALUDE_soup_feeding_theorem_l2166_216626


namespace NUMINAMATH_CALUDE_intersection_implies_value_l2166_216667

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, a^2+1, 2*a-1}
  (A ∩ B = {-3}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_value_l2166_216667


namespace NUMINAMATH_CALUDE_best_play_wins_probability_best_play_always_wins_more_than_two_plays_l2166_216607

/-- The probability that the best play wins in a competition with two plays -/
def probability_best_play_wins (n : ℕ) : ℚ :=
  1 - (n.factorial * n.factorial : ℚ) / ((2 * n).factorial : ℚ)

/-- The setup of the competition -/
structure Competition :=
  (n : ℕ)  -- number of students in each play
  (honest_mothers : ℕ)  -- number of mothers voting honestly
  (biased_mothers : ℕ)  -- number of mothers voting for their child's play

/-- The conditions of the competition -/
def competition_conditions (c : Competition) : Prop :=
  c.honest_mothers = c.n ∧ c.biased_mothers = c.n

/-- The theorem stating the probability of the best play winning -/
theorem best_play_wins_probability (c : Competition) 
  (h : competition_conditions c) : 
  probability_best_play_wins c.n = 1 - (c.n.factorial * c.n.factorial : ℚ) / ((2 * c.n).factorial : ℚ) :=
sorry

/-- For more than two plays, the best play always wins -/
theorem best_play_always_wins_more_than_two_plays (c : Competition) (s : ℕ) 
  (h1 : competition_conditions c) (h2 : s > 2) : 
  probability_best_play_wins c.n = 1 :=
sorry

end NUMINAMATH_CALUDE_best_play_wins_probability_best_play_always_wins_more_than_two_plays_l2166_216607


namespace NUMINAMATH_CALUDE_kaleb_chocolate_bars_l2166_216621

/-- The number of chocolate bars Kaleb needs to sell -/
def total_chocolate_bars (bars_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  bars_per_box * num_boxes

/-- Theorem stating the total number of chocolate bars Kaleb needs to sell -/
theorem kaleb_chocolate_bars :
  total_chocolate_bars 5 142 = 710 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_chocolate_bars_l2166_216621


namespace NUMINAMATH_CALUDE_smallest_fraction_divides_exactly_l2166_216628

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21
def smallestFraction : Rat := 1 / 42

theorem smallest_fraction_divides_exactly :
  (∃ (n1 n2 n3 : ℕ), fraction1 * n1 = smallestFraction ∧
                     fraction2 * n2 = smallestFraction ∧
                     fraction3 * n3 = smallestFraction) ∧
  (∀ (f : Rat), f > 0 ∧ (∃ (m1 m2 m3 : ℕ), fraction1 * m1 = f ∧
                                           fraction2 * m2 = f ∧
                                           fraction3 * m3 = f) →
                f ≥ smallestFraction) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_divides_exactly_l2166_216628


namespace NUMINAMATH_CALUDE_equation_proof_l2166_216695

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2166_216695


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2166_216636

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2166_216636


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2166_216602

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the point P
structure Point where
  x : Real
  y : Real

-- Define the theorem
theorem point_in_fourth_quadrant (abc : Triangle) (p : Point) :
  abc.A > Real.pi / 2 →  -- Angle A is obtuse
  p.x = Real.tan abc.B →  -- x-coordinate is tan B
  p.y = Real.cos abc.A →  -- y-coordinate is cos A
  p.x > 0 ∧ p.y < 0  -- Point is in fourth quadrant
:= by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2166_216602


namespace NUMINAMATH_CALUDE_f_shifted_l2166_216697

/-- Given a function f(x) = 3x - 5, prove that f(x - 4) = 3x - 17 for any real number x -/
theorem f_shifted (x : ℝ) : (fun x => 3 * x - 5) (x - 4) = 3 * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l2166_216697


namespace NUMINAMATH_CALUDE_a_plus_b_value_l2166_216676

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l2166_216676


namespace NUMINAMATH_CALUDE_fraction_17_39_415th_digit_l2166_216641

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def nth_digit (n d k : ℕ) : ℕ :=
  sorry

theorem fraction_17_39_415th_digit :
  nth_digit 17 39 415 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_17_39_415th_digit_l2166_216641


namespace NUMINAMATH_CALUDE_opposite_expressions_l2166_216657

theorem opposite_expressions (x : ℝ) : (x + 1) + (3 * x - 5) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l2166_216657


namespace NUMINAMATH_CALUDE_problem_solution_l2166_216610

theorem problem_solution : (3/4)^2017 * (-1-1/3)^2018 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2166_216610


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2166_216608

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (z : ℝ), z = x^2 + 3*x*y + 2*y^2 ∧ z ≤ 120 - 30*Real.sqrt 3 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 10 ∧
  x'^2 + 3*x'*y' + 2*y'^2 = 120 - 30*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2166_216608


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l2166_216633

theorem fixed_point_power_function (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 4 = 2 →
  f 16 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l2166_216633


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2166_216686

/-- A quadratic function with specific properties -/
def QuadraticFunction (m n : ℝ) : ℝ → ℝ := fun x ↦ m * x^2 - 2 * m * x + n + 1

/-- The derived function f based on g -/
def DerivedFunction (g : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ g x + (2 - a) * x

/-- The theorem statement -/
theorem quadratic_function_properties 
  (m n : ℝ) 
  (h_m : m > 0)
  (h_max : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≥ QuadraticFunction m n y)
  (h_min : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≤ QuadraticFunction m n y)
  (h_max_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 4)
  (h_min_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 0)
  : 
  (∀ x, QuadraticFunction m n x = x^2 - 2*x + 1) ∧
  (∃ a, (a = -5 ∨ a = 4) ∧ 
    (∃ x ∈ Set.Icc (-1) 2, ∀ y ∈ Set.Icc (-1) 2, 
      DerivedFunction (QuadraticFunction m n) a x ≤ DerivedFunction (QuadraticFunction m n) a y) ∧
    (∃ x ∈ Set.Icc (-1) 2, DerivedFunction (QuadraticFunction m n) a x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2166_216686


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2166_216611

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def complementary (t : Triangle) : Prop :=
  t.A + t.B = 90

def pythagorean (t : Triangle) : Prop :=
  (t.a + t.b) * (t.a - t.b) = t.c^2

def angle_ratio (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.A / t.C = 1

-- Theorem statement
theorem triangle_is_right_angled (t : Triangle) 
  (h1 : complementary t) 
  (h2 : pythagorean t) 
  (h3 : angle_ratio t) : 
  t.A = 45 ∧ t.B = 90 ∧ t.C = 45 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l2166_216611


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2166_216681

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2166_216681


namespace NUMINAMATH_CALUDE_bug_probability_after_six_steps_l2166_216669

/-- Represents a vertex of the tetrahedron -/
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

/-- The probability of the bug being at a given vertex after n steps -/
def prob_at_vertex (v : Vertex) (n : ℕ) : ℚ :=
  sorry

/-- The probability of the bug choosing a non-opposite vertex -/
def prob_non_opposite : ℚ := 1/2

/-- The probability of the bug choosing the opposite vertex -/
def prob_opposite : ℚ := 1/6

/-- The edge length of the tetrahedron -/
def edge_length : ℝ := 1

theorem bug_probability_after_six_steps :
  prob_at_vertex Vertex.A 6 = 53/324 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_six_steps_l2166_216669


namespace NUMINAMATH_CALUDE_abcd_power_2018_l2166_216680

theorem abcd_power_2018 (a b c d : ℝ) 
  (ha : (5 : ℝ) ^ a = 4)
  (hb : (4 : ℝ) ^ b = 3)
  (hc : (3 : ℝ) ^ c = 2)
  (hd : (2 : ℝ) ^ d = 5) :
  (a * b * c * d) ^ 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_abcd_power_2018_l2166_216680


namespace NUMINAMATH_CALUDE_equation_solution_l2166_216683

theorem equation_solution (x : ℝ) : 3 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2166_216683


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2166_216638

/-- The equation of a line passing through (3,4) and tangent to x^2 + y^2 = 25 is 3x + 4y - 25 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (x^2 + y^2 = 25) →  -- Circle equation
  ((3:ℝ)^2 + 4^2 = 25) →  -- Point (3,4) lies on the circle
  (∃ k : ℝ, y - 4 = k * (x - 3)) →  -- Line passes through (3,4)
  (∀ p : ℝ × ℝ, p.1^2 + p.2^2 = 25 → (3 * p.1 + 4 * p.2 - 25 = 0 → p = (3, 4))) →  -- Line touches circle at only one point
  (3 * x + 4 * y - 25 = 0) -- Equation of the tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2166_216638


namespace NUMINAMATH_CALUDE_scientific_notation_8350_l2166_216675

theorem scientific_notation_8350 : 
  8350 = 8.35 * (10 : ℝ)^3 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_8350_l2166_216675


namespace NUMINAMATH_CALUDE_five_digit_number_divisible_by_37_and_173_l2166_216600

theorem five_digit_number_divisible_by_37_and_173 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) →  -- five-digit number
  n % 37 = 0 →  -- divisible by 37
  n % 173 = 0 →  -- divisible by 173
  (n / 1000) % 10 = 3 →  -- thousands digit is 3
  (n / 100) % 10 = 2  -- hundreds digit is 2
  := by sorry

end NUMINAMATH_CALUDE_five_digit_number_divisible_by_37_and_173_l2166_216600


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l2166_216684

theorem fraction_power_simplification :
  9 * (1 / 7)^4 = 9 / 2401 :=
by sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l2166_216684


namespace NUMINAMATH_CALUDE_odd_function_sum_l2166_216660

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_sum : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2166_216660


namespace NUMINAMATH_CALUDE_mod_eight_congruence_l2166_216693

theorem mod_eight_congruence (m : ℕ) : 
  12^7 % 8 = m → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_congruence_l2166_216693


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2166_216615

/-- The coefficient of x² in the expansion of (2x³ + 5x² - 3x)(3x² - 5x + 1) -/
def coefficient_x_squared : ℤ := 20

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 2 * x^3 + 5 * x^2 - 3 * x

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 3 * x^2 - 5 * x + 1

theorem coefficient_x_squared_proof :
  ∃ (a b c d e f : ℚ),
    poly1 x * poly2 x = a * x^5 + b * x^4 + c * x^3 + coefficient_x_squared * x^2 + e * x + f :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2166_216615


namespace NUMINAMATH_CALUDE_morse_alphabet_size_l2166_216631

/-- The number of signals in each letter -/
def signal_length : Nat := 7

/-- The number of possible signals (dot and dash) -/
def signal_types : Nat := 2

/-- The number of possible alterations for each sequence (including the original) -/
def alterations_per_sequence : Nat := signal_length + 1

/-- The total number of possible sequences -/
def total_sequences : Nat := signal_types ^ signal_length

/-- The maximum number of unique letters in the alphabet -/
def max_letters : Nat := total_sequences / alterations_per_sequence

theorem morse_alphabet_size :
  max_letters = 16 := by sorry

end NUMINAMATH_CALUDE_morse_alphabet_size_l2166_216631


namespace NUMINAMATH_CALUDE_age_difference_l2166_216606

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2166_216606


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2166_216664

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2166_216664


namespace NUMINAMATH_CALUDE_inverse_composition_equals_one_third_l2166_216625

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Theorem statement
theorem inverse_composition_equals_one_third :
  g_inv (g_inv 11) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_one_third_l2166_216625


namespace NUMINAMATH_CALUDE_kimberly_skittles_bought_l2166_216619

/-- The number of Skittles Kimberly bought -/
def skittles_bought : ℕ := sorry

/-- Kimberly's initial number of Skittles -/
def initial_skittles : ℕ := 5

/-- Kimberly's total number of Skittles after buying more -/
def total_skittles : ℕ := 12

theorem kimberly_skittles_bought :
  skittles_bought = total_skittles - initial_skittles :=
sorry

end NUMINAMATH_CALUDE_kimberly_skittles_bought_l2166_216619


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2166_216672

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) : 
  A = Real.pi * r^2 → r = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2166_216672


namespace NUMINAMATH_CALUDE_quadratic_functions_theorem_l2166_216670

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Condition that a function is quadratic -/
def IsQuadratic (f : QuadraticFunction) : Prop := 
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def VertexX (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a function -/
def XIntercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_theorem 
  (f g : QuadraticFunction)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (h_relation : ∀ x, g x = -f (75 - x))
  (h_vertex : VertexX f ∈ XIntercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h_intercepts : {x₁, x₂, x₃, x₄} ⊆ XIntercepts f ∪ XIntercepts g)
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h_diff : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_theorem_l2166_216670


namespace NUMINAMATH_CALUDE_sally_initial_cards_l2166_216616

def initial_cards : ℕ := 27
def cards_from_dan : ℕ := 41
def cards_bought : ℕ := 20
def total_cards : ℕ := 88

theorem sally_initial_cards : 
  initial_cards + cards_from_dan + cards_bought = total_cards :=
by sorry

end NUMINAMATH_CALUDE_sally_initial_cards_l2166_216616


namespace NUMINAMATH_CALUDE_unique_solution_l2166_216624

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the conditions of the problem
structure DrawResult where
  alice : ℕ
  bob : ℕ
  alice_valid : valid_number alice
  bob_valid : valid_number bob
  alice_uncertain : ∀ n, valid_number n → n ≠ alice → (n < alice ∨ n > alice)
  bob_certain : bob < alice ∨ bob > alice
  bob_prime : is_prime bob
  product_multiple_of_10 : (alice * bob) % 10 = 0
  perfect_square : ∃ k : ℕ, 100 * bob + alice = k * k

-- Theorem statement
theorem unique_solution (d : DrawResult) : d.alice = 29 ∧ d.bob = 5 ∧ d.alice + d.bob = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2166_216624


namespace NUMINAMATH_CALUDE_sector_area_l2166_216605

/-- Given a circular sector with central angle 60° and arc length 4, its area is 24/π -/
theorem sector_area (r : ℝ) : 
  (π / 3 : ℝ) = 4 / r →   -- Central angle in radians = Arc length / radius
  (1 / 2) * r^2 * (π / 3) = 24 / π := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2166_216605


namespace NUMINAMATH_CALUDE_multiplicative_inverse_800_mod_7801_l2166_216687

theorem multiplicative_inverse_800_mod_7801 
  (h1 : 28^2 + 195^2 = 197^2) -- Pythagorean triple condition
  : ∃ n : ℕ, n < 7801 ∧ (800 * n) % 7801 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_800_mod_7801_l2166_216687


namespace NUMINAMATH_CALUDE_M_mod_1000_l2166_216663

/-- The number of distinguishable flagpoles -/
def num_flagpoles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 21

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 9

/-- The minimum number of flags required on each flagpole -/
def min_flags_per_pole : ℕ := 3

/-- The function to calculate the number of distinguishable arrangements -/
def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l2166_216663


namespace NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2166_216622

/-- A function that checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

/-- A function that reverses the digits of a number -/
def reverseDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that 10989 is the only five-digit number
    that when multiplied by 9, results in its reverse -/
theorem unique_reverse_multiple_of_nine :
  ∀ n : ℕ, isFiveDigit n → (9 * n = reverseDigits n) → n = 10989 :=
sorry

end NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2166_216622


namespace NUMINAMATH_CALUDE_coles_return_speed_l2166_216673

/-- Calculates the average speed of the return journey given the conditions of Cole's trip -/
theorem coles_return_speed (total_time : Real) (outbound_time : Real) (outbound_speed : Real) :
  total_time = 2 ∧ outbound_time = 72 / 60 ∧ outbound_speed = 70 →
  (2 * outbound_speed * outbound_time) / (total_time - outbound_time) = 105 := by
  sorry

#check coles_return_speed

end NUMINAMATH_CALUDE_coles_return_speed_l2166_216673


namespace NUMINAMATH_CALUDE_g_zero_value_l2166_216662

def f (x : ℝ) : ℝ := 2 * x + 3

theorem g_zero_value (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = f x) : g 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_value_l2166_216662


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2166_216650

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeating_decimal (n : ℕ) : ℚ :=
  n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeating_decimal (n : ℕ) : ℚ :=
  n / 99

/-- The sum of 0.3̄ and 0.0̄2̄ equals 35/99 -/
theorem sum_of_repeating_decimals :
  single_repeating_decimal 3 + double_repeating_decimal 2 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2166_216650


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_16x_l2166_216618

theorem factorization_xy_squared_minus_16x (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_16x_l2166_216618
