import Mathlib

namespace NUMINAMATH_CALUDE_find_x_l1456_145615

-- Define the # operation
def sharp (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

-- Theorem statement
theorem find_x : 
  ∃ (x : ℤ), 
    (∀ (p : ℤ), sharp (sharp (sharp p x) x) x = -4) ∧ 
    (sharp (sharp (sharp 18 x) x) x = -4) → 
    x = -21 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1456_145615


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l1456_145660

structure LetterSet where
  letters : Finset Char
  vowels : Finset Char
  vowels_subset : vowels ⊆ letters

def probability_no_vowel (s : LetterSet) : ℚ :=
  (s.letters.card - s.vowels.card : ℚ) / s.letters.card

def set1 : LetterSet := {
  letters := {'a', 'b', 'c', 'd', 'e'},
  vowels := {'a', 'e'},
  vowels_subset := by simp
}

def set2 : LetterSet := {
  letters := {'k', 'l', 'm', 'n', 'o', 'p'},
  vowels := ∅,
  vowels_subset := by simp
}

def set3 : LetterSet := {
  letters := {'r', 's', 't', 'u', 'v'},
  vowels := ∅,
  vowels_subset := by simp
}

def set4 : LetterSet := {
  letters := {'w', 'x', 'y', 'z', 'i'},
  vowels := {'i'},
  vowels_subset := by simp
}

theorem probability_at_least_one_vowel :
  1 - (probability_no_vowel set1 * probability_no_vowel set2 * 
       probability_no_vowel set3 * probability_no_vowel set4) = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l1456_145660


namespace NUMINAMATH_CALUDE_integer_product_condition_l1456_145655

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) :=
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l1456_145655


namespace NUMINAMATH_CALUDE_income_increase_is_fifty_percent_l1456_145657

/-- Represents the financial situation of a person over two years -/
structure FinancialData where
  income1 : ℝ
  savingsRate1 : ℝ
  incomeIncrease : ℝ

/-- The conditions of the problem -/
def problemConditions (d : FinancialData) : Prop :=
  d.savingsRate1 = 0.5 ∧
  d.income1 > 0 ∧
  d.incomeIncrease > 0 ∧
  let savings1 := d.savingsRate1 * d.income1
  let expenditure1 := d.income1 - savings1
  let income2 := d.income1 * (1 + d.incomeIncrease)
  let savings2 := 2 * savings1
  let expenditure2 := income2 - savings2
  expenditure1 + expenditure2 = 2 * expenditure1

/-- The theorem stating that under the given conditions, 
    the income increase in the second year is 50% -/
theorem income_increase_is_fifty_percent (d : FinancialData) :
  problemConditions d → d.incomeIncrease = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_income_increase_is_fifty_percent_l1456_145657


namespace NUMINAMATH_CALUDE_marble_196_is_green_l1456_145687

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 => MarbleColor.Red
  | 3 | 4 | 5 | 6 | 7 => MarbleColor.Green
  | _ => MarbleColor.Blue

/-- Theorem stating that the 196th marble is green -/
theorem marble_196_is_green : marbleColor 196 = MarbleColor.Green := by
  sorry


end NUMINAMATH_CALUDE_marble_196_is_green_l1456_145687


namespace NUMINAMATH_CALUDE_unique_solution_club_l1456_145656

/-- The ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

/-- Theorem stating that 21 is the unique solution to A ♣ 7 = 82 -/
theorem unique_solution_club : ∃! A : ℝ, club A 7 = 82 ∧ A = 21 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_club_l1456_145656


namespace NUMINAMATH_CALUDE_ediths_books_l1456_145625

/-- The total number of books Edith has, given the number of novels and their relation to writing books -/
theorem ediths_books (novels : ℕ) (writing_books : ℕ) 
  (h1 : novels = 80) 
  (h2 : novels = writing_books / 2) : 
  novels + writing_books = 240 := by
  sorry

end NUMINAMATH_CALUDE_ediths_books_l1456_145625


namespace NUMINAMATH_CALUDE_smallest_cube_ending_2016_l1456_145648

theorem smallest_cube_ending_2016 :
  ∃ (n : ℕ), n^3 % 10000 = 2016 ∧ ∀ (m : ℕ), m < n → m^3 % 10000 ≠ 2016 :=
by
  use 856
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_2016_l1456_145648


namespace NUMINAMATH_CALUDE_chocolates_on_square_perimeter_l1456_145658

/-- The number of chocolates on one side of the square -/
def chocolates_per_side : ℕ := 6

/-- The number of sides in a square -/
def sides_of_square : ℕ := 4

/-- The number of corners in a square -/
def corners_of_square : ℕ := 4

/-- The total number of chocolates around the perimeter of the square -/
def chocolates_on_perimeter : ℕ := chocolates_per_side * sides_of_square - corners_of_square

theorem chocolates_on_square_perimeter : chocolates_on_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_on_square_perimeter_l1456_145658


namespace NUMINAMATH_CALUDE_cube_root_of_three_cubed_l1456_145652

theorem cube_root_of_three_cubed (b : ℝ) : b^3 = 3 → b = 3^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_three_cubed_l1456_145652


namespace NUMINAMATH_CALUDE_nickel_chocolates_l1456_145611

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 10) 
  (h2 : robert = nickel + 5) : 
  nickel = 5 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l1456_145611


namespace NUMINAMATH_CALUDE_modulus_of_special_complex_l1456_145639

/-- The modulus of a complex number Z = 3a - 4ai where a < 0 is equal to -5a -/
theorem modulus_of_special_complex (a : ℝ) (ha : a < 0) :
  Complex.abs (Complex.mk (3 * a) (-4 * a)) = -5 * a := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_special_complex_l1456_145639


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1456_145604

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  (1.15 * l) * (1.25 * w) = 1.4375 * (l * w) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1456_145604


namespace NUMINAMATH_CALUDE_monomial_exponent_sum_l1456_145632

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

theorem monomial_exponent_sum (m n : ℕ) (h : like_terms m n) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_sum_l1456_145632


namespace NUMINAMATH_CALUDE_henry_kombucha_consumption_l1456_145624

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle in dollars -/
def bottle_cost : ℚ := 3

/-- The cash refund for each bottle in dollars -/
def bottle_refund : ℚ := 1/10

/-- The number of bottles Henry can buy with his yearly refund -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem henry_kombucha_consumption :
  bottles_per_month * bottle_refund * months_in_year = bottles_from_refund * bottle_cost :=
sorry

end NUMINAMATH_CALUDE_henry_kombucha_consumption_l1456_145624


namespace NUMINAMATH_CALUDE_square_difference_equality_l1456_145605

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1456_145605


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l1456_145619

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_prime (a x : ℝ) : ℝ := 3*x^2 - 2*a*x

-- Theorem statement
theorem tangent_line_inclination (a b : ℝ) :
  (f_prime a 1 = -1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l1456_145619


namespace NUMINAMATH_CALUDE_area_triangle_PF1F2_is_sqrt15_l1456_145654

/-- The ellipse with equation x²/9 + y²/5 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) + (p.2^2 / 5) = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse satisfying the given condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle PF₁F₂ is √15 -/
theorem area_triangle_PF1F2_is_sqrt15 
  (h_P_on_ellipse : P ∈ Ellipse)
  (h_PF1_eq_2PF2 : distance P F1 = 2 * distance P F2) :
  (1/2) * distance F1 F2 * Real.sqrt (distance P F1 ^ 2 - (distance F1 F2 / 2) ^ 2) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_PF1F2_is_sqrt15_l1456_145654


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1456_145671

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 - 12*x + a = (2*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1456_145671


namespace NUMINAMATH_CALUDE_log_properties_l1456_145650

theorem log_properties :
  (Real.log 5 + Real.log 2 = 1) ∧
  (Real.log 5 / Real.log 2 = Real.log 5 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_properties_l1456_145650


namespace NUMINAMATH_CALUDE_circle_center_line_segment_length_l1456_145699

/-- Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- Line l with equation x - y = 0 -/
def LineL (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of circle C is at (1, 1) -/
theorem circle_center : ∃ (x y : ℝ), CircleC x y ∧ x = 1 ∧ y = 1 :=
sorry

/-- The length of line segment AB, where A and B are intersection points of circle C and line l, is 2√2 -/
theorem line_segment_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧
    LineL x₁ y₁ ∧ LineL x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_circle_center_line_segment_length_l1456_145699


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1456_145600

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1456_145600


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l1456_145663

theorem digit_sum_puzzle :
  ∀ (A B : Nat),
    A < 10 →
    B < 10 →
    8000 + 100 * A + 10 * A + 4 - (100 * B + 10 * B + B) = 1000 * B + 100 * B + 10 * B + B →
    A = 8 ∧ B = 4 ∧ A + B = 12 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l1456_145663


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1456_145640

theorem min_value_quadratic_sum (x y : ℝ) (h : x + y = 1) :
  ∀ z w : ℝ, z + w = 1 → 2 * x^2 + 3 * y^2 ≤ 2 * z^2 + 3 * w^2 ∧
  ∃ a b : ℝ, a + b = 1 ∧ 2 * a^2 + 3 * b^2 = 6/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1456_145640


namespace NUMINAMATH_CALUDE_clothing_calculation_l1456_145674

/-- Calculates the remaining pieces of clothing after donations and disposal --/
def remaining_clothing (initial : ℕ) (first_donation : ℕ) (disposal : ℕ) : ℕ :=
  initial - (first_donation + 3 * first_donation) - disposal

/-- Theorem stating the remaining pieces of clothing --/
theorem clothing_calculation :
  remaining_clothing 100 5 15 = 65 := by
  sorry

end NUMINAMATH_CALUDE_clothing_calculation_l1456_145674


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1456_145653

theorem rationalize_denominator : 
  ∃ (A B C D E F G H I : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11) = 
      (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + G * Real.sqrt H) / I ∧
    I > 0 ∧
    A = -6 ∧ B = 5 ∧ C = -8 ∧ D = 3 ∧ E = 3 ∧ F = 11 ∧ G = 1 ∧ H = 165 ∧ I = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1456_145653


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1456_145634

/-- A rational point in the Cartesian plane -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- A coloring function that assigns colors to rational points -/
def ColoringFunction (n : ℕ) := RationalPoint → Fin n

/-- Predicate to check if a point lies on a line segment between two other points -/
def OnLineSegment (A B P : RationalPoint) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
    P.x = A.x + t * (B.x - A.x) ∧
    P.y = A.y + t * (B.y - A.y)

/-- Main theorem: For any positive n, there exists a coloring function such that
    any line segment between two rational points contains all n colors -/
theorem exists_valid_coloring (n : ℕ) (h : 0 < n) :
  ∃ f : ColoringFunction n,
    ∀ A B : RationalPoint, A ≠ B →
      ∀ c : Fin n, ∃ P : RationalPoint,
        OnLineSegment A B P ∧ f P = c :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1456_145634


namespace NUMINAMATH_CALUDE_coin_distribution_l1456_145694

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- Total 5 coins
  a + b = c + d + e →  -- Sum condition
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d →  -- Arithmetic sequence
  e = 2/3 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l1456_145694


namespace NUMINAMATH_CALUDE_train_length_l1456_145693

/-- Given a train that crosses a signal post in 40 seconds and takes 600 seconds
    to cross a 9000-meter long bridge at a constant speed, prove that the length
    of the train is 642.857142857... meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 600 →
  bridge_length = 9000 →
  ∃ (train_length : ℝ), train_length = 360000 / 560 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1456_145693


namespace NUMINAMATH_CALUDE_factor_expression_l1456_145690

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1456_145690


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l1456_145684

def nickel_value : ℚ := 5
def quarter_value : ℚ := 25
def half_dollar_value : ℚ := 50

def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 40
def num_quarters : ℕ := 30

def total_value : ℚ := 
  num_nickels * nickel_value + 
  num_half_dollars * half_dollar_value + 
  num_quarters * quarter_value

def half_dollar_total : ℚ := num_half_dollars * half_dollar_value

theorem half_dollar_percentage : 
  (half_dollar_total / total_value) * 100 = 64 := by sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l1456_145684


namespace NUMINAMATH_CALUDE_detect_non_conforming_probability_l1456_145642

/-- The number of cans in a box -/
def total_cans : ℕ := 5

/-- The number of non-conforming cans in the box -/
def non_conforming_cans : ℕ := 2

/-- The number of cans selected for testing -/
def selected_cans : ℕ := 2

/-- The probability of detecting at least one non-conforming product -/
def probability_detect : ℚ := 7 / 10

theorem detect_non_conforming_probability :
  probability_detect = (Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) 1 + 
                        Nat.choose non_conforming_cans 2) / 
                       Nat.choose total_cans selected_cans :=
by sorry

end NUMINAMATH_CALUDE_detect_non_conforming_probability_l1456_145642


namespace NUMINAMATH_CALUDE_triangle_table_height_l1456_145637

theorem triangle_table_height (a b c : ℝ) (h_a : a = 25) (h_b : b = 31) (h_c : c = 34) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_max := 2 * area / (a + b + c)
  h_max = 4 * Real.sqrt 231 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_table_height_l1456_145637


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1456_145630

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1456_145630


namespace NUMINAMATH_CALUDE_decimal_sum_as_fraction_l1456_145682

theorem decimal_sum_as_fraction : 
  (0.2 + 0.03 + 0.004 + 0.0006 + 0.00007 + 0.000008 + 0.0000009 : ℚ) = 2340087/10000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_as_fraction_l1456_145682


namespace NUMINAMATH_CALUDE_expand_expression_l1456_145689

theorem expand_expression (x y : ℝ) : (3*x - 15) * (4*y + 20) = 12*x*y + 60*x - 60*y - 300 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1456_145689


namespace NUMINAMATH_CALUDE_triangle_ratio_equality_l1456_145696

/-- Given a triangle ABC with sides a, b, c, height ha corresponding to side a,
    and inscribed circle radius r, prove that (a + b + c) / a = ha / r -/
theorem triangle_ratio_equality (a b c ha r : ℝ) (ha_pos : ha > 0) (r_pos : r > 0) 
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) : (a + b + c) / a = ha / r :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equality_l1456_145696


namespace NUMINAMATH_CALUDE_farm_ploughing_problem_l1456_145638

/-- Calculates the actual ploughing rate given the conditions of the farm problem -/
def actualPloughingRate (totalArea plannedRate extraDays unploughedArea : ℕ) : ℕ :=
  let plannedDays := totalArea / plannedRate
  let actualDays := plannedDays + extraDays
  let ploughedArea := totalArea - unploughedArea
  ploughedArea / actualDays

/-- Theorem stating the actual ploughing rate for the given farm problem -/
theorem farm_ploughing_problem :
  actualPloughingRate 3780 90 2 40 = 85 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_problem_l1456_145638


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l1456_145647

theorem major_premise_incorrect : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l1456_145647


namespace NUMINAMATH_CALUDE_associated_equation_part1_associated_equation_part2_associated_equation_part3_l1456_145623

-- Part 1
theorem associated_equation_part1 (x : ℝ) : 
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) → 
  (x - (3 * x + 1) = -5) :=
sorry

-- Part 2
theorem associated_equation_part2 (x : ℤ) : 
  (x - 1/4 < 1 ∧ 4 + 2 * x > -7 * x + 5) → 
  (x - 1 = 0) :=
sorry

-- Part 3
theorem associated_equation_part3 (m : ℝ) : 
  (∀ x : ℝ, (x < 2 * x - m ∧ x - 2 ≤ m) → 
  (2 * x - 1 = x + 2 ∨ 3 + x = 2 * (x + 1/2))) → 
  (1 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_associated_equation_part1_associated_equation_part2_associated_equation_part3_l1456_145623


namespace NUMINAMATH_CALUDE_mean_median_difference_is_03_l1456_145673

/-- Represents a frequency histogram bin -/
structure HistogramBin where
  lowerBound : ℝ
  upperBound : ℝ
  frequency : ℕ

/-- Calculates the median of a dataset represented by a frequency histogram -/
def calculateMedian (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

/-- Calculates the mean of a dataset represented by a frequency histogram -/
def calculateMean (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

theorem mean_median_difference_is_03 (histogram : List HistogramBin) : 
  let totalStudents := 20
  let h := [
    ⟨0, 1, 4⟩,
    ⟨2, 3, 2⟩,
    ⟨4, 5, 6⟩,
    ⟨6, 7, 3⟩,
    ⟨8, 9, 5⟩
  ]
  calculateMean h totalStudents - calculateMedian h totalStudents = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_is_03_l1456_145673


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l1456_145636

/-- Gravitational force function -/
noncomputable def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / d^2

theorem gravitational_force_at_distance
  (k : ℝ)
  (h1 : gravitational_force k 5000 = 500)
  (h2 : k > 0) :
  gravitational_force k 25000 = 1/5 := by
  sorry

#check gravitational_force_at_distance

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l1456_145636


namespace NUMINAMATH_CALUDE_tara_spent_more_on_ice_cream_l1456_145628

/-- The amount Tara spent more on ice cream than on yogurt -/
def ice_cream_yogurt_difference : ℕ :=
  let ice_cream_cartons : ℕ := 19
  let yogurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yogurt_price : ℕ := 1
  (ice_cream_cartons * ice_cream_price) - (yogurt_cartons * yogurt_price)

/-- Theorem stating that Tara spent $129 more on ice cream than on yogurt -/
theorem tara_spent_more_on_ice_cream : ice_cream_yogurt_difference = 129 := by
  sorry

end NUMINAMATH_CALUDE_tara_spent_more_on_ice_cream_l1456_145628


namespace NUMINAMATH_CALUDE_cube_cutting_l1456_145627

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l1456_145627


namespace NUMINAMATH_CALUDE_exists_prime_number_of_ones_l1456_145644

/-- A number consisting of q ones in decimal notation -/
def number_of_ones (q : ℕ) : ℕ := (10^q - 1) / 9

/-- Theorem stating that there exists a natural number k such that
    a number consisting of (6k-1) ones is prime -/
theorem exists_prime_number_of_ones :
  ∃ k : ℕ, Nat.Prime (number_of_ones (6*k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_number_of_ones_l1456_145644


namespace NUMINAMATH_CALUDE_seashell_collection_l1456_145629

/-- The number of seashells collected by Stefan, Vail, and Aiguo -/
theorem seashell_collection (stefan vail aiguo : ℕ) 
  (h1 : stefan = vail + 16)
  (h2 : vail = aiguo - 5)
  (h3 : aiguo = 20) :
  stefan + vail + aiguo = 66 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l1456_145629


namespace NUMINAMATH_CALUDE_lawn_mowing_l1456_145613

theorem lawn_mowing (mary_rate tom_rate : ℚ)
  (h1 : mary_rate = 1 / 3)
  (h2 : tom_rate = 1 / 6)
  (total_lawn : ℚ)
  (h3 : total_lawn = 1) :
  let combined_rate := mary_rate + tom_rate
  let mowed_together := combined_rate * 1
  let mowed_mary_alone := mary_rate * 1
  let total_mowed := mowed_together + mowed_mary_alone
  total_lawn - total_mowed = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_l1456_145613


namespace NUMINAMATH_CALUDE_not_all_same_graph_l1456_145672

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x - 2
def equation_II (x y : ℝ) : Prop := y = (x^2 - 4) / (x + 2)
def equation_III (x y : ℝ) : Prop := (x + 2) * y = x^2 - 4

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem stating that the three equations do not all represent the same graph
theorem not_all_same_graph :
  ¬(same_graph equation_I equation_II ∧ same_graph equation_II equation_III ∧ same_graph equation_I equation_III) :=
by sorry

end NUMINAMATH_CALUDE_not_all_same_graph_l1456_145672


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1456_145683

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, 2^n + 65 = m^2) ↔ (n = 10 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1456_145683


namespace NUMINAMATH_CALUDE_female_democrats_count_l1456_145681

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 720 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1456_145681


namespace NUMINAMATH_CALUDE_probability_two_from_ten_with_two_defective_l1456_145691

/-- The probability of drawing at least one defective product -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (draw : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) draw : ℚ) / (Nat.choose total draw : ℚ)

/-- Theorem stating the probability of drawing at least one defective product -/
theorem probability_two_from_ten_with_two_defective :
  probability_at_least_one_defective 10 2 2 = 17 / 45 := by
sorry

end NUMINAMATH_CALUDE_probability_two_from_ten_with_two_defective_l1456_145691


namespace NUMINAMATH_CALUDE_cos_equality_implies_45_l1456_145664

theorem cos_equality_implies_45 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (315 * π / 180) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_45_l1456_145664


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1456_145698

/-- A rectangle with integer dimensions satisfying the given condition has a perimeter of 26. -/
theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- not a square
  4 * (a + b) - a * b = 12 →  -- twice perimeter minus area equals 12
  2 * (a + b) = 26 := by  -- perimeter equals 26
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1456_145698


namespace NUMINAMATH_CALUDE_average_book_price_l1456_145620

theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 65 →
  books2 = 55 →
  price1 = 1280 →
  price2 = 880 →
  (price1 + price2) / (books1 + books2 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_book_price_l1456_145620


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1456_145617

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3 →
  x ≥ 40.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1456_145617


namespace NUMINAMATH_CALUDE_unique_solution_l1456_145646

theorem unique_solution : ∃! x : ℝ, x^29 * 4^15 = 2 * 10^29 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1456_145646


namespace NUMINAMATH_CALUDE_coprime_in_ten_consecutive_integers_l1456_145695

theorem coprime_in_ten_consecutive_integers (k : ℤ) :
  ∃ n ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ n → Int.gcd (k + n) (k + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_in_ten_consecutive_integers_l1456_145695


namespace NUMINAMATH_CALUDE_arrow_pointing_theorem_l1456_145602

/-- Represents the direction of an arrow -/
inductive Direction
| Left
| Right

/-- Represents an arrangement of arrows -/
def ArrowArrangement (n : ℕ) := Fin n → Direction

/-- The number of arrows pointing to the i-th arrow -/
def pointingTo (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

/-- The number of arrows that the i-th arrow is pointing to -/
def pointingFrom (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

theorem arrow_pointing_theorem (n : ℕ) (h : Odd n) (h1 : n ≥ 1) (arr : ArrowArrangement n) :
  ∃ i : Fin n, pointingTo arr i = pointingFrom arr i := by sorry

end NUMINAMATH_CALUDE_arrow_pointing_theorem_l1456_145602


namespace NUMINAMATH_CALUDE_probability_three_ones_two_twos_l1456_145606

def roll_probability : ℕ → ℚ
  | 1 => 1/6
  | 2 => 1/6
  | _ => 2/3

def num_rolls : ℕ := 5

theorem probability_three_ones_two_twos :
  (Nat.choose num_rolls 3) * (roll_probability 1)^3 * (roll_probability 2)^2 = 5/3888 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_ones_two_twos_l1456_145606


namespace NUMINAMATH_CALUDE_max_ab_value_l1456_145659

theorem max_ab_value (a b c : ℝ) : 
  (∀ x : ℝ, 2*x + 2 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 2*x^2 - 2*x + 4) →
  a*b ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l1456_145659


namespace NUMINAMATH_CALUDE_combined_area_rhombus_circle_l1456_145616

/-- The combined area of a rhombus and a circle -/
theorem combined_area_rhombus_circle (d1 d2 r : ℝ) (h1 : d1 = 40) (h2 : d2 = 30) (h3 : r = 10) :
  (d1 * d2 / 2) + (π * r^2) = 600 + 100 * π := by
  sorry

end NUMINAMATH_CALUDE_combined_area_rhombus_circle_l1456_145616


namespace NUMINAMATH_CALUDE_x_cube_plus_four_x_equals_eight_l1456_145612

theorem x_cube_plus_four_x_equals_eight (x : ℝ) (h : x^3 + 4*x = 8) :
  x^7 + 64*x^2 = 128 := by
sorry

end NUMINAMATH_CALUDE_x_cube_plus_four_x_equals_eight_l1456_145612


namespace NUMINAMATH_CALUDE_additional_students_score_l1456_145635

/-- Given a class with the following properties:
  * There are 17 students in total
  * The average grade of 15 students is 85
  * After including two more students, the new average becomes 87
  This theorem proves that the combined score of the two additional students is 204. -/
theorem additional_students_score (total_students : ℕ) (initial_students : ℕ) 
  (initial_average : ℝ) (final_average : ℝ) : 
  total_students = 17 → 
  initial_students = 15 → 
  initial_average = 85 → 
  final_average = 87 → 
  (total_students * final_average - initial_students * initial_average : ℝ) = 204 := by
  sorry

#check additional_students_score

end NUMINAMATH_CALUDE_additional_students_score_l1456_145635


namespace NUMINAMATH_CALUDE_nearest_significant_place_is_ten_thousands_l1456_145622

/-- Represents the place value of a digit in a number -/
inductive DigitPlace
  | Hundreds
  | Thousands
  | TenThousands

/-- The given number -/
def givenNumber : ℝ := 3.02 * 10^6

/-- Determines if a digit place is significant for a given number -/
def isSignificantPlace (n : ℝ) (place : DigitPlace) : Prop :=
  match place with
  | DigitPlace.Hundreds => n ≥ 100 ∧ n < 1000
  | DigitPlace.Thousands => n ≥ 1000 ∧ n < 10000
  | DigitPlace.TenThousands => n ≥ 10000 ∧ n < 100000

/-- The nearest significant digit place for the given number -/
def nearestSignificantPlace : DigitPlace := DigitPlace.TenThousands

theorem nearest_significant_place_is_ten_thousands :
  isSignificantPlace givenNumber nearestSignificantPlace :=
by sorry

end NUMINAMATH_CALUDE_nearest_significant_place_is_ten_thousands_l1456_145622


namespace NUMINAMATH_CALUDE_complex_modulus_l1456_145633

theorem complex_modulus (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (1 + a * i) * i = 2 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l1456_145633


namespace NUMINAMATH_CALUDE_prop_q_not_necessary_nor_sufficient_l1456_145643

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def PropP (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- Proposition Q: The coefficients of two quadratic expressions are proportional -/
def PropQ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂

theorem prop_q_not_necessary_nor_sufficient :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropQ a₁ b₁ c₁ a₂ b₂ c₂ → PropP a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropP a₁ b₁ c₁ a₂ b₂ c₂ → PropQ a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_prop_q_not_necessary_nor_sufficient_l1456_145643


namespace NUMINAMATH_CALUDE_division_remainder_l1456_145692

theorem division_remainder : ∀ (dividend divisor quotient : ℕ),
  dividend = 166 →
  divisor = 18 →
  quotient = 9 →
  dividend = divisor * quotient + 4 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l1456_145692


namespace NUMINAMATH_CALUDE_circle_and_trajectory_l1456_145603

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x - y + 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, -2)

-- Define point D
def D : ℝ × ℝ := (4, 3)

theorem circle_and_trajectory :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ Line ∧
    A ∈ Circle C r ∧
    B ∈ Circle C r ∧
    (∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ (x, y) ∈ Circle C r) ∧
    (∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
      ∃ (E : ℝ × ℝ), E ∈ Circle C r ∧ (x, y) = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_trajectory_l1456_145603


namespace NUMINAMATH_CALUDE_correct_operation_l1456_145685

theorem correct_operation (a b : ℝ) : 4 * a^2 * b - 2 * b * a^2 = 2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1456_145685


namespace NUMINAMATH_CALUDE_forty_five_candies_cost_candies_for_fifty_l1456_145697

-- Define the cost of one candy in rubles
def cost_per_candy : ℝ := 1

-- Define the relationship between 45 candies and their cost
theorem forty_five_candies_cost (c : ℝ) : c * 45 = 45 := by sorry

-- Define the number of candies that can be bought for 20 rubles
def candies_for_twenty : ℝ := 20

-- Theorem to prove
theorem candies_for_fifty : ℝ := by
  -- The number of candies that can be bought for 50 rubles is 50
  exact 50

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_forty_five_candies_cost_candies_for_fifty_l1456_145697


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1456_145610

theorem opposite_of_negative_two : (-(- 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1456_145610


namespace NUMINAMATH_CALUDE_system_solution_l1456_145614

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 20 ∧ 9 * x - 8 * y = 36 ∧ x = 76 / 15 ∧ y = 18 / 15 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1456_145614


namespace NUMINAMATH_CALUDE_problem_statement_l1456_145676

theorem problem_statement (x y : ℝ) (h1 : 2*x + 5*y = 10) (h2 : x*y = -10) :
  4*x^2 + 25*y^2 = 300 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1456_145676


namespace NUMINAMATH_CALUDE_gwen_recycling_problem_l1456_145662

/-- Represents the problem of calculating unrecycled bags. -/
def unrecycled_bags_problem (total_bags : ℕ) (points_per_bag : ℕ) (total_points : ℕ) : Prop :=
  let recycled_bags := total_points / points_per_bag
  total_bags - recycled_bags = 2

/-- Theorem stating the solution to Gwen's recycling problem. -/
theorem gwen_recycling_problem :
  unrecycled_bags_problem 4 8 16 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycling_problem_l1456_145662


namespace NUMINAMATH_CALUDE_correct_number_value_l1456_145666

/-- Given 10 numbers with an initial average of 21, where one number was wrongly read as 26,
    and the correct average is 22, prove that the correct value of the wrongly read number is 36. -/
theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 ∧ 
  initial_avg = 21 ∧ 
  correct_avg = 22 ∧ 
  wrong_value = 26 →
  ∃ (correct_value : ℚ), 
    n * correct_avg - (n * initial_avg - wrong_value) = correct_value ∧
    correct_value = 36 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_value_l1456_145666


namespace NUMINAMATH_CALUDE_select_three_from_seven_eq_210_l1456_145669

/-- The number of ways to select 3 distinct individuals from a group of 7 people to fill 3 distinct positions. -/
def select_three_from_seven : ℕ :=
  7 * 6 * 5

/-- Theorem stating that selecting 3 distinct individuals from a group of 7 people to fill 3 distinct positions can be done in 210 ways. -/
theorem select_three_from_seven_eq_210 :
  select_three_from_seven = 210 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_seven_eq_210_l1456_145669


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l1456_145688

theorem xiaolis_estimate (x y z w : ℝ) (hx : x > y) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  (x + z) - (y - w) > x - y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l1456_145688


namespace NUMINAMATH_CALUDE_specific_kite_area_l1456_145686

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite shape -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a square -/
structure Square where
  center : Point
  sideLength : ℝ

/-- Calculates the area of a kite with an internal square -/
def kiteArea (k : Kite) (s : Square) : ℝ :=
  sorry

/-- The theorem stating the area of the specific kite -/
theorem specific_kite_area :
  let k : Kite := {
    v1 := {x := 1, y := 6},
    v2 := {x := 4, y := 7},
    v3 := {x := 7, y := 6},
    v4 := {x := 4, y := 0}
  }
  let s : Square := {
    center := {x := 4, y := 3},
    sideLength := 2
  }
  kiteArea k s = 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_kite_area_l1456_145686


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l1456_145667

theorem mod_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 8173 [ZMOD 15] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l1456_145667


namespace NUMINAMATH_CALUDE_find_number_l1456_145609

theorem find_number (A B : ℕ+) (h1 : B = 286) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 26) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1456_145609


namespace NUMINAMATH_CALUDE_intersection_of_D_sets_nonempty_l1456_145621

def D (n : ℕ) : Set ℕ :=
  {x | ∃ a b : ℕ, a * b = n ∧ a > b ∧ b > 0 ∧ x = a - b}

theorem intersection_of_D_sets_nonempty (k : ℕ) (hk : k > 1) :
  ∃ (n : Fin k → ℕ), (∀ i, n i > 1) ∧ 
  (∀ i j, i ≠ j → n i ≠ n j) ∧
  (∃ x y : ℕ, x ≠ y ∧ ∀ i, x ∈ D (n i) ∧ y ∈ D (n i)) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_D_sets_nonempty_l1456_145621


namespace NUMINAMATH_CALUDE_power_equation_solution_l1456_145645

theorem power_equation_solution (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1456_145645


namespace NUMINAMATH_CALUDE_debate_tournament_participants_l1456_145661

theorem debate_tournament_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end NUMINAMATH_CALUDE_debate_tournament_participants_l1456_145661


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1456_145626

theorem algebraic_expression_value (a b : ℝ) (h : 3 * a * b - 3 * b^2 - 2 = 0) :
  (1 - (2 * a * b - b^2) / a^2) / ((a - b) / (a^2 * b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1456_145626


namespace NUMINAMATH_CALUDE_inequality_hold_l1456_145678

theorem inequality_hold (x : ℝ) : 
  x ≥ -1/2 → x ≠ 0 → 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
   (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 24)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_hold_l1456_145678


namespace NUMINAMATH_CALUDE_recipe_change_l1456_145677

/-- Represents the recipe for the apple-grape drink -/
structure Recipe where
  apple_proportion : ℚ  -- Proportion of an apple juice container used per can
  grape_proportion : ℚ  -- Proportion of a grape juice container used per can

/-- The total volume of juice per can -/
def total_volume (r : Recipe) : ℚ :=
  r.apple_proportion + r.grape_proportion

theorem recipe_change (old_recipe new_recipe : Recipe) :
  old_recipe.apple_proportion = 1/6 →
  old_recipe.grape_proportion = 1/10 →
  new_recipe.apple_proportion = 1/5 →
  total_volume old_recipe = total_volume new_recipe →
  new_recipe.grape_proportion = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_recipe_change_l1456_145677


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1456_145618

theorem rectangular_prism_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f) = 720 →
  a + b + c + d + e + f = 27 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1456_145618


namespace NUMINAMATH_CALUDE_max_t_is_one_l1456_145679

/-- The function f(x) = x^2 - ax + a - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a - 1

/-- The theorem stating that the maximum value of t is 1 -/
theorem max_t_is_one (t : ℝ) :
  (∀ a ∈ Set.Ioo 0 4, ∃ x₀ ∈ Set.Icc 0 2, t ≤ |f a x₀|) →
  t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_t_is_one_l1456_145679


namespace NUMINAMATH_CALUDE_cork_mass_proof_l1456_145631

/-- The density of platinum in kg/m^3 -/
def platinum_density : ℝ := 2.15e4

/-- The density of cork wood in kg/m^3 -/
def cork_density : ℝ := 2.4e2

/-- The density of the combined system in kg/m^3 -/
def system_density : ℝ := 4.8e2

/-- The mass of the piece of platinum in kg -/
def platinum_mass : ℝ := 86.94

/-- The mass of the piece of cork wood in kg -/
def cork_mass : ℝ := 85

theorem cork_mass_proof :
  ∃ (cork_volume platinum_volume : ℝ),
    cork_volume > 0 ∧
    platinum_volume > 0 ∧
    cork_density = cork_mass / cork_volume ∧
    platinum_density = platinum_mass / platinum_volume ∧
    system_density = (cork_mass + platinum_mass) / (cork_volume + platinum_volume) :=
by sorry

end NUMINAMATH_CALUDE_cork_mass_proof_l1456_145631


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1456_145607

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1456_145607


namespace NUMINAMATH_CALUDE_special_line_equation_l1456_145680

/-- A line passing through the point (3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ
  /-- The line passes through the point (3, -4) -/
  passes_through_point : equation 3 (-4) 0 = 0
  /-- The x-intercept and y-intercept are opposite numbers -/
  opposite_intercepts : ∃ (a : ℝ), equation a 0 0 = 0 ∧ equation 0 (-a) 0 = 0

/-- The equation of the special line is either 4x + 3y = 0 or x - y - 7 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, l.equation x y 0 = 4*x + 3*y) ∨
  (∀ x y, l.equation x y 0 = x - y - 7) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1456_145680


namespace NUMINAMATH_CALUDE_expression_equality_l1456_145651

theorem expression_equality : (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1456_145651


namespace NUMINAMATH_CALUDE_games_given_away_l1456_145601

def initial_games : ℕ := 183
def remaining_games : ℕ := 92

theorem games_given_away : initial_games - remaining_games = 91 := by
  sorry

end NUMINAMATH_CALUDE_games_given_away_l1456_145601


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_difference_sum_of_roots_specific_equation_l1456_145675

theorem sum_of_roots_squared_difference (a c : ℝ) :
  let f := fun x : ℝ => (x - a)^2 - c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 2*a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => (x - 5)^2 - 9
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_difference_sum_of_roots_specific_equation_l1456_145675


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1456_145665

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define point W on side XZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
axiom XW_length : ∀ t : Triangle, dist (t.X) (W t) = 9
axiom WZ_length : ∀ t : Triangle, dist (W t) (t.Z) = 15

-- Define the areas of triangles XYW and WYZ
def area_XYW (t : Triangle) : ℝ := sorry
def area_WYZ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) :
  (area_XYW t) / (area_WYZ t) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1456_145665


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1456_145668

theorem simplify_fraction_product : 15 * (18 / 5) * (-42 / 45) = -50.4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1456_145668


namespace NUMINAMATH_CALUDE_square_angle_problem_l1456_145649

/-- In a square ABCD with a segment CE, if two angles formed are 7α and 8α, then α = 9°. -/
theorem square_angle_problem (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_angle_problem_l1456_145649


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1456_145641

/-- The function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The monotonic increasing interval of f(x) = -x^2 + 2x - 2 is (-∞, 1) -/
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x < y → y ≤ 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1456_145641


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1456_145670

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1456_145670


namespace NUMINAMATH_CALUDE_distinct_colorings_tetrahedron_l1456_145608

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the symmetry group size of a tetrahedron
def symmetry_group_size : ℕ := 12

-- Define the number of vertices in a tetrahedron
def num_vertices : ℕ := 4

-- Define the number of colorings fixed by identity rotation
def fixed_by_identity : ℕ := num_colors ^ num_vertices

-- Define the number of colorings fixed by 180° rotations
def fixed_by_180_rotation : ℕ := num_colors ^ 2

-- Define the number of 180° rotations
def num_180_rotations : ℕ := 3

-- Theorem statement
theorem distinct_colorings_tetrahedron :
  (fixed_by_identity + num_180_rotations * fixed_by_180_rotation) / symmetry_group_size = 36 :=
by sorry

end NUMINAMATH_CALUDE_distinct_colorings_tetrahedron_l1456_145608
