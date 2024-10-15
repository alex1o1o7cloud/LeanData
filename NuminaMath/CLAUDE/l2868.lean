import Mathlib

namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2868_286891

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  ∃ (z : ℂ), z = i^2017 / (1 + i) ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2868_286891


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l2868_286860

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length 1 -/
def UnitCube : Set Point3D :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1}

/-- The diagonal vertices of the cube -/
def A : Point3D := ⟨0, 0, 0⟩
def C : Point3D := ⟨1, 1, 1⟩

/-- The midpoints of two opposite edges not containing A or C -/
def B : Point3D := ⟨0.5, 0, 1⟩
def D : Point3D := ⟨0.5, 1, 0⟩

/-- The plane passing through A, B, C, and D -/
def InterceptingPlane : Set Point3D :=
  {p | ∃ (s t : ℝ), p = ⟨s, t, 1 - s - t⟩ ∧ 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1}

/-- The quadrilateral ABCD formed by the intersection of the plane and the cube -/
def QuadrilateralABCD : Set Point3D :=
  UnitCube ∩ InterceptingPlane

/-- The area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point3D) : ℝ := sorry

theorem area_of_quadrilateral_ABCD :
  quadrilateralArea A B C D = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l2868_286860


namespace NUMINAMATH_CALUDE_base_b_divisibility_l2868_286875

theorem base_b_divisibility (b : ℤ) : 
  let diff := 2 * b^3 + 2 * b - (2 * b^2 + 2 * b + 1)
  (b = 8 ∧ ¬(diff % 3 = 0)) ∨
  ((b = 3 ∨ b = 4 ∨ b = 6 ∨ b = 7) ∧ (diff % 3 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l2868_286875


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2868_286866

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m : ℕ), m > 0 ∧ 
    9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) → n ≤ k) ∧
  (∃ (m : ℕ), m > 0 ∧ 9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2868_286866


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2868_286876

theorem no_integer_solutions : 
  ¬ ∃ (x : ℤ), (x^2 - 3*x + 2)^2 - 3*(x^2 - 3*x) - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2868_286876


namespace NUMINAMATH_CALUDE_student_scores_l2868_286881

theorem student_scores (M P C : ℝ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 30) : 
  M + P = 40 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l2868_286881


namespace NUMINAMATH_CALUDE_number_placement_theorem_l2868_286885

-- Define the function f that maps integer coordinates to natural numbers
def f : ℤ × ℤ → ℕ := fun (x, y) => Nat.gcd (Int.natAbs x) (Int.natAbs y)

-- Define the property that every natural number appears at some point
def surjective (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), f (x, y) = n

-- Define the property of periodicity along a line
def periodic_along_line (f : ℤ × ℤ → ℕ) (a b c : ℤ) : Prop :=
  c ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℤ), (a * x₁ + b * y₁ = c) ∧ (a * x₂ + b * y₂ = c) ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  ∃ (dx dy : ℤ), ∀ (x y : ℤ), 
    (a * x + b * y = c) → f (x, y) = f (x + dx, y + dy)

theorem number_placement_theorem :
  surjective f ∧ 
  (∀ (a b c : ℤ), periodic_along_line f a b c) :=
sorry

end NUMINAMATH_CALUDE_number_placement_theorem_l2868_286885


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2868_286884

/-- Given a configuration of squares and a rectangle, prove the ratio of rectangle's length to width -/
theorem rectangle_ratio (s : ℝ) (h1 : s > 0) : 
  let large_square_side := 3 * s
  let small_square_side := s
  let rectangle_width := s
  let rectangle_length := large_square_side
  (rectangle_length / rectangle_width : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2868_286884


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2868_286869

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}

-- Define the set N
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2868_286869


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2868_286882

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^5 - 24*x^4 + 5*x^3 + 15*x^2 - 18*x + 12 = 
  (x - 3) * (x^5 + 5*x^4 - 9*x^3 - 22*x^2 - 51*x - 171) - 501 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2868_286882


namespace NUMINAMATH_CALUDE_problem_statement_l2868_286868

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 40) / (x - 3)) →
  P + Q = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2868_286868


namespace NUMINAMATH_CALUDE_boat_license_combinations_l2868_286813

/-- The number of possible letters for a boat license. -/
def num_letters : ℕ := 4

/-- The number of digits in a boat license. -/
def num_digits : ℕ := 6

/-- The number of possible digits for each position (0-9). -/
def digits_per_position : ℕ := 10

/-- The total number of possible boat license combinations. -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

/-- Theorem stating that the total number of boat license combinations is 4,000,000. -/
theorem boat_license_combinations :
  total_combinations = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l2868_286813


namespace NUMINAMATH_CALUDE_small_months_not_remainder_l2868_286883

/-- The number of months in a year -/
def total_months : ℕ := 12

/-- The number of big months in a year -/
def big_months : ℕ := 7

/-- The number of small months in a year -/
def small_months : ℕ := 4

/-- February is a special case and not counted as either big or small -/
def february_special : Prop := True

theorem small_months_not_remainder :
  small_months ≠ total_months - big_months :=
by sorry

end NUMINAMATH_CALUDE_small_months_not_remainder_l2868_286883


namespace NUMINAMATH_CALUDE_max_initial_happy_citizens_l2868_286853

/-- Represents the state of happiness for a citizen --/
inductive MoodState
| Happy
| Unhappy

/-- Represents a citizen in Happy City --/
structure Citizen where
  id : Nat
  mood : MoodState

/-- Represents the state of Happy City --/
structure HappyCity where
  citizens : List Citizen
  day : Nat

/-- Function to simulate a day of smiling in Happy City --/
def smileDay (city : HappyCity) : HappyCity :=
  sorry

/-- Function to count happy citizens --/
def countHappy (city : HappyCity) : Nat :=
  sorry

/-- Theorem stating the maximum initial number of happy citizens --/
theorem max_initial_happy_citizens :
  ∀ (initialCity : HappyCity),
    initialCity.citizens.length = 2014 →
    (∃ (finalCity : HappyCity),
      finalCity = (smileDay ∘ smileDay ∘ smileDay ∘ smileDay) initialCity ∧
      countHappy finalCity = 2000) →
    countHappy initialCity ≤ 32 :=
  sorry

end NUMINAMATH_CALUDE_max_initial_happy_citizens_l2868_286853


namespace NUMINAMATH_CALUDE_integral_identity_l2868_286890

theorem integral_identity : ∫ (x : ℝ) in -Real.arctan (1/3)..0, (3 * Real.tan x + 1) / (2 * Real.sin (2*x) - 5 * Real.cos (2*x) + 1) = (1/4) * Real.log (6/5) := by
  sorry

end NUMINAMATH_CALUDE_integral_identity_l2868_286890


namespace NUMINAMATH_CALUDE_max_different_ages_l2868_286834

/-- The maximum number of different integer ages within one standard deviation of the average -/
theorem max_different_ages (average_age : ℝ) (std_dev : ℝ) : average_age = 31 → std_dev = 9 →
  (Set.Icc (average_age - std_dev) (average_age + std_dev) ∩ Set.range (Int.cast : ℤ → ℝ)).ncard = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_different_ages_l2868_286834


namespace NUMINAMATH_CALUDE_sixth_term_term_1994_l2868_286805

-- Define the sequence
def a (n : ℕ+) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end NUMINAMATH_CALUDE_sixth_term_term_1994_l2868_286805


namespace NUMINAMATH_CALUDE_greatest_area_difference_l2868_286825

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the maximum length available for the rotated rectangle's diagonal. -/
def maxDiagonal : ℕ := 50

theorem greatest_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 100 ∧ 
    perimeter r2 = 100 ∧ 
    r2.length * r2.length + r2.width * r2.width ≤ maxDiagonal * maxDiagonal ∧
    ∀ (s1 s2 : Rectangle), 
      perimeter s1 = 100 → 
      perimeter s2 = 100 → 
      s2.length * s2.length + s2.width * s2.width ≤ maxDiagonal * maxDiagonal →
      (area r1 - area r2) ≥ (area s1 - area s2) ∧
      (area r1 - area r2) = 373 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l2868_286825


namespace NUMINAMATH_CALUDE_product_mod_seven_l2868_286865

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2868_286865


namespace NUMINAMATH_CALUDE_mork_mindy_tax_rate_l2868_286828

/-- The combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (mindy_income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : mindy_income_ratio = 2) : 
  (mork_rate + mindy_rate * mindy_income_ratio) / (1 + mindy_income_ratio) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_mork_mindy_tax_rate_l2868_286828


namespace NUMINAMATH_CALUDE_only_prime_three_squared_plus_eight_prime_l2868_286888

theorem only_prime_three_squared_plus_eight_prime :
  ∀ p : ℕ, Prime p ∧ Prime (p^2 + 8) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_three_squared_plus_eight_prime_l2868_286888


namespace NUMINAMATH_CALUDE_largest_three_digit_arithmetic_sequence_l2868_286824

/-- Checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- Checks if the digits of a three-digit number form an arithmetic sequence -/
def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) - ((n / 10) % 10) = ((n / 10) % 10) - (n % 10)

/-- The main theorem stating that 789 is the largest three-digit number
    with distinct digits forming an arithmetic sequence -/
theorem largest_three_digit_arithmetic_sequence :
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 →
    has_distinct_digits m ∧ digits_form_arithmetic_sequence m →
    m ≤ 789) ∧
  has_distinct_digits 789 ∧ digits_form_arithmetic_sequence 789 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_arithmetic_sequence_l2868_286824


namespace NUMINAMATH_CALUDE_radical_simplification_l2868_286839

theorem radical_simplification : 
  Real.sqrt ((16^12 + 8^14) / (16^5 + 8^16 + 2^24)) = 2^11 * Real.sqrt (65/17) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2868_286839


namespace NUMINAMATH_CALUDE_second_month_sale_is_11860_l2868_286837

/-- Represents the sales data for a grocer over 6 months -/
structure GrocerSales where
  first_month : ℕ
  third_month : ℕ
  fourth_month : ℕ
  fifth_month : ℕ
  sixth_month : ℕ
  average_sale : ℕ

/-- Calculates the sale in the second month given the sales data -/
def second_month_sale (sales : GrocerSales) : ℕ :=
  6 * sales.average_sale - (sales.first_month + sales.third_month + sales.fourth_month + sales.fifth_month + sales.sixth_month)

/-- Theorem stating that the second month sale is 11860 given the specific sales data -/
theorem second_month_sale_is_11860 :
  let sales : GrocerSales := {
    first_month := 5420,
    third_month := 6350,
    fourth_month := 6500,
    fifth_month := 6200,
    sixth_month := 8270,
    average_sale := 6400
  }
  second_month_sale sales = 11860 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_is_11860_l2868_286837


namespace NUMINAMATH_CALUDE_cube_volume_increase_l2868_286864

theorem cube_volume_increase (a : ℝ) (ha : a > 0) :
  (2 * a)^3 - a^3 = 7 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l2868_286864


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2868_286809

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2868_286809


namespace NUMINAMATH_CALUDE_range_of_m_l2868_286845

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x m : ℝ) : Prop := x^2 + x + m - m^2 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (m > 1) →
  (∀ x : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ (q y m)) →
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2868_286845


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2868_286827

theorem min_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  (1 / (a + 2*b)) + (4 / (2*a + b)) ≥ 3 :=
by sorry

theorem min_value_achieved (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ a₀ + b₀ = 1 ∧ (1 / (a₀ + 2*b₀)) + (4 / (2*a₀ + b₀)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2868_286827


namespace NUMINAMATH_CALUDE_mary_screws_problem_l2868_286867

theorem mary_screws_problem (initial_screws : Nat) (buy_multiplier : Nat) (num_sections : Nat) : 
  initial_screws = 8 → buy_multiplier = 2 → num_sections = 4 → 
  (initial_screws + initial_screws * buy_multiplier) / num_sections = 6 := by
sorry

end NUMINAMATH_CALUDE_mary_screws_problem_l2868_286867


namespace NUMINAMATH_CALUDE_cube_root_of_2_plus_11i_l2868_286829

def complex_cube_root (z : ℂ) : Prop :=
  z^3 = (2 : ℂ) + Complex.I * 11

theorem cube_root_of_2_plus_11i :
  complex_cube_root (2 + Complex.I) ∧
  ∃ (z₁ z₂ : ℂ), 
    complex_cube_root z₁ ∧
    complex_cube_root z₂ ∧
    z₁ ≠ z₂ ∧
    z₁ ≠ (2 + Complex.I) ∧
    z₂ ≠ (2 + Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_2_plus_11i_l2868_286829


namespace NUMINAMATH_CALUDE_range_of_fraction_l2868_286838

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3/2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2868_286838


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2868_286857

theorem sum_of_cubes (a b c d e : ℝ) 
  (sum_zero : a + b + c + d + e = 0)
  (sum_products : a*b*c + a*b*d + a*b*e + a*c*d + a*c*e + a*d*e + b*c*d + b*c*e + b*d*e + c*d*e = 2008) :
  a^3 + b^3 + c^3 + d^3 + e^3 = -12048 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2868_286857


namespace NUMINAMATH_CALUDE_quadratic_root_implies_d_l2868_286871

theorem quadratic_root_implies_d (d : ℚ) : 
  (∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = -7 + Real.sqrt 15 ∨ x = -7 - Real.sqrt 15) →
  d = 181 / 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_d_l2868_286871


namespace NUMINAMATH_CALUDE_seating_theorem_l2868_286851

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange a pair of people -/
def pairArrangements : ℕ := 2

/-- The number of ways to seat 10 people around a round table with two specific people next to each other -/
def seatingArrangements : ℕ := roundTableArrangements 9 * pairArrangements

theorem seating_theorem : seatingArrangements = 80640 := by sorry

end NUMINAMATH_CALUDE_seating_theorem_l2868_286851


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2868_286887

/-- Given Mike's current age m and Dan's current age d, prove that the number of years
    until their age ratio is 3:2 is 97, given the initial conditions. -/
theorem age_ratio_problem (m d : ℕ) (h1 : m - 3 = 4 * (d - 3)) (h2 : m - 8 = 5 * (d - 8)) :
  ∃ x : ℕ, x = 97 ∧ (m + x : ℚ) / (d + x) = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2868_286887


namespace NUMINAMATH_CALUDE_fraction_simplification_l2868_286843

theorem fraction_simplification : (4 : ℚ) / (2 - 4 / 5) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2868_286843


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l2868_286815

theorem quadratic_roots_sum_reciprocals (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → (b^2 + 8*b + 4 = 0) → (a / b + b / a = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l2868_286815


namespace NUMINAMATH_CALUDE_probability_sum_seven_l2868_286801

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_is_seven (roll : ℕ × ℕ) : Prop :=
  roll.1 + roll.2 = 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 6), (6, 1), (2, 5), (5, 2), (3, 4), (4, 3)}

theorem probability_sum_seven :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card (die_faces.product die_faces) : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_seven_l2868_286801


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l2868_286823

theorem cubic_root_sum_product (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let p : ℝ → ℝ := λ x => α * x^3 - α * x^2 + β * x + β
  ∀ x₁ x₂ x₃ : ℝ, (p x₁ = 0 ∧ p x₂ = 0 ∧ p x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l2868_286823


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2868_286863

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I * (((a + 3 * Complex.I) / (1 - 2 * Complex.I)).im) = ((a + 3 * Complex.I) / (1 - 2 * Complex.I))) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2868_286863


namespace NUMINAMATH_CALUDE_candy_box_count_candy_box_theorem_l2868_286835

theorem candy_box_count : ℝ → Prop :=
  fun x =>
    let day1_eaten := 0.2 * x + 16
    let day1_remaining := x - day1_eaten
    let day2_eaten := 0.3 * day1_remaining + 20
    let day2_remaining := day1_remaining - day2_eaten
    let day3_eaten := 0.75 * day2_remaining + 30
    day3_eaten = day2_remaining ∧ x = 270

theorem candy_box_theorem : ∃ x : ℝ, candy_box_count x :=
  sorry

end NUMINAMATH_CALUDE_candy_box_count_candy_box_theorem_l2868_286835


namespace NUMINAMATH_CALUDE_roger_money_theorem_l2868_286898

def roger_money_problem (initial_amount gift_amount spent_amount : ℕ) : Prop :=
  initial_amount + gift_amount - spent_amount = 19

theorem roger_money_theorem :
  roger_money_problem 16 28 25 := by
  sorry

end NUMINAMATH_CALUDE_roger_money_theorem_l2868_286898


namespace NUMINAMATH_CALUDE_seven_digit_divisibility_l2868_286810

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem seven_digit_divisibility :
  ∀ (a b c d e f : ℕ),
    (is_divisible_by_8 (2300000 + a * 10000 + b * 1000 + 372) = false) ∧
    (is_divisible_by_8 (5300000 + c * 10000 + d * 1000 + 164) = false) ∧
    (is_divisible_by_8 (5000000 + e * 10000 + f * 1000 + 3416) = true) ∧
    (is_divisible_by_8 (7100000 + a * 10000 + b * 1000 + 172) = false) :=
by
  sorry

#check seven_digit_divisibility

end NUMINAMATH_CALUDE_seven_digit_divisibility_l2868_286810


namespace NUMINAMATH_CALUDE_spade_evaluation_l2868_286859

-- Define the ♠ operation
def spade (a b : ℚ) : ℚ := (3 * a + b) / (a + b)

-- Theorem statement
theorem spade_evaluation :
  spade (spade 5 (spade 3 6)) 1 = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spade_evaluation_l2868_286859


namespace NUMINAMATH_CALUDE_cube_sum_of_conjugate_fractions_l2868_286852

theorem cube_sum_of_conjugate_fractions :
  let x := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
  let y := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)
  x^3 + y^3 = 2702 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_conjugate_fractions_l2868_286852


namespace NUMINAMATH_CALUDE_abc_inequality_l2868_286897

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 11/6 * c < a + b ∧ a + b < 2 * c)
  (h2 : 3/2 * a < b + c ∧ b + c < 5/3 * a)
  (h3 : 5/2 * b < a + c ∧ a + c < 11/4 * b) :
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2868_286897


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_22_l2868_286874

-- Define the triangle
def triangle_side_a : ℝ := 5
def triangle_side_b : ℝ := 12
def triangle_side_c : ℝ := 13

-- Define the rectangle
def rectangle_width : ℝ := 5

-- Theorem statement
theorem rectangle_perimeter_equals_22 :
  let triangle_area := (1/2) * triangle_side_a * triangle_side_b
  let rectangle_length := triangle_area / rectangle_width
  2 * (rectangle_width + rectangle_length) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_22_l2868_286874


namespace NUMINAMATH_CALUDE_laptop_price_l2868_286833

def original_price : ℝ → Prop :=
  fun x => (0.80 * x - 50 = 0.70 * x - 30) ∧ (x > 0)

theorem laptop_price : ∃ x, original_price x ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l2868_286833


namespace NUMINAMATH_CALUDE_power_of_seven_l2868_286844

theorem power_of_seven (k : ℕ) : (7 : ℝ) ^ (4 * k + 2) = 784 → (7 : ℝ) ^ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l2868_286844


namespace NUMINAMATH_CALUDE_sum_of_powers_divisibility_l2868_286822

theorem sum_of_powers_divisibility 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → (6 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisibility_l2868_286822


namespace NUMINAMATH_CALUDE_spring_scale_reading_comparison_l2868_286841

/-- Represents the angular velocity of Earth's rotation -/
def earth_angular_velocity : ℝ := sorry

/-- Represents the radius of the Earth at the equator -/
def earth_equator_radius : ℝ := sorry

/-- Represents the acceleration due to gravity at the equator -/
def gravity_equator : ℝ := sorry

/-- Represents the acceleration due to gravity at the poles -/
def gravity_pole : ℝ := sorry

/-- Calculates the centrifugal force at the equator for an object of mass m -/
def centrifugal_force (m : ℝ) : ℝ :=
  m * earth_angular_velocity^2 * earth_equator_radius

/-- Calculates the apparent weight of an object at the equator -/
def apparent_weight_equator (m : ℝ) : ℝ :=
  m * gravity_equator - centrifugal_force m

/-- Calculates the apparent weight of an object at the pole -/
def apparent_weight_pole (m : ℝ) : ℝ :=
  m * gravity_pole

theorem spring_scale_reading_comparison (m : ℝ) (m_pos : m > 0) :
  apparent_weight_pole m > apparent_weight_equator m :=
by
  sorry

end NUMINAMATH_CALUDE_spring_scale_reading_comparison_l2868_286841


namespace NUMINAMATH_CALUDE_more_boys_than_girls_is_two_l2868_286808

/-- The number of more boys than girls in a field day competition -/
def more_boys_than_girls : ℕ :=
  let fourth_grade_class1_girls := 12
  let fourth_grade_class1_boys := 13
  let fourth_grade_class2_girls := 15
  let fourth_grade_class2_boys := 11
  let fifth_grade_class1_girls := 9
  let fifth_grade_class1_boys := 13
  let fifth_grade_class2_girls := 10
  let fifth_grade_class2_boys := 11

  let total_girls := fourth_grade_class1_girls + fourth_grade_class2_girls +
                     fifth_grade_class1_girls + fifth_grade_class2_girls
  let total_boys := fourth_grade_class1_boys + fourth_grade_class2_boys +
                    fifth_grade_class1_boys + fifth_grade_class2_boys

  total_boys - total_girls

theorem more_boys_than_girls_is_two :
  more_boys_than_girls = 2 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_is_two_l2868_286808


namespace NUMINAMATH_CALUDE_social_media_to_phone_ratio_l2868_286858

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 8

/-- Represents the weekly social media usage in hours -/
def weekly_social_media : ℝ := 28

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem stating that the ratio of daily social media usage to total daily phone usage is 1:2 -/
theorem social_media_to_phone_ratio :
  (weekly_social_media / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_to_phone_ratio_l2868_286858


namespace NUMINAMATH_CALUDE_diana_eraser_sharing_l2868_286817

theorem diana_eraser_sharing (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 3840) (h2 : erasers_per_friend = 80) :
  total_erasers / erasers_per_friend = 48 := by
  sorry

end NUMINAMATH_CALUDE_diana_eraser_sharing_l2868_286817


namespace NUMINAMATH_CALUDE_sunday_school_three_year_olds_l2868_286830

/-- The number of 4-year-olds in the Sunday school -/
def four_year_olds : ℕ := 20

/-- The number of 5-year-olds in the Sunday school -/
def five_year_olds : ℕ := 15

/-- The number of 6-year-olds in the Sunday school -/
def six_year_olds : ℕ := 22

/-- The average class size -/
def average_class_size : ℕ := 35

/-- The number of classes -/
def num_classes : ℕ := 2

theorem sunday_school_three_year_olds :
  ∃ (three_year_olds : ℕ),
    (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / num_classes = average_class_size ∧
    three_year_olds = 13 := by
  sorry

end NUMINAMATH_CALUDE_sunday_school_three_year_olds_l2868_286830


namespace NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l2868_286856

/-- Given two curves and their tangent lines, prove the range of parameter a -/
theorem tangent_lines_perpendicular_range (a : ℝ) : 
  ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 3/2 ∧
  let f₁ (x : ℝ) := (a * x - 1) * Real.exp x
  let f₂ (x : ℝ) := (1 - x) * Real.exp (-x)
  let k₁ := (a * x₀ + a - 1) * Real.exp x₀
  let k₂ := (x₀ - 2) * Real.exp (-x₀)
  k₁ * k₂ = -1 →
  1 ≤ a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l2868_286856


namespace NUMINAMATH_CALUDE_triangle_vector_parallel_l2868_286802

theorem triangle_vector_parallel (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a, Real.sqrt 3 * b) = (Real.cos A, Real.sin B) →
  A = π / 3 ∧
  (a = 2 → 2 < b + c ∧ b + c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_parallel_l2868_286802


namespace NUMINAMATH_CALUDE_geometric_sum_l2868_286840

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_l2868_286840


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2868_286804

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2868_286804


namespace NUMINAMATH_CALUDE_rehab_centers_multiple_l2868_286861

/-- The number of rehabilitation centers visited by each person and the total visited --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ
  total : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.total = 27 ∧
  rc.jane = rc.total - (rc.lisa + rc.jude + rc.han)

/-- The theorem to be proved --/
theorem rehab_centers_multiple (rc : RehabCenters) 
  (h : problem_conditions rc) : ∃ x : ℕ, x = 2 ∧ rc.jane = x * rc.han + 6 := by
  sorry

end NUMINAMATH_CALUDE_rehab_centers_multiple_l2868_286861


namespace NUMINAMATH_CALUDE_only_sample_size_statement_correct_l2868_286895

/-- Represents a statistical study with a population and a sample. -/
structure StatisticalStudy where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a statement about the statistical study. -/
inductive Statement
  | sample_is_population
  | sample_average_is_population_average
  | examinees_are_population
  | sample_size_is_1000

/-- Checks if a statement is correct for the given statistical study. -/
def is_correct_statement (study : StatisticalStudy) (stmt : Statement) : Prop :=
  match stmt with
  | Statement.sample_is_population => False
  | Statement.sample_average_is_population_average => False
  | Statement.examinees_are_population => False
  | Statement.sample_size_is_1000 => study.sample_size = 1000

/-- The main theorem stating that only the sample size statement is correct. -/
theorem only_sample_size_statement_correct (study : StatisticalStudy) 
    (h1 : study.population_size = 70000)
    (h2 : study.sample_size = 1000) :
    ∀ (stmt : Statement), is_correct_statement study stmt ↔ stmt = Statement.sample_size_is_1000 := by
  sorry

end NUMINAMATH_CALUDE_only_sample_size_statement_correct_l2868_286895


namespace NUMINAMATH_CALUDE_odd_number_pattern_l2868_286878

/-- Represents the number of odd numbers in a row of the pattern -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the sum of odd numbers up to the nth row -/
def sum_to_row (n : ℕ) : ℕ := n^2

/-- Represents the nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The problem statement -/
theorem odd_number_pattern :
  let total_previous_rows := sum_to_row 20
  let position_in_row := 6
  nth_odd (total_previous_rows + position_in_row) = 811 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_pattern_l2868_286878


namespace NUMINAMATH_CALUDE_katies_journey_distance_l2868_286873

/-- The total distance of Katie's journey to the island -/
def total_distance (leg1 leg2 leg3 : ℕ) : ℕ :=
  leg1 + leg2 + leg3

/-- Theorem stating that the total distance of Katie's journey is 436 miles -/
theorem katies_journey_distance :
  total_distance 132 236 68 = 436 := by
  sorry

end NUMINAMATH_CALUDE_katies_journey_distance_l2868_286873


namespace NUMINAMATH_CALUDE_actual_speed_is_22_5_l2868_286849

/-- Proves that the actual average speed is 22.5 mph given the conditions of the problem -/
theorem actual_speed_is_22_5 (v t : ℝ) (h : v > 0) (h' : t > 0) :
  (v * t = (v + 37.5) * (3/8 * t)) → v = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_22_5_l2868_286849


namespace NUMINAMATH_CALUDE_roses_mary_added_l2868_286806

/-- The number of roses Mary put in the vase -/
def roses_added : ℕ := sorry

/-- The initial number of roses in the vase -/
def initial_roses : ℕ := 6

/-- The final number of roses in the vase -/
def final_roses : ℕ := 22

theorem roses_mary_added : roses_added = 16 := by
  sorry

end NUMINAMATH_CALUDE_roses_mary_added_l2868_286806


namespace NUMINAMATH_CALUDE_bill_receives_26_l2868_286880

/-- Given a sum of money M to be divided among Allan, Bill, and Carol, prove that Bill receives $26 --/
theorem bill_receives_26 (M : ℚ) : 
  (∃ (allan_share bill_share carol_share : ℚ),
    -- Allan's share
    allan_share = 1 + (1/3) * (M - 1) ∧
    -- Bill's share
    bill_share = 6 + (1/3) * (M - allan_share - 6) ∧
    -- Carol's share
    carol_share = M - allan_share - bill_share ∧
    -- Carol receives $40
    carol_share = 40 ∧
    -- Bill's share is $26
    bill_share = 26) :=
by sorry

end NUMINAMATH_CALUDE_bill_receives_26_l2868_286880


namespace NUMINAMATH_CALUDE_smallest_chord_length_l2868_286889

/-- The circle equation x^2 + y^2 + 4x - 6y + 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y + 4 = 0

/-- The line equation mx - y + 1 = 0 -/
def line_equation (m x y : ℝ) : Prop :=
  m*x - y + 1 = 0

/-- The chord length for a given m -/
noncomputable def chord_length (m : ℝ) : ℝ :=
  sorry  -- Definition of chord length in terms of m

theorem smallest_chord_length :
  ∃ (m : ℝ), ∀ (m' : ℝ), chord_length m ≤ chord_length m' ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_chord_length_l2868_286889


namespace NUMINAMATH_CALUDE_rectangle_dimension_solution_l2868_286854

theorem rectangle_dimension_solution :
  ∃! x : ℚ, (3 * x - 4 > 0) ∧ 
             (2 * x + 7 > 0) ∧ 
             ((3 * x - 4) * (2 * x + 7) = 18 * x - 10) ∧
             x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_solution_l2868_286854


namespace NUMINAMATH_CALUDE_truncated_pyramid_sphere_area_relation_l2868_286893

/-- Given a regular n-gonal truncated pyramid circumscribed around a sphere:
    S1 is the area of the base surface,
    S2 is the area of the lateral surface,
    S is the total surface area,
    σ is the area of the polygon whose vertices are the tangential points of the sphere and the lateral faces of the pyramid.
    This theorem states that σS = 4S1S2 cos²(π/n). -/
theorem truncated_pyramid_sphere_area_relation (n : ℕ) (S1 S2 S σ : ℝ) :
  n ≥ 3 →
  S1 > 0 →
  S2 > 0 →
  S = S1 + S2 →
  σ > 0 →
  σ * S = 4 * S1 * S2 * (Real.cos (π / n : ℝ))^2 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_sphere_area_relation_l2868_286893


namespace NUMINAMATH_CALUDE_rhombus_area_from_overlapping_strips_l2868_286899

/-- The area of a rhombus formed by two overlapping strips -/
theorem rhombus_area_from_overlapping_strips (β : Real) (h : β ≠ 0) : 
  let strip_width : Real := 2
  let diagonal1 : Real := strip_width
  let diagonal2 : Real := strip_width / Real.sin β
  let rhombus_area : Real := (1 / 2) * diagonal1 * diagonal2
  rhombus_area = 2 / Real.sin β :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_from_overlapping_strips_l2868_286899


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l2868_286848

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0 ∧ 
   Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin θ ∧
   ∀ φ, φ > 0 ∧ Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin φ → θ ≤ φ) →
  θ = 70 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l2868_286848


namespace NUMINAMATH_CALUDE_team_wins_l2868_286832

theorem team_wins (current_percentage : ℚ) (future_wins future_games : ℕ) 
  (new_percentage : ℚ) (h1 : current_percentage = 45/100) 
  (h2 : future_wins = 6) (h3 : future_games = 8) (h4 : new_percentage = 1/2) : 
  ∃ (total_games : ℕ) (current_wins : ℕ), 
    (current_wins : ℚ) / total_games = current_percentage ∧
    ((current_wins + future_wins) : ℚ) / (total_games + future_games) = new_percentage ∧
    current_wins = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_wins_l2868_286832


namespace NUMINAMATH_CALUDE_lawrence_average_work_hours_l2868_286836

def lawrence_work_hours (full_days : ℕ) (partial_days : ℕ) (full_day_hours : ℝ) (partial_day_hours : ℝ) : ℝ :=
  (full_days : ℝ) * full_day_hours + (partial_days : ℝ) * partial_day_hours

theorem lawrence_average_work_hours :
  let total_days : ℕ := 5
  let full_days : ℕ := 3
  let partial_days : ℕ := 2
  let full_day_hours : ℝ := 8
  let partial_day_hours : ℝ := 5.5
  let total_hours := lawrence_work_hours full_days partial_days full_day_hours partial_day_hours
  total_hours / total_days = 7 := by
sorry

end NUMINAMATH_CALUDE_lawrence_average_work_hours_l2868_286836


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2868_286803

/-- A type representing a 17-digit number -/
def Digit17 := Fin 10 → Fin 10

/-- Reverses a 17-digit number -/
def reverse (n : Digit17) : Digit17 :=
  fun i => n (16 - i)

/-- Adds two 17-digit numbers -/
def add (a b : Digit17) : Digit17 :=
  sorry

/-- Checks if a number has at least one even digit -/
def hasEvenDigit (n : Digit17) : Prop :=
  ∃ i, (n i).val % 2 = 0

/-- Main theorem: For any 17-digit number, when added to its reverse, 
    the resulting sum contains at least one even digit -/
theorem sum_with_reverse_has_even_digit (n : Digit17) : 
  hasEvenDigit (add n (reverse n)) := by
  sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2868_286803


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l2868_286886

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) --/
def length (n : ℕ) : ℕ := sorry

theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → a + 3 * b ≤ x + 3 * y ∧
  x + 3 * y = 98305 := by sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l2868_286886


namespace NUMINAMATH_CALUDE_smallest_semicircle_area_l2868_286862

/-- Given a right-angled triangle with semicircles on each side, prove that the smallest semicircle has area 144 -/
theorem smallest_semicircle_area (x : ℝ) : 
  x > 0 ∧ x^2 < 180 ∧ 3*x < 180 ∧ x^2 + 3*x = 180 → x^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_semicircle_area_l2868_286862


namespace NUMINAMATH_CALUDE_triangle_angle_property_l2868_286894

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is inside a triangle -/
def isInsideTriangle (P : Point3D) (T : Triangle3D) : Prop := sorry

/-- Check if a point is outside the plane of a triangle -/
def isOutsidePlane (D : Point3D) (T : Triangle3D) : Prop := sorry

/-- Angle between three points in 3D space -/
def angle (A B C : Point3D) : ℝ := sorry

/-- An angle is acute if it's less than 90 degrees -/
def isAcute (θ : ℝ) : Prop := θ < Real.pi / 2

/-- An angle is obtuse if it's greater than 90 degrees -/
def isObtuse (θ : ℝ) : Prop := θ > Real.pi / 2

theorem triangle_angle_property (T : Triangle3D) (P D : Point3D) :
  isInsideTriangle P T →
  isOutsidePlane D T →
  (isAcute (angle T.A P D) ∨ isAcute (angle T.B P D) ∨ isAcute (angle T.C P D)) →
  (isObtuse (angle T.A P D) ∨ isObtuse (angle T.B P D) ∨ isObtuse (angle T.C P D)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_property_l2868_286894


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2868_286892

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 5
  let φ : ℝ → ℝ := λ x ↦ 5 * x + 4
  ∀ x : ℝ, δ (φ x) = 9 → x = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2868_286892


namespace NUMINAMATH_CALUDE_largest_x_value_l2868_286812

theorem largest_x_value (x : ℝ) : 
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2 →
  x ≤ 9/4 ∧ ∃ y, y > 9/4 → (16 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y ≠ 8 * y - 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2868_286812


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_equals_two_sqrt_three_l2868_286814

theorem sqrt_two_times_sqrt_six_equals_two_sqrt_three :
  Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_equals_two_sqrt_three_l2868_286814


namespace NUMINAMATH_CALUDE_high_low_game_combinations_l2868_286847

/-- Represents the types of cards in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  totalCards : Nat
  highCards : Nat
  lowCards : Nat
  highCardPoints : Nat
  lowCardPoints : Nat
  targetPoints : Nat

/-- Calculates the total points for a given combination of high and low cards -/
def calculatePoints (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Nat :=
  highCount * game.highCardPoints + lowCount * game.lowCardPoints

/-- Checks if a given combination of high and low cards achieves the target points -/
def isValidCombination (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Prop :=
  calculatePoints game highCount lowCount = game.targetPoints

/-- Theorem: In the high-low game, to earn exactly 5 points, 
    the number of low cards drawn must be either 1, 3, or 5 -/
theorem high_low_game_combinations (game : HighLowGame) 
    (h1 : game.totalCards = 52)
    (h2 : game.highCards = game.lowCards)
    (h3 : game.highCards + game.lowCards = game.totalCards)
    (h4 : game.highCardPoints = 2)
    (h5 : game.lowCardPoints = 1)
    (h6 : game.targetPoints = 5) :
    ∀ (highCount lowCount : Nat), 
      isValidCombination game highCount lowCount → 
      lowCount = 1 ∨ lowCount = 3 ∨ lowCount = 5 :=
  sorry


end NUMINAMATH_CALUDE_high_low_game_combinations_l2868_286847


namespace NUMINAMATH_CALUDE_sneakers_cost_l2868_286826

/-- The cost of sneakers given initial savings, action figure sales, and remaining money --/
theorem sneakers_cost
  (initial_savings : ℕ)
  (action_figures_sold : ℕ)
  (price_per_figure : ℕ)
  (money_left : ℕ)
  (h1 : initial_savings = 15)
  (h2 : action_figures_sold = 10)
  (h3 : price_per_figure = 10)
  (h4 : money_left = 25) :
  initial_savings + action_figures_sold * price_per_figure - money_left = 90 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_cost_l2868_286826


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2868_286870

def line1 (a : ℝ) (x y : ℝ) : Prop := 2*x + a*y = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := x - (a+1)*y = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ → line2 a x₂ y₂ → 
    (x₁ * x₂ + y₁ * y₂ = 0 ∨ (x₁ = 0 ∧ y₁ = 0) ∨ (x₂ = 0 ∧ y₂ = 0))

theorem perpendicular_lines (a : ℝ) : perpendicular a → (a = -2 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2868_286870


namespace NUMINAMATH_CALUDE_staircase_climbing_l2868_286818

/-- Number of ways to ascend n steps by jumping 1 or 2 steps at a time -/
def ascend (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => ascend (n+1) + ascend n

/-- Number of ways to descend n steps with option to skip steps -/
def descend (n : ℕ) : ℕ := 2^(n-1)

theorem staircase_climbing :
  (ascend 10 = 89) ∧ (descend 10 = 512) := by
  sorry


end NUMINAMATH_CALUDE_staircase_climbing_l2868_286818


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l2868_286816

open Real

theorem triangle_side_ratio_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 3 * B ∧  -- Given condition
  a / sin A = b / sin B ∧  -- Sine rule
  a / sin A = c / sin C →  -- Sine rule
  1 < a / b ∧ a / b < 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_range_l2868_286816


namespace NUMINAMATH_CALUDE_miami_ny_temp_difference_l2868_286850

/-- Represents the temperatures of three cities and their relationships -/
structure CityTemperatures where
  new_york : ℝ
  miami : ℝ
  san_diego : ℝ
  ny_temp_is_80 : new_york = 80
  miami_cooler_than_sd : miami = san_diego - 25
  average_temp : (new_york + miami + san_diego) / 3 = 95

/-- The temperature difference between Miami and New York -/
def temp_difference (ct : CityTemperatures) : ℝ :=
  ct.miami - ct.new_york

/-- Theorem stating that the temperature difference between Miami and New York is 10 degrees -/
theorem miami_ny_temp_difference (ct : CityTemperatures) : temp_difference ct = 10 := by
  sorry

end NUMINAMATH_CALUDE_miami_ny_temp_difference_l2868_286850


namespace NUMINAMATH_CALUDE_marble_difference_l2868_286877

theorem marble_difference (red_marbles : ℕ) (red_bags : ℕ) (blue_marbles : ℕ) (blue_bags : ℕ)
  (h1 : red_marbles = 288)
  (h2 : red_bags = 12)
  (h3 : blue_marbles = 243)
  (h4 : blue_bags = 9)
  (h5 : red_bags ≠ 0)
  (h6 : blue_bags ≠ 0) :
  blue_marbles / blue_bags - red_marbles / red_bags = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2868_286877


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2868_286819

/-- The sum of all terms in the geometric sequence {(2/3)^n, n ∈ ℕ*} is 2. -/
theorem geometric_sequence_sum : 
  let a : ℕ → ℝ := fun n => (2/3)^n
  ∑' n, a n = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2868_286819


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2868_286842

theorem average_weight_decrease (initial_count : ℕ) (initial_avg : ℝ) (new_weight : ℝ) : 
  initial_count = 20 → 
  initial_avg = 57 → 
  new_weight = 48 → 
  let new_avg := (initial_count * initial_avg + new_weight) / (initial_count + 1)
  initial_avg - new_avg = 0.43 := by sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2868_286842


namespace NUMINAMATH_CALUDE_shaded_area_hexagon_with_semicircles_l2868_286821

/-- The area of the shaded region in a regular hexagon with inscribed semicircles -/
theorem shaded_area_hexagon_with_semicircles (s : ℝ) (h : s = 3) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area := π * (s/2)^2 / 2
  let total_semicircle_area := 3 * semicircle_area
  hexagon_area - total_semicircle_area = 13.5 * Real.sqrt 3 - 27 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_hexagon_with_semicircles_l2868_286821


namespace NUMINAMATH_CALUDE_base_nine_digits_of_2048_l2868_286872

theorem base_nine_digits_of_2048 : ∃ n : ℕ, n > 0 ∧ 9^(n-1) ≤ 2048 ∧ 2048 < 9^n :=
by sorry

end NUMINAMATH_CALUDE_base_nine_digits_of_2048_l2868_286872


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2868_286896

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)
  Complex.im z = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2868_286896


namespace NUMINAMATH_CALUDE_continuous_fraction_value_l2868_286820

theorem continuous_fraction_value :
  ∃ x : ℝ, x = 1 + 1 / (2 + 1 / x) ∧ x = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_value_l2868_286820


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l2868_286855

theorem expand_polynomial_product : 
  ∀ x : ℝ, (3*x^2 - 2*x + 4) * (4*x^2 + 3*x - 6) = 12*x^4 + x^3 - 8*x^2 + 24*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l2868_286855


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2868_286846

theorem unique_square_divisible_by_three_in_range : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧
  x % 3 = 0 ∧
  60 < x ∧ x < 130 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2868_286846


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2868_286879

theorem complement_intersection_equals_set (U M N : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 3, 4} →
  N = {2, 4, 5} →
  (U \ M) ∩ N = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2868_286879


namespace NUMINAMATH_CALUDE_cylinder_intersection_angle_l2868_286807

theorem cylinder_intersection_angle (r b a : ℝ) (h_r : r = 1) (h_b : b = r) 
  (h_e : (Real.sqrt 5) / 3 = Real.sqrt (1 - (b / a)^2)) :
  Real.arccos (2 / 3) = Real.arccos (b / a) := by sorry

end NUMINAMATH_CALUDE_cylinder_intersection_angle_l2868_286807


namespace NUMINAMATH_CALUDE_circle_radius_satisfies_condition_l2868_286800

/-- The radius of a circle satisfying the given condition -/
def circle_radius : ℝ := 8

/-- The condition that the product of four inches and the circumference equals the area -/
def circle_condition (r : ℝ) : Prop := 4 * (2 * Real.pi * r) = Real.pi * r^2

/-- Theorem stating that the radius satisfies the condition -/
theorem circle_radius_satisfies_condition : 
  circle_condition circle_radius := by sorry

end NUMINAMATH_CALUDE_circle_radius_satisfies_condition_l2868_286800


namespace NUMINAMATH_CALUDE_cyclic_power_inequality_l2868_286811

theorem cyclic_power_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4*b + b^4*c + c^4*d + d^4*a ≥ a*b*c*d*(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_power_inequality_l2868_286811


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2868_286831

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ S ↔ x^2 - y^2 = 1988) ∧ 
    Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2868_286831
