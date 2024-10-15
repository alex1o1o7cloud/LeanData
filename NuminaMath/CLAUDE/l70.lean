import Mathlib

namespace NUMINAMATH_CALUDE_james_has_43_oreos_l70_7092

/-- The number of Oreos James has -/
def james_oreos (jordan_oreos : ℕ) : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_has_43_oreos :
  ∃ (jordan_oreos : ℕ), 
    james_oreos jordan_oreos + jordan_oreos = total_oreos ∧
    james_oreos jordan_oreos = 43 := by
  sorry

end NUMINAMATH_CALUDE_james_has_43_oreos_l70_7092


namespace NUMINAMATH_CALUDE_mikeys_leaves_mikeys_leaves_specific_l70_7096

/-- The number of leaves Mikey has after receiving more leaves -/
def total_leaves (initial : ℝ) (new : ℝ) : ℝ :=
  initial + new

/-- Theorem stating that Mikey's total leaves is the sum of initial and new leaves -/
theorem mikeys_leaves (initial : ℝ) (new : ℝ) :
  total_leaves initial new = initial + new := by
  sorry

/-- Specific instance of Mikey's leaves problem -/
theorem mikeys_leaves_specific :
  total_leaves 356.0 112.0 = 468.0 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_mikeys_leaves_specific_l70_7096


namespace NUMINAMATH_CALUDE_last_digit_of_n_l70_7019

/-- Represents a natural number with its digits -/
structure DigitNumber where
  value : ℕ
  num_digits : ℕ
  greater_than_ten : value > 10

/-- Represents the transformation from N to M -/
structure Transformation where
  increase_by_two : ℕ  -- position of the digit increased by 2
  increase_by_odd : List ℕ  -- list of odd numbers added to other digits

/-- Main theorem statement -/
theorem last_digit_of_n (N M : DigitNumber) (t : Transformation) :
  M.value = 3 * N.value →
  M.num_digits = N.num_digits →
  (∃ (transformed_N : ℕ), transformed_N = N.value + t.increase_by_two + t.increase_by_odd.sum) →
  N.value % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_n_l70_7019


namespace NUMINAMATH_CALUDE_special_function_max_l70_7016

open Real

/-- A continuous function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x y, f (x + y) * f (x - y) = f x ^ 2 - f y ^ 2) ∧
  (∀ x, f (x + 2 * π) = f x) ∧
  (∀ a, 0 < a → a < 2 * π → ∃ x, f (x + a) ≠ f x)

/-- The main theorem to be proved -/
theorem special_function_max (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, |f (π / 2)| ≥ f x :=
sorry

end NUMINAMATH_CALUDE_special_function_max_l70_7016


namespace NUMINAMATH_CALUDE_two_heroes_two_villains_l70_7064

/-- Represents the type of an inhabitant -/
inductive InhabitantType
| Hero
| Villain

/-- Represents an inhabitant on the island -/
structure Inhabitant where
  type : InhabitantType

/-- Represents the table with four inhabitants -/
structure Table where
  inhabitants : Fin 4 → Inhabitant

/-- Defines what it means for an inhabitant to tell the truth -/
def tellsTruth (i : Inhabitant) : Prop :=
  i.type = InhabitantType.Hero

/-- Defines what an inhabitant says about themselves -/
def claimsSelfHero (i : Inhabitant) : Prop :=
  true

/-- Defines what an inhabitant says about the person on their right -/
def claimsRightVillain (t : Table) (pos : Fin 4) : Prop :=
  true

/-- The main theorem stating that the only valid configuration is 2 Heroes and 2 Villains alternating -/
theorem two_heroes_two_villains (t : Table) :
  (∀ (pos : Fin 4), claimsSelfHero (t.inhabitants pos)) →
  (∀ (pos : Fin 4), claimsRightVillain t pos) →
  (∃ (pos : Fin 4),
    tellsTruth (t.inhabitants pos) ∧
    ¬tellsTruth (t.inhabitants (pos + 1)) ∧
    tellsTruth (t.inhabitants (pos + 2)) ∧
    ¬tellsTruth (t.inhabitants (pos + 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_two_heroes_two_villains_l70_7064


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l70_7004

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube : 
  (∀ y : ℕ, y > 0 ∧ y < 7350 → ¬ is_perfect_cube (1260 * y)) ∧ 
  is_perfect_cube (1260 * 7350) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l70_7004


namespace NUMINAMATH_CALUDE_vector_sum_and_scalar_mult_l70_7017

/-- Prove that the sum of the vector (3, -2, 5) and 2 times the vector (-1, 4, -3) is equal to the vector (1, 6, -1). -/
theorem vector_sum_and_scalar_mult :
  let v₁ : Fin 3 → ℝ := ![3, -2, 5]
  let v₂ : Fin 3 → ℝ := ![-1, 4, -3]
  v₁ + 2 • v₂ = ![1, 6, -1] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_and_scalar_mult_l70_7017


namespace NUMINAMATH_CALUDE_circle_op_inequality_l70_7054

def circle_op (x y : ℝ) : ℝ := x * (1 - y)

theorem circle_op_inequality (a : ℝ) : 
  (∀ x : ℝ, circle_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_inequality_l70_7054


namespace NUMINAMATH_CALUDE_kite_area_theorem_l70_7067

/-- A symmetrical quadrilateral kite -/
structure Kite where
  base : ℝ
  height : ℝ

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := k.base * k.height

/-- Theorem: The area of a kite with base 35 and height 15 is 525 -/
theorem kite_area_theorem (k : Kite) (h1 : k.base = 35) (h2 : k.height = 15) :
  kite_area k = 525 := by
  sorry

#check kite_area_theorem

end NUMINAMATH_CALUDE_kite_area_theorem_l70_7067


namespace NUMINAMATH_CALUDE_four_digit_equation_solutions_l70_7010

/-- Represents a four-digit number ABCD as a pair of two-digit numbers (AB, CD) -/
def FourDigitNumber := Nat × Nat

/-- Checks if a pair of numbers represents a valid four-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  10 ≤ n.1 ∧ n.1 ≤ 99 ∧ 10 ≤ n.2 ∧ n.2 ≤ 99

/-- Converts a pair of two-digit numbers to a four-digit number -/
def toNumber (n : FourDigitNumber) : Nat :=
  100 * n.1 + n.2

/-- The equation that the four-digit number must satisfy -/
def satisfiesEquation (n : FourDigitNumber) : Prop :=
  toNumber n = n.1 * n.2 + n.1 * n.1

theorem four_digit_equation_solutions :
  ∀ n : FourDigitNumber, 
    isValidFourDigitNumber n ∧ satisfiesEquation n ↔ 
    n = (12, 96) ∨ n = (34, 68) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equation_solutions_l70_7010


namespace NUMINAMATH_CALUDE_carl_teaches_six_periods_l70_7081

-- Define the given conditions
def cards_per_student : ℕ := 10
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3
def total_spent : ℚ := 108

-- Define the number of periods
def periods : ℕ := 6

-- Theorem statement
theorem carl_teaches_six_periods :
  (total_spent / cost_per_pack) * cards_per_pack =
  periods * students_per_class * cards_per_student :=
by sorry

end NUMINAMATH_CALUDE_carl_teaches_six_periods_l70_7081


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l70_7043

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * seq.a₁ + (k - 1) * seq.d) / 2

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * (seq.a₁ + (seq.n - k) * seq.d) + (k - 1) * seq.d) / 2

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n * (2 * seq.a₁ + (seq.n - 1) * seq.d) / 2

theorem arithmetic_sequence_unique_n (seq : ArithmeticSequence) :
  sum_first_k seq 4 = 40 →
  sum_last_k seq 4 = 80 →
  sum_all seq = 210 →
  seq.n = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l70_7043


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_upper_bound_l70_7003

open Real

theorem function_inequality_implies_a_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_f : ∀ x, f x = x - a * x * log x) :
  (∃ x₀ ∈ Set.Icc (exp 1) (exp 2), f x₀ ≤ (1/4) * log x₀) →
  a ≤ 1 - 1 / (4 * exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_upper_bound_l70_7003


namespace NUMINAMATH_CALUDE_even_mono_increasing_inequality_l70_7045

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is monotonically increasing on [0, +∞) if f(x) ≤ f(y) for 0 ≤ x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem even_mono_increasing_inequality (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_mono : MonoIncreasing f) : 
    f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_increasing_inequality_l70_7045


namespace NUMINAMATH_CALUDE_x_range_for_quartic_equation_l70_7098

theorem x_range_for_quartic_equation (k x : ℝ) :
  x^4 - 2*k*x^2 + k^2 + 2*k - 3 = 0 → -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_quartic_equation_l70_7098


namespace NUMINAMATH_CALUDE_equation_solution_l70_7026

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l70_7026


namespace NUMINAMATH_CALUDE_abc_inequality_l70_7041

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l70_7041


namespace NUMINAMATH_CALUDE_quadratic_function_solution_set_l70_7079

/-- Given a quadratic function f(x) = ax^2 - (a+2)x - b, where a and b are real numbers,
    if the solution set of f(x) > 0 is (-3,2), then a + b = -7. -/
theorem quadratic_function_solution_set (a b : ℝ) :
  (∀ x, (a * x^2 - (a + 2) * x - b > 0) ↔ (-3 < x ∧ x < 2)) →
  a + b = -7 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_solution_set_l70_7079


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l70_7085

/-- Given two lines that pass through a common point, prove that the line passing through
    the points defined by the coefficients of these lines has a specific equation. -/
theorem line_through_coefficient_points (a₁ a₂ b₁ b₂ : ℝ) : 
  (a₁ * 2 + b₁ * 3 + 1 = 0) →
  (a₂ * 2 + b₂ * 3 + 1 = 0) →
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l70_7085


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l70_7035

def blue_eggs : ℕ := 12
def pink_eggs : ℕ := 5
def golden_eggs : ℕ := 3

def blue_points : ℕ := 2
def pink_points : ℕ := 3
def golden_points : ℕ := 5

def total_people : ℕ := 4

theorem easter_egg_distribution :
  let total_points := blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points
  (total_points / total_people = 13) ∧ (total_points % total_people = 2) := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l70_7035


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l70_7075

/-- A Mersenne number is of the form 2^p - 1 for some positive integer p -/
def mersenne_number (p : ℕ) : ℕ := 2^p - 1

/-- A Mersenne prime is a Mersenne number that is also prime -/
def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = mersenne_number p ∧ Nat.Prime n

theorem largest_mersenne_prime_under_500 :
  ∀ n : ℕ, is_mersenne_prime n ∧ n < 500 → n ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l70_7075


namespace NUMINAMATH_CALUDE_gianna_savings_l70_7089

def total_savings : ℕ := 14235
def days_in_year : ℕ := 365
def daily_savings : ℚ := total_savings / days_in_year

theorem gianna_savings : daily_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l70_7089


namespace NUMINAMATH_CALUDE_sum_236_83_base4_l70_7099

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 4 representation -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of 236 and 83 in base 4 is [1, 3, 3, 2, 3] -/
theorem sum_236_83_base4 :
  addBase4 (toBase4 236) (toBase4 83) = [1, 3, 3, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_236_83_base4_l70_7099


namespace NUMINAMATH_CALUDE_benny_seashells_l70_7076

/-- Represents the number of seashells Benny has after giving some to Jason -/
def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proves that Benny has 14 seashells remaining -/
theorem benny_seashells : seashells_remaining 66 52 = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l70_7076


namespace NUMINAMATH_CALUDE_count_valid_words_l70_7015

/-- The number of letters in each word -/
def word_length : ℕ := 4

/-- The number of available letters -/
def alphabet_size : ℕ := 5

/-- The number of letters that must be included -/
def required_letters : ℕ := 2

/-- The number of 4-letter words that can be formed using the letters A, B, C, D, and E, 
    with repetition allowed, and including both A and E at least once -/
def valid_words : ℕ := alphabet_size^word_length - 2*(alphabet_size-1)^word_length + (alphabet_size-2)^word_length

theorem count_valid_words : valid_words = 194 := by sorry

end NUMINAMATH_CALUDE_count_valid_words_l70_7015


namespace NUMINAMATH_CALUDE_tetrahedron_passage_l70_7006

/-- The minimal radius through which a regular tetrahedron with edge length 1 can pass -/
def min_radius : ℝ := 0.4478

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- A circular hole -/
structure CircularHole where
  radius : ℝ

/-- Predicate for whether a tetrahedron can pass through a hole -/
def can_pass_through (t : RegularTetrahedron) (h : CircularHole) : Prop :=
  h.radius ≥ min_radius

/-- Theorem stating the condition for a regular tetrahedron to pass through a circular hole -/
theorem tetrahedron_passage (t : RegularTetrahedron) (h : CircularHole) :
  can_pass_through t h ↔ h.radius ≥ min_radius :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_passage_l70_7006


namespace NUMINAMATH_CALUDE_part_one_part_two_l70_7056

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Part 1
theorem part_one (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x ≥ 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2) : 
  a = 2 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x + |x - 3| ≥ 1) : 
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l70_7056


namespace NUMINAMATH_CALUDE_curve_symmetry_condition_l70_7008

/-- Given a curve y = (px + q) / (rx - s) with nonzero p, q, r, s,
    if y = -x is an axis of symmetry, then r + s = 0 -/
theorem curve_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x - s) ↔ (-x) = (p * (-y) + q) / (r * (-y) - s)) →
  r + s = 0 :=
by sorry

end NUMINAMATH_CALUDE_curve_symmetry_condition_l70_7008


namespace NUMINAMATH_CALUDE_cube_root_squared_times_fifth_root_l70_7022

theorem cube_root_squared_times_fifth_root (x : ℝ) (h : x > 0) :
  (x^(1/3))^2 * x^(1/5) = x^(13/15) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_squared_times_fifth_root_l70_7022


namespace NUMINAMATH_CALUDE_tailor_time_ratio_l70_7059

theorem tailor_time_ratio (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) 
  (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (pants_time : ℚ), 
    pants_time / shirt_time = 2 ∧
    total_cost = hourly_rate * (num_shirts * shirt_time + num_pants * pants_time) :=
by sorry

end NUMINAMATH_CALUDE_tailor_time_ratio_l70_7059


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l70_7024

theorem triangle_side_lengths (n : ℕ) : 
  (n + 4 + n + 10 > 4*n + 7) ∧ 
  (n + 4 + 4*n + 7 > n + 10) ∧ 
  (n + 10 + 4*n + 7 > n + 4) ∧ 
  (4*n + 7 > n + 10) ∧ 
  (n + 10 > n + 4) →
  (∃ (count : ℕ), count = 2 ∧ 
    (∀ m : ℕ, (m + 4 + m + 10 > 4*m + 7) ∧ 
              (m + 4 + 4*m + 7 > m + 10) ∧ 
              (m + 10 + 4*m + 7 > m + 4) ∧ 
              (4*m + 7 > m + 10) ∧ 
              (m + 10 > m + 4) ↔ 
              (m = n ∨ m = n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l70_7024


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l70_7042

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial : trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l70_7042


namespace NUMINAMATH_CALUDE_factor_expression_l70_7044

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l70_7044


namespace NUMINAMATH_CALUDE_marked_price_calculation_l70_7048

/-- Given a pair of articles bought for $50 with a 30% discount, 
    prove that the marked price of each article is 50 / 1.4 -/
theorem marked_price_calculation (total_price : ℝ) (discount_percent : ℝ) 
    (h1 : total_price = 50)
    (h2 : discount_percent = 30) : 
  (total_price / (2 * (1 - discount_percent / 100))) = 50 / 1.4 := by
  sorry

#eval (50 : Float) / 1.4

end NUMINAMATH_CALUDE_marked_price_calculation_l70_7048


namespace NUMINAMATH_CALUDE_volume_of_T_l70_7023

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of T is 32/3 -/
theorem volume_of_T : volume T = 32/3 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l70_7023


namespace NUMINAMATH_CALUDE_martha_apples_theorem_l70_7080

def apples_to_give_away (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) (final_apples : ℕ) : ℕ :=
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - final_apples

theorem martha_apples_theorem :
  apples_to_give_away 20 5 2 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_martha_apples_theorem_l70_7080


namespace NUMINAMATH_CALUDE_equation_solutions_l70_7094

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁ = 1 ∧ x₂ = 4) ∧ 
    (∀ x : ℝ, (x - 1)^2 = 3*(x - 1) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ = 2 + Real.sqrt 3 ∧ y₂ = 2 - Real.sqrt 3) ∧ 
    (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l70_7094


namespace NUMINAMATH_CALUDE_triangle_median_altitude_equations_l70_7069

/-- Triangle ABC with vertices A(-5, 0), B(4, -4), and C(0, 2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-5, 0)
    B := (4, -4)
    C := (0, 2) }

/-- The equation of the line on which the median to side BC lies -/
def median_equation (t : Triangle) : LineEquation :=
  { a := 1
    b := 7
    c := 5 }

/-- The equation of the line on which the altitude from A to side BC lies -/
def altitude_equation (t : Triangle) : LineEquation :=
  { a := 2
    b := -3
    c := 10 }

theorem triangle_median_altitude_equations :
  (median_equation triangle_ABC).a = 1 ∧
  (median_equation triangle_ABC).b = 7 ∧
  (median_equation triangle_ABC).c = 5 ∧
  (altitude_equation triangle_ABC).a = 2 ∧
  (altitude_equation triangle_ABC).b = -3 ∧
  (altitude_equation triangle_ABC).c = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_altitude_equations_l70_7069


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l70_7095

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (k a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x y : ℝ), k * x + y - Real.sqrt 2 * k = 0 → 
      x^2 / a^2 - y^2 / b^2 = 1 → 
      (∃ (m : ℝ), y = m * x ∧ abs (Real.sqrt 2 * k / Real.sqrt (1 + k^2)) = 4/3))) → 
  Real.sqrt (1 + b^2 / a^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l70_7095


namespace NUMINAMATH_CALUDE_range_of_a_l70_7029

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l70_7029


namespace NUMINAMATH_CALUDE_prob_black_then_red_is_15_59_l70_7087

/-- A deck of cards with specific properties -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = 60)
  (h_black : black = 30)
  (h_red : red = 30)
  (h_sum : black + red = total)

/-- The probability of drawing a black card first and a red card second -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * d.red / (d.total - 1)

/-- Theorem stating the probability is equal to 15/59 -/
theorem prob_black_then_red_is_15_59 (d : Deck) :
  prob_black_then_red d = 15 / 59 := by
  sorry


end NUMINAMATH_CALUDE_prob_black_then_red_is_15_59_l70_7087


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l70_7002

theorem oak_trees_in_park (x : ℕ) : x + 4 = 9 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l70_7002


namespace NUMINAMATH_CALUDE_jack_needs_more_money_l70_7037

def sock_price : ℝ := 9.50
def shoe_price : ℝ := 92
def jack_money : ℝ := 40
def num_socks : ℕ := 2

theorem jack_needs_more_money :
  let total_cost := num_socks * sock_price + shoe_price
  total_cost - jack_money = 71 := by sorry

end NUMINAMATH_CALUDE_jack_needs_more_money_l70_7037


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l70_7018

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_2 : a + b + c = 2) : 
  1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) ≥ 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l70_7018


namespace NUMINAMATH_CALUDE_helen_raisin_cookie_difference_l70_7046

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_today

theorem helen_raisin_cookie_difference : raisin_cookie_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookie_difference_l70_7046


namespace NUMINAMATH_CALUDE_range_of_a_l70_7001

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x y : ℝ, x < y → (2 * a - 1) ^ x > (2 * a - 1) ^ y) →
  1/2 < a ∧ a ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l70_7001


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l70_7066

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l70_7066


namespace NUMINAMATH_CALUDE_set_operations_l70_7031

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x < 7}) ∧
  (Aᶜ = {x | x < 3 ∨ 7 ≤ x}) ∧
  ((A ∪ B)ᶜ = {x | x ≤ 2 ∨ 10 ≤ x}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l70_7031


namespace NUMINAMATH_CALUDE_line_through_points_l70_7065

/-- Given a line y = cx + d passing through the points (3, -3) and (6, 9), prove that c + d = -11 -/
theorem line_through_points (c d : ℝ) : 
  (-3 : ℝ) = c * 3 + d → 
  9 = c * 6 + d → 
  c + d = -11 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l70_7065


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l70_7052

def U : Finset ℕ := {1,2,3,4,5,6}
def A : Finset ℕ := {2,4,5}
def B : Finset ℕ := {1,3}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l70_7052


namespace NUMINAMATH_CALUDE_circular_seating_sum_l70_7040

theorem circular_seating_sum (n : ℕ) (h : n ≥ 3) :
  (∀ (girl_sum : ℕ → ℕ) (boy_cards : ℕ → ℕ) (girl_cards : ℕ → ℕ),
    (∀ i : ℕ, i ∈ Finset.range n → 1 ≤ boy_cards i ∧ boy_cards i ≤ n) →
    (∀ i : ℕ, i ∈ Finset.range n → n + 1 ≤ girl_cards i ∧ girl_cards i ≤ 2*n) →
    (∀ i : ℕ, i ∈ Finset.range n → 
      girl_sum i = girl_cards i + boy_cards i + boy_cards ((i + 1) % n)) →
    (∀ i j : ℕ, i ∈ Finset.range n → j ∈ Finset.range n → girl_sum i = girl_sum j)) ↔
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_circular_seating_sum_l70_7040


namespace NUMINAMATH_CALUDE_percentage_problem_l70_7097

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 660 = 12 / 100 * 1500 - 15 → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l70_7097


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l70_7012

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 10) :
  let a := S * (1 - r)
  a = 40/3 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l70_7012


namespace NUMINAMATH_CALUDE_probability_ratio_l70_7053

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_two (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_number 3 * Nat.choose per_number 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l70_7053


namespace NUMINAMATH_CALUDE_fill_pipe_fraction_l70_7072

/-- Represents the fraction of a cistern that can be filled in a given time -/
def FractionFilled (time : ℝ) : ℝ := sorry

theorem fill_pipe_fraction :
  let fill_time : ℝ := 30
  let fraction := FractionFilled fill_time
  (∃ (f : ℝ), FractionFilled fill_time = f ∧ f * fill_time = fill_time) →
  fraction = 1 := by sorry

end NUMINAMATH_CALUDE_fill_pipe_fraction_l70_7072


namespace NUMINAMATH_CALUDE_pencil_count_problem_l70_7084

/-- The number of pencils in a drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (sara_adds : ℕ) (john_adds : ℕ) (ben_removes : ℕ) : ℕ :=
  initial + sara_adds + john_adds - ben_removes

/-- Theorem stating that given the initial number of pencils and the changes made by Sara, John, and Ben, the final number of pencils is 245. -/
theorem pencil_count_problem : final_pencil_count 115 100 75 45 = 245 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_problem_l70_7084


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l70_7013

theorem sin_2alpha_value (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l70_7013


namespace NUMINAMATH_CALUDE_triangle_parallelepiped_analogy_inappropriate_l70_7025

/-- A shape in a geometric space -/
inductive GeometricShape
  | Triangle
  | Parallelepiped
  | TriangularPyramid

/-- The dimension of a geometric space -/
inductive Dimension
  | Plane
  | Space

/-- A function that determines if two shapes form an appropriate analogy across dimensions -/
def appropriateAnalogy (shape1 : GeometricShape) (dim1 : Dimension) 
                       (shape2 : GeometricShape) (dim2 : Dimension) : Prop :=
  sorry

/-- Theorem stating that comparing a triangle in a plane to a parallelepiped in space 
    is not an appropriate analogy -/
theorem triangle_parallelepiped_analogy_inappropriate :
  ¬(appropriateAnalogy GeometricShape.Triangle Dimension.Plane 
                       GeometricShape.Parallelepiped Dimension.Space) :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallelepiped_analogy_inappropriate_l70_7025


namespace NUMINAMATH_CALUDE_min_value_of_y_l70_7011

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x z : ℝ, x > 0 → z > 0 → x + z = 2 → 1/x + 4/z ≥ 1/a + 4/b) ∧ 1/a + 4/b = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l70_7011


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l70_7083

def smallest_two_digit_multiple_of_3 : ℕ → Prop :=
  λ n => n ≥ 10 ∧ n < 100 ∧ 3 ∣ n ∧ ∀ m, m ≥ 10 ∧ m < 100 ∧ 3 ∣ m → n ≤ m

def smallest_three_digit_multiple_of_4 : ℕ → Prop :=
  λ n => n ≥ 100 ∧ n < 1000 ∧ 4 ∣ n ∧ ∀ m, m ≥ 100 ∧ m < 1000 ∧ 4 ∣ m → n ≤ m

theorem sum_of_smallest_multiples : 
  ∀ a b : ℕ, smallest_two_digit_multiple_of_3 a → smallest_three_digit_multiple_of_4 b → 
  a + b = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l70_7083


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l70_7007

theorem min_sum_absolute_values (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| + 2 ≥ 8 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| + 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l70_7007


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l70_7071

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 - 3*z + 2) ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l70_7071


namespace NUMINAMATH_CALUDE_exists_favorable_config_for_second_player_l70_7088

/-- Represents a square on the game board -/
structure Square :=
  (hasArrow : Bool)

/-- Represents the game board -/
def Board := List Square

/-- Calculates the probability of the second player winning given a board configuration and game parameters -/
def secondPlayerWinProbability (board : Board) (s₁ : ℕ) (s₂ : ℕ) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that there exists a board configuration where the second player has a winning probability greater than 1/2, even when s₁ > s₂ -/
theorem exists_favorable_config_for_second_player :
  ∃ (board : Board) (s₁ s₂ : ℕ), s₁ > s₂ ∧ secondPlayerWinProbability board s₁ s₂ > (1/2 : ℝ) :=
sorry


end NUMINAMATH_CALUDE_exists_favorable_config_for_second_player_l70_7088


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l70_7033

theorem difference_of_squares_example (x y : ℤ) (hx : x = 12) (hy : y = 5) :
  (x - y) * (x + y) = 119 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l70_7033


namespace NUMINAMATH_CALUDE_equal_boy_girl_division_theorem_l70_7047

/-- Represents a student arrangement as a list of integers, where 1 represents a boy and -1 represents a girl -/
def StudentArrangement := List Int

/-- Checks if a given arrangement can be divided into two parts with equal number of boys and girls -/
def canBeDivided (arrangement : StudentArrangement) : Bool :=
  sorry

/-- Counts the number of arrangements where division is impossible -/
def countImpossibleDivisions (n : Nat) : Nat :=
  sorry

/-- Counts the number of arrangements where exactly one division is possible -/
def countSingleDivisions (n : Nat) : Nat :=
  sorry

theorem equal_boy_girl_division_theorem (n : Nat) (h : n ≥ 2) :
  countSingleDivisions (2 * n) = 2 * countImpossibleDivisions (2 * n) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_boy_girl_division_theorem_l70_7047


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l70_7009

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.base = 4 ∧ t.leg^2 - 5*t.leg + 6 = 0

-- Define the perimeter
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, triangle_conditions t → perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l70_7009


namespace NUMINAMATH_CALUDE_cats_not_liking_either_l70_7060

theorem cats_not_liking_either (total : ℕ) (cheese : ℕ) (tuna : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cheese : cheese = 25)
  (h_tuna : tuna = 70)
  (h_both : both = 15) :
  total - (cheese + tuna - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_cats_not_liking_either_l70_7060


namespace NUMINAMATH_CALUDE_apple_orchard_problem_l70_7082

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧ 
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem apple_orchard_problem :
  ∀ o : Orchard, orchard_conditions o → o.pure_gala = 27 := by
  sorry

end NUMINAMATH_CALUDE_apple_orchard_problem_l70_7082


namespace NUMINAMATH_CALUDE_coupon_value_l70_7039

/-- Calculates the value of a coupon for eyeglass frames -/
theorem coupon_value (frame_cost lens_cost insurance_percentage total_cost_after : ℚ) : 
  frame_cost = 200 →
  lens_cost = 500 →
  insurance_percentage = 80 / 100 →
  total_cost_after = 250 →
  (frame_cost + lens_cost * (1 - insurance_percentage)) - total_cost_after = 50 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l70_7039


namespace NUMINAMATH_CALUDE_sum_of_integers_l70_7014

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 145) 
  (h2 : x.val * y.val = 72) : 
  x.val + y.val = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l70_7014


namespace NUMINAMATH_CALUDE_range_of_a_l70_7027

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l70_7027


namespace NUMINAMATH_CALUDE_min_sum_squares_l70_7093

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l70_7093


namespace NUMINAMATH_CALUDE_mihaly_third_day_foxes_l70_7086

/-- Represents the number of animals hunted by a person on a specific day -/
structure DailyHunt where
  rabbits : ℕ
  foxes : ℕ
  pheasants : ℕ

/-- Represents the total hunt over three days for a person -/
structure ThreeDayHunt where
  day1 : DailyHunt
  day2 : DailyHunt
  day3 : DailyHunt

def Karoly : ThreeDayHunt := sorry
def Laszlo : ThreeDayHunt := sorry
def Mihaly : ThreeDayHunt := sorry

def total_animals : ℕ := 86

def first_day_foxes : ℕ := 12
def first_day_rabbits : ℕ := 14

def second_day_total : ℕ := 44

def total_pheasants : ℕ := 12

theorem mihaly_third_day_foxes :
  (∀ d : DailyHunt, d.rabbits ≥ 1 ∧ d.foxes ≥ 1 ∧ d.pheasants ≥ 1) →
  (∀ d : DailyHunt, d ≠ Laszlo.day2 → Even d.rabbits ∧ Even d.foxes ∧ Even d.pheasants) →
  Laszlo.day2.foxes = 5 →
  (Karoly.day1.foxes + Laszlo.day1.foxes + Mihaly.day1.foxes = first_day_foxes) →
  (Karoly.day1.rabbits + Laszlo.day1.rabbits + Mihaly.day1.rabbits = first_day_rabbits) →
  (Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants = second_day_total) →
  (Karoly.day1.pheasants + Karoly.day2.pheasants + Karoly.day3.pheasants +
   Laszlo.day1.pheasants + Laszlo.day2.pheasants + Laszlo.day3.pheasants +
   Mihaly.day1.pheasants + Mihaly.day2.pheasants + Mihaly.day3.pheasants = total_pheasants) →
  (Karoly.day1.rabbits + Karoly.day1.foxes + Karoly.day1.pheasants +
   Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Karoly.day3.rabbits + Karoly.day3.foxes + Karoly.day3.pheasants +
   Laszlo.day1.rabbits + Laszlo.day1.foxes + Laszlo.day1.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Laszlo.day3.rabbits + Laszlo.day3.foxes + Laszlo.day3.pheasants +
   Mihaly.day1.rabbits + Mihaly.day1.foxes + Mihaly.day1.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants +
   Mihaly.day3.rabbits + Mihaly.day3.foxes + Mihaly.day3.pheasants = total_animals) →
  Mihaly.day3.foxes = 1 := by
  sorry

end NUMINAMATH_CALUDE_mihaly_third_day_foxes_l70_7086


namespace NUMINAMATH_CALUDE_base_5_to_base_7_conversion_l70_7074

def base_5_to_decimal (n : ℕ) : ℕ := 
  2 * 5^0 + 1 * 5^1 + 4 * 5^2

def decimal_to_base_7 (n : ℕ) : List ℕ :=
  [2, 1, 2]

theorem base_5_to_base_7_conversion :
  decimal_to_base_7 (base_5_to_decimal 412) = [2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_base_7_conversion_l70_7074


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l70_7049

def repeating_decimal_1 : ℚ := 1 / 9
def repeating_decimal_2 : ℚ := 2 / 99
def repeating_decimal_3 : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 134 / 999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l70_7049


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_squared_l70_7021

theorem roots_reciprocal_sum_squared (a b c r s : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hr : a * r^2 + b * r + c = 0) (hs : a * s^2 + b * s + c = 0) :
  1 / r^2 + 1 / s^2 = (b^2 - 2*a*c) / c^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_squared_l70_7021


namespace NUMINAMATH_CALUDE_stick_pieces_l70_7034

theorem stick_pieces (n₁ n₂ : ℕ) (h₁ : n₁ = 12) (h₂ : n₂ = 18) : 
  (n₁ - 1) + (n₂ - 1) - (n₁.lcm n₂ / n₁.gcd n₂ - 1) + 1 = 24 := by sorry

end NUMINAMATH_CALUDE_stick_pieces_l70_7034


namespace NUMINAMATH_CALUDE_teachers_on_field_trip_l70_7030

theorem teachers_on_field_trip 
  (num_students : ℕ) 
  (student_ticket_cost : ℕ) 
  (teacher_ticket_cost : ℕ) 
  (total_cost : ℕ) :
  num_students = 12 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (num_teachers : ℕ), 
    num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = total_cost ∧
    num_teachers = 4 := by
sorry

end NUMINAMATH_CALUDE_teachers_on_field_trip_l70_7030


namespace NUMINAMATH_CALUDE_locus_of_P_l70_7051

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y - 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define point Q
def Q : ℝ × ℝ := (2, -1)

-- Define the condition for a point P to be on the locus
def on_locus (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 - 1 = 0 ∧ P ≠ (3, 4) ∧ P ≠ (-2, -3) ∧ P ≠ (1, 0)

-- State the theorem
theorem locus_of_P (P A B : ℝ × ℝ) :
  l₁ A.1 A.2 →
  l₂ B.1 B.2 →
  (∃ (t : ℝ), P = (1 - t) • A + t • B) →
  P ≠ Q →
  (P.1 - A.1) / (B.1 - P.1) = (Q.1 - A.1) / (B.1 - Q.1) →
  (P.2 - A.2) / (B.2 - P.2) = (Q.2 - A.2) / (B.2 - Q.2) →
  on_locus P :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_l70_7051


namespace NUMINAMATH_CALUDE_extra_flowers_l70_7070

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l70_7070


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l70_7050

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l70_7050


namespace NUMINAMATH_CALUDE_number_of_recitation_orders_l70_7061

/-- The number of high school seniors --/
def total_students : ℕ := 7

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of special students (A, B, C) --/
def special_students : ℕ := 3

/-- Function to calculate the number of recitation orders --/
def recitation_orders : ℕ := sorry

/-- Theorem stating the number of recitation orders --/
theorem number_of_recitation_orders :
  recitation_orders = 768 := by sorry

end NUMINAMATH_CALUDE_number_of_recitation_orders_l70_7061


namespace NUMINAMATH_CALUDE_boxes_sold_theorem_l70_7062

/-- Represents the number of boxes sold on each day --/
structure BoxesSold where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total number of boxes sold over three days --/
def totalBoxesSold (boxes : BoxesSold) : ℕ :=
  boxes.friday + boxes.saturday + boxes.sunday

/-- Theorem stating the total number of boxes sold over three days --/
theorem boxes_sold_theorem (boxes : BoxesSold) 
  (h1 : boxes.friday = 40)
  (h2 : boxes.saturday = 2 * boxes.friday - 10)
  (h3 : boxes.sunday = boxes.saturday / 2) :
  totalBoxesSold boxes = 145 := by
  sorry

#check boxes_sold_theorem

end NUMINAMATH_CALUDE_boxes_sold_theorem_l70_7062


namespace NUMINAMATH_CALUDE_mango_buying_rate_l70_7068

/-- Represents the rate at which mangoes are bought and sold -/
structure MangoRate where
  buy : ℚ  -- Buying rate (rupees per x mangoes)
  sell : ℚ  -- Selling rate (mangoes per rupee)

/-- Calculates the profit percentage given buying and selling rates -/
def profit_percentage (rate : MangoRate) : ℚ :=
  (rate.sell⁻¹ / rate.buy - 1) * 100

/-- Proves that the buying rate is 2 rupees for x mangoes given the conditions -/
theorem mango_buying_rate (rate : MangoRate) 
  (h_sell : rate.sell = 3)
  (h_profit : profit_percentage rate = 50) :
  rate.buy = 2 := by
  sorry

end NUMINAMATH_CALUDE_mango_buying_rate_l70_7068


namespace NUMINAMATH_CALUDE_free_throw_contest_ratio_l70_7073

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  sandra = 3 * alex →
  alex + sandra + hector = 80 →
  hector / sandra = 2 := by
sorry

end NUMINAMATH_CALUDE_free_throw_contest_ratio_l70_7073


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l70_7020

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  has_top_icing : Bool
  has_side_icing : Bool
  middle_icing_height : Rat

/-- Counts cubes with exactly two iced sides in an iced cake -/
def count_double_iced_cubes (cake : IcedCake) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem double_iced_cubes_count (cake : IcedCake) : 
  cake.size = 5 ∧ 
  cake.has_top_icing = true ∧ 
  cake.has_side_icing = true ∧ 
  cake.middle_icing_height = 5/2 →
  count_double_iced_cubes cake = 72 :=
by sorry

end NUMINAMATH_CALUDE_double_iced_cubes_count_l70_7020


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l70_7055

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  201000 ≤ Nat.lcm k l ∧ 
  ∃ (k' l' : ℕ), 1000 ≤ k' ∧ k' < 10000 ∧ 
                 1000 ≤ l' ∧ l' < 10000 ∧ 
                 Nat.gcd k' l' = 5 ∧ 
                 Nat.lcm k' l' = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l70_7055


namespace NUMINAMATH_CALUDE_laura_friends_count_l70_7057

def total_blocks : ℕ := 28
def blocks_per_friend : ℕ := 7

theorem laura_friends_count : total_blocks / blocks_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_laura_friends_count_l70_7057


namespace NUMINAMATH_CALUDE_scaled_right_triangle_area_l70_7036

theorem scaled_right_triangle_area :
  ∀ (a b : ℝ) (scale : ℝ),
    a = 50 →
    b = 70 →
    scale = 2 →
    (1/2 : ℝ) * (a * scale) * (b * scale) = 7000 := by
  sorry

end NUMINAMATH_CALUDE_scaled_right_triangle_area_l70_7036


namespace NUMINAMATH_CALUDE_rectangle_to_square_possible_l70_7077

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a rectangular piece cut from the original rectangle -/
structure Piece where
  length : ℕ
  width : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

def Square.area (s : Square) : ℕ := s.side * s.side

def can_form_square (r : Rectangle) (s : Square) (pieces : List Piece) : Prop :=
  r.area = s.area ∧
  (pieces.foldl (fun acc p => acc + p.length * p.width) 0 = r.area) ∧
  (∀ p ∈ pieces, p.length ≤ r.length ∧ p.width ≤ r.width)

theorem rectangle_to_square_possible (r : Rectangle) (h1 : r.length = 16) (h2 : r.width = 9) :
  ∃ (s : Square) (pieces : List Piece), can_form_square r s pieces ∧ pieces.length ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_possible_l70_7077


namespace NUMINAMATH_CALUDE_circle_and_ngons_inequalities_l70_7028

/-- Given a circle and two regular n-gons (one inscribed, one circumscribed),
    prove the relationships between their areas and perimeters. -/
theorem circle_and_ngons_inequalities 
  (n : ℕ) 
  (S : ℝ) 
  (L : ℝ) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (P₁ : ℝ) 
  (P₂ : ℝ) 
  (h_n : n ≥ 3) 
  (h_S : S > 0) 
  (h_L : L > 0) 
  (h_S₁ : S₁ > 0) 
  (h_S₂ : S₂ > 0) 
  (h_P₁ : P₁ > 0) 
  (h_P₂ : P₂ > 0) 
  (h_inscribed : S₁ < S) 
  (h_circumscribed : S₂ > S) : 
  (S^2 > S₁ * S₂) ∧ (L^2 < P₁ * P₂) := by
  sorry


end NUMINAMATH_CALUDE_circle_and_ngons_inequalities_l70_7028


namespace NUMINAMATH_CALUDE_smaller_number_proof_l70_7078

theorem smaller_number_proof (x y : ℝ) 
  (h1 : y = 71.99999999999999)
  (h2 : y - x = (1/3) * y) : 
  x = 48 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l70_7078


namespace NUMINAMATH_CALUDE_factorial_ones_divisibility_l70_7091

/-- Definition of [n]! as the product of numbers consisting of n ones -/
def factorial_ones (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => (10^(i+1) - 1) / 9)

/-- Theorem stating that [n+m]! is divisible by [n]! * [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ones_divisibility_l70_7091


namespace NUMINAMATH_CALUDE_equation_solution_l70_7032

theorem equation_solution : 
  ∃ x : ℝ, (-3 * x - 9 = 6 * x + 18) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l70_7032


namespace NUMINAMATH_CALUDE_bank_deposit_calculation_l70_7005

theorem bank_deposit_calculation (initial_amount : ℝ) : 
  (0.20 * 0.25 * 0.30 * initial_amount = 750) → initial_amount = 50000 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_calculation_l70_7005


namespace NUMINAMATH_CALUDE_johns_age_l70_7090

theorem johns_age (j d m : ℕ) 
  (h1 : j = d - 20)
  (h2 : j = m - 15)
  (h3 : j + d = 80)
  (h4 : m = d + 5) :
  j = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l70_7090


namespace NUMINAMATH_CALUDE_pentagon_area_l70_7063

theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25)
  (h₆ : a * b / 2 + (b + c) * d / 2 = 995) : 
  ∃ (pentagon_area : ℝ), pentagon_area = 995 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l70_7063


namespace NUMINAMATH_CALUDE_smallest_integer_with_gcd_lcm_constraint_l70_7000

theorem smallest_integer_with_gcd_lcm_constraint (x : ℕ) (m n : ℕ) 
  (h1 : x > 0)
  (h2 : m = 30)
  (h3 : Nat.gcd m n = x + 3)
  (h4 : Nat.lcm m n = x * (x + 3)) :
  n ≥ 162 ∧ ∃ (x : ℕ), x > 0 ∧ 
    Nat.gcd 30 162 = x + 3 ∧ 
    Nat.lcm 30 162 = x * (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_gcd_lcm_constraint_l70_7000


namespace NUMINAMATH_CALUDE_rhombus_area_l70_7038

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 21*d₁ + 30 = 0 → 
  d₂^2 - 21*d₂ + 30 = 0 → 
  d₁ ≠ d₂ →
  (1/2 : ℝ) * d₁ * d₂ = 15 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l70_7038


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l70_7058

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(3/4) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l70_7058
