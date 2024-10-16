import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2743_274389

theorem arithmetic_sequence_common_difference 
  (n : ℕ) 
  (total_sum : ℝ) 
  (even_sum : ℝ) 
  (h1 : n = 20) 
  (h2 : total_sum = 75) 
  (h3 : even_sum = 25) : 
  (even_sum - (total_sum - even_sum)) / 10 = -2.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2743_274389


namespace NUMINAMATH_CALUDE_x_over_z_equals_five_l2743_274341

theorem x_over_z_equals_five (x y z : ℚ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / z = 5 := by sorry

end NUMINAMATH_CALUDE_x_over_z_equals_five_l2743_274341


namespace NUMINAMATH_CALUDE_remaining_fruits_theorem_l2743_274342

/-- Represents the number of fruits in a bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  mangoes : ℕ

/-- Calculates the total number of fruits in the bag -/
def FruitBag.total (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.mangoes

/-- Represents Luisa's actions on the fruit bag -/
def luisa_action (bag : FruitBag) : FruitBag :=
  { apples := bag.apples - 2,
    oranges := bag.oranges - 4,
    mangoes := bag.mangoes - (2 * bag.mangoes / 3) }

/-- The theorem to be proved -/
theorem remaining_fruits_theorem (initial_bag : FruitBag)
    (h1 : initial_bag.apples = 7)
    (h2 : initial_bag.oranges = 8)
    (h3 : initial_bag.mangoes = 15) :
    (luisa_action initial_bag).total = 14 := by
  sorry


end NUMINAMATH_CALUDE_remaining_fruits_theorem_l2743_274342


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l2743_274386

def f (x : ℝ) : ℝ := x^2 - 5*x + 1

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ 
    x = (5 + Real.sqrt 21) / 2 ∨
    x = (5 - Real.sqrt 21) / 2 ∨
    x = (11 + Real.sqrt 101) / 2 ∨
    x = (11 - Real.sqrt 101) / 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l2743_274386


namespace NUMINAMATH_CALUDE_farm_bulls_count_l2743_274379

/-- Given a farm with cattle and a cow-to-bull ratio, calculates the number of bulls -/
def calculate_bulls (total_cattle : ℕ) (cow_ratio : ℕ) (bull_ratio : ℕ) : ℕ :=
  (total_cattle * bull_ratio) / (cow_ratio + bull_ratio)

/-- Theorem: On a farm with 555 cattle and a cow-to-bull ratio of 10:27, there are 405 bulls -/
theorem farm_bulls_count : calculate_bulls 555 10 27 = 405 := by
  sorry

end NUMINAMATH_CALUDE_farm_bulls_count_l2743_274379


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_l2743_274365

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a ten-digit number -/
def isTenDigitNumber (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_square (N : ℕ) :
  isTenDigitNumber N →
  sumOfDigits N = 4 →
  (sumOfDigits (N^2) = 7 ∨ sumOfDigits (N^2) = 16) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_l2743_274365


namespace NUMINAMATH_CALUDE_polygon_sides_l2743_274345

theorem polygon_sides (n : ℕ) : n > 2 → (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2743_274345


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l2743_274333

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents the grade of a product -/
inductive Grade
| A
| B
| C
| D

/-- Processing fee for each grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch -/
def processingCost (b : Branch) : ℝ :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch and grade -/
def frequency (b : Branch) (g : Grade) : ℝ :=
  match b, g with
  | Branch.A, Grade.A => 0.4
  | Branch.A, Grade.B => 0.2
  | Branch.A, Grade.C => 0.2
  | Branch.A, Grade.D => 0.2
  | Branch.B, Grade.A => 0.28
  | Branch.B, Grade.B => 0.17
  | Branch.B, Grade.C => 0.34
  | Branch.B, Grade.D => 0.21

/-- Average profit for a branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry


end NUMINAMATH_CALUDE_branch_A_more_profitable_l2743_274333


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2743_274310

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2743_274310


namespace NUMINAMATH_CALUDE_second_smallest_packs_l2743_274385

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 10

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The number of hot dogs left over -/
def leftover_hot_dogs : ℕ := 4

/-- A function that checks if a given number of packs satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

/-- The theorem stating that 6 is the second smallest number of packs satisfying the condition -/
theorem second_smallest_packs : 
  ∃ (m : ℕ), m < 6 ∧ satisfies_condition m ∧ 
  (∀ (k : ℕ), k < m → ¬satisfies_condition k) ∧
  (∀ (k : ℕ), m < k → k < 6 → ¬satisfies_condition k) ∧
  satisfies_condition 6 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_packs_l2743_274385


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l2743_274322

theorem smallest_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2024 * a + 48048 * b) ∧ 
  (∀ (l : ℕ) (c d : ℤ), l > 0 ∧ l = 2024 * c + 48048 * d → k ≤ l) ∧ 
  k = 88 := by
sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l2743_274322


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l2743_274372

theorem decimal_equivalent_one_fourth_power_one : (1 / 4 : ℚ) ^ 1 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l2743_274372


namespace NUMINAMATH_CALUDE_cherry_sweets_count_l2743_274395

/-- The number of cherry-flavored sweets initially in the packet -/
def initial_cherry : ℕ := 30

/-- The number of strawberry-flavored sweets initially in the packet -/
def initial_strawberry : ℕ := 40

/-- The number of pineapple-flavored sweets initially in the packet -/
def initial_pineapple : ℕ := 50

/-- The number of cherry-flavored sweets Aaron gives to his friend -/
def given_away : ℕ := 5

/-- The total number of sweets left in the packet after Aaron's actions -/
def remaining_total : ℕ := 55

theorem cherry_sweets_count :
  initial_cherry = 30 ∧
  (initial_cherry / 2 - given_away) + (initial_strawberry / 2) + (initial_pineapple / 2) = remaining_total :=
by sorry

end NUMINAMATH_CALUDE_cherry_sweets_count_l2743_274395


namespace NUMINAMATH_CALUDE_dorothy_age_ratio_l2743_274374

/-- Given Dorothy's sister's age and the condition about their future ages,
    prove that Dorothy is currently 3 times as old as her sister. -/
theorem dorothy_age_ratio (sister_age : ℕ) (dorothy_age : ℕ) : 
  sister_age = 5 →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age / sister_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_age_ratio_l2743_274374


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2743_274367

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 4*x - 1 = 0) ↔ ((x + 2)^2 = 5) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2743_274367


namespace NUMINAMATH_CALUDE_hakimi_age_l2743_274355

/-- Given three friends with an average age of 40, where Jared is ten years older than Hakimi
    and Molly is 30 years old, prove that Hakimi's age is 40. -/
theorem hakimi_age (average_age : ℝ) (molly_age : ℝ) (jared_hakimi_age_diff : ℝ) 
  (h1 : average_age = 40)
  (h2 : molly_age = 30)
  (h3 : jared_hakimi_age_diff = 10) : 
  ∃ (hakimi_age : ℝ), hakimi_age = 40 ∧ 
    (hakimi_age + (hakimi_age + jared_hakimi_age_diff) + molly_age) / 3 = average_age :=
by sorry

end NUMINAMATH_CALUDE_hakimi_age_l2743_274355


namespace NUMINAMATH_CALUDE_digit_properties_l2743_274347

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem digit_properties :
  ∀ (a b : Nat), a ∈ Digits → b ∈ Digits → a ≠ b →
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (a + b) * (a * b) ≥ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (0 + 1) * (0 * 1) ≤ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x + y = 10 ↔ 
      ((x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∨ (x = 3 ∧ y = 7) ∨ (x = 4 ∧ y = 6) ∨
       (x = 9 ∧ y = 1) ∨ (x = 8 ∧ y = 2) ∨ (x = 7 ∧ y = 3) ∨ (x = 6 ∧ y = 4))) :=
by
  sorry

end NUMINAMATH_CALUDE_digit_properties_l2743_274347


namespace NUMINAMATH_CALUDE_weight_equivalence_l2743_274340

-- Define the weights of shapes as real numbers
variable (triangle circle square : ℝ)

-- Define the conditions from the problem
axiom weight_relation1 : 5 * triangle = 3 * circle
axiom weight_relation2 : circle = triangle + 2 * square

-- Theorem to prove
theorem weight_equivalence : triangle + circle = 3 * square := by
  sorry

end NUMINAMATH_CALUDE_weight_equivalence_l2743_274340


namespace NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2743_274398

-- Define what it means for four numbers to be in arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  ∃ d : ℝ, y = x + d ∧ z = y + d ∧ w = z + d

-- State the theorem
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression 10 a b (a * b) ↔ 
  ((a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2743_274398


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l2743_274393

/-- The probability of all Russian players being paired exclusively with other Russian players
    in a random pairing of 10 tennis players, where 4 are from Russia. -/
theorem russian_players_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let probability : ℚ := (russian_players - 1) / (total_players - 1) *
                         (russian_players - 3) / (total_players - 3)
  probability = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l2743_274393


namespace NUMINAMATH_CALUDE_descending_order_xy_xy2_x_l2743_274328

theorem descending_order_xy_xy2_x
  (x y : ℝ)
  (hx : x < 0)
  (hy : -1 < y ∧ y < 0) :
  xy > xy^2 ∧ xy^2 > x :=
by sorry

end NUMINAMATH_CALUDE_descending_order_xy_xy2_x_l2743_274328


namespace NUMINAMATH_CALUDE_unique_a_for_nonnegative_f_l2743_274396

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → x^2 * (Real.log x - a) + a ≥ 0 ∧ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_nonnegative_f_l2743_274396


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2743_274381

theorem shaded_area_calculation (R : ℝ) (r : ℝ) (h1 : R = 9) (h2 : r = R / 4) :
  π * R^2 - 2 * (π * r^2) = 70.875 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2743_274381


namespace NUMINAMATH_CALUDE_base_7_representation_l2743_274368

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits are consecutive -/
def isConsecutive (digits : List ℕ) : Bool :=
  sorry

theorem base_7_representation :
  let base7Digits := toBase7 143
  base7Digits = [2, 6, 3] ∧
  base7Digits.length = 3 ∧
  isConsecutive base7Digits = true :=
by sorry

end NUMINAMATH_CALUDE_base_7_representation_l2743_274368


namespace NUMINAMATH_CALUDE_cube_vertex_shapes_l2743_274315

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

-- Define the types of shapes we're interested in
inductive ShapeType
  | Rectangle
  | NonRectangleParallelogram
  | IsoscelesRightTetrahedron
  | RegularTetrahedron
  | RightTetrahedron

-- Function to check if 4 vertices form a specific shape
def formsShape (c : Cube) (v : Finset (Fin 8)) (s : ShapeType) : Prop :=
  v.card = 4 ∧ v ⊆ c.vertices ∧ match s with
    | ShapeType.Rectangle => sorry
    | ShapeType.NonRectangleParallelogram => sorry
    | ShapeType.IsoscelesRightTetrahedron => sorry
    | ShapeType.RegularTetrahedron => sorry
    | ShapeType.RightTetrahedron => sorry

-- Theorem statement
theorem cube_vertex_shapes (c : Cube) :
  (∃ v, formsShape c v ShapeType.Rectangle) ∧
  (∃ v, formsShape c v ShapeType.IsoscelesRightTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RegularTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RightTetrahedron) ∧
  (¬ ∃ v, formsShape c v ShapeType.NonRectangleParallelogram) :=
sorry

end NUMINAMATH_CALUDE_cube_vertex_shapes_l2743_274315


namespace NUMINAMATH_CALUDE_percentage_problem_l2743_274352

theorem percentage_problem (P : ℝ) : P = 20 → 0.25 * 1280 = (P / 100) * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2743_274352


namespace NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l2743_274388

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def satisfiesCondition (n : ℕ) : Prop :=
  isTwoDigit n ∧ Nat.Prime (n - 7 * sumOfDigits n)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | satisfiesCondition n} = {10, 31, 52, 73, 94} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l2743_274388


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_implies_m_values_l2743_274373

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - (m-1)*x + m - 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 m = 0 ∧ quadratic x2 m = 0 :=
sorry

-- Theorem 2: When the difference between the roots is 3, m = 0 or m = 6
theorem roots_difference_implies_m_values :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 0 = 0 ∧ quadratic x2 0 = 0 ∧ |x1 - x2| = 3 ∨
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 6 = 0 ∧ quadratic x2 6 = 0 ∧ |x1 - x2| = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_implies_m_values_l2743_274373


namespace NUMINAMATH_CALUDE_homework_problem_count_l2743_274397

theorem homework_problem_count 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : problems_per_page = 3) : 
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l2743_274397


namespace NUMINAMATH_CALUDE_find_a20_l2743_274335

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem find_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 20 = -30 := by
  sorry

end NUMINAMATH_CALUDE_find_a20_l2743_274335


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2743_274376

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ+, (n : ℕ) = 210 ∧ 
  (∀ m : ℕ+, m < n → ¬(∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  (m : ℕ) % p = 0 ∧ (m : ℕ) % q = 0 ∧ (m : ℕ) % r = 0 ∧ (m : ℕ) % s = 0)) ∧
  (∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  210 % p = 0 ∧ 210 % q = 0 ∧ 210 % r = 0 ∧ 210 % s = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2743_274376


namespace NUMINAMATH_CALUDE_new_computer_cost_new_computer_cost_is_600_l2743_274305

theorem new_computer_cost (used_computers_cost : ℕ) (savings : ℕ) : ℕ :=
  let new_computer_cost := used_computers_cost + savings
  new_computer_cost

#check new_computer_cost 400 200

theorem new_computer_cost_is_600 :
  new_computer_cost 400 200 = 600 := by sorry

end NUMINAMATH_CALUDE_new_computer_cost_new_computer_cost_is_600_l2743_274305


namespace NUMINAMATH_CALUDE_triangle_inequality_special_l2743_274361

theorem triangle_inequality_special (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + y^2 - x*y) + Real.sqrt (y^2 + z^2 - y*z) ≥ Real.sqrt (z^2 + x^2 - z*x) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_special_l2743_274361


namespace NUMINAMATH_CALUDE_square_area_proof_l2743_274391

theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l2743_274391


namespace NUMINAMATH_CALUDE_inequality_proof_l2743_274366

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2743_274366


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_first_six_l2743_274371

/-- A geometric sequence with positive terms satisfying a_{n+2} + 2a_{n+1} = 8a_n -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 1 = 1) ∧
  (∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n)

/-- The sum of the first 6 terms of the geometric sequence -/
def SumFirstSixTerms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem geometric_sequence_sum_first_six (a : ℕ → ℝ) 
  (h : GeometricSequence a) : SumFirstSixTerms a = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_first_six_l2743_274371


namespace NUMINAMATH_CALUDE_area_of_RQST_l2743_274326

/-- Square with side length 3 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- Points on the square -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (3, 3)
def D : ℝ × ℝ := (0, 3)

/-- Points E and F divide AB into three segments with ratios 1:2 -/
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (2, 0)

/-- Points G and H divide CD similarly -/
def G : ℝ × ℝ := (3, 1)
def H : ℝ × ℝ := (3, 2)

/-- S is the midpoint of AB -/
def S : ℝ × ℝ := (1.5, 0)

/-- Q is the midpoint of CD -/
def Q : ℝ × ℝ := (3, 1.5)

/-- R and T divide the square into two equal areas -/
def R : ℝ × ℝ := (0, 1.5)
def T : ℝ × ℝ := (3, 1.5)

/-- Area of a quadrilateral given its vertices -/
def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2
           - (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1))

theorem area_of_RQST :
  quadrilateralArea R Q S T = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_RQST_l2743_274326


namespace NUMINAMATH_CALUDE_ranked_choice_voting_theorem_l2743_274377

theorem ranked_choice_voting_theorem 
  (initial_votes_A initial_votes_B initial_votes_C initial_votes_D initial_votes_E : ℚ)
  (redistribution_D_to_A redistribution_D_to_B redistribution_D_to_C : ℚ)
  (redistribution_E_to_A redistribution_E_to_B redistribution_E_to_C : ℚ)
  (majority_difference : ℕ)
  (h1 : initial_votes_A = 35/100)
  (h2 : initial_votes_B = 25/100)
  (h3 : initial_votes_C = 20/100)
  (h4 : initial_votes_D = 15/100)
  (h5 : initial_votes_E = 5/100)
  (h6 : redistribution_D_to_A = 60/100)
  (h7 : redistribution_D_to_B = 25/100)
  (h8 : redistribution_D_to_C = 15/100)
  (h9 : redistribution_E_to_A = 50/100)
  (h10 : redistribution_E_to_B = 30/100)
  (h11 : redistribution_E_to_C = 20/100)
  (h12 : majority_difference = 1890) :
  ∃ (total_votes : ℕ),
    total_votes = 11631 ∧
    (initial_votes_A + redistribution_D_to_A * initial_votes_D + redistribution_E_to_A * initial_votes_E) * total_votes -
    (initial_votes_B + redistribution_D_to_B * initial_votes_D + redistribution_E_to_B * initial_votes_E) * total_votes =
    majority_difference := by
  sorry


end NUMINAMATH_CALUDE_ranked_choice_voting_theorem_l2743_274377


namespace NUMINAMATH_CALUDE_total_spent_on_decks_l2743_274313

/-- The cost of a trick deck in dollars -/
def deck_cost : ℝ := 8

/-- The discount rate for buying 5 or more decks -/
def discount_rate : ℝ := 0.1

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Alice bought -/
def alice_decks : ℕ := 4

/-- The number of decks Bob bought -/
def bob_decks : ℕ := 3

/-- The minimum number of decks to qualify for a discount -/
def discount_threshold : ℕ := 5

/-- Function to calculate the cost of decks with potential discount -/
def calculate_cost (num_decks : ℕ) : ℝ :=
  let base_cost := (num_decks : ℝ) * deck_cost
  if num_decks ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the total amount spent on trick decks -/
theorem total_spent_on_decks : 
  calculate_cost victor_decks + calculate_cost alice_decks + calculate_cost bob_decks = 99.20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_decks_l2743_274313


namespace NUMINAMATH_CALUDE_parallel_lines_problem_l2743_274317

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, m1 * x + y = b1 ↔ m2 * x + y = b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_problem (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - 1 = 0 ↔ 6 * x + a * y + 2 = 0) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_problem_l2743_274317


namespace NUMINAMATH_CALUDE_max_value_of_f_l2743_274336

open Real

noncomputable def f (x : ℝ) := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2743_274336


namespace NUMINAMATH_CALUDE_intersection_distance_l2743_274383

/-- The circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

/-- The line l passing through (-4, 0) with slope angle π/4 -/
def l (x y : ℝ) : Prop := y = x + 4

/-- The intersection points of l and C₁ -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | C₁ p.1 p.2 ∧ l p.1 p.2}

/-- The theorem stating that the distance between the intersection points is √2 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2743_274383


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2743_274384

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -1/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 5 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y - 3 = -6 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2743_274384


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2743_274302

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 3 * Real.sin (π - α) = -2 * Real.cos (π + α)) : 
  ((4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2/21) ∧ 
  (Real.cos (2*α) + Real.sin (α + π/2) = (5 + 3 * Real.sqrt 13) / 13) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2743_274302


namespace NUMINAMATH_CALUDE_ten_digit_number_exists_l2743_274360

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem ten_digit_number_exists : ∃ n : ℕ, 
  1000000000 ≤ n ∧ n < 10000000000 ∧ 
  (∀ d, d ∣ n → d ≠ 0) ∧
  product_of_digits (n + product_of_digits n) = product_of_digits n :=
sorry

end NUMINAMATH_CALUDE_ten_digit_number_exists_l2743_274360


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l2743_274359

/-- A structure representing a sphere packing in a cube -/
structure SpherePacking where
  cube_side_length : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ
  is_valid : Prop

/-- The theorem stating the radius of spheres in the given packing configuration -/
theorem sphere_packing_radius (packing : SpherePacking) : 
  packing.cube_side_length = 2 ∧ 
  packing.num_spheres = 10 ∧ 
  packing.is_valid →
  packing.sphere_radius = 0.5 :=
sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l2743_274359


namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l2743_274316

theorem nested_sqrt_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l2743_274316


namespace NUMINAMATH_CALUDE_complex_number_properties_l2743_274304

def z : ℂ := 4 + 3 * Complex.I

theorem complex_number_properties :
  Complex.abs z = 5 ∧ (1 + Complex.I) / z = (7 + Complex.I) / 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2743_274304


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l2743_274318

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l2743_274318


namespace NUMINAMATH_CALUDE_configurations_formula_l2743_274358

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_configurations (n : ℕ) : ℕ :=
  factorial (n * (n + 1) / 2) /
  (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1

theorem configurations_formula (n : ℕ) :
  num_configurations n = factorial (n * (n + 1) / 2) /
    (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1 :=
by sorry

end NUMINAMATH_CALUDE_configurations_formula_l2743_274358


namespace NUMINAMATH_CALUDE_ratio_problem_l2743_274380

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2743_274380


namespace NUMINAMATH_CALUDE_alicia_art_collection_l2743_274382

/-- The number of medieval art pieces Alicia donated -/
def donated : ℕ := 46

/-- The number of medieval art pieces Alicia had left after donating -/
def left_after_donating : ℕ := 24

/-- The original number of medieval art pieces Alicia had -/
def original_pieces : ℕ := donated + left_after_donating

theorem alicia_art_collection : original_pieces = 70 := by
  sorry

end NUMINAMATH_CALUDE_alicia_art_collection_l2743_274382


namespace NUMINAMATH_CALUDE_divisibility_problem_l2743_274399

theorem divisibility_problem (x y z : ℕ) (h1 : x = 987654) (h2 : y = 456) (h3 : z = 222) :
  (x + z) % y = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2743_274399


namespace NUMINAMATH_CALUDE_barycentric_centroid_relation_l2743_274363

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C X M : V}
variable (α β γ : ℝ)

/-- Given a triangle ABC and a point X with barycentric coordinates (α : β : γ),
    where α + β + γ = 1, and M is the centroid of triangle ABC,
    prove that 3 * vector(XM) = (α - β) * vector(AB) + (β - γ) * vector(BC) + (γ - α) * vector(CA) -/
theorem barycentric_centroid_relation
  (h1 : X = α • A + β • B + γ • C)
  (h2 : α + β + γ = 1)
  (h3 : M = (1/3 : ℝ) • (A + B + C)) :
  3 • (X - M) = (α - β) • (A - B) + (β - γ) • (B - C) + (γ - α) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_centroid_relation_l2743_274363


namespace NUMINAMATH_CALUDE_product_of_primes_even_l2743_274351

theorem product_of_primes_even (P Q : ℕ+) : 
  Prime P.val → Prime Q.val → Prime (P.val - Q.val) → Prime (P.val + Q.val) → 
  Even (P.val * Q.val * (P.val - Q.val) * (P.val + Q.val)) := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_even_l2743_274351


namespace NUMINAMATH_CALUDE_extreme_values_and_inequality_l2743_274350

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + 1) / Real.exp x

theorem extreme_values_and_inequality (m : ℝ) (h₁ : m ≥ 0) :
  (m > 0 → (∃ (min_x max_x : ℝ), min_x = 1 - m ∧ max_x = 1 ∧
    ∀ x, f m x ≥ f m min_x ∧ f m x ≤ f m max_x)) ∧
  (m ∈ Set.Ioo 1 2 → ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 1 m → x₂ ∈ Set.Icc 1 m →
    f m x₁ > -x₂ + 1 + 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_inequality_l2743_274350


namespace NUMINAMATH_CALUDE_largest_three_digit_in_pascal_l2743_274300

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's triangle entry at row n and position k -/
def pascal (n k : ℕ) : ℕ := binomial n k

/-- The largest three-digit number -/
def largest_three_digit : ℕ := 999

/-- The row where the largest three-digit number first appears -/
def first_appearance_row : ℕ := 1000

/-- The position in the row where the largest three-digit number first appears -/
def first_appearance_pos : ℕ := 500

theorem largest_three_digit_in_pascal :
  (∀ n k, n < first_appearance_row → pascal n k ≤ largest_three_digit) ∧
  pascal first_appearance_row first_appearance_pos = largest_three_digit ∧
  (∀ n k, n > first_appearance_row → pascal n k > largest_three_digit) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_in_pascal_l2743_274300


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l2743_274370

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Properties of the isosceles triangle -/
def IsoscelesTriangle.properties (t : IsoscelesTriangle) : Prop :=
  t.base = 16 ∧ t.side = 10

/-- Inradius of the triangle -/
def inradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Circumradius of the triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Distance between the centers of inscribed and circumscribed circles -/
def centerDistance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem about the properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) 
  (h : t.properties) : 
  inradius t = 8/3 ∧ 
  circumradius t = 25/3 ∧ 
  centerDistance t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l2743_274370


namespace NUMINAMATH_CALUDE_brothers_selection_probability_l2743_274323

theorem brothers_selection_probability (p_x p_y p_both : ℚ) :
  p_x = 1/3 → p_y = 2/5 → p_both = p_x * p_y → p_both = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_brothers_selection_probability_l2743_274323


namespace NUMINAMATH_CALUDE_largest_root_of_g_l2743_274392

-- Define the function g(x)
def g (x : ℝ) : ℝ := 12 * x^4 - 17 * x^2 + 5

-- State the theorem
theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l2743_274392


namespace NUMINAMATH_CALUDE_age_difference_l2743_274344

/-- Given information about Lexie and her siblings' ages, prove the age difference between her brother and sister. -/
theorem age_difference (lexie_age : ℕ) (brother_age_diff : ℕ) (sister_age_factor : ℕ) 
  (h1 : lexie_age = 8)
  (h2 : lexie_age = brother_age_diff + lexie_age - 6)
  (h3 : sister_age_factor * lexie_age = 2 * lexie_age) :
  2 * lexie_age - (lexie_age - 6) = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2743_274344


namespace NUMINAMATH_CALUDE_initial_pencils_count_l2743_274348

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Sally took out -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the drawer after Sally took some out -/
def pencils_left : ℕ := 5

/-- Theorem stating that the initial number of pencils is 9 -/
theorem initial_pencils_count : initial_pencils = 9 := by sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l2743_274348


namespace NUMINAMATH_CALUDE_tangent_lines_count_l2743_274332

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add appropriate fields for a line

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of lines tangent to both circles -/
def countTangentLines (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem tangent_lines_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 5) 
  (h2 : c2.radius = 8) 
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 13) :
  countTangentLines c1 c2 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l2743_274332


namespace NUMINAMATH_CALUDE_rome_trip_notes_l2743_274309

/-- Represents the number of notes carried by a person -/
structure Notes where
  euros : ℕ
  dollars : ℕ

/-- The total number of notes carried by both people -/
def total_notes (donald : Notes) (mona : Notes) : ℕ :=
  donald.euros + donald.dollars + mona.euros + mona.dollars

theorem rome_trip_notes :
  ∀ (donald : Notes),
    donald.euros + donald.dollars = 39 →
    donald.euros = donald.dollars →
    ∃ (mona : Notes),
      mona.euros = 3 * donald.euros ∧
      mona.dollars = donald.dollars ∧
      donald.euros + mona.euros = 2 * (donald.dollars + mona.dollars) ∧
      total_notes donald mona = 118 := by
  sorry


end NUMINAMATH_CALUDE_rome_trip_notes_l2743_274309


namespace NUMINAMATH_CALUDE_only_opening_window_is_translational_l2743_274301

-- Define the type for phenomena
inductive Phenomenon
  | wipingCarWindows
  | openingClassroomDoor
  | openingClassroomWindow
  | swingingOnSwing

-- Define the property of being a translational motion
def isTranslationalMotion (p : Phenomenon) : Prop :=
  match p with
  | .wipingCarWindows => False
  | .openingClassroomDoor => False
  | .openingClassroomWindow => True
  | .swingingOnSwing => False

-- Theorem statement
theorem only_opening_window_is_translational :
  ∀ (p : Phenomenon), isTranslationalMotion p ↔ p = Phenomenon.openingClassroomWindow :=
by sorry

end NUMINAMATH_CALUDE_only_opening_window_is_translational_l2743_274301


namespace NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l2743_274307

theorem sin_pi_fourth_plus_alpha (α : ℝ) (h : Real.cos (π/4 - α) = 1/3) :
  Real.sin (π/4 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l2743_274307


namespace NUMINAMATH_CALUDE_c_value_is_198_l2743_274378

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i)^3 - 107 * i

-- State the theorem
theorem c_value_is_198 :
  ∀ a b c : ℕ+, c_equation a b = c → c = 198 := by
  sorry

end NUMINAMATH_CALUDE_c_value_is_198_l2743_274378


namespace NUMINAMATH_CALUDE_remaining_fence_is_48_feet_l2743_274339

/-- The length of fence remaining to be whitewashed after three people have worked on it. -/
def remaining_fence (total_length : ℝ) (first_length : ℝ) (second_fraction : ℝ) (third_fraction : ℝ) : ℝ :=
  let remaining_after_first := total_length - first_length
  let remaining_after_second := remaining_after_first * (1 - second_fraction)
  remaining_after_second * (1 - third_fraction)

/-- Theorem stating that the remaining fence to be whitewashed is 48 feet. -/
theorem remaining_fence_is_48_feet :
  remaining_fence 100 10 (1/5) (1/3) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fence_is_48_feet_l2743_274339


namespace NUMINAMATH_CALUDE_parallelogram_area_l2743_274334

/-- Represents a parallelogram with given base, height, and one angle -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  angle : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 20 and height 4 is 80 -/
theorem parallelogram_area :
  ∀ (p : Parallelogram), p.base = 20 ∧ p.height = 4 ∧ p.angle = 60 → area p = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2743_274334


namespace NUMINAMATH_CALUDE_vaccination_deworming_cost_is_500_l2743_274324

/-- Calculates the cost of vaccinating and deworming a cow given the purchase price,
    daily food cost, number of days, selling price, and profit. -/
def vaccination_deworming_cost (purchase_price : ℕ) (daily_food_cost : ℕ) (days : ℕ)
    (selling_price : ℕ) (profit : ℕ) : ℕ :=
  selling_price - profit - (purchase_price + daily_food_cost * days)

/-- Proves that the cost of vaccinating and deworming is $500 given the specific conditions. -/
theorem vaccination_deworming_cost_is_500 :
    vaccination_deworming_cost 600 20 40 2500 600 = 500 := by
  sorry

#eval vaccination_deworming_cost 600 20 40 2500 600

end NUMINAMATH_CALUDE_vaccination_deworming_cost_is_500_l2743_274324


namespace NUMINAMATH_CALUDE_carnation_bouquets_problem_l2743_274353

/-- Proves that given five bouquets of carnations with specified conditions,
    the sum of carnations in the fourth and fifth bouquets is 34. -/
theorem carnation_bouquets_problem (b1 b2 b3 b4 b5 : ℕ) : 
  b1 = 9 → b2 = 14 → b3 = 18 → 
  (b1 + b2 + b3 + b4 + b5) / 5 = 15 →
  b4 + b5 = 34 := by
sorry

end NUMINAMATH_CALUDE_carnation_bouquets_problem_l2743_274353


namespace NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2743_274320

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Checks if all digits in a number are unique -/
def all_digits_unique (n : ℕ) : Prop :=
  ∀ (d₁ d₂ : Digit), d₁ ≠ d₂ → (n / d₁.val % 10 ≠ n / d₂.val % 10)

/-- The cryptarithm solution -/
def cryptarithm_solution (six : ThreeDigitNumber) (two : ThreeDigitNumber) (twelve : FiveDigitNumber) : Prop :=
  (six.val * two.val = twelve.val) ∧
  (two.val ≥ 23) ∧
  all_digits_unique six.val ∧
  all_digits_unique two.val ∧
  all_digits_unique twelve.val

theorem cryptarithm_unique_solution :
  ∃! (six : ThreeDigitNumber) (two : ThreeDigitNumber) (twelve : FiveDigitNumber),
    cryptarithm_solution six two twelve ∧
    six.val = 986 ∧
    two.val = 34 ∧
    twelve.val = 34848 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2743_274320


namespace NUMINAMATH_CALUDE_parallelogram_area_36_18_l2743_274319

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_36_18 :
  parallelogram_area 36 18 = 648 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_36_18_l2743_274319


namespace NUMINAMATH_CALUDE_cos_160_eq_neg_cos_20_l2743_274356

/-- Proves that cos 160° equals -cos 20° --/
theorem cos_160_eq_neg_cos_20 : 
  Real.cos (160 * π / 180) = - Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_160_eq_neg_cos_20_l2743_274356


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l2743_274390

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem range_of_x_for_inequality (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  {x : ℝ | ∀ a ∈ Set.Icc (-1) 1, f x a > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l2743_274390


namespace NUMINAMATH_CALUDE_remainder_problem_l2743_274303

theorem remainder_problem (m : ℤ) (h : m % 24 = 23) : m % 288 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2743_274303


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2743_274364

theorem polynomial_sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 + 2 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2743_274364


namespace NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l2743_274387

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 7 + a 9 = 16
  fourth_term : a 4 = 1

/-- The 12th term of the arithmetic sequence is 15 -/
theorem twelfth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l2743_274387


namespace NUMINAMATH_CALUDE_angle_around_point_l2743_274325

theorem angle_around_point (x : ℝ) : 
  (120 : ℝ) + 80 + x + x = 360 → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l2743_274325


namespace NUMINAMATH_CALUDE_speedster_convertible_fraction_l2743_274338

theorem speedster_convertible_fraction :
  ∀ (total_inventory : ℕ) (speedsters : ℕ) (speedster_convertibles : ℕ),
    speedsters = total_inventory / 3 →
    total_inventory - speedsters = 30 →
    speedster_convertibles = 12 →
    (speedster_convertibles : ℚ) / speedsters = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertible_fraction_l2743_274338


namespace NUMINAMATH_CALUDE_stamps_leftover_l2743_274331

theorem stamps_leftover (olivia parker quinn : ℕ) (album_capacity : ℕ) 
  (h1 : olivia = 52) 
  (h2 : parker = 66) 
  (h3 : quinn = 23) 
  (h4 : album_capacity = 15) : 
  (olivia + parker + quinn) % album_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_stamps_leftover_l2743_274331


namespace NUMINAMATH_CALUDE_profit_percentage_is_18_percent_l2743_274346

def cost_price : ℝ := 460
def selling_price : ℝ := 542.8

theorem profit_percentage_is_18_percent :
  (selling_price - cost_price) / cost_price * 100 = 18 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_is_18_percent_l2743_274346


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l2743_274329

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = Real.sqrt 3 →
  ‖b‖ = 2 →
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = Real.cos (π / 6) →
  (a.1 - m * b.1) * a.1 + (a.2 - m * b.2) * a.2 = 0 →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l2743_274329


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2743_274308

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2743_274308


namespace NUMINAMATH_CALUDE_range_of_a_l2743_274327

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 0

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition that A is outside the circle M
def A_outside_M (a : ℝ) : Prop :=
  ∀ x y, circle_M a x y → (x - 0)^2 + (y - 2)^2 > (x - a)^2 + (y - a)^2

-- Define the existence of point T
def exists_T (a : ℝ) : Prop :=
  ∃ x y, circle_M a x y ∧ 
    Real.cos (Real.pi/4) * (x - 0) + Real.sin (Real.pi/4) * (y - 2) = 
    Real.sqrt ((x - 0)^2 + (y - 2)^2) * Real.sqrt ((x - a)^2 + (y - a)^2)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, a > 0 → A_outside_M a → exists_T a → Real.sqrt 3 - 1 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2743_274327


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2743_274362

theorem fractional_equation_solution :
  ∃! x : ℚ, x ≠ 1 ∧ x ≠ -1 ∧ (x / (x + 1) - 1 = 3 / (x - 1)) :=
by
  use (-1/2)
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2743_274362


namespace NUMINAMATH_CALUDE_decimal_rep_1_13_150th_digit_l2743_274375

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimalRep : ℕ → Fin 10 := 
  fun n => match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem decimal_rep_1_13_150th_digit : decimalRep 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_rep_1_13_150th_digit_l2743_274375


namespace NUMINAMATH_CALUDE_system_solutions_l2743_274314

theorem system_solutions :
  let S : Set (ℝ × ℝ × ℝ) := { (x, y, z) | x^5 = y^3 + 2*z ∧ y^5 = z^3 + 2*x ∧ z^5 = x^3 + 2*y }
  S = {(0, 0, 0), (Real.sqrt 2, Real.sqrt 2, Real.sqrt 2), (-Real.sqrt 2, -Real.sqrt 2, -Real.sqrt 2)} := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2743_274314


namespace NUMINAMATH_CALUDE_exam_average_l2743_274330

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 70 / 100 →
  avg_total = 80 / 100 →
  ∃ avg₂ : ℚ, 
    (n₁.cast * avg₁ + n₂.cast * avg₂) / (n₁ + n₂).cast = avg_total ∧
    avg₂ = 95 / 100 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_l2743_274330


namespace NUMINAMATH_CALUDE_prime_quadratic_l2743_274312

theorem prime_quadratic (a : ℕ) : 
  Nat.Prime (a^2 - 10*a + 21) ↔ a = 2 ∨ a = 8 := by sorry

end NUMINAMATH_CALUDE_prime_quadratic_l2743_274312


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_equals_A_l2743_274337

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Theorem statement
theorem intersection_A_complement_B_equals_A : A ∩ (U \ B) = A := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_equals_A_l2743_274337


namespace NUMINAMATH_CALUDE_john_scores_42_points_l2743_274349

/-- The number of points John scores in a game -/
def john_total_points : ℕ :=
  let points_per_4_min : ℕ := 2 * 2 + 3
  let period_length : ℕ := 12
  let num_periods : ℕ := 2
  let intervals_per_period : ℕ := period_length / 4
  let points_per_period : ℕ := points_per_4_min * intervals_per_period
  points_per_period * num_periods

/-- Theorem stating that John scores 42 points in the game -/
theorem john_scores_42_points : john_total_points = 42 := by
  sorry

end NUMINAMATH_CALUDE_john_scores_42_points_l2743_274349


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2743_274394

theorem fixed_point_on_line (m : ℝ) : (2 : ℝ) + 1 = m * ((2 : ℝ) - 2) := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2743_274394


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l2743_274321

/-- The line to which the circle is tangent -/
def line (x y : ℝ) : ℝ := x - y - 4

/-- The circle to which the target circle is tangent -/
def given_circle (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - 2*y

/-- The equation of the target circle -/
def target_circle (x y : ℝ) : ℝ := (x - 1)^2 + (y + 1)^2 - 2

/-- Theorem stating that the target circle is the smallest circle tangent to both the line and the given circle -/
theorem smallest_tangent_circle :
  ∀ r > 0, ∀ a b : ℝ,
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → line x y ≠ 0) ∧
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → given_circle x y ≠ 0) →
    r^2 ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l2743_274321


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2743_274311

theorem sum_of_quadratic_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℚ := -48
  let b : ℚ := 108
  let c : ℚ := 162
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2743_274311


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2743_274306

theorem fractional_equation_solution (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → ((x + m) / (x - 2) + 1 / (2 - x) = 3)) →
  ((2 + m) / (2 - 2) + 1 / (2 - 2) = 3) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2743_274306


namespace NUMINAMATH_CALUDE_part_one_part_two_l2743_274357

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2743_274357


namespace NUMINAMATH_CALUDE_sum_le_product_plus_two_l2743_274343

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x * y * z + 2 := by
sorry

end NUMINAMATH_CALUDE_sum_le_product_plus_two_l2743_274343


namespace NUMINAMATH_CALUDE_binary_253_property_l2743_274369

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Counts the number of zeros in a binary representation -/
def countZeros (bin : BinaryRepresentation) : Nat :=
  sorry

/-- Counts the number of ones in a binary representation -/
def countOnes (bin : BinaryRepresentation) : Nat :=
  sorry

theorem binary_253_property :
  let bin := toBinary 253
  let a := countZeros bin
  let b := countOnes bin
  2 * b - a = 13 := by sorry

end NUMINAMATH_CALUDE_binary_253_property_l2743_274369


namespace NUMINAMATH_CALUDE_train_length_l2743_274354

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 69) 
  (h2 : man_speed = 3) 
  (h3 : passing_time = 10) : 
  train_speed * (5/18) * passing_time + man_speed * (5/18) * passing_time = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2743_274354
