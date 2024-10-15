import Mathlib

namespace NUMINAMATH_CALUDE_euler_formula_imaginary_part_l948_94889

open Complex

theorem euler_formula_imaginary_part :
  let z : ℂ := Complex.exp (I * Real.pi / 4)
  let w : ℂ := z / (1 - I)
  Complex.im w = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_euler_formula_imaginary_part_l948_94889


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l948_94843

theorem gasoline_tank_capacity : ∃ (x : ℝ),
  x > 0 ∧
  (7/8 * x - 15 = 2/3 * x) ∧
  x = 72 := by
sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l948_94843


namespace NUMINAMATH_CALUDE_smallest_x_congruence_l948_94848

theorem smallest_x_congruence :
  ∃ (x : ℕ), x > 0 ∧ (725 * x) % 35 = (1165 * x) % 35 ∧
  ∀ (y : ℕ), y > 0 → (725 * y) % 35 = (1165 * y) % 35 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_congruence_l948_94848


namespace NUMINAMATH_CALUDE_multiplication_sum_equality_l948_94818

theorem multiplication_sum_equality : 45 * 25 + 55 * 45 + 20 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_equality_l948_94818


namespace NUMINAMATH_CALUDE_total_necklaces_is_1942_l948_94853

/-- Represents the production of necklaces for a single machine on a given day -/
structure DailyProduction where
  machine : Nat
  day : Nat
  amount : Nat

/-- Calculates the total number of necklaces produced over three days -/
def totalNecklaces (productions : List DailyProduction) : Nat :=
  productions.map (·.amount) |>.sum

/-- The production data for all machines over three days -/
def necklaceProduction : List DailyProduction := [
  -- Sunday
  { machine := 1, day := 1, amount := 45 },
  { machine := 2, day := 1, amount := 108 },
  { machine := 3, day := 1, amount := 230 },
  { machine := 4, day := 1, amount := 184 },
  -- Monday
  { machine := 1, day := 2, amount := 59 },
  { machine := 2, day := 2, amount := 54 },
  { machine := 3, day := 2, amount := 230 },
  { machine := 4, day := 2, amount := 368 },
  -- Tuesday
  { machine := 1, day := 3, amount := 59 },
  { machine := 2, day := 3, amount := 108 },
  { machine := 3, day := 3, amount := 276 },
  { machine := 4, day := 3, amount := 221 }
]

/-- Theorem: The total number of necklaces produced over three days is 1942 -/
theorem total_necklaces_is_1942 : totalNecklaces necklaceProduction = 1942 := by
  sorry

end NUMINAMATH_CALUDE_total_necklaces_is_1942_l948_94853


namespace NUMINAMATH_CALUDE_dinitrogen_pentoxide_weight_l948_94824

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen pentoxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen pentoxide -/
def O_count : ℕ := 5

/-- The molecular weight of Dinitrogen pentoxide in g/mol -/
def molecular_weight_N2O5 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

theorem dinitrogen_pentoxide_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_pentoxide_weight_l948_94824


namespace NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l948_94880

theorem remainder_not_always_power_of_four :
  ∃ n : ℕ, n ≥ 2 ∧ ¬∃ k : ℕ, (2^(2^n) : ℕ) % (2^n - 1) = 4^k := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l948_94880


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l948_94829

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l948_94829


namespace NUMINAMATH_CALUDE_rational_as_cube_sum_ratio_l948_94844

theorem rational_as_cube_sum_ratio (q : ℚ) (hq : 0 < q) : 
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
    q = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_as_cube_sum_ratio_l948_94844


namespace NUMINAMATH_CALUDE_linear_function_properties_l948_94872

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ),
    (linear_function k b (-4) = -9) ∧
    (linear_function k b 3 = 5) ∧
    (k = 2 ∧ b = -1) ∧
    (∃ x, linear_function k b x = 0 ∧ x = 1/2) ∧
    (linear_function k b 0 = -1) ∧
    (1/2 * 1/2 * 1 = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l948_94872


namespace NUMINAMATH_CALUDE_ellipse_parallelogram_condition_l948_94852

/-- The condition for an ellipse to have a parallelogram with a vertex on the ellipse, 
    tangent to the unit circle externally, and inscribed in the ellipse for any point on the ellipse -/
theorem ellipse_parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → 
    ∃ (p q r s : ℝ × ℝ), 
      (p.1^2 + p.2^2 = 1) ∧ 
      (q.1^2 + q.2^2 = 1) ∧ 
      (r.1^2 + r.2^2 = 1) ∧ 
      (s.1^2 + s.2^2 = 1) ∧
      (x^2/a^2 + y^2/b^2 = 1) ∧
      (p.1 - x = s.1 - r.1) ∧ 
      (p.2 - y = s.2 - r.2) ∧
      (q.1 - x = r.1 - s.1) ∧ 
      (q.2 - y = r.2 - s.2)) ↔ 
  (1/a^2 + 1/b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_ellipse_parallelogram_condition_l948_94852


namespace NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l948_94891

/-- The product of the digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * ones

/-- The sum of the digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

/-- A two-digit number N satisfying N = P(N)^2 + S(N) has 1 as its tens digit -/
theorem tens_digit_of_special_two_digit_number :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧ 
    N = (P N)^2 + S N ∧
    N / 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l948_94891


namespace NUMINAMATH_CALUDE_simplify_expression_l948_94805

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4*z^2 + 2*z - 5) = 8 - 9*z^2 - 2*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l948_94805


namespace NUMINAMATH_CALUDE_parabola_sequence_property_l948_94840

/-- Sequence of parabolas Lₙ with general form y² = (2/Sₙ)(x - Tₙ/Sₙ) -/
def T (n : ℕ) : ℚ := (3^n - 1) / 2

def S (n : ℕ) : ℚ := 3^(n-1)

/-- The expression 2Tₙ - 3Sₙ always equals -1 for any positive integer n -/
theorem parabola_sequence_property (n : ℕ) (h : n > 0) : 2 * T n - 3 * S n = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sequence_property_l948_94840


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l948_94895

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem even_odd_sum_difference : 
  (sumFirstEvenIntegers 25) - (sumFirstOddIntegers 20) = 250 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l948_94895


namespace NUMINAMATH_CALUDE_key_pairs_and_drawers_l948_94810

/-- Given 10 distinct keys, prove the following:
1. The number of possible pairs of keys
2. The number of copies of each key needed to form all possible pairs
3. The minimum number of drawers to open to ensure possession of all 10 different keys
-/
theorem key_pairs_and_drawers (n : ℕ) (h : n = 10) :
  let num_pairs := n.choose 2
  let copies_per_key := n - 1
  let total_drawers := num_pairs
  let min_drawers := total_drawers - copies_per_key + 1
  (num_pairs = 45) ∧ (copies_per_key = 9) ∧ (min_drawers = 37) := by
  sorry


end NUMINAMATH_CALUDE_key_pairs_and_drawers_l948_94810


namespace NUMINAMATH_CALUDE_triangle_equality_l948_94898

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l948_94898


namespace NUMINAMATH_CALUDE_sum_of_products_l948_94811

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 36) :
  x*y + y*z + x*z = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l948_94811


namespace NUMINAMATH_CALUDE_not_all_zero_deriv_is_critical_point_l948_94808

open Set
open Function
open Filter

/-- A point x₀ is a critical point of a differentiable function f if f'(x₀) = 0 
    and f'(x) changes sign in any neighborhood of x₀. -/
def IsCriticalPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ ∧ 
  (deriv f) x₀ = 0 ∧
  ∀ ε > 0, ∃ x₁ x₂, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    abs (x₁ - x₀) < ε ∧ abs (x₂ - x₀) < ε ∧
    (deriv f) x₁ * (deriv f) x₂ < 0

/-- The statement "For all differentiable functions f, if f'(x₀) = 0, 
    then x₀ is a critical point of f" is false. -/
theorem not_all_zero_deriv_is_critical_point :
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), DifferentiableAt ℝ f x₀ → (deriv f) x₀ = 0 → IsCriticalPoint f x₀) :=
by sorry

end NUMINAMATH_CALUDE_not_all_zero_deriv_is_critical_point_l948_94808


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l948_94866

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  Complex.abs (1 / (1 + Complex.I * Real.sqrt 3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l948_94866


namespace NUMINAMATH_CALUDE_cardinality_of_B_l948_94860

def A : Finset ℚ := {1, 2, 3, 4, 6}

def B : Finset ℚ := Finset.image (λ (p : ℚ × ℚ) => p.1 / p.2) (A.product A)

theorem cardinality_of_B : Finset.card B = 13 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_B_l948_94860


namespace NUMINAMATH_CALUDE_closest_integer_to_35_4_l948_94839

theorem closest_integer_to_35_4 : ∀ n : ℤ, |n - (35 : ℚ) / 4| ≥ |9 - (35 : ℚ) / 4| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_35_4_l948_94839


namespace NUMINAMATH_CALUDE_seventh_roots_of_unity_polynomial_factorization_l948_94835

theorem seventh_roots_of_unity_polynomial_factorization (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b₁*x + c₁)*(x^2 + b₂*x + c₂)*(x^2 + b₃*x + c₃)) :
  b₁*c₁ + b₂*c₂ + b₃*c₃ = 1 := by
sorry

end NUMINAMATH_CALUDE_seventh_roots_of_unity_polynomial_factorization_l948_94835


namespace NUMINAMATH_CALUDE_bob_rock_skips_bob_rock_skips_solution_l948_94855

theorem bob_rock_skips (jim_skips : ℕ) (rocks_each : ℕ) (total_skips : ℕ) : ℕ :=
  let bob_skips := (total_skips - jim_skips * rocks_each) / rocks_each
  bob_skips

#check @bob_rock_skips

theorem bob_rock_skips_solution :
  bob_rock_skips 15 10 270 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bob_rock_skips_bob_rock_skips_solution_l948_94855


namespace NUMINAMATH_CALUDE_fairCoin_threeFlips_oneTwoTails_l948_94820

/-- Probability of getting k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- A fair coin has probability 0.5 of landing tails -/
def fairCoinProbability : ℝ := 0.5

/-- The number of consecutive coin flips -/
def numberOfFlips : ℕ := 3

theorem fairCoin_threeFlips_oneTwoTails :
  binomialProbability numberOfFlips 1 fairCoinProbability +
  binomialProbability numberOfFlips 2 fairCoinProbability = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_fairCoin_threeFlips_oneTwoTails_l948_94820


namespace NUMINAMATH_CALUDE_annual_pension_correct_l948_94838

/-- Represents the annual pension calculation for an employee -/
noncomputable def annual_pension 
  (a b p q : ℝ) 
  (h1 : b ≠ a) : ℝ :=
  (q * a^2 - p * b^2)^2 / (4 * (p * b - q * a)^2)

/-- Theorem stating the annual pension calculation is correct -/
theorem annual_pension_correct 
  (a b p q : ℝ) 
  (h1 : b ≠ a)
  (h2 : ∃ (k x : ℝ), 
    k * (x - a)^2 = k * x^2 - p ∧ 
    k * (x + b)^2 = k * x^2 + q) :
  ∃ (k x : ℝ), k * x^2 = annual_pension a b p q h1 := by
  sorry

end NUMINAMATH_CALUDE_annual_pension_correct_l948_94838


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_equals_zero_one_l948_94877

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_A_complement_B_equals_zero_one :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_equals_zero_one_l948_94877


namespace NUMINAMATH_CALUDE_even_function_derivative_odd_function_derivative_l948_94890

variable (f : ℝ → ℝ)

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem for even functions
theorem even_function_derivative (h : IsEven f) (h' : Differentiable ℝ f) :
  IsOdd (deriv f) := by sorry

-- Theorem for odd functions
theorem odd_function_derivative (h : IsOdd f) (h' : Differentiable ℝ f) :
  IsEven (deriv f) := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_odd_function_derivative_l948_94890


namespace NUMINAMATH_CALUDE_daniels_purchase_l948_94858

/-- Given the costs of a magazine and a pencil, and a coupon discount,
    calculate the total amount spent. -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given the specific costs and discount,
    the total amount spent is $1.00. -/
theorem daniels_purchase :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_daniels_purchase_l948_94858


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l948_94809

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem linear_function_not_in_second_quadrant :
  ¬ ∃ x : ℝ, in_second_quadrant x (f x) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l948_94809


namespace NUMINAMATH_CALUDE_solution_set_a_eq_one_range_of_a_l948_94857

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f x 1 < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Theorem 2: Range of a when f(x) < 3 has a solution
theorem range_of_a (a : ℝ) :
  (∃ x, f x a < 3) ↔ -3 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_one_range_of_a_l948_94857


namespace NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l948_94879

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define a point P on circle E
def point_P (x y : ℝ) : Prop := circle_E x y

-- Define point Q as the intersection of perpendicular bisector of PF and radius PE
def point_Q (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), point_P px py ∧
  (x - px)^2 + (y - py)^2 = (x - 1)^2 + y^2 ∧
  (x + px + 2) * (x - 1) + y * py = 0

-- Theorem stating the locus of Q is an ellipse
theorem locus_of_Q_is_ellipse :
  ∀ (x y : ℝ), point_Q x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l948_94879


namespace NUMINAMATH_CALUDE_clothes_fraction_l948_94887

def incentive : ℚ := 240
def food_fraction : ℚ := 1/3
def savings_fraction : ℚ := 3/4
def savings_amount : ℚ := 84

theorem clothes_fraction (clothes_amount : ℚ) 
  (h1 : clothes_amount = incentive - food_fraction * incentive - savings_amount / savings_fraction) 
  (h2 : clothes_amount / incentive = 1/5) : 
  clothes_amount / incentive = 1/5 := by
sorry

end NUMINAMATH_CALUDE_clothes_fraction_l948_94887


namespace NUMINAMATH_CALUDE_root_problem_l948_94878

-- Define the polynomials
def p (c d : ℝ) (x : ℝ) : ℝ := (x + c) * (x + d) * (x + 15)
def q (c d : ℝ) (x : ℝ) : ℝ := (x + 3 * c) * (x + 5) * (x + 9)

-- State the theorem
theorem root_problem (c d : ℝ) :
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    p c d r1 = 0 ∧ p c d r2 = 0 ∧ p c d r3 = 0) ∧
  (∃! (r : ℝ), q c d r = 0) ∧
  c ≠ d ∧ c ≠ 4 ∧ c ≠ 15 ∧ d ≠ 4 ∧ d ≠ 15 ∧ d ≠ 5 →
  100 * c + d = 157 := by
sorry

end NUMINAMATH_CALUDE_root_problem_l948_94878


namespace NUMINAMATH_CALUDE_first_group_size_l948_94847

/-- Given a work that takes some men 80 days to complete and 20 men 32 days to complete,
    prove that the number of men in the first group is 8. -/
theorem first_group_size (work : ℕ) : ∃ (x : ℕ), x * 80 = 20 * 32 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_first_group_size_l948_94847


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l948_94894

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_x_axis :
  let P : Point3D := { x := 1, y := 3, z := 6 }
  symmetricPointXAxis P = { x := 1, y := -3, z := -6 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l948_94894


namespace NUMINAMATH_CALUDE_students_not_in_biology_l948_94897

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l948_94897


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l948_94868

/-- Given a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between its asymptotes is 45°, then a/b = √2 + 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (2 * (b/a) / (1 - (b/a)^2)) = π/4) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l948_94868


namespace NUMINAMATH_CALUDE_inequality_implication_l948_94833

theorem inequality_implication (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l948_94833


namespace NUMINAMATH_CALUDE_sum_frequencies_equals_total_data_l948_94828

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  groups : List ℕ  -- List of frequencies for each group
  total_data : ℕ   -- Total number of data points

/-- 
Theorem: In a frequency distribution table, the sum of the frequencies 
of all groups is equal to the total number of data points.
-/
theorem sum_frequencies_equals_total_data (table : FrequencyDistributionTable) : 
  table.groups.sum = table.total_data := by
  sorry


end NUMINAMATH_CALUDE_sum_frequencies_equals_total_data_l948_94828


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l948_94896

-- Define the initial amount of peanut butter in tablespoons
def initial_amount : ℚ := 35 + 2/3

-- Define the amount used for the recipe in tablespoons
def amount_used : ℚ := 5 + 1/3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3

-- Theorem to prove
theorem peanut_butter_servings :
  ⌊(initial_amount - amount_used) / serving_size⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l948_94896


namespace NUMINAMATH_CALUDE_cube_constructions_l948_94846

/-- The number of rotational symmetries of a cube -/
def cubeRotations : ℕ := 24

/-- The total number of ways to place 3 blue cubes in 8 positions -/
def totalPlacements : ℕ := Nat.choose 8 3

/-- The number of invariant configurations under 180° rotation around edge axes -/
def edgeRotationInvariants : ℕ := 4

/-- The number of invariant configurations under 180° rotation around face axes -/
def faceRotationInvariants : ℕ := 4

/-- The sum of all fixed points under different rotations -/
def sumFixedPoints : ℕ := totalPlacements + 6 * edgeRotationInvariants + 3 * faceRotationInvariants

/-- The number of unique constructions of a 2x2x2 cube with 5 white and 3 blue unit cubes -/
def uniqueConstructions : ℕ := sumFixedPoints / cubeRotations

theorem cube_constructions : uniqueConstructions = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_constructions_l948_94846


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l948_94869

theorem sum_of_reciprocals_of_roots (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a + 1011 = 0 →
  b^3 - 2022*b + 1011 = 0 →
  c^3 - 2022*c + 1011 = 0 →
  1/a + 1/b + 1/c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l948_94869


namespace NUMINAMATH_CALUDE_simplify_expression_l948_94884

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x - 4 + 5 = 4*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l948_94884


namespace NUMINAMATH_CALUDE_sum_positive_given_difference_abs_l948_94826

theorem sum_positive_given_difference_abs (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_given_difference_abs_l948_94826


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l948_94802

/-- Represents the number of balls of each color in the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
inductive DrawResult
| Red
| Blue

/-- Represents a sequence of 6 draw operations -/
def DrawSequence := Vector DrawResult 6

/-- Initial state of the urn -/
def initial_state : UrnState := ⟨1, 2⟩

/-- Final number of balls in the urn after 6 operations -/
def final_ball_count : ℕ := 8

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of sequences that result in 4 red and 4 blue balls -/
def favorable_sequence_count : ℕ :=
  sorry

/-- Theorem: The probability of having 4 red and 4 blue balls after 6 operations is 5/14 -/
theorem urn_probability_theorem :
  (favorable_sequence_count : ℚ) * sequence_probability (Vector.replicate 6 DrawResult.Red) = 5/14 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l948_94802


namespace NUMINAMATH_CALUDE_quadratic_solutions_l948_94874

/-- Given a quadratic function f(x) = x^2 + mx, if its axis of symmetry is x = 3,
    then the solutions to x^2 + mx = 0 are 0 and 6. -/
theorem quadratic_solutions (m : ℝ) :
  (∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∃ k, ∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∀ x, x^2 + m*x = 0 ↔ x = 0 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_l948_94874


namespace NUMINAMATH_CALUDE_zero_function_satisfies_equation_zero_function_is_solution_l948_94861

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 2*y) * f (x - 2*y) = (f x + f y)^2 - 16 * y^2 * f x

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ x => 0

/-- Theorem: The zero function satisfies the functional equation -/
theorem zero_function_satisfies_equation : SatisfiesFunctionalEquation ZeroFunction := by
  sorry

/-- Theorem: The zero function is a solution to the functional equation -/
theorem zero_function_is_solution :
  ∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f ∧ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_function_satisfies_equation_zero_function_is_solution_l948_94861


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l948_94831

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l948_94831


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l948_94807

/-- Given a natural number, returns the sum of its digits. -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number. -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, isThreeDigit n → n % 9 = 0 → digitSum n = 27 → n ≤ 999 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l948_94807


namespace NUMINAMATH_CALUDE_rohan_age_multiple_l948_94815

def rohan_current_age : ℕ := 25

def rohan_past_age : ℕ := rohan_current_age - 15

def rohan_future_age : ℕ := rohan_current_age + 15

theorem rohan_age_multiple : 
  ∃ (x : ℚ), rohan_future_age = x * rohan_past_age ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_rohan_age_multiple_l948_94815


namespace NUMINAMATH_CALUDE_A_power_101_l948_94813

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]]

theorem A_power_101 :
  A ^ 101 = ![![0, 1, 0],
              ![0, 0, 1],
              ![1, 0, 0]] := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l948_94813


namespace NUMINAMATH_CALUDE_wednesday_water_intake_l948_94871

/-- Represents the water intake for a week -/
structure WeeklyWaterIntake where
  total : ℕ
  mon_thu_sat : ℕ
  tue_fri_sun : ℕ
  wed : ℕ

/-- The water intake on Wednesday can be determined from the other data -/
theorem wednesday_water_intake (w : WeeklyWaterIntake)
  (h_total : w.total = 60)
  (h_mon_thu_sat : w.mon_thu_sat = 9)
  (h_tue_fri_sun : w.tue_fri_sun = 8)
  (h_balance : w.total = 3 * w.mon_thu_sat + 3 * w.tue_fri_sun + w.wed) :
  w.wed = 9 := by
  sorry

#check wednesday_water_intake

end NUMINAMATH_CALUDE_wednesday_water_intake_l948_94871


namespace NUMINAMATH_CALUDE_rationalize_denominator_l948_94842

theorem rationalize_denominator :
  ∃ (A B C D E : ℚ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 5 * Real.sqrt 3) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 12 ∧
    B = 7 ∧
    C = -15 ∧
    D = 3 ∧
    E = 37 ∧
    (∀ k : ℚ, k ≠ 0 → (k * A * Real.sqrt B + k * C * Real.sqrt D) / (k * E) = (A * Real.sqrt B + C * Real.sqrt D) / E) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, B = m^2 * n)) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, D = m^2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l948_94842


namespace NUMINAMATH_CALUDE_unique_twelve_times_digit_sum_l948_94825

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_twelve_times_digit_sum :
  ∀ n : ℕ, n > 0 → (n = 12 * sum_of_digits n ↔ n = 108) := by
  sorry

end NUMINAMATH_CALUDE_unique_twelve_times_digit_sum_l948_94825


namespace NUMINAMATH_CALUDE_expand_product_l948_94870

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y^2 + 2*y + 4) = 4*y^3 - 4*y^2 - 8*y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l948_94870


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l948_94876

theorem dining_bill_calculation (total_spent : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h_total : total_spent = 132)
  (h_tip : tip_rate = 0.20)
  (h_tax : tax_rate = 0.10) :
  ∃ (original_price : ℝ),
    original_price = 100 ∧
    total_spent = original_price * (1 + tax_rate) * (1 + tip_rate) := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l948_94876


namespace NUMINAMATH_CALUDE_nested_sqrt_unique_solution_l948_94841

/-- Recursive function representing the nested square root expression -/
def nestedSqrt (x : ℤ) : ℕ → ℤ
  | 0 => x
  | n + 1 => (nestedSqrt x n).sqrt

/-- Theorem stating that the only integer solution to the nested square root equation is (0, 0) -/
theorem nested_sqrt_unique_solution :
  ∀ x y : ℤ, (nestedSqrt x 1998 : ℤ) = y → x = 0 ∧ y = 0 := by
  sorry

#check nested_sqrt_unique_solution

end NUMINAMATH_CALUDE_nested_sqrt_unique_solution_l948_94841


namespace NUMINAMATH_CALUDE_green_hat_cost_l948_94867

theorem green_hat_cost (total_hats : ℕ) (green_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) :
  total_hats = 85 →
  green_hats = 20 →
  blue_hat_cost = 6 →
  total_price = 530 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end NUMINAMATH_CALUDE_green_hat_cost_l948_94867


namespace NUMINAMATH_CALUDE_smallest_positive_period_cos_l948_94819

/-- The smallest positive period of cos(π/3 - 2x/5) is 5π -/
theorem smallest_positive_period_cos (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (π/3 - 2*x/5)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, f (t + T) = f t) ∧ 
  (∀ S, S > 0 ∧ (∀ t, f (t + S) = f t) → T ≤ S) ∧
  T = 5*π :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_cos_l948_94819


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l948_94862

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn simultaneously -/
def CardsDrawn : ℕ := 2

/-- Number of Ace of Hearts in a standard deck -/
def AceOfHearts : ℕ := 1

/-- Probability of drawing the Ace of Hearts when 2 cards are drawn simultaneously from a standard 52-card deck -/
theorem ace_of_hearts_probability :
  (AceOfHearts * (StandardDeck - CardsDrawn)) / (StandardDeck.choose CardsDrawn) = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l948_94862


namespace NUMINAMATH_CALUDE_almond_butter_servings_l948_94822

def container_amount : ℚ := 34 + 3/5
def serving_size : ℚ := 5 + 1/2

theorem almond_butter_servings :
  (container_amount / serving_size : ℚ) = 6 + 21/55 := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l948_94822


namespace NUMINAMATH_CALUDE_paco_cookies_problem_l948_94886

/-- Calculates the initial number of salty cookies Paco had --/
def initial_salty_cookies (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ) : ℕ :=
  initial_sweet - difference

theorem paco_cookies_problem (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ)
  (h1 : initial_sweet = 39)
  (h2 : sweet_eaten = 32)
  (h3 : salty_eaten = 23)
  (h4 : difference = sweet_eaten - salty_eaten)
  (h5 : difference = 9) :
  initial_salty_cookies initial_sweet sweet_eaten salty_eaten difference = 30 := by
  sorry

#eval initial_salty_cookies 39 32 23 9

end NUMINAMATH_CALUDE_paco_cookies_problem_l948_94886


namespace NUMINAMATH_CALUDE_james_older_brother_age_l948_94854

/-- Given information about John and James' ages, prove James' older brother's age -/
theorem james_older_brother_age :
  ∀ (john_age james_age : ℕ),
  john_age = 39 →
  john_age - 3 = 2 * (james_age + 6) →
  ∃ (james_brother_age : ℕ),
  james_brother_age = james_age + 4 ∧
  james_brother_age = 16 := by
sorry

end NUMINAMATH_CALUDE_james_older_brother_age_l948_94854


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l948_94851

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b - a)^2 * (c^2 + 4*a*b)^2 ≤ 2*c^6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l948_94851


namespace NUMINAMATH_CALUDE_band_repertoire_average_l948_94849

theorem band_repertoire_average (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (encore : ℕ) : 
  total_songs = 30 →
  first_set = 5 →
  second_set = 7 →
  encore = 2 →
  (total_songs - (first_set + second_set + encore)) / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_band_repertoire_average_l948_94849


namespace NUMINAMATH_CALUDE_ellipse_and_chord_problem_l948_94881

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}

-- Define the circle
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2}

theorem ellipse_and_chord_problem 
  (e : ℝ) (f : ℝ × ℝ) 
  (h_e : e = 2 * Real.sqrt 2 / 3)
  (h_f : f = (0, 2 * Real.sqrt 2))
  (h_foci : ∃ (f' : ℝ × ℝ), f'.1 = 0 ∧ f'.2 = -f.2) :
  -- Standard equation of the ellipse
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Ellipse a b = Ellipse 3 1 ∧
  -- Maximum length of CP
  ∃ (C : ℝ × ℝ) (P : ℝ × ℝ), 
    C ∈ Ellipse 3 1 ∧ 
    P = (-1, 0) ∧
    ∀ (C' : ℝ × ℝ), C' ∈ Ellipse 3 1 → 
      Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) ≥ 
      Real.sqrt ((C'.1 - P.1)^2 + (C'.2 - P.2)^2) ∧
    Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) = 9 * Real.sqrt 2 / 4 ∧
  -- Length of AB when CP is maximum
  ∃ (A B : ℝ × ℝ),
    A ∈ Circle 2 ∧
    B ∈ Circle 2 ∧
    (A.2 - B.2) * (C.1 - P.1) + (B.1 - A.1) * (C.2 - P.2) = 0 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 62 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_problem_l948_94881


namespace NUMINAMATH_CALUDE_find_number_l948_94827

theorem find_number : ∃ n : ℝ, 7 * n - 15 = 2 * n + 10 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l948_94827


namespace NUMINAMATH_CALUDE_sufficient_condition_quadratic_l948_94882

theorem sufficient_condition_quadratic (a : ℝ) :
  a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_quadratic_l948_94882


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l948_94837

/-- Given two vectors in ℝ³, prove that the magnitude of their difference is 3 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ × ℝ) :
  a = (1, 0, 2) → b = (0, 1, 2) →
  ‖a - 2 • b‖ = 3 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l948_94837


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l948_94834

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l948_94834


namespace NUMINAMATH_CALUDE_range_of_a_l948_94865

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- Define the property of f being increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : is_increasing (f a)) :
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l948_94865


namespace NUMINAMATH_CALUDE_croissant_count_is_two_l948_94806

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  expensive : ℕ
  cheap : ℕ

/-- Calculates the total cost given the number of items at each price point -/
def totalCost (counts : ItemCounts) : ℚ :=
  1.5 * counts.expensive + 1.2 * counts.cheap

/-- Checks if a rational number is a whole number -/
def isWholeNumber (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

/-- The main theorem to be proved -/
theorem croissant_count_is_two :
  ∀ counts : ItemCounts,
    counts.expensive + counts.cheap = 7 →
    isWholeNumber (totalCost counts) →
    counts.expensive = 2 :=
by sorry

end NUMINAMATH_CALUDE_croissant_count_is_two_l948_94806


namespace NUMINAMATH_CALUDE_equal_roots_condition_l948_94856

/-- The quadratic equation x^2 - nx + 9 = 0 has two equal real roots if and only if n = 6 or n = -6 -/
theorem equal_roots_condition (n : ℝ) : 
  (∃ x : ℝ, x^2 - n*x + 9 = 0 ∧ (∀ y : ℝ, y^2 - n*y + 9 = 0 → y = x)) ↔ 
  (n = 6 ∨ n = -6) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l948_94856


namespace NUMINAMATH_CALUDE_harry_terry_calculation_harry_terry_calculation_proof_l948_94803

theorem harry_terry_calculation : ℤ → ℤ → Prop :=
  fun (H T : ℤ) =>
    (H = 8 - (2 + 5)) ∧ (T = 8 - 2 + 5) → H - T = -10

-- The proof is omitted
theorem harry_terry_calculation_proof : harry_terry_calculation 1 11 := by
  sorry

end NUMINAMATH_CALUDE_harry_terry_calculation_harry_terry_calculation_proof_l948_94803


namespace NUMINAMATH_CALUDE_factorial_expression_l948_94864

theorem factorial_expression : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_l948_94864


namespace NUMINAMATH_CALUDE_A_inter_B_l948_94814

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2}

theorem A_inter_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l948_94814


namespace NUMINAMATH_CALUDE_solutions_eq_divisors_l948_94883

/-- The number of integer solutions to the equation xy + ax + by = c -/
def num_solutions (a b c : ℤ) : ℕ :=
  2 * (Nat.divisors (a * b + c).natAbs).card

/-- The number of divisors (positive and negative) of an integer n -/
def num_divisors (n : ℤ) : ℕ :=
  2 * (Nat.divisors n.natAbs).card

theorem solutions_eq_divisors (a b c : ℤ) :
  num_solutions a b c = num_divisors (a * b + c) :=
sorry

end NUMINAMATH_CALUDE_solutions_eq_divisors_l948_94883


namespace NUMINAMATH_CALUDE_max_distance_ellipse_line_l948_94845

/-- The maximum distance between a point on the ellipse x²/12 + y²/4 = 1 and the line x + √3y - 6 = 0 is √6 + 3, occurring at the point (-√6, -√2) -/
theorem max_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 12 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let distance (p : ℝ × ℝ) := |p.1 + Real.sqrt 3 * p.2 - 6| / 2
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q ∈ ellipse, distance q ≤ distance p) ∧
    distance p = Real.sqrt 6 + 3 ∧
    p = (-Real.sqrt 6, -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_line_l948_94845


namespace NUMINAMATH_CALUDE_roots_of_equation_correct_description_l948_94801

theorem roots_of_equation (x : ℝ) : 
  (x^2 + 4) * (x^2 - 4) = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem correct_description : 
  ∀ x : ℝ, (x^2 + 4) * (x^2 - 4) = 0 → 
  ∃ y : ℝ, y = 2 ∨ y = -2 ∧ (y^2 + 4) * (y^2 - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_correct_description_l948_94801


namespace NUMINAMATH_CALUDE_equation_solution_l948_94885

theorem equation_solution :
  ∃! y : ℝ, (7 : ℝ) ^ (y + 6) = 343 ^ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l948_94885


namespace NUMINAMATH_CALUDE_class_transfer_problem_l948_94823

/-- Proof of the class transfer problem -/
theorem class_transfer_problem :
  -- Define the total number of students
  ∀ (total : ℕ),
  -- Define the number of students transferred from A to B
  ∀ (transfer_a_to_b : ℕ),
  -- Define the number of students transferred from B to C
  ∀ (transfer_b_to_c : ℕ),
  -- Condition: total students is 92
  total = 92 →
  -- Condition: 5 students transferred from A to B
  transfer_a_to_b = 5 →
  -- Condition: 32 students transferred from B to C
  transfer_b_to_c = 32 →
  -- Condition: After transfers, students in A = 3 * students in B
  ∃ (final_a final_b : ℕ),
    final_a = 3 * final_b ∧
    final_a + final_b = total - transfer_b_to_c →
  -- Conclusion: Originally 45 students in A and 47 in B
  ∃ (original_a original_b : ℕ),
    original_a = 45 ∧
    original_b = 47 ∧
    original_a + original_b = total :=
by sorry

end NUMINAMATH_CALUDE_class_transfer_problem_l948_94823


namespace NUMINAMATH_CALUDE_cube_surface_area_l948_94863

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (v : ℝ) (h : v = 1728) : 
  (6 * ((v ^ (1/3)) ^ 2)) = 864 :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l948_94863


namespace NUMINAMATH_CALUDE_circle_radius_l948_94899

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2) → 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2 ∧ r = Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l948_94899


namespace NUMINAMATH_CALUDE_complex_sum_equality_l948_94836

theorem complex_sum_equality : 
  5 * Complex.exp (Complex.I * Real.pi / 12) + 5 * Complex.exp (Complex.I * 13 * Real.pi / 24) 
  = 10 * Real.cos (11 * Real.pi / 48) * Complex.exp (Complex.I * 5 * Real.pi / 16) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l948_94836


namespace NUMINAMATH_CALUDE_even_number_of_even_scores_l948_94859

/-- Represents a team's score in the basketball competition -/
structure TeamScore where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- The total number of teams in the competition -/
def numTeams : ℕ := 10

/-- The number of games each team plays -/
def gamesPerTeam : ℕ := numTeams - 1

/-- Calculate the total score for a team -/
def totalScore (ts : TeamScore) : ℕ :=
  2 * ts.wins + ts.draws

/-- The scores of all teams in the competition -/
def allTeamScores : Finset TeamScore :=
  sorry

/-- The sum of all team scores is equal to the total number of games multiplied by 2 -/
axiom total_score_sum : 
  (allTeamScores.sum totalScore) = (numTeams * gamesPerTeam)

/-- Theorem: There must be an even number of teams with an even total score -/
theorem even_number_of_even_scores : 
  Even (Finset.filter (fun ts => Even (totalScore ts)) allTeamScores).card :=
sorry

end NUMINAMATH_CALUDE_even_number_of_even_scores_l948_94859


namespace NUMINAMATH_CALUDE_parabolas_intersection_l948_94800

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℚ := {-5/3, 0}

/-- First parabola function -/
def f (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- Second parabola function -/
def g (x : ℚ) : ℚ := 9 * x^2 + 6 * x + 2

/-- Theorem stating that the two parabolas intersect at the given points -/
theorem parabolas_intersection :
  ∀ x ∈ intersection_x, f x = g x ∧ 
  (x = -5/3 → f x = 17) ∧ 
  (x = 0 → f x = 2) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l948_94800


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l948_94812

/-- Given a geometric sequence {a_n} where a_{2020} = 8a_{2017}, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2020 = 8 * a 2017 →         -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l948_94812


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_proof_l948_94893

theorem inequality_and_minimum_value_proof 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (min_val : ℝ) (min_x : ℝ),
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧ 
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
    (∀ (x : ℝ) (hx : 0 < x ∧ x < 1/2), 
      2 / x + 9 / (1 - 2*x) ≥ min_val) ∧
    (0 < min_x ∧ min_x < 1/2) ∧
    (2 / min_x + 9 / (1 - 2*min_x) = min_val) ∧
    min_val = 25 ∧
    min_x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_proof_l948_94893


namespace NUMINAMATH_CALUDE_find_number_l948_94832

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l948_94832


namespace NUMINAMATH_CALUDE_complement_union_theorem_l948_94888

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l948_94888


namespace NUMINAMATH_CALUDE_log_relationship_l948_94816

theorem log_relationship (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  6 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log b) →
  (a = b^(5/3) ∨ a = b^(3/5)) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_l948_94816


namespace NUMINAMATH_CALUDE_triangular_pyramid_least_faces_triangular_pyramid_faces_l948_94892

-- Define the shapes
inductive Shape
  | TriangularPrism
  | QuadrangularPrism
  | TriangularPyramid
  | QuadrangularPyramid
  | TruncatedQuadrangularPyramid

-- Function to count the number of faces for each shape
def numFaces (s : Shape) : ℕ :=
  match s with
  | Shape.TriangularPrism => 5
  | Shape.QuadrangularPrism => 6
  | Shape.TriangularPyramid => 4
  | Shape.QuadrangularPyramid => 5
  | Shape.TruncatedQuadrangularPyramid => 6  -- Assuming the truncated pyramid has a top face

-- Theorem stating that the triangular pyramid has the least number of faces
theorem triangular_pyramid_least_faces :
  ∀ s : Shape, numFaces Shape.TriangularPyramid ≤ numFaces s :=
by
  sorry

-- Theorem stating that the number of faces of a triangular pyramid is 4
theorem triangular_pyramid_faces :
  numFaces Shape.TriangularPyramid = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_least_faces_triangular_pyramid_faces_l948_94892


namespace NUMINAMATH_CALUDE_solve_for_A_l948_94804

theorem solve_for_A : ∃ A : ℚ, 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 ∧ A = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l948_94804


namespace NUMINAMATH_CALUDE_rectangular_box_diagonal_l948_94821

/-- Proves that a rectangular box with given surface area and edge length sum has a specific longest diagonal --/
theorem rectangular_box_diagonal (x y z : ℝ) : 
  (2 * (x*y + y*z + z*x) = 150) →  -- Total surface area
  (4 * (x + y + z) = 60) →         -- Sum of all edge lengths
  Real.sqrt (x^2 + y^2 + z^2) = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonal_l948_94821


namespace NUMINAMATH_CALUDE_factorial_simplification_l948_94817

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * Nat.factorial 8) / (10 * 9 * Nat.factorial 8 + 3 * 9 * Nat.factorial 8) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l948_94817


namespace NUMINAMATH_CALUDE_average_income_problem_l948_94830

/-- Given the average incomes of different pairs of individuals and the income of one individual,
    prove that the average income of a specific pair is as stated. -/
theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (N + O) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_problem_l948_94830


namespace NUMINAMATH_CALUDE_circle_equation_l948_94873

/-- The equation of a circle with center (1, 1) and radius 1 -/
theorem circle_equation : 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 1 ↔ 
  ((x, y) : ℝ × ℝ) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l948_94873


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l948_94875

/-- Two similar right triangles with legs 12 and 9 in one, and x and 6 in the other, have x = 8 -/
theorem similar_right_triangles_leg_length : ∀ x : ℝ,
  (12 : ℝ) / x = 9 / 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l948_94875


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l948_94850

theorem largest_solution_of_equation : 
  ∃ (b : ℝ), (3 * b + 4) * (b - 2) = 9 * b ∧ 
  ∀ (x : ℝ), (3 * x + 4) * (x - 2) = 9 * x → x ≤ b ∧ 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l948_94850
