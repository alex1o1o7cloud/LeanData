import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3583_358300

theorem arithmetic_sequence_50th_term : 
  let start : ℤ := -48
  let diff : ℤ := 2
  let n : ℕ := 50
  let sequence := fun i : ℕ => start + diff * (i - 1)
  sequence n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3583_358300


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l3583_358365

def f (a x : ℝ) : ℝ := -x^2 - 2*a*x

theorem max_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f a x = a^2) →
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l3583_358365


namespace NUMINAMATH_CALUDE_line_parameterization_l3583_358341

/-- The line equation y = (3/4)x - 2 parameterized as (x, y) = (-3, v) + u(m, -8) -/
def line_equation (x y : ℝ) : Prop :=
  y = (3/4) * x - 2

/-- The parametric form of the line -/
def parametric_form (x y u v m : ℝ) : Prop :=
  x = -3 + u * m ∧ y = v - 8 * u

/-- Theorem stating that v = -17/4 and m = -16/9 satisfy the line equation and parametric form -/
theorem line_parameterization :
  ∃ (v m : ℝ), v = -17/4 ∧ m = -16/9 ∧
  (∀ (x y u : ℝ), parametric_form x y u v m → line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3583_358341


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3583_358312

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3583_358312


namespace NUMINAMATH_CALUDE_last_digit_89_base_4_l3583_358343

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_89_base_4 : last_digit_base_4 89 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base_4_l3583_358343


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3583_358363

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x + y/2 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3583_358363


namespace NUMINAMATH_CALUDE_more_men_than_women_count_l3583_358360

def num_men : ℕ := 6
def num_women : ℕ := 4
def group_size : ℕ := 5

def select_group (m w : ℕ) : ℕ := Nat.choose num_men m * Nat.choose num_women w

theorem more_men_than_women_count : 
  (select_group 3 2) + (select_group 4 1) + (select_group 5 0) = 186 :=
by sorry

end NUMINAMATH_CALUDE_more_men_than_women_count_l3583_358360


namespace NUMINAMATH_CALUDE_complex_magnitude_l3583_358369

/-- Given a complex number z satisfying z(1+i) = 1-2i, prove that |z| = √10/2 -/
theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3583_358369


namespace NUMINAMATH_CALUDE_seulgi_stack_higher_l3583_358399

/-- Represents the stack of boxes for each person -/
structure BoxStack where
  numBoxes : ℕ
  boxHeight : ℝ

/-- Calculates the total height of a stack of boxes -/
def totalHeight (stack : BoxStack) : ℝ :=
  stack.numBoxes * stack.boxHeight

theorem seulgi_stack_higher (hyunjeong seulgi : BoxStack)
  (h1 : hyunjeong.numBoxes = 15)
  (h2 : hyunjeong.boxHeight = 4.2)
  (h3 : seulgi.numBoxes = 20)
  (h4 : seulgi.boxHeight = 3.3) :
  totalHeight seulgi > totalHeight hyunjeong := by
  sorry

end NUMINAMATH_CALUDE_seulgi_stack_higher_l3583_358399


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3583_358347

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) 
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 25)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l3583_358347


namespace NUMINAMATH_CALUDE_juan_number_operations_l3583_358361

theorem juan_number_operations (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 9) → (n = 7) := by
  sorry

end NUMINAMATH_CALUDE_juan_number_operations_l3583_358361


namespace NUMINAMATH_CALUDE_sum_inequality_l3583_358374

theorem sum_inequality (m n : ℕ+) (a : Fin m → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ i, a i ∈ Finset.range n)
  (h_sum : ∀ i j, i ≤ j → a i + a j ≤ n → ∃ k, a i + a j = a k) :
  (Finset.sum (Finset.range m) (λ i => a i)) / m ≥ (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3583_358374


namespace NUMINAMATH_CALUDE_expected_digits_is_31_20_l3583_358340

/-- A fair 20-sided die numbered from 1 to 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun i => numDigits (i + 1)) / icosahedralDie.card

/-- Theorem stating the expected number of digits -/
theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_31_20_l3583_358340


namespace NUMINAMATH_CALUDE_triangle_has_obtuse_angle_l3583_358331

/-- A triangle with vertices A(1, 2), B(-3, 4), and C(0, -2) has an obtuse angle. -/
theorem triangle_has_obtuse_angle :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (0, -2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  BC^2 > AB^2 + AC^2 := by sorry

end NUMINAMATH_CALUDE_triangle_has_obtuse_angle_l3583_358331


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3583_358351

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_6 + a_8 = 4, prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometricSequence a) 
    (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3583_358351


namespace NUMINAMATH_CALUDE_max_value_is_72_l3583_358301

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- The maximum weight Carl can carry -/
def maxWeight : ℕ := 24

/-- The available types of rocks -/
def rocks : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 5 }
]

/-- A function to calculate the maximum value of rocks that can be carried -/
def maxValue (rocks : List Rock) (maxWeight : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum value Carl can transport is $72 -/
theorem max_value_is_72 : maxValue rocks maxWeight = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_72_l3583_358301


namespace NUMINAMATH_CALUDE_mobile_chip_transistor_count_l3583_358324

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem mobile_chip_transistor_count :
  toScientificNotation 15300000000 = ScientificNotation.mk 1.53 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_mobile_chip_transistor_count_l3583_358324


namespace NUMINAMATH_CALUDE_speed_increase_time_l3583_358353

/-- Represents the journey of Xavier from point P to point Q -/
structure Journey where
  initialSpeed : ℝ
  increasedSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ

/-- Theorem stating that Xavier increases his speed after 24 minutes -/
theorem speed_increase_time (j : Journey)
  (h1 : j.initialSpeed = 40)
  (h2 : j.increasedSpeed = 60)
  (h3 : j.totalDistance = 56)
  (h4 : j.totalTime = 0.8) : 
  ∃ t : ℝ, t * j.initialSpeed + (j.totalTime - t) * j.increasedSpeed = j.totalDistance ∧ t = 0.4 := by
  sorry

#check speed_increase_time

end NUMINAMATH_CALUDE_speed_increase_time_l3583_358353


namespace NUMINAMATH_CALUDE_algebra_sum_is_5_l3583_358321

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 8 => -3
  | 9 => -2
  | 0 => -1
  | _ => 0  -- This case should never occur

def alphabet_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'l' => 12
  | 'g' => 7
  | 'e' => 5
  | 'b' => 2
  | 'r' => 18
  | _ => 0  -- This case should never occur for valid input

theorem algebra_sum_is_5 :
  (letter_value (alphabet_position 'a') +
   letter_value (alphabet_position 'l') +
   letter_value (alphabet_position 'g') +
   letter_value (alphabet_position 'e') +
   letter_value (alphabet_position 'b') +
   letter_value (alphabet_position 'r') +
   letter_value (alphabet_position 'a')) = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebra_sum_is_5_l3583_358321


namespace NUMINAMATH_CALUDE_least_positive_value_cubic_equation_l3583_358303

/-- The least positive integer value of a cubic equation with prime number constraints -/
theorem least_positive_value_cubic_equation (x y z w : ℕ) : 
  Prime x → Prime y → Prime z → Prime w →
  x + y + z + w < 50 →
  (∀ a b c d : ℕ, Prime a → Prime b → Prime c → Prime d → 
    a + b + c + d < 50 → 
    24 * a^3 + 16 * b^3 - 7 * c^3 + 5 * d^3 ≥ 24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3) →
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_value_cubic_equation_l3583_358303


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3583_358308

/-- Proves that given a journey of 225 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed for the second half of the journey is 26.25 km/hr. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 225 →
  total_time = 10 →
  first_half_speed = 21 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := first_half_distance / second_half_time
  second_half_speed = 26.25 := by
  sorry


end NUMINAMATH_CALUDE_journey_speed_calculation_l3583_358308


namespace NUMINAMATH_CALUDE_binomial_p_value_l3583_358394

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If a binomial random variable has E[ξ] = 8 and D[ξ] = 1.6, then p = 0.8 -/
theorem binomial_p_value (X : BinomialRV) 
  (h1 : expectedValue X = 8) 
  (h2 : variance X = 1.6) : 
  X.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_p_value_l3583_358394


namespace NUMINAMATH_CALUDE_negative_integer_solution_exists_l3583_358391

theorem negative_integer_solution_exists : ∃ (x : ℤ), x < 0 ∧ 3 * x + 13 ≥ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_exists_l3583_358391


namespace NUMINAMATH_CALUDE_percentage_both_correct_l3583_358325

theorem percentage_both_correct (total : ℝ) (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ) :
  total > 0 →
  first_correct / total = 0.75 →
  second_correct / total = 0.30 →
  neither_correct / total = 0.20 →
  (first_correct + second_correct - (total - neither_correct)) / total = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l3583_358325


namespace NUMINAMATH_CALUDE_mean_proportional_81_100_l3583_358381

theorem mean_proportional_81_100 : ∃ x : ℝ, x^2 = 81 * 100 ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_81_100_l3583_358381


namespace NUMINAMATH_CALUDE_least_possible_third_side_l3583_358357

theorem least_possible_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  Real.sqrt 161 ≤ min a (min b c) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l3583_358357


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l3583_358388

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l3583_358388


namespace NUMINAMATH_CALUDE_constant_term_of_liams_polynomial_l3583_358342

/-- Represents a polynomial with degree 5 -/
structure Poly5 where
  coeffs : Fin 6 → ℝ
  monic : coeffs 5 = 1

/-- The product of two polynomials -/
def poly_product (p q : Poly5) : Fin 11 → ℝ := sorry

theorem constant_term_of_liams_polynomial 
  (serena_poly liam_poly : Poly5)
  (same_constant : serena_poly.coeffs 0 = liam_poly.coeffs 0)
  (positive_constant : serena_poly.coeffs 0 > 0)
  (same_z2_coeff : serena_poly.coeffs 2 = liam_poly.coeffs 2)
  (product : poly_product serena_poly liam_poly = 
    fun i => match i with
    | 0 => 9  | 1 => 5  | 2 => 10 | 3 => 4  | 4 => 9
    | 5 => 6  | 6 => 5  | 7 => 4  | 8 => 3  | 9 => 2
    | 10 => 1
  ) :
  liam_poly.coeffs 0 = 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_liams_polynomial_l3583_358342


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_and_sum_l3583_358364

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 9 →
  a 2 + a 3 = 18 →
  (∀ n, b n = a n + 2 * n) →
  (∀ n, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n, S n = 3 * 2^n + n * (n + 1) - 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_and_sum_l3583_358364


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3583_358398

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3583_358398


namespace NUMINAMATH_CALUDE_tangent_line_and_function_inequality_l3583_358323

open Real

theorem tangent_line_and_function_inequality (a b m : ℝ) : 
  (∀ x, x = -π/4 → (tan x = a*x + b + π/2)) →
  (∀ x, x ∈ Set.Icc 1 2 → m ≤ (exp x + b*x^2 + a) ∧ (exp x + b*x^2 + a) ≤ m^2 - 2) →
  (∃ m_max : ℝ, m_max = exp 1 + 1 ∧ m ≤ m_max) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_function_inequality_l3583_358323


namespace NUMINAMATH_CALUDE_main_theorem_l3583_358373

/-- The set of real numbers c > 0 for which exactly one of two statements is true --/
def C : Set ℝ := {c | c > 0 ∧ (c ≤ 1/2 ∨ c ≥ 1)}

/-- Statement p: The function y = c^x is monotonically decreasing on ℝ --/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

/-- Statement q: The solution set of x + |x - 2c| > 1 is ℝ --/
def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2*c| > 1

/-- Main theorem: c is in set C if and only if exactly one of p(c) or q(c) is true --/
theorem main_theorem (c : ℝ) : c ∈ C ↔ (p c ∧ ¬q c) ∨ (¬p c ∧ q c) := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3583_358373


namespace NUMINAMATH_CALUDE_simplify_xy_squared_l3583_358366

theorem simplify_xy_squared (x y : ℝ) : 5 * x * y^2 - 6 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_xy_squared_l3583_358366


namespace NUMINAMATH_CALUDE_num_assignments_is_15000_l3583_358346

/-- Represents a valid assignment of students to events -/
structure Assignment where
  /-- The mapping of students to events -/
  student_to_event : Fin 7 → Fin 5
  /-- Ensures that students 0 and 1 (representing A and B) are not in the same event -/
  students_separated : student_to_event 0 ≠ student_to_event 1
  /-- Ensures that each event has at least one participant -/
  events_nonempty : ∀ e : Fin 5, ∃ s : Fin 7, student_to_event s = e

/-- The number of valid assignments -/
def num_valid_assignments : ℕ := sorry

/-- The main theorem stating that the number of valid assignments is 15000 -/
theorem num_assignments_is_15000 : num_valid_assignments = 15000 := by sorry

end NUMINAMATH_CALUDE_num_assignments_is_15000_l3583_358346


namespace NUMINAMATH_CALUDE_sisters_ages_l3583_358316

theorem sisters_ages (s g : ℕ) : 
  (s > 0) → 
  (g > 0) → 
  (1000 ≤ g * 100 + s) → 
  (g * 100 + s < 10000) → 
  (∃ a : ℕ, g * 100 + s = a * a) →
  (∃ b : ℕ, (g + 13) * 100 + (s + 13) = b * b) →
  s + g = 55 := by
sorry

end NUMINAMATH_CALUDE_sisters_ages_l3583_358316


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_cubic_inequality_negation_l3583_358329

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ (∃ x, ¬p x) := by sorry

theorem cubic_inequality_negation :
  (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_cubic_inequality_negation_l3583_358329


namespace NUMINAMATH_CALUDE_wire_cutting_l3583_358304

/-- Given a wire of length 80 cm, if it's cut into two pieces such that the longer piece
    is 3/5 of the shorter piece longer, then the length of the shorter piece is 400/13 cm. -/
theorem wire_cutting (total_length : ℝ) (shorter_piece : ℝ) :
  total_length = 80 ∧
  total_length = shorter_piece + (shorter_piece + 3/5 * shorter_piece) →
  shorter_piece = 400/13 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3583_358304


namespace NUMINAMATH_CALUDE_pool_cost_is_90000_l3583_358380

/-- The cost to fill a rectangular pool with bottled water -/
def pool_fill_cost (length width depth : ℝ) (liters_per_cubic_foot : ℝ) (cost_per_liter : ℝ) : ℝ :=
  length * width * depth * liters_per_cubic_foot * cost_per_liter

/-- Theorem: The cost to fill the specified pool is $90,000 -/
theorem pool_cost_is_90000 :
  pool_fill_cost 20 6 10 25 3 = 90000 := by
  sorry

#eval pool_fill_cost 20 6 10 25 3

end NUMINAMATH_CALUDE_pool_cost_is_90000_l3583_358380


namespace NUMINAMATH_CALUDE_pencil_count_in_10x10_grid_l3583_358305

/-- Represents a grid of items -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of items on the perimeter of a grid -/
def perimeterCount (g : Grid) : ℕ :=
  2 * (g.rows + g.cols) - 4

/-- Calculates the number of items inside a grid (excluding the perimeter) -/
def innerCount (g : Grid) : ℕ :=
  (g.rows - 2) * (g.cols - 2)

/-- The main theorem stating that in a 10x10 grid, the number of pencils inside is 64 -/
theorem pencil_count_in_10x10_grid :
  let g : Grid := { rows := 10, cols := 10 }
  innerCount g = 64 := by sorry

end NUMINAMATH_CALUDE_pencil_count_in_10x10_grid_l3583_358305


namespace NUMINAMATH_CALUDE_change_calculation_l3583_358328

def egg_cost : ℕ := 3
def pancake_cost : ℕ := 2
def cocoa_cost : ℕ := 2
def tax : ℕ := 1
def initial_order_cost : ℕ := egg_cost + pancake_cost + 2 * cocoa_cost + tax
def additional_order_cost : ℕ := pancake_cost + cocoa_cost
def total_paid : ℕ := 15

theorem change_calculation :
  total_paid - (initial_order_cost + additional_order_cost) = 1 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l3583_358328


namespace NUMINAMATH_CALUDE_cut_pentagon_area_l3583_358377

/-- Represents a pentagon created by cutting a triangular corner from a rectangular sheet. -/
structure CutPentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating that a pentagon with specific side lengths has an area of 770. -/
theorem cut_pentagon_area : ∃ (p : CutPentagon), p.sides = {14, 21, 22, 28, 35} ∧ p.area = 770 := by
  sorry

end NUMINAMATH_CALUDE_cut_pentagon_area_l3583_358377


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l3583_358314

-- Define the original length of the rectangle
def original_length : ℝ := 140

-- Define the length increase factor
def length_increase : ℝ := 1.30

-- Define the width decrease factor
def width_decrease : ℝ := 0.8230769230769231

-- Define the approximate width we want to prove
def approximate_width : ℝ := 130.91

-- Theorem statement
theorem rectangle_width_proof :
  ∃ (original_width : ℝ),
    (original_length * original_width = original_length * length_increase * original_width * width_decrease) ∧
    (abs (original_width - approximate_width) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l3583_358314


namespace NUMINAMATH_CALUDE_orange_juice_revenue_l3583_358387

/-- Represents the number of trees each sister owns -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating that the total revenue from selling orange juice is $220,000 -/
theorem orange_juice_revenue :
  (trees * gabriela_oranges + trees * alba_oranges + trees * maricela_oranges) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_revenue_l3583_358387


namespace NUMINAMATH_CALUDE_circle_equation_tangent_lines_center_x_range_l3583_358350

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the circle C
def circle_C : Circle :=
  { center := (3, 2), radius := 1 }

-- Theorem 1
theorem circle_equation (C : Circle) (h1 : C.center.2 = line_l C.center.1) 
  (h2 : C.center.2 = -C.center.1 + 5) (h3 : C.radius = 1) :
  ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1 :=
sorry

-- Theorem 2
theorem tangent_lines (C : Circle) (h : ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1) :
  (∀ x, x = 3) ∨ (∀ x y, 3*x + 4*y - 12 = 0) :=
sorry

-- Theorem 3
theorem center_x_range (C : Circle) 
  (h : ∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = C.radius^2 ∧ 
                    (M.1 - point_A.1)^2 + (M.2 - point_A.2)^2 = M.1^2 + M.2^2) :
  9/4 ≤ C.center.1 ∧ C.center.1 ≤ 13/4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_lines_center_x_range_l3583_358350


namespace NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l3583_358384

-- Define the points
def M : ℝ × ℝ := (-5, 3)
def A1 : ℝ × ℝ := (-8, -1)
def A2 : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 6)

-- Define the circle equations
def circle1_eq (x y : ℝ) : Prop := (x + 5)^2 + (y - 3)^2 = 25
def circle2_eq (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 5

-- Theorem for the first circle
theorem circle1_correct : 
  (∀ x y : ℝ, circle1_eq x y ↔ 
    ((x, y) = M ∨ (∃ t : ℝ, (x, y) = M + t • (A1 - M) ∧ 0 < t ∧ t < 1))) := by sorry

-- Theorem for the second circle
theorem circle2_correct : 
  (∀ x y : ℝ, circle2_eq x y ↔ 
    ((x, y) = A2 ∨ (x, y) = B ∨ (x, y) = C ∨ 
    (∃ t : ℝ, ((x, y) = A2 + t • (B - A2) ∨ 
               (x, y) = B + t • (C - B) ∨ 
               (x, y) = C + t • (A2 - C)) ∧ 
    0 < t ∧ t < 1))) := by sorry

end NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l3583_358384


namespace NUMINAMATH_CALUDE_percentage_difference_l3583_358376

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3583_358376


namespace NUMINAMATH_CALUDE_contrapositive_prop2_true_l3583_358393

theorem contrapositive_prop2_true : 
  (∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_prop2_true_l3583_358393


namespace NUMINAMATH_CALUDE_base_conversion_2869_to_base_7_l3583_358348

theorem base_conversion_2869_to_base_7 :
  2869 = 1 * (7^4) + 1 * (7^3) + 2 * (7^2) + 3 * (7^1) + 6 * (7^0) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2869_to_base_7_l3583_358348


namespace NUMINAMATH_CALUDE_books_read_l3583_358318

theorem books_read (total_books : ℕ) (total_movies : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  total_books = 10 →
  total_movies = 11 →
  movies_watched = 12 →
  books_read = (min movies_watched total_movies) + 1 →
  books_read = 12 := by
sorry

end NUMINAMATH_CALUDE_books_read_l3583_358318


namespace NUMINAMATH_CALUDE_min_value_x_l3583_358354

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2*y = (x + 16*y) / (2*x*y)) : 
  x ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ = 4 ∧ y₀ > 0 ∧ x₀ - 2*y₀ = (x₀ + 16*y₀) / (2*x₀*y₀) := by
  sorry

#check min_value_x

end NUMINAMATH_CALUDE_min_value_x_l3583_358354


namespace NUMINAMATH_CALUDE_line_relationship_l3583_358327

-- Define a type for lines in space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder definition

-- Define parallel relationship between lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of parallel lines

-- Define intersection relationship between lines
def intersects (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of intersecting lines

-- Define skew relationship between lines
def skew (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of skew lines

-- Theorem statement
theorem line_relationship (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3583_358327


namespace NUMINAMATH_CALUDE_special_quadrilateral_angles_l3583_358320

/-- A quadrilateral with specific angle relationships and side equality -/
structure SpecialQuadrilateral where
  A : ℝ  -- Angle at vertex A
  B : ℝ  -- Angle at vertex B
  C : ℝ  -- Angle at vertex C
  D : ℝ  -- Angle at vertex D
  angle_B_triple_A : B = 3 * A
  angle_C_triple_B : C = 3 * B
  angle_D_triple_C : D = 3 * C
  sum_of_angles : A + B + C + D = 360
  sides_equal : True  -- Representing AD = BC (not used in angle calculations)

/-- The angles in the special quadrilateral are 9°, 27°, 81°, and 243° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  q.A = 9 ∧ q.B = 27 ∧ q.C = 81 ∧ q.D = 243 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_angles_l3583_358320


namespace NUMINAMATH_CALUDE_kho_kho_only_players_l3583_358379

theorem kho_kho_only_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) : 
  total = 45 → kabadi = 10 → both = 5 → kho_kho_only = total - kabadi + both :=
by
  sorry

end NUMINAMATH_CALUDE_kho_kho_only_players_l3583_358379


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_radius_l3583_358358

theorem equilateral_triangle_circle_radius (r : ℝ) 
  (h : r > 0) : 
  (3 * (r * Real.sqrt 3) = π * r^2) → 
  r = (3 * Real.sqrt 3) / π := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_radius_l3583_358358


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l3583_358335

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (-2, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 10 = 0

-- Define the two possible perpendicular lines
def perp_line₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def perp_line₂ (x y : ℝ) : Prop := x - 2 * y = 0

theorem intersection_and_parallel_line :
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  parallel_line P.1 P.2 ∧
  (∃ (x y : ℝ), parallel_line x y ∧ 3 * x - 2 * y + 4 = 0) ∧
  (perp_line₁ 0 0 ∧ (∃ (x y : ℝ), perp_line₁ x y ∧ l₁ x y ∧ l₂ x y) ∨
   perp_line₂ 0 0 ∧ (∃ (x y : ℝ), perp_line₂ x y ∧ l₁ x y ∧ l₂ x y)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l3583_358335


namespace NUMINAMATH_CALUDE_arrangements_equal_l3583_358311

/-- The number of arrangements when adding 2 books to 3 existing books while keeping their relative order --/
def arrangements_books : ℕ := 20

/-- The number of arrangements for 7 people with height constraints --/
def arrangements_people : ℕ := 20

/-- Theorem stating that both arrangement problems result in 20 different arrangements --/
theorem arrangements_equal : arrangements_books = arrangements_people := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_l3583_358311


namespace NUMINAMATH_CALUDE_vector_simplification_l3583_358333

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B O M : V)

-- Define the theorem
theorem vector_simplification (A B O M : V) :
  (B - A) + (O - B) + (M - O) + (B - M) = B - A :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3583_358333


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3583_358339

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n % 2 = 1) → -- n is odd
  (n + (n + 4) = 150) → -- sum of first and third is 150
  (n + (n + 2) + (n + 4) = 225) -- sum of all three is 225
:= by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3583_358339


namespace NUMINAMATH_CALUDE_f_is_even_l3583_358362

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : is_odd_function g) :
  is_even_function (fun x ↦ |g (x^5)|) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_l3583_358362


namespace NUMINAMATH_CALUDE_divisibility_by_p_squared_l3583_358371

theorem divisibility_by_p_squared (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_three : p > 3) :
  ∃ k : ℤ, (p + 1 : ℤ)^(p - 1) - 1 = k * p^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_p_squared_l3583_358371


namespace NUMINAMATH_CALUDE_stratified_sampling_most_representative_l3583_358397

-- Define a type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a type for high school grades
inductive HighSchoolGrade
  | First
  | Second
  | Third

-- Define a population with subgroups
structure Population where
  subgroups : List HighSchoolGrade

-- Define a characteristic being studied
structure Characteristic where
  name : String
  hasSignificantDifferences : Bool

-- Define a function to determine the most representative sampling method
def mostRepresentativeSamplingMethod (pop : Population) (char : Characteristic) : SamplingMethod :=
  if char.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

-- Theorem statement
theorem stratified_sampling_most_representative 
  (pop : Population) 
  (char : Characteristic) 
  (h1 : pop.subgroups = [HighSchoolGrade.First, HighSchoolGrade.Second, HighSchoolGrade.Third]) 
  (h2 : char.name = "Understanding of Jingma") 
  (h3 : char.hasSignificantDifferences = true) :
  mostRepresentativeSamplingMethod pop char = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_representative_l3583_358397


namespace NUMINAMATH_CALUDE_equidistant_point_l3583_358313

/-- The distance between two points in a 2D plane -/
def distance (x1 y1 x2 y2 : ℚ) : ℚ :=
  ((x2 - x1)^2 + (y2 - y1)^2).sqrt

/-- The point C with coordinates (3, 0) -/
def C : ℚ × ℚ := (3, 0)

/-- The point D with coordinates (5, 6) -/
def D : ℚ × ℚ := (5, 6)

/-- The y-coordinate of the point on the y-axis -/
def y : ℚ := 13/3

theorem equidistant_point : 
  distance 0 y C.1 C.2 = distance 0 y D.1 D.2 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_l3583_358313


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l3583_358395

theorem travel_distance_ratio :
  ∀ (total_distance plane_distance bus_distance train_distance : ℕ),
    total_distance = 900 →
    plane_distance = total_distance / 3 →
    bus_distance = 360 →
    train_distance = total_distance - (plane_distance + bus_distance) →
    (train_distance : ℚ) / bus_distance = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l3583_358395


namespace NUMINAMATH_CALUDE_fraction_over_65_l3583_358315

theorem fraction_over_65 (total_people : ℕ) (people_under_21 : ℕ) (people_over_65 : ℕ) :
  (people_under_21 : ℚ) / total_people = 3 / 7 →
  50 < total_people →
  total_people < 100 →
  people_under_21 = 24 →
  (people_over_65 : ℚ) / total_people = people_over_65 / 56 :=
by sorry

end NUMINAMATH_CALUDE_fraction_over_65_l3583_358315


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3583_358383

/-- Calculate simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof that the simple interest for the given conditions is 4016.25 -/
theorem simple_interest_calculation :
  let principal : ℚ := 44625
  let rate : ℚ := 1
  let time : ℚ := 9
  simpleInterest principal rate time = 4016.25 := by
  sorry

#eval simpleInterest 44625 1 9

end NUMINAMATH_CALUDE_simple_interest_calculation_l3583_358383


namespace NUMINAMATH_CALUDE_all_terms_even_l3583_358370

theorem all_terms_even (p q : ℤ) (hp : Even p) (hq : Even q) :
  ∀ k : ℕ, k ≤ 8 → Even (Nat.choose 8 k * p^(8 - k) * q^k) := by sorry

end NUMINAMATH_CALUDE_all_terms_even_l3583_358370


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3583_358355

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3583_358355


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3583_358306

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 1) + (m - 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = -1 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3583_358306


namespace NUMINAMATH_CALUDE_dried_fruit_business_theorem_l3583_358326

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -80 * x + 560

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 3) * (sales_quantity x) - 80

theorem dried_fruit_business_theorem 
  (cost_per_bag : ℝ) 
  (other_expenses : ℝ) 
  (min_price max_price : ℝ) :
  cost_per_bag = 3 →
  other_expenses = 80 →
  min_price = 3.5 →
  max_price = 5.5 →
  sales_quantity 3.5 = 280 →
  sales_quantity 5.5 = 120 →
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    sales_quantity x = -80 * x + 560) →
  daily_profit 4 = 160 ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    daily_profit x ≤ 240) ∧
  daily_profit 5 = 240 := by
sorry

end NUMINAMATH_CALUDE_dried_fruit_business_theorem_l3583_358326


namespace NUMINAMATH_CALUDE_shorter_stick_length_l3583_358367

theorem shorter_stick_length (longer shorter : ℝ) 
  (h1 : longer - shorter = 12)
  (h2 : (2/3) * longer = shorter) : 
  shorter = 24 := by
  sorry

end NUMINAMATH_CALUDE_shorter_stick_length_l3583_358367


namespace NUMINAMATH_CALUDE_difference_of_squares_l3583_358382

theorem difference_of_squares (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3583_358382


namespace NUMINAMATH_CALUDE_math_contest_scores_l3583_358310

theorem math_contest_scores : 
  let scores : List ℕ := [59, 67, 97, 103, 109, 113]
  let total_sum : ℕ := scores.sum
  let four_students_avg : ℕ := 94
  let four_students_sum : ℕ := 4 * four_students_avg
  let remaining_sum : ℕ := total_sum - four_students_sum
  remaining_sum / 2 = 86 := by
  sorry

end NUMINAMATH_CALUDE_math_contest_scores_l3583_358310


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3583_358390

theorem arithmetic_expression_equality : 
  8.1 * 1.3 + 8 / 1.3 + 1.9 * 1.3 - 11.9 / 1.3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3583_358390


namespace NUMINAMATH_CALUDE_brad_balloons_l3583_358356

theorem brad_balloons (total red blue : ℕ) (h1 : total = 50) (h2 : red = 12) (h3 : blue = 7) :
  total - (red + blue) = 31 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l3583_358356


namespace NUMINAMATH_CALUDE_centripetal_acceleration_proportionality_l3583_358385

/-- Centripetal acceleration proportionality -/
theorem centripetal_acceleration_proportionality
  (a v r ω T : ℝ) (h1 : a = v^2 / r) (h2 : a = r * ω^2) (h3 : a = 4 * Real.pi^2 * r / T^2) :
  (∃ k1 : ℝ, a = k1 * (v^2 / r)) ∧
  (∃ k2 : ℝ, a = k2 * (r * ω^2)) ∧
  (∃ k3 : ℝ, a = k3 * (r / T^2)) :=
by sorry

end NUMINAMATH_CALUDE_centripetal_acceleration_proportionality_l3583_358385


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l3583_358375

/-- The number of students in both drama and science clubs at Lincoln High School -/
theorem students_in_both_clubs 
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 250)
  (h2 : drama_club = 100)
  (h3 : science_club = 130)
  (h4 : either_club = 210) :
  drama_club + science_club - either_club = 20 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l3583_358375


namespace NUMINAMATH_CALUDE_skittles_bought_proof_l3583_358372

/-- The number of Skittles Brenda initially had -/
def initial_skittles : ℕ := 7

/-- The number of Skittles Brenda ended up with -/
def final_skittles : ℕ := 15

/-- The number of Skittles Brenda bought -/
def bought_skittles : ℕ := final_skittles - initial_skittles

theorem skittles_bought_proof :
  bought_skittles = final_skittles - initial_skittles :=
by sorry

end NUMINAMATH_CALUDE_skittles_bought_proof_l3583_358372


namespace NUMINAMATH_CALUDE_chess_tournament_boys_l3583_358349

theorem chess_tournament_boys (n : ℕ) (k : ℚ) : 
  n > 2 →  -- There are more than 2 boys
  (6 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →  -- Total points equation
  (∀ m : ℕ, m > 2 ∧ m ≠ n → (6 : ℚ) + m * ((m + 2) * (m + 1) / 2 - 6) / m ≠ (m + 2) * (m + 1) / 2) →  -- n is the only solution > 2
  n = 5 ∨ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_boys_l3583_358349


namespace NUMINAMATH_CALUDE_modified_system_solution_l3583_358309

theorem modified_system_solution
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)
  (h₁ : a₁ * 4 + b₁ * 6 = c₁)
  (h₂ : a₂ * 4 + b₂ * 6 = c₂)
  : ∃ (x y : ℝ), x = 5 ∧ y = 10 ∧ 4 * a₁ * x + 3 * b₁ * y = 5 * c₁ ∧ 4 * a₂ * x + 3 * b₂ * y = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_modified_system_solution_l3583_358309


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l3583_358368

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, x^3 + 64 = 0 ↔ x = -4) ∧
  (∃ x : ℝ, (x - 2)^2 = 81 ↔ x = 11 ∨ x = -7) := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l3583_358368


namespace NUMINAMATH_CALUDE_age_difference_is_four_l3583_358359

/-- Gladys' current age -/
def gladys_age : ℕ := 40 - 10

/-- Juanico's current age -/
def juanico_age : ℕ := 41 - 30

/-- The difference between half of Gladys' age and Juanico's age -/
def age_difference : ℕ := (gladys_age / 2) - juanico_age

theorem age_difference_is_four : age_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l3583_358359


namespace NUMINAMATH_CALUDE_prime_sum_product_l3583_358330

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3583_358330


namespace NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l3583_358378

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B

/-- The perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The theorem to be proved -/
theorem max_perimeter_of_special_triangle :
  ∀ t : Triangle,
    satisfiesCondition t →
    t.c = Real.sqrt 3 →
    ∃ maxPerimeter : ℝ,
      maxPerimeter = 3 * Real.sqrt 3 ∧
      ∀ t' : Triangle,
        satisfiesCondition t' →
        t'.c = Real.sqrt 3 →
        perimeter t' ≤ maxPerimeter :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l3583_358378


namespace NUMINAMATH_CALUDE_greatest_integer_c_for_all_real_domain_l3583_358344

theorem greatest_integer_c_for_all_real_domain : 
  (∃ c : ℤ, (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_for_all_real_domain_l3583_358344


namespace NUMINAMATH_CALUDE_police_catch_thief_time_police_catch_thief_time_equals_two_l3583_358322

/-- Proves that the time taken for a police officer to catch a thief is 2 hours,
    given specific initial conditions. -/
theorem police_catch_thief_time (thief_speed : ℝ) (police_speed : ℝ) 
  (initial_distance : ℝ) (delay_time : ℝ) : ℝ :=
  by
  -- Define the conditions
  have h1 : thief_speed = 20 := by sorry
  have h2 : police_speed = 40 := by sorry
  have h3 : initial_distance = 60 := by sorry
  have h4 : delay_time = 1 := by sorry

  -- Calculate the distance covered by the thief during the delay
  let thief_distance := thief_speed * delay_time

  -- Calculate the remaining distance between the police and thief
  let remaining_distance := initial_distance - thief_distance

  -- Calculate the relative speed between police and thief
  let relative_speed := police_speed - thief_speed

  -- Calculate the time taken to catch the thief
  let catch_time := remaining_distance / relative_speed

  -- Prove that catch_time equals 2
  sorry

/-- The time taken for the police officer to catch the thief -/
def catch_time : ℝ := 2

-- Proof that the theorem result equals the defined catch_time
theorem police_catch_thief_time_equals_two :
  police_catch_thief_time 20 40 60 1 = catch_time := by sorry

end NUMINAMATH_CALUDE_police_catch_thief_time_police_catch_thief_time_equals_two_l3583_358322


namespace NUMINAMATH_CALUDE_hook_all_of_one_color_l3583_358332

/-- Represents a square sheet on the table -/
structure Sheet where
  color : Nat
  deriving Repr

/-- Represents the rectangular table with sheets -/
structure Table where
  sheets : List Sheet
  num_colors : Nat
  deriving Repr

/-- Two sheets can be hooked together -/
def can_hook (s1 s2 : Sheet) : Prop := sorry

/-- All sheets of the same color can be hooked together using the given number of hooks -/
def can_hook_color (t : Table) (c : Nat) (hooks : Nat) : Prop := sorry

/-- For any k different colored sheets, two can be hooked together -/
axiom hook_property (t : Table) :
  ∀ (diff_colored_sheets : List Sheet),
    diff_colored_sheets.length = t.num_colors →
    (∀ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets → s2 ∈ diff_colored_sheets → s1 ≠ s2 → s1.color ≠ s2.color) →
    ∃ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets ∧ s2 ∈ diff_colored_sheets ∧ s1 ≠ s2 ∧ can_hook s1 s2

/-- Main theorem: It's possible to hook all sheets of a certain color using 2k-2 hooks -/
theorem hook_all_of_one_color (t : Table) (h : t.num_colors ≥ 2) :
  ∃ (c : Nat), can_hook_color t c (2 * t.num_colors - 2) := by
  sorry

end NUMINAMATH_CALUDE_hook_all_of_one_color_l3583_358332


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3583_358392

/-- If the solution set of ax^2 + bx + c < 0 (a ≠ 0) is R, then a < 0 and b^2 - 4ac < 0 -/
theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) → 
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3583_358392


namespace NUMINAMATH_CALUDE_travel_time_calculation_l3583_358307

theorem travel_time_calculation (total_distance : ℝ) (foot_speed : ℝ) (bicycle_speed : ℝ) (foot_distance : ℝ)
  (h1 : total_distance = 61)
  (h2 : foot_speed = 4)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  (foot_distance / foot_speed) + ((total_distance - foot_distance) / bicycle_speed) = 9 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l3583_358307


namespace NUMINAMATH_CALUDE_count_terminating_decimals_is_40_l3583_358336

/-- 
Counts the number of integers n between 1 and 120 inclusive 
for which the decimal representation of n/120 terminates.
-/
def count_terminating_decimals : ℕ :=
  let max : ℕ := 120
  let prime_factors : Multiset ℕ := {2, 2, 2, 3, 5}
  sorry

/-- The count of terminating decimals is 40 -/
theorem count_terminating_decimals_is_40 : 
  count_terminating_decimals = 40 := by sorry

end NUMINAMATH_CALUDE_count_terminating_decimals_is_40_l3583_358336


namespace NUMINAMATH_CALUDE_nellie_uncle_rolls_l3583_358396

/-- Prove that Nellie sold 10 rolls to her uncle -/
theorem nellie_uncle_rolls : 
  ∀ (total_rolls grandmother_rolls neighbor_rolls remaining_rolls : ℕ),
  total_rolls = 45 →
  grandmother_rolls = 1 →
  neighbor_rolls = 6 →
  remaining_rolls = 28 →
  total_rolls - remaining_rolls - grandmother_rolls - neighbor_rolls = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_nellie_uncle_rolls_l3583_358396


namespace NUMINAMATH_CALUDE_sum_of_primes_below_1000_l3583_358352

-- Define a function that checks if a number is prime
def isPrime (n : Nat) : Prop := sorry

-- Define a function that counts the number of primes below a given number
def countPrimesBelow (n : Nat) : Nat := sorry

-- Define a function that sums all primes below a given number
def sumPrimesBelow (n : Nat) : Nat := sorry

-- Theorem statement
theorem sum_of_primes_below_1000 :
  (countPrimesBelow 1000 = 168) → (sumPrimesBelow 1000 = 76127) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_below_1000_l3583_358352


namespace NUMINAMATH_CALUDE_alice_favorite_number_l3583_358386

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alice_favorite_number :
  ∃! n : ℕ, 30 < n ∧ n < 150 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 55 := by
  sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l3583_358386


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_expression_simplification_l3583_358338

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by sorry

-- Problem 2
theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (3 * a^2 - 3 * b^2) / (a^2 * b + a * b^2) / (1 - (a^2 + b^2) / (2 * a * b)) = -6 / (a - b) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_expression_simplification_l3583_358338


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3583_358337

/-- The fixed point that the line (a+3)x + (2a-1)y + 7 = 0 passes through for all real a -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line equation as a function of a, x, and y -/
def line_equation (a x y : ℝ) : ℝ := (a + 3) * x + (2 * a - 1) * y + 7

theorem fixed_point_on_line :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3583_358337


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3583_358334

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ

/-- The other asymptote of the hyperbola -/
def otherAsymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x => 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x => -2 * x) 
  (h2 : h.foci_x = -4) : 
  otherAsymptote h = fun x => 2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3583_358334


namespace NUMINAMATH_CALUDE_expression_equality_l3583_358302

theorem expression_equality : -1^2023 + |Real.sqrt 3 - 2| - 3 * Real.tan (π / 3) = 1 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3583_358302


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l3583_358319

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of P(-3,2) reflected across the x-axis -/
theorem reflect_P_across_x_axis : 
  reflect_x (-3, 2) = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l3583_358319


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3583_358317

theorem quadratic_equation_roots (k : ℝ) (h : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, k * x₁^2 + (k + 3) * x₁ + 3 = 0 ∧ k * x₂^2 + (k + 3) * x₂ + 3 = 0) ∧
  (∀ x : ℤ, k * x^2 + (k + 3) * x + 3 = 0 → k = 1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3583_358317


namespace NUMINAMATH_CALUDE_a_spending_percentage_l3583_358345

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_percentage : ℝ) 
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : b_spending_percentage = 0.85)
  (h4 : ∃ (a_spending_percentage : ℝ), 
    a_salary * (1 - a_spending_percentage) = (total_salary - a_salary) * (1 - b_spending_percentage)) :
  ∃ (a_spending_percentage : ℝ), a_spending_percentage = 0.95 := by
sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l3583_358345


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3583_358389

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ f (-1)) ∧  -- f(-1) is the minimum value
  f (-1) = -3 ∧          -- f(-1) = -3
  f 1 = 5                -- f(1) = 5
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3583_358389
