import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l3390_339047

theorem right_triangle_sin_A (A B C : ℝ) : 
  -- ABC is a right triangle
  A + B + C = Real.pi ∧ A = Real.pi / 2 →
  -- sin B = 3/5
  Real.sin B = 3 / 5 →
  -- sin C = 4/5
  Real.sin C = 4 / 5 →
  -- sin A = 1
  Real.sin A = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l3390_339047


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3390_339039

/-- The x-intercept of the line 2x + 3y = 6 is 3 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x + 3 * y = 6 → y = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3390_339039


namespace NUMINAMATH_CALUDE_complex_equation_real_solutions_l3390_339064

theorem complex_equation_real_solutions :
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) ∧
                     (∀ a : ℝ, (∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) → a ∈ s) ∧
                     Finset.card s = 5 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_solutions_l3390_339064


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3390_339095

theorem digit_sum_problem (P Q : ℕ) (h1 : P < 10) (h2 : Q < 10) 
  (h3 : 1013 + 1000 * P + 100 * Q + 10 * P + Q = 2023) : P + Q = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3390_339095


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3390_339025

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100 * x = 9 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3390_339025


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3390_339098

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y = 32 ∧ 2 * x - y = 0) → (x = 8 ∧ y = 16) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x - 1 < 5 - 2 * x ∧ 5 * x + 1 ≥ 2 * x + 3) →
  (2 / 3 ≤ x ∧ x < 6 / 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3390_339098


namespace NUMINAMATH_CALUDE_aiyanna_cookies_l3390_339017

def alyssa_cookies : ℕ := 129
def cookie_difference : ℕ := 11

theorem aiyanna_cookies : ℕ := alyssa_cookies + cookie_difference

#check aiyanna_cookies -- This should return 140

end NUMINAMATH_CALUDE_aiyanna_cookies_l3390_339017


namespace NUMINAMATH_CALUDE_square_root_of_neg_five_squared_l3390_339075

theorem square_root_of_neg_five_squared : Real.sqrt ((-5)^2) = 5 ∨ Real.sqrt ((-5)^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_neg_five_squared_l3390_339075


namespace NUMINAMATH_CALUDE_omega_value_l3390_339018

-- Define the complex numbers z and ω
variable (z ω : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the conditions
axiom pure_imaginary : ∃ (y : ℝ), (1 + 3 * i) * z = i * y
axiom omega_def : ω = z / (2 + i)
axiom omega_abs : Complex.abs ω = 5 * Real.sqrt 2

-- State the theorem to be proved
theorem omega_value : ω = 7 - i ∨ ω = -(7 - i) := by sorry

end NUMINAMATH_CALUDE_omega_value_l3390_339018


namespace NUMINAMATH_CALUDE_fraction_simplification_l3390_339084

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) :
  ((a + b)^2 * (a^3 - b^3)) / ((a^2 - b^2)^2) = (a^2 + a*b + b^2) / (a - b) ∧
  (6*a^2*b^2 - 3*a^3*b - 3*a*b^3) / (a*b^3 - a^3*b) = 3 * (a - b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3390_339084


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l3390_339092

def vector_a : Fin 2 → ℝ := ![4, 2]

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (vector_a 0 ^ 2 + vector_a 1 ^ 2)
  let opposite_unit_vector := fun i => -vector_a i / magnitude
  opposite_unit_vector 0 = -2 * Real.sqrt 5 / 5 ∧
  opposite_unit_vector 1 = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l3390_339092


namespace NUMINAMATH_CALUDE_range_of_m_l3390_339065

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) (h : p m ∧ q m) : -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3390_339065


namespace NUMINAMATH_CALUDE_sin_negative_nineteen_sixths_pi_l3390_339072

theorem sin_negative_nineteen_sixths_pi : 
  Real.sin (-19/6 * Real.pi) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_nineteen_sixths_pi_l3390_339072


namespace NUMINAMATH_CALUDE_cubic_identity_l3390_339021

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l3390_339021


namespace NUMINAMATH_CALUDE_range_of_k_l3390_339053

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- State the theorem
theorem range_of_k (k : ℝ) : A ∩ B k = B k → -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3390_339053


namespace NUMINAMATH_CALUDE_horner_operations_l3390_339061

-- Define the polynomial coefficients
def coeffs : List ℝ := [8, 7, 6, 5, 4, 3, 2]

-- Define Horner's method
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

-- Define a function to count operations in Horner's method
def count_operations (coeffs : List ℝ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

-- Theorem statement
theorem horner_operations :
  let (mults, adds) := count_operations coeffs
  mults = 6 ∧ adds = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_operations_l3390_339061


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3390_339067

/-- The distance of a fly from the ceiling in a room -/
theorem fly_distance_from_ceiling :
  ∀ (z : ℝ),
  (2 : ℝ)^2 + 5^2 + z^2 = 7^2 →
  z = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3390_339067


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3390_339038

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 →
    1 / (x^3 + 2*x^2 - 17*x - 30) = A / (x - 5) + B / (x + 2) + C / (x + 2)^2) →
  A = 1 / 49 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3390_339038


namespace NUMINAMATH_CALUDE_m_range_l3390_339074

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3390_339074


namespace NUMINAMATH_CALUDE_brian_bought_22_pencils_l3390_339043

/-- The number of pencils Brian bought -/
def pencils_bought (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem stating that Brian bought 22 pencils -/
theorem brian_bought_22_pencils :
  pencils_bought 39 18 43 = 22 := by
  sorry

end NUMINAMATH_CALUDE_brian_bought_22_pencils_l3390_339043


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_conditions_l3390_339016

theorem least_positive_integer_satisfying_conditions : ∃ (N : ℕ), 
  (N > 1) ∧ 
  (∃ (a : ℕ), a > 0 ∧ N = a * (2 * a - 1)) ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (N - 1))) % k = 0) ∧
  (∀ (M : ℕ), M > 1 ∧ M < N → 
    (∃ (b : ℕ), b > 0 ∧ M = b * (2 * b - 1)) → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (M - 1))) % k = 0) → False) ∧
  N = 2016 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_conditions_l3390_339016


namespace NUMINAMATH_CALUDE_b_income_less_than_others_l3390_339052

structure Income where
  c : ℝ
  a : ℝ
  b_salary : ℝ
  b_commission : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

def Income.b_total (i : Income) : ℝ := i.b_salary + i.b_commission

def Income.others_total (i : Income) : ℝ := i.a + i.c + i.d + i.e + i.f

def valid_income (i : Income) : Prop :=
  i.a = i.c * 1.2 ∧
  i.b_salary = i.a * 1.25 ∧
  i.b_commission = (i.a + i.c) * 0.05 ∧
  i.d = i.b_total * 0.85 ∧
  i.e = i.c * 1.1 ∧
  i.f = (i.b_total + i.e) / 2

theorem b_income_less_than_others (i : Income) (h : valid_income i) :
  i.b_total < i.others_total ∧ i.b_commission = i.c * 0.11 :=
sorry

end NUMINAMATH_CALUDE_b_income_less_than_others_l3390_339052


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3390_339041

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ * x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3390_339041


namespace NUMINAMATH_CALUDE_roberts_cash_amount_l3390_339013

theorem roberts_cash_amount (raw_materials_cost machinery_cost : ℝ) 
  (h1 : raw_materials_cost = 100)
  (h2 : machinery_cost = 125)
  (total_amount : ℝ) :
  raw_materials_cost + machinery_cost + 0.1 * total_amount = total_amount →
  total_amount = 250 := by
sorry

end NUMINAMATH_CALUDE_roberts_cash_amount_l3390_339013


namespace NUMINAMATH_CALUDE_coffee_stock_calculation_l3390_339034

/-- Represents the initial amount of coffee in stock -/
def initial_stock : ℝ := sorry

/-- The fraction of initial stock that is decaffeinated -/
def initial_decaf_fraction : ℝ := 0.4

/-- The amount of new coffee purchased -/
def new_purchase : ℝ := 100

/-- The fraction of new purchase that is decaffeinated -/
def new_decaf_fraction : ℝ := 0.6

/-- The fraction of total stock that is decaffeinated after the purchase -/
def final_decaf_fraction : ℝ := 0.44

theorem coffee_stock_calculation :
  initial_stock = 400 ∧
  final_decaf_fraction * (initial_stock + new_purchase) =
    initial_decaf_fraction * initial_stock + new_decaf_fraction * new_purchase :=
sorry

end NUMINAMATH_CALUDE_coffee_stock_calculation_l3390_339034


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l3390_339048

theorem smallest_marble_collection : ∀ n : ℕ,
  (n % 4 = 0) →  -- one fourth are red
  (n % 5 = 0) →  -- one fifth are blue
  (n ≥ 8 + 5) →  -- at least 8 white and 5 green
  (∃ r b w g : ℕ, 
    r + b + w + g = n ∧
    r = n / 4 ∧
    b = n / 5 ∧
    w = 8 ∧
    g = 5) →
  n ≥ 220 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l3390_339048


namespace NUMINAMATH_CALUDE_four_five_equality_and_precision_l3390_339062

/-- Represents a decimal number with its value and precision -/
structure Decimal where
  value : ℚ
  precision : ℕ

/-- 4.5 as a Decimal -/
def d1 : Decimal := { value := 4.5, precision := 1 }

/-- 4.50 as a Decimal -/
def d2 : Decimal := { value := 4.5, precision := 2 }

/-- Two Decimals are equal in magnitude if their values are equal -/
def equal_magnitude (a b : Decimal) : Prop := a.value = b.value

/-- Two Decimals differ in precision if their precisions are different -/
def differ_precision (a b : Decimal) : Prop := a.precision ≠ b.precision

/-- Theorem stating that 4.5 and 4.50 are equal in magnitude but differ in precision -/
theorem four_five_equality_and_precision : 
  equal_magnitude d1 d2 ∧ differ_precision d1 d2 := by
  sorry

end NUMINAMATH_CALUDE_four_five_equality_and_precision_l3390_339062


namespace NUMINAMATH_CALUDE_floor_T_equals_120_l3390_339059

-- Define positive real numbers p, q, r, s
variable (p q r s : ℝ)

-- Define the conditions
axiom p_pos : p > 0
axiom q_pos : q > 0
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom sum_squares_pq : p^2 + q^2 = 2500
axiom sum_squares_rs : r^2 + s^2 = 2500
axiom product_pr : p * r = 1200
axiom product_qs : q * s = 1200

-- Define T
def T : ℝ := p + q + r + s

-- Theorem to prove
theorem floor_T_equals_120 : ⌊T p q r s⌋ = 120 := by sorry

end NUMINAMATH_CALUDE_floor_T_equals_120_l3390_339059


namespace NUMINAMATH_CALUDE_circle_equation_specific_l3390_339094

/-- The standard equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (2, -1) and radius 3 -/
theorem circle_equation_specific : ∀ x y : ℝ,
  circle_equation x y 2 (-1) 3 ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l3390_339094


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l3390_339029

theorem fraction_multiplication_addition : (2 / 9 : ℚ) * (5 / 6 : ℚ) + (1 / 18 : ℚ) = (13 / 54 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l3390_339029


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3390_339093

/-- The range of a, given the conditions in the problem -/
def range_of_a : Set ℝ :=
  {a | a ≤ -2 ∨ a = 1}

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

/-- Proposition q: There exists x₀ ∈ ℝ such that x₀^2 + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The main theorem stating that given the conditions, the range of a is as defined -/
theorem range_of_a_theorem (a : ℝ) :
  (prop_p a ∧ prop_q a) → a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3390_339093


namespace NUMINAMATH_CALUDE_trig_identity_l3390_339019

theorem trig_identity (α : ℝ) : 
  Real.sin (π + α)^2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3390_339019


namespace NUMINAMATH_CALUDE_largest_divisor_of_factorial_l3390_339076

theorem largest_divisor_of_factorial (m n : ℕ) (hm : m ≥ 3) (hn : n > m * (m - 2)) :
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d)) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d) ∧
    ∀ d' > 0, d' ∣ n.factorial → (∀ k ∈ Finset.Icc m n, ¬(k ∣ d')) → d' ≤ d) →
  (m - 1 : ℕ) > 0 ∧ (m - 1 : ℕ) ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ (m - 1 : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_factorial_l3390_339076


namespace NUMINAMATH_CALUDE_four_Z_three_equals_one_l3390_339007

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

-- Theorem to prove
theorem four_Z_three_equals_one : Z 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_one_l3390_339007


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l3390_339083

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leap_year_frequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_frequency

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40,
    given that leap years occur every 5 years -/
theorem max_leap_years_in_period :
  max_leap_years = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l3390_339083


namespace NUMINAMATH_CALUDE_cos_36_cos_24_minus_sin_36_sin_24_l3390_339014

theorem cos_36_cos_24_minus_sin_36_sin_24 :
  Real.cos (36 * π / 180) * Real.cos (24 * π / 180) -
  Real.sin (36 * π / 180) * Real.sin (24 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_cos_24_minus_sin_36_sin_24_l3390_339014


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3390_339066

theorem sum_of_three_numbers : 300 + 2020 + 10001 = 12321 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3390_339066


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3390_339080

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 1 → parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The condition that a = 1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The main theorem stating that a = 1 is sufficient but not necessary -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 1 → parallel 1 a (-1) (2*a - 1) a (-2)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2)) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3390_339080


namespace NUMINAMATH_CALUDE_exists_nonperiodic_sequence_satisfying_property_l3390_339087

/-- A sequence of natural numbers. -/
def Sequence := ℕ → ℕ

/-- A sequence satisfies the given property if for any k, there exists a t such that
    the sequence remains constant when we add multiples of t to k. -/
def SatisfiesProperty (a : Sequence) : Prop :=
  ∀ k, ∃ t, ∀ m, a k = a (k + m * t)

/-- A sequence is periodic if there exists a period T such that
    for all k, a(k) = a(k + T). -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T, ∀ k, a k = a (k + T)

/-- There exists a sequence that satisfies the property but is not periodic. -/
theorem exists_nonperiodic_sequence_satisfying_property :
  ∃ a : Sequence, SatisfiesProperty a ∧ ¬IsPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_exists_nonperiodic_sequence_satisfying_property_l3390_339087


namespace NUMINAMATH_CALUDE_polynomial_root_difference_sum_l3390_339006

theorem polynomial_root_difference_sum (a b c d : ℝ) (x₁ x₂ : ℝ) : 
  a + b + c = 0 →
  a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 →
  a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 →
  x₁ = 1 →
  a ≥ b →
  b ≥ c →
  a > 0 →
  c < 0 →
  ∃ (min_val max_val : ℝ),
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≥ min_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = min_val) ∧
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≤ max_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = max_val) ∧
    min_val + max_val = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_difference_sum_l3390_339006


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3390_339045

theorem complex_magnitude_problem (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3390_339045


namespace NUMINAMATH_CALUDE_runner_stop_time_l3390_339060

theorem runner_stop_time (total_distance : ℝ) (first_pace second_pace stop_time : ℝ) 
  (h1 : total_distance = 10)
  (h2 : first_pace = 8)
  (h3 : second_pace = 7)
  (h4 : stop_time = 8)
  (h5 : first_pace > second_pace)
  (h6 : stop_time / (first_pace - second_pace) + 
        (stop_time / (first_pace - second_pace)) * second_pace = total_distance) :
  (stop_time / (first_pace - second_pace)) * second_pace = 56 := by
  sorry


end NUMINAMATH_CALUDE_runner_stop_time_l3390_339060


namespace NUMINAMATH_CALUDE_speed_ratio_with_head_start_l3390_339086

/-- The ratio of speeds in a race where one runner has a head start -/
theorem speed_ratio_with_head_start (vA vB : ℝ) (h : vA > 0 ∧ vB > 0) : 
  (120 / vA = 60 / vB) → vA / vB = 2 := by
  sorry

#check speed_ratio_with_head_start

end NUMINAMATH_CALUDE_speed_ratio_with_head_start_l3390_339086


namespace NUMINAMATH_CALUDE_male_athletes_to_sample_l3390_339030

def total_athletes : ℕ := 98
def female_athletes : ℕ := 42
def selection_probability : ℚ := 2/7

def male_athletes : ℕ := total_athletes - female_athletes

theorem male_athletes_to_sample :
  ⌊(male_athletes : ℚ) * selection_probability⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_male_athletes_to_sample_l3390_339030


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3390_339027

theorem quadratic_equation_equivalence : 
  (∀ x, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3390_339027


namespace NUMINAMATH_CALUDE_pencil_theorem_l3390_339015

def pencil_problem (anna_pencils : ℕ) (harry_pencils : ℕ) (lost_pencils : ℕ) : Prop :=
  anna_pencils = 50 ∧
  harry_pencils = 2 * anna_pencils ∧
  harry_pencils - lost_pencils = 81

theorem pencil_theorem : 
  ∃ (anna_pencils harry_pencils lost_pencils : ℕ),
    pencil_problem anna_pencils harry_pencils lost_pencils ∧ lost_pencils = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_theorem_l3390_339015


namespace NUMINAMATH_CALUDE_nathan_total_earnings_l3390_339032

/-- Nathan's hourly wage in dollars -/
def hourly_wage : ℝ := 6

/-- Hours worked in the second week of July -/
def hours_week2 : ℝ := 12

/-- Hours worked in the third week of July -/
def hours_week3 : ℝ := 18

/-- Earnings difference between the third and second week -/
def earnings_difference : ℝ := 36

theorem nathan_total_earnings : 
  hourly_wage * hours_week2 + hourly_wage * hours_week3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_nathan_total_earnings_l3390_339032


namespace NUMINAMATH_CALUDE_andy_position_after_2021_moves_l3390_339050

-- Define the ant's position as a pair of integers
def Position := ℤ × ℤ

-- Define the direction as an enumeration
inductive Direction
| North
| East
| South
| West

-- Define the initial position and direction
def initial_position : Position := (10, -10)
def initial_direction : Direction := Direction.North

-- Define a function to calculate the movement distance for a given move number
def movement_distance (move_number : ℕ) : ℕ :=
  (move_number / 4 : ℕ) + 1

-- Define a function to update the direction after a right turn
def turn_right (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

-- Define a function to update the position based on direction and distance
def move (pos : Position) (dir : Direction) (dist : ℤ) : Position :=
  match dir with
  | Direction.North => (pos.1, pos.2 + dist)
  | Direction.East => (pos.1 + dist, pos.2)
  | Direction.South => (pos.1, pos.2 - dist)
  | Direction.West => (pos.1 - dist, pos.2)

-- Define a function to simulate the ant's movement for a given number of moves
def simulate_movement (num_moves : ℕ) : Position :=
  sorry -- Actual implementation would go here

-- State the theorem
theorem andy_position_after_2021_moves :
  simulate_movement 2021 = (10, 496) := by sorry

end NUMINAMATH_CALUDE_andy_position_after_2021_moves_l3390_339050


namespace NUMINAMATH_CALUDE_negation_of_union_l3390_339003

theorem negation_of_union (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_l3390_339003


namespace NUMINAMATH_CALUDE_cone_section_properties_l3390_339033

/-- Given a right circular cone with base radius 25 cm and slant height 42 cm,
    when cut by a plane parallel to the base such that the volumes of the two resulting parts are equal,
    the radius of the circular intersection is 25 * (1/2)^(1/3) cm
    and the height of the smaller cone is sqrt(1139) * (1/2)^(1/3) cm. -/
theorem cone_section_properties :
  let base_radius : ℝ := 25
  let slant_height : ℝ := 42
  let cone_height : ℝ := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  let section_radius : ℝ := base_radius * (1/2) ^ (1/3)
  let small_cone_height : ℝ := cone_height * (1/2) ^ (1/3)
  (1/3) * Real.pi * base_radius ^ 2 * cone_height = 2 * ((1/3) * Real.pi * section_radius ^ 2 * small_cone_height) →
  section_radius = 25 * (1/2) ^ (1/3) ∧ small_cone_height = Real.sqrt 1139 * (1/2) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_cone_section_properties_l3390_339033


namespace NUMINAMATH_CALUDE_dividend_calculation_l3390_339044

theorem dividend_calculation (D d Q R : ℕ) 
  (eq_condition : D = d * Q + R)
  (d_value : d = 17)
  (Q_value : Q = 9)
  (R_value : R = 9) :
  D = 162 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3390_339044


namespace NUMINAMATH_CALUDE_committee_size_l3390_339091

theorem committee_size (n : ℕ) : 
  (n * (n - 1) = 42) → n = 7 := by
  sorry

#check committee_size

end NUMINAMATH_CALUDE_committee_size_l3390_339091


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l3390_339022

def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8

def total_pieces : ℕ := chairs + tables

def total_time : ℕ := total_pieces * time_per_piece

theorem furniture_assembly_time : total_time = 32 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l3390_339022


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3390_339097

/-- Given two parallel vectors a and b, prove that the magnitude of b is √13 -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![(-4), 6]
  let b : Fin 2 → ℝ := ![2, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →  -- Parallel vectors condition
  Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3390_339097


namespace NUMINAMATH_CALUDE_periodic_function_property_l3390_339056

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) where f(2009) = 3, 
    prove that f(2010) = -3 -/
theorem periodic_function_property (a b α β : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β))
  (h2 : f 2009 = 3) : 
  f 2010 = -3 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_property_l3390_339056


namespace NUMINAMATH_CALUDE_collinear_probability_5x4_l3390_339081

/-- Represents a rectangular array of dots. -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of collinear sets of 4 dots in a 5x4 array. -/
def collinearSets (arr : DotArray) : ℕ := arr.cols * choose arr.rows 4

/-- The total number of ways to choose 4 dots from the array. -/
def totalChoices (arr : DotArray) : ℕ := choose (arr.rows * arr.cols) 4

/-- The probability of choosing 4 collinear dots. -/
def collinearProbability (arr : DotArray) : ℚ :=
  collinearSets arr / totalChoices arr

/-- Theorem: The probability of choosing 4 collinear dots in a 5x4 array is 4/969. -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 4 / 969 := by
  sorry

end NUMINAMATH_CALUDE_collinear_probability_5x4_l3390_339081


namespace NUMINAMATH_CALUDE_intersection_cylinders_in_sphere_l3390_339054

/-- Theorem: Intersection of three perpendicular unit cylinders is contained in a sphere of radius √(3/2) -/
theorem intersection_cylinders_in_sphere (a b c d e f : ℝ) (x y z : ℝ) : 
  ((x - a)^2 + (y - b)^2 ≤ 1) →
  ((y - c)^2 + (z - d)^2 ≤ 1) →
  ((z - e)^2 + (x - f)^2 ≤ 1) →
  ∃ (center_x center_y center_z : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_cylinders_in_sphere_l3390_339054


namespace NUMINAMATH_CALUDE_P_speed_is_8_l3390_339085

/-- Represents the cycling speed of P in kmph -/
def P_speed : ℝ := 8

/-- J's walking speed in kmph -/
def J_speed : ℝ := 6

/-- Time (in hours) J walks before P starts -/
def time_before_P_starts : ℝ := 1.5

/-- Total time (in hours) from J's start to the point where J is 3 km behind P -/
def total_time : ℝ := 7.5

/-- Distance (in km) J is behind P at the end -/
def distance_behind : ℝ := 3

theorem P_speed_is_8 :
  P_speed = 8 :=
sorry

#check P_speed_is_8

end NUMINAMATH_CALUDE_P_speed_is_8_l3390_339085


namespace NUMINAMATH_CALUDE_range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l3390_339069

/-- The function f(x) = x^2 - (m-1)x + 2m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

/-- Theorem 1: f(x) > 0 for all x in (0, +∞) iff -2√6 + 5 ≤ m ≤ 2√6 + 5 -/
theorem range_of_m_for_positive_f (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 :=
sorry

/-- Theorem 2: f(x) has a zero point in (0, 1) iff m ∈ (-2, 0) -/
theorem range_of_m_for_zero_in_interval (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m > -2 ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l3390_339069


namespace NUMINAMATH_CALUDE_mario_earnings_l3390_339011

/-- Mario's work hours and earnings over two weeks in July --/
theorem mario_earnings :
  ∀ (third_week_hours second_week_hours : ℕ) 
    (hourly_rate third_week_earnings second_week_earnings : ℚ),
  third_week_hours = 28 →
  third_week_hours = second_week_hours + 10 →
  third_week_earnings = second_week_earnings + 68 →
  hourly_rate * (third_week_hours : ℚ) = third_week_earnings →
  hourly_rate * (second_week_hours : ℚ) = second_week_earnings →
  hourly_rate * ((third_week_hours + second_week_hours) : ℚ) = 312.8 :=
by sorry

end NUMINAMATH_CALUDE_mario_earnings_l3390_339011


namespace NUMINAMATH_CALUDE_last_infected_on_fifth_exam_l3390_339089

def total_mice : ℕ := 10
def infected_mice : ℕ := 3
def healthy_mice : ℕ := 7

-- The number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to arrange k items from n items
def arrange (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem last_infected_on_fifth_exam :
  choose infected_mice 2 * arrange 4 2 * choose healthy_mice 2 * arrange 2 2 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_last_infected_on_fifth_exam_l3390_339089


namespace NUMINAMATH_CALUDE_alpha_beta_range_l3390_339026

theorem alpha_beta_range (α β : ℝ) 
  (h1 : 0 < α - β) (h2 : α - β < π) 
  (h3 : 0 < α + 2*β) (h4 : α + 2*β < π) : 
  0 < α + β ∧ α + β < π :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l3390_339026


namespace NUMINAMATH_CALUDE_goldbach_negation_equiv_l3390_339040

-- Define Goldbach's Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define the negation of Goldbach's Conjecture
def not_goldbach : Prop :=
  ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Theorem stating the equivalence
theorem goldbach_negation_equiv :
  ¬goldbach_conjecture ↔ not_goldbach := by sorry

end NUMINAMATH_CALUDE_goldbach_negation_equiv_l3390_339040


namespace NUMINAMATH_CALUDE_b_completion_time_l3390_339023

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℚ := 15

/-- The number of days A actually works -/
def a_worked_days : ℚ := 5

/-- The number of days B needs to complete the entire work -/
def b_total_days : ℚ := 9/2

/-- The fraction of work completed by A -/
def a_work_completed : ℚ := a_worked_days / a_total_days

/-- The fraction of work B needs to complete -/
def b_work_to_complete : ℚ := 1 - a_work_completed

/-- The fraction of work B completes per day -/
def b_work_per_day : ℚ := 1 / b_total_days

/-- The number of days B needs to complete the remaining work -/
def b_days_needed : ℚ := b_work_to_complete / b_work_per_day

theorem b_completion_time : b_days_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l3390_339023


namespace NUMINAMATH_CALUDE_robins_hair_length_l3390_339042

/-- Given that Robin's hair is currently 13 inches long after cutting off 4 inches,
    prove that his initial hair length was 17 inches. -/
theorem robins_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 13)
  (h2 : cut_length = 4) : 
  current_length + cut_length = 17 := by
sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3390_339042


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3390_339070

theorem right_angled_triangle (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  a ^ 2 + b ^ 2 = c ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3390_339070


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3390_339035

/-- The perimeter of a rectangle with length 100 and breadth 500 is 1200. -/
theorem rectangle_perimeter : 
  ∀ (length breadth perimeter : ℕ), 
    length = 100 → 
    breadth = 500 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 1200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3390_339035


namespace NUMINAMATH_CALUDE_equation_proof_l3390_339058

theorem equation_proof (x : ℚ) : x = 5 → 65 + (x * 12) / 60 = 66 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3390_339058


namespace NUMINAMATH_CALUDE_problem_1_l3390_339099

theorem problem_1 (α : Real) (h : Real.sin α - 2 * Real.cos α = 0) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3390_339099


namespace NUMINAMATH_CALUDE_total_musicians_l3390_339002

/-- Represents a musical group with a specific number of male and female musicians. -/
structure MusicGroup where
  males : Nat
  females : Nat

/-- The total number of musicians in a group is the sum of males and females. -/
def MusicGroup.total (g : MusicGroup) : Nat :=
  g.males + g.females

/-- The orchestra has 11 males and 12 females. -/
def orchestra : MusicGroup :=
  { males := 11, females := 12 }

/-- The band has twice the number of musicians as the orchestra. -/
def band : MusicGroup :=
  { males := 2 * orchestra.males, females := 2 * orchestra.females }

/-- The choir has 12 males and 17 females. -/
def choir : MusicGroup :=
  { males := 12, females := 17 }

/-- Theorem: The total number of musicians in the orchestra, band, and choir is 98. -/
theorem total_musicians :
  orchestra.total + band.total + choir.total = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_musicians_l3390_339002


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3390_339012

/-- Given an isosceles triangle with base b and height h, and a rectangle with base b and height 2b,
    if their areas are equal, then the height of the triangle is 4 times the base. -/
theorem isosceles_triangle_rectangle_equal_area (b h : ℝ) (b_pos : 0 < b) :
  (1 / 2 : ℝ) * b * h = b * (2 * b) → h = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3390_339012


namespace NUMINAMATH_CALUDE_interesting_quartet_inequality_l3390_339088

theorem interesting_quartet_inequality (p a b c : ℕ) : 
  Nat.Prime p → p % 2 = 1 →
  a ≠ b → b ≠ c → a ≠ c →
  (ab + 1) % p = 0 →
  (ac + 1) % p = 0 →
  (bc + 1) % p = 0 →
  (p : ℚ) + 2 ≤ (a + b + c : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_interesting_quartet_inequality_l3390_339088


namespace NUMINAMATH_CALUDE_total_pears_is_five_l3390_339079

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

/-- Theorem: The total number of pears picked is 5 -/
theorem total_pears_is_five : total_pears = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_five_l3390_339079


namespace NUMINAMATH_CALUDE_bird_nest_difference_l3390_339055

theorem bird_nest_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 := by sorry

end NUMINAMATH_CALUDE_bird_nest_difference_l3390_339055


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_three_l3390_339031

theorem six_digit_multiple_of_three : ∃ (n : ℕ), 325473 = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_three_l3390_339031


namespace NUMINAMATH_CALUDE_chefs_wage_difference_l3390_339073

/-- Proves that the difference between the total hourly wage of 3 managers
    and the total hourly wage of 3 chefs is $3.9375, given the specified conditions. -/
theorem chefs_wage_difference (manager_wage : ℝ) (num_chefs num_dishwashers : ℕ) :
  manager_wage = 8.5 →
  num_chefs = 3 →
  num_dishwashers = 4 →
  let first_dishwasher_wage := manager_wage / 2
  let dishwasher_wages := [
    first_dishwasher_wage,
    first_dishwasher_wage + 1.5,
    first_dishwasher_wage + 3,
    first_dishwasher_wage + 4.5
  ]
  let chef_wages := (List.take num_chefs dishwasher_wages).map (λ w => w * 1.25)
  (3 * manager_wage - chef_wages.sum) = 3.9375 := by
  sorry

end NUMINAMATH_CALUDE_chefs_wage_difference_l3390_339073


namespace NUMINAMATH_CALUDE_dance_cost_theorem_l3390_339063

/-- Represents the cost calculation for dance shoes and fans. -/
structure DanceCost where
  x : ℝ  -- Number of fans per pair of shoes
  yA : ℝ -- Cost at supermarket A
  yB : ℝ -- Cost at supermarket B

/-- Calculates the cost for dance shoes and fans given the conditions. -/
def calculate_cost (x : ℝ) : DanceCost :=
  { x := x
  , yA := 27 * x + 270
  , yB := 30 * x + 240 }

/-- Theorem stating the relationship between costs and number of fans. -/
theorem dance_cost_theorem (x : ℝ) (h : x ≥ 2) :
  let cost := calculate_cost x
  cost.yA = 27 * x + 270 ∧
  cost.yB = 30 * x + 240 ∧
  (x < 10 → cost.yB < cost.yA) ∧
  (x = 10 → cost.yB = cost.yA) ∧
  (x > 10 → cost.yA < cost.yB) := by
  sorry

#check dance_cost_theorem

end NUMINAMATH_CALUDE_dance_cost_theorem_l3390_339063


namespace NUMINAMATH_CALUDE_sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l3390_339010

theorem sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two :
  Real.sqrt 50 - Real.sqrt 32 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l3390_339010


namespace NUMINAMATH_CALUDE_complex_circle_range_l3390_339068

theorem complex_circle_range (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (Complex.abs (z - Complex.mk 3 4) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_range_l3390_339068


namespace NUMINAMATH_CALUDE_number_equation_solution_l3390_339096

theorem number_equation_solution : 
  ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3390_339096


namespace NUMINAMATH_CALUDE_add_base6_example_l3390_339005

/-- Represents a number in base 6 --/
def Base6 : Type := Fin 6 → ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def fromBase6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- The number 5 in base 6 --/
def five_base6 : Base6 := toBase6 5

/-- The number 23 in base 6 --/
def twentythree_base6 : Base6 := toBase6 23

/-- The number 32 in base 6 --/
def thirtytwo_base6 : Base6 := toBase6 32

theorem add_base6_example : addBase6 five_base6 twentythree_base6 = thirtytwo_base6 := by
  sorry

end NUMINAMATH_CALUDE_add_base6_example_l3390_339005


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3390_339037

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m n : ℝ) 
  (hmn : m * n = 2 / 9) :
  (((m + n) * c)^2 / a^2 - ((m - n) * b * c / a)^2 / b^2 = 1) →
  (c^2 / a^2 - 1 = (3 * Real.sqrt 2 / 4)^2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3390_339037


namespace NUMINAMATH_CALUDE_rectangles_not_always_similar_l3390_339049

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem statement
theorem rectangles_not_always_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end NUMINAMATH_CALUDE_rectangles_not_always_similar_l3390_339049


namespace NUMINAMATH_CALUDE_september_1_2017_is_friday_l3390_339009

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def march_19_2017 : Date :=
  { year := 2017, month := 3, day := 19 }

def september_1_2017 : Date :=
  { year := 2017, month := 9, day := 1 }

/-- Returns the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Calculates the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

theorem september_1_2017_is_friday :
  dayOfWeek march_19_2017 = DayOfWeek.Sunday →
  dayOfWeek september_1_2017 = DayOfWeek.Friday :=
by
  sorry

#check september_1_2017_is_friday

end NUMINAMATH_CALUDE_september_1_2017_is_friday_l3390_339009


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3390_339004

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_externally_tangent :
  let d := Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)
  d = radius₁ + radius₂ := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3390_339004


namespace NUMINAMATH_CALUDE_sapling_growth_l3390_339051

/-- The height of a sapling after n years -/
def sapling_height (n : ℕ) : ℝ :=
  1.5 + 0.2 * n

/-- Theorem: The height of the sapling after n years is 1.5 + 0.2n meters -/
theorem sapling_growth (n : ℕ) :
  sapling_height n = 1.5 + 0.2 * n := by
  sorry

end NUMINAMATH_CALUDE_sapling_growth_l3390_339051


namespace NUMINAMATH_CALUDE_win_sector_area_l3390_339028

/-- The area of the WIN sector on a circular spinner with given radius and win probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 8) (h_p : p = 3/7) :
  p * π * r^2 = (192 * π) / 7 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l3390_339028


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3390_339024

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 2 = 5 → 4*y - 2*x + 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3390_339024


namespace NUMINAMATH_CALUDE_square_transformation_2007_l3390_339078

-- Define the vertex order as a list of characters
def VertexOrder := List Char

-- Define the transformation operations
def rotate90Clockwise (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, a, b, c]
  | _ => order

def reflectVertical (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, c, b, a]
  | _ => order

def reflectHorizontal (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [c, b, a, d]
  | _ => order

-- Define the complete transformation sequence
def transformSequence (order : VertexOrder) : VertexOrder :=
  reflectHorizontal (reflectVertical (rotate90Clockwise order))

-- Define a function to apply the transformation sequence n times
def applyTransformSequence (order : VertexOrder) (n : Nat) : VertexOrder :=
  match n with
  | 0 => order
  | n + 1 => applyTransformSequence (transformSequence order) n

-- Theorem statement
theorem square_transformation_2007 :
  applyTransformSequence ['A', 'B', 'C', 'D'] 2007 = ['D', 'C', 'B', 'A'] := by
  sorry


end NUMINAMATH_CALUDE_square_transformation_2007_l3390_339078


namespace NUMINAMATH_CALUDE_frequency_calculation_l3390_339020

/-- Given a sample capacity and a frequency rate, calculate the frequency of a group of samples. -/
def calculate_frequency (sample_capacity : ℕ) (frequency_rate : ℚ) : ℚ :=
  frequency_rate * sample_capacity

/-- Theorem: Given a sample capacity of 32 and a frequency rate of 0.125, the frequency is 4. -/
theorem frequency_calculation :
  let sample_capacity : ℕ := 32
  let frequency_rate : ℚ := 1/8
  calculate_frequency sample_capacity frequency_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_frequency_calculation_l3390_339020


namespace NUMINAMATH_CALUDE_simon_treasures_l3390_339008

/-- The number of sand dollars Simon collected -/
def sand_dollars : ℕ := sorry

/-- The number of sea glass pieces Simon collected -/
def sea_glass : ℕ := sorry

/-- The number of seashells Simon collected -/
def seashells : ℕ := sorry

/-- The total number of treasures Simon collected -/
def total_treasures : ℕ := 190

theorem simon_treasures : 
  sea_glass = 3 * sand_dollars ∧ 
  seashells = 5 * sea_glass ∧
  total_treasures = sand_dollars + sea_glass + seashells →
  sand_dollars = 10 := by sorry

end NUMINAMATH_CALUDE_simon_treasures_l3390_339008


namespace NUMINAMATH_CALUDE_triangle_tangent_identity_l3390_339000

theorem triangle_tangent_identity (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  Real.tan (A/2) * Real.tan (B/2) + Real.tan (B/2) * Real.tan (C/2) + Real.tan (C/2) * Real.tan (A/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_identity_l3390_339000


namespace NUMINAMATH_CALUDE_three_cones_problem_l3390_339071

/-- A cone with vertex A -/
structure Cone (A : Point) where
  vertex_angle : ℝ

/-- A plane passing through a point -/
structure Plane (A : Point)

/-- Three cones touching each other externally -/
def touching_cones (A : Point) (c1 c2 c3 : Cone A) : Prop :=
  sorry

/-- Two cones are identical -/
def identical_cones (c1 c2 : Cone A) : Prop :=
  c1.vertex_angle = c2.vertex_angle

/-- A cone touches a plane -/
def cone_touches_plane (c : Cone A) (p : Plane A) : Prop :=
  sorry

/-- A cone lies on one side of a plane -/
def cone_on_one_side (c : Cone A) (p : Plane A) : Prop :=
  sorry

theorem three_cones_problem (A : Point) (c1 c2 c3 : Cone A) (p : Plane A) :
  touching_cones A c1 c2 c3 →
  identical_cones c1 c2 →
  c3.vertex_angle = π / 2 →
  cone_touches_plane c1 p →
  cone_touches_plane c2 p →
  cone_touches_plane c3 p →
  cone_on_one_side c1 p →
  cone_on_one_side c2 p →
  cone_on_one_side c3 p →
  c1.vertex_angle = 2 * Real.arctan (4 / 5) :=
sorry

end NUMINAMATH_CALUDE_three_cones_problem_l3390_339071


namespace NUMINAMATH_CALUDE_g_is_even_l3390_339077

open Real

/-- A function F is odd if F(-x) = -F(x) for all x -/
def IsOdd (F : ℝ → ℝ) : Prop := ∀ x, F (-x) = -F x

/-- A function G is even if G(-x) = G(x) for all x -/
def IsEven (G : ℝ → ℝ) : Prop := ∀ x, G (-x) = G x

/-- Given a > 0, a ≠ 1, and F is an odd function, prove that G(x) = F(x) * (1 / (a^x - 1) + 1/2) is an even function -/
theorem g_is_even (a : ℝ) (ha : a > 0) (hna : a ≠ 1) (F : ℝ → ℝ) (hF : IsOdd F) :
  IsEven (fun x ↦ F x * (1 / (a^x - 1) + 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l3390_339077


namespace NUMINAMATH_CALUDE_power_function_through_point_l3390_339057

/-- Given a power function f(x) = x^α that passes through the point (9,3), prove that f(100) = 10 -/
theorem power_function_through_point (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x ^ α) 
  (h2 : f 9 = 3) : 
  f 100 = 10 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3390_339057


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3390_339082

theorem quadratic_equation_condition (x : ℝ) :
  (x^2 + 2*x - 3 = 0 ↔ (x = -3 ∨ x = 1)) →
  (x = 1 → x^2 + 2*x - 3 = 0) ∧
  ¬(x^2 + 2*x - 3 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3390_339082


namespace NUMINAMATH_CALUDE_problem_solution_l3390_339001

theorem problem_solution :
  ∀ (E F D : ℕ),
    E + F + D = 15 →
    F + E + 1 = 12 →
    E < 10 ∧ F < 10 ∧ D < 10 →
    E ≠ F ∧ F ≠ D ∧ E ≠ D →
    D = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3390_339001


namespace NUMINAMATH_CALUDE_problem_solution_l3390_339036

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + 2 * y - 6 = 0
def equation2 (x y m : ℝ) : Prop := x - 2 * y + m * x + 5 = 0

theorem problem_solution :
  -- Part 1: Positive integer solutions
  (∀ x y : ℕ+, equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1)) ∧
  -- Part 2: Value of m when x + y = 0
  (∀ x y m : ℝ, x + y = 0 → equation1 x y → equation2 x y m → m = -13/6) ∧
  -- Part 3: Fixed solution regardless of m
  (∀ m : ℝ, equation2 0 (5/2) m) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3390_339036


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3390_339046

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) :
  Set.Ioo (-1/2 : ℝ) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3390_339046


namespace NUMINAMATH_CALUDE_pi_comparison_l3390_339090

theorem pi_comparison : -Real.pi < -3.14 := by sorry

end NUMINAMATH_CALUDE_pi_comparison_l3390_339090
