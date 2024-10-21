import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daniel_drive_distance_l694_69478

/-- The distance Daniel drives back from work every day -/
noncomputable def D : ℝ := sorry

/-- Daniel's speed on Sunday in miles per hour -/
noncomputable def x : ℝ := sorry

/-- Time taken to drive back on Sunday in hours -/
noncomputable def T_sunday : ℝ := D / x

/-- Time taken to drive back on Monday in hours -/
noncomputable def T_monday : ℝ := (2 * D - 48) / x

/-- Theorem stating that Daniel drives 100 miles back from work every day -/
theorem daniel_drive_distance :
  (T_monday = 1.52 * T_sunday) → D = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daniel_drive_distance_l694_69478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l694_69430

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define the properties of the triangle
def TriangleProperties (t : AcuteTriangle) : Prop :=
  2 * (Real.cos ((t.B + t.C) / 2))^2 + Real.sin (2 * t.A) = 1 ∧
  t.a = 2 * Real.sqrt 3 - 2 ∧
  1/2 * t.b * t.c * Real.sin t.A = 2

-- State the theorem
theorem triangle_theorem (t : AcuteTriangle) 
  (h : TriangleProperties t) : t.A = Real.pi/6 ∧ t.b + t.c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l694_69430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_21_factors_l694_69456

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The smallest positive integer with exactly 21 factors -/
def smallest_with_21_factors : ℕ := 576

theorem smallest_21_factors :
  (∀ m : ℕ, m > 0 → m < smallest_with_21_factors → num_factors m ≠ 21) ∧
  num_factors smallest_with_21_factors = 21 :=
sorry

#eval num_factors smallest_with_21_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_21_factors_l694_69456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l694_69492

/-- The function f(x) = x + 1/(x-3) for x > 3 -/
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 3)

/-- The minimum value of f(x) is 5 for x > 3 -/
theorem f_min_value :
  (∀ x > 3, f x ≥ 5) ∧ (∃ x > 3, f x = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l694_69492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_quadrant_iv_l694_69471

/-- The angle in radians -/
noncomputable def angle : ℝ := -29 / 12 * Real.pi

/-- Function to determine the quadrant of an angle in radians -/
noncomputable def quadrant (θ : ℝ) : Nat :=
  let θ_normalized := θ % (2 * Real.pi)
  if 0 ≤ θ_normalized && θ_normalized < Real.pi / 2 then 1
  else if Real.pi / 2 ≤ θ_normalized && θ_normalized < Real.pi then 2
  else if Real.pi ≤ θ_normalized && θ_normalized < 3 * Real.pi / 2 then 3
  else 4

theorem angle_in_quadrant_iv :
  quadrant angle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_quadrant_iv_l694_69471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l694_69475

open Real

theorem trigonometric_expressions :
  (Real.sin (-π/2) + 3 * Real.cos 0 - 2 * Real.tan (3*π/4) - 4 * Real.cos (5*π/3) = 2) ∧
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) → Real.tan θ = 4/3 → Real.sin θ - Real.cos θ = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l694_69475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_functions_on_real_line_l694_69463

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x

noncomputable def h (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem continuous_functions_on_real_line 
  (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  Continuous f ∧ Continuous (g a) ∧ Continuous h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_functions_on_real_line_l694_69463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_condition_l694_69491

/-- The condition b²-4ac<0 is neither sufficient nor necessary for ax²+bx+c>0 to hold for all real x. -/
theorem quadratic_inequality_condition :
  ∃ a b c : ℝ, (b^2 - 4*a*c < 0 ∧ ∃ x, a*x^2 + b*x + c ≤ 0) ∧
  ∃ a b c : ℝ, (∀ x, a*x^2 + b*x + c > 0) ∧ b^2 - 4*a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_condition_l694_69491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l694_69445

def our_sequence (n : ℕ) : ℚ :=
  (3^n - 1) / 2

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 3^n) : 
    ∀ n : ℕ, n ≥ 1 → a n = our_sequence n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l694_69445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_difference_l694_69410

theorem integral_bounds_difference (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_f0 : f 0 = 0) 
  (h_f1 : f 1 = 1) 
  (h_deriv_bound : ∀ x, |deriv f x| ≤ 2) :
  ∃ a b, a < b ∧ 
    (∀ y, (∃ g : ℝ → ℝ, (∀ x, g x = f x) ∧ (∫ x in (0:ℝ)..1, g x) = y) ↔ a < y ∧ y < b) ∧
    b - a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_difference_l694_69410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_digit_divisible_by_30_l694_69442

def digits : ℕ → List ℕ
  | n => if n < 10 then [n] else (n % 10) :: digits (n / 10)

theorem no_three_digit_divisible_by_30 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d, d ∈ digits n → d > 6) → ¬(n % 30 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_digit_divisible_by_30_l694_69442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radicals_l694_69477

-- Define the type for quadratic radicals
noncomputable def QuadraticRadical := ℝ → ℝ

-- Define the given quadratic radicals
noncomputable def radical1 : QuadraticRadical := λ x => Real.sqrt (x^2 + 1)
noncomputable def radical2 : QuadraticRadical := λ x => Real.sqrt (x^2 * (Real.sqrt x)^5)
noncomputable def radical3 : QuadraticRadical := λ _ => Real.sqrt 13
noncomputable def radical4 : QuadraticRadical := λ _ => 2 * Real.sqrt 3
noncomputable def radical5 : QuadraticRadical := λ _ => Real.sqrt (1/2)
noncomputable def radical6 : QuadraticRadical := λ _ => Real.sqrt 6

-- Define the property of being a simplest quadratic radical
def is_simplest (r : QuadraticRadical) : Prop := sorry

-- Theorem statement
theorem simplest_quadratic_radicals :
  {r : QuadraticRadical | is_simplest r} = {radical1, radical3, radical4, radical6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radicals_l694_69477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l694_69462

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then
    if -3 ≤ x then x^2 + 2*x - 1 else 0
  else
    if x ≤ 5 then x - 1 else 0

-- State the theorem about the range of f
theorem range_of_f :
  ∃ (y : ℝ), y ∈ Set.range f ↔ -2 ≤ y ∧ y ≤ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l694_69462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l694_69485

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (2*t - 1, -4*t - 2)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 / (1 - Real.cos θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Statement of the theorem
theorem min_distance_C₁_C₂ : 
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 ∧ 
  ∀ (t θ : ℝ), distance (C₁ t) (C₂ θ) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l694_69485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_circle_equation_with_chord_l694_69495

-- Define the circle C
def circle_C (r : ℝ) := {(x, y) : ℝ × ℝ | x^2 + (y - 4)^2 = r^2}

-- Define the point M
def point_M : ℝ × ℝ := (-2, 0)

-- Define the line l passing through M with slope k
def line_l (k : ℝ) := {(x, y) : ℝ × ℝ | y = k * (x + 2)}

-- Helper function to check if a line is tangent to a circle
def is_tangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∧ p ∈ c

-- Part I
theorem tangent_line_equation :
  ∃ (k : ℝ), is_tangent (line_l k) (circle_C 2) ∧
  (∀ (x y : ℝ), (x, y) ∈ line_l k ↔ (x = -2 ∨ 3*x - 4*y - 6 = 0)) :=
sorry

-- Part II
theorem circle_equation_with_chord :
  let l := {(x, y) : ℝ × ℝ | x + y + 2 = 0}
  ∃ (A B : ℝ × ℝ), A ∈ l ∧ B ∈ l ∧ A ∈ circle_C (2*Real.sqrt 5) ∧ B ∈ circle_C (2*Real.sqrt 5) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  ∀ (x y : ℝ), (x, y) ∈ circle_C (2*Real.sqrt 5) ↔ x^2 + (y - 4)^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_circle_equation_with_chord_l694_69495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equality_l694_69427

theorem sine_sum_equality (A B : Real) 
  (h1 : 0 < A) (h2 : A < Real.pi/2) 
  (h3 : 0 < B) (h4 : B < Real.pi/2)
  (h5 : Real.sin A ^ 2 + Real.sin B ^ 2 = (Real.sin (A + B) ^ 3) ^ (1/2)) : 
  A + B = Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equality_l694_69427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_26_years_l694_69469

/-- Tom's current age -/
def t : ℕ := sorry

/-- Jerry's current age -/
def j : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:1 -/
def x : ℕ := sorry

/-- Tom's age 4 years ago was 5 times Jerry's age 4 years ago -/
axiom past_condition_1 : t - 4 = 5 * (j - 4)

/-- Tom's age 10 years ago was 6 times Jerry's age 10 years ago -/
axiom past_condition_2 : t - 10 = 6 * (j - 10)

/-- The ratio of their ages will be 3:1 after x years -/
axiom future_ratio : (t + x) = 3 * (j + x)

/-- Theorem: It will take 26 years for the ratio of Tom's age to Jerry's age to become 3:1 -/
theorem age_ratio_in_26_years : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_26_years_l694_69469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_equals_three_l694_69433

/-- The probability mass function for a binomial distribution -/
noncomputable def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The random variable ξ follows a Binomial distribution B(6, 1/2) -/
noncomputable def ξ : ℕ → ℝ := binomial_pmf 6 (1/2)

/-- Theorem: The probability that ξ equals 3 is 5/16 -/
theorem prob_xi_equals_three : ξ 3 = 5/16 := by
  -- Expand the definition of ξ
  unfold ξ binomial_pmf
  -- Simplify the expression
  simp [Nat.choose]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_equals_three_l694_69433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l694_69434

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) (n : ℕ)
  (h1 : sum seq 9 = 18)
  (h2 : sum seq n = 240)
  (h3 : seq.a (n - 4) = 30) :
  n = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l694_69434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l694_69458

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 4) : Real.tan (3 * θ) = 52 / 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l694_69458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l694_69428

noncomputable section

variable (a : ℝ) (x : ℝ)

noncomputable def f (a x : ℝ) : ℝ := (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2

theorem f_negative (h : 0 < x ∧ x < a) : f a x < 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l694_69428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l694_69406

noncomputable section

/-- A line passing through a point (x₀, y₀) with slope m has the equation y - y₀ = m(x - x₀) -/
def line_equation (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

/-- Two non-vertical lines with slopes m₁ and m₂ are perpendicular if and only if m₁ * m₂ = -1 -/
def perpendicular_slopes (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

/-- The slope of a line ax + by = c, where b ≠ 0, is -a/b -/
noncomputable def slope_from_general_form (a b c : ℝ) : ℝ :=
  -a / b

theorem perpendicular_line_through_point 
  (x₀ y₀ : ℝ) -- The point (-1, 2) through which the line passes
  (a b c : ℝ) -- Coefficients of the given line 3x - 6y = 9
  (h₁ : b ≠ 0) -- Ensure the given line is not vertical
  (h₂ : line_equation x₀ y₀ (-2) x₀ y₀) -- The line passes through (-1, 2)
  (h₃ : perpendicular_slopes (slope_from_general_form a b c) (-2)) -- The slopes are perpendicular
  : ∀ x y, y = -2 * x ↔ line_equation x₀ y₀ (-2) x y :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l694_69406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l694_69404

def F : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | (n + 3) => F (n + 2) + F (n + 1)

theorem fibonacci_inequality (n : ℕ) (hn : n > 0) :
  (F (n + 1) : ℝ) ^ (1 / n : ℝ) > 1 + 1 / ((F n : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l694_69404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_coordinates_A_and_C_l694_69424

-- Define the coordinate system
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the given conditions
def B : Point := (-1, 0)

-- Theorem statement
theorem find_coordinates_A_and_C :
  ∃ (A C : Point),
    distance origin A = 4 * distance origin B ∧
    distance origin C = 4 * distance origin B ∧
    (A.2 = 0 ∨ A.1 = 0) ∧
    (C.2 = 0 ∨ C.1 = 0) ∧
    A = (4, 0) ∧
    C = (0, -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_coordinates_A_and_C_l694_69424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_a_b_c_l694_69490

-- Define the constants
noncomputable def a : ℝ := (2 : ℝ) ^ (5/2 : ℝ)
noncomputable def b : ℝ := Real.log 2.5 / Real.log 10
def c : ℝ := 1

-- State the theorem
theorem ordering_of_a_b_c : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_a_b_c_l694_69490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_proof_l694_69401

/-- Calculates the discount percentage given profit percentages with and without discount -/
noncomputable def calculate_discount_percentage (profit_with_discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  let selling_price_with_discount := 1 + profit_with_discount
  let selling_price_without_discount := 1 + profit_without_discount
  let discount := selling_price_without_discount - selling_price_with_discount
  (discount / selling_price_without_discount) * 100

/-- Theorem stating that given the specific profit percentages, the discount is approximately 46.304% -/
theorem discount_percentage_proof :
  let profit_with_discount := 0.235
  let profit_without_discount := 1.30
  let calculated_discount := calculate_discount_percentage profit_with_discount profit_without_discount
  abs (calculated_discount - 46.304) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_proof_l694_69401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l694_69479

/-- A quadratic function f(x) = ax^2 + bx + 4 with a < 0 -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 4

/-- The x-coordinate of point A -/
def x_A : ℝ := -2

/-- The x-coordinate of point B -/
def x_B : ℝ := 4

/-- The y-coordinate of points A and B -/
def y_AB : ℝ := 0

/-- The x-coordinate of point C (y-intercept) -/
def x_C : ℝ := 0

/-- The x-coordinate of the axis of symmetry -/
noncomputable def x_sym (a b : ℝ) : ℝ := -b / (2 * a)

/-- The point D is on the axis of symmetry -/
noncomputable def x_D (a b : ℝ) : ℝ := x_sym a b

/-- The y-coordinate of point D -/
noncomputable def y_D (a b : ℝ) : ℝ := f a b (x_D a b)

/-- The point M is on the axis of symmetry -/
noncomputable def x_M (a b : ℝ) : ℝ := x_sym a b

theorem quadratic_function_proof (a b : ℝ) (h_a : a < 0) :
  f a b x_A = y_AB ∧ 
  f a b x_B = y_AB ∧ 
  ∃ (y_M : ℝ), (x_M a b = 1 ∧ y_M = -1 ∨ x_M a b = 1 ∧ y_M = 7) ∧
    (y_D a b - f a b x_C = f a b x_C - y_M) →
  a = -1/2 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l694_69479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_areas_l694_69486

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/4, 0)

-- Define the condition for points A and B
def pointsOnParabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define the condition for A and B being on opposite sides of x-axis
def oppositeSides (A B : ℝ × ℝ) : Prop :=
  A.2 > 0 ∧ B.2 < 0

-- Define the dot product condition
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 2

-- Define the sum of areas of triangles ABO and AFO
noncomputable def sumOfAreas (A B : ℝ × ℝ) : ℝ :=
  1/2 * 2 * (A.2 - B.2) + 1/2 * 1/4 * A.2

-- Theorem statement
theorem min_sum_of_areas :
  ∀ A B : ℝ × ℝ,
  pointsOnParabola A B →
  oppositeSides A B →
  dotProductCondition A B →
  (∀ C D : ℝ × ℝ, pointsOnParabola C D → oppositeSides C D → dotProductCondition C D →
    sumOfAreas A B ≤ sumOfAreas C D) →
  sumOfAreas A B = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_areas_l694_69486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_domain_range_constraint_l694_69454

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem sine_domain_range_constraint (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, -1 ≤ f x ∧ f x ≤ 2) →
  b - a ≠ 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_domain_range_constraint_l694_69454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_range_l694_69408

theorem curve_line_intersection_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    Real.sqrt (1 - x₁^2) = k * (x₁ - 1) + 1 ∧
    Real.sqrt (1 - x₂^2) = k * (x₂ - 1) + 1) →
  0 < k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_range_l694_69408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_green_time_maximizes_probability_l694_69412

/-- Probability of passing through a traffic light without stopping -/
noncomputable def pass_probability (green_time highway_time : ℝ) : ℝ :=
  green_time / (green_time + highway_time)

/-- Total probability of passing both intersections without stopping -/
noncomputable def total_probability (x : ℝ) : ℝ :=
  (pass_probability x 30) * (pass_probability 120 x)

/-- The optimal green time for maximizing the probability of passing without stopping -/
def optimal_green_time : ℝ := 60

theorem optimal_green_time_maximizes_probability :
  ∀ x > 0, total_probability x ≤ total_probability optimal_green_time ∧
  total_probability optimal_green_time = 4/9 := by
  sorry

#eval optimal_green_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_green_time_maximizes_probability_l694_69412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_cosine_l694_69493

theorem largest_angle_cosine (A B C : ℝ) 
  (h : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (h_sin : ∃ (k : ℝ), k > 0 ∧ Real.sin A = 2 * k ∧ Real.sin B = 3 * k ∧ Real.sin C = 4 * k) :
  Real.cos (max A (max B C)) = -1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_cosine_l694_69493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l694_69443

noncomputable def f (x : ℝ) : ℝ := (3 : ℝ)^(x^2 - 3) - Real.sqrt (x^2)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l694_69443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l694_69466

-- Define the function f
noncomputable def f (x : ℝ) := Real.log (x^2 + 3*x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -2 ∨ x > -1}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.range Real.log} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l694_69466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_p_zero_l694_69429

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

structure MonicCubicPolynomial where
  d : ℤ
  e : ℤ
  f : ℤ
  p : ℤ → ℤ
  h_monic : ∀ x, p x = x^3 + d*x^2 + e*x + f
  h_integer_roots : ∃ r₁ r₂ r₃ : ℤ, ∀ x, p x = (x - r₁) * (x - r₂) * (x - r₃)

def has_six_integer_divisors (P : MonicCubicPolynomial) : Prop :=
  ∃ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ, ∀ a : ℤ, (P.p a % a = 0) ↔ a ∈ ({a₁, a₂, a₃, a₄, a₅, a₆} : Finset ℤ)

def exactly_one_is_perfect_square (P : MonicCubicPolynomial) : Prop :=
  (is_perfect_square ((P.p 1 + P.p (-1)) / 2) ∧ ¬is_perfect_square ((P.p 1 - P.p (-1)) / 2)) ∨
  (¬is_perfect_square ((P.p 1 + P.p (-1)) / 2) ∧ is_perfect_square ((P.p 1 - P.p (-1)) / 2))

theorem second_smallest_p_zero
  (P : MonicCubicPolynomial)
  (h_six_divisors : has_six_integer_divisors P)
  (h_one_perfect_square : exactly_one_is_perfect_square P)
  (h_p_zero_positive : P.p 0 > 0) :
  ∃ P₁ : MonicCubicPolynomial,
    has_six_integer_divisors P₁ ∧
    exactly_one_is_perfect_square P₁ ∧
    P₁.p 0 > 0 ∧
    P₁.p 0 < P.p 0 ∧
    ∀ P₂ : MonicCubicPolynomial,
      has_six_integer_divisors P₂ →
      exactly_one_is_perfect_square P₂ →
      P₂.p 0 > 0 →
      P₂.p 0 ≠ P₁.p 0 →
      P.p 0 ≤ P₂.p 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_p_zero_l694_69429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_trajectory_is_circle_l694_69431

theorem complex_trajectory_is_circle (z : ℂ) : 
  (Complex.abs (2 * z + 1) = Complex.abs (z - Complex.I)) ↔ 
  ∃ (center : ℂ) (radius : ℝ), 
    (z.re - center.re)^2 + (z.im - center.im)^2 = radius^2 :=
by
  sorry

#check complex_trajectory_is_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_trajectory_is_circle_l694_69431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_speed_calculation_l694_69400

/-- The speed of a bike given its distance traveled and time taken -/
noncomputable def bike_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: A bike traveling 350 meters in 7 seconds has a speed of 50 meters per second -/
theorem bike_speed_calculation : bike_speed 350 7 = 50 := by
  -- Unfold the definition of bike_speed
  unfold bike_speed
  -- Perform the division
  norm_num
  -- QED
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_speed_calculation_l694_69400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l694_69405

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x + 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l694_69405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_300_l694_69488

noncomputable section

/-- Revenue function for x ≤ 390 -/
def R1 (x : ℝ) : ℝ := -x^3 / 900 + 400 * x

/-- Revenue function for x > 390 -/
def R2 : ℝ := 90090

/-- Total revenue function -/
def R (x : ℝ) : ℝ :=
  if x ≤ 390 then R1 x else R2

/-- Fixed cost -/
def F : ℝ := 20000

/-- Variable cost per unit -/
def V : ℝ := 100

/-- Total cost function -/
def C (x : ℝ) : ℝ := F + V * x

/-- Profit function -/
def W (x : ℝ) : ℝ := R x - C x

/-- Theorem: The profit is maximized when x = 300 -/
theorem profit_maximized_at_300 :
  ∀ x : ℝ, x ≥ 0 → W 300 ≥ W x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_300_l694_69488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_even_g_l694_69413

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

noncomputable def g (n : ℝ) (x : ℝ) : ℝ := f (x + n)

theorem min_n_for_even_g :
  ∀ n : ℝ, n > 0 → (∀ x : ℝ, g n x = g n (-x)) → n ≥ 5 * Real.pi / 6 := by
  sorry

#check min_n_for_even_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_even_g_l694_69413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l694_69496

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given an ellipse with specific foci and one x-intercept, prove the other x-intercept -/
theorem ellipse_x_intercept (e : Ellipse) (h1 : e.focus1 = ⟨0, 3⟩) (h2 : e.focus2 = ⟨4, 0⟩) 
    (h3 : ∃ (p : Point), p.y = 0 ∧ distance p e.focus1 + distance p e.focus2 = distance ⟨0, 0⟩ e.focus1 + distance ⟨0, 0⟩ e.focus2) :
  ∃ (p : Point), p = ⟨56/11, 0⟩ ∧ distance p e.focus1 + distance p e.focus2 = distance ⟨0, 0⟩ e.focus1 + distance ⟨0, 0⟩ e.focus2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l694_69496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_segments_l694_69451

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}

-- Define the perpendicular diameters
def AB : Set (ℝ × ℝ) := {p | p.2 = 0 ∧ p ∈ Circle}
def CD : Set (ℝ × ℝ) := {p | p.1 = 0 ∧ p ∈ Circle}

-- Define the chord CH
def CH : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t, Real.sqrt (36 - t^2)) ∧ p ∈ Circle}

-- Define point K
def K : ℝ × ℝ := (8, 0)

-- State the theorem
theorem diameter_segments :
  ∃ (x y : ℝ), x + y = 20 ∧ x * y = 36 ∧ (x = 2 ∨ x = 18) ∧ (y = 2 ∨ y = 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_segments_l694_69451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l694_69483

-- Define the two parabolas
def f (x : ℝ) : ℝ := x^2 + 6*x + 5
def g (x : ℝ) : ℝ := x^2 - 4*x + 8

-- Define the vertices of the parabolas
def C : ℝ × ℝ := (-3, f (-3))
def D : ℝ × ℝ := (2, g 2)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem vertex_distance : distance C D = Real.sqrt 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l694_69483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_diagonal_planes_l694_69414

/-- A rectangular parallelepiped -/
structure RectangularParallelepiped where
  faces : Fin 6 → Face
  vertices : Fin 8 → Vertex

/-- A face of a rectangular parallelepiped -/
structure Face where
  diagonals : Fin 2 → Line

/-- A vertex of a rectangular parallelepiped -/
structure Vertex

/-- A line in 3D space -/
structure Line

/-- A plane in 3D space -/
structure Plane

/-- The number of planes determined by the diagonals on each face of a rectangular parallelepiped -/
def num_planes (rp : RectangularParallelepiped) : ℕ := 14

/-- Theorem stating that the number of planes determined by the diagonals on each face of a rectangular parallelepiped is 14 -/
theorem rectangular_parallelepiped_diagonal_planes (rp : RectangularParallelepiped) :
  num_planes rp = 14 := by
  -- Proof to be implemented
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_diagonal_planes_l694_69414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_degrees_theorem_l694_69497

/-- Represents an isosceles triangle PQR with a rolling circle --/
structure RollingCircleTriangle where
  s : ℝ  -- Length of PR and QR
  t : ℝ  -- Length of PQ
  h : ℝ  -- Altitude from P to QR
  r : ℝ  -- Radius of the rolling circle
  t_gt_s : t > s
  r_eq_half_h : r = h / 2
  h_eq_sqrt : h = Real.sqrt (s^2 - (t/2)^2)

/-- The number of degrees in arc MTN --/
noncomputable def arc_degrees (triangle : RollingCircleTriangle) : ℝ :=
  360 - 4 * Real.arcsin (triangle.t / (2 * triangle.s))

/-- Theorem stating the relationship between the triangle's properties and the arc degrees --/
theorem arc_degrees_theorem (triangle : RollingCircleTriangle) :
  arc_degrees triangle = 360 - 4 * Real.arcsin (triangle.t / (2 * triangle.s)) := by
  -- Unfold the definition of arc_degrees
  unfold arc_degrees
  -- The equality follows directly from the definition
  rfl

/-- Lemma: The arc_degrees is always positive --/
lemma arc_degrees_positive (triangle : RollingCircleTriangle) :
  arc_degrees triangle > 0 := by
  sorry

/-- Lemma: The arc_degrees is always less than 360 degrees --/
lemma arc_degrees_less_than_360 (triangle : RollingCircleTriangle) :
  arc_degrees triangle < 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_degrees_theorem_l694_69497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transport_time_l694_69465

/-- The minimum time required to transport trucks under given conditions -/
theorem min_transport_time (v : ℝ) (h : v > 0) : 
  (∀ t : ℝ, t ≥ 400 / v + 25 * (v / 20)^2 / v → t ≥ 10) :=
by
  intro t ht
  -- Define the transport time function
  let transport_time := 400 / v + 25 * (v / 20)^2 / v
  
  -- Apply AM-GM inequality
  have h_am_gm : transport_time ≥ 2 * Real.sqrt (400 / v * v / 16) := by
    -- Proof of AM-GM inequality goes here
    sorry
  
  -- Simplify the right-hand side of AM-GM
  have h_simplify : 2 * Real.sqrt (400 / v * v / 16) = 10 := by
    -- Algebraic simplification goes here
    sorry
  
  -- Combine the inequalities
  calc
    t ≥ transport_time := ht
    _ ≥ 2 * Real.sqrt (400 / v * v / 16) := h_am_gm
    _ = 10 := h_simplify

  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transport_time_l694_69465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_unique_root_l694_69460

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp (2 * x) else Real.exp (-2 * x)

theorem k_range_for_unique_root (k : ℝ) : 
  (∀ x : ℝ, f (-x) = f x) → 
  (∀ x : ℝ, x ≥ 0 → f x = Real.exp (2 * x)) → 
  (∃! x : ℝ, f x - |x - 1| - k * x = 0) → 
  k ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_unique_root_l694_69460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_base_circumference_equals_height_l694_69472

/-- A cylinder is a three-dimensional solid with circular bases and a curved lateral surface. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The circumference of a circle is given by 2πr where r is the radius. -/
noncomputable def circumference (c : Cylinder) : ℝ := 2 * Real.pi * c.radius

/-- The lateral surface area of a cylinder is given by 2πrh where r is the radius and h is the height. -/
noncomputable def lateralSurfaceArea (c : Cylinder) : ℝ := 2 * Real.pi * c.radius * c.height

/-- A property that holds when the lateral surface of a cylinder unfolds into a square. -/
def lateralSurfaceUnfoldsToSquare (c : Cylinder) : Prop :=
  ∃ (side : ℝ), lateralSurfaceArea c = side * side

theorem cylinder_base_circumference_equals_height (c : Cylinder) 
  (h : lateralSurfaceUnfoldsToSquare c) : circumference c = c.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_base_circumference_equals_height_l694_69472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_is_logarithm_l694_69452

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem stating that g is the inverse of f
theorem inverse_exponential_is_logarithm :
  ∀ x : ℝ, x > 0 → f (g x) = x ∧ g (f x) = x := by
  sorry

#check inverse_exponential_is_logarithm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_is_logarithm_l694_69452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_P_l694_69403

def P : ℕ := 54^6 + 6 * 54^5 + 15 * 54^4 + 20 * 54^3 + 15 * 54^2 + 6 * 54 + 1

theorem number_of_factors_of_P : 
  (Finset.filter (λ n : ℕ => P % n = 0) (Finset.range (P + 1))).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_P_l694_69403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sum_condition_l694_69494

noncomputable def f (x : ℝ) : ℝ := 1 / (x^3 + 3*x^2 + 2*x)

noncomputable def sum_f (n : ℕ) : ℝ := (1/2) * (1 - 1/(n+1 : ℝ) + 1/(n+2 : ℝ))

theorem smallest_n_for_sum_condition :
  ∃ n : ℕ, n ≥ 1 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k < n → sum_f k ≤ 503/2014) ∧
    sum_f n > 503/2014 ∧
    n = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sum_condition_l694_69494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l694_69416

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0
def line2 (x y : ℝ) : Prop := y = -x - 1

-- Define points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (P : ℝ × ℝ), line2 P.1 P.2 ∧
  (∀ (Q : ℝ × ℝ), line2 Q.1 Q.2 →
    distance P A + distance P B ≤ distance Q A + distance Q B) ∧
  distance P A + distance P B = Real.sqrt 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l694_69416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_characterization_l694_69484

/-- Define a function to calculate the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Define what it means for a number to be a friend of another number. -/
def is_friend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N > 0 ∧ (n ∣ N) ∧ (sum_of_digits N = u)

/-- The main theorem characterizing numbers with finitely many non-friends. -/
theorem friend_characterization (n : ℕ) (hn : n > 0) :
  (∃ S : Finset ℕ, ∀ u : ℕ, u > 0 → u ∉ S → is_friend u n) ↔ ¬(3 ∣ n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_characterization_l694_69484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_to_june_increase_is_50_percent_l694_69467

-- Define the profit changes
noncomputable def march_to_april_increase : ℝ := 0.35
noncomputable def april_to_may_decrease : ℝ := 0.20
noncomputable def march_to_june_increase : ℝ := 0.62000000000000014

-- Define the function to calculate the percent increase from May to June
noncomputable def may_to_june_increase : ℝ :=
  let april_profit := 1 + march_to_april_increase
  let may_profit := april_profit * (1 - april_to_may_decrease)
  let june_profit := 1 + march_to_june_increase
  (june_profit / may_profit) - 1

-- Theorem statement
theorem may_to_june_increase_is_50_percent :
  may_to_june_increase = 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_to_june_increase_is_50_percent_l694_69467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_nonnegative_f_l694_69498

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * x - Real.log x) * Real.log x - 2 * a * x + 2

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → (deriv (f a) 1 = 0 ∧ f a 1 = 0) := by sorry

-- Theorem for the range of a
theorem range_of_a_for_nonnegative_f :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ↔ a ∈ Set.Icc (Real.exp (-2)) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_nonnegative_f_l694_69498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_pricing_l694_69468

/-- Represents a cylindrical bottle -/
structure Bottle where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the price of a new bottle based on the original bottle and the new height -/
noncomputable def calculate_new_price (original : Bottle) (new_height : ℝ) : ℝ :=
  original.price * (new_height / original.height)

/-- Theorem statement for the bottle pricing problem -/
theorem bottle_pricing 
  (original : Bottle) 
  (new_height : ℝ) 
  (h_diameter : original.diameter > 0)
  (h_height : original.height > 0)
  (h_price : original.price > 0)
  (h_new_height : new_height > 0) :
  calculate_new_price original new_height = 
    original.price * (new_height / original.height) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_pricing_l694_69468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l694_69489

/-- Given a hyperbola C₁ with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity 2,
    prove that the eccentricity of hyperbola C₂ with equation x²/b² - y²/a² = 1 is 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c₁ := fun x y ↦ x^2 / a^2 - y^2 / b^2
  let e₁ := 2
  let c₂ := fun x y ↦ x^2 / b^2 - y^2 / a^2
  c₁ 0 0 = 1 → e₁ = 2 → ∃ e₂, e₂ = (2 * Real.sqrt 3) / 3 ∧ 
    (∀ x y, c₂ x y = 1 → e₂ = Real.sqrt ((x/b)^2 + (y/a)^2) / (x/b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l694_69489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l694_69446

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x

noncomputable def tangent_line (x : ℝ) : ℝ → ℝ := λ y ↦ 2 * (y - 1) + 4/3

noncomputable def x_intercept : ℝ := 1/3
noncomputable def y_intercept : ℝ := -2/3

theorem tangent_triangle_area :
  (1/2) * x_intercept * (-y_intercept) = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l694_69446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_common_difference_l694_69449

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y ∧ x ≠ 0 ∧ y ≠ 0

def arithmetic_common_difference (x y z : ℝ) : ℝ :=
  y - x

theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a b : ℝ),
  (is_arithmetic_sequence 1 a b) →
  (is_geometric_sequence 3 (a + 2) (b + 5)) →
  (arithmetic_common_difference 1 a b = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_common_difference_l694_69449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l694_69419

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ 
  a 2 = 1011 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n

theorem sequence_1000th_term (a : ℕ → ℕ) (h : my_sequence a) : a 1000 = 1676 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l694_69419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_restaurant_bill_l694_69448

/-- Calculates the total bill for a group at Tom's Restaurant -/
theorem toms_restaurant_bill (adults children : ℕ) (regular_price discount service_charge : ℚ) : 
  adults = 2 → 
  children = 5 → 
  regular_price = 8 → 
  discount = 0.2 → 
  service_charge = 0.1 → 
  (adults * regular_price + children * regular_price) * (1 + service_charge) - 
    (children * regular_price * discount) = 53.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_restaurant_bill_l694_69448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ten_eleven_l694_69470

theorem log_ten_eleven (r s : ℝ) (hr : (2 : ℝ) = 7^r) (hs : (11 : ℝ) = 2^s) : 
  (11 : ℝ) = 10^(r * s / (r + 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ten_eleven_l694_69470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_puzzle_solution_l694_69402

/-- Represents the direction of an arrow --/
inductive Direction
  | Up
  | Down
  | Left
  | Right
  | UpLeft
  | UpRight
  | DownLeft
  | DownRight

/-- Represents a cell in the grid --/
structure Cell where
  number : Nat
  arrow : Option Direction

/-- Represents the 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Cell

/-- Check if a number is valid (between 1 and 9) --/
def isValidNumber (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Check if all numbers in the grid are unique and valid --/
def validNumbers (g : Grid) : Prop :=
  ∀ i j, isValidNumber (g i j).number ∧
  ∀ i j i' j', g i j = g i' j' → (i = i' ∧ j = j') ∨ (g i j).number ≠ (g i' j').number

/-- Helper function to determine the direction between two cells --/
def sequenceDirection : (Fin 3 × Fin 3) → (Fin 3 × Fin 3) → Direction := by
  sorry

/-- Check if the arrows form a valid sequence from 1 to 9 --/
def validSequence (g : Grid) : Prop :=
  ∃ seq : Fin 9 → (Fin 3 × Fin 3),
    (∀ k, g (seq k).1 (seq k).2 = Cell.mk (k + 1) (some (sequenceDirection (seq k) (seq (k + 1))))) ∧
    seq 0 = ⟨2, 2⟩ ∧ seq 8 = ⟨0, 2⟩

/-- The theorem to prove --/
theorem grid_puzzle_solution (g : Grid) 
  (h1 : validNumbers g)
  (h2 : validSequence g)
  (h3 : g 2 2 = Cell.mk 1 (some Direction.UpLeft))
  (h4 : g 0 2 = Cell.mk 9 none)
  (h5 : g 0 1 = Cell.mk 8 (some Direction.Right))
  (h6 : g 1 0 = Cell.mk 5 (some Direction.Down))
  (h7 : g 1 1 = Cell.mk 4 (some Direction.Left))
  (h8 : g 2 1 = Cell.mk 2 (some Direction.UpLeft))
  (h9 : g 2 0 = Cell.mk 3 (some Direction.Left))
  : g 0 0 = Cell.mk 6 (some Direction.Up) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_puzzle_solution_l694_69402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_theorem_l694_69447

noncomputable def calculate_discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then 200 + 0.9 * (amount - 200)
  else 200 + 0.9 * 300 + 0.7 * (amount - 500)

theorem shopping_discount_theorem (trip1 trip2 : ℝ) 
  (h1 : trip1 = 168) (h2 : trip2 = 423) : 
  calculate_discount (trip1 + trip2) = 546.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_theorem_l694_69447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_water_shortage_l694_69482

/-- Calculates the water shortage for a football team given specific hydration requirements and spills. -/
theorem football_team_water_shortage : 
  let total_players : ℕ := 50
  let coach_water : ℚ := 6.5  -- in liters
  let players_150ml : ℕ := 20
  let players_200ml : ℕ := 15
  let players_250ml : ℕ := total_players - players_150ml - players_200ml
  let spill_150ml : ℕ := 7
  let spill_200ml : ℕ := 5
  let spill_250ml : ℕ := 3
  let spill_amount_150ml : ℕ := 25
  let spill_amount_200ml : ℕ := 40
  let spill_amount_250ml : ℕ := 30

  let total_water_needed : ℕ := 
    players_150ml * 150 + players_200ml * 200 + players_250ml * 250

  let total_water_spilled : ℕ := 
    spill_150ml * spill_amount_150ml + 
    spill_200ml * spill_amount_200ml + 
    spill_250ml * spill_amount_250ml

  let water_actually_used : ℕ := total_water_needed - total_water_spilled
  let coach_water_ml : ℤ := (coach_water * 1000).floor

  (coach_water_ml : ℚ) < water_actually_used ∧ 
  (water_actually_used : ℤ) - coach_water_ml = 2285 := by
  sorry

#check football_team_water_shortage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_water_shortage_l694_69482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_existence_l694_69411

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rotation function
noncomputable def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  sorry

-- Define the annulus
structure Annulus where
  center : ℝ × ℝ
  inner_radius : ℝ
  outer_radius : ℝ

-- Define the intersection of annulus and circle
def intersect (a : Annulus) (c : Circle) : Prop :=
  sorry

theorem equilateral_triangle_existence 
  (α β γ : Circle)
  (h : γ.radius ≤ min α.radius β.radius) :
  (∃ (p q r : ℝ × ℝ), 
    p ∈ Metric.sphere α.center α.radius ∧ 
    q ∈ Metric.sphere β.center β.radius ∧ 
    r ∈ Metric.sphere γ.center γ.radius ∧ 
    ‖p - q‖ = ‖q - r‖ ∧ ‖r - p‖ = ‖p - q‖) ↔ 
  intersect 
    { center := rotate γ.center α.center (π / 3),
      inner_radius := α.radius - γ.radius,
      outer_radius := α.radius + γ.radius }
    β :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_existence_l694_69411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_condition_equivalent_to_range_of_c_l694_69459

theorem max_condition_equivalent_to_range_of_c :
  ∀ c : ℝ,
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 
    max (|x + c / x|) (|x + c / x + 2|) ≥ 5) ↔ 
  c ∈ Set.Iic (-6) ∪ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_condition_equivalent_to_range_of_c_l694_69459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_tangent_circles_l694_69421

-- Define the radius of the two circles
def r₁ : ℝ := 2
def r₂ : ℝ := 3

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of being tangent
def isTangent (t : Triangle) (c : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ [t.A, t.B, t.C] ∧ 
    Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) = c.radius

-- Define the property of congruent sides
def hasCongruentSides (t : Triangle) : Prop :=
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 
  Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)

-- Define the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_area_with_tangent_circles 
  (t : Triangle) (c₁ c₂ : Circle) : 
  c₁.radius = r₁ →
  c₂.radius = r₂ →
  isTangent t c₁ →
  isTangent t c₂ →
  hasCongruentSides t →
  triangleArea t = 169 / Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_tangent_circles_l694_69421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l694_69407

noncomputable def f (x : ℝ) : ℝ := (15*x^5 + 10*x^4 + 5*x^3 + 7*x^2 + 6*x + 2) / (5*x^5 + 3*x^4 + 9*x^3 + 4*x^2 + 2*x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 3| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l694_69407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l694_69432

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log (x + 2) / Real.log 3 else Real.exp x - 1

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
    (h : m + n = f (f (Real.log 2))) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l694_69432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l694_69436

/-- A right prism with triangular bases -/
structure TriangularPrism where
  -- Base sides
  a : ℝ
  b : ℝ
  -- Height of the prism
  h : ℝ
  -- Angle between sides a and b
  θ : ℝ
  -- Conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_h : 0 < h
  angle_bounds : 0 < θ ∧ θ < π

/-- The sum of areas of three mutually adjacent faces is 30 -/
noncomputable def adjacent_faces_area_sum (p : TriangularPrism) : ℝ :=
  p.a * p.h + p.b * p.h + 1/2 * p.a * p.b * Real.sin p.θ

/-- The volume of the triangular prism -/
noncomputable def volume (p : TriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_triangular_prism :
  ∀ p : TriangularPrism, adjacent_faces_area_sum p = 30 →
  volume p ≤ 10 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l694_69436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_digits_fraction_l694_69418

-- Helper function to count non-zero digits after the decimal point
def number_of_nonzero_decimal_digits (x : ℚ) : ℕ :=
  sorry

theorem nonzero_digits_fraction (n : ℕ) (d : ℕ) (h : d = 2^4 * 5^9) :
  number_of_nonzero_decimal_digits (n / d : ℚ) = 3 :=
by
  sorry

#check nonzero_digits_fraction 120 (2^4 * 5^9) (by rfl)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_digits_fraction_l694_69418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XPQ_l694_69480

noncomputable section

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d X Y = 8 ∧ d Y Z = 13 ∧ d X Z = 15

-- Define points P and Q
noncomputable def P (X Y : ℝ × ℝ) : ℝ × ℝ :=
  let v := (Y.1 - X.1, Y.2 - X.2)
  let u := Real.sqrt (v.1^2 + v.2^2)
  (X.1 + 3 * v.1 / u, X.2 + 3 * v.2 / u)

noncomputable def Q (X Z : ℝ × ℝ) : ℝ × ℝ :=
  let v := (Z.1 - X.1, Z.2 - X.2)
  let u := Real.sqrt (v.1^2 + v.2^2)
  (X.1 + 10 * v.1 / u, X.2 + 10 * v.2 / u)

-- Define the area of a triangle given three points
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  |((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))| / 2

-- Theorem statement
theorem area_of_XPQ (X Y Z : ℝ × ℝ) :
  Triangle X Y Z →
  TriangleArea X (P X Y) (Q X Z) = 60 * Real.sqrt 3 / 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XPQ_l694_69480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_010101_subsequence_l694_69464

def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (mySequence n + mySequence (n+1) + mySequence (n+2) + 
              mySequence (n+3) + mySequence (n+4) + mySequence (n+5)) % 10

def subsequence_010101 (n : ℕ) : Prop :=
  mySequence n = 0 ∧ 
  mySequence (n+1) = 1 ∧ 
  mySequence (n+2) = 0 ∧ 
  mySequence (n+3) = 1 ∧ 
  mySequence (n+4) = 0 ∧ 
  mySequence (n+5) = 1

theorem no_010101_subsequence : ∀ n, ¬ subsequence_010101 n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_010101_subsequence_l694_69464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_areas_l694_69499

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a frustum -/
noncomputable def lateral_surface_area (f : Frustum) : ℝ :=
  Real.pi * (f.lower_radius + f.upper_radius) * Real.sqrt (f.height^2 + (f.lower_radius - f.upper_radius)^2)

/-- Calculates the area of the cross-sectional face of a frustum -/
def cross_sectional_area (f : Frustum) : ℝ :=
  f.height * (2 * f.upper_radius)

theorem frustum_areas (f : Frustum) 
    (h_lower : f.lower_radius = 8)
    (h_upper : f.upper_radius = 4)
    (h_height : f.height = 5) :
    lateral_surface_area f = 12 * Real.pi * Real.sqrt 41 ∧
    cross_sectional_area f = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_areas_l694_69499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l694_69420

-- Define the universal set
def U := ℝ

-- Define set P
def P : Set ℝ := {x : ℝ | Real.log (x^2) ≤ 1}

-- Define set Q
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) ∧ y = Real.sin x + Real.tan x}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = Set.Icc (-Real.sqrt (Real.exp 1)) ((Real.sqrt 2 + 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l694_69420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l694_69437

noncomputable def vector_product (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

noncomputable def m : ℝ × ℝ := (1, 1/2)
noncomputable def n : ℝ × ℝ := (0, 1)

noncomputable def f (x : ℝ) : ℝ := 1/2 * Real.sin (x/2) + 1

theorem f_properties :
  (∀ x, f x ≤ 3/2) ∧
  (∀ x, f (x + 4 * Real.pi) = f x) ∧
  (∀ T, 0 < T → T < 4 * Real.pi → ∃ x, f (x + T) ≠ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l694_69437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l694_69438

-- Define a regular hexagon
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Function to calculate the area of a circle
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

-- Define the problem setup
variable (hexagon : RegularHexagon)
variable (sideLength : ℝ)
axiom sideLength_eq : sideLength = 2

-- Define the two circles
variable (circle1 : Circle)
variable (circle2 : Circle)

-- Define a predicate for tangency
def TangentToSide (h : RegularHexagon) (i : Fin 6) (c : Circle) : Prop := sorry

-- Axioms for the tangent conditions
axiom circle1_tangent_AB : TangentToSide hexagon 0 circle1
axiom circle2_tangent_CD : TangentToSide hexagon 2 circle2
axiom circles_tangent_BC : TangentToSide hexagon 1 circle1 ∧ TangentToSide hexagon 1 circle2
axiom circles_tangent_FA : TangentToSide hexagon 5 circle1 ∧ TangentToSide hexagon 5 circle2

-- The theorem to be proved
theorem circle_area_ratio :
  circleArea circle2 / circleArea circle1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l694_69438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l694_69453

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem statement
theorem inverse_proportion_quadrants :
  (∀ x > 0, f x > 0) ∧ (∀ x < 0, f x < 0) := by
  constructor
  · intro x hx
    unfold f
    apply div_pos
    · exact six_pos
    · exact hx
  · intro x hx
    unfold f
    apply div_neg_of_pos_of_neg
    · exact six_pos
    · exact hx
  where
    six_pos : (6 : ℝ) > 0 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l694_69453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_tan_alpha_value_l694_69417

-- Part 1
theorem simplify_trig_expression (α : ℝ) :
  (Real.sin (π - α) * Real.cos (π + α) * Real.sin (π / 2 + α)) / (Real.sin (-α) * Real.sin (3 * π / 2 + α)) = -Real.cos α := by
  sorry

-- Part 2
theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (π - α) + Real.cos α = 7 / 13) :
  Real.tan α = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_tan_alpha_value_l694_69417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l694_69409

/-- Represents a cricket player's performance -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ

/-- Calculates the average runs per innings -/
def average (player : CricketPlayer) : ℚ :=
  player.totalRuns / player.innings

/-- Calculates the new average after playing an additional innings -/
def newAverage (player : CricketPlayer) : ℚ :=
  (player.totalRuns + player.nextInningsRuns) / (player.innings + 1)

/-- The main theorem about the increase in average -/
theorem average_increase (player : CricketPlayer) 
    (h1 : player.innings = 10)
    (h2 : average player = 32)
    (h3 : player.nextInningsRuns = 76) :
  newAverage player - average player = 4 := by
  sorry

/-- Example calculation -/
def examplePlayer : CricketPlayer := ⟨10, 320, 76⟩

#eval average examplePlayer
#eval newAverage examplePlayer
#eval newAverage examplePlayer - average examplePlayer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l694_69409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a3_S10_l694_69457

/-- An arithmetic sequence with positive terms satisfying a specific condition -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  positive : ∀ n : ℕ, a n > 0
  condition : a 1 + a 3 + a 8 = (a 4) ^ 2

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem stating the maximum value of a₃ · S₁₀ -/
theorem max_value_a3_S10 (seq : ArithmeticSequence) :
  ∃ (M : ℝ), M = 375 / 4 ∧ seq.a 3 * S seq 10 ≤ M ∧
  ∃ (seq' : ArithmeticSequence), seq'.a 3 * S seq' 10 = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a3_S10_l694_69457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l694_69440

/-- The focus of the parabola x² = 8y has coordinates (0, 2) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | x^2 = 8*y}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (0, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l694_69440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l694_69450

/-- A function f(x) = sin(ωx) - 1 with ω > 0 has exactly 3 zeros in [0, 2π] -/
def has_three_zeros (ω : ℝ) : Prop :=
  ω > 0 ∧ (∃ (z₁ z₂ z₃ : ℝ), 
    0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ < z₃ ∧ z₃ ≤ 2*Real.pi ∧
    (∀ x ∈ Set.Icc 0 (2*Real.pi), Real.sin (ω*x) - 1 = 0 ↔ x = z₁ ∨ x = z₂ ∨ x = z₃))

/-- The range of ω values for which f(x) = sin(ωx) - 1 has exactly 3 zeros in [0, 2π] -/
theorem omega_range (ω : ℝ) : has_three_zeros ω ↔ 9/4 ≤ ω ∧ ω < 13/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l694_69450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l694_69481

/-- A coloring function that assigns one of three colors (0, 1, or 2) to each positive integer. -/
def ColoringFunction := ℕ+ → Fin 3

/-- The number of divisors of a positive integer n with a given color. -/
def coloredDivisorsCount (f : ColoringFunction) (n : ℕ+) (color : Fin 3) : ℕ := sorry

/-- Theorem stating the existence of a coloring function satisfying the required property. -/
theorem exists_valid_coloring : ∃ (f : ColoringFunction), 
  ∀ (n : ℕ+) (i j : Fin 3), 
    (coloredDivisorsCount f n i : Int) - (coloredDivisorsCount f n j : Int) ≤ 2 ∧
    (coloredDivisorsCount f n i : Int) - (coloredDivisorsCount f n j : Int) ≥ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l694_69481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_nonpositive_l694_69441

/-- A function f is increasing if for all x and y, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Our function f(x) = e^x - ax - 1 -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ Real.exp x - a * x - 1

theorem increasing_f_implies_a_nonpositive :
  ∀ a : ℝ, IsIncreasing (f a) → a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_nonpositive_l694_69441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_regions_bound_l694_69426

/-- Represents a colored region in the plane --/
structure ColoredRegion where

/-- Represents a line in the plane --/
structure Line where

/-- Predicate to determine if two regions are adjacent --/
def AdjacentRegions (r1 r2 : ColoredRegion) : Prop := sorry

/-- Represents the plane divided by lines --/
structure DividedPlane where
  lines : Finset Line
  coloredRegions : Finset ColoredRegion
  n : ℕ
  n_ge_two : n ≥ 2
  line_count : lines.card = n
  no_adjacent_colored : ∀ r1 r2 : ColoredRegion, r1 ∈ coloredRegions → r2 ∈ coloredRegions → ¬AdjacentRegions r1 r2

theorem colored_regions_bound (plane : DividedPlane) :
  plane.coloredRegions.card ≤ (plane.n^2 + plane.n) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_regions_bound_l694_69426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l694_69487

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 16

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = 8*x

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_parabola :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      circle_eq x1 y1 → parabola_eq x2 y2 →
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_eq x1 y1 ∧ parabola_eq x2 y2 ∧
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l694_69487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_4_and_7_is_108_l694_69476

/-- Represents a clock face with equally spaced rays -/
structure ClockFace where
  /-- The number of equally spaced rays on the clock face -/
  num_rays : ℕ
  /-- Assertion that the number of rays is positive -/
  rays_positive : 0 < num_rays

/-- Calculates the angle between two rays on a clock face -/
noncomputable def angle_between_rays (clock : ClockFace) (ray1 : ℕ) (ray2 : ℕ) : ℝ :=
  (((ray2 - ray1 + clock.num_rays) % clock.num_rays) : ℝ) * (360 / clock.num_rays)

/-- Theorem stating that the angle between 4 o'clock and 7 o'clock on a 10-ray clock is 108° -/
theorem angle_between_4_and_7_is_108 :
  let clock : ClockFace := ⟨10, by norm_num⟩
  angle_between_rays clock 4 7 = 108 := by
  sorry

-- Remove the #eval statement as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_4_and_7_is_108_l694_69476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l694_69455

noncomputable def A (m : ℝ) : Set ℝ := {y | ∃ x, y = Real.sin x - Real.cos (x + Real.pi/6) + m}

def B : Set ℝ := {y | ∃ x ∈ Set.Icc 1 2, y = -x^2 + 2*x}

def p (x : ℝ) (m : ℝ) : Prop := x ∈ A m

def q (x : ℝ) : Prop := x ∈ B

theorem range_of_m :
  ∀ m : ℝ,
  (∀ x, q x → p x m) ∧
  (∃ x, p x m ∧ ¬q x) →
  m ∈ Set.Icc (1 - Real.sqrt 3) (Real.sqrt 3) :=
by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l694_69455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l694_69444

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- A line passing through a point at a given angle -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- Theorem about the specific ellipse and triangle -/
theorem ellipse_and_triangle_properties :
  ∀ (E : Ellipse),
    eccentricity E = 1/2 →
    (∃ (P : ℝ × ℝ), ellipse_equation E P.1 P.2 ∧
      ∃ (F : ℝ × ℝ), F.1 > 0 ∧ Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 1) →
    (ellipse_equation E = λ x y ↦ x^2/4 + y^2/3 = 1) ∧
    (∃ (A B : ℝ × ℝ),
      ellipse_equation E A.1 A.2 ∧
      ellipse_equation E B.1 B.2 ∧
      (∃ (L : Line), L.point = (0, 2) ∧ L.angle = π/3 ∧
        (A.2 - L.point.2 = (Real.tan L.angle) * (A.1 - L.point.1)) ∧
        (B.2 - L.point.2 = (Real.tan L.angle) * (B.1 - L.point.1))) ∧
      (1/2 * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 1 = 2 * Real.sqrt 177 / 15)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l694_69444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l694_69422

theorem contrapositive_sin_equality (x y : ℝ) : 
  (¬(Real.sin x = Real.sin y) → ¬(x = y)) ↔ (x = y → Real.sin x = Real.sin y) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l694_69422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l694_69439

/-- Represents the speed of the firetruck in different terrains -/
structure FiretruckSpeed where
  highway : ℝ
  field : ℝ

/-- Calculates the area reachable by a firetruck in a given time -/
noncomputable def reachableArea (speed : FiretruckSpeed) (time : ℝ) : ℝ :=
  4 * Real.pi * (speed.highway * time)^3 / (48 * speed.field)

/-- Theorem stating the area reachable by the firetruck in 5 minutes -/
theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 15
  let time := 5 / 60
  reachableArea speed time = 125 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l694_69439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_start_time_l694_69415

/-- Represents time in hours from midnight -/
def Time := Nat

/-- Converts a time from 24-hour format to 12-hour format with AM/PM -/
def to12HourFormat (t : Nat) : String :=
  let hour := t % 24
  if hour = 0 then "12 AM"
  else if hour < 12 then s!"{hour} AM"
  else if hour = 12 then "12 PM"
  else s!"{hour - 12} PM"

/-- The end time of the period (5 PM) -/
def endTime : Nat := 17

/-- Duration of rain in hours -/
def rainDuration : Nat := 4

/-- Duration without rain in hours -/
def noRainDuration : Nat := 5

/-- Theorem stating that the start time of the period is 8 AM -/
theorem period_start_time :
  let totalDuration := rainDuration + noRainDuration
  let startTime := endTime - totalDuration
  to12HourFormat startTime = "8 AM" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_start_time_l694_69415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_property_l694_69423

open Matrix

theorem scaling_matrix_property (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (N = ![![7, 0, 0], ![0, 7, 0], ![0, 0, 7]]) →
  (∀ w : Fin 3 → ℝ, N.mulVec w = (7 : ℝ) • w) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_property_l694_69423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_expression_simplification_l694_69473

theorem factorial_expression_simplification (n : ℕ) (h : n ≥ 7) :
  (Nat.factorial (n + 3) + Nat.factorial (n + 1)) / Nat.factorial (n + 2) =
  (n^2 + 5*n + 7) / (n + 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_expression_simplification_l694_69473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l694_69461

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
structure LineThroughFocus where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_focus : a * focus.1 + b * focus.2 + c = 0

-- Define the intersection of the line with the parabola
def intersects (l : LineThroughFocus) (p : PointOnParabola) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parabola_intersection_length 
  (l : LineThroughFocus) 
  (m n : PointOnParabola) 
  (intersects_m : intersects l m)
  (intersects_n : intersects l n)
  (midpoint_x : (m.x + n.x) / 2 = 3) :
  Real.sqrt ((m.x - n.x)^2 + (m.y - n.y)^2) = 8 := by
  sorry

-- Additional lemma to help with the proof
lemma parabola_focus_directrix_property (p : PointOnParabola) :
  p.x + 1 = Real.sqrt (p.x^2 + p.y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l694_69461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_G_largest_increase_l694_69435

structure City where
  name : String
  pop1990 : Nat
  pop2000 : Nat

def percentageIncrease (c : City) : Rat :=
  Rat.ofInt c.pop2000 / Rat.ofInt c.pop1990

def cities : List City := [
  ⟨"F", 50, 60⟩,
  ⟨"G", 60, 80⟩,
  ⟨"H", 90, 110⟩,
  ⟨"I", 120, 150⟩,
  ⟨"J", 150, 190⟩
]

theorem city_G_largest_increase :
  ∃ c ∈ cities, c.name = "G" ∧
    ∀ other ∈ cities, percentageIncrease c ≥ percentageIncrease other := by
  sorry

#eval cities.map (fun c => (c.name, percentageIncrease c))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_G_largest_increase_l694_69435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l694_69474

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l694_69474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_pirate_coins_l694_69425

/-- The number of pirates --/
def num_pirates : ℕ := 15

/-- The fraction of remaining coins that the kth pirate takes --/
def pirate_fraction (k : ℕ) : ℚ := (k + 1) / num_pirates

/-- The number of coins remaining after the kth pirate takes their share --/
def coins_remaining (initial_coins : ℕ) : ℕ → ℚ
  | 0 => initial_coins
  | k + 1 => (1 - pirate_fraction k) * coins_remaining initial_coins k

/-- The smallest number of initial coins that ensures each pirate gets a whole number of coins --/
def smallest_initial_coins : ℕ := 208080 * num_pirates^14 / Nat.factorial 14

/-- The theorem to be proved --/
theorem fifteenth_pirate_coins :
  ∀ k < num_pirates, (coins_remaining smallest_initial_coins k).num > 0 ∧
  (coins_remaining smallest_initial_coins 14).num = 208080 ∧
  (coins_remaining smallest_initial_coins 14).den = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_pirate_coins_l694_69425
