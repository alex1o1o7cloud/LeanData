import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l611_61187

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that a polynomial satisfies the given equation -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ z : ℝ, z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 →
    1 / P z + 1 / P (1/z) = z + 1/z

/-- The identity polynomial -/
def IdentityPolynomial : RealPolynomial := λ x ↦ x

/-- The theorem statement -/
theorem polynomial_equation_solution
  (P : RealPolynomial)
  (h_nonconstant : ∃ x y, P x ≠ P y)
  (h_satisfies : SatisfiesEquation P) :
  P = IdentityPolynomial := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l611_61187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l611_61136

-- Define the roots α and β
noncomputable def α : ℝ := Real.sqrt 2 -- placeholder value
noncomputable def β : ℝ := Real.pi / 2 -- placeholder value

-- State the conditions
axiom α_positive : α > 0
axiom β_positive : β > 0
axiom α_β_distinct : α ≠ β
axiom α_root : 2 * α = Real.tan α
axiom β_root : 2 * β = Real.tan β

-- Define the integral
noncomputable def integral : ℝ := ∫ x in Set.Icc 0 1, Real.sin (α * x) * Real.sin (β * x)

-- State the theorem
theorem integral_equals_zero : integral = 0 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l611_61136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_distance_l611_61186

/-- Predicate to check if a point is a vertex of a regular hexagon -/
def is_vertex_of_regular_hexagon (point : ℝ × ℝ) (side_length : ℝ) : Prop :=
  sorry

/-- Predicate to check if a point is on the perimeter of a regular hexagon -/
def is_on_perimeter_of_regular_hexagon (point : ℝ × ℝ) (side_length : ℝ) : Prop :=
  sorry

/-- Function to calculate the distance between two points along the perimeter of a regular hexagon -/
def perimeter_distance_between (start_point end_point : ℝ × ℝ) (side_length : ℝ) : ℝ :=
  sorry

/-- The distance between a vertex and a point on the perimeter of a regular hexagon -/
theorem hexagon_distance (side_length : ℝ) (perimeter_distance : ℝ) : 
  side_length = 2 →
  perimeter_distance = 5 →
  ∃ (start_point end_point : ℝ × ℝ),
    is_vertex_of_regular_hexagon start_point side_length ∧
    is_on_perimeter_of_regular_hexagon end_point side_length ∧
    perimeter_distance_between start_point end_point side_length = perimeter_distance ∧
    dist start_point end_point = Real.sqrt 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_distance_l611_61186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l611_61122

def sequence_a (n : ℕ+) : ℚ := sorry

def sequence_S (n : ℕ+) : ℚ := 2 * n - sequence_a n

theorem sequence_a_formula :
  ∀ n : ℕ+, sequence_a n = (2^n.val - 1) / 2^(n.val-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l611_61122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l611_61149

-- Define the function f(x) = m + log₂x
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.log x / Real.log 2

-- Define what it means for f to have a zero point
def has_zero_point (m : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 1 ∧ f m x = 0

-- State the theorem
theorem m_negative_sufficient_not_necessary :
  (∀ m : ℝ, m < 0 → has_zero_point m) ∧
  ¬(∀ m : ℝ, has_zero_point m → m < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l611_61149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l611_61117

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 3 ∧ y / x = b / a) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l611_61117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conclusion_l611_61111

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the intersection operation for planes
variable (intersection : Plane → Plane → Line)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the coplanar relation for points
variable (coplanar : Point → Point → Point → Point → Prop)

-- Define the projection of a line onto a plane
variable (projection : Line → Plane → Line)

-- Define a function to create a line from two points
variable (line_from_points : Point → Point → Line)

-- Theorem statement
theorem perpendicular_conclusion 
  (α β : Plane) (A B C D E F : Point) :
  let AB := line_from_points A B
  let CD := line_from_points C D
  let EF := line_from_points E F
  let BD := line_from_points B D
  let AC := line_from_points A C
  intersection α β = EF →
  perpendicular_line_plane AB α →
  perpendicular_line_plane CD α →
  (perpendicular_line_plane AC β ∨ 
   ∃ (l : Line), projection AC β = l ∧ projection CD β = l) →
  perpendicular_line_line BD EF :=
by
  -- Introduce the local definitions
  intro AB CD EF BD AC
  -- Introduce the hypotheses
  intro h_intersection h_AB_perp_α h_CD_perp_α h_additional
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conclusion_l611_61111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_min_distance_l611_61133

/-- A parabola passing through two points with a specific area condition -/
structure Parabola where
  a : ℝ
  b : ℝ
  y₀ : ℝ
  h_a_pos : a > 0
  h_point_A : y₀ = a * (-1)^2 + b * (-1)
  h_point_B : 0 = a * 2^2 + b * 2
  h_area : 1 = (1/2) * |2 * y₀|

/-- The line to which we're finding the minimum distance -/
noncomputable def line_l (x : ℝ) : ℝ := x - 25/4

/-- The point on the parabola with minimum distance to line_l -/
noncomputable def min_distance_point : ℝ × ℝ := (5/2, 5/12)

/-- Main theorem stating the properties of the parabola and the minimum distance point -/
theorem parabola_and_min_distance (p : Parabola) :
  p.a = 1/3 ∧ p.b = -2/3 ∧ min_distance_point.1 = 5/2 ∧ min_distance_point.2 = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_min_distance_l611_61133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_C_right_triangle_area_l611_61152

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = 2 * Real.sin t.A * Real.sin t.C

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b

def hasRightAngleAtB (t : Triangle) : Prop :=
  t.B = Real.pi / 2

-- Theorem 1
theorem isosceles_triangle_cos_C (t : Triangle) 
  (h1 : satisfiesCondition t) (h2 : isIsosceles t) : 
  Real.cos t.C = 7/8 := by sorry

-- Theorem 2
theorem right_triangle_area (t : Triangle) 
  (h1 : satisfiesCondition t) (h2 : hasRightAngleAtB t) (h3 : t.c = Real.sqrt 2) :
  (1/2) * t.a * t.c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_C_right_triangle_area_l611_61152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l611_61179

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = -3) : Real.tan (3 * θ) = -9/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l611_61179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_arithmetic_mean_of_special_pairs_l611_61116

-- Define the set of pairs of two-digit natural numbers
def two_digit_pairs : Set (ℕ × ℕ) :=
  {(a, b) | 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99}

-- Define the condition for the special relationship between arithmetic and geometric means
noncomputable def special_mean_condition (a b : ℕ) : Prop :=
  (a + b : ℝ) / 2 = (25 / 24) * Real.sqrt (a * b)

-- Define the set of pairs satisfying the special mean condition
def special_pairs : Set (ℕ × ℕ) :=
  {p ∈ two_digit_pairs | special_mean_condition p.1 p.2}

-- Define the arithmetic mean function
noncomputable def arithmetic_mean (a b : ℕ) : ℝ := (a + b : ℝ) / 2

-- State the theorem
theorem largest_arithmetic_mean_of_special_pairs :
  ∃ (max_mean : ℝ), 
    (∀ (a b : ℕ), (a, b) ∈ special_pairs → arithmetic_mean a b ≤ max_mean) ∧
    (∃ (a b : ℕ), (a, b) ∈ special_pairs ∧ arithmetic_mean a b = max_mean) ∧
    max_mean = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_arithmetic_mean_of_special_pairs_l611_61116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_approx_30_percent_l611_61126

/-- Calculates the overall percentage change of an investment over three years -/
noncomputable def investment_change (initial : ℝ) (loss1 gain2 gain3 : ℝ) : ℝ :=
  let year1 := initial * (1 - loss1)
  let year2 := year1 * (1 + gain2)
  let year3 := year2 * (1 + gain3)
  (year3 - initial) / initial * 100

/-- Theorem stating that the investment change is approximately 30% -/
theorem investment_change_approx_30_percent :
  ∃ ε > 0, |investment_change 200 0.1 0.15 0.25 - 30| < ε :=
by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_approx_30_percent_l611_61126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_haircut_spending_l611_61120

/-- Represents the hair growth and haircut information for John --/
structure HaircutInfo where
  growth_rate : ℝ  -- Hair growth rate in inches per month
  max_length : ℝ   -- Maximum hair length before cutting
  min_length : ℝ   -- Hair length after cutting
  cost : ℝ         -- Cost of a haircut in dollars
  tip_rate : ℝ     -- Tip rate as a decimal

/-- Calculates the annual spending on haircuts --/
noncomputable def annual_haircut_spending (info : HaircutInfo) : ℝ :=
  let months_between_cuts := (info.max_length - info.min_length) / info.growth_rate
  let cuts_per_year := 12 / months_between_cuts
  let cost_per_cut := info.cost * (1 + info.tip_rate)
  cuts_per_year * cost_per_cut

/-- Theorem stating that John spends $324 on haircuts annually --/
theorem john_haircut_spending :
  let john_info : HaircutInfo := {
    growth_rate := 1.5,
    max_length := 9,
    min_length := 6,
    cost := 45,
    tip_rate := 0.2
  }
  annual_haircut_spending john_info = 324 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_haircut_spending_l611_61120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l611_61107

-- Define the function f(x) = 2^x + x - 4
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + x - 4

-- State the theorem
theorem unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l611_61107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_2_l611_61135

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle represented by polar equation -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Given a parametric line and a polar circle, 
    calculate the distance from the circle's center to the line -/
def distanceFromCircleCenterToLine (l : ParametricLine) (c : PolarCircle) : ℝ :=
  sorry

/-- The specific line and circle from the problem -/
def problemLine : ParametricLine :=
  { x := λ t => t
  , y := λ t => t + 1 }

noncomputable def problemCircle : PolarCircle :=
  { ρ := λ θ => -6 * Real.cos θ }

theorem distance_is_sqrt_2 : 
  distanceFromCircleCenterToLine problemLine problemCircle = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_2_l611_61135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l611_61138

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + (y - Real.sqrt 3)^2 = 3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  A.2 = line_l A.1 ∧ B.2 = line_l B.1 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance_product (A B : ℝ × ℝ) :
  intersection_points A B →
  (A.1^2 + A.2^2) * (B.1^2 + B.2^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l611_61138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l611_61141

/-- Represents a monomial with real coefficient and integer exponents -/
structure Monomial where
  coeff : ℝ
  exponents : List ℕ

/-- Calculate the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.exponents.sum

/-- The monomial -3π * x * y^2 * z^3 -/
noncomputable def example_monomial : Monomial :=
  { coeff := -3 * Real.pi
  , exponents := [1, 2, 3] }

theorem example_monomial_properties :
  example_monomial.coeff = -3 * Real.pi ∧
  degree example_monomial = 6 := by
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l611_61141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_for_A_l611_61160

/-- Represents the probability of A winning when A shoots first at position x -/
noncomputable def A_shoots_first (x : ℝ) : ℝ := x^2

/-- Represents the probability of A winning when B shoots first at position x -/
noncomputable def B_shoots_first (x : ℝ) : ℝ := 1 - x

/-- The optimal shooting position for contestant A -/
noncomputable def optimal_position : ℝ := (Real.sqrt 5 - 1) / 2

theorem optimal_strategy_for_A :
  ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    (x < optimal_position → B_shoots_first x > A_shoots_first x) ∧
    (x > optimal_position → A_shoots_first x > B_shoots_first x) ∧
    (x = optimal_position → A_shoots_first x = B_shoots_first x) :=
by sorry

#check optimal_strategy_for_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_for_A_l611_61160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_times_value_range_for_f_l611_61112

open Real

-- Define the function f(x) = ln x + x
noncomputable def f (x : ℝ) : ℝ := log x + x

-- Define the property of being a k-times value function
def is_k_times_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ k > 0 ∧
  ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (k * a) (k * b)

-- Theorem statement
theorem k_times_value_range_for_f :
  {k : ℝ | is_k_times_value_function f k} = Set.Ioo 1 (1 + 1 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_times_value_range_for_f_l611_61112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_120_l611_61174

/-- An arithmetic progression with a given property -/
structure ArithmeticProgression where
  /-- First term of the progression -/
  a : ℝ
  /-- Common difference of the progression -/
  d : ℝ
  /-- Sum of 4th and 12th terms is 16 -/
  sum_property : a + 3*d + a + 11*d = 16

/-- Sum of first n terms of an arithmetic progression -/
noncomputable def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- The sum of the first 15 terms is 120 -/
theorem sum_15_terms_is_120 (ap : ArithmeticProgression) :
  sum_n_terms ap 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_120_l611_61174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l611_61132

theorem min_lambda_value (x y l : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → a + 2 * Real.sqrt (2 * a * b) ≤ l * (a + b)) : 
  l ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l611_61132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_x_axis_l611_61106

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem statement -/
theorem parabola_point_distance_to_x_axis
  (P : ParabolaPoint)
  (h : distance P.x P.y 0 1 = 8) :
  |P.y| = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_x_axis_l611_61106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_minimum_value_l611_61165

theorem circle_center_minimum_value (m n : ℝ) (h1 : m * n > 0) : 
  (m * (-2) + 2 * n * (-1) + 1 = 0) → (1 / m + 1 / n ≥ 8) := by
  intro h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_minimum_value_l611_61165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l611_61178

/-- Calculates the final amount after compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (compoundingsPerYear : ℕ) (years : ℝ) : ℝ :=
  principal * (1 + rate / (compoundingsPerYear : ℝ)) ^ ((compoundingsPerYear : ℝ) * years)

/-- Theorem stating that an $800 investment at 10% interest compounded semiannually yields $882 after one year -/
theorem investment_growth : 
  let principal := (800 : ℝ)
  let rate := (0.10 : ℝ)
  let compoundingsPerYear := 2
  let years := (1 : ℝ)
  compoundInterest principal rate compoundingsPerYear years = 882 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l611_61178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_range_of_f_l611_61151

noncomputable def ω : ℝ := 5/6

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x) - Real.sin (ω * x), Real.sin (ω * x))

noncomputable def b (x : ℝ) : ℝ × ℝ := (-Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sqrt 3 * Real.cos (ω * x))

noncomputable def f (x : ℝ) : ℝ := 
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - Real.sqrt 2

theorem period_and_range_of_f :
  (∀ x, f (x + 6 * π / 5) = f x) ∧ 
  (∀ x ∈ Set.Icc 0 (3 * π / 5), -1 - Real.sqrt 2 ≤ f x ∧ f x ≤ 2 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_range_of_f_l611_61151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_g_value_l611_61167

-- Define the divisor function d(n)
def d (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define the function g(n)
noncomputable def g (n : ℕ) : ℝ := (d n : ℝ) / n^(1/4 : ℝ)

-- State the theorem
theorem max_g_value (n : ℕ) : n ≠ 1440 → g 1440 > g n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_g_value_l611_61167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_roots_l611_61155

theorem quadratic_equations_roots (a b c : ℝ) (α β γ δ : ℝ) :
  (5 * α^2 - a * α + b = 0) →
  (5 * β^2 - a * β + b = 0) →
  (γ^2 - b * γ + c = 0) →
  (δ^2 - b * δ + c = 0) →
  (α ≠ β) → (α ≠ γ) → (α ≠ δ) → (β ≠ γ) → (β ≠ δ) → (γ ≠ δ) →
  let M : Set ℝ := {α, β, γ, δ}
  let S : Set ℝ := {x | ∃ u v, u ∈ M ∧ v ∈ M ∧ u ≠ v ∧ x = u + v}
  let P : Set ℝ := {x | ∃ u v, u ∈ M ∧ v ∈ M ∧ u ≠ v ∧ x = u * v}
  S = {5, 7, 8, 9, 10, 12} →
  P = {6, 10, 14, 15, 21, 35} →
  (a = 35 ∧ b = 50 ∧ c = 21) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_roots_l611_61155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_reasonable_l611_61109

/-- Represents a sampling method -/
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic
| RandomNumber

/-- Represents a population with distinct subgroups -/
structure Population (α : Type u) where
  subgroups : List (Set α)
  nonempty_subgroups : ∀ s ∈ subgroups, Set.Nonempty s

/-- Represents a survey requirement -/
structure SurveyRequirement (α : Type u) where
  population : Population α
  need_subgroup_representation : Bool

/-- Determines if a sampling method is reasonable for a given survey requirement -/
def is_reasonable_method {α : Type u} (method : SamplingMethod) (req : SurveyRequirement α) : Prop :=
  match method with
  | SamplingMethod.Stratified => 
      req.need_subgroup_representation ∧ 
      req.population.subgroups.length > 1
  | _ => False

/-- Theorem: Stratified sampling is the most reasonable method for the given survey -/
theorem stratified_sampling_most_reasonable {α : Type u} (req : SurveyRequirement α) 
    (h1 : req.need_subgroup_representation = true)
    (h2 : req.population.subgroups.length = 3) :
    ∀ (method : SamplingMethod), 
      is_reasonable_method method req → method = SamplingMethod.Stratified := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_reasonable_l611_61109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approximately_852_l611_61123

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ (time : ℝ)

def angela_deposit : ℝ := 9000
def bob_deposit : ℝ := 10000
def angela_rate : ℝ := 0.05
def bob_rate : ℝ := 0.045
def time_period : ℕ := 25

theorem balance_difference_approximately_852 :
  let angela_balance := compound_interest angela_deposit angela_rate time_period
  let bob_balance := compound_interest bob_deposit bob_rate time_period
  ∃ ε > 0, |angela_balance - bob_balance - 852| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approximately_852_l611_61123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_value_l611_61177

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

-- Define the sum of the series
noncomputable def fibonacciSum : ℝ := ∑' n, (fibonacci n : ℝ) / 7^n

-- State the theorem
theorem fibonacci_sum_value : fibonacciSum = 49 / 287 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_value_l611_61177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l611_61148

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^2 + 2*m*x + 1 = 0 ∧ y^2 + 2*m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

-- Define the set of m values that satisfy the conditions
def M : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : M = Set.Iic (-2) ∪ Set.Ico (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l611_61148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l611_61144

noncomputable def f (A w φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (w * x + φ)

theorem function_and_range 
  (A w φ : ℝ) 
  (h1 : A > 0) 
  (h2 : w > 0) 
  (h3 : 0 < φ ∧ φ < Real.pi / 2) 
  (h4 : ∀ x, f A w φ (x + Real.pi / w) = f A w φ x) 
  (h5 : f A w φ (2 * Real.pi / 3) = -2) :
  (∃ g : ℝ → ℝ, g = λ x ↦ 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∀ y, y ∈ Set.Icc 0 (Real.pi / 2) →
    (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f A w φ x = y) ↔ y ∈ Set.Icc (-1) 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l611_61144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l611_61125

/-- Represents the average speed of a car's journey with multiple segments and stops. -/
noncomputable def average_speed (d1 d2 d3 : ℝ) (v1 v2 v3 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2 + d3) / (d1/v1 + d2/v2 + d3/v3 + t1 + t2)

/-- Theorem stating that the average speed of the car's journey is approximately 86.37 km/h. -/
theorem car_journey_average_speed :
  let d1 := (120 : ℝ) -- km
  let d2 := (180 : ℝ) -- km
  let d3 := (75 : ℝ)  -- km
  let v1 := (80 : ℝ)  -- km/h
  let v2 := (100 : ℝ) -- km/h
  let v3 := (120 : ℝ) -- km/h
  let t1 := (15/60 : ℝ) -- hours (15 minutes)
  let t2 := (10/60 : ℝ) -- hours (10 minutes)
  ∃ ε > 0, |average_speed d1 d2 d3 v1 v2 v3 t1 t2 - 86.37| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l611_61125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l611_61100

-- Define the curves
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - a * x
def g (x : ℝ) : ℝ := -x^3 + x^2

-- Define the theorem
theorem a_range_theorem (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ x₂ < 0 ∧
   -- A and B exist on f and g
   ∃ y₁ y₂ : ℝ, y₁ = f a x₁ ∧ y₂ = g x₂ ∧
   -- Triangle AOB is right-angled at O
   x₁ * x₂ + y₁ * y₂ = 0 ∧
   -- AC = (1/2)CB
   x₂ = -2 * x₁) →
  a ∈ Set.Ioo (1 / (10 * (Real.exp 2 - 2))) (1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l611_61100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_infinity_to_zero_a_range_for_bounded_f_l611_61114

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/3)^x + (1/9)^x

-- Part 1: f(x) is not bounded on (-∞, 0) when a = -1/2
theorem f_not_bounded_neg_infinity_to_zero :
  ¬ ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), x < 0 → |f (-1/2) x| ≤ M :=
by sorry

-- Part 2: Range of a for f(x) to be bounded with upper bound 4 on [0, +∞)
theorem a_range_for_bounded_f (a : ℝ) :
  (∀ (x : ℝ), x ≥ 0 → |f a x| ≤ 4) ↔ a ∈ Set.Icc (-6) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_infinity_to_zero_a_range_for_bounded_f_l611_61114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l611_61119

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sin (3 * x)) / (Real.cos x + Real.cos (3 * x))

theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l611_61119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_clothing_profit_increase_l611_61169

theorem winter_clothing_profit_increase :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    let sept_price := a
    let sept_cost := 0.75 * a
    let sept_profit := sept_price - sept_cost
    let oct_price := 0.9 * a
    let oct_cost := sept_cost
    let oct_profit := oct_price - oct_cost
    let sept_sales := b
    let oct_sales := 1.8 * b
    let total_sept_profit := sept_profit * sept_sales
    let total_oct_profit := oct_profit * oct_sales
    total_oct_profit = 1.62 * total_sept_profit :=
by
  intros a b ha hb
  simp [ha, hb]
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_clothing_profit_increase_l611_61169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emmett_pushups_l611_61184

theorem emmett_pushups (jumping_jacks situps : ℕ) (pushup_percentage : ℚ) :
  jumping_jacks = 12 →
  situps = 20 →
  pushup_percentage = 1/5 →
  ∃ pushups : ℕ, 
    pushups = 8 ∧
    pushups = (pushup_percentage * (jumping_jacks + situps + pushups : ℚ)).floor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emmett_pushups_l611_61184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_configuration_l611_61192

def first_10_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def fifteenth_prime : Nat := 47

def second_prime : Nat := 3

def fifth_prime : Nat := 11

structure Configuration where
  numbers : List Nat
  single_line_sums : List (Nat × Nat × Nat)
  double_line_sums : List (Nat × Nat × Nat)

def is_valid_configuration (c : Configuration) : Prop :=
  c.numbers.length = 11 ∧
  c.numbers.toFinset = ({1} ∪ first_10_primes.toFinset) ∧
  (∀ t ∈ c.single_line_sums, t.1 + t.2.1 + t.2.2 = fifteenth_prime) ∧
  (∀ t ∈ c.double_line_sums, t.1 + t.2.1 + t.2.2 = second_prime * fifth_prime)

def is_mirror_symmetric (c1 c2 : Configuration) : Prop :=
  -- Define mirror symmetry condition here
  sorry

theorem unique_configuration :
  ∃! c : Configuration, is_valid_configuration c ∧
    ∀ c' : Configuration, is_valid_configuration c' →
      is_mirror_symmetric c c' → c = c' := by
  sorry

#check unique_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_configuration_l611_61192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_sequences_l611_61101

theorem distinct_floor_sequences (a : ℝ) (N M : ℕ) (hN : N > 0) (hM : M > 0) :
  (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i * a⌋ ≠ ⌊j * a⌋) ∧
  (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ M → ⌊i / a⌋ ≠ ⌊j / a⌋) ↔
  (N - 1 : ℝ) / N ≤ |a| ∧ |a| ≤ M / (M - 1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_sequences_l611_61101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l611_61104

theorem train_speed_problem (train_length : ℝ) (passing_time : ℝ) (speed_difference : ℝ) :
  train_length = 50 →
  passing_time = 36 →
  speed_difference = 36 →
  let relative_speed := (2 * train_length) / (passing_time / 3600)
  let faster_train_speed := relative_speed + speed_difference
  faster_train_speed = 46 := by
  intro h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l611_61104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_270_feet_l611_61168

/-- Represents the scenario with Jack, Christina, and Lindy --/
structure Scenario where
  initialDistance : ℝ
  jackSpeed : ℝ
  christinaSpeed : ℝ
  lindySpeed : ℝ

/-- Calculates the time taken for Jack and Christina to meet --/
noncomputable def meetingTime (s : Scenario) : ℝ :=
  s.initialDistance / (s.jackSpeed + s.christinaSpeed)

/-- Calculates the total distance traveled by Lindy --/
noncomputable def lindyDistance (s : Scenario) : ℝ :=
  s.lindySpeed * meetingTime s

/-- Theorem stating that Lindy travels 270 feet --/
theorem lindy_travels_270_feet (s : Scenario) 
    (h1 : s.initialDistance = 240)
    (h2 : s.jackSpeed = 5)
    (h3 : s.christinaSpeed = 3)
    (h4 : s.lindySpeed = 9) : 
  lindyDistance s = 270 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_270_feet_l611_61168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_alone_time_l611_61156

/-- The number of days it takes for Thomas to complete the task alone -/
def T : ℝ := sorry

/-- The number of days it takes for Edward to complete the task alone -/
def E : ℝ := sorry

/-- Thomas and Edward together can finish the task in 8 days -/
axiom together_rate : 1 / T + 1 / E = 1 / 8

/-- Thomas working alone for 13 days followed by Edward working alone for 6 days can finish the task -/
axiom sequential_work : 13 / T + 6 / E = 1

/-- Theorem: It takes Thomas 14 days to complete the task alone -/
theorem thomas_alone_time : T = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_alone_time_l611_61156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l611_61110

-- Define the original expression
noncomputable def original_expression (x : ℝ) : ℝ :=
  (x - 1) / ((x^2 - 2*x + 1) * ((x^2 + x - 1) / (x - 1) - x - 1)) - 1 / (x - 2)

-- Define the simplified expression
noncomputable def simplified_expression (x : ℝ) : ℝ :=
  -2 / (x * (x - 2))

-- State the theorem
theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 0 → x ≠ 1 → x ≠ 2 →
  (original_expression x = simplified_expression x) ∧
  (simplified_expression (-1) = -2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l611_61110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_when_fourth_powers_sum_to_one_l611_61195

theorem sin_cos_sum_when_fourth_powers_sum_to_one (α : ℝ) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 1 → Real.sin α + Real.cos α = 1 ∨ Real.sin α + Real.cos α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_when_fourth_powers_sum_to_one_l611_61195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_theorem_l611_61193

/-- The ratio of the area of a regular octagon circumscribed about a circle 
    to the area of a regular octagon inscribed in the same circle -/
noncomputable def octagon_area_ratio : ℝ := 12 + 8 * Real.sqrt 2

/-- Theorem stating that the ratio of the areas of circumscribed and inscribed regular octagons 
    is equal to 12 + 8√2 -/
theorem octagon_area_ratio_theorem (r : ℝ) (r_pos : r > 0) : 
  (8 * (1 + Real.sqrt 2)^3 * r^2) / (2 * (1 + Real.sqrt 2) * r^2) = octagon_area_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_theorem_l611_61193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_ellipses_l611_61171

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem intersection_of_ellipses (P : Point) : 
  let A : Point := ⟨-4, 0⟩
  let B : Point := ⟨-3, 2⟩
  let C : Point := ⟨3, 2⟩
  let D : Point := ⟨4, 0⟩
  (distance P A + distance P D = 10) →
  (distance P B + distance P C = 8) →
  P.y = (18 - 12 * Real.sqrt 2) / 2 := by
  sorry

#check intersection_of_ellipses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_ellipses_l611_61171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_sqrt2_inequality_no_exact_n_for_optimal_c_l611_61145

open Real

-- Define the fractional part function as noncomputable
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem largest_c_for_sqrt2_inequality :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (n : ℕ), n > 0 → frac (n * Real.sqrt 2) ≥ c / n) ∧ 
    (∀ (c' : ℝ), c' > c → 
      ∃ (n : ℕ), n > 0 ∧ frac (n * Real.sqrt 2) < c' / n)) ∧
  (∀ (c : ℝ), c > 0 ∧ 
    (∀ (n : ℕ), n > 0 → frac (n * Real.sqrt 2) ≥ c / n) →
    c ≤ 1 / (2 * Real.sqrt 2)) :=
by sorry

-- State that there's no n for which the equality holds
theorem no_exact_n_for_optimal_c :
  ¬∃ (n : ℕ), n > 0 ∧ frac (n * Real.sqrt 2) = 1 / (2 * Real.sqrt 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_sqrt2_inequality_no_exact_n_for_optimal_c_l611_61145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_area_is_three_halves_of_rectangle_l611_61197

/-- Represents a rectangular piece of metal with length twice its width -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq_twice_width : length = 2 * width

/-- Represents a circular piece cut from the rectangle -/
structure Circle where
  radius : ℝ

/-- Represents a square piece cut from the circle -/
structure Square where
  side : ℝ

/-- Calculates the area of the rectangle -/
def rectangle_area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the area of the circle -/
noncomputable def circle_area (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Calculates the area of the square -/
def square_area (s : Square) : ℝ := s.side^2

/-- Theorem stating that the wasted area is 3/2 of the original rectangle's area -/
theorem wasted_area_is_three_halves_of_rectangle
  (rect : Rectangle)
  (circ : Circle)
  (sq : Square)
  (h1 : circ.radius = rect.width / 2)  -- Maximum circle
  (h2 : sq.side = circ.radius * Real.sqrt 2)  -- Maximum square in circle
  : rectangle_area rect - square_area sq = (3/2) * rectangle_area rect :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_area_is_three_halves_of_rectangle_l611_61197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l611_61146

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a triangle COB where:
  C is on the y-axis with coordinates (0,p)
  O is at the origin (0,0)
  B is on the x-axis with coordinates (24,0)
  Prove that the area of triangle COB is 12p -/
theorem triangle_COB_area (p : ℝ) : 
  let C : ℝ × ℝ := (0, p)
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (24, 0)
  area_triangle C O B = 12 * p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l611_61146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_count_l611_61191

def sequence_condition (a : Fin 50 → ℤ) : Prop :=
  (∀ i, a i = -1 ∨ a i = 0 ∨ a i = 1) ∧
  (Finset.sum Finset.univ a) = 9 ∧
  (Finset.sum Finset.univ (λ i => (a i + 1)^2)) = 107

theorem zero_count (a : Fin 50 → ℤ) (h : sequence_condition a) :
  (Finset.filter (λ i => a i = 0) Finset.univ).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_count_l611_61191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l611_61137

/-- A complex number z is defined as z = (a^2 - 1) + (a - 2)i, where a is a real number. -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

/-- A complex number is pure imaginary if its real part is zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

/-- The statement "a = 1 is a sufficient but not necessary condition for z to be pure imaginary" -/
theorem a_eq_one_sufficient_not_necessary :
  (∃ a : ℝ, a = 1 ∧ is_pure_imaginary (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ is_pure_imaginary (z a)) := by
  sorry

#check a_eq_one_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l611_61137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l611_61129

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 3*a else a^x

theorem a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) →
  (1/4 < a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l611_61129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_product_remainder_one_l611_61161

theorem triple_product_remainder_one :
  ∀ a b c : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a * b % c = 1) ∧ (b * c % a = 1) ∧ (c * a % b = 1) →
  ((a = 2 ∧ b = 3 ∧ c = 5) ∨
   (a = 2 ∧ b = 5 ∧ c = 3) ∨
   (a = 3 ∧ b = 2 ∧ c = 5) ∨
   (a = 3 ∧ b = 5 ∧ c = 2) ∨
   (a = 5 ∧ b = 2 ∧ c = 3) ∨
   (a = 5 ∧ b = 3 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_product_remainder_one_l611_61161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l611_61176

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2023 * x + Real.pi / 4) + Real.sin (2023 * x - Real.pi / 4)

theorem min_value_theorem (A : ℝ) (h₁ : ∀ x, f x ≤ A) (h₂ : ∃ x, f x = A)
  (h₃ : ∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) :
  ∃ x₁ x₂, A * |x₁ - x₂| = 2 * Real.pi / 2023 ∧ ∀ y₁ y₂, A * |y₁ - y₂| ≥ 2 * Real.pi / 2023 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l611_61176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l611_61159

-- Define the linear regression model
structure LinearRegressionModel where
  R_squared : ℝ
  h_R_squared_range : 0 ≤ R_squared ∧ R_squared ≤ 1

-- Define the properties we want to prove
def good_fitting (model : LinearRegressionModel) : Prop :=
  model.R_squared > 0.9

def explanatory_variable_contribution (model : LinearRegressionModel) : Prop :=
  ⌊model.R_squared * 100⌋ = 96

def random_error_impact (model : LinearRegressionModel) : Prop :=
  ⌊(1 - model.R_squared) * 100⌋ = 4

def points_on_regression_line (model : LinearRegressionModel) (percentage : ℝ) : Prop :=
  percentage ≠ model.R_squared * 100

-- The main theorem
theorem linear_regression_properties (model : LinearRegressionModel) 
  (h : model.R_squared = 0.96) : 
  good_fitting model ∧ 
  explanatory_variable_contribution model ∧ 
  random_error_impact model ∧ 
  points_on_regression_line model 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l611_61159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_2_sqrt_5_l611_61166

structure RectangularParallelepiped where
  ab : ℝ
  bc : ℝ
  cg : ℝ

-- Remove the midpoint definition as it's already defined in Mathlib

noncomputable def pyramidVolume (base_area height : ℝ) : ℝ := (1/3) * base_area * height

theorem pyramid_volume_is_2_sqrt_5 (r : RectangularParallelepiped)
  (h_ab : r.ab = 4)
  (h_bc : r.bc = 2)
  (h_cg : r.cg = 3) :
  ∃ (base_area height : ℝ),
    base_area = 4 * Real.sqrt 5 ∧
    height = 3/2 ∧
    pyramidVolume base_area height = 2 * Real.sqrt 5 := by
  sorry

#check pyramid_volume_is_2_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_2_sqrt_5_l611_61166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l611_61189

theorem negation_of_sin_inequality :
  (¬∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l611_61189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l611_61173

-- Define the two circles
def circle1 : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∃ (min_dist : ℝ), min_dist = 3 ∧
  ∀ (p q : ℝ × ℝ), p ∈ circle1 → q ∈ circle2 → distance p q ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l611_61173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_decrease_second_year_l611_61153

def initial_population : ℚ := 20000
def first_year_increase_percent : ℚ := 25 / 100
def final_population : ℚ := 18750

theorem population_decrease_second_year :
  let first_year_population : ℚ := initial_population * (1 + first_year_increase_percent)
  let second_year_decrease_percent : ℚ := (first_year_population - final_population) / first_year_population
  second_year_decrease_percent = 25 / 100 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_decrease_second_year_l611_61153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_l611_61103

theorem consecutive_integers_sum (a : ℕ → ℕ) : 
  (∀ i : ℕ, i ∈ Finset.range 8 → a (i+1) = a i + 1) →  -- consecutive integers
  (∃ m : ℕ, a 1 + a 3 + a 5 + a 7 + a 9 = m^2) →  -- sum of odd-indexed terms is a square
  (∃ n : ℕ, a 2 + a 4 + a 6 + a 8 = n^3) →        -- sum of even-indexed terms is a cube
  (∀ k : ℕ → ℕ, 
    (∀ i : ℕ, i ∈ Finset.range 8 → k (i+1) = k i + 1) →
    (∃ m : ℕ, k 1 + k 3 + k 5 + k 7 + k 9 = m^2) →
    (∃ n : ℕ, k 2 + k 4 + k 6 + k 8 = n^3) →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 ≤ 
     k 1 + k 2 + k 3 + k 4 + k 5 + k 6 + k 7 + k 8 + k 9)) →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_l611_61103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_BC_l611_61150

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  A : Point
  B : Point
  C : Point
  D : Point
  is_square : A.x = 0 ∧ A.y = 0 ∧ B.x = s ∧ B.y = 0 ∧ C.x = s ∧ C.y = s ∧ D.x = 0 ∧ D.y = s

/-- The intersection point of two quarter-circle arcs -/
noncomputable def intersectionPoint (s : ℝ) : Point :=
  { x := s * Real.sqrt 3 / 2, y := s / 2 }

/-- The theorem stating the distance from the intersection point to side BC -/
theorem distance_to_BC (s : ℝ) (square : Square s) :
  let X := intersectionPoint s
  s - X.y = s / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_BC_l611_61150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisecting_intersections_l611_61172

noncomputable section

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define point P
def P : ℝ × ℝ := (3, 0)

-- Define a general line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 3)

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := ((2 - 3*k) / (2 - k), -6*k / (2 - k))
def B (k : ℝ) : ℝ × ℝ := ((3*k - 3) / (k + 1), -6*k / (k + 1))

-- State the theorem
theorem line_bisecting_intersections :
  ∃ k : ℝ, 
    (∀ x y : ℝ, line_through_P k x y ↔ 8 * x - y - 24 = 0) ∧
    l₁ (A k).1 (A k).2 ∧
    l₂ (B k).1 (B k).2 ∧
    P = ((A k).1 + (B k).1, (A k).2 + (B k).2) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisecting_intersections_l611_61172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l611_61124

open Real

/-- The volume of a right circular cone with height h and base radius r -/
noncomputable def cone_volume (h r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The total volume of ice cream in a cone topped with a hemisphere -/
noncomputable def ice_cream_volume (h r : ℝ) : ℝ := cone_volume h r + hemisphere_volume r

theorem ice_cream_volume_calculation :
  ice_cream_volume 12 3 = 54 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l611_61124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l611_61118

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (1, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through the given point
  perpendicular_line point.1 point.2 ∧
  -- The perpendicular line is indeed perpendicular to the given line
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    given_line x₁ y₁ → given_line x₂ y₂ → 
    perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (point.1 - x₁) + (y₂ - y₁) * (point.2 - y₁)) *
    ((x₂ - x₁) * (x₁ - x₁) + (y₂ - y₁) * (y₁ - y₁)) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l611_61118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_value_l611_61162

-- Define the set B
def B : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 2}

-- Define the function h
def h (x : ℚ) : ℚ := 2 - 1 / (x + 1)

-- Define the properties of function f
def f_property (f : B → ℝ) : Prop :=
  ∀ x : ℚ, x ∈ B → f ⟨x, by sorry⟩ + f ⟨h x, by sorry⟩ = Real.log (|x + 1|)

-- Theorem statement
theorem f_2008_value (f : B → ℝ) (hf : f_property f) :
  f ⟨2008, by sorry⟩ = Real.log ((2009 * 2002) / 2005) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_value_l611_61162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l611_61127

theorem complex_fraction_equality (i : ℂ) : i * i = -1 → (2 * i) / (1 - i) = -1 + i := by
  intro h
  sorry

#check complex_fraction_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l611_61127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ecommerce_sales_theorem_l611_61147

/-- Represents the e-commerce company's sales model -/
structure SalesModel where
  cost_price : ℤ
  max_price : ℤ
  base_price : ℤ
  base_volume : ℤ
  volume_increase : ℤ
  price_decrease : ℤ

/-- Calculates the daily sales volume for a given price -/
def daily_sales_volume (model : SalesModel) (price : ℤ) : ℤ :=
  model.base_volume + model.volume_increase * (model.base_price - price)

/-- Represents the functional relationship between price and sales volume -/
def sales_function (model : SalesModel) (x : ℤ) : ℤ :=
  model.base_volume + model.volume_increase * (model.base_price - x)

/-- Calculates the daily profit for a given price -/
def daily_profit (model : SalesModel) (price : ℤ) : ℤ :=
  (price - model.cost_price) * (daily_sales_volume model price)

/-- Main theorem proving the three parts of the problem -/
theorem ecommerce_sales_theorem (model : SalesModel)
    (h_cost : model.cost_price = 70)
    (h_max_price : model.max_price = 99)
    (h_base_price : model.base_price = 95)
    (h_base_volume : model.base_volume = 50)
    (h_volume_increase : model.volume_increase = 2)
    (h_price_decrease : model.price_decrease = 1) :
    (daily_sales_volume model 80 = 80) ∧
    (∀ x, model.cost_price ≤ x ∧ x ≤ model.max_price →
      sales_function model x = -2 * x + 240) ∧
    (∃ x, model.cost_price ≤ x ∧ x ≤ model.max_price ∧
      daily_profit model x = 1200 ∧ x = 90) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ecommerce_sales_theorem_l611_61147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_speeds_theorem_l611_61105

/-- Represents the route from B to C -/
structure Route where
  uphill : ℝ
  horizontal : ℝ
  downhill : ℝ

/-- Represents the traveler's speeds -/
structure TravelerSpeeds where
  horizontal : ℝ
  uphill : ℝ
  downhill : ℝ

/-- Calculates the time taken for a journey given distance and speed -/
noncomputable def timeTaken (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Theorem stating that given the route and travel times, the traveler's speeds are as calculated -/
theorem traveler_speeds_theorem (route : Route) (halfwayTime minutes207 minutes231 : ℝ) 
  (h1 : route.uphill = 3)
  (h2 : route.horizontal = 5)
  (h3 : route.downhill = 6)
  (h4 : halfwayTime = 216)
  (h5 : minutes207 = 207)
  (h6 : minutes231 = 231) :
  ∃ (speeds : TravelerSpeeds),
    speeds.horizontal = 4 ∧ 
    speeds.uphill = 3 ∧ 
    speeds.downhill = 5 ∧
    timeTaken route.uphill speeds.uphill + 
    timeTaken route.horizontal speeds.horizontal + 
    timeTaken route.downhill speeds.downhill = minutes207 ∧
    timeTaken route.uphill speeds.uphill + 
    timeTaken route.horizontal speeds.horizontal + 
    timeTaken (route.uphill / 2) speeds.downhill = halfwayTime / 2 ∧
    timeTaken route.downhill speeds.uphill + 
    timeTaken route.horizontal speeds.horizontal + 
    timeTaken route.uphill speeds.downhill = minutes231 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_speeds_theorem_l611_61105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_center_cut_through_adjacent_sides_l611_61140

/-- A square in 2D space -/
structure Square where
  side : ℝ
  center : ℝ × ℝ

/-- A line segment in 2D space -/
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Predicate to check if a point is on the perimeter of a square -/
def isOnSquarePerimeter (s : Square) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  let half_side := s.side / 2
  (x = cx - half_side ∨ x = cx + half_side) ∧ (cy - half_side ≤ y ∧ y ≤ cy + half_side) ∨
  (y = cy - half_side ∨ y = cy + half_side) ∧ (cx - half_side ≤ x ∧ x ≤ cx + half_side)

/-- Predicate to check if a point is a vertex of a square -/
def isVertex (s : Square) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  let half_side := s.side / 2
  (x = cx - half_side ∨ x = cx + half_side) ∧ (y = cy - half_side ∨ y = cy + half_side)

/-- Predicate to check if two points are on adjacent sides of a square -/
def areOnAdjacentSides (s : Square) (p1 p2 : ℝ × ℝ) : Prop :=
  isOnSquarePerimeter s p1 ∧ isOnSquarePerimeter s p2 ∧
  ¬(isVertex s p1) ∧ ¬(isVertex s p2) ∧
  (p1.1 = s.center.1 - s.side / 2 ∧ p2.2 = s.center.2 - s.side / 2 ∨
   p1.1 = s.center.1 + s.side / 2 ∧ p2.2 = s.center.2 + s.side / 2 ∨
   p1.2 = s.center.2 - s.side / 2 ∧ p2.1 = s.center.1 - s.side / 2 ∨
   p1.2 = s.center.2 + s.side / 2 ∧ p2.1 = s.center.1 + s.side / 2)

/-- Theorem: It is impossible to cut a line through the center of a square
    that intersects two adjacent sides at non-vertex points -/
theorem no_center_cut_through_adjacent_sides (s : Square) :
  ¬∃ (l : LineSegment), areOnAdjacentSides s l.start l.endpoint ∧
                        (∃ t : ℝ, 0 < t ∧ t < 1 ∧
                         s.center = (1 - t) • l.start + t • l.endpoint) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_center_cut_through_adjacent_sides_l611_61140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l611_61182

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) : ℝ × ℝ := (4 * t, 3 * t - 1)

noncomputable def C₂_polar (θ : ℝ) : ℝ := 8 * Real.cos θ / (1 - Real.cos (2 * θ))

-- Define point P
def P : ℝ × ℝ := C₁ 0

-- Define the Cartesian equations of C₁ and C₂
def C₁_cartesian (x y : ℝ) : Prop := 3 * x - 4 * y - 4 = 0

def C₂_cartesian (x y : ℝ) : Prop := y^2 = 4 * x

-- Theorem statement
theorem intersection_distance_product : 
  ∃ (A B : ℝ × ℝ), 
    C₁_cartesian A.1 A.2 ∧ C₂_cartesian A.1 A.2 ∧
    C₁_cartesian B.1 B.2 ∧ C₂_cartesian B.1 B.2 ∧
    A ≠ B ∧
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)) * 
    (Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) = 25 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l611_61182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l611_61143

noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

def line (x y : ℝ) : Prop := 2*x + y - 10 = 0

noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |2*x + y - 10| / Real.sqrt 5

theorem min_distance_to_line :
  ∀ x y : ℝ, ellipse x y → 
    (∀ x' y' : ℝ, ellipse x' y' → 
      distance_to_line x y ≤ distance_to_line x' y') ∧
    distance_to_line x y = Real.sqrt 5 :=
by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l611_61143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_center_of_gravity_l611_61131

/-- Represents a spherical segment -/
structure SphericalSegment where
  R : ℝ  -- radius of the sphere
  h : ℝ  -- height of the segment
  density_prop : ℝ → ℝ  -- density function

/-- The z-coordinate of the center of gravity of a spherical segment -/
noncomputable def center_of_gravity (s : SphericalSegment) : ℝ :=
  (20 * s.R^2 - 15 * s.R * s.h + 3 * s.h^2) / (5 * (4 * s.R - s.h))

/-- Theorem: The z-coordinate of the center of gravity of a spherical segment
    with radius R and height h, where the volumetric density at each point
    is proportional to its distance from the base of the segment,
    is equal to (20R^2 - 15Rh + 3h^2) / (5(4R - h)) -/
theorem spherical_segment_center_of_gravity (s : SphericalSegment)
  (h_density : ∃ k > 0, ∀ z, s.density_prop z = k * (z - (s.R - s.h))) :
  center_of_gravity s = (20 * s.R^2 - 15 * s.R * s.h + 3 * s.h^2) / (5 * (4 * s.R - s.h)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_center_of_gravity_l611_61131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_line_through_point_l611_61181

/-- Given a line y = -2x + 1 shifted m units downward (m > 0), 
    if the shifted line passes through the point (1, -3), then m = 2 -/
theorem shifted_line_through_point (m : ℝ) : 
  m > 0 → 
  (λ x => -2*x + 1 - m) 1 = -3 → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_line_through_point_l611_61181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosA_l611_61198

theorem triangle_cosA (A B C : ℝ) (h : ℝ) : 
  B = π/4 → 
  C - B = 4 → 
  h = (C - B)/4 → 
  (1/2) * (C - B) * h = (1/2) * (A - B) * (C - B) * Real.sin B →
  Real.cos A = -Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosA_l611_61198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_in_y_approximation_l611_61154

/-- The function we're analyzing -/
def f (x : ℝ) := x^3 + 2*x

/-- The derivative of our function -/
def f' (x : ℝ) := 3*x^2 + 2

theorem change_in_y_approximation :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1.1
  let dx := x₂ - x₁
  let dy := f' x₁ * dx
  ∃ ε > 0, |dy - 0.5| < ε := by
  sorry

#eval f' 1 * 0.1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_in_y_approximation_l611_61154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_wings_cost_l611_61180

def lee_money : ℕ := 10
def friend_money : ℕ := 8
def salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def soda_quantity : ℕ := 2
def tax : ℕ := 3
def change : ℕ := 3

theorem chicken_wings_cost (chicken_wings : ℕ) : chicken_wings = 6 := by
  let total_money : ℕ := lee_money + friend_money
  let total_spent : ℕ := total_money - change
  let other_costs : ℕ := salad_cost + soda_cost * soda_quantity + tax
  have h1 : chicken_wings + other_costs = total_spent := by sorry
  have h2 : chicken_wings = total_spent - other_costs := by sorry
  have h3 : total_spent = 15 := by sorry
  have h4 : other_costs = 9 := by sorry
  calc
    chicken_wings = total_spent - other_costs := h2
    _ = 15 - 9 := by rw [h3, h4]
    _ = 6 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_wings_cost_l611_61180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop4_correct_l611_61183

-- Define the necessary types and functions
def is_ellipse (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∀ (x y : ℝ), x^2 / (m+1) - y^2 / (m-2) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

def is_hyperbola (F₁ F₂ : ℝ × ℝ) (diff : ℝ) (P : ℝ × ℝ → Prop) : Prop :=
  ∀ p, P p → |dist p F₁ - dist p F₂| = diff → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), P (x, y) → x^2 / a^2 - y^2 / b^2 = 1

noncomputable def dist (p₁ p₂ : ℝ × ℝ) : ℝ := Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the propositions
def prop1 : Prop := ∀ m : ℝ, -1 < m ∧ m < 2 ↔ is_ellipse m

def prop2 : Prop := ¬(∀ P : ℝ × ℝ → Prop, is_hyperbola (-4, 0) (4, 0) 8 P) →
                    ∃ P : ℝ × ℝ → Prop, is_hyperbola (-4, 0) (4, 0) 8 P

def prop3 : Prop := ∀ p q : Prop, ¬(p ∧ q) → ¬p ∧ ¬q

def prop4 : Prop := ∀ a : ℝ, (∀ x : ℝ, (x ≥ -3 ∧ x ≤ 1 → x ≤ a) ∧ (x ≤ a → ¬(x < -3 ∨ x > 1))) → a ≥ 1

-- The main theorem
theorem only_prop4_correct : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop4_correct_l611_61183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_value_from_arguments_l611_61175

noncomputable def arg (z : ℂ) : ℝ := Real.arctan (z.im / z.re)

theorem complex_value_from_arguments (z : ℂ) :
  arg (z + 2) = π / 3 ∧ arg (z - 2) = 5 * π / 6 → z = -1 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_value_from_arguments_l611_61175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_not_unique_l611_61157

/-- Represents a train with a speed and length -/
structure Train where
  speed : ℝ
  length : ℝ

/-- The problem setup -/
structure TrainProblem where
  train1 : Train
  train2 : Train
  crossTime1 : ℝ
  crossTime2 : ℝ
  sameDirCrossTime : ℝ

/-- The conditions of the problem -/
def problemConditions (p : TrainProblem) : Prop :=
  p.crossTime1 = 10 ∧
  p.crossTime2 = 15 ∧
  p.sameDirCrossTime = 135 ∧
  p.train1.length = p.train1.speed * p.crossTime1 ∧
  p.train2.length = p.train2.speed * p.crossTime2 ∧
  p.sameDirCrossTime = (p.train1.length + p.train2.length) / (p.train1.speed - p.train2.speed)

/-- The theorem stating that the length of the first train cannot be uniquely determined -/
theorem train_length_not_unique (p : TrainProblem) : 
  problemConditions p → ¬∃! l : ℝ, p.train1.length = l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_not_unique_l611_61157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_education_funding_growth_and_total_l611_61194

/-- Represents the education funding in millions of yuan -/
structure EducationFunding where
  amount : ℝ

/-- The education funding in 2017 -/
def funding_2017 : EducationFunding := ⟨25⟩

/-- The education funding in 2019 -/
def funding_2019 : EducationFunding := ⟨30.25⟩

/-- The average annual growth rate of education funding from 2017 to 2019 -/
def avg_growth_rate : ℝ := 0.1

/-- The total education funding from 2017 to 2019 -/
def total_funding : EducationFunding := ⟨82.75⟩

theorem education_funding_growth_and_total :
  (funding_2017.amount * (1 + avg_growth_rate)^2 = funding_2019.amount) ∧
  (funding_2017.amount + funding_2017.amount * (1 + avg_growth_rate) + funding_2019.amount = total_funding.amount) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_education_funding_growth_and_total_l611_61194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l611_61130

-- Define the points and hexagon
def A : ℝ × ℝ := (0, 0)
def B : ℝ → ℝ × ℝ := λ b => (b, 3)

structure Hexagon (b : ℝ) where
  vertices : Fin 6 → ℝ × ℝ
  convex : Bool
  equilateral : Bool
  angle_FAB : ℝ
  parallel_AB_DE : Bool
  parallel_BC_EF : Bool
  parallel_CD_FA : Bool
  distinct_y_coords : Bool

noncomputable def hexagon_area (h : Hexagon b) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_theorem (b : ℝ) (h : Hexagon b) 
  (h_convex : h.convex = true)
  (h_equilateral : h.equilateral = true)
  (h_angle : h.angle_FAB = 120)
  (h_parallel1 : h.parallel_AB_DE = true)
  (h_parallel2 : h.parallel_BC_EF = true)
  (h_parallel3 : h.parallel_CD_FA = true)
  (h_distinct : h.distinct_y_coords = true)
  (h_y_coords : ∀ i, (h.vertices i).2 ∈ ({0, 3, 6, 9, 12, 15} : Set ℝ))
  (h_A : h.vertices 0 = A)
  (h_B : h.vertices 1 = B b) :
  ∃ (m n : ℕ), hexagon_area h = m * Real.sqrt n ∧ m + n = 75 ∧ 
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l611_61130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flynn_tv_time_l611_61163

/-- The number of minutes of TV watched every weekday night -/
def weekday_tv_minutes (weeks_per_year : ℕ) (weekdays_per_week : ℕ) (weekend_additional_hours : ℕ) (total_annual_hours : ℕ) : ℕ :=
  let total_weekdays := weeks_per_year * weekdays_per_week
  let total_weekend_hours := weeks_per_year * weekend_additional_hours
  let total_weekday_hours := total_annual_hours - total_weekend_hours
  let weekday_hours := total_weekday_hours / total_weekdays
  weekday_hours * 60

theorem flynn_tv_time :
  weekday_tv_minutes 52 5 2 234 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flynn_tv_time_l611_61163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l611_61158

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) (h3 : f 0 φ = 1 / 2) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) φ = f x φ ∧ ∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x : ℝ, f (x + T') φ ≠ f x φ) ∧
  (φ = Real.pi / 6) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x φ ≥ -1 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x φ = -1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l611_61158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_quadrilateral_area_l611_61170

/-- Helper function to calculate the area of a quadrilateral given its four vertices -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  sorry -- Implementation of area calculation

/-- Given a hyperbola with the equation (x²/(16+k)) - (y²/(8-k)) = 1,
    where -16 < k < 8, and one of its asymptotic lines is y = -√3 x,
    prove that the area of quadrilateral F₁QF₂P is 12√6,
    where P(3, y₀) and Q are symmetric points on the hyperbola with respect to the origin. -/
theorem hyperbola_quadrilateral_area :
  ∀ (k : ℝ) (y₀ : ℝ),
  -16 < k →
  k < 8 →
  (∃ (x y : ℝ), x^2 / (16 + k) - y^2 / (8 - k) = 1 ∧ y = -Real.sqrt 3 * x) →
  (∃ (x y : ℝ), x^2 / (16 + k) - y^2 / (8 - k) = 1 ∧ x = 3 ∧ y = y₀) →
  (∃ (x y : ℝ), x^2 / (16 + k) - y^2 / (8 - k) = 1 ∧ x = -3 ∧ y = -y₀) →
  ∃ (F₁ F₂ : ℝ × ℝ), area_quadrilateral F₁ (3, y₀) F₂ (-3, -y₀) = 12 * Real.sqrt 6 :=
by
  sorry -- Proof implementation


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_quadrilateral_area_l611_61170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l611_61128

/-- Given a hyperbola with equation x²/9 - y²/a = 1 and right focus at (√13, 0),
    its asymptotes are given by the equation y = ±(2/3)x -/
theorem hyperbola_asymptotes (a : ℝ) :
  (∃ (x y : ℝ), x^2 / 9 - y^2 / a = 1) →  -- Hyperbola equation
  (Real.sqrt 13, 0) ∈ {z : ℝ × ℝ | z.1^2 / 9 - z.2^2 / a = 1} →  -- Right focus condition
  {z : ℝ × ℝ | z.2 = (2/3) * z.1 ∨ z.2 = -(2/3) * z.1} =
    {z : ℝ × ℝ | ∃ (t : ℝ), (z.1 / (3 * t))^2 - (z.2 / t)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l611_61128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_planes_division_l611_61188

/-- A plane in 3D space -/
structure Plane where

/-- The number of parts that two planes divide space into -/
def divide_space (p1 p2 : Plane) : ℕ :=
  sorry

/-- Two planes are non-coincident -/
def non_coincident (p1 p2 : Plane) : Prop :=
  sorry

theorem two_planes_division (p1 p2 : Plane) (h : non_coincident p1 p2) :
  divide_space p1 p2 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_planes_division_l611_61188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_25_cubes_l611_61139

/-- Represents a three-dimensional arrangement of unit cubes -/
structure CubeArrangement where
  cubes : Nat
  surface_area : Nat

/-- The minimum surface area arrangement for a given number of unit cubes -/
def min_surface_area (n : Nat) : Nat :=
  sorry -- Placeholder for the actual implementation

/-- Properties of the minimum surface area arrangement -/
axiom min_surface_area_properties (n : Nat) :
  ∃ (arr : CubeArrangement), 
    arr.cubes = n ∧ 
    arr.surface_area = min_surface_area n ∧
    ∀ (other : CubeArrangement), other.cubes = n → arr.surface_area ≤ other.surface_area

/-- Theorem: The minimum surface area for 25 unit cubes is 54 -/
theorem min_surface_area_25_cubes : 
  min_surface_area 25 = 54 := by
  sorry

#check min_surface_area_25_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_25_cubes_l611_61139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2y_value_l611_61199

theorem sin_2y_value (x y : ℝ) 
  (h1 : Real.sin x = (3/2) * Real.sin y - (2/3) * Real.cos y)
  (h2 : Real.cos x = (3/2) * Real.cos y - (2/3) * Real.sin y) : 
  Real.sin (2 * y) = 25/72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2y_value_l611_61199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l611_61102

theorem divisibility_conditions (n : ℕ+) :
  ((∀ a : ℕ, Odd a → (4 : ℕ) ∣ (a ^ n.val - 1)) ↔ Even n.val) ∧
  ((∀ a : ℕ, Odd a → (2^2017 : ℕ) ∣ (a ^ n.val - 1)) ↔ ∃ m : ℕ+, n.val = 2^2015 * m.val) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l611_61102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l611_61190

def has_solution_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f x = 0

def unique_solution (f : ℝ → ℝ) : Prop :=
  ∃! x, f x ≤ 0

def proposition_p (a : ℝ) : Prop :=
  has_solution_in_interval (fun x ↦ 2 * x^2 + a * x - a^2) (-1) 1

def proposition_q (a : ℝ) : Prop :=
  unique_solution (fun x ↦ x^2 + 2 * a * x + 2 * a)

theorem range_of_a (a : ℝ) :
  ¬(proposition_p a ∨ proposition_q a) → a ∈ Set.Iic (-2) ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l611_61190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_circle_l611_61196

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the given circles and line
noncomputable def circle1 : Circle := ⟨(0, 0), 1⟩
noncomputable def circle2 : Circle := ⟨(2, 0), 1⟩
noncomputable def intersectingLine : Line := ⟨(0, 1), (2, 1)⟩

-- Define the intersection points
noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (2, 1)
noncomputable def D : ℝ × ℝ := (2, -1)

-- Define the tangent lines (simplified representation)
noncomputable def tangentA : Line := ⟨A, (1, 1)⟩
noncomputable def tangentB : Line := ⟨B, (1, -1)⟩
noncomputable def tangentC : Line := ⟨C, (3, 1)⟩
noncomputable def tangentD : Line := ⟨D, (3, -1)⟩

-- Define the intersection points of tangents
noncomputable def K : ℝ × ℝ := (1, 2)
noncomputable def L : ℝ × ℝ := (1, -2)
noncomputable def M : ℝ × ℝ := (3, 2)
noncomputable def N : ℝ × ℝ := (3, -2)

-- Define the line connecting centers of the given circles
noncomputable def centerLine : Line := ⟨(0, 0), (2, 0)⟩

-- Define a membership relation for points in a circle
def pointInCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a membership relation for points on a line
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  (p.2 - l.point1.2) * (l.point2.1 - l.point1.1) = 
  (p.1 - l.point1.1) * (l.point2.2 - l.point1.2)

-- Theorem statement
theorem tangent_intersection_circle :
  ∃ (newCircle : Circle),
    pointInCircle K newCircle ∧ pointInCircle L newCircle ∧ 
    pointInCircle M newCircle ∧ pointInCircle N newCircle ∧
    pointOnLine newCircle.center centerLine := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_circle_l611_61196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_range_l611_61164

/-- A function that represents (x-5)/(x-a-2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 5) / (x - a - 2)

/-- The theorem stating that if f is monotonically increasing on (-1, +∞), then a ∈ (-∞, -3] -/
theorem monotone_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y → f a x < f a y) →
  a ≤ -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_range_l611_61164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l611_61134

theorem intersection_distance (x₁ x₂ : ℝ) (p q : ℕ) : 
  2 = 3 * x₁^2 + 2 * x₁ - 1 →
  2 = 3 * x₂^2 + 2 * x₂ - 1 →
  x₁ ≠ x₂ →
  Nat.Coprime p q →
  p > 0 →
  q > 0 →
  |x₁ - x₂| = Real.sqrt p / q →
  p - q = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l611_61134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l611_61108

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := (1 + 1/x^2) * (1 + x)^6

-- Theorem statement
theorem coefficient_of_x_squared : 
  (deriv (deriv expression) 0) / 2 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l611_61108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mission_duration_l611_61185

theorem mission_duration (planned_days : ℕ) (overtime_percent : ℚ) (second_mission_days : ℕ) : 
  planned_days = 5 → 
  overtime_percent = 60 / 100 → 
  second_mission_days = 3 → 
  planned_days + (planned_days : ℚ) * overtime_percent + second_mission_days = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mission_duration_l611_61185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_circle_equation_proof_l611_61121

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

def A : Point := ⟨4, 6⟩
def B : Point := ⟨-2, 4⟩

def linePassingThroughWithEqualIntercepts (p : Point) : Line :=
  ⟨1, 1, p.x + p.y⟩

noncomputable def circleWithDiameter (p q : Point) : Circle :=
  let center : Point := ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩
  let radius : ℝ := Real.sqrt (((p.x - q.x)^2 + (p.y - q.y)^2) / 4)
  ⟨center, radius⟩

theorem line_equation_proof :
  linePassingThroughWithEqualIntercepts A = ⟨1, 1, 10⟩ := by sorry

theorem circle_equation_proof :
  circleWithDiameter A B = ⟨⟨1, 5⟩, Real.sqrt 10⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_circle_equation_proof_l611_61121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_standard_deviation_l611_61142

noncomputable def dart_scores : List ℝ := [8, 9, 10, 10, 8]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

noncomputable def standard_deviation (xs : List ℝ) : ℝ := (variance xs).sqrt

theorem dart_standard_deviation :
  standard_deviation dart_scores = Real.sqrt (4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_standard_deviation_l611_61142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_l611_61113

theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 - p.2^2 = 4 ∧ p.2 = k * p.1 - 1) → 
  (k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2) ∧ 
  ¬(∀ k' : ℝ, (∃! p : ℝ × ℝ, p.1^2 - p.2^2 = 4 ∧ p.2 = k' * p.1 - 1) → 
    (k' = Real.sqrt 5 / 2 ∨ k' = -Real.sqrt 5 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_l611_61113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_is_32_l611_61115

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 4 * (2 * Real.cos t - Real.cos (2 * t))
noncomputable def y (t : ℝ) : ℝ := 4 * (2 * Real.sin t - Real.sin (2 * t))

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- State the theorem
theorem arc_length_is_32 :
  arcLength 0 Real.pi = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_is_32_l611_61115
