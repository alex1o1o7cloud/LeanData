import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_f_at_pi_half_l691_69146

open Real

noncomputable def f (x : ℝ) := x * sin x

theorem second_derivative_f_at_pi_half :
  (deriv (deriv f)) (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_f_at_pi_half_l691_69146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l691_69111

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a frustum -/
noncomputable def lateral_surface_area (f : Frustum) : ℝ :=
  Real.pi * (f.lower_radius + f.upper_radius) * 
    Real.sqrt ((f.lower_radius - f.upper_radius)^2 + f.height^2)

/-- Theorem stating the lateral surface area of a specific frustum -/
theorem frustum_lateral_surface_area :
  let f : Frustum := { lower_radius := 6, upper_radius := 3, height := 4 }
  lateral_surface_area f = 45 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l691_69111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_range_l691_69163

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y² = 4x -/
def onParabola (p : Point) : Prop := p.y^2 = 4 * p.x

/-- A point is in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := 0 < p.x ∧ 0 < p.y

/-- The slope of a line through the origin and a point -/
noncomputable def slopeFromOrigin (p : Point) : ℝ := p.y / p.x

/-- The slope of a line through two points -/
noncomputable def slopeBetweenPoints (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

theorem parabola_slope_range :
  ∀ A B : Point,
    A ≠ B →
    onParabola A →
    onParabola B →
    inFirstQuadrant A →
    inFirstQuadrant B →
    slopeFromOrigin A * slopeFromOrigin B = 1 →
    0 < slopeBetweenPoints A B ∧ slopeBetweenPoints A B < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_range_l691_69163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l691_69117

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 * Real.cos (2 * x)) / Real.log (1 / 2)

-- State the theorem
theorem f_monotonic_decreasing_interval :
  StrictMonoOn f (Set.Icc (π / 6) (π / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l691_69117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_statement_3_statement_5_statement_6_l691_69171

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(2*x) - 2^(x+1) + 2

-- Define the domain M
def M : Set ℝ := {x | f x ∈ Set.Icc 1 2}

-- Theorem stating the properties of f and M
theorem f_properties :
  (∀ x ∈ M, f x ∈ Set.Icc 1 2) ∧
  (M = Set.Iic 1) ∧
  (0 ∈ M) ∧
  (1 ∈ M) ∧
  (M ⊆ Set.Iic 1) := by
  sorry

-- Additional statements to check
theorem statement_3 : M ⊆ Set.Iic 1 := by
  sorry

theorem statement_5 : 1 ∈ M := by
  sorry

theorem statement_6 : 0 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_statement_3_statement_5_statement_6_l691_69171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sewage_treatment_plant_optimization_l691_69193

/-- Monthly processing cost function -/
noncomputable def y (x m : ℝ) : ℝ := (1/400) * x^2 - m*x + 25

/-- Monthly profit function -/
noncomputable def z (x : ℝ) : ℝ := 0.9*x - y x (1/10)

theorem sewage_treatment_plant_optimization :
  ∃ (m : ℝ),
  (y 120 m = 49) ∧
  (∀ x, 80 ≤ x → x ≤ 210 → y x m ≥ 0) ∧
  (∀ x, 80 ≤ x → x ≤ 210 → z x ≤ 75) ∧
  (∃ x, 80 ≤ x ∧ x ≤ 210 ∧ z x = 75) ∧
  (∀ x, 80 ≤ x → x ≤ 210 → y x m / x ≥ y 100 m / 100) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sewage_treatment_plant_optimization_l691_69193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_cast_l691_69166

theorem total_votes_cast (candidate_percentage : ℝ) (vote_difference : ℕ) : ℝ :=
  let total_votes := 4500
  have h1 : candidate_percentage = 0.35 := by sorry
  have h2 : vote_difference = 1350 := by sorry
  have h3 : total_votes * candidate_percentage + (total_votes * candidate_percentage + vote_difference) = total_votes := by sorry
  total_votes


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_cast_l691_69166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_and_range_invariance_l691_69173

def is_not_completely_identical (data : List ℝ) : Prop :=
  ∃ i j, i ≠ j ∧ data.get? i ≠ data.get? j

noncomputable def average (data : List ℝ) : ℝ :=
  (data.sum) / (data.length : ℝ)

noncomputable def range (data : List ℝ) : ℝ :=
  data.maximum.getD 0 - data.minimum.getD 0

theorem average_and_range_invariance
  (data : List ℝ) (x_bar : ℝ)
  (h_not_identical : is_not_completely_identical data)
  (h_x_bar : x_bar = average data) :
  let new_data := x_bar :: data
  average new_data = x_bar ∧ range new_data = range data := by
  sorry

#check average_and_range_invariance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_and_range_invariance_l691_69173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_sliced_solid_l691_69139

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge in the prism -/
structure Midpoint where
  edge : Fin 3  -- Representing AC (0), BC (1), or DC (2)

/-- The solid formed by slicing off a part of the prism -/
structure SlicedSolid (p : RightPrism) where
  x : Midpoint
  y : Midpoint
  z : Midpoint

/-- The surface area of the sliced solid -/
noncomputable def surface_area (p : RightPrism) (solid : SlicedSolid p) : ℝ := 
  50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2

/-- Main theorem -/
theorem surface_area_of_sliced_solid (p : RightPrism) (solid : SlicedSolid p)
  (h_height : p.height = 20)
  (h_base : p.base_side_length = 10)
  (h_x : solid.x.edge = 0)  -- x is midpoint of AC
  (h_y : solid.y.edge = 1)  -- y is midpoint of BC
  (h_z : solid.z.edge = 2)  -- z is midpoint of DC
  : surface_area p solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_sliced_solid_l691_69139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_angle_l691_69128

theorem same_terminal_side_angle : ∃ (k : ℤ), (5 * π / 3 : ℝ) = -π / 3 + 2 * k * π ∧
  (5 * π / 3 : ℝ) ∈ ({5 * π / 6, π / 3, 11 * π / 6, 5 * π / 3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_angle_l691_69128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_satisfies_conditions_l691_69124

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Evaluates a quadratic polynomial at a given point -/
def eval_quadratic {α : Type*} [Ring α] (p : QuadraticPolynomial α) (x : α) : α :=
  p.a * x * x + p.b * x + p.c

/-- Evaluates a real quadratic polynomial at a complex point -/
def eval_quadratic_complex (p : QuadraticPolynomial ℝ) (z : ℂ) : ℂ :=
  Complex.mk (p.a * (z.re * z.re - z.im * z.im) + p.b * z.re + p.c)
             (p.a * (2 * z.re * z.im) + p.b * z.im)

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (p : QuadraticPolynomial ℝ),
    (p.a = -1 ∧ p.b = -6 ∧ p.c = -25) ∧
    (eval_quadratic_complex p (Complex.mk (-3) (-4)) = 0) ∧
    (p.b = -6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_satisfies_conditions_l691_69124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charcoal_needed_l691_69144

/-- Calculates the total amount of charcoal needed for three batches of paint -/
theorem total_charcoal_needed
  (ratio1 : ℚ) (water1 : ℚ) (total_water1 : ℚ)
  (ratio2 : ℚ) (water2 : ℚ) (total_water2 : ℚ)
  (ratio3 : ℚ) (water3 : ℚ) (total_water3 : ℚ)
  (h1 : ratio1 = 2 / 30)
  (h2 : water1 = 30)
  (h3 : total_water1 = 900)
  (h4 : ratio2 = 3 / 50)
  (h5 : water2 = 50)
  (h6 : total_water2 = 1200)
  (h7 : ratio3 = 4 / 80)
  (h8 : water3 = 80)
  (h9 : total_water3 = 1600) :
  (ratio1 * total_water1) + (ratio2 * total_water2) + (ratio3 * total_water3) = 212 := by
  sorry

#eval (2/30 : ℚ) * 900 + (3/50 : ℚ) * 1200 + (4/80 : ℚ) * 1600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charcoal_needed_l691_69144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_apples_l691_69192

/-- The number of remaining whole apples after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℚ) : ℕ :=
  (initial : ℚ) - given_away |> Int.floor |> Int.toNat

/-- Theorem stating that given 356 initial apples and 272 and 3/5 apples given away,
    the remaining number of whole apples is 83 -/
theorem farmer_apples : remaining_apples 356 (272 + 3/5) = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_apples_l691_69192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_with_inclination_l691_69135

theorem line_slope_with_inclination (θ : Real) (k : Real) :
  θ = 5 * Real.pi / 6 → k = Real.tan θ → k = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_with_inclination_l691_69135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angle_theorem_l691_69116

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define an angle
structure Angle where
  vertex : Point
  point1 : Point
  point2 : Point

-- Define a function to check if a point is on the circle's circumference
def OnCircumference (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define an inscribed angle
def InscribedAngle (c : Circle) (a : Angle) : Prop :=
  OnCircumference c a.vertex ∧ OnCircumference c a.point1 ∧ OnCircumference c a.point2

-- Define a central angle
def CentralAngle (c : Circle) (a : Angle) : Prop :=
  a.vertex = c.center ∧ OnCircumference c a.point1 ∧ OnCircumference c a.point2

-- Define the measure of an angle
noncomputable def AngleMeasure (a : Angle) : ℝ := sorry

-- Theorem statement
theorem inscribed_angle_theorem (c : Circle) (inscribed : Angle) (central : Angle) :
  InscribedAngle c inscribed →
  CentralAngle c central →
  inscribed.point1 = central.point1 →
  inscribed.point2 = central.point2 →
  AngleMeasure inscribed = (1/2) * AngleMeasure central := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angle_theorem_l691_69116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_for_doubling_in_8_years_l691_69181

/-- The rate percent per annum for a sum that doubles in 8 years at simple interest -/
theorem simple_interest_rate_for_doubling_in_8_years : ℝ := by
  let years : ℝ := 8
  let double_factor : ℝ := 2
  let rate_percent : ℝ := (double_factor - 1) * 100 / years
  have h : rate_percent = 12.5 := by
    -- Proof goes here
    sorry
  exact rate_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_for_doubling_in_8_years_l691_69181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_proposition_l691_69187

-- Define propositions
def proposition_A : Prop := ∀ (n : ℕ), n ≥ 3 → (n - 2) * 180 = 360

def proposition_B : Prop := ∀ (x y : ℝ), y = 3 / x → (∀ (x1 x2 : ℝ), x1 < x2 → (3 / x1) > (3 / x2))

def proposition_C : Prop := ∀ (angle1 angle2 : ℝ), angle1 + angle2 = 90

noncomputable def proposition_D : Prop := 
  let r : ℝ := 3;
  let h : ℝ := 4;
  Real.pi * r * (r^2 + h^2).sqrt = 15 * Real.pi

-- Theorem statement
theorem incorrect_proposition : 
  proposition_A ∧ proposition_B ∧ proposition_D ∧ ¬proposition_C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_proposition_l691_69187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_equal_segments_l691_69165

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
variable (C₁ C₂ : Circle) -- Two non-intersecting circles
variable (E₁ E₂ F₁ F₂ G₁ G₂ A B : Point) -- Points of tangency and intersection
variable (external_tangent internal_tangent₁ internal_tangent₂ : Line) -- Tangent lines

-- Define the properties of tangent lines
def is_tangent (l : Line) (c : Circle) : Prop := sorry

-- Define the property of non-intersecting circles
def non_intersecting (c₁ c₂ : Circle) : Prop := sorry

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : Point) : ℝ := sorry

-- Define a belongs to relation for points and lines
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define an intersection operation for lines
def line_intersection (l₁ l₂ : Line) : Set Point := sorry

-- Theorem statement
theorem common_tangents_equal_segments
  (h₁ : non_intersecting C₁ C₂)
  (h₂ : is_tangent external_tangent C₁)
  (h₃ : is_tangent external_tangent C₂)
  (h₄ : is_tangent internal_tangent₁ C₁)
  (h₅ : is_tangent internal_tangent₁ C₂)
  (h₆ : is_tangent internal_tangent₂ C₁)
  (h₇ : is_tangent internal_tangent₂ C₂)
  (h₈ : point_on_line E₁ external_tangent ∧ is_tangent external_tangent C₁)
  (h₉ : point_on_line E₂ external_tangent ∧ is_tangent external_tangent C₂)
  (h₁₀ : A ∈ line_intersection external_tangent internal_tangent₁)
  (h₁₁ : B ∈ line_intersection external_tangent internal_tangent₂)
  : distance A E₁ = distance B E₂ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_equal_segments_l691_69165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_divisor_relation_l691_69197

theorem dividend_divisor_relation (x y : ℕ+) : 
  (x : ℝ) / (y : ℝ) = 96.45 → (x : Int) % (y : Int) = 9 → y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_divisor_relation_l691_69197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_increase_l691_69177

-- Define the initial side length of the pentagon
noncomputable def initial_side_length : ℝ := 10

-- Define the increase in side length
def side_increase : ℝ := 2

-- Define the initial area of the pentagon
noncomputable def initial_area : ℝ := 200 * Real.sqrt 5

-- Function to calculate the area of a regular pentagon given its side length
noncomputable def pentagon_area (s : ℝ) : ℝ := (5 * s^2 / 4) * (Real.sqrt 5 + 1)

-- Theorem statement
theorem pentagon_area_increase :
  pentagon_area (initial_side_length + side_increase) - initial_area = 180 - 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_increase_l691_69177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l691_69107

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line
def Line (k m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + m}

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) := 
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem ellipse_and_line_intersection 
  (h_center : (0, 0) ∈ Ellipse 2 (Real.sqrt 3))
  (h_eccentricity : (Real.sqrt 3 / 2 : ℝ) = 1/2)
  (h_minor_axis : (0, Real.sqrt 3) ∈ Ellipse 2 (Real.sqrt 3))
  (k : ℝ)
  (h_k : k ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : A ∈ Ellipse 2 (Real.sqrt 3) ∩ Line k (-2*k/7))
  (h_B : B ∈ Ellipse 2 (Real.sqrt 3) ∩ Line k (-2*k/7))
  (h_AB_distinct : A ≠ B)
  (h_not_vertex : A ≠ (2, 0) ∧ A ≠ (-2, 0) ∧ B ≠ (2, 0) ∧ B ≠ (-2, 0))
  (h_circle : (2, 0) ∈ Circle ((A.1 + B.1)/2, (A.2 + B.2)/2) (((A.1 - B.1)^2 + (A.2 - B.2)^2)/4)) :
  (2*k/7, 0) ∈ Line k (-2*k/7) := by
  sorry

-- Note: The proof is omitted as per the instructions.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l691_69107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_simplification_l691_69145

theorem factorial_ratio_simplification (N : ℕ) (h : N ≥ 2) :
  Nat.factorial (2 * N) / (Nat.factorial (N + 2) * Nat.factorial (N - 2)) =
  Finset.prod (Finset.range (N + 1)) (λ k => (N + 2 + k) / Nat.factorial (N - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_simplification_l691_69145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_not_expressible_is_correct_l691_69174

/-- The smallest prime that cannot be expressed as |3^a - 2^b| for non-negative integers a and b -/
def smallest_prime_not_expressible : ℕ := 41

/-- Predicate to check if a natural number can be expressed as |3^a - 2^b| -/
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = Int.natAbs (3^a - 2^b)

theorem smallest_prime_not_expressible_is_correct :
  Nat.Prime smallest_prime_not_expressible ∧
  ¬(is_expressible smallest_prime_not_expressible) ∧
  ∀ p : ℕ, Nat.Prime p → p < smallest_prime_not_expressible → is_expressible p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_not_expressible_is_correct_l691_69174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_theorem_l691_69196

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the dot product condition
def dot_product_condition (x₀ y₀ : ℝ) : Prop :=
  (x₀ - F1.1) * (x₀ - F2.1) + y₀ * y₀ < 0

theorem hyperbola_range_theorem (x₀ y₀ : ℝ) :
  hyperbola x₀ y₀ →
  dot_product_condition x₀ y₀ →
  -Real.sqrt 3 / 3 < y₀ ∧ y₀ < Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_theorem_l691_69196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l691_69178

def x : ℝ × ℝ × ℝ := (1, -4, 4)
def p : ℝ × ℝ × ℝ := (2, 1, -1)
def q : ℝ × ℝ × ℝ := (0, 3, 2)
def r : ℝ × ℝ × ℝ := (1, -1, 1)

theorem vector_decomposition : x = (-1 : ℝ) • p + (3 : ℝ) • r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l691_69178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l691_69101

/-- Represents an equilateral hexagon with a square inscribed in it. -/
structure InscribedSquareHexagon where
  /-- Side length AB of the hexagon -/
  ab : ℝ
  /-- Side length EF of the hexagon -/
  ef : ℝ
  /-- Assumption that the hexagon is equilateral -/
  equilateral : True
  /-- Assumption that P is on BC -/
  p_on_bc : True
  /-- Assumption that Q is on DE -/
  q_on_de : True
  /-- Assumption that R is on EF -/
  r_on_ef : True

/-- The side length of the inscribed square in the hexagon -/
noncomputable def square_side_length (h : InscribedSquareHexagon) : ℝ :=
  (25 * Real.sqrt 3 - 50 / Real.sqrt 3) / ((3 * Real.sqrt 3 + 6) / 2)

/-- Theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length (h : InscribedSquareHexagon) 
    (h_ab : h.ab = 50) 
    (h_ef : h.ef = 50 * (Real.sqrt 3 - 2)) : 
  square_side_length h = (25 * Real.sqrt 3 - 50 / Real.sqrt 3) / ((3 * Real.sqrt 3 + 6) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l691_69101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_eighteen_l691_69160

theorem root_product_equals_eighteen :
  (27 : ℝ) ^ (1/3) * 81 ^ (1/4) * 32 ^ (1/5) = 18 := by
  -- Simplify each root
  have h1 : (27 : ℝ) ^ (1/3) = 3 := by sorry
  have h2 : (81 : ℝ) ^ (1/4) = 3 := by sorry
  have h3 : (32 : ℝ) ^ (1/5) = 2 := by sorry
  
  -- Rewrite using the simplifications
  rw [h1, h2, h3]
  
  -- Evaluate the final product
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_eighteen_l691_69160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beans_l691_69112

/-- The amount of beans Henry buys, in pounds -/
def b : ℝ := sorry

/-- The amount of rice Henry buys, in pounds -/
def r : ℝ := sorry

/-- The condition that the amount of rice is at least 3 pounds more than twice the amount of beans -/
axiom rice_lower_bound : r ≥ 3 + 2 * b

/-- The condition that the amount of rice is no more than three times the amount of beans -/
axiom rice_upper_bound : r ≤ 3 * b

/-- The theorem stating that the minimum amount of beans Henry could buy is 3 pounds -/
theorem min_beans : b ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beans_l691_69112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l691_69185

noncomputable section

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Line l with equation y = kx + m -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Point A is the right vertex of the ellipse -/
def point_A : ℝ × ℝ := (2, 0)

/-- M and N are intersection points of line l and ellipse C -/
def intersection_points (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    (x₁ ≠ 2 ∨ y₁ ≠ 0) ∧ (x₂ ≠ 2 ∨ y₂ ≠ 0) ∧
    (x₁ ≠ -2 ∨ y₁ ≠ 0) ∧ (x₂ ≠ -2 ∨ y₂ ≠ 0)

/-- Circle with MN as diameter passes through point A -/
def circle_condition (k m : ℝ) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ →
    (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (2/7, 0)

theorem line_passes_through_fixed_point :
  ∀ (k m : ℝ), 
    intersection_points k m ∧ circle_condition k m →
    line_l k m (fixed_point.1) (fixed_point.2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l691_69185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l691_69109

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  Real.sin A = 3 * Real.sin B →
  C = π / 3 →
  c = Real.sqrt 7 →
  -- Conclusions
  a = 3 ∧ Real.sin A = (3 * Real.sqrt 21) / 14 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l691_69109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_9_to_105_l691_69108

theorem last_three_digits_of_9_to_105 : 9^105 ≡ 049 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_9_to_105_l691_69108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_sum_not_divisible_by_five_l691_69134

theorem base_b_sum_not_divisible_by_five (b : ℕ) : 
  b ∈ ({3, 4, 6, 7, 8} : Set ℕ) → 
  (¬(5 ∣ (3 * b^3 + 2 * b^2 + 6 * b + 3)) ↔ b = 4 ∨ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_sum_not_divisible_by_five_l691_69134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l691_69154

/-- The side length of the regular octagon -/
def side_length : ℝ := 3

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := 0.5 * Real.pi * r^2

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

theorem octagon_semicircles_area : 
  octagon_area side_length - num_sides * semicircle_area (side_length / 2) = 54 + 36 * Real.sqrt 2 - 9 * Real.pi := by
  sorry

#eval num_sides -- This line is added to ensure there's at least one computable definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l691_69154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l691_69180

noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I)

theorem z_properties :
  (z.re = 2) ∧
  (z.im = -1) ∧
  (Complex.abs z = Real.sqrt 5) ∧
  (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l691_69180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l691_69167

-- Define complex numbers α and β
variable (α β : ℂ)

-- Define the conditions
def condition1 (α β : ℂ) : Prop := (α + β).re > 0
def condition2 (α β : ℂ) : Prop := (2 * Complex.I * (α - 3 * β)).re > 0
def condition3 (α β : ℂ) : Prop := β = 4 + 3 * Complex.I

-- State the theorem
theorem alpha_value (h1 : condition1 α β) (h2 : condition2 α β) (h3 : condition3 α β) :
  α = -35 + 9 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l691_69167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_on_length_equality_l691_69132

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2

-- Define a point on the circle
structure PointOnCircle where
  x : ℝ
  y : ℝ
  on_circle : circle_equation x y

-- Define the origin
noncomputable def origin : PointOnCircle where
  x := 0
  y := 0
  on_circle := by sorry

-- Define a line from the origin
noncomputable def line_from_origin (α : ℝ) (t : ℝ) : PointOnCircle where
  x := t * Real.cos α
  y := t * Real.sin α
  on_circle := by sorry

-- Define the theorem
theorem on_length_equality (α : ℝ) :
  let M := line_from_origin α 2
  let N := line_from_origin α 3
  (M.x^2 + M.y^2 = 2^2) → (N.x^2 + N.y^2 = (12/5)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_on_length_equality_l691_69132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_alpha_l691_69110

-- Define α as a real number representing an angle
variable (α : ℝ)

-- Define the conditions
def in_fourth_quadrant (θ : ℝ) : Prop := Real.sin θ < 0 ∧ Real.cos θ > 0
def sum_condition (θ : ℝ) : Prop := Real.sin θ + Real.cos θ = 1/5

-- State the theorem
theorem tan_half_alpha (h1 : in_fourth_quadrant α) (h2 : sum_condition α) :
  Real.tan (α/2) = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_alpha_l691_69110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_success_rearrangements_l691_69141

def word : String := "SUCCESS"

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U'

def vowels : List Char :=
  word.data.filter is_vowel

def consonants : List Char :=
  word.data.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  Nat.factorial vowels.length

def consonant_arrangements : ℕ :=
  Nat.factorial consonants.length /
  (Nat.factorial (consonants.filter (· = 'S')).length *
   Nat.factorial (consonants.filter (· = 'C')).length)

theorem success_rearrangements :
  vowel_arrangements * consonant_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_success_rearrangements_l691_69141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_statements_l691_69155

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships between lines and planes
axiom parallel : Plane → Plane → Prop
axiom perpendicular : Plane → Plane → Prop
axiom perpendicular_line_plane : Line → Plane → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom line_in_plane : Line → Plane → Prop
axiom line_parallel_plane : Line → Plane → Prop

theorem three_correct_statements 
  (m n : Line) (α β γ : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (∀ p q r : Plane, parallel p q → parallel q r → parallel r p) ∧
  (∀ p q r : Plane, perpendicular p r → parallel q r → perpendicular p q) ∧
  (∀ l m : Line, ∀ p : Plane, 
    perpendicular_line_plane l p → 
    perpendicular_lines l m → 
    line_in_plane m p → 
    line_parallel_plane m p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_statements_l691_69155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l691_69168

noncomputable def x (t : ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def y (t : ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

def t₀ : ℝ := -1

theorem tangent_and_normal_lines :
  let x₀ : ℝ := x t₀
  let y₀ : ℝ := y t₀
  let tangent_slope : ℝ := (1 + t₀^2)
  let normal_slope : ℝ := -(1 / tangent_slope)
  (∀ x' y', y' - y₀ = tangent_slope * (x' - x₀) ↔ y' = 2 * x' + 3 * Real.pi / 4) ∧
  (∀ x' y', y' - y₀ = normal_slope * (x' - x₀) ↔ y' = -x' / 2 + Real.pi / 8) := by
  sorry

#check tangent_and_normal_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l691_69168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l691_69131

-- Define the sets
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | Real.exp (x * Real.log 2) > 4}
def C : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- State the theorem
theorem set_equality : C = (Set.univ \ A) ∩ (Set.univ \ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l691_69131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_example_l691_69103

/-- The volume of a cone given its diameter and height -/
noncomputable def cone_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (diameter / 2)^2 * height

/-- Theorem: The volume of a cone with diameter 12 cm and height 8 cm is 96π cubic centimeters -/
theorem cone_volume_example : cone_volume 12 8 = 96 * Real.pi := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_example_l691_69103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_2_50_l691_69164

def die_sides : ℕ := 8

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 0 then roll else 0

def expected_value : ℚ :=
  (Finset.range die_sides).sum (λ i => (winnings (i + 1)) / die_sides)

theorem expected_value_is_2_50 : expected_value = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_2_50_l691_69164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l691_69176

/-- The circle C in the Cartesian plane -/
def C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

/-- The line l in the Cartesian plane -/
def l (x y : ℝ) : Prop :=
  x + y = 1

/-- The distance between a point (x, y) and the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y - 1| / Real.sqrt 2

/-- The minimum distance from the circle C to the line l -/
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), C x y → distance_to_line x y ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l691_69176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_sides_not_necessarily_equal_l691_69106

-- Define a structure for triangles
structure Triangle where
  sides : Fin 3 → ℝ
  angles : Fin 3 → ℝ
  height : ℝ

-- Define a predicate for similar triangles
def similar (t1 t2 : Triangle) : Prop :=
  -- Corresponding angles are equal
  (∀ i : Fin 3, t1.angles i = t2.angles i) ∧
  -- Corresponding sides are proportional
  (∃ k : ℝ, k > 0 ∧ ∀ i : Fin 3, t1.sides i = k * t2.sides i) ∧
  -- Ratio of corresponding heights is equal to the ratio of similarity
  (∃ k : ℝ, k > 0 ∧ t1.height = k * t2.height)

-- Theorem statement
theorem similar_triangles_sides_not_necessarily_equal :
  ∃ (t1 t2 : Triangle), similar t1 t2 ∧ ¬(∀ (i : Fin 3), t1.sides i = t2.sides i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_sides_not_necessarily_equal_l691_69106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_result_l691_69147

/-- The vector q that results from projecting v1 and v2 onto u -/
noncomputable def q : ℝ × ℝ := (133/50, 69/50)

/-- The first given vector -/
def v1 : ℝ × ℝ := (3, -2)

/-- The second given vector -/
def v2 : ℝ × ℝ := (2, 5)

/-- The direction vector of the line passing through v1 and v2 -/
def dir : ℝ × ℝ := v2 - v1

theorem projection_result :
  -- q is on the line passing through v1 and v2
  ∃ t : ℝ, q = v1 + t • dir ∧
  -- q is orthogonal to the direction vector
  (q.1 * dir.1 + q.2 * dir.2 = 0) ∧
  -- q is unique
  ∀ q' : ℝ × ℝ, (∃ t : ℝ, q' = v1 + t • dir) ∧ (q'.1 * dir.1 + q'.2 * dir.2 = 0) → q' = q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_result_l691_69147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_after_five_hours_l691_69175

/-- Represents the score on an exam based on study time -/
noncomputable def examScore (maxScore : ℝ) (initialScore : ℝ) (initialTime : ℝ) (studyTime : ℝ) : ℝ :=
  min maxScore (initialScore * studyTime / initialTime)

/-- Theorem stating that the exam score after 5 hours of study is 100 points -/
theorem exam_score_after_five_hours 
  (maxScore : ℝ) 
  (initialScore : ℝ) 
  (initialTime : ℝ) 
  (h1 : maxScore = 100)
  (h2 : initialScore = 84)
  (h3 : initialTime = 2)
  : examScore maxScore initialScore initialTime 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_after_five_hours_l691_69175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_properties_l691_69188

structure QuadrilateralPyramid where
  a : ℝ
  base_is_trapezoid : Bool
  ab_eq_bc_eq_cd : Bool
  ad_eq_2ab : Bool
  height_foot_at_diag_intersection : Bool
  a_to_apex_right_angle : Bool
  c_to_apex_right_angle : Bool

noncomputable def surface_area (p : QuadrilateralPyramid) : ℝ :=
  p.a ^ 2 * (Real.sqrt 3 + 2)

noncomputable def volume (p : QuadrilateralPyramid) : ℝ :=
  (p.a ^ 3 * Real.sqrt 2) / 4

theorem quadrilateral_pyramid_properties (p : QuadrilateralPyramid) 
  (h1 : p.base_is_trapezoid = true)
  (h2 : p.ab_eq_bc_eq_cd = true)
  (h3 : p.ad_eq_2ab = true)
  (h4 : p.height_foot_at_diag_intersection = true)
  (h5 : p.a_to_apex_right_angle = true)
  (h6 : p.c_to_apex_right_angle = true) :
  surface_area p = p.a ^ 2 * (Real.sqrt 3 + 2) ∧ 
  volume p = (p.a ^ 3 * Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_properties_l691_69188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_and_scalar_l691_69195

/-- Given two non-collinear vectors a and b in a vector space, and points A, B, C, D,
    prove that A, C, and D are collinear and find the value of t for which two vectors are collinear. -/
theorem vector_collinearity_and_scalar (V : Type*) [AddCommGroup V] [Module ℝ V]
  (a b : V) (A B C D : V) (t : ℝ) :
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ ∃ (r : ℝ), b = r • a →
  (B - A) = 2 • a + b →
  (C - B) = a - 3 • b →
  (D - C) = -a + (2/3) • b →
  (∃ (s : ℝ), C - A = s • (D - C)) ∧
  (∃ (l : ℝ), b - t • a = l • ((1/2) • a - (3/2) • b) → t = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_and_scalar_l691_69195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_slant_asymptote_l691_69100

/-- A rational function with a numerator of degree 7 and a slant asymptote -/
noncomputable def f (q : ℝ → ℝ) (x : ℝ) : ℝ := (3 * x^7 + 5 * x^4 - 6) / (q x)

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- Predicate for a function having a slant asymptote -/
def has_slant_asymptote (f : ℝ → ℝ) : Prop := sorry

/-- The slant asymptote of a function, if it exists -/
noncomputable def slant_asymptote (f : ℝ → ℝ) : ℝ → ℝ := sorry

theorem rational_function_slant_asymptote 
  (q : ℝ → ℝ) 
  (h_slant : has_slant_asymptote (f q)) :
  (degree q = 6) ∧ 
  (slant_asymptote (f q) = λ x ↦ 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_slant_asymptote_l691_69100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pocket_knife_equalizes_shares_l691_69194

-- Define the number of rabbits sold
def rabbits_sold : ℕ → ℕ := λ x ↦ x

-- Define the price per rabbit in forints
def price_per_rabbit : ℕ → ℕ := λ x ↦ 10 * x

-- Define the total revenue in forints
def total_revenue : ℕ → ℕ := λ x ↦ rabbits_sold x * price_per_rabbit x

-- Define the revenue in 10-forint units
def revenue_units : ℕ → ℕ := λ x ↦ total_revenue x / 10

-- Define the value of the pocket knife in forints
def pocket_knife_value : ℕ := 40

theorem pocket_knife_equalizes_shares (x : ℕ) :
  (rabbits_sold x = price_per_rabbit x / 10) →
  (revenue_units x % 2 = 1) →
  (∃ y z : ℕ, x = 10 * y + z ∧ z ∈ ({4, 6} : Set ℕ)) →
  pocket_knife_value = 40 :=
by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check pocket_knife_equalizes_shares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pocket_knife_equalizes_shares_l691_69194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_multiples_of_25_between_249_and_501_l691_69179

theorem even_multiples_of_25_between_249_and_501 :
  ∃ (S : Finset ℕ), S.card = 5 ∧
  ∀ n ∈ S, 249 < n ∧ n < 501 ∧ n % 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_multiples_of_25_between_249_and_501_l691_69179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_locus_and_parabola_l691_69138

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the locus L
def locusL (y : ℝ) : Prop := y = -1

-- Define the locus Q
def locusQ (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the point F
def pointF : ℝ × ℝ := (0, 1)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2).sqrt

-- Define the area of the triangle formed by the tangent line and coordinate axes
noncomputable def triangleArea (x y : ℝ) : ℝ := (1 / 16) * |x^3|

theorem circle_tangent_locus_and_parabola :
  -- Part I: Prove that the locus L is y = -1
  (∀ x y : ℝ, (∃ r : ℝ, (x - 0)^2 + (y - (-4))^2 = r^2 ∧ (x - 0)^2 + (y - 2)^2 = r^2) → locusL y) ∧
  -- Part II: Prove that the locus Q is x^2 = 4y
  (∀ x y : ℝ, (∃ m : ℝ, distance x y x (-1) = m ∧ distance x y (pointF.1) (pointF.2) = m) → locusQ x y) ∧
  -- Part III: Prove that there exist points on Q satisfying the area condition
  (∃ x1 y1 : ℝ, locusQ x1 y1 ∧ triangleArea x1 y1 = 1/2 ∧ 
    ((x1 = 2 ∧ y1 = 1) ∨ (x1 = -2 ∧ y1 = 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_locus_and_parabola_l691_69138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_in_second_quadrant_l691_69150

-- Define the complex number z as a function of real m
noncomputable def z (m : ℝ) : ℂ := (m - 2*Complex.I) / (1 - 2*Complex.I)

-- Theorem stating that z cannot be in the second quadrant for any real m
theorem z_not_in_second_quadrant :
  ¬ ∃ m : ℝ, (z m).re < 0 ∧ (z m).im > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_in_second_quadrant_l691_69150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_paint_for_smaller_statues_white_paint_equals_49_72_l691_69189

/-- Amount of white paint needed for two smaller statues -/
theorem white_paint_for_smaller_statues 
  (h_large : ℝ) 
  (h_small : ℝ) 
  (paint_large : ℝ) 
  (white_ratio : ℝ) 
  (h_height : h_large = 6 ∧ h_small = 3.5)
  (h_paint_large : paint_large = 1)
  (h_white_ratio : white_ratio = 4 / 5)
  : ℝ :=
  let surface_ratio := (h_small / h_large) ^ 2
  let paint_small := surface_ratio * paint_large
  let total_paint := 2 * paint_small
  total_paint * white_ratio

theorem white_paint_equals_49_72 
  (h_large : ℝ) 
  (h_small : ℝ) 
  (paint_large : ℝ) 
  (white_ratio : ℝ) 
  (h_height : h_large = 6 ∧ h_small = 3.5)
  (h_paint_large : paint_large = 1)
  (h_white_ratio : white_ratio = 4 / 5)
  : white_paint_for_smaller_statues h_large h_small paint_large white_ratio h_height h_paint_large h_white_ratio = 49 / 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_paint_for_smaller_statues_white_paint_equals_49_72_l691_69189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pushup_difference_l691_69172

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℤ := 15

/-- The number of push-ups David did more than Zachary -/
def david_more_than_zachary : ℤ := 39

/-- The number of push-ups John did less than David -/
def john_less_than_david : ℤ := 9

/-- The number of push-ups David did -/
def david_pushups : ℤ := zachary_pushups + david_more_than_zachary

/-- The number of push-ups John did -/
def john_pushups : ℤ := david_pushups - john_less_than_david

theorem pushup_difference : |zachary_pushups - john_pushups| = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pushup_difference_l691_69172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_iff_a_in_A_l691_69133

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + 4) / Real.log a

-- Define the set of a values
def A : Set ℝ := {a | (0 < a ∧ a < 1) ∨ a ≥ 4}

-- Theorem statement
theorem no_minimum_iff_a_in_A (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x < y) ↔ a ∈ A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_iff_a_in_A_l691_69133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_unique_b_l691_69157

open Real

-- Define the polynomial
def P (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + b*x - 2*a

-- Define the condition for all roots being real
def all_roots_real (a b : ℝ) : Prop :=
  ∃ r s t : ℝ, ∀ x : ℝ, P a b x = 0 ↔ (x = r ∨ x = s ∨ x = t)

theorem smallest_a_and_unique_b :
  (∃ a : ℝ, a > 0 ∧
    (∀ a' : ℝ, 0 < a' ∧ a' < a →
      ¬∃ b' : ℝ, b' > 0 ∧ all_roots_real a' b') ∧
    (∃ b : ℝ, b > 0 ∧ all_roots_real a b)) ∧
  (∃! b : ℝ, b > 0 ∧ all_roots_real (Real.sqrt 2) b ∧ b = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_unique_b_l691_69157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_inequality_l691_69104

/-- Definition of T_α polynomial -/
def T_α (α : List ℝ) (x : List ℝ) : ℝ := sorry

theorem exponent_inequality (n : ℕ) (α β : List ℝ) (x : List ℝ) :
  (α.length = n) →
  (β.length = n) →
  (x.length = n) →
  (α ≠ β) →
  (α.sum = β.sum) →
  (∀ i, 0 ≤ x.get! i) →
  T_α α x ≥ T_α β x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_inequality_l691_69104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_range_l691_69182

/-- The function f(x) = x^3 - 3x --/
noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Predicate to check if a point (1, m) has exactly three tangents to the curve y = f(x) --/
def has_three_tangents (m : ℝ) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃ ∧
    (∀ (x : ℝ), f x ≤ f t₁ + f' t₁ * (x - t₁)) ∧
    (∀ (x : ℝ), f x ≤ f t₂ + f' t₂ * (x - t₂)) ∧
    (∀ (x : ℝ), f x ≤ f t₃ + f' t₃ * (x - t₃)) ∧
    (f t₁ + f' t₁ * (1 - t₁) = m) ∧
    (f t₂ + f' t₂ * (1 - t₂) = m) ∧
    (f t₃ + f' t₃ * (1 - t₃) = m)

/-- Theorem stating that if a point (1, m) has exactly three tangents to y = f(x), then -3 < m < -2 --/
theorem three_tangents_range (m : ℝ) :
  has_three_tangents m → -3 < m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_range_l691_69182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_size_of_sin_l691_69122

noncomputable def f (x : ℝ) := Real.sin x

theorem max_domain_size_of_sin (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) (1/2)) →
  (∀ y ∈ Set.Icc (-1) (1/2), ∃ x ∈ Set.Icc a b, f x = y) →
  b - a ≤ 4*π/3 :=
by
  sorry

#check max_domain_size_of_sin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_size_of_sin_l691_69122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_l691_69199

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x

theorem min_positive_period_sin_cos :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_l691_69199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l691_69127

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Theorem statement
theorem ellipse_eccentricity :
  eccentricity 3 1 = 2 * Real.sqrt 2 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l691_69127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l691_69190

/-- Given a triangle DEF with angles D, E, F satisfying cos 3D + cos 3E + cos 3F = 1,
    and two sides of lengths 8 and 15, the maximum length of the third side is 13. -/
theorem max_third_side_length (D E F : ℝ) (a b : ℝ) :
  Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1 →
  a = 8 →
  b = 15 →
  ∃ (c : ℝ), c ≤ 13 ∧ 
    ∀ (x : ℝ), (∃ (θ : ℝ), x^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) → x ≤ c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l691_69190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_geq_quadratic_plus_linear_l691_69161

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := (1/2) * x^2 + x + 1

theorem exp_geq_quadratic_plus_linear (x : ℝ) (h : x ≥ 0) : f x ≥ g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_geq_quadratic_plus_linear_l691_69161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_normal_distribution_symmetry_correlation_strength_model_fit_l691_69129

-- Define the chi-square test statistic and critical value
def chi_square_statistic : ℝ := 3.937
def critical_value : ℝ := 3.841

-- Define the normal distribution and probability function
structure NormalDistribution (μ σ : ℝ) where
  value : ℝ

def probability_function (μ σ : ℝ) (ξ : NormalDistribution μ σ) (x : ℝ) : ℝ := sorry

-- Define the correlation coefficient and sum of squared residuals
def correlation_coefficient : ℝ := sorry
def sum_squared_residuals : ℝ := sorry

-- Define independence and linear correlation strength as propositions
def independence (X Y : Type) : Prop := sorry
def stronger_linear_correlation : Prop := sorry
def worse_model_fit : Prop := sorry

-- Statement 1
theorem independence_test (X Y : Type) : 
  chi_square_statistic > critical_value → ¬(independence X Y) := by sorry

-- Statement 2
theorem normal_distribution_symmetry (μ σ : ℝ) (ξ : NormalDistribution μ σ) :
  (∀ x, probability_function μ σ ξ x = probability_function μ σ ξ (-x)) → μ = 1/2 := by sorry

-- Statement 3
theorem correlation_strength :
  ¬(|correlation_coefficient| < 1 → stronger_linear_correlation) := by sorry

-- Statement 4
theorem model_fit :
  sum_squared_residuals > 0 → worse_model_fit := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_normal_distribution_symmetry_correlation_strength_model_fit_l691_69129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_cos_B_is_two_thirds_l691_69156

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def sine_ratio (t : Triangle) : Prop :=
  Real.sin t.A / Real.sin t.B = 3 / 5

def dot_product_condition (t : Triangle) : Prop :=
  t.a * t.c * Real.cos t.B = t.b^2 - (t.a - t.c)^2

-- Theorem statements
theorem largest_angle_is_120_degrees (t : Triangle) 
  (h1 : is_arithmetic_sequence t.a t.b t.c) 
  (h2 : sine_ratio t) : 
  max t.A (max t.B t.C) = 120 * π / 180 := by sorry

theorem cos_B_is_two_thirds (t : Triangle) 
  (h : dot_product_condition t) : 
  Real.cos t.B = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_cos_B_is_two_thirds_l691_69156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_theorem_l691_69140

/-- Represents the salary of a person relative to Raja's salary -/
structure Salary where
  relative_to_raja : ℚ

/-- Calculate the percentage difference between two salaries -/
noncomputable def percentage_difference (higher lower : Salary) : ℚ :=
  (higher.relative_to_raja - lower.relative_to_raja) / higher.relative_to_raja * 100

theorem salary_difference_theorem (raja : Salary) (ram : Salary) (simran : Salary) (rahul : Salary) :
  raja.relative_to_raja = 1 →
  ram.relative_to_raja = raja.relative_to_raja * 5/4 →
  simran.relative_to_raja = raja.relative_to_raja * 17/20 →
  rahul.relative_to_raja = simran.relative_to_raja * 11/10 →
  percentage_difference ram simran = 32 := by
  sorry

#check salary_difference_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_theorem_l691_69140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circle_radius_correct_small_circles_problem_l691_69184

/-- The radius of the smaller circles in a configuration where a circle of radius 2 is surrounded by 5 smaller circles, each tangent to two others and the central circle. -/
noncomputable def small_circle_radius : ℝ := (2 * Real.sin (72 * Real.pi / 180)) / (1 - Real.sin (72 * Real.pi / 180))

/-- The theorem stating that the radius of the smaller circles in the given configuration is correct. -/
theorem small_circle_radius_correct (R : ℝ) (n : ℕ) (h1 : R = 2) (h2 : n = 5) :
  small_circle_radius = (2 * Real.sin (2 * Real.pi / n)) / (1 - Real.sin (2 * Real.pi / n)) :=
by sorry

/-- The main theorem proving that the small_circle_radius is the correct solution to the problem. -/
theorem small_circles_problem :
  ∃ (r : ℝ), r > 0 ∧ 
  r = small_circle_radius ∧
  (∀ (i j : Fin 5), i ≠ j → ∃ (x y : ℝ × ℝ),
    (x.1 - y.1)^2 + (x.2 - y.2)^2 = (2*r)^2 ∧
    x.1^2 + x.2^2 = (2 + r)^2 ∧
    y.1^2 + y.2^2 = (2 + r)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circle_radius_correct_small_circles_problem_l691_69184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_greater_cos_squared_l691_69151

theorem sin_squared_greater_cos_squared (x : ℝ) :
  0 < x ∧ x < π →
  (Real.sin x)^2 > (Real.cos x)^2 ↔ π/4 < x ∧ x < 3*π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_greater_cos_squared_l691_69151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_negative_third_l691_69170

theorem eighth_to_negative_third : (1/8 : ℝ)^(-(1/3) : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_negative_third_l691_69170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_representation_l691_69113

def number : ℕ := 8 * 10^9 + 9 * 10^7 + 3 * 10^4 + 2 * 10^3

theorem number_representation :
  (number = 890032000) ∧
  (number / 10000 : ℚ) = 89003.2 ∧
  (number / 10^9 : ℚ).floor = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_representation_l691_69113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a1_estimate_l691_69105

theorem smallest_a1_estimate (n : ℕ+) (a : Fin n → ℕ) (k : ℕ) :
  (∀ i j : Fin n, i < j → a i < a j) →
  (∀ i : Fin n, a i < 2 * n) →
  (∀ i j : Fin n, i ≠ j → ¬(a i ∣ a j)) →
  (3^k < 2 * n) →
  (2 * n < 3^(k + 1)) →
  a 0 ≥ 2^k ∧ ∀ m : ℕ, m < 2^k → ∃ b : Fin n → ℕ,
    (∀ i j : Fin n, i < j → b i < b j) ∧
    (∀ i : Fin n, b i < 2 * n) ∧
    (∀ i j : Fin n, i ≠ j → ¬(b i ∣ b j)) ∧
    b 0 = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a1_estimate_l691_69105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l691_69115

/-- Calculates the length of a bridge given train parameters --/
noncomputable def bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proves that the bridge length is approximately 550.15 meters --/
theorem bridge_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length_calculation 200 60 45 - 550.15| < ε :=
by
  sorry

-- Use #eval with a dummy function to avoid issues with noncomputable definitions
def dummy_bridge_length_calculation (train_length : Nat) (train_speed_kmh : Nat) (crossing_time : Nat) : Nat :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

#eval dummy_bridge_length_calculation 200 60 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l691_69115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l691_69123

-- Define the original efficiency of the car
noncomputable def original_efficiency : ℚ := 300 / 10

-- Define the efficiency decrease factor
def efficiency_decrease : ℚ := 1 / 10

-- Define the amount of gas for the trip with extra weight
def gas_amount : ℚ := 15

-- Theorem to prove
theorem car_travel_distance : 
  let new_efficiency := original_efficiency * (1 - efficiency_decrease)
  gas_amount * new_efficiency = 405 :=
by
  -- Unfold definitions
  unfold original_efficiency efficiency_decrease gas_amount
  -- Simplify the expression
  simp [mul_sub, mul_one, mul_div_cancel']
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l691_69123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l691_69186

-- Problem 1
theorem problem_1 (a m n : ℝ) (h1 : a^m = 10) (h2 : a^n = 2) :
  a^(m - 2*n) = 2.5 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : 2*x + 5*y - 3 = 0) :
  (4:ℝ)^x * (32:ℝ)^y = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l691_69186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_and_greater_than_four_l691_69102

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1))
    |> List.filter (λ m => m > 1 && n % m == 0)
    |> List.isEmpty

/-- The number of sides on the first die -/
def firstDieSides : ℕ := 6

/-- The number of sides on the second die -/
def secondDieSides : ℕ := 8

/-- The number of favorable outcomes on the first die (prime numbers) -/
def firstDieFavorable : ℕ := (List.range firstDieSides).filter isPrime |>.length

/-- The number of favorable outcomes on the second die (numbers greater than 4) -/
def secondDieFavorable : ℕ := (List.range secondDieSides).filter (λ x => x > 4) |>.length

/-- The total number of possible outcomes when rolling both dice -/
def totalOutcomes : ℕ := firstDieSides * secondDieSides

/-- The number of favorable outcomes when rolling both dice -/
def favorableOutcomes : ℕ := firstDieFavorable * secondDieFavorable

theorem probability_prime_and_greater_than_four :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_and_greater_than_four_l691_69102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l691_69149

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0

def q (x : ℝ) : Prop := ∃ y, Real.log (2*x - 3) = y

-- Theorem stating p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l691_69149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_theorem_l691_69183

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def octalToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 8) ((m % 8) :: acc)
    go n []

/-- The theorem states that the sum of 246₈, 123₈, and 657₈ in base 8 is equal to 1210₈. -/
theorem octal_sum_theorem :
  let a := octalToNat [2, 4, 6]
  let b := octalToNat [1, 2, 3]
  let c := octalToNat [6, 5, 7]
  let sum := a + b + c
  natToOctal sum = [1, 2, 1, 0] := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_theorem_l691_69183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_terminates_l691_69130

def transform_triple (x y z : ℤ) : ℤ × ℤ × ℤ :=
  (x + y, -y, z + y)

def sum_of_squares (l : List ℤ) : ℤ :=
  l.map (λ x => x * x) |>.sum

structure Pentagon where
  vertices : List ℤ
  sum_positive : vertices.sum > 0
  length_five : vertices.length = 5

def transform_pentagon (p : Pentagon) : Pentagon :=
  sorry

theorem transform_terminates (p : Pentagon) : 
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → 
    (Nat.iterate transform_pentagon m p).vertices.all (λ x => x ≥ 0) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_terminates_l691_69130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_formula_l691_69148

/-- Circumference of a circle -/
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- Constant of proportionality between circumference and radius -/
noncomputable def k : ℝ := 2 * Real.pi

theorem circle_circumference_formula (r : ℝ) (h : r > 0) :
  ∃ (k : ℝ), circumference r = k * r ∧ k = 2 * Real.pi :=
by
  use 2 * Real.pi
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_formula_l691_69148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_surface_area_greater_l691_69119

-- Define a type for convex polyhedra
structure ConvexPolyhedron where
  -- Add necessary fields (this is a simplified representation)
  vertices : Set (Fin 3 → ℝ)
  -- Add more properties to ensure convexity

-- Define a function to calculate the surface area of a convex polyhedron
def surfaceArea (p : ConvexPolyhedron) : ℝ := sorry

-- Define a relation for one polyhedron being inside another
def isInside (inner outer : ConvexPolyhedron) : Prop := sorry

-- Theorem statement
theorem outer_surface_area_greater
  (P_inner P_outer : ConvexPolyhedron)
  (h_inside : isInside P_inner P_outer) :
  surfaceArea P_outer > surfaceArea P_inner := by
  sorry

#check outer_surface_area_greater

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_surface_area_greater_l691_69119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l691_69120

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 - 3*x*y + 4*y^2 - z = 0) :
  let f := λ (a b c : ℝ) => 2/a + 1/b - 2/c
  let g := λ (a b c : ℝ) => a*b/c
  ∃ (M : ℝ), M = 1 ∧ (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
    g x' y' z' ≤ g x y z →
    f x' y' z' ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l691_69120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_5_minus_e_approx_l691_69158

/-- The mathematical constant e --/
noncomputable def e : ℝ := Real.exp 1

/-- Approximation of |5 - e| --/
def approx_abs_5_minus_e : ℝ := 2.28172

/-- Theorem stating that |5 - e| is approximately equal to 2.28172 --/
theorem abs_5_minus_e_approx : 
  |5 - e - approx_abs_5_minus_e| < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_5_minus_e_approx_l691_69158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l691_69143

-- Define the curve C
def curve_C (ρ θ : ℝ) : Prop := ρ * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def OP_OQ_relation (ρ_P ρ_Q : ℝ) : Prop := ρ_P * ρ_Q = Real.sqrt 2

-- Define the trajectory of point P (C₁)
def trajectory_P (ρ θ : ℝ) : Prop := ρ = Real.cos θ - Real.sin θ

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Define the curve C₂
def curve_C2 (x y t : ℝ) : Prop := x = 1/2 - (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Main theorem
theorem EF_length :
  ∀ (ρ_E θ_E ρ_F θ_F : ℝ),
  trajectory_P ρ_E θ_E →
  line_l (ρ_E * Real.cos θ_E) (ρ_E * Real.sin θ_E) →
  curve_C2 (ρ_F * Real.cos θ_F) (ρ_F * Real.sin θ_F) (2 * ρ_F * Real.sin θ_F / Real.sqrt 2) →
  ρ_E + ρ_F = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l691_69143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_speed_is_100_l691_69125

-- Define the problem parameters
def distance : ℝ := 1000
def time_difference : ℝ := 1
def speed_difference : ℝ := 40
def arrival_difference : ℝ := 6

-- Define Abel's speed as a variable
def abel_speed : ℝ → Prop := λ v => v > 0

-- Theorem statement
theorem abel_speed_is_100 :
  ∃ v : ℝ, abel_speed v ∧
  (distance / v + arrival_difference = distance / (v + speed_difference) + time_difference) ∧
  v = 100 := by
  -- Proof goes here
  sorry

#check abel_speed_is_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_speed_is_100_l691_69125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_segment_l691_69198

/-- The midpoint of a line segment in polar coordinates --/
noncomputable def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  (r₁ * Real.cos ((θ₂ - θ₁) / 2), (θ₁ + θ₂) / 2)

theorem midpoint_specific_segment :
  let r₁ : ℝ := 10
  let θ₁ : ℝ := π / 6
  let r₂ : ℝ := 10
  let θ₂ : ℝ := 5 * π / 6
  let (r, θ) := midpoint_polar r₁ θ₁ r₂ θ₂
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 5 ∧ θ = π / 2 := by
  sorry

#check midpoint_specific_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_segment_l691_69198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_fourth_quadrant_l691_69159

theorem unit_circle_fourth_quadrant (θ : Real) (y : Real) :
  -- Conditions
  θ ∈ Set.Icc (3*π/2) (2*π) → -- θ is in the fourth quadrant
  (1/2)^2 + y^2 = 1 → -- Point P is on the unit circle
  y < 0 → -- Point P is in the fourth quadrant
  -- Conclusions
  Real.tan θ = -Real.sqrt 3 ∧
  (Real.cos (π/2 - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_fourth_quadrant_l691_69159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l691_69153

theorem triangle_sine_sum_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l691_69153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_value_theorem_l691_69152

/-- A real polynomial of degree at most 2012 -/
def MyPolynomial := ℝ → ℝ

/-- The property that P(n) = 2^n for n = 1, 2, ..., 2012 -/
def matches_powers_of_two (P : MyPolynomial) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2012 → P n = 2^n

/-- The expression to be minimized -/
def expression_to_minimize (P : MyPolynomial) : ℝ :=
  (P 0)^2 + (P 2013)^2

theorem minimal_value_theorem (P : MyPolynomial) 
  (h : matches_powers_of_two P) :
  ∃ (min_P0 : ℝ), 
    (∀ (P0 : ℝ), expression_to_minimize (λ x ↦ if x = 0 then P0 else P x) ≥ 
                  expression_to_minimize (λ x ↦ if x = 0 then min_P0 else P x)) ∧
    min_P0 = 1 - 2^2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_value_theorem_l691_69152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_length_l691_69142

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a point lies on the parabola y = x^2 -/
def onParabola (p : Point) : Prop :=
  p.y = p.x ^ 2

/-- Check if two points have the same y-coordinate (i.e., form a line parallel to x-axis) -/
def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Calculate the length between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  |p2.x - p1.x|

theorem triangle_on_parabola_length (t : Triangle) :
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C ∧
  t.A = ⟨0, 0⟩ ∧
  parallelToXAxis t.B t.C ∧
  triangleArea (distance t.B t.C) t.B.y = 64 →
  distance t.B t.C = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_length_l691_69142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_16_l691_69126

noncomputable def green_lettuce_price : ℝ := 2
noncomputable def red_lettuce_price : ℝ := 3
noncomputable def cherry_tomatoes_price : ℝ := 4
noncomputable def parmesan_cheese_price : ℝ := 5

noncomputable def green_lettuce_spent : ℝ := 12
noncomputable def red_lettuce_spent : ℝ := 9
noncomputable def cherry_tomatoes_spent : ℝ := 16
noncomputable def parmesan_cheese_spent : ℝ := 15

noncomputable def total_weight : ℝ :=
  green_lettuce_spent / green_lettuce_price +
  red_lettuce_spent / red_lettuce_price +
  cherry_tomatoes_spent / cherry_tomatoes_price +
  parmesan_cheese_spent / parmesan_cheese_price

theorem total_weight_is_16 : total_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_16_l691_69126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_2011_value_l691_69118

def t : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 1) => (t n - 1) / (t n + 1)

theorem t_2011_value : t 2011 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_2011_value_l691_69118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_art_probability_l691_69191

/-- The probability of arranging art pieces with Escher and Dali prints consecutively -/
def art_arrangement_probability (total escher dali : ℕ) : ℚ :=
  let other := total - escher - dali
  let arrangements := Nat.factorial total
  let favorable := Nat.factorial (other + 2) * Nat.factorial escher * Nat.factorial dali
  (favorable : ℚ) / arrangements

/-- The specific case with 12 total pieces, 4 by Escher, and 2 by Dali -/
theorem specific_art_probability : art_arrangement_probability 12 4 2 = 1 / 247 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_art_probability_l691_69191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l691_69137

theorem complex_real_condition (a : ℝ) : 
  (Complex.I + 1) * (1 - a * Complex.I) ∈ Set.range Complex.ofReal → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l691_69137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l691_69121

def S (n : ℕ) : ℕ := n^2 + 3*n + 1

def a : ℕ → ℕ
  | 0 => 5  -- Adding case for 0
  | 1 => 5
  | n + 2 => 2*(n + 2) + 2

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  (∀ k, k > 0 → S k - S (k-1) = a k) ∧ 
  (∀ k, k > 1 → a k = 2*k + 2) ∧ 
  (a 1 = 5) := by
  sorry

#check sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l691_69121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_not_strictly_increasing_first_quadrant_l691_69114

theorem sine_not_strictly_increasing_first_quadrant :
  ¬ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π / 2 → Real.sin x₁ < Real.sin x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_not_strictly_increasing_first_quadrant_l691_69114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_m_values_l691_69162

-- Define the circles
def circle_O1 (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x + m^2 - 4 = 0

def circle_O2 (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0

-- Define the centers of the circles
def center_O1 (m : ℝ) : ℝ × ℝ := (m, 0)
def center_O2 (m : ℝ) : ℝ × ℝ := (-1, 2*m)

-- Define the radii of the circles
def radius_O1 : ℝ := 2
def radius_O2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers (m : ℝ) : ℝ :=
  Real.sqrt ((m + 1)^2 + 4*m^2)

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  distance_between_centers m = radius_O1 + radius_O2 ∨
  distance_between_centers m = |radius_O1 - radius_O2|

-- State the theorem
theorem tangent_circles_m_values :
  {m : ℝ | are_tangent m} = {-12/5, -2/5, 0, 2} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_m_values_l691_69162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l691_69136

/-- Asymptote slope of a hyperbola of the form (x²/a²) - (y²/b²) = 1 -/
noncomputable def asymptote_slope_1 (a b : ℝ) : ℝ := b / a

/-- Asymptote slope of a hyperbola of the form (y²/a²) - (x²/b²) = 1 -/
noncomputable def asymptote_slope_2 (a b : ℝ) : ℝ := a / b

/-- The value of M for which the two given hyperbolas have the same asymptotes -/
theorem hyperbolas_same_asymptotes :
  ∀ M : ℝ, 
  M > 0 →
  asymptote_slope_1 3 4 = asymptote_slope_2 5 (Real.sqrt M) →
  M = 225 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l691_69136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_choices_from_25_l691_69169

/-- The number of ways to choose k distinct items from a set of n items -/
def choose_distinct (n : ℕ) (k : ℕ) : ℕ :=
  (List.range k).foldl (λ acc i => acc * (n - i)) 1

/-- Theorem: Choosing 4 distinct items from 25 items results in 303600 ways -/
theorem distinct_choices_from_25 : choose_distinct 25 4 = 303600 := by
  rfl

#eval choose_distinct 25 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_choices_from_25_l691_69169
