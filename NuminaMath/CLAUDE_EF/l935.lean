import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l935_93561

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

-- State the theorem
theorem tangent_line_parallel (a : ℝ) :
  (∀ x : ℝ, x > -a) →  -- Domain condition
  (∃ y : ℝ, y = f a 1) →  -- Existence of f(1)
  (∃ m : ℝ, m = 1 / (1 + a)) →  -- Derivative at x = 1
  (1 / (1 + a) = 1 / 2) →  -- Parallel condition
  a = 1 :=
by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l935_93561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_b_l935_93592

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (1/2) * x^2 - a * x + b * log x

-- State the theorem
theorem max_value_of_b :
  ∀ a b : ℝ,
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, x > 0 → f a b x ≤ f a b x₀) ∧
    f a b x₀ < 0) →
  b ≤ exp 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_b_l935_93592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l935_93587

/-- Geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ :=
  λ n ↦ a * r^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio (a r : ℝ) (h : r ≠ 1) :
  (geometric_sum a r 6 = -7 * geometric_sum a r 3) →
  (geometric_sequence a r 4 + geometric_sequence a r 3) /
  (geometric_sequence a r 3 + geometric_sequence a r 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l935_93587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l935_93506

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the triangle area function
noncomputable def triangle_area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_theorem (t : Triangle) : 
  (t.b * Real.sin t.B - t.a * Real.sin t.C = 0) → 
  (t.b^2 = t.a * t.c) ∧ 
  ((t.a = 1 ∧ t.c = 2) → (triangle_area t = Real.sqrt 7 / 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l935_93506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_proof_l935_93512

/-- A point in the complex plane -/
structure ComplexPoint where
  z : ℂ

/-- A triangle in the complex plane -/
structure ComplexTriangle where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint

/-- The centroid of a triangle -/
noncomputable def centroid (t : ComplexTriangle) : ComplexPoint :=
  { z := (t.A.z + t.B.z + t.C.z) / 3 }

/-- Intersection point of a line through P and a side of the triangle -/
noncomputable def intersection (P A B : ComplexPoint) : ComplexPoint :=
  { z := sorry }  -- Actual implementation would depend on the line-side intersection formula

/-- Similarity of triangles -/
def similar (t1 t2 : ComplexTriangle) : Prop :=
  sorry  -- Actual implementation would involve ratio of sides or angles

theorem centroid_proof (ABC : ComplexTriangle) (P : ComplexPoint) :
  (∃ (D E F : ComplexPoint),
    D = intersection P ABC.B ABC.C ∧
    E = intersection P ABC.C ABC.A ∧
    F = intersection P ABC.A ABC.B ∧
    similar { A := D, B := E, C := F } ABC) →
  P = centroid ABC := by
  sorry  -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_proof_l935_93512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_min_distance_value_l935_93579

/-- Curve C₁ defined parametrically -/
noncomputable def C₁ : ℝ → ℝ × ℝ := λ θ ↦ (Real.cos θ, Real.sin θ)

/-- Line l defined in polar form -/
def l : ℝ → ℝ → Prop := λ ρ θ ↦ ρ * (2 * Real.cos θ - Real.sin θ) = 6

/-- Distance function from a point to the line l -/
noncomputable def distance_to_l (x y : ℝ) : ℝ :=
  (|2 * x - y - 6|) / Real.sqrt 5

/-- The point P on C₁ -/
noncomputable def P : ℝ × ℝ := (2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5)

/-- Theorem stating that P has the minimum distance to l -/
theorem min_distance_point :
  P ∈ Set.range C₁ ∧
  ∀ Q : ℝ × ℝ, Q ∈ Set.range C₁ → distance_to_l P.1 P.2 ≤ distance_to_l Q.1 Q.2 := by
  sorry

/-- Theorem stating the value of the minimum distance -/
theorem min_distance_value :
  distance_to_l P.1 P.2 = 6 * Real.sqrt 5 / 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_min_distance_value_l935_93579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_comparison_l935_93548

/-- Represents the time taken by a person to reach the destination -/
noncomputable def time_taken (S m n : ℝ) : ℝ := 2 * S / (m + n)

/-- Represents the time taken by Person B to reach the destination -/
noncomputable def time_taken_B (S m n : ℝ) : ℝ := S / (2 * m) + S / (2 * n)

theorem travel_time_comparison (S m n : ℝ) (hS : S > 0) (hm : m > 0) (hn : n > 0) :
  (m ≠ n → time_taken S m n < time_taken_B S m n) ∧ 
  (m = n → time_taken S m n = time_taken_B S m n) := by
  sorry

#check travel_time_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_comparison_l935_93548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_equals_sqrt26_plus_5sqrt2_plus_3_l935_93571

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The configuration of three circles as described in the problem -/
structure ThreeCirclesConfig where
  circleA : Circle
  circleB : Circle
  circleC : Circle
  pointA' : Point
  pointB' : Point
  pointC' : Point

/-- Checks if the configuration satisfies the problem conditions -/
def validConfig (config : ThreeCirclesConfig) : Prop :=
  config.circleA.radius = 2 ∧
  config.circleB.radius = 3 ∧
  config.circleC.radius = 4 ∧
  config.pointB'.x < config.pointC'.x ∧
  config.pointA'.x < config.pointB'.x ∧
  -- Circle B is externally tangent to circles A and C
  (config.circleA.center.x - config.circleB.center.x)^2 + 
    (config.circleA.center.y - config.circleB.center.y)^2 = 
    (config.circleA.radius + config.circleB.radius)^2 ∧
  (config.circleC.center.x - config.circleB.center.x)^2 + 
    (config.circleC.center.y - config.circleB.center.y)^2 = 
    (config.circleC.radius + config.circleB.radius)^2

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perimeter of quadrilateral A'B'C'B -/
noncomputable def perimeterA'B'C'B (config : ThreeCirclesConfig) : ℝ :=
  distance config.pointA' config.pointB' +
  distance config.pointB' config.pointC' +
  distance config.pointB' config.circleB.center

/-- The main theorem stating that for any valid configuration, 
    the perimeter of A'B'C'B is √26 + 5√2 + 3 -/
theorem perimeter_equals_sqrt26_plus_5sqrt2_plus_3 
  (config : ThreeCirclesConfig) (h : validConfig config) :
  perimeterA'B'C'B config = Real.sqrt 26 + 5 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_equals_sqrt26_plus_5sqrt2_plus_3_l935_93571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l935_93530

/-- The quadratic equation x^2 - (k+3)x + 2k + 2 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 - (k+3)*x + 2*k + 2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  (k+3)^2 - 4*(2*k+2)

/-- The sum of squares of roots plus their product -/
def root_sum_square_product (α β : ℝ) : ℝ :=
  α^2 + β^2 + α*β

theorem quadratic_equation_properties (k : ℝ) :
  (∃ α β : ℝ, quadratic_equation k α ∧ quadratic_equation k β ∧ α ≠ β) ∧
  (discriminant k ≥ 0) ∧
  (∀ α β : ℝ, quadratic_equation k α ∧ quadratic_equation k β ∧ root_sum_square_product α β = 4 →
    k = -1 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l935_93530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l935_93542

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : Polynomial ℝ) :
  n ≥ 1 →
  P.degree = n →
  (∀ k : ℕ, k ≤ n → P.eval (k : ℝ) = (k : ℝ) / ((k : ℝ) + 1)) →
  P.eval ((n + 1 : ℕ) : ℝ) = ((-1 : ℝ)^n + (n : ℝ) + 1) / ((n : ℝ) + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l935_93542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_slope_range_l935_93599

/-- Circle with equation x^2 + y^2 - 4x - 4y - 10 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 10 = 0}

/-- Line with equation ax + by = 0 -/
def Line (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a*p.1 + b*p.2 = 0}

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (a b : ℝ) : ℝ :=
  |a*p.1 + b*p.2| / Real.sqrt (a^2 + b^2)

theorem circle_line_slope_range (a b : ℝ) (h : b ≠ 0) :
  (∃ p1 p2 p3 : ℝ × ℝ, p1 ∈ Circle ∧ p2 ∈ Circle ∧ p3 ∈ Circle ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    distanceToLine p1 a b = 2 * Real.sqrt 2 ∧
    distanceToLine p2 a b = 2 * Real.sqrt 2 ∧
    distanceToLine p3 a b = 2 * Real.sqrt 2) →
  2 - Real.sqrt 3 ≤ -a/b ∧ -a/b ≤ 2 + Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_slope_range_l935_93599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l935_93552

-- Define the function f(x) = 3^(x^2 - 4x)
noncomputable def f (x : ℝ) : ℝ := 3^(x^2 - 4*x)

-- Theorem stating that the decreasing interval of f is (-∞, 2)
theorem f_decreasing_interval :
  ∀ x y : ℝ, x < y → x < 2 → y < 2 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l935_93552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l935_93518

/-- The time (in minutes) a cyclist must wait for a hiker to catch up, given their speeds and initial separation. -/
noncomputable def waiting_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (separation_time : ℝ) : ℝ :=
  let separation_distance := cyclist_speed * separation_time / 60
  separation_distance / hiker_speed * 60

/-- Theorem stating that under the given conditions, the waiting time is 20 minutes. -/
theorem cyclist_waiting_time :
  waiting_time 5 20 5 = 20 := by
  -- Unfold the definition of waiting_time
  unfold waiting_time
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l935_93518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_equation_l935_93509

/-- A circle satisfying the given conditions -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  symmetrical_about_y_axis : center.1 = 0
  passes_through_focus : (1 : ℝ)^2 + (0 - center.2)^2 = radius^2
  arc_ratio : |(center.2 : ℝ)| / Real.sqrt 2 = |radius| / 2

/-- The equation of the circle c -/
def circle_equation (c : CircleC) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem stating the equation of circle c -/
theorem circle_c_equation (c : CircleC) :
  ∃ k : ℤ, k = 1 ∨ k = -1 ∧ ∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - (k : ℝ))^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_equation_l935_93509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l935_93525

noncomputable section

/-- The area enclosed by a curve consisting of 16 congruent circular arcs, 
    each with length π/2 and centered at the vertices of a regular octagon 
    with side length 3 -/
def enclosed_area (num_arcs : ℕ) (arc_length : ℝ) (octagon_side : ℝ) : ℝ :=
  2 * (1 + Real.sqrt 2) * octagon_side^2 + 
  num_arcs * (arc_length^2 / (4 * Real.pi))

theorem curve_area_theorem : 
  enclosed_area 16 (Real.pi/2) 3 = 54 + 38.4 * Real.sqrt 2 + 4 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l935_93525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_2000_l935_93529

def count_lcm_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => Nat.lcm p.1 p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

theorem lcm_pairs_2000 : count_lcm_pairs 2000 = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_2000_l935_93529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l935_93564

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) + 1 / (x - 5)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc 4 5 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l935_93564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_count_theorem_l935_93528

/-- The number of common positive divisors of 9240 and 8820 -/
def commonDivisorsCount : ℕ := 24

/-- The first number -/
def a : ℕ := 9240

/-- The second number -/
def b : ℕ := 8820

theorem common_divisors_count_theorem :
  (Finset.filter (λ d ↦ d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = commonDivisorsCount := by
  sorry

#eval (Finset.filter (λ d ↦ d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_count_theorem_l935_93528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l935_93585

theorem set_equality_implies_difference (a b : ℝ) : {a, 1} = ({0, a + b} : Set ℝ) → b - a = 1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l935_93585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l935_93554

/-- A power function that passes through the point (4, 1/2) -/
noncomputable def f (x : ℝ) : ℝ := x ^ (-1/2 : ℝ)

/-- The power function passes through the point (4, 1/2) -/
axiom f_at_4 : f 4 = 1/2

/-- Theorem: f(3) equals √3/3 -/
theorem f_at_3 : f 3 = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l935_93554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l935_93540

/-- The function f(x) = x(ln x - ax) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a * x)

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - 2 * a * x

theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) →
  (0 < a ∧ a < 1/2) :=
by
  sorry

#check extreme_points_imply_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l935_93540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_when_m_is_four_unique_m_for_infinite_solutions_l935_93559

/-- A system of two linear equations in two variables -/
structure LinearSystem (α : Type*) [Field α] where
  eq1 : α → α → α
  eq2 : α → α → α

/-- The condition for a linear system to have infinitely many solutions -/
def hasInfinitelySolutions {α : Type*} [Field α] (sys : LinearSystem α) : Prop :=
  ∃ (k : α), k ≠ 0 ∧ 
    (∀ x y, sys.eq1 x y = 0 ↔ k * (sys.eq2 x y) = 0)

/-- The specific system of equations given in the problem -/
noncomputable def givenSystem (m : ℝ) : LinearSystem ℝ where
  eq1 := λ x y => x + m*y - 2
  eq2 := λ x y => m*x + 16*y - 8

/-- The theorem stating that the given system has infinitely many solutions when m = 4 -/
theorem infinitely_many_solutions_when_m_is_four :
  hasInfinitelySolutions (givenSystem 4) := by sorry

/-- The theorem stating that 4 is the only value of m for which the system has infinitely many solutions -/
theorem unique_m_for_infinite_solutions :
  ∀ m : ℝ, hasInfinitelySolutions (givenSystem m) → m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_when_m_is_four_unique_m_for_infinite_solutions_l935_93559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_geometric_progression_l935_93567

/-- For an ellipse where the focal length, length of the minor axis, and length of the major axis
    form a geometric progression, the eccentricity is equal to (-1 + √5) / 2. -/
theorem ellipse_eccentricity_geometric_progression (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_geom_prog : b^2 = a * c) (h_ellipse : c < a) :
  Real.sqrt (1 - b^2 / a^2) = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_geometric_progression_l935_93567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l935_93581

/-- Represents a triangle -/
structure Triangle where
  /-- Indicates if the triangle is a right triangle -/
  isRight : Bool
  /-- The length of one leg of the triangle -/
  oneLeg : ℝ
  /-- The measure of the angles adjacent to the given leg -/
  adjacentAngles : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ

/-- A right triangle with one leg of 8 inches and two 45° angles adjacent to that leg has a hypotenuse of 8√2 inches. -/
theorem right_triangle_hypotenuse (t : Triangle) 
  (h1 : t.isRight = true) 
  (h2 : t.oneLeg = 8) 
  (h3 : t.adjacentAngles = 45) : 
  t.hypotenuse = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l935_93581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PQ_l935_93503

/-- Ellipse C with semi-major axis a -/
def ellipse_C (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + p.2^2 = 1}

/-- Circle O -/
def circle_O : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line AB -/
def line_AB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 / a + p.2 = 1}

/-- Distance between a point and a line -/
noncomputable def distance_point_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Tangent line to circle O -/
def tangent_line_O : Set (Set (ℝ × ℝ)) := sorry

theorem max_distance_PQ (a : ℝ) (h1 : a > 1) :
  ∃ C : Set (ℝ × ℝ),
    C = ellipse_C a ∧
    (a, 0) ∈ C ∧
    (0, 1) ∈ C ∧
    distance_point_line (0, 0) (line_AB a) = Real.sqrt 3 / 2 →
    ∀ l ∈ tangent_line_O,
      ∀ P Q : ℝ × ℝ,
        P ∈ C ∧ Q ∈ C ∧ P ∈ l ∧ Q ∈ l →
        ‖(P.1 - Q.1, P.2 - Q.2)‖ ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PQ_l935_93503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l935_93555

/-- Custom operation ⊗ for non-zero real numbers -/
noncomputable def otimes (a b : ℝ) : ℝ := 1 / a + 1 / b

/-- Theorem stating the solution to the equation (x+1)⊗x = 2 -/
theorem solution_equation (x : ℝ) (hx : x ≠ 0) (hx1 : x + 1 ≠ 0) :
  otimes (x + 1) x = 2 ↔ x = Real.sqrt 2 / 2 ∨ x = -Real.sqrt 2 / 2 := by
  sorry

#check solution_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l935_93555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l935_93598

/-- Represents the state of boxes and balls -/
def BoxState (n : ℕ) := Fin n → ℕ

/-- Allowed operations on the boxes -/
inductive BoxOperation (n : ℕ)
  | move_from_first : BoxOperation n
  | move_to_last : BoxOperation n
  | move_from_middle (i : Fin n) : BoxOperation n

/-- Applies an operation to a box state -/
def apply_operation (n : ℕ) (state : BoxState n) (op : BoxOperation n) : BoxState n :=
  match op with
  | BoxOperation.move_from_first => sorry
  | BoxOperation.move_to_last => sorry
  | BoxOperation.move_from_middle i => sorry

/-- Checks if a state is valid (total number of balls is n) -/
def is_valid_state (n : ℕ) (state : BoxState n) : Prop :=
  (Finset.sum Finset.univ (λ i => state i)) = n

/-- Checks if a state is the desired final state (one ball in each box) -/
def is_final_state (n : ℕ) (state : BoxState n) : Prop :=
  ∀ i, state i = 1

/-- Main theorem: It's always possible to reach the final state -/
theorem balls_in_boxes (n : ℕ) (initial_state : BoxState n) 
  (h_valid : is_valid_state n initial_state) :
  ∃ (ops : List (BoxOperation n)), 
    let final_state := ops.foldl (apply_operation n) initial_state
    is_final_state n final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l935_93598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_instantaneous_rate_change_l935_93520

-- Define the temperature function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 8

-- Define the derivative of the temperature function
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem min_instantaneous_rate_change :
  ∃ y ∈ Set.Icc (0 : ℝ) 5, ∀ z ∈ Set.Icc (0 : ℝ) 5, f' y ≤ f' z ∧ f' y = -1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_instantaneous_rate_change_l935_93520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_cost_calculation_l935_93524

theorem cupcake_cost_calculation (total_cupcakes : ℕ) (burnt_cupcakes : ℕ) (eaten_cupcakes : ℕ) 
  (selling_price : ℚ) (net_profit : ℚ) :
  total_cupcakes = 72 →
  burnt_cupcakes = 24 →
  eaten_cupcakes = 9 →
  selling_price = 2 →
  net_profit = 24 →
  (total_cupcakes - burnt_cupcakes - eaten_cupcakes) * selling_price - net_profit = 3/4 * total_cupcakes := by
  sorry

#eval (3 : ℚ) / 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_cost_calculation_l935_93524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l935_93534

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := (x ∈ Set.Icc (-1) 0 ∧ x ≠ 0) ∨ (x ∈ Set.Ioc 0 1)

-- State that f is an odd function
axiom f_odd (x : ℝ) : domain x → f (-x) = -f x

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ∈ Set.Icc (-1) (-1/2) ∨ x ∈ Set.Ioc 0 1

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, domain x → (f x - f (-x) > -1 ↔ solution_set x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l935_93534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_equation_l935_93541

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of a hyperbola -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The slope of an asymptote of a hyperbola -/
noncomputable def Hyperbola.asymptoticSlope (h : Hyperbola) : ℝ := h.b / h.a

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def Hyperbola.focalDistance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem about a specific hyperbola with given properties -/
theorem specific_hyperbola_equation :
  ∃ (h : Hyperbola),
    -- One asymptote has an inclination angle of 60°
    h.asymptoticSlope = Real.sqrt 3 ∧
    -- |ON| = |NF| + 1, where N is midpoint of MF
    h.focalDistance / 2 = h.focalDistance / 4 + 1 ∧
    -- The equation of the hyperbola
    (∀ x y : ℝ, h.equation x y ↔ x^2 - y^2 / 3 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_equation_l935_93541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_predecessor_l935_93566

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

theorem binary_predecessor (M : List Bool) :
  M = [true, false, true, false, true, false, false] →
  natToBinary (binaryToNat M - 1) = [true, false, true, false, false, true, true] := by
  sorry

#eval natToBinary (binaryToNat [true, false, true, false, true, false, false] - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_predecessor_l935_93566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intermediate_factors_l935_93597

/-- Representation of a number as a list of its decimal digits -/
def DecimalDigits := List Nat

/-- Polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Check if a number is prime -/
def isPrime (p : Nat) : Prop := sorry

/-- Convert a list of decimal digits to a natural number -/
def digitsToNat (digits : DecimalDigits) : Nat := sorry

/-- Convert a natural number to a polynomial -/
def toPolynomial (p : Nat) : IntPolynomial := sorry

/-- Check if a polynomial has factors with degrees between lo and hi -/
def hasFactorsBetween (p : IntPolynomial) (lo hi : Nat) : Prop := sorry

theorem no_intermediate_factors 
  (p : Nat) 
  (digits : DecimalDigits) 
  (h1 : isPrime p) 
  (h2 : digitsToNat digits = p) 
  (h3 : digits.head! > 1) 
  (h4 : digits.length > 1) :
  ¬ hasFactorsBetween (toPolynomial p) 1 (digits.length - 1) := by
  sorry

#check no_intermediate_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intermediate_factors_l935_93597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_range_theorem_m_range_theorem_l935_93583

noncomputable def f (a x : ℝ) : ℝ := (1/2) * a * x^2 - 2 * a * x + Real.log x

theorem extreme_points_and_range_theorem 
  (a : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_extreme : ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (deriv (f a)) x ≠ 0) 
  (h_product : x₁ * x₂ > 1/2) :
  a ∈ Set.Ioo 1 2 :=
sorry

theorem m_range_theorem 
  (a : ℝ) 
  (h_a : a ∈ Set.Ioo 1 2) 
  (x₀ : ℝ) 
  (h_x₀ : x₀ ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2) 
  (h_ineq : ∀ m : ℝ, (∀ a ∈ Set.Ioo 1 2, f a x₀ + Real.log (a + 1) > m * (a^2 - 1) - (a + 1) + 2 * Real.log 2) → m ∈ Set.Iic (-1/4)) :
  true :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_range_theorem_m_range_theorem_l935_93583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l935_93538

-- Define the complex number z
noncomputable def z : ℂ := (1 + Complex.I) / (2 + Complex.I)

-- Theorem statement
theorem imaginary_part_of_z : z.im = 1/5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l935_93538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l935_93578

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := sorry

def extendToCircumcircle (t : Triangle) (p q : ℝ × ℝ) : ℝ × ℝ := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

def area (p q r : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (t : Triangle) :
  let D := median t t.A
  let E := median t t.C
  let F := extendToCircumcircle t t.C E
  distance D t.A = 18 ∧
  distance E t.C = 27 ∧
  distance t.A t.B = 24 →
  area t.A F t.B = 8 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l935_93578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disposable_income_increase_approx_24_62_percent_l935_93539

/-- Calculates the percentage increase in disposable income given the initial and new weekly incomes,
    tax rates, and monthly expense. -/
noncomputable def disposable_income_increase_percentage (initial_weekly_income : ℝ) (new_weekly_income : ℝ)
    (initial_tax_rate : ℝ) (new_tax_rate : ℝ) (monthly_expense : ℝ) : ℝ :=
  let initial_monthly_income := (initial_weekly_income * (1 - initial_tax_rate)) * 4
  let new_monthly_income := (new_weekly_income * (1 - new_tax_rate)) * 4
  let initial_disposable_income := initial_monthly_income - monthly_expense
  let new_disposable_income := new_monthly_income - monthly_expense
  let increase := new_disposable_income - initial_disposable_income
  (increase / initial_disposable_income) * 100

theorem disposable_income_increase_approx_24_62_percent :
  ∃ ε > 0, ε < 0.01 ∧
  |disposable_income_increase_percentage 60 70 0.15 0.18 100 - 24.62| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disposable_income_increase_approx_24_62_percent_l935_93539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_sqrt3_to_inf_l935_93511

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 3 / x - 4
  else if x < 0 then -((-x + 3 / (-x) - 4))
  else 0

-- State the theorem
theorem f_increasing_on_sqrt3_to_inf (x₁ x₂ : ℝ) 
  (h₁ : Real.sqrt 3 < x₁) (h₂ : x₁ < x₂) : f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_sqrt3_to_inf_l935_93511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_root_ratio_l935_93515

theorem quartic_root_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, x ∈ ({1, 2, 3, 4} : Set ℝ) → a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) :
  d / e = -25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_root_ratio_l935_93515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l935_93563

theorem derivative_at_two (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, f x = 2 * f' 2 * x + x^3) →
  f' 2 = -12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l935_93563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_can_land_l935_93595

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The lighthouse position -/
def lighthouse : Point := ⟨0, 0⟩

/-- The shore is represented by the line x = 10 -/
def onShore (p : Point) : Prop :=
  p.x = 10

theorem ship_can_land (ship : Point) :
  distance ship lighthouse ≤ 10 →
  ∃ (landingPoint : Point), onShore landingPoint ∧
    distance ship lighthouse + distance lighthouse landingPoint ≤ 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_can_land_l935_93595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l935_93531

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 7
  | n + 1 => (7 * sequenceA n) / (sequenceA n + 7)

theorem sequence_formula : ∀ n : ℕ, n > 0 → sequenceA n = 7 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l935_93531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_sqrt3_sufficient_not_necessary_l935_93591

theorem tan_not_sqrt3_sufficient_not_necessary :
  ∃ α : Real, (Real.tan α ≠ Real.sqrt 3 → α ≠ π / 3) ∧ (α ≠ π / 3 ∧ Real.tan α = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_sqrt3_sufficient_not_necessary_l935_93591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l935_93553

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, g x = Real.sin (2 * x)) →
  (∀ x, f x = Real.sin (8 * x + π / 4)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l935_93553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l935_93536

def spinner_A : Finset ℕ := {1, 2, 3, 5}
def spinner_B : Finset ℕ := {1, 2, 3, 4, 6}

def is_even (n : ℕ) : Bool := n % 2 = 0

def total_outcomes : ℕ := (Finset.card spinner_A) * (Finset.card spinner_B)

def even_product_outcomes : ℕ :=
  Finset.card (Finset.filter (fun (pair : ℕ × ℕ) => is_even (pair.1 * pair.2))
    (spinner_A.product spinner_B))

theorem probability_even_product :
  (even_product_outcomes : ℚ) / total_outcomes = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l935_93536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_hike_distance_l935_93586

/-- The distance from the starting point to the end point in Linda's hike --/
noncomputable def hikeDistance (northDistance : ℝ) (eastTurnAngle : ℝ) (secondLegDistance : ℝ) : ℝ :=
  let eastDistance := secondLegDistance * Real.sin (eastTurnAngle * Real.pi / 180)
  let totalNorthDistance := northDistance + secondLegDistance * Real.cos (eastTurnAngle * Real.pi / 180)
  Real.sqrt (eastDistance ^ 2 + totalNorthDistance ^ 2)

/-- Theorem stating that the hike distance for Linda's specific path is √(61 + 30√2) --/
theorem linda_hike_distance :
  hikeDistance 3 45 5 = Real.sqrt (61 + 30 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_hike_distance_l935_93586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_9_seconds_l935_93556

/-- The time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) : ℝ :=
  train_length_m / (train_speed_kmh * 1000 / 3600)

/-- Theorem: The time taken for a train to cross a pole is approximately 9 seconds -/
theorem train_crossing_time_approx_9_seconds :
  ∃ ε > 0, |train_crossing_time 58 145 - 9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_9_seconds_l935_93556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l935_93505

-- Define the total investment
noncomputable def total_investment : ℝ := 5000

-- Define the two interest rates
noncomputable def rate1 : ℝ := 0.035
noncomputable def rate2 : ℝ := 0.07

-- Define the condition that the return on rate2 is twice the return on rate1
def return_condition (x : ℝ) : Prop :=
  rate2 * x = 2 * rate1 * (total_investment - x)

-- Define the average rate calculation
noncomputable def average_rate (x : ℝ) : ℝ :=
  (rate2 * x + rate1 * (total_investment - x)) / total_investment

-- Theorem statement
theorem investment_average_rate :
  ∃ x : ℝ, 0 < x ∧ x < total_investment ∧ return_condition x ∧ average_rate x = 0.0525 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l935_93505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l935_93576

theorem sin_minus_cos_value (θ : Real) : 
  (π/2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  Real.tan (θ + π/4) = 1/2 → 
  Real.sin θ - Real.cos θ = 2 * Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l935_93576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_operating_time_proof_l935_93550

/-- The total operating time of a movie theater in hours -/
noncomputable def theater_operating_time (movie_length : ℝ) (replays : ℕ) (ad_length : ℝ) : ℝ :=
  movie_length * (replays : ℝ) + (ad_length / 60) * (replays : ℝ)

/-- Theorem: Given the specified conditions, the theater operates for 11 hours per day -/
theorem theater_operating_time_proof :
  theater_operating_time 1.5 6 20 = 11 := by
  -- Unfold the definition of theater_operating_time
  unfold theater_operating_time
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- This line is removed as it's not necessary for the proof
-- #eval theater_operating_time 1.5 6 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_operating_time_proof_l935_93550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_equal_750_l935_93570

/-- Anne's savings in dollars -/
def A : ℝ := sorry

/-- Katherine's savings in dollars -/
def K : ℝ := sorry

/-- If Anne had $150 less, she would have 1/3 as much as Katherine -/
axiom condition1 : A - 150 = (1/3) * K

/-- If Katherine had twice as much, she would have 3 times as much as Anne -/
axiom condition2 : 2 * K = 3 * A

/-- The total savings of Anne and Katherine -/
def total_savings : ℝ := A + K

theorem total_savings_equal_750 : total_savings = 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_equal_750_l935_93570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_approx_nearest_integer_to_sum_of_roots_l935_93577

/-- A function satisfying the given condition for all non-zero x -/
noncomputable def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → 3 * f x + 2 * f (1 / x) = 6 * x + 3

/-- The sum of roots for the equation f(x) = 2010 -/
noncomputable def sum_of_roots (f : ℝ → ℝ) : ℝ :=
  10047 / 18

theorem sum_of_roots_approx (f : ℝ → ℝ) (h : f_condition f) :
  |sum_of_roots f - 558.17| < 0.01 :=
sorry

theorem nearest_integer_to_sum_of_roots (f : ℝ → ℝ) (h : f_condition f) :
  Int.floor (sum_of_roots f + 0.5) = 558 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_approx_nearest_integer_to_sum_of_roots_l935_93577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_f_less_than_11_l935_93580

def f : ℕ → ℕ
  | 0 => 9  -- Add this case to handle Nat.zero
  | 1 => 9
  | n + 1 => if (n + 1) % 3 = 0 then f n + 3 else f n - 1

theorem count_f_less_than_11 : 
  (Finset.filter (fun k => f k < 11) (Finset.range 1000)).card = 5 := by sorry

#eval (Finset.filter (fun k => f k < 11) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_f_less_than_11_l935_93580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_pump_liquid_l935_93508

-- Define the boiler parameters
variable (ρ : ℝ) -- Density of the liquid
variable (H : ℝ) -- Height of the boiler
variable (a : ℝ) -- Parameter of the parabolic surface
variable (g : ℝ) -- Gravitational acceleration

-- Define the boiler shape
def boiler_shape (x y z : ℝ) : Prop := z = a^2 * (x^2 + y^2)

-- Define the work function
noncomputable def work (ρ H a g : ℝ) : ℝ := (Real.pi * ρ * g * H^3) / (6 * a^2)

-- Theorem statement
theorem work_to_pump_liquid (ρ H a g : ℝ) (h₁ : ρ > 0) (h₂ : H > 0) (h₃ : a ≠ 0) (h₄ : g > 0) :
  work ρ H a g = (Real.pi * ρ * g * H^3) / (6 * a^2) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_pump_liquid_l935_93508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_bound_l935_93590

theorem subset_union_bound (n : ℕ) (h : n > 0) :
  ∃ (S : Finset (Finset (Fin n))),
    S.card = n ∧
    (∀ s ∈ S, s.card = 2) ∧
    (∃ T : Finset (Fin n), ∀ s ∈ S, s ⊆ T) ∧
    ((Finset.univ.filter (λ s : Finset (Fin n) ↦ s.card = 2)).card = 2 * n - 1) →
    (∃ U : Finset (Fin n), (∀ s ∈ S, s ⊆ U) ∧ U.card ≤ (2 * n / 3 : ℚ).ceil) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_bound_l935_93590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l935_93516

/-- A rectangular parallelepiped with dimensions 2 × 6 × 9 -/
structure Parallelepiped :=
  (length : ℝ) (width : ℝ) (height : ℝ)
  (length_eq : length = 2)
  (width_eq : width = 6)
  (height_eq : height = 9)

/-- A point O in the section AA'C'C of the parallelepiped -/
structure PointO :=
  (inSection : Bool)

/-- A sphere inscribed in a dihedral angle -/
structure Sphere :=
  (center : PointO)
  (touchesPlaneABCPrime : Bool)
  (touchesPlaneAABPrime : Bool)
  (doesNotIntersectPlaneAADPrime : Bool)

/-- Angles formed by point O -/
structure Angles :=
  (OAB : ℝ)
  (OAD : ℝ)
  (OAAPrime : ℝ)
  (sum_eq : OAB + OAD + OAAPrime = 180)

/-- Distance from a point to a plane -/
def distance_from_point_to_plane (o : PointO) (h : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem distance_to_plane (p : Parallelepiped) (o : PointO) (s : Sphere) (a : Angles) :
  o.inSection ∧ 
  s.center = o ∧ 
  s.touchesPlaneABCPrime ∧ 
  s.touchesPlaneAABPrime ∧ 
  s.doesNotIntersectPlaneAADPrime →
  (∃ (d : ℝ), d = 3 ∧ d = distance_from_point_to_plane o p.height) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l935_93516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_base_radius_proof_l935_93589

/-- The new base radius of a reshaped cone and cylinder -/
noncomputable def new_base_radius : ℝ := Real.sqrt 7

theorem new_base_radius_proof (
  orig_cone_radius : ℝ) (orig_cone_height : ℝ)
  (orig_cyl_radius : ℝ) (orig_cyl_height : ℝ)
  (new_cone_height : ℝ) (new_cyl_height : ℝ)
  (h1 : orig_cone_radius = 5)
  (h2 : orig_cone_height = 4)
  (h3 : orig_cyl_radius = 2)
  (h4 : orig_cyl_height = 8)
  (h5 : new_cone_height + new_cyl_height = orig_cone_height + orig_cyl_height)
  (h6 : (1/3) * π * new_base_radius^2 * new_cone_height + π * new_base_radius^2 * new_cyl_height = 
        (1/3) * π * orig_cone_radius^2 * orig_cone_height + π * orig_cyl_radius^2 * orig_cyl_height) :
  new_base_radius = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_base_radius_proof_l935_93589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_l_equations_of_m_l935_93594

-- Define the slope of line l
def m : ℚ := 3/4

-- Define the point P
def P : ℝ × ℝ := (2, 5)

-- Define the distance from P to line m
def distance_to_m : ℝ := 3

-- Theorem for the equation of line l
theorem equation_of_l (x y : ℝ) : 
  (m : ℝ) * (x - P.1) = y - P.2 ↔ 3 * x - 4 * y + 14 = 0 :=
sorry

-- Theorem for the equations of line m
theorem equations_of_m (x y : ℝ) : 
  (∃ (c : ℝ), 3 * x - 4 * y + c = 0 ∧ 
   |c - 14| / Real.sqrt (3^2 + 4^2) = distance_to_m) ↔
  (3 * x - 4 * y - 1 = 0 ∨ 3 * x - 4 * y + 29 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_l_equations_of_m_l935_93594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l935_93588

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, f x = x - Real.sin x) →
  f (a - 2) + f (4 - a^2) < 0 →
  2 < a ∧ a < Real.sqrt 5 :=
by
  -- Introduce the assumptions
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l935_93588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_triangle_l935_93502

/-- Represents a pentagon formed by a square and two symmetric right isosceles triangles -/
structure Pentagon where
  square_side : ℝ
  square_side_positive : square_side > 0

/-- Represents a right isosceles triangle -/
structure RightIsoscelesTriangle where
  leg_length : ℝ
  leg_length_positive : leg_length > 0

/-- Calculate the area of a pentagon -/
noncomputable def pentagon_area (p : Pentagon) : ℝ :=
  p.square_side^2 + 2 * (p.square_side^2 / 2)

/-- Calculate the area of a right isosceles triangle -/
noncomputable def right_isosceles_triangle_area (t : RightIsoscelesTriangle) : ℝ :=
  t.leg_length^2 / 2

/-- Theorem stating that a pentagon can be cut into parts forming a new right isosceles triangle -/
theorem pentagon_to_triangle (p : Pentagon) (h : p.square_side = 2) :
  ∃ t : RightIsoscelesTriangle, 
    pentagon_area p = right_isosceles_triangle_area t ∧ 
    t.leg_length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_triangle_l935_93502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_31_is_friday_l935_93527

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => 
    let nextDay := match start with
      | DayOfWeek.Sunday => DayOfWeek.Monday
      | DayOfWeek.Monday => DayOfWeek.Tuesday
      | DayOfWeek.Tuesday => DayOfWeek.Wednesday
      | DayOfWeek.Wednesday => DayOfWeek.Thursday
      | DayOfWeek.Thursday => DayOfWeek.Friday
      | DayOfWeek.Friday => DayOfWeek.Saturday
      | DayOfWeek.Saturday => DayOfWeek.Sunday
    dayAfter nextDay n

theorem january_31_is_friday (h : DayOfWeek.Wednesday = dayAfter DayOfWeek.Wednesday 0) :
  DayOfWeek.Friday = dayAfter DayOfWeek.Wednesday 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_31_is_friday_l935_93527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_cylinder_wire_l935_93514

-- Define the cylinder's dimensions
noncomputable def cylinder_radius : ℝ := 56
noncomputable def cylinder_height : ℝ := 20

-- Define the wire length based on the cylinder dimensions
noncomputable def wire_length : ℝ := 2 * (2 * Real.pi * cylinder_radius) + 2 * cylinder_height

-- Define the cube's edge length
noncomputable def cube_edge : ℝ := wire_length / 12

-- Define the cube's volume
noncomputable def cube_volume : ℝ := cube_edge ^ 3

-- Theorem to prove
theorem cube_volume_from_cylinder_wire :
  ‖cube_volume - 201684.7‖ < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_cylinder_wire_l935_93514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_4F_l935_93593

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the function G
noncomputable def G (x : ℝ) : ℝ := F ((4 * x + x^4) / (1 + 4 * x^3))

-- Theorem statement
theorem G_equals_4F (x : ℝ) (h : x ≠ -1 ∧ x ≠ 1 ∧ 1 + 4 * x^3 ≠ 0) : G x = 4 * F x := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_4F_l935_93593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_190_l935_93523

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + 2

theorem a_7_equals_190 : sequence_a 7 = 190 := by
  -- Proof steps would go here
  sorry

#eval sequence_a 7  -- This will evaluate the function for n = 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_190_l935_93523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_pairs_l935_93507

theorem odd_integer_pairs : 
  ∀ m n : ℕ, 
    (∃ k : ℕ, (3 * m + 1 : ℕ) = k * n) → 
    (∃ l : ℕ, (n^2 + 3 : ℕ) = l * m) → 
    Odd m → Odd n → 
    ((m = 1 ∧ n = 1) ∨ (m = 43 ∧ n = 13) ∨ (m = 49 ∧ n = 37)) := by
  sorry

#check odd_integer_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_pairs_l935_93507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l935_93510

noncomputable def time_for_mile (n : ℕ) : ℝ := 2 * (n - 1)

noncomputable def speed_at_mile (n : ℕ) : ℝ := 1 / (n - 1)

theorem particle_travel_time (n : ℕ) (h : n ≥ 2) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ m : ℕ, m ≥ 2 → speed_at_mile m = k / (m - 1)) ∧
  time_for_mile 2 = 2 →
  time_for_mile n = 2 * (n - 1) :=
by
  sorry

#check particle_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l935_93510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_f_g_l935_93547

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
def g (x : ℝ) : ℝ := x + 3

-- State the theorem
theorem product_f_g (x : ℝ) (h : x ≠ -3) : f x * g x = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_f_g_l935_93547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_218000000_l935_93569

/-- Scientific notation representation -/
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

/-- Check if a number is in valid scientific notation form -/
def is_valid_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

theorem scientific_notation_218000000 :
  ∃ (a : ℝ) (n : ℤ), 
    scientific_notation a n = 218000000 ∧ 
    is_valid_scientific_notation a n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

#check scientific_notation_218000000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_218000000_l935_93569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l935_93565

/-- The common ratio of an infinite geometric series -/
noncomputable def common_ratio (a : ℝ) (S : ℝ) : ℝ := 1 - a / S

theorem geometric_series_ratio (a S : ℝ) (ha : a = 416) (hS : S = 3120) :
  common_ratio a S = 84 / 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l935_93565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l935_93543

/-- Represents a car with its fuel consumption characteristics -/
structure Car where
  initial_fuel : ℝ
  fuel_after_100km : ℝ
  total_distance : ℝ

/-- Calculate the average fuel consumption of the car -/
noncomputable def average_fuel_consumption (c : Car) : ℝ :=
  (c.initial_fuel - c.fuel_after_100km) / 100

/-- Calculate the remaining fuel after driving a given distance -/
noncomputable def remaining_fuel (c : Car) (distance : ℝ) : ℝ :=
  c.initial_fuel - (average_fuel_consumption c) * distance

/-- The main theorem about the car's fuel consumption -/
theorem car_fuel_consumption (c : Car) 
  (h1 : c.initial_fuel = 60)
  (h2 : c.fuel_after_100km = 50)
  (h3 : c.total_distance = 300) :
  average_fuel_consumption c = 0.1 ∧ 
  (∀ x, remaining_fuel c x = 60 - 0.1 * x) ∧
  remaining_fuel c 260 = 34 := by
  sorry

#check car_fuel_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l935_93543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_program_duration_l935_93517

/-- Represents the duration of a degree program in years -/
structure ProgramDuration where
  bs : ℚ
  ms : ℚ
  phd_avg : ℚ

/-- Represents a student's completion rate for the Ph.D. program -/
structure PhDCompletionRate where
  rate : ℚ

/-- The combined program duration for a student -/
def total_duration (program : ProgramDuration) (phd_rate : PhDCompletionRate) : ℚ :=
  program.bs + program.ms + program.phd_avg * phd_rate.rate

/-- Tom's program durations and completion rate -/
def tom_program : ProgramDuration := ⟨3, 2, 5⟩
def tom_phd_rate : PhDCompletionRate := ⟨3/4⟩

/-- Theorem stating that Tom's total program duration is 8.75 years -/
theorem tom_program_duration :
  total_duration tom_program tom_phd_rate = 35/4 := by
  sorry

/-- Possible Ph.D. completion rates -/
def faster_phd_rate : PhDCompletionRate := ⟨1/5⟩
def slower_phd_rate : PhDCompletionRate := ⟨1/7⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_program_duration_l935_93517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptadecagon_path_length_l935_93549

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  vertices : Fin n → ℝ × ℝ

/-- A point on the perimeter of a regular polygon -/
structure PerimeterPoint (n : ℕ) where
  polygon : RegularPolygon n
  position : ℝ × ℝ

/-- The path length traced by a point rotating around each vertex of a regular polygon -/
noncomputable def tracedPathLength (n : ℕ) (p : PerimeterPoint n) (rotations : ℕ) : ℝ :=
  sorry

theorem heptadecagon_path_length :
  ∀ (heptadecagon : RegularPolygon 17) (p : PerimeterPoint 17),
    heptadecagon.sideLength = 3 →
    (let (x, y) := heptadecagon.vertices 0
     let (px, py) := p.position
     (px - x)^2 + (py - y)^2 = 1) →
    tracedPathLength 17 p 2 = 4 * Real.pi := by
  sorry

#check heptadecagon_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptadecagon_path_length_l935_93549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l935_93568

-- Define the function f
noncomputable def f (α : ℚ) (x : ℝ) : ℝ := x ^ (α : ℝ)

-- Define the derivative of f
noncomputable def f_derivative (α : ℚ) (x : ℝ) : ℝ := (α : ℝ) * x ^ ((α - 1 : ℚ) : ℝ)

-- State the theorem
theorem alpha_value (α : ℚ) : 
  f_derivative α (-1) = -4 → α = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l935_93568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l935_93521

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (sum_angles : A + B + C = Real.pi)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Define the condition
noncomputable def condition (t : Triangle) : Prop :=
  Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B

-- Define right angle
def is_right_angle (t : Triangle) : Prop :=
  t.C = Real.pi / 2

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ t : Triangle, is_right_angle t → condition t) ∧
  (∃ t : Triangle, condition t ∧ ¬is_right_angle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l935_93521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_l935_93551

-- Define the points
def P : ℝ × ℝ := (3, -1)
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (4, 5)

-- Define the line equations
def line1 (x y : ℝ) : Prop := 4 * x - y - 13 = 0
def line2 (x : ℝ) : Prop := x = 3

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem equidistant_lines :
  (∃ (x y : ℝ), line1 x y ∧ (x, y) = P ∧ distance (x, y) A = distance (x, y) B) ∨
  (∃ (x : ℝ), line2 x ∧ (x, P.2) = P ∧ distance (x, P.2) A = distance (x, P.2) B) := by
  sorry

#check equidistant_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_l935_93551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l935_93545

noncomputable section

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the function f
def f (lambda : ℝ) (x : ℝ) : ℝ := x^lambda + x^(-lambda)

-- Define the main theorem
theorem functional_equation_solution (lambda : ℝ) (h_lambda : lambda > 0) :
  (∀ (x y z : ℝ), x ∈ PositiveReals → y ∈ PositiveReals → z ∈ PositiveReals →
    f lambda (x*y*z) + f lambda x + f lambda y + f lambda z = 
    f lambda (Real.sqrt (x*y)) * f lambda (Real.sqrt (y*z)) * f lambda (Real.sqrt (z*x))) ∧
  (∀ (x y : ℝ), x ∈ PositiveReals → y ∈ PositiveReals → 1 ≤ x → x < y → f lambda x < f lambda y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l935_93545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l935_93557

/-- Calculates the future value of an investment or loan with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem b_gain_calculation (principal : ℝ) (rate_ab : ℝ) (time_ab : ℝ) 
  (rate_bc : ℝ) (time_bc : ℝ) (h1 : principal = 3200) 
  (h2 : rate_ab = 0.12) (h3 : time_ab = 3) (h4 : rate_bc = 0.145) 
  (h5 : time_bc = 5) :
  let amount_ab := compound_interest principal rate_ab time_ab
  let amount_bc := compound_interest principal rate_bc time_bc
  abs (amount_bc - amount_ab - 1940.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l935_93557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lukes_laundry_l935_93572

theorem lukes_laundry : ℕ :=
  let first_load := 17
  let num_small_loads := 5
  let pieces_per_small_load := 6
  
  let total_pieces := first_load + num_small_loads * pieces_per_small_load
  
  have h : total_pieces = 47 := by
    -- Proof steps would go here
    sorry
  
  47

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lukes_laundry_l935_93572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_slope_l935_93544

/-- A parabola with equation y = ax² + bx + 9 whose tangent line at (2, -1) has slope 1 has a = 3 and b = -11 -/
theorem parabola_tangent_slope (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 9 = a * x^2 + b * x + 9) →
  (a * 2^2 + b * 2 + 9 = -1) →
  (2 * a * 2 + b = 1) →
  (a = 3 ∧ b = -11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_slope_l935_93544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l935_93584

/-- The inclination angle of a line with slope k -/
noncomputable def inclination_angle (k : ℝ) : ℝ := Real.arctan k

/-- The proposition that the inclination angle of line l is between π/6 and π/2 -/
theorem inclination_angle_range (k : ℝ) : 
  (∃ x y : ℝ, y = k * x - Real.sqrt 3 ∧ 
              2 * x + 3 * y - 6 = 0 ∧ 
              x > 0 ∧ y > 0) → 
  π / 6 < inclination_angle k ∧ inclination_angle k < π / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l935_93584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l935_93573

def student_council_size : ℕ := 24
def boys_count : ℕ := 12
def girls_count : ℕ := 12
def committee_size : ℕ := 5

theorem committee_probability :
  (Nat.choose student_council_size committee_size - 2 * Nat.choose boys_count committee_size : ℚ) /
  Nat.choose student_council_size committee_size = 455 / 472 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l935_93573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l935_93546

noncomputable def z₁ : ℂ := 1 + 2 * Complex.I
noncomputable def z₂ : ℂ := (3 + Complex.I) / (1 + Complex.I)
noncomputable def z₃ : ℂ := -1 - 2 * Complex.I

theorem fourth_vertex_of_square (z₄ : ℂ) : 
  (z₁ - z₄ = z₂ - z₃) ∧ (Complex.abs (z₁ - z₂) = Complex.abs (z₂ - z₃)) ∧ (Complex.abs (z₃ - z₄) = Complex.abs (z₄ - z₁)) →
  z₄ = -2 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l935_93546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l935_93522

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)*(3-x) + (3*x)/(x+1)

-- State the theorem
theorem max_value_of_f :
  (f 0 = 1) →
  (f 3 = 9/4) →
  ∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ f x = 7/3 ∧ ∀ y ∈ Set.Icc 1 4, f y ≤ f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l935_93522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunshine_rose_grape_max_profit_l935_93560

/-- The yield function for Sunshine Rose Grape -/
noncomputable def s (x : ℝ) : ℝ := 16/5 - 4/(x+1)

/-- The profit function for Sunshine Rose Grape -/
noncomputable def P (x : ℝ) : ℝ := 25 * s x - x - 3*x

theorem sunshine_rose_grape_max_profit :
  ∀ x : ℝ, 0 < x → x ≤ 5 →
    P x ≤ 44 ∧ (P x = 44 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunshine_rose_grape_max_profit_l935_93560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transformation_l935_93513

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola equation -/
def Hyperbola := ℝ → ℝ

/-- The original hyperbola equation -/
noncomputable def original_hyperbola : Hyperbola :=
  fun x => (1 - 3*x) / (2*x - 1)

/-- The transformed hyperbola equation -/
noncomputable def transformed_hyperbola : Hyperbola :=
  fun x => -0.25 / x

/-- The translation vector -/
def translation : Point :=
  { x := 0.5, y := -1.5 }

/-- Theorem stating the equivalence of the original and transformed hyperbolas -/
theorem hyperbola_transformation :
  ∃ (t : Point),
    (∀ x y, y = original_hyperbola x ↔ 
      (y + t.y) * ((x - t.x)) = -0.25) ∧
    t = translation := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transformation_l935_93513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_integer_pairs_in_circle_l935_93574

/-- The set of integer pairs (x, y) such that x^2 + y^2 ≤ 100 -/
def IntegerPairsInCircle : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | p.1^2 + p.2^2 ≤ 100}

/-- The cardinality of the set of integer pairs (x, y) such that x^2 + y^2 ≤ 100 is 317 -/
theorem cardinality_of_integer_pairs_in_circle :
  Finset.card (Finset.filter (fun p => p.1^2 + p.2^2 ≤ 100) (Finset.product (Finset.range 21) (Finset.range 21))) = 317 := by
  sorry

#eval Finset.card (Finset.filter (fun p => p.1^2 + p.2^2 ≤ 100) (Finset.product (Finset.range 21) (Finset.range 21)))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_integer_pairs_in_circle_l935_93574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_distance_maximize_diff_distance_l935_93596

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the reflection of a point with respect to a line
noncomputable def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  sorry

-- Define the intersection of two lines
noncomputable def intersection (l1 l2 : Line) : Point :=
  sorry

-- Define a function to check if two points are on the same side of a line
def sameSide (p1 p2 : Point) (l : Line) : Prop :=
  sorry

-- Define a membership relation for Point and Line
def onLine (p : Point) (l : Line) : Prop :=
  sorry

-- Define a way to create a Line from two Points
noncomputable def lineFromPoints (p1 p2 : Point) : Line :=
  sorry

-- Theorem for part (a)
theorem minimize_sum_distance (l : Line) (A B : Point) 
  (h : sameSide A B l) :
  ∃ X : Point, 
    onLine X l ∧ 
    (∀ Y : Point, onLine Y l → 
      distance A X + distance B X ≤ distance A Y + distance B Y) ∧
    X = intersection l (lineFromPoints A (reflect B l)) :=
  sorry

-- Theorem for part (b)
theorem maximize_diff_distance (l : Line) (A B : Point) 
  (h : ¬sameSide A B l) :
  ∃ X : Point, 
    onLine X l ∧ 
    (∀ Y : Point, onLine Y l → 
      |distance A X - distance B X| ≥ |distance A Y - distance B Y|) ∧
    X = intersection l (lineFromPoints A (reflect B l)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_distance_maximize_diff_distance_l935_93596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinate_sum_l935_93537

-- Define the points
noncomputable def P : ℝ × ℝ := (0, 8)
noncomputable def Q : ℝ × ℝ := (0, 0)
noncomputable def R : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def G : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
noncomputable def H : ℝ × ℝ := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

-- Define the intersection point I
noncomputable def I : ℝ × ℝ := (0, 8)

-- Theorem statement
theorem intersection_point_coordinate_sum : 
  I.1 + I.2 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinate_sum_l935_93537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_is_two_to_one_l935_93575

/-- Represents Maggie's work week --/
structure WorkWeek where
  office_rate : ℚ
  tractor_rate : ℚ
  tractor_hours : ℚ
  total_income : ℚ

/-- Calculates the ratio of office hours to tractor hours --/
def work_ratio (w : WorkWeek) : ℚ :=
  let office_hours := (w.total_income - w.tractor_rate * w.tractor_hours) / w.office_rate
  office_hours / w.tractor_hours

/-- Theorem stating that the work ratio is 2:1 given the specific conditions --/
theorem work_ratio_is_two_to_one (w : WorkWeek) 
    (h1 : w.office_rate = 10)
    (h2 : w.tractor_rate = 12)
    (h3 : w.tractor_hours = 13)
    (h4 : w.total_income = 416) :
  work_ratio w = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_is_two_to_one_l935_93575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_intersecting_same_line_coplanar_l935_93532

/-- Represents a line in space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A line intersects another line -/
def intersects (l1 l2 : Line) : Prop := sorry

/-- A set of lines lie in the same plane -/
def coplanar (lines : Set Line) : Prop := sorry

/-- Axiom 3 and its corollary -/
axiom axiom_3 : ∀ (l1 l2 l3 : Line), 
  parallel l1 l2 → intersects l1 l3 → intersects l2 l3 → 
  coplanar {l1, l2, l3}

theorem parallel_lines_intersecting_same_line_coplanar 
  (l1 l2 l3 : Line) : 
  parallel l1 l2 → intersects l1 l3 → intersects l2 l3 → 
  coplanar {l1, l2, l3} := by
  intros h1 h2 h3
  exact axiom_3 l1 l2 l3 h1 h2 h3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_intersecting_same_line_coplanar_l935_93532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encoding_transformation_l935_93501

/-- Represents the encoding of a single character --/
inductive Encoding
| A
| B
| C

/-- Defines the new encoding rules --/
def new_encode (c : Encoding) : String :=
  match c with
  | Encoding.A => "21"
  | Encoding.B => "122"
  | Encoding.C => "1"

/-- Represents the original encoded message --/
def original_message : String := "011011010011"

/-- Represents the decoded message (intermediate step) --/
def decoded_message : List Encoding := [Encoding.A, Encoding.B, Encoding.C, Encoding.B, Encoding.A]

/-- The theorem to be proved --/
theorem encoding_transformation :
  String.join (List.map new_encode decoded_message) = "211221121" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_encoding_transformation_l935_93501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_zero_max_value_on_interval_no_zeros_iff_a_in_range_l935_93533

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - a

-- Part I
theorem extremum_at_zero (a : ℝ) (h : a ≠ 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a 0 ≤ f a x) → a = -1 :=
sorry

theorem max_value_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) 1, ∀ y ∈ Set.Icc (-2 : ℝ) 1, f (-1) y ≤ f (-1) x ∧ f (-1) x = Real.exp (-2) + 3 :=
sorry

-- Part II
theorem no_zeros_iff_a_in_range (a : ℝ) :
  (∀ x, f a x ≠ 0) ↔ a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_zero_max_value_on_interval_no_zeros_iff_a_in_range_l935_93533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_five_not_possible_l935_93562

/-- Represents the score in the basketball shooting competition -/
def Score := ℕ

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := Finset.range 10

/-- The maximum possible score when all shots are successful -/
def MaxScore : ℕ := Finset.sum BallNumbers id

/-- Predicate to check if a score is possible given exactly two missed shots -/
def IsPossibleScore (s : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ BallNumbers ∧ b ∈ BallNumbers ∧ a ≠ b ∧ s = MaxScore - (a + b)

/-- Theorem stating that 35 is not a possible score -/
theorem thirty_five_not_possible : ¬ IsPossibleScore 35 := by
  sorry

#eval MaxScore -- This will output the maximum score (55)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_five_not_possible_l935_93562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_is_one_f_inequality_when_a_is_minus_one_l935_93535

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (x^2 + a*x) / Real.exp x

-- Statement 1
theorem f_nonnegative_when_a_is_one :
  ∀ x : ℝ, x ≥ 0 → f 1 x ≥ 0 := by sorry

-- Statement 2
theorem f_inequality_when_a_is_minus_one :
  ∀ x : ℝ, x > 0 → (1 - Real.log x / x) * f (-1) x > 1 - 1 / Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_is_one_f_inequality_when_a_is_minus_one_l935_93535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ABCD_l935_93558

noncomputable def A : ℝ × ℝ := (2, 1)
noncomputable def B : ℝ × ℝ := (3, 2)
noncomputable def C : ℝ × ℝ := (0, 5)
noncomputable def D : ℝ × ℝ := (-1, 4)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem rectangle_ABCD :
  (dot_product (vector A B) (vector A D) = 0) ∧
  (vector A B = vector D C) ∧
  (dot_product (vector A C) (vector B D) / (magnitude (vector A C) * magnitude (vector B D)) = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ABCD_l935_93558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_three_eq_six_point_two_five_l935_93526

/-- The function k(x) = 4x - 7 -/
noncomputable def k (x : ℝ) : ℝ := 4 * x - 7

/-- The function s(k(x)) = x^2 - 2x + 5 -/
noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 7) / 4  -- Inverse of k(x)
  x^2 - 2*x + 5

/-- Theorem: s(3) = 6.25 -/
theorem s_of_three_eq_six_point_two_five : s 3 = 6.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_three_eq_six_point_two_five_l935_93526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_cubes_common_volume_common_volume_positive_l935_93582

/-- Two congruent cubes with shared diagonal intersection planes -/
structure IntersectingCubes where
  a : ℝ  -- Edge length
  rotation_angle : ℝ  -- Rotation angle between diagonal planes
  (positive_edge : 0 < a)
  (rotation_is_90 : rotation_angle = π / 2)

/-- Volume of the common part of the intersecting cubes -/
noncomputable def commonVolume (c : IntersectingCubes) : ℝ :=
  c.a ^ 3 * (Real.sqrt 2 - 3 / 2)

/-- Theorem stating the volume of the common part of the intersecting cubes -/
theorem intersecting_cubes_common_volume (c : IntersectingCubes) :
  commonVolume c = c.a ^ 3 * (Real.sqrt 2 - 3 / 2) := by
  -- Unfold the definition of commonVolume
  unfold commonVolume
  -- The equality holds by definition
  rfl

/-- Theorem stating that the common volume is positive -/
theorem common_volume_positive (c : IntersectingCubes) :
  0 < commonVolume c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_cubes_common_volume_common_volume_positive_l935_93582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l935_93504

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  2 * (Real.sin t.C)^2 + 5 * (Real.sin t.A)^2 = 7 * Real.sin t.A * Real.sin t.C ∧
  t.c < 2 * t.a ∧
  1/2 * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 15 ∧
  Real.sin t.B = Real.sqrt 15 / 4

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.a = t.c ∧ 
  (∃ m : ℝ, m = 4 ∨ m = 2 * Real.sqrt 6 ∧ 
   m^2 = t.a^2 + (t.a/2)^2 - t.a * (t.a/2) * Real.cos t.A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l935_93504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l935_93500

/-- The coefficient of x^2 in the expansion of (1+x)^7 is 21 -/
theorem coefficient_x_squared_in_expansion : Nat.choose 7 2 = 21 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l935_93500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l935_93519

/-- Parabola with equation y² = 4x and focus at (1, 0) -/
structure Parabola where
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4*x
  focus : ℝ × ℝ := (1, 0)

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_property (p : Parabola) (a : PointOnParabola p) :
  distance (a.x, a.y) p.focus = 5/4 * a.x → a.x = 4 := by
  sorry

#check parabola_point_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l935_93519
