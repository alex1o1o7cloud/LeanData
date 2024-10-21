import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l697_69707

theorem law_of_sines (A B C a b c R : Real) :
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) → (R > 0) →
  (a = 2 * R * Real.sin A) → (b = 2 * R * Real.sin B) → (c = 2 * R * Real.sin C) →
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧ c / Real.sin C = 2 * R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l697_69707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l697_69736

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_and_range (A ω φ : ℝ) 
  (h1 : A > 0) (h2 : ω > 0) (h3 : |φ| < Real.pi / 2)
  (h4 : ∀ x, f A ω φ x ≥ -2)
  (h5 : f A ω φ 0 = Real.sqrt 3)
  (h6 : f A ω φ (5 * Real.pi / 6) = 0)
  (h7 : ∀ x ∈ Set.Icc 0 (Real.pi / 6), 
    ∀ y ∈ Set.Icc 0 (Real.pi / 6), x < y → f A ω φ x < f A ω φ y) :
  (∀ x, f A ω φ x = 2 * Real.sin (4 / 5 * x + Real.pi / 3)) ∧
  (Set.Icc 1 2 = {y | ∃ x ∈ Set.Icc 0 (5 * Real.pi / 8), f A ω φ x = y}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l697_69736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_statement_A_is_not_axiom_l697_69795

-- Define the basic concepts
structure Plane : Type
structure Line : Type
structure Point : Type

-- Define the relations
def Parallel (p q : Plane) : Prop := sorry
def PointOn (pt : Point) (l : Line) : Prop := sorry
def LineIn (l : Line) (p : Plane) : Prop := sorry
def PointIn (pt : Point) (p : Plane) : Prop := sorry

-- Define the axioms
axiom axiom_B : ∀ (p1 p2 p3 : Point), 
  (¬∃ (l : Line), PointOn p1 l ∧ PointOn p2 l ∧ PointOn p3 l) → 
  ∃! (p : Plane), PointIn p1 p ∧ PointIn p2 p ∧ PointIn p3 p

axiom axiom_C : ∀ (l : Line) (p : Plane) (p1 p2 : Point),
  PointOn p1 l → PointOn p2 l → PointIn p1 p → PointIn p2 p →
  ∀ (q : Point), PointOn q l → PointIn q p

axiom axiom_D : ∀ (p1 p2 : Plane) (pt : Point),
  p1 ≠ p2 → PointIn pt p1 → PointIn pt p2 →
  ∃! (l : Line), LineIn l p1 ∧ LineIn l p2 ∧ PointOn pt l

-- Define the statement A as a theorem
theorem parallel_transitivity : ∀ (p1 p2 p3 : Plane),
  Parallel p1 p3 → Parallel p2 p3 → Parallel p1 p2 :=
by sorry

-- The main theorem stating that A is not an axiom
theorem statement_A_is_not_axiom : 
  ¬(∀ (p1 p2 p3 : Plane), Parallel p1 p3 → Parallel p2 p3 → Parallel p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_statement_A_is_not_axiom_l697_69795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_shirts_left_l697_69777

/-- Given that Kim has 4 dozen shirts and gives 1/3 of them to her sister, prove that she has 32 shirts left. -/
theorem kim_shirts_left : ℕ := by
  let initial_shirts : ℕ := 4 * 12
  let sister_share : ℕ := initial_shirts / 3
  have h : initial_shirts - sister_share = 32 := by
    -- Proof goes here
    sorry
  exact 32


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_shirts_left_l697_69777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_is_necessary_l697_69750

-- Define the variables A, B, C, and D as real numbers
variable (A B C D : ℝ)

-- Define the implication from condition ① to condition ②
def condition_implication (A B C D : ℝ) : Prop := A > B → C < D

-- Define what it means for condition ① to be a necessary condition for condition ②
def is_necessary_condition (A B C D : ℝ) : Prop := C < D → A > B

-- Theorem statement
theorem condition_is_necessary : 
  ∀ A B C D : ℝ, condition_implication A B C D → is_necessary_condition A B C D :=
by
  intros A B C D h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_is_necessary_l697_69750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_browser_tabs_remaining_l697_69796

theorem browser_tabs_remaining (initial_tabs : ℕ) 
  (h1 : initial_tabs = 400) : 
  (initial_tabs - (initial_tabs / 4) - 
   ((initial_tabs - (initial_tabs / 4)) * 2 / 5) - 
   ((initial_tabs - (initial_tabs / 4) - 
     ((initial_tabs - (initial_tabs / 4)) * 2 / 5)) / 2)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_browser_tabs_remaining_l697_69796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l697_69734

/-- The vertex of the parabola y = x^2 - 4x + 1 has coordinates (2, -3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 4*x + 1 → (∃ (x₀ y₀ : ℝ), (x₀, y₀) = (2, -3) ∧ ∀ x', y ≤ x'^2 - 4*x' + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l697_69734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_butcher_distance_l697_69740

/-- The distance between the bakery and the butcher shop in kilometers -/
noncomputable def total_distance : ℝ := 2.5

/-- The extra distance walked by the butcher's son in kilometers -/
noncomputable def extra_distance : ℝ := 0.5

/-- The time left for the butcher's son to reach his destination in hours -/
noncomputable def butcher_son_time_left : ℝ := 1/6

/-- The time left for the baker's son to reach his destination in hours -/
noncomputable def baker_son_time_left : ℝ := -3/8

/-- The distance walked by the baker's son when they meet in kilometers -/
noncomputable def baker_son_distance : ℝ := 1

/-- The distance walked by the butcher's son when they meet in kilometers -/
noncomputable def butcher_son_distance : ℝ := baker_son_distance + extra_distance

theorem bakery_butcher_distance :
  (baker_son_distance + butcher_son_distance = total_distance) ∧
  (butcher_son_distance = baker_son_distance + extra_distance) ∧
  (baker_son_distance = butcher_son_time_left * (baker_son_distance / butcher_son_time_left)) ∧
  (butcher_son_distance = -baker_son_time_left * (butcher_son_distance / -baker_son_time_left)) :=
by sorry

#check bakery_butcher_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_butcher_distance_l697_69740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparisons_l697_69725

theorem sine_comparisons :
  (0 < Real.pi/10 ∧ Real.pi/10 < Real.pi/8 ∧ Real.pi/8 < Real.pi/2) →
  (0 < Real.pi/8 ∧ Real.pi/8 < 3*Real.pi/8 ∧ 3*Real.pi/8 < Real.pi/2) →
  (Real.sin (-Real.pi/10) > Real.sin (-Real.pi/8)) ∧
  (Real.sin (7*Real.pi/8) < Real.sin (5*Real.pi/8)) := by
  sorry

#check sine_comparisons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparisons_l697_69725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_diag_quad_side_length_l697_69773

/-- A quadrilateral with perpendicular diagonals -/
structure PerpDiagQuad where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  diag_perp : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem to be proved -/
theorem perp_diag_quad_side_length (Q : PerpDiagQuad) 
  (h1 : distance Q.A Q.B = 5)
  (h2 : distance Q.B Q.C = 4)
  (h3 : distance Q.C Q.D = 3) :
  distance Q.D Q.A = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_diag_quad_side_length_l697_69773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_polynomials_l697_69767

/-- Define the complex number z as a 2011th root of unity -/
noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 2011)

/-- Define the polynomial P(x) -/
noncomputable def P (x : ℂ) : ℂ := 
  x^2008 + 3*x^2007 + 6*x^2006 + 
  (Finset.sum (Finset.range 2006) (λ k ↦ ((k+3)*(k+4)/2) * x^(2007-k))) + 
  (2008 * 2009 / 2) * x + (2009 * 2010 / 2)

/-- Theorem statement -/
theorem product_of_polynomials :
  (Finset.prod (Finset.range 2010) (λ k ↦ P (z^(k+1)))) = 
    2011^2009 * (1005^2011 - 1004^2011) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_polynomials_l697_69767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_sin_2alpha_value_l697_69761

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2

-- Theorem for the smallest positive period
theorem smallest_positive_period (x : ℝ) : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (y : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 ∧ (∀ (y : ℝ), f (x + q) = f x) → p ≤ q) ∧ 
  p = Real.pi :=
sorry

-- Theorem for the value of sin(2α)
theorem sin_2alpha_value (α : ℝ) 
  (h1 : -Real.pi / 2 < α) (h2 : α < 0) (h3 : f α = 5 / 6) : 
  Real.sin (2 * α) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_sin_2alpha_value_l697_69761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BAC_l697_69711

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (5, 0)

-- Define the angle BAC
noncomputable def angle_BAC : ℝ := Real.arccos (
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) /
  (((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt * ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt)
)

-- Theorem statement
theorem sin_angle_BAC : Real.sin angle_BAC = 3 * (10 : ℝ).sqrt / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BAC_l697_69711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_six_l697_69701

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the area of a quadrilateral using the Shoelace formula -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * |p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y -
         p2.x * p1.y - p3.x * p2.y - p4.x * p3.y - p1.x * p4.y|

/-- The vertices of the polygon -/
def vertices : List Point := [
  { x := 0, y := 0 },
  { x := 4, y := 0 },
  { x := 2, y := 3 },
  { x := 4, y := 6 }
]

theorem polygon_area_is_six :
  quadrilateralArea (vertices[0]!) (vertices[1]!) (vertices[2]!) (vertices[3]!) = 6 := by
  sorry

#eval quadrilateralArea (vertices[0]!) (vertices[1]!) (vertices[2]!) (vertices[3]!)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_six_l697_69701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_seaplane_average_speed_approx_l697_69742

/-- Calculate the average speed for a round trip with equal distances -/
noncomputable def averageSpeed (speed1 speed2 : ℝ) : ℝ :=
  (2 * speed1 * speed2) / (speed1 + speed2)

/-- Theorem: The average speed for a round trip with equal distances is (2 * speed1 * speed2) / (speed1 + speed2) -/
theorem average_speed_round_trip (speed1 speed2 : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) :
  averageSpeed speed1 speed2 = (2 * speed1 * speed2) / (speed1 + speed2) := by
  sorry

/-- The average speed for the seaplane trip -/
noncomputable def seaplaneAverageSpeed : ℝ :=
  averageSpeed 110 88

/-- Theorem: The average speed for the seaplane trip is approximately 97.78 mph -/
theorem seaplane_average_speed_approx :
  ∃ ε > 0, |seaplaneAverageSpeed - 97.78| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_seaplane_average_speed_approx_l697_69742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_value_l697_69719

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2+m)

-- State the theorem
theorem odd_power_function_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f m x = -(f m (-x))) →  -- f is odd on [-1, m]
  f m (m+1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_value_l697_69719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l697_69788

-- Define the constants
noncomputable def a : ℝ := Real.log 6 / Real.log (1/3)
noncomputable def b : ℝ := (1/4) ^ (8/10 : ℝ)
noncomputable def c : ℝ := Real.log Real.pi

-- State the theorem
theorem order_of_expressions : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l697_69788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l697_69790

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the product of slopes of lines connecting any point on the hyperbola (except vertices)
    to the vertices is 3, then the equations of the asymptotes are y = ±√3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → x ≠ a → x ≠ -a → 
    (y / (x + a)) * (y / (x - a)) = 3) →
  (∃ k : ℝ, k^2 = 3 ∧ 
    (∀ x y : ℝ, (y = k*x ∨ y = -k*x) ↔ 
      (∀ ε > 0, ∃ x' y', x'^2/a^2 - y'^2/b^2 = 1 ∧ 
        ((x - x')^2 + (y - y')^2 < ε^2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l697_69790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_speaking_probability_l697_69747

def probability_of_speaking : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def at_least_two_speaking (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n + Nat.choose n 1 * p * (1 - p)^(n - 1))

theorem at_least_two_speaking_probability :
  at_least_two_speaking probability_of_speaking number_of_babies = 473 / 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_speaking_probability_l697_69747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l697_69792

theorem cos_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 4/5)
  (h4 : Real.tan (α - β) = -1/3) :
  Real.cos β = 9 * Real.sqrt 10 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l697_69792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l697_69768

-- Define the volume of the cylindrical sculpture
noncomputable def sculpture_volume : ℝ := 90 * Real.pi

-- Define the volume of one clay block
def block_volume : ℝ := 48

-- Theorem statement
theorem blocks_needed_for_sculpture :
  ⌈sculpture_volume / block_volume⌉ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l697_69768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentLinesProperties_l697_69726

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- Add necessary fields
  mk :: -- Constructor

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields
  mk :: -- Constructor

/-- A point in 3D space -/
structure Point3D where
  -- Add necessary fields
  mk :: -- Constructor

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields
  mk :: -- Constructor

/-- An edge of a polyhedron -/
structure Edge where
  mk :: -- Constructor

/-- A face of a polyhedron -/
structure Face where
  mk :: -- Constructor

/-- A vertex of a polyhedron -/
structure Vertex where
  mk :: -- Constructor

/-- The midsphere of a regular polyhedron -/
def midsphere (T : RegularPolyhedron) : Sphere := sorry

/-- The midpoint of an edge of a regular polyhedron -/
def edgeMidpoint (T : RegularPolyhedron) (edge : Edge) : Point3D := sorry

/-- The tangent line to the midsphere through the midpoint of an edge, perpendicular to the edge -/
def tangentLine (T : RegularPolyhedron) (edge : Edge) : Line3D := sorry

/-- The set of tangent lines for all edges of a face -/
def faceTangents (T : RegularPolyhedron) (face : Face) : Set Line3D := sorry

/-- The set of tangent lines for all edges meeting at a vertex -/
def vertexTangents (T : RegularPolyhedron) (vertex : Vertex) : Set Line3D := sorry

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Predicate to check if a line lies on a plane -/
def lineOnPlane (l : Line3D) (plane : Plane3D) : Prop := sorry

/-- Main theorem -/
theorem tangentLinesProperties (T : RegularPolyhedron) :
  (∀ face : Face, ∃ p : Point3D, ∀ l ∈ faceTangents T face, pointOnLine p l) ∧
  (∀ vertex : Vertex, ∃ plane : Plane3D, ∀ l ∈ vertexTangents T vertex, lineOnPlane l plane) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentLinesProperties_l697_69726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_package_volume_correct_l697_69757

/-- The volume needed to package the university's fine arts collection -/
def packageVolume : ℕ :=
  let boxLength : ℕ := 20
  let boxWidth : ℕ := 20
  let boxHeight : ℕ := 15
  let boxVolume : ℕ := boxLength * boxWidth * boxHeight
  let boxCost : ℚ := 120 / 100  -- $1.20 expressed as a rational number
  let totalCost : ℚ := 612  -- $612 expressed as a rational number
  let numBoxes : ℕ := (totalCost / boxCost).floor.toNat  -- Convert to ℕ using floor and toNat
  numBoxes * boxVolume

theorem package_volume_correct : packageVolume = 3060000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_package_volume_correct_l697_69757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l697_69780

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the measure of an angle
noncomputable def angle_measure (p q r : ℝ × ℝ) : ℝ := 
  sorry

-- Define the theorem
theorem angle_C_measure 
  (ABCD : Quadrilateral)
  (h : angle_measure ABCD.A ABCD.B ABCD.C / angle_measure ABCD.B ABCD.C ABCD.D = 3) :
  angle_measure ABCD.B ABCD.C ABCD.D = 135 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l697_69780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_vertical_shift_l697_69706

/-- Given a sinusoidal function y = a * Real.sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then d = 1 -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -3)
  (h_osc : ∃ x y, a * Real.sin (b * x + c) + d = 5 ∧ a * Real.sin (b * y + c) + d = -3) :
  d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_vertical_shift_l697_69706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l697_69735

noncomputable def f (x : ℝ) := Real.sqrt (x * (70 - x)) + Real.sqrt (x * (5 - x))

theorem max_value_of_f :
  ∃ (x₀ : ℝ) (M : ℝ),
    x₀ ∈ Set.Icc 0 5 ∧
    M = f x₀ ∧
    (∀ x ∈ Set.Icc 0 5, f x ≤ M) ∧
    x₀ = 14/3 ∧
    M = 5 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l697_69735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ranges_l697_69743

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a * x^2 - 3 * x + 3) / x

-- State the theorem
theorem function_ranges (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (s t : ℝ), s > 0 → g a s ≥ f t) : 
  (∀ (y : ℝ), y ∈ Set.Icc (-3) 3 ↔ ∃ (x : ℝ), f x = y) ∧ 
  (a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ranges_l697_69743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_infected_count_l697_69772

/-- 
Given a town where 1/4 of the population suffers from a viral infection,
prove that the expected number of infected people in a randomly selected
group of 500 is 125.
-/
theorem expected_infected_count 
  (town_infection_rate : ℚ) 
  (sample_size : ℕ) 
  (h1 : town_infection_rate = 1 / 4) 
  (h2 : sample_size = 500) : 
  town_infection_rate * (sample_size : ℚ) = 125 := by
  sorry

#eval (1 / 4 : ℚ) * 500  -- This will evaluate to 125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_infected_count_l697_69772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l697_69720

theorem problem_statement :
  (¬(∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l697_69720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_stabilizing_k_A_19_pow_86_stabilizes_at_19_l697_69752

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Computes f(A) as defined in the problem -/
def f (digits : Digits) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^(digits.length - 1 - i)) 0

/-- Generates the sequence A_i -/
def sequenceA (A : Nat) : Nat → Nat
  | 0 => A
  | n + 1 => f (Nat.digits (sequenceA A n) 10)

theorem existence_of_stabilizing_k (A : Nat) :
  ∃ k : Nat, sequenceA A (k + 1) = sequenceA A k := by
  sorry

#check existence_of_stabilizing_k

theorem A_19_pow_86_stabilizes_at_19 :
  ∃ k : Nat, sequenceA (19^86) k = 19 := by
  sorry

#check A_19_pow_86_stabilizes_at_19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_stabilizing_k_A_19_pow_86_stabilizes_at_19_l697_69752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_one_std_is_84_percent_l697_69784

/-- A symmetric distribution with a given percentage within one standard deviation -/
structure SymmetricDistribution where
  /-- The percentage of the distribution within one standard deviation of the mean -/
  within_one_std : ℝ
  /-- Assumption that the distribution is symmetric -/
  symmetric : Prop
  /-- Assumption that 68% of the distribution is within one standard deviation -/
  within_one_std_is_68_percent : within_one_std = 68

/-- The percentage of the distribution less than one standard deviation above the mean -/
noncomputable def percent_less_than_one_std_above_mean (d : SymmetricDistribution) : ℝ :=
  50 + d.within_one_std / 2

/-- Theorem stating that the percentage less than one standard deviation above the mean is 84% -/
theorem percent_less_than_one_std_is_84_percent (d : SymmetricDistribution) :
  percent_less_than_one_std_above_mean d = 84 := by
  unfold percent_less_than_one_std_above_mean
  rw [d.within_one_std_is_68_percent]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_one_std_is_84_percent_l697_69784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l697_69716

/-- The set S containing elements from 1 to 10 -/
def S : Finset ℕ := Finset.range 10

/-- A family of subsets of S -/
def A : ℕ → Finset ℕ := sorry

/-- The number of subsets in the family -/
def k : ℕ := sorry

/-- Each subset has exactly 5 elements -/
axiom subset_size : ∀ i, (A i).card = 5

/-- The intersection of any two distinct subsets has at most 2 elements -/
axiom intersection_size : ∀ i j, i < j → (A i ∩ A j).card ≤ 2

/-- All subsets are contained in S -/
axiom subset_of_S : ∀ i, A i ⊆ S

/-- The maximum value of k is 6 -/
theorem max_k_is_six : k ≤ 6 ∧ ∃ A : ℕ → Finset ℕ, 
  (∀ i, (A i).card = 5) ∧ 
  (∀ i j, i < j → (A i ∩ A j).card ≤ 2) ∧
  (∀ i, A i ⊆ S) ∧
  k = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l697_69716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_count_l697_69717

theorem floor_sqrt_count : 
  (Finset.filter (fun x : ℕ => Int.floor (Real.sqrt (x : ℝ)) = 8) 
    (Finset.range 81 \ Finset.range 64)).card = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_count_l697_69717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l697_69762

noncomputable section

-- Define the two lines
def line1 (x y : ℝ) : Prop := 3 * x + y - 3 = 0
def line2 (x y : ℝ) : Prop := 3 * x + y + 1/2 = 0

-- Define the distance between two parallel lines
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_between_given_lines :
  distance_between_parallel_lines 3 1 (-3) (1/2) = (7 / 20) * Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l697_69762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l697_69766

noncomputable def f (x : ℝ) : ℝ := x - 4 / x

noncomputable def F (x a : ℝ) : ℝ := x^2 + 16 / x^2 - 2 * a * (x - 4 / x)

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -3 then 6 * a + 17
  else if a < 0 then 8 - a^2
  else 8

theorem main_theorem :
  (∀ x ∈ Set.Icc 1 2, f x ∈ Set.Icc (-3) 0) ∧
  (∀ a x, x ∈ Set.Icc 1 2 → F x a ≥ g a) ∧
  (∀ a ∈ Set.Ioo (-3) 0, ∀ t, g a > -2 * a^2 + a * t + 4 → t > -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l697_69766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_ratio_l697_69731

theorem cylinder_height_ratio (h : ℝ) (h_pos : h > 0) : 
  ∃ (h_new : ℝ), 
    (5/6 : ℝ) * π * h = (3/5 : ℝ) * π * (1.25^2) * h_new ∧ 
    h_new = (32/45 : ℝ) * h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_ratio_l697_69731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_simplification_l697_69787

noncomputable def θ : ℝ := 2 * Real.pi / 2015

theorem product_simplification : 
  ∃ (a : ℕ) (b : ℤ), 
    (Finset.prod (Finset.range 1440) (λ k => Real.cos (2^k * θ) - 1/2) = b / (2:ℝ)^a) ∧
    (b % 2 = 1) ∧
    (a + b = 1441) := by
  sorry

#check product_simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_simplification_l697_69787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l697_69770

theorem absolute_value_puzzle (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) + abs x) + x = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l697_69770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l697_69782

theorem sufficient_not_necessary_condition :
  ∃ (α : ℝ) (k : ℤ), 
    (α = Real.pi / 6 + 2 * k * Real.pi → Real.cos (2 * α) = 1 / 2) ∧
    ¬(Real.cos (2 * α) = 1 / 2 → α = Real.pi / 6 + 2 * k * Real.pi) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l697_69782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_z_l697_69705

theorem triangle_cosine_z (X Y Z : ℝ) : 
  X + Y + Z = Real.pi → 
  Real.sin X = 4/5 → 
  Real.cos Y = 12/13 → 
  Real.cos Z = -16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_z_l697_69705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_value_l697_69702

/-- The minimum value of √2y₀ + |x₀ - y₀ - 2| for any point (x₀, y₀) on the parabola x² = y -/
theorem parabola_min_value : 
  ∀ x₀ y₀ : ℝ, 
  x₀^2 = y₀ → 
  (∀ x y : ℝ, x^2 = y → Real.sqrt 2 * y + |x - y - 2| ≥ Real.sqrt 2 * y₀ + |x₀ - y₀ - 2|) → 
  Real.sqrt 2 * y₀ + |x₀ - y₀ - 2| = 9/4 - Real.sqrt 2/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_value_l697_69702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l697_69756

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the left focus F
def F : ℝ × ℝ := (-4, 0)

-- Define the point A
def A : ℝ × ℝ := (1, 4)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum (P : ℝ × ℝ) (h : hyperbola P.1 P.2) (h_right : P.1 > 0) :
  distance P F + distance P A ≥ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l697_69756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l697_69741

/-- Represents an ellipse with center O, major axis AB, minor axis CD, and focus F. -/
structure Ellipse where
  O : ℝ × ℝ  -- Center of the ellipse
  A : ℝ × ℝ  -- One end of the major axis
  B : ℝ × ℝ  -- Other end of the major axis
  C : ℝ × ℝ  -- One end of the minor axis
  D : ℝ × ℝ  -- Other end of the minor axis
  F : ℝ × ℝ  -- Focus of the ellipse

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the product of major and minor axes of an ellipse -/
theorem ellipse_axes_product (e : Ellipse) 
  (h1 : distance e.O e.F = 8)
  (h2 : distance e.O e.C + distance e.C e.F - distance e.O e.F = 3) :
  (distance e.A e.B) * (distance e.C e.D) = 138.84 := by
  sorry

#check ellipse_axes_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l697_69741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_ab_plus_c_count_l697_69712

def S : Finset ℕ := {1, 2, 3, 4, 5}

theorem odd_ab_plus_c_count :
  ∃! (T : Finset ℕ),
    (∀ n ∈ T, ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b + c ∧ Odd n) ∧
    (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → Odd (a * b + c) → a * b + c ∈ T) ∧
    T.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_ab_plus_c_count_l697_69712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l697_69769

noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

def equation (x : ℝ) : Prop :=
  2 + geometric_sum (5*x) x = 100

def converges (x : ℝ) : Prop := abs x < 1

theorem solution_exists :
  ∃ x : ℝ, equation x ∧ converges x ∧ x = 2/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l697_69769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l697_69745

noncomputable def g (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_g :
  ∀ y : ℝ, (∃ t : ℝ, g t = y) ↔ -1/4 ≤ y ∧ y ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l697_69745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_trajectory_equation_l697_69781

open Real

noncomputable section

-- Define the circle C
def circle_center : ℝ × ℝ := (3, π/6)
def circle_radius : ℝ := 3

-- Define the polar coordinates
def polar_coords (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

-- Define a point on the circle
def point_on_circle (θ : ℝ) : ℝ × ℝ := polar_coords (6 * cos (θ - π/6)) θ

-- Define the ratio condition
def ratio_condition (ρ : ℝ) : Prop := ∃ (k : ℝ), k > 0 ∧ (3 * k) / (2 * k) = 3 / 2 ∧ ρ = 5 * k

-- Theorem for the circle equation
theorem circle_equation (θ : ℝ) : 
  point_on_circle θ = polar_coords (6 * cos (θ - π/6)) θ := by sorry

-- Theorem for the trajectory of point P
theorem trajectory_equation (ρ θ : ℝ) : 
  ratio_condition ρ → polar_coords ρ θ = polar_coords (10 * cos (θ - π/6)) θ := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_trajectory_equation_l697_69781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l697_69779

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    let A := line_l t₁
    let B := line_l t₂
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    2 * Real.sqrt 7 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l697_69779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l697_69728

-- Define the circles and their properties
noncomputable def large_circle_area : ℝ := 100 * Real.pi
def large_circle_sectors : ℕ := 4
def small_circle_halves : ℕ := 2

-- Theorem statement
theorem shaded_area_calculation :
  let large_circle_radius : ℝ := Real.sqrt (large_circle_area / Real.pi)
  let small_circle_radius : ℝ := large_circle_radius / 2
  let large_circle_shaded_area : ℝ := large_circle_area * 2 / large_circle_sectors
  let small_circle_area : ℝ := Real.pi * small_circle_radius ^ 2
  let small_circle_shaded_area : ℝ := small_circle_area / small_circle_halves
  large_circle_shaded_area + small_circle_shaded_area = 62.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l697_69728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_theorem_l697_69798

/-- Represents the bus travel scenario -/
structure BusTravel where
  speed_without_stops : ℝ
  speed_with_stops : ℝ
  total_time : ℝ
  stop_time : ℝ

/-- Calculates the stop time per hour for a given bus travel scenario -/
noncomputable def stop_time_per_hour (bt : BusTravel) : ℝ :=
  (bt.stop_time / bt.total_time) * 60

/-- Theorem stating that for the given speeds, the bus stops for 30 minutes per hour -/
theorem bus_stop_time_theorem (bt : BusTravel) 
  (h1 : bt.speed_without_stops = 80)
  (h2 : bt.speed_with_stops = 40)
  (h3 : bt.total_time > 0) :
  stop_time_per_hour bt = 30 := by
  sorry

#check bus_stop_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_theorem_l697_69798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_implies_one_l697_69733

theorem divisibility_condition_implies_one (x y : ℕ) :
  x > 0 → y > 0 →
  (x^3 + y) % (x^2 + y^2) = 0 ∧ 
  (y^3 + x) % (x^2 + y^2) = 0 → 
  x = 1 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_implies_one_l697_69733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_avg_age_l697_69732

-- Define the ratio of women to men
def women_to_men_ratio : ℚ := 3 / 2

-- Define the average age of women
def avg_age_women : ℚ := 36

-- Define the average age of men
def avg_age_men : ℚ := 30

-- Theorem to prove
theorem population_avg_age : 
  (women_to_men_ratio * avg_age_women + avg_age_men) / (women_to_men_ratio + 1) = 33 + 3/5 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_avg_age_l697_69732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l697_69776

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def Ray (start : Point) (direction : Point) :=
  {p : Point | ∃ t : ℝ, t ≥ 0 ∧ p.x = start.x + t * (direction.x - start.x) ∧ p.y = start.y + t * (direction.y - start.y)}

theorem locus_is_ray (M N : Point) (h1 : M.x = 1 ∧ M.y = 0) (h2 : N.x = 3 ∧ N.y = 0) :
  ∀ P : Point, distance P M - distance P N = 2 → P ∈ Ray N M := by
  sorry

#check locus_is_ray

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l697_69776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_and_sin_abs_properties_l697_69749

-- Define the absolute value of sine function
noncomputable def abs_sin (x : ℝ) : ℝ := |Real.sin x|

-- Define the sine of absolute value function
noncomputable def sin_abs (x : ℝ) : ℝ := Real.sin |x|

-- Statement of the theorem
theorem abs_sin_and_sin_abs_properties :
  (∀ x : ℝ, abs_sin (x + π) = abs_sin x) ∧
  (∀ x : ℝ, sin_abs (-x) = sin_abs x) := by
  sorry

#check abs_sin_and_sin_abs_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_and_sin_abs_properties_l697_69749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_squares_l697_69783

theorem cosine_difference_squares : 
  Real.cos (π / 12) ^ 2 - Real.cos (5 * π / 12) ^ 2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_squares_l697_69783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_sqrt_29_l697_69759

/-- Given a circle with center on the x-axis passing through points (1,5) and (2,4),
    the radius of this circle is √29. -/
theorem circle_radius_sqrt_29 (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∀ p ∈ C, ∃ r : ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2) →  -- C is a circle
  center.2 = 0 →  -- center is on x-axis
  (1, 5) ∈ C →  -- (1,5) is on the circle
  (2, 4) ∈ C →  -- (2,4) is on the circle
  ∃ r : ℝ, r^2 = 29 ∧ ∀ p ∈ C, (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_sqrt_29_l697_69759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l697_69764

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (heq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (y = x / 2 ∨ y = -x / 2)) :
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l697_69764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l697_69723

theorem complex_equation_solution (z : ℂ) : 
  (z + 2*Complex.I)*(z - 2*Complex.I) = 2 → 
  z = Complex.I * Real.sqrt 2 ∨ z = -Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l697_69723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_diagonal_incommensurable_l697_69794

/- Define a square -/
structure Square where
  side : ℝ
  side_positive : side > 0

/- Define the diagonal of a square -/
noncomputable def diagonal (s : Square) : ℝ := s.side * Real.sqrt 2

/- Define commensurability -/
def commensurable (a b : ℝ) : Prop := ∃ (q : ℚ), a = q * b

/- Theorem statement -/
theorem square_side_diagonal_incommensurable (s : Square) : 
  ¬ commensurable s.side (diagonal s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_diagonal_incommensurable_l697_69794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marigolds_is_89_l697_69799

/-- The number of marigolds sold during a 3-day garden center sale -/
def marigolds_sold : ℕ → ℕ := sorry

/-- The total number of marigolds sold during the 3-day sale -/
def total_marigolds : ℕ := sorry

/-- Conditions of the sale -/
axiom day1 : marigolds_sold 1 = 14
axiom day2 : marigolds_sold 2 = marigolds_sold 1 + 25
axiom day3 : marigolds_sold 3 = 2 * marigolds_sold 2

/-- Definition of total marigolds sold -/
axiom total_def : total_marigolds = marigolds_sold 1 + marigolds_sold 2 + marigolds_sold 3

/-- Theorem: The total number of marigolds sold during the 3-day sale is 89 -/
theorem total_marigolds_is_89 : total_marigolds = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marigolds_is_89_l697_69799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_isosceles_triangle_l697_69771

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    where a > b > 0, and focal distance c, if the triangle formed by
    its left, right, and upper vertices is an isosceles triangle with
    a base angle of 30°, then c/b = √2 -/
theorem ellipse_isosceles_triangle (a b c : ℝ) :
  a > b ∧ b > 0 →  -- Condition: a > b > 0
  (∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) →  -- Condition: Ellipse equation
  (∃ A B C : ℝ × ℝ,  -- Condition: A, B, C are vertices of the ellipse
    A.1 = -a ∧ A.2 = 0 ∧
    B.1 = a ∧ B.2 = 0 ∧
    C.1 = 0 ∧ C.2 = b) →
  (∃ θ : ℝ,  -- Condition: Triangle ABC is isosceles with base angle 30°
    θ = 30 * π / 180 ∧
    Real.tan θ = b / a) →
  c^2 = a^2 - b^2 →  -- Definition of focal distance
  c / b = Real.sqrt 2 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_isosceles_triangle_l697_69771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_intersection_radius_is_sqrt_59_l697_69765

/-- A sphere intersecting three coordinate planes -/
structure IntersectingSphere where
  /-- Center of intersection with xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of intersection with xy-plane -/
  xy_radius : ℝ
  /-- Center of intersection with yz-plane -/
  yz_center : ℝ × ℝ × ℝ
  /-- Center of intersection with xz-plane -/
  xz_center : ℝ × ℝ × ℝ

/-- The radius of the circle where the sphere intersects the yz-plane -/
noncomputable def yz_intersection_radius (s : IntersectingSphere) : ℝ :=
  Real.sqrt 59

/-- Theorem stating that for a sphere with given intersections, 
    the radius of its intersection with the yz-plane is √59 -/
theorem yz_intersection_radius_is_sqrt_59 (s : IntersectingSphere) 
    (h1 : s.xy_center = (3, 5, 0)) 
    (h2 : s.xy_radius = 2) 
    (h3 : s.yz_center = (0, 5, -8)) 
    (h4 : s.xz_center = (3, 0, -8)) : 
  yz_intersection_radius s = Real.sqrt 59 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_intersection_radius_is_sqrt_59_l697_69765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_trip_time_calculation_l697_69730

/-- Represents the time taken for a boat trip -/
structure TripTime where
  downstream : ℝ
  upstream : ℝ

/-- Calculates the time taken for downstream and upstream trips -/
noncomputable def calculateTripTime (distanceDown distanceUp streamSpeed boatSpeed : ℝ) : TripTime :=
  { downstream := distanceDown / (boatSpeed + streamSpeed),
    upstream := distanceUp / (boatSpeed - streamSpeed) }

/-- Theorem stating the correct calculation of trip times -/
theorem correct_trip_time_calculation 
  (distanceDown distanceUp streamSpeed boatSpeed : ℝ)
  (h1 : distanceDown = 78)
  (h2 : distanceUp = 50)
  (h3 : streamSpeed = 7)
  (h4 : boatSpeed > streamSpeed) :
  let tripTime := calculateTripTime distanceDown distanceUp streamSpeed boatSpeed
  tripTime.downstream = 78 / (boatSpeed + 7) ∧
  tripTime.upstream = 50 / (boatSpeed - 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_trip_time_calculation_l697_69730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_main_volume_theorem_l697_69778

/-- Represents a cube with edge length 2 units -/
def Cube : Set (Fin 3 → ℝ) :=
  {c | ∀ i, 0 ≤ c i ∧ c i ≤ 2}

/-- Represents the first cut plane -/
def FirstCut : Set (Fin 3 → ℝ) :=
  {p | ∃ t, p 0 + p 1 = 2*t ∧ 0 ≤ t ∧ t ≤ 1}

/-- Represents the second cut plane -/
def SecondCut : Set (Fin 3 → ℝ) :=
  {p | ∃ t, p 0 + p 1 = 2*t + 2/3 ∧ 0 ≤ t ∧ t ≤ 1}

/-- Represents the third cut plane -/
def ThirdCut : Set (Fin 3 → ℝ) :=
  {p | ∃ t, p 0 + p 1 = 2*t + 1 ∧ 0 ≤ t ∧ t ≤ 1}

/-- The volume of the pyramid-shaped section including vertex A -/
noncomputable def PyramidVolume : ℝ := 8 * Real.sqrt 2 / 9

/-- Theorem stating the volume of the pyramid-shaped section -/
theorem pyramid_volume_theorem :
  ∃ (v : ℝ), v = PyramidVolume ∧ v = 8 * Real.sqrt 2 / 9 :=
by
  use PyramidVolume
  constructor
  · rfl
  · rfl

/-- The main theorem about the volume of the intersected region -/
theorem main_volume_theorem (c : Cube) (f : FirstCut) (s : SecondCut) (t : ThirdCut) :
  ∃ (v : ℝ), v = PyramidVolume :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_main_volume_theorem_l697_69778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_problem_l697_69797

/-- Represents the recycling problem -/
theorem recycling_problem (cans_collected : ℕ) (total_received : ℚ) : 
  cans_collected = 144 → total_received = 12 → ∃ newspaper_kg : ℚ, newspaper_kg = 20 :=
by
  intro h_cans h_total
  let cans_rate : ℚ := 1/2
  let newspaper_rate : ℚ := 3/2
  let cans_money := (cans_collected / 12 : ℚ) * cans_rate
  let newspaper_money := total_received - cans_money
  let newspaper_units := newspaper_money / newspaper_rate
  let newspaper_kg := newspaper_units * 5
  exists newspaper_kg
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_problem_l697_69797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixtape_runtime_l697_69775

-- Define the song lengths for Side A and Side B
def side_a_songs : List Float := [3, 4, 5, 6, 3, 7]
def side_b_songs : List Float := [6, 6, 8, 5]

-- Define transition times
def transition_a : Float := 0.5  -- 30 seconds in minutes
def transition_b : Float := 0.75 -- 45 seconds in minutes

-- Define silence and bonus track lengths
def silence : Float := 2
def bonus_track : Float := 4

-- Define speed factors
def normal_speed : Float := 1
def fast_speed : Float := 1.5

-- Calculate total runtime
def calculate_runtime (songs : List Float) (transition : Float) : Float :=
  songs.sum + (songs.length.toFloat - 1) * transition

-- Theorem statement
theorem mixtape_runtime :
  let side_a_runtime := calculate_runtime side_a_songs transition_a + silence + bonus_track
  let side_b_runtime := calculate_runtime side_b_songs transition_b
  let total_runtime_normal := side_a_runtime + side_b_runtime
  let total_runtime_fast := total_runtime_normal / fast_speed
  total_runtime_normal = 63.75 ∧ (total_runtime_fast - 42.5).abs < 0.01 := by
  sorry

#eval let side_a_runtime := calculate_runtime side_a_songs transition_a + silence + bonus_track
      let side_b_runtime := calculate_runtime side_b_songs transition_b
      let total_runtime_normal := side_a_runtime + side_b_runtime
      let total_runtime_fast := total_runtime_normal / fast_speed
      (total_runtime_normal, total_runtime_fast)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixtape_runtime_l697_69775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_five_l697_69727

-- Define lg as logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_five : lg 25 + Real.log 27 / Real.log 3 + lg 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_five_l697_69727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_possible_l697_69751

/-- Represents a person's position on a circular track -/
structure Position where
  angle : ℝ
  deriving Inhabited

/-- Represents a move on the circular track -/
structure Move where
  person1 : Fin n
  person2 : Fin n
  distance : ℝ

/-- Checks if the positions form an equal distribution on the circle -/
def is_equal_distribution {n : ℕ} (positions : Fin n → Position) : Prop :=
  ∀ i j : Fin n, (positions i).angle - (positions j).angle = (2 * Real.pi / n) * (i - j)

/-- Applies a move to the current positions -/
def apply_move {n : ℕ} (positions : Fin n → Position) (move : Move) : Fin n → Position :=
  sorry

/-- Theorem stating that it's possible to achieve an equal distribution in at most n-1 moves -/
theorem equal_distribution_possible (n : ℕ) (h : n > 0) :
  ∃ (initial_positions : Fin n → Position) (moves : List Move),
    moves.length ≤ n - 1 ∧
    is_equal_distribution (moves.foldl apply_move initial_positions) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_possible_l697_69751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_f_expression_prove_k_value_prove_k_range_l697_69758

noncomputable section

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4^x + 1, 2^x)
def b (y k : ℝ) : ℝ × ℝ := (y - 1, y - k)

-- Define the perpendicularity condition
def perpendicular (x y k : ℝ) : Prop :=
  (4^x + 1) * (y - 1) + 2^x * (y - k) = 0

-- Define the function f
noncomputable def f (x k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Theorem 1: Prove the expression for f(x)
theorem prove_f_expression (x y k : ℝ) :
  perpendicular x y k → y = f x k := by
  sorry

-- Theorem 2: Prove the value of k when the minimum of f(x) is -3
theorem prove_k_value (k : ℝ) :
  (∃ x, f x k = -3) ∧ (∀ x, f x k ≥ -3) → k = -11 := by
  sorry

-- Define the triangle inequality condition
def triangle_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃, f x₁ + f x₂ > f x₃ ∧ f x₂ + f x₃ > f x₁ ∧ f x₃ + f x₁ > f x₂

-- Theorem 3: Prove the range of k
theorem prove_k_range (k : ℝ) :
  triangle_inequality (f · k) → k ∈ Set.Icc (-1/2) 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_f_expression_prove_k_value_prove_k_range_l697_69758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_five_digit_number_with_product_36_l697_69704

/-- Represents a five-digit number as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Checks if a given list represents a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.all (λ d => d ≥ 0 ∧ d ≤ 9) ∧ n.head! ≠ 0

/-- Calculates the product of digits -/
def digitProduct (n : FiveDigitNumber) : Nat :=
  n.foldl (·*·) 1

/-- Calculates the sum of digits -/
def digitSum (n : FiveDigitNumber) : Nat :=
  n.sum

/-- Compares two five-digit numbers -/
def greaterThan (a b : FiveDigitNumber) : Prop :=
  a.reverse.foldl (fun acc d => acc * 10 + d) 0 > b.reverse.foldl (fun acc d => acc * 10 + d) 0

theorem greatest_five_digit_number_with_product_36 :
  ∃ (M : FiveDigitNumber),
    isValidFiveDigitNumber M ∧
    digitProduct M = 36 ∧
    (∀ (N : FiveDigitNumber), isValidFiveDigitNumber N → digitProduct N = 36 → greaterThan M N) ∧
    digitSum M = 15 := by
  sorry

#eval digitProduct [9, 2, 2, 1, 1]
#eval digitSum [9, 2, 2, 1, 1]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_five_digit_number_with_product_36_l697_69704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_quadrilateral_area_one_implies_n_zero_l697_69786

/-- A quadrilateral with vertices on the curve y = e^x -/
structure ExpQuadrilateral where
  n : ℝ
  area : ℝ
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ
  vertex4 : ℝ × ℝ

/-- The area of the quadrilateral is calculated using the Shoelace formula -/
noncomputable def calculateArea (q : ExpQuadrilateral) : ℝ :=
  (1/2) * abs (
    q.vertex1.1 * q.vertex2.2 + q.vertex2.1 * q.vertex3.2 + 
    q.vertex3.1 * q.vertex4.2 + q.vertex4.1 * q.vertex1.2 - 
    (q.vertex2.1 * q.vertex1.2 + q.vertex3.1 * q.vertex2.2 + 
     q.vertex4.1 * q.vertex3.2 + q.vertex1.1 * q.vertex4.2)
  )

/-- The theorem stating that if the area is 1, then n must be 0 -/
theorem exp_quadrilateral_area_one_implies_n_zero (q : ExpQuadrilateral) 
  (h1 : q.vertex1 = (q.n, Real.exp q.n))
  (h2 : q.vertex2 = (q.n + 0.5, Real.exp (q.n + 0.5)))
  (h3 : q.vertex3 = (q.n + 1, Real.exp (q.n + 1)))
  (h4 : q.vertex4 = (q.n + 1.5, Real.exp (q.n + 1.5)))
  (h_area : q.area = 1)
  (h_calc_area : q.area = calculateArea q) :
  q.n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_quadrilateral_area_one_implies_n_zero_l697_69786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l697_69738

noncomputable def sequence_a (n : ℕ+) : ℝ := 2 * n.val - 1

noncomputable def partial_sum_S (n : ℕ+) : ℝ := n.val^2

noncomputable def sequence_b (n : ℕ+) : ℝ := 
  (sequence_a (n + 2)) / (sequence_a n * sequence_a (n + 1) * 2^n.val)

noncomputable def partial_sum_T (n : ℕ+) : ℝ := 1 - 1 / (2^n.val * (2 * n.val + 1))

theorem sequence_properties :
  (∀ n : ℕ+, sequence_a n + 1 = 2 * (partial_sum_S n).sqrt) →
  (∀ n : ℕ+, sequence_a n = 2 * n.val - 1) ∧
  (∀ n : ℕ+, 5/6 ≤ partial_sum_T n ∧ partial_sum_T n < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l697_69738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l697_69724

noncomputable def f : ℝ → ℝ := fun x ↦ -x^2 + 2*x + 15

noncomputable def g (m : ℝ) : ℝ → ℝ := fun x ↦ (1 - 2*m)*x - f x

noncomputable def g_min (m : ℝ) : ℝ :=
  if m ≤ -1/2 then -15
  else if m < 3/2 then -m^2 - m - 61/4
  else -4*m - 13

theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = -2*x + 1) ∧
  f 2 = 15 ∧
  (∀ x, f x = -x^2 + 2*x + 15) ∧
  (∀ m, IsGreatest { y | ∃ x ∈ Set.Icc 0 2, g m x = y } (g_min m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l697_69724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BEDC_is_76_5_l697_69744

/-- Represents a parallelogram ABCD with point E on BC -/
structure Parallelogram where
  -- Base length of the parallelogram
  base : ℝ
  -- Height of the parallelogram
  height : ℝ
  -- Length of BE
  be_length : ℝ
  -- Length of ED
  ed_length : ℝ
  -- Condition that BE + ED equals the base
  base_division : base = be_length + ed_length

/-- Calculates the area of the shaded region BEDC in the given parallelogram -/
noncomputable def area_BEDC (p : Parallelogram) : ℝ :=
  p.base * p.height - (1/2 * p.be_length * p.height)

/-- Theorem stating that the area of BEDC is 76.5 for the given parallelogram -/
theorem area_BEDC_is_76_5 (p : Parallelogram) 
  (h1 : p.base = 12)
  (h2 : p.height = 9)
  (h3 : p.be_length = 7)
  (h4 : p.ed_length = 5) :
  area_BEDC p = 76.5 := by
  sorry

#check area_BEDC_is_76_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BEDC_is_76_5_l697_69744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l697_69793

-- Define the circle ω
noncomputable def ω : Set (ℝ × ℝ) := sorry

-- Define points A, B, and C
def A : ℝ × ℝ := (8, 17)
def B : ℝ × ℝ := (16, 15)
def C : ℝ × ℝ := (7, 0)

-- Define necessary concepts
def IsTangentLine (s : Set (ℝ × ℝ)) (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry
def Line.throughPoints (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem circle_area (hA : A ∈ ω) (hB : B ∈ ω) 
  (hC : C ∉ ω) 
  (hTangentA : IsTangentLine ω A (Line.throughPoints A C))
  (hTangentB : IsTangentLine ω B (Line.throughPoints B C)) :
  Area ω = 4930 * Real.pi / 281 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l697_69793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l697_69739

/-- Proves that for a hyperbola with given properties, its asymptotes have a specific equation -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := (13 : ℝ).sqrt / 2
  e ^ 2 * a ^ 2 = a ^ 2 + b ^ 2 →
  (∀ x y, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1) →
  ∃ k, k = 3 / 2 ∧ (∀ x y, y = k * x ∨ y = -k * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l697_69739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_over_one_minus_tan_22_5_squared_l697_69729

theorem tan_22_5_over_one_minus_tan_22_5_squared (θ : ℝ) (h : θ = 22.5 * π / 180) :
  Real.tan θ / (1 - Real.tan θ ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_over_one_minus_tan_22_5_squared_l697_69729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l697_69774

/-- A function f that is odd and satisfies f(-1) = -2 -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 + 1) / (a * x + b)

theorem f_properties (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) →  -- f is odd
  f a b (-1) = -2 →                 -- f(-1) = -2
  (∀ x, f a b x = x + 1/x) ∧        -- Analytical expression
  StrictMono (fun x ↦ f a b x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l697_69774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_picnic_l697_69755

/-- Represents the percentage of employees who are men -/
noncomputable def percentMen : ℝ := 0.55

/-- Represents the percentage of all employees who attended the picnic -/
noncomputable def percentTotalAttended : ℝ := 0.29

/-- Represents the percentage of women who attended the picnic -/
noncomputable def percentWomenAttended : ℝ := 0.4

/-- Represents the percentage of men who attended the picnic -/
noncomputable def percentMenAttended : ℝ := (percentTotalAttended - (1 - percentMen) * percentWomenAttended) / percentMen

theorem company_picnic :
  percentMenAttended = 0.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_picnic_l697_69755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_pyramid_volume_l697_69753

/-- The volume of a pyramid with square base -/
noncomputable def pyramid_volume (s : ℝ) (h : ℝ) : ℝ := (1/3) * s^2 * h

theorem modified_pyramid_volume :
  ∀ s h : ℝ,
  s > 0 → h > 0 →
  pyramid_volume s h = 60 →
  pyramid_volume (3*s) (2*h) = 1080 :=
by
  intros s h hs hh hv
  unfold pyramid_volume at *
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_pyramid_volume_l697_69753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_four_thirds_l697_69760

noncomputable def surface (x y : ℝ) : ℝ := x * y^2

noncomputable def volume : ℝ := ∫ x in Set.Icc 0 1, ∫ y in Set.Icc 0 2, surface x y

theorem volume_equals_four_thirds : volume = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_four_thirds_l697_69760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l697_69721

/-- The distance between two points M₁(-1, 0, 2) and M₂(0, 3, 1) in three-dimensional space is √11 -/
theorem distance_between_points : 
  let M₁ : Fin 3 → ℝ := ![-1, 0, 2]
  let M₂ : Fin 3 → ℝ := ![0, 3, 1]
  Real.sqrt ((M₂ 0 - M₁ 0)^2 + (M₂ 1 - M₁ 1)^2 + (M₂ 2 - M₁ 2)^2) = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l697_69721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_no_negative_integer_a_with_positive_max_exists_max_and_min_l697_69748

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (a * Real.exp x) / x + x

-- Part 1
theorem tangent_line_condition :
  (∃ m : ℝ, m * (0 - 1) + f a 1 = -1) → a = -1 / Real.exp 1 := by
  sorry

-- Part 2
theorem no_negative_integer_a_with_positive_max :
  ¬ ∃ (a : ℤ), a < 0 ∧ ∃ (x : ℝ), (∀ y : ℝ, f a y ≤ f a x) ∧ f a x > 0 := by
  sorry

-- Part 3
theorem exists_max_and_min (h : a > 0) :
  ∃ (x_max x_min : ℝ),
    (∀ y : ℝ, f a y ≤ f a x_max) ∧
    (∀ y : ℝ, f a y ≥ f a x_min) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_no_negative_integer_a_with_positive_max_exists_max_and_min_l697_69748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_345_ratio_l697_69789

/-- A triangle with sides in the ratio 3:4:5 is a right triangle -/
theorem right_triangle_345_ratio (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k) : 
  a^2 + b^2 = c^2 := by
  cases' h_ratio with k hk
  have h1 : a = 3*k := hk.2.1
  have h2 : b = 4*k := hk.2.2.1
  have h3 : c = 5*k := hk.2.2.2
  calc
    a^2 + b^2 = (3*k)^2 + (4*k)^2 := by rw [h1, h2]
    _         = 9*k^2 + 16*k^2    := by ring
    _         = 25*k^2            := by ring
    _         = (5*k)^2           := by ring
    _         = c^2               := by rw [h3]

#check right_triangle_345_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_345_ratio_l697_69789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangles_intersection_l697_69709

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points on the sides of ABC
variable (C1 C2 B1 B2 A1 A2 : ℝ × ℝ)

-- Define the centroid of a triangle
def centroid (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Define a line segment
def line_segment (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {R | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))}

-- Define the theorem
theorem inscribed_triangles_intersection
  (h_C1 : C1 ∈ line_segment A B) (h_C2 : C2 ∈ line_segment A B)
  (h_B1 : B1 ∈ line_segment A C) (h_B2 : B2 ∈ line_segment A C)
  (h_A1 : A1 ∈ line_segment B C) (h_A2 : A2 ∈ line_segment B C)
  (h_centroid : centroid A1 B1 C1 = centroid A2 B2 C2) :
  (∃ P, P ∈ line_segment A1 B1 ∧ P ∈ line_segment A2 B2) ∧
  (∃ Q, Q ∈ line_segment B1 C1 ∧ Q ∈ line_segment B2 C2) ∧
  (∃ R, R ∈ line_segment C1 A1 ∧ R ∈ line_segment C2 A2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangles_intersection_l697_69709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l697_69763

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalAreaPyramid (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge ^ 2 - (baseEdge / 2) ^ 2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid 
    with base edges of 8 units and lateral edges of 10 units is equal to 32√21 square units -/
theorem pyramid_area_theorem : totalAreaPyramid 8 10 = 32 * Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l697_69763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_area_range_l697_69700

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_pos : 0 < r

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The area of a quadrilateral formed by certain intersections with the ellipse and circle -/
noncomputable def quadrilateral_area (e : Ellipse) (c : Circle) : ℝ → ℝ := sorry

theorem ellipse_circle_area_range (e : Ellipse) (c : Circle) :
  (∀ x y, x^2 / e.a^2 + y^2 / e.b^2 = 1 → x = 1 → y = eccentricity e) →
  c.r = e.a →
  (∃ x y, x^2 + y^2 = (2 * c.r)^2 ∧ x - y + 2 = 0) →
  (∀ S, ∃ k, quadrilateral_area e c k = S) →
  (∀ S, 2 ≤ S ∧ S ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_area_range_l697_69700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verbose_language_word_count_l697_69791

/-- The number of letters in the Verbose alphabet -/
def alphabet_size : ℕ := 24

/-- The maximum word length in the Verbose language -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of a given length that contain 'A' at least once -/
def words_with_a (length : ℕ) : ℕ :=
  alphabet_size ^ length - (alphabet_size - 1) ^ length

/-- The total number of valid words in the Verbose language -/
def total_words : ℕ :=
  (List.range max_word_length).map (λ i => words_with_a (i + 1)) |>.sum

/-- Theorem stating the total number of words in the Verbose language -/
theorem verbose_language_word_count :
  total_words = 1580921 := by
  sorry

#eval total_words

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verbose_language_word_count_l697_69791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l697_69718

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The result of the calculation rounded to the nearest hundredth -/
noncomputable def calculationResult : ℝ :=
  roundToHundredth (53.463 * 12.9873 + 10.253)

theorem calculation_proof : calculationResult = 705.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l697_69718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_money_left_l697_69713

/-- Calculates the amount of money James has left after paying for parking tickets. -/
def money_left (initial_balance : ℚ) (ticket_cost : ℚ) (num_full_cost_tickets : ℕ) 
  (fourth_ticket_fraction : ℚ) (fifth_ticket_fraction : ℚ) (roommate_contribution_fraction : ℚ) : ℚ :=
  let total_cost := ticket_cost * (num_full_cost_tickets : ℚ) + 
    ticket_cost * fourth_ticket_fraction + 
    ticket_cost * fifth_ticket_fraction
  let roommate_contribution := total_cost * roommate_contribution_fraction
  let james_payment := total_cost - roommate_contribution
  initial_balance - james_payment

/-- Theorem stating that James has $500 left after paying for parking tickets. -/
theorem james_money_left : 
  money_left 800 200 3 (1/4) (1/2) (3/5) = 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_money_left_l697_69713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_volume_ratio_l697_69714

/-- The ratio of surface area to volume of the remaining solid after cutting off smaller tetrahedra from each vertex of a unit regular tetrahedron -/
noncomputable def surface_volume_ratio (n : ℝ) : ℝ :=
  (1 - 2 / n^2) / (1 - 4 / n^3)

/-- The theorem stating that the surface area to volume ratio is minimized when n = 3 -/
theorem min_surface_volume_ratio :
  ∀ n : ℝ, n > 1 → surface_volume_ratio 3 ≤ surface_volume_ratio n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_volume_ratio_l697_69714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l697_69710

noncomputable def a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

noncomputable def b : ℝ × ℝ := (-1/2, Real.sqrt 3/2)

theorem vector_problem (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ 2*Real.pi) 
  (h2 : ∃ (k : ℝ), a α ≠ k • b) :
  (∃ (x y : ℝ × ℝ), x = a α + b ∧ y = a α - b ∧ x.1 * y.1 + x.2 * y.2 = 0) ∧
  (‖(Real.sqrt 3 • a α + b)‖ = ‖(a α - Real.sqrt 3 • b)‖ → 
    α = Real.pi/6 ∨ α = 7*Real.pi/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l697_69710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l697_69737

noncomputable def short_distance (x y : ℝ) : ℝ := min (abs x) (abs y)

def equidistant (x1 y1 x2 y2 : ℝ) : Prop :=
  short_distance x1 y1 = short_distance x2 y2

theorem problem_solution :
  -- Part 1
  short_distance (-5) (-2) = 2 ∧
  -- Part 2
  (∀ m : ℝ, short_distance (-2) (-2*m+1) = 1 ↔ m = 0 ∨ m = 1) ∧
  -- Part 3
  (∀ k : ℝ, equidistant (-1) (k+3) 4 (2*k-3) ↔ k = 1 ∨ k = 2) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l697_69737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l697_69785

/-- The function g(x) with parameter k -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - 3*x - 18)

/-- Theorem stating the condition for g(x) to have exactly one vertical asymptote -/
theorem g_one_vertical_asymptote (k : ℝ) :
  (∃! x, (x^2 - 3*x - 18 = 0 ∧ x^2 + 3*x + k ≠ 0)) ↔ (k = -54 ∨ k = 0) := by
  sorry

#check g_one_vertical_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l697_69785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copy_pages_problem_l697_69703

/-- The maximum number of pages that can be copied given a cost per page and a budget -/
def max_pages_copied (cost_per_page : ℚ) (budget : ℚ) : ℕ :=
  (budget * 100 / cost_per_page).floor.toNat

/-- Theorem: Given a cost of 5 cents per page and a budget of $50, 
    the maximum number of pages that can be copied is 1000 -/
theorem copy_pages_problem :
  max_pages_copied (5 / 100) 50 = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_copy_pages_problem_l697_69703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_valid_solutions_l697_69715

/-- A function that checks if a set of four positive integers satisfies the given condition -/
def satisfies_condition (a b c d : ℕ) : Prop :=
  (b * c * d) % a = 1 ∧
  (a * c * d) % b = 1 ∧
  (a * b * d) % c = 1 ∧
  (a * b * c) % d = 1

/-- The theorem stating the only valid solutions -/
theorem only_valid_solutions :
  ∀ a b c d : ℕ,
    a > 0 → b > 0 → c > 0 → d > 0 →
    satisfies_condition a b c d ↔ 
      ((a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
       (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13)) :=
by
  sorry

#check only_valid_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_valid_solutions_l697_69715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_relatively_prime_l697_69708

/-- Defines the polynomial P_n(x) = 1 + 2x + 3x^2 + ... + nx^(n-1) -/
noncomputable def P (n : ℕ+) : Polynomial ℚ :=
  (Finset.range n).sum (fun i => (↑(i + 1) : ℚ) • (Polynomial.X : Polynomial ℚ) ^ i)

/-- Theorem stating that P_j(x) and P_k(x) are relatively prime for distinct positive integers j and k -/
theorem P_relatively_prime (j k : ℕ+) (h : j ≠ k) : IsCoprime (P j) (P k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_relatively_prime_l697_69708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_length_l697_69746

/-- A circle in the 2D plane --/
class IsCircle (S : Set (ℝ × ℝ)) : Prop

/-- A line is tangent to a circle at a point --/
def IsTangentAt (S : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Given a triangle DEF with side lengths DE = 8, EF = 10, and FD = 6,
    and circles ω3 and ω4 as described, prove that DL = √2 / 2 --/
theorem triangle_circle_intersection_length 
  (D E F L : ℝ × ℝ) 
  (ω3 ω4 : Set (ℝ × ℝ)) :
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let fd := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  let dl := Real.sqrt ((L.1 - D.1)^2 + (L.2 - D.2)^2)
  IsCircle ω3 → IsCircle ω4 →
  de = 8 → ef = 10 → fd = 6 →
  E ∈ ω3 → F ∈ ω4 →
  IsTangentAt ω3 {p : ℝ × ℝ | ∃ t, p = (1-t) • F + t • D} D →
  IsTangentAt ω4 {p : ℝ × ℝ | ∃ t, p = (1-t) • D + t • E} D →
  L ∈ ω3 ∩ ω4 → L ≠ D →
  dl = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_length_l697_69746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_is_1500_l697_69754

/-- Represents the temperature change rate with respect to altitude -/
noncomputable def temp_change_rate : ℝ := 6 / 1000

/-- Represents the ground temperature in °C -/
noncomputable def ground_temp : ℝ := 8

/-- Represents the high altitude temperature in °C -/
noncomputable def high_altitude_temp : ℝ := -1

/-- Calculates the height of the hot air balloon above the ground -/
noncomputable def balloon_height : ℝ := (ground_temp - high_altitude_temp) / temp_change_rate

/-- Theorem stating that the height of the hot air balloon is 1500m -/
theorem balloon_height_is_1500 : balloon_height = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_is_1500_l697_69754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_given_chords_l697_69722

-- Define a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of a chord
def isChord (c : Circle) (l : Line) : Prop :=
  sorry -- Placeholder for the actual definition

-- Define the length of a chord
def chordLength (c : Circle) (l : Line) : ℝ :=
  sorry -- Placeholder for the actual definition

-- Main theorem
theorem circle_with_given_chords 
  (a b c : Line) (d : ℝ) :
  ∃ (circles : Finset Circle), 
    (∀ circ ∈ circles, 
      (isChord circ a ∧ chordLength circ a = d) ∧
      (isChord circ b ∧ chordLength circ b = d) ∧
      (isChord circ c ∧ chordLength circ c = d)) ∧
    circles.card ≤ 4 := by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_given_chords_l697_69722
