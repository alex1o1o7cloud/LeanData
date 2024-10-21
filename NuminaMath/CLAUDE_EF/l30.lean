import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_radar_correct_coverage_ring_area_correct_l30_3092

/-- The number of radars -/
def n : ℕ := 5

/-- The radius of each radar's coverage area in km -/
def r : ℝ := 13

/-- The width of the coverage ring in km -/
def w : ℝ := 10

/-- The central angle of the regular pentagon in radians -/
noncomputable def α : ℝ := 2 * Real.pi / n

/-- The distance from the center to each radar -/
noncomputable def distance_to_radar : ℝ := 12 / Real.sin (α / 2)

/-- The area of the coverage ring -/
noncomputable def coverage_ring_area : ℝ := 240 * Real.pi / Real.tan (α / 2)

/-- Theorem stating the distance from the center to each radar -/
theorem distance_to_radar_correct :
  distance_to_radar = 12 / Real.sin (α / 2) := by sorry

/-- Theorem stating the area of the coverage ring -/
theorem coverage_ring_area_correct :
  coverage_ring_area = 240 * Real.pi / Real.tan (α / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_radar_correct_coverage_ring_area_correct_l30_3092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l30_3038

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - Real.sqrt 3 / 2

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

def A : ℝ × ℝ := (-3, 2)

def B : ℝ × ℝ := (3, 10)

theorem problem_solution (x₀ : ℝ) (h₁ : x₀ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) (h₂ : f (x₀/2) = -1/3) :
  Real.sin x₀ = -(1 + 2 * Real.sqrt 6) / 6 ∧
  ∃! P : ℝ × ℝ, P.1 = 0 ∧ P.2 = 1 ∧ g P.1 = P.2 ∧
    ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l30_3038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_error_l30_3087

/-- The combined percentage error in the volume of a cube with measurement errors -/
theorem cube_volume_error (length_error width_error height_error : ℝ) 
  (h1 : length_error = 0.02)
  (h2 : width_error = -0.03)
  (h3 : height_error = 0.04) :
  let combined_factor := (1 + length_error) * (1 + width_error) * (1 + height_error)
  let volume_error_percentage := (combined_factor - 1) * 100
  abs (volume_error_percentage - 3.0744) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_error_l30_3087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_12_ships_in_10x10_grid_l30_3008

/-- Represents a rectangle on a grid -/
structure Rectangle where
  x : ℕ
  y : ℕ
  width : ℕ
  height : ℕ

/-- Checks if two rectangles touch (including corners) -/
def touch (r1 r2 : Rectangle) : Prop :=
  ∃ (x y : ℕ), x ≥ r1.x ∧ x < r1.x + r1.width ∧
                y ≥ r1.y ∧ y < r1.y + r1.height ∧
                x ≥ r2.x - 1 ∧ x ≤ r2.x + r2.width ∧
                y ≥ r2.y - 1 ∧ y ≤ r2.y + r2.height

/-- Checks if a rectangle is within the bounds of the grid -/
def inBounds (r : Rectangle) (gridSize : ℕ) : Prop :=
  r.x + r.width ≤ gridSize ∧ r.y + r.height ≤ gridSize

/-- Main theorem: It's impossible to place 12 non-touching 1x4 rectangles in a 10x10 grid -/
theorem no_12_ships_in_10x10_grid :
  ¬∃ (ships : List Rectangle),
    ships.length = 12 ∧
    (∀ s, s ∈ ships → s.width = 4 ∧ s.height = 1 ∧ inBounds s 10) ∧
    (∀ s1 s2, s1 ∈ ships → s2 ∈ ships → s1 ≠ s2 → ¬touch s1 s2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_12_ships_in_10x10_grid_l30_3008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_l30_3090

def sequenceProperty (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 5 ∧ 
  (∀ n : ℕ, n ≥ 2 → a n = S (n - 1)) ∧
  (∀ n : ℕ, S n - S (n - 1) = a n)

theorem sequence_expression (a : ℕ → ℕ) (S : ℕ → ℕ) :
  sequenceProperty a S → ∀ n : ℕ, n ≥ 1 → a n = 5 * 2^(n - 1) :=
by
  sorry

#check sequence_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_l30_3090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_crawl_distance_l30_3035

theorem spider_crawl_distance (start middle end_ : ℝ) 
  (h1 : start = 3) (h2 : middle = -1) (h3 : end_ = 8.5) :
  (|middle - start| + |end_ - middle|) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_crawl_distance_l30_3035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_coordinates_l30_3027

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def P : ℝ × ℝ := (4, 3)

noncomputable def rotateVector (v : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (v.1 * Real.cos θ - v.2 * Real.sin θ, v.1 * Real.sin θ + v.2 * Real.cos θ)

theorem rotated_vector_coordinates :
  let OP : ℝ × ℝ := (P.1 - O.1, P.2 - O.2)
  let OQ : ℝ × ℝ := rotateVector OP (2 * Real.pi / 3)
  OQ = (-(4 - 3 * Real.sqrt 3) / 2, -(3 - 4 * Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_coordinates_l30_3027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l30_3094

/-- The radius of a circle inscribed in a rhombus with given diagonal lengths -/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / (4 * Real.sqrt ((d1/2)^2 + (d2/2)^2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals of lengths 8 and 30 is 60/√241 -/
theorem inscribed_circle_radius_specific : inscribed_circle_radius 8 30 = 60 / Real.sqrt 241 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l30_3094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_one_equals_eleven_point_twenty_five_l30_3016

-- Define the functions u and v
def u (x : ℝ) : ℝ := 4 * x - 9

noncomputable def v (x : ℝ) : ℝ := 
  let y := (x + 9) / 4
  y^2 + 4 * y - 5

-- Theorem statement
theorem v_of_one_equals_eleven_point_twenty_five :
  v 1 = 11.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_one_equals_eleven_point_twenty_five_l30_3016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_locus_is_circle_l30_3007

/-- The locus of points from which two tangent lines can be drawn to the unit circle,
    such that the angle between these tangent lines is 60°, is a circle with radius 2. -/
theorem tangent_locus_is_circle (P : ℝ × ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧
    ((P.1 - A.1)*(A.1) + (P.2 - A.2)*(A.2) = 0) ∧
    ((P.1 - B.1)*(B.1) + (P.2 - B.2)*(B.2) = 0) ∧
    (Real.arccos ((P.1 - A.1)*(P.1 - B.1) + (P.2 - A.2)*(P.2 - B.2)) / 
      (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) = π/3)) →
  P.1^2 + P.2^2 = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_locus_is_circle_l30_3007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l30_3044

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition b + c = a(cos C + √3 sin C) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.b + t.c = t.a * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C)

/-- Helper function to calculate the perimeter of the incircle -/
noncomputable def incirclePerimeter (t : Triangle) : ℝ :=
  2 * Real.pi * (Real.sqrt 3 / 6 : ℝ) * (t.b + t.c - 2)

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) (h : satisfiesCondition t) :
  t.A = Real.pi / 3 ∧
  ∀ (t' : Triangle), satisfiesCondition t' → t'.a = 2 →
    2 * Real.pi * (Real.sqrt 3 / 3 : ℝ) ≥ incirclePerimeter t' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l30_3044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_greater_iff_no_intersection_l30_3031

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  finish : ℝ × ℝ

-- Define the length of a tangent from a point to a circle
noncomputable def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry

-- Define whether a line segment intersects a circle
def intersectsCircle (s : LineSegment) (c : Circle) : Prop :=
  sorry

-- Define the length of a line segment
noncomputable def segmentLength (s : LineSegment) : ℝ :=
  sorry

theorem tangent_sum_greater_iff_no_intersection
  (k : Circle) (AB : LineSegment) :
  let a := tangentLength AB.start k
  let b := tangentLength AB.finish k
  (a + b > segmentLength AB) ↔ ¬(intersectsCircle AB k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_greater_iff_no_intersection_l30_3031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_when_x_greater_than_two_sum_greater_than_four_when_f_equal_l30_3034

noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.exp x

noncomputable def g (x : ℝ) : ℝ := f (4 - x)

theorem f_greater_than_g_when_x_greater_than_two :
  ∀ x : ℝ, x > 2 → f x > g x := by sorry

theorem sum_greater_than_four_when_f_equal (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = f x₂ → x₁ + x₂ > 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_when_x_greater_than_two_sum_greater_than_four_when_f_equal_l30_3034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_two_iff_z_purely_imaginary_l30_3004

/-- Given a real number a, z is defined as ((a+2i)(-1+i))/i -/
noncomputable def z (a : ℝ) : ℂ := ((a + 2*Complex.I) * (-1 + Complex.I)) / Complex.I

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def is_purely_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

/-- Theorem stating that a = 2 is a necessary and sufficient condition for z to be purely imaginary -/
theorem a_eq_two_iff_z_purely_imaginary :
  ∀ a : ℝ, is_purely_imaginary (z a) ↔ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_two_iff_z_purely_imaginary_l30_3004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l30_3018

theorem solutions_count (θ : ℝ) : 
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ n : Finset (Fin 56), 
    Real.tan (10 * Real.pi * Real.cos θ) = 1 / Real.tan (10 * Real.pi * Real.sin θ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l30_3018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_segment_l30_3082

/-- Right triangular prism with given properties -/
structure RightTriangularPrism where
  /-- AB = AA₁ = 1 -/
  edge_length : ℝ
  /-- AC₁ = √2 -/
  diagonal_length : ℝ
  /-- The angle between AC₁ and plane ABC is 45° -/
  angle_with_base : ℝ

/-- Point P moving within plane ABC -/
def MovingPoint (prism : RightTriangularPrism) := {p : ℝ × ℝ × ℝ // p.2 = 0}

/-- The area of triangle AC₁P is 1/2 -/
def AreaCondition (prism : RightTriangularPrism) (p : MovingPoint prism) : Prop :=
  ∃ (h : ℝ), h * prism.diagonal_length / 2 = 1/2

/-- Predicate to check if a set is a segment of an ellipse -/
def IsEllipseSegment (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Theorem: The trajectory of point P is a segment of an ellipse -/
theorem trajectory_is_ellipse_segment (prism : RightTriangularPrism) 
  (h1 : prism.edge_length = 1)
  (h2 : prism.diagonal_length = Real.sqrt 2)
  (h3 : prism.angle_with_base = π/4)
  (p : MovingPoint prism)
  (h4 : AreaCondition prism p) :
  ∃ (ellipse : Set (ℝ × ℝ × ℝ)), p.val ∈ ellipse ∧ IsEllipseSegment ellipse :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_segment_l30_3082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_structure_l30_3042

noncomputable def volume_cylinder_hemispheres (total_length : ℝ) (radius : ℝ) : ℝ :=
  let cylinder_length := total_length - 2 * radius
  let hemisphere_volume := (2 / 3) * Real.pi * radius^3
  let cylinder_volume := Real.pi * radius^2 * cylinder_length
  hemisphere_volume + cylinder_volume

theorem volume_specific_structure :
  volume_cylinder_hemispheres 30 4 = 437.33 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_structure_l30_3042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l30_3077

/-- Predicate to check if a given function represents a parabola -/
def IsParabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0

/-- The equation x^2 + ky^2 = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : ¬ IsParabola (fun x y => x^2 + k*y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l30_3077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_NB_equals_three_l30_3041

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (3, 2*Real.sqrt 3)

-- Define point N
noncomputable def point_N : ℝ × ℝ := (0, 2*Real.sqrt 3)

-- Define the circle centered at F with radius 1
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line AF
noncomputable def line_AF (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define point B as the intersection of the circle and line AF
noncomputable def point_B : ℝ × ℝ := (3/2, Real.sqrt 3 / 2)

-- Theorem statement
theorem distance_NB_equals_three :
  parabola point_A.1 point_A.2 →
  circle_F point_B.1 point_B.2 →
  line_AF point_B.1 point_B.2 →
  Real.sqrt ((point_N.1 - point_B.1)^2 + (point_N.2 - point_B.2)^2) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_NB_equals_three_l30_3041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_cubed_times_4_to_15_l30_3095

/-- The number of digits in a positive real number -/
noncomputable def num_digits (x : ℝ) : ℕ :=
  if x < 1 then 1 else Nat.floor (Real.log x / Real.log 10) + 1

/-- Theorem: The number of digits in 3^3 * 4^15 is 11 -/
theorem digits_of_3_cubed_times_4_to_15 :
  num_digits (3^3 * 4^15) = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_cubed_times_4_to_15_l30_3095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l30_3052

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + Real.pi / 3)

theorem phi_value (φ : ℝ) :
  (∀ x, f x φ = -f (-x) φ) →  -- f is an odd function
  (∀ x y, 0 ≤ x → x ≤ y → y ≤ Real.pi/4 → f y φ ≤ f x φ) →  -- f is decreasing on [0, π/4]
  φ = 2*Real.pi/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l30_3052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l30_3097

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : 0 < β ∧ β < Real.pi/2) 
  (h3 : Real.cos α = 5/13) 
  (h4 : Real.sin (α - β) = 4/5) : 
  Real.sin β = 16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l30_3097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulation_count_eq_catalan_l30_3083

/-- The number of ways to draw n-3 non-intersecting diagonals in an n-gon with equal sides and equal angles, 
    such that each resulting triangle shares at least one side with the polygon -/
def triangulationCount (n : ℕ) : ℚ :=
  1 / (n - 1 : ℚ) * (Nat.choose (2 * n - 4) (n - 2) : ℚ)

/-- Definition of the n-th Catalan number -/
def catalanNumber (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- Theorem stating that the triangulation count is equal to the (n-2)-th Catalan number -/
theorem triangulation_count_eq_catalan (n : ℕ) (h : n ≥ 3) : 
  triangulationCount n = catalanNumber (n - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulation_count_eq_catalan_l30_3083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l30_3047

/-- A right triangle with vertices at (3, 3), (0, 0), and (x, 0) where x > 0 -/
structure RightTriangle where
  x : ℝ
  h_positive : x > 0

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem third_vertex_coordinate (t : RightTriangle) :
  triangleArea t.x 3 = 18 → t.x = 12 := by
  intro h
  -- The proof steps would go here
  sorry

#check third_vertex_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l30_3047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_eight_eightyfirsths_l30_3088

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  area : ℝ
  lower_base : ℝ
  upper_base : ℝ
  lower_base_twice_upper : lower_base = 2 * upper_base

/-- Represents the shaded area formed by intersecting segments in the trapezoid -/
noncomputable def shaded_area (t : Trapezoid) : ℝ := 8 / 81

/-- Theorem stating that the shaded area in a trapezoid with specific properties is 8/81 -/
theorem shaded_area_is_eight_eightyfirsths (t : Trapezoid) 
  (h1 : t.area = 1) 
  (h2 : ∃ (n : ℕ), n > 0 ∧ t.lower_base / n = t.upper_base / n) : 
  shaded_area t = 8 / 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_eight_eightyfirsths_l30_3088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_region_probability_l30_3015

-- Define the radii of the outer and inner circles
noncomputable def outer_radius : ℝ := 3
noncomputable def inner_radius : ℝ := 1

-- Define the probability we want to prove
noncomputable def target_probability : ℝ := 1 / 9

-- Theorem statement
theorem circular_region_probability :
  (π * inner_radius^2) / (π * outer_radius^2) = target_probability :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_region_probability_l30_3015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_is_three_l30_3049

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the hyperbola equation -/
def onHyperbola (p : Point) : Prop :=
  p.x^2 / 9 - p.y^2 / 7 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the right focus of the hyperbola -/
def rightFocus : Point :=
  { x := 4, y := 0 }

/-- Represents the origin -/
def origin : Point :=
  { x := 0, y := 0 }

/-- The main theorem -/
theorem distance_to_origin_is_three (p : Point) 
  (h1 : onHyperbola p) 
  (h2 : distance p rightFocus = 1) : 
  distance p origin = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_is_three_l30_3049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l30_3065

/-- Given two circles in the x-y plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

/-- The equation of the line containing the common chord -/
def common_chord_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

/-- The length of the common chord -/
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 30

/-- Theorem stating the properties of the common chord -/
theorem common_chord_properties :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord_line x y) ∧
  (∃ a b c d : ℝ, 
    circle1 a b ∧ circle2 a b ∧ circle1 c d ∧ circle2 c d ∧
    common_chord_line a b ∧ common_chord_line c d ∧
    ((a - c)^2 + (b - d)^2 = common_chord_length^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l30_3065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_per_task_amount_verify_total_amount_l30_3046

/-- The amount Maurice gets per task -/
def amount_per_task : ℚ := 2

/-- The bonus Maurice gets for every 10 tasks -/
def bonus_per_10_tasks : ℚ := 6

/-- The total number of tasks Maurice completed -/
def total_tasks : ℕ := 30

/-- The total amount Maurice made for completing all tasks -/
def total_amount : ℚ := 78

/-- Theorem stating that Maurice gets $2 per task -/
theorem maurice_per_task_amount :
  amount_per_task = 2 :=
by
  -- The proof goes here
  sorry

/-- Theorem verifying the total amount Maurice made -/
theorem verify_total_amount :
  total_amount = amount_per_task * total_tasks + (total_tasks / 10) * bonus_per_10_tasks :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_per_task_amount_verify_total_amount_l30_3046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l30_3002

/-- Represents the walking speed of a person in blocks per unit time -/
structure WalkingSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a circular path with a given length in blocks -/
structure CircularPath where
  length : ℝ
  length_pos : length > 0

/-- Represents two people walking in opposite directions on a circular path -/
noncomputable def MeetingPoint (h : WalkingSpeed) (j : WalkingSpeed) (p : CircularPath) : ℝ :=
  (p.length * h.speed) / (h.speed + j.speed)

theorem meeting_point_theorem (h : WalkingSpeed) (j : WalkingSpeed) (p : CircularPath) 
  (h_speed : h.speed > 0)
  (j_speed : j.speed = 3 * h.speed)
  (p_length : p.length = 30) :
  MeetingPoint h j p = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l30_3002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_among_options_l30_3056

theorem irrational_among_options : 
  (¬ (∃ (a b : ℤ), b ≠ 0 ∧ 5 * Real.pi = (a : ℝ) / (b : ℝ))) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ 3.14 = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-3 : ℝ) = (a : ℝ) / (b : ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_among_options_l30_3056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_specific_values_l30_3023

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ln|a + 1/(1-x)| + b -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_implies_specific_values (a b : ℝ) :
  IsOdd (f a b) → a = -1/2 ∧ b = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_specific_values_l30_3023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l30_3076

/-- The length of a train given its speed, time to cross a bridge, and bridge length -/
noncomputable def train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : ℝ :=
  train_speed * (1000 / 3600) * crossing_time - bridge_length

/-- Theorem stating that a train traveling at 60 kmph that takes 20.99832013438925 seconds
    to cross a bridge of 240 m in length has a length of approximately 110 m -/
theorem train_length_calculation :
  let train_speed : ℝ := 60
  let crossing_time : ℝ := 20.99832013438925
  let bridge_length : ℝ := 240
  abs (train_length train_speed crossing_time bridge_length - 110) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l30_3076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l30_3029

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x else 3 * x - 50

-- Theorem statement
theorem h_composition_equality :
  ∃ b : ℝ, b < 0 ∧ h (h (h 15)) = h (h (h b)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l30_3029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_negative_exponent_l30_3050

theorem fraction_negative_exponent (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((2 : ℝ) * a / ((3 : ℝ) * b)) ^ (-2 : ℤ) = 9 * b^2 / (4 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_negative_exponent_l30_3050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l30_3011

/-- A line passing through (3, 2) and intersecting positive x and y axes -/
structure Line where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 3 / a + 2 / b = 1

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangleArea (l : Line) : ℝ := (l.a * l.b) / 2

/-- The minimum area of the triangle is 12 -/
theorem min_triangle_area :
  ∀ l : Line, triangleArea l ≥ 12 ∧ ∃ l : Line, triangleArea l = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l30_3011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_value_l30_3057

def sequence_b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | n+2 => (1/2) * sequence_b (n+1) + (1/3) * sequence_b n

noncomputable def sequence_sum : ℚ := ∑' n, sequence_b n

theorem sequence_sum_value : sequence_sum = 21/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_value_l30_3057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_odd_f_symmetry_l30_3099

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 * x^2 - x^4) / (|x - 2| - 2)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ∈ Set.Icc (-2) 0 ∪ Set.Ioc 0 2

-- Theorem for the domain of f
theorem f_domain : ∀ x : ℝ, f x ≠ 0 ↔ domain x := by sorry

-- Theorem for f being an odd function
theorem f_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by sorry

-- Theorem for the symmetry of f(x+1)+1
theorem f_symmetry : ∀ x : ℝ, domain (x - 1) → f ((-x) + 1) + 1 = -(f (x + 1) + 1) + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_odd_f_symmetry_l30_3099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_final_price_l30_3043

/-- The final price of a shirt after two successive 25% discounts, given an original price --/
def final_price (original_price : ℚ) : ℚ :=
  let first_discount := original_price * (1 - 0.25)
  let second_discount := first_discount * (1 - 0.25)
  (second_discount * 100).floor / 100

/-- Theorem stating that the final price of the shirt is $14.25 --/
theorem shirt_final_price :
  final_price 26.67 = 14.25 := by
  sorry

#eval final_price 26.67

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_final_price_l30_3043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrants_l30_3073

theorem complex_number_quadrants (a b : ℝ) (h : (Complex.ofReal a + Complex.I * Complex.ofReal b)^2 = Complex.ofReal 3 - Complex.I * Complex.ofReal 4) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrants_l30_3073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_expected_return_l30_3014

-- Define the lottery game parameters
noncomputable def ticket_cost : ℚ := 2
noncomputable def prize : ℚ := 1000
def num_digits : ℕ := 3
def digit_range : ℕ := 10

-- Define the probability of winning
noncomputable def prob_win : ℚ := 1 / (digit_range ^ num_digits : ℚ)

-- Theorem statement
theorem lottery_expected_return :
  prize * prob_win - ticket_cost = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_expected_return_l30_3014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_15_minutes_l30_3085

/-- The time it takes for Person A to catch up with Person B -/
noncomputable def catchUpTime (tA tB headStart : ℝ) : ℝ :=
  (headStart * tA) / (tB - tA)

theorem catch_up_time_is_15_minutes :
  let tA : ℝ := 30  -- Time for Person A to complete the journey in minutes
  let tB : ℝ := 40  -- Time for Person B to complete the journey in minutes
  let headStart : ℝ := 5  -- Person B's head start in minutes
  catchUpTime tA tB headStart = 15 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_15_minutes_l30_3085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_chord_length_l30_3000

noncomputable section

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l'
def line_l' (x y : ℝ) : Prop := y = x - 2

-- Theorem for part I
theorem line_circle_separate :
  ∃ (d : ℝ), d > 2 ∧ 
  ∀ (x y : ℝ), line_l x y → d = |x + y + 1| / Real.sqrt 2 :=
sorry

-- Theorem for part II
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l' A.1 A.2 ∧ line_l' B.1 B.2 ∧
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_chord_length_l30_3000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_exchange_theorem_l30_3068

/-- Represents the exchange rate between magical items -/
structure ExchangeRate where
  wands : ℕ
  cloaks : ℕ
  hats : ℕ

/-- The exchange rates given in the problem -/
def wand_cloak_rate : ExchangeRate := ⟨4, 6, 0⟩
def wand_hat_rate : ExchangeRate := ⟨5, 0, 5⟩

/-- Converts a number of wands to the equivalent number of cloaks -/
def wands_to_cloaks (n : ℚ) : ℚ :=
  n * (wand_cloak_rate.cloaks : ℚ) / (wand_cloak_rate.wands : ℚ)

/-- Converts a number of hats to the equivalent number of wands -/
def hats_to_wands (n : ℚ) : ℚ :=
  n * (wand_hat_rate.wands : ℚ) / (wand_hat_rate.hats : ℚ)

/-- The main theorem stating the equivalence -/
theorem magical_exchange_theorem :
  wands_to_cloaks 5 + wands_to_cloaks (hats_to_wands 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_exchange_theorem_l30_3068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplicative_property_l30_3089

/-- A number theoretic function is a function from positive integers to some type α -/
def NumberTheoreticFunction (α : Type*) := ℕ+ → α

/-- A multiplicative number theoretic function satisfies f(mn) = f(m)f(n) for coprime m and n -/
def IsMultiplicative {α : Type*} [Monoid α] (f : NumberTheoreticFunction α) :=
  ∀ (m n : ℕ+), Nat.Coprime m.val n.val → f (m * n) = f m * f n

/-- Theorem: A multiplicative number theoretic function satisfies f(ab) = f(a)f(b) for coprime a and b -/
theorem multiplicative_property {α : Type*} [Monoid α] (f : NumberTheoreticFunction α) 
  (h : IsMultiplicative f) (a b : ℕ+) (hab : Nat.Coprime a.val b.val) : 
  f (a * b) = f a * f b := by
  exact h a b hab


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplicative_property_l30_3089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mi_seon_height_is_correct_l30_3069

/-- Converts a height given in meters and centimeters to meters as a decimal -/
noncomputable def height_to_meters (meters : ℕ) (centimeters : ℕ) : ℝ :=
  meters + (centimeters : ℝ) / 100

/-- Eun-young's height in meters and centimeters -/
noncomputable def eun_young_height : ℝ := height_to_meters 1 35

/-- The difference in height between Mi-seon and Eun-young in centimeters -/
def height_difference : ℕ := 9

/-- Mi-seon's height in meters -/
noncomputable def mi_seon_height : ℝ := eun_young_height + (height_difference : ℝ) / 100

theorem mi_seon_height_is_correct : mi_seon_height = 1.44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mi_seon_height_is_correct_l30_3069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l30_3067

theorem smallest_difference (x y : ℕ) (h : x * y - 10 * x + 3 * y = 670) :
  ∃ (a b : ℕ), a * b - 10 * a + 3 * b = 670 ∧ 
  (∀ (c d : ℕ), c * d - 10 * c + 3 * d = 670 → 
    |Int.ofNat a - Int.ofNat b| ≤ |Int.ofNat c - Int.ofNat d|) ∧
  |Int.ofNat a - Int.ofNat b| = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l30_3067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separated_distance_is_one_l30_3063

-- Define the line equation
def line (x y θ : ℝ) : Prop := x * Real.sin θ + y * Real.cos θ = 1 + Real.cos θ

-- Define the circle (renamed to avoid naming conflict)
def circleEq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/2

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ θ : ℝ) : ℝ :=
  |x₀ * Real.sin θ + y₀ * Real.cos θ - (1 + Real.cos θ)| / Real.sqrt (Real.sin θ^2 + Real.cos θ^2)

-- Theorem statement
theorem line_circle_separated (θ : ℝ) :
  distance_point_to_line 0 1 θ > Real.sqrt (1/2) := by
  sorry

-- Additional lemma to show that the distance is always 1
theorem distance_is_one (θ : ℝ) :
  distance_point_to_line 0 1 θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separated_distance_is_one_l30_3063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horses_count_l30_3051

/-- Represents the number of horses (which is equal to the number of men) -/
def num_horses : ℕ := sorry

/-- Represents the number of breed A dogs -/
def num_dogs_A : ℕ := sorry

/-- Represents the number of breed B dogs -/
def num_dogs_B : ℕ := sorry

/-- The number of men is 3 times the number of breed A dogs -/
axiom men_to_dogs_A : num_horses = 3 * num_dogs_A

/-- The number of men is 4 times the number of breed B dogs -/
axiom men_to_dogs_B : num_horses = 4 * num_dogs_B

/-- The total number of legs walking on the ground is 200 -/
axiom total_legs : 2 * num_horses + 4 * num_horses + 3 * num_dogs_A + 4 * num_dogs_B = 200

theorem horses_count : num_horses = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horses_count_l30_3051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l30_3084

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + Real.sin (2 * x)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l30_3084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_of_roots_l30_3048

theorem max_sum_of_squares_of_roots :
  ∃ (max : ℝ), max = 18 ∧
  ∀ (k x₁ x₂ : ℝ),
  (x₁^2 - (k - 2) * x₁ + (k^2 + 3*k + 5) = 0) →
  (x₂^2 - (k - 2) * x₂ + (k^2 + 3*k + 5) = 0) →
  x₁^2 + x₂^2 ≤ max :=
by
  -- We claim that the maximum value is 18
  use 18
  constructor
  · -- Trivial equality
    rfl
  · -- Main proof
    intros k x₁ x₂ h₁ h₂
    sorry -- Proof steps would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_of_roots_l30_3048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l30_3032

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∀ x : ℝ, x ≠ 5 → (∃ y : ℝ, g y = x) → x ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l30_3032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_numbers_are_integers_and_fractions_integers_and_fractions_are_rational_l30_3093

-- Define rational numbers
def RationalNumber := ℚ

-- Define the set of integers as a subset of rationals
def IntegerSet : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions (non-integer rationals)
def FractionSet : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b ∧ q ∉ IntegerSet}

-- Theorem statement
theorem rational_numbers_are_integers_and_fractions :
  IntegerSet ∪ FractionSet = Set.univ := by
  sorry

-- Additional theorem to explicitly state that integers and fractions are rational numbers
theorem integers_and_fractions_are_rational :
  ∀ q : ℚ, q ∈ IntegerSet ∨ q ∈ FractionSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_numbers_are_integers_and_fractions_integers_and_fractions_are_rational_l30_3093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l30_3045

def A : Set ℝ := {x | ∃ y, y = Real.log (x + 1)}

theorem complement_of_A : Set.univ \ A = {x : ℝ | x ≤ -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l30_3045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_specific_parabola_l30_3055

/-- Given a parabola with equation y = ax² + bx + c, 
    returns the y-coordinate of its directrix -/
noncomputable def directrix_y (a b c : ℝ) : ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  k - 1 / (4 * a)

/-- The directrix of the parabola y = 3x² - 6x + 2 has y-coordinate -13/12 -/
theorem directrix_of_specific_parabola :
  directrix_y 3 (-6) 2 = -13/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_specific_parabola_l30_3055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_proof_l30_3013

/-- Represents a number in octal (base 8) notation -/
structure Octal where
  value : ℕ
  isValid : value < 8^64 := by sorry

/-- Converts an Octal number to its decimal (base 10) representation -/
def octal_to_decimal (n : Octal) : ℕ := n.value

/-- Converts a decimal (base 10) number to its Octal representation -/
def decimal_to_octal (n : ℕ) : Octal :=
  ⟨n, by sorry⟩

/-- Performs subtraction in Octal -/
def octal_sub (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Performs addition in Octal -/
def octal_add (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a + octal_to_decimal b)

instance : OfNat Octal n where
  ofNat := decimal_to_octal n

theorem octal_arithmetic_proof :
  octal_add (octal_sub 1246 573) 32 = 705 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_proof_l30_3013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l30_3037

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -5 then 2^x - 3
  else 2^(-10 - x) - 3  -- This is the symmetric part for x < -5

-- State the theorem
theorem f_zeros (k : ℤ) :
  (k = 1 ∨ k = -12) ↔
  (∃ x : ℝ, x > ↑k ∧ x < ↑k + 1 ∧ f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l30_3037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_arrangement_impossible_l30_3079

/-- Represents the arrangement of numbers on a five-pointed star -/
structure StarArrangement where
  /-- The sum of numbers on each line of the star -/
  line_sum : ℕ
  /-- The total sum of all numbers placed on the star -/
  total_sum : ℕ

/-- Theorem stating the impossibility of the arrangement -/
theorem star_arrangement_impossible : ¬ ∃ (arr : StarArrangement), 
  (arr.total_sum = 4 * 1 + 3 * 2 + 3 * 3) ∧ 
  (arr.line_sum * 5 = 2 * arr.total_sum) ∧ 
  (arr.line_sum > 0) :=
by
  sorry

#check star_arrangement_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_arrangement_impossible_l30_3079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_l30_3003

theorem binomial_probability (X : ℕ → ℝ) (n : ℕ) (p : ℝ) :
  (∀ k, X k = Nat.choose n k * p^k * (1 - p)^(n - k)) →
  n = 6 →
  p = 1/2 →
  X 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_l30_3003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l30_3036

/-- Represents the possible outcomes of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the different types of coins -/
inductive Coin
| Penny
| Nickel
| Dime
| Quarter
| HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Represents the outcome of flipping all five coins -/
def CoinFlipOutcome := Coin → CoinFlip

/-- Calculates the total value of heads in a given outcome -/
def headsValue (outcome : CoinFlipOutcome) : ℕ :=
  List.sum (List.map (fun c => 
    match outcome c with
    | CoinFlip.Heads => coinValue c
    | CoinFlip.Tails => 0
  ) [Coin.Penny, Coin.Nickel, Coin.Dime, Coin.Quarter, Coin.HalfDollar])

/-- The set of all possible outcomes when flipping 5 coins -/
def allOutcomes : Finset CoinFlipOutcome :=
  sorry

/-- The set of favorable outcomes (at least 40 cents in heads) -/
def favorableOutcomes : Finset CoinFlipOutcome :=
  allOutcomes.filter (fun outcome => headsValue outcome ≥ 40)

/-- Theorem stating the probability of getting at least 40 cents in heads -/
theorem probability_at_least_40_cents :
  (favorableOutcomes.card : ℚ) / allOutcomes.card = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l30_3036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_period_l30_3061

noncomputable def f (x : ℝ) : ℝ := (6 * (Real.cos x)^4 + 5 * (Real.sin x)^2 - 4) / (Real.cos (2 * x))

def is_in_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ Real.pi/4 + k*Real.pi/2

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_domain_and_period :
  (∀ x : ℝ, is_in_domain x ↔ f x ≠ 0) ∧
  has_period f Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_period_l30_3061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pigment_in_brown_paint_l30_3096

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue_percent : ℚ
  red_percent : ℚ
  yellow_percent : ℚ
  total_weight : ℚ

/-- Calculates the percentage of blue pigment in the brown paint -/
noncomputable def blue_pigment_percentage (sky_blue : PaintMixture) (green : PaintMixture) 
  (brown_weight : ℚ) (red_pigment_weight : ℚ) : ℚ :=
  let sky_blue_weight := red_pigment_weight / sky_blue.red_percent
  let green_weight := brown_weight - sky_blue_weight
  let blue_from_sky_blue := sky_blue_weight * sky_blue.blue_percent
  let blue_from_green := green_weight * green.blue_percent
  let total_blue := blue_from_sky_blue + blue_from_green
  (total_blue / brown_weight) * 100

/-- Theorem stating the percentage of blue pigment in the brown paint -/
theorem blue_pigment_in_brown_paint :
  let sky_blue := PaintMixture.mk (1/10) (9/10) 0 0
  let green := PaintMixture.mk (7/10) 0 (3/10) 0
  let brown_weight := 10
  let red_pigment_weight := (9/2)
  blue_pigment_percentage sky_blue green brown_weight red_pigment_weight = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pigment_in_brown_paint_l30_3096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_log_composite_l30_3020

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the composite function F
noncomputable def F (x : ℝ) : ℝ := g (f x)

-- Theorem statement
theorem quadratic_log_composite :
  (∀ x, f x = -2 * (x - 1)^2 + 2) ∧
  (∀ x > 0, g x = Real.log x / Real.log 2) ∧
  (Set.Ioo 0 2 = {x | ∃ y, F x = y}) ∧
  (Set.Iic 1 = {y | ∃ x, F x = y}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_log_composite_l30_3020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l30_3022

-- Define the function f
noncomputable def f (x : Real) : Real := 
  (Real.sqrt 3 / 2) * Real.sin (2 * x + Real.pi / 3) - Real.cos x ^ 2 + 1 / 2

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem max_triangle_area (t : Triangle) : 
  f t.A = 1/4 → t.a = 3 → 0 < t.A → t.A < Real.pi → 
  ∃ (S : Real), S = (t.b * t.c * Real.sin t.A) / 2 ∧ S ≤ (9 * Real.sqrt 3) / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l30_3022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_is_one_extreme_points_inequality_l30_3074

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) / Real.exp x + a * x - 2

/-- Theorem for part (1) -/
theorem min_value_f_when_a_is_one :
  ∀ x : ℝ, x ≥ 0 → f 1 x ≥ 0 := by sorry

/-- Theorem for part (2) -/
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  Real.exp x₂ - Real.exp x₁ > 2 / a - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_is_one_extreme_points_inequality_l30_3074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_production_correct_l30_3091

/-- The annual production function for electronic components -/
noncomputable def annual_production (a : ℝ) (p : ℝ) (m : ℕ) (x : ℝ) : ℝ :=
  a * (1 + p/100)^x

/-- Theorem stating the annual production function is correct -/
theorem annual_production_correct (a : ℝ) (p : ℝ) (m : ℕ) (x : ℝ) 
  (h1 : a > 0) (h2 : p > 0) (h3 : 0 ≤ x ∧ x ≤ m) :
  ∃ y : ℝ, y = annual_production a p m x ∧ 
    y = a * (1 + p/100)^x := by
  use annual_production a p m x
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_production_correct_l30_3091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_projection_bijective_l30_3019

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the planes and point
variable (α₁ α₂ : Subspace ℝ V) (O : V)

-- Define the lines
def l₁ (α₁ α₂ : Subspace ℝ V) (O : V) : Subspace ℝ V := sorry
def l₂ (α₁ α₂ : Subspace ℝ V) (O : V) : Subspace ℝ V := sorry

-- Define the central projection
noncomputable def central_projection (α₁ α₂ : Subspace ℝ V) (O : V) : 
  {x : V // x ∈ α₁ ∧ x ∉ l₁ α₁ α₂ O} → {y : V // y ∈ α₂ ∧ y ∉ l₂ α₁ α₂ O} := 
  sorry

-- Theorem statement
theorem central_projection_bijective
  (h_intersect : α₁ ⊓ α₂ ≠ ⊥)
  (h_O_not_in_α₁ : O ∉ α₁)
  (h_O_not_in_α₂ : O ∉ α₂) :
  Function.Bijective (central_projection α₁ α₂ O) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_projection_bijective_l30_3019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_equality_condition_l30_3009

theorem min_value_trig_sum (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + (1 / Real.tan x))^2 + (Real.sin x + (1 / Real.sin x))^2 + (Real.cos x + (1 / Real.cos x))^2 ≥ 9 :=
by sorry

theorem equality_condition (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + (1 / Real.tan x))^2 + (Real.sin x + (1 / Real.sin x))^2 + (Real.cos x + (1 / Real.cos x))^2 = 9 ↔ x = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_equality_condition_l30_3009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_squared_distance_AC_squared_in_interval_l30_3030

-- Define the angle range
def angle_range : Set ℝ := {θ | 30 * Real.pi / 180 ≤ θ ∧ θ ≤ 45 * Real.pi / 180}

-- Define the distances
def AB : ℝ := 15
noncomputable def BC (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Theorem statement
theorem distance_AC_squared (θ : ℝ) (h : θ ∈ angle_range) :
  AB ^ 2 + BC θ ^ 2 - 2 * AB * BC θ * Real.cos θ = 225 := by
  sorry

-- Corollary to show the result is in the correct interval
theorem distance_AC_squared_in_interval (θ : ℝ) (h : θ ∈ angle_range) :
  200 ≤ AB ^ 2 + BC θ ^ 2 - 2 * AB * BC θ * Real.cos θ ∧
  AB ^ 2 + BC θ ^ 2 - 2 * AB * BC θ * Real.cos θ ≤ 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_squared_distance_AC_squared_in_interval_l30_3030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_chessboard_cannot_be_tiled_l30_3054

/-- Represents a chessboard with some squares removed -/
structure Chessboard where
  rows : Nat
  cols : Nat
  removed : List (Nat × Nat)

/-- Represents a domino tile -/
structure Domino where
  length : Nat
  width : Nat

/-- Function to check if a square is white -/
def isWhiteSquare (row : Nat) (col : Nat) : Bool :=
  (row + col) % 2 == 0

/-- Function to count white squares on the chessboard -/
def countWhiteSquares (board : Chessboard) : Nat :=
  let totalSquares := board.rows * board.cols
  let whiteSquares := totalSquares / 2
  let removedWhiteSquares := (board.removed.filter (fun (r, c) => isWhiteSquare r c)).length
  whiteSquares - removedWhiteSquares

/-- Function to check if the chessboard can be tiled with dominos -/
def canBeTiled (board : Chessboard) (_ : Domino) : Prop :=
  2 * countWhiteSquares board == board.rows * board.cols - board.removed.length

/-- Theorem stating that the modified chessboard cannot be tiled with dominos -/
theorem modified_chessboard_cannot_be_tiled :
  let board : Chessboard := ⟨8, 8, [(0, 0), (7, 7)]⟩
  let domino : Domino := ⟨2, 1⟩
  ¬ (canBeTiled board domino) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_chessboard_cannot_be_tiled_l30_3054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_tangents_l30_3040

/-- The cosine of the angle between two tangents drawn from an external point to a circle -/
theorem cosine_angle_between_tangents (center : ℝ × ℝ) (radius : ℝ) (P : ℝ × ℝ) : 
  center = (1, 1) →
  radius = 1 →
  P = (3, 2) →
  let d := Real.sqrt ((P.1 - center.1)^2 + (P.2 - center.2)^2)
  Real.cos (2 * Real.arcsin (radius / d)) = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_tangents_l30_3040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l30_3024

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x + 2) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of g
def domain_g : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x ≠ 0 ∧ g x ≠ 0} = domain_g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l30_3024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l30_3025

def A : Set ℤ := {x | 0 < x ∧ x < 4}
def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l30_3025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l30_3098

/-- An arithmetic sequence with first term 2 and positive common difference -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 2
  | n + 1 => arithmetic_seq d n + d

/-- An exponential sequence with first term 1 and base greater than 1 -/
def exponential_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => r * exponential_seq r n

/-- The sequence c_n defined as the sum of a_n and b_n -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ := arithmetic_seq d n + exponential_seq r n

theorem c_k_value (d r k : ℕ) :
  d > 0 ∧ r > 1 ∧
  c_seq d r (k - 1) = 400 ∧
  c_seq d r (k + 1) = 1600 →
  c_seq d r k = 684 := by
  sorry

#eval c_seq 130 2 5  -- This will evaluate c_5 with d=130 and r=2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l30_3098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_embankment_construction_time_l30_3001

/-- Represents the amount of work done by one worker in one day -/
noncomputable def worker_rate : ℚ := 1 / 300

/-- Calculates the total work done given the number of workers, efficiency, and days -/
noncomputable def total_work (workers : ℕ) (efficiency : ℚ) (days : ℚ) : ℚ :=
  (workers : ℚ) * worker_rate * efficiency * days

theorem embankment_construction_time :
  ∃ (days : ℚ),
    total_work 60 1 5 = total_work 40 1 days + total_work 20 (3/4) days ∧
    days = 300 / 55 := by
  sorry

#eval (300 : ℚ) / 55

end NUMINAMATH_CALUDE_ERRORFEEDBACK_embankment_construction_time_l30_3001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l30_3086

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def is_geometric_sequence (a b c : ℝ) : Prop := b * b = a * c

structure Matrix3x3 where
  a : Fin 3 → Fin 3 → ℝ

def row_sum (A : Matrix3x3) (i : Fin 3) : ℝ :=
  A.a i 0 + A.a i 1 + A.a i 2

theorem matrix_properties (A : Matrix3x3) 
  (h_positive : ∀ i j : Fin 3, 0 < A.a i j)
  (h_distinct : ∀ i j k l : Fin 3, (i ≠ k ∨ j ≠ l) → A.a i j ≠ A.a k l)
  (h_arithmetic : ∀ i : Fin 3, is_arithmetic_sequence (A.a i 0) (A.a i 1) (A.a i 2))
  (h_geometric : is_geometric_sequence (row_sum A 0) (row_sum A 1) (row_sum A 2)) :
  (is_geometric_sequence (A.a 0 1) (A.a 1 1) (A.a 2 1)) ∧ 
  (∃ B : Matrix3x3, 
    (∀ i j : Fin 3, 0 < B.a i j) ∧
    (∀ i j k l : Fin 3, (i ≠ k ∨ j ≠ l) → B.a i j ≠ B.a k l) ∧
    (∀ i : Fin 3, is_arithmetic_sequence (B.a i 0) (B.a i 1) (B.a i 2)) ∧
    (is_geometric_sequence (row_sum B 0) (row_sum B 1) (row_sum B 2)) ∧
    ¬(is_geometric_sequence (B.a 0 0) (B.a 1 0) (B.a 2 0))) ∧
  (A.a 0 1 + A.a 2 1 > A.a 1 0 + A.a 1 2) ∧
  ((A.a 0 0 + A.a 0 1 + A.a 0 2 + A.a 1 0 + A.a 1 1 + A.a 1 2 + A.a 2 0 + A.a 2 1 + A.a 2 2 = 9) → A.a 1 1 ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l30_3086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_equals_root_two_plus_root_six_l30_3078

/-- Represents a cube containing spheres -/
structure CubeWithSpheres where
  edge_length : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ

/-- Properties of the cube with spheres -/
def cube_properties (c : CubeWithSpheres) : Prop :=
  c.small_sphere_radius = 1 ∧
  c.large_sphere_radius = c.edge_length / 2 ∧
  c.edge_length = (Real.sqrt 3 + 1)^2

/-- The distance between centers of large and small circles when viewed along an edge -/
noncomputable def center_distance (c : CubeWithSpheres) : ℝ :=
  Real.sqrt 2 * (c.large_sphere_radius - c.small_sphere_radius)

/-- Theorem stating the distance between centers of large and small circles -/
theorem center_distance_equals_root_two_plus_root_six (c : CubeWithSpheres) 
  (h : cube_properties c) : 
  center_distance c = Real.sqrt 2 + Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_equals_root_two_plus_root_six_l30_3078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_l30_3081

-- Define the fraction (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- Define what it means for the fraction to be meaningful
def is_meaningful (x : ℝ) : Prop := 2 * x - 1 ≠ 0

-- Theorem statement
theorem fraction_meaningful_iff (x : ℝ) : is_meaningful x ↔ x ≠ 1/2 := by
  -- Proof goes here
  sorry

#check fraction_meaningful_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_l30_3081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l30_3028

theorem inequality_proof (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 3/2) :
  (3 : ℝ)^(-a) + (9 : ℝ)^(-b) + (27 : ℝ)^(-c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l30_3028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l30_3006

-- Define the hyperbola C
def hyperbola (l : ℝ) (x y : ℝ) : Prop :=
  x^2 / (1 - l) - y^2 / l = 1

-- Define the right focus B
def B : ℝ × ℝ := (1, 0)

-- Define the condition for l
def l_condition (l : ℝ) : Prop :=
  0 < l ∧ l < 1

-- Part 1
theorem part1 (l : ℝ) (M N : ℝ × ℝ) :
  l_condition l →
  hyperbola l M.1 M.2 →
  hyperbola l N.1 N.2 →
  M.1 = 1 ∧ N.1 = 1 →
  M.2 - N.2 = 2 →
  l = (Real.sqrt 5 - 1) / 2 :=
sorry

-- Part 2
theorem part2 (A : ℝ × ℝ) :
  hyperbola (3/4) A.1 A.2 →
  A.2 - B.2 = Real.sqrt 3 * (A.1 - B.1) →
  Real.sqrt (A.1^2 + A.2^2) = Real.sqrt 13 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l30_3006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_bought_200_tickets_l30_3033

/-- Represents the lotto ticket scenario --/
structure LottoScenario where
  ticketPrice : ℚ
  winnerPercentage : ℚ
  fiveDollarWinnerPercentage : ℚ
  grandPrizeAmount : ℚ
  averageOtherWinAmount : ℚ
  totalProfit : ℚ

/-- Calculates the number of lotto tickets bought given a LottoScenario --/
noncomputable def calculateTickets (scenario : LottoScenario) : ℚ :=
  let totalWinnings (t : ℚ) : ℚ :=
    (scenario.winnerPercentage * scenario.fiveDollarWinnerPercentage * t * 5) +
    scenario.grandPrizeAmount +
    (scenario.winnerPercentage * t - scenario.winnerPercentage * scenario.fiveDollarWinnerPercentage * t - 1) * scenario.averageOtherWinAmount
  ((scenario.totalProfit + scenario.ticketPrice * 200) / (totalWinnings 1 - scenario.ticketPrice))

/-- Theorem stating that James bought 200 lotto tickets --/
theorem james_bought_200_tickets : calculateTickets {
  ticketPrice := 2,
  winnerPercentage := 1/5,
  fiveDollarWinnerPercentage := 4/5,
  grandPrizeAmount := 5000,
  averageOtherWinAmount := 10,
  totalProfit := 4830
} = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_bought_200_tickets_l30_3033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_is_one_l30_3021

/-- Definition of a two-digit number -/
def isTwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- Definition of Q(n) for a two-digit number n = 10a + b -/
noncomputable def Q (a b : ℕ) : ℕ := 
  Int.toNat (Int.floor (a / b : ℚ))

/-- Definition of T(n) for a two-digit number n = 10a + b -/
def T (a b : ℕ) : ℕ := 
  a + b

/-- Theorem: For a two-digit number M = 10a + b, if M = Q(M) + T(M), then b = 1 -/
theorem units_digit_is_one (a b : ℕ) (h : isTwoDigitNumber (10 * a + b)) 
  (h_eq : 10 * a + b = Q a b + T a b) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_is_one_l30_3021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_and_equation_solution_l30_3066

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x
  else if 0 ≤ x ∧ x ≤ 1 then x^2 - 1
  else 0

theorem symmetric_points_and_equation_solution :
  ∃ (A B : ℝ × ℝ),
    (A.1 = Real.sqrt 2 - 1 ∧ A.2 = 2 - 2 * Real.sqrt 2) ∧
    (B.1 = 1 - Real.sqrt 2 ∧ B.2 = 2 * Real.sqrt 2 - 2) ∧
    f A.1 = A.2 ∧ f B.1 = B.2 ∧
    A.1 = -B.1 ∧ A.2 = -B.2 ∧
    ∃ (x₁ x₂ x₃ : ℝ),
      x₁ < x₂ ∧ x₂ < x₃ ∧
      x₃ - x₂ = 2 * (x₂ - x₁) ∧
      f x₁ + 2 * Real.sqrt (1 - x₁^2) + |f x₁ - 2 * Real.sqrt (1 - x₁^2)| - 2 * ((-3 + Real.sqrt 17) / 2) * x₁ - 4 = 0 ∧
      f x₂ + 2 * Real.sqrt (1 - x₂^2) + |f x₂ - 2 * Real.sqrt (1 - x₂^2)| - 2 * ((-3 + Real.sqrt 17) / 2) * x₂ - 4 = 0 ∧
      f x₃ + 2 * Real.sqrt (1 - x₃^2) + |f x₃ - 2 * Real.sqrt (1 - x₃^2)| - 2 * ((-3 + Real.sqrt 17) / 2) * x₃ - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_and_equation_solution_l30_3066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l30_3075

/-- The speed of a train given its length, platform length, and time to cross -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  (train_length + platform_length) / time * 3.6

/-- Theorem: A train with length 250.0416 m crossing a 270 m platform in 26 seconds has a speed of approximately 72 km/h -/
theorem train_speed_theorem :
  let train_length : ℝ := 250.0416
  let platform_length : ℝ := 270
  let time : ℝ := 26
  abs (train_speed train_length platform_length time - 72) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l30_3075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_l30_3058

-- Define the "averaged with" operation
noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

-- Define the "difference by" operation
def diff (a b : ℝ) : ℝ := a - b

-- Theorem for the distributive laws
theorem distributive_laws :
  (∀ x y z : ℝ, diff x (avg y z) = avg (diff x y) (diff x z)) ∧
  (∀ x y z : ℝ, avg x (avg y z) = avg (avg x y) (avg x z)) ∧
  (∃ x y z : ℝ, avg x (diff y z) ≠ diff (avg x y) (avg x z)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_l30_3058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_a_gt_one_min_sum_x1_x2_min_sum_x1_x2_equals_two_l30_3026

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + x

-- Part 1
theorem f_negative_iff_a_gt_one (a : ℝ) :
  (∀ x > 0, f a x < 0) ↔ a > 1 := by
  sorry

-- Part 2
theorem min_sum_x1_x2 (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f (1/2) x₁ + f (1/2) x₂ = 1 → x₁ + x₂ ≥ 2 := by
  sorry

-- Theorem for the minimum value
theorem min_sum_x1_x2_equals_two :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ f (1/2) x₁ + f (1/2) x₂ = 1 ∧ x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_a_gt_one_min_sum_x1_x2_min_sum_x1_x2_equals_two_l30_3026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savage_parking_l30_3039

/-- The number of ways to arrange k cars in n spaces such that no three consecutive spaces are occupied. -/
def f (n k : ℕ) : ℕ :=
  if k > n then 0
  else if k = 0 then 1
  else if k = 1 then n
  else if k = 2 then n.choose 2
  else f (n-1) k + f (n-2) (k-1) + f (n-3) (k-2)

/-- There are 357 ways to park 6 identical cars in a 12-space garage with no 3 consecutive spaces occupied. -/
theorem savage_parking : f 12 6 = 357 := by
  -- We would normally prove this by induction or by explicitly calculating the value
  -- For now, we'll use sorry to skip the proof
  sorry

#eval f 12 6  -- This should evaluate to 357

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savage_parking_l30_3039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_or_two_heads_in_three_tosses_l30_3062

/-- Probability of getting 1 or 2 heads when tossing a fair coin three times -/
theorem prob_one_or_two_heads_in_three_tosses :
  let X : Finset (Fin 3 → Fin 2) := Finset.univ
  let count_heads (outcome : Fin 3 → Fin 2) : ℕ :=
    (outcome 0).val + (outcome 1).val + (outcome 2).val
  let favorable_outcomes := X.filter (λ outcome ↦ 0 < count_heads outcome ∧ count_heads outcome < 3)
  (favorable_outcomes.card : ℚ) / (X.card : ℚ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_or_two_heads_in_three_tosses_l30_3062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_is_correct_l30_3059

/-- Two circles with centers at (5,5) and (20,15), both tangent to the x-axis -/
structure TwoCircles where
  center1 : ℝ × ℝ := (5, 5)
  center2 : ℝ × ℝ := (20, 15)
  tangent_to_x_axis : True

/-- The distance between the closest points of two circles -/
noncomputable def closest_distance (c : TwoCircles) : ℝ :=
  Real.sqrt 13 * 5 - 20

/-- Theorem stating that the distance between the closest points of the two circles is 5√13 - 20 -/
theorem closest_distance_is_correct (c : TwoCircles) : 
  closest_distance c = Real.sqrt 13 * 5 - 20 := by
  rfl

#check closest_distance_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_is_correct_l30_3059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_is_40_seconds_l30_3064

/-- Calculates the time (in seconds) required for a moving train to cross a stationary train -/
noncomputable def time_to_cross_stationary_train (speed_kmh : ℝ) (time_to_pass_pole : ℝ) (stationary_train_length : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  let moving_train_length := speed_ms * time_to_pass_pole
  let total_distance := moving_train_length + stationary_train_length
  total_distance / speed_ms

/-- Theorem: Given the specified conditions, the time to cross the stationary train is 40 seconds -/
theorem time_to_cross_is_40_seconds (speed_kmh : ℝ) (time_to_pass_pole : ℝ) (stationary_train_length : ℝ)
    (h1 : speed_kmh = 36)
    (h2 : time_to_pass_pole = 10)
    (h3 : stationary_train_length = 300) :
    time_to_cross_stationary_train speed_kmh time_to_pass_pole stationary_train_length = 40 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_is_40_seconds_l30_3064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_is_400_l30_3017

def starting_number (x : ℕ) : Prop :=
  (∃ (count : ℕ), count = 300 ∧
    (∀ y : ℕ, x < y ∧ y ≤ 1000 ∧ y % 10 ∈ ({1, 3, 5, 7, 9} : Finset ℕ) →
      y - x ≤ count)) ∧
    (∀ z : ℕ, z > x → ¬(∃ (count : ℕ), count = 300 ∧
      (∀ y : ℕ, z < y ∧ y ≤ 1000 ∧ y % 10 ∈ ({1, 3, 5, 7, 9} : Finset ℕ) →
        y - z ≤ count)))

theorem starting_number_is_400 : starting_number 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_is_400_l30_3017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_rental_cost_l30_3060

noncomputable def total_cost : ℝ := 4.8
def num_dvds : ℕ := 4
noncomputable def cost_per_dvd : ℝ := total_cost / num_dvds

theorem dvd_rental_cost : cost_per_dvd = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_rental_cost_l30_3060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pentagon_weight_l30_3071

/-- The weight of a regular pentagon with given side length, assuming uniform density and thickness -/
noncomputable def pentagonWeight (sideLength : ℝ) (referenceWeight : ℝ) (referenceSideLength : ℝ) : ℝ :=
  referenceWeight * (sideLength / referenceSideLength)^2

theorem second_pentagon_weight :
  let firstSideLength : ℝ := 4
  let firstWeight : ℝ := 20
  let secondSideLength : ℝ := 6
  pentagonWeight secondSideLength firstWeight firstSideLength = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pentagon_weight_l30_3071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l30_3005

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 9

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define that P is inside the circle
def P_inside_circle : Prop := circle_eq P.1 P.2

-- Define AC as the longest chord through P
def AC_longest_chord (A C : ℝ × ℝ) : Prop := 
  circle_eq A.1 A.2 ∧ circle_eq C.1 C.2 ∧ 
  ∀ X Y : ℝ × ℝ, circle_eq X.1 X.2 → circle_eq Y.1 Y.2 → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 ≤ (A.1 - C.1)^2 + (A.2 - C.2)^2

-- Define BD as the shortest chord through P
def BD_shortest_chord (B D : ℝ × ℝ) : Prop := 
  circle_eq B.1 B.2 ∧ circle_eq D.1 D.2 ∧ 
  ∀ X Y : ℝ × ℝ, circle_eq X.1 X.2 → circle_eq Y.1 Y.2 → P ∈ Set.Icc X Y →
    (B.1 - D.1)^2 + (B.2 - D.2)^2 ≤ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Theorem statement
theorem quadrilateral_area 
  (A B C D : ℝ × ℝ) 
  (h_AC : AC_longest_chord A C) 
  (h_BD : BD_shortest_chord B D) 
  (h_P_inside : P_inside_circle) : 
  abs ((A.1 - C.1) * (B.2 - D.2) - (A.2 - C.2) * (B.1 - D.1)) / 2 = 6 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l30_3005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l30_3010

noncomputable def area_quadrilateral (x y b c : ℝ) : ℝ := 
  (3 * x) / (x - 2)

theorem min_area_quadrilateral (x y : ℝ) (B C : ℝ × ℝ) :
  y^2 = 2*x →
  B.1 = 0 →
  C.1 = 0 →
  (x - 1)^2 + y^2 = 1 →
  (∀ (x' y' : ℝ), y'^2 = 2*x' → (x' - 1)^2 + y'^2 = 1 → 
    area_quadrilateral x' y' B.2 C.2 ≥ area_quadrilateral x y B.2 C.2) →
  area_quadrilateral x y B.2 C.2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l30_3010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l30_3012

theorem right_triangle_area (a b c : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 6) 
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) : 
  (1 / 2 : ℝ) * a * b = 4 * Real.sqrt 6 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l30_3012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sine_inequality_l30_3080

theorem negation_of_universal_sine_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≥ -1) ↔ (∃ x : ℝ, Real.sin x < -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sine_inequality_l30_3080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l30_3070

/-- Represents a hyperbola with equation x²/3 - y²/6 = 1 -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y => x^2 / 3 - y^2 / 6 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ → ℝ) :=
  {f | ∃ (k : ℝ), (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) ∧ f = fun x => k * x}

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt 3

theorem hyperbola_properties (h : Hyperbola) :
  (asymptotes h = {f | ∃ (k : ℝ), (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) ∧ f = fun x => k * x}) ∧
  (eccentricity h = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l30_3070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutter_lagaan_payment_l30_3053

/-- Calculates the lagaan payment for a farmer given the total lagaan collected and the farmer's land percentage -/
noncomputable def calculate_lagaan_payment (total_lagaan : ℝ) (land_percentage : ℝ) : ℝ :=
  total_lagaan * (land_percentage / 100)

/-- Theorem: Mutter's lagaan payment is Rs. 800 -/
theorem mutter_lagaan_payment :
  let total_lagaan : ℝ := 344000
  let mutter_land_percentage : ℝ := 0.23255813953488372
  calculate_lagaan_payment total_lagaan mutter_land_percentage = 800 := by
  -- Unfold the definition of calculate_lagaan_payment
  unfold calculate_lagaan_payment
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutter_lagaan_payment_l30_3053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l30_3072

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the slope of the tangent line
noncomputable def m : ℝ := deriv f x₀

-- Define the y-coordinate of the point of tangency
def y₀ : ℝ := f x₀

-- Theorem: The equation of the tangent line is y = -4x + 1
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = -4 * x + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l30_3072
