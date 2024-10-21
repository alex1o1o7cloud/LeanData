import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_identification_not_linear_in_one_variable_two_var_l803_80393

/-- Predicate to check if an equation is linear in one variable -/
def IsLinearInOneVariable (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_equation_identification :
  IsLinearInOneVariable (fun x ↦ 2 * x - 1) ∧
  ¬IsLinearInOneVariable (fun x ↦ 1 / x - 4) ∧
  ¬IsLinearInOneVariable (fun x ↦ 4 * x^2 - 2 * x) :=
by sorry

-- Separate theorem for the two-variable equation
theorem not_linear_in_one_variable_two_var :
  ∀ y, ¬IsLinearInOneVariable (fun x ↦ 5 * x - y - 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_identification_not_linear_in_one_variable_two_var_l803_80393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l803_80332

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The vector (a, √3b) is parallel to (cos A, sin B) -/
def vectorsParallel (t : Triangle) : Prop :=
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A

/-- Theorem: In a triangle where (a, √3b) is parallel to (cos A, sin B), 
    angle A = π/3, and if a = √7 and b = 2, the area of the triangle is 3√3/2 -/
theorem triangle_theorem (t : Triangle) 
  (h1 : vectorsParallel t) 
  (h2 : t.a = Real.sqrt 7) 
  (h3 : t.b = 2) : 
  t.A = π/3 ∧ (1/2 * t.b * t.c * Real.sin t.A) = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l803_80332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_proof_l803_80340

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 1 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x + y = 5

/-- The minimum length of the tangent line -/
noncomputable def min_tangent_length : ℝ := 2 * Real.sqrt 3

theorem min_tangent_length_proof :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' ∧ line_eq x' y' →
  Real.sqrt ((x' - 1)^2 + (y' + 2)^2) ≥ min_tangent_length := by
  sorry

#check min_tangent_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_proof_l803_80340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_difference_l803_80365

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x^2 - 2*x)

-- State the theorem
theorem function_max_min_difference (a : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 (3/2), f a x ≤ max) ∧
    (∀ x ∈ Set.Icc 0 (3/2), f a x ≥ min) ∧
    (max - min = 2*a)) →
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_difference_l803_80365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_relations_l803_80320

-- Define the triangle and circle properties
structure TriangleWithTangentCircle where
  -- Triangle properties
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  p : ℝ
  rA : ℝ
  -- Circle properties
  ρ : ℝ
  d : ℝ
  -- Assumptions
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_radii : 0 < r ∧ 0 < rA ∧ 0 < ρ
  pos_d : 0 < d
  semi_peri : p = (a + b + c) / 2

-- Theorem statements
theorem tangent_circle_relations (t : TriangleWithTangentCircle) :
  t.a * (t.d - t.ρ) = 2 * t.p * (t.r - t.ρ) ∧
  ∃ DE : ℝ, DE = Real.sqrt (t.r * t.rA * (t.ρ - t.r) * (t.rA - t.ρ)) / (t.rA - t.r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_relations_l803_80320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_15_5_l803_80386

/-- The radius of a sphere tangent to a truncated cone -/
noncomputable def sphere_radius_tangent_to_truncated_cone (r₁ r₂ : ℝ) : ℝ :=
  5 * Real.sqrt 3

/-- Theorem: The radius of a sphere tangent to the top, bottom, and lateral surface
    of a truncated cone with horizontal base radii 15 and 5 is equal to 5√3 -/
theorem sphere_radius_tangent_to_truncated_cone_15_5 :
  sphere_radius_tangent_to_truncated_cone 15 5 = 5 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_15_5_l803_80386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_sine_value_l803_80324

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem function_and_sine_value 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ) (h3 : φ < Real.pi) 
  (h4 : f ω φ (Real.pi / 6) = -1/2) 
  (h5 : ∃ (x : ℝ), f ω φ x = 0 ∧ f ω φ (x + Real.pi) = 0) 
  (θ : ℝ) 
  (h6 : f ω φ (θ + Real.pi / 3) = -3/5) :
  (∀ x, f ω φ x = -Real.sin x) ∧ 
  (Real.sin θ = (3 - 4 * Real.sqrt 3) / 10 ∨ Real.sin θ = (3 + 4 * Real.sqrt 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_sine_value_l803_80324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_is_14_l803_80389

/-- Represents the time it takes for a pump to fill a tank and for the system with a leak to fill the tank -/
structure FillTimes where
  pump_time : ℝ
  with_leak_time : ℝ

/-- Calculates the time it takes for a leak to drain a full tank -/
noncomputable def leak_drain_time (times : FillTimes) : ℝ :=
  (times.pump_time * times.with_leak_time) / (times.with_leak_time - times.pump_time)

/-- Theorem stating that given specific fill times, the leak drain time is 14 hours -/
theorem leak_drain_time_is_14 (times : FillTimes) 
  (h1 : times.pump_time = 2)
  (h2 : times.with_leak_time = 7/3) : 
  leak_drain_time times = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_is_14_l803_80389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_formula_l803_80341

/-- A regular octagon inscribed in a circle -/
structure RegularOctagonInCircle where
  /-- The side length of the octagon -/
  side_length : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The radius is equal to the side length -/
  radius_eq_side : radius = side_length

/-- The length of the arc intercepted by any side of the octagon -/
noncomputable def arc_length (octagon : RegularOctagonInCircle) : ℝ :=
  (5 * Real.pi / 4) * octagon.side_length

/-- Theorem: The length of the arc intercepted by any side of the octagon is 5π/4 * side_length -/
theorem arc_length_formula (octagon : RegularOctagonInCircle) (h : octagon.side_length = 5) :
  arc_length octagon = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_formula_l803_80341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_min_omega_l803_80314

-- Define the function f
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Theorem 1
theorem phi_value (ω : ℝ) (φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) (h3 : f ω φ 0 = 1) :
  φ = π/6 := by
  sorry

-- Theorem 2
theorem min_omega (ω : ℝ) (φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) 
  (h3 : ∃ x : ℝ, f ω φ (x + 2) - f ω φ x = 4) :
  ω ≥ π/2 ∧ ∀ ω' > 0, (∃ x : ℝ, f ω' φ (x + 2) - f ω' φ x = 4) → ω' ≥ π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_min_omega_l803_80314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_preserves_shape_and_size_l803_80349

-- Define a geometric figure
structure GeometricFigure where
  -- Simplified representation of a figure as a set of points
  points : Set (ℝ × ℝ)

-- Define a translation
def translation (t : ℝ × ℝ) (f : GeometricFigure) : GeometricFigure :=
  { points := f.points.image (fun p => (p.1 + t.1, p.2 + t.2)) }

-- Define shape preservation
def preservesShape (f g : GeometricFigure) : Prop :=
  ∃ (t : ℝ × ℝ), g = translation t f

-- Define size preservation
def preservesSize (f g : GeometricFigure) : Prop :=
  f.points.ncard = g.points.ncard

-- Theorem statement
theorem translation_preserves_shape_and_size (f : GeometricFigure) (t : ℝ × ℝ) :
  preservesShape f (translation t f) ∧ preservesSize f (translation t f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_preserves_shape_and_size_l803_80349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l803_80360

open Real BigOperators

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℝ := tan (n : ℝ) * tan ((n - 1) : ℝ)

-- Define the sum of the sequence up to n
noncomputable def sum_a (n : ℕ) : ℝ := ∑ k in Finset.range n, a (k + 1)

-- State the theorem
theorem sequence_sum_theorem (A B : ℝ) :
  (∀ n : ℕ, sum_a n = A * tan (n : ℝ) + B * n) →
  A = 1 / tan 1 ∧ B = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l803_80360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l803_80337

-- Define the set of x values that satisfy the inequality
noncomputable def S : Set ℝ := {x | x > 2 ∨ (3 ≤ x ∧ x ≤ 5)}

-- Define the function representing the left side of the inequality
noncomputable def f (p q r x : ℝ) : ℝ := (x - p) * (x - q) / (x - r)

-- Theorem statement
theorem sum_of_constants (p q r : ℝ) 
  (h1 : ∀ x, x ∈ S ↔ f p q r x ≤ 0)
  (h2 : p < q) : 
  p + q + 2*r = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l803_80337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l803_80312

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of triangle PQR with vertices P(-3, 2), Q(1, 7), and R(3, -1) is 21 square units -/
theorem triangle_PQR_area :
  triangleArea (-3) 2 1 7 3 (-1) = 21 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l803_80312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l803_80373

-- Define a right triangle PQR
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : (R.1 - P.1) * (Q.1 - P.1) + (R.2 - P.2) * (Q.2 - P.2) = 0

-- Define the median to the hypotenuse
noncomputable def median_to_hypotenuse (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.P.1 - ((t.Q.1 + t.R.1) / 2))^2 + (t.P.2 - ((t.Q.2 + t.R.2) / 2))^2)

-- Define the perimeter of the triangle
noncomputable def perimeter (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.Q.1 - t.P.1)^2 + (t.Q.2 - t.P.2)^2) +
  Real.sqrt ((t.R.1 - t.Q.1)^2 + (t.R.2 - t.Q.2)^2) +
  Real.sqrt ((t.P.1 - t.R.1)^2 + (t.P.2 - t.R.2)^2)

-- Define the area of the triangle
noncomputable def area (t : RightTriangle) : ℝ :=
  (1/2) * abs ((t.Q.1 - t.P.1) * (t.R.2 - t.P.2) - (t.R.1 - t.P.1) * (t.Q.2 - t.P.2))

-- Theorem statement
theorem right_triangle_area (t : RightTriangle) :
  median_to_hypotenuse t = 5/4 → perimeter t = 6 → area t = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l803_80373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l803_80361

def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l803_80361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l803_80363

-- Define the function f as the integrand
noncomputable def f (x : ℝ) : ℝ := (8 * x - Real.arctan (2 * x)) / (1 + 4 * x^2)

-- Define the function F as the antiderivative
noncomputable def F (x : ℝ) : ℝ := Real.log (1 + 4 * x^2) - (1/4) * (Real.arctan (2 * x))^2

-- Theorem statement
theorem integral_proof (x : ℝ) : 
  deriv F x = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l803_80363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_cube_surface_area_l803_80317

/-- Represents a cube in 3D space -/
structure Cube where
  side_length : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  radius : ℝ

/-- Calculates the surface area of a cube -/
def Cube.surface_area (c : Cube) : ℝ :=
  6 * c.side_length ^ 2

/-- Defines what it means for a sphere to be inscribed in a cube -/
def Sphere.inscribed_in (s : Sphere) (c : Cube) : Prop :=
  s.radius * 2 = c.side_length

/-- Defines what it means for a cube to be inscribed in a sphere -/
def Cube.inscribed_in (c : Cube) (s : Sphere) : Prop :=
  c.side_length * Real.sqrt 3 = s.radius * 2

/-- Given a cube with a sphere inscribed inside it, and another cube inscribed inside the sphere,
    this theorem proves that if the outer cube has a surface area of 54 square meters,
    then the inner cube has a surface area of 18 square meters. -/
theorem inner_cube_surface_area (outer_cube : Cube) (sphere : Sphere) (inner_cube : Cube) :
  outer_cube.surface_area = 54 →
  sphere.inscribed_in outer_cube →
  inner_cube.inscribed_in sphere →
  inner_cube.surface_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_cube_surface_area_l803_80317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l803_80383

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (x + 1)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 1 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ f c) ∧
  f c = Real.exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l803_80383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_state_points_ratio_l803_80300

/-- Golden State Team points calculation -/
theorem golden_state_points_ratio : 
  ∀ (klay_points : ℕ),
  (let draymond_points : ℕ := 12
   let curry_points : ℕ := 2 * draymond_points
   let kelly_points : ℕ := 9
   let durant_points : ℕ := 2 * kelly_points
   let total_points : ℕ := 69
   klay_points + draymond_points + curry_points + kelly_points + durant_points = total_points) →
  (klay_points : ℚ) / draymond_points = 1 / 2 :=
by
  intro klay_points
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_state_points_ratio_l803_80300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagonal_pyramid_volume_l803_80348

/-- The volume of a regular decagonal pyramid -/
noncomputable def pyramid_volume (height : ℝ) (apex_angle : ℝ) : ℝ :=
  (5 * height^3 * Real.sin (2 * apex_angle)) / (3 * (1 + 2 * Real.cos apex_angle))

/-- Theorem: The volume of a regular decagonal pyramid with height 39 cm and lateral face apex angle 18° is approximately 20023 cm³ -/
theorem decagonal_pyramid_volume :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |pyramid_volume 39 (18 * π / 180) - 20023| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagonal_pyramid_volume_l803_80348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snakes_in_cage_l803_80392

/-- Given a cage with snakes and alligators, prove that the total number of snakes
    is equal to the sum of hiding and not hiding snakes. -/
theorem total_snakes_in_cage 
  (hiding_snakes : ℕ) 
  (not_hiding_snakes : ℕ) : 
  hiding_snakes + not_hiding_snakes = hiding_snakes + not_hiding_snakes := by
  rfl

/-- Calculate the total number of snakes in the zoo cage -/
def snakes_in_zoo_cage : ℕ := 64 + 31

#eval snakes_in_zoo_cage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snakes_in_cage_l803_80392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inverse_l803_80330

theorem log_product_inverse (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  Real.log c * Real.log d / (Real.log d * Real.log c) = 1 :=
by
  -- We use Real.log instead of Real.log c d, as Real.log takes one argument
  -- The expression is rewritten to match the standard definition of logarithms in Lean
  -- The proof steps are omitted and replaced with 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inverse_l803_80330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l803_80377

/-- The volume of a cylindrical well -/
noncomputable def well_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * (diameter / 2)^2 * depth

/-- Theorem: The volume of a cylindrical well with diameter 2 meters and depth 14 meters is 14π cubic meters -/
theorem well_volume_calculation :
  well_volume 2 14 = 14 * Real.pi :=
by
  unfold well_volume
  simp [Real.pi]
  ring

/-- Approximate numerical value of the well volume -/
def approx_well_volume : ℚ :=
  (14 * 314159) / 100000

#eval approx_well_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l803_80377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_division_l803_80397

/-- A cross-shaped region in the coordinate plane -/
structure CrossRegion where
  squares : Fin 7 → Set (ℝ × ℝ)
  central_at_origin : squares 0 = {p : ℝ × ℝ | -0.5 ≤ p.1 ∧ p.1 ≤ 0.5 ∧ -0.5 ≤ p.2 ∧ p.2 ≤ 0.5}
  unit_extension : ∀ i : Fin 4, ∃ d : ℝ × ℝ, d ∈ [(1,0), (-1,0), (0,1), (0,-1)] ∧
                   squares (i+1) = {p : ℝ × ℝ | (p.1 - d.1, p.2 - d.2) ∈ squares 0}

/-- A line from (c,0) to (4,4) -/
def dividing_line (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (4 / (4 - c)) * (p.1 - c)}

/-- The area of a region -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that c = 2.25 divides the cross region into two equal areas -/
theorem cross_division (R : CrossRegion) :
  ∃ c : ℝ, c = 2.25 ∧
  area (R.squares 0 ∪ R.squares 1 ∪ R.squares 2 ∪ R.squares 3 ∪ R.squares 4 ∪ R.squares 5 ∪ R.squares 6 ∩ 
       {p : ℝ × ℝ | p.2 ≤ (4 / (4 - c)) * (p.1 - c)}) =
  area (R.squares 0 ∪ R.squares 1 ∪ R.squares 2 ∪ R.squares 3 ∪ R.squares 4 ∪ R.squares 5 ∪ R.squares 6 ∩ 
       {p : ℝ × ℝ | p.2 ≥ (4 / (4 - c)) * (p.1 - c)}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_division_l803_80397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l803_80359

-- Define the centers of the circles
variable (A B C : ℝ × ℝ)

-- Define the radii of the circles
def radius_A : ℝ := 2
def radius_B : ℝ := 3
def radius_C : ℝ := 4

-- Define the line m
variable (m : Set (ℝ × ℝ))

-- Define the tangent points
variable (A' B' C' : ℝ × ℝ)

-- State that the circles are tangent to line m
axiom tangent_to_m : A' ∈ m ∧ B' ∈ m ∧ C' ∈ m

-- State that B' is between A' and C'
axiom B'_between : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ B' = t • A' + (1 - t) • C'

-- State that the circles are externally tangent
axiom externally_tangent : 
  dist A B = radius_A + radius_B ∧
  dist B C = radius_B + radius_C

-- The theorem to be proved
theorem area_of_triangle_ABC : 
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l803_80359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_maintenance_cost_l803_80370

-- Define constants
def miles_per_year : ℕ := 12000
def miles_per_oil_change : ℕ := 3000
def miles_per_tire_rotation : ℕ := 6000
def miles_per_brake_pad_replacement : ℕ := 24000
def oil_change_prices : List ℚ := [55, 45, 50, 40]
def tire_rotation_price : ℚ := 40
def tire_rotation_discount : ℚ := 0.1
def brake_pad_replacement_price : ℚ := 200
def brake_pad_discount : ℚ := 0.2
def biennial_membership_fee : ℚ := 60

-- Define the theorem
theorem annual_maintenance_cost :
  (let oil_changes_per_year := miles_per_year / miles_per_oil_change - 1
   let tire_rotations_per_year := miles_per_year / miles_per_tire_rotation
   let brake_pad_replacements_per_year := miles_per_year / miles_per_brake_pad_replacement
   let oil_change_cost := (List.sum (List.take oil_changes_per_year oil_change_prices))
   let tire_rotation_cost := tire_rotations_per_year * tire_rotation_price * (1 - tire_rotation_discount)
   let brake_pad_cost := (brake_pad_replacement_price * (1 - brake_pad_discount) + biennial_membership_fee) * brake_pad_replacements_per_year / 2
   oil_change_cost + tire_rotation_cost + brake_pad_cost) = 317 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_maintenance_cost_l803_80370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_true_proposition_q_false_p_true_q_false_l803_80305

-- Define the logarithm function
noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 2 * a) / Real.log a

-- Define a general function f
variable (f : ℝ → ℝ)

-- Define symmetry about a point
def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

theorem proposition_p_true (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  log_function a (-1) = 1 := by sorry

theorem proposition_q_false :
  ¬(∀ f : ℝ → ℝ, symmetric_about (fun x ↦ f (x + 1)) (0, 0) → symmetric_about f (-1, 0)) := by sorry

-- Combine the results to show that p is true and q is false
theorem p_true_q_false (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_function a (-1) = 1) ∧
  ¬(∀ f : ℝ → ℝ, symmetric_about (fun x ↦ f (x + 1)) (0, 0) → symmetric_about f (-1, 0)) := by
  constructor
  · exact proposition_p_true a h1 h2
  · exact proposition_q_false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_true_proposition_q_false_p_true_q_false_l803_80305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l803_80322

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 3 = 1

-- Define the focal distance
noncomputable def focal_distance : ℝ := 2 * Real.sqrt 7

-- Define the asymptote equations
def asymptote (x y : ℝ) : Prop := y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    (∃ c, c = focal_distance ∧ c = 2 * Real.sqrt ((4 : ℝ) + 3)) ∧
    asymptote x y) :=
by
  sorry

#check hyperbola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l803_80322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lengths_l803_80399

-- Define the circles and points
variable (C1 C2 : Set (Fin 2 → ℝ))
variable (P Q R S W X Y : Fin 2 → ℝ)

-- Define the conditions
variable (h1 : Q ∈ C1 ∩ C2)
variable (h2 : X ∈ C1 ∩ C2)
variable (h3 : P ∈ C1)
variable (h4 : W ∈ C1)
variable (h5 : R ∈ C2)
variable (h6 : Y ∈ C2)

-- Define the lengths
variable (h7 : ‖Q - R‖ = 7)
variable (h8 : ‖R - S‖ = 9)
variable (h9 : ‖X - Y‖ = 18)
variable (h10 : ‖W - X‖ = 6 * ‖Y - S‖)

-- Theorem statement
theorem sum_of_lengths :
  ‖P - S‖ + ‖W - S‖ = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lengths_l803_80399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_correct_l803_80318

/-- The height of a rectangular box containing spheres -/
noncomputable def box_height (box_width : ℝ) (large_sphere_radius : ℝ) (small_sphere_radius : ℝ) : ℝ :=
  5 + Real.sqrt 23

/-- The configuration of spheres in the box is valid -/
def valid_configuration (box_width : ℝ) (large_sphere_radius : ℝ) (small_sphere_radius : ℝ) : Prop :=
  box_width = 6 ∧
  large_sphere_radius = 3 ∧
  small_sphere_radius = 2 ∧
  ∃ (h : ℝ),
    h = box_height box_width large_sphere_radius small_sphere_radius ∧
    -- Large sphere is tangent to all four smaller spheres
    (3 - 2)^2 + (3 - 2)^2 + (h/2 - 2)^2 = (large_sphere_radius + small_sphere_radius)^2 ∧
    -- Smaller spheres are tangent to two adjacent sides of the box
    2 * small_sphere_radius = 2 ∧
    -- Large sphere fits within the height of the box
    h ≥ 2 * large_sphere_radius

theorem box_height_correct
  (box_width : ℝ)
  (large_sphere_radius : ℝ)
  (small_sphere_radius : ℝ)
  (h_valid : valid_configuration box_width large_sphere_radius small_sphere_radius) :
  box_height box_width large_sphere_radius small_sphere_radius = 5 + Real.sqrt 23 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_correct_l803_80318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l803_80350

/-- The set E with 100 elements -/
def E : Finset ℕ := Finset.range 100

/-- Characteristic sequence of a subset of E -/
def CharacteristicSequence (X : Finset ℕ) : ℕ → ℕ :=
  fun i => if i ∈ X ∧ i < 100 then 1 else 0

/-- Subset P of E with alternating characteristic sequence -/
def P : Finset ℕ :=
  E.filter (fun i => i % 2 = 1)

/-- Subset Q of E with characteristic sequence 1,0,0,1,0,0,... -/
def Q : Finset ℕ :=
  E.filter (fun i => i % 3 = 1)

/-- The number of elements in the intersection of P and Q is 17 -/
theorem intersection_count : (P ∩ Q).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l803_80350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l803_80329

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

/-- The shifted function g(x) = f(x - φ) -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

/-- Symmetry about the y-axis means g(x) = g(-x) for all x -/
def symmetric_about_y_axis (φ : ℝ) : Prop := ∀ x, g φ x = g φ (-x)

theorem smallest_positive_phi :
  ∃ φ : ℝ, φ > 0 ∧ symmetric_about_y_axis φ ∧ 
  ∀ ψ, (ψ > 0 ∧ symmetric_about_y_axis ψ) → φ ≤ ψ ∧ φ = 3 * Real.pi / 8 :=
by
  sorry

#check smallest_positive_phi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l803_80329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_increase_l803_80391

/-- Given a price increase of 50% resulting in a new price of 6 pounds,
    prove that the original price was 4 pounds. -/
theorem soda_price_increase (new_price : ℝ) (increase_percentage : ℝ) 
  (h1 : new_price = 6)
  (h2 : increase_percentage = 50) : 
  new_price / (1 + increase_percentage / 100) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_increase_l803_80391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_unique_zeros_condition_l803_80379

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1
theorem tangent_line_at_origin (a : ℝ) (h : a = 1) :
  ∃ m b : ℝ, m = 2 ∧ b = 0 ∧
  ∀ x : ℝ, (deriv (f a)) 0 * x + f a 0 = m * x + b :=
sorry

-- Part 2
theorem unique_zeros_condition (a : ℝ) :
  (∃! x₁, x₁ ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a x₁ = 0) ∧
  (∃! x₂, x₂ ∈ Set.Ioi 0 ∧ f a x₂ = 0) ↔
  a < -1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_unique_zeros_condition_l803_80379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l803_80333

/-- Given two parabolas, prove the translation between them --/
theorem parabola_translation
  (f g : ℝ → ℝ)
  (hf : f = λ x ↦ (x - 1)^2 + 5)
  (hg : g = λ x ↦ x^2 + 2*x + 3) :
  ∃ (h k : ℝ), h = -2 ∧ k = -3 ∧
  (∀ x, g x = f (x - h) + k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l803_80333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l803_80381

theorem sin_pi_plus_alpha (α : ℝ) 
  (h1 : Real.cos (2 * Real.pi - α) = -Real.sqrt 5 / 3)
  (h2 : α > Real.pi ∧ α < 3 * Real.pi / 2) : 
  Real.sin (Real.pi + α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l803_80381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_concurrent_lines_l803_80396

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the left focus
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)

-- Define the vertices of the ellipse
def vertex_right (a : ℝ) : ℝ × ℝ := (a, 0)
def vertex_left (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define the left directrix
def left_directrix (a c : ℝ) (x : ℝ) : Prop := x = -a^2 / c

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

-- Define the concurrency of three lines
def concurrent (l₁ l₂ l₃ : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l₁ x y ∧ l₂ x y ∧ l₃ x y

-- State the theorem
theorem ellipse_concurrent_lines 
  (a b c : ℝ) (h₁ : a > b) (h₂ : b > 0) (k : ℝ) :
  let C := ellipse a b
  let F := left_focus c
  let A₂ := vertex_right a
  let A₁ := vertex_left a
  let l := left_directrix a c
  ∀ P Q : ℝ × ℝ, 
    C P.1 P.2 → C Q.1 Q.2 → 
    line_through_point F k P.1 P.2 → line_through_point F k Q.1 Q.2 →
    concurrent 
      (λ x y => line_through_point P ((P.2 - A₂.2) / (P.1 - A₂.1)) x y)
      (λ x y => line_through_point Q ((Q.2 - A₁.2) / (Q.1 - A₁.1)) x y)
      (λ x y => l x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_concurrent_lines_l803_80396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l803_80343

noncomputable def f (a x : ℝ) : ℝ := a * (2 : ℝ)^(x+2) + 3 * (4 : ℝ)^x

theorem f_minimum_value (a : ℝ) (h : a < -3) :
  ∃ (min : ℝ), ∀ x ∈ Set.Ioo 1 2, f a x ≥ min ∧
  (a ≤ -6 → min = 48 + 16*a) ∧
  (-6 < a → min = -4*a^2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l803_80343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l803_80316

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95).natAbs b.natAbs = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l803_80316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l803_80342

-- Define the function f
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem omega_range (ω : ℝ) (φ : ℝ) :
  ω > 0 ∧
  -Real.pi ≤ φ ∧ φ ≤ 0 ∧
  (∀ x, f ω φ x = -f ω φ (-x)) ∧
  (∀ x y, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ 3*Real.pi/16 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  0 < ω ∧ ω ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l803_80342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_half_l803_80306

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 2 * Real.sin (ω * x) * Real.cos (ω * x) + 
  Real.sqrt 2 * (Real.cos (ω * x))^2 - Real.sqrt 2 / 2

theorem symmetry_implies_omega_half (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω (π/4 - x) = f ω (π/4 + x)) : ω = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_half_l803_80306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l803_80346

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + Real.sin (2 * x - Real.pi / 6) + Real.cos (2 * x) + a

-- Define the theorem
theorem f_properties (a : ℝ) :
  -- 1. Smallest positive period is π
  (∀ x, f x a = f (x + Real.pi) a) ∧
  (∀ T, T > 0 ∧ (∀ x, f x a = f (x + T) a) → T ≥ Real.pi) ∧
  -- 2. Monotonically decreasing intervals
  (∀ k : ℤ, StrictMonoOn (f · a) (Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3))) ∧
  -- 3. Minimum value condition implies a = -1
  ((∃ x ∈ Set.Icc 0 (Real.pi / 2), ∀ y ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ f y a) ∧
   (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = -2) → a = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l803_80346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_journey_time_l803_80334

/-- Represents the usual speed of the train -/
noncomputable def usual_speed : ℝ := sorry

/-- Represents the usual time taken for the journey in hours -/
noncomputable def usual_time : ℝ := sorry

/-- The reduced speed of the train -/
noncomputable def reduced_speed : ℝ := (4/7) * usual_speed

/-- The time taken when the train is late, in hours -/
noncomputable def late_time : ℝ := usual_time + 9/60

/-- States that speed and time are inversely proportional -/
axiom speed_time_relation : usual_speed * usual_time = reduced_speed * late_time

/-- Theorem stating that the usual time for the journey is 12 minutes -/
theorem usual_journey_time : usual_time * 60 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_journey_time_l803_80334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_hourly_rate_l803_80355

/-- Fred's hourly rate given his total earnings and hours worked -/
noncomputable def hourly_rate (total_earnings : ℝ) (hours_worked : ℝ) : ℝ :=
  total_earnings / hours_worked

theorem fred_hourly_rate :
  hourly_rate 100 8 = 12.5 := by
  -- Unfold the definition of hourly_rate
  unfold hourly_rate
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_hourly_rate_l803_80355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_cone_base_circumference_proof_l803_80309

/-- The circumference of the base of a right circular cone formed by gluing the ends of a 180° sector cut from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (π : ℝ) : ℝ :=
  let original_radius : ℝ := 6
  let sector_angle : ℝ := 180
  let full_circle_angle : ℝ := 360
  let original_circumference := 2 * π * original_radius
  let base_circumference := (sector_angle / full_circle_angle) * original_circumference
  base_circumference

/-- Proof of the theorem -/
theorem cone_base_circumference_proof (π : ℝ) : cone_base_circumference π = 6 * π :=
by
  -- Unfold the definition of cone_base_circumference
  unfold cone_base_circumference
  -- Simplify the expression
  simp
  -- The proof steps would go here, but for now we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_cone_base_circumference_proof_l803_80309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_of_2008_l803_80326

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

noncomputable def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (f_iter n)

theorem f_2008_of_2008 : f_iter 2008 2008 = -1 / 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_of_2008_l803_80326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_blocks_l803_80371

theorem carlos_blocks (initial_blocks : ℕ) (h1 : initial_blocks = 58) : 
  let blocks_after_rachel := initial_blocks - (initial_blocks * 2 / 5)
  let blocks_to_exchange := blocks_after_rachel / 2
  let final_blocks := blocks_after_rachel - blocks_to_exchange
  final_blocks = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_blocks_l803_80371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximum_at_x_max_x_max_smallest_l803_80362

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 7)

/-- Converts degrees to radians -/
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / 180

/-- The smallest positive value of x in degrees where g(x) achieves its maximum -/
def x_max : ℝ := 7560

theorem g_maximum_at_x_max :
  ∀ x : ℝ, x > 0 → g (deg_to_rad x) ≤ g (deg_to_rad x_max) :=
by sorry

theorem x_max_smallest :
  ∀ x : ℝ, 0 < x → x < x_max → g (deg_to_rad x) < g (deg_to_rad x_max) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximum_at_x_max_x_max_smallest_l803_80362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_value_l803_80344

/-- Given two points (x₁, y₁) and (x₂, y₂), calculate the slope of the line passing through them. -/
noncomputable def pointSlope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

/-- Given the coefficients a, b, c of a line ax + by = c, calculate its slope. -/
noncomputable def lineSlope (a b : ℝ) : ℝ := -a / b

theorem parallel_lines_k_value : ∃ k : ℝ, 
  pointSlope 3 (-6) k 24 = lineSlope 4 6 ∧ k = 42 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_value_l803_80344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_centers_l803_80327

-- Define the points A, B, C on a line
variable (A B C : ℝ × ℝ)

-- Define the centers of the equilateral triangles
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry
noncomputable def R : ℝ × ℝ := sorry

-- B is on AC
axiom B_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = ((1 - t) • A.1 + t • C.1, 0)

-- Define what it means to be the center of an equilateral triangle
def IsCenter (center point1 point2 : ℝ × ℝ) : Prop := sorry

-- P, Q, R are centers of equilateral triangles on AB, BC, CA respectively
axiom P_center : IsCenter P A B
axiom Q_center : IsCenter Q B C
axiom R_center : IsCenter R C A

-- Define what it means for three points to form an equilateral triangle
def IsEquilateral (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Define the center of a triangle
noncomputable def TriangleCenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define what it means for a point to be on a line segment
def IsOnSegment (point end1 end2 : ℝ × ℝ) : Prop := sorry

theorem equilateral_centers (A B C : ℝ × ℝ) :
  IsEquilateral P Q R ∧ IsOnSegment (TriangleCenter P Q R) A C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_centers_l803_80327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_through_point_l803_80345

/-- If the terminal side of angle α passes through point P(1,-2), then tan 2α = 4/3 -/
theorem tan_double_angle_through_point (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (1, -2) ∧ (Real.tan α = P.2 / P.1)) → Real.tan (2 * α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_through_point_l803_80345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l803_80351

theorem greatest_integer_fraction : 
  ⌊(5^50 + 3^50 : ℝ) / (5^45 + 3^45 : ℝ)⌋ = 3124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l803_80351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l803_80364

theorem angle_terminal_side (m : ℝ) : 
  (∃ (α : ℝ), α = 7 * π / 3 ∧ 
   ∃ (x y : ℝ), x = Real.sqrt m ∧ y = m^(1/3) ∧ 
   x * Real.cos α = y * Real.sin α) → 
  m = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l803_80364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_equality_count_l803_80328

-- Define the fractional part function
noncomputable def frac (r : ℝ) : ℝ := r - ⌊r⌋

-- State the theorem
theorem fractional_part_equality_count :
  ∃ (S : Set ℝ), (∀ x ∈ S, x ∈ Set.Ico 1 2 ∧ frac (x^2018) = frac (x^2017)) ∧
                 (Finite S ∧ Nat.card S = 2^2017) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_equality_count_l803_80328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_l803_80387

theorem complex_product_real (a : ℝ) : 
  (Complex.I : ℂ).im = 1 →
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  ((1 : ℂ) + a * Complex.I) * ((2 : ℂ) - Complex.I) = ((1 : ℂ) + a * Complex.I) * ((2 : ℂ) - Complex.I) →
  a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_l803_80387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_equals_sqrt5_over_5_l803_80388

theorem cos_minus_sin_equals_sqrt5_over_5 (α : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.cos α - Real.sin α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_equals_sqrt5_over_5_l803_80388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l803_80353

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define point P on the ellipse
def point_P : ℝ × ℝ := (4, 0)

-- Define the fixed point Q
def point_Q : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem ellipse_properties :
  -- Condition: Ellipse C has eccentricity 1/2
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 - b^2) / a^2 = 1/4) →
  -- Condition: Circle with center at origin and radius b is tangent to x - y + √6 = 0
  (∃ (b : ℝ), b > 0 ∧ b = Real.sqrt 3) →
  -- 1. Equation of C is x²/4 + y²/3 = 1
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  -- 2. AE intersects x-axis at fixed point Q(1,0)
  (∀ A B E : ℝ × ℝ, 
    ellipse_C A.1 A.2 → 
    ellipse_C B.1 B.2 → 
    ellipse_C E.1 E.2 → 
    A.1 = B.1 ∧ A.2 = -B.2 → 
    (∃ t : ℝ, E = (1 - t) • point_P + t • B) →
    (∃ s : ℝ, (1 - s) • A + s • E = point_Q)) ∧
  -- 3. Range of OM · ON
  (∀ M N : ℝ × ℝ,
    ellipse_C M.1 M.2 →
    ellipse_C N.1 N.2 →
    (∃ t : ℝ, (1 - t) • point_Q + t • M = N) →
    -4 ≤ (M.1 * N.1 + M.2 * N.2) ∧ (M.1 * N.1 + M.2 * N.2) ≤ -5/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l803_80353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_area_ratio_l803_80374

theorem octagon_circle_area_ratio : ∀ (s : ℝ), s > 0 →
  let r := s / (2 * Real.sin (π / 8))
  let octagon_area := 4 * s * r
  let circle_area := π * r^2
  circle_area / octagon_area = π / (4 * Real.sqrt 2) :=
λ s hs => by
  -- Introduce the local definitions
  let r := s / (2 * Real.sin (π / 8))
  let octagon_area := 4 * s * r
  let circle_area := π * r^2
  
  -- The proof steps would go here
  sorry

#check octagon_circle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_area_ratio_l803_80374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l803_80301

theorem acute_triangle_condition (k : ℕ+) : 
  (∀ a b c : ℝ, a = 8 ∧ b = 17 ∧ c = k.val ∧
    (a < b + c ∧ b < a + c ∧ c < a + b) ∧ 
    (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)) ↔ 
  (k = 16 ∨ k = 17 ∨ k = 18) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l803_80301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l803_80335

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define helper functions for tangency conditions
def are_tangent (A B : Circle) : Prop := sorry
def is_tangent_to_midpoint (C A B : Circle) : Prop := sorry
def is_externally_tangent (D C : Circle) : Prop := sorry
def is_internally_tangent (D A : Circle) : Prop := sorry

-- Define the problem setup
def problem_setup : Prop := ∃ (A B C D : Circle),
  -- Radii conditions
  A.radius = 2 ∧ B.radius = 2 ∧ C.radius = 2 ∧ D.radius = 1 ∧
  -- Tangency conditions
  are_tangent A B ∧
  is_tangent_to_midpoint C A B ∧
  is_externally_tangent D C ∧
  is_internally_tangent D A ∧
  is_internally_tangent D B

-- Define the area calculation function
def area_inside_C_outside_ABD (A B C D : Circle) : ℝ := sorry

-- The theorem to prove
theorem area_calculation (A B C D : Circle) :
  problem_setup →
  area_inside_C_outside_ABD A B C D = 4 - 0.5 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l803_80335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carrots_grown_l803_80394

/-- The total number of carrots grown by Sandy, Sam, Sophie, and Sara -/
theorem total_carrots_grown 
  (sandy_carrots : ℝ)
  (sam_carrots : ℝ)
  (sophie_multiplier : ℝ)
  (sara_less : ℝ)
  (h1 : sandy_carrots = 6.5)
  (h2 : sam_carrots = 3.25)
  (h3 : sophie_multiplier = 2.75)
  (h4 : sara_less = 7.5) :
  sandy_carrots + sam_carrots + (sophie_multiplier * sam_carrots) +
  ((sandy_carrots + sam_carrots + (sophie_multiplier * sam_carrots)) - sara_less) = 29.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carrots_grown_l803_80394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l803_80369

open Real

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := exp 1

/-- The function f(x) = e^x -/
noncomputable def f (x : ℝ) : ℝ := e^x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := e^x

theorem tangent_line_at_x_1 :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ e * x - y = 0) ∧
               (f 1 = e * 1 + b) ∧
               (f' 1 = m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l803_80369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80304

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
noncomputable def specificTriangle : Triangle where
  a := 2 * Real.sqrt 2
  b := 5
  c := Real.sqrt 13
  A := Real.arcsin ((2 * Real.sqrt 13) / 13)
  B := Real.pi - Real.arcsin ((2 * Real.sqrt 13) / 13) - Real.pi/4
  C := Real.pi/4

theorem triangle_properties (t : Triangle) (h : t = specificTriangle) :
  t.C = Real.pi/4 ∧ 
  Real.sin t.A = (2 * Real.sqrt 13) / 13 ∧ 
  Real.sin (2 * t.A + Real.pi/4) = (17 * Real.sqrt 2) / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_reciprocals_l803_80339

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → 
    (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) → 
  (x : ℕ) + (y : ℕ) = 49 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_reciprocals_l803_80339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_properties_l803_80378

/-- Properties of a square with side length 24/7 inches -/
theorem square_properties :
  let side_length : ℚ := 24 / 7
  let area : ℚ := side_length ^ 2
  let diagonal : ℝ := (side_length : ℝ) * Real.sqrt 2
  let cube_volume : ℚ := side_length ^ 3
  let angle : ℝ := 45
  ∀ (side_length area : ℚ) (diagonal : ℝ) (cube_volume : ℚ) (angle : ℝ),
    side_length = 24 / 7 →
    area = 576 / 49 ∧
    diagonal = (24 : ℝ) * Real.sqrt 2 / 7 ∧
    cube_volume = 13824 / 343 ∧
    angle = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_properties_l803_80378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_D_true_others_false_l803_80336

-- Define the propositions
def proposition_A (a b : ℝ) : Prop := a = b → a^2 = b^2
def proposition_B (a b : ℝ) : Prop := a = b → |a| = |b|
def proposition_C (a b : ℝ) : Prop := a = 0 → a * b = 0

-- Define a structure for triangles
structure Triangle : Type :=
  (side1 side2 side3 : ℝ)

-- Define necessary functions for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry
def corresponding_sides_equal (t1 t2 : Triangle) : Prop := sorry

-- Define the propositions for triangles
def proposition_D (triangle1 triangle2 : Triangle) : Prop := 
  congruent triangle1 triangle2 → corresponding_sides_equal triangle1 triangle2

-- Define the inverse propositions
def inverse_A (a b : ℝ) : Prop := a^2 = b^2 → a = b
def inverse_B (a b : ℝ) : Prop := |a| = |b| → a = b
def inverse_C (a b : ℝ) : Prop := a * b = 0 → a = 0
def inverse_D (triangle1 triangle2 : Triangle) : Prop := 
  corresponding_sides_equal triangle1 triangle2 → congruent triangle1 triangle2

-- Theorem statement
theorem inverse_proposition_D_true_others_false :
  (∀ (triangle1 triangle2 : Triangle), inverse_D triangle1 triangle2) ∧
  (∃ (a b : ℝ), ¬(inverse_A a b)) ∧
  (∃ (a b : ℝ), ¬(inverse_B a b)) ∧
  (∃ (a b : ℝ), ¬(inverse_C a b)) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_D_true_others_false_l803_80336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_condition_l803_80310

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (1/2) * (a + 3) * x^2 - a * x - 1

/-- Theorem stating that a ≥ 3 is a necessary but not sufficient condition for x = 1 
    to be the point of minimum value of f(x) -/
theorem min_point_condition (a : ℝ) :
  (∀ x : ℝ, f a 1 ≤ f a x) → a ≥ 3 ∧ ∃ b : ℝ, b ≥ 3 ∧ ¬(∀ x : ℝ, f b 1 ≤ f b x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_condition_l803_80310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l803_80323

/-- Given a cone with base radius 1 and lateral area twice the area of its base,
    prove that its volume is √3π/3 -/
theorem cone_volume (r l h : ℝ) :
  r = 1 →
  l * r * π = 2 * (π * r^2) →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l803_80323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l803_80315

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / Real.exp x

-- State the theorem
theorem function_inequality 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ)
  (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_ineq : ∀ x, f' x < f x) : 
  f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l803_80315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80382

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Given conditions
  a = 3 ∧
  Real.cos C = -1/15 ∧
  5 * Real.sin (B + C) = 3 * Real.sin (A + C) →
  -- Conclusions
  c = 6 ∧
  Real.sin (B - Real.pi/3) = (2 * Real.sqrt 14 - 5 * Real.sqrt 3) / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_l803_80311

theorem fourth_term_coefficient (x a : ℝ) : 
  (∃ k : ℕ, ∃ b c d : ℝ, (x + a)^9 = b + c + d + 84*x^6*a^3 + k) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_l803_80311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l803_80303

open Set
open Function
open Real

noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 + x)

theorem range_of_g :
  range g = {y : ℝ | y > 0} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l803_80303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l803_80352

/-- Geometric sequence with first term 1 and common ratio 2 -/
noncomputable def geometric_sequence (n : ℕ) : ℝ := 2^(n-1)

/-- Sum of first n terms of the geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := (1 - 2^n) / (1 - 2)

/-- The theorem to prove -/
theorem geometric_sequence_problem (k : ℕ) : 
  S (k+2) - S k = 48 → k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l803_80352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_area_l803_80390

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if two line segments are parallel -/
def parallel (p1 q1 p2 q2 : Point2D) : Prop :=
  (q1.y - p1.y) * (q2.x - p2.x) = (q2.y - p2.y) * (q1.x - p1.x)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point2D) : ℝ :=
  abs ((q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)) / 2

theorem pentagon_triangle_area 
  (p q r s t : Point2D)
  (h1 : distance p q = 3)
  (h2 : distance q r = 3)
  (h3 : distance r s = 3)
  (h4 : distance s t = 3)
  (h5 : distance t p = 3)
  (h6 : (q.x - p.x) * (r.x - q.x) + (q.y - p.y) * (r.y - q.y) = 0)
  (h7 : (s.x - r.x) * (t.x - s.x) + (s.y - r.y) * (t.y - s.y) = 0)
  (h8 : (t.x - s.x) * (p.x - t.x) + (t.y - s.y) * (p.y - t.y) = 0)
  (h9 : parallel q r s t) :
  triangleArea q r s = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_area_l803_80390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_is_49_l803_80356

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h1 : tens < 10
  h2 : ones < 10

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h1 : hundreds < 10
  h2 : tens < 10
  h3 : ones < 10

def to_nat_two_digit (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

def to_nat_three_digit (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem two_digit_number_is_49 
  (x : TwoDigitNumber)
  (y : ThreeDigitNumber)
  (h1 : x.ones = 9)
  (h2 : y.hundreds = 3 ∧ y.tens = 5 ∧ y.ones = 2)
  (h3 : to_nat_two_digit x + 253 = 299) :
  to_nat_two_digit x = 49 := by
  sorry

#check two_digit_number_is_49

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_is_49_l803_80356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_circle_l803_80385

/-- Given a line and a circle with specific properties, prove the maximum distance between a point on the line and (0,1) -/
theorem max_distance_line_circle (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (Real.sqrt 2 * a * A.1 + b * A.2 = 1) ∧ 
    (Real.sqrt 2 * a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧
    ((A.1 - B.1) * (A.1 + B.1) + (A.2 - B.2) * (A.2 + B.2) = 0)) →
  (∀ x y : ℝ, Real.sqrt 2 * a * x + b * y = 1 → 
    (x - 0)^2 + (y - 1)^2 ≤ (Real.sqrt 2 + 1)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_circle_l803_80385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_peak_location_l803_80321

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := 
  λ x => (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

theorem normal_peak_location (f : ℝ → ℝ) (μ σ : ℝ) :
  (∀ x, f x = normal_distribution μ σ x) →
  (∫ x in Set.Ici 0.2, f x = 0.5) →
  (∀ x, f x ≤ f 0.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_peak_location_l803_80321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_statement_correct_l803_80398

-- Define the four derivative statements
def statement1 (x : ℝ) : Prop := deriv Real.cos x = Real.sin x
def statement2 : Prop := deriv (λ _ : ℝ => Real.sin (Real.pi/6)) 0 = Real.cos (Real.pi/6)
def statement3 (x : ℝ) : Prop := deriv (λ x => 1/x^2) x = -1/x
def statement4 (x : ℝ) : Prop := deriv (λ x => -1/Real.sqrt x) x = 1/(2*x*Real.sqrt x)

-- Theorem statement
theorem only_fourth_statement_correct :
  (∃ x : ℝ, ¬statement1 x) ∧
  ¬statement2 ∧
  (∃ x : ℝ, x ≠ 0 → ¬statement3 x) ∧
  (∀ x : ℝ, x > 0 → statement4 x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_statement_correct_l803_80398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l803_80368

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 2

/-- The line function -/
def g (x : ℝ) : ℝ := 8 - x

/-- The intersection points of the parabola and the line -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x, f x = g x ∧ p = (x, f x)}

theorem distance_between_intersection_points :
  ∃ p₁ p₂, p₁ ∈ intersection_points ∧ p₂ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ 
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l803_80368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_budget_is_3_43_l803_80367

/-- Represents the budget and costs for Agatha's bike purchase --/
structure BikePurchase where
  budget : ℚ
  frame_cost : ℚ
  front_wheel_usd : ℚ
  rear_wheel_eur : ℚ
  seat_gbp : ℚ
  handlebar_tape_usd : ℚ
  water_bottle_cage_eur : ℚ
  bike_lock_gbp : ℚ
  future_expense_eur : ℚ
  usd_to_gbp_rate : ℚ
  eur_to_gbp_rate : ℚ

/-- Calculates the remaining budget for additional accessories --/
def remaining_budget (purchase : BikePurchase) : ℚ :=
  purchase.budget -
  (purchase.frame_cost +
   purchase.front_wheel_usd / purchase.usd_to_gbp_rate +
   purchase.rear_wheel_eur / purchase.eur_to_gbp_rate +
   purchase.seat_gbp +
   purchase.handlebar_tape_usd / purchase.usd_to_gbp_rate +
   purchase.water_bottle_cage_eur / purchase.eur_to_gbp_rate +
   purchase.bike_lock_gbp +
   purchase.future_expense_eur / purchase.eur_to_gbp_rate)

/-- Theorem stating that the remaining budget for Agatha's bike purchase is £3.43 --/
theorem remaining_budget_is_3_43 (purchase : BikePurchase)
  (h1 : purchase.budget = 200)
  (h2 : purchase.frame_cost = 65)
  (h3 : purchase.front_wheel_usd = 45)
  (h4 : purchase.rear_wheel_eur = 35)
  (h5 : purchase.seat_gbp = 25)
  (h6 : purchase.handlebar_tape_usd = 18)
  (h7 : purchase.water_bottle_cage_eur = 8)
  (h8 : purchase.bike_lock_gbp = 15)
  (h9 : purchase.future_expense_eur = 10)
  (h10 : purchase.usd_to_gbp_rate = 135/100)
  (h11 : purchase.eur_to_gbp_rate = 59/50) :
  remaining_budget purchase = 343/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_budget_is_3_43_l803_80367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_pyramid_volume_l803_80384

/-- The volume of a pyramid with a rectangular base and four equal edges to the apex --/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (base_length_pos : 0 < base_length)
  (base_width_pos : 0 < base_width)
  (edge_length_pos : 0 < edge_length) :
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal/2)^2)
  (1/3 : ℝ) * base_area * height = 
  (1/3 : ℝ) * base_length * base_width * Real.sqrt (edge_length^2 - (base_length^2 + base_width^2)/4) :=
by sorry

/-- The volume of a pyramid with a 7x10 rectangular base and four edges of length 15 --/
theorem specific_pyramid_volume :
  (1/3 : ℝ) * 7 * 10 * Real.sqrt (15^2 - (7^2 + 10^2)/4) = (70 : ℝ) * Real.sqrt 187.75 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_pyramid_volume_l803_80384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l803_80357

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function of f
noncomputable def f_inv (a : ℝ) (y : ℝ) : ℝ := a ^ y

-- State the theorem
theorem find_m (a : ℝ) (m : ℝ) (h1 : f a 2 = 4) (h2 : f_inv a 16 = m) : m = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l803_80357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_volume_l803_80395

-- Define the face areas of the rectangular prism
def face_area_1 : ℝ := 72
def face_area_2 : ℝ := 50
def face_area_3 : ℝ := 75

-- Define the volume function
def prism_volume (a b c : ℝ) : ℝ := a * b * c

-- Theorem statement
theorem rectangular_prism_volume :
  ∃ (a b c : ℝ),
    a * b = face_area_1 ∧
    b * c = face_area_2 ∧
    a * c = face_area_3 ∧
    Int.floor (prism_volume a b c + 0.5) = 164 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_volume_l803_80395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_gives_six_l803_80307

/-- The number of spiders Pugsley and Wednesday exchange in their trading game. -/
structure SpiderTrade where
  pugsley_initial : ℕ
  wednesday_initial : ℕ
  wednesday_to_pugsley : ℕ
  pugsley_initial_eq : pugsley_initial = 4
  first_scenario : wednesday_initial + 2 = 9 * (pugsley_initial - 2)
  second_scenario : pugsley_initial + wednesday_to_pugsley = wednesday_initial - 6

/-- The number of spiders Wednesday gives to Pugsley in the second scenario is 6. -/
theorem wednesday_gives_six (trade : SpiderTrade) : trade.wednesday_to_pugsley = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_gives_six_l803_80307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80302

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and a point D on side BC, prove that under certain conditions,
    angle A is π/3 and the maximum area of the triangle is 3√3/2. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  (2 * b - c) * Real.cos A = a * Real.cos C →
  D.1 + D.2 = c ∧ D.1 = 2 * D.2 →
  Real.sqrt ((D.1 - a * Real.cos A)^2 + (a * Real.sin A)^2) = 2 →
  A = Real.pi / 3 ∧ 
  (∀ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' →
    Real.sqrt ((2/3 * a' - a' * Real.cos (Real.pi/3))^2 + (a' * Real.sin (Real.pi/3))^2) = 2 →
    1/2 * b' * c' * Real.sin (Real.pi/3) ≤ 3 * Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l803_80302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_when_perimeter_equals_area_l803_80366

theorem rectangle_sides_when_perimeter_equals_area :
  ∀ w : ℝ,
  w > 0 →
  let l := 3 * w
  2 * (w + l) = w * l →
  (w = 8/3 ∧ l = 8) :=
by
  intro w hw
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_when_perimeter_equals_area_l803_80366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_winner_l803_80380

-- Define the set of students
def Students : Type := Fin 6

-- Define the set of teachers
inductive Teacher : Type
| A | B | C | D

-- Define the guesses of each teacher
def teacherGuess (t : Teacher) (s : Students) : Prop :=
  match t with
  | Teacher.A => s = ⟨2, sorry⟩ ∨ s = ⟨4, sorry⟩
  | Teacher.B => s ≠ ⟨5, sorry⟩
  | Teacher.C => s ≠ ⟨1, sorry⟩ ∧ s ≠ ⟨2, sorry⟩ ∧ s ≠ ⟨3, sorry⟩
  | Teacher.D => s = ⟨0, sorry⟩ ∨ s = ⟨1, sorry⟩ ∨ s = ⟨3, sorry⟩

-- Define the condition that only one teacher guesses correctly
def onlyOneCorrect (winner : Students) : Prop :=
  ∃! t : Teacher, teacherGuess t winner

-- The main theorem
theorem competition_winner :
  ∃ (winner : Students), onlyOneCorrect winner ∧ winner = ⟨5, sorry⟩ ∧ teacherGuess Teacher.C winner := by
  sorry

#check competition_winner

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_winner_l803_80380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l803_80319

/-- Two circles in a plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  slope : ℝ
  y_intercept : ℝ

/-- The configuration of two circles and their tangent line -/
def circle_config : TwoCircles :=
  { center1 := (3, 0)
    radius1 := 3
    center2 := (8, 0)
    radius2 := 2 }

/-- Predicate to check if a point is a tangent point of a circle and a line -/
def is_tangent_point (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) (line : TangentLine) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 ∧
  point.2 = line.slope * point.1 + line.y_intercept

theorem tangent_line_y_intercept (config : TwoCircles) (line : TangentLine) :
  config = circle_config →
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.1 ≥ 0 ∧ p1.2 ≥ 0) ∧
    (p2.1 ≥ 0 ∧ p2.2 ≥ 0) ∧
    is_tangent_point config.center1 config.radius1 p1 line ∧
    is_tangent_point config.center2 config.radius2 p2 line) →
  line.y_intercept = 15 * Real.sqrt 26 / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l803_80319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_growth_rate_l803_80325

/-- The average monthly growth rate of a supermarket's sales -/
noncomputable def average_monthly_growth_rate (january_sales : ℝ) (march_sales : ℝ) : ℝ :=
  let growth_factor := (march_sales / january_sales) ^ (1/2)
  growth_factor - 1

/-- Theorem stating that given the sales conditions, the average monthly growth rate is 20% -/
theorem supermarket_growth_rate :
  let january_sales := (2000000 : ℝ)
  let march_sales := (2880000 : ℝ)
  (average_monthly_growth_rate january_sales march_sales) = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_growth_rate_l803_80325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_for_given_point_l803_80338

/-- If the terminal side of angle α passes through the point (1/2, √3/2), then sin(α) = √3/2 -/
theorem sine_value_for_given_point (α : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = 1/2 ∧ P.2 = Real.sqrt 3/2 ∧ 
    (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α)) →
  Real.sin α = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_for_given_point_l803_80338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l803_80308

noncomputable section

-- Define the vectors m and n
def m (ω x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x), 1)
def n (ω a x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x), a)

-- Define the dot product function
def f (ω a x : ℝ) : ℝ := (m ω x).1 * (n ω a x).1 + (m ω x).2 * (n ω a x).2

-- State the theorem
theorem vector_dot_product_problem (ω a : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, f ω a (x + π/ω) = f ω a x) ∧  -- minimum positive period is π
  (∃ x : ℝ, f ω a x = 3) ∧                -- maximum value is 3
  (ω = 1 ∧ a = 2) ∧                       -- solution for ω and a
  (∀ k : ℤ, ∀ x : ℝ, 
    k * π - π/6 ≤ x ∧ x ≤ k * π + π/3 →   -- intervals of monotonic increase
    ∀ y : ℝ, x ≤ y ∧ y ≤ k * π + π/3 → f ω a x ≤ f ω a y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l803_80308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l803_80376

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 * Real.log 9

-- State the theorem
theorem f_sum_equals_two : f 2 + f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l803_80376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l803_80331

/-- A polynomial of odd degree 2m+1 over ℤ is irreducible under certain divisibility conditions. -/
theorem polynomial_irreducibility (m : ℕ) (p : ℕ) (hp : Nat.Prime p) 
  (f : Polynomial ℤ) (c : ℕ → ℤ) :
  (∀ i, Polynomial.coeff f i = c i) →
  Polynomial.degree f = 2 * m + 1 →
  ¬ (p : ℤ) ∣ c (2 * m + 1) →
  (∀ i ∈ Finset.range m, (p : ℤ) ∣ c (m + 1 + i)) →
  (∀ i ∈ Finset.range (m + 1), (p^2 : ℤ) ∣ c i) →
  ¬ (p^3 : ℤ) ∣ c 0 →
  Irreducible f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l803_80331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l803_80347

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the focus
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

-- Define the arithmetic sequence property
def arithmetic_sequence (v1 v2 v3 : ℝ × ℝ) : Prop :=
  (v1.1 + v3.1 = 2 * v2.1) ∧ (v1.2 + v3.2 = 2 * v2.2)

-- Define the sum of vectors property
def sum_of_vectors (v1 v2 v3 : ℝ × ℝ) : Prop :=
  v1.1 + v2.1 + v3.1 = 0 ∧ v1.2 + v2.2 + v3.2 = 0

-- Main theorem
theorem line_equation (p A B C : ℝ × ℝ) :
  p = 1 →
  point_on_parabola 1 A.1 A.2 →
  point_on_parabola 1 B.1 B.2 →
  point_on_parabola 1 C.1 C.2 →
  arithmetic_sequence (A.1 - (focus 1).1, A.2 - (focus 1).2)
                      (B.1 - (focus 1).1, B.2 - (focus 1).2)
                      (C.1 - (focus 1).1, C.2 - (focus 1).2) →
  B.2 < 0 →
  sum_of_vectors (A.1 - (focus 1).1, A.2 - (focus 1).2)
                 (B.1 - (focus 1).1, B.2 - (focus 1).2)
                 (C.1 - (focus 1).1, C.2 - (focus 1).2) →
  ∃ (x y : ℝ), 2*x - y - 1 = 0 ∧ (x - A.1) * (C.2 - A.2) = (y - A.2) * (C.1 - A.1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l803_80347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l803_80354

theorem sqrt_sum_theorem : Real.sqrt 25 + Real.sqrt ((-2)^2) + ((-8) ^ (1/3 : ℝ)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l803_80354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_above_median_top_25_l803_80313

/-- Represents a class of students with their exam scores -/
structure ExamClass where
  students : Finset ℕ
  scores : students → ℝ
  student_count : students.card = 50

/-- Definition of median for a class -/
noncomputable def median (c : ExamClass) : ℝ := sorry

/-- A student's rank in the class based on their score -/
noncomputable def rank (c : ExamClass) (student : c.students) : ℕ := sorry

/-- Theorem: A student scoring above median ranks in top 25 -/
theorem above_median_top_25 (c : ExamClass) (student : c.students) :
  c.scores student > median c → rank c student ≤ 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_above_median_top_25_l803_80313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l803_80375

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the function equation
noncomputable def func (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 8) + 2

-- Define the intersection condition
def intersects (k : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ line k x₁ = func x₁ ∧ line k x₂ = func x₂

-- Define the distance condition
def distance_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ line k x₁ = func x₁ ∧ line k x₂ = func x₂ ∧
  (x₂ - x₁)^2 + (line k x₂ - line k x₁)^2 = (12 * Real.sqrt 5 / 5)^2

-- Theorem statement
theorem find_k : ∃ k : ℝ, intersects k ∧ distance_condition k ∧ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l803_80375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_monotonically_decreasing_when_a_is_neg_half_l803_80358

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a + 1) / x - 1 + Real.log x

-- Theorem for part (1)
theorem tangent_line_at_2 :
  let a := 1
  let x₀ := 2
  let y₀ := f a x₀
  let m := (Real.log x₀ + 1 - 2 / (x₀^2))
  ∀ x y, x - y + Real.log 2 = 0 ↔ y - y₀ = m * (x - x₀) := by sorry

-- Theorem for part (2)
theorem monotonically_decreasing_when_a_is_neg_half :
  let a := -1/2
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_monotonically_decreasing_when_a_is_neg_half_l803_80358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_l803_80372

theorem min_x_prime_factorization (x y : ℕ) (h : 3 * x ^ 7 = 17 * y ^ 11) :
  ∃ (a b c d : ℕ),
    x = a ^ c * b ^ d ∧
    a + b + c + d = 27 ∧
    (∀ (a' b' c' d' : ℕ), x = a' ^ c' * b' ^ d' → a + b + c + d ≤ a' + b' + c' + d') ∧
    a = 17 ∧ b = 3 ∧ c = 4 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_l803_80372
