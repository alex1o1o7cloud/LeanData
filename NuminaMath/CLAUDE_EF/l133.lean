import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l133_13365

/-- Given vectors a and b in ℝ², prove that the angle between them is π -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a + 2 • b = (2, -4) →
  3 • a - b = (-8, 16) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l133_13365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l133_13340

-- Define the parabola and circle
def parabola (x y : ℝ) : Prop := y^2 = x
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_parabola_circle :
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ myCircle x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      parabola x3 y3 → myCircle x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = Real.sqrt 11 / 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l133_13340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_third_digit_sqrt10_plus3_pow2001_l133_13381

theorem thirty_third_digit_sqrt10_plus3_pow2001 : 
  ∃ (x : ℝ), (Real.sqrt 10 + 3)^2001 = ⌊(Real.sqrt 10 + 3)^2001⌋ + x ∧ 
  ∃ (n : ℕ), x * 10^33 = n + 0 / 10 ∧ n < 10^33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_third_digit_sqrt10_plus3_pow2001_l133_13381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l133_13342

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + Real.log x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
  (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x ≤ f c) ∧
  f c = -1/2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l133_13342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_M_property_l133_13347

noncomputable section

/-- The polar equation of curve M -/
def curve_M (θ : ℝ) : ℝ := 4 * Real.cos θ

/-- The point A on curve M -/
def point_A (φ : ℝ) : ℝ × ℝ := (curve_M φ * Real.cos φ, curve_M φ * Real.sin φ)

/-- The point B on curve M -/
def point_B (φ : ℝ) : ℝ × ℝ := 
  (curve_M (φ + Real.pi/4) * Real.cos (φ + Real.pi/4), 
   curve_M (φ + Real.pi/4) * Real.sin (φ + Real.pi/4))

/-- The point C on curve M -/
def point_C (φ : ℝ) : ℝ × ℝ := 
  (curve_M (φ - Real.pi/4) * Real.cos (φ - Real.pi/4), 
   curve_M (φ - Real.pi/4) * Real.sin (φ - Real.pi/4))

/-- The distance from O to a point -/
def distance_from_O (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

theorem curve_M_property (φ : ℝ) :
  distance_from_O (point_B φ) + distance_from_O (point_C φ) = Real.sqrt 2 * distance_from_O (point_A φ) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_M_property_l133_13347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_bug_ratio_l133_13327

theorem garden_bug_ratio (initial_plants : ℕ) (eaten_day1 : ℕ) (eaten_day3 : ℕ) (final_plants : ℕ) :
  initial_plants = 30 →
  eaten_day1 = 20 →
  eaten_day3 = 1 →
  final_plants = 4 →
  (initial_plants - eaten_day1 - (final_plants + eaten_day3) : ℚ) / (initial_plants - eaten_day1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_bug_ratio_l133_13327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_c_range_l133_13310

/-- The function f(x) = 2x^2 + cx + ln x -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + c * x + Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (c : ℝ) (x : ℝ) : ℝ := 4 * x + c + 1 / x

theorem extremum_implies_c_range :
  ∀ c : ℝ, (∃ x : ℝ, x > 0 ∧ f_derivative c x = 0 ∧
    (∀ y : ℝ, y > 0 → (f_derivative c y < 0 ∧ y < x) ∨ (f_derivative c y > 0 ∧ y > x)))
  → c < -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_c_range_l133_13310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l133_13309

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x + 2)

theorem symmetry_of_f : ∀ t : ℝ, f (-2 - t) + 1 = -f (-2 + t) + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l133_13309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_per_meter_l133_13391

/-- The cost price of one meter of cloth given the selling conditions -/
theorem cost_price_per_meter
  (cloth_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : cloth_length = 92)
  (h2 : total_selling_price = 9890)
  (h3 : profit_per_meter = 24) :
  (total_selling_price - cloth_length * profit_per_meter : ℚ) / cloth_length = 83.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_per_meter_l133_13391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l133_13328

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * log x - (1/2) * x^2 + b * x

-- State the theorem
theorem min_value_of_a (a b : ℝ) :
  (∃ x, ∀ y, f a b x ≤ f a b y) →  -- f has a minimum value
  (∀ b, ∃ x, f a b x > 0) →        -- for all b, the minimum value of f is greater than 0
  a ≥ -exp 3 :=                    -- the minimum value of a is -e^3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l133_13328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l133_13399

noncomputable def f (x : ℝ) := |Real.sin x|

theorem f_properties : 
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) ∧ 
  (∀ x, f (-x) = f x) ∧
  (∀ x, f (x + Real.pi) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l133_13399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_intersection_points_max_distance_C2_to_l_l133_13331

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 6) * t)

-- Define curve C₁
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define curve C₂
noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ := ((1/2) * Real.cos θ, (Real.sqrt 3 / 2) * Real.sin θ)

-- Theorem for the distance between intersection points
theorem distance_intersection_points :
  ∃ A B : ℝ × ℝ, (∃ θ : ℝ, curve_C1 θ = A) ∧ (∃ θ : ℝ, curve_C1 θ = B) ∧
  (∃ t : ℝ, line_l t = A) ∧ (∃ t : ℝ, line_l t = B) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 := by sorry

-- Theorem for the maximum distance from C₂ to line l
theorem max_distance_C2_to_l :
  ∃ d : ℝ, d = Real.sqrt 10 / 4 + 1 / 2 ∧
  (∀ P : ℝ × ℝ, (∃ θ : ℝ, curve_C2 θ = P) →
    ∃ Q : ℝ × ℝ, (∃ t : ℝ, line_l t = Q) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d) ∧
  (∃ P : ℝ × ℝ, (∃ θ : ℝ, curve_C2 θ = P) ∧
    ∃ Q : ℝ × ℝ, (∃ t : ℝ, line_l t = Q) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = d) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_intersection_points_max_distance_C2_to_l_l133_13331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_volume_difference_l133_13308

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (sphere_radius : ℝ) (cylinder_base_radius : ℝ) 
  (h_sphere : sphere_radius = 7)
  (h_cylinder : cylinder_base_radius = 4)
  (h_inscribed : cylinder_base_radius ≤ sphere_radius) :
  (4/3 * Real.pi * sphere_radius^3) - 
  (Real.pi * cylinder_base_radius^2 * (2 * Real.sqrt (sphere_radius^2 - cylinder_base_radius^2))) = 
  ((1372/3) - 32 * Real.sqrt 33) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_volume_difference_l133_13308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l133_13368

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being tangent
def areTangent (c1 c2 : Circle) : Prop :=
  (c1.radius + c2.radius = dist c1.center c2.center) ∨
  (|c1.radius - c2.radius| = dist c1.center c2.center)

-- State the theorem
theorem tangent_circles_radius (A B : Circle) :
  areTangent A B →
  dist A.center B.center = 5 →
  A.radius = 2 →
  B.radius = 3 ∨ B.radius = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l133_13368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QRS_l133_13329

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The angle between three points in 3D space -/
noncomputable def angle (p q r : Point3D) : ℝ := sorry

/-- Check if two vectors are perpendicular -/
def isPerpendicular (v w : Point3D) : Prop :=
  v.x * w.x + v.y * w.y + v.z * w.z = 0

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point3D) : ℝ := sorry

theorem area_of_triangle_QRS (P Q R S T : Point3D) : 
  distance P Q = 3 →
  distance Q R = 3 →
  distance R S = 3 →
  distance S T = 3 →
  distance T P = 3 →
  angle P Q R = 2 * Real.pi / 3 →
  angle R S T = 2 * Real.pi / 3 →
  angle S T P = 2 * Real.pi / 3 →
  isPerpendicular (Point3D.mk (R.x - P.x) (R.y - P.y) (R.z - P.z)) (Point3D.mk (S.x - T.x) (S.y - T.y) (S.z - T.z)) →
  triangleArea Q R S = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QRS_l133_13329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_exposed_surface_area_l133_13304

/-- Represents a cube with its volume and position in the stack -/
structure Cube where
  volume : Nat
  position : Nat

/-- Calculates the exposed surface area of a cube in the stack -/
noncomputable def exposedSurfaceArea (cube : Cube) (totalCubes : Nat) : ℚ :=
  let sideLength := (cube.volume : ℚ) ^ (1/3)
  if cube.position = 1 then
    5 * sideLength * sideLength  -- Bottom cube
  else if cube.position = totalCubes then
    6 * sideLength * sideLength  -- Top cube
  else
    5 * sideLength * sideLength  -- Middle cubes

/-- The stack of cubes -/
def cubeStack : List Cube := [
  ⟨512, 1⟩, ⟨216, 2⟩, ⟨343, 3⟩, ⟨125, 4⟩,
  ⟨64, 5⟩, ⟨27, 6⟩, ⟨8, 7⟩, ⟨1, 8⟩
]

/-- Theorem stating the total exposed surface area of the cube stack -/
theorem total_exposed_surface_area :
  (cubeStack.map (fun c => exposedSurfaceArea c cubeStack.length)).sum = 1021 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_exposed_surface_area_l133_13304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l133_13320

/-- The function f(x) = a*ln(x) - x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

/-- The theorem stating the range of a given the conditions -/
theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → 
    (f a p - f a q) / (p - q) > 1) →
  a ∈ Set.Ici 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l133_13320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l133_13369

noncomputable section

/-- Curve C defined by parametric equations -/
def curve_C (α : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 2 * Real.cos α, 1 + Real.sqrt 2 * Real.sin α)

/-- Polar equation of curve C -/
def polar_equation_C (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 2 * Real.sin θ

/-- Point A in polar coordinates -/
def point_A : ℝ × ℝ := (2, Real.pi / 4)

/-- Transformation function for curve C to C' -/
def transform_C (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1, p.2 / 2)

theorem curve_C_properties :
  /- The polar equation of curve C is ρ = 2cos(θ) + 2sin(θ) -/
  (∀ θ, polar_equation_C θ = Real.sqrt ((curve_C θ).1^2 + (curve_C θ).2^2)) ∧
  /- Point A(2, π/4) in polar coordinates is inside curve C -/
  ((Real.sqrt 2 - 1)^2 + (Real.sqrt 2 - 1)^2 < 2) ∧
  /- The equation of curve C' after transformation is (x-2)^2 + (y-1/2)^2 = 2 -/
  (∀ α, let (x, y) := transform_C (curve_C α) ; (x - 2)^2 + (y - 1/2)^2 = 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l133_13369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_implies_a_ge_one_l133_13352

/-- A function f is decreasing on an interval if for any two points x and y in that interval,
    x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

theorem quadratic_decreasing_implies_a_ge_one :
  ∀ a : ℝ,
  let f := fun x : ℝ ↦ x^2 - 2*a*x + 2
  DecreasingOn f { x : ℝ | x ≤ 1 } →
  a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_implies_a_ge_one_l133_13352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_false_l133_13312

-- Define purely imaginary complex numbers
def PurelyImaginary (z : ℂ) : Prop := z.re = 0

-- Define propositions p and q
def p : Prop := ∀ x : ℝ, x ∈ [0, 1] → Real.exp x ≥ 1
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define the main theorem
theorem exactly_one_false : ∃! n : Fin 4, ¬ (
  -- Statement 1
  (∀ z : ℂ, PurelyImaginary z → z.re = 0) ∧ 
  (∃ z : ℂ, z.re = 0 ∧ ¬ PurelyImaginary z) ∧
  -- Statement 2
  (p ∨ q) ∧
  -- Statement 3
  (∀ a b m : ℝ, (a ≥ b → a * m^2 ≥ b * m^2)) ∧
  -- Statement 4
  (∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q))
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_false_l133_13312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_reciprocal_sum_l133_13313

/-- The value of 1/(ρ₁² + ρ₂²) for two points on the ellipse x²/16 + y²/4 = 1 -/
theorem ellipse_points_reciprocal_sum (θ ρ₁ ρ₂ : ℝ) :
  (ρ₁ * Real.cos θ)^2 / 16 + (ρ₁ * Real.sin θ)^2 / 4 = 1 →
  (ρ₂ * Real.cos (θ + π/2))^2 / 16 + (ρ₂ * Real.sin (θ + π/2))^2 / 4 = 1 →
  1 / (ρ₁^2 + ρ₂^2) = 5/16 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_reciprocal_sum_l133_13313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l133_13380

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism, given specific dimensions. -/
theorem cone_prism_volume_ratio (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  (1 / 3) * Real.pi * (3 * r / 2)^2 * h / (24 * r^2 * h) = Real.pi / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l133_13380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_and_coeff_sum_l133_13361

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)]

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the perimeter of the hexagon
noncomputable def hexagon_perimeter (vertices : List (ℝ × ℝ)) : ℝ :=
  let pairs := List.zip vertices (vertices.rotateRight 1)
  List.sum (List.map (fun (p : (ℝ × ℝ) × (ℝ × ℝ)) => distance p.1 p.2) pairs)

-- Theorem stating the perimeter and coefficient sum
theorem hexagon_perimeter_and_coeff_sum :
  hexagon_perimeter hexagon_vertices = 2 + 4 * Real.sqrt 2 ∧
  (2 : ℤ) + (4 : ℤ) + (0 : ℤ) = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_and_coeff_sum_l133_13361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l133_13305

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden --/
structure SquareGarden where
  side : ℝ

/-- Calculates the area of a rectangular garden --/
def area_rectangular (g : RectangularGarden) : ℝ :=
  g.length * g.width

/-- Calculates the perimeter of a rectangular garden --/
def perimeter_rectangular (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a square garden --/
def area_square (g : SquareGarden) : ℝ :=
  g.side ^ 2

/-- Transforms a rectangular garden to a square garden with the same perimeter --/
noncomputable def to_square_garden (g : RectangularGarden) : SquareGarden :=
  SquareGarden.mk ((perimeter_rectangular g) / 4)

/-- The main theorem stating the increase in area --/
theorem garden_area_increase (g : RectangularGarden)
    (h1 : g.length = 60)
    (h2 : g.width = 20) :
    area_square (to_square_garden g) - area_rectangular g = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l133_13305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l133_13397

/-- The probability of getting heads on a single flip -/
noncomputable def p_heads : ℝ := 1/3

/-- The probability of getting tails on a single flip -/
noncomputable def p_tails : ℝ := 2/3

/-- The number of players -/
def num_players : ℕ := 4

/-- The probability that all players flip their coins the same number of times -/
noncomputable def prob_same_flips : ℝ := 81/65

/-- Theorem stating the probability that all players flip their coins the same number of times -/
theorem coin_flip_probability :
  let p := p_heads
  let q := p_tails
  let n := num_players
  (∑' k, (q^n)^k * p^n) = prob_same_flips := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l133_13397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l133_13366

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l133_13366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_problem_l133_13333

theorem investment_rate_problem (total_investment : ℝ) (first_amount : ℝ) (first_rate : ℝ)
  (second_amount : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_amount = 5000 →
  first_rate = 3.5 / 100 →
  second_amount = 4000 →
  second_rate = 4.5 / 100 →
  desired_income = 600 →
  ∃ (remaining_rate : ℝ),
    (remaining_rate ≥ 8.15 / 100 ∧ remaining_rate ≤ 8.25 / 100) ∧
    (first_amount * first_rate + second_amount * second_rate +
     (total_investment - first_amount - second_amount) * remaining_rate) = desired_income :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_problem_l133_13333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_trip_time_l133_13354

/-- 
Given a family's trip with the following conditions:
- Total trip distance: 43.25 km
- Canoeing speed: 12 km/h
- Hiking speed: 5 km/h
- Hiking distance: 27 km

This theorem proves that the total trip time is approximately 6.75 hours.
-/
theorem family_trip_time (total_distance : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (h1 : total_distance = 43.25)
  (h2 : canoe_speed = 12)
  (h3 : hike_speed = 5)
  (h4 : hike_distance = 27) :
  ∃ (trip_time : ℝ), (trip_time ≥ 6.74 ∧ trip_time ≤ 6.76) ∧ 
  trip_time = (total_distance - hike_distance) / canoe_speed + hike_distance / hike_speed :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_trip_time_l133_13354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l133_13359

theorem cube_root_simplification :
  (((8 : ℝ) + 27) ^ (1/3 : ℝ)) * ((27 + 27 ^ (1/3 : ℝ)) ^ (1/3 : ℝ)) = 1050 ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l133_13359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l133_13346

/-- Represents a parabolic arch -/
structure ParabolicArch where
  a : ℝ
  k : ℝ

/-- The height of the arch at a given x coordinate -/
noncomputable def ParabolicArch.height (arch : ParabolicArch) (x : ℝ) : ℝ :=
  arch.a * x^2 + arch.k

/-- Constructs a parabolic arch given its center height and span -/
noncomputable def makeParabolicArch (centerHeight : ℝ) (span : ℝ) : ParabolicArch :=
  { a := -4 * centerHeight / span^2,
    k := centerHeight }

theorem parabolic_arch_height_at_10 :
  let arch := makeParabolicArch 20 50
  arch.height 10 = 16.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l133_13346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_four_ninths_l133_13362

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 3*y^2 = 5

-- Define the line
def line (k x : ℝ) : ℝ := k*(x + 1)

-- Define point M
def M : ℝ × ℝ := (-7/3, 0)

-- Define the dot product of MA and MB
def dot_product (k : ℝ) (A B : ℝ × ℝ) : ℝ :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2)

theorem dot_product_equals_four_ninths (k : ℝ) :
  ∀ A B : ℝ × ℝ,
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  A.2 = line k A.1 →
  B.2 = line k B.1 →
  A ≠ B →
  dot_product k A B = 4/9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_four_ninths_l133_13362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_AlI3_l133_13395

/-- The atomic mass of aluminum in g/mol -/
noncomputable def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of iodine in g/mol -/
noncomputable def atomic_mass_I : ℝ := 126.90

/-- The number of aluminum atoms in AlI3 -/
def num_Al : ℕ := 1

/-- The number of iodine atoms in AlI3 -/
def num_I : ℕ := 3

/-- The molar mass of AlI3 in g/mol -/
noncomputable def molar_mass_AlI3 : ℝ := num_Al * atomic_mass_Al + num_I * atomic_mass_I

/-- The mass percentage of iodine in AlI3 -/
noncomputable def mass_percentage_I : ℝ := (num_I * atomic_mass_I) / molar_mass_AlI3 * 100

theorem iodine_mass_percentage_in_AlI3 :
  abs (mass_percentage_I - 93.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_AlI3_l133_13395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l133_13396

-- Define the space and points
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable (A B C P M : E)
variable (x y z : ℝ)

-- Define the conditions
axiom not_coplanar : P ∉ affineSpan ℝ {A, B, C}
axiom pm_eq_2mc : P - M = (2 : ℝ) • (M - C)
axiom bm_eq_xab_yac_zap : B - M = x • (A - B) + y • (A - C) + z • (A - P)

-- State the theorem
theorem sum_of_coefficients_is_zero : x + y + z = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l133_13396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_theorem_l133_13390

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

-- Define the probability of X ≥ 1 for X ~ B(2, p)
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

-- Define the probability of Y ≥ 1 for Y ~ B(3, p)
def prob_Y_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^3

theorem binomial_probability_theorem :
  ∀ p : ℝ,
  0 ≤ p ∧ p ≤ 1 →
  (prob_X_geq_1 p = 3/4) →
  prob_Y_geq_1 p = 7/8 :=
by
  intro p h1 h2
  sorry

#check binomial_probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_theorem_l133_13390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_l133_13316

-- Define set A (now using ℝ instead of ℤ)
def A : Set ℝ := {-2, -1, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Theorem statement
theorem A_intersect_B_equals : A ∩ B = {-2, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_l133_13316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l133_13330

theorem product_of_numbers_with_ratio (a b : ℝ) : 
  (a - b) / 2 = (a + b) / 8 ∧ (a + b) / 8 = (2 * a * b) / 30 → a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l133_13330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_circumcircle_radius_l133_13379

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1

-- Define the domain of x
def domain (x : ℝ) : Prop := 0 < x ∧ x < Real.pi

-- Theorem for the monotonically decreasing interval
theorem monotonic_decreasing_interval (x : ℝ) (h : domain x) :
  f x ≤ f (Real.pi / 8) ↔ Real.pi / 8 ≤ x ∧ x ≤ 5 * Real.pi / 8 :=
sorry

-- Theorem for the radius of the circumcircle
theorem circumcircle_radius (A B : ℝ) (h1 : domain A) (h2 : domain B) (h3 : A ≠ B) (h4 : f A = f B) :
  let AB : ℝ := Real.sqrt 2
  let R : ℝ := 1
  ∃ (C : ℝ), domain C ∧ R = (AB / 2) / Real.sin ((Real.pi - A - B) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_circumcircle_radius_l133_13379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_curves_range_l133_13337

open Real

-- Define the functions f and g
noncomputable def f (a c x : ℝ) : ℝ := exp x - a * log x + c

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := exp x - a * log x

-- State the theorem
theorem parallel_curves_range (a c : ℝ) (h1 : a > 0) (h2 : c ≠ 0) :
  (∀ x : ℝ, x > 0 → ∃ h, f a c x + h = g a x) →
  g a 1 = exp 1 →
  (∃! x, x ∈ Set.Ioo 2 3 ∧ g a x = 0) →
  a ∈ Set.Ioo (exp 2 / log 2) (exp 3 / log 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_curves_range_l133_13337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_length_range_l133_13306

theorem obtuse_triangle_side_length_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = a + 1 ∧ z = a + 2 ∧
   x + y > z ∧ y + z > x ∧ z + x > y ∧
   z^2 > x^2 + y^2 ∧
   x > 0 ∧ y > 0 ∧ z > 0)
  ↔ 
  (1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_length_range_l133_13306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l133_13335

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- State the theorem
theorem triangle_problem (ABC : Triangle) (h1 : f ABC.C = 1) 
  (h2 : ABC.area = (3 * Real.sqrt 3) / 4) (h3 : ABC.c = Real.sqrt 7) :
  (∀ x, f x ≥ -1/2) ∧ (ABC.a + ABC.b + ABC.c = 4 + Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l133_13335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_given_cone_l133_13398

-- Define the right triangular cone
structure RightTriangularCone where
  sa : ℝ
  ab : ℝ
  bc : ℝ
  sa_perpendicular_to_base : Prop  -- Represents SA ⊥ ABC
  right_angle_abc : Prop           -- Represents ∠ABC = 90°

-- Define the given cone
def given_cone : RightTriangularCone :=
  { sa := 2
  , ab := 2
  , bc := 1
  , sa_perpendicular_to_base := True
  , right_angle_abc := True }

-- Function to calculate the volume of a right triangular cone
noncomputable def cone_volume (c : RightTriangularCone) : ℝ :=
  (1 / 3) * (1 / 2) * c.ab * c.bc * c.sa

-- Function to calculate the volume of the circumscribed sphere
noncomputable def circumscribed_sphere_volume (c : RightTriangularCone) : ℝ :=
  (4 / 3) * Real.pi * ((3 / 2) ^ 3)

-- Theorem statement
theorem volume_ratio_of_given_cone :
  (cone_volume given_cone) / (circumscribed_sphere_volume given_cone) = 4 / (27 * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_given_cone_l133_13398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l133_13387

-- Define the variables
noncomputable def a : ℝ := Real.log 2 / Real.log (1/2)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10)

-- State the theorem
theorem ascending_order : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l133_13387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l133_13350

-- Define the rational function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + x - 4) / (x^2 - 5*x + 6)

-- Define the domain of f(x)
def domain_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ 3}

-- Theorem statement
theorem domain_of_f : 
  domain_f = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l133_13350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_sin_l133_13372

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 3 * Real.pi / 4)

noncomputable def stretch (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f (x / k)

noncomputable def shift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ f (x + c)

noncomputable def g : ℝ → ℝ := shift (stretch f 2) (Real.pi / 4)

theorem g_equals_sin : g = Real.sin := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_sin_l133_13372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l133_13386

-- Define the complex number
noncomputable def z : ℂ := ((-5 : ℝ) + (1 : ℝ) * Complex.I) / ((2 : ℝ) - (3 : ℝ) * Complex.I)

-- State the theorem
theorem modulus_of_complex_fraction : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l133_13386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_range_of_a_l133_13374

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := x + 1 / x

-- Theorem for the minimum and maximum values of f when a = 3
theorem f_min_max_on_interval :
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f 3 x ≤ f 3 y) ∧
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f 3 y ≤ f 3 x) ∧
  (⨅ x ∈ Set.Icc 0 1, f 3 x) = 2/3 ∧
  (⨆ x ∈ Set.Icc 0 1, f 3 x) = 2 :=
by sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧ a ≤ 5/6) ↔
    ∀ x₁ ∈ Set.Ioo 2 3, ∃ x₂ ∈ Set.Icc 1 2, f a x₁ < g x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_range_of_a_l133_13374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ambiguity_l133_13323

noncomputable def has_two_solutions (a b A : ℝ) : Prop :=
  ∃ (B₁ B₂ : ℝ), B₁ ≠ B₂ ∧ 
    Real.sin B₁ = (b * Real.sin A) / a ∧
    Real.sin B₂ = (b * Real.sin A) / a ∧
    0 < B₁ ∧ B₁ < Real.pi ∧ 0 < B₂ ∧ B₂ < Real.pi

theorem triangle_ambiguity (a b : ℝ) (h_a : a = 2 * Real.sqrt 3) (h_b : b = 4) :
  (∃ (A : ℝ), A = Real.pi / 6 ∧ has_two_solutions a b A) ∧
  (∃ (A : ℝ), Real.cos A = 3 / 5 ∧ has_two_solutions a b A) ∧
  (∀ (C : ℝ), C = Real.pi / 6 → ¬ has_two_solutions a b C) ∧
  (∀ (B : ℝ), B = Real.pi / 6 → ¬ has_two_solutions a b B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ambiguity_l133_13323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_and_optimal_batch_size_l133_13382

/-- The total annual cost for shipping and storage fees as a function of batch size -/
noncomputable def total_cost (x : ℝ) : ℝ :=
  (4000 / x) * 360 + 100 * x

/-- The theorem stating the minimum total annual cost and the optimal batch size -/
theorem min_cost_and_optimal_batch_size :
  (∀ x > 0, total_cost x ≥ 24000) ∧
  (total_cost 120 = 24000) := by
  sorry

#check min_cost_and_optimal_batch_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_and_optimal_batch_size_l133_13382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danielle_popsicle_sticks_l133_13339

/-- Calculates the number of popsicle sticks left after making popsicles --/
def popsicle_sticks_left (total_money : ℚ) (mold_cost : ℚ) (stick_pack_cost : ℚ) 
  (stick_pack_size : ℕ) (juice_cost : ℚ) (popsicles_per_juice : ℕ) : ℕ :=
  let remaining_money := total_money - mold_cost - stick_pack_cost
  let juice_bottles := (remaining_money / juice_cost).floor
  let popsicles_made := juice_bottles * popsicles_per_juice
  stick_pack_size - popsicles_made.toNat

/-- Theorem stating that Danielle will be left with 40 popsicle sticks --/
theorem danielle_popsicle_sticks : 
  popsicle_sticks_left 10 3 1 100 2 20 = 40 := by
  -- Unfold the definition of popsicle_sticks_left
  unfold popsicle_sticks_left
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_danielle_popsicle_sticks_l133_13339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_savings_paper_towels_l133_13314

/-- The percent savings per roll when buying a package of 12 rolls for $9 compared to buying 12 rolls individually at $1 each. -/
theorem percent_savings_paper_towels 
  (package_price : ℝ)
  (individual_price : ℝ)
  (num_rolls : ℝ)
  (h1 : package_price = 9)
  (h2 : individual_price = 1)
  (h3 : num_rolls = 12) :
  (1 - package_price / (num_rolls * individual_price)) * 100 = 25 := by
  sorry

#check percent_savings_paper_towels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_savings_paper_towels_l133_13314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_nine_solution_l133_13367

/-- Represents a digit in base h --/
def Digit (h : ℕ) := {d : ℕ // d < h}

/-- Converts a list of digits in base h to a natural number --/
def toNatBase (h : ℕ) (digits : List (Digit h)) : ℕ :=
  digits.foldr (fun d acc => acc * h + d.val) 0

/-- The equation in base h --/
def baseHEquation (h : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 : Digit h),
    toNatBase h [d1, d2, d3, d4] + toNatBase h [d5, d6, d7, d8] = toNatBase h [d9, d10, d11, d12] ∧
    d1.val = 3 ∧ d2.val = 6 ∧ d3.val = 8 ∧ d4.val = 4 ∧
    d5.val = 4 ∧ d6.val = 1 ∧ d7.val = 7 ∧ d8.val = 5 ∧
    d9.val = 1 ∧ d10.val = 0 ∧ d11.val = 2 ∧ d12.val = 9

/-- Theorem stating that 9 is the only solution --/
theorem base_nine_solution :
  ∃! h : ℕ, h > 1 ∧ baseHEquation h :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_nine_solution_l133_13367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l133_13332

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (0, -3)
def D : ℝ × ℝ := (8, -3)
def E : ℝ × ℝ := (8, 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem laser_beam_distance :
  distance A B + distance B C + distance C D + distance D E = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l133_13332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l133_13371

theorem b_range (a b : ℝ) : 
  (∀ x, (1 - a) * x^2 - 4 * x + 6 > 0 ↔ -3 < x ∧ x < 1) →
  (∀ x, a * x^2 + b * x + 3 ≥ 0) →
  b ∈ Set.Icc (-6) 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l133_13371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l133_13353

theorem line_equation (A B C : ℝ × ℝ) : 
  A.1 = 0 →  -- A is on y-axis
  B.2 = 2 * B.1 + 2 →  -- B is on y = 2x + 2
  C.2 = 0 →  -- C is on x-axis
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 32 →  -- AC length is 4√2
  B.1 = (A.1 + C.1) / 2 →  -- B is midpoint of AC
  B.2 = (A.2 + C.2) / 2 →
  0 < B.1 ∧ 0 < B.2 →  -- B is in first quadrant
  ∃ m b : ℝ, m = -7 ∧ b = 28/5 ∧ 
    (∀ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2) → y = m * x + b) :=
by
  sorry

#check line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l133_13353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l133_13317

-- Define the binomial coefficient sum function
def binomialCoefficientSum (n : ℕ) : ℕ := 2^n

-- Define the constant term function
def constantTerm (n : ℕ) (m : ℚ) : ℚ := (Nat.choose n (n/2)) * m^(n/2)

-- Define a function to check if 6th and 7th terms have maximum coefficients
def sixthAndSeventhMaxCoeff (n : ℕ) (m : ℚ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → (Nat.choose n k) * m^k ≤ (Nat.choose n 5) * m^5 ∧
                        (Nat.choose n k) * m^k ≤ (Nat.choose n 6) * m^6

theorem binomial_expansion_properties :
  ∀ n : ℕ, ∀ m : ℚ,
    (binomialCoefficientSum n = 256 → n = 8) ∧
    (n = 8 ∧ constantTerm n m = 35/8 → m = 1/2 ∨ m = -1/2) ∧
    (n = 8 ∧ sixthAndSeventhMaxCoeff n m → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l133_13317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l133_13315

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-15 * x^2 - 14 * x + 21)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≤ -7/5 ∨ x ≥ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l133_13315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_three_l133_13394

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
  Real.arccos ((distance p q)^2 + (distance q r)^2 - (distance p r)^2) / (2 * distance p q * distance q r)

/-- Represents a broken line with n+1 points -/
structure BrokenLine (n : ℕ) where
  points : Fin (n+1) → Point
  segment_length_eq_one : ∀ i : Fin n, distance (points i) (points (i+1)) = 1
  angles_increasing : ∀ i : Fin (n-2), 
    angle (points (i+3)) (points (i+1)) (points (i+2)) ≤ 
    angle (points (i+1)) (points (i+2)) (points (i+3))
  min_angle : ∀ i : Fin (n-2), π/3 ≤ angle (points (i+3)) (points (i+1)) (points (i+2))
  max_angle : ∀ i : Fin (n-2), angle (points (i+1)) (points (i+2)) (points (i+3)) ≤ 2*π/3

/-- The main theorem -/
theorem distance_less_than_three {n : ℕ} (bl : BrokenLine n) :
  ∀ i : Fin (n+1), distance (bl.points 0) (bl.points i) < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_three_l133_13394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_lines_sixteen_regions_l133_13307

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a straight line in the grid -/
inductive Line where
  | Vertical : Nat → Line
  | Horizontal : Nat → Line
  | Diagonal : GridPoint → GridPoint → Line

/-- A configuration of lines on the grid -/
structure GridConfiguration where
  lines : List Line

/-- Checks if two points are in the same region -/
def sameRegion (config : GridConfiguration) (p1 p2 : GridPoint) : Prop :=
  sorry

/-- Counts the number of distinct regions in the grid -/
def countRegions (config : GridConfiguration) : Nat :=
  sorry

/-- The main theorem: there exists a configuration of 5 lines that creates 16 regions -/
theorem five_lines_sixteen_regions :
  ∃ (config : GridConfiguration),
    config.lines.length = 5 ∧
    countRegions config = 16 := by
  sorry

#check five_lines_sixteen_regions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_lines_sixteen_regions_l133_13307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l133_13378

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => sequence_a n + 1 / (2^(n+1))

theorem sequence_a_2019 : sequence_a 2019 = 3/2 - 1/(2^2018) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l133_13378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l133_13389

theorem binomial_expansion_coefficient :
  let expansion := (λ x : ℝ => (x^2 + 1/(2*x))^5)
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → expansion x = c*x + (λ y => y^5 + y^4 + y^3 + y^2 + 1) (x^2)) ∧
  (∀ c : ℝ, (∀ x : ℝ, x ≠ 0 → expansion x = c*x + (λ y => y^5 + y^4 + y^3 + y^2 + 1) (x^2)) → c = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l133_13389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l133_13384

noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 1) / (x^2 - 5*x + 6)

theorem vertical_asymptotes_of_f :
  ∀ x : ℝ, (x = 2 ∨ x = 3) ↔ 
    (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l133_13384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l133_13322

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 170 meters traveling at 45 km/hr 
    takes 30 seconds to cross a bridge of length 205 meters -/
theorem train_bridge_crossing_time :
  time_to_cross_bridge 170 205 45 = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cross_bridge 170 205 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l133_13322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l133_13375

/-- A circle passes through (0,2) and is tangent to y = x^2 at (1,1). Its center is (1/2, 3/2). -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - 1)^2 + (center.2 - 1)^2) →
  (0, 2) ∈ C →
  (1, 1) ∈ C →
  (∀ (p : ℝ × ℝ), p ∈ C → p.2 ≥ p.1^2) →
  (∀ (p : ℝ × ℝ), p.1 ≠ 1 → p ∈ C → p.2 > p.1^2) →
  center = (1/2, 3/2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l133_13375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoveling_time_theorem_l133_13344

/-- Represents Julia's snow shoveling rate at a given hour -/
def shoveling_rate (hour : ℕ) : ℕ :=
  max (30 - 2 * (hour - 1)) 0

/-- Calculates the total volume of snow removed up to a given hour -/
def total_removed (hours : ℕ) : ℕ :=
  Finset.sum (Finset.range hours) (λ h => shoveling_rate (h + 1))

/-- Represents the dimensions of Julia's driveway -/
def driveway_volume : ℕ := 5 * 12 * 4

/-- Theorem stating that it takes 15 hours to shovel the driveway clean -/
theorem shoveling_time_theorem :
  (∀ h : ℕ, h < 15 → total_removed h < driveway_volume) ∧
  total_removed 15 ≥ driveway_volume := by
  sorry

#eval driveway_volume
#eval total_removed 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoveling_time_theorem_l133_13344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_fixed_point_l133_13383

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 4

-- State the theorem
theorem inverse_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv (f a) ∧ f_inv 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_fixed_point_l133_13383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_possible_l133_13348

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length 1 -/
def UnitSquare : Type := Unit

/-- The area covered by the "protection zone" of a unit square for a unit diameter circle -/
noncomputable def protectionZoneArea : ℝ := 3 + Real.pi / 4

/-- The number of unit squares in the rectangle -/
def numSquares : ℕ := 120

/-- The main rectangle in the problem -/
def mainRectangle : Rectangle := { width := 20, height := 25 }

/-- The interior area of the rectangle available for placing the circle's center -/
def interiorArea (r : Rectangle) : ℝ := (r.width - 1) * (r.height - 1)

/-- Statement of the theorem -/
theorem circle_placement_possible (squares : Fin numSquares → UnitSquare) :
  (numSquares : ℝ) * protectionZoneArea < interiorArea mainRectangle := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_possible_l133_13348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_sum_l133_13355

noncomputable def F₁ : ℝ × ℝ := (-4, 2 - Real.sqrt 2)
noncomputable def F₂ : ℝ × ℝ := (-4, 2 + Real.sqrt 2)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  |dist P F₁ - dist P F₂| = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

noncomputable def center : ℝ × ℝ := ((F₁.1 + F₂.1) / 2, (F₁.2 + F₂.2) / 2)

noncomputable def h : ℝ := center.1
noncomputable def k : ℝ := center.2

noncomputable def c : ℝ := dist F₁ F₂ / 2

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_parameters_sum :
  h + k + a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_sum_l133_13355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sand_mass_112_tons_l133_13338

/-- Represents the properties of a sand-loaded platform -/
structure SandPlatform where
  length : ℝ
  width : ℝ
  max_angle : ℝ
  sand_density : ℝ

/-- Calculates the maximum mass of sand that can be loaded onto the platform -/
noncomputable def max_sand_mass (platform : SandPlatform) : ℝ :=
  let height := platform.width / 2
  let prism_volume := platform.length * platform.width * height
  let pyramid_volume := 2 * (1/3 * platform.width * height * height)
  let total_volume := prism_volume + pyramid_volume
  total_volume * platform.sand_density

/-- Theorem stating the maximum mass of sand on the given platform -/
theorem max_sand_mass_112_tons (platform : SandPlatform) 
  (h_length : platform.length = 8)
  (h_width : platform.width = 4)
  (h_angle : platform.max_angle = 45)
  (h_density : platform.sand_density = 1500) :
  max_sand_mass platform = 112000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sand_mass_112_tons_l133_13338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_24_l133_13303

theorem multiple_of_24 (n : ℕ) : 
  ∃ k₁ k₂ : ℤ, (6*n - 1)^2 - 1 = 24 * k₁ ∧ (6*n + 1)^2 - 1 = 24 * k₂ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_24_l133_13303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_81_l133_13321

/-- The sum of the proper divisors of 81 is 40. -/
theorem sum_proper_divisors_81 : (Finset.filter (λ x ↦ x ∣ 81 ∧ x ≠ 81) (Finset.range 82)).sum id = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_81_l133_13321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_conversion_l133_13364

noncomputable def spherical_to_rectangular (r θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.sin θ * Real.cos φ, r * Real.sin θ * Real.sin φ, r * Real.cos θ)

noncomputable def spherical_to_cylindrical (r θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.sin θ, φ, r * Real.cos θ)

theorem spherical_coordinate_conversion :
  let r : ℝ := 2
  let θ : ℝ := Real.pi / 6
  let φ : ℝ := Real.pi / 3
  
  let (x, y, z) := spherical_to_rectangular r θ φ
  let (ρ, t, z') := spherical_to_cylindrical r θ φ
  
  (x = 1/2 ∧ y = Real.sqrt 3 / 2 ∧ z = Real.sqrt 3) ∧
  (ρ = 1 ∧ t = Real.pi / 3 ∧ z' = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_conversion_l133_13364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_10_l133_13370

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetric point with respect to xoy plane -/
def symmetricPointXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: The length of line segment AB is 10 -/
theorem length_AB_is_10 : 
  let A : Point3D := { x := 2, y := -3, z := 5 }
  let B : Point3D := symmetricPointXOY A
  distance A B = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_10_l133_13370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l133_13300

-- Define the ∇ operation for positive real numbers
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  nabla (nabla (nabla 2 3) 4) 5 = 7/8 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l133_13300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_trapezoid_l133_13358

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 6)
def C : ℝ × ℝ := (3, 6)
def D : ℝ × ℝ := (3, 0)

-- Define the angles
noncomputable def angle1 : ℝ := 45 * Real.pi / 180
noncomputable def angle2 : ℝ := 75 * Real.pi / 180

-- Define the lines from A
noncomputable def lineA1 (x : ℝ) : ℝ := x * Real.tan angle1
noncomputable def lineA2 (x : ℝ) : ℝ := x * Real.tan angle2

-- Define the lines from B
noncomputable def lineB1 (x : ℝ) : ℝ := 6 - x * Real.tan angle1
noncomputable def lineB2 (x : ℝ) : ℝ := 6 - x * Real.tan angle2

-- Define the intersection points
def P : ℝ × ℝ := (3, 3)
noncomputable def Q : ℝ × ℝ := (6 / (2 * Real.tan angle2), 3)

-- Define a Trapezoid type
def Trapezoid (points : List (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem intersection_points_form_trapezoid :
  let points := [A, B, P, Q]
  Trapezoid points :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_trapezoid_l133_13358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l133_13356

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-m^2 + 2*m + 3)

theorem power_function_value (m : ℤ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x < f m y) →  -- monotonically increasing in (0, +∞)
  (∀ x : ℝ, f m x = f m (-x)) →                -- symmetric about y-axis
  f m (-2) = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l133_13356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_four_l133_13351

theorem non_negative_integers_less_than_four :
  {n : ℕ | n < 4} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_four_l133_13351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l133_13373

theorem acute_triangle_side_range :
  ∀ a : ℝ,
  (∃ (A B C : ℝ × ℝ),
    let d := λ p q : ℝ × ℝ ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    d A B = 1 ∧
    d B C = 3 ∧
    d C A = a ∧
    (d A B)^2 + (d B C)^2 > (d C A)^2 ∧
    (d B C)^2 + (d C A)^2 > (d A B)^2 ∧
    (d C A)^2 + (d A B)^2 > (d B C)^2)
  ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l133_13373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_responding_friends_l133_13360

theorem percentage_of_responding_friends
  (initial_friends : ℕ)
  (kept_percentage : ℚ)
  (final_friends : ℕ)
  (h1 : initial_friends = 100)
  (h2 : kept_percentage = 40 / 100)
  (h3 : final_friends = 70) :
  (final_friends - (kept_percentage * initial_friends).floor) /
  (initial_friends - (kept_percentage * initial_friends).floor) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_responding_friends_l133_13360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_A_correct_l133_13376

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (a_initial : ℕ) (b_initial : ℕ) (a_change : ℤ) (b_change : ℕ) 
  (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let a_investment_months : ℤ := a_initial * change_month + (a_initial + a_change) * (total_months - change_month)
  let b_investment_months : ℕ := b_initial * change_month + (b_initial + b_change) * (total_months - change_month)
  let total_investment_months : ℤ := a_investment_months + b_investment_months
  ((a_investment_months * total_profit) / total_investment_months).toNat

theorem share_A_correct : 
  calculate_share_A 3000 4000 (-1000) 1000 12 8 630 = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_A_correct_l133_13376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_sequence_593_to_598_l133_13343

def arrow_sequence (n : ℕ) : Fin 5 :=
  match n % 5 with
  | 0 => 0  -- A
  | 1 => 1  -- B
  | 2 => 2  -- C
  | 3 => 3  -- D
  | _ => 4  -- E

theorem arrow_sequence_593_to_598 :
  (List.range 6).map (λ i => arrow_sequence (593 + i)) = [2, 3, 4, 0, 1, 2] :=
by
  simp [arrow_sequence]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_sequence_593_to_598_l133_13343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l133_13326

noncomputable section

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

def RightFocus : ℝ × ℝ := (2, 0)

noncomputable def PointP : ℝ × ℝ := (2, Real.sqrt 6 / 3)

theorem ellipse_equation (e : Ellipse) 
  (h1 : Ellipse.equation e RightFocus.1 RightFocus.2)
  (h2 : Ellipse.equation e PointP.1 PointP.2) :
  e.a^2 = 6 ∧ e.b^2 = 2 := by sorry

theorem line_equation (e : Ellipse)
  (h1 : e.a^2 = 6 ∧ e.b^2 = 2)
  (A B M : ℝ × ℝ)
  (hA : Ellipse.equation e A.1 A.2)
  (hB : Ellipse.equation e B.1 B.2)
  (hM : Ellipse.equation e M.1 M.2)
  (hABF : ∃ k : ℝ, A.2 - RightFocus.2 = k * (A.1 - RightFocus.1) ∧
                   B.2 - RightFocus.2 = k * (B.1 - RightFocus.1))
  (hCentroid : M.1 + A.1 + B.1 = 0 ∧ M.2 + A.2 + B.2 = 0) :
  ∃ k : ℝ, k^2 = 1/5 ∧ 
    (∀ x y : ℝ, y = k * (x - 2) → Ellipse.equation e x y) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l133_13326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_y_axis_l133_13324

def M (x : ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n.val).prod (λ i => x + i)

def f (x : ℝ) : ℝ :=
  x * M (x - 9) 19

theorem f_symmetric_y_axis : ∀ x : ℝ, f x = f (-x) := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_y_axis_l133_13324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_quadrant_theorem_l133_13377

/-- Represents a circle quadrant with semi-circles on two sides -/
structure CircleQuadrant where
  radius : ℝ
  p : ℝ
  q : ℝ

/-- The area of the quadrant -/
noncomputable def quadrant_area (cq : CircleQuadrant) : ℝ := Real.pi * cq.radius^2

/-- The area of one semi-circle -/
noncomputable def semicircle_area (cq : CircleQuadrant) : ℝ := (Real.pi * cq.radius^2) / 2

/-- The theorem stating that if p = 1 and q = c in a circle quadrant with semi-circles, then c = 1 -/
theorem circle_quadrant_theorem (cq : CircleQuadrant) 
  (h1 : cq.p = 1)
  (h2 : ∃ c, cq.q = c) :
  cq.q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_quadrant_theorem_l133_13377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_zero_l133_13301

noncomputable section

open Real Matrix

-- Define the matrix entries
def matrixEntry (i j : Nat) : ℝ := sin (π / 4 + (i - 1) * 3 + j)

-- Define the 3x3 matrix
def matrix : Matrix (Fin 3) (Fin 3) ℝ := fun i j => matrixEntry i.val j.val

-- Theorem statement
theorem matrix_determinant_zero : det matrix = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_zero_l133_13301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_sided_polygon_with_108_degree_angles_l133_13319

/-- Definition: The measure of an interior angle of a regular n-gon. -/
noncomputable def interior_angle (n : ℕ) (i : ℕ) : ℝ :=
  (n - 2) * 180 / n

/-- Theorem: For an n-sided polygon where each interior angle is 108°, n = 5. -/
theorem n_sided_polygon_with_108_degree_angles (n : ℕ) : 
  (n ≥ 3) → (∀ i : ℕ, i < n → interior_angle n i = 108) → n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_sided_polygon_with_108_degree_angles_l133_13319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_l133_13341

noncomputable def powerSum (n : ℕ) : ℝ := (37.5 : ℝ)^n + (26.5 : ℝ)^n

theorem integer_power_sum (n : ℕ) : 
  ∃ (m : ℤ), powerSum n = m ↔ n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_l133_13341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l133_13311

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8*y + 13 = 0

/-- The line equation -/
def line_eq (a x y : ℝ) : Prop := a*x + y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 4)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (a : ℝ) (p : ℝ × ℝ) : ℝ :=
  |a * p.1 + p.2 - 1| / Real.sqrt (a^2 + 1)

/-- Main theorem -/
theorem circle_line_distance (a : ℝ) : 
  distance_point_to_line a circle_center = 1 → a = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l133_13311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_power_tower_exceeds_iterated_factorial_l133_13363

-- Define power tower function
def powerTower : ℕ → ℕ
| 0 => 1
| n + 1 => 2^(powerTower n)

-- Define iterated factorial function
def iteratedFactorial : ℕ → ℕ → ℕ
| n, 0 => n
| n, k + 1 => Nat.factorial (iteratedFactorial n k)

theorem smallest_n_power_tower_exceeds_iterated_factorial :
  ∃ n : ℕ, (n = 104) ∧ 
    (powerTower n > iteratedFactorial 100 100) ∧
    (∀ m : ℕ, m < n → powerTower m ≤ iteratedFactorial 100 100) := by
  sorry

#eval powerTower 3  -- To check if the function works
#eval iteratedFactorial 5 2  -- To check if the function works

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_power_tower_exceeds_iterated_factorial_l133_13363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_elements_not_equal_l133_13349

def arithmetic_seq (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ + d * i)

def allowed_operation (l₁ l₂ : List ℤ) : List (List ℤ × List ℤ) :=
  l₁.bind (fun x => l₁.bind (fun y =>
    if x ≠ y then
      [(l₁.filter (fun z => z ≠ x ∧ z ≠ y) ++ [((x + y) / 3)], l₂)]
    else
      []))

def sum_mod_4 (l : List ℤ) : ℤ :=
  (l.foldl (· + ·) 0) % 4

theorem final_elements_not_equal :
  let l₁ := arithmetic_seq 1 5 10
  let l₂ := arithmetic_seq 4 5 10
  ∀ (l₁' l₂' : List ℤ),
    (∃ (ops : List (List ℤ × List ℤ)),
      (ops.foldl (fun acc op => allowed_operation op.1 op.2 ++ allowed_operation op.2 op.1) [(l₁, l₂)]).any
        (fun p => p.1 = l₁' ∧ p.2 = l₂')) →
    l₁'.length = 1 →
    l₂'.length = 1 →
    sum_mod_4 l₁' = sum_mod_4 l₁ →
    sum_mod_4 l₂' = sum_mod_4 l₂ →
    l₁'.head? ≠ l₂'.head? :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_elements_not_equal_l133_13349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l133_13385

theorem rhombus_longest_diagonal 
  (area : ℝ) 
  (ratio_long : ℝ) 
  (ratio_short : ℝ) 
  (h_area : area = 150) 
  (h_ratio : ratio_long / ratio_short = 4 / 3) : 
  ratio_long * (2 * area / (ratio_long * ratio_short))^(1/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l133_13385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_value_of_sum_l133_13302

/-- The function f(t) = 1 / (t² - 4t + 9) -/
noncomputable def f (t : ℝ) : ℝ := 1 / (t^2 - 4*t + 9)

theorem greatest_value_of_sum (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  f x + f y + f z ≤ 7/18 ∧ ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1 ∧ f x + f y + f z = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_value_of_sum_l133_13302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_range_of_g_alt_l133_13334

/-- The function g(x) = x / (x^2 + x + 1) -/
noncomputable def g (x : ℝ) : ℝ := x / (x^2 + x + 1)

/-- The range of g is [-1, 1/3] -/
theorem range_of_g : Set.range g = { z : ℝ | -1 ≤ z ∧ z ≤ 1/3 } := by
  sorry

/-- An alternative formulation of the range of g -/
theorem range_of_g_alt : ∀ z : ℝ, (∃ x : ℝ, g x = z) ↔ -1 ≤ z ∧ z ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_range_of_g_alt_l133_13334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l133_13393

/-- The total number of hours charged to the project -/
def total_hours : ℝ := 500

/-- Kate's hours as a variable -/
def k : ℝ := sorry

/-- Pat's hours in terms of Kate's -/
noncomputable def pat_hours : ℝ := 2 * k

/-- Mark's hours in terms of Kate's -/
noncomputable def mark_hours : ℝ := (2/3) * k

/-- Alex's hours in terms of Kate's -/
noncomputable def alex_hours : ℝ := 1.5 * k

/-- Jen's hours in terms of Kate's -/
noncomputable def jen_hours : ℝ := (1/2) * k

/-- The sum of all hours should equal the total_hours -/
axiom hours_sum : 
  k + pat_hours + mark_hours + alex_hours + jen_hours = total_hours

/-- The theorem to be proved -/
theorem project_hours_difference :
  ∃ (diff : ℝ), abs (diff + 322.57) < 0.01 ∧ 
  diff = (mark_hours + jen_hours) - (pat_hours + k + alex_hours) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l133_13393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_112_point_5_l133_13336

/-- A square and a right triangle positioned side-by-side on the x-axis -/
structure GeometricSetup where
  /-- The side length of the square -/
  square_side : ℝ
  /-- The base of the right triangle -/
  triangle_base : ℝ
  /-- The height of the right triangle -/
  triangle_height : ℝ
  /-- The x-coordinate where the square and triangle meet -/
  intersection_x : ℝ
  /-- Constraint: The square side length is 15 -/
  h_square_side : square_side = 15
  /-- Constraint: The triangle base is 15 -/
  h_triangle_base : triangle_base = 15
  /-- Constraint: The triangle height is 15 -/
  h_triangle_height : triangle_height = 15
  /-- Constraint: The intersection point is at x = 15 -/
  h_intersection_x : intersection_x = 15

/-- The area of the shaded region in the geometric setup -/
noncomputable def shadedArea (setup : GeometricSetup) : ℝ :=
  (setup.triangle_base * setup.triangle_height) / 2

/-- Theorem: The shaded area is 112.5 square units -/
theorem shaded_area_is_112_point_5 (setup : GeometricSetup) :
  shadedArea setup = 112.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_112_point_5_l133_13336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_congruence_count_l133_13318

theorem four_digit_congruence_count : 
  let count := Finset.filter (fun x => 1000 ≤ x ∧ x ≤ 9999 ∧ (3874 * x + 481) % 31 = 1205 % 31) (Finset.range 10000)
  Finset.card count = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_congruence_count_l133_13318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circles_angle_sine_l133_13345

noncomputable def angle_between_external_common_tangents (R r : ℝ) : ℝ :=
  2 * Real.arcsin ((R - r) / (R + r))

theorem external_tangent_circles_angle_sine 
  (R r θ : ℝ) 
  (h1 : R > r) 
  (h2 : R > 0) 
  (h3 : r > 0) 
  (h4 : θ = angle_between_external_common_tangents R r) : 
  Real.sin θ = (4 * (R - r) * Real.sqrt (R * r)) / ((R + r) ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circles_angle_sine_l133_13345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_expression_two_l133_13388

-- Define x and y as noncomputable
noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

-- Theorem for the first expression
theorem expression_one : x^2 * y - x * y^2 = 4 := by sorry

-- Theorem for the second expression
theorem expression_two : x^2 - y^2 = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_expression_two_l133_13388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_division_theorem_l133_13392

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Whether it's possible to divide weights 1 to n into two equal piles -/
def can_divide_equally (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s ⊆ Finset.range n ∧ 
    2 * (s.sum id) = sum_first_n n

theorem weight_division_theorem :
  (can_divide_equally 99 ∧ ¬can_divide_equally 98) := by
  sorry

#check weight_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_division_theorem_l133_13392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_division_theorem_l133_13325

/-- The number of ways to divide 5 boys and 3 girls into three groups,
    with each group having at least 1 boy and 1 girl -/
def group_division_ways : ℕ := 150

/-- The number of boys -/
def num_boys : ℕ := 5

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of groups -/
def num_groups : ℕ := 3

theorem group_division_theorem :
  (∀ (division : List (List ℕ)),
    division.length = num_groups ∧
    (∀ group ∈ division, group.length ≥ 2) ∧
    (division.map (λ group => (group.filter (λ x => x ≤ num_boys)).length)).sum = num_boys ∧
    (division.map (λ group => (group.filter (λ x => x > num_boys)).length)).sum = num_girls ∧
    (∀ group ∈ division,
      (group.filter (λ x => x ≤ num_boys)).length > 0 ∧
      (group.filter (λ x => x > num_boys)).length > 0)
  ) →
  (division_ways : ℕ) →
  division_ways = group_division_ways :=
by
  sorry

#eval group_division_ways

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_division_theorem_l133_13325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l133_13357

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (3 * x)

theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l133_13357
