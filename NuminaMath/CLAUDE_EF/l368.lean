import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l368_36846

/-- The surface area of a cone given its slant height and base circumference -/
theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 8 →
  base_circumference = 6 * Real.pi →
  Real.pi * (base_circumference / (2 * Real.pi))^2 + Real.pi * (base_circumference / (2 * Real.pi)) * slant_height = 33 * Real.pi :=
by
  intro h_slant h_circ
  have base_radius : ℝ := base_circumference / (2 * Real.pi)
  have surface_area : ℝ := Real.pi * base_radius^2 + Real.pi * base_radius * slant_height
  
  -- Substitute the given values
  rw [h_slant, h_circ]
  
  -- Simplify the expression
  simp [Real.pi]
  
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l368_36846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_interval_l368_36802

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - 1

-- State the theorem
theorem even_function_interval (a : ℝ) :
  (∀ x ∈ Set.Icc (1 - a) 3, f a x = f a (-x)) →
  (Set.Icc (1 - a) 3 = Set.Icc (-3) 3) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_interval_l368_36802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l368_36875

theorem vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  ‖a‖ = 1 → ‖b‖ = 1 → a • b = -1/2 → ‖2 • a - b‖ = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l368_36875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_property_l368_36899

theorem divisor_sum_property (n : ℕ) : 
  (∀ (a b : ℕ), a ∣ n → b ∣ n → Nat.Coprime a b → (a + b - 1) ∣ n) ↔ 
  (∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p^k) ∨ n = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_property_l368_36899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_surface_area_is_6pi_l368_36894

/-- The surface area of a solid with a cylindrical part (radius 1, height 3) and 
    an isosceles triangular part (legs of length 3) that doesn't contribute to the surface area -/
noncomputable def solidSurfaceArea : ℝ := 6 * Real.pi

/-- Theorem stating that the surface area of the described solid is 6π -/
theorem solid_surface_area_is_6pi : solidSurfaceArea = 6 * Real.pi := by
  -- Unfold the definition of solidSurfaceArea
  unfold solidSurfaceArea
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_surface_area_is_6pi_l368_36894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_row_l368_36819

theorem tiles_in_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 324 → tile_size = 9 → (Real.sqrt room_area * 12) / tile_size = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_row_l368_36819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_awesome_points_l368_36860

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Definition of a right triangle with sides 3, 4, and 5 -/
def rightTriangle : Triangle :=
  { a := { x := 0, y := 0 }
    b := { x := 3, y := 0 }
    c := { x := 0, y := 4 } }

/-- The boundary of a triangle -/
def boundary (t : Triangle) : Set Point := sorry

/-- Predicate to check if a point is the center of a parallelogram -/
def isParallelogramCenter (p v1 v2 v3 v4 : Point) : Prop := sorry

/-- A point is awesome if it's the center of a parallelogram with vertices on the triangle's boundary -/
def isAwesome (p : Point) (t : Triangle) : Prop :=
  ∃ (v1 v2 v3 v4 : Point),
    (v1 ∈ boundary t) ∧ (v2 ∈ boundary t) ∧ (v3 ∈ boundary t) ∧ (v4 ∈ boundary t) ∧
    isParallelogramCenter p v1 v2 v3 v4

/-- The set of awesome points -/
def awesomeSet (t : Triangle) : Set Point :=
  { p | isAwesome p t }

/-- The area of a set of points -/
noncomputable def area (s : Set Point) : ℝ := sorry

theorem area_of_awesome_points :
  area (awesomeSet rightTriangle) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_awesome_points_l368_36860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l368_36896

def A : Finset ℕ := {0, 1, 2, 3}
def B : Finset ℕ := {2, 3, 4, 5}

theorem union_cardinality : Finset.card (A ∪ B) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l368_36896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_12_sample_size_l368_36831

/-- Represents the ratio of students in grades 10, 11, and 12 respectively -/
def student_ratio : Fin 3 → ℕ := ![4, 3, 3]

/-- The total sample size -/
def sample_size : ℕ := 200

/-- The index of grade 12 in our vector (0-based) -/
def grade_12_index : Fin 3 := 2

theorem grade_12_sample_size :
  (student_ratio grade_12_index * sample_size) / (student_ratio 0 + student_ratio 1 + student_ratio 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_12_sample_size_l368_36831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l368_36807

-- Define the original function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) + 1

-- Define the inverse function f
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt (x^2 - 2*x)

-- State the theorem
theorem inverse_functions (x : ℝ) (hx : x > 2) :
  f (g (-Real.sqrt (x - 2))) = x ∧ g (f x) = x := by
  sorry

#check inverse_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l368_36807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_surface_area_ratio_equality_l368_36810

variable (R : ℝ) -- Radius of the sphere

-- Define the sphere
noncomputable def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3
noncomputable def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- Define the truncated cone (inscribed in the sphere)
noncomputable def truncated_cone_volume (R : ℝ) : ℝ := sorry
noncomputable def truncated_cone_surface_area (R : ℝ) : ℝ := sorry

-- The theorem to prove
theorem volume_surface_area_ratio_equality (R : ℝ) (h : R > 0) :
  sphere_volume R / truncated_cone_volume R = 
  sphere_surface_area R / truncated_cone_surface_area R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_surface_area_ratio_equality_l368_36810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_plus_50_l368_36898

theorem abs_diff_squares_plus_50 : |(105 : ℤ)^2 - (95 : ℤ)^2 + 50| = 2050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_plus_50_l368_36898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l368_36886

/-- Definition of a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Two spheres are tangent to each other -/
def spheres_tangent (s₁ s₂ : Sphere) : Prop :=
  let (x₁, y₁, z₁) := s₁.center
  let (x₂, y₂, z₂) := s₂.center
  (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 = (s₁.radius + s₂.radius)^2

/-- A sphere is tangent to the base of the cone -/
def spheres_tangent_to_cone_base (s : Sphere) : Prop :=
  let (_, _, z) := s.center
  z = s.radius

/-- A sphere is tangent to the side of the cone -/
def spheres_tangent_to_cone_side (base_radius height : ℝ) (s : Sphere) : Prop :=
  let (x, y, z) := s.center
  (x^2 + y^2) / base_radius^2 + z^2 / height^2 = ((base_radius - s.radius)^2 + (height - s.radius)^2) / (base_radius^2 + height^2)

/-- The radius of each sphere in a right circular cone with specific conditions -/
theorem sphere_radius_in_cone (base_radius : ℝ) (height : ℝ) (r : ℝ) : 
  base_radius = 8 →
  height = 15 →
  (∃ (s₁ s₂ s₃ : Sphere), 
    (s₁.center ≠ s₂.center ∧ s₁.center ≠ s₃.center ∧ s₂.center ≠ s₃.center) ∧
    (s₁.radius = r ∧ s₂.radius = r ∧ s₃.radius = r) ∧
    (spheres_tangent s₁ s₂ ∧ spheres_tangent s₁ s₃ ∧ spheres_tangent s₂ s₃) ∧
    (spheres_tangent_to_cone_base s₁ ∧ spheres_tangent_to_cone_base s₂ ∧ spheres_tangent_to_cone_base s₃) ∧
    (spheres_tangent_to_cone_side base_radius height s₁ ∧ 
     spheres_tangent_to_cone_side base_radius height s₂ ∧ 
     spheres_tangent_to_cone_side base_radius height s₃)) →
  r = (280 - 100 * Real.sqrt 3) / 121 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l368_36886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_two_to_negative_three_l368_36883

theorem sixteen_to_negative_two_to_negative_three (x : ℝ) :
  x = 16 → x^(-(2:ℝ)^(-(3:ℝ))) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_two_to_negative_three_l368_36883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l368_36845

/-- The polar equation of the curve -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 3 * Real.tan θ * (1 / Real.cos θ)

/-- The Cartesian equation of the curve -/
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 = 3 * y

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_equation_is_parabola :
  ∀ (r θ x y : ℝ), polar_equation r θ → 
    (x = r * Real.cos θ ∧ y = r * Real.sin θ) → 
    cartesian_equation x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l368_36845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_arrangement_theorem_l368_36869

/-- Represents a coin arrangement as a list of booleans, where true represents heads and false represents tails. -/
def CoinArrangement := List Bool

/-- Defines the operation of flipping two neighboring coins with the same orientation. -/
def flip_neighbors (arrangement : CoinArrangement) (i : Nat) : CoinArrangement :=
  sorry

/-- Defines the equivalence relation between two coin arrangements. -/
def equivalent (a b : CoinArrangement) : Prop :=
  sorry

/-- The number of distinct coin arrangements that cannot be obtained from each other by flipping. -/
def distinct_arrangements (n : Nat) : Nat :=
  if n % 2 = 0 then n + 1 else 2

theorem coin_arrangement_theorem (n : Nat) :
  n > 0 →
  ∃ (arrangements : Finset CoinArrangement),
    (∀ a, a ∈ arrangements → a.length = n) ∧
    (∀ a b, a ∈ arrangements → b ∈ arrangements → a ≠ b → ¬equivalent a b) ∧
    arrangements.card = distinct_arrangements n :=
  sorry

#check coin_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_arrangement_theorem_l368_36869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_october_temperature_l368_36833

-- Define the temperature function
noncomputable def T (a A x : ℝ) : ℝ := a + A * Real.cos (Real.pi / 6 * (x - 6))

-- State the theorem
theorem october_temperature 
  (a A : ℝ) 
  (h1 : T a A 6 = 28) 
  (h2 : T a A 12 = 18) : 
  T a A 10 = 20.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_october_temperature_l368_36833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circumference_l368_36815

/-- Predicate to check if a rectangle ABCD is inscribed in a circle with given center -/
def IsInscribed (A B C D center : ℝ × ℝ) : Prop :=
  dist center A = dist center B ∧
  dist center B = dist center C ∧
  dist center C = dist center D ∧
  dist center D = dist center A

/-- Function to calculate the circumference of a circle given its center and a point on the circle -/
noncomputable def CircumferenceOfCircle (center point : ℝ × ℝ) : ℝ :=
  2 * Real.pi * dist center point

/-- Given a rectangle ABCD inscribed in a circle with AB = 8 and AD = 6,
    prove that the circumference of the circle is 30 when π = 3. -/
theorem inscribed_rectangle_circumference
  (A B C D : ℝ × ℝ)  -- Points of the rectangle
  (center : ℝ × ℝ)   -- Center of the circle
  (h_inscribed : IsInscribed A B C D center)
  (h_AB : dist A B = 8)
  (h_AD : dist A D = 6)
  (h_pi : Real.pi = 3) :
  CircumferenceOfCircle center A = 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circumference_l368_36815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_y_intercept_l368_36843

/-- The number of data points -/
def n : ℕ := 8

/-- The sum of x values -/
noncomputable def sum_x : ℝ := 6

/-- The sum of y values -/
noncomputable def sum_y : ℝ := 3

/-- The slope of the regression line -/
noncomputable def m : ℝ := 1/3

/-- Theorem: The y-intercept of the regression line -/
theorem regression_y_intercept : 
  ∃ (a : ℝ), (sum_y / n) = m * (sum_x / n) + a ∧ a = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_y_intercept_l368_36843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagstaff_height_l368_36808

/-- The height of the flagstaff given similar shadow conditions with a building -/
theorem flagstaff_height 
  (flagstaff_shadow building_shadow building_height flagstaff_height : ℝ)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75)
  (h_building_height : building_height = 12.5)
  (h_similar_conditions : building_height / building_shadow = flagstaff_height / flagstaff_shadow) :
  flagstaff_height = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagstaff_height_l368_36808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_alpha_l368_36872

noncomputable def P (α : Real) : Real × Real := (Real.sin α - Real.cos α, Real.tan α)

theorem range_of_alpha (α : Real) :
  α ∈ Set.Icc 0 (2 * Real.pi) →
  P α ∈ Set.prod (Set.Ioi 0) (Set.Ioi 0) →
  α ∈ Set.union (Set.Ioo (Real.pi / 4) (Real.pi / 2)) (Set.Ioo Real.pi (5 * Real.pi / 4)) :=
by
  sorry

#check range_of_alpha

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_alpha_l368_36872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l368_36847

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b * sin(C) + c * sin(B) = 4a * sin(B) * sin(C) and b^2 + c^2 - a^2 = 8,
    then the area of triangle ABC is 2√3/3. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C →
  b^2 + c^2 - a^2 = 8 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l368_36847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_curves_l368_36889

/-- An ellipse centered at the origin with foci at (-1,0) and (1,0) -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- A parabola with vertex at the origin and focus at (1,0) -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The intersection point of the ellipse and parabola -/
noncomputable def A : ℝ × ℝ := (3/2, Real.sqrt 6)

theorem intersection_point_on_curves : 
  ellipse A.1 A.2 ∧ parabola A.1 A.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_curves_l368_36889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_correct_l368_36832

/-- A natural number that satisfies the given remainder conditions -/
def SatisfiesConditions (y : ℕ) : Prop :=
  y % 5 = 4 ∧ y % 7 = 6 ∧ y % 8 = 7

/-- The smallest natural number satisfying the conditions -/
def SmallestSatisfyingNumber : ℕ := 279

/-- Theorem stating that SmallestSatisfyingNumber is indeed the smallest number satisfying the conditions -/
theorem smallest_satisfying_number_correct :
  SatisfiesConditions SmallestSatisfyingNumber ∧
  ∀ y : ℕ, 0 < y → y < SmallestSatisfyingNumber → ¬SatisfiesConditions y :=
by
  sorry

#eval SmallestSatisfyingNumber % 5
#eval SmallestSatisfyingNumber % 7
#eval SmallestSatisfyingNumber % 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_correct_l368_36832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_odd_function_property_l368_36849

-- Define the function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Theorem 1
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  ∀ x y : ℝ, x < y → f x > f y := by
  sorry

-- Define odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem 2
theorem odd_function_property (h : is_odd f) :
  is_odd (λ x ↦ f x - f (abs x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_odd_function_property_l368_36849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_pi_l368_36851

theorem angle_sum_pi (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi/2)
  (h_acute_β : 0 < β ∧ β < Real.pi/2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi/2)
  (h_cos_sum : Real.cos α + Real.cos β + Real.cos γ = 1 + 4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2)) :
  α + β + γ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_pi_l368_36851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l368_36850

theorem arithmetic_calculations :
  ((-14) - 14 + (-5) - (-30) - 2 = -5) ∧
  (-1^2008 - (5 * (-2) - (-4)^2 / (-8)) = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l368_36850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l368_36859

/-- Calculates the total cost for filling up three mini-vans and two trucks at a fuel station -/
theorem total_cost_calculation (base_minivan_capacity : ℝ) (minivan_price_per_liter : ℝ) (truck_price_per_liter : ℝ) (service_fee : ℝ) : 
  base_minivan_capacity = 65 →
  minivan_price_per_liter = 0.75 →
  truck_price_per_liter = 0.85 →
  service_fee = 2.10 →
  (let minivan1_capacity := base_minivan_capacity * 1.10
   let minivan2_capacity := base_minivan_capacity * 0.95
   let minivan3_capacity := base_minivan_capacity
   let truck_base_capacity := base_minivan_capacity * 2.20
   let truck1_capacity := truck_base_capacity
   let truck2_capacity := truck_base_capacity * 1.15
   (minivan1_capacity * minivan_price_per_liter + service_fee) +
   (minivan2_capacity * minivan_price_per_liter + service_fee) +
   (minivan3_capacity * minivan_price_per_liter + service_fee) +
   (truck1_capacity * truck_price_per_liter + service_fee) +
   (truck2_capacity * truck_price_per_liter + service_fee)) = 420.52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l368_36859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_four_l368_36828

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C₁ : Circle := { center := (0, 0), radius := 5 }
def C₂ : Circle := { center := (12, 0), radius := 7 }
def C₃ : Circle := { center := (15, 0), radius := 10 }

-- Define the property of external tangency between C₁ and C₂
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

-- Define the property of internal tangency of C₁ and C₂ to C₃
def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c2.radius - c1.radius)^2

-- Define the collinearity of the centers
def collinear (c1 c2 c3 : Circle) : Prop :=
  c1.center.2 = c2.center.2 ∧ c2.center.2 = c3.center.2

-- Define the length of the tangent from C₃ to C₂
noncomputable def tangent_length (c2 c3 : Circle) : ℝ :=
  Real.sqrt ((c3.radius + c2.radius)^2 - (c3.center.1 - c2.center.1)^2)

-- Theorem statement
theorem tangent_length_is_four :
  externally_tangent C₁ C₂ ∧
  internally_tangent C₁ C₃ ∧
  internally_tangent C₂ C₃ ∧
  collinear C₁ C₂ C₃ →
  tangent_length C₂ C₃ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_four_l368_36828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_odd_integers_sum_l368_36861

theorem no_five_consecutive_odd_integers_sum : 
  ¬ ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -379 := by
  sorry

#check no_five_consecutive_odd_integers_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_odd_integers_sum_l368_36861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l368_36841

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- Returns true if the given line is tangent to the given circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry  -- Definition of tangency

/-- The number of distinct lines tangent to both circles -/
def numTangentLines (c1 c2 : Circle) : ℕ :=
  sorry  -- Definition to count tangent lines

theorem four_tangent_lines (c1 c2 : Circle) :
  c1.radius = 3 →
  c2.radius = 5 →
  ((c1.center.fst - c2.center.fst)^2 + (c1.center.snd - c2.center.snd)^2) = 7^2 →
  numTangentLines c1 c2 = 4 :=
by
  sorry

#check four_tangent_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l368_36841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l368_36881

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the lines AE and CD
noncomputable def line_AE (x : ℝ) : ℝ := (A.2 - E.2) / (A.1 - E.1) * (x - A.1) + A.2
noncomputable def line_CD (x : ℝ) : ℝ := (C.2 - D.2) / (C.1 - D.1) * (x - C.1) + C.2

-- Define the intersection point F
noncomputable def F : ℝ × ℝ := 
  let x := (C.2 - D.2) * (A.1 - E.1) * C.1 - (A.2 - E.2) * (C.1 - D.1) * A.1
           / ((C.2 - D.2) * (A.1 - E.1) - (A.2 - E.2) * (C.1 - D.1))
  let y := line_AE x
  (x, y)

-- Theorem statement
theorem intersection_point_sum : F.1 + F.2 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l368_36881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l368_36811

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/8)

theorem graph_translation (x : ℝ) : g x = 3/4 - (1/4) * Real.sin (4*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l368_36811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_NH4_3PO4_approx_l368_36857

/-- Molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of phosphorus in g/mol -/
def molar_mass_P : ℝ := 30.97

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Number of NH4 groups in (NH4)3PO4 -/
def num_NH4_groups : ℕ := 3

/-- Number of PO4 groups in (NH4)3PO4 -/
def num_PO4_groups : ℕ := 1

/-- Calculates the molar mass of (NH4)3PO4 in g/mol -/
def molar_mass_NH4_3PO4 : ℝ :=
  num_NH4_groups * (molar_mass_N + 4 * molar_mass_H) +
  num_PO4_groups * (molar_mass_P + 4 * molar_mass_O)

/-- Calculates the mass of nitrogen in (NH4)3PO4 in g/mol -/
def mass_N_in_NH4_3PO4 : ℝ := num_NH4_groups * molar_mass_N

/-- Theorem: The mass percentage of N in (NH4)3PO4 is approximately 28.19% -/
theorem mass_percentage_N_in_NH4_3PO4_approx :
  ∃ ε > 0, abs ((mass_N_in_NH4_3PO4 / molar_mass_NH4_3PO4) * 100 - 28.19) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_NH4_3PO4_approx_l368_36857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_comparison_l368_36823

theorem revenue_comparison (R : ℝ) (hR : R > 0) : 
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_comparison_l368_36823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_convex_1000gon_whole_angles_l368_36803

-- Define a polygon type
structure Polygon where
  sides : ℕ
  interior_angles : Fin sides → ℕ
  exterior_angles : Fin sides → ℕ

-- Define properties for a convex polygon
def is_convex (p : Polygon) : Prop :=
  ∀ i : Fin p.sides, p.interior_angles i + p.exterior_angles i = 180

-- Define the sum of interior angles for a polygon
def sum_interior_angles (p : Polygon) : ℕ :=
  (p.sides - 2) * 180

-- Define the sum of exterior angles for a polygon
def sum_exterior_angles : ℕ := 360

-- Theorem: No convex 1000-gon exists with all angles as whole numbers
theorem no_convex_1000gon_whole_angles :
  ¬ ∃ (p : Polygon), p.sides = 1000 ∧ is_convex p ∧
  (∀ i : Fin p.sides, p.interior_angles i ∈ Set.univ) ∧
  (∀ i : Fin p.sides, p.exterior_angles i ∈ Set.univ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_convex_1000gon_whole_angles_l368_36803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_polynomial_to_y_expression_l368_36830

/-- Given y = x + 1/x, there exists a polynomial P(y) and a natural number n
    such that x^6 + x^5 - 5x^4 + x^3 + x + 1 = x^n * P(y) -/
theorem transform_polynomial_to_y_expression 
  (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  ∃ (P : ℝ → ℝ) (n : ℕ), x^6 + x^5 - 5*x^4 + x^3 + x + 1 = x^n * P y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_polynomial_to_y_expression_l368_36830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l368_36804

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a line with a given slope -/
structure Line where
  slope : ℝ

/-- Represents the number of intersections between a line and a branch of the hyperbola -/
noncomputable def intersections (h : Hyperbola) (l : Line) (right_branch : Bool) : ℕ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + (h.b / h.a) ^ 2)

/-- The theorem statement -/
theorem hyperbola_eccentricity_range (h : Hyperbola) :
  (∃ l1 : Line, l1.slope = 1 ∧ 
    intersections h l1 true = 1 ∧ 
    intersections h l1 false = 1) →
  (∃ l2 : Line, l2.slope = 2 ∧ 
    intersections h l2 true = 2) →
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l368_36804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l368_36800

/-- Sum of first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Theorem: For a geometric sequence with sum S(n), if S(2) = 4 and S(4) = 16, then S(8) = 160 -/
theorem geometric_sequence_sum :
  S 2 = 4 →
  S 4 = 16 →
  S 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l368_36800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_binomial_coefficient_in_expansion_l368_36890

theorem max_binomial_coefficient_in_expansion :
  ∃ (r : ℕ), 
    let n : ℕ := 7;
    let C : ℕ → ℕ → ℕ := fun k m => Nat.choose k m;
    let coeff : ℕ → ℚ := fun k => (C n k : ℚ) * (2^k);
    coeff r = 2 * coeff (r-1) ∧
    coeff r = (5/6) * coeff (r+1) ∧
    coeff 4 = 560 ∧
    ∀ k, k ≠ 4 → coeff k ≤ coeff 4 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_binomial_coefficient_in_expansion_l368_36890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_voters_for_tall_to_win_l368_36885

/-- Represents the voting structure and conditions of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : ℕ
  num_districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section
  num_districts_pos : num_districts > 0
  sections_per_district_pos : sections_per_district > 0
  voters_per_section_pos : voters_per_section > 0

/-- The minimum number of voters required to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : ℕ :=
  2 * (contest.num_districts / 2 + 1) * (contest.sections_per_district / 2 + 1)

/-- Theorem stating the minimum number of voters required for Tall to win the contest -/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h_total : contest.total_voters = 105)
  (h_districts : contest.num_districts = 5)
  (h_sections : contest.sections_per_district = 7)
  (h_voters : contest.voters_per_section = 3) :
  min_voters_to_win contest = 24 := by
  sorry

def main : IO Unit := do
  let contest : GiraffeContest := {
    total_voters := 105,
    num_districts := 5,
    sections_per_district := 7,
    voters_per_section := 3,
    total_voters_eq := by rfl,
    num_districts_pos := by norm_num,
    sections_per_district_pos := by norm_num,
    voters_per_section_pos := by norm_num
  }
  IO.println s!"Minimum voters required: {min_voters_to_win contest}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_voters_for_tall_to_win_l368_36885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l368_36879

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsPeriodic (f : ℝ → ℝ) : Prop := ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x

def IsStrictlyMonotonicDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem function_properties (f : ℝ → ℝ) :
  (IsOdd f → IsOdd (f ∘ f)) ∧
  (IsPeriodic f → IsPeriodic (f ∘ f)) ∧
  (IsStrictlyMonotonicDecreasing f → ∀ x y, x < y → f (f x) < f (f y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l368_36879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l368_36884

noncomputable def f (x : ℝ) := Real.sin x ^ 4 + 2 * Real.sin x * Real.cos x - Real.cos x ^ 4

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m) ∧
  (∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ π → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l368_36884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_l368_36865

open Real

/-- The height of a pole given three points on the ground and their angles of elevation -/
theorem pole_height (α β γ : ℝ) (h : α + β + γ = π / 2) : 
  ∃ (height : ℝ), 
    height > 0 ∧ 
    tan α = height / 10 ∧ 
    tan β = height / 20 ∧ 
    tan γ = height / 30 ∧ 
    height = 10 := by
  sorry

#check pole_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_l368_36865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l368_36813

-- Define x and y
noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 2 - Real.sqrt 3

-- Theorem 1
theorem problem_1 : 3 * x^2 + 5 * x * y + 3 * y^2 = 47 := by
  sorry

-- Theorem 2
theorem problem_2 : Real.sqrt (x / y) + Real.sqrt (y / x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l368_36813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l368_36826

/-- The angle between two lines given by their equations -/
noncomputable def angle_between_lines (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ :=
  Real.arctan (abs ((a2 * b1 - a1 * b2) / (a1 * a2 + b1 * b2)))

/-- Theorem: The angle between lines x-3y+3=0 and x-y+1=0 is arctan(1/2) -/
theorem angle_between_specific_lines :
  angle_between_lines 1 (-3) 3 1 (-1) 1 = Real.arctan (1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l368_36826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l368_36852

theorem tan_alpha_value (α : ℝ) 
    (h1 : Real.cos (π/2 + α) = 3/5) 
    (h2 : α ∈ Set.Ioo (π/2) (3*π/2)) : 
    Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l368_36852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l368_36863

noncomputable def a : ℤ := 5^2019 - 10 * ⌊(5^2019 : ℚ) / 10⌋
noncomputable def b : ℤ := 7^2020 - 10 * ⌊(7^2020 : ℚ) / 10⌋
noncomputable def c : ℤ := 13^2021 - 10 * ⌊(13^2021 : ℚ) / 10⌋

theorem abc_order : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l368_36863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_product_l368_36882

theorem xy_product (x y : ℝ) 
  (h1 : (2 : ℝ) ^ x = (16 : ℝ) ^ (y + 1))
  (h2 : (27 : ℝ) ^ y = (3 : ℝ) ^ (x - 2)) : 
  x * y = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_product_l368_36882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_50pi_l368_36836

/-- Configuration of three circles -/
structure CircleConfig where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of each smaller circle
  touching_internally : r < R
  touching_externally : r > 0

/-- The area of the shaded region in the circle configuration -/
noncomputable def shaded_area (c : CircleConfig) : ℝ :=
  c.R^2 * Real.pi - 2 * c.r^2 * Real.pi

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_50pi (c : CircleConfig) 
  (h1 : c.R = 10) 
  (h2 : c.r^2 - 10*c.r + 25 = 0) : 
  shaded_area c = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_50pi_l368_36836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l368_36806

theorem order_of_abc (a b c : ℝ) (ha : a = (0.3 : ℝ)^2) (hb : b = 2^(0.3 : ℝ)) (hc : c = Real.log 2 / Real.log 0.3) :
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l368_36806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l368_36874

theorem triangle_area_ratio (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3/4) (h5 : x^2 + y^2 + z^2 = 1/3) :
  1 - (x * (1 - z) + y * (1 - x) + z * (1 - y)) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l368_36874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l368_36868

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hour_to_m_per_sec (speed_km_h : ℝ) : ℝ :=
  speed_km_h * 1000 / 3600

/-- Calculates the time (in seconds) it takes for an object to travel a given distance -/
noncomputable def time_to_cross (length_m : ℝ) (speed_m_s : ℝ) : ℝ :=
  length_m / speed_m_s

theorem train_crossing_time :
  let train_length : ℝ := 240
  let train_speed_km_h : ℝ := 54
  let train_speed_m_s : ℝ := km_per_hour_to_m_per_sec train_speed_km_h
  time_to_cross train_length train_speed_m_s = 16 := by
  sorry

-- Remove #eval statements as they are not computable
-- #eval km_per_hour_to_m_per_sec 54
-- #eval time_to_cross 240 (km_per_hour_to_m_per_sec 54)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l368_36868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_attendance_l368_36840

theorem wedding_attendance (expected_attendees : ℕ) (no_show_rate : ℚ) 
  (h1 : expected_attendees = 220)
  (h2 : no_show_rate = 5 / 100) :
  ⌊(expected_attendees : ℚ) * (1 - no_show_rate)⌋ = 209 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_attendance_l368_36840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_connection_probability_l368_36818

-- Define the probability of error for each digit
noncomputable def p : ℝ := 0.02

-- Define a function to check if a number is correct according to the rule
def is_correct (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100 = (((n / 10) % 10 + n % 10) % 10))

-- Define the probability of a correct number with exactly two erroneous digits
noncomputable def r₂ : ℝ := 1 / 9

-- Define the probability of a correct number with all three erroneous digits
noncomputable def r₃ : ℝ := 8 / 81

-- State the theorem
theorem incorrect_connection_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ 
  abs ((3 * p^2 * (1 - p) * r₂ + p^3 * r₃) - 0.000131) < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_connection_probability_l368_36818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l368_36820

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 1

-- Define the reflection of a point about the y-axis
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

-- Define a line given a point and slope
def my_line (x₀ y₀ k : ℝ) (x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the distance between a point and a line
noncomputable def distance_point_line (x₀ y₀ k : ℝ) (x y : ℝ) : ℝ :=
  abs (k * x - y + y₀ - k * x₀) / Real.sqrt (k^2 + 1)

theorem reflected_ray_slope :
  ∃ (k : ℝ), (k = -4/3 ∨ k = -3/4) ∧
  (∀ (x y : ℝ), my_line 2 (-3) k x y →
    (∃ (x' y' : ℝ), my_circle x' y' ∧
      distance_point_line 2 (-3) k x' y' = 1)) ∧
  (reflect_y (-2) (-3) = (2, -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l368_36820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_b_necessary_not_sufficient_l368_36892

-- Define a complex number
def complex (a b : ℝ) : ℂ := (a - b) + (a + b) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem a_eq_b_necessary_not_sufficient :
  (∀ a b : ℝ, is_purely_imaginary (complex a b) → a = b) ∧
  (∃ a b : ℝ, a = b ∧ ¬is_purely_imaginary (complex a b)) :=
by
  constructor
  · intro a b h
    sorry -- Proof that the condition is necessary
  · use 0, 0
    constructor
    · rfl
    · intro h
      sorry -- Proof that the condition is not sufficient


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_b_necessary_not_sufficient_l368_36892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_polygons_with_A₁_l368_36858

/-- A type representing a point on a circle -/
def Point : Type := Unit

/-- A convex polygon is represented as a set of points -/
def ConvexPolygon : Type := Set Point

/-- The set of all points on the circle -/
def allPoints : Finset Point := sorry

/-- The specific point A₁ -/
def A₁ : Point := sorry

/-- The set of all convex polygons that can be formed from the points -/
def allPolygons : Finset ConvexPolygon := sorry

/-- The set of polygons that include A₁ -/
def polygonsWithA₁ : Finset ConvexPolygon := sorry

/-- The set of polygons that do not include A₁ -/
def polygonsWithoutA₁ : Finset ConvexPolygon := sorry

theorem more_polygons_with_A₁ :
  Finset.card polygonsWithA₁ > Finset.card polygonsWithoutA₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_polygons_with_A₁_l368_36858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_E_l368_36827

-- Define a cube type
structure Cube where
  faces : Fin 6 → Char

-- Define adjacency relation
def adjacent (c : Cube) (f1 f2 : Fin 6) : Prop :=
  f1 ≠ f2 ∧ ∃ (edge : Fin 12), (∃ (connects : Fin 12 → Fin 6 × Fin 6), connects edge = (f1, f2))

-- Define opposite relation
def opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  f1 ≠ f2 ∧ ∀ (f : Fin 6), f ≠ f1 ∧ f ≠ f2 → (adjacent c f1 f ↔ ¬adjacent c f2 f)

-- Theorem statement
theorem opposite_face_of_E (c : Cube) 
  (h1 : c.faces 0 = 'A')
  (h2 : c.faces 1 = 'Б')
  (h3 : c.faces 2 = 'В')
  (h4 : c.faces 3 = 'Г')
  (h5 : c.faces 4 = 'Д')
  (h6 : c.faces 5 = 'Е')
  (adj1 : adjacent c 3 0)
  (adj2 : adjacent c 3 1)
  (adj3 : adjacent c 3 2)
  (adj4 : adjacent c 3 4) :
  opposite c 3 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_E_l368_36827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l368_36897

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a / Real.exp x

-- State the theorem
theorem tangent_point_abscissa 
  (a : ℝ) 
  (h1 : ∀ x, f_deriv a x = -f_deriv a (-x)) -- f' is an odd function
  (h2 : ∃ x, f_deriv a x = 3/2) -- there exists a point where f'(x) = 3/2
  : ∃ x, f_deriv a x = 3/2 ∧ x = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l368_36897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_calculation_l368_36891

/-- The area of a trapezoid with given dimensions -/
noncomputable def trapezoidArea (upperSide lowerSide height : ℝ) : ℝ :=
  (1/2) * (upperSide + lowerSide) * height

/-- Theorem: The area of a trapezoid with an upper side of 12 cm, 
    a lower side 4 cm longer than the upper side, and a height of 10 cm 
    is equal to 140 square centimeters. -/
theorem trapezoid_area_calculation :
  let upperSide : ℝ := 12
  let lowerSide : ℝ := upperSide + 4
  let height : ℝ := 10
  trapezoidArea upperSide lowerSide height = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_calculation_l368_36891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_two_rays_points_on_curve_l368_36873

noncomputable section

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ :=
  (t + 1/t, 2)

-- Theorem statement
theorem curve_is_two_rays :
  ∀ (x y : ℝ), (∃ t : ℝ, t ≠ 0 ∧ curve t = (x, y)) →
  ((x ≥ 2 ∨ x ≤ -2) ∧ y = 2) :=
by
  sorry

-- Additional theorem to show that all points satisfying the condition are on the curve
theorem points_on_curve :
  ∀ (x y : ℝ), ((x ≥ 2 ∨ x ≤ -2) ∧ y = 2) →
  (∃ t : ℝ, t ≠ 0 ∧ curve t = (x, y)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_two_rays_points_on_curve_l368_36873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l368_36880

noncomputable section

-- Define the given circle
def given_circle : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 - 2*x = 0

-- Define the given line
def given_line : ℝ × ℝ → Prop := λ (x, y) ↦ x + Real.sqrt 3 * y = 0

-- Define the tangent point
def tangent_point : ℝ × ℝ := (3, -Real.sqrt 3)

-- Define the property of being externally tangent
def externally_tangent (c1 c2 : ℝ × ℝ → Prop) : Prop := sorry

-- Define the property of being tangent to a line at a point
def tangent_to_line_at_point (c : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop := sorry

-- Define the equation of circle C
def circle_C : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 4)^2 + y^2 = 4 ∨ x^2 + (y + 4*Real.sqrt 3)^2 = 36

theorem circle_equation :
  ∀ (C : ℝ × ℝ → Prop),
  externally_tangent C given_circle ∧
  tangent_to_line_at_point C given_line tangent_point →
  ∀ x y, C (x, y) ↔ circle_C (x, y) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l368_36880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_point_on_circle_l368_36854

-- Define the ellipse structure
structure Ellipse where
  f1 : EuclideanSpace ℝ (Fin 2) -- First focus
  f2 : EuclideanSpace ℝ (Fin 2) -- Second focus
  a : ℝ                         -- Semi-major axis length

-- Define a point on the ellipse
def PointOnEllipse (e : Ellipse) (p : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist p e.f1 + dist p e.f2 = 2 * e.a

-- Define the extension point Q
def ExtensionPoint (e : Ellipse) (p q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  PointOnEllipse e p ∧ dist p q = dist p e.f2

-- Theorem statement
theorem extension_point_on_circle (e : Ellipse) (p q : EuclideanSpace ℝ (Fin 2)) :
  ExtensionPoint e p q → dist q e.f1 = 2 * e.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_point_on_circle_l368_36854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l368_36834

open Real

-- Define the curve C
def C (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (x - arcsin a) * (x - arccos a) + (y - arcsin a) * (y + arccos a) = 0

-- Define the intersection line
def intersectionLine : ℝ → Prop :=
  λ x => x = π / 4

-- Define the chord length
noncomputable def chordLength (a : ℝ) : ℝ := 
  sqrt (2 * ((arcsin a)^2 + (arccos a)^2))

-- Theorem statement
theorem min_chord_length :
  ∀ a : ℝ, chordLength a ≥ π / 2 ∧ ∃ a₀ : ℝ, chordLength a₀ = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l368_36834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l368_36878

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  ∃ (d : ℝ), d = 4 ∧ 
  (∀ (x y : ℝ), x^2 - 10*x + y^2 - 4*y - 7 = 0 → 
   ∀ (x' y' : ℝ), x'^2 + 14*x' + y'^2 + 6*y' + 49 = 0 → 
   (x - x')^2 + (y - y')^2 ≥ d^2) := by
  -- First circle equation: x^2 - 10*x + y^2 - 4*y - 7 = 0
  -- Second circle equation: x^2 + 14*x + y^2 + 6*y + 49 = 0
  -- The shortest distance between the circles is 4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l368_36878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_m_range_for_two_roots_l368_36876

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.exp x - x * Real.sin x

-- Define the equation h(x) = 0
def h (m x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 - x - 1 - m * (x * Real.cos x - Real.sin x)

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 Real.pi → f x ≤ f y ∧ f x = 2 := by
  sorry

-- Theorem for the range of m
theorem m_range_for_two_roots :
  ∀ (m : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi/2) ∧ x₂ ∈ Set.Icc 0 (Real.pi/2) ∧ 
    x₁ ≠ x₂ ∧ h m x₁ = 0 ∧ h m x₂ = 0) ↔ 
    m ∈ Set.Ioo (-(Real.exp (Real.pi/2)) + (Real.pi^2/8) + (Real.pi/2) + 1) (-1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_m_range_for_two_roots_l368_36876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_g_odd_l368_36870

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem f_decreasing_and_g_odd :
  (∀ x y, x ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) → y ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) → x < y → f x > f y) ∧
  (∀ x : ℝ, g (-x) = -g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_g_odd_l368_36870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_l368_36809

-- Define the circle and points
def Circle : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 50 }

variable (A B C : ℝ × ℝ)

-- Define the conditions
axiom A_on_circle : A ∈ Circle
axiom C_on_circle : C ∈ Circle
axiom B_inside_circle : B.1^2 + B.2^2 < 50
axiom angle_ABC_90 : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
axiom AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36
axiom BC_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4

-- Define the center of the circle
def O : ℝ × ℝ := (0, 0)

-- State the theorem
theorem distance_B_to_center :
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_l368_36809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_i_l368_36893

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_of_i : dilation (1 + 2*I) 2 (0 + I) = -1 := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_i_l368_36893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l368_36862

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := Real.rpow 3 t - 4
noncomputable def y (t : ℝ) : ℝ := Real.rpow 3 (2 * t) - 7 * Real.rpow 3 t - 2

-- Theorem statement
theorem points_form_parabola :
  ∃ (a b c : ℝ), ∀ (t : ℝ), y t = a * (x t)^2 + b * (x t) + c :=
by
  -- Introduce the constants a, b, and c
  let a := 1
  let b := 1
  let c := -6
  
  -- Prove the existence of a, b, and c
  use a, b, c
  
  -- Prove that the equation holds for all t
  intro t
  
  -- Expand the definitions of x and y
  simp [x, y]
  
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l368_36862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_matrix_property_l368_36816

theorem scalar_matrix_property (M : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, M.mulVec v = (7 : ℝ) • v) ↔
  M = ![![7, 0, 0], ![0, 7, 0], ![0, 0, 7]] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_matrix_property_l368_36816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_count_l368_36829

def M : Finset Nat := {0, 1, 2, 3, 4}
def N : Finset Nat := {1, 3, 5}

theorem intersection_subset_count : Finset.card (Finset.powerset (M ∩ N)) = 4 := by
  -- Compute the intersection
  have h_intersection : M ∩ N = {1, 3} := by rfl
  
  -- Count the subsets
  calc
    Finset.card (Finset.powerset (M ∩ N)) = Finset.card (Finset.powerset {1, 3}) := by rw [h_intersection]
    _ = 2^2 := by rfl
    _ = 4 := by rfl

  -- Alternative concise proof:
  -- rw [h_intersection]
  -- rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_count_l368_36829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l368_36822

/-- Given a regular triangular pyramid with apex angle α, 
    the dihedral angle β at the base is arccos(√3 * tan(α/2)) -/
theorem dihedral_angle_regular_triangular_pyramid (α : Real) :
  ∃ β : Real, β = Real.arccos (Real.sqrt 3 * Real.tan (α / 2)) ∧
  ∃ (pyramid : {p : Real × Real // p.1 > 0 ∧ p.2 > 0}),
    pyramid.val.1 = α ∧ pyramid.val.2 = β :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l368_36822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l368_36824

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  -- Vectors m and n are perpendicular
  Real.cos t.A * (2 * t.c + t.b) + t.a * Real.cos t.B = 0 ∧
  -- Side a is given
  t.a = 4 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : problem_conditions t) : 
  t.A = 2 * Real.pi / 3 ∧ 
  (∀ t' : Triangle, problem_conditions t' → 
    t'.a * t'.b * Real.sin t'.C / 2 ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l368_36824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_slope_l368_36864

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the slope of a line
def line_slope (m : ℝ) (x y : ℝ) : Prop := y = m * x + (1 - m)

-- Theorem statement
theorem correct_slope :
  ∀ x y : ℝ, line_equation x y → line_slope 1 x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_slope_l368_36864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l368_36837

/-- Triangle ABC with midpoints A', B', C' -/
structure Triangle (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C : V)
  (A' B' C' : V)
  (midpoint_A' : A' = (1/2 : ℝ) • (B + C))
  (midpoint_B' : B' = (1/2 : ℝ) • (A + C))
  (midpoint_C' : C' = (1/2 : ℝ) • (A + B))

/-- The statement of the problem -/
theorem midpoint_theorem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V] 
  (t : Triangle V) :
  ∃ (X : V), ∀ (P P' : V),
    (‖P - t.A‖ = ‖P' - t.A'‖) →
    (‖P - t.B‖ = ‖P' - t.B'‖) →
    (‖P - t.C‖ = ‖P' - t.C'‖) →
    ∃ (r : ℝ), X = (1 - r) • P + r • P' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l368_36837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l368_36856

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

-- Define the theorem
theorem function_and_triangle_properties 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2)
  (h_max : ∃ x, f x 3 = 6 ∧ ∀ y, f y 3 ≤ 6)
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Sides of the triangle
  (h_f_A : f A 3 = 5)
  (h_a : a = 4)
  (h_area : 1/2 * b * c * Real.sin A = Real.sqrt 3) :
  b + c = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l368_36856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_theorem_l368_36801

/-- A 2000 x 2000 table filled with 1 or -1 -/
def Table := Fin 2000 → Fin 2000 → Int

/-- Predicate to check if a table contains only 1 or -1 -/
def validTable (t : Table) : Prop :=
  ∀ i j, t i j = 1 ∨ t i j = -1

/-- The sum of all numbers in the table -/
def tableSum (t : Table) : Int :=
  Finset.sum (Finset.univ : Finset (Fin 2000)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 2000)) fun j =>
      t i j

/-- Subset of 1000 rows -/
def RowSubset := Fin 1000 → Fin 2000

/-- Subset of 1000 columns -/
def ColSubset := Fin 1000 → Fin 2000

/-- Sum of numbers at the intersections of given row and column subsets -/
def intersectionSum (t : Table) (rows : RowSubset) (cols : ColSubset) : Int :=
  Finset.sum (Finset.univ : Finset (Fin 1000)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 1000)) fun j =>
      t (rows i) (cols j)

theorem intersection_sum_theorem (t : Table) 
  (h1 : validTable t) 
  (h2 : tableSum t ≥ 0) :
  ∃ (rows : RowSubset) (cols : ColSubset), intersectionSum t rows cols ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_theorem_l368_36801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_m_range_l368_36848

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m - x^2 + 2 * Real.log x

-- Define the interval
def interval : Set ℝ := { x | 1 / Real.exp 2 ≤ x ∧ x ≤ Real.exp 1 }

-- State the theorem
theorem two_zeros_m_range (m : ℝ) :
  (∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0) →
  1 < m ∧ m ≤ 4 + 1 / Real.exp 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_m_range_l368_36848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_valid_routes_l368_36805

/-- Represents a city in the road network -/
inductive City : Type
| P | Q | R | S | T
deriving DecidableEq, BEq

/-- Represents a road between two cities -/
inductive Road : Type
| PQ | PR | PT | QS | QR | RS | ST
deriving DecidableEq, BEq

/-- The road network connecting the cities -/
def road_network : List Road :=
  [Road.PQ, Road.PR, Road.PT, Road.QS, Road.QR, Road.RS, Road.ST]

/-- A route is a list of roads -/
def Route := List Road

/-- Function to check if a route is valid (uses each road exactly once) -/
def is_valid_route (r : Route) : Bool :=
  r.toFinset == road_network.toFinset && r.length == road_network.length

/-- Function to check if a route starts at P and ends at Q -/
def is_PQ_route (r : Route) : Bool :=
  match r.head? with
  | some Road.PQ => true
  | some Road.PR => true
  | some Road.PT => true
  | _ => false

/-- The number of valid routes from P to Q -/
def num_valid_PQ_routes : Nat :=
  (List.filter (λ r => is_valid_route r && is_PQ_route r) (List.permutations road_network)).length

/-- Theorem stating that there are 16 valid routes from P to Q -/
theorem sixteen_valid_routes : num_valid_PQ_routes = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_valid_routes_l368_36805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_white_squares_correct_max_white_squares_surrounded_l368_36887

/-- The maximum number of white squares that can be surrounded by n black squares -/
def max_white_squares (n : ℕ) : ℕ :=
  let k := n / 4
  match n % 4 with
  | 0 => 2 * k^2 - 2 * k + 1
  | 1 => 2 * k^2 - k
  | 2 => 2 * k^2
  | 3 => 2 * k^2 + k
  | _ => 0  -- This case should never occur, but it's needed for exhaustiveness

theorem max_white_squares_correct (n : ℕ) :
  max_white_squares n = 
    let k := n / 4
    match n % 4 with
    | 0 => 2 * k^2 - 2 * k + 1
    | 1 => 2 * k^2 - k
    | 2 => 2 * k^2
    | 3 => 2 * k^2 + k
    | _ => 0  -- This case should never occur, but it's needed for exhaustiveness
:= by sorry

/-- The maximum number of white squares is achieved when white squares are surrounded by black squares -/
theorem max_white_squares_surrounded (n : ℕ) :
  ∀ (white_count : ℕ), white_count ≤ max_white_squares n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_white_squares_correct_max_white_squares_surrounded_l368_36887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l368_36853

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (0, -5)
def F₂ : ℝ × ℝ := (0, 5)

-- Define the distance function as noncomputable
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points M satisfying the condition
def M : Set (ℝ × ℝ) := {m : ℝ × ℝ | distance m F₁ + distance m F₂ = 10}

-- Theorem statement
theorem locus_is_line_segment : 
  M = {m : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ m = (0, -5 + 10*t)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l368_36853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_power_function_l368_36814

/-- A function f is a power function if there exist constants a and b such that f(x) = ax^b for all x in the domain of f, where a ≠ 0. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x ^ b

/-- Given function -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 1) * x^(-4*α - 2)

/-- Theorem: If f(α) is a power function, then α = 2 -/
theorem alpha_value_for_power_function (α : ℝ) :
  IsPowerFunction (f α) → α = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_power_function_l368_36814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l368_36855

/-- Given a group of people with an initial average age and size, and a new group joining with their own average age and size, calculate the new average age of the combined group. -/
theorem new_average_age 
  (initial_group_size : ℕ) 
  (initial_average_age : ℚ) 
  (new_group_size : ℕ) 
  (new_group_average_age : ℚ) : 
  initial_group_size = 12 → 
  initial_average_age = 16 → 
  new_group_size = 12 → 
  new_group_average_age = 15 → 
  (initial_group_size * initial_average_age + new_group_size * new_group_average_age) / 
    (initial_group_size + new_group_size) = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l368_36855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l368_36877

theorem trig_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 47/127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l368_36877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l368_36888

/-- Represents the work rate of a single man per day -/
def r : ℝ := sorry

/-- Represents the total work to be completed -/
def W : ℝ := sorry

/-- The number of men initially working -/
def initial_men : ℕ := 5

/-- The number of days the initial group works -/
def initial_days : ℕ := 15

/-- The number of men after some drop out -/
def remaining_men : ℕ := 3

/-- The number of days the remaining men work to complete the job -/
def remaining_days : ℕ := 25

/-- The work completed in the initial period -/
def work_completed : ℝ := initial_men * initial_days * r

/-- The total work is the sum of initial work and remaining work -/
axiom total_work : W = initial_men * initial_days * r + remaining_men * remaining_days * r

/-- The theorem stating the ratio of initial work to total work -/
theorem work_ratio : work_completed / W = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l368_36888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_containing_points_l368_36821

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The theorem statement -/
theorem circle_containing_points (A : Finset Point) (n : ℕ) (h : n ≥ 2) (h' : A.card = n) :
  ∃ (c : Circle), ∃ (p q : Point), p ∈ A ∧ q ∈ A ∧
    (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2 ∧
    (c.center.x - q.x)^2 + (c.center.y - q.y)^2 = c.radius^2 ∧
    (p.x - q.x)^2 + (p.y - q.y)^2 = (2 * c.radius)^2 ∧
    (A.filter (fun r => (c.center.x - r.x)^2 + (c.center.y - r.y)^2 ≤ c.radius^2)).card ≥ n / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_containing_points_l368_36821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_99_l368_36895

def a : ℕ → ℕ
  | 0 => 10
  | 1 => 10
  | 2 => 10
  | 3 => 10
  | 4 => 10
  | 5 => 10
  | 6 => 10
  | 7 => 10
  | 8 => 10
  | 9 => 10
  | 10 => 10
  | n+11 => 100 * a (n+10) + (n+11)

theorem least_multiple_of_99 : 
  (∀ k, 10 < k ∧ k < 45 → ¬ (99 ∣ a k)) ∧ (99 ∣ a 45) := by
  sorry

#eval a 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_99_l368_36895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crescent_moon_area_l368_36839

noncomputable section

/-- The area of a quarter circle with radius r -/
def quarterCircleArea (r : ℝ) : ℝ := (Real.pi * r^2) / 4

/-- The area of the crescent moon -/
def crescentMoonArea : ℝ := quarterCircleArea 4 - quarterCircleArea 2

theorem crescent_moon_area : crescentMoonArea = 3 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crescent_moon_area_l368_36839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_colors_l368_36844

/-- Modified sum of prime factors, replacing 5 with 1 and 7 with 6 -/
def S (a : ℕ+) : ℕ := sorry

/-- Coloring function based on S(a) mod 7 -/
def f (a : ℕ+) : Fin 7 := 
  Fin.ofNat (S a % 7)

theorem distinct_colors (a : ℕ+) : 
  (({a, 2*a, 3*a, 4*a, 5*a, 6*a, 7*a} : Finset ℕ+).image f).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_colors_l368_36844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_result_l368_36871

variable (a b c : ℂ)

def equation1 (a b c : ℂ) : Prop := 
  (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -4

def equation2 (a b c : ℂ) : Prop := 
  (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 7

theorem result (h1 : equation1 a b c) (h2 : equation2 a b c) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_result_l368_36871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l368_36866

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_ordering (k x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h_k : k > 0)
  (h_x : x₁ < 0 ∧ 0 < x₂ ∧ x₂ < x₃)
  (h_y₁ : y₁ = inverse_proportion k x₁)
  (h_y₂ : y₂ = inverse_proportion k x₂)
  (h_y₃ : y₃ = inverse_proportion k x₃) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l368_36866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_for_specific_ellipse_l368_36835

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a parabola with focus (p, 0) and directrix x = -p -/
structure Parabola where
  p : ℝ
  h_positive : 0 < p

/-- The distance between intersection points of an ellipse and a parabola
    sharing a focus, where the directrix of the parabola is the minor axis of the ellipse -/
noncomputable def intersection_distance (e : Ellipse) (p : Parabola) : ℝ :=
  4 * Real.sqrt 14 / 3

/-- Theorem stating the distance between intersection points for the given ellipse and parabola -/
theorem intersection_distance_for_specific_ellipse :
  ∃ (e : Ellipse) (p : Parabola),
    e.a = 5 ∧ e.b = 3 ∧
    p.p = 4 ∧
    intersection_distance e p = 4 * Real.sqrt 14 / 3 := by
  sorry

#check intersection_distance_for_specific_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_for_specific_ellipse_l368_36835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l368_36825

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Theorem stating that there are no intersection points
theorem no_intersection : ¬∃ (x y : ℝ), line_eq x y ∧ circle_eq x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l368_36825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l368_36812

theorem problem_statement : (7/16 : ℝ) - (7/8 : ℝ) * (Real.sin (15 * π / 180))^2 = (7 * Real.sqrt 3) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l368_36812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_survey_size_correct_l368_36867

/-- The number of customers who responded to the original survey -/
def original_responses : ℕ := 7

/-- The number of customers sent the redesigned survey -/
def redesigned_total : ℕ := 63

/-- The number of customers who responded to the redesigned survey -/
def redesigned_responses : ℕ := 9

/-- The increase in response rate (in percentage points) -/
def response_rate_increase : ℚ := 4

/-- The number of customers sent the original survey -/
def original_total : ℕ := 68

/-- Theorem stating that the calculated number of customers sent the original survey is approximately correct -/
theorem original_survey_size_correct : 
  ∃ ε : ℚ, ε > 0 ∧ ε < 1 ∧ 
  abs ((original_responses : ℚ) / original_total * 100 + response_rate_increase - 
       (redesigned_responses : ℚ) / redesigned_total * 100) < ε := by
  sorry

#check original_survey_size_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_survey_size_correct_l368_36867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_horse_odds_l368_36817

/-- Represents the odds of a horse winning as a ratio of two natural numbers -/
structure Odds where
  forOdds : ℕ
  againstOdds : ℕ

/-- Represents a horse race with three horses -/
structure HorseRace where
  x_odds : Odds
  y_odds : Odds

/-- Calculates the probability of winning given the odds -/
def probability (odds : Odds) : ℚ :=
  odds.againstOdds / (odds.forOdds + odds.againstOdds)

/-- Theorem stating the odds of the third horse winning -/
theorem third_horse_odds (race : HorseRace) 
  (h1 : race.x_odds = { forOdds := 3, againstOdds := 1 })
  (h2 : race.y_odds = { forOdds := 2, againstOdds := 3 }) :
  ∃ z_odds : Odds, 
    z_odds = { forOdds := 3, againstOdds := 17 } ∧ 
    probability race.x_odds + probability race.y_odds + probability z_odds = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_horse_odds_l368_36817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_equal_sum_groups_l368_36842

theorem partition_into_equal_sum_groups
  (n m k : ℕ) (h_m_ge_n : m ≥ n) (h_sum : (Finset.range n).sum id = m * k) :
  ∃ (partition : Finset (Finset ℕ)),
    partition.card = k ∧
    (∀ S ∈ partition, S.sum id = m) ∧
    (Finset.biUnion partition id : Finset ℕ) = Finset.range n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_equal_sum_groups_l368_36842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_equals_1_l368_36838

def sequence_a : ℕ → ℤ
  | 0 => -1  -- Added case for 0
  | 1 => -1
  | 2 => 2
  | n + 3 => sequence_a (n + 2) + sequence_a (n + 1)

theorem a_2008_equals_1 : sequence_a 2008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_equals_1_l368_36838
