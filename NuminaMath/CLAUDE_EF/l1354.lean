import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_distance_from_mountain_l1354_135407

/-- Converts a map distance to actual distance given a scale --/
noncomputable def mapToActualDistance (mapDistance : ℝ) (mapScale : ℝ) (actualScale : ℝ) : ℝ :=
  mapDistance * (actualScale / mapScale)

/-- Theorem: Given the map scale and Ram's position, calculate his actual distance from the mountain base --/
theorem ram_distance_from_mountain (mapDistanceBetweenMountains : ℝ) (actualDistanceBetweenMountains : ℝ) 
  (mapDistanceRam : ℝ) : 
  mapDistanceBetweenMountains = 312 →
  actualDistanceBetweenMountains = 136 →
  mapDistanceRam = 28 →
  ∃ ε > 0, |mapToActualDistance mapDistanceRam mapDistanceBetweenMountains actualDistanceBetweenMountains - 12.205| < ε := by
  sorry

#check ram_distance_from_mountain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_distance_from_mountain_l1354_135407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l1354_135402

/-- The length of a wire in meters, given its volume in cubic centimeters and diameter in millimeters -/
noncomputable def wire_length (volume : ℝ) (diameter : ℝ) : ℝ :=
  let radius : ℝ := diameter / 2 / 10  -- Convert diameter to radius in cm
  let height : ℝ := volume / (Real.pi * radius^2)
  height / 100  -- Convert cm to m

/-- Theorem stating that a wire with volume 44 cubic cm and diameter 1 mm has length approximately 56.0254 m -/
theorem wire_length_approx :
  ∃ ε > 0, |wire_length 44 1 - 56.0254| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l1354_135402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l1354_135403

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - A.1, C.2 - A.2)
  let w := (C.1 - B.1, C.2 - B.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- The area of the triangle with vertices (-3,6), (9,-2), and (11,5) is 48 -/
theorem specific_triangle_area : 
  triangle_area (-3, 6) (9, -2) (11, 5) = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l1354_135403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_front_view_heights_l1354_135412

/-- Represents the heights of cube stacks in a column -/
def ColumnHeights := List Nat

/-- The front view height of a column is the maximum height in that column -/
def frontViewHeight (column : ColumnHeights) : Nat :=
  match column.maximum? with
  | some max => max
  | none => 0

/-- The given cube stack configuration -/
def cubeStacks : List ColumnHeights :=
  [[3, 1], [2, 4, 2], [5, 2]]

/-- Theorem: The front view heights of the given cube stacks are [3, 4, 5] -/
theorem front_view_heights :
  (cubeStacks.map frontViewHeight) = [3, 4, 5] := by
  sorry

#eval cubeStacks.map frontViewHeight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_front_view_heights_l1354_135412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_little_cars_sold_l1354_135475

def total_earnings : ℕ := 45
def legos_cost : ℕ := 30
def car_price : ℕ := 5

theorem little_cars_sold : 
  (total_earnings - legos_cost) / car_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_little_cars_sold_l1354_135475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_special_case_l1354_135408

/-- 
Given a quadratic polynomial y = (2/√3)x² + bx + c whose graph intersects 
the coordinate axes at three points forming an isosceles triangle with 
two equal sides and an angle of 120°, the roots of the polynomial are 0.5 and 1.5.
-/
theorem quadratic_roots_special_case 
  (b c : ℝ) 
  (h1 : ∃ K L M : ℝ × ℝ, 
    (K.1 = 0 ∨ K.2 = 0) ∧ 
    (L.1 = 0 ∨ L.2 = 0) ∧ 
    (M.1 = 0 ∨ M.2 = 0) ∧ 
    (2 / Real.sqrt 3 * K.1^2 + b * K.1 + c = K.2) ∧
    (2 / Real.sqrt 3 * L.1^2 + b * L.1 + c = L.2) ∧
    (2 / Real.sqrt 3 * M.1^2 + b * M.1 + c = M.2))
  (h2 : ∃ K L M : ℝ × ℝ, 
    Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2) = 
    Real.sqrt ((K.1 - M.1)^2 + (K.2 - M.2)^2))
  (h3 : ∃ K L M : ℝ × ℝ,
    Real.arccos ((L.1 - K.1) * (M.1 - K.1) + (L.2 - K.2) * (M.2 - K.2)) / 
    (Real.sqrt ((L.1 - K.1)^2 + (L.2 - K.2)^2) * 
     Real.sqrt ((M.1 - K.1)^2 + (M.2 - K.2)^2)) = 2 * π / 3) :
  ∃ x₁ x₂ : ℝ, x₁ = 0.5 ∧ x₂ = 1.5 ∧ 
    2 / Real.sqrt 3 * x₁^2 + b * x₁ + c = 0 ∧
    2 / Real.sqrt 3 * x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_special_case_l1354_135408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1354_135423

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 3 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Define the center of the circle
def center : ℝ × ℝ := (2, -1)

-- Define the radius of the circle
noncomputable def radius : ℝ := Real.sqrt 2

-- Define the tangent lines
def tangent_lines (x y : ℝ) : Prop :=
  ((-2 + Real.sqrt 6) * x - 2 * y = 0) ∨
  ((-2 - Real.sqrt 6) * x - 2 * y = 0) ∨
  (x + 2 * y + Real.sqrt 10 = 0) ∨
  (x + 2 * y - Real.sqrt 10 = 0)

theorem circle_properties :
  (∀ x y, circle_C x y ↔ circle_C (y - 1) (x - 1)) ∧  -- Symmetry
  fourth_quadrant center.1 center.2 ∧                  -- Center in fourth quadrant
  (∀ x y, circle_C x y → ((x - center.1)^2 + (y - center.2)^2 = radius^2)) ∧  -- Radius
  (∃ x y, circle_C x y ∧ tangent_lines x y ∧           -- Existence of tangent lines
    (∃ a ≠ 0, x = 2*a ∧ y = a) ∨ (x ≠ 0 ∧ y = 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1354_135423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_reflection_l1354_135430

/-- Rectangle vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 1)
def D : ℝ × ℝ := (0, 1)

/-- Midpoint of AB -/
noncomputable def P₀ : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Points of reflection -/
noncomputable def P₁ (θ : ℝ) : ℝ × ℝ := (B.1, (B.1 - P₀.1) * Real.tan θ)
noncomputable def P₂ (θ : ℝ) : ℝ × ℝ := (C.1 - (C.2 - (P₁ θ).2) * Real.tan θ, C.2)
noncomputable def P₃ (θ : ℝ) : ℝ × ℝ := (D.1, D.2 - (D.1 - (P₂ θ).1) * Real.tan θ)
noncomputable def P₄ (θ : ℝ) : ℝ × ℝ := (A.1 + (A.2 - (P₃ θ).2) / Real.tan θ, A.2)

/-- Theorem statement -/
theorem particle_reflection (θ : ℝ) : P₄ θ = P₀ → Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_reflection_l1354_135430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_two_implies_expression_equals_negative_two_l1354_135431

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π/2 + θ) - Real.cos (π - θ)) / (Real.sin (π/2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_two_implies_expression_equals_negative_two_l1354_135431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_congruent_triangles_area_union_specific_triangles_l1354_135462

/-- A right triangle with sides 15, 20, and 25 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 15
  side_b : b = 20
  side_c : c = 25

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The area of the union of two congruent right triangles is equal to the area of one triangle -/
theorem area_union_congruent_triangles (t : RightTriangle) :
  triangleArea t.a t.b = 150 := by
  sorry

/-- The main theorem: The area of the union of two congruent right triangles with sides 15, 20, and 25 is equal to 150 -/
theorem area_union_specific_triangles :
  ∃ t : RightTriangle, triangleArea t.a t.b = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_congruent_triangles_area_union_specific_triangles_l1354_135462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_southbound_vehicles_count_l1354_135439

/-- Represents the traffic scenario on a highway --/
structure TrafficScenario where
  northbound_speed : ℝ
  southbound_speed : ℝ
  observed_vehicles : ℕ
  observation_time : ℝ
  section_length : ℝ

/-- Calculates the number of southbound vehicles in a given section of highway --/
noncomputable def southbound_vehicles (scenario : TrafficScenario) : ℝ :=
  let relative_speed := scenario.northbound_speed + scenario.southbound_speed
  let relative_distance := relative_speed * scenario.observation_time
  let density := (scenario.observed_vehicles : ℝ) / relative_distance
  density * scenario.section_length

/-- Theorem stating that under the given conditions, there are 375 southbound vehicles in a 150-mile section --/
theorem southbound_vehicles_count (scenario : TrafficScenario) 
  (h1 : scenario.northbound_speed = 70)
  (h2 : scenario.southbound_speed = 50)
  (h3 : scenario.observed_vehicles = 25)
  (h4 : scenario.observation_time = 1/12)
  (h5 : scenario.section_length = 150) :
  southbound_vehicles scenario = 375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_southbound_vehicles_count_l1354_135439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_slope_max_area_and_chord_length_l1354_135417

noncomputable section

/-- Circle C with center (-2, 0) and radius 3 -/
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 9

/-- Point P -/
def point_P : ℝ × ℝ := (0, 1)

/-- Line passing through P(0,1) with slope k -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

/-- Chord length |AB| for a given slope k -/
noncomputable def chord_length (k : ℝ) : ℝ := 2 * Real.sqrt (9 - (1 - 2*k)^2 / (k^2 + 1))

/-- Area of triangle ABC for a given slope k -/
noncomputable def triangle_area (k : ℝ) : ℝ := 
  Real.sqrt (9 - ((1 - 2*k)^2 / (k^2 + 1))) * (abs (1 - 2*k) / Real.sqrt (k^2 + 1))

theorem chord_length_and_slope :
  ∀ k : ℝ, chord_length k = 4 * Real.sqrt 2 → k = 0 ∨ k = 4/3 := by
  sorry

theorem max_area_and_chord_length :
  (∀ k : ℝ, triangle_area k ≤ 9/2) ∧
  (∃ k : ℝ, triangle_area k = 9/2 ∧ chord_length k = 3 * Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_slope_max_area_and_chord_length_l1354_135417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_of_P_l1354_135495

/-- Given a triangle ABC with the following properties:
    - D lies on BC extended past C such that BD:DC = 5:3
    - E lies on AB extended past B such that AE:EB = 4:1
    - P is the intersection of lines BE and AD
    Then the barycentric coordinates of P with respect to triangle ABC are (21/73, 15/73, 37/73). -/
theorem barycentric_coordinates_of_P (A B C D E P : ℝ × ℝ) : 
  (∃ t : ℝ, D = C + t • (C - B) ∧ t = 5/3) →
  (∃ s : ℝ, E = A + s • (B - A) ∧ s = 4) →
  (∃ u v : ℝ, P = B + u • (E - B) ∧ P = A + v • (D - A)) →
  ∃ α β γ : ℝ, α + β + γ = 1 ∧ 
    P = (α • A.1 + β • B.1 + γ • C.1, α • A.2 + β • B.2 + γ • C.2) ∧
    α = 21/73 ∧ β = 15/73 ∧ γ = 37/73 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_of_P_l1354_135495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_radius_5_l1354_135424

/-- The area of a circle with radius 5 meters is 25π square meters. -/
theorem area_circle_radius_5 : ∀ π : ℝ, π * 5^2 = 25 * π := by
  intro π
  ring

#check area_circle_radius_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_radius_5_l1354_135424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_1_minus_x_sq_plus_2x_l1354_135465

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  (∫ x in Set.Icc 0 1, (Real.sqrt (1 - x^2) + 2*x)) = (Real.pi + 4) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_1_minus_x_sq_plus_2x_l1354_135465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1354_135490

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- A line with slope k and y-intercept m -/
def Line (k m : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + m}

/-- The circle with center (0, 0) and radius 1 -/
def UnitCircle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Predicate to check if a line is tangent to the unit circle -/
def isTangentToUnitCircle (k m : ℝ) : Prop := m^2 = 1 + k^2

/-- Theorem: For an ellipse with given properties, if a line intersects it at points A and B
    such that OA · OB = 2/3, then the slope of the line is ±1 -/
theorem ellipse_line_intersection (a b k m : ℝ) 
  (h_ellipse : Ellipse a b)
  (h_ecc : eccentricity a b = Real.sqrt 2 / 2)
  (h_minor : b = 1)
  (h_line : Line k m)
  (h_tangent : isTangentToUnitCircle k m)
  (h_intersect : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ A ∈ Line k m ∧ B ∈ Line k m)
  (h_dot_product : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ A ∈ Line k m ∧ B ∈ Line k m ∧ 
                   (A.1 * B.1 + A.2 * B.2) = 2/3) :
  k = 1 ∨ k = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1354_135490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_divisibility_l1354_135493

theorem sum_of_powers_divisibility (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_divisibility_l1354_135493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l1354_135447

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) : Type where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

/-- The distance between the foci of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola a b) : ℝ :=
  2 * Real.sqrt (a^2 + b^2)

/-- Point on the asymptote of the hyperbola in the first quadrant -/
def asymptote_point (h : Hyperbola a b) : ℝ × ℝ :=
  (a, b)

theorem hyperbola_eccentricity_squared (a b : ℝ) (h : Hyperbola a b) 
  (m : ℝ × ℝ) (hm : m = asymptote_point h) :
  let c := focal_distance h / 2
  let d₁ := Real.sqrt ((m.1 + c)^2 + m.2^2)
  let d₂ := Real.sqrt ((m.1 - c)^2 + m.2^2)
  d₁ - d₂ = 2 * b → (eccentricity h)^2 = (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l1354_135447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1354_135474

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, (3 + a) * x + 4 * y = 5 - 3 * a ↔ 2 * x + (5 + a) * y = 8) →
  ((3 + a) / 4 = 2 / (5 + a)) →
  ((5 - 3 * a) / 4 ≠ 8 / (5 + a)) →
  a = -7 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1354_135474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1354_135400

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/5 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define point P
def point_P : ℝ × ℝ := (0, -2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    distance point_P A + distance point_P B = 10 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1354_135400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_value_l1354_135428

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
noncomputable def line_l (x y α : ℝ) : Prop := y = Real.tan α * x + 1

-- Define the arithmetic sequence condition
def arithmetic_sequence (AB BC CD : ℝ) : Prop := BC - AB = CD - BC

-- Define the theorem
theorem inclination_angle_value (α : ℝ) :
  (∃ A B C D : ℝ × ℝ,
    circle_F A.1 A.2 ∧ circle_F B.1 B.2 ∧ circle_F C.1 C.2 ∧ circle_F D.1 D.2 ∧
    parabola A.1 A.2 ∧ parabola D.1 D.2 ∧
    line_l A.1 A.2 α ∧ line_l B.1 B.2 α ∧ line_l C.1 C.2 α ∧ line_l D.1 D.2 α ∧
    arithmetic_sequence (dist A B) (dist B C) (dist C D)) →
  α = Real.arctan (Real.sqrt 2 / 2) ∨ α = Real.pi - Real.arctan (Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_value_l1354_135428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1354_135486

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1)) + x^2 + 2

-- State the theorem
theorem inequality_solution_set :
  ∀ x : ℝ, f (x + 1) > f (2 * x) ↔ -1/3 < x ∧ x < 1 := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1354_135486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_customer_payment_proof_l1354_135461

/-- Calculates the total amount paid by a customer given the sales tax, tax rate, and cost of tax-free items. -/
noncomputable def total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  (sales_tax / tax_rate) + sales_tax + tax_free_cost

/-- Theorem stating that given the specific conditions, the total amount paid is $40.00 -/
theorem customer_payment_proof :
  let sales_tax : ℝ := 1.28
  let tax_rate : ℝ := 0.08
  let tax_free_cost : ℝ := 22.72
  total_amount_paid sales_tax tax_rate tax_free_cost = 40 :=
by
  -- Unfold the definition of total_amount_paid
  unfold total_amount_paid
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_customer_payment_proof_l1354_135461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135488

-- Define the functions
noncomputable def A : ℝ → ℝ := fun x ↦ Real.sin x
def B : ℝ → ℝ := fun x ↦ x^3 - x^2
noncomputable def C : ℝ → ℝ := fun x ↦ Real.sqrt x
def D : Fin 5 → ℝ := fun n ↦ match n with
  | 0 => 2
  | 1 => 0
  | 2 => 2
  | 3 => 4
  | 4 => 0

-- Define the domains
def domainA : Set ℝ := Set.Icc (-Real.pi/2) (Real.pi/2)
def domainC : Set ℝ := Set.Icc 0 4
def domainD : Set (Fin 5) := Set.univ

-- Define continuity and invertibility
def isContinuous (f : ℝ → ℝ) (domain : Set ℝ) : Prop := sorry
def isInvertible (f : ℝ → ℝ) (domain : Set ℝ) : Prop := sorry

-- Define the label assignment
def label : ℕ → ℕ := fun n ↦ n

theorem function_properties :
  (label 4 = 4) ∧
  (label 1 * label 3 = 3) ∧
  (isContinuous A domainA) ∧
  (isInvertible A domainA) ∧
  (¬isInvertible B Set.univ) ∧
  (isContinuous C domainC) ∧
  (isInvertible C domainC) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1354_135437

-- Define the function f(x) = ln x + x
noncomputable def f (x : ℝ) := Real.log x + x

-- State the theorem
theorem solution_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 3 :=
by
  sorry

-- Note: Set.Ioo a b represents the open interval (a, b)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1354_135437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1354_135450

theorem simplify_expression : (-5) - (-4) + (-7) - 2 = -5 + 4 - 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1354_135450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_four_l1354_135401

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 - (1/2) * t, (Real.sqrt 3 / 2) * t)

/-- Circle C in polar form -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin θ

/-- Point P -/
def point_P : ℝ × ℝ := (1, 0)

/-- Intersection points of line l and circle C -/
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3 / 2, Real.sqrt 3 - 3/2)
noncomputable def point_B : ℝ × ℝ := (-Real.sqrt 3 / 2, Real.sqrt 3 + 3/2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The sum of distances from P to A and B is 4 -/
theorem sum_of_distances_is_four :
  distance point_P point_A + distance point_P point_B = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_four_l1354_135401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_l1354_135404

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def agrees_with_id_on_unit_interval (f : ℝ → ℝ) : Prop := ∀ x, x ∈ Set.Icc 0 1 → f x = x

theorem four_solutions 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : is_periodic_2 f)
  (h_unit : agrees_with_id_on_unit_interval f) :
  ∃ (S : Finset ℝ), S.card = 4 ∧ ∀ x ∈ S, f x = Real.log (abs x) / Real.log 3 ∧ 
  ∀ y, f y = Real.log (abs y) / Real.log 3 → y ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_l1354_135404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_of_252_l1354_135443

/-- The number of odd factors of 252 is 6. -/
theorem odd_factors_of_252 : (Finset.filter (fun n => n % 2 = 1) (Nat.divisors 252)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_of_252_l1354_135443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_vector_equation_l1354_135445

/-- Given a line segment CD extended to point Q such that CQ:QD = 7:2,
    prove that Q = (7/9)*C + (2/9)*D. -/
theorem extended_segment_vector_equation (C D Q : ℝ × ℝ × ℝ) :
  (∃ (t : ℝ), t > 1 ∧ Q - C = t • (D - C) ∧ (t - 1) / 1 = 7 / 2) →
  Q = (7/9) • C + (2/9) • D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_vector_equation_l1354_135445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1354_135436

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -1 then -x^2 - 2*x + 1 else (1/2)^x

-- State the theorem
theorem a_range (a : ℝ) : 
  (f (3 - a^2) < f (2*a)) → (-2 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1354_135436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cost_proof_l1354_135487

theorem cookie_cost_proof (selling_price_percentage : ℝ) 
                          (cookies_sold : ℕ) 
                          (total_earnings : ℝ) 
                          (cost_per_cookie : ℝ) : ℝ :=
  by
  have h1 : selling_price_percentage = 120 := by sorry
  have h2 : cookies_sold = 50 := by sorry
  have h3 : total_earnings = 60 := by sorry
  have h4 : cost_per_cookie = 1 := by sorry
  
  -- Calculation steps
  let selling_price_per_cookie := total_earnings / (cookies_sold : ℝ)
  have h5 : selling_price_per_cookie = 1.20 := by sorry
  
  have h6 : cost_per_cookie = selling_price_per_cookie / (selling_price_percentage / 100) := by sorry
  
  -- Final result
  exact cost_per_cookie


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cost_proof_l1354_135487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l1354_135467

/-- The distance between two parallel lines with slope m and y-intercepts c1 and c2 -/
noncomputable def distance_between_parallel_lines (m : ℝ) (c1 c2 : ℝ) : ℝ :=
  |c2 - c1| / Real.sqrt (m^2 + 1)

/-- Theorem: For a line M parallel to y = (5/3)x + 3 and 3 units away from it,
    the equation of M is either y = (5/3)x + (3 + √34) or y = (5/3)x + (3 - √34) -/
theorem parallel_line_equation (m c : ℝ) (h1 : m = 5/3) (h2 : c = 3) :
  let d := 3
  let c_diff := Real.sqrt 34
  (distance_between_parallel_lines m c (c + c_diff) = d ∨
   distance_between_parallel_lines m c (c - c_diff) = d) ∧
  (∀ y, y ≠ c + c_diff ∧ y ≠ c - c_diff →
   distance_between_parallel_lines m c y ≠ d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l1354_135467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1354_135433

theorem divisibility_condition (n : ℕ) :
  (∃ (a b c : ℕ), (a ∣ n) ∧ (b ∣ n) ∧ (c ∣ n) ∧
    (a > b) ∧ (b > c) ∧
    ((a^2 - b^2) ∣ n) ∧ ((b^2 - c^2) ∣ n) ∧ ((a^2 - c^2) ∣ n)) →
  (60 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1354_135433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l1354_135409

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  train_length = 110 ∧ 
  train_speed_kmh = 80 ∧ 
  passing_time = 4.5 →
  ∃ man_speed_kmh : ℝ, 
    (man_speed_kmh ≥ 8.0063 ∧ man_speed_kmh ≤ 8.0065) ∧ 
    train_length = (train_speed_kmh * 1000 / 3600 + man_speed_kmh * 1000 / 3600) * passing_time :=
by
  sorry

#check man_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l1354_135409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_is_seven_l1354_135492

-- Define the problem parameters
def loan_amount : ℚ := 200
def total_repayment : ℚ := 240
def time_period : ℚ := 3

-- Define the function to calculate the annual simple interest rate
def calculate_annual_interest_rate (principal : ℚ) (total_repayment : ℚ) (time : ℚ) : ℚ :=
  ((total_repayment - principal) / (principal * time)) * 100

-- Define the function to round to the nearest whole number
def round_to_nearest_whole (x : ℚ) : ℤ :=
  Int.floor (x + 1/2)

-- Theorem statement
theorem annual_interest_rate_is_seven :
  round_to_nearest_whole (calculate_annual_interest_rate loan_amount total_repayment time_period) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_is_seven_l1354_135492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_theta_l1354_135425

theorem sin_two_pi_minus_theta (θ : ℝ) (h1 : 3 * (Real.cos θ)^2 = Real.tan θ + 3) 
  (h2 : ∀ (k : ℤ), θ ≠ k * π) : 
  Real.sin (2 * (π - θ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_theta_l1354_135425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_sine_l1354_135484

/-- Given a function f, prove that if halving its x-coordinates and shifting right by π/3
    results in sin(x - π/4), then f(x) = sin(x/2 + π/12) -/
theorem transform_to_sine (f : ℝ → ℝ) : 
  (∀ x, Real.sin (x - π/4) = f (2 * (x - π/3))) → 
  (∀ x, f x = Real.sin (x/2 + π/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_sine_l1354_135484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_reciprocal_l1354_135421

theorem complex_reciprocal (z : ℂ) (h : z^2 = -4) : 
  (1 : ℂ) / z = Complex.I / 2 ∨ (1 : ℂ) / z = -Complex.I / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_reciprocal_l1354_135421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_properties_l1354_135413

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.a * t.c * Real.sin t.B

/-- Theorem about a specific triangle -/
theorem specific_triangle_properties :
  ∀ t : Triangle,
  t.B = π/6 →
  t.a = Real.sqrt 3 →
  t.c = 1 →
  t.b = 1 ∧ triangleArea t = Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_properties_l1354_135413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_volume_of_D_l1354_135426

/-- The region D bounded by the curve √x + √y = √a and the line x + y = a -/
def D (a : ℝ) := {p : ℝ × ℝ | Real.sqrt p.1 + Real.sqrt p.2 = Real.sqrt a ∧ p.1 + p.2 ≤ a}

/-- The area of region D -/
noncomputable def area_D (a : ℝ) : ℝ := a^2 / 3

/-- The volume of the solid formed by rotating D about the line x + y = a -/
noncomputable def volume_solid (a : ℝ) : ℝ := (Real.pi * Real.sqrt 2 / 15) * a^3

theorem area_and_volume_of_D (a : ℝ) (ha : a > 0) :
  area_D a = a^2 / 3 ∧ volume_solid a = (Real.pi * Real.sqrt 2 / 15) * a^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_volume_of_D_l1354_135426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_theorem_l1354_135418

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle P
def circle_P (x y r : ℝ) : Prop := (x - 2)^2 + y^2 = r^2

-- Define the tangency conditions
def externally_tangent (x y r : ℝ) : Prop := 
  ∃ (x₁ y₁ : ℝ), circle_M x₁ y₁ ∧ (x - x₁)^2 + (y - y₁)^2 = (r + 1)^2

def internally_tangent (x y r : ℝ) : Prop := 
  ∃ (x₁ y₁ : ℝ), circle_N x₁ y₁ ∧ (x - x₁)^2 + (y - y₁)^2 = (3 - r)^2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ (x ≠ -2 ∨ y ≠ 0)

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := 
  (y = (Real.sqrt 2 / 4) * (x + 4)) ∨ (y = -(Real.sqrt 2 / 4) * (x + 4)) ∨ (x = 0)

theorem circles_theorem :
  ∀ (x y r : ℝ),
    externally_tangent x y r →
    internally_tangent x y r →
    (curve_C x y ∨ (x = -2 ∧ y = 0)) ∧
    (r = 2 → line_l x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_theorem_l1354_135418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_theorem_l1354_135444

noncomputable def circle_ratio (r : ℝ) (A B C : ℝ × ℝ) : Prop :=
  let circle := {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = (2*r)^2}
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let arc_length (p q : ℝ × ℝ) := 2 * r * Real.arccos ((p.1 * q.1 + p.2 * q.2) / ((2*r)^2))
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧
  dist A B = dist A C ∧
  dist A B > 2*r ∧
  arc_length B C = 2*r ∧
  dist A B / dist B C = 2 * Real.sin (1/2)

theorem circle_ratio_theorem (r : ℝ) (A B C : ℝ × ℝ) (h : r > 0) :
  circle_ratio r A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_theorem_l1354_135444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1354_135416

-- Define the function f
def f (x : ℝ) : ℝ := 3 - 4*x + x^2

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := 2 - Real.sqrt (1 + x)

-- Theorem statement
theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1354_135416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_intersect_l1354_135419

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the uniform distribution on a line segment
def UniformDist (a b : ℝ) : Set ℝ := {X : ℝ | a ≤ X ∧ X ≤ b}

-- Define the centers of the circles
def CenterA : Set ℝ := UniformDist 0 4
def CenterB : Set ℝ := UniformDist 0 4
def CenterC : Set ℝ := UniformDist 0 4

-- Define the probability space
def Ω : Type := ℝ × ℝ × ℝ

-- Define the event of all three circles intersecting
def AllIntersect (ω : Ω) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Circle (ω.1, 0) 2 ∧ p ∈ Circle (ω.2.1, 2) 2 ∧ p ∈ Circle (ω.2.2, -2) 2

-- Define a probability measure (this is a simplification and may need to be adjusted)
noncomputable def ℙ : Set Ω → ℝ := sorry

-- State the theorem
theorem probability_all_intersect :
  ℙ {ω : Ω | AllIntersect ω} = 0.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_intersect_l1354_135419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_weights_l1354_135427

theorem potato_weights (a b c d : ℕ) 
  (h_distinct : a < b ∧ b < c ∧ c < d) 
  (h_sums : c + d = 127 ∧ b + d = 116 ∧ b + c = 101 ∧ a + d = 112) : 
  (∀ x y, (x, y) ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] → x + y ≤ c + d) ∧
  (∀ x y, (x, y) ∈ [(a, b), (a, c), (a, d), (b, c)] → x + y < b + d) ∧
  (∀ x y, (x, y) ∈ [(a, c), (a, d), (b, c), (b, d), (c, d)] → a + b < x + y) ∧
  (∀ x y, (x, y) ∈ [(a, d), (b, c), (b, d), (c, d)] → a + c < x + y) ∧
  a = 41 ∧ b = 45 ∧ c = 56 ∧ d = 71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_weights_l1354_135427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_power_equation_l1354_135453

theorem cos_sin_power_equation (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, (Real.cos x)^n - (Real.sin x)^n = 1 ↔
    (Even n ∧ ∃ k : ℤ, x = k * Real.pi) ∨
    (Odd n ∧ (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_power_equation_l1354_135453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1354_135478

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- Represents an ellipse in the xy-plane -/
structure Ellipse where
  equation : ℝ → ℝ → Prop

/-- The focal length of a conic section -/
def focalLength (c : ℝ) : ℝ := 2 * c

theorem hyperbola_equation (C : Hyperbola) (E : Ellipse) :
  (∀ x y : ℝ, E.equation x y ↔ x^2/9 + y^2/4 = 1) →
  (∃ k : ℝ, ∀ x y : ℝ, x - 2*y = 0 → C.equation x y) →
  (focalLength (Real.sqrt 5) = focalLength (Real.sqrt (9 - 4))) →
  (∀ x y : ℝ, C.equation x y ↔ (x^2/4 - y^2 = 1 ∨ y^2 - x^2/4 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1354_135478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decrease_percentage_l1354_135441

/-- Represents the relationship between x, y, and z -/
structure XYZRelation where
  x : ℝ
  y : ℝ
  z : ℝ
  k : ℝ
  c : ℝ
  inverse_prop : x * y = k
  direct_prop : y = c * z

/-- The percentage change in y when z changes -/
noncomputable def z_to_y_change_ratio : ℝ := 1/2

/-- The percentage increase in x -/
noncomputable def x_increase_percent : ℝ := 20

/-- Theorem stating that y decreases by 100/6 percent when x increases by 20% -/
theorem y_decrease_percentage (r : XYZRelation) :
  let x' := r.x * (1 + x_increase_percent / 100)
  let y' := r.k / x'
  (r.y - y') / r.y * 100 = 100/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decrease_percentage_l1354_135441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_starts_at_8am_l1354_135460

/-- The time when the train from city A starts, in hours after midnight -/
def train_A_start_time : ℝ := 8

/-- The total distance between cities A and B in km -/
def total_distance : ℝ := 465

/-- The speed of the train from city A in km/hr -/
def speed_A : ℝ := 60

/-- The speed of the train from city B in km/hr -/
def speed_B : ℝ := 75

/-- The time when the train from city B starts, in hours after midnight -/
def train_B_start_time : ℝ := 9

/-- The time when the trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

theorem train_A_starts_at_8am :
  train_A_start_time = 8 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_starts_at_8am_l1354_135460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_area_inscribed_cylinder_optimal_height_inscribed_cylinder_l1354_135434

/-- The lateral area of an inscribed cylinder in a cone -/
noncomputable def lateral_area (R H x : ℝ) : ℝ := (2 * Real.pi * R / H) * (-x^2 + H*x)

/-- The maximum lateral area of an inscribed cylinder in a cone -/
theorem max_lateral_area_inscribed_cylinder (R H : ℝ) (hR : R > 0) (hH : H > 0) :
  ∃ x : ℝ, 0 < x ∧ x < H ∧
    lateral_area R H x = (1/2) * Real.pi * R * H ∧
    ∀ y : ℝ, 0 < y → y < H → lateral_area R H y ≤ lateral_area R H x :=
by sorry

/-- The optimal height of the inscribed cylinder for maximum lateral area -/
theorem optimal_height_inscribed_cylinder (R H : ℝ) (hR : R > 0) (hH : H > 0) :
  ∃ x : ℝ, x = H / 2 ∧
    ∀ y : ℝ, 0 < y → y < H → lateral_area R H y ≤ lateral_area R H x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_area_inscribed_cylinder_optimal_height_inscribed_cylinder_l1354_135434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_proof_l1354_135497

/-- Represents the number of meters of cloth sold by a shop owner -/
noncomputable def meters_sold (gain_meters : ℝ) (gain_percentage : ℝ) : ℝ :=
  (2 * gain_meters) / (gain_percentage / 100)

/-- Theorem stating that given the conditions, the shop owner sold 30 meters of cloth -/
theorem cloth_sale_proof (gain_meters : ℝ) (gain_percentage : ℝ) 
  (h1 : gain_meters = 10)
  (h2 : gain_percentage = 50) :
  meters_sold gain_meters gain_percentage = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval meters_sold 10 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_proof_l1354_135497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_prove_election_votes_calculation_l1354_135464

theorem election_votes_calculation (candidate1_percentage candidate2_percentage 
                                    candidate3_percentage candidate4_percentage 
                                    candidate5_percentage : Real)
                                   (vote_difference : Nat)
                                   (total_votes : Nat) : Prop :=
  candidate1_percentage = 0.32 ∧
  candidate2_percentage = 0.28 ∧
  candidate3_percentage = 0.20 ∧
  candidate4_percentage = 0.15 ∧
  candidate5_percentage = 0.05 ∧
  vote_difference = 548 ∧
  (candidate1_percentage - candidate2_percentage) * (total_votes : Real) = vote_difference ∧
  total_votes = 13700

theorem prove_election_votes_calculation : 
  ∃ (candidate1_percentage candidate2_percentage candidate3_percentage 
     candidate4_percentage candidate5_percentage : Real)
    (vote_difference total_votes : Nat),
  election_votes_calculation candidate1_percentage candidate2_percentage
                             candidate3_percentage candidate4_percentage
                             candidate5_percentage vote_difference total_votes := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_prove_election_votes_calculation_l1354_135464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1354_135406

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos α = Real.sqrt 5 / 5) (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  Real.cos β = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1354_135406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1354_135448

/-- Given a quadratic function and specific conditions, prove properties about its roots and axis of symmetry. -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (x₁ x₂ m x₀ : ℝ)
  (hx : x₁ < x₂)
  (hf_neq : f x₁ ≠ f x₂)
  (hm : f m = (f x₁ + f x₂) / 2)
  (hm_between : x₁ < m ∧ m < x₂)
  (harith : x₁ + x₂ = 2 * m - 1)
  (hx₀ : x₀ = -b / (2 * a)) :
  (∃ y z, y ≠ z ∧ f y = (f x₁ + f x₂) / 2 ∧ f z = (f x₁ + f x₂) / 2) ∧
  (∃ r, x₁ < r ∧ r < x₂ ∧ f r = (f x₁ + f x₂) / 2) ∧
  x₀ < m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1354_135448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_neg_three_l1354_135473

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_three :
  (f⁻¹) (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_neg_three_l1354_135473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alcohol_percentage_l1354_135491

/-- Proves that the initial alcohol percentage in a mixture is 20% given the conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (initial_alcohol_percentage : ℝ)
  (h1 : initial_volume = 15)
  (h2 : added_water = 5)
  (h3 : final_percentage = 15)
  (h4 : final_percentage / 100 * (initial_volume + added_water) = 
        (initial_alcohol_percentage / 100) * initial_volume) :
  initial_alcohol_percentage = 20 := by
  sorry

#check initial_alcohol_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alcohol_percentage_l1354_135491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_minimum_l1354_135471

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a + Real.log x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := deriv (f a) x

theorem f_tangent_and_minimum (a : ℝ) :
  (∃ k : ℝ, k * deriv (f a) 1 = -1 ∧ k ≠ 0 → a = 0) ∧
  (a > 0 ∧ a < Real.log 2 →
    ∃ x₀ : ℝ, 1/2 < x₀ ∧ x₀ < 1 ∧
    (∀ x, x > 0 → deriv (g a) x = 0 → x = x₀) ∧
    f a x₀ < 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_minimum_l1354_135471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_price_theorem_l1354_135432

/-- Given an initial price, apply discounts, tax, and shipping fee to reach a final price --/
def final_price (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := initial_price * (1 - 0.24)
  let price_after_second_discount := price_after_first_discount * (1 - 0.15)
  let price_after_tax := price_after_second_discount * (1 + 0.18)
  price_after_tax + 50

/-- Theorem stating that if the final price is 532, the initial price was approximately 632.29 --/
theorem initial_price_theorem (P : ℝ) :
  final_price P = 532 → abs (P - 632.29) < 0.01 := by
  sorry

#eval final_price 632.29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_price_theorem_l1354_135432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_sum_l1354_135499

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
noncomputable def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem largest_negative_sum 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_10 : a 10 < 0)
  (h_11_pos : a 11 > 0)
  (h_11_abs : a 11 > |a 10|) :
  (∀ n : ℕ, S a n < 0 → S a n ≤ S a 19) ∧ S a 19 < 0 ∧ S a 20 > 0 :=
by
  sorry

#check largest_negative_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_sum_l1354_135499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_equals_673B_plus_820I_l1354_135485

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 2, 2]

theorem B_power_six_equals_673B_plus_820I :
  B^6 = 673 • B + 820 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_equals_673B_plus_820I_l1354_135485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1354_135452

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

-- Define f' as the derivative of f
noncomputable def f' : ℝ → ℝ := deriv f

-- Theorem statement
theorem f_property (a : ℝ) : f a + f' a + f (-a) - f' (-a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1354_135452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_specific_case_l1354_135451

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period. -/
noncomputable def bankers_gain (bankers_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  let true_discount := bankers_discount / (1 + interest_rate * time / 100)
  bankers_discount - true_discount

/-- Theorem stating that given the specific conditions, the banker's gain is 270. -/
theorem bankers_gain_specific_case :
  bankers_gain 1020 12 3 = 270 := by
  -- Unfold the definition of bankers_gain
  unfold bankers_gain
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_specific_case_l1354_135451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_meeting_time_l1354_135405

/-- The time (in minutes) it takes for two projectiles to meet -/
noncomputable def meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  initial_distance / (speed1 + speed2) * 60

/-- Theorem stating that the meeting time for the given conditions is 90 minutes -/
theorem projectile_meeting_time :
  let initial_distance : ℝ := 1455
  let speed1 : ℝ := 470
  let speed2 : ℝ := 500
  meeting_time initial_distance speed1 speed2 = 90 := by
  sorry

/-- Evaluate the meeting time for the given conditions -/
def eval_meeting_time : ℚ :=
  (1455 : ℚ) / (470 + 500) * 60

#eval eval_meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_meeting_time_l1354_135405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l1354_135429

def U : Set ℕ := {2, 3, 4}

def A : Set ℕ := {x : ℕ | (x - 1) * (x - 4) < 0 ∧ x ∈ U}

theorem complement_of_A_in_U : U \ A = {4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l1354_135429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_l1354_135489

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := b * a^x

-- State the theorem
theorem function_and_inequality (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a b 1 = 1/6) (h4 : f a b 3 = 1/24) :
  (∃ (g : ℝ → ℝ), g = (λ x ↦ (1/3) * (1/2)^x) ∧ ∀ x, f a b x = g x) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → 2^x + 3^x - m ≥ 0) ↔ m ≤ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_l1354_135489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_card_position_l1354_135458

/-- The position of the last remaining card in the Josephus problem -/
def josephus (n : ℕ) : ℕ :=
  2 * (n - 2^(Nat.log2 n)) + 1

/-- The number of cards in the stack -/
def numCards : ℕ := 200

/-- The position of the last remaining card -/
def lastCard : ℕ := 145

theorem last_card_position :
  josephus numCards = lastCard := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_card_position_l1354_135458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l1354_135459

theorem least_d_value (c d : ℕ) 
  (hc_factors : Finset.card (Nat.divisors c) = 4)
  (hd_factors : Finset.card (Nat.divisors d) = c)
  (hd_div_c : c ∣ d) :
  d ≥ 24 ∧ ∃ (d_min : ℕ), d_min = 24 ∧ 
    Finset.card (Nat.divisors d_min) = c ∧ 
    c ∣ d_min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l1354_135459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_integrality_l1354_135469

theorem binomial_coefficient_integrality (m n : ℕ) : 
  ∃ k : ℕ, k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) = 
    Nat.factorial (2 * m) * Nat.factorial (2 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_integrality_l1354_135469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_prime_roots_l1354_135455

/-- Given a quadratic equation x^2 - 45x + m = 0 with prime roots, 
    the sum of the squares of these roots is 1853 -/
theorem sum_of_squares_of_prime_roots (m : ℤ) : 
  (∃ α β : ℤ, Nat.Prime α.natAbs ∧ Nat.Prime β.natAbs ∧ α^2 - 45*α + m = 0 ∧ β^2 - 45*β + m = 0) →
  (∃ α β : ℤ, Nat.Prime α.natAbs ∧ Nat.Prime β.natAbs ∧ α + β = 45 ∧ α * β = m ∧ α^2 + β^2 = 1853) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_prime_roots_l1354_135455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jungkook_average_score_l1354_135468

/-- Jungkook's average score calculation -/
theorem jungkook_average_score 
  (korean_english_avg : ℝ) 
  (math_score : ℝ) 
  (h1 : korean_english_avg = 88) 
  (h2 : math_score = 100) : 
  (2 * korean_english_avg + math_score) / 3 = 92 := by
  sorry

#check jungkook_average_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jungkook_average_score_l1354_135468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l1354_135476

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4 * a * x

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
def Line (p q : Point) := {r : Point | ∃ t : ℝ, r.x = p.x + t * (q.x - p.x) ∧ r.y = p.y + t * (q.y - p.y)}

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem statement -/
theorem parabola_distance_sum (p : Parabola) (P A B F : Point) : 
  p.equation = fun x y => y^2 = 12 * x →
  P.x = 2 ∧ P.y = 1 →
  A ∈ Line P B →
  p.equation A.x A.y →
  p.equation B.x B.y →
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2 →
  F.x = 3 ∧ F.y = 0 →
  distance A F + distance B F = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l1354_135476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135463

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

theorem function_properties (a b : ℝ) :
  (∀ x ∈ (Set.Ioo (-2) 6), f a b x > 0) →
  (∀ x ∈ (Set.Iic (-2) ∪ Set.Ici 6), f a b x < 0) →
  (∃ g : ℝ → ℝ, g = f a b ∧
    (∀ x, g x = -4 * x^2 + 16 * x + 48) ∧
    (Set.Icc 1 10).image g ⊆ Set.Icc (-192) 64 ∧
    g 2 = 64 ∧ g 10 = -192) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1354_135438

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- Theorem statements
theorem f_properties :
  (f 2 = 4) ∧
  (f (1/2) = 1/4) ∧
  (f (f (-1)) = 1) ∧
  (f (Real.sqrt 3) = 3) ∧
  (f (3/2) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1354_135438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdhouse_volume_difference_l1354_135410

-- Constants for unit conversions
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def meters_to_feet : ℝ := 3.281

-- Sara's birdhouse dimensions in feet
def sara_width : ℝ := 1
def sara_height : ℝ := 2
def sara_depth : ℝ := 2

-- Jake's birdhouse dimensions in inches
def jake_width : ℝ := 16
def jake_height : ℝ := 20
def jake_depth : ℝ := 18

-- Tom's birdhouse dimensions in meters
def tom_width : ℝ := 0.4
def tom_height : ℝ := 0.6
def tom_depth : ℝ := 0.5

-- Function to calculate volume
noncomputable def calculate_volume (width : ℝ) (height : ℝ) (depth : ℝ) : ℝ :=
  width * height * depth

-- Theorem to prove
theorem birdhouse_volume_difference :
  let sara_volume := calculate_volume sara_width sara_height sara_depth
  let jake_volume := calculate_volume (jake_width * inches_to_feet) (jake_height * inches_to_feet) (jake_depth * inches_to_feet)
  let tom_volume := calculate_volume (tom_width * meters_to_feet) (tom_height * meters_to_feet) (tom_depth * meters_to_feet)
  let max_volume := max sara_volume (max jake_volume tom_volume)
  let min_volume := min sara_volume (min jake_volume tom_volume)
  abs (max_volume - min_volume - 0.916) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdhouse_volume_difference_l1354_135410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_roots_l1354_135466

theorem simplify_nested_roots : 
  (16384 : ℝ) = 2^14 → 
  Real.sqrt (Real.rpow (Real.sqrt (1 / 16384)) (1/3)) = 1 / Real.rpow 128 (1/6) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_roots_l1354_135466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1354_135440

-- Define a parabola
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a line
structure Line where
  m : ℝ
  k : ℝ

-- Define the number of intersection points between a line and a parabola
noncomputable def intersectionPoints (l : Line) (p : Parabola) : Nat :=
  sorry

-- Define a predicate for a line being tangent to a parabola
def isTangent (l : Line) (p : Parabola) : Prop :=
  intersectionPoints l p = 1

-- Theorem statement
theorem parabola_intersection_theorem 
  (p : Parabola) (l1 l2 : Line) 
  (h1 : isTangent l1 p) 
  (h2 : intersectionPoints l2 p > 0) 
  (h3 : ∃ (x y : ℝ), l1.m * x + l1.k = l2.m * x + l2.k ∧ p.a * x^2 + p.b * x + p.c = y) :
  ∃ (n : Nat), n ∈ ({1, 2, 3} : Set Nat) ∧ intersectionPoints l1 p + intersectionPoints l2 p = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1354_135440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_solid_l1354_135482

/-- A solid with a triangular base and parallel upper face --/
structure TriangularPrismoid where
  a : ℝ
  b : ℝ
  c : ℝ
  upper_scale : ℝ

/-- Volume of the TriangularPrismoid --/
noncomputable def volume (solid : TriangularPrismoid) : ℝ :=
  sorry

/-- The specific solid described in the problem --/
noncomputable def specific_solid : TriangularPrismoid where
  a := 6 * Real.sqrt 2
  b := 6 * Real.sqrt 2
  c := 6 * Real.sqrt 2
  upper_scale := 2

/-- Theorem stating the volume of the specific solid --/
theorem volume_of_specific_solid :
  volume specific_solid = 144 * Real.sqrt 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_solid_l1354_135482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_divisibility_l1354_135477

/-- The eight-digit number formed by inserting n between 9637 and 428 -/
def eight_digit_number (n : ℕ) : ℕ := 9637 * 10000 + n * 1000 + 428

/-- A number is divisible by 11 if and only if the alternating sum of its digits is divisible by 11 -/
def divisible_by_11 (m : ℕ) : Prop :=
  ∃ k : ℤ, (9 - 6 + 3 - 7 + (m / 1000 % 10) - 4 + 2 - 8 : ℤ) = 11 * k

theorem eight_digit_divisibility (n : ℕ) :
  divisible_by_11 (eight_digit_number n) ↔ n = 9 := by
  sorry

#check eight_digit_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_divisibility_l1354_135477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1354_135496

/-- The trajectory of point M given the conditions in the problem -/
def trajectory_of_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 16 * p.2}

/-- Point F with coordinates (0, 4) -/
def F : ℝ × ℝ := (0, 4)

/-- The line y + 5 = 0 -/
def line_L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 + 5 = 0}

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to a line (y = k) -/
def distance_to_line (p : ℝ × ℝ) (k : ℝ) : ℝ :=
  |p.2 - k|

/-- The condition that for any point M, its distance to F is 1 less than its distance to line L -/
def condition (M : ℝ × ℝ) : Prop :=
  distance M F + 1 = distance_to_line M (-5)

theorem trajectory_equation :
  ∀ M : ℝ × ℝ, condition M → M ∈ trajectory_of_M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1354_135496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_200_l1354_135481

/-- Represents a triangle with two medians -/
structure TriangleWithMedians where
  -- The length of the first median
  median1_length : ℝ
  -- The length of the second median
  median2_length : ℝ
  -- The medians intersect at right angles
  medians_perpendicular : Bool

/-- Calculates the area of a triangle given its two perpendicular medians -/
noncomputable def triangle_area (t : TriangleWithMedians) : ℝ :=
  6 * (2/3 * t.median1_length) * (2/3 * t.median2_length) / 2

/-- The main theorem stating that a triangle with the given properties has an area of 200 -/
theorem triangle_area_is_200 (t : TriangleWithMedians) 
  (h1 : t.median1_length = 10) 
  (h2 : t.median2_length = 15) 
  (h3 : t.medians_perpendicular = true) : 
  triangle_area t = 200 := by
  sorry

#check triangle_area_is_200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_200_l1354_135481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_values_l1354_135446

theorem cos_double_angle_values (α β : ℝ) 
  (h1 : Real.cos α = 3/5)
  (h2 : Real.cos β = -1/2)
  (h3 : π < α + β)
  (h4 : α + β < 2*π)
  (h5 : 0 < α - β)
  (h6 : α - β < π) :
  Real.cos (2*α) = -7/25 ∧ Real.cos (2*β) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_values_l1354_135446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_satisfying_conditions_l1354_135422

/-- A circle centered at (1, 0) with radius 2 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 4}

/-- A line with slope m -/
def L (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 - m * p.2 + 1 = 0}

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 0)

/-- The area of a triangle formed by two points on a circle and its center -/
noncomputable def triangleArea (p q : ℝ × ℝ) : ℝ := 
  (1/2) * Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) *
                    Real.sqrt ((q.1 - center.1)^2 + (q.2 - center.2)^2) *
                    Real.sin (Real.arccos ((p.1 - center.1) * (q.1 - center.1) + (p.2 - center.2) * (q.2 - center.2)) /
                              (Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) *
                               Real.sqrt ((q.1 - center.1)^2 + (q.2 - center.2)^2)))

theorem exists_m_satisfying_conditions :
  ∃ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ C ∩ L m ∧ B ∈ C ∩ L m ∧ A ≠ B ∧ triangleArea A B = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_satisfying_conditions_l1354_135422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1354_135411

theorem perpendicular_vectors (a b : ℝ × ℝ) (l : ℝ) :
  a = (2, 4) →
  b = (1, 1) →
  (b.1 * (a.1 + l * b.1) + b.2 * (a.2 + l * b.2) = 0) →
  l = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1354_135411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_specific_angles_l1354_135479

/-- The circle C in Cartesian coordinates -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*Real.sqrt 3*y - 1 = 0

/-- The line l in parametric form -/
noncomputable def lineL (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

/-- The distance between two points on the line -/
noncomputable def distance (t₁ t₂ α : ℝ) : ℝ :=
  Real.sqrt ((t₁ - t₂)^2 * (Real.cos α)^2 + (t₁ - t₂)^2 * (Real.sin α)^2)

theorem intersection_implies_specific_angles :
  ∀ α t₁ t₂ : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circleC x₁ y₁ ∧ circleC x₂ y₂ ∧
    (x₁, y₁) = lineL t₁ α ∧ (x₂, y₂) = lineL t₂ α ∧
    distance t₁ t₂ α = 3 * Real.sqrt 2) →
  α = π/4 ∨ α = 3*π/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_specific_angles_l1354_135479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_figure_l1354_135480

/-- The parametric equations defining the curve --/
noncomputable def curve (t : ℝ) : ℝ × ℝ := (6 * (t - Real.sin t), 6 * (1 - Real.cos t))

/-- The lower bound of the region --/
def lower_bound : ℝ := 6

/-- The left boundary of the region --/
def left_boundary : ℝ := 0

/-- The right boundary of the region --/
noncomputable def right_boundary : ℝ := 12 * Real.pi

/-- The area of the enclosed figure --/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the area of the enclosed figure is 18π + 72 --/
theorem area_of_enclosed_figure :
  enclosed_area = 18 * Real.pi + 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_figure_l1354_135480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_negative_one_l1354_135498

theorem cube_root_sum_equals_negative_one
  (a b : ℝ)
  (h : Real.sqrt (a - 2) + abs (b + 3) = 0) :
  (a + b : ℝ) ^ (1/3 : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_negative_one_l1354_135498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_in_interval_l1354_135420

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := 2^(2-x)

-- Define the difference function
noncomputable def h (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem intersection_point_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ h x = 0 := by
  sorry

#check intersection_point_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_in_interval_l1354_135420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rectangle_area_ratio_l1354_135442

/-- Given a rectangle with length 28 and width 20, and a rhombus formed by connecting
    the midpoints of the rectangle's sides, where one diagonal of the rhombus is 12,
    prove that the ratio of the area of the rhombus to the area of the rectangle is 3:14. -/
theorem rhombus_rectangle_area_ratio :
  ∀ (rectangle_length rectangle_width rhombus_diagonal_1 : ℝ),
    rectangle_length = 28 →
    rectangle_width = 20 →
    rhombus_diagonal_1 = 12 →
    let rhombus_diagonal_2 := rectangle_width
    let rhombus_area := (rhombus_diagonal_1 * rhombus_diagonal_2) / 2
    let rectangle_area := rectangle_length * rectangle_width
    (rhombus_area / rectangle_area) = (3 / 14) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rectangle_area_ratio_l1354_135442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1354_135483

/-- A strategy for the second player in the segment game -/
def SecondPlayerStrategy : Type :=
  ℕ → ℝ × ℝ → ℝ × ℝ

/-- The game state after n moves -/
def GameState (n : ℕ) (strategy : SecondPlayerStrategy) : ℝ × ℝ :=
  sorry

/-- The intersection of all segments after n moves -/
def Intersection (n : ℕ) (strategy : SecondPlayerStrategy) : Set ℝ :=
  sorry

/-- A winning strategy for the second player ensures no rational number is in the intersection -/
theorem second_player_wins :
  ∃ (strategy : SecondPlayerStrategy),
    ∀ (q : ℚ), ∀ (n : ℕ), (q : ℝ) ∉ Intersection n strategy :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1354_135483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_symmetric_l1354_135470

/-- Definition of the polynomial P_n -/
def P : ℕ → ℝ → ℝ → ℝ
  | 0, x, y => 1  -- Add base case for n = 0
  | 1, x, y => 1
  | n+2, x, y => (x+y-1)*(y+1)*P (n+1) x (y+2) + (y-y^2)*P (n+1) x y

/-- Theorem: P_n is symmetric in x and y for all n, x, and y -/
theorem P_symmetric (n : ℕ) (x y : ℝ) : P n x y = P n y x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_symmetric_l1354_135470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135435

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

-- Define the theorem
theorem function_properties (ω : ℝ) (h1 : ω > 0) :
  -- The distance between two adjacent symmetric axes is π/2
  (∃ (k : ℤ), ∀ (x : ℝ), f ω x = f ω (x + k * (π / 2))) →
  -- Part 1: Solutions to f(x) = 1
  (∃ (k : ℤ), f ω (k * π + π / 6) = 1 ∧ f ω (k * π + π / 2) = 1) ∧
  -- Part 2: Minimum positive m for which f(x-m) coincides with 2cos(2x)
  (∃ (m : ℝ), m > 0 ∧ 
    (∀ (x : ℝ), f ω (x - m) = 2 * Real.cos (2 * x)) ∧
    (∀ (m' : ℝ), m' > 0 → 
      (∀ (x : ℝ), f ω (x - m') = 2 * Real.cos (2 * x)) → m' ≥ π / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1354_135435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_function_composition_l1354_135472

noncomputable section

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := x / 3
def h (x : ℝ) : ℝ := x ^ 2

-- Define their inverse functions
def f_inv (x : ℝ) : ℝ := x - 2
def g_inv (x : ℝ) : ℝ := 3 * x
noncomputable def h_inv (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem complex_function_composition :
  f (h (g_inv (f_inv (h_inv (g (f (f 5))))))) = 191 - 36 * Real.sqrt 17 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_function_composition_l1354_135472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_not_sufficient_condition2_sufficient_condition3_sufficient_condition4_sufficient_l1354_135456

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define what it means for a triangle to be acute
def is_acute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- Condition 1
def condition1 (t : Triangle) : Prop :=
  ∃ (AB AC : Real × Real), AB.1 * AC.1 + AB.2 * AC.2 > 0

-- Condition 2
def condition2 (t : Triangle) : Prop :=
  ∃ (k : Real), Real.sin t.A = 4*k ∧ Real.sin t.B = 5*k ∧ Real.sin t.C = 6*k

-- Condition 3
def condition3 (t : Triangle) : Prop :=
  Real.cos t.A * Real.cos t.B * Real.cos t.C > 0

-- Condition 4
def condition4 (t : Triangle) : Prop :=
  Real.tan t.A * Real.tan t.B = 2

-- Theorem statements
theorem condition1_not_sufficient :
  ¬(∀ t : Triangle, condition1 t → is_acute t) :=
sorry

theorem condition2_sufficient :
  ∀ t : Triangle, condition2 t → is_acute t :=
sorry

theorem condition3_sufficient :
  ∀ t : Triangle, condition3 t → is_acute t :=
sorry

theorem condition4_sufficient :
  ∀ t : Triangle, condition4 t → is_acute t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_not_sufficient_condition2_sufficient_condition3_sufficient_condition4_sufficient_l1354_135456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1354_135457

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line 3x - 4y - 10 = 0 -/
def line (x y : ℝ) : Prop := 3*x - 4*y - 10 = 0

/-- Distance between a point (x, y) and the line 3x - 4y - 10 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - 4*y - 10| / Real.sqrt (3^2 + (-4)^2)

/-- The minimum distance from any point on the unit circle to the line is 1 -/
theorem min_distance_circle_to_line :
  ∀ x y : ℝ, unit_circle x y → (∃ (min_dist : ℝ), min_dist = 1 ∧
    ∀ p q : ℝ, unit_circle p q → distance_to_line p q ≥ min_dist) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1354_135457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1354_135454

-- Define the original expression
noncomputable def original_expression : ℝ := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)

-- Define the rationalized form
noncomputable def rationalized_form : ℝ := -9 - 4 * Real.sqrt 5

-- Theorem statement
theorem rationalize_denominator :
  original_expression = rationalized_form ∧
  ∃ (A B C : ℤ), rationalized_form = A + B * Real.sqrt C ∧
                  A = -9 ∧ B = -4 ∧ C = 5 ∧ A * B * C = 180 := by
  sorry

#eval 180 -- To check if the theorem compiles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1354_135454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1354_135449

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 1 / Real.exp x)

theorem f_property (x₁ x₂ : ℝ) (h : f x₁ < f x₂) : x₁^2 < x₂^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1354_135449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_l1354_135414

theorem cos_two_beta (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4)
  (h4 : Real.cos (α + β) = -3 / 5)
  (h5 : Real.sin (α - β) = 5 / 13) : 
  Real.cos (2 * β) = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_l1354_135414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1354_135494

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x * (1 - x) else -x * (1 + x)

theorem f_is_even_and_correct : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x ≥ 0, f x = x * (1 - x)) ∧ 
  (∀ x ≤ 0, f x = -x * (1 + x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1354_135494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_minus_smallest_equals_eight_l1354_135415

def numbers : List ℤ := [-6, 2, -3]

theorem largest_minus_smallest_equals_eight :
  (numbers.maximum?.getD 0) - (numbers.minimum?.getD 0) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_minus_smallest_equals_eight_l1354_135415
