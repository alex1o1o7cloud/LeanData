import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l817_81771

open Real Set

-- Define the function f(x) = cos²x + 3sin²x - 2
noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + 3 * sin x ^ 2 - 2

-- Define the interval (-10, 10)
def I : Set ℝ := Ioo (-10) 10

-- Theorem statement
theorem solution_count : ∃ (S : Finset ℝ), S.card = 12 ∧ (∀ x ∈ S, x ∈ I ∧ f x = 0) ∧ (∀ x ∈ I, f x = 0 → x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l817_81771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_polyhedron_properties_l817_81778

/-- A convex polyhedron formed by intersecting a sphere with lines parallel to tetrahedron edges -/
structure TetrahedronSpherePolyhedron where
  -- Sphere radius
  radius : ℝ
  -- Assumption that radius is √2
  radius_is_sqrt_2 : radius = Real.sqrt 2
  -- Number of lines drawn through the center
  num_lines : ℕ
  -- Assumption that there are 6 lines
  num_lines_is_6 : num_lines = 6
  -- Assumption that lines are parallel to edges of a regular tetrahedron
  lines_parallel_to_tetrahedron : True

/-- The volume of the TetrahedronSpherePolyhedron -/
noncomputable def volume (p : TetrahedronSpherePolyhedron) : ℝ := 12 + 4 * Real.sqrt 3

/-- The surface area of the TetrahedronSpherePolyhedron -/
def surface_area (p : TetrahedronSpherePolyhedron) : ℝ := 4

/-- Theorem stating the volume and surface area of the TetrahedronSpherePolyhedron -/
theorem tetrahedron_sphere_polyhedron_properties (p : TetrahedronSpherePolyhedron) :
  volume p = 12 + 4 * Real.sqrt 3 ∧ surface_area p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_polyhedron_properties_l817_81778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l817_81775

def S : Set ℝ := {x | (x - 2) * (x - 3) ≥ 0}
def T : Set ℝ := {x | x > 0}

theorem intersection_S_T : S ∩ T = Set.Ioo 0 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l817_81775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l817_81715

def A (a : ℝ) := {x : ℝ | x^2 + x + a ≤ 0}
def B (a : ℝ) := {x : ℝ | x^2 - x + 2*a*x - 1 < 0}
def C (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 4*a - 9}

theorem range_of_a :
  ∀ a : ℝ, (Set.Nonempty (A a) ∨ Set.Nonempty (B a) ∨ Set.Nonempty (C a)) ↔ 
  (a < 5/8 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l817_81715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_in_unit_interval_l817_81767

theorem three_numbers_in_unit_interval (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ < 1) 
  (h₂ : 0 ≤ x₂ ∧ x₂ < 1) 
  (h₃ : 0 ≤ x₃ ∧ x₃ < 1) : 
  ∃ (a b : ℝ), a ∈ ({x₁, x₂, x₃} : Set ℝ) ∧ b ∈ ({x₁, x₂, x₃} : Set ℝ) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_in_unit_interval_l817_81767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l817_81706

-- Define the function g on the domain {a, b, c}
def g (x : ℝ) : ℝ := sorry

-- Define the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem transformed_triangle_area
  (a b c : ℝ) (h : triangleArea (a, g a) (b, g b) (c, g c) = 50) :
  triangleArea (a/3, 3 * g a) (b/3, 3 * g b) (c/3, 3 * g c) = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l817_81706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_square_is_512_l817_81743

/-- Two circles are externally tangent -/
def are_externally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A smaller circle is internally tangent to a larger circle -/
def is_internally_tangent (r R : ℝ) : Prop := sorry

/-- A line is a common external tangent to two smaller circles and a chord of the larger circle -/
def is_common_external_tangent_and_chord (r₁ r₂ R : ℝ) : Prop := sorry

/-- The square of the length of the chord -/
def chord_length_square (r₁ r₂ R : ℝ) : ℝ := sorry

/-- Given three circles where:
    - Two smaller circles have radii 4 and 8 respectively
    - The two smaller circles are externally tangent to each other
    - Both smaller circles are internally tangent to a larger circle with radius 12
    - A common external tangent to the two smaller circles is a chord of the larger circle
    This theorem states that the square of the chord's length is 512 -/
theorem chord_length_square_is_512 (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : R = 12)
    (h₄ : are_externally_tangent r₁ r₂)
    (h₅ : is_internally_tangent r₁ R)
    (h₆ : is_internally_tangent r₂ R)
    (h₇ : is_common_external_tangent_and_chord r₁ r₂ R) :
  chord_length_square r₁ r₂ R = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_square_is_512_l817_81743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_three_l817_81723

/-- Calculates the height of a door given room dimensions and whitewashing costs -/
noncomputable def door_height (room_length room_width room_height : ℝ)
                (whitewash_cost_per_sqft : ℝ)
                (door_width : ℝ)
                (window_width window_height : ℝ)
                (num_windows : ℕ)
                (total_cost : ℝ) : ℝ :=
  let room_perimeter := 2 * (room_length + room_width)
  let wall_area := room_perimeter * room_height
  let window_area := (num_windows : ℝ) * window_width * window_height
  let whitewashed_area := total_cost / whitewash_cost_per_sqft
  (wall_area - whitewashed_area - window_area) / door_width

/-- Theorem stating that the door height is 3 feet given the problem parameters -/
theorem door_height_is_three :
  door_height 25 15 12 8 6 4 3 3 7248 = 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_three_l817_81723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_l817_81735

-- Define a non-obtuse triangle ABC
structure NonObtuseTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  nonObtuse : A ≤ π/2 ∧ B ≤ π/2 ∧ C ≤ π/2
  angleSum : A + B + C = π

-- State the theorem
theorem angle_B_is_pi_over_three (triangle : NonObtuseTriangle) 
  (h : 2 * triangle.b * Real.sin triangle.A = Real.sqrt 3 * triangle.a) : 
  triangle.B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_l817_81735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l817_81798

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 is √(1 + b^2/a^2) -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The hyperbola x^2/4 - y^2/2 = 1 has an eccentricity of √6/2 -/
theorem hyperbola_eccentricity : eccentricity 2 (Real.sqrt 2) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l817_81798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l817_81769

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def binomial_expansion_constant_term : ℚ := 5 / 3

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 = binomial_expansion_constant_term →
  a 3 * a 7 = (25 : ℚ) / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l817_81769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l817_81763

-- Define the points A and B
def A : ℝ × ℝ := (-1, -5)
def B : ℝ × ℝ := (3, -2)

-- Define the slope of line AB
noncomputable def slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the angle of inclination of line AB
noncomputable def angle_AB : ℝ := Real.arctan slope_AB

-- Define the angle of inclination of line l
noncomputable def angle_l : ℝ := 2 * angle_AB

-- State the theorem
theorem slope_of_line_l : 
  Real.tan angle_l = 24 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l817_81763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_custom_sequence_l817_81728

def custom_sequence (n : ℕ) : ℕ := n.factorial + n

def sum_custom_sequence : ℕ := (List.range 12).map (λ i => custom_sequence (i + 1)) |>.sum

theorem units_digit_of_sum_custom_sequence : sum_custom_sequence % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_custom_sequence_l817_81728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l817_81770

/-- The number of days required for a group to complete a work together, given their individual completion times. -/
def daysRequired (a b c d : ℕ) : ℕ :=
  (((10080 : ℚ) / 2582).ceil).toNat

/-- Theorem stating that given the individual work rates, the number of days required to complete the work together is as calculated. -/
theorem work_completion_time (a b c d : ℕ) 
  (ha : a = 15) (hb : b = 14) (hc : c = 16) (hd : d = 18) : 
  daysRequired a b c d = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l817_81770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_approx_29_l817_81752

/-- Calculates the average miles per gallon for a round trip with different fuel efficiencies for each leg -/
noncomputable def average_mpg (total_distance : ℝ) (distance1 : ℝ) (mpg1 : ℝ) (distance2 : ℝ) (mpg2 : ℝ) : ℝ :=
  total_distance / ((distance1 / mpg1) + (distance2 / mpg2))

/-- Theorem stating that the average mpg for the given round trip is approximately 29 -/
theorem round_trip_mpg_approx_29 :
  let total_distance := (300 : ℝ)
  let distance1 := (150 : ℝ)
  let mpg1 := (35 : ℝ)
  let distance2 := (150 : ℝ)
  let mpg2 := (25 : ℝ)
  let result := average_mpg total_distance distance1 mpg1 distance2 mpg2
  ∃ ε > 0, |result - 29| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_approx_29_l817_81752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_tree_problem_l817_81760

/-- Factor tree problem -/
theorem factor_tree_problem (X Y Z W : ℕ) : 
  Y = 2 * 7 →
  Z = 11 * W →
  W = 11 * 2 →
  X = Y * Z →
  X = 3388 := by
  sorry

#check factor_tree_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_tree_problem_l817_81760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l817_81756

open Matrix

variable {n : ℕ} {K : Type*} [CommRing K]
variable (N : Matrix (Fin n) (Fin n) K)

theorem det_cube (h : det N = 3) : det (N ^ 3) = 27 := by
  have h1 : det (N ^ 3) = (det N) ^ 3 := by
    sorry -- Proof of this property would go here
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l817_81756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_one_zero_point_m_values_l817_81708

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.exp x * (2 * x - 1) - m * x + m

-- Define the property of having exactly one zero point
def has_exactly_one_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Theorem statement
theorem function_with_one_zero_point_m_values (m : ℝ) :
  has_exactly_one_zero (f · m) → m = 1 ∨ m = 4 * Real.exp (3/2) ∨ m ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_one_zero_point_m_values_l817_81708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l817_81764

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (2, 1) to the line 3x + 4y - 2 = 0 is 8/5 -/
theorem distance_specific_point_to_line :
  distance_point_to_line 2 1 3 4 (-2) = 8/5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l817_81764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_relations_l817_81702

noncomputable section

open Real

theorem triangle_trigonometric_relations (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with internal angles A, B, C and opposite sides a, b, c
  A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0 →
  -- Statement 1
  ¬(∃ (A B C : ℝ), A + B + C = π ∧ (A > π/2 ∨ B > π/2 ∨ C > π/2) → 
    Real.tan A + Real.tan B + Real.tan C > 0) ∧
  -- Statement 2
  ¬(∀ (A B C : ℝ), A + B + C = π ∧ A < π/2 ∧ B < π/2 ∧ C < π/2 → 
    Real.cos A + Real.cos B > Real.sin A + Real.sin B) ∧
  -- Statement 3
  ¬(∀ (A B : ℝ), A < B → Real.cos (Real.sin A) < Real.cos (Real.tan B)) ∧
  -- Statement 4
  (Real.sin B = 2/5 ∧ Real.tan C = 3/4 → A > C ∧ C > B) := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_relations_l817_81702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_digits_sum_l817_81766

theorem base_conversion_digits_sum : 
  ∀ n : ℕ, 
  (6^3 ≤ n ∧ n < 6^4) → 
  (∃ d : ℕ, 8 ≤ d ∧ d ≤ 11 ∧ 2^(d-1) ≤ n ∧ n < 2^d) ∧
  (Finset.sum {8, 9, 10, 11} id = 38) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_digits_sum_l817_81766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_root_three_l817_81701

theorem expression_equals_root_three : 
  |Real.sqrt 3 - 1| + (-8 : ℝ) ^ (1/3 : ℝ) + Real.sqrt 9 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_root_three_l817_81701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_magnitude_l817_81741

noncomputable def a : ℝ × ℝ := (-1, 2)
noncomputable def b : ℝ × ℝ := (3, 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def orthogonal (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_c_magnitude :
  ∀ c : ℝ × ℝ,
    parallel a c →
    orthogonal a (b.1 + c.1, b.2 + c.2) →
    magnitude c = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_magnitude_l817_81741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_PC_l817_81779

/-- IsEquilateral predicate for a triangle -/
def IsEquilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - A‖

/-- Given an equilateral triangle ABC and a point P, prove the maximum length of PC -/
theorem max_length_PC (A B C P : EuclideanSpace ℝ (Fin 2)) : 
  IsEquilateral A B C →  -- ABC is an equilateral triangle
  ‖A - P‖ = 2 →         -- AP = 2
  ‖B - P‖ = 3 →         -- BP = 3
  ∃ (C' : EuclideanSpace ℝ (Fin 2)), IsEquilateral A B C' ∧ ‖C' - P‖ ≤ 5 ∧ 
    ∀ (C'' : EuclideanSpace ℝ (Fin 2)), IsEquilateral A B C'' → ‖C'' - P‖ ≤ ‖C' - P‖ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_PC_l817_81779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l817_81772

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the circle C
def circle_C (center : PolarPoint) (P : PolarPoint) : Prop :=
  center.ρ = 1 ∧ 
  center.θ = 0 ∧ 
  P.ρ = Real.sqrt 3 ∧ 
  P.θ = Real.pi / 6 ∧
  center.ρ * Real.sin (Real.pi / 3 - center.θ) = Real.sqrt 3 / 2

-- Theorem for the radius and equation of circle C
theorem circle_C_properties (center : PolarPoint) (P : PolarPoint) 
  (h : circle_C center P) : 
  (∃ (r : ℝ), r = 1 ∧ ∀ (θ : ℝ), ∃ (ρ : ℝ), ρ = 2 * Real.cos θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l817_81772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_fourth_zero_not_before_l817_81787

def u (n : ℕ) : ℕ := n^4 + n^2

def delta : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0, f => f
  | k+1, f => fun n => delta k f (n+1) - delta k f n

theorem delta_fourth_zero_not_before (n : ℕ) :
  delta 4 u n = 0 ∧ ∀ k < 4, delta k u n ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_fourth_zero_not_before_l817_81787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l817_81747

theorem inequality_proof (n : ℕ) (x : ℝ) (hn : n > 1) :
  let f := fun (t : ℝ) => (1 + 1 / (n : ℝ))^t
  ((f (2 * x) + f 2) / 2) > (f x * Real.log (1 + 1 / (n : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l817_81747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l817_81792

-- Define the functions
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x + 1)

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → f m x ≤ f m y
def q (m : ℝ) : Prop := ∀ x, ∃ y, g m x = y

-- State the theorem
theorem f_g_properties :
  (¬(p 2)) ∧ 
  (∀ m, (p m ∧ ¬(q m)) ∨ (¬(p m) ∧ q m) ↔ m ≤ -2 ∨ (1 < m ∧ m < 2)) :=
by
  sorry

#check f_g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l817_81792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l817_81781

-- Define a plane
structure Plane where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

-- Define a line
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def is_perpendicular_line_plane (l : Line) (α : Plane) : Prop :=
  ∃ (k : ℝ), l.direction = k • α.normal

-- Define perpendicularity between two lines
def is_perpendicular_lines (l1 l2 : Line) : Prop :=
  let (x1, y1, z1) := l1.direction
  let (x2, y2, z2) := l2.direction
  x1 * x2 + y1 * y2 + z1 * z2 = 0

-- Theorem statement
theorem perpendicular_line_plane (l : Line) (α : Plane) :
  (∀ (m : Line), m.point ∈ Set.range (λ p : ℝ × ℝ × ℝ => p) → is_perpendicular_lines l m) →
  is_perpendicular_line_plane l α :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l817_81781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_decrease_approx_19_96_l817_81734

-- Define the revenue data for each company
noncomputable def company_A_old : ℝ := 69.0
noncomputable def company_A_new : ℝ := 48.0
noncomputable def company_B_old : ℝ := 120.0
noncomputable def company_B_new : ℝ := 100.0
noncomputable def company_C_old : ℝ := 172.0
noncomputable def company_C_new : ℝ := 150.0

-- Define the function to calculate percentage decrease
noncomputable def percentage_decrease (old_revenue new_revenue : ℝ) : ℝ :=
  ((old_revenue - new_revenue) / old_revenue) * 100

-- Define the theorem
theorem average_percentage_decrease_approx_19_96 :
  let decrease_A := percentage_decrease company_A_old company_A_new
  let decrease_B := percentage_decrease company_B_old company_B_new
  let decrease_C := percentage_decrease company_C_old company_C_new
  let average_decrease := (decrease_A + decrease_B + decrease_C) / 3
  abs (average_decrease - 19.96) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_decrease_approx_19_96_l817_81734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l817_81785

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_right_focus_to_line :
  distance_point_to_line right_focus.1 right_focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l817_81785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l817_81737

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x > 0 ∧ f (-1) < f (Real.log x / Real.log 10)}

-- State the theorem
theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : monotone_decreasing_on f {x | x < 0}) :
  solution_set f = {x | 0 < x ∧ x < (1/10) ∨ x > 10} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l817_81737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l817_81722

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem statement
theorem correct_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_line_plane m α ∧ parallel_line_plane n β ∧ parallel_plane_plane α β) → 
    perpendicular_line_line m n) ∧
  ((perpendicular_line_plane m α ∧ perpendicular_line_plane n β ∧ perpendicular_plane_plane α β) → 
    perpendicular_line_line m n) :=
by
  constructor
  · intro h
    sorry
  · intro h
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l817_81722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l817_81745

/-- Circle C in the Cartesian plane -/
def Circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

/-- Line l passing through (0,2) with slope √3 -/
def Line (x y : ℝ) : Prop := y - 2 = Real.sqrt 3 * x

/-- Point P -/
def P : ℝ × ℝ := (0, 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_line_intersection_theorem :
  ∃ (A B : ℝ × ℝ),
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
    Line A.1 A.2 ∧ Line B.1 B.2 ∧
    A ≠ B ∧
    |1 / distance P A - 1 / distance P B| = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l817_81745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l817_81768

noncomputable def shaded_area (r₁ r₂ : ℝ) : ℝ :=
  let θ := Real.arccos (r₁ / r₂)
  let sector_area := 2 * (θ / (2 * Real.pi)) * Real.pi * r₂^2
  let triangle_area := r₁ * Real.sqrt (r₂^2 - r₁^2)
  let small_sector_area := (θ / (2 * Real.pi)) * Real.pi * r₁^2
  4 * (sector_area - triangle_area - small_sector_area)

theorem shaded_area_calculation :
  ∃ ε > 0, |shaded_area 2 3 - ((5.65 / 3) * Real.pi - 2.9724 * Real.sqrt 5)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l817_81768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l817_81733

/-- A function f: ℝ → ℝ has a positive extremum if there exists a point c > 0 such that
    f'(c) = 0 and for all x in a neighborhood of c, f(x) ≤ f(c) (for maximum)
    or f(x) ≥ f(c) (for minimum) -/
def has_positive_extremum (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ deriv f c = 0 ∧
  ((∃ ε > 0, ∀ x, |x - c| < ε → f x ≤ f c) ∨
   (∃ ε > 0, ∀ x, |x - c| < ε → f x ≥ f c))

/-- The main theorem stating that if e^(ax) + 2x has a positive extremum, then a < -2 -/
theorem extremum_condition (a : ℝ) :
  has_positive_extremum (λ x ↦ Real.exp (a * x) + 2 * x) → a < -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l817_81733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_seven_terms_proof_of_seven_terms_l817_81796

/-- The number of terms in the expansion of [(a + 2b)³(a - 2b)³]² -/
def num_terms : ℕ := 7

/-- The expansion of [(a + 2b)³(a - 2b)³]² has 7 distinct terms -/
theorem expansion_has_seven_terms :
  num_terms = 7 := by rfl

/-- Proof that the expansion has exactly 7 terms -/
theorem proof_of_seven_terms (a b : ℝ) :
  ((a + 2*b)^3 * (a - 2*b)^3)^2 =
    a^12 - 24*a^10*b^2 + 240*a^8*b^4 - 1344*a^6*b^6 + 3840*a^4*b^8 - 6144*a^2*b^10 + 4096*b^12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_seven_terms_proof_of_seven_terms_l817_81796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_sortable_l817_81784

/-- Represents a person with their height and position in the line -/
structure Person where
  height : ℝ
  position : ℕ

/-- Represents the line of people -/
def Line := Vector Person 1998

/-- Checks if two positions in the line can be swapped -/
def canSwap (i j : ℕ) : Prop :=
  i < 1998 ∧ j < 1998 ∧ (i + 2 = j ∨ j + 2 = i)

/-- Represents a permutation of the line that can be achieved through allowed swaps -/
def ValidPermutation (l : Line) (p : Equiv.Perm (Fin 1998)) : Prop :=
  ∀ i j, p.toFun i ≠ i → (canSwap i.val j.val ∨ i = j)

/-- States that it's not always possible to sort the line by height using only allowed swaps -/
theorem not_always_sortable : 
  ∃ l : Line, ¬ (∀ p : Equiv.Perm (Fin 1998), ValidPermutation l p → 
    (∀ i j : Fin 1998, i < j → (l.get i).height ≤ (l.get j).height)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_sortable_l817_81784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l817_81758

-- Define the function f(x) parametrized by k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (3 * k * x^2 - x + 2) / (-3 * x^2 + 4 * x + k)

-- State the theorem
theorem domain_all_reals (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ k < -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l817_81758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_calculation_l817_81727

/-- The amount of dog food in pounds that Melody initially bought -/
noncomputable def initial_dog_food : ℚ := 30

/-- The number of dogs Melody has -/
def num_dogs : ℕ := 3

/-- The amount of dog food in pounds that each dog eats per meal -/
noncomputable def food_per_meal : ℚ := 1/2

/-- The number of meals each dog eats per day -/
def meals_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The amount of dog food in pounds left after a week -/
noncomputable def food_left : ℚ := 9

theorem dog_food_calculation :
  initial_dog_food = 
    (num_dogs : ℚ) * food_per_meal * (meals_per_day : ℚ) * (days_in_week : ℚ) + food_left :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_calculation_l817_81727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l817_81746

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, k = 2^m

theorem power_of_two_characterization (k : ℕ+) : 
  (∀ n : ℕ+, ∃ m : ℕ, 2^((k.val-1)*n.val+1) * (k.val*n.val).factorial / n.val.factorial = m) ↔ is_power_of_two k.val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l817_81746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_below_g_implies_k_range_l817_81754

/-- The function f(x) = 1 - kx^2 --/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 1 - k * x^2

/-- The function g(x) = (√3/3)[sin(x + 2018π/3) - sin(x - 2018π/3)] --/
noncomputable def g (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * (Real.sin (x + 2018 * Real.pi / 3) - Real.sin (x - 2018 * Real.pi / 3))

/-- Theorem: If f(x) is always below g(x) for all real x, then k ≥ 1/2 --/
theorem f_below_g_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, f k x < g x) → k ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_below_g_implies_k_range_l817_81754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xichang_dunk_probability_l817_81791

-- Define the number of teams
def num_teams : ℕ := 4

-- Define the favorable positions for Xichang player (2nd and 3rd)
def favorable_positions : ℕ := 2

-- Define the target probability
def target_probability : ℚ := 1 / 2

-- Theorem statement
theorem xichang_dunk_probability :
  (favorable_positions * Nat.factorial (num_teams - 1)) / Nat.factorial num_teams = target_probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xichang_dunk_probability_l817_81791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_is_sqrt_13_l817_81799

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a line in 2D space using two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculates the slope of a line -/
noncomputable def slopeLine (l : Line) : ℝ :=
  (l.p2.y - l.p1.y) / (l.p2.x - l.p1.x)

/-- Calculates the y-intercept of a line -/
noncomputable def yIntercept (l : Line) : ℝ :=
  l.p1.y - slopeLine l * l.p1.x

/-- Finds the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  let m1 := slopeLine l1
  let m2 := slopeLine l2
  let b1 := yIntercept l1
  let b2 := yIntercept l2
  let x := (b2 - b1) / (m1 - m2)
  let y := m1 * x + b1
  { x := x, y := y }

theorem length_of_AE_is_sqrt_13 (A B C D : Point) (h1 : A = { x := 0, y := 4 })
    (h2 : B = { x := 6, y := 0 }) (h3 : C = { x := 2, y := 1 }) (h4 : D = { x := 5, y := 4 }) : 
    let AB : Line := { p1 := A, p2 := B }
    let CD : Line := { p1 := C, p2 := D }
    let E : Point := intersectionPoint AB CD
    distance A E = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_is_sqrt_13_l817_81799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l817_81749

/-- The length of the chord intercepted by a circle on a line -/
theorem chord_length_circle_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ((x₁ - 3)^2 + y₁^2 = 9) ∧
    ((x₂ - 3)^2 + y₂^2 = 9) ∧
    (3*x₁ - 4*y₁ - 4 = 0) ∧
    (3*x₂ - 4*y₂ - 4 = 0) ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (4 * Real.sqrt 2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l817_81749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l817_81748

/-- The sum of the infinite series Σ(k=1 to ∞) k/(3^k) -/
noncomputable def seriesSum : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

/-- The theorem stating that the sum of the infinite series Σ(k=1 to ∞) k/(3^k) is equal to 3/4 -/
theorem series_sum_equals_three_fourths : seriesSum = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l817_81748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l817_81718

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  deriving Repr

/-- The grid configuration -/
def Grid := Cell → Nat

/-- Distance between two cells -/
noncomputable def distance (c1 c2 : Cell) : Real :=
  Real.sqrt (((c1.row : Real) - (c2.row : Real))^2 + ((c1.col : Real) - (c2.col : Real))^2)

/-- Predicate to check if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Theorem stating the maximum value of S -/
theorem max_S_value (grid : Grid) : 
  (∀ c : Cell, c.row ≤ 100 ∧ c.col ≤ 100) →
  (∀ n : Nat, ∃! c : Cell, grid c = n ∧ n ≤ 10000) →
  (∀ c1 c2 : Cell, adjacent c1 c2 → (grid c1 : Int) - (grid c2 : Int) = 1 ∨ (grid c2 : Int) - (grid c1 : Int) = 1) →
  (∃ S : Real, ∀ c1 c2 : Cell, (grid c1 : Int) - (grid c2 : Int) = 5000 → distance c1 c2 ≥ S) →
  (∀ S' : Real, (∀ c1 c2 : Cell, (grid c1 : Int) - (grid c2 : Int) = 5000 → distance c1 c2 ≥ S') → S' ≤ 50 * Real.sqrt 2) :=
by
  sorry

#check max_S_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l817_81718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_values_l817_81709

def polynomial (x b : ℤ) : ℤ := x^3 + 5*x^2 + b*x + 9

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ ({-127, -74, -27, -24, -15, -13} : Finset ℤ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_values_l817_81709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l817_81730

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a*x
def g (a x : ℝ) : ℝ := (3*a + 1)*x - (a^2 + a)*x^2

-- State the theorem
theorem function_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f a x > f a y) ∧ (a ≤ -1/2 ∨ a ≥ 1) ∧
  (∀ x > 1, f a x < g a x) ∧ (-1 ≤ a ∧ a < 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l817_81730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brother_lower_limit_l817_81721

/-- Arun's weight in kilograms -/
def W : ℝ := sorry

/-- Lower limit of Arun's weight according to his brother's opinion -/
def B : ℝ := sorry

/-- Arun's weight is greater than 66 kg but less than 72 kg -/
axiom arun_opinion : 66 < W ∧ W < 72

/-- Arun's weight is greater than B but less than 70 kg (according to his brother) -/
axiom brother_opinion : B < W ∧ W < 70

/-- Arun's weight cannot be greater than 69 kg (according to his mother) -/
axiom mother_opinion : W ≤ 69

/-- The average of different probable weights is 68 kg -/
axiom average_weight : (max 66 B + 69) / 2 = 68

/-- The lower limit of Arun's weight according to his brother's opinion is 67 kg -/
theorem brother_lower_limit : B = 67 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brother_lower_limit_l817_81721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l817_81750

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n + a (n + 1) + a (n + 2) > 0) ∧
  (∀ n, a n + a (n + 1) + a (n + 2) + a (n + 3) < 0)

theorem max_sequence_length (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ n : ℕ, ∀ m : ℕ, m > n → ¬(sequence_property (fun i => if i ≤ m then a i else 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l817_81750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_zeros_l817_81725

/-- A set of numbers satisfying the given conditions -/
def BoardNumbers : Type := {s : Finset ℤ // s.card = 2011 ∧ ∀ a b c, a ∈ s → b ∈ s → c ∈ s → (a + b + c) ∈ s}

/-- The number of zeros in a set of numbers -/
def countZeros (s : Finset ℤ) : ℕ := (s.filter (· = 0)).card

/-- The theorem stating the minimum number of zeros -/
theorem min_zeros (numbers : BoardNumbers) : 
  countZeros numbers.1 ≥ 2009 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_zeros_l817_81725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_equivalence_vertical_shift_l817_81738

-- Define the original function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 3
  else if x < 1 then Real.sqrt (1 - x^2) + 1
  else -x + 3

-- Define the shifted function
noncomputable def g_shifted (x : ℝ) : ℝ := g x + 2

-- Theorem statement
theorem g_shift_equivalence :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → g_shifted x = g x + 2 := by
  intro x h
  unfold g_shifted
  rfl

-- Theorem to show that g_shifted is indeed a vertical shift of g
theorem vertical_shift :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → g_shifted x = g x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_equivalence_vertical_shift_l817_81738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_sulfate_formation_l817_81762

-- Define the chemical species
inductive ChemicalSpecies
| NH3
| H2SO4
| NH4_2SO4

-- Define a function to represent the number of moles
noncomputable def moles : ChemicalSpecies → ℚ
| ChemicalSpecies.NH3 => 4
| ChemicalSpecies.H2SO4 => 2
| ChemicalSpecies.NH4_2SO4 => 0  -- Initial amount, to be calculated

-- Define the stoichiometric coefficients
def stoichiometry : ChemicalSpecies → ℚ
| ChemicalSpecies.NH3 => 2
| ChemicalSpecies.H2SO4 => 1
| ChemicalSpecies.NH4_2SO4 => 1

-- Define the reaction yield function
noncomputable def reaction_yield (reactant : ChemicalSpecies) : ℚ :=
  (moles reactant) / (stoichiometry reactant)

-- Theorem to prove
theorem ammonium_sulfate_formation :
  min (reaction_yield ChemicalSpecies.NH3) (reaction_yield ChemicalSpecies.H2SO4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_sulfate_formation_l817_81762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_monotonicity_l817_81788

-- Define the linear function f
def f : ℝ → ℝ := λ x => 4*x + 1

-- Define g in terms of f
def g (m : ℝ) : ℝ → ℝ := λ x => f x * (x + m)

-- Theorem statement
theorem linear_function_and_monotonicity :
  (∀ x y : ℝ, x < y → f x < f y) ∧  -- f is increasing on ℝ
  (∀ x : ℝ, f (f x) = 16 * x + 5) ∧  -- f(f(x)) = 16x + 5
  (∀ m : ℝ, (∀ x y : ℝ, 1 < x ∧ x < y → g m x < g m y) ↔ m ≥ -9/4)  -- g is monotonically increasing on (1, +∞) iff m ≥ -9/4
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_monotonicity_l817_81788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l817_81732

/-- Given two functions f and g, and a real number k, we prove three statements about their relationship. -/
theorem function_inequalities (k : ℝ) :
  let f := fun (x : ℝ) => 8 * x^2 + 16 * x - k
  let g := fun (x : ℝ) => 2 * x^3 + 5 * x^2 + 4 * x
  ((∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ≤ g x) ↔ k ≥ 45) ∧
  ((∃ x, x ∈ Set.Icc (-3 : ℝ) 3 ∧ f x ≤ g x) ↔ k ≥ -7) ∧
  ((∀ x1 x2, x1 ∈ Set.Icc (-3 : ℝ) 3 → x2 ∈ Set.Icc (-3 : ℝ) 3 → f x1 ≤ g x2) ↔ k ≥ 141) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l817_81732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_period_l817_81736

/-- The period of a sine function y = sin(bx) -/
noncomputable def period (b : ℝ) : ℝ := 2 * Real.pi / b

/-- The function y = sin(3x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem sin_3x_period :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ p = period 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_period_l817_81736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roulette_even_product_probability_l817_81717

/-- Represents the numbers on the roulette wheel -/
def roulette_numbers : Finset ℕ :=
  Finset.range 10 ∪ {11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

/-- The probability of getting an even product when spinning the roulette wheel twice -/
def prob_even_product : ℚ :=
  1 - (1 - (Finset.filter (fun n => n % 2 = 0) roulette_numbers).card / roulette_numbers.card) ^ 2

theorem roulette_even_product_probability :
  prob_even_product = 7 / 16 := by
  sorry

#eval prob_even_product.num
#eval prob_even_product.den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roulette_even_product_probability_l817_81717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l817_81740

-- Define the functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
noncomputable def g (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem inequality_proof 
  (a b c d : ℝ) 
  (h1 : a > b) (h2 : b ≥ 1) 
  (h3 : c > d) (h4 : d > 0) 
  (h5 : f a - f b = Real.pi) 
  (h6 : g c - g d = Real.pi / 10) : 
  a + d - b - c < (9 * Real.pi) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l817_81740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l817_81795

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6) - 1

theorem f_properties :
  -- Minimum positive period
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  -- Monotonically decreasing interval
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    k * Real.pi + Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 5 * Real.pi / 3 
    → f y < f x) ∧
  -- Range when x ∈ [0, π/2]
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → -2 ≤ f x ∧ f x ≤ 1) ∧
  (∃ (x y : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 0 ≤ y ∧ y ≤ Real.pi / 2 ∧ 
    f x = -2 ∧ f y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l817_81795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_seating_value_l817_81744

/-- Represents a circular table with chairs -/
structure CircularTable where
  total_chairs : Nat
  unoccupied_chairs : Nat
  seating_condition : Nat → Nat → Prop

/-- The smallest number of people that can be seated under given conditions -/
def smallest_seating (table : CircularTable) : Nat :=
  sorry

/-- Theorem stating the smallest number of people that can be seated -/
theorem smallest_seating_value (table : CircularTable) 
  (h1 : table.total_chairs = 72)
  (h2 : table.unoccupied_chairs = 12)
  (h3 : table.seating_condition = λ n m => 
    m > n → m ≤ table.total_chairs - table.unoccupied_chairs → 
    ∃ k, k < m ∧ k ≥ n ∧ (m - k = 1 ∨ k - n = 1)) :
  smallest_seating table = 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_seating_value_l817_81744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_angle_sum_l817_81759

/-- A point lies on a circle -/
def OnCircle (P : Point) (c : Circle) : Prop := sorry

/-- A quadrilateral is inscribed in a circle -/
def QuadrilateralInscribed (E F G H : Point) (c : Circle) : Prop :=
  OnCircle E c ∧ OnCircle F c ∧ OnCircle G c ∧ OnCircle H c

/-- The measure of an angle in degrees -/
noncomputable def MeasureAngle (A B C : Point) : ℝ := sorry

/-- Given a quadrilateral EFGH inscribed in a circle, if ∠EGH = 50° and ∠EFG = 20°, then ∠GEF + ∠EHG = 110° -/
theorem inscribed_quadrilateral_angle_sum (E F G H : Point) (c : Circle) :
  QuadrilateralInscribed E F G H c →
  MeasureAngle E G H = 50 →
  MeasureAngle E F G = 20 →
  MeasureAngle G E F + MeasureAngle E H G = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_angle_sum_l817_81759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cutting_perimeter_difference_l817_81757

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents a way to cut the plywood -/
structure CutConfiguration where
  rectangles : Fin 6 → Rectangle
  is_valid : ∀ i, (rectangles i).length * (rectangles i).width = 9

/-- The theorem statement -/
theorem plywood_cutting_perimeter_difference :
  ∃ (max_config min_config : CutConfiguration),
    (∀ config : CutConfiguration, perimeter (config.rectangles 0) ≤ perimeter (max_config.rectangles 0)) ∧
    (∀ config : CutConfiguration, perimeter (min_config.rectangles 0) ≤ perimeter (config.rectangles 0)) ∧
    perimeter (max_config.rectangles 0) - perimeter (min_config.rectangles 0) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cutting_perimeter_difference_l817_81757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_limit_x_bounded_above_x_satisfies_equation_limit_satisfies_equation_l817_81782

/-- The sequence defined by the nested radical -/
noncomputable def x : ℕ → ℝ
  | 0 => Real.sqrt 86
  | n + 1 => Real.sqrt (86 + 41 * x n)

/-- The theorem stating that the limit of the sequence is 43 -/
theorem nested_radical_limit : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - 43| < ε := by
  sorry

/-- The theorem stating that the sequence is bounded above by 43 -/
theorem x_bounded_above : ∀ n, x n ≤ 43 := by
  sorry

/-- The theorem stating that x satisfies the equation F = sqrt(86 + 41F) -/
theorem x_satisfies_equation : ∀ n, (x n)^2 = 86 + 41 * (x n) := by
  sorry

/-- The theorem stating that the limit of x is a solution to F^2 = 86 + 41F -/
theorem limit_satisfies_equation : 
  ∃ L, (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - L| < ε) ∧ L^2 = 86 + 41 * L := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_limit_x_bounded_above_x_satisfies_equation_limit_satisfies_equation_l817_81782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_l817_81712

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

theorem train_pass_time_approx :
  let ε := 0.01
  ∃ t : ℝ, 
    |t - train_pass_time 480 260 85| < ε ∧
    |t - 31.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_l817_81712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_52_63487_to_nearest_tenth_l817_81714

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The original number to be rounded -/
def original_number : ℝ := 52.63487

/-- Theorem stating that rounding 52.63487 to the nearest tenth results in 52.6 -/
theorem round_52_63487_to_nearest_tenth :
  round_to_nearest_tenth original_number = 52.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_52_63487_to_nearest_tenth_l817_81714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l817_81731

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def C₂ (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the distance function
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem distance_range :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  1 ≤ distance x₁ y₁ x₂ y₂ ∧ distance x₁ y₁ x₂ y₂ ≤ 5 := by
  sorry

#check distance_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l817_81731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l817_81713

/-- The optimal selection method concept -/
inductive OptimalSelectionConcept
| GoldenRatio
| Mean
| Mode
| Median

/-- Properties of the mathematical constant used in the optimal selection method -/
structure MathConstantProperties where
  unique : Bool
  universal : Bool
  aesthetic : Bool
  proportional : Bool

/-- The optimal selection method -/
def optimalSelectionMethod (concept : OptimalSelectionConcept) (properties : MathConstantProperties) : Prop := 
  match concept with
  | OptimalSelectionConcept.GoldenRatio => properties.unique ∧ properties.universal ∧ properties.aesthetic ∧ properties.proportional
  | _ => False

/-- Theorem: The optimal selection method uses the Golden ratio -/
theorem optimal_selection_uses_golden_ratio 
  (h : MathConstantProperties) 
  (hunique : h.unique = true) 
  (huniversal : h.universal = true) 
  (haesthetic : h.aesthetic = true) 
  (hproportional : h.proportional = true) :
  optimalSelectionMethod OptimalSelectionConcept.GoldenRatio h := by
  simp [optimalSelectionMethod]
  exact ⟨hunique, huniversal, haesthetic, hproportional⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l817_81713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_ratio_theorem_l817_81724

noncomputable def sum_of_coefficients (f : ℝ → ℝ) (x : ℝ) : ℝ := f x

def sum_of_binomial_coefficients (n : ℕ) : ℝ := 2^n

theorem expansion_ratio_theorem (n : ℕ) : 
  (∀ x : ℝ, x > 0 → 
    (sum_of_coefficients (λ y => (Real.sqrt y + 3 * Real.rpow y (-1/3))^n) 1) / 
    (sum_of_binomial_coefficients n) = 64) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_ratio_theorem_l817_81724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l817_81797

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

def graph1 (x y : ℝ) : Prop := (frac x)^2 + y^2 = 2 * (frac x)

def graph2 (x y : ℝ) : Prop := y = (1/3) * x

def n_values : Set ℤ := {-2, -1, 0, 1, 2}

def num_intersections : ℕ := 10

theorem intersection_count :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ S ↔ (graph1 p.1 p.2 ∧ graph2 p.1 p.2 ∧ (floor p.1) ∈ n_values)) ∧
    (Finite S ∧ Nat.card S = num_intersections) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l817_81797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_covering_impossibility_l817_81774

theorem cube_face_covering_impossibility :
  ∀ (cube_side_length : ℕ) (strip_width : ℕ) (strip_length : ℕ) (num_strips : ℕ),
    cube_side_length = 4 →
    strip_width = 1 →
    strip_length = 3 →
    num_strips = 16 →
    ¬ ∃ (arrangement : List (ℕ × ℕ × ℕ)),
      (arrangement.length = num_strips) ∧
      (∀ (strip : ℕ × ℕ × ℕ), strip ∈ arrangement →
        strip.1 < cube_side_length ∧ strip.2.1 < cube_side_length ∧ strip.2.2 < cube_side_length) ∧
      (∀ (i j : ℕ), i < cube_side_length → j < cube_side_length →
        (∃ (strip : ℕ × ℕ × ℕ), strip ∈ arrangement ∧
          ((strip.1 = i ∧ strip.2.1 = j ∧ strip.2.2 = 0) ∨
           (strip.1 = i ∧ strip.2.1 = 0 ∧ strip.2.2 = j) ∨
           (strip.1 = 0 ∧ strip.2.1 = i ∧ strip.2.2 = j)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_covering_impossibility_l817_81774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_siblings_six_people_l817_81794

/-- Represents a group of people in a room with siblings -/
structure SiblingGroup where
  people : Finset Nat
  sibling : {x // x ∈ people} → {x // x ∈ people}
  sibling_pairs : Nat

/-- The probability of selecting two non-siblings from a SiblingGroup -/
def prob_non_siblings (g : SiblingGroup) : ℚ :=
  let total_pairs := (g.people.card * (g.people.card - 1)) / 2
  (total_pairs - g.sibling_pairs) / total_pairs

theorem prob_non_siblings_six_people (g : SiblingGroup) 
  (h1 : g.people.card = 6)
  (h2 : ∀ p : {x // x ∈ g.people}, g.sibling p ≠ p)
  (h3 : ∀ p : {x // x ∈ g.people}, g.sibling (g.sibling p) = p)
  (h4 : g.sibling_pairs = 3) :
  prob_non_siblings g = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_siblings_six_people_l817_81794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_in_ratio_l817_81726

def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (4, -5)
def C : ℝ × ℝ := (7.5, -8.5)
def lambda : ℝ := -3

theorem divide_segment_in_ratio :
  (C.1 - A.1) / (B.1 - C.1) = lambda ∧ (C.2 - A.2) / (B.2 - C.2) = lambda := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_in_ratio_l817_81726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l817_81789

/-- Represents the number of miles Bill ran on Saturday -/
def billSaturday : ℕ := sorry

/-- Represents the number of miles Bill ran on Sunday -/
def billSunday : ℕ := sorry

/-- Represents the number of miles Julia ran on Sunday -/
def juliaSunday : ℕ := sorry

/-- Bill ran 4 more miles on Sunday than on Saturday -/
axiom bill_sunday_more : billSunday = billSaturday + 4

/-- Julia ran twice the number of miles on Sunday that Bill ran on Sunday -/
axiom julia_sunday : juliaSunday = 2 * billSunday

/-- Bill and Julia ran a total of 20 miles on Saturday and Sunday -/
axiom total_miles : billSaturday + billSunday + juliaSunday = 20

/-- Theorem: Bill ran 6 miles on Sunday -/
theorem bill_sunday_miles : billSunday = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l817_81789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l817_81780

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l817_81780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_squared_plus_floor_n_over_k_squared_l817_81776

theorem min_k_squared_plus_floor_n_over_k_squared (n : ℕ) : 
  (∃ k : ℕ+, ∀ j : ℕ+, k^2 + n / k^2 ≤ j^2 + n / j^2) ∧
  (∃ k : ℕ+, k^2 + n / k^2 = 1991) ↔ 
  967 * 1024 ≤ n ∧ n < 968 * 1024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_squared_plus_floor_n_over_k_squared_l817_81776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l817_81705

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  f1 : Point
  f2 : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  distance p e.f1 + distance p e.f2 = distance e.f1 (Point.mk 0 0) + distance e.f2 (Point.mk 0 0)

theorem ellipse_intersection : 
  let e := Ellipse.mk (Point.mk 0 2) (Point.mk 4 0)
  onEllipse e (Point.mk 0 0) → 
  (∃ x : ℝ, x ≠ 0 ∧ onEllipse e (Point.mk x 0)) →
  onEllipse e (Point.mk (24/5) 0) :=
by sorry

#check ellipse_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l817_81705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tricolor_triangle_l817_81700

/-- A coloring of edges in a complete graph with three colors -/
def Coloring (n : ℕ) := Fin (3*n + 1) → Fin (3*n + 1) → Fin 3

/-- A coloring is valid if each vertex has exactly n edges of each color -/
def valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ v : Fin (3*n + 1), ∀ color : Fin 3,
    (Finset.filter (fun w ↦ c v w = color) (Finset.univ.erase v)).card = n

/-- Main theorem: In a valid coloring, there exists a triangle with all three colors -/
theorem exists_tricolor_triangle (n : ℕ) (c : Coloring n) (h : valid_coloring n c) :
  ∃ u v w : Fin (3*n + 1), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c u v ≠ c v w ∧ c v w ≠ c u w ∧ c u w ≠ c u v := by
  sorry

#check exists_tricolor_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tricolor_triangle_l817_81700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l817_81765

-- Define the circle's area
noncomputable def circle_area : ℝ := 256 * Real.pi

-- Define the octagon's area
noncomputable def octagon_area : ℝ := 256 * Real.sqrt 2

-- Theorem statement
theorem inscribed_octagon_area :
  let r := Real.sqrt (circle_area / Real.pi)
  octagon_area = 8 * (1/2 * r^2 * Real.sin (Real.pi/4)) :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l817_81765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_distance_l817_81777

/-- Given two lines symmetric about the y-axis, proves the distance from a point to one of the lines -/
theorem symmetric_lines_distance (l₁ l₂ : Set (ℝ × ℝ)) :
  (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (-x, y) ∈ l₂) →  -- l₁ and l₂ are symmetric about y-axis
  (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ 2*x - 3*y = 0) →  -- equation of l₁
  let d := |2*2 + 3*(-1)| / Real.sqrt (2^2 + 3^2)
  d = 1 / Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_distance_l817_81777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l817_81711

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / (2 * sequence_a (n + 1) + 3)

theorem fourth_term_value : sequence_a 4 = 1 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l817_81711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l817_81742

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b d e : ℝ) : Prop :=
  a / b = d / e

/-- The condition for a = 1 to hold true when two lines are parallel -/
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, a * x + y + 1 = 0 ↔ x + a * y - 1 = 0) ↔ 
  parallel_lines a 1 1 a ∧ a = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l817_81742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l817_81755

/-- Represents the speed of a cyclist -/
structure CyclistSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- The problem setup -/
structure CyclistProblem where
  k : ℝ
  r : ℝ
  t : ℝ
  faster : CyclistSpeed
  slower : CyclistSpeed
  distance : ℝ
  same_direction_time : ℝ
  opposite_direction_time : ℝ
  k_pos : k > 0
  r_pos : r > 0
  t_pos : t > 0
  distance_eq : distance = 2 * k
  same_direction_eq : distance = (faster.speed - slower.speed) * same_direction_time
  opposite_direction_eq : distance = (faster.speed + slower.speed) * opposite_direction_time
  same_direction_time_eq : same_direction_time = 3 * r
  opposite_direction_time_eq : opposite_direction_time = 2 * t
  faster_twice_slower : faster.speed = 2 * slower.speed

/-- The theorem to prove -/
theorem cyclist_speed_ratio (prob : CyclistProblem) :
  prob.faster.speed / prob.slower.speed = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l817_81755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l817_81710

-- Define the given conditions
noncomputable def skateboard_time : ℝ := 45 / 60  -- Convert 45 minutes to hours
noncomputable def skateboard_speed : ℝ := 20
noncomputable def jog_time : ℝ := 75 / 60  -- Convert 75 minutes to hours
noncomputable def jog_speed : ℝ := 6

-- Define the total distance and time
noncomputable def total_distance : ℝ := skateboard_time * skateboard_speed + jog_time * jog_speed
noncomputable def total_time : ℝ := skateboard_time + jog_time

-- Theorem to prove
theorem overall_average_speed :
  total_distance / total_time = 11.25 := by
  -- Expand the definitions
  unfold total_distance total_time skateboard_time skateboard_speed jog_time jog_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l817_81710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l817_81716

/-- The parabola C with a point (1, y₀) at distance 3 from its focus has equation y² = 8x -/
theorem parabola_equation (p : ℝ) :
  (∃ y₀ : ℝ, y₀^2 = 2*p*1) ∧  -- Point (1, y₀) lies on the parabola
  (∃ y₀ : ℝ, (1 - (-p/2))^2 + y₀^2 = 3^2) -- Distance from (1, y₀) to focus (p/2, 0) is 3
  →
  (∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 8*x) -- The parabola equation is y² = 8x
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l817_81716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_origin_centered_circle_l817_81753

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 2)^2

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 16

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 = p.2 ∧ circle_equation p.1 p.2}

-- Theorem statement
theorem intersection_points_on_origin_centered_circle :
  ∃ R : ℝ, R = Real.sqrt 2 ∧
  ∀ p ∈ intersection_points, p.1^2 + p.2^2 = R^2 := by
  sorry

#check intersection_points_on_origin_centered_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_origin_centered_circle_l817_81753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_k_l817_81720

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

noncomputable def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a 1) * (1 - q^n) / (1 - q)

theorem geometric_sequence_max_k 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geo : geometric_sequence a q)
  (h_a1_neg : a 1 < 0)
  (h_arith : arithmetic_sequence (λ n ↦ match n with
    | 1 => a 1
    | 2 => a 4
    | 3 => a 3 - a 1
    | _ => 0
  ))
  (h_sum_ineq : ∀ k : ℕ, k > 4 → sum_of_geometric_sequence a q k < 
    5 * sum_of_geometric_sequence a q (k - 4)) :
  (∀ k : ℕ, k > 0 → sum_of_geometric_sequence a q k < 
    5 * sum_of_geometric_sequence a q (k - 4)) → k ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_k_l817_81720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_force_magnitude_l817_81790

-- Define the force type as a 2D vector
def Force := ℝ × ℝ

-- Define the magnitude of a force
noncomputable def magnitude (f : Force) : ℝ := Real.sqrt (f.1^2 + f.2^2)

-- Define the angle between two forces
noncomputable def angle (f1 f2 : Force) : ℝ := Real.arccos ((f1.1 * f2.1 + f1.2 * f2.2) / (magnitude f1 * magnitude f2))

-- State the theorem
theorem equilibrium_force_magnitude
  (F1 F2 F3 : Force)
  (h_equilibrium : F1.1 + F2.1 + F3.1 = 0 ∧ F1.2 + F2.2 + F3.2 = 0)
  (h_angle : angle F1 F2 = 2 * Real.pi / 3)
  (h_magnitude_F1 : magnitude F1 = 6)
  (h_magnitude_F2 : magnitude F2 = 6) :
  magnitude F3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_force_magnitude_l817_81790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l817_81707

/-- The sum of the infinite series Σ(n=0 to ∞) (-1)^n * (2n+1) * x^n -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (-1)^n * (2*n+1) * x^n

/-- Theorem stating that if S(x) = -8, then x = -9/5 -/
theorem series_solution (x : ℝ) (h : S x = -8) : x = -9/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l817_81707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_on_two_lines_l817_81719

/-- Represents a square on a plane -/
structure Square where
  side : ℝ
  center : ℝ × ℝ

/-- Generates the nth Fibonacci number -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

/-- Generates the sequence of squares -/
def generateSquares : ℕ → Square
  | 0 => { side := 1, center := (0, 0) }
  | 1 => { side := 1, center := (1, 0) }
  | n + 2 => sorry -- Implementation details omitted

/-- Checks if a point lies on a line given by two other points -/
def isOnLine (p1 p2 p : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x, y) := p
  (y2 - y1) * (x - x1) = (x2 - x1) * (y - y1)

/-- The main theorem to be proved -/
theorem centers_on_two_lines :
  ∃ (l1 l2 : ℝ × ℝ → Prop),
    (∀ n : ℕ, l1 ((generateSquares (2 * n)).center) ∨ l2 ((generateSquares (2 * n + 1)).center)) ∧
    (∀ p : ℝ × ℝ, l1 p → ¬l2 p) ∧
    (∀ p1 p2 p3 : ℝ × ℝ, l1 p1 → l1 p2 → l1 p3 → isOnLine p1 p2 p3) ∧
    (∀ p1 p2 p3 : ℝ × ℝ, l2 p1 → l2 p2 → l2 p3 → isOnLine p1 p2 p3) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_on_two_lines_l817_81719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l817_81739

theorem trig_problem (α : Real) 
  (h1 : Real.cos (α - π/6) + Real.sin α = (4 * Real.sqrt 3) / 5)
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.sin (α + π/3) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l817_81739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_25_pounds_apples_l817_81704

/-- Calculates the cost of apples given the weight in pounds -/
noncomputable def appleCost (pounds : ℝ) : ℝ :=
  let baseRate := 4 / 5  -- $4 per 5 pounds
  let baseCost := baseRate * pounds
  if pounds > 20 then
    baseCost * (1 - 0.2)  -- 20% discount
  else
    baseCost

/-- Theorem stating that 25 pounds of apples cost $16 -/
theorem cost_of_25_pounds_apples : appleCost 25 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_25_pounds_apples_l817_81704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_is_2_sqrt_5_l817_81793

-- Define the vectors
def a : ℝ × ℝ := (6, 2)
def b : ℝ → ℝ × ℝ := λ t => (t, 3)

-- Define the perpendicularity condition
def perpendicular (t : ℝ) : Prop := a.1 * (b t).1 + a.2 * (b t).2 = 0

-- Define the magnitude of the sum of vectors
noncomputable def magnitude_sum (t : ℝ) : ℝ := 
  Real.sqrt ((a.1 + (b t).1)^2 + (a.2 + (b t).2)^2)

-- Theorem statement
theorem magnitude_of_sum_is_2_sqrt_5 :
  ∃ t : ℝ, perpendicular t ∧ magnitude_sum t = 2 * Real.sqrt 5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_is_2_sqrt_5_l817_81793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_six_parts_intersection_lines_l817_81729

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents the 3D space -/
structure Space where

/-- Represents an intersection line between planes -/
structure IntersectionLine where

/-- Given three planes, returns the number of parts they divide the space into -/
def number_of_parts (p1 p2 p3 : Plane) : ℕ := sorry

/-- Given three planes, returns the number of intersection lines between them -/
def number_of_intersection_lines (p1 p2 p3 : Plane) : ℕ := sorry

theorem three_planes_six_parts_intersection_lines 
  (p1 p2 p3 : Plane) (s : Space) :
  number_of_parts p1 p2 p3 = 6 →
  (number_of_intersection_lines p1 p2 p3 = 1 ∨ 
   number_of_intersection_lines p1 p2 p3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_six_parts_intersection_lines_l817_81729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l817_81783

-- Define points A, B, and M
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, -3)

-- Define the line AB
def line_AB (x : ℝ) : ℝ := -x + 1

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

theorem min_distance_to_line :
  distance_point_to_line M.1 M.2 1 (-1) 1 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l817_81783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pumps_filling_time_l817_81786

noncomputable def pool_filling_time (slower_pump_rate : ℝ) (faster_pump_rate : ℝ) : ℝ :=
  1 / (slower_pump_rate + faster_pump_rate)

theorem two_pumps_filling_time :
  ∀ (slower_pump_rate : ℝ) (faster_pump_rate : ℝ) (slower_pump_alone_time : ℝ),
    slower_pump_rate > 0 →
    faster_pump_rate = 1.5 * slower_pump_rate →
    slower_pump_alone_time = 12.5 →
    slower_pump_rate = 1 / slower_pump_alone_time →
    pool_filling_time slower_pump_rate faster_pump_rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pumps_filling_time_l817_81786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l817_81703

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

/-- The radius of a sphere given its volume -/
noncomputable def sphereRadius (v : ℝ) : ℝ := (v / ((4 / 3) * Real.pi)) ^ (1 / 3)

theorem original_sphere_radius (R : ℝ) :
  (sphereVolume R = 27 * sphereVolume 1) → R = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l817_81703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l817_81773

theorem lambda_range (l : ℝ) : 
  (∀ x : ℝ, x > 0 → (Real.exp x + 1) * x > (Real.log x - Real.log l) * (x / l + 1)) → 
  l > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l817_81773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l817_81761

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.a * t.c * Real.sin t.B

theorem triangle_max_area (t : Triangle) (hb : t.b = 2) (hB : t.B = π/3) :
  ∃ (S : ℝ), S = Real.sqrt 3 ∧ ∀ (t' : Triangle), t'.b = 2 → t'.B = π/3 → triangleArea t' ≤ S := by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l817_81761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_distances_l817_81751

-- Define the line l₁
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0

-- Define the circle C
def C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x - 8 * y + 9 = 0

-- Define the fixed point of l₁
def fixed_point : ℝ × ℝ :=
  (2, 2)

-- Define l₂ as perpendicular to l₁ and passing through the fixed point
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  (4 * k + 1) * (x - 2) + (3 * k - 2) * (y - 2) = 0

-- Define the intersection points
noncomputable def A (k : ℝ) : ℝ × ℝ := sorry
noncomputable def B (k : ℝ) : ℝ × ℝ := sorry
noncomputable def E (k : ℝ) : ℝ × ℝ := sorry
noncomputable def F (k : ℝ) : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem max_sum_of_distances :
  ∀ k : ℝ, distance (A k) (B k) + distance (E k) (F k) ≤ 6 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_distances_l817_81751
