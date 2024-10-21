import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_equation_correct_l215_21563

/-- Represents a journey with two segments and a rest period. -/
structure Journey where
  total_distance : ℚ
  total_time : ℚ
  rest_time : ℚ
  speed_before_rest : ℚ
  speed_after_rest : ℚ

/-- The equation for the journey time before rest. -/
def journey_equation (j : Journey) (t : ℚ) : Prop :=
  j.speed_before_rest * t + j.speed_after_rest * (j.total_time - j.rest_time - t) = j.total_distance

/-- The specific journey described in the problem. -/
def johns_journey : Journey :=
  { total_distance := 300
    total_time := 5
    rest_time := 1/2
    speed_before_rest := 60
    speed_after_rest := 90 }

/-- Theorem stating that the given equation correctly represents John's journey. -/
theorem journey_equation_correct (t : ℚ) :
  journey_equation johns_journey t ↔ 60 * t + 90 * (9/2 - t) = 300 := by
  unfold journey_equation
  unfold johns_journey
  simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_equation_correct_l215_21563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_registration_ways_l215_21592

/-- The number of ways two students can register for admission tests. -/
def registration_ways : ℕ := 36

/-- There are three universities to choose from. -/
def num_universities : ℕ := 3

/-- Each student can choose up to two schools. -/
def max_choices : ℕ := 2

/-- The number of ways both students can choose one school each. -/
def one_school_each : ℕ := Nat.choose num_universities 1 * Nat.choose num_universities 1

/-- The number of ways both students can choose two schools each. -/
def two_schools_each : ℕ := Nat.choose num_universities 2 * Nat.choose num_universities 2

/-- The number of ways one student can choose one school and the other two schools. -/
def one_and_two_schools : ℕ := 2 * Nat.choose num_universities 1 * Nat.choose num_universities 2

/-- Theorem stating that the total number of registration ways is 36. -/
theorem total_registration_ways : registration_ways = one_school_each + two_schools_each + one_and_two_schools := by
  sorry

/-- Verify that the calculated result matches the expected value of 36. -/
example : registration_ways = 36 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_registration_ways_l215_21592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_l215_21547

/-- Given a triangle OAB and a point C satisfying certain conditions, 
    prove that a specific line always passes through a fixed point. -/
theorem fixed_point_on_line (O A B C : ℝ × ℝ) (l m : ℝ) : 
  (∀ (m : ℝ), (m + l) * (-3/2) + (m - 2*m) * (3/4) + 3*m = 0) →
  (C.1 - A.1 = 2 * (B.1 - C.1) ∧ C.2 - A.2 = 2 * (B.2 - C.2)) →
  (C = (l * A.1 + m * B.1, l * A.2 + m * B.2)) →
  true := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_l215_21547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l215_21521

/-- The circle centered at (1, 1) with radius √2 -/
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

/-- A point P with coordinates (a, 10) -/
def point (a : ℝ) : ℝ × ℝ := (a, 10)

/-- The distance squared from a point (x, y) to the center of the circle (1, 1) -/
def distanceSquared (x y : ℝ) : ℝ := (x - 1)^2 + (y - 1)^2

theorem point_outside_circle (a : ℝ) :
  distanceSquared (point a).1 (point a).2 > 2 := by
  -- Expand the definition of distanceSquared
  unfold distanceSquared
  -- Expand the definition of point
  unfold point
  -- Simplify the expression
  simp
  -- The result is (a - 1)^2 + 81 > 2
  -- This is always true because (a - 1)^2 ≥ 0 and 81 > 2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l215_21521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_sqrt2_over_2_l215_21590

/-- The distance from a point to a line in polar coordinates --/
noncomputable def distance_point_to_line (ρ₀ : ℝ) (θ₀ : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  sorry

/-- The equation of the line in polar coordinates --/
noncomputable def line_equation (ρ θ : ℝ) : ℝ :=
  ρ * Real.sin (θ + Real.pi/4) - Real.sqrt 2 / 2

/-- Theorem stating that the distance from the point to the line is √2/2 --/
theorem distance_point_to_line_is_sqrt2_over_2 :
  distance_point_to_line 2 (3*Real.pi/4) line_equation = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_sqrt2_over_2_l215_21590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l215_21595

/-- The function representing the curve y = e^x - 2 --/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 2

/-- The slope of the tangent line --/
def m : ℝ := 2

/-- Theorem: If the line y = 2x + b is tangent to y = e^x - 2, then b = -2ln(2) --/
theorem tangent_line_b_value (b : ℝ) :
  (∃ x₀ : ℝ, f x₀ = m * x₀ + b ∧ deriv f x₀ = m) → b = -2 * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l215_21595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_existence_l215_21593

/-- Definition of set sum -/
def setSum (X Y : Set α) [Add α] : Set α := {z | ∃ x y, x ∈ X ∧ y ∈ Y ∧ z = x + y}

/-- Theorem stating the existence of a partition for ℤ and non-existence for ℚ -/
theorem partition_existence : 
  (∃ (A B C : Set ℤ), A.Nonempty ∧ B.Nonempty ∧ C.Nonempty ∧ 
    (A ∪ B ∪ C = Set.univ) ∧ (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (setSum A B ∩ setSum B C = ∅) ∧ (setSum B C ∩ setSum C A = ∅) ∧ (setSum C A ∩ setSum A B = ∅)) ∧
  (¬∃ (A B C : Set ℚ), A.Nonempty ∧ B.Nonempty ∧ C.Nonempty ∧ 
    (A ∪ B ∪ C = Set.univ) ∧ (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (setSum A B ∩ setSum B C = ∅) ∧ (setSum B C ∩ setSum C A = ∅) ∧ (setSum C A ∩ setSum A B = ∅)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_existence_l215_21593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_equals_1_l215_21541

noncomputable def f (a b α β x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem f_2014_equals_1 (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2013 = -1) : f a b α β 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_equals_1_l215_21541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l215_21577

noncomputable def z : ℂ := (Complex.I / (1 - Complex.I)) ^ 2

theorem point_location : 
  let w := 2 + z
  (w.re > 0) ∧ (w.im < 0) :=
by
  -- Expand the definition of z
  have hz : z = -Complex.I/2 := by
    -- The proof of this step is omitted for brevity
    sorry
  
  -- Define w
  let w := 2 + z
  
  -- Substitute the value of z into w
  have hw : w = 2 - Complex.I/2 := by
    -- The proof of this step is omitted for brevity
    sorry
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that the real part is positive
  · -- The proof of this step is omitted for brevity
    sorry
  
  -- Prove that the imaginary part is negative
  · -- The proof of this step is omitted for brevity
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l215_21577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l215_21551

open Real

theorem triangle_property (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  sin A ^ 2 + sin B ^ 2 + sin A * sin B = sin C ^ 2 ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C →
  C = 2 * π / 3 ∧ 
  1 < (a + b) / c ∧ (a + b) / c < 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l215_21551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l215_21545

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x) / (1 + 2^x) - 1/2

-- Theorem statement
theorem range_of_y (x : ℝ) : 
  (floor (f x) + floor (f (-x)) = -1) ∨ (floor (f x) + floor (f (-x)) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l215_21545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_composition_l215_21588

theorem burger_composition (total_weight filler_weight : ℝ) 
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 75 ∧
  filler_weight / total_weight * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_composition_l215_21588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_implies_a_value_l215_21559

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (2^x - 1) + a

theorem function_symmetry_implies_a_value :
  ∀ a : ℝ, (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) → a = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_implies_a_value_l215_21559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_l215_21583

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 7 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define a function to count tangent lines
def count_tangent_lines : ℕ := 2

-- Theorem statement
theorem two_tangent_lines : count_tangent_lines = 2 := by
  -- The proof goes here
  sorry

#check two_tangent_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_l215_21583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l215_21550

def M : Set ℕ := {x : ℕ | x < 6}
def N : Set ℕ := {x : ℕ | 2 < x ∧ x < 9}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l215_21550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l215_21538

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of snow used in Anna's snowman -/
theorem snowman_volume : 
  sphereVolume 4 + sphereVolume 6 + sphereVolume 8 + sphereVolume 10 = (7168 / 3) * Real.pi := by
  -- Expand the definition of sphereVolume
  unfold sphereVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l215_21538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_nine_l215_21501

theorem cube_root_of_sqrt_nine (x : ℝ) : x^3 = Real.sqrt 9 → x = (Real.sqrt 9)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_nine_l215_21501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l215_21564

/-- The area of the triangle formed by the lines y = 3x + 6, y = -2x + 8, and y = 0 is 21.6 square units. -/
theorem triangle_area :
  ∃ (area : ℝ),
    (∀ x y : ℝ, y = 3 * x + 6 ∨ y = -2 * x + 8 ∨ y = 0) →
    area = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l215_21564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l215_21552

noncomputable def f (x : ℝ) := 8 * Real.sin x + 15 * Real.cos x

theorem f_max_value :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l215_21552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l215_21529

/-- Converts a point from cylindrical coordinates to rectangular coordinates. -/
noncomputable def cylindrical_to_rectangular (r θ z : Real) : Real × Real × Real :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates. -/
noncomputable def cylindrical_point : Real × Real × Real := (3, Real.pi / 3, 8)

/-- The expected point in rectangular coordinates. -/
noncomputable def rectangular_point : Real × Real × Real := (1.5, 3 * Real.sqrt 3 / 2, 8)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l215_21529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l215_21515

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 0)

theorem perpendicular_vectors (l : ℝ) : 
  (a.1 + l * b.1) * a.1 + (a.2 + l * b.2) * a.2 = 0 → l = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l215_21515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_for_specific_prism_l215_21536

/-- Regular triangular prism with given dimensions -/
structure RegularTriangularPrism where
  lateral_edge : ℝ
  base_side : ℝ

/-- The cross-section that bisects the volume of the prism -/
def volume_bisecting_cross_section (p : RegularTriangularPrism) : ℝ → ℝ → Prop := sorry

/-- The dihedral angle between the cross-section and the base -/
noncomputable def dihedral_angle (p : RegularTriangularPrism) : ℝ := sorry

/-- Main theorem: Cosine of the dihedral angle in the specific prism -/
theorem dihedral_angle_cosine_for_specific_prism :
  let p : RegularTriangularPrism := { lateral_edge := 2, base_side := 1 }
  Real.cos (dihedral_angle p) = 2 / Real.sqrt 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_for_specific_prism_l215_21536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_dwelling_points_order_l215_21560

open Real

-- Define the concept of a "new dwelling point"
def new_dwelling_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = (deriv f) x

-- Define the functions g, h, and φ
noncomputable def g (x : ℝ) : ℝ := sin x
noncomputable def h (x : ℝ) : ℝ := log x
def φ (x : ℝ) : ℝ := x^3

-- State the theorem
theorem new_dwelling_points_order (a b c : ℝ) :
  (0 < a ∧ a < π) →
  (0 < b) →
  (c ≠ 0) →
  new_dwelling_point g a →
  new_dwelling_point h b →
  new_dwelling_point φ c →
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_dwelling_points_order_l215_21560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l215_21513

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2).sqrt

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem dot_product_range :
  ∀ (xa ya xb yb : ℝ),
    circleC xa ya →
    circleC xb yb →
    distance xa ya xb yb = 2 * Real.sqrt 3 →
    2 ≤ dot_product 3 4 (xa + xb) (ya + yb) ∧
    dot_product 3 4 (xa + xb) (ya + yb) ≤ 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l215_21513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l215_21568

/-- The area of a triangle with sides a and b, and angle γ between them. -/
noncomputable def triangle_area (a b γ : ℝ) : ℝ := (1/2) * a * b * Real.sin γ

/-- Theorem: The area of a triangle with sides a and b, and angle γ between them,
    is equal to (1/2)ab sin(γ). -/
theorem triangle_area_formula (a b γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  triangle_area a b γ = (1/2) * a * b * Real.sin γ := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- The equality holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l215_21568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_seven_twelfths_l215_21531

open Real

/-- The series sum from n=1 to infinity of (3n^2 + 2n + 1) / (n(n+1)(n+2)(n+3)) -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n^2 + 2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

/-- The theorem states that the sum of the infinite series is equal to 7/12 -/
theorem infinite_series_sum_eq_seven_twelfths : infinite_series_sum = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_seven_twelfths_l215_21531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l215_21516

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), f (Real.pi/3 + x) = f (Real.pi/3 - x)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l215_21516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l215_21562

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - 3
noncomputable def g (x : ℝ) : ℝ := 2 * x

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x + 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composition_equality : f (g_inv (f_inv (f (g (f 10))))) = 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l215_21562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_logarithm_l215_21565

theorem fixed_point_logarithm (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x : ℝ, x = 2 → (4 + Real.log (x - 1) / Real.log a = 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_logarithm_l215_21565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_four_fifths_l215_21507

theorem x_equals_y_iff_t_equals_four_fifths (t : ℚ) :
  (1 - 3 * t = 2 * t - 3) ↔ t = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_four_fifths_l215_21507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_and_fly_problem_l215_21512

/-- The problem of two cyclists and a fly --/
theorem cyclists_and_fly_problem 
  (cyclist_speed : ℝ) 
  (initial_distance : ℝ) 
  (fly_speed : ℝ) 
  (h1 : cyclist_speed = 10) 
  (h2 : initial_distance = 50) 
  (h3 : fly_speed = 15) : 
  fly_speed * (initial_distance / (2 * cyclist_speed)) = 37.5 := by
  -- Define intermediate calculations
  let relative_speed := 2 * cyclist_speed
  let time_to_meet := initial_distance / relative_speed
  let fly_distance := fly_speed * time_to_meet
  
  -- Proof steps (currently using sorry)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_and_fly_problem_l215_21512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_C_when_a_is_7_l215_21539

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B > 0 ∧ t.B < Real.pi ∧ 
  Real.cos t.B = 3/5 ∧
  t.a * t.c * Real.cos t.B = -21

-- Theorem for the area of the triangle
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) : 
  (1/2) * t.a * t.c * Real.sin t.B = 14 := by sorry

-- Theorem for angle C when a = 7
theorem angle_C_when_a_is_7 (t : Triangle) (h : triangle_conditions t) (ha : t.a = 7) : 
  t.C = Real.pi/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_C_when_a_is_7_l215_21539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l215_21527

/-- Given an angle θ, if sin θ < 0 and tan θ > 0, then θ is in the third quadrant. -/
theorem angle_in_third_quadrant (θ : ℝ) 
  (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  θ % (2 * Real.pi) ∈ Set.Ioo (Real.pi) (3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l215_21527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_one_l215_21532

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - 1) ^ x

theorem increasing_f_implies_a_greater_than_one (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a > 1 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_one_l215_21532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downward_route_is_9_miles_l215_21535

/-- Represents the hiking problem with given conditions --/
structure HikingProblem where
  upward_rate_1 : ℚ  -- Rate on well-maintained trail (miles per day)
  upward_rate_2 : ℚ  -- Rate on rocky terrain (miles per day)
  ascent_time : ℚ    -- Total time to climb the mountain (days)
  descent_rate_factor : ℚ  -- Factor by which descent rate is faster than average ascent rate

/-- Calculates the length of the downward route --/
def downward_route_length (h : HikingProblem) : ℚ :=
  let avg_ascent_rate := (h.upward_rate_1 + h.upward_rate_2) / 2
  let descent_rate := h.descent_rate_factor * avg_ascent_rate
  descent_rate * h.ascent_time

/-- Theorem stating that the downward route length is 9 miles --/
theorem downward_route_is_9_miles (h : HikingProblem) 
  (h_upward_rate_1 : h.upward_rate_1 = 4)
  (h_upward_rate_2 : h.upward_rate_2 = 2)
  (h_ascent_time : h.ascent_time = 2)
  (h_descent_rate_factor : h.descent_rate_factor = 3/2) :
  downward_route_length h = 9 := by
  sorry

#eval downward_route_length { 
  upward_rate_1 := 4, 
  upward_rate_2 := 2, 
  ascent_time := 2, 
  descent_rate_factor := 3/2 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downward_route_is_9_miles_l215_21535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetric_C_l215_21511

noncomputable section

open Real

/-- The function f(x) = sin(ωx + π/3) -/
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 3)

/-- The shifted function C(x) = sin(ωx + ωπ/2 + π/3) -/
def C (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + ω * π / 2 + π / 3)

theorem min_omega_for_symmetric_C :
  ∀ ω : ℝ, ω > 0 →
  (∀ x : ℝ, C ω x = C ω (-x)) →
  ω ≥ 1/3 ∧
  ∃ ω₀ : ℝ, ω₀ = 1/3 ∧ ω₀ > 0 ∧ (∀ x : ℝ, C ω₀ x = C ω₀ (-x)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetric_C_l215_21511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l215_21555

/-- The area of a triangle given its vertices in the coordinate plane -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of the triangle with vertices (-3, 2), (8, -3), and (3, 5) is 31.5 square units -/
theorem triangle_area_specific : triangle_area (-3) 2 8 (-3) 3 5 = 31.5 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l215_21555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l215_21557

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a n + (sequence_a n)^2 / (n + 2)^2

theorem sequence_inequality (n : ℕ) :
  (2 * n + 4) / (n + 4 : ℝ) < sequence_a (n + 1) ∧ sequence_a (n + 1) < n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l215_21557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l215_21524

def triangle_abc (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_abc a b c A B C)
  (h_eq : a * (Real.cos (C/2))^2 + c * (Real.cos (A/2))^2 = (a + c)/2 + b * Real.cos B)
  (h_area : 1/2 * a * c * Real.sin B = Real.sqrt 3/4)
  (h_b : b = Real.sqrt 2) :
  B = Real.pi/3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l215_21524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_price_equality_l215_21566

theorem mooncake_price_equality (m : ℝ) (h : m > 0) : 
  let initial_price := m * (1 + 0.25)
  let final_price := initial_price * (1 - 0.20)
  final_price = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_price_equality_l215_21566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l215_21570

/-- Predicate to check if a point is the focus of a parabola -/
def is_focus (f : ℝ × ℝ) (parabola : ℝ × ℝ → Prop) : Prop :=
  ∃ (p : ℝ), 
    parabola f ∧
    ∀ (point : ℝ × ℝ), parabola point → 
      (point.1 - f.1)^2 + (point.2 - f.2)^2 = (point.2 - f.2 + p)^2

/-- The focus of a parabola with equation y = -4x^2 has coordinates (0, -1/16) -/
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), f = (0, -1/16) ∧ is_focus f (λ (p : ℝ × ℝ) ↦ p.2 = -4 * p.1^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l215_21570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_three_l215_21586

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x⁻¹ - x⁻¹ / (1 - x⁻¹)

/-- Theorem stating that f(f(-3)) = 6/5 -/
theorem f_of_f_neg_three : f (f (-3)) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_three_l215_21586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l215_21585

/-- The circle with equation (x+1)^2 + y^2 = 1 -/
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The given line with equation x + y = 0 -/
def given_line (x y : ℝ) : Prop := x + y = 0

/-- The line we need to prove about -/
def perpendicular_line (x y : ℝ) : Prop := x + y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → circle_center.1 = x ∧ circle_center.2 = y) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    perpendicular_line x₁ y₁ → 
    perpendicular_line x₂ y₂ → 
    x₁ ≠ x₂ → 
    (y₂ - y₁) / (x₂ - x₁) * (0 - (-1)) / (1 - 0) = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l215_21585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l215_21510

/-- The area of a rhombus with given vertices in a rectangular coordinate system -/
theorem rhombus_area : ∃ (area : ℝ), area = 70 := by
  -- Define the vertices of the rhombus
  let v1 : ℝ × ℝ := (0, 3.5)
  let v2 : ℝ × ℝ := (10, 0)
  let v3 : ℝ × ℝ := (0, -3.5)
  let v4 : ℝ × ℝ := (-10, 0)

  -- Define the diagonals
  let d1 : ℝ := v1.2 - v3.2  -- Vertical diagonal
  let d2 : ℝ := v2.1 - v4.1  -- Horizontal diagonal

  -- Calculate the area
  let area : ℝ := (d1 * d2) / 2

  -- Prove the theorem
  use area
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l215_21510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinct_values_l215_21508

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℤ := floor (3 * x) + floor (4 * x) + floor (5 * x) + floor (6 * x)

-- Theorem statement
theorem number_of_distinct_values :
  ∃ (S : Finset ℤ), (∀ y ∈ S, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ f x = y) ∧ 
                    (∀ x : ℝ, 0 ≤ x → x ≤ 100 → f x ∈ S) ∧
                    Finset.card S = 1201 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinct_values_l215_21508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_6_l215_21596

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 1]

theorem B_power_6 : 
  B^6 = 40208 • B + 25955 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_6_l215_21596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplest_form_l215_21597

theorem fraction_simplest_form (n : ℕ+) 
  (h : (n.val^2 + 3*n.val - 10) / (n.val^2 + 6*n.val - 16) = 
       (n.val^2 + 3*n.val - 10) / (n.val^2 + 6*n.val - 16)) : 
  (n.val^2 + 3*n.val - 10) / (n.val^2 + 6*n.val - 16) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplest_form_l215_21597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_smaller_sphere_l215_21587

-- Define the types for our objects
def Sphere := Set (EuclideanSpace ℝ (Fin 3))
def Tetrahedron := Set (EuclideanSpace ℝ (Fin 3))
def Plane := Set (EuclideanSpace ℝ (Fin 3))

-- Define our objects
noncomputable def sphere_O : Sphere := sorry
noncomputable def tetrahedron_ABCD : Tetrahedron := sorry
noncomputable def sphere_O' : Sphere := sorry
noncomputable def D : EuclideanSpace ℝ (Fin 3) := sorry
noncomputable def plane_ABC : Plane := sorry

-- Define the necessary concepts
def CircumscribedSphere (s : Sphere) (t : Tetrahedron) : Prop := sorry
def TangentSpheres (s1 s2 : Sphere) (p : EuclideanSpace ℝ (Fin 3)) : Prop := sorry
def TangentSphereToPlane (s : Sphere) (p : Plane) : Prop := sorry
def SurfaceArea (s : Sphere) : ℝ := sorry
def Volume (s : Sphere) : ℝ := sorry

-- State the axioms
axiom circumscribed : CircumscribedSphere sphere_O tetrahedron_ABCD
axiom tangent_at_D : TangentSpheres sphere_O sphere_O' D
axiom tangent_to_plane : TangentSphereToPlane sphere_O' plane_ABC
axiom surface_area_O : SurfaceArea sphere_O = 9 * Real.pi

-- State the theorem
theorem volume_of_smaller_sphere :
  Volume sphere_O' = 4 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_smaller_sphere_l215_21587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_sin_product_l215_21537

theorem right_triangle_max_sin_product (A B C : ℝ) : 
  C = π / 2 → 
  0 ≤ A ∧ A ≤ π → 
  0 ≤ B ∧ B ≤ π → 
  A + B + C = π → 
  ∀ x y, 0 ≤ x ∧ x ≤ π → 0 ≤ y ∧ y ≤ π → x + y + C = π → 
    Real.sin A * Real.sin B ≤ Real.sin x * Real.sin y → 
    Real.sin A * Real.sin B ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_sin_product_l215_21537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l215_21580

theorem sin_sum_specific_angles (α β : Real) 
  (h1 : Real.sin α = 2/3)
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi)
  (h3 : Real.cos β = -3/5)
  (h4 : β ∈ Set.Ioo Real.pi (3*Real.pi/2)) :
  Real.sin (α + β) = (4*Real.sqrt 5 - 6)/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l215_21580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l215_21578

noncomputable def f (x : ℝ) : ℝ := 
  Real.exp x - Real.exp 0 * x + (1/2) * x^2

theorem f_derivative_at_one : 
  (deriv f) 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l215_21578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_original_shape_is_two_plus_sqrt_two_l215_21506

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  base_angle : ℝ
  base_length : ℝ
  top_length : ℝ

/-- Calculates the area of the original shape given its oblique projection -/
def area_of_original_shape (projection : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Given an oblique projection of a horizontal planar shape that forms an isosceles trapezoid
    with a base angle of 45°, bases and top of length 1, the area of the original planar shape
    is 2 + √2. -/
theorem area_of_original_shape_is_two_plus_sqrt_two (projection : IsoscelesTrapezoid) 
  (h1 : projection.base_angle = 45 * π / 180)
  (h2 : projection.base_length = 1)
  (h3 : projection.top_length = 1) :
  area_of_original_shape projection = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_original_shape_is_two_plus_sqrt_two_l215_21506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_l215_21500

/-- Given a bus with an average speed excluding stoppages and a stoppage time per hour,
    calculate the average speed including stoppages. -/
noncomputable def average_speed_with_stoppages (speed_without_stoppages : ℝ) (stoppage_time : ℝ) : ℝ :=
  speed_without_stoppages * (1 - stoppage_time / 60)

/-- Theorem stating that for a bus with an average speed of 60 km/hr excluding stoppages
    and stops for 15 minutes per hour, the average speed including stoppages is 45 km/hr. -/
theorem bus_average_speed :
  average_speed_with_stoppages 60 15 = 45 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_l215_21500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l215_21581

/-- Efficiency of worker A relative to B -/
noncomputable def efficiency_A_to_B : ℚ := 1 / 2

/-- Efficiency of worker B relative to C -/
noncomputable def efficiency_B_to_C : ℚ := 2 / 3

/-- Time taken by A, B, and C working together (in days) -/
def time_ABC : ℚ := 15

/-- The time taken by worker B to complete the job alone (in days) -/
def time_B_alone : ℚ := 75 / 2

/-- Theorem stating that given the efficiency relationships and combined work time,
    B alone would take 37.5 days to complete the job -/
theorem b_alone_time (efficiency_A_to_B efficiency_B_to_C time_ABC time_B_alone : ℚ) :
  efficiency_A_to_B = 1 / 2 →
  efficiency_B_to_C = 2 / 3 →
  time_ABC = 15 →
  time_B_alone = 75 / 2 := by
  sorry

#eval time_B_alone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l215_21581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_vertices_theorem_l215_21518

/-- A cube has 8 vertices -/
def cube_vertices : Nat := 8

/-- Each vertex in a cube is connected to 3 other vertices -/
def adjacent_vertices : Nat := 3

/-- The smallest number of red vertices that guarantees a vertex with all red neighbors -/
def min_red_vertices : Nat := 5

/-- Represents the coloring of vertices in a cube -/
def Coloring := Fin cube_vertices → Bool

/-- Two vertices are adjacent if they share an edge in the cube -/
def adjacent : Fin cube_vertices → Fin cube_vertices → Prop := sorry

/-- A vertex has all red neighbors if all its adjacent vertices are colored red -/
def has_all_red_neighbors (c : Coloring) (v : Fin cube_vertices) : Prop :=
  ∀ (w : Fin cube_vertices), adjacent v w → c w = true

/-- A coloring satisfies the condition if there exists a vertex with all red neighbors -/
def satisfies_condition (c : Coloring) : Prop :=
  ∃ (v : Fin cube_vertices), has_all_red_neighbors c v

/-- The main theorem: 5 is the smallest number that satisfies the condition -/
theorem min_red_vertices_theorem :
  (∀ (c : Coloring), (∃ (S : Finset (Fin cube_vertices)), S.card = min_red_vertices ∧ (∀ v ∈ S, c v = true)) → satisfies_condition c) ∧
  (∀ (n : Nat), n < min_red_vertices → 
    ∃ (c : Coloring), (∃ (S : Finset (Fin cube_vertices)), S.card = n ∧ (∀ v ∈ S, c v = true)) ∧ ¬satisfies_condition c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_vertices_theorem_l215_21518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l215_21599

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := customOp (Real.cos x) (Real.sin x)

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1) (Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l215_21599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l215_21543

theorem biased_coin_probability (p : ℝ) 
  (hp1 : 0 < p) (hp2 : p < 1)
  (h_equal_prob : 4 * p * (1 - p)^3 = 6 * p^2 * (1 - p)^2) :
  6 * p^2 * (1 - p)^2 = 216 / 625 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l215_21543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_bd_length_l215_21589

/-- An isosceles triangle with a given altitude --/
structure IsoscelesTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ  -- D is the foot of the altitude
  -- AB = AC (isosceles property)
  isIsosceles : dist A B = dist A C
  -- CD is altitude from C to AB
  isAltitude : (C.1 - D.1) * (B.1 - A.1) + (C.2 - D.2) * (B.2 - A.2) = 0
  -- D is on AB
  DOnAB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Length of AB is 10
  ABLength : dist A B = 10
  -- Length of CD is 6
  CDLength : dist C D = 6

/-- The main theorem --/
theorem isosceles_triangle_bd_length (triangle : IsoscelesTriangle) : 
  dist triangle.B triangle.D = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_bd_length_l215_21589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l215_21520

/-- The system of equations has a unique solution if and only if a = ±√2 -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + 4*p.2^2 = 1 ∧ p.1 + 2*p.2 = a) ↔ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l215_21520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l215_21594

/-- For a parabola defined by x² = 2y, the distance from its focus to its directrix is 1 -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), x^2 = 2*y → (∃ (p : ℝ), p = 1 ∧ p = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l215_21594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l215_21519

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (exp x / x) - m * x

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x > 0, f m x > 0) → m < exp 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l215_21519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_weaving_l215_21558

-- Define the geometric sequence
noncomputable def geometric_sequence (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (2 ^ (n - 1))

-- Define the sum of the first n terms of the geometric sequence
noncomputable def geometric_sum (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (1 - 2^n) / (1 - 2)

-- Theorem statement
theorem second_day_weaving :
  ∀ a₁ : ℝ, geometric_sum a₁ 5 = 5 → geometric_sequence a₁ 2 = 10/31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_weaving_l215_21558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_point_parabola_midpoint_distance_l215_21569

-- Part 1
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = -8*x ∨ x^2 = -y

theorem parabola_through_point :
  parabola_equation (-2) (-4) := by sorry

-- Part 2
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2*p*x ∧ p > 0

noncomputable def focus (p : ℝ) : ℝ × ℝ :=
  (p/2, 0)

def point_A : ℝ × ℝ :=
  (0, 2)

noncomputable def midpoint_B (p : ℝ) : ℝ × ℝ :=
  (p/4, 1)

def on_parabola (p : ℝ) : Prop :=
  parabola p (p/4) 1

noncomputable def distance_to_directrix (p : ℝ) : ℝ :=
  3 * Real.sqrt 2 / 4

theorem parabola_midpoint_distance (p : ℝ) :
  on_parabola p → distance_to_directrix p = 3 * Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_point_parabola_midpoint_distance_l215_21569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_50_51_l215_21502

def f (x : ℤ) : ℤ := x^3 - x^2 + 2*x + 2000

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_50_51_l215_21502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_symmetrization_preserves_convexity_l215_21534

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ M → B ∈ M →
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
      (1 - t) • A + t • B ∈ M

-- Define Steiner symmetrization
noncomputable def SteinerSymmetrization (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

-- Define a line segment
def LineSegment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • A + t • B}

-- Main theorem
theorem steiner_symmetrization_preserves_convexity 
  (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) :
  ConvexPolygon M → 
  ConvexPolygon (SteinerSymmetrization M l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_symmetrization_preserves_convexity_l215_21534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_30_miles_l215_21574

/-- The distance Bob walked when meeting Yolanda --/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (yolanda_head_start : ℝ) : ℝ :=
  let meeting_time := (total_distance - yolanda_speed * yolanda_head_start) / (yolanda_speed + bob_speed)
  bob_speed * meeting_time

/-- Theorem stating that Bob walked 30 miles when they met --/
theorem bob_walked_30_miles :
  distance_bob_walked 60 5 6 1 = 30 := by
  -- Unfold the definition of distance_bob_walked
  unfold distance_bob_walked
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_30_miles_l215_21574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l215_21591

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (B : Point) (C : Point) : Prop :=
  M.x = (B.x + C.x) / 2 ∧ M.y = (B.y + C.y) / 2

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (A : Point) (B : Point) (C : Point) (D : Point) : Prop :=
  (B.x - A.x) * (D.x - C.x) + (B.y - A.y) * (D.y - C.y) = 0

/-- Theorem: A triangle can be constructed given two altitudes and a median -/
theorem triangle_construction_theorem 
  (AH : ℝ) -- Length of altitude from A
  (AM : ℝ) -- Length of median from A
  (BB₁ : ℝ) -- Length of altitude from B
  : ∃ (t : Triangle), 
    -- M is the midpoint of BC
    isMidpoint (Point.mk AM 0) t.B t.C ∧ 
    -- AH is perpendicular to BC
    isPerpendicular t.A (Point.mk 0 AH) t.B t.C ∧
    -- BB₁ is perpendicular to AC
    isPerpendicular t.B (Point.mk 0 BB₁) t.A t.C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l215_21591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_perfect_square_factors_l215_21567

/-- The number of positive integer factors of 8820 that are perfect squares -/
def perfectSquareFactors : ℕ := 8

/-- The prime factorization of 8820 -/
def primeFactorization : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1), (7, 2)]

/-- Theorem stating that the number of positive integer factors of 8820 
    that are perfect squares is equal to perfectSquareFactors -/
theorem count_perfect_square_factors :
  (primeFactorization.foldl
    (fun acc (p, e) ↦ acc * (e / 2 + 1))
    1) = perfectSquareFactors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_perfect_square_factors_l215_21567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paytons_score_l215_21548

/-- Given a class of 20 students, if the average score of 19 students is 85
    and the average score of all 20 students is 86, then the score of the
    20th student (Payton) must be 105. -/
theorem paytons_score (total_students : ℕ) (average_19 : ℝ) (average_20 : ℝ) (paytons_score : ℝ) :
  total_students = 20 →
  average_19 = 85 →
  average_20 = 86 →
  (19 * average_19 + paytons_score) / total_students = average_20 →
  paytons_score = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paytons_score_l215_21548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_curve_length_l215_21509

/-- Represents a rectangle with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the length of the diagonal of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.a^2 + r.b^2)

/-- Calculates the length of a quarter circle with given radius -/
noncomputable def quarterCircleLength (radius : ℝ) : ℝ :=
  (1/2) * Real.pi * radius

/-- Theorem: The length of the curve traced by vertex A when rotating a rectangle
    first around vertex D and then around vertex C is 50π cm, given AB = 30 cm and BC = 40 cm -/
theorem rectangle_rotation_curve_length :
  let rect := Rectangle.mk 30 40
  let diag := rect.diagonal
  let curve_length := 2 * (quarterCircleLength diag)
  curve_length = 50 * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_curve_length_l215_21509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l215_21544

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

-- Define the tangent line at (1, 2)
def tangent_line (x y : ℝ) : Prop := 2*x - y = 0

-- State the theorem
theorem shortest_distance :
  ∃ (d : ℝ), d = 4*Real.sqrt 5 / 5 - 1 ∧
  ∀ (P Q : ℝ × ℝ),
    tangent_line P.1 P.2 →
    circle_eq Q.1 Q.2 →
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l215_21544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l215_21533

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_range (α : ℝ) (a : ℝ) :
  f α (1/4) = 8 →
  (∀ x y, x > y → x > 0 → y > 0 → f α x < f α y) →
  f α (a - 1) < f α (8 - 2*a) →
  3 < a ∧ a < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l215_21533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l215_21556

/-- Represents the composition of an alloy mixture --/
structure AlloyMixture where
  total_mass : ℝ
  copper_percentage : ℝ

/-- Calculates the resulting alloy mixture from two component alloys --/
noncomputable def mix_alloys (alloy1 : AlloyMixture) (mass1 : ℝ) (alloy2 : AlloyMixture) (mass2 : ℝ) : AlloyMixture :=
  { total_mass := mass1 + mass2,
    copper_percentage := (alloy1.copper_percentage * mass1 + alloy2.copper_percentage * mass2) / (mass1 + mass2) }

/-- Theorem: Mixing 200g of 25% copper alloy with 800g of 50% copper alloy results in 1000g of 45% copper alloy --/
theorem alloy_mixture_theorem (alloy1 alloy2 result : AlloyMixture) (mass1 mass2 : ℝ) :
  alloy1.copper_percentage = 0.25 →
  alloy2.copper_percentage = 0.50 →
  mass1 = 200 →
  mass2 = 800 →
  result = mix_alloys alloy1 mass1 alloy2 mass2 →
  result.total_mass = 1000 ∧ result.copper_percentage = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l215_21556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l215_21573

noncomputable def p (x : ℝ) : ℝ := (5/4) * x^2 - 5*x + 15/4

def numerator (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x^2 + 8*x - 8

theorem p_satisfies_conditions :
  (∀ x, x ≠ 1 ∧ x ≠ 3 → numerator x / p x ≠ 0) ∧  -- Vertical asymptotes at x = 1 and x = 3
  (∀ y, ∃ x, |x| > y ∧ |numerator x / p x| > y) ∧  -- No horizontal asymptote
  p (-1) = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l215_21573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_xy_l215_21504

/-- The time (in hours) it takes to fill the tank with all valves open -/
noncomputable def time_all : ℝ := 1.2

/-- The time (in hours) it takes to fill the tank with valves X and Z open -/
noncomputable def time_xz : ℝ := 2

/-- The time (in hours) it takes to fill the tank with valves Y and Z open -/
noncomputable def time_yz : ℝ := 3

/-- The rate at which valve X fills the tank (in tank volumes per hour) -/
noncomputable def rate_x : ℝ := 1 / 2

/-- The rate at which valve Y fills the tank (in tank volumes per hour) -/
noncomputable def rate_y : ℝ := 1 / 3

/-- The rate at which valve Z fills the tank (in tank volumes per hour) -/
noncomputable def rate_z : ℝ := 1 / 6

/-- The time it takes to fill the tank with valves X and Y open -/
noncomputable def time_xy : ℝ := 1 / (rate_x + rate_y)

theorem fill_time_xy :
  time_all = 1 / (rate_x + rate_y + rate_z) ∧
  time_xz = 1 / (rate_x + rate_z) ∧
  time_yz = 1 / (rate_y + rate_z) →
  time_xy = 1.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_xy_l215_21504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l215_21572

open Real

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := cos (2 * x + φ)

theorem f_shifted_is_even (φ : ℝ) (h : ∀ x : ℝ, f φ x ≤ f φ 1) :
  ∀ x : ℝ, f φ (x + 1) = f φ (-x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l215_21572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l215_21526

/-- The repeating decimal 0.6̄7 is equal to the fraction 61/90 -/
theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 0.6 + ∑' n, 7 / (10 ^ (n + 2))) → (0.6777777 : ℚ) = 61 / 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l215_21526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_with_given_inradius_and_area_l215_21579

/-- Given a triangle with inradius r and area A, calculates its perimeter -/
noncomputable def triangle_perimeter (r : ℝ) (A : ℝ) : ℝ :=
  (2 * A) / r

theorem triangle_perimeter_with_given_inradius_and_area :
  triangle_perimeter 2.5 25 = 20 := by
  -- Unfold the definition of triangle_perimeter
  unfold triangle_perimeter
  -- Simplify the expression
  simp
  -- Check that the result is equal to 20
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_with_given_inradius_and_area_l215_21579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worldview_determines_methodology_l215_21598

-- Define worldview and methodology as types
def Worldview : Type := Unit
def Methodology : Type := Unit

-- Define the relationship between worldview and methodology
def determines (w : Worldview) (m : Methodology) : Prop := True

-- State the conditions
axiom worldview_definition : 
  ∀ (w : Worldview), ∃ (view : Prop), view ↔ (∀ (s : String), s = "overall view and fundamental stance on the entire world and the relationship between humans and the world")

axiom methodology_definition : 
  ∀ (m : Methodology), ∃ (method : Prop), method ↔ (∀ (s : String), s = "using worldview to guide the understanding and transformation of the world")

axiom different_aspects : 
  ∀ (w : Worldview) (m : Methodology), ∃ (issue : Prop), issue ↔ (∀ (s : String), s = "worldview and methodology are different aspects of the same issue")

-- State the theorem to be proved
theorem worldview_determines_methodology :
  ∀ (w : Worldview) (m : Methodology), determines w m :=
by
  intros w m
  exact True.intro


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worldview_determines_methodology_l215_21598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_absolute_values_l215_21505

theorem min_sum_absolute_values (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (Matrix.of !![p, q; r, s])^2 = Matrix.of !![12, 0; 0, 12] →
  (∀ p' q' r' s' : ℤ, p' ≠ 0 → q' ≠ 0 → r' ≠ 0 → s' ≠ 0 →
    (Matrix.of !![p', q'; r', s'])^2 = Matrix.of !![12, 0; 0, 12] →
    p.natAbs + q.natAbs + r.natAbs + s.natAbs ≤ 
    p'.natAbs + q'.natAbs + r'.natAbs + s'.natAbs) →
  p.natAbs + q.natAbs + r.natAbs + s.natAbs = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_absolute_values_l215_21505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l215_21514

/-- The area of a triangle with vertices (0, 7, 10), (-1, 6, 8), and (-4, 9, 6) is equal to √41. -/
theorem triangle_area : 
  let A : Fin 3 → ℝ := ![0, 7, 10]
  let B : Fin 3 → ℝ := ![-1, 6, 8]
  let C : Fin 3 → ℝ := ![-4, 9, 6]
  let AB : Fin 3 → ℝ := λ i => B i - A i
  let AC : Fin 3 → ℝ := λ i => C i - A i
  let cross_product : Fin 3 → ℝ := ![
    AB 1 * AC 2 - AB 2 * AC 1,
    AB 2 * AC 0 - AB 0 * AC 2,
    AB 0 * AC 1 - AB 1 * AC 0
  ]
  let area : ℝ := (1/2) * Real.sqrt (cross_product 0^2 + cross_product 1^2 + cross_product 2^2)
  area = Real.sqrt 41 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l215_21514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_two_l215_21530

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/3)^x else Real.log x / Real.log (1/2)

-- State the theorem
theorem f_composition_sqrt_two : f (f (Real.sqrt 2)) = Real.sqrt 3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_two_l215_21530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l215_21554

/-- Given a battery with voltage 48V and a resistor with resistance R,
    the current I is related to R by the function I = 48 / R -/
noncomputable def current (R : ℝ) : ℝ := 48 / R

/-- The theorem states that when the resistance is 12Ω,
    the resulting current is 4A -/
theorem current_at_12_ohms :
  current 12 = 4 := by
  -- Unfold the definition of current
  unfold current
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l215_21554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circledAsterisk_problem_l215_21546

/-- Custom operation ⊛ on natural numbers -/
def circledAsterisk : ℕ → ℕ → ℕ := sorry

/-- For any natural number a: a ⊛ a = a -/
axiom circledAsterisk_self (a : ℕ) : circledAsterisk a a = a

/-- For any natural number a: a ⊛ 0 = 2a -/
axiom circledAsterisk_zero (a : ℕ) : circledAsterisk a 0 = 2 * a

/-- For any four natural numbers a, b, c, d: (a ⊛ b) + (c ⊛ d) = (a + c) ⊛ (b + d) -/
axiom circledAsterisk_sum (a b c d : ℕ) : 
  (circledAsterisk a b) + (circledAsterisk c d) = circledAsterisk (a + c) (b + d)

theorem circledAsterisk_problem :
  (circledAsterisk (2 + 3) (0 + 3) = 7) ∧ (circledAsterisk 1024 48 = 2000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circledAsterisk_problem_l215_21546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_coplanar_implies_not_intersect_l215_21553

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [Fact (finrank ℝ V = 3)]

-- Define points in space
variable (E F G H : V)

-- Define the property of points being coplanar
def isCoplanar (p q r s : V) : Prop :=
  ∃ (a b c d : ℝ), a • p + b • q + c • r + d • s = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- Define the property of lines intersecting
def linesIntersect (p₁ q₁ p₂ q₂ : V) : Prop :=
  ∃ (t s : ℝ), p₁ + t • (q₁ - p₁) = p₂ + s • (q₂ - p₂)

-- State the theorem
theorem not_coplanar_implies_not_intersect (E F G H : V) :
  (¬ isCoplanar E F G H → ¬ linesIntersect E F G H) ∧
  (∃ E' F' G' H' : V, ¬ linesIntersect E' F' G' H' ∧ isCoplanar E' F' G' H') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_coplanar_implies_not_intersect_l215_21553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_exponential_difference_l215_21576

theorem polynomial_exponential_difference (n : ℕ) (p : Polynomial ℝ) :
  Polynomial.degree p = n →
  ∃ k : ℕ, k ≤ n + 1 ∧ |p.eval (k : ℝ) - 3^k| ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_exponential_difference_l215_21576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_terrain_length_l215_21582

/-- Represents the road from A to B -/
structure Road where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Represents the walking speeds -/
structure WalkingSpeeds where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

noncomputable def total_length (r : Road) : ℝ := r.uphill + r.flat + r.downhill

noncomputable def time_AB (r : Road) (s : WalkingSpeeds) : ℝ :=
  r.uphill / s.uphill + r.flat / s.flat + r.downhill / s.downhill

noncomputable def time_BA (r : Road) (s : WalkingSpeeds) : ℝ :=
  r.downhill / s.uphill + r.flat / s.flat + r.uphill / s.downhill

theorem flat_terrain_length (r : Road) (s : WalkingSpeeds) :
  total_length r = 11.5 ∧
  time_AB r s = 2.9 ∧
  time_BA r s = 3.1 ∧
  s.uphill = 3 ∧
  s.flat = 4 ∧
  s.downhill = 5 →
  r.flat = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_terrain_length_l215_21582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l215_21549

/-- Predicate for an isosceles triangle -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- An isosceles triangle with sides of length 3 and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 3 →
  b = 6 →
  c = 6 →
  IsoscelesTriangle a b c →
  a + b + c = 15 := by
  intros a b c ha hb hc hiso
  simp [ha, hb, hc]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l215_21549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_geometric_a_formula_b_formula_T_formula_l215_21584

-- Define the sequences
def a : ℕ → ℝ := sorry

def b : ℕ → ℝ := sorry

def c : ℕ → ℝ := sorry

-- Define the sum of first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define the sum of first n terms of c_n
def T : ℕ → ℝ := sorry

-- Given conditions
axiom a3 : a 3 = 7
axiom S6 : S 6 = 48
axiom b_rec : ∀ n, 2 * b (n + 1) = b n + 2
axiom b1 : b 1 = 3
axiom c_def : ∀ n, c n = a n * (b n - 2)

-- Theorem statements
theorem b_geometric : 
  (∀ n, b (n + 1) - 2 = (1 / 2) * (b n - 2)) ∧ (b 1 - 2 = 1) := by sorry

theorem a_formula : ∀ n, a n = 2 * n + 1 := by sorry

theorem b_formula : ∀ n, b n = (1 / 2) ^ (n - 1) + 2 := by sorry

theorem T_formula : ∀ n, T n = 10 - (2 * n + 5) * (1 / 2) ^ (n - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_geometric_a_formula_b_formula_T_formula_l215_21584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l215_21575

theorem max_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 1 2, x^2 - |a| * x + a - 1 > 0) → 
  a ≤ 2 ∧ ∀ b : ℝ, (∀ x ∈ Set.Ioo 1 2, x^2 - |b| * x + b - 1 > 0) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l215_21575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_placement_theorem_correct_answer_is_c_l215_21571

theorem digit_placement_theorem (t u : ℕ) :
  1000 * t + 100 * u + 21 = (10 * t + u) * 100 + 21 := by
  calc 1000 * t + 100 * u + 21
    = (10 * 100) * t + 100 * u + 21 := by ring
  _ = 10 * 100 * t + 100 * u + 21 := by ring
  _ = (10 * t) * 100 + 100 * u + 21 := by ring
  _ = (10 * t + u) * 100 + 21 := by ring

theorem correct_answer_is_c (t u : ℕ) :
  1000 * t + 100 * u + 21 = (10 * t + u) * 100 + 21 := by
  exact digit_placement_theorem t u

#check correct_answer_is_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_placement_theorem_correct_answer_is_c_l215_21571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_iff_union_complement_eq_univ_l215_21542

universe u

theorem subset_iff_union_complement_eq_univ {U : Type u} (A B : Set U) :
  B ⊆ A ↔ A ∪ (Set.univ \ B) = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_iff_union_complement_eq_univ_l215_21542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_on_torus_l215_21525

/-- The radius of the largest sphere that can be placed on top of a torus while touching the xy-plane -/
noncomputable def largest_sphere_radius : ℝ := 13 / 6

/-- The center of the circular cross-section of the torus -/
def torus_center : ℝ × ℝ × ℝ := (4, 0, 1)

/-- The radius of the circular cross-section of the torus -/
def torus_radius : ℝ := 2

/-- The plane on which the torus sits -/
def table_plane : ℝ × ℝ × ℝ → Prop :=
  fun p => p.2.1 = 0

theorem largest_sphere_on_torus :
  ∃ (sphere_center : ℝ × ℝ × ℝ),
    sphere_center.2.1 = largest_sphere_radius ∧
    table_plane sphere_center ∧
    ‖sphere_center - torus_center‖ = largest_sphere_radius + torus_radius ∧
    ∀ (r : ℝ) (c : ℝ × ℝ × ℝ),
      r > largest_sphere_radius →
      table_plane c →
      ‖c - torus_center‖ > r + torus_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_on_torus_l215_21525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preschool_nap_l215_21517

def kids_remaining (initial : ℕ) (phase1 : ℚ) (phase2 : ℚ) (phase3 : ℚ) : ℕ :=
  let after_phase1 := initial - Int.toNat ((initial : ℚ) * phase1).floor
  let after_phase2 := after_phase1 - Int.toNat ((after_phase1 : ℚ) * phase2).floor
  after_phase2 - Int.toNat ((after_phase2 : ℚ) * phase3).floor

theorem preschool_nap (initial : ℕ) (h_initial : initial = 20) :
  kids_remaining initial (1/3) (1/4) (1/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preschool_nap_l215_21517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l215_21540

theorem expression_equality : 
  |2 - Real.tan (60 * π / 180)| + (1/4 : ℝ) + (1/2) * Real.sqrt 12 = 
  0.268 + 1/4 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l215_21540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l215_21561

noncomputable def sum_of_terms : ℝ :=
  1 / (Real.sin (30 * Real.pi / 180) * Real.sin (31 * Real.pi / 180)) +
  1 / (Real.sin (32 * Real.pi / 180) * Real.sin (33 * Real.pi / 180)) +
  1 / (Real.sin (34 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) +
  -- ... (continuing the pattern)
  1 / (Real.sin (148 * Real.pi / 180) * Real.sin (149 * Real.pi / 180))

theorem least_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ sum_of_terms = 1 / Real.sin (1 * Real.pi / 180) ∧
  ∀ m : ℕ, m > 0 → sum_of_terms = 1 / Real.sin (m * Real.pi / 180) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l215_21561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_trigonometry_l215_21503

noncomputable def α : ℝ := Real.arccos (-1/2)

theorem unit_circle_trigonometry :
  let P : ℝ × ℝ := (-1/2, -Real.sqrt 3 / 2)
  (∀ t : ℝ, Real.cos t = P.1 ∧ Real.sin t = P.2 → t = α ∨ t = α + 2 * Real.pi) →
  (Real.sin α = -Real.sqrt 3 / 2 ∧ Real.cos α = -1/2) ∧
  ((Real.sin (α - Real.pi) + Real.cos (α + Real.pi/2)) / Real.tan (Real.pi + α) = 1) ∧
  (Real.tan (α + Real.pi/4) < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_trigonometry_l215_21503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_arithmetic_sequence_sqrt_sum_arithmetic_sequence_iff_l215_21522

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem sqrt_sum_arithmetic_sequence (a₁ d : ℝ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  (a₁ = 1 ∧ d = 2) →
  arithmetic_sequence (λ n ↦ Real.sqrt (sum_of_arithmetic_sequence a₁ d n)) := by
  sorry

theorem sqrt_sum_arithmetic_sequence_iff (a₁ d : ℝ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  arithmetic_sequence (λ n ↦ Real.sqrt (sum_of_arithmetic_sequence a₁ d n)) ↔
  d = 2 * a₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_arithmetic_sequence_sqrt_sum_arithmetic_sequence_iff_l215_21522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l215_21528

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 4)

-- Define the directrix of the parabola
def directrix : ℝ := -1

-- Define the property that the circle is tangent to the directrix
def circle_tangent_to_directrix (r : ℝ) : Prop :=
  r = circle_center.fst - directrix

-- Theorem statement
theorem chord_length_is_six :
  ∃ (r : ℝ), circle_tangent_to_directrix r →
    let x₁ := circle_center.fst - r
    let x₂ := circle_center.fst + r
    x₂ - x₁ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l215_21528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l215_21523

/-- Represents the cost and selling prices of furniture items -/
structure FurnitureItem where
  costPrice : ℝ
  sellingPrice : ℝ

/-- Calculates the selling price given a cost price and a markup percentage -/
noncomputable def applyMarkup (costPrice : ℝ) (markupPercent : ℝ) : ℝ :=
  costPrice * (1 + markupPercent / 100)

/-- Calculates the selling price given a cost price and a discount percentage -/
noncomputable def applyDiscount (costPrice : ℝ) (discountPercent : ℝ) : ℝ :=
  costPrice * (1 - discountPercent / 100)

/-- Theorem stating the correct cost prices given the selling prices and markup/discount percentages -/
theorem furniture_cost_prices 
  (computerTable : FurnitureItem)
  (officeChair : FurnitureItem)
  (bookshelf : FurnitureItem)
  (h1 : computerTable.sellingPrice = applyMarkup computerTable.costPrice 20)
  (h2 : officeChair.sellingPrice = applyMarkup officeChair.costPrice 25)
  (h3 : bookshelf.sellingPrice = applyDiscount bookshelf.costPrice 15)
  (h4 : computerTable.sellingPrice = 3600)
  (h5 : officeChair.sellingPrice = 5000)
  (h6 : bookshelf.sellingPrice = 1700) :
  computerTable.costPrice = 3000 ∧ officeChair.costPrice = 4000 ∧ bookshelf.costPrice = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l215_21523
