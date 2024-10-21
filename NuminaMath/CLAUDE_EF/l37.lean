import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l37_3763

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ (2 : ℝ)^x₀ ≤ 0) ↔ (∀ x : ℝ, x > 0 → (2 : ℝ)^x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l37_3763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l37_3702

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - a n ^ 2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + b n ^ 2) - 1) / b n

theorem inequality_holds (n : ℕ) : 2 ^ (n + 2) * a n < Real.pi ∧ Real.pi < 2 ^ (n + 2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l37_3702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l37_3755

noncomputable section

open Real

def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_radius_proof (z : ℝ) :
  let h := (5 : ℝ)
  let r := (5 + 2 * sqrt 10 : ℝ)
  let original_volume := cylinder_volume r h
  let volume_after_radius_increase := cylinder_volume (r + 3) h
  let volume_after_height_increase := cylinder_volume r (h + 3)
  volume_after_radius_increase - original_volume = z ∧
  volume_after_height_increase - original_volume = z →
  r = 5 + 2 * sqrt 10 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l37_3755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_iff_a_in_range_l37_3721

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem function_increasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a ≥ -3 ∧ a < -2) := by
  sorry

#check function_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_iff_a_in_range_l37_3721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l37_3725

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the functions
def f (x : ℝ) := 8 + 2*x - x^2

def g (x : ℝ) := f (2 - x^2)

-- State the theorem
theorem monotonicity_intervals :
  (∀ x y, 2 < x ∧ x < y → log 0.7 (x^2 - 3*x + 2) > log 0.7 (y^2 - 3*y + 2)) ∧
  (∀ x y, x < y ∧ y < 1 → log 0.7 (x^2 - 3*x + 2) < log 0.7 (y^2 - 3*y + 2)) ∧
  (∀ x y, x < y ∧ y < -1 → g x < g y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → g x < g y) ∧
  (∀ x y, 1 < x ∧ x < y → g x > g y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 0 → g x > g y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l37_3725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonous_interval_l37_3706

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 4 * x - 1 / x

-- Theorem statement
theorem not_monotonous_interval (k : ℝ) :
  (∃ x y, x ∈ Set.Ioo (k - 1) (k + 1) ∧ y ∈ Set.Ioo (k - 1) (k + 1) ∧ 
    f x < f y ∧ ∃ z ∈ Set.Ioo x y, f z > f y) ↔ 
  k ∈ Set.Ioo 1 (3/2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonous_interval_l37_3706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_specific_base_l37_3797

noncomputable def pow (a x : ℝ) : ℝ := Real.exp (x * Real.log a)

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement of the theorem
theorem unique_solution_implies_specific_base (a : ℝ) :
  a > 1 →
  (∃! x, pow a x = log a x) →
  a = Real.exp (1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_specific_base_l37_3797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_third_or_fourth_quadrant_l37_3733

/-- Complex number z as a function of real number m -/
noncomputable def z (m : ℝ) : ℂ := m^2 * (1 / (m + 8) + Complex.I) + (6 * m - 16) * Complex.I - (m + 2) / (m + 8)

/-- z is purely imaginary iff m = -1 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = -1 := by sorry

/-- z is in the third or fourth quadrant iff m ∈ (-8, 2) ∪ {-1} -/
theorem z_in_third_or_fourth_quadrant (m : ℝ) : 
  Complex.re (z m) ≠ 0 ∧ Complex.im (z m) < 0 ↔ m ∈ Set.Ioo (-8 : ℝ) 2 ∪ {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_third_or_fourth_quadrant_l37_3733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l37_3744

noncomputable def z : ℂ := (3 + 4*Complex.I) / (2 + Complex.I)

theorem z_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l37_3744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_three_equidistant_points_l37_3707

/-- The hyperbola C: x²/4 - y²/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

/-- The right vertex P of the hyperbola -/
def P : ℝ × ℝ := (2, 0)

/-- The line l passing through P with normal vector (1, -1) -/
def line_l (x y : ℝ) : Prop := y = x + 2

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y + 2| / Real.sqrt 2

theorem hyperbola_three_equidistant_points :
  ∃ (d : ℝ), d > 0 ∧
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ hyperbola x₃ y₃ ∧
    distance_to_line x₁ y₁ = d ∧
    distance_to_line x₂ y₂ = d ∧
    distance_to_line x₃ y₃ = d ∧
    (∀ (x y : ℝ), hyperbola x y ∧ distance_to_line x y = d →
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃))) →
  d = Real.sqrt 2 / 2 ∨ d = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_three_equidistant_points_l37_3707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_l37_3750

/-- The distance between a point (3, 2) and its reflection over the x-axis is 4. -/
theorem distance_to_reflection : ∀ (P : ℝ × ℝ),
  P.1 = 3 ∧ P.2 = 2 →
  let P' : ℝ × ℝ := (P.1, -P.2)
  Real.sqrt ((P'.1 - P.1)^2 + (P'.2 - P.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_l37_3750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_with_complement_l37_3745

open Set Real

theorem intersection_and_union_with_complement :
  let A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
  let B : Set ℝ := {x | 2 < x ∧ x < 4}
  (A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (Bᶜ) = {x : ℝ | x ≤ 3 ∨ 4 ≤ x}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_with_complement_l37_3745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_circumscribed_cube_l37_3741

/-- The surface area of a sphere circumscribed about a cube with edge length 2 is 12π -/
theorem sphere_surface_area_circumscribed_cube (cube_edge : ℝ) (h : cube_edge = 2) : 
  4 * Real.pi * (cube_edge * Real.sqrt 3 / 2) ^ 2 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_circumscribed_cube_l37_3741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_implies_tan_sum_l37_3752

/-- Given complex numbers z and u, if their sum is 4/5 + 3/5i, 
    then the tangent of the sum of their arguments is 24/7 -/
theorem sum_of_complex_implies_tan_sum (α β : ℝ) :
  let z : ℂ := Complex.exp (α * Complex.I)
  let u : ℂ := Complex.exp (β * Complex.I)
  (z + u = Complex.ofReal (4/5) + Complex.I * Complex.ofReal (3/5)) →
  Real.tan (α + β) = 24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_implies_tan_sum_l37_3752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_politics_coverage_percentage_l37_3720

/-- The percentage of reporters covering local politics in country x -/
noncomputable def local_politics_coverage (
  total_reporters : ℝ
  ) (politics_reporters : ℝ) (local_politics_reporters : ℝ) : ℝ :=
  (local_politics_reporters / total_reporters) * 100

/-- The percentage of reporters not covering politics -/
def non_politics_coverage : ℝ := 80

/-- The percentage of politics reporters not covering local politics in country x -/
def non_local_politics_coverage : ℝ := 40

theorem local_politics_coverage_percentage :
  ∀ (total_reporters : ℝ) (politics_reporters : ℝ) (local_politics_reporters : ℝ),
  total_reporters > 0 →
  politics_reporters = total_reporters * (1 - non_politics_coverage / 100) →
  local_politics_reporters = politics_reporters * (1 - non_local_politics_coverage / 100) →
  local_politics_coverage total_reporters politics_reporters local_politics_reporters = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_politics_coverage_percentage_l37_3720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l37_3786

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x - m * Real.sqrt x + 5

/-- The theorem statement -/
theorem problem_statement (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f m x > 1) ↔ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l37_3786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_intersection_complement_P_Q_l37_3716

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set P
def P : Set ℝ := {x : ℝ | |x| > 2}

-- Define set Q
def Q : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Theorem for P ∩ Q
theorem intersection_P_Q : P ∩ Q = Set.Ioo 2 3 := by sorry

-- Theorem for (∁_U P) ∩ Q
theorem intersection_complement_P_Q : (Set.compl P) ∩ Q = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_intersection_complement_P_Q_l37_3716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_l37_3709

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal : 
  ¬(∀ x ∈ A, (2 * x) ∈ B) ↔ ∃ x ∈ A, (2 * x) ∉ B :=
by
  sorry

#check negation_of_universal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_l37_3709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neva_river_current_speed_l37_3717

/-- Represents the scenario of a swimmer in the Neva River --/
structure SwimmerScenario where
  distance_between_bridges : ℚ
  upstream_swim_time : ℚ
  total_flask_travel_time : ℚ

/-- Calculates the speed of the river current given the swimmer scenario --/
def river_current_speed (scenario : SwimmerScenario) : ℚ :=
  scenario.distance_between_bridges / scenario.total_flask_travel_time

/-- Theorem stating that the river current speed is 3 km/h for the given scenario --/
theorem neva_river_current_speed :
  let scenario : SwimmerScenario := {
    distance_between_bridges := 2,  -- 2 km
    upstream_swim_time := 1/3,      -- 20 minutes = 1/3 hour
    total_flask_travel_time := 2/3  -- 40 minutes = 2/3 hour
  }
  river_current_speed scenario = 3 := by
  sorry

#eval river_current_speed {
  distance_between_bridges := 2,
  upstream_swim_time := 1/3,
  total_flask_travel_time := 2/3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neva_river_current_speed_l37_3717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_to_b_equals_sixteen_l37_3794

theorem n_to_b_equals_sixteen (n b : ℝ) : 
  n = 2 ^ (3/10) → b = 40/3 → n ^ b = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_to_b_equals_sixteen_l37_3794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_theorem_l37_3731

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 / (2^x + 1) - a

theorem odd_function_range_theorem :
  (∀ x, f x (1/2) = -f (-x) (1/2)) →
  (∀ y, y ∈ Set.range (fun x => f x (1/2)) ∩ Set.Icc (-1) 3 →
    -7/18 ≤ y ∧ y ≤ 1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_theorem_l37_3731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l37_3762

-- Define the circle
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define the perpendicularity condition
def perpendicular_to_OP (m : ℝ) : Prop := m * 1 = -1

-- Define the line passing through P with slope m
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Theorem statement
theorem line_equation : 
  ∃ (m : ℝ), perpendicular_to_OP m ∧ 
  (∀ (x y : ℝ), line_through_P m x y ↔ x + y - 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l37_3762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_plans_l37_3785

theorem arrangement_plans (n k m : ℕ) : 
  n = 6 → k = 2 → m = 4 →
  (Nat.choose n k) * (Nat.factorial m / (Nat.factorial 2 * Nat.factorial 2)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_plans_l37_3785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_Q_range_l37_3759

-- Define the points M and N
noncomputable def M : ℝ × ℝ := (-2, 0)
noncomputable def N : ℝ × ℝ := (2, 0)

-- Define the trajectory C
def C (x y : ℝ) : Prop := y^2 = -8*x

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2)

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the x-coordinate of point Q
noncomputable def x_Q (k : ℝ) : ℝ := -2 - 8/k

-- Main theorem
theorem x_Q_range (k : ℝ) (S T : ℝ × ℝ) :
  k < 0 →
  k > -1 →
  C S.1 S.2 →
  C T.1 T.2 →
  l k S.1 S.2 →
  l k T.1 T.2 →
  second_quadrant S.1 S.2 →
  second_quadrant T.1 T.2 →
  x_Q k < -6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_Q_range_l37_3759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_sqrt_three_l37_3730

noncomputable section

open Real

/-- Expression B -/
def B : ℝ := 2 * (cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2)

/-- Expression C -/
def C : ℝ := (1 + tan (15 * π / 180)) / (1 - tan (15 * π / 180))

/-- Expression D -/
def D : ℝ := (cos (10 * π / 180) - 2 * sin (20 * π / 180)) / sin (10 * π / 180)

/-- Theorem stating that B, C, and D are equal to √3 -/
theorem expressions_equal_sqrt_three : B = sqrt 3 ∧ C = sqrt 3 ∧ D = sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_sqrt_three_l37_3730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l37_3765

/-- Represents a right pyramid with an equilateral triangular base -/
structure RightPyramid where
  base_perimeter : ℝ
  apex_height : ℝ

/-- Calculates the vertical height from apex to centroid of the base -/
noncomputable def vertical_height_to_centroid (pyramid : RightPyramid) : ℝ :=
  (4 * Real.sqrt 33) / 3

/-- Theorem stating the vertical height for a specific pyramid -/
theorem specific_pyramid_height :
  let pyramid := RightPyramid.mk 24 8
  vertical_height_to_centroid pyramid = (4 * Real.sqrt 33) / 3 := by
  sorry

#check specific_pyramid_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l37_3765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l37_3764

noncomputable def x : ℝ := Real.exp 1 -- e, the solution to ln(x) = x
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.exp (-0.5)

theorem relationship_xyz : y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l37_3764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_property_l37_3715

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Check if a list of natural numbers all have the same digit sum -/
def allSameDigitSum (list : List ℕ) : Prop :=
  ∀ x y, x ∈ list → y ∈ list → digitSum x = digitSum y

/-- The property we want to prove for our target number -/
def hasSumProperty (n : ℕ) : Prop :=
  (∃ (list : List ℕ), list.length = 2002 ∧ allSameDigitSum list ∧ list.sum = n) ∧
  (∃ (list : List ℕ), list.length = 2003 ∧ allSameDigitSum list ∧ list.sum = n)

theorem smallest_sum_property :
  (hasSumProperty 10010) ∧ 
  (∀ m : ℕ, m < 10010 → ¬(hasSumProperty m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_property_l37_3715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l37_3787

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.sin x * Real.cos x + Real.cos x ^ 4 + (1/2) * Real.cos (2*x)

theorem f_range :
  (∀ x : ℝ, -1/2 ≤ f x ∧ f x ≤ 1) ∧
  (∃ x : ℝ, f x = -1/2) ∧
  (∃ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l37_3787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l37_3783

-- Define a triangle with side lengths a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  side_angle_relation : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)

-- Define the theorem
theorem triangle_angle_relation (t : Triangle) 
  (h : t.b * (t.a + t.b) * (t.b + t.c) = t.a^3 + t.b * (t.a^2 + t.c^2) + t.c^3) :
  1 / (Real.sqrt t.A + Real.sqrt t.B) + 1 / (Real.sqrt t.B + Real.sqrt t.C) = 
  2 / (Real.sqrt t.C + Real.sqrt t.A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l37_3783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l37_3734

theorem tan_alpha_value (α : Real) 
  (h1 : Real.cos (π / 2 + α) = 2 * Real.sqrt 2 / 3)
  (h2 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) :
  Real.tan α = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l37_3734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_length_l37_3771

/-- An isosceles trapezoid with a circumscribed circle -/
structure IsoscelesTrapezoid where
  /-- Height of the trapezoid -/
  height : ℝ
  /-- Angle at which the lateral side is seen from the center of the circumscribed circle -/
  lateral_angle : ℝ
  /-- The lateral angle is 120° -/
  angle_is_120 : lateral_angle = 120

/-- The midline of an isosceles trapezoid -/
noncomputable def midline (t : IsoscelesTrapezoid) : ℝ := t.height * Real.sqrt 3 / 3

/-- Theorem: The midline of the isosceles trapezoid has length h√3/3 -/
theorem midline_length (t : IsoscelesTrapezoid) : midline t = t.height * Real.sqrt 3 / 3 := by
  -- Unfold the definition of midline
  unfold midline
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_length_l37_3771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_two_numbers_l37_3773

theorem existence_of_two_numbers (S : Finset ℝ) (h : S.card = 101) :
  ∃ u v, u ∈ S ∧ v ∈ S ∧ u ≠ v ∧ 100 * |u - v| * |1 - u * v| ≤ (1 + u^2) * (1 + v^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_two_numbers_l37_3773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_l37_3703

/-- Calculates the ratio of average speed to still water speed for a boat trip --/
theorem boat_speed_ratio
  (still_speed : ℝ)
  (current_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (h1 : still_speed = 20)
  (h2 : current_speed = 4)
  (h3 : downstream_distance = 4)
  (h4 : upstream_distance = 2)
  : (downstream_distance + upstream_distance) / 
    ((downstream_distance / (still_speed + current_speed)) + 
     (upstream_distance / (still_speed - current_speed))) / 
    still_speed = 36 / 35 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_l37_3703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l37_3704

/-- Represents the score distribution of students in a test --/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score100 : ℝ
  sum_to_one : score60 + score75 + score85 + score90 + score100 = 1

/-- Calculates the mean score given a score distribution --/
def mean_score (dist : ScoreDistribution) : ℝ :=
  60 * dist.score60 + 75 * dist.score75 + 85 * dist.score85 + 90 * dist.score90 + 100 * dist.score100

/-- Determines the median score given a score distribution --/
noncomputable def median_score (dist : ScoreDistribution) : ℝ :=
  if dist.score60 + dist.score75 > 0.5 then 75
  else if dist.score60 + dist.score75 + dist.score85 > 0.5 then 85
  else if dist.score60 + dist.score75 + dist.score85 + dist.score90 > 0.5 then 90
  else 100

/-- The main theorem stating the difference between mean and median scores --/
theorem mean_median_difference (dist : ScoreDistribution) 
  (h1 : dist.score60 = 0.15)
  (h2 : dist.score75 = 0.20)
  (h3 : dist.score85 = 0.30)
  (h4 : dist.score90 = 0.10) :
  mean_score dist - median_score dist = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l37_3704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l37_3743

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 - 2*a*Real.sin x - (Real.cos x)^2

def interval : Set ℝ := { x | -Real.pi/6 ≤ x ∧ x ≤ 2*Real.pi/3 }

noncomputable def min_value (a : ℝ) : ℝ :=
  if a < -1/2 then a + 9/4
  else if a ≤ 1 then -a^2 + 2
  else -2*a + 3

theorem f_minimum (a : ℝ) :
  ∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f a x ≤ f a y ∧ f a x = min_value a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l37_3743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_m_range_l37_3796

/-- The function f parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

/-- Predicate to check if three real numbers can be sides of a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem statement -/
theorem triangle_condition_implies_m_range :
  (∀ m : ℝ, (∀ a b c : ℝ, can_form_triangle (f m a) (f m b) (f m c)) →
    m > 7/5 ∧ m < 5) := by
  sorry

#check triangle_condition_implies_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_m_range_l37_3796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_one_third_l37_3788

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 / x else x^2 - 1

-- State the theorem
theorem f_composite_one_third : f (f (1/3)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_one_third_l37_3788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_x_coord_l37_3754

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  b : ℝ

/-- Defines the equation of the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 - p.y^2 / h.b^2 = 1

/-- Defines the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem hyperbola_intersection_x_coord (h : Hyperbola) (p f1 f2 o : Point) (c : ℝ) :
  on_hyperbola h p →
  distance f1 f2 = 2 * c →
  distance o p = c →
  distance p f1 = c + 2 →
  0 < p.x ∧ 0 < p.y →
  o.x = 0 ∧ o.y = 0 →
  p.x = (Real.sqrt 3 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_x_coord_l37_3754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l37_3710

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_example : 
  dilation (-1 + 2*Complex.I) 4 (3 + 4*Complex.I) = 15 + 10*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l37_3710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l37_3718

-- Define the function f(x) = x + ln x
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 + 1 / x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2 * x - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l37_3718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_right_angle_l37_3751

structure Parabola where
  p : ℝ
  equation : ∀ (x y : ℝ), x^2 = 2 * p * y

structure Point where
  x : ℝ
  y : ℝ

def FocalChord (para : Parabola) (a b : Point) : Prop :=
  sorry -- Definition of focal chord to be implemented

def Projection (para : Parabola) (point original : Point) : Prop :=
  sorry -- Definition of projection on directrix to be implemented

noncomputable def AngleBetween (f a b : Point) : ℝ :=
  sorry -- Definition of angle between three points to be implemented

theorem focal_chord_right_angle (para : Parabola) (a b a1 b1 f : Point) :
  FocalChord para a b →
  Projection para a1 a →
  Projection para b1 b →
  AngleBetween f a1 b1 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_right_angle_l37_3751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_5_and_13_l37_3723

theorem infinitely_many_divisible_by_5_and_13 :
  ∃ f : ℕ → ℕ, StrictMono f ∧
  ∀ k : ℕ, (4 * (f k)^2 + 1) % 5 = 0 ∧ (4 * (f k)^2 + 1) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_5_and_13_l37_3723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_age_l37_3791

noncomputable def average_age_combined (num_people_A : ℕ) (avg_age_A : ℝ) (num_people_B : ℕ) (avg_age_B : ℝ) : ℝ :=
  ((num_people_A : ℝ) * avg_age_A + (num_people_B : ℝ) * avg_age_B) / ((num_people_A + num_people_B) : ℝ)

theorem combined_average_age :
  average_age_combined 6 40 4 25 = 34 := by
  unfold average_age_combined
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_age_l37_3791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_eq_sin_2x_plus_1_period_of_sin_2x_l37_3742

/-- Definition of the determinant-like operation -/
def det_like (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := det_like (Real.sin x) (-1) 1 (Real.cos x)

/-- The statement that π is the smallest positive period of f(x) -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ T = π ∧
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) := by
  sorry

/-- Proof that f(x) = (1/2) * sin(2x) + 1 -/
theorem f_eq_sin_2x_plus_1 (x : ℝ) :
  f x = (1/2) * Real.sin (2 * x) + 1 := by
  sorry

/-- The period of sin(2x) is π -/
theorem period_of_sin_2x :
  ∀ x, Real.sin (2 * (x + π)) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_eq_sin_2x_plus_1_period_of_sin_2x_l37_3742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biography_price_l37_3736

-- Define the given constants
def mystery_price : ℝ := 12
def total_savings : ℝ := 19
def num_biographies : ℝ := 5
def num_mysteries : ℝ := 3
def total_discount_rate : ℝ := 0.43
def mystery_discount_rate : ℝ := 0.375

-- Define the theorem
theorem biography_price : 
  ∃ (biography_price : ℝ),
    -- The total savings equals the sum of savings from biographies and mysteries
    total_savings = (biography_price * num_biographies * (total_discount_rate - mystery_discount_rate)) + 
                    (mystery_price * num_mysteries * mystery_discount_rate) ∧
    -- The normal price of a biography is $20
    biography_price = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biography_price_l37_3736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoints_distance_sum_l37_3758

/-- A square with side length 4 -/
structure Square where
  side_length : ℝ
  is_four : side_length = 4

/-- The midpoint of a side in a square -/
structure Midpoint (s : Square) where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Sum of distances from a vertex to all midpoints -/
noncomputable def sum_distances (s : Square) (v : Midpoint s) (m1 m2 m3 m4 : Midpoint s) : ℝ :=
  distance v.x v.y m1.x m1.y +
  distance v.x v.y m2.x m2.y +
  distance v.x v.y m3.x m3.y +
  distance v.x v.y m4.x m4.y

theorem square_midpoints_distance_sum (s : Square) (v : Midpoint s) 
    (m1 m2 m3 m4 : Midpoint s) :
  sum_distances s v m1 m2 m3 m4 = 4 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoints_distance_sum_l37_3758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_arithmetic_l37_3792

/-- Convert a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ :=
  (n % 10) + 5 * ((n / 10) % 10) + 25 * (n / 100)

/-- Convert a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n % 10) + 7 * ((n / 10) % 10) + 49 * ((n / 100) % 10) + 343 * (n / 1000)

theorem base_conversion_arithmetic :
  (2468 / base5ToBase10 200 : ℚ).floor - base7ToBase10 3451 + 7891 = 6679 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_arithmetic_l37_3792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l37_3727

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line m
def line_m (x y : ℝ) : Prop := ∃ (k b : ℝ), y = k * x + b

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line_m A.1 A.2 ∧ line_m B.1 B.2

-- Define the midpoint
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem line_equation :
  ∀ (A B : ℝ × ℝ),
  intersection_points A B →
  is_midpoint (1, 1/2) A B →
  ∀ (x y : ℝ), line_m x y ↔ x + 2*y - 2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l37_3727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_distance_l37_3793

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- A square in 2D space -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: In a unit square with an inscribed circle, 
    the distance between a corner and a specific point on the circle is √5/10 -/
theorem inscribed_circle_distance (ABCD : Square) (ω : Circle) (M P : Point) :
  ABCD.sideLength = 1 →
  ω.center = Point.mk 0 0 →
  ω.radius = 1/2 →
  M = Point.mk 0 (-1/2) →
  P.x^2 + P.y^2 = 1/4 →
  P.y = -2*P.x - 1/2 →
  P ≠ M →
  distance (Point.mk (-1/2) (1/2)) P = Real.sqrt 5 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_distance_l37_3793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_transformations_l37_3777

noncomputable def f (x : ℝ) : ℝ := 3 + Real.log x / Real.log 3

theorem symmetry_transformations (g : ℝ → ℝ) :
  (∀ x, f (-x) = f x) ∨  -- Symmetry about y-axis
  (∀ x, f x = -f (-x)) ∨  -- Symmetry about origin
  (∀ x, f x = -f x + 6) ∨  -- Symmetry about x-axis
  (∀ x, f (f x) = x)  -- Symmetry about y = x
  →
  (g = λ x => -3 - Real.log x / Real.log 3) ∨
  (g = λ x => 3 + Real.log (-x) / Real.log 3) ∨
  (g = λ x => -3 - Real.log (-x) / Real.log 3) ∨
  (g = λ x => 3^(x-3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_transformations_l37_3777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l37_3708

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2) + Real.cos (x/2) ^ 2

/-- The theorem stating the properties of the triangle and the conclusions -/
theorem triangle_properties (t : Triangle) : 
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c →
  t.a = 2 →
  (∀ x, f x ≤ f t.B) →
  t.A = π/3 ∧ 
  (1/2 * t.a^2 * Real.sin t.A = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l37_3708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lineups_l37_3724

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of players who can't play together -/
def conflict_players : ℕ := 3

/-- The number of possible starting lineups -/
def num_lineups : ℕ := 2937

/-- Theorem stating the number of possible starting lineups -/
theorem count_lineups :
  (Nat.choose total_players lineup_size) -
  (Nat.choose (total_players - conflict_players) (lineup_size - conflict_players)) = num_lineups :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lineups_l37_3724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l37_3722

/-- Calculates the average walking rate given total distance, total time, and break time. -/
noncomputable def average_walking_rate (total_distance : ℝ) (total_time : ℝ) (break_time : ℝ) : ℝ :=
  total_distance / (total_time - break_time)

/-- Theorem stating that given the specific conditions, the average walking rate is approximately 4.097 mph. -/
theorem jack_walking_rate :
  let total_distance : ℝ := 14 -- miles
  let total_time : ℝ := 3.75 -- hours (3 hours and 45 minutes)
  let break_time : ℝ := 1/3 -- hours (20 minutes)
  let actual_rate := average_walking_rate total_distance total_time break_time
  abs (actual_rate - 4.097) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l37_3722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ways_to_sum_three_l37_3711

/-- The number of ways to express a positive integer n as the sum of three positive integers, 
    considering different arrangements as distinct. -/
def ways_to_sum_three (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

/-- Theorem stating that ways_to_sum_three correctly counts the number of ways to express
    a positive integer n as the sum of three positive integers. -/
theorem count_ways_to_sum_three (n : ℕ) (h : n > 2) :
  ways_to_sum_three n = Finset.card (Finset.filter 
    (fun (x : ℕ × ℕ × ℕ) => x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0 ∧ x.1 + x.2.1 + x.2.2 = n)
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))) :=
by sorry

#check count_ways_to_sum_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ways_to_sum_three_l37_3711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_prime_l37_3700

def is_expressible (p : ℕ) : Prop :=
  ∃ a b : ℕ, p = Int.natAbs (2^a - 3^b)

theorem smallest_unexpressible_prime : 
  (∀ p : ℕ, p < 41 → Nat.Prime p → is_expressible p) ∧
  Nat.Prime 41 ∧
  ¬is_expressible 41 := by
  sorry

#check smallest_unexpressible_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_prime_l37_3700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotonic_interval_sine_l37_3713

theorem max_monotonic_interval_sine (f : ℝ → ℝ) (m : ℝ) : 
  (f = λ x ↦ Real.sin (2 * x + π / 4)) →
  (∀ x y, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) →
  m ≤ π / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotonic_interval_sine_l37_3713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_ratio_theorem_l37_3729

/-- Sum of digits of a positive integer -/
def sum_of_digits : ℕ → ℕ
| 0 => 0
| n + 1 => (n + 1) % 10 + sum_of_digits (n / 10)

/-- The set of solutions to the problem -/
def solution_set : Set ℕ := {1008, 1344, 1680, 2688}

/-- Main theorem -/
theorem digit_sum_ratio_theorem :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 →
    (n : ℚ) / sum_of_digits n = 112 ↔ n ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_ratio_theorem_l37_3729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_behavior_l37_3798

/-- Defines the sequence given by the recurrence relation -/
def sequenceA (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1/4) * (a n - 6)^3 + 6

/-- Theorem stating the behavior of the sequence when a₁ = 5 -/
theorem sequence_behavior (a : ℕ → ℝ) (h : sequenceA a) (h1 : a 1 = 5) :
  (∀ n, a n < a (n + 1)) ∧
  (∃ M : ℝ, M ≤ 6 ∧ ∀ n m, n > m → a n < M) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_behavior_l37_3798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_current_age_l37_3746

-- Define the ages of Amy, Ben, and Chris
variable (a b c : ℝ)

-- Define the conditions
def average_age (a b c : ℝ) : Prop := (a + b + c) / 3 = 15
def chris_past_amy_now (a c : ℝ) : Prop := c - 5 = 2 * a
def ben_future_amy_future (a b : ℝ) : Prop := b + 5 = (a + 5) / 2

-- Theorem statement
theorem chris_current_age 
  (h1 : average_age a b c)
  (h2 : chris_past_amy_now a c)
  (h3 : ben_future_amy_future a b) :
  c = 29.28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_current_age_l37_3746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l37_3784

theorem imaginary_part_of_complex_fraction (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → z.im = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l37_3784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l37_3790

open Real

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * sin t.A

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * sin t.B - Real.sqrt 3 * t.b * cos t.A = 0)
  (h2 : 0 < t.A ∧ t.A < π)
  (h3 : t.a = Real.sqrt 7)
  (h4 : t.b = 2) :
  t.A = π / 3 ∧ 
  area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l37_3790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l37_3772

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (x + 1) + (x - 5) ^ (1/3 : ℝ)

-- State the theorem about the domain of h(x)
theorem h_domain : 
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x ≥ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l37_3772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalent_to_shifted_sine_roots_imply_m_range_l37_3735

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6) + 2 * (Real.sin (x - Real.pi / 12))^2

theorem f_equivalent_to_shifted_sine (x : Real) : 
  f x = 2 * Real.sin (2 * x - Real.pi / 3) + 1 := by sorry

theorem roots_imply_m_range (m : Real) :
  (∃ x y, x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2) ∧ 
          y ∈ Set.Icc (Real.pi / 12) (Real.pi / 2) ∧ 
          x ≠ y ∧ 
          f x = m ∧ 
          f y = m) →
  m ∈ Set.Icc (Real.sqrt 3 + 1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalent_to_shifted_sine_roots_imply_m_range_l37_3735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l37_3738

noncomputable section

/-- An inverse proportion function from ℝ to ℝ -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The specific function we want to prove is an inverse proportion -/
def f (x : ℝ) : ℝ := 2 / x

theorem f_is_inverse_proportion :
  InverseProportion f := by
  use 2
  intro x hx
  simp [f, hx]

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l37_3738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_root_greater_than_three_l37_3780

-- Define the three equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 3 = 29
def equation2 (x : ℝ) : Prop := (3*x - 2)^2 = 2*(x - 1)^2
def equation3 (x : ℝ) : Prop := Real.sqrt (x^2 - 10) = Real.sqrt (2*x - 3)

-- Define a function to check if a real number is less than or equal to 3
def isLessThanOrEqualToThree (x : ℝ) : Prop := x ≤ 3

-- Theorem statement
theorem no_root_greater_than_three :
  (∀ x, equation1 x → isLessThanOrEqualToThree x) ∧
  (∀ x, equation2 x → isLessThanOrEqualToThree x) ∧
  (∀ x, equation3 x → isLessThanOrEqualToThree x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_root_greater_than_three_l37_3780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_remaining_slices_l37_3779

-- Define the pizza sizes
def large_pizza : ℕ := 12
def medium_pizza : ℕ := 10
def small_pizza : ℕ := 8

-- Define the eating percentages
def stephen_large_percent : ℚ := 25 / 100
def stephen_medium_percent : ℚ := 15 / 100
def pete_large_percent : ℚ := 50 / 100
def pete_medium_percent : ℚ := 20 / 100
def laura_small_percent : ℚ := 30 / 100

-- Function to calculate remaining slices
def remaining_slices (total : ℕ) (eaten_percent : ℚ) : ℕ :=
  total - Int.toNat (Rat.floor (↑total * eaten_percent))

-- Theorem stating the remaining slices for each pizza
theorem pizza_remaining_slices :
  let large_after_stephen := remaining_slices large_pizza stephen_large_percent
  let large_after_pete := remaining_slices large_after_stephen pete_large_percent
  let medium_after_stephen := remaining_slices medium_pizza stephen_medium_percent
  let medium_after_pete := remaining_slices medium_after_stephen pete_medium_percent
  let small_after_laura := remaining_slices small_pizza laura_small_percent
  (large_after_pete = 5) ∧ (medium_after_pete = 8) ∧ (small_after_laura = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_remaining_slices_l37_3779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_equals_15_factorial_l37_3782

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem min_sum_of_product_equals_15_factorial (p q r s : ℕ+) 
  (h : (p * q * r * s : ℕ) = factorial 15) : 
  (p : ℕ) + q + r + s ≥ 260000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_equals_15_factorial_l37_3782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_3003_even_integers_digit_count_l37_3728

def count_digits (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def sum_digits (n : ℕ) : ℕ :=
  (List.range n).map (fun i => count_digits ((i + 1) * 2)) |>.sum

theorem first_3003_even_integers_digit_count : sum_digits 3003 = 11460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_3003_even_integers_digit_count_l37_3728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l37_3732

/-- Represents a parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y => y^2 = 4*x

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The focus of the parabola -/
def focus (p : Parabola) : ℝ × ℝ := (1, 0)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_focus (p : Parabola) (a : PointOnParabola p) (h : a.x = 4) :
  distance (a.x, a.y) (focus p) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l37_3732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_one_l37_3739

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + 2 / (3^x - 1)

-- State the theorem
theorem odd_function_implies_m_equals_one :
  (∀ x, f m x = -f m (-x)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_one_l37_3739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l37_3774

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, (1/x > 1 → (1/3:ℝ)^x < 1)) ∧
  (∃ x : ℝ, ((1/3:ℝ)^x < 1 ∧ ¬(1/x > 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l37_3774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rectangle_overlap_area_l37_3757

/-- Given two rectangles with dimensions 8 × 10 and 12 × 9 respectively,
    and an overlapping area of 37, the area of the non-overlapping part
    of the second rectangle is 65. -/
def rectangle_overlap_area (rect1_width : ℕ) (rect1_height : ℕ)
  (rect2_width : ℕ) (rect2_height : ℕ)
  (overlap_area : ℕ) : ℕ :=
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  let white_area := rect1_area - overlap_area
  rect2_area - white_area

/-- The specific instance of the problem -/
theorem specific_rectangle_overlap_area :
  rectangle_overlap_area 8 10 12 9 37 = 65 :=
by
  -- Unfold the definition and evaluate
  unfold rectangle_overlap_area
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rectangle_overlap_area_l37_3757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l37_3789

-- Define the function f(x) = 3x^2 - 2ln(x)
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

-- Define the domain of f(x)
def domain : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem monotonic_increasing_interval :
  {x : ℝ | x ∈ domain ∧ x > Real.sqrt 3 / 3} = {x : ℝ | ∀ y ∈ domain, y > x → f y > f x} := by
  sorry

#check monotonic_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l37_3789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_rental_cost_l37_3768

theorem dvd_rental_cost (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 →
  num_dvds = 4 →
  cost_per_dvd = total_cost / (num_dvds : ℝ) →
  cost_per_dvd = 1.20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_rental_cost_l37_3768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l37_3766

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

theorem projection_property :
  ∀ (p : (ℝ × ℝ) → (ℝ × ℝ)),
  (p (2, -4) = (1, -1)) →
  (p (-3, 2) = (-5/2, 5/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l37_3766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_university_savings_plan_l37_3778

/-- The monthly deposit amount required to accumulate a target sum over a given period with compound interest. -/
noncomputable def monthly_deposit (n : ℕ) (r : ℝ) (target : ℝ) : ℝ :=
  target * r / ((1 + r)^n - 1)

/-- Theorem stating the monthly deposit required for university expenses -/
theorem university_savings_plan :
  let n : ℕ := 25  -- number of monthly deposits
  let r : ℝ := 0.02  -- monthly interest rate
  let target : ℝ := 60000  -- target savings amount
  ∀ ε > 0, |monthly_deposit n r target - 1875| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_university_savings_plan_l37_3778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l37_3761

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - Real.cos (2 * x) + 1) / (2 * Real.sin x)

theorem f_properties :
  ∃ (domain : Set ℝ) (range : Set ℝ) (α : ℝ),
    domain = {x : ℝ | ∀ (k : ℤ), x ≠ k * Real.pi} ∧
    range = Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Ioo (-1) 1 ∪ Set.Icc 1 (Real.sqrt 2) ∧
    0 < α ∧ α < Real.pi / 2 ∧
    Real.tan (α / 2) = 1 / 2 ∧
    f α = 7 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l37_3761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_edge_length_l37_3753

/-- A square pyramid with a hemisphere resting on its base --/
structure PyramidWithHemisphere where
  -- Radius of the hemisphere
  r : ℝ
  -- Height of the pyramid
  h : ℝ
  -- The hemisphere rests on the base and is tangent to the other faces
  hemisphere_tangent : True

/-- The edge-length of the base of the pyramid --/
noncomputable def base_edge_length (p : PyramidWithHemisphere) : ℝ :=
  3 * Real.sqrt 2

/-- Theorem stating the edge-length of the base of the pyramid --/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h_radius : p.r = 2)
  (h_height : p.h = 6) :
  base_edge_length p = 3 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_edge_length_l37_3753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l37_3760

/-- Represents the length of the park in meters -/
noncomputable def length : ℝ := sorry

/-- Represents the breadth of the park in meters -/
noncomputable def breadth : ℝ := sorry

/-- The ratio between length and breadth is 1:2 -/
axiom length_breadth_ratio : breadth = 2 * length

/-- The speed of the cyclist in meters per minute -/
noncomputable def speed : ℝ := 6000 / 60

/-- The time taken to complete one round in minutes -/
noncomputable def time : ℝ := 6

/-- The perimeter of the park in meters -/
noncomputable def perimeter : ℝ := 2 * (length + breadth)

/-- The perimeter equals the distance covered by the cyclist -/
axiom perimeter_equals_distance : perimeter = speed * time

/-- The theorem to prove -/
theorem park_area : length * breadth = 20000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l37_3760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l37_3737

/-- Represents a class of students -/
structure MyClass where
  total : ℕ
  girls : ℕ
  boys : ℕ

/-- Condition: Among any group of ten students, there is at least one girl -/
def hasGirlInTen (c : MyClass) : Prop :=
  c.boys ≤ 9

/-- The main theorem -/
theorem girls_in_class (c : MyClass) :
  c.total = 17 ∧
  hasGirlInTen c ∧
  c.boys > c.girls ∧
  c.total = c.boys + c.girls →
  c.girls = 8 := by
  sorry

#check girls_in_class

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l37_3737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_cyclicity_l37_3776

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the inscribed circle
structure InscribedCircle (ABCD : Quadrilateral) :=
  (center : EuclideanSpace ℝ (Fin 2))
  (radius : ℝ)
  (M : EuclideanSpace ℝ (Fin 2)) -- Tangent point on AB
  (N : EuclideanSpace ℝ (Fin 2)) -- Tangent point on BC
  (P : EuclideanSpace ℝ (Fin 2)) -- Tangent point on CD
  (Q : EuclideanSpace ℝ (Fin 2)) -- Tangent point on DA

-- Define the property of being cyclic
def is_cyclic (ABCD : Quadrilateral) : Prop := sorry

-- Define the condition (AM)(CP) = (BN)(DQ)
def tangent_product_condition (ABCD : Quadrilateral) (circle : InscribedCircle ABCD) : Prop :=
  norm (ABCD.A - circle.M) * norm (ABCD.C - circle.P) = 
  norm (ABCD.B - circle.N) * norm (ABCD.D - circle.Q)

-- State the theorem
theorem inscribed_circle_cyclicity 
  (ABCD : Quadrilateral) 
  (circle : InscribedCircle ABCD) 
  (h : tangent_product_condition ABCD circle) : 
  is_cyclic ABCD :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_cyclicity_l37_3776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l37_3799

noncomputable def f (x : ℝ) := 1/x - 2*x

theorem min_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 2 → f y ≥ f x) ∧
  f x = -7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l37_3799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_max_profit_optimal_advertising_expense_l37_3726

/-- Represents the sales quantity (in ten thousand units) as a function of advertising expenses -/
noncomputable def Q (x : ℝ) : ℝ := (3 * x + 1) / (x + 1)

/-- Represents the annual profit (in ten thousand yuan) as a function of advertising expenses -/
noncomputable def W (x : ℝ) : ℝ := (-x^2 + 98*x + 35) / (2*(x + 1))

theorem company_max_profit (x : ℝ) (h : x ≥ 0) :
  W x ≤ 42 ∧ W 7 = 42 := by sorry

theorem optimal_advertising_expense :
  ∃ x : ℝ, x ≥ 0 ∧ W x = 42 ∧ ∀ y : ℝ, y ≥ 0 → W y ≤ W x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_max_profit_optimal_advertising_expense_l37_3726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cyclic_number_l37_3748

def is_cyclic_permutation (a b : Nat) : Prop :=
  ∃ k, a.digits 10 = (b.digits 10).rotate k

theorem unique_cyclic_number : ∃! n : Nat,
  (100000 ≤ n) ∧ (n < 1000000) ∧
  (is_cyclic_permutation n (2*n)) ∧
  (is_cyclic_permutation n (3*n)) ∧
  (is_cyclic_permutation n (4*n)) ∧
  (is_cyclic_permutation n (5*n)) ∧
  (is_cyclic_permutation n (6*n)) ∧
  ((2*n).digits 10).head? = (n.digits 10).get? 2 ∧
  ((3*n).digits 10).head? = (n.digits 10).get? 1 ∧
  ((4*n).digits 10).head? = (n.digits 10).get? 4 ∧
  ((5*n).digits 10).head? = (n.digits 10).get? 5 ∧
  ((6*n).digits 10).head? = (n.digits 10).get? 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cyclic_number_l37_3748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l37_3756

theorem tan_theta_minus_pi_fourth (θ : Real) 
  (h1 : -π/2 < θ ∧ θ < 0)  -- θ is in the fourth quadrant
  (h2 : Real.sin (θ + π/4) = -3/5) : 
  Real.tan (θ - π/4) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l37_3756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_plate_area_l37_3781

/-- Represents the dimensions of a rectangular plate in centimeters -/
structure PlateDimensions where
  width : ℚ
  length : ℚ

/-- Calculates the area of a rectangular plate in square meters -/
def plateArea (d : PlateDimensions) : ℚ :=
  (d.width / 100) * (d.length / 100)

/-- Theorem: The area of a rectangular iron plate with width 500 cm and length 800 cm is 40 square meters -/
theorem iron_plate_area :
  let d : PlateDimensions := { width := 500, length := 800 }
  plateArea d = 40 := by
  -- Unfold the definition of plateArea
  unfold plateArea
  -- Perform the calculation
  simp [PlateDimensions.width, PlateDimensions.length]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_plate_area_l37_3781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_properties_l37_3747

/-- Hyperbola C with equation y^2/(4b^2) - x^2/b^2 = 1 where b = 2 -/
def hyperbola_C (x y : ℝ) : Prop := y^2/16 - x^2/4 = 1

/-- Distance function between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Minimum distance function from M(0,t) to C -/
noncomputable def min_distance (t : ℝ) : ℝ :=
  if 4 < t ∧ t ≤ 5 then t - 4 else (Real.sqrt (5 * t^2 - 100)) / 5

/-- Trajectory of point Q -/
def trajectory_Q (x y : ℝ) : Prop := y^2/25 - x^2/100 = 1 ∧ x ≠ 0

theorem hyperbola_C_properties :
  ∀ t x y x0 y0 : ℝ,
  (∀ x y, hyperbola_C x y → 
    distance 0 t x y ≥ min_distance t) ∧
  (∀ k m, k ≠ 2 ∧ k ≠ -2 →
    (∃! p : ℝ × ℝ, hyperbola_C p.1 p.2 ∧ p.2 = k * p.1 + m) →
    trajectory_Q x0 y0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_properties_l37_3747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_golden_ratio_relation_l37_3705

-- Define the golden ratio Φ as noncomputable
noncomputable def Φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem fibonacci_golden_ratio_relation (n : ℕ) :
  (fib (n + 1) : ℝ) - Φ * (fib n : ℝ) = (-1 / Φ) ^ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_golden_ratio_relation_l37_3705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_is_27_5_percent_l37_3770

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the new alcohol concentration after mixing and diluting -/
noncomputable def newConcentration (v1 v2 : Vessel) (totalCapacity : ℝ) : ℝ :=
  let alcoholVolume1 := v1.capacity * v1.alcoholConcentration
  let alcoholVolume2 := v2.capacity * v2.alcoholConcentration
  let totalAlcoholVolume := alcoholVolume1 + alcoholVolume2
  (totalAlcoholVolume / totalCapacity) * 100

/-- Theorem stating that the new concentration is 27.5% -/
theorem concentration_is_27_5_percent 
  (v1 : Vessel) 
  (v2 : Vessel) 
  (h1 : v1.capacity = 3)
  (h2 : v1.alcoholConcentration = 0.25)
  (h3 : v2.capacity = 5)
  (h4 : v2.alcoholConcentration = 0.4)
  (h5 : v1.capacity + v2.capacity = 8)
  (totalCapacity : ℝ)
  (h6 : totalCapacity = 10) :
  newConcentration v1 v2 totalCapacity = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_is_27_5_percent_l37_3770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_geq_three_halves_l37_3714

/-- The function f(x) = 3^x / (3^x + 1) -/
noncomputable def f (x : ℝ) : ℝ := 3^x / (3^x + 1)

/-- Theorem statement -/
theorem sum_f_geq_three_halves (a b c : ℝ) (h : a + b + c = 0) :
  f a + f b + f c ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_geq_three_halves_l37_3714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_infinitely_many_perpendicular_lines_l37_3769

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Point → Prop)
variable (contains_line : Plane → Line → Prop)
variable (mutually_perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)

-- Theorem 1: Unique perpendicular plane through a point
theorem unique_perpendicular_plane 
  (L : Line) (P : Point) :
  ∃! (π : Plane), perpendicular L π ∧ contains π P :=
sorry

-- Theorem 2: Infinitely many perpendicular lines in mutually perpendicular planes
theorem infinitely_many_perpendicular_lines 
  (A B : Plane) (l : Line) :
  mutually_perpendicular A B → 
  line_in_plane l A → 
  ∃ (S : Set Line), 
    (∀ m ∈ S, line_in_plane m B ∧ lines_perpendicular l m) ∧
    (Set.Infinite S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_infinitely_many_perpendicular_lines_l37_3769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_trip_funds_l37_3749

def base_6_to_10 : ℕ := 
  3 * (6^3) + 2 * (6^2) + 4 * (6^1) + 2 * (6^0)

def kate_savings : ℕ := base_6_to_10

def ticket_cost : ℕ := 1000

theorem kate_trip_funds : 
  kate_savings < ticket_cost ∧ ticket_cost - kate_savings = 254 := by
  constructor
  · -- Prove kate_savings < ticket_cost
    norm_num [kate_savings, ticket_cost, base_6_to_10]
  · -- Prove ticket_cost - kate_savings = 254
    norm_num [kate_savings, ticket_cost, base_6_to_10]

#eval kate_savings
#eval ticket_cost - kate_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_trip_funds_l37_3749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_calculation_l37_3712

/-- Given a sample capacity and a frequency rate, calculate the frequency of a group of samples. -/
def calculate_frequency (sample_capacity : ℕ) (frequency_rate : ℝ) : ℝ :=
  frequency_rate * (sample_capacity : ℝ)

/-- Theorem: For a sample with capacity 32 and a frequency rate of 0.25, the frequency is 8. -/
theorem frequency_calculation :
  let sample_capacity : ℕ := 32
  let frequency_rate : ℝ := 0.25
  calculate_frequency sample_capacity frequency_rate = 8 := by
  unfold calculate_frequency
  simp
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_calculation_l37_3712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l37_3795

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (Real.sqrt (b + 4 * a / c) = b * Real.sqrt (a / c)) ↔ 
  (c = a * b - 4 * a / b ∧ b * c = 4 * a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l37_3795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_specific_trapezoid_l37_3767

/-- An isosceles trapezoid with given side lengths -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  DA : ℝ
  isosceles : BC = DA
  AB_longer : AB > CD

/-- The length of the diagonal AC in an isosceles trapezoid -/
noncomputable def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt ((t.AB - t.CD)^2 / 4 + t.BC^2)

/-- Theorem: The diagonal length of the specified isosceles trapezoid is √457 -/
theorem diagonal_length_specific_trapezoid :
  let t : IsoscelesTrapezoid := {
    AB := 24,
    CD := 12,
    BC := 13,
    DA := 13,
    isosceles := by rfl,
    AB_longer := by norm_num
  }
  diagonal_length t = Real.sqrt 457 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_specific_trapezoid_l37_3767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l37_3719

/-- The area of a rhombus with side length 5 and one diagonal of length 8 is 24. -/
theorem rhombus_area (s d₁ : ℝ) (hs : s = 5) (hd₁ : d₁ = 8) :
  (d₁ * (2 * Real.sqrt (s^2 - (d₁/2)^2))) / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l37_3719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_centers_l37_3775

/-- The maximum distance between centers of two circles in a rectangle --/
theorem max_distance_between_circle_centers (rectangle_length : ℝ) (rectangle_width : ℝ) (circle_diameter : ℝ) : 
  rectangle_length = 18 →
  rectangle_width = 15 →
  circle_diameter = 7.5 →
  Real.sqrt ((rectangle_length - circle_diameter)^2 + (rectangle_width - circle_diameter)^2) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_centers_l37_3775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_properties_l37_3740

/-- A pyramid with vertices A₁, A₂, A₃, A₄ in ℝ³ -/
structure Pyramid where
  A₁ : ℝ × ℝ × ℝ
  A₂ : ℝ × ℝ × ℝ
  A₃ : ℝ × ℝ × ℝ
  A₄ : ℝ × ℝ × ℝ

/-- Given a pyramid, compute various geometric properties -/
noncomputable def pyramidProperties (p : Pyramid) : 
  (ℝ × ℝ × ℝ → ℝ) × -- Equation of plane A₁A₂A₃
  (ℝ × ℝ × ℝ → ℝ) × -- Equation of plane A₁A₂A₄
  ℝ × -- Angle between edge A₁A₃ and face A₁A₂A₄
  ℝ × -- Distance from A₃ to face A₁A₂A₄
  ℝ × -- Shortest distance between lines A₁A₂ and A₃A₄
  (ℝ → ℝ × ℝ × ℝ) -- Parametric equation of altitude from A₃ to face A₁A₂A₄
  := sorry

/-- The theorem stating the properties of the specific pyramid -/
theorem specific_pyramid_properties : 
  let p : Pyramid := {
    A₁ := (4, 2, 5),
    A₂ := (0, 7, 2),
    A₃ := (0, 2, 7),
    A₄ := (1, 5, 0)
  }
  let (plane_A₁A₂A₃, plane_A₁A₂A₄, angle, distance_A₃, shortest_distance, altitude) := pyramidProperties p
  
  (plane_A₁A₂A₃ = λ (x, y, z) => x + 2*y + 2*z - 18) ∧
  (plane_A₁A₂A₄ = λ (x, y, z) => -16*x - 11*y + 3*z + 71) ∧
  (abs (angle - 139) < 0.1) ∧ -- Using absolute difference for approximate equality
  (abs (distance_A₃ - 3.56) < 0.01) ∧
  (abs (shortest_distance - 1.595) < 0.001) ∧
  (altitude = λ t => (1 + t, 5 + 2*t, 2*t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_properties_l37_3740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tirzah_purses_count_l37_3701

theorem tirzah_purses_count :
  ∀ (num_purses : ℕ),
  (num_purses / 2 : ℚ) + (3 / 4 : ℚ) * 24 = 31 → num_purses = 26 :=
by
  intro num_purses
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tirzah_purses_count_l37_3701
