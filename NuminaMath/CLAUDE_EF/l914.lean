import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_quadrant_IV_l914_91466

noncomputable def powerFunction (a : ℝ) (x : ℝ) : ℝ := x^a

def inQuadrantIV (f : ℝ → ℝ) : Prop :=
  ∃ x, x < 0 ∧ f x < 0

theorem power_function_not_in_quadrant_IV :
  ∀ a ∈ ({-1, (1/2 : ℝ), 2, 3} : Set ℝ), ¬(inQuadrantIV (powerFunction a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_quadrant_IV_l914_91466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l914_91483

noncomputable def f (x : ℝ) := x + 1

noncomputable def g (a x : ℝ) := Real.exp (Real.log 2 * |x + 2|) + a

theorem a_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 3 4, ∃ x₂ ∈ Set.Icc (-3) 1, f x₁ ≥ g a x₂) → 
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l914_91483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_square_with_integer_l914_91443

theorem complete_square_with_integer (y : ℝ) : ∃ k : ℤ, y^2 + 14*y + 60 = (y+7)^2 + k := by
  use 11
  ring

-- The following line is not necessary for the theorem, but if you want to keep it:
-- #eval (11 : ℤ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_square_with_integer_l914_91443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_2_range_of_a_for_positive_f_l914_91436

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x + 2

-- Theorem 1: Minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (m : ℝ), m = 5 ∧ ∀ x > 1, f 2 x ≥ m :=
sorry

-- Theorem 2: Range of a for f(x) > 0
theorem range_of_a_for_positive_f :
  ∀ a : ℝ, (∀ x > 1, f a x > 0) ↔ a > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_2_range_of_a_for_positive_f_l914_91436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_reach_necklace_simultaneously_l914_91455

/-- Represents the speed of a swimmer relative to the water --/
def SwimmerSpeed : Type := ℝ

/-- Represents the speed of the river current --/
def CurrentSpeed : Type := ℝ

/-- Represents the time spent swimming --/
def SwimTime : Type := ℝ

/-- Calculates the distance traveled by a swimmer in the direction of the current --/
def distanceWithCurrent (swimSpeed : ℝ) (current : ℝ) (time : ℝ) : ℝ :=
  (swimSpeed + current) * time

/-- Calculates the distance traveled by a swimmer against the current --/
def distanceAgainstCurrent (swimSpeed : ℝ) (current : ℝ) (time : ℝ) : ℝ :=
  (swimSpeed - current) * time

/-- Calculates the distance traveled by the necklace --/
def necklaceDistance (current : ℝ) (time : ℝ) : ℝ :=
  current * time

theorem swimmers_reach_necklace_simultaneously 
  (swimSpeed : ℝ) 
  (current : ℝ) 
  (time : ℝ) 
  (h1 : swimSpeed > 0) 
  (h2 : current > 0) 
  (h3 : time > 0) : 
  distanceWithCurrent swimSpeed current time + 
  distanceAgainstCurrent swimSpeed current time = 
  2 * necklaceDistance current time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_reach_necklace_simultaneously_l914_91455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l914_91497

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Theorem stating that the line intersects the circle
theorem line_intersects_circle : ∃ (x y : ℝ), line x y ∧ circle_eq x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l914_91497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_at_one_f_increasing_for_nonneg_a_g_concave_l914_91481

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 1 / x + a * x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / x - 1 / (x^2) + a

-- Define g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 * f' a x

-- Theorem 1: The derivative of f(x) at x=1 is zero when a = 0
theorem tangent_parallel_at_one (a : ℝ) : f' a 1 = 0 ↔ a = 0 := by sorry

-- Theorem 2: For a ≥ 0, f'(x) > 0 for all x > 0
theorem f_increasing_for_nonneg_a (a : ℝ) (x : ℝ) (ha : a ≥ 0) (hx : x > 0) : f' a x > 0 := by sorry

-- Theorem 3: For a > 0, g(x) is concave on its domain (0, +∞)
theorem g_concave (a : ℝ) (ha : a > 0) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
    g a ((x₁ + x₂) / 2) ≤ (g a x₁ + g a x₂) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_at_one_f_increasing_for_nonneg_a_g_concave_l914_91481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_l914_91432

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_plane_perpendicular (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicular_lines a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_l914_91432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l914_91430

theorem polynomial_sum_theorem (p q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x : ℝ, p x = a * x^2 + b * x + c) →  -- p is quadratic
  (∃ d e : ℝ, d ≠ 0 ∧ ∀ x : ℝ, q x = d * (x - 1)^2 * (x - e)) →  -- q is cubic with double root at x=1
  p 4 = 8 →  -- given condition
  q 3 = 3 →  -- given condition
  (∀ x : ℝ, p x + q x = x^3 - (19/4) * x^2 + (13/4) * x - 5/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l914_91430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_signs_l914_91458

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the derivatives of f and g
def f' : ℝ → ℝ := sorry
def g' : ℝ → ℝ := sorry

-- State the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → HasDerivAt f (f' x) x ∧ f' x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → HasDerivAt g (g' x) x ∧ g' x > 0

-- State the theorem to be proved
theorem f_g_deriv_signs : ∀ x : ℝ, x < 0 → f' x > 0 ∧ g' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_signs_l914_91458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l914_91498

theorem triangle_properties (a b c A B C : ℝ) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B))
  (h_relation : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0) :
  A = π / 3 ∧ 
  (a = Real.sqrt 3 → 
    let S := (1 / 2) * a * b * Real.sin C
    Real.sqrt 3 / 2 < S ∧ S ≤ 3 * Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l914_91498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l914_91491

/-- Represents a person's age and its properties -/
structure Age where
  value : ℕ
  isOneDigit : Prop := value < 10
  isTwoDigits : Prop := 10 ≤ value ∧ value < 100

/-- The problem statement -/
theorem age_difference_proof 
  (john : Age) 
  (wilson : Age) 
  (h1 : wilson.isOneDigit)
  (h2 : john.isTwoDigits)
  (h3 : ∃ (j w : ℕ), john.value = 10 * j + w ∧ wilson.value = w)
  (h4 : john.value + 21 = 2 * (wilson.value + 21)) :
  john.value - wilson.value = 30 := by
  sorry

#check age_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l914_91491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_exist_l914_91446

noncomputable def sign (x : ℝ) : ℝ := 
  if x > 0 then 1 else if x < 0 then -1 else 0

theorem four_solutions_exist :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔ 
      (x = 4036 - 4037 * sign (y - z) ∧
       y = 4036 - 4037 * sign (z - x) ∧
       z = 4036 - 4037 * sign (x - y))) ∧
    Finset.card solutions = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_exist_l914_91446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_PQRS_l914_91479

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry  -- The actual implementation would go here

/-- The specific tetrahedron PQRS from the problem -/
noncomputable def PQRS : Tetrahedron :=
  { PQ := 6
    PR := 5
    PS := 4
    QR := 7
    QS := 5
    RS := 15/4 * Real.sqrt 2 }

/-- Theorem stating that the volume of tetrahedron PQRS is 10/3 -/
theorem volume_of_PQRS : tetrahedronVolume PQRS = 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_PQRS_l914_91479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l914_91408

open Real

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := log x + 3/2 * x^2 - b*x

-- State the theorem
theorem increasing_function_condition (b : ℝ) :
  (∀ x > 0, Monotone (f b)) → b ≤ 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l914_91408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l914_91409

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def equation (x : ℝ) : Prop :=
  (1 : ℝ) / (floor x) + 1 / (x - floor x) = 8 / 3

theorem equation_solutions :
  {x : ℝ | equation x} = {29/12, 19/6, 97/24} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l914_91409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l914_91424

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 5 * Real.cos α, Real.sin α)

-- Define the line l in polar form
noncomputable def line_l (θ : ℝ) : ℝ := Real.sqrt 2 / Real.cos (θ + Real.pi / 4)

-- Define the point P
def point_P : ℝ × ℝ := (0, -2)

-- Define the intersection points A and B (we don't know their exact coordinates)
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry

-- Theorem statement
theorem distance_sum_theorem :
  let dist := fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist point_P point_A + dist point_P point_B = 10 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l914_91424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_binomial_expansion_l914_91485

theorem fourth_term_binomial_expansion 
  (x : ℝ) (h : x > 0) : 
  let a := 2
  let b := -x^(-(1/3 : ℝ))
  let n := 6
  let r := 3
  (n.choose r) * a^(n-r) * b^r = -160/x :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_binomial_expansion_l914_91485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l914_91404

noncomputable section

-- Define the common radius
variable (r : ℝ)

-- Define the volumes of the cylinder, sphere, and cone
noncomputable def cylinder_volume : ℝ := 2 * Real.pi * r^3
noncomputable def sphere_volume : ℝ := (4/3) * Real.pi * r^3
noncomputable def cone_volume : ℝ := (2/3) * Real.pi * r^3

-- Theorem statement
theorem volume_ratio (r : ℝ) (hr : r > 0) :
  (cylinder_volume r) / (sphere_volume r) = 3/2 ∧
  (sphere_volume r) / (sphere_volume r) = 1 ∧
  (cone_volume r) / (sphere_volume r) = 1/2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l914_91404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_movement_l914_91456

def roll_distances : List ℚ := [12, -10, 9, -6, 8.5, -6, 8, -7]

def time_per_mm : ℚ := 1/50  -- 0.02 as a rational number

theorem ball_movement (roll_distances : List ℚ) (time_per_mm : ℚ) :
  let final_position := roll_distances.sum
  let total_distance := (roll_distances.map abs).sum
  let total_time := total_distance * time_per_mm
  (final_position = 17/2 ∧ total_time = 133/100) := by
  sorry

#eval roll_distances.sum
#eval (roll_distances.map abs).sum * time_per_mm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_movement_l914_91456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l914_91454

/-- Given a function f such that f(x^2 + 1) = x^4 + 4x^2 for all real x,
    the minimum value of f(x) within its domain is 0. -/
theorem min_value_of_f (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 4*x^2) : 
    ∃ y : ℝ, f y = 0 ∧ ∀ x : ℝ, f x ≥ 0 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l914_91454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l914_91439

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l914_91439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_correct_l914_91447

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of g
def domain_g : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that domain_g is the correct domain for g
theorem g_domain_correct :
  ∀ x : ℝ, x ∈ domain_g ↔ (∃ y : ℝ, g x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_correct_l914_91447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_kmph_equals_five_ms_l914_91478

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

theorem eighteen_kmph_equals_five_ms :
  kmph_to_ms 18 = 5 := by
  unfold kmph_to_ms
  norm_num

#eval (18 : ℚ) * (1000 / 3600)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_kmph_equals_five_ms_l914_91478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_l914_91435

noncomputable def f (n : ℕ+) : ℝ := (3/2)^(n : ℝ) - 2*(n : ℝ) - 7

theorem project_profitability :
  (∀ n : ℕ+, n < 8 → f n < 0) ∧
  (∀ n : ℕ+, n ≥ 8 → f n > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_l914_91435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l914_91499

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.pi ^ (1/3))
  (hb : b = Real.log 3 / Real.log Real.pi)
  (hc : c = Real.log (Real.sqrt 3 - 1)) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l914_91499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l914_91407

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- The first sphere -/
def sphere1 : Sphere :=
  { center := { x := -5, y := -15, z := 10 },
    radius := 23 }

/-- The second sphere -/
def sphere2 : Sphere :=
  { center := { x := 7, y := 12, z := -20 },
    radius := 95 }

/-- Theorem stating the largest possible distance between points on the two spheres -/
theorem largest_distance_between_spheres :
  ∃ (p1 : Point3D) (p2 : Point3D),
    (distance p1 sphere1.center = sphere1.radius) ∧
    (distance p2 sphere2.center = sphere2.radius) ∧
    (∀ (q1 q2 : Point3D),
      (distance q1 sphere1.center = sphere1.radius) →
      (distance q2 sphere2.center = sphere2.radius) →
      distance q1 q2 ≤ distance p1 p2) ∧
    distance p1 p2 = 118 + Real.sqrt 1773 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l914_91407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l914_91462

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + (1 + a) * y = 3
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (a + 1) * x + (3 - 2 * a) * y = 2

-- Define the condition for perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, (perpendicular (-a / (1 + a)) (-(a + 1) / (3 - 2 * a))) ↔ (a = -1 ∨ a = 3) := by
  sorry

#check perpendicular_lines_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l914_91462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_arithmetic_l914_91448

/-- Proves the equation involving base conversions and arithmetic operations -/
theorem base_conversion_arithmetic : 
  let base_5_to_10 : ℕ := 6
  let base_7_to_10 : ℕ := 520
  (2468 : ℚ) / base_5_to_10 - base_7_to_10 + 3456 = 3347 + 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_arithmetic_l914_91448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_n_zeros_l914_91482

/-- Represents an infinite sheet of squared paper -/
def Sheet := ℤ → ℤ → ℤ

/-- Condition that each square is the sum of the squares above and to the left -/
def is_valid_sheet (s : Sheet) : Prop :=
  ∀ i j : ℤ, s i j = s (i-1) j + s i (j-1)

/-- A row in the sheet -/
def Row (s : Sheet) (r : ℤ) : ℤ → ℤ := λ j => s r j

/-- Count of zeros in a row -/
noncomputable def zero_count (row : ℤ → ℤ) : ℕ :=
  sorry

theorem at_most_n_zeros 
  (s : Sheet) 
  (h_valid : is_valid_sheet s) 
  (r : ℤ) 
  (h_positive : ∀ j : ℤ, (Row s r j) > 0) 
  (n : ℕ) :
  zero_count (Row s (r + ↑n)) ≤ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_n_zeros_l914_91482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_average_speed_l914_91467

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem carmen_average_speed :
  let total_distance : ℝ := 35
  let total_time : ℝ := 7
  average_speed total_distance total_time = 5 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_average_speed_l914_91467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_bisector_foot_distance_l914_91426

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  F₁.1 = -F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0 ∧ F₂.1 > 0

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem hyperbola_bisector_foot_distance 
  (F₁ F₂ P H : ℝ × ℝ) :
  foci F₁ F₂ →
  on_hyperbola P →
  -- H is on the bisector of ∠F₁PF₂
  distance P H = distance P F₁ →
  -- H is perpendicular to F₁H
  (H.1 - F₁.1) * (P.1 - H.1) + (H.2 - F₁.2) * (P.2 - H.2) = 0 →
  distance origin H = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_bisector_foot_distance_l914_91426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_path_max_distance_l914_91469

/-- Represents a turn direction -/
inductive Turn
| Left
| Right

/-- Represents a path on a plane -/
structure SnailPath where
  segments : ℕ
  turns : List Turn

/-- Calculates the maximum possible distance between start and end points of a path -/
noncomputable def max_distance (p : SnailPath) : ℝ :=
  sorry

theorem snail_path_max_distance :
  let p : SnailPath := {
    segments := 300,
    turns := List.replicate 99 Turn.Left ++ List.replicate 200 Turn.Right
  }
  max_distance p = 100 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_path_max_distance_l914_91469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l914_91464

/-- A line in the xy-plane with y-intercept 2 and passing through (259, 520) has slope 2 -/
theorem line_slope (m b : ℝ) : 
  b = 2 ∧ (∃ (x y : ℝ), x = 259 ∧ y = 520 ∧ y = m * x + b) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l914_91464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l914_91492

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_theorem (e : Ellipse) 
  (focus1 : Point) (focus2 : Point) (p : Point) :
  focus1 = ⟨0, -1⟩ →
  focus2 = ⟨0, -3⟩ →
  p = ⟨5, -2⟩ →
  (∀ x y : ℝ, (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1 ↔ 
    distance ⟨x, y⟩ focus1 + distance ⟨x, y⟩ focus2 = 2 * e.a) →
  e.a + e.k = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l914_91492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_exists_l914_91406

/-- The slope of line e passing through point E -/
def m : ℝ := sorry

/-- Point E on the circle -/
noncomputable def E : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Point P where line e intersects AB -/
noncomputable def P : ℝ × ℝ := (1, m * (1 - Real.sqrt 2))

/-- Point Q where line e intersects CD -/
noncomputable def Q : ℝ × ℝ := (-1, -m * (1 + Real.sqrt 2))

/-- Midpoint of PQ -/
noncomputable def K : ℝ × ℝ := (0, -m * Real.sqrt 2)

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 - 2 * m * Real.sqrt 2 * y = 1 - m^2

/-- The unit circle equation -/
def unit_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Theorem stating the conditions for the existence of the hyperbola -/
theorem hyperbola_exists : 
  (∃ (x y : ℝ), hyperbola_equation x y ∧ unit_circle_equation x y) ↔ 
  (m^2 ≤ 2 ∧ m ≠ 1 ∧ m ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_exists_l914_91406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l914_91405

noncomputable def purchase_price : ℝ := 800
noncomputable def repair_costs : ℝ := 200
noncomputable def selling_price : ℝ := 1200

noncomputable def total_cost : ℝ := purchase_price + repair_costs
noncomputable def gain : ℝ := selling_price - total_cost
noncomputable def gain_percent : ℝ := (gain / total_cost) * 100

theorem scooter_gain_percent : gain_percent = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l914_91405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_equipment_usage_l914_91422

/-- The average annual cost function for the equipment -/
noncomputable def average_annual_cost (x : ℝ) : ℝ := 1 + 10 / x + x / 10

/-- Theorem stating the optimal usage time and minimum cost -/
theorem optimal_equipment_usage :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ y : ℝ, y > 0 → average_annual_cost x ≤ average_annual_cost y) ∧
  average_annual_cost x = 3 ∧
  x = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_equipment_usage_l914_91422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zumish_word_count_remainder_l914_91465

/-- Definition of Zumish language letters -/
inductive ZumishLetter
| M
| O
| P
| Z

/-- Function to check if a letter is a vowel -/
def isVowel (l : ZumishLetter) : Bool :=
  match l with
  | ZumishLetter.O => true
  | ZumishLetter.Z => true
  | _ => false

/-- Function to check if a word is valid in Zumish -/
def isValidZumishWord (word : List ZumishLetter) : Bool :=
  let rec checkValid (w : List ZumishLetter) (consonantCount : Nat) : Bool :=
    match w with
    | [] => true
    | h :: t =>
      if isVowel h then
        if consonantCount < 2 then false
        else checkValid t 0
      else
        checkValid t (consonantCount + 1)
  checkValid word 0

/-- Function to generate all possible words of length n -/
def generateAllWords (n : Nat) : List (List ZumishLetter) :=
  let allLetters := [ZumishLetter.M, ZumishLetter.O, ZumishLetter.P, ZumishLetter.Z]
  List.foldl (fun acc _ => List.join (acc.map (fun w => allLetters.map (fun l => l :: w))))
    [[]] (List.range n)

/-- Function to count valid n-letter Zumish words -/
def countValidZumishWords (n : Nat) : Nat :=
  let allWords := generateAllWords n
  (allWords.filter isValidZumishWord).length

/-- Theorem stating the remainder of 12-letter Zumish words when divided by 1000 -/
theorem zumish_word_count_remainder :
  countValidZumishWords 12 % 1000 = 322 := by
  sorry

#eval countValidZumishWords 12 % 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zumish_word_count_remainder_l914_91465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_t_value_l914_91433

noncomputable section

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def intersectionPoint (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  let x := (t - p1.2) * (p2.1 - p1.1) / (p2.2 - p1.2) + p1.1
  (x, t)

def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem triangle_intersection_t_value (ABC : Triangle) (t : ℝ) : 
  ABC.A = (0, 10) → ABC.B = (3, 0) → ABC.C = (9, 0) →
  let T := intersectionPoint ABC.A ABC.B t
  let U := intersectionPoint ABC.A ABC.C t
  triangleArea ABC.A T U = 15 →
  abs (t - 2.93) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_t_value_l914_91433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_equation_l914_91403

/-- Rotate a line counterclockwise around a point -/
def rotate_line (a b c : ℝ) (θ : ℝ) (x₀ y₀ : ℝ) : ℝ → ℝ → Prop :=
  λ x y => sorry

theorem rotated_line_equation :
  let original_line := λ x y => x - y + Real.sqrt 3 - 1 = 0
  let rotation_angle := 15 * π / 180
  let rotation_center := (1, Real.sqrt 3)
  let rotated_line := rotate_line 1 (-1) (Real.sqrt 3 - 1) rotation_angle 1 (Real.sqrt 3)
  ∀ x y, rotated_line x y ↔ y = Real.sqrt 3 * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_equation_l914_91403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_partitions_eq_43252_remainder_mod_1000_l914_91411

def S : Finset Nat := Finset.range 12

def valid_partition (A B : Finset Nat) : Prop :=
  A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.card > 0 ∧ B.card > 0 ∧ A.card = B.card + 3

def count_valid_partitions : Nat :=
  Finset.sum (Finset.range 5) (fun k =>
    Nat.choose 12 (k + 3) * Nat.choose (9 - k) k)

theorem count_valid_partitions_eq_43252 :
  count_valid_partitions = 43252 :=
sorry

theorem remainder_mod_1000 :
  count_valid_partitions % 1000 = 252 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_partitions_eq_43252_remainder_mod_1000_l914_91411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_iff_power_of_two_l914_91495

/-- Represents the three colors of cubes -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents a circular arrangement of cubes -/
def Arrangement (n : ℕ) := Fin n → Color

/-- The operation performed by the robot -/
def robotOperation (a b : Color) : Color :=
  match a, b with
  | Color.White, Color.White => Color.White
  | Color.Blue, Color.Blue => Color.Blue
  | Color.Red, Color.Red => Color.Red
  | _, _ => Color.Red  -- If colors are different, return the third color (arbitrarily chosen as Red)

/-- Helper function: Perform robot operation until one cube is left -/
def robotOperationUntilOne (n : ℕ) (start : Fin n) (arr : Arrangement n) : Color :=
  sorry  -- Implementation details omitted for brevity

/-- Checks if an arrangement is good -/
def isGoodArrangement (n : ℕ) (arr : Arrangement n) : Prop :=
  ∀ start₁ start₂, 
    (robotOperationUntilOne n start₁ arr) = (robotOperationUntilOne n start₂ arr)

/-- Checks if a number is lucky -/
def isLucky (n : ℕ) : Prop :=
  ∀ arr : Arrangement n, isGoodArrangement n arr

/-- A number is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The main theorem: A number is lucky if and only if it's a power of 2 -/
theorem lucky_iff_power_of_two (n : ℕ) :
  isLucky n ↔ isPowerOfTwo n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_iff_power_of_two_l914_91495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_increase_l914_91472

/-- Represents the travel conditions for Daniel's commute --/
structure TravelConditions where
  total_distance : ℝ
  sunday_speed : ℝ
  monday_speed1 : ℝ
  monday_speed2 : ℝ
  monday_speed3 : ℝ
  monday_speed4 : ℝ
  monday_distance1 : ℝ
  monday_distance2 : ℝ
  monday_distance3 : ℝ

/-- Calculates the percentage increase in travel time from Sunday to Monday --/
noncomputable def percentage_increase (tc : TravelConditions) : ℝ :=
  let sunday_time := tc.total_distance / tc.sunday_speed
  let monday_time1 := tc.monday_distance1 / tc.monday_speed1
  let monday_time2 := tc.monday_distance2 / tc.monday_speed2
  let monday_time3 := tc.monday_distance3 / tc.monday_speed3
  let monday_time4 := (tc.total_distance - tc.monday_distance1 - tc.monday_distance2 - tc.monday_distance3) / tc.monday_speed4
  let monday_time := monday_time1 + monday_time2 + monday_time3 + monday_time4
  ((monday_time - sunday_time) / sunday_time) * 100

/-- The main theorem stating the percentage increase in travel time --/
theorem travel_time_increase 
  (tc : TravelConditions)
  (h1 : tc.total_distance = 150)
  (h2 : tc.sunday_speed = x)
  (h3 : tc.monday_speed1 = 2 * x)
  (h4 : tc.monday_speed2 = x / 3)
  (h5 : tc.monday_speed3 = 3 * x / 2)
  (h6 : tc.monday_speed4 = x / 2)
  (h7 : tc.monday_distance1 = 32)
  (h8 : tc.monday_distance2 = 50)
  (h9 : tc.monday_distance3 = 40)
  (h10 : x > 0) :
  abs (percentage_increase tc - 65.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_increase_l914_91472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_1_ellipse_equation_2_l914_91489

/-- Definition of an ellipse with foci on the x-axis -/
structure EllipseOnXAxis where
  a : ℝ
  b : ℝ
  h : a > b

/-- Check if a point (x, y) is on the ellipse -/
def EllipseOnXAxis.contains (e : EllipseOnXAxis) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Focal distance of an ellipse -/
noncomputable def EllipseOnXAxis.focalDistance (e : EllipseOnXAxis) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Eccentricity of an ellipse -/
noncomputable def EllipseOnXAxis.eccentricity (e : EllipseOnXAxis) : ℝ :=
  e.focalDistance / e.a

theorem ellipse_equation_1 (e : EllipseOnXAxis) :
  e.focalDistance = 4 → e.contains 3 (-2 * Real.sqrt 6) →
  e.a^2 = 36 ∧ e.b^2 = 32 := by sorry

theorem ellipse_equation_2 (e : EllipseOnXAxis) :
  e.focalDistance = 8 → e.eccentricity = 0.8 →
  (e.a^2 = 25 ∧ e.b^2 = 9) ∨ (e.a^2 = 9 ∧ e.b^2 = 25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_1_ellipse_equation_2_l914_91489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l914_91484

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through : (center.1 + 2)^2 + (center.2 - 1)^2 = 2

theorem circle_equation (c : Circle) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ∨ (x + 3)^2 + y^2 = 2 ↔
   (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l914_91484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l914_91421

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.sin (2 * ω * x + Real.pi / 3) - 4 * (Real.cos (ω * x))^2 + 3

theorem problem_solution (ω : ℝ) (h_ω : 0 < ω ∧ ω < 2) 
  (h_symmetry : ∀ x, f ω x = f ω (Real.pi / 3 - x)) :
  (ω = 1 ∧ ∃ x, ∀ y, f ω x ≤ f ω y ∧ f ω x = -1) ∧
  (∀ A B C : ℝ, 
    0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
    1 * Real.sin B / 2 = Real.sqrt 3 / 4 →
    f ω A = 2 →
    1 + Real.sqrt (2 - 2 * Real.cos A) + Real.sqrt (2 + 2 * Real.cos A) = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l914_91421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_distance_after_3_minutes_l914_91402

/-- The distance between two vehicles after a given time -/
noncomputable def distance_between (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Conversion factor from minutes to hours -/
noncomputable def minutes_to_hours (m : ℝ) : ℝ :=
  m / 60

theorem vehicle_distance_after_3_minutes :
  let truck_speed := (65 : ℝ)
  let car_speed := (85 : ℝ)
  let time_minutes := (3 : ℝ)
  distance_between truck_speed car_speed (minutes_to_hours time_minutes) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_distance_after_3_minutes_l914_91402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l914_91415

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a
  else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  6/5 ≤ a ∧ a < 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l914_91415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_Z_eq_sqrt2_div_2_l914_91437

noncomputable def Z : ℂ := (1 : ℂ) / (1 - Complex.I) - Complex.I

theorem abs_Z_eq_sqrt2_div_2 : Complex.abs Z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_Z_eq_sqrt2_div_2_l914_91437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_pfq_approx_48_l914_91420

/-- Hyperbola with equation x²/9 - y²/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- Left focus of the hyperbola -/
def F : ℝ × ℝ := (-5, 0)

/-- Point on the x-axis -/
def A : ℝ × ℝ := (5, 0)

/-- Points on the right branch of the hyperbola -/
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

/-- The length of PQ is twice the length of the imaginary axis -/
axiom pq_length : dist P Q = 16

/-- A is on the line segment PQ -/
axiom a_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • P + t • Q

/-- P and Q are on the hyperbola -/
axiom p_on_hyperbola : hyperbola P.1 P.2
axiom q_on_hyperbola : hyperbola Q.1 Q.2

/-- Theorem: The perimeter of triangle PFQ is approximately 48 -/
theorem perimeter_pfq_approx_48 : 
  |((dist P F + dist Q F + dist P Q) - 48)| < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_pfq_approx_48_l914_91420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_in_Y_l914_91490

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  other : ℝ
  sum_to_one : ryegrass + other = 1

/-- The composition of seed mixture X -/
def X : SeedMixture where
  ryegrass := 0.4
  other := 0.6
  sum_to_one := by norm_num

/-- The composition of seed mixture Y -/
def Y (y : ℝ) : SeedMixture where
  ryegrass := y
  other := 0.75
  sum_to_one := sorry

/-- The proportion of X in the final mixture -/
noncomputable def x_proportion : ℝ := 1 / 3

theorem ryegrass_in_Y (y : ℝ) :
  (Y y).ryegrass = y →
  x_proportion * X.ryegrass + (1 - x_proportion) * y = 0.3 →
  y = 0.25 := by
  sorry

#check ryegrass_in_Y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_in_Y_l914_91490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_vertex_orthocenter_circumcenter_l914_91475

/-- Predicate to check if three points form a triangle -/
def IsTriangle (s : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
sorry

/-- Orthocenter of a triangle -/
def Orthocenter (triangle : Set (EuclideanSpace ℝ (Fin 2))) : EuclideanSpace ℝ (Fin 2) :=
sorry

/-- Circumcenter of a triangle -/
def Circumcenter (triangle : Set (EuclideanSpace ℝ (Fin 2))) : EuclideanSpace ℝ (Fin 2) :=
sorry

/-- Given three points A, H, and O in a plane, prove that there exists a unique triangle ABC
    such that A is a vertex, H is the orthocenter, and O is the circumcenter of the triangle. -/
theorem triangle_construction_from_vertex_orthocenter_circumcenter
  (A H O : EuclideanSpace ℝ (Fin 2)) :
  ∃! (B C : EuclideanSpace ℝ (Fin 2)), 
    let triangle := {A, B, C}
    IsTriangle triangle ∧
    Orthocenter triangle = H ∧
    Circumcenter triangle = O :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_vertex_orthocenter_circumcenter_l914_91475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_sum_l914_91445

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the function h as the sum of f and g
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Define the fundamental period of a function
def is_fundamental_period (T : ℝ) (func : ℝ → ℝ) : Prop :=
  ∀ x, func (x + T) = func x ∧ ∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x, func (x + T') ≠ func x

-- State the theorem
theorem smallest_period_of_sum :
  (is_fundamental_period 1 f) →
  (is_fundamental_period 1 g) →
  ∃ k : ℕ, is_fundamental_period (1 / k) h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_sum_l914_91445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_reaches_60m_l914_91452

/-- Represents the monkey's climbing pattern -/
def MonkeyClimb := ℕ → ℤ

/-- The monkey's height after n minutes -/
def monkeyHeight (climb : MonkeyClimb) (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (n / 2) * 3
  else
    (n / 2) * 3 + 6

/-- The climbing pattern of the monkey -/
def monkeyClimb : MonkeyClimb :=
  λ n => if n % 2 = 0 then -3 else 6

theorem monkey_reaches_60m :
  ∃ (n : ℕ), monkeyHeight monkeyClimb n ≥ 60 ∧ 
  (∀ (m : ℕ), m < n → monkeyHeight monkeyClimb m < 60) ∧
  n = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_reaches_60m_l914_91452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l914_91477

/-- Calculates the average speed of a round trip given the one-way distance, uphill time, and downhill time -/
noncomputable def averageSpeed (distance : ℝ) (uphillTime downhillTime : ℝ) : ℝ :=
  (2 * distance) / ((uphillTime + downhillTime) / 60)

/-- Theorem stating that for the given conditions, the average speed is 2.5 km/hr -/
theorem mary_average_speed :
  let distance : ℝ := 1 -- km
  let uphillTime : ℝ := 40 -- minutes
  let downhillTime : ℝ := 8 -- minutes
  averageSpeed distance uphillTime downhillTime = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l914_91477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_six_digit_number_l914_91400

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (t b : ℕ) : ℕ := 199000 + t * 10 + b

theorem unique_prime_six_digit_number :
  ∃! b : ℕ, b ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
  ∃ t : ℕ, t < 10 ∧ is_prime (six_digit_number t b) ∧
  ∀ b' : ℕ, b' ∈ ({1, 2, 3, 4, 5} : Finset ℕ) → b' ≠ b →
  ∀ t' : ℕ, t' < 10 → ¬(is_prime (six_digit_number t' b')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_six_digit_number_l914_91400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l914_91442

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 + a ≤ (a + 1) * x

-- Define the solution set of the inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Define the given interval
def given_interval : Set ℝ := Set.Icc (-3) 2

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, solution_set a ⊆ given_interval) ↔ a ∈ Set.Icc (-3) 2 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l914_91442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_45_degree_angle_l914_91488

/-- Given a line with a slope angle of 45 degrees, its slope is 1 -/
theorem slope_of_45_degree_angle (a : Real) (h : Real.tan a = Real.tan (π/4)) : Real.tan a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_45_degree_angle_l914_91488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_angles_l914_91463

/-- Given a triangle ABC with sides a = 2, b = √2, and c = √3 + 1, 
    prove that its angles are A = π/4, B = π/6, and C = 7π/12 -/
theorem triangle_abc_angles (a b c : ℝ) (h_a : a = 2) (h_b : b = Real.sqrt 2) 
    (h_c : c = Real.sqrt 3 + 1) : 
  ∃ (A B C : ℝ), 
    A = π/4 ∧ 
    B = π/6 ∧ 
    C = 7*π/12 ∧
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧
    a / Real.sin A = b / Real.sin B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_angles_l914_91463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_swim_time_l914_91461

/-- Represents the swimming scenario with given distances and speeds -/
structure SwimmingScenario where
  downstream_distance : ℝ
  upstream_distance : ℝ
  swimmer_speed : ℝ
  current_speed : ℝ

/-- Calculates the time taken to swim a given distance with a given speed -/
noncomputable def swim_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that under the given conditions, the swim time is 3 hours in each direction -/
theorem equal_swim_time (scenario : SwimmingScenario) 
  (h1 : scenario.downstream_distance = 45)
  (h2 : scenario.upstream_distance = 18)
  (h3 : scenario.swimmer_speed = 10.5)
  (h4 : swim_time scenario.downstream_distance (scenario.swimmer_speed + scenario.current_speed) = 
        swim_time scenario.upstream_distance (scenario.swimmer_speed - scenario.current_speed)) :
  swim_time scenario.downstream_distance (scenario.swimmer_speed + scenario.current_speed) = 3 ∧
  swim_time scenario.upstream_distance (scenario.swimmer_speed - scenario.current_speed) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_swim_time_l914_91461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l914_91457

theorem matrix_vector_computation 
  (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (u z : Fin 2 → ℝ) 
  (hu : M.mulVec u = ![3, -2])
  (hz : M.mulVec z = ![-1, 4]) :
  M.mulVec (3 • u - 2 • z) = ![11, -14] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l914_91457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_trig_matrix_zero_l914_91459

open Matrix Real

theorem det_trig_matrix_zero (α β φ : ℝ) : 
  det !![0, sin (α + φ), -cos (α + φ);
        -sin (α + φ), 0, sin (β + φ);
        cos (α + φ), -sin (β + φ), 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_trig_matrix_zero_l914_91459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_multiplication_l914_91460

theorem unique_digit_multiplication (B : ℕ) : 
  B < 10 → 
  (10 * B + 5) * (90 + B) = 9045 → 
  B = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_multiplication_l914_91460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_for_Q_necessary_not_sufficient_for_P_l914_91431

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sets P and Q
def P (t : ℝ) : Set ℝ := {x | |f (x + t) - 1| < 1}
def Q : Set ℝ := {x | f x < -2}

-- State the theorem
theorem t_range_for_Q_necessary_not_sufficient_for_P :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x y : ℝ, x < y → f y < f x) →  -- f is decreasing
  f 3 = -2 →
  (∀ t : ℝ, (∀ x : ℝ, x ∈ Q → x ∈ P t) ∧ 
    (∃ x : ℝ, x ∈ P t ∧ x ∉ Q)) →
  {t : ℝ | t ≤ -6} = {t : ℝ | ∀ x : ℝ, x ∈ Q → x ∈ P t} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_for_Q_necessary_not_sufficient_for_P_l914_91431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sin_double_angle_l914_91487

theorem inverse_sin_double_angle (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sin_double_angle_l914_91487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l914_91496

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * cos t.B = 2 * t.a - t.b) 
  (h2 : 0 < t.A ∧ t.A < π)
  (h3 : 0 < t.B ∧ t.B < π)
  (h4 : 0 < t.C ∧ t.C < π) :
  t.C = π / 3 ∧ 
  (t.c = 3 → t.a + t.b + t.c ≤ 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l914_91496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l914_91414

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  prop1 : a 3 * a 7 = -16
  prop2 : a 4 + a 6 = 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) (fun i => seq.a (i + 1))

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = n * (n - 9) ∨ sum_n seq n = -n * (n - 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l914_91414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l914_91412

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x

noncomputable def k : ℝ := f' 1

def perp_line (m : ℝ) (x y : ℝ) : Prop := 2 * x + m * y + 1 = 0

theorem tangent_perpendicular_line (m : ℝ) :
  (∃ y, perp_line m 1 y ∧ k * (y - f 1) = y - f 1) → m = 2 * Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l914_91412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l914_91419

theorem number_of_factors (n : ℕ) (h : n = 2^4 * 3^5 * 5^6 * 7^7) :
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 1680 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l914_91419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_halfway_fraction_l914_91416

theorem double_halfway_fraction : (5 : ℚ) / 12 = 2 * ((1 / 6 + 1 / 4) / 2) := by
  -- Define the two initial fractions
  let a : ℚ := 1/6
  let b : ℚ := 1/4
  
  -- Define the halfway point between a and b
  let halfway : ℚ := (a + b) / 2
  
  -- Define the fraction that is double the distance of the halfway point
  let result : ℚ := 2 * halfway
  
  -- The proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_halfway_fraction_l914_91416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l914_91494

/-- Custom operation ⊗ -/
noncomputable def custom_op (a b : ℝ) : ℝ := 
  if a < b then (b - 1) / a else (a + 1) / b

/-- Theorem statement -/
theorem custom_op_example : 
  let log_10000 : ℝ := 4
  let half_inv_square : ℝ := 4
  custom_op log_10000 half_inv_square = 5/4 := by
  -- Unfold the definitions
  unfold custom_op
  -- Since log_10000 = half_inv_square, we use the second case of custom_op
  simp [if_neg]
  -- Simplify the arithmetic
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l914_91494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_procedures_correct_l914_91471

/-- The maximum number of procedures needed to make all boxes have a multiple of 3 balls -/
def max_procedures (n : ℕ) : ℕ :=
  let m := n / 3
  if n % 3 = 0 then 2 * m + 1 else 2 * m + 2

/-- The procedure to add balls to boxes -/
def add_balls (n : ℕ) (i j : ℕ) (boxes : Fin n → ℕ) : Fin n → ℕ :=
  λ k ↦ if j ≤ k.val ∧ k.val ≤ i then boxes k + 1 else boxes k

/-- The minimum number of procedures to make all boxes have a multiple of 3 balls -/
noncomputable def min_procedures (n : ℕ) (initial_boxes : Fin n → ℕ) : ℕ :=
  sorry

theorem max_procedures_correct (n : ℕ) :
  ∀ initial_boxes : Fin n → ℕ, min_procedures n initial_boxes ≤ max_procedures n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_procedures_correct_l914_91471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_count_l914_91413

/-- The number of bottles in a box containing water, orange juice, and apple juice bottles -/
theorem bottle_count : ℕ := by
  let water_bottles : ℕ := 2 * 12
  let orange_juice_bottles : ℕ := (7 * 12) / 4
  let apple_juice_bottles : ℕ := water_bottles + 6
  have h : water_bottles + orange_juice_bottles + apple_juice_bottles = 75 := by
    sorry
  exact 75


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_count_l914_91413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l914_91440

theorem smallest_odd_with_five_prime_factors : 
  ∀ n : ℕ, Odd n → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) → 
  n ≥ 15015 := by
  sorry

#check smallest_odd_with_five_prime_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l914_91440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_in_terms_of_p_q_l914_91401

theorem determinant_in_terms_of_p_q (p q a b c : ℝ) : 
  (a^3 - 3*p*a + 2*q = 0) →
  (b^3 - 3*p*b + 2*q = 0) →
  (c^3 - 3*p*c + 2*q = 0) →
  Matrix.det ![![2 + a, 1, 1],
                ![1, 2 + b, 1],
                ![1, 1, 2 + c]] = -3*p - 2*q + 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_in_terms_of_p_q_l914_91401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_population_after_9_days_l914_91451

/-- Calculates the number of cells after a given number of days, given an initial population,
    doubling period, and survival rate. -/
def cellPopulation (initialCells : ℕ) (doublingPeriod : ℕ) (survivalRate : ℚ) (days : ℕ) : ℕ :=
  let cycles := days / doublingPeriod
  let finalPopulation := (initialCells : ℚ) * (2 * survivalRate) ^ cycles
  Int.floor finalPopulation |>.toNat

/-- Theorem stating that given 5 initial cells, doubling every 3 days with a 90% survival rate,
    the population after 9 days is 28 cells. -/
theorem cell_population_after_9_days :
  cellPopulation 5 3 (9/10) 9 = 28 := by
  sorry

#eval cellPopulation 5 3 (9/10) 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_population_after_9_days_l914_91451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_championship_draw_matches_l914_91441

theorem football_championship_draw_matches 
  (num_teams : ℕ) 
  (num_rounds : ℕ) 
  (total_points : ℕ) 
  (points_for_win : ℕ) 
  (points_for_draw : ℕ) 
  (h1 : num_teams = 16)
  (h2 : num_rounds = 16)
  (h3 : total_points = 222)
  (h4 : points_for_win = 2)
  (h5 : points_for_draw = 1) :
  (num_teams * num_rounds / 2) * points_for_win - total_points = 34 := by
  sorry

#check football_championship_draw_matches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_championship_draw_matches_l914_91441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_five_l914_91480

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * (1 / (x - 1)) + 3

-- State the theorem
theorem f_of_two_equals_five : f 2 = 5 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_five_l914_91480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_70_l914_91428

/-- The 150th digit to the right of the decimal point in the decimal representation of 17/70 is 2. -/
theorem digit_150_of_17_70 : ∃ (s : List Nat), 
  (∀ n, n ≥ 1 → n ≤ 150 → s.get? (n - 1) = some ((17 * 10^n / 70) % 10)) ∧ 
  s.get? 149 = some 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_70_l914_91428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_to_raisins_ratio_l914_91468

/-- The ratio of the cost of nuts to raisins in Chris's mixture -/
theorem nuts_to_raisins_ratio (raisin_cost nut_cost : ℝ) :
  raisin_cost > 0 → nut_cost > 0 →
  (3 * raisin_cost) / (3 * raisin_cost + 4 * nut_cost) = 3 / 11 →
  nut_cost / raisin_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_to_raisins_ratio_l914_91468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_occupancy_problem_l914_91449

/-- The number of people on the bus at each stage --/
structure BusOccupancy where
  initial : ℕ
  after_first : ℕ
  after_second : ℕ
  after_third : ℕ
  final : ℕ

/-- The changes in occupancy at each stop --/
structure BusStops where
  first_off : ℕ
  first_on : ℕ
  second_off : ℕ
  second_on : ℕ
  third_off : ℕ
  third_on : ℕ
  final_off : ℕ

def bus_journey (occupancy : BusOccupancy) (stops : BusStops) : Prop :=
  occupancy.after_first = occupancy.initial - stops.first_off + stops.first_on ∧
  occupancy.after_second = occupancy.after_first - stops.second_off + stops.second_on ∧
  occupancy.after_third = occupancy.after_second - stops.third_off + stops.third_on ∧
  occupancy.final = occupancy.after_third - stops.final_off

theorem bus_occupancy_problem (stops : BusStops) 
  (h_first_off : stops.first_off = 75)
  (h_first_on : stops.first_on = 62)
  (h_second_off : stops.second_off = 59)
  (h_second_on : stops.second_on = 88)
  (h_third_off : stops.third_off = 96)
  (h_third_on : stops.third_on = 53)
  (h_final_off : stops.final_off = 112) :
  ∃ (occupancy : BusOccupancy), bus_journey occupancy stops ∧ occupancy.initial = 178 ∧ occupancy.final = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_occupancy_problem_l914_91449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_216_l914_91486

def sequenceQ (n : ℕ) : ℚ := (n : ℚ) ^ (n - 3)

theorem sixth_term_is_216 : sequenceQ 6 = 216 := by
  -- Unfold the definition of sequenceQ
  unfold sequenceQ
  -- Simplify the expression
  simp
  -- Evaluate the exponentiation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_216_l914_91486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l914_91425

/-- Given a real number a ≥ -2, prove that if C ⊆ B, then 1/2 ≤ a ≤ 3 -/
theorem range_of_a (a : ℝ) (h_a : a ≥ -2) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
  let B : Set ℝ := {y | ∃ x ∈ A, y = 2 * x + 3}
  let C : Set ℝ := {z | ∃ x ∈ A, z = x^2}
  (C ⊆ B) → (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l914_91425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l914_91418

theorem salary_change (S : ℝ) (hS : S > 0) : 
  let change1 := S * (1 - 0.35)
  let change2 := change1 * (1 + 0.20)
  let change3 := change2 * (1 - 0.15)
  let final_salary := change3 * (1 + 0.25)
  (final_salary - S) / S * 100 = -17.125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l914_91418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_35_decomposition_l914_91410

theorem power_35_decomposition (a b : ℤ) :
  (5 : ℝ)^(a : ℝ)^(b : ℝ) * (7 : ℝ)^(b : ℝ)^(a : ℝ) = (35 : ℝ)^((a * b) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_35_decomposition_l914_91410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l914_91423

/-- Curve C is defined by x = 3cosθ, y = 2sinθ -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

/-- Line l has polar equation ρ(cosθ - 2sinθ) = 12 -/
def line_l (x y : ℝ) : Prop := x - 2*y = 12

/-- The distance from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3 * x - 4 * y - 12| / Real.sqrt 5

/-- The theorem stating the minimum distance from curve C to line l -/
theorem min_distance_curve_to_line :
  ∀ θ : ℝ, ∃ d : ℝ, d = distance_to_line (curve_C θ).1 (curve_C θ).2 ∧
  d ≥ (7 * Real.sqrt 5) / 5 ∧
  ∃ θ₀ : ℝ, distance_to_line (curve_C θ₀).1 (curve_C θ₀).2 = (7 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l914_91423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l914_91474

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)^(-x^2 - 4*x + 3)

-- State the theorem
theorem f_monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y ∧ y ≤ -2 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l914_91474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completes_in_eight_days_l914_91453

/-- The number of days Tanya takes to complete a piece of work, given Sakshi's completion time and Tanya's efficiency relative to Sakshi. -/
noncomputable def tanya_completion_time (sakshi_days : ℝ) (tanya_efficiency : ℝ) : ℝ :=
  sakshi_days / (1 + tanya_efficiency)

/-- Theorem stating that Tanya completes the work in 8 days given the problem conditions. -/
theorem tanya_completes_in_eight_days (sakshi_days : ℝ) (tanya_efficiency : ℝ)
  (h1 : sakshi_days = 10)
  (h2 : tanya_efficiency = 0.25) :
  tanya_completion_time sakshi_days tanya_efficiency = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completes_in_eight_days_l914_91453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_horizontal_distance_l914_91450

-- Define the parabola function
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Define the set of x-coordinates for point P
def P : Set ℝ := {x | f x = 10}

-- Define the set of x-coordinates for point Q
def Q : Set ℝ := {x | f x = 0}

-- Theorem statement
theorem smallest_horizontal_distance :
  ∃ (p q : ℝ), p ∈ P ∧ q ∈ Q ∧
    (∀ (p' q' : ℝ), p' ∈ P → q' ∈ Q → |p - q| ≤ |p' - q'|) ∧
    |p - q| = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_horizontal_distance_l914_91450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l914_91429

/-- Given a point P(m, -√3) on the terminal side of angle α, where m ≠ 0 and cos α = (√2 * m) / 4,
    prove the values of m, sin α, and tan α. -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) (h1 : m ≠ 0) (h2 : Real.cos α = (Real.sqrt 2 * m) / 4) :
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.sin α = -(Real.sqrt 6) / 4 ∧
  Real.tan α = -(Real.sqrt 15) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l914_91429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l914_91473

noncomputable section

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- Define the tangent line l
def line_l (x y : ℝ) : Prop :=
  y = x + 2

-- Define the circle O
def circle_O (x y b : ℝ) : Prop :=
  x^2 + y^2 = b^2

-- Main theorem
theorem ellipse_properties (a b : ℝ) 
  (h_focal : focal_length a b = 2)
  (h_tangent : ∃ x y, line_l x y ∧ circle_O x y b) :
  -- 1) Equation of ellipse C
  (∀ x y, ellipse_C x y a b ↔ x^2/3 + y^2/2 = 1) ∧
  -- 2) Value of k
  (∀ x₀ y₀ k, 
    ellipse_C x₀ y₀ a b → 
    y₀ = k * x₀ → 
    k > 0 → 
    x₀ * Real.sqrt 2 + y₀ * 1 = Real.sqrt 6 → 
    k = Real.sqrt 2) ∧
  -- 3) Maximum area of triangle AOD
  (∃ max_area : ℝ, 
    (∀ x₀ y₀, 
      ellipse_C x₀ y₀ a b → 
      x₀ > 0 → 
      y₀ > 0 → 
      x₀ * y₀ ≤ max_area) ∧
    max_area = Real.sqrt 6 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l914_91473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l914_91438

/-- Predicate indicating that a, b, c form a triangle -/
def IsTriangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate indicating that angles A, B, C are opposite to sides a, b, c respectively -/
def OppositeAngles (a b c A B C : ℝ) : Prop :=
  sorry

/-- Predicate indicating that a, b, c form a right triangle -/
def RightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if cos²(A/2) = (b+c)/(2c), then ABC is a right triangle. -/
theorem triangle_right_angle (a b c : ℝ) (A B C : ℝ) 
    (h_triangle : IsTriangle a b c)
    (h_opposite : OppositeAngles a b c A B C)
    (h_cos_sq : (Real.cos (A / 2))^2 = (b + c) / (2 * c)) :
    RightTriangle a b c :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l914_91438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_distances_l914_91427

/-- The point that minimizes the sum of distances from two fixed points in a plane lies on the line connecting the reflection of one point across the y-axis and the other point. --/
theorem minimize_sum_of_distances (A B C : ℝ × ℝ) (k : ℝ) : 
  A = (3, 4) → B = (6, 2) → C = (-2, k) →
  (∀ k' : ℝ, Real.sqrt ((3 - (-2))^2 + (4 - k)^2) + Real.sqrt ((6 - (-2))^2 + (2 - k)^2) 
            ≤ Real.sqrt ((3 - (-2))^2 + (4 - k')^2) + Real.sqrt ((6 - (-2))^2 + (2 - k')^2)) →
  k = 34/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_distances_l914_91427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_four_eq_neg_seven_l914_91493

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 5 else 6 - x

-- Theorem statement
theorem g_neg_four_eq_neg_seven : g (-4) = -7 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the if-then-else expression
  simp [show (-4 : ℝ) < 0 from by norm_num]
  -- Perform arithmetic simplification
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_four_eq_neg_seven_l914_91493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_speed_calculation_l914_91470

/-- Proves that given a distance of 9.999999999999993 km, if a person walks at 5 kmph and arrives 10 minutes late, then to arrive 10 minutes early, they need to walk at approximately 6 kmph. -/
theorem bus_stop_speed_calculation (distance : ℝ) (initial_speed : ℝ) (late_time : ℝ) (early_time : ℝ) :
  distance = 9.999999999999993 →
  initial_speed = 5 →
  late_time = 1/6 →
  early_time = 1/6 →
  ∃ (final_speed : ℝ), 
    (distance / initial_speed - late_time) - early_time = distance / final_speed ∧
    abs (final_speed - 6) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_speed_calculation_l914_91470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_satisfies_conditions_l914_91476

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the conditions
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 = 11 ∧ a 1 * a 6 = 32 / 9

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) > a n

def condition3 (a : ℕ → ℝ) : Prop :=
  ∃ m : ℕ, m > 4 ∧ 
    (2 / 3 * a (m - 1) - a m ^ 2 = a m ^ 2 - (a (m + 1) + 4 / 9))

-- The main theorem
theorem no_geometric_sequence_satisfies_conditions :
  ¬ ∃ a : ℕ → ℝ, geometric_sequence a ∧ condition1 a ∧ condition2 a ∧ condition3 a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_satisfies_conditions_l914_91476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l914_91444

theorem expression_value : ((((3 : ℝ) + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2 = 65 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l914_91444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_reductions_equal_single_reduction_l914_91417

theorem two_reductions_equal_single_reduction (P : ℝ) (h : P > 0) :
  P * (1 - 0.25) * (1 - 0.60) = P * (1 - 0.70) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_reductions_equal_single_reduction_l914_91417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_patch_cost_ratio_l914_91434

def quilt_length : ℝ := 16
def quilt_width : ℝ := 20
def patch_area : ℝ := 4
def first_ten_cost : ℝ := 10
def total_patches_cost : ℝ := 450

theorem quilt_patch_cost_ratio :
  (let total_area : ℝ := quilt_length * quilt_width
   let total_patches : ℝ := total_area / patch_area
   let first_ten_total : ℝ := 10 * first_ten_cost
   let remaining_patches_cost : ℝ := total_patches_cost - first_ten_total
   remaining_patches_cost / first_ten_total) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_patch_cost_ratio_l914_91434
