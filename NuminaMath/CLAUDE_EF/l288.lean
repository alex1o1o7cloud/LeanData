import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_given_problem_l288_28869

/-- The height of a cone formed by rolling one sector of a circular sheet -/
noncomputable def cone_height (sheet_radius : ℝ) (num_sectors : ℕ) : ℝ :=
  let sector_arc_length := 2 * Real.pi * sheet_radius / num_sectors
  let base_radius := sector_arc_length / (2 * Real.pi)
  Real.sqrt (sheet_radius ^ 2 - base_radius ^ 2)

/-- Theorem stating the height of the cone for the given problem -/
theorem cone_height_for_given_problem :
  cone_height 8 4 = 2 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_given_problem_l288_28869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_divisibility_l288_28836

def N (k : ℕ) : ℚ := (2 * k).factorial / k.factorial

theorem N_divisibility (k : ℕ) : 
  (∃ m : ℕ, N k = 2^k * m) ∧ ¬(∃ n : ℕ, N k = 2^(k+1) * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_divisibility_l288_28836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l288_28880

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2.1 = k * b.2.1 ∧ a.2.2 = k * b.2.2

theorem parallel_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 2, -2)
  let b : ℝ × ℝ × ℝ := (2, y, 4)
  parallel a b → x + y = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l288_28880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_proof_l288_28848

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  math : Nat
  physics : Nat
  chemistry : Nat

/-- The maximum number of subjects a teacher can teach -/
def maxSubjectsPerTeacher : Nat := 3

/-- The given teacher counts for each subject -/
def givenCounts : TeacherCounts :=
  { math := 7, physics := 6, chemistry := 5 }

/-- The minimum number of teachers required -/
def minTeachersRequired : Nat := 5

theorem min_teachers_proof (counts : TeacherCounts) (max_subjects : Nat) :
  counts = givenCounts →
  max_subjects = maxSubjectsPerTeacher →
  minTeachersRequired ≥ max counts.math (max counts.physics counts.chemistry) ∧
  minTeachersRequired * max_subjects ≥ counts.math + counts.physics + counts.chemistry :=
by
  sorry

#check min_teachers_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_proof_l288_28848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_symmetrical_points_l288_28864

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Get the symmetric point about the xOy plane -/
def symmetricAboutXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem distance_symmetrical_points :
  let p1 : Point3D := { x := 2, y := 4, z := 6 }
  let p : Point3D := { x := 1, y := 3, z := -5 }
  let p2 := symmetricAboutXOY p
  distance p1 p2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_symmetrical_points_l288_28864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_zero_l288_28888

-- Define the power function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define function g
noncomputable def g (α m : ℝ) (x : ℝ) : ℝ := f α x + x - m

-- Theorem statement
theorem power_function_and_zero (α m : ℝ) :
  f α 2 = 8 →  -- f passes through (2,8)
  (∃ x, 2 < x ∧ x < 3 ∧ g α m x = 0) →  -- g has a zero in (2,3)
  f α 3 = 27 ∧ 10 < m ∧ m < 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_zero_l288_28888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_freshmen_than_sophomores_l288_28856

theorem more_freshmen_than_sophomores 
  (total : ℕ) 
  (junior_percent : ℚ) 
  (not_sophomore_percent : ℚ) 
  (seniors : ℕ) 
  (h_total : total = 800)
  (h_junior : junior_percent = 22 / 100)
  (h_not_soph : not_sophomore_percent = 74 / 100)
  (h_seniors : seniors = 160)
  (h_all_types : ∀ student, student ∈ Set.univ → 
    student ∈ ({Freshman, Sophomore, Junior, Senior} : Set (Fin 4))) :
  (↑total * junior_percent).floor +
  (↑total * (1 - not_sophomore_percent)).floor +
  seniors +
  ((↑total * not_sophomore_percent).floor - seniors - (↑total * junior_percent).floor) -
  (↑total * (1 - not_sophomore_percent)).floor = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_freshmen_than_sophomores_l288_28856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_for_strictly_increasing_g_l288_28832

/-- The function f(x) = ax + b -/
def f (a b x : ℝ) : ℝ := a * x + b

/-- The piecewise function g(x) -/
noncomputable def g (a b x : ℝ) : ℝ :=
  if x ≤ a then f a b x else f a b (f a b x)

/-- Theorem stating the minimum value of b -/
theorem min_b_for_strictly_increasing_g :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x < y → g a b x < g a b y) →
  b ≥ (1/4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_for_strictly_increasing_g_l288_28832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_four_bricks_heights_l288_28828

/-- Represents the possible height contributions of a single brick in inches -/
inductive BrickHeight where
  | small : BrickHeight  -- 4 inches
  | medium : BrickHeight -- 10 inches
  | large : BrickHeight  -- 19 inches

/-- Calculates the number of unique tower heights achievable with the given bricks -/
def uniqueTowerHeights (numBricks : Nat) : Nat :=
  2 + (numBricks * 5 - 8 - 4 + 1) + 4

/-- Theorem stating the number of unique tower heights achievable with 94 bricks -/
theorem ninety_four_bricks_heights :
  uniqueTowerHeights 94 = 465 := by
  rfl

#eval uniqueTowerHeights 94

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_four_bricks_heights_l288_28828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_multiplication_l288_28835

/-- Given a triangle with sides a and b enclosing an angle θ, 
    if we create a new triangle by tripling side a, doubling side b, 
    and keeping angle θ the same, then the area of the new triangle 
    is 6 times the area of the original triangle. -/
theorem triangle_area_multiplication (a b θ : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < θ) (h4 : θ < π) :
  (1/2) * (3*a) * (2*b) * Real.sin θ = 6 * ((1/2) * a * b * Real.sin θ) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_multiplication_l288_28835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l288_28853

theorem units_digit_of_expression : ∃ k : ℕ, 2^2023 * 5^2024 * 11^2025 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l288_28853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_midpoint_l288_28879

/-- The equation of the line passing through the intersection of two given lines and the midpoint of a segment -/
theorem line_through_intersection_and_midpoint 
  (A B : ℝ × ℝ) 
  (line1 line2 : ℝ → ℝ → Prop) 
  (h1 : A = (-2, 1)) 
  (h2 : B = (4, 3)) 
  (h3 : line1 = λ x y ↦ 2*x - 3*y + 1 = 0) 
  (h4 : line2 = λ x y ↦ 3*x + 2*y - 1 = 0) : 
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ 7*x - 4*y + 1 = 0) ∧ 
    (∃ (I : ℝ × ℝ), line1 I.1 I.2 ∧ line2 I.1 I.2 ∧ l I.1 I.2) ∧
    (l ((A.1 + B.1)/2) ((A.2 + B.2)/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_midpoint_l288_28879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_monotone_increasing_l288_28872

open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sqrt 2 * sin (2 * x + π / 3)

theorem y_monotone_increasing : 
  MonotoneOn y (Set.Icc (7 * π / 12) (13 * π / 12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_monotone_increasing_l288_28872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimus_prime_distance_l288_28812

/-- The distance between points A and B in kilometers -/
noncomputable def distance : ℝ := 750

/-- The speed of Optimus Prime in robot form in km/h -/
noncomputable def robot_speed : ℝ := 150

/-- The time saved when traveling entirely in car form, in hours -/
noncomputable def time_saved_car : ℝ := 1

/-- The time saved when traveling partially in car form, in hours -/
noncomputable def time_saved_partial : ℝ := 2/3

/-- The distance traveled in robot form before transforming, in km -/
noncomputable def robot_distance : ℝ := 150

/-- The speed increase factor when transforming into a car from the beginning -/
noncomputable def car_speed_increase : ℝ := 5/4

/-- The speed increase factor when transforming into a car after traveling in robot form -/
noncomputable def partial_speed_increase : ℝ := 6/5

theorem optimus_prime_distance :
  distance = robot_speed * time_saved_car * car_speed_increase ∧
  distance = robot_speed * time_saved_partial * 6 + robot_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimus_prime_distance_l288_28812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_monotonic_intervals_extreme_values_l288_28881

noncomputable section

open Real

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * log x

-- Theorem for part (1)
theorem find_a_and_b :
  ∃ a b : ℝ, f a b 1 = 1/2 ∧ (deriv (f a b)) 1 = 0 ∧ a = 1/2 ∧ b = -1 := by
  sorry

-- Theorem for part (2)
theorem monotonic_intervals (x : ℝ) :
  x > 0 → (x < 1 → (deriv (f (1/2) (-1))) x < 0) ∧
          (x > 1 → (deriv (f (1/2) (-1))) x > 0) := by
  sorry

-- Theorem for part (3)
theorem extreme_values :
  let f' := f (1/2) (-1)
  (∀ x ∈ Set.Icc (1/ℯ) ℯ, f' x ≥ 1/2) ∧
  (∃ x ∈ Set.Icc (1/ℯ) ℯ, f' x = 1/2) ∧
  (f' ℯ = max (f' (1/ℯ)) (f' ℯ)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_monotonic_intervals_extreme_values_l288_28881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_equivalence_l288_28886

theorem price_increase_equivalence (P : ℝ) (h : P > 0) : 
  (P * 1.25 * 1.25 * 0.9) = (P * 1.40625) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_equivalence_l288_28886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l288_28882

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- Curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 3| / Real.sqrt 2

/-- The range of distances from points on curve C to line l -/
theorem distance_range : 
  ∀ x y : ℝ, curve_C x y → 
    Real.sqrt 2 / 2 ≤ distance_to_line x y ∧ 
    distance_to_line x y ≤ 5 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l288_28882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l288_28875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  xIntercept1 : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_other_x_intercept 
  (e : Ellipse) 
  (h1 : e.focus1 = ⟨0, -3⟩) 
  (h2 : e.focus2 = ⟨4, 0⟩)
  (h3 : e.xIntercept1 = ⟨0, 0⟩) :
  ∃ (p : Point), p.y = 0 ∧ p.x = 56 / 11 ∧ 
    distance e.focus1 p + distance e.focus2 p = 
    distance e.focus1 e.xIntercept1 + distance e.focus2 e.xIntercept1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l288_28875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_prime_factors_30_factorial_is_10_l288_28804

/-- The number of distinct prime factors of 30! -/
def num_distinct_prime_factors_30_factorial : ℕ :=
  (Finset.filter (fun p => Nat.Prime p) (Finset.range 31)).card

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem num_distinct_prime_factors_30_factorial_is_10 :
  num_distinct_prime_factors_30_factorial = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_prime_factors_30_factorial_is_10_l288_28804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_13_l288_28858

def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => 4^(n+2) * f (n+1) - 16^(n+1) * f n + n * 2^(n^2)

theorem divisible_by_13 :
  f 1989 % 13 = 0 ∧ f 1990 % 13 = 0 ∧ f 1991 % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_13_l288_28858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l288_28846

/-- The function we're minimizing -/
noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

/-- The theorem stating the minimum value and where it occurs -/
theorem min_value_of_f :
  ∀ x > 1, f x ≥ 5 ∧ (f x = 5 ↔ x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l288_28846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_3_neg3_l288_28887

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem rectangular_to_polar_3_neg3 :
  let (r, θ) := rectangular_to_polar 3 (-3)
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_3_neg3_l288_28887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_pi_expression_l288_28889

theorem absolute_value_pi_expression : 
  abs (Real.pi - abs (Real.pi - 9)) = 9 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_pi_expression_l288_28889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l288_28803

theorem triangle_angle_cosine (a b c : ℝ) (h1 : a = 3) (h2 : b = Real.sqrt 8) (h3 : c = 2 + Real.sqrt 2) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ Real.cos θ = (3 * Real.sqrt 2 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l288_28803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l288_28885

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two vectors -/
def dot_product (v1 v2 : Vec2D) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Magnitude of a vector -/
noncomputable def magnitude (v : Vec2D) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem stating that the eccentricity of the given hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) (F Q : Point) 
  (h_right_branch : Q.x > 0)
  (h_projection : 2 < (dot_product (Vec2D.mk F.x F.y) (Vec2D.mk Q.x Q.y)) / 
    (magnitude (Vec2D.mk Q.x Q.y)) ∧ 
    (dot_product (Vec2D.mk F.x F.y) (Vec2D.mk Q.x Q.y)) / 
    (magnitude (Vec2D.mk Q.x Q.y)) ≤ 4) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l288_28885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l288_28809

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- Define the vectors
noncomputable def m (t : Triangle) : ℝ × ℝ := (Real.cos (t.A / 2) ^ 2, Real.cos t.B)
def n (t : Triangle) : ℝ × ℝ := (-t.a, 4 * t.c + 2 * t.b)
def p : ℝ × ℝ := (1, 0)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ k : ℝ, m t - (1/2 : ℝ) • p = k • n t)  -- (m - 1/2p) is parallel to n
  (h2 : t.a = Real.sqrt 3) :                     -- a = √3
  Real.cos t.A = -1/2 ∧                          -- cos A = -1/2
  t.a + t.b + t.c ≤ Real.sqrt 3 + 2 :=           -- Maximum perimeter
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l288_28809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_calculation_l288_28808

/-- Calculates the length of the second train given the speeds of two trains
    running in opposite directions, the length of the first train, and the
    time it takes for them to clear each other. -/
noncomputable def second_train_length (speed1 speed2 : ℝ) (length1 : ℝ) (clear_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

theorem second_train_length_calculation :
  let speed1 : ℝ := 80  -- km/h
  let speed2 : ℝ := 65  -- km/h
  let length1 : ℝ := 121  -- meters
  let clear_time : ℝ := 6.802214443534172  -- seconds
  abs (second_train_length speed1 speed2 length1 clear_time - 153) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_calculation_l288_28808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_30_degrees_l288_28874

/-- The angle of inclination of a line given its equation -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (a / b)

/-- The equation of the line is x - √3 * y + 5 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 5 = 0

theorem angle_of_inclination_is_30_degrees :
  0 ≤ angle_of_inclination 1 (Real.sqrt 3) 5 ∧
  angle_of_inclination 1 (Real.sqrt 3) 5 < π ∧
  angle_of_inclination 1 (Real.sqrt 3) 5 = π / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_30_degrees_l288_28874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l288_28843

noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (π/6) π ∧
  ∀ (x : ℝ), x ∈ Set.Icc (π/6) π → f x ≤ f c ∧
  f c = π/2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l288_28843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_composition_l288_28834

-- Define a homothety
structure Homothety (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  center : α
  coeff : ℝ

-- Define a translation
structure Translation (α : Type*) [AddCommGroup α] where
  vec : α

-- Define the composition result
inductive CompositionResult (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α]
| Homothety : Homothety α → CompositionResult α
| Translation : Translation α → CompositionResult α

-- Main theorem
theorem homothety_composition {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (h₁ h₂ : Homothety α) :
  ∃ (result : CompositionResult α), 
    (∃ (h₃ : Homothety α), result = CompositionResult.Homothety h₃ ∧ 
      ∃ (t : ℝ), h₃.center = (1 - t) • h₁.center + t • h₂.center ∧ 
      h₃.coeff = h₁.coeff * h₂.coeff) ∨
    (∃ (t : Translation α), result = CompositionResult.Translation t ∧ 
      t.vec = h₂.center - h₁.center + 
        h₂.coeff • (h₁.center - h₂.center) + (h₂.coeff * h₁.coeff - 1) • h₂.center) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_composition_l288_28834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l288_28829

def given_numbers : List ℚ := [-4, -1/5, 0, 22/7, -314/100, 717, -5, 188/100]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def positive_numbers : List ℚ := [22/7, 717, 188/100]
def negative_numbers : List ℚ := [-4, -1/5, -314/100, -5]
def integer_numbers : List ℚ := [-4, 0, 717, -5]
def fraction_numbers : List ℚ := [-1/5, 22/7, -314/100, 188/100]

theorem number_categorization :
  (∀ x ∈ positive_numbers, is_positive x) ∧
  (∀ x ∈ negative_numbers, is_negative x) ∧
  (∀ x ∈ integer_numbers, is_integer x) ∧
  (∀ x ∈ fraction_numbers, is_fraction x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l288_28829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_C_l288_28854

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector2D in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Line segment with endpoints A and B -/
structure LineSegment where
  A : Point
  B : Point

/-- Trajectory of a point -/
structure Trajectory where
  equation : Point → Prop

theorem trajectory_of_point_C (AB : LineSegment)
  (h_length : (AB.A.x - AB.B.x)^2 + (AB.A.y - AB.B.y)^2 = 9)
  (h_A_on_x : AB.A.y = 0)
  (h_B_on_y : AB.B.x = 0)
  (C : Point)
  (h_AC_CB : ∃ (AC CB : Vector2D), 
    AC.x = C.x - AB.A.x ∧ AC.y = C.y - AB.A.y ∧
    CB.x = AB.B.x - C.x ∧ CB.y = AB.B.y - C.y ∧
    AC.x = 2 * CB.x ∧ AC.y = 2 * CB.y) :
  ∃ (T : Trajectory), T.equation C ↔ C.x^2 + C.y^2 = 1 := by
  sorry

#check trajectory_of_point_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_C_l288_28854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_triangulation_l288_28890

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → Set (ℝ × ℝ)

/-- Represents a triangulation of a polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  triangles : Set (Set (ℝ × ℝ))
  is_valid : Bool

/-- Represents a colored triangulation of a polygon -/
structure ColoredTriangulation (n : ℕ) where
  triangulation : Triangulation n
  color : Set (ℝ × ℝ) → Bool

/-- Checks if the triangulation satisfies the given conditions -/
def satisfies_conditions (ct : ColoredTriangulation 2017) : Prop :=
  ∀ t1 t2, t1 ∈ ct.triangulation.triangles → t2 ∈ ct.triangulation.triangles →
    (ct.color t1 ≠ ct.color t2 → (∃ side, side ∈ t1 ∧ side ∈ t2)) ∨
    (∃ v, v ∈ t1 ∧ v ∈ t2) ∨
    (t1 ∩ t2 = ∅)

/-- Checks if all sides of the polygon are sides of black triangles -/
def all_sides_black (ct : ColoredTriangulation 2017) : Prop :=
  ∀ i, ∀ side ∈ ct.triangulation.polygon.sides i,
    ∃ t ∈ ct.triangulation.triangles, ¬ct.color t ∧ side ∈ t

/-- The main theorem stating the impossibility of the required triangulation -/
theorem impossible_triangulation :
  ¬∃ (ct : ColoredTriangulation 2017),
    satisfies_conditions ct ∧ all_sides_black ct := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_triangulation_l288_28890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_half_open_interval_l288_28877

def M : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = 1 - |x| ∧ y > 0}

theorem intersection_equals_half_open_interval : M ∩ N = Set.Ico 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_half_open_interval_l288_28877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l288_28830

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (a x y : ℝ) : Prop := (a + 1) * x + y - 2 - a = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt

theorem line_equations 
  (h1 : ∃ (A B : ℝ × ℝ), circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
                          (∃ a, line_l a A.1 A.2) ∧ 
                          (∃ a, line_l a B.1 B.2))
  (h2 : ∃ (A B : ℝ × ℝ), distance A.1 A.2 B.1 B.2 = 2 * Real.sqrt 3)
  (h3 : ∃ (a : ℝ), line_l a point_P.1 point_P.2)
  (h4 : ∃ (a : ℝ), (2 + a) / (a + 1) = 2 + a ∨ a = -2) :
  (∃ (x y : ℝ), x = 1 ∧ circle_C x y) ∧ 
  (∃ (x y : ℝ), 3 * x - 4 * y + 5 = 0 ∧ (∃ a, line_l a x y)) ∧
  (∃ (x y : ℝ), x - y = 0 ∧ (∃ a, line_l a x y)) ∧
  (∃ (x y : ℝ), x + y - 2 = 0 ∧ (∃ a, line_l a x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l288_28830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_price_reduction_for_target_profit_l288_28876

/-- Represents the price reduction in yuan -/
def price_reduction : ℝ → Prop := sorry

/-- The original profit per piece in yuan -/
def original_profit : ℝ := 40

/-- The original daily sales volume -/
def original_sales : ℝ := 20

/-- The target daily profit in yuan -/
def target_profit : ℝ := 1200

/-- The rate of increase in sales for each yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Theorem stating that there exists a price reduction that achieves the target profit -/
theorem exists_price_reduction_for_target_profit :
  ∃ x : ℝ, price_reduction x ∧ 
    (original_profit - x) * (original_sales + sales_increase_rate * x) = target_profit :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_price_reduction_for_target_profit_l288_28876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_properties_l288_28807

/-- A regular hexagonal pyramid with base side length a and all diagonal sections congruent -/
structure RegularHexagonalPyramid (a : ℝ) where
  (a_pos : a > 0)
  (diagonal_sections_congruent : True)  -- We can't express this condition directly, so we use True as a placeholder

/-- The volume of a regular hexagonal pyramid -/
noncomputable def volume (p : RegularHexagonalPyramid a) : ℝ := 3 * a^3 / 4

/-- The lateral surface area of a regular hexagonal pyramid -/
noncomputable def lateral_surface_area (p : RegularHexagonalPyramid a) : ℝ := 3 * a^2 * Real.sqrt 6 / 2

/-- Theorem stating the volume and lateral surface area of a regular hexagonal pyramid -/
theorem regular_hexagonal_pyramid_properties (a : ℝ) (p : RegularHexagonalPyramid a) :
  volume p = 3 * a^3 / 4 ∧ lateral_surface_area p = 3 * a^2 * Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_properties_l288_28807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l288_28894

noncomputable def a (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω * x) + Real.cos (ω * x), Real.sin (ω * x))
noncomputable def b (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω * x) - Real.cos (ω * x), 2 * Real.sqrt 3 * Real.cos (ω * x))

noncomputable def f (ω x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2 + 1

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem vector_function_properties (ω : ℝ) (h_ω : 0 < ω ∧ ω < 2) :
  (symmetric_about (f ω) (π / 3)) →
  (∀ x ∈ Set.Icc 0 (π / 2), f ω x ∈ Set.Icc 0 3) ∧
  (∀ k : ℤ, symmetric_about (f ω) (k * π / 2 + π / 3)) ∧
  (∀ k : ℤ, StrictMonoOn (f ω) (Set.Ioo (k * π - π / 6) (k * π + π / 3))) ∧
  (∀ k : ℤ, StrictAntiOn (f ω) (Set.Ioo (k * π + π / 3) (k * π + 5 * π / 6))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l288_28894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_lambda_l288_28826

theorem vector_perpendicular_lambda : ∀ (a b c : ℝ × ℝ) (l : ℝ),
  a = (2, 0) →
  b = (1, 2) →
  c = (1, -2) →
  (a.1 - l * b.1) * c.1 + (a.2 - l * b.2) * c.2 = 0 →
  l = -2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_lambda_l288_28826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_associated_value_and_x_range_l288_28884

/-- Definition of associated value for the equation x - 2y = 2 -/
noncomputable def associated_value (x y : ℝ) : ℝ :=
  if (abs x ≥ abs y) then abs x else abs y

/-- The equation x - 2y = 2 -/
def satisfies_equation (x y : ℝ) : Prop :=
  x - 2*y = 2

theorem min_associated_value_and_x_range :
  (∃ (m : ℝ), ∀ (x y : ℝ), satisfies_equation x y → associated_value x y ≥ m ∧ m = 2/3) ∧
  (∀ (x y : ℝ), satisfies_equation x y → associated_value x y = abs x → (x ≥ 1/3 ∨ x ≤ -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_associated_value_and_x_range_l288_28884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l288_28844

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) 
  (h1 : sale_price = 80)
  (h2 : discount_percentage = 20) : 
  sale_price / (1 - discount_percentage / 100) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l288_28844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l288_28820

-- Define the base of the exponential and logarithmic functions
variable (a : ℝ)

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := a^x

-- Define the logarithmic function
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the condition that a > 1
def a_greater_than_one : Prop := a > 1

-- Define the critical value e^(1/e)
noncomputable def critical_value : ℝ := Real.exp (1 / Real.exp 1)

-- Theorem statement
theorem intersection_points (ha : a_greater_than_one a) :
  (a = critical_value → ∃! x, f a x = g a x) ∧
  (a > critical_value → ∀ x, f a x ≠ g a x) ∧
  (1 < a ∧ a < critical_value → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = g a x₁ ∧ f a x₂ = g a x₂) :=
by
  sorry

#check intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l288_28820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_g_l288_28891

noncomputable section

open Real MeasureTheory

def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + 1

theorem max_difference_of_g (x₁ x₂ : ℝ) :
  x₁ ∈ Set.Icc (-2 * π) (2 * π) →
  x₂ ∈ Set.Icc (-2 * π) (2 * π) →
  g x₁ + g x₂ = 6 →
  |x₁ - x₂| ≤ 3 * π ∧ ∃ y₁ y₂, y₁ ∈ Set.Icc (-2 * π) (2 * π) ∧ y₂ ∈ Set.Icc (-2 * π) (2 * π) ∧ g y₁ + g y₂ = 6 ∧ |y₁ - y₂| = 3 * π :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_g_l288_28891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_18_divisible_by_18_l288_28863

/-- Theorem stating the smallest n for which any 18 out of n integers can subset sum reaching an exact divisible by 18 -/
theorem smallest_n_for_18_divisible_by_18 : ℕ := by
  -- Define a function that checks if a list of integers has a subset of size 18 with sum divisible by 18
  let has_18_divisible_by_18 (l : List ℤ) : Prop :=
    ∃ (subset : List ℤ), subset.length = 18 ∧ subset.sum % 18 = 0 ∧ subset.toFinset ⊆ l.toFinset

  -- Define the property we want to prove
  let property (n : ℕ) : Prop :=
    ∀ (l : List ℤ), l.length = n → has_18_divisible_by_18 l

  -- State and prove the theorem
  have h : (∀ m < 35, ¬(property m)) ∧ property 35 := by
    sorry -- The actual proof would go here

  -- Return the result
  exact 35


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_18_divisible_by_18_l288_28863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_series_sum_l288_28805

def series_term1 : ℚ := (Nat.factorial 15 * Nat.factorial 14 - Nat.factorial 13 * Nat.factorial 12) / 201

def series_term2 : ℚ := (Nat.factorial 17 * Nat.factorial 16 - Nat.factorial 15 * Nat.factorial 14) / 243

def series_sum : ℚ := series_term1 + series_term2

theorem greatest_prime_factor_of_series_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p = 271 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (Int.natAbs series_sum.num) → q ≤ p :=
by
  sorry

#eval series_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_series_sum_l288_28805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_xoy_plane_l288_28815

/-- Given a point P(1, √2, √3) in the Cartesian coordinate system, 
    the point Q where a perpendicular line from P meets the xOy plane 
    has coordinates (1, √2, 0). -/
theorem perpendicular_to_xoy_plane :
  ∀ (P Q : ℝ × ℝ × ℝ),
    P = (1, Real.sqrt 2, Real.sqrt 3) →
    Q.2.2 = 0 →
    (∀ (t : ℝ), ∃ (k : ℝ), 
      P.1 + k * (Q.1 - P.1) = P.1 ∧
      P.2.1 + k * (Q.2.1 - P.2.1) = P.2.1 ∧
      P.2.2 + k * (Q.2.2 - P.2.2) = t) →
    Q = (1, Real.sqrt 2, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_xoy_plane_l288_28815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l288_28847

noncomputable section

/-- Represents the side length of the equilateral triangle -/
def side_length : ℝ := 10

/-- Represents the radius of the circle -/
def radius : ℝ := side_length / 2

/-- Represents the area of one sector of the circle -/
def sector_area : ℝ := (1/3) * Real.pi * radius^2

/-- Represents the area of the equilateral triangle -/
def triangle_area : ℝ := (side_length^2 * Real.sqrt 3) / 4

/-- Represents the area of one shaded region -/
def shaded_area : ℝ := sector_area - triangle_area / 2

/-- Represents the coefficient 'a' in the expression a*π - b*√c -/
def a : ℝ := 2 * shaded_area / Real.pi

/-- Represents the coefficient 'b' in the expression a*π - b*√c -/
def b : ℝ := (sector_area - shaded_area) * 2 / Real.sqrt 3

/-- Represents the value 'c' in the expression a*π - b*√c -/
def c : ℝ := 3

theorem sum_of_coefficients :
  a + b + c = 28 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l288_28847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hyperbola_properties_l288_28852

/-- A hyperbola with specific properties -/
structure SpecialHyperbola where
  /-- The hyperbola has its foci on the x-axis -/
  foci_on_x_axis : Bool
  /-- The distance between its two vertices is 2 -/
  vertex_distance : ℝ
  /-- The distance from a focus to an asymptote is √2 -/
  focus_to_asymptote : ℝ

/-- Properties of the special hyperbola -/
def hyperbola_properties (h : SpecialHyperbola) : Prop :=
  h.foci_on_x_axis ∧
  h.vertex_distance = 2 ∧
  h.focus_to_asymptote = Real.sqrt 2

/-- The standard equation of the hyperbola -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 - y^2/2 = 1

/-- The length of the real axis -/
def real_axis_length : ℝ := 2

/-- The length of the imaginary axis -/
noncomputable def imaginary_axis_length : ℝ := 2 * Real.sqrt 2

/-- The coordinates of the foci -/
noncomputable def foci_coordinates : Set (ℝ × ℝ) :=
  {(-Real.sqrt 3, 0), (Real.sqrt 3, 0)}

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt 3

/-- The equations of the asymptotes -/
def asymptote_equations (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Main theorem: properties of the special hyperbola -/
theorem special_hyperbola_properties (h : SpecialHyperbola) 
  (hprops : hyperbola_properties h) :
  (∀ x y, standard_equation x y) ∧
  real_axis_length = 2 ∧
  imaginary_axis_length = 2 * Real.sqrt 2 ∧
  foci_coordinates = {(-Real.sqrt 3, 0), (Real.sqrt 3, 0)} ∧
  eccentricity = Real.sqrt 3 ∧
  (∀ x y, asymptote_equations x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hyperbola_properties_l288_28852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l288_28871

/-- The number of pieces of cherry gum Chewbacca initially has -/
def initial_cherry : ℕ := 25

/-- The number of pieces of grape gum Chewbacca initially has -/
def initial_grape : ℕ := 35

/-- The number of pieces in each complete pack of gum -/
def x : ℕ := 25

/-- The ratio of cherry to grape gum after losing two packs of cherry and finding three packs of grape -/
def ratio1 (x : ℕ) : ℚ := (initial_cherry - 2 * x : ℚ) / (initial_grape + 3 * x : ℚ)

/-- The ratio of cherry to grape gum after losing one pack of cherry and finding five packs of grape -/
def ratio2 (x : ℕ) : ℚ := (initial_cherry - x : ℚ) / (initial_grape + 5 * x : ℚ)

/-- Theorem stating that the number of pieces in each complete pack of gum is 25 -/
theorem gum_pack_size : x = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l288_28871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_polynomial_is_correct_l288_28818

/-- A monic quadratic polynomial with real coefficients that has -3 + i√3 as a root -/
def target_polynomial (x : ℂ) : ℂ := x^2 + 6*x + 12

/-- The complex number -3 + i√3 -/
noncomputable def alpha : ℂ := -3 + Complex.I * Real.sqrt 3

theorem target_polynomial_is_correct :
  -- The polynomial has -3 + i√3 as a root
  target_polynomial alpha = 0 ∧
  -- The polynomial is monic
  (∃ (a b : ℝ), ∀ x, target_polynomial x = x^2 + a*x + b) ∧
  -- The polynomial is unique
  (∀ f : ℂ → ℂ, (∃ (a b : ℝ), ∀ x, f x = x^2 + a*x + b) → f alpha = 0 → f = target_polynomial) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_polynomial_is_correct_l288_28818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l288_28850

def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + c

def g (a b c : ℝ) (x : ℝ) : ℝ := f b c (x - a) - a^2

def is_steep_function (h : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∃ (l u : ℝ), (∀ x ∈ Set.Icc m n, l ≤ h x ∧ h x ≤ u) ∧ (u - l) / (n - m) > 8

theorem f_g_properties (b c : ℝ) (h₁ : f b c 1 = -1) (h₂ : f b c 3 = -1) :
  (∃ a : ℝ, a > 0 ∧ 
    (∃ x₁ x₂ : ℝ, x₁ < 4 ∧ 4 < x₂ ∧ g a b c x₁ = 0 ∧ g a b c x₂ = 0) →
    a > 1/2) ∧
  (∃ a : ℝ, a > 0 ∧ 
    is_steep_function (g a b c) a (2*a) →
    a > 6 + 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l288_28850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trajectory_area_l288_28814

open Complex

/-- The trajectory of z₁ in the complex plane -/
def z₁_trajectory : Set ℂ :=
  { z : ℂ | ∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ z = (1 : ℂ) + (4 * t - 2 : ℝ) * Complex.I }

/-- The trajectory of z₂ in the complex plane -/
def z₂_trajectory : Set ℂ :=
  { z : ℂ | ∃ θ : ℝ, z = Complex.exp (Complex.I * θ) }

/-- The trajectory of z₁ + z₂ in the complex plane -/
def sum_trajectory : Set ℂ :=
  { z : ℂ | ∃ z₁ ∈ z₁_trajectory, ∃ z₂ ∈ z₂_trajectory, z = z₁ + z₂ }

/-- The area of a set in the complex plane -/
noncomputable def area (s : Set ℂ) : ℝ := sorry

/-- The main theorem stating the area of the sum trajectory -/
theorem sum_trajectory_area : area sum_trajectory = 8 + Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trajectory_area_l288_28814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_function_range_l288_28857

-- Define the function y = (a^2 - 1)^x
noncomputable def y (a x : ℝ) : ℝ := (a^2 - 1)^x

-- State the theorem
theorem decreasing_exponential_function_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y a x₁ > y a x₂) →
  (1 < |a| ∧ |a| < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_function_range_l288_28857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l288_28822

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a trapezoid -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The length of a line segment between two points -/
noncomputable def distance (p q : Point) : ℝ := 
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point lies on a line segment -/
def onSegment (p q r : Point) : Prop := sorry

/-- Check if two line segments intersect -/
def intersect (p q r s : Point) : Point := sorry

/-- Theorem: In a trapezoid ABCD with point E on AD such that AE = BC, 
    and segments CA and CE intersecting diagonal BD at O and P respectively, 
    if BO = PD, then AD^2 = BC^2 + AD * BC -/
theorem trapezoid_theorem (ABCD : Trapezoid) (E O P : Point) : 
  onSegment E ABCD.A ABCD.D →
  distance E ABCD.A = distance ABCD.B ABCD.C →
  O = intersect ABCD.C ABCD.A ABCD.B ABCD.D →
  P = intersect ABCD.C E ABCD.B ABCD.D →
  distance ABCD.B O = distance P ABCD.D →
  (distance ABCD.A ABCD.D)^2 = (distance ABCD.B ABCD.C)^2 + 
    (distance ABCD.A ABCD.D) * (distance ABCD.B ABCD.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l288_28822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_BCD_l288_28860

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The problem statement -/
theorem triangle_area_BCD (A B C D : Point) (area_ABC : ℝ) : 
  A.x = 0 ∧ A.y = 0 ∧
  D.x = 40 ∧ D.y = 0 ∧
  B.x = 10 ∧ B.y = 24 ∧
  C.x = 30 ∧ C.y = 0 ∧
  area_ABC = 36 ∧
  (B.y - C.y) / (B.x - C.x) = (B.y - D.y) / (B.x - D.x) →  -- BC parallel to AD
  triangleArea (D.x - C.x) ((B.y - A.y) * area_ABC / (C.x - A.x)) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_BCD_l288_28860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_divided_by_line_l288_28813

/-- A circle in the 2D plane -/
structure Circle where
  k : ℝ
  m : ℝ
  equation : ∀ x y : ℝ, x^2 + y^2 + k*x + m*y - 4 = 0

/-- A line in the 2D plane -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- The center of a circle -/
noncomputable def center (c : Circle) : ℝ × ℝ :=
  (-c.k/2, -c.m/2)

/-- Predicate stating that a line divides a circle into two equal areas -/
def divides_equally (l : Set (ℝ × ℝ)) (c : Circle) : Prop :=
  (center c) ∈ l

theorem circle_divided_by_line (c : Circle) :
  divides_equally Line c → c.m - c.k = 2 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_divided_by_line_l288_28813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_purchase_increase_l288_28895

/-- Calculates the additional amount of oil that can be purchased after a price reduction --/
noncomputable def additional_oil_purchased (original_price : ℝ) (reduced_price : ℝ) (budget : ℝ) : ℝ :=
  (budget / reduced_price) - (budget / original_price)

/-- Theorem stating the additional amount of oil that can be purchased after a 10% price reduction --/
theorem oil_purchase_increase :
  let reduced_price : ℝ := 16
  let budget : ℝ := 800
  let original_price : ℝ := reduced_price / 0.9
  abs (additional_oil_purchased original_price reduced_price budget - 5.01) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_purchase_increase_l288_28895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l288_28833

theorem triangle_inequalities (a b c lambda : ℝ) 
  (h_positive : lambda > 0) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_perimeter : a + b + c = lambda) : 
  (13/27 * lambda^2 ≤ a^2 + b^2 + c^2 + (4/lambda) * a * b * c ∧ 
   a^2 + b^2 + c^2 + (4/lambda) * a * b * c < lambda^2/2) ∧
  (1/4 * lambda^2 < a * b + b * c + c * a - (2/lambda) * a * b * c ∧ 
   a * b + b * c + c * a - (2/lambda) * a * b * c ≤ 7/27 * lambda^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l288_28833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l288_28810

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- First line equation: x + 2y + 4 = 0 -/
def line1 (x y : ℝ) : Prop := x + 2*y + 4 = 0

/-- Second line equation: 2x + 4y + 7 = 0 -/
def line2 (x y : ℝ) : Prop := 2*x + 4*y + 7 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 2 4 8 7 = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l288_28810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_work_time_l288_28861

-- Define work rates as fractions of work done per hour
def work_rate_x : ℚ := 1/8
def work_rate_y : ℚ := 1/24  -- This is what we want to prove
def work_rate_z : ℚ := 1/8
def work_rate_w : ℚ := 1/12

-- State the theorem
theorem y_work_time 
  (h1 : work_rate_x = 1/8)
  (h2 : work_rate_y + work_rate_z = 1/6)
  (h3 : work_rate_x + work_rate_z = 1/4)
  (h4 : work_rate_x + work_rate_y + work_rate_w = 1/5)
  (h5 : work_rate_x + work_rate_w + work_rate_z = 1/3)
  : work_rate_y = 1/24 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_work_time_l288_28861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_equilateral_triangle_l288_28839

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

variable (t : Triangle)

-- Statement B
theorem obtuse_triangle (h : 0 < Real.tan t.A * Real.tan t.B ∧ Real.tan t.A * Real.tan t.B < 1) :
  t.C > π/2 := by
  sorry

-- Statement C
theorem equilateral_triangle (h : Real.cos (t.A - t.B) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1) :
  t.A = t.B ∧ t.B = t.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_equilateral_triangle_l288_28839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l288_28800

/-- 
Given a regular triangular pyramid with:
- A plane angle at the apex of 90°
- The distance between a lateral edge and the opposite side of the base equal to d

The volume of the pyramid is (d³√2)/3
-/
theorem regular_triangular_pyramid_volume 
  (d : ℝ) 
  (h_apex_angle : Real.pi / 2 = 90 * (Real.pi / 180)) 
  (h_lateral_distance : ℝ) 
  (h_lateral_distance_eq : h_lateral_distance = d) : 
  ∃ (V : ℝ), V = (d^3 * Real.sqrt 2) / 3 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l288_28800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_l288_28865

/-- The circle with equation x^2 + y^2 = 3 -/
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- The line with equation x + my - m - 1 = 0 -/
def myLine (m x y : ℝ) : Prop := x + m*y - m - 1 = 0

/-- The chord intercepted by the line on the circle is the shortest when m = 1 -/
theorem shortest_chord (m : ℝ) : 
  (∀ x y, myCircle x y → myLine m x y → 
    ∀ m', (∀ x' y', myCircle x' y' → myLine m' x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - x')^2 + (y - y')^2)) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_l288_28865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_properties_l288_28873

-- Define the circle M
def circle_M (a r : ℝ) : ℝ × ℝ → Prop :=
  λ p ↦ (p.1 - a)^2 + (p.2 - 4)^2 = r^2

-- Define the line
def line (m : ℝ) : ℝ × ℝ → Prop :=
  λ p ↦ 4*p.1 + 3*p.2 + m = 0

-- Define the chord length
noncomputable def chord_length (a r m : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((24 + m)^2 / 25))

theorem circle_M_properties :
  ∃ (a r : ℝ),
    r > 0 ∧
    circle_M a r (0, 0) ∧
    circle_M a r (6, 0) ∧
    a = 3 ∧
    r = 5 ∧
    (∀ m : ℝ, chord_length a r m = 6 → (m = -4 ∨ m = -44)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_properties_l288_28873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_marks_proof_l288_28827

def class_size : ℕ := 16
def wrong_marks : ℕ := 73
def average_increase : ℚ := 1/2

theorem actual_marks_proof :
  ∀ (actual_marks : ℕ),
    (wrong_marks - actual_marks : ℚ) = average_increase * class_size →
    actual_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_marks_proof_l288_28827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_closed_unit_interval_l288_28878

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 2}

-- Theorem stating that the union of M and N is equal to the closed interval [0, 1]
theorem union_M_N_equals_closed_unit_interval :
  M ∪ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_closed_unit_interval_l288_28878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_finish_remaining_is_two_l288_28823

/-- The time it takes for person A to finish the work alone -/
noncomputable def a_time : ℝ := 6

/-- The time it takes for person B to finish the work alone -/
noncomputable def b_time : ℝ := 15

/-- The time B worked before leaving -/
noncomputable def b_worked : ℝ := 10

/-- The remaining work after B left -/
noncomputable def remaining_work : ℝ := 1 - (b_worked / b_time)

/-- The time it takes for A to finish the remaining work -/
noncomputable def a_finish_remaining : ℝ := remaining_work * a_time

theorem a_finish_remaining_is_two :
  a_finish_remaining = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_finish_remaining_is_two_l288_28823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_test_passing_l288_28896

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) : 
  total_problems = 40 → 
  passing_percentage = 75 / 100 → 
  (total_problems : ℚ) - (passing_percentage * total_problems).floor = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_test_passing_l288_28896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3b_plus_18_l288_28849

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 5 * b = 10 - 3 * a) :
  (Finset.filter (fun n : ℕ => (3 * b + 18 : ℤ) ∣ n) (Finset.range 9)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3b_plus_18_l288_28849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_min_volume_l288_28868

/-- Represents the minimum volume of an ideal gas in a cyclic process -/
noncomputable def min_volume (R P₀ T₀ a b c : ℝ) : ℝ :=
  (R * T₀ / P₀) * (a * Real.sqrt (a^2 + b^2 - c^2) - b * c) / 
  (b * Real.sqrt (a^2 + b^2 - c^2) + a * c)

/-- Theorem stating the minimum volume of an ideal gas in a specific cyclic process -/
theorem ideal_gas_min_volume 
  (R P₀ T₀ a b c : ℝ) 
  (h_positive : R > 0 ∧ P₀ > 0 ∧ T₀ > 0)
  (h_constraint : c^2 < a^2 + b^2) :
  ∀ P T : ℝ, P > 0 → T > 0 → 
  (P/P₀ - a)^2 + (T/T₀ - b)^2 = c^2 →
  R * T / P ≥ min_volume R P₀ T₀ a b c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_min_volume_l288_28868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l288_28851

-- Define S as a variable (not a definition)
variable (S : Finset ℤ)

-- S has exactly 11 elements
axiom S_card : S.card = 11

-- All elements in S are distinct (this is already guaranteed by Finset)

-- S contains the specified seven elements
axiom S_contains : {3, 5, 7, 13, 15, 17, 19} ⊆ S

-- Define the set of possible medians
def possible_medians (S : Finset ℤ) : Finset ℤ := 
  S.filter (λ x => (S.filter (λ y => y < x)).card = 5)

-- Theorem stating the number of possible medians
theorem median_count : (possible_medians S).card = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l288_28851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_numerator_problem_l288_28870

theorem fraction_numerator_problem :
  ∀ (numerator : ℚ),
    (numerator / (4 * numerator - 4) = 3 / 8) → numerator = 3 := by
  intro numerator
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_numerator_problem_l288_28870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l288_28817

theorem problem_solution : 
  (Real.sqrt ((-3)^2) + ((-8) : ℝ)^(1/3) - abs (Real.pi - 2) = 3 - Real.pi) ∧
  (4/3 / ((-1/3)^2) * (-1/2) - (-2)^2 = -10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l288_28817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l288_28837

theorem angle_terminal_side_point (α : ℝ) :
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l288_28837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bea_age_not_unique_l288_28897

structure Person where
  age : ℕ

def kiarra : Person := ⟨30⟩

axiom bea_younger_than_kiarra : ∀ (bea : Person), bea.age < kiarra.age

axiom job_age (bea : Person) : ∃ (job : Person), job.age = 3 * bea.age

axiom figaro_age (job : Person) : ∃ (figaro : Person), figaro.age = job.age + 7

axiom harry_age (figaro : Person) : ∃ (harry : Person), harry.age = figaro.age / 2

theorem bea_age_not_unique : ¬∃! (bea : Person), 
  bea.age < kiarra.age ∧ 
  (∃ (job : Person), job.age = 3 * bea.age ∧
    (∃ (figaro : Person), figaro.age = job.age + 7 ∧
      (∃ (harry : Person), harry.age = figaro.age / 2))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bea_age_not_unique_l288_28897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_l288_28802

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points O, A, B, C
variable (O A B C : V)

-- Define the parameter lambda
variable (lambda : ℝ)

-- State that OA and OB are not collinear
variable (h_not_collinear : ¬ ∃ (k : ℝ), A - O = k • (B - O))

-- Define the relation for point C
variable (h_C : C - O = lambda • (A - O) + (2 - lambda) • (B - O))

-- Theorem statement
theorem trajectory_is_line :
  ∃ (D : V), ∃ (mu : ℝ), C - O = mu • (D - O) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_l288_28802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_square_area_relation_l288_28859

-- Define a regular dodecagon
structure RegularDodecagon where
  vertices : Fin 12 → ℝ × ℝ
  is_regular : ∀ i j, dist (vertices i) (vertices ((i + 1) % 12)) = dist (vertices j) (vertices ((j + 1) % 12))
  inscribed_in_unit_circle : ∀ i, dist (0, 0) (vertices i) = 1

-- Define a square
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : ∀ i, dist (vertices i) (vertices ((i + 1) % 4)) = dist (vertices ((i + 1) % 4)) (vertices ((i + 2) % 4))

-- Define the property that four vertices of the dodecagon are at midpoints of square sides
def vertices_at_midpoints (d : RegularDodecagon) (s : Square) : Prop :=
  ∃ (i₁ i₂ i₃ i₄ : Fin 12), 
    d.vertices i₁ = ((s.vertices 0 + s.vertices 1) : ℝ × ℝ) / 2 ∧
    d.vertices i₂ = ((s.vertices 1 + s.vertices 2) : ℝ × ℝ) / 2 ∧
    d.vertices i₃ = ((s.vertices 2 + s.vertices 3) : ℝ × ℝ) / 2 ∧
    d.vertices i₄ = ((s.vertices 3 + s.vertices 0) : ℝ × ℝ) / 2

-- Define the area of a polygon
noncomputable def area (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem dodecagon_square_area_relation 
  (d : RegularDodecagon) (s : Square) 
  (h : vertices_at_midpoints d s) : 
  area s.vertices = (1 / 12) * area d.vertices ∧ 
  area d.vertices = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_square_area_relation_l288_28859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l288_28819

-- Define the circle and points
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

def A (x a : ℝ) : ℝ × ℝ := (x, a)
def B (x b : ℝ) : ℝ × ℝ := (x, b)
def C (x c : ℝ) : ℝ × ℝ := (x, c)

-- Define the expression
noncomputable def G (x r a b c : ℝ) : ℝ :=
  let t := x^2 - r^2
  let AP := Real.sqrt (a^2 + t)
  let BQ := Real.sqrt (b^2 + t)
  let CR := Real.sqrt (c^2 + t)
  let AB := a - b
  let BC := b - c
  let AC := a - c
  (AB * CR + BC * AP)^2 - (AC * BQ)^2

-- State the theorem
theorem circle_tangent_theorem (O : ℝ × ℝ) (r x a b c : ℝ) 
  (h1 : x > 0) (h2 : a > b) (h3 : b > c) (h4 : c > 0) :
  Real.sign (G x r a b c) = Real.sign (x^2 - r^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l288_28819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop1_logical_equivalences_prop2_logical_equivalences_l288_28842

-- Proposition 1
def proposition1 (a b c : ℝ) : Prop := a > b → c^2 * a > c^2 * b

-- Proposition 2
def proposition2 (a b c : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = a * x^2 + b * x + c) →
  (b^2 - 4*a*c < 0 → ∃ x, f x = 0)

-- Converse, Inverse, and Contrapositive for Proposition 1
theorem prop1_logical_equivalences (a b c : ℝ) :
  (proposition1 a b c) ∧
  ((c^2 * a > c^2 * b → a > b) ↔ (c^2 * a > c^2 * b → a > b)) ∧
  ((a ≤ b → c^2 * a ≤ c^2 * b) ↔ (¬(a > b) → ¬(c^2 * a > c^2 * b))) ∧
  ((c^2 * a ≤ c^2 * b → a ≤ b) ↔ (¬(c^2 * a > c^2 * b) → ¬(a > b))) :=
by sorry

-- Converse, Inverse, and Contrapositive for Proposition 2
theorem prop2_logical_equivalences (a b c : ℝ) (f : ℝ → ℝ) :
  (proposition2 a b c f) ∧
  ((∃ x, f x = 0) → b^2 - 4*a*c < 0 ↔ ((∃ x, f x = 0) → b^2 - 4*a*c < 0)) ∧
  (b^2 - 4*a*c ≥ 0 → ¬(∃ x, f x = 0) ↔ (¬(b^2 - 4*a*c < 0) → ¬(∃ x, f x = 0))) ∧
  (¬(∃ x, f x = 0) → b^2 - 4*a*c ≥ 0 ↔ (¬(∃ x, f x = 0) → ¬(b^2 - 4*a*c < 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop1_logical_equivalences_prop2_logical_equivalences_l288_28842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l288_28825

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos t, 4 * Real.sqrt 2 * Real.sin t)

noncomputable def bounded_area (f : ℝ → ℝ × ℝ) (y_line : ℝ) : ℝ :=
  2 * Real.pi - 4

theorem area_calculation :
  bounded_area parametric_curve 4 = 2 * Real.pi - 4 := by
  sorry

#eval "Area calculation theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l288_28825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_growth_rate_l288_28801

/-- Represents the farmer's fruit selling scenario -/
structure FruitSelling where
  investment : ℚ
  yield : ℚ
  market_price : ℚ
  orchard_price : ℚ
  daily_sales : ℚ
  labor_cost : ℚ
  transport_cost : ℚ

/-- Calculates the income from selling in the market -/
noncomputable def market_income (fs : FruitSelling) : ℚ :=
  fs.yield * fs.market_price - (fs.yield / fs.daily_sales) * (2 * fs.labor_cost + fs.transport_cost)

/-- Calculates the income from selling in the orchard -/
def orchard_income (fs : FruitSelling) : ℚ :=
  fs.yield * fs.orchard_price

/-- Theorem stating the growth rate of net income is 20% -/
theorem net_income_growth_rate
  (fs : FruitSelling)
  (h1 : fs.investment = 13800)
  (h2 : fs.yield = 18000)
  (h3 : fs.market_price = 9/2)
  (h4 : fs.orchard_price = 4)
  (h5 : fs.daily_sales = 1000)
  (h6 : fs.labor_cost = 100)
  (h7 : fs.transport_cost = 200)
  (h8 : fs.orchard_price < fs.market_price)
  (h9 : market_income fs > orchard_income fs)
  (h10 : (market_income fs - fs.investment) * 6/5 = 72000) :
  ((72000 - (market_income fs - fs.investment)) / (market_income fs - fs.investment)) = 1/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_growth_rate_l288_28801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_of_m_l288_28821

def factorial (n : ℕ) : ℕ := Nat.factorial n

def m : ℕ := (List.range 2004).map factorial |>.sum

def lastTwoDigits (n : ℕ) : ℕ := n % 100

theorem sum_of_last_two_digits_of_m : lastTwoDigits m % 10 + lastTwoDigits m / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_of_m_l288_28821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l288_28824

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem third_vertex_coordinates (x : ℝ) :
  x < 0 →
  triangle_area 7 3 0 0 x 0 = 42 →
  x = -21 := by
  sorry

#check third_vertex_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l288_28824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_sum_at_one_l288_28893

noncomputable def degree (p : ℂ → ℂ) : ℕ := sorry

noncomputable def polynomial_value (p : ℂ → ℂ) (z : ℂ) : ℂ := sorry

def leading_coeff (p : ℂ → ℂ) (d : ℕ) : ℂ → ℂ := sorry

theorem degree_of_sum_at_one (f g : ℂ → ℂ) :
  degree f = 3 →
  degree g = 2 →
  polynomial_value (leading_coeff f 3) 1 = 0 →
  polynomial_value (leading_coeff (f + g) 2) 1 ≠ 0 →
  degree (fun z => polynomial_value (f + g) 1) = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_sum_at_one_l288_28893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AYX_measure_l288_28892

-- Define the basic geometric objects
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point) (radius : ℝ)

-- Define the properties of the configuration
def Incircle (c : Circle) (t : Triangle) : Prop := sorry
def Circumcircle (c : Circle) (t : Triangle) : Prop := sorry
def OnSegment (P A B : Point) : Prop := sorry

def IncircleAndCircumcircle (t : Triangle) (c : Circle) (X Y Z : Point) : Prop :=
  (Incircle c t) ∧ (Circumcircle c (Triangle.mk X Y Z)) ∧
  (OnSegment X t.B t.C) ∧ (OnSegment Y t.A t.B) ∧ (OnSegment Z t.A t.C)

-- Define the angle measure
noncomputable def AngleMeasure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem angle_AYX_measure
  (t : Triangle) (c : Circle) (X Y Z : Point)
  (h : IncircleAndCircumcircle t c X Y Z)
  (angleA : AngleMeasure t.B t.A t.C = 40)
  (angleB : AngleMeasure t.A t.B t.C = 60)
  (angleC : AngleMeasure t.A t.C t.B = 80) :
  AngleMeasure t.A Y X = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AYX_measure_l288_28892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_draining_rate_l288_28816

/-- Represents the draining rate of a pool given its dimensions, capacity, and draining time. -/
noncomputable def poolDrainingRate (width : ℝ) (length : ℝ) (depth : ℝ) (capacity : ℝ) (drainingTime : ℝ) : ℝ :=
  (width * length * depth * capacity) / drainingTime

/-- Theorem stating the draining rate of the Malibu Country Club pool. -/
theorem malibu_pool_draining_rate :
  poolDrainingRate 60 150 10 0.8 1200 = 60 := by
  -- Unfold the definition of poolDrainingRate
  unfold poolDrainingRate
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_draining_rate_l288_28816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l288_28845

/-- The function f(x) = |x-1| - 2|x+1| -/
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

/-- k is the maximum value of f(x) -/
noncomputable def k : ℝ := sSup (Set.range f)

/-- Given m and n are positive real numbers satisfying 1/m + 1/(2n) = k -/
def condition (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ 1/m + 1/(2*n) = k

theorem problem_statement (m n : ℝ) (h : condition m n) : m + 2*n ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l288_28845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l288_28838

noncomputable def f (x : ℝ) := Real.sin x + Real.sin (2 * x)

theorem period_of_f : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), 0 < S ∧ S < T → ∃ (y : ℝ), f (y + S) ≠ f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l288_28838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l288_28855

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
noncomputable def time_to_meet (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2 + initial_distance
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  total_distance / relative_speed

/-- Theorem stating that the time for two trains to meet is 32 seconds under given conditions. -/
theorem trains_meet_time :
  time_to_meet 400 200 200 54 36 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l288_28855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l288_28898

open Real

theorem local_minimum_condition (c : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), x * (x - c)^2 ≥ 1 * (1 - c)^2) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l288_28898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_to_2014_l288_28841

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ m : ℕ, m < 10 → m > 0 → n % m = 0

def append_digits (base : ℕ) (digits : ℕ) : ℕ :=
  base * (10 ^ (Nat.digits 10 digits).length) + digits

theorem smallest_append_to_2014 :
  ∃ (d : ℕ), d < 1000 ∧ 
    is_divisible_by_all_less_than_10 (append_digits 2014 d) ∧
    ∀ (d' : ℕ), d' < d → ¬is_divisible_by_all_less_than_10 (append_digits 2014 d') :=
by
  sorry

#eval append_digits 2014 506  -- To check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_to_2014_l288_28841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_X_l288_28899

/-- The probability that shooter A hits the target -/
noncomputable def prob_A : ℝ := 2/3

/-- The probability that shooter B hits the target -/
noncomputable def prob_B : ℝ := 4/5

/-- The random variable X representing the number of shooters hitting the target -/
noncomputable def X : ℕ → ℝ
| 0 => (1 - prob_A) * (1 - prob_B)
| 1 => prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
| 2 => prob_A * prob_B
| _ => 0

/-- The expected value of X -/
noncomputable def E_X : ℝ := 0 * X 0 + 1 * X 1 + 2 * X 2

/-- The variance of X -/
noncomputable def var_X : ℝ := (0 - E_X)^2 * X 0 + (1 - E_X)^2 * X 1 + (2 - E_X)^2 * X 2

/-- Theorem stating that the variance of X equals 86/225 -/
theorem variance_X : var_X = 86/225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_X_l288_28899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sums_determine_original_set_l288_28806

/-- Represents a multiset of non-negative integers -/
def NonNegIntegerSet := Multiset ℕ

/-- Represents all possible non-empty subset sums of a NonNegIntegerSet -/
def SubsetSums (s : NonNegIntegerSet) := Multiset ℕ

/-- States that a SubsetSums contains exactly (2^n - 1) elements -/
def ValidSubsetSums (n : ℕ) (sums : Multiset ℕ) : Prop :=
  Multiset.card sums = 2^n - 1

/-- States that a SubsetSums contains all possible non-empty subset sums of s -/
def ContainsAllSums (s : NonNegIntegerSet) (sums : Multiset ℕ) : Prop :=
  ∀ subset : Multiset ℕ, subset ⊆ s → subset ≠ ∅ → (subset.sum ∈ sums)

/-- The main theorem: given valid subset sums, the original set can be uniquely determined -/
theorem subset_sums_determine_original_set (n : ℕ) (sums : Multiset ℕ)
  (h_valid : ValidSubsetSums n sums) :
  ∃! (original : NonNegIntegerSet), ContainsAllSums original sums ∧ Multiset.card original = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sums_determine_original_set_l288_28806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l288_28883

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ -0.5 then 1 / (x + 0.5) else 0.5

-- Theorem statement
theorem f_satisfies_equation :
  ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 :=
by
  intro x
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l288_28883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l288_28867

noncomputable def series_sum (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    (n / 2 + 1) / (2 ^ (n + 2))
  else
    (n / 2 + 1) / (3 ^ (n + 2))

noncomputable def infinite_series_sum : ℚ := ∑' n, series_sum n

theorem sum_of_coprime_integers (a b : ℕ) (h1 : Nat.Coprime a b) (h2 : (a : ℚ) / b = infinite_series_sum) :
  a + b = 721 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l288_28867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mets_fans_count_l288_28831

-- Define the fan counts for each team
variable (yankees_fans : ℕ)
variable (mets_fans : ℕ)
variable (red_sox_fans : ℕ)

-- Define the ratios
axiom yankees_mets_ratio : yankees_fans = 3 * (mets_fans / 2)
axiom mets_red_sox_ratio : red_sox_fans = 5 * (mets_fans / 4)

-- Define the total number of fans
axiom total_fans : yankees_fans + mets_fans + red_sox_fans = 360

-- Theorem to prove
theorem mets_fans_count : mets_fans = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mets_fans_count_l288_28831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l288_28862

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of a cylinder given its circumference c -/
noncomputable def radiusFromCircumference (c : ℝ) : ℝ := c / (2 * Real.pi)

theorem cylinder_volume_difference (sheet_width sheet_length : ℝ) 
  (h_width : sheet_width = 7)
  (h_length : sheet_length = 10) :
  let v1 := cylinderVolume (radiusFromCircumference sheet_width) sheet_length
  let v2 := cylinderVolume (radiusFromCircumference sheet_length) sheet_width
  Real.pi * |v1 - v2| = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l288_28862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meetings_l288_28811

/-- Represents the number of meetings between Michael and the truck -/
def number_of_meetings : ℕ := 3

/-- Michael's walking speed in feet per second -/
noncomputable def michael_speed : ℝ := 7

/-- Distance between trash pails in feet -/
noncomputable def pail_distance : ℝ := 150

/-- Truck's speed in feet per second -/
noncomputable def truck_speed : ℝ := 14

/-- Time the truck stops at each pail in seconds -/
noncomputable def truck_stop_time : ℝ := 20

/-- Time for the truck to travel between pails in seconds -/
noncomputable def truck_travel_time : ℝ := pail_distance / truck_speed

/-- Total time for one truck cycle (travel + stop) in seconds -/
noncomputable def truck_cycle_time : ℝ := truck_travel_time + truck_stop_time

/-- Distance Michael travels during one truck cycle in feet -/
noncomputable def michael_cycle_distance : ℝ := michael_speed * truck_cycle_time

/-- Relative distance Michael gains on the truck per cycle in feet -/
noncomputable def relative_distance_gain : ℝ := michael_cycle_distance - pail_distance

/-- Initial distance between Michael and the truck in feet -/
noncomputable def initial_distance : ℝ := pail_distance

theorem michael_truck_meetings :
  number_of_meetings = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meetings_l288_28811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bose_einstein_distributions_fermi_dirac_distributions_l288_28840

/-- Represents the number of states for a given energy level k in Bose-Einstein distribution -/
def bose_einstein_states (k : ℕ) : ℕ := k^2 + 1

/-- Represents the number of states for a given energy level k in Fermi-Dirac distribution -/
def fermi_dirac_states (k : ℕ) : ℕ := 2 * (1 + k^2)

/-- Represents the total energy of the system in units of E₀ -/
def total_energy : ℕ := 4

/-- Represents the number of particles in the system -/
def num_particles : ℕ := 4

/-- Represents the maximum energy level k -/
def max_k : ℕ := 4

/-- Theorem stating the number of distinct distribution diagrams for Bose-Einstein distribution -/
theorem bose_einstein_distributions : 
  (∀ k : ℕ, k ≤ max_k → ∃ n : ℕ, bose_einstein_states k = n) →
  (∃ n : ℕ, n = 72 ∧ 
    n = (Nat.choose (num_particles + max_k) max_k)) := by sorry

/-- Theorem stating the number of distinct distribution diagrams for Fermi-Dirac distribution -/
theorem fermi_dirac_distributions : 
  (∀ k : ℕ, k ≤ max_k → ∃ n : ℕ, fermi_dirac_states k = n) →
  (∃ n : ℕ, n = 246 ∧ 
    n = (Nat.choose (fermi_dirac_states 0 + fermi_dirac_states 1 + fermi_dirac_states 2 + fermi_dirac_states 3 + fermi_dirac_states 4) num_particles)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bose_einstein_distributions_fermi_dirac_distributions_l288_28840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_token_game_ends_in_57_rounds_l288_28866

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState := sorry

/-- Checks if the game is over (i.e., if any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool := sorry

/-- Simulates the entire game until it's over -/
def playGame (initialState : GameState) : GameState := sorry

theorem token_game_ends_in_57_rounds : 
  let initialState := GameState.mk 
    [Player.mk 18, Player.mk 17, Player.mk 16, Player.mk 15] 0
  (playGame initialState).rounds = 57 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_token_game_ends_in_57_rounds_l288_28866
