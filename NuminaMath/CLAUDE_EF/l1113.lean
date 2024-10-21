import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_and_area_l1113_111347

/-- Truncated cone parameters -/
structure TruncatedCone where
  R : ℝ  -- Large base radius
  r : ℝ  -- Small base radius
  h : ℝ  -- Height
  R_positive : R > 0
  r_positive : r > 0
  h_positive : h > 0
  r_less_R : r < R

/-- Volume of a truncated cone -/
noncomputable def volume (tc : TruncatedCone) : ℝ :=
  (Real.pi / 3) * tc.h * (tc.R^2 + tc.r^2 + tc.R * tc.r)

/-- Lateral surface area of a truncated cone -/
noncomputable def lateral_surface_area (tc : TruncatedCone) : ℝ :=
  Real.pi * (tc.R + tc.r) * Real.sqrt ((tc.R - tc.r)^2 + tc.h^2)

/-- Theorem about the volume and lateral surface area of a specific truncated cone -/
theorem truncated_cone_volume_and_area :
  ∃ (tc : TruncatedCone),
    tc.R = 10 ∧ tc.r = 5 ∧ tc.h = 8 ∧
    volume tc = (1400 * Real.pi) / 3 ∧
    lateral_surface_area tc = 15 * Real.pi * Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_and_area_l1113_111347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_minkowski_sum_properties_l1113_111396

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  -- Add necessary fields and properties
  mk :: -- This is a placeholder for the constructor

/-- The Minkowski sum of two convex polygons -/
def minkowskiSum (l1 : ℝ) (M1 : ConvexPolygon) (l2 : ℝ) (M2 : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- The number of sides of a convex polygon -/
def numSides (M : ConvexPolygon) : ℕ :=
  sorry

/-- The perimeter of a convex polygon -/
def perimeter (M : ConvexPolygon) : ℝ :=
  sorry

theorem convex_polygon_minkowski_sum_properties
  (M1 M2 : ConvexPolygon) (l1 l2 : ℝ) (hl : l1 + l2 = 1) (hl1 : l1 ≥ 0) (hl2 : l2 ≥ 0) :
  let M := minkowskiSum l1 M1 l2 M2
  numSides M ≤ numSides M1 + numSides M2 ∧
  perimeter M = l1 * perimeter M1 + l2 * perimeter M2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_minkowski_sum_properties_l1113_111396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1113_111334

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else (- x)^2 - 4*(- x)

-- State the theorem
theorem solution_set_of_inequality :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x ≥ 0, f x = x^2 - 4*x) →  -- definition for x ≥ 0
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-7) 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1113_111334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_knots_equations_l1113_111388

/-- Represents the number of students in Class 7(1) -/
def m : ℕ := sorry

/-- Represents the planned number of Chinese Knots to make -/
def n : ℕ := sorry

/-- If each person makes 4 knots, they make 2 more than planned -/
axiom four_knots_per_person : 4 * m = n + 2

/-- If each person makes 2 knots, they make 58 fewer than planned -/
axiom two_knots_per_person : 2 * m = n - 58

/-- The equations 4m - 2 = 2m + 58 and (n + 2) / 4 = (n - 58) / 2 correctly represent the relationship between m and n -/
theorem chinese_knots_equations : 
  (4 * m - 2 = 2 * m + 58) ∧ ((n + 2) / 4 = (n - 58) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_knots_equations_l1113_111388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l1113_111303

theorem percentage_calculation : 
  (77 + 2/5 : ℚ) / 100 * 2510 - (56 + 1/4 : ℚ) / 100 * 1680 = 998.34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l1113_111303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_30_degrees_l1113_111367

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (2 * R^2 * α) / 2

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R
    around one of its ends by an angle of 30° (π/6 radians) is equal to πR²/3 -/
theorem rotated_semicircle_area_30_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (π/6) = π * R^2 / 3 := by
  sorry

#check rotated_semicircle_area_30_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_30_degrees_l1113_111367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_perimeter_l1113_111307

-- Define the lines
def line_through_origin (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1}

def line_x_equals_1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1}

def line_y_equals_neg_x_plus_1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 + 1}

-- Define the intersection points
def intersection_point_1 : ℝ × ℝ := (1, 0)

def intersection_point_2 : ℝ × ℝ := (1, -1)

-- Define the perimeter of the triangle
noncomputable def triangle_perimeter : ℝ :=
  Real.sqrt (1^2 + 0^2) +
  Real.sqrt (1^2 + 1^2) +
  abs (0 - (-1))

-- The theorem to prove
theorem isosceles_right_triangle_perimeter :
  triangle_perimeter = 1 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_perimeter_l1113_111307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_g_range_l1113_111349

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Opposite sides

-- Define the condition from the problem
axiom triangle_condition : (Real.sqrt 2 * a - b) / c = (Real.cos B) / (Real.cos C)

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + C)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi/4)

-- Theorem statements
theorem angle_C_measure : C = Real.pi/4 := by sorry

theorem g_range : 
  Set.Icc (0 : ℝ) (Real.pi/3) ⊆ g ⁻¹' (Set.Icc ((Real.sqrt 6 - Real.sqrt 2)/4) 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_g_range_l1113_111349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1113_111377

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

def angle_of_inclination (α : ℝ) : Prop := 0 ≤ α ∧ α < Real.pi

theorem line_inclination :
  ∃ α, angle_of_inclination α ∧
    (∀ x y, line_equation x y → Real.tan α = -(1 / Real.sqrt 3)) ∧
    α = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1113_111377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_equality_iff_a_ge_e_l1113_111370

/-- The function f(x) = ax * e^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

/-- The composition of f with itself -/
noncomputable def f_comp (a : ℝ) (x : ℝ) : ℝ := f a (f a x)

theorem range_equality_iff_a_ge_e (a : ℝ) (ha : a > 0) :
  (Set.range (f a) = Set.range (f_comp a)) ↔ a ≥ Real.exp 1 := by
  sorry

#check range_equality_iff_a_ge_e

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_equality_iff_a_ge_e_l1113_111370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l1113_111328

theorem tan_double_angle (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin (α - π/2) = 3/5) : 
  Real.tan (2 * α) = 24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l1113_111328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1113_111356

-- Define the point (a, b) in the first quadrant
variable (a b : ℝ)

-- Define the conditions
def first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Symmetric point (x₀, y₀) lies on 2x + y + 3 = 0
def symmetric_point_condition (a b : ℝ) : Prop := ∃ x₀ y₀ : ℝ, 2 * x₀ + y₀ + 3 = 0

-- Midpoint of (a, b) and (x₀, y₀) lies on x + y - 2 = 0
def midpoint_condition (a b : ℝ) : Prop := ∃ x₀ y₀ : ℝ, (a + x₀) / 2 + (b + y₀) / 2 - 2 = 0

-- Theorem to prove
theorem min_value_theorem : 
  (∀ a b : ℝ, first_quadrant a b → symmetric_point_condition a b → midpoint_condition a b → 
    (1 / a + 8 / b) ≥ 25 / 9) ∧ 
  (∃ a b : ℝ, first_quadrant a b ∧ symmetric_point_condition a b ∧ midpoint_condition a b ∧
    1 / a + 8 / b = 25 / 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1113_111356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1113_111358

noncomputable def line (k : ℝ) (x : ℝ) : ℝ := k * x

def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

noncomputable def chord_length (k : ℝ) : ℝ :=
  2 * Real.sqrt (4 - ((2 * k + 1) / Real.sqrt (k^2 + 1))^2)

theorem k_range (k : ℝ) : 
  (∀ x, circle_eq x (line k x)) → chord_length k ≥ 2 * Real.sqrt 3 → -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1113_111358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l1113_111346

/-- Proves that y = Ce^(-x^2) is a solution to the differential equation dy/y + 2x dx = 0 -/
theorem solution_verification (C : ℝ) : 
  let y : ℝ → ℝ := λ x => C * Real.exp (-x^2)
  ∀ x, (deriv y x / y x) + 2 * x = 0 := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l1113_111346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l1113_111398

theorem alcohol_mixture_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_coprime : Nat.Coprime m n) :
  (2 * (40 : ℚ) / 100 + 3 * 60 / 100) / 5 = (4 * 30 / 100 + (m / n : ℚ) * 80 / 100) / (4 + m / n) →
  m + n = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l1113_111398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l1113_111380

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 3/5

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

-- Define the angle of inclination
noncomputable def α (x : ℝ) : ℝ := Real.arctan (f' x)

-- Theorem statement
theorem angle_range :
  ∀ x : ℝ, (α x ∈ Set.Icc 0 (π/2)) ∨ (α x ∈ Set.Ioc (2*π/3) π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l1113_111380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_angle_range_l1113_111325

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line passing through the right focus
noncomputable def line (k x : ℝ) : ℝ := k * (x - Real.sqrt 2)

-- Define the condition for intersection
def intersects_right_branch (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧
  hyperbola x1 (line k x1) ∧ hyperbola x2 (line k x2)

-- Theorem statement
theorem hyperbola_intersection_angle_range :
  ∀ k : ℝ, intersects_right_branch k →
  (k < -1 ∨ k > 1) ∧ 
  (Real.arctan k > π/4 ∧ Real.arctan k < 3*π/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_angle_range_l1113_111325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1113_111374

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 1)

-- State the theorem
theorem f_minimum_value :
  (∀ x > -1, f x ≥ 0) ∧ (∃ x > -1, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1113_111374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1113_111382

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (3*a + 2)^(1/3) = 2
def condition2 : Prop := (3*a + b - 1)^(1/2) = 3
def condition3 : Prop := c = Int.floor (Real.sqrt 2)

-- Define the theorem
theorem problem_solution 
  (h1 : condition1 a)
  (h2 : condition2 a b)
  (h3 : condition3 c) :
  a = 2 ∧ b = 4 ∧ c = 1 ∧ (a + b - c)^(1/2) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1113_111382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l1113_111300

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + 2 * b = 2) :
  8 ≤ (4 : ℝ)^a + (16 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l1113_111300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_squares_l1113_111387

noncomputable def square_area (perimeter : ℝ) : ℝ :=
  (perimeter / 4) ^ 2

theorem area_ratio_of_squares (p1 p2 p3 p4 : ℝ) 
  (h1 : p1 = 16) (h2 : p2 = 24) (h3 : p3 = 12) (h4 : p4 = 28) :
  square_area p2 / square_area p4 = 36 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_squares_l1113_111387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_condition_l1113_111342

theorem cube_root_condition (x : ℝ) :
  (∀ x, x < 0 → (x^2)^(1/3) > 0) ∧
  (∃ x, x ≥ 0 ∧ (x^2)^(1/3) > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_condition_l1113_111342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_of_factorial_product_l1113_111371

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def factorial_product (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc * factorial (i + 1)) 1

def is_perfect_square (n : ℕ) : Bool := 
  match n.sqrt with
  | m => m * m = n

def count_perfect_square_divisors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ i => n % i = 0 && is_perfect_square i) |>.length

theorem perfect_square_divisors_of_factorial_product :
  count_perfect_square_divisors (factorial_product 10) = 1120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_of_factorial_product_l1113_111371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_on_y_axis_l1113_111322

theorem ellipse_on_y_axis (α : Real) (h : α ∈ Set.Ioo (π/2) (3*π/4)) :
  ∃ (a b : Real), a > 0 ∧ b > 0 ∧
  ∀ (x y : Real), x^2 * Real.sin α - y^2 * Real.cos α = 1 ↔
    (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a < b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_on_y_axis_l1113_111322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_moment_of_inertia_l1113_111302

/-- Represents a square plate with side length and mass -/
structure SquarePlate where
  sideLength : ℝ
  mass : ℝ

/-- The moment of inertia of a square plate about an axis passing through
    the midpoints of two adjacent sides -/
noncomputable def momentOfInertia (plate : SquarePlate) : ℝ :=
  (5 / 12) * plate.mass * plate.sideLength^2

/-- Theorem stating that the moment of inertia of a square plate about an axis
    passing through the midpoints of two adjacent sides is (5/12)mL^2 -/
theorem square_plate_moment_of_inertia (plate : SquarePlate) :
  momentOfInertia plate = (5 / 12) * plate.mass * plate.sideLength^2 := by
  sorry

#check square_plate_moment_of_inertia

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_moment_of_inertia_l1113_111302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l1113_111309

noncomputable section

def vector2D := ℝ × ℝ

def dot_product (v w : vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def vector_magnitude (v : vector2D) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def cosine_angle (v w : vector2D) : ℝ :=
  dot_product v w / (vector_magnitude v * vector_magnitude w)

theorem cosine_angle_specific_vectors :
  let a : vector2D := (4, 3)
  let b : vector2D := (3 - 8, 18 - 6)
  cosine_angle a b = 16 / 65 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l1113_111309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1113_111326

/-- Represents a valid 8-digit sequence where no two adjacent digits have the same parity -/
def ValidSequence := Fin 8 → Fin 10

/-- Returns true if two natural numbers have different parity -/
def differentParity (a b : ℕ) : Prop := a % 2 ≠ b % 2

/-- A sequence is valid if no two adjacent digits have the same parity -/
def isValidSequence (s : ValidSequence) : Prop :=
  ∀ i : Fin 7, differentParity (s i) (s (i.succ))

/-- The number of valid 8-digit sequences -/
noncomputable def numValidSequences : ℕ := 781250

/-- The theorem stating the number of valid 8-digit sequences -/
theorem count_valid_sequences : numValidSequences = 781250 := by
  -- The proof is omitted
  sorry

-- Additional lemmas to support the main theorem
lemma valid_sequence_count_positive : numValidSequences > 0 := by
  -- The proof is omitted
  sorry

lemma valid_sequence_upper_bound : numValidSequences ≤ 10^8 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1113_111326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_timing_l1113_111317

theorem bus_journey_timing (T : ℝ) (hT : T > 0) : 
  let first_third_time := (3/2) * (T/3)
  let remaining_distance_time := (2*T/3) / (4/3)
  first_third_time + remaining_distance_time = T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_timing_l1113_111317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1113_111350

noncomputable def curve (x : ℝ) : ℝ := Real.log x

def line (x y : ℝ) : Prop := x - y + 3 = 0

noncomputable def distance_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ - c| / Real.sqrt (a^2 + b^2)

theorem shortest_distance_curve_to_line :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  ∀ (x : ℝ), x > 0 →
    distance_to_line x (curve x) 1 (-1) (-3) ≥ distance_to_line x₀ (curve x₀) 1 (-1) (-3) ∧
    distance_to_line x₀ (curve x₀) 1 (-1) (-3) = 2 * Real.sqrt 2 := by
  sorry

#check shortest_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1113_111350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l1113_111320

theorem cos_two_theta (θ : Real) : 
  (2 : Real)^(-5/3 + 3 * Real.cos θ) + 1 = (2 : Real)^(1/3 + Real.cos θ) → 
  Real.cos (2 * θ) = -31/81 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l1113_111320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortar_shell_ground_time_l1113_111355

-- Define the relationship between flight height and time
noncomputable def flight_height (x : ℝ) : ℝ := -1/5 * x^2 + 10*x

-- Theorem: The mortar shell falls to the ground after 50 seconds
theorem mortar_shell_ground_time : 
  ∃ (t : ℝ), t = 50 ∧ flight_height t = 0 :=
by
  -- We'll use 50 as our witness for t
  use 50
  constructor
  · -- First part: t = 50
    rfl
  · -- Second part: flight_height 50 = 0
    -- We'll leave this as sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortar_shell_ground_time_l1113_111355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1113_111353

/-- The tax rate function based on income in thousands of dollars -/
noncomputable def tax_rate (x : ℝ) : ℝ := x / 2

/-- The take-home pay function based on income in thousands of dollars -/
noncomputable def take_home_pay (x : ℝ) : ℝ := 1000 * x - (tax_rate x / 100) * 1000 * x

/-- The income that maximizes take-home pay -/
def max_income : ℝ := 100

theorem max_take_home_pay :
  ∀ x : ℝ, x > 0 → take_home_pay x ≤ take_home_pay max_income := by
  sorry

#eval max_income -- This will not actually evaluate due to noncomputability, but it's here for completeness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1113_111353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_l1113_111369

/-- Represents the total worth of a stock in Rupees -/
def TotalWorth (x : ℝ) : Prop := x ≥ 0

/-- Given conditions about stock sales and overall loss -/
axiom stock_conditions (x : ℝ) :
  TotalWorth x →
  (0.2 * x * 1.1 + 0.8 * x * 0.95 = x - 300)

/-- Theorem: The total worth of the stock is 15000 Rupees -/
theorem stock_worth : TotalWorth 15000 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_l1113_111369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_property_l1113_111344

/-- Prove that for a parabola and a line with specific properties, λ + μ = -1 -/
theorem parabola_line_intersection_property
  (p : ℝ) (m : ℝ) (hm : m ≠ 0) (hp : p > 0)
  (E M N P : ℝ × ℝ)
  (hE : E = (m, 0))
  (hM : M.2^2 = 2 * p * M.1)
  (hN : N.2^2 = 2 * p * N.1)
  (hP : P.1 = 0)
  (hline : ∃ (a b : ℝ), (M.2 - E.2) = a * (M.1 - E.1) ∧
                        (N.2 - E.2) = a * (N.1 - E.1) ∧
                        (P.2 - E.2) = a * (P.1 - E.1) ∧
                        M.2 = a * M.1 + b ∧
                        N.2 = a * N.1 + b ∧
                        P.2 = a * P.1 + b)
  (lambda mu : ℝ)
  (hPM : (M.1 - P.1, M.2 - P.2) = lambda • (E.1 - M.1, E.2 - M.2))
  (hPN : (N.1 - P.1, N.2 - P.2) = mu • (E.1 - N.1, E.2 - N.2)) :
  lambda + mu = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_property_l1113_111344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_bike_time_l1113_111365

noncomputable def weekly_commute (bike_time : ℝ) : ℝ :=
  bike_time + 3 * (bike_time + 10) + (1/3) * bike_time

theorem ryan_bike_time :
  ∃ (bike_time : ℝ),
    weekly_commute bike_time = 160 ∧
    bike_time = 30 := by
  use 30
  constructor
  · simp [weekly_commute]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_bike_time_l1113_111365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_joined_l1113_111321

/-- Proves that 8 new students joined the class given the conditions -/
theorem new_students_joined (original_avg : ℝ) (new_avg : ℝ) (avg_decrease : ℝ) (original_strength : ℕ) : 
  original_avg = 40 →
  new_avg = 32 →
  avg_decrease = 4 →
  original_strength = 8 →
  ∃ (n : ℕ), 
    n = 8 ∧
    (original_avg * (original_strength : ℝ) + new_avg * (n : ℝ)) / ((original_strength : ℝ) + (n : ℝ)) = original_avg - avg_decrease :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_joined_l1113_111321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_two_max_min_condition_l1113_111393

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b * Real.log (x + 1)

-- Part 1: Minimum value when b = -12
theorem min_value_at_two (x : ℝ) (h : x ∈ Set.Icc 1 3) : 
  f (-12) x ≥ f (-12) 2 := by sorry

-- Part 2: Range of b for max and min
theorem max_min_condition (b : ℝ) :
  (∃ (x y : ℝ), x < y ∧ 
    (∀ z, f b z ≤ f b x) ∧ 
    (∀ z, f b z ≥ f b y)) ↔ 
  b < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_two_max_min_condition_l1113_111393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_l1113_111329

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the quadrilateral
def is_cyclic_quadrilateral (q : Quadrilateral) : Prop := sorry

def is_diameter (q : Quadrilateral) : Prop :=
  is_cyclic_quadrilateral q ∧ q.B.1 = -q.C.1 ∧ q.B.2 = -q.C.2

noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ab_length (q : Quadrilateral) :
  is_diameter q →
  length q.B q.C = 8 →
  length q.B q.D = 4 * Real.sqrt 2 →
  angle q.D q.C q.A / angle q.A q.C q.B = 2 →
  length q.A q.B = 2 * (Real.sqrt 6 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_l1113_111329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_pairs_l1113_111301

theorem infinite_square_pairs : ∀ k : ℕ, k > 0 → ∃ a b : ℕ,
  let a_squared := a ^ 2
  let b_squared := b ^ 2
  let concatenated := a_squared * 10^(2*k) + b_squared
  (10^(2*k-1) ≤ a_squared) ∧ (a_squared < 10^(2*k)) ∧
  (10^(2*k-1) ≤ b_squared) ∧ (b_squared < 10^(2*k)) ∧
  (∃ c : ℕ, concatenated = c ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_pairs_l1113_111301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1113_111379

/-- A sequence satisfying the given conditions -/
def satisfies_conditions (a : ℕ → ℚ) : Prop :=
  a 1 = 7/6 ∧
  ∀ n : ℕ, n > 0 → ∃ α β : ℚ,
    a n * α^2 - a (n+1) * α + 1 = 0 ∧
    a n * β^2 - a (n+1) * β + 1 = 0 ∧
    6 * α - 2 * α * β + 6 * β = 3

/-- The theorem stating the general formula for the sequence -/
theorem sequence_formula (a : ℕ → ℚ) (h : satisfies_conditions a) :
  ∀ n : ℕ, n > 0 → a n = 1 / (2^n) + 2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1113_111379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unjoinable_pair_exists_l1113_111390

-- Define the type for points in the plane
def Point : Type := ℝ × ℝ

-- Define the set of points Z
variable (Z : Set Point)

-- Define what it means for a pair of points to be unjoinable
def Unjoinable (Z : Set Point) (p q : Point) : Prop :=
  ∀ (path : List Point), path.head? = some p ∧ path.getLast? = some q →
    ∃ (z : Point), z ∈ Z ∧ z ∈ path

-- Assume the existence of at least one unjoinable pair
variable (h : ∃ (p q : Point), Unjoinable Z p q)

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem unjoinable_pair_exists (r : ℝ) (hr : r > 0) :
  ∃ (p q : Point), Unjoinable Z p q ∧ distance p q = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unjoinable_pair_exists_l1113_111390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_sin_l1113_111359

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => deriv (f_n n)

theorem f_2012_eq_sin : f_n 2012 = f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_sin_l1113_111359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1113_111354

theorem problem_statement :
  (∀ x : ℝ, (2 : ℝ)^x > 0) ∧
  ¬(∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)) ∧
  ((∀ x : ℝ, (2 : ℝ)^x > 0) ∧ ¬(∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1113_111354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectiles_meet_time_l1113_111352

/-- The time (in minutes) it takes for two projectiles to meet -/
noncomputable def time_to_meet (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance / (speed1 + speed2)) * 60

theorem projectiles_meet_time :
  let distance : ℝ := 1386
  let speed1 : ℝ := 445
  let speed2 : ℝ := 545
  time_to_meet distance speed1 speed2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectiles_meet_time_l1113_111352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_theorem_l1113_111383

/-- Calculates the distance run by a thief before being overtaken by a policeman. -/
noncomputable def distance_run_by_thief (initial_distance : ℝ) (thief_speed : ℝ) (policeman_speed : ℝ) : ℝ :=
  let relative_speed := policeman_speed - thief_speed
  let time := initial_distance / relative_speed
  thief_speed * time

/-- Theorem stating that given the specific conditions, the thief will run 2000 meters. -/
theorem thief_distance_theorem :
  distance_run_by_thief 500 (12 * 1000 / 3600) (15 * 1000 / 3600) = 2000 := by
  -- Unfold the definition of distance_run_by_thief
  unfold distance_run_by_thief
  -- Simplify the expression
  simp
  -- The proof is incomplete, so we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_theorem_l1113_111383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1113_111372

/-- Predicate to determine if a given equation represents an ellipse -/
def IsEllipse (x y : ℝ) : Prop := sorry

/-- If the equation (x^2)/(k-1) + (y^2)/(k^2-3) = -1 represents an ellipse, 
    then k is in the interval (-√3, -1) ∪ (-1, 1) -/
theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k - 1) + y^2 / (k^2 - 3) = -1) → IsEllipse x y) → 
  k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1113_111372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1113_111363

noncomputable def h (x : ℝ) : ℝ := (4 * x - 2) / (2 * x + 10)

theorem h_domain : Set.range h = {y | ∃ x : ℝ, x ≠ -5 ∧ y = h x} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1113_111363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isaac_age_is_14_l1113_111318

def siblings_ages : List Nat := [2, 4, 6, 8, 10, 12, 14]

def park_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b = 18

def soccer_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b ≤ 14

def stayed_home (isaac_age : Nat) : Prop :=
  isaac_age ∈ siblings_ages ∧ 4 ∈ siblings_ages

theorem isaac_age_is_14 :
  park_pair siblings_ages →
  soccer_pair siblings_ages →
  stayed_home 14 →
  (∀ age ∈ siblings_ages, age ≠ 14 → age ≠ 4 → 
    (∃ pair : List Nat, pair = [age, 4] ∨ 
             (park_pair pair ∧ pair.toFinset ⊆ siblings_ages.toFinset) ∨
             (soccer_pair pair ∧ pair.toFinset ⊆ siblings_ages.toFinset))) →
  14 ∈ siblings_ages :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isaac_age_is_14_l1113_111318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1113_111394

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of triangle DEF with vertices D(-5, 2), E(8, 2), and F(6, -4) is 39 square units -/
theorem triangle_DEF_area :
  triangleArea (-5) 2 8 2 6 (-4) = 39 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1113_111394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_price_is_76_l1113_111397

/-- The amount made from selling jerseys -/
noncomputable def jersey_total : ℚ := 152

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 2

/-- The amount made per jersey -/
noncomputable def jersey_price : ℚ := jersey_total / jerseys_sold

theorem jersey_price_is_76 : jersey_price = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_price_is_76_l1113_111397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l1113_111376

/-- The surface area of a cone with given slant height and base radius -/
noncomputable def cone_surface_area (slant_height : ℝ) (base_radius : ℝ) : ℝ :=
  Real.pi * base_radius^2 + Real.pi * base_radius * slant_height

/-- Theorem: The surface area of a cone with slant height 2 and base radius 1 is 3π -/
theorem cone_surface_area_specific : 
  cone_surface_area 2 1 = 3 * Real.pi := by
  unfold cone_surface_area
  simp [Real.pi]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l1113_111376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l1113_111304

/-- The distance from a point in polar coordinates to a line in polar form -/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let A := 1
  let B := -Real.sqrt 3
  let C := 2
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating that the distance from the point (2, π/6) to the line ρ sin(θ - π/6) = 1 is 1 -/
theorem distance_point_to_line_is_one :
  distance_point_to_line 2 (Real.pi / 6) (fun ρ θ ↦ ρ * Real.sin (θ - Real.pi / 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l1113_111304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_areas_is_113pi_l1113_111319

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a right triangle with circles at its vertices -/
structure TriangleWithCircles where
  sideA : ℝ
  sideB : ℝ
  sideC : ℝ
  circleA : Circle
  circleB : Circle
  circleC : Circle

/-- The property of circles being mutually externally tangent -/
def areMutuallyExternallyTangent (t : TriangleWithCircles) : Prop :=
  t.circleA.radius + t.circleB.radius = t.sideC ∧
  t.circleA.radius + t.circleC.radius = t.sideB ∧
  t.circleB.radius + t.circleC.radius = t.sideA

/-- The sum of the areas of the circles -/
noncomputable def sumOfAreas (t : TriangleWithCircles) : ℝ :=
  Real.pi * (t.circleA.radius^2 + t.circleB.radius^2 + t.circleC.radius^2)

/-- The main theorem -/
theorem sum_of_areas_is_113pi (t : TriangleWithCircles) 
  (h1 : t.sideA = 5) (h2 : t.sideB = 12) (h3 : t.sideC = 13) 
  (h4 : areMutuallyExternallyTangent t) : sumOfAreas t = 113 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_areas_is_113pi_l1113_111319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l1113_111336

/-- A particle moves in a 2D plane. Its position at time t is given by (3t + 5, 5t - 8). -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 5 * t - 8)

/-- The velocity of the particle is the rate of change of its position with respect to time. -/
def particle_velocity : ℝ × ℝ := (3, 5)

/-- The speed of the particle is the magnitude of its velocity vector. -/
noncomputable def particle_speed : ℝ := Real.sqrt (3^2 + 5^2)

/-- Theorem: The speed of the particle is √34 units per unit time. -/
theorem particle_speed_is_sqrt_34 : particle_speed = Real.sqrt 34 := by
  -- Unfold the definition of particle_speed
  unfold particle_speed
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l1113_111336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_surface_area_l1113_111389

/-- Represents a cube with a given edge length -/
structure Cube where
  edge_length : ℝ

/-- Calculates the surface area of a cube -/
noncomputable def surface_area (c : Cube) : ℝ := 6 * c.edge_length ^ 2

/-- Represents the composite shape formed by attaching two smaller cubes to a larger cube -/
structure CompositeShape where
  large_cube : Cube
  small_cube1 : Cube
  small_cube2 : Cube

/-- Calculates the surface area of the composite shape -/
noncomputable def composite_surface_area (shape : CompositeShape) : ℝ :=
  surface_area shape.large_cube +
  2 * (4 * shape.small_cube1.edge_length ^ 2 + shape.small_cube1.edge_length ^ 2) +
  2 * (4 * shape.small_cube2.edge_length ^ 2 + shape.small_cube2.edge_length ^ 2) -
  2 * (4 * shape.large_cube.edge_length ^ 2 / 5)

/-- The theorem stating that the surface area of the composite shape is 270 -/
theorem composite_shape_surface_area :
  ∀ (shape : CompositeShape),
    shape.large_cube.edge_length = 5 →
    shape.small_cube1.edge_length ≠ shape.small_cube2.edge_length →
    shape.small_cube1.edge_length ≠ shape.large_cube.edge_length →
    shape.small_cube2.edge_length ≠ shape.large_cube.edge_length →
    composite_surface_area shape = 270 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_surface_area_l1113_111389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l1113_111362

theorem impossibility_of_arrangement : ¬ ∃ (a b : Fin 1986 → Fin 3972), 
  (∀ k : Fin 1986, a k < b k ∧ b k - a k = k + 1) ∧
  (∀ i j : Fin 1986, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l1113_111362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l1113_111310

/-- Represents a triangle with a perimeter -/
structure Triangle where
  perimeter : ℝ

/-- Represents a large triangle divided into smaller triangles -/
structure DividedTriangle where
  largeTriangle : Triangle
  smallTriangles : Finset Triangle
  h_nine : smallTriangles.card = 9
  h_equal_perimeters : ∀ t1 t2, t1 ∈ smallTriangles → t2 ∈ smallTriangles → t1.perimeter = t2.perimeter

/-- The theorem statement -/
theorem divided_triangle_perimeter 
  (dt : DividedTriangle) 
  (h_large_perimeter : dt.largeTriangle.perimeter = 120) :
  ∀ t, t ∈ dt.smallTriangles → t.perimeter = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l1113_111310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l1113_111338

def T : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 1 => 3^(T n)

theorem t_50_mod_7 : T 50 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l1113_111338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_of_1_and_1008_closest_to_2_l1113_111378

/-- The harmonic mean of two numbers -/
noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

/-- Function to calculate the distance between two real numbers -/
noncomputable def distance (x y : ℝ) : ℝ := abs (x - y)

theorem harmonic_mean_of_1_and_1008_closest_to_2 :
  ∀ (n : ℤ), n ≠ 2 →
  distance (harmonicMean 1 1008) 2 < distance (harmonicMean 1 1008) (n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_of_1_and_1008_closest_to_2_l1113_111378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1113_111381

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties : 
  (∀ x, f (Real.pi/3 - x) = f (Real.pi/3 + x)) ∧ 
  (∀ x, f x ≤ 3) ∧ 
  (∃ x, f x = 3) ∧ 
  (¬ ∀ x, f (-x) = -f x) ∧
  (¬ ∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), ∀ y ∈ Set.Icc (-Real.pi/4) (Real.pi/4), x < y → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1113_111381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_count_l1113_111311

theorem election_votes_count :
  ∃ (V : ℕ),
  let invalid_votes : ℕ := 200
  let winner_percent : ℚ := 33 / 100
  let runner_up_percent : ℚ := 32 / 100
  let remaining_percent : ℚ := 32 / 100
  winner_percent + runner_up_percent + remaining_percent = 97 / 100 ∧
  winner_percent - runner_up_percent = 1 / 100 ∧
  (winner_percent - runner_up_percent) * V = 700 ∧
  V + invalid_votes = 70200 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_count_l1113_111311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l1113_111315

-- Define L(m) as the x-coordinate of the left endpoint of the intersection
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as [L(-m) - L(m)] / m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ → |r m - 1/2| < ε :=
by
  sorry

#check limit_of_r_as_m_approaches_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l1113_111315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_channel_data_volume_l1113_111337

/-- Calculates the total data volume in KiB for a stereo audio channel with metadata -/
noncomputable def total_data_volume (session_duration_minutes : ℕ) (sampling_rate : ℕ) (sampling_depth : ℕ) (metadata_bytes : ℕ) (metadata_interval_kib : ℕ) : ℝ :=
  let session_duration_seconds := session_duration_minutes * 60
  let mono_data_rate := sampling_rate * sampling_depth
  let stereo_data_rate := 2 * mono_data_rate
  let audio_data_volume := stereo_data_rate * session_duration_seconds
  let audio_data_kib := (audio_data_volume : ℝ) / (8 * 1024)
  let metadata_chunks := audio_data_kib / metadata_interval_kib
  let metadata_volume := metadata_chunks * (metadata_bytes * 8 : ℝ)
  let total_bits := (audio_data_volume : ℝ) + metadata_volume
  total_bits / (8 * 1024)

/-- The total data volume for the given audio channel specifications is approximately 807.53 KiB -/
theorem audio_channel_data_volume :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_data_volume 51 63 17 47 5 - 807.53| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_channel_data_volume_l1113_111337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1113_111375

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

def point : ℝ × ℝ := (-1, -1)

theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := (2 : ℝ) / (x₀ + 2)^2  -- Derivative of f at x₀
  (2 : ℝ) * x - y + 1 = 0 ↔ y - y₀ = m * (x - x₀) :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1113_111375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1113_111306

theorem complex_modulus :
  Complex.abs ((1 + Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1113_111306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tint_percentage_l1113_111392

/-- Proves that adding 8 liters of blue tint to a 40-liter mixture with 20% blue tint
    results in a new mixture with 1/3 blue tint. -/
theorem blue_tint_percentage (original_volume : ℝ) (original_blue_percent : ℝ) 
  (added_blue : ℝ) : 
  original_volume = 40 → 
  original_blue_percent = 0.20 → 
  added_blue = 8 → 
  (original_volume * original_blue_percent + added_blue) / (original_volume + added_blue) = 1/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tint_percentage_l1113_111392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_sphere_intersection_l1113_111330

def origin : ℝ × ℝ × ℝ := (0, 0, 0)

def plane_intersect (a b c : ℝ) (O A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ),
    A = (α, 0, 0) ∧
    B = (0, β, 0) ∧
    C = (0, 0, γ) ∧
    A ≠ O ∧ B ≠ O ∧ C ≠ O ∧
    (a / α + b / β + c / γ = 1)

def sphere_center (p q r : ℝ) (O A B C : ℝ × ℝ × ℝ) : Prop :=
  let center := (p, q, r)
  dist center O = dist center A ∧
  dist center O = dist center B ∧
  dist center O = dist center C

theorem plane_sphere_intersection
  (a b c p q r : ℝ) (O A B C : ℝ × ℝ × ℝ) :
  O = origin →
  plane_intersect a b c O A B C →
  sphere_center p q r O A B C →
  a / p + b / q + c / r = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_sphere_intersection_l1113_111330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l1113_111341

theorem equation_solution_set : 
  {x : ℝ | (4 : ℝ)^x + 2^(x+1) - 3 = 0} = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l1113_111341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_eq_90_l1113_111327

/-- A sequence defined by a₁ = 0 and aₙ₊₁ = aₙ + 2n for n ≥ 1 -/
def a : ℕ → ℕ
  | 0 => 0  -- This case is added to handle the base case
  | 1 => 0
  | n + 1 => a n + 2 * n

/-- The 10th term of the sequence equals 90 -/
theorem a_10_eq_90 : a 10 = 90 := by
  -- Proof will be added here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_eq_90_l1113_111327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_properties_l1113_111340

/-- Linear equation type -/
structure LinearEquation where
  a : ℝ
  b : ℝ

/-- Root of a linear equation -/
noncomputable def root (eq : LinearEquation) : ℝ := -eq.b / eq.a

/-- Homogeneous function of degree zero -/
def isHomogeneousZero (f : LinearEquation → ℝ) : Prop :=
  ∀ (c : ℝ) (eq : LinearEquation), c ≠ 0 → f {a := c * eq.a, b := c * eq.b} = f eq

/-- Theorem stating that the root function is homogeneous of degree zero -/
theorem root_properties (eq : LinearEquation) (h : eq.a ≠ 0) :
  isHomogeneousZero root := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_properties_l1113_111340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_length_greater_than_three_l1113_111308

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- C is a right angle
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AC = 3
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 9

-- Define a point P on side BC
def PointOnBC (B C P : ℝ × ℝ) : Prop :=
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)

-- Length of AP
noncomputable def LengthAP (A P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

-- Theorem statement
theorem ap_length_greater_than_three 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (P : ℝ × ℝ) 
  (h_p_on_bc : PointOnBC B C P) : 
  LengthAP A P > 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_length_greater_than_three_l1113_111308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l1113_111351

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point2D) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point (x, y) is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point2D) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if two conics share the same foci -/
def shareFoci (e : Ellipse) (h : Hyperbola) : Prop :=
  e.a^2 - e.b^2 = h.a^2 + h.b^2

/-- The main theorem -/
theorem trajectory_is_straight_line (n m : ℝ) (h₁ : 0 < n) (h₂ : n < 4) :
  let e : Ellipse := ⟨2, Real.sqrt n⟩
  let h : Hyperbola := ⟨2 * Real.sqrt 2, Real.sqrt m⟩
  shareFoci e h →
  ∃ (a b c : ℝ), a * n + b * m + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l1113_111351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1113_111305

/-- The time taken for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  total_length / relative_speed

/-- Theorem stating that the time taken for the given trains to cross is approximately 11.16 seconds -/
theorem train_crossing_time_approx :
  ∃ ε > 0, |train_crossing_time 150 160 60 40 - 11.16| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1113_111305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_with_negative_coeff_and_positive_powers_l1113_111332

/-- A polynomial with real coefficients -/
def MyPolynomial := ℕ → ℝ

/-- Check if all coefficients of a polynomial are positive -/
def all_positive (p : MyPolynomial) : Prop :=
  ∀ n, p n > 0

/-- The n-th power of a polynomial -/
noncomputable def pow_poly (p : MyPolynomial) (n : ℕ) : MyPolynomial :=
  sorry

theorem exists_polynomial_with_negative_coeff_and_positive_powers :
  ∃ (P : MyPolynomial),
    (∃ k, P k < 0) ∧
    (∀ n > 1, all_positive (pow_poly P n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_with_negative_coeff_and_positive_powers_l1113_111332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l1113_111312

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the area of a circular segment -/
noncomputable def segmentArea (r d : ℝ) : ℝ :=
  r^2 * Real.arccos (d / r) - d * Real.sqrt (r^2 - d^2)

theorem circle_area_problem (A B C : Circle) (M : ℝ × ℝ) : 
  A.radius = 1 →
  B.radius = 1 →
  C.radius = 2 →
  distance A.center B.center = 3 →
  M = ((A.center.1 + B.center.1) / 2, (A.center.2 + B.center.2) / 2) →
  distance C.center M = C.radius →
  let d := 0.5
  let segArea := segmentArea C.radius d
  let areaInside := π * C.radius^2 - 2 * segArea
  areaInside = 4 * π - 2 * segArea := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l1113_111312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1113_111323

noncomputable def data_set : List ℝ := [6, 7, 8, 8, 9, 10]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem variance_of_dataset :
  variance data_set = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1113_111323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_ratio_l1113_111368

/-- Predicate indicating that a point (x, y) lies on an ellipse -/
def IsEllipse (m : ℝ) (x y : ℝ) : Prop := x^2 + m*y^2 = 1

/-- Predicate indicating that the foci of the ellipse are on the y-axis -/
def FociOnYAxis (m : ℝ) : Prop := sorry

/-- Predicate indicating that the length of the major axis is twice the length of the minor axis -/
def MajorAxisTwiceMinorAxis (m : ℝ) : Prop := sorry

/-- Given an ellipse with equation x^2 + my^2 = 1, where the foci are on the y-axis
    and the length of the major axis is twice the length of the minor axis,
    prove that m = 1/4 -/
theorem ellipse_axis_ratio (m : ℝ) :
  (∀ x y : ℝ, IsEllipse m x y) →
  FociOnYAxis m →
  MajorAxisTwiceMinorAxis m →
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_ratio_l1113_111368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_constant_term_and_integral_l1113_111391

theorem binomial_constant_term_and_integral (m : ℝ) : 
  (m = (Nat.choose 6 4) * (Real.sqrt 5 / 5)^2) →
  (∫ (x : ℝ) in (1 : ℝ)..m, x^2 - 2*x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_constant_term_and_integral_l1113_111391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l1113_111343

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
noncomputable def foci : ℝ × ℝ × ℝ × ℝ := (-Real.sqrt 3, 0, Real.sqrt 3, 0)

-- Define the condition for vectors being perpendicular
def vectors_perpendicular (x y : ℝ) : Prop :=
  let (f1x, f1y, f2x, f2y) := foci
  (x - f1x) * (x - f2x) + (y - f1y) * (y - f2y) = 0

-- Theorem statement
theorem distance_to_y_axis (x y : ℝ) :
  is_on_ellipse x y → vectors_perpendicular x y → abs x = (2 * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l1113_111343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l1113_111348

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 8

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop :=
  x + y - 1 = 0 ∨ x - y + 3 = 0

-- Theorem statement
theorem circle_and_chord_properties :
  -- Given conditions
  (circle_eq 1 2) ∧
  (circle_eq (-3) 2) ∧
  (circle_eq (-1) (2 * Real.sqrt 2)) ∧
  (∃ x y : ℝ, chord_AB x y ∧ (x + 1)^2 + (y - 2)^2 = 7) →
  -- Conclusions
  (∀ x y : ℝ, circle_eq x y ↔ (x + 1)^2 + y^2 = 8) ∧
  (∀ x y : ℝ, chord_AB x y ↔ (x + y - 1 = 0 ∨ x - y + 3 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l1113_111348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_inverse_x_l1113_111339

open Real MeasureTheory

theorem integral_x_plus_inverse_x : 
  ∫ x in Set.Icc 1 (exp 1), (x + 1/x) = (exp 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_inverse_x_l1113_111339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_system_properties_l1113_111386

/-- Represents a gear with a radius and rotation direction -/
structure Gear where
  radius : ℚ
  clockwise : Bool

/-- Represents a system of three gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear

/-- The gear ratio between two gears -/
def gearRatio (g1 g2 : Gear) : ℚ := g1.radius / g2.radius

theorem gear_system_properties (s : GearSystem) 
  (h1 : s.A.radius = 15)
  (h2 : s.B.radius = 10)
  (h3 : s.C.radius = 5)
  (h4 : s.A.clockwise = true) :
  (s.C.clockwise = true) ∧ 
  (gearRatio s.A s.C = 3) := by
  sorry

#check gear_system_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_system_properties_l1113_111386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_range_l1113_111373

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := c^x

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def inequality_holds (c : ℝ) : Prop :=
  ∀ x, x + |x - 2*c| > 1

theorem c_value_range (c : ℝ) (h : c > 0) :
  (monotonically_decreasing (f c) ∨ inequality_holds c) ∧
  ¬(monotonically_decreasing (f c) ∧ inequality_holds c) ↔
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 := by
  sorry

#check c_value_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_range_l1113_111373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l1113_111345

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 1/2) / Real.log a

theorem f_positive_iff_a_in_range :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x > 0) ↔ 
  a ∈ Set.union (Set.Ioo (1/2) (5/8)) (Set.Ioi (3/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l1113_111345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1113_111335

theorem min_value_theorem (n : ℝ) (hn : n > 0) : 
  n / 2 + 50 / n ≥ 10 ∧ (n / 2 + 50 / n = 10 ↔ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1113_111335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_max_value_l1113_111361

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 2)
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorems
theorem f_increasing : 
  ∀ x y, x ∈ Set.Icc 0 Real.pi → y ∈ Set.Icc 0 Real.pi → x < y → f x < f y :=
by sorry

theorem g_max_value : 
  ∃ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), g x = 3 / 2 ∧ 
  ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), g y ≤ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_max_value_l1113_111361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l1113_111357

open Real MeasureTheory

/-- The area bounded by the graph of y = arccos(cos x) and the x-axis on the interval [0, 2π] is π² -/
theorem area_arccos_cos : ∫ x in Set.Icc 0 (2 * π), arccos (cos x) = π ^ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l1113_111357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1113_111385

-- Define the line
noncomputable def line (x : ℝ) : ℝ := 2 * x + 7

-- Define the vector parameterizations
noncomputable def param_A (t : ℝ) : ℝ × ℝ := (0 + 2*t, 7 + t)
noncomputable def param_B (t : ℝ) : ℝ × ℝ := (-7/2 - t, 0 - 2*t)
noncomputable def param_C (t : ℝ) : ℝ × ℝ := (1 + 6*t, 9 + 3*t)
noncomputable def param_D (t : ℝ) : ℝ × ℝ := (2 + t/2, -1 + t)
noncomputable def param_E (t : ℝ) : ℝ × ℝ := (-7 + t/10, -7 + t/5)

-- Define a function to check if a parameterization is valid
def is_valid_param (param : ℝ → ℝ × ℝ) : Prop :=
  ∀ t, (param t).2 = line (param t).1

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  is_valid_param param_B ∧ 
  is_valid_param param_E ∧ 
  ¬is_valid_param param_A ∧ 
  ¬is_valid_param param_C ∧ 
  ¬is_valid_param param_D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1113_111385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l1113_111324

theorem two_solutions_for_equation : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n > 0 ∧ ((n + 500) / 50 : ℕ) = ⌊Real.sqrt n⌋) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l1113_111324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_symmetry_l1113_111395

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

-- Define the center of the ellipse
def center : ℝ × ℝ := (1, 0)

-- Define the slopes of the axes of symmetry
noncomputable def k₁ : ℝ := (13 + 5 * Real.sqrt 17) / 16
noncomputable def k₂ : ℝ := (13 - 5 * Real.sqrt 17) / 16

-- Define the axes of symmetry equations
def axis_equation₁ (x y : ℝ) : Prop := y = k₁ * (x - 1)
def axis_equation₂ (x y : ℝ) : Prop := y = k₂ * (x - 1)

-- State the theorem
theorem ellipse_symmetry :
  (ellipse_equation 1 1 ∧ ellipse_equation 1 (-1)) →
  (∀ x y : ℝ, ellipse_equation x y →
    (∃ x' y', ellipse_equation x' y' ∧
      ((x + x') / 2, (y + y') / 2) = center) ∧
    (axis_equation₁ x y ∨ axis_equation₂ x y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_symmetry_l1113_111395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_of_equation_l1113_111360

theorem positive_integer_solutions_of_equation :
  ∀ x y : ℕ,
  x > 0 ∧ y > 0 →
  (4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0) ↔
  (∃ a : ℕ, a > 0 ∧ x = 2 * a ∧ y = a) ∨ (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_of_equation_l1113_111360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_monotonicity_of_f_l1113_111384

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - b * x + Real.log x

-- Part 1: Tangent line at x = 1 when a = b = 1
theorem tangent_line_at_x_1 :
  let a : ℝ := 1
  let b : ℝ := 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := f a b x₀
  let m : ℝ := (2 * a * x₀ - b + 1 / x₀)
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 2 * x - y - 2 = 0 := by
  sorry

-- Part 2: Monotonicity when a ≤ 0 and b = 2a + 1
theorem monotonicity_of_f (a : ℝ) (h : a ≤ 0) :
  let b : ℝ := 2 * a + 1
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a b x₁ < f a b x₂) ∧
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f a b x₁ > f a b x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_monotonicity_of_f_l1113_111384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_statues_l1113_111366

/-- The amount of paint required for a group of statues -/
noncomputable def paint_required (num_statues : ℕ) (statue_height : ℝ) (reference_height : ℝ) (reference_paint : ℝ) : ℝ :=
  (num_statues : ℝ) * reference_paint * (statue_height / reference_height) ^ 2

/-- Theorem: The amount of paint required for 1000 statues, each 3 feet high, 
    is 1000/9 pints, given that 1 pint is required for a 9-foot statue -/
theorem paint_for_statues : 
  paint_required 1000 3 9 1 = 1000 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_statues_l1113_111366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravellingRateIsPointSix_l1113_111333

/-- Represents the dimensions and cost of a rectangular plot with a gravel path -/
structure PlotWithPath where
  length : ℝ
  width : ℝ
  pathWidth : ℝ
  totalCost : ℝ

/-- Calculates the rate per square meter for gravelling the path -/
noncomputable def gravellingRate (plot : PlotWithPath) : ℝ :=
  let totalArea := plot.length * plot.width
  let innerLength := plot.length - 2 * plot.pathWidth
  let innerWidth := plot.width - 2 * plot.pathWidth
  let innerArea := innerLength * innerWidth
  let pathArea := totalArea - innerArea
  plot.totalCost / pathArea

/-- Theorem stating that for the given plot dimensions and cost, the gravelling rate is 0.6 -/
theorem gravellingRateIsPointSix : 
  let plot := PlotWithPath.mk 110 65 2.5 510
  gravellingRate plot = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravellingRateIsPointSix_l1113_111333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1113_111331

-- Define the spade operation for positive real numbers
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

-- Theorem statement
theorem spade_nested_calculation :
  ∀ (x y : ℝ), x > 0 → y > 0 →
  spade 3 (spade 3 3) = 21 / 8 :=
by
  intros x y hx hy
  -- Unfold the definition of spade
  simp [spade]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1113_111331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_three_greater_than_negative_two_l1113_111313

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (abs x) / Real.log a

theorem f_negative_three_greater_than_negative_two 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) :
  f a (-3) > f a (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_three_greater_than_negative_two_l1113_111313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficientOfX2Is30_l1113_111316

noncomputable def polynomialExpansion (x : ℝ) : ℝ := (1 + 1/x^2) * (1 + x)^6

noncomputable def coefficientOfX2 (f : ℝ → ℝ) : ℝ :=
  (λ x => (deriv (deriv f) x) / 2) 0

theorem coefficientOfX2Is30 : coefficientOfX2 polynomialExpansion = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficientOfX2Is30_l1113_111316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l1113_111314

/-- A line with slope k -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The circle (x-1)^2 + y^2 = 1 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line of symmetry x - y + b = 0 -/
def symmetry_line (b : ℝ) (x y : ℝ) : Prop := x - y + b = 0

/-- Two points are symmetric about a line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) (b : ℝ) : Prop :=
  symmetry_line b ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

theorem intersection_symmetry (k b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    y₁ = line k x₁ ∧ my_circle x₁ y₁ ∧
    y₂ = line k x₂ ∧ my_circle x₂ y₂ ∧
    symmetric_points x₁ y₁ x₂ y₂ b) →
  k = -1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l1113_111314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sequence_derivative_bound_l1113_111399

def harmonic_sequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 / (n + 2 : ℚ)

def derive_sequence (a : ℕ → ℚ) : ℕ → ℚ :=
  λ i => (a i + a (i + 1)) / 2

def nth_derivative : (ℕ → ℚ) → ℕ → (ℕ → ℚ)
  | a, 0 => a
  | a, n + 1 => derive_sequence (nth_derivative a n)

theorem harmonic_sequence_derivative_bound (n : ℕ) (h : n > 0):
  ∃ x : ℚ, nth_derivative harmonic_sequence (n - 1) 0 = x ∧ x < 2 / n :=
by
  sorry

#check harmonic_sequence_derivative_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sequence_derivative_bound_l1113_111399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_distribution_l1113_111364

/-- The number of people in the second distribution -/
def x : ℕ := sorry

/-- The total amount in the first distribution -/
def amount1 : ℕ := 100

/-- The total amount in the second distribution -/
def amount2 : ℕ := 150

/-- The difference in the number of people between the two distributions -/
def diff : ℕ := 5

/-- The theorem representing the problem -/
theorem fibonacci_distribution (h : x > diff) :
  (amount1 : ℚ) / (x - diff) = amount2 / x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_distribution_l1113_111364
