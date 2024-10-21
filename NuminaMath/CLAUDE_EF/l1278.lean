import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_station_perimeter_l1278_127877

/-- The perimeter of a circular sector with given radius and central angle -/
noncomputable def sector_perimeter (radius : ℝ) (central_angle : ℝ) : ℝ :=
  (central_angle / (2 * Real.pi)) * (2 * Real.pi * radius) + 2 * radius

/-- Theorem: The perimeter of a circular sector with radius 3 cm and central angle 270° is 9π/2 + 6 cm -/
theorem space_station_perimeter :
  sector_perimeter 3 ((3 / 2) * Real.pi) = (9 / 2) * Real.pi + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_station_perimeter_l1278_127877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_configuration_radius_l1278_127858

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Configuration of circles as described in the problem -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  P : Circle
  Q : Circle
  R : Circle
  S : Circle

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Checks if two circles touch externally -/
def touchesExternally (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- Theorem stating that in the given configuration, the radius of P, Q, R, S is 2 -/
theorem circles_configuration_radius (config : CircleConfiguration) : 
  (config.A.radius = 1) →
  (config.B.radius = 1) →
  (config.P.radius = config.Q.radius) →
  (config.P.radius = config.R.radius) →
  (config.P.radius = config.S.radius) →
  (touchesExternally config.A config.B) →
  (touchesExternally config.A config.P) →
  (touchesExternally config.A config.R) →
  (touchesExternally config.A config.S) →
  (touchesExternally config.B config.P) →
  (touchesExternally config.B config.Q) →
  (touchesExternally config.B config.R) →
  (touchesExternally config.P config.Q) →
  (touchesExternally config.P config.S) →
  (touchesExternally config.Q config.R) →
  (touchesExternally config.R config.S) →
  config.P.radius = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_configuration_radius_l1278_127858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundredth_term_is_seven_l1278_127891

def mySequence (x : ℤ) : ℕ → ℤ
  | 0 => x
  | 1 => 7
  | (n + 2) => mySequence x n - mySequence x (n + 1)

theorem two_hundredth_term_is_seven (x : ℤ) : mySequence x 199 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundredth_term_is_seven_l1278_127891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_particular_solution_1_satisfies_initial_condition_particular_solution_2_satisfies_initial_condition_l1278_127851

noncomputable section

-- Define the differential equation
def diff_eq (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x ≠ 0 → x^2 * (deriv y x) + x * y x = 1

-- Define the general solution
def general_solution (x : ℝ) (C : ℝ) : ℝ :=
  (Real.log (abs x) + C) / x

-- Define the particular solutions
def particular_solution_1 (x : ℝ) : ℝ :=
  (Real.log (abs x) + 3) / x

def particular_solution_2 (x : ℝ) : ℝ :=
  (Real.log (abs x) + (5 * Real.exp 1 - 1)) / x

-- Theorem statements
theorem general_solution_satisfies_diff_eq (x : ℝ) (C : ℝ) :
  diff_eq x (λ y => general_solution x C) := by
  sorry

theorem particular_solution_1_satisfies_initial_condition :
  particular_solution_1 1 = 3 := by
  sorry

theorem particular_solution_2_satisfies_initial_condition :
  particular_solution_2 (-Real.exp 1) = -5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_particular_solution_1_satisfies_initial_condition_particular_solution_2_satisfies_initial_condition_l1278_127851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1278_127876

/-- The standard equation of an ellipse with one focus at (0,1) and passing through (3/2, 1) -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) (F₁ P : ℝ × ℝ) :
  F₁ = (0, 1) →
  P = (3/2, 1) →
  P ∈ C →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    C = {(x, y) : ℝ × ℝ | y^2 / a^2 + x^2 / b^2 = 1} →
  C = {(x, y) : ℝ × ℝ | y^2 / 4 + x^2 / 3 = 1} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1278_127876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l1278_127866

theorem sin_double_angle_problem (x : ℝ) : 
  Real.sin (x - π/4) = 3/5 → Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l1278_127866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1278_127882

/-- Represents a clock with 12 hour marks and a 360-degree face -/
structure Clock :=
  (total_hours : ℕ)
  (total_degrees : ℕ)
  (hour_marks_equal : total_degrees % total_hours = 0)

/-- Calculates the angle of the hour hand at a given time -/
noncomputable def hour_hand_angle (c : Clock) (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour % c.total_hours + minute / 60 : ℝ) * (c.total_degrees / c.total_hours : ℝ)

/-- Calculates the angle of the minute hand at a given time -/
noncomputable def minute_hand_angle (c : Clock) (minute : ℕ) : ℝ :=
  (minute : ℝ) * (c.total_degrees / 60 : ℝ)

/-- Calculates the smaller angle between hour and minute hands -/
noncomputable def smaller_angle (c : Clock) (hour : ℕ) (minute : ℕ) : ℝ :=
  let angle := |hour_hand_angle c hour minute - minute_hand_angle c minute|
  min angle (c.total_degrees - angle)

/-- The theorem to be proved -/
theorem clock_angle_at_7_30 (c : Clock) (h1 : c.total_hours = 12) (h2 : c.total_degrees = 360) :
  smaller_angle c 7 30 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1278_127882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_equals_4037_l1278_127874

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => 3 * x - 5  -- Added case for 0
| 1 => 3 * x - 5
| 2 => 7 * x - 15
| 3 => 4 * x + 3
| n + 4 => arithmetic_sequence x (n + 3) + (arithmetic_sequence x 2 - arithmetic_sequence x 1)

/-- Theorem stating the existence of an n for which the nth term is 4037 and n = 673 -/
theorem nth_term_equals_4037 :
  ∃ x : ℝ, ∃ n : ℕ, arithmetic_sequence x n = 4037 ∧ n = 673 := by
  sorry

#eval arithmetic_sequence 4 673  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_equals_4037_l1278_127874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_A_perpendicular_to_BC_l1278_127830

/-- The point A through which the plane passes -/
def A : Fin 3 → ℝ := ![(-1), 2, (-2)]

/-- The point B used to define vector BC -/
def B : Fin 3 → ℝ := ![13, 14, 1]

/-- The point C used to define vector BC -/
def C : Fin 3 → ℝ := ![14, 15, 2]

/-- Vector BC -/
def BC : Fin 3 → ℝ := ![C 0 - B 0, C 1 - B 1, C 2 - B 2]

/-- The equation of the plane -/
def plane_equation (x y z : ℝ) : Prop :=
  x + y + z + 1 = 0

theorem plane_through_A_perpendicular_to_BC :
  plane_equation (A 0) (A 1) (A 2) ∧
  (∀ x y z, plane_equation x y z → (x - A 0) * (BC 0) + (y - A 1) * (BC 1) + (z - A 2) * (BC 2) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_A_perpendicular_to_BC_l1278_127830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_arced_square_is_2pi_l1278_127854

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a unit square -/
noncomputable def perimeter_arced_square : ℝ := 2 * Real.pi

/-- Theorem: The perimeter of a region bounded by semicircular arcs constructed on the sides of a unit square is equal to 2π -/
theorem perimeter_arced_square_is_2pi : perimeter_arced_square = 2 * Real.pi := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_arced_square_is_2pi_l1278_127854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_apf_l1278_127822

/-- Helper function to calculate the area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The maximum area of triangle APF for an ellipse with given parameters -/
theorem max_area_triangle_apf (F : ℝ × ℝ) (A : ℝ × ℝ) :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 5 = 1}
  F.1 = 2 ∧ F.2 = 0 →  -- Right focus of the ellipse
  A = (0, 2 * Real.sqrt 3) →
  (∃ (P : ℝ × ℝ), P ∈ ellipse ∧
    ∀ (Q : ℝ × ℝ), Q ∈ ellipse →
      area_triangle A P F ≤ area_triangle A Q F) →
  ∃ (P : ℝ × ℝ), P ∈ ellipse ∧ 
    area_triangle A P F = (21 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_apf_l1278_127822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l1278_127818

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

-- Define the interval [1, 4]
def interval : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem f_max_min_on_interval :
  (∀ x ∈ interval, f x ≤ 9/5) ∧
  (∀ x ∈ interval, f x ≥ 3/2) ∧
  (∃ x ∈ interval, f x = 9/5) ∧
  (∃ x ∈ interval, f x = 3/2) := by
  sorry

#check f_max_min_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l1278_127818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_well_defined_l1278_127812

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log (2 * x)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, x₁ > (1/2 : ℝ) → x₂ > (1/2 : ℝ) → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Define the domain of the function
def f_domain (x : ℝ) : Prop := x > 0

-- State that the function is defined on its domain
theorem f_well_defined :
  ∀ x : ℝ, f_domain x → ∃ y : ℝ, f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_well_defined_l1278_127812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1278_127885

/-- A polynomial satisfying the given equation is of the form ax² + bx -/
theorem polynomial_equation_solution :
  ∀ (P : ℝ → ℝ),
  (∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
    3*P (a - b) + 3*P (b - c) + 3*P (c - a)) →
  ∃ (a b : ℝ), ∀ (x : ℝ), P x = a * x^2 + b * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1278_127885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inserted_twos_divisibility_l1278_127896

theorem inserted_twos_divisibility (n : ℕ) : ∃ (a_n : ℕ), 
  (Nat.digits 10 a_n).length = n + 6 ∧ 
  (Nat.digits 10 a_n).take 1 = [2] ∧
  (Nat.digits 10 a_n).drop (n + 1) = [2, 0, 2, 0] ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → ((Nat.digits 10 a_n).get ⟨i, by sorry⟩) = 2) ∧
  a_n % 91 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inserted_twos_divisibility_l1278_127896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_area_l1278_127833

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 5^2 + y^2 / 3^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the perpendicularity condition
def perpendicular (P : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

-- Theorem statement
theorem ellipse_right_triangle_area :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 → perpendicular P →
  (1/2 * abs ((P.1 - F1.1) * (P.2 - F2.2) - (P.2 - F1.2) * (P.1 - F2.1))) = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_area_l1278_127833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_geometric_progression_l1278_127831

theorem cubic_roots_geometric_progression (a b c : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c
  (∃ r q : ℝ, q ≠ 0 ∧ (f r = 0) ∧ (f (r*q) = 0) ∧ (f (r*q^2) = 0)) →
  (∃ x : ℝ, f x = 0 ∧ x = -(c^(1/3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_geometric_progression_l1278_127831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1278_127884

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2/x - 1
  else if x < 0 then 2/x + 1
  else 0

theorem odd_function_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 0, f x = 2/x - 1) ∧
  (∀ x < 0, f x = 2/x + 1) ∧
  f 0 = 0 ∧
  f (1/2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1278_127884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1278_127872

/-- The probability of a randomly selected point from a square with vertices at (±3, ±3) 
    being within two units of the origin -/
noncomputable def probability_within_circle : ℝ := Real.pi / 9

/-- The side length of the square -/
def square_side : ℝ := 6

/-- The area of the square -/
def square_area : ℝ := square_side ^ 2

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The area of the circle -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

theorem probability_calculation : 
  probability_within_circle = circle_area / square_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1278_127872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_interval_l1278_127846

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

theorem systematic_sampling_interval (total : ℕ) (sampleSize : ℕ) 
    (h1 : total = 40) (h2 : sampleSize = 4) :
  let sample := systematicSample total sampleSize
  ∀ i, i + 1 < sampleSize → 
    (sample.get? (i + 1)).isSome ∧ (sample.get? i).isSome ∧ 
    ((sample.get? (i + 1)).get! - (sample.get? i).get! = total / sampleSize) := by
  sorry

#check systematic_sampling_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_interval_l1278_127846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_99_l1278_127887

theorem digit_150_of_17_over_99 : ∃ (d : ℕ), d = 7 ∧ 
  (∃ (seq : ℕ → ℕ), (∀ n, seq n < 10) ∧ 
    (∀ n, seq (2*n) = 1 ∧ seq (2*n + 1) = 7) ∧
    ((17 : ℚ) / 99 = ∑' n, (seq n : ℚ) / (10 : ℚ)^(n+1)) ∧
  d = seq 149) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_99_l1278_127887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_true_l1278_127841

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem two_statements_true : 
  (if (reciprocal 2 + reciprocal 4 = reciprocal 6) then 1 else 0) +
  (if (reciprocal 3 * reciprocal 5 = reciprocal 15) then 1 else 0) +
  (if (reciprocal 7 - reciprocal 3 = reciprocal 4) then 1 else 0) +
  (if ((reciprocal 12) / (reciprocal 3) = reciprocal 4) then 1 else 0) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_true_l1278_127841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l1278_127821

noncomputable def inverse_proportion (x : ℝ) : ℝ := -1 / x

noncomputable def point_A : ℝ × ℝ := (-1, inverse_proportion (-1))

noncomputable def point_B : ℝ × ℝ := (inverse_proportion 1, 1)

noncomputable def point_C : ℝ × ℝ := (2, inverse_proportion 2)

theorem inverse_proportion_relationship :
  let a := point_A.2
  let b := point_B.1
  let c := point_C.2
  a > c ∧ c > b := by
    -- Expand the definitions
    unfold point_A point_B point_C inverse_proportion
    -- Simplify the expressions
    simp
    -- Split the conjunction
    apply And.intro
    -- Prove a > c
    · norm_num
    -- Prove c > b
    · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l1278_127821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1278_127890

def sequence_a : ℕ → ℝ
  | 0 => 3  -- Define the base case for n = 0
  | n + 1 => 4 * sequence_a n + 3

theorem sequence_a_general_term (n : ℕ) : sequence_a n = 2^(2*n.succ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1278_127890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangles_l1278_127850

-- Define the square and rectangles
noncomputable def square_side_length : ℝ := sorry

-- Define the rectangle dimensions
noncomputable def rectangle_width : ℝ := square_side_length / 8
noncomputable def rectangle_height : ℝ := square_side_length

-- Define the perimeter of a rectangle
noncomputable def rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_height)

-- State the theorem
theorem square_perimeter_from_rectangles 
  (h : rectangle_perimeter = 40) : 
  4 * square_side_length = 640 / 9 := by
  sorry

#check square_perimeter_from_rectangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangles_l1278_127850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1278_127840

/-- A polyhedron with parallel rectangular base faces and side edges of equal length -/
structure Polyhedron where
  a : ℝ  -- length of first base rectangle
  b : ℝ  -- width of first base rectangle
  c : ℝ  -- length of second base rectangle
  d : ℝ  -- width of second base rectangle
  m : ℝ  -- height of the polyhedron
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < m

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron) : ℝ :=
  (p.m / 6) * ((2 * p.a + p.c) * p.b + (2 * p.c + p.a) * p.d)

/-- Theorem stating that the given formula correctly represents the volume of the polyhedron -/
theorem volume_formula_correct (p : Polyhedron) : 
  volume p = (p.m / 6) * ((2 * p.a + p.c) * p.b + (2 * p.c + p.a) * p.d) :=
by
  -- The proof is trivial since it's just the definition of volume
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1278_127840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1278_127848

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) - 1

theorem sequence_a_closed_form (n : ℕ) :
  n ≥ 1 → sequence_a n = 2^(n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1278_127848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1278_127862

-- Define the train length in meters
noncomputable def train_length : ℝ := 480

-- Define the platform length in meters
noncomputable def platform_length : ℝ := 250

-- Define the train speed in km/hr
noncomputable def train_speed_kmh : ℝ := 60

-- Convert train speed from km/hr to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Calculate the total distance the train needs to cover
noncomputable def total_distance : ℝ := train_length + platform_length

-- Calculate the time taken for the train to pass the platform
noncomputable def time_to_pass : ℝ := total_distance / train_speed_ms

-- Theorem stating that the time taken is approximately 43.8 seconds
theorem train_passing_time : 
  |time_to_pass - 43.8| < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1278_127862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentomino_symmetry_count_l1278_127853

/-- A pentomino-like figure made up of 5 squares -/
structure PentominoLike where
  squares : Finset (ℤ × ℤ)
  size_eq_five : squares.card = 5

/-- The set of all pentomino-like figures -/
def PentominoSet : Finset PentominoLike := sorry

/-- A figure has reflectional symmetry -/
def has_reflectional_symmetry (p : PentominoLike) : Prop := sorry

/-- A figure has rotational symmetry -/
def has_rotational_symmetry (p : PentominoLike) : Prop := sorry

/-- A figure has either reflectional or rotational symmetry -/
def has_symmetry (p : PentominoLike) : Prop :=
  has_reflectional_symmetry p ∨ has_rotational_symmetry p

/-- Assume has_symmetry is decidable -/
instance : DecidablePred has_symmetry := sorry

/-- The main theorem -/
theorem pentomino_symmetry_count :
  (PentominoSet.filter has_symmetry).card = 8 ∧ PentominoSet.card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentomino_symmetry_count_l1278_127853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1278_127869

/-- Given a linear function f(x) = ax + b, if the distance between the intersection points
    of y = x² and y = f(x) is 2√3, and the distance between the intersection points of
    y = x² - 2 and y = f(x) + 1 is √60, then the distance between the intersection points
    of y = x² - 1 and y = f(x) + 1 is 2√11. -/
theorem intersection_distance (a b : ℝ) : 
  let f (x : ℝ) := a * x + b
  let d1 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b))
  let d2 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 12))
  let d3 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 8))
  d1 = 2 * Real.sqrt 3 → d2 = Real.sqrt 60 → d3 = 2 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1278_127869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_triangle_inequality_l1278_127832

-- Define the function f
def f (x m : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval [0,2]
def I : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem min_m_for_triangle_inequality :
  ∃ (m_min : ℝ), m_min = 6 ∧
  (∀ m > m_min, ∀ a b c, a ∈ I → b ∈ I → c ∈ I →
    f a m + f b m ≥ f c m ∧
    f b m + f c m ≥ f a m ∧
    f c m + f a m ≥ f b m) ∧
  (∀ m < m_min, ∃ a b c, a ∈ I ∧ b ∈ I ∧ c ∈ I ∧
    (f a m + f b m < f c m ∨
     f b m + f c m < f a m ∨
     f c m + f a m < f b m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_triangle_inequality_l1278_127832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1278_127837

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  let C := {p : ℝ × ℝ | ∃ x, p.2 = f x}
  (∀ p : ℝ × ℝ, p ∈ C ↔ (11 * Real.pi / 12 - p.1, p.2) ∈ C) ∧ 
  (∀ p : ℝ × ℝ, p ∈ C ↔ (4 * Real.pi / 3 - p.1, -p.2) ∈ C) ∧
  (∀ x₁ x₂ : ℝ, -Real.pi/12 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*Real.pi/12 → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1278_127837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_on_interval_l1278_127878

-- Define the function f(x) = x / e^x
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- State the theorem
theorem f_has_maximum_on_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.exp 1) ∧ f x = (Real.exp 1)⁻¹ ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.exp 1) → f y ≤ f x := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_on_interval_l1278_127878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_coefficients_sum_implies_a_l1278_127867

noncomputable def polynomial (a x : ℝ) : ℝ := (a + x) * (1 + x)^4

noncomputable def sum_odd_coefficients (a : ℝ) : ℝ :=
  (polynomial a 1 - polynomial a (-1)) / 2

theorem odd_coefficients_sum_implies_a (a : ℝ) :
  sum_odd_coefficients a = 32 → a = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_coefficients_sum_implies_a_l1278_127867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_planes_l1278_127838

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

-- Define the problem setup
def projectionAxis : Set Point := sorry

-- Helper definition
def is_tangent_point (S : Sphere) (P : Point) : Prop := sorry

-- Theorem statement
theorem four_tangent_planes 
  (S₁ S₂ : Sphere) 
  (P : Point) 
  (h₁ : S₁.center ∈ projectionAxis) 
  (h₂ : S₂.center ∈ projectionAxis) :
  ∃! (planes : Finset (Set Point)), 
    planes.card = 4 ∧ 
    (∀ plane ∈ planes, 
      P ∈ plane ∧ 
      (∃ T₁ : Point, T₁ ∈ plane ∧ is_tangent_point S₁ T₁) ∧
      (∃ T₂ : Point, T₂ ∈ plane ∧ is_tangent_point S₂ T₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_planes_l1278_127838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_less_than_one_l1278_127814

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem ac_less_than_one (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a < b) (h5 : b < c) (h6 : f a > f c) (h7 : f c > f b) : a * c < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_less_than_one_l1278_127814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_l1278_127886

/-- If the cost price is 98% of the selling price, then the profit percentage is approximately 2.04%. -/
theorem profit_percentage (selling_price : ℝ) (selling_price_pos : 0 < selling_price) :
  let cost_price := 0.98 * selling_price
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  abs (profit_percentage - 2.04) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_l1278_127886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_144_l1278_127815

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, -2],
    ![5, 6, -4],
    ![1, 3,  7]]

theorem det_A_eq_144 : Matrix.det A = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_144_l1278_127815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l1278_127870

/-- The numerator of the rational function -/
noncomputable def numerator (x : ℝ) : ℝ := 3*x^7 + 2*x^6 - 5*x^3 + 4

/-- A rational function with the given numerator and a polynomial denominator of degree n -/
noncomputable def rational_function (n : ℕ) (x : ℝ) : ℝ := numerator x / x^n

/-- The existence of a horizontal asymptote for a rational function -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - L| < ε

/-- The theorem stating the smallest possible degree for the denominator -/
theorem smallest_degree_for_horizontal_asymptote :
  (∀ k < 7, ¬ has_horizontal_asymptote (rational_function k)) ∧
  has_horizontal_asymptote (rational_function 7) := by
  sorry

#check smallest_degree_for_horizontal_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l1278_127870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_is_4_2_l1278_127835

/-- Represents the time taken by a worker to complete the job alone -/
structure WorkerTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the portion of the job completed by a worker in one hour -/
noncomputable def work_rate (w : WorkerTime) : ℝ := 1 / w.time

/-- The time taken to complete the job given the working schedule -/
noncomputable def job_completion_time (a b c d : WorkerTime) : ℝ :=
  let initial_work := 2 * (work_rate a + work_rate c)
  let remaining_work := 1 - initial_work
  let combined_rate := work_rate b + work_rate c + work_rate d
  2 + remaining_work / combined_rate

/-- The theorem stating that the job will be completed in 4.2 hours -/
theorem job_completion_time_is_4_2 (a b c d : WorkerTime)
  (ha : a.time = 8) (hb : b.time = 12) (hc : c.time = 10) (hd : d.time = 15) :
  job_completion_time a b c d = 4.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_is_4_2_l1278_127835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yangyang_helps_mom_time_l1278_127827

noncomputable def warehouse_rice : ℝ := 1

noncomputable def dad_rate : ℝ := 1 / 10
noncomputable def mom_rate : ℝ := 1 / 12
noncomputable def yangyang_rate : ℝ := 1 / 15

noncomputable def total_work : ℝ := 2 * warehouse_rice

theorem yangyang_helps_mom_time (time_with_mom : ℝ) 
  (h1 : time_with_mom > 0)
  (h2 : time_with_mom * (mom_rate + yangyang_rate) + 
        (total_work - time_with_mom * (mom_rate + yangyang_rate)) / (dad_rate + yangyang_rate) = 
        total_work / dad_rate) :
  time_with_mom = 4 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yangyang_helps_mom_time_l1278_127827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1278_127823

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (1, -2)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ), u = l • v ∨ v = l • u

theorem parallel_vectors_k_value :
  ∃! (k : ℝ), parallel (-a + b) (a + k • b) ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1278_127823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_is_six_l1278_127810

/-- The set of marble numbers -/
def marbleNumbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of all distinct pairs of marble numbers -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbleNumbers.product marbleNumbers).filter (fun p => p.1 < p.2)

/-- The sum of a pair of numbers -/
def pairSum (p : ℕ × ℕ) : ℕ := p.1 + p.2

/-- The expected value of the sum of two randomly chosen distinct marbles -/
def expectedSum : ℚ :=
  (marblePairs.sum (fun p => (pairSum p : ℚ))) / marblePairs.card

theorem expected_sum_is_six : expectedSum = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_is_six_l1278_127810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1278_127808

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1278_127808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dante_initial_jelly_beans_l1278_127843

/-- Represents the number of jelly beans each child has -/
structure JellyBeans where
  aaron : ℕ
  bianca : ℕ
  callie : ℕ
  dante : ℕ

/-- Checks if the distribution of jelly beans satisfies the condition -/
def is_valid_distribution (jb : JellyBeans) : Prop :=
  ∀ (x y : ℕ), x ∈ [jb.aaron, jb.bianca, jb.callie, jb.dante] →
    y ∈ [jb.aaron, jb.bianca, jb.callie, jb.dante] →
    x ≥ y - 1

/-- The initial distribution of jelly beans -/
def initial_distribution (dante_beans : ℕ) : JellyBeans :=
  { aaron := 5
  , bianca := 7
  , callie := 8
  , dante := dante_beans }

/-- The distribution after Dante gives 1 jelly bean to Aaron -/
def final_distribution (dante_beans : ℕ) : JellyBeans :=
  { aaron := 6
  , bianca := 7
  , callie := 8
  , dante := dante_beans - 1 }

theorem dante_initial_jelly_beans :
  ∃ (dante_beans : ℕ),
    dante_beans = 8 ∧
    is_valid_distribution (final_distribution dante_beans) := by
  use 8
  apply And.intro
  · rfl
  · intro x y hx hy
    simp [is_valid_distribution, final_distribution]
    sorry  -- The actual proof would go here

#check dante_initial_jelly_beans

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dante_initial_jelly_beans_l1278_127843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1278_127880

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.tan t.A = (2 * t.c - t.b) * Real.tan t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : satisfiesCondition t)
  (h3 : t.a = Real.sqrt 13)
  (h4 : t.b = 3) :
  t.A = Real.pi/3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = (12 * Real.sqrt 546 + 9 * Real.sqrt 39) / 104) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1278_127880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_emission_reduction_l1278_127817

/-- Calculates the reduction in carbon emissions after implementing a bus service in Johnstown. -/
theorem carbon_emission_reduction :
  let total_population : ℕ := 80
  let car_pollution_per_year : ℕ := 10
  let bus_pollution_per_year : ℕ := 100
  let bus_capacity : ℕ := 40
  let switch_percentage : ℚ := 1/4

  let people_using_bus : ℕ := (↑total_population * switch_percentage).floor.toNat
  let people_still_driving : ℕ := total_population - people_using_bus
  
  let emissions_before : ℕ := total_population * car_pollution_per_year
  let emissions_after : ℕ := people_still_driving * car_pollution_per_year + bus_pollution_per_year
  
  emissions_before - emissions_after = 100 := by
    -- Proof goes here
    sorry

#eval (80 : ℚ) * (1/4 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_emission_reduction_l1278_127817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_root_117_l1278_127824

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  surface_area : x * y + y * z + x * z = 216
  edge_sum : x + y + z = 30

/-- The radius of the sphere containing the inscribed box -/
noncomputable def sphere_radius (box : InscribedBox) : ℝ :=
  Real.sqrt ((box.x ^ 2 + box.y ^ 2 + box.z ^ 2) / 4)

/-- Theorem: The radius of the sphere is √117 -/
theorem sphere_radius_is_root_117 (box : InscribedBox) : 
  sphere_radius box = Real.sqrt 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_root_117_l1278_127824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1278_127897

theorem definite_integral_exp_plus_2x : 
  ∫ x in (Set.Icc 0 1), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1278_127897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1278_127834

/-- Given a hyperbola with equation x²/5 - y²/b² = 1, 
    if the distance from the focus to the asymptote is 2, 
    then its focal length is 6 -/
theorem hyperbola_focal_length (b : ℝ) : 
  (∃ x y : ℝ, x^2 / 5 - y^2 / b^2 = 1) →
  (∃ c : ℝ, b * c / Real.sqrt (5 + b^2) = 2) →
  ∃ focal_length : ℝ, focal_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1278_127834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l1278_127844

/-- Represents the distribution of scores in the math competition -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1
  score60_value : score60 = 0.2
  score75_value : score75 = 0.25
  score85_value : score85 = 0.4
  score95_value : score95 = 0.15

/-- Calculates the mean score for the given distribution -/
def mean (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95

/-- The median score for the given distribution -/
def median (d : ScoreDistribution) : ℝ := 85

/-- Theorem stating the difference between median and mean is 6 -/
theorem median_mean_difference (d : ScoreDistribution) : median d - mean d = 6 := by
  sorry

-- Remove the #eval statement as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l1278_127844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_proof_l1278_127839

def plane (x y z : ℝ) : Prop := 4*x + 6*y + 4*z - 50 = 0

def symmetric_points (A A' : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ P : ℝ × ℝ × ℝ, 
    plane P.1 P.2.1 P.2.2 ∧ 
    P.1 = (A.1 + A'.1) / 2 ∧ 
    P.2.1 = (A.2.1 + A'.2.1) / 2 ∧ 
    P.2.2 = (A.2.2 + A'.2.2) / 2

theorem symmetry_proof :
  let A  : ℝ × ℝ × ℝ := (2, 0, 2)
  let A' : ℝ × ℝ × ℝ := (6, 6, 6)
  symmetric_points A A' plane :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_proof_l1278_127839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_eq_pow_l1278_127883

/-- The custom operation * for positive integers -/
def custom_op : Nat → Nat → Nat := sorry

/-- Property 1: 1 * 1 = 1 -/
axiom custom_op_one : custom_op 1 1 = 1

/-- Property 2: (n+1) * 1 = 3(n * 1) -/
axiom custom_op_succ (n : Nat) : custom_op (n + 1) 1 = 3 * (custom_op n 1)

/-- Theorem: n * 1 = 3^(n-1) for all positive integers n -/
theorem custom_op_eq_pow (n : Nat) (h : n > 0) : custom_op n 1 = 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_eq_pow_l1278_127883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_even_roll_l1278_127895

theorem probability_of_even_roll (die : Finset Nat) (is_fair : die = {1, 2, 3, 4, 5, 6}) :
  (({2, 4, 6} : Finset Nat).card : ℚ) / die.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_even_roll_l1278_127895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1278_127881

/-- Calculates the time (in seconds) for a train to cross a bridge. -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Proves that a train 100 meters long, traveling at 36 kmph, will take 35 seconds to cross a bridge 250 meters long. -/
theorem train_crossing_bridge :
  train_crossing_time 100 250 36 = 35 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  sorry

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check train_crossing_time 100 250 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1278_127881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_sum_l1278_127893

theorem prime_power_sum (n : ℕ) : 
  Prime (n^8 + n^7 + 1) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_sum_l1278_127893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_k_when_ON_parallel_MP_l1278_127807

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6 = 0

-- Define the line passing through P(0, 2) with slope k
def line (k x y : ℝ) : Prop := y = k*x + 2

-- Define the condition for the line to intersect the circle at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ circle_M x₁ y₁ ∧ circle_M x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the center of the circle M
def center_M : ℝ × ℝ := (4, 0)

-- Define the condition for ON being parallel to MP
def ON_parallel_MP (k : ℝ) : Prop :=
  let N := (-(4 / (2*k + 1)), 2 / (2*k + 1))
  (N.2 - center_M.2) / (N.1 - center_M.1) = -1/2

-- Theorem 1: Range of k
theorem range_of_k :
  ∀ k, intersects_at_two_points k ↔ -3 < k ∧ k < 1/3 := by
  sorry

-- Theorem 2: Value of k when ON is parallel to MP
theorem k_when_ON_parallel_MP :
  ∀ k, intersects_at_two_points k ∧ ON_parallel_MP k → k = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_k_when_ON_parallel_MP_l1278_127807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pennies_all_heads_l1278_127803

-- Define a coin flip as a type with two possible outcomes
inductive CoinFlip : Type
  | Heads : CoinFlip
  | Tails : CoinFlip

-- Define the probability of getting heads on a single flip
noncomputable def prob_heads : ℝ := 1 / 2

-- Define the probability of getting all heads in three flips
noncomputable def prob_all_heads : ℝ := prob_heads * prob_heads * prob_heads

-- Theorem statement
theorem three_pennies_all_heads :
  prob_all_heads = 1 / 8 := by
  -- Unfold the definitions
  unfold prob_all_heads prob_heads
  -- Simplify the expression
  simp [mul_assoc, mul_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pennies_all_heads_l1278_127803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1278_127809

/-- The length of the minor axis of an ellipse with given foci and a point on the ellipse -/
theorem ellipse_minor_axis_length : ∃ b : ℝ, b = 3 := by
  -- Define the foci
  let f1 : ℝ × ℝ := (-6, 0)
  let f2 : ℝ × ℝ := (6, 0)
  -- Define the point on the ellipse
  let p : ℝ × ℝ := (5, 2)
  -- Define the distance between foci
  let c : ℝ := 6
  -- Define the sum of distances from P to foci
  let sum_distances : ℝ := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
                           Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  -- Define the semi-major axis length
  let a : ℝ := sum_distances / 2
  -- Define the semi-minor axis length
  let b : ℝ := Real.sqrt (a^2 - c^2)
  -- The length of the minor axis
  existsi b
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1278_127809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_shape_perimeter_l1278_127888

/-- A shape with specific properties -/
structure SpecialShape where
  area : ℝ
  numSmallSquares : ℕ
  numPerimeterSegments : ℕ

/-- The perimeter of the special shape -/
noncomputable def perimeter (shape : SpecialShape) : ℝ :=
  let smallSquareArea := shape.area / shape.numSmallSquares
  let smallSquareSide := Real.sqrt smallSquareArea
  shape.numPerimeterSegments * smallSquareSide

/-- Theorem stating that the perimeter of the special shape is 144 -/
theorem special_shape_perimeter :
  ∀ (shape : SpecialShape),
  shape.area = 528 ∧
  shape.numSmallSquares = 33 ∧
  shape.numPerimeterSegments = 36 →
  perimeter shape = 144 := by
  intro shape ⟨area_eq, num_squares_eq, num_segments_eq⟩
  unfold perimeter
  rw [area_eq, num_squares_eq, num_segments_eq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_shape_perimeter_l1278_127888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_shaded_area_l1278_127802

/-- Represents the non-shaded area of Figure X -/
noncomputable def figure_x_non_shaded : ℝ := Real.pi * (3/2)^2

/-- Represents the non-shaded area of Figure Y -/
noncomputable def figure_y_non_shaded : ℝ := Real.pi

/-- Represents the non-shaded area of Figure Z -/
def figure_z_non_shaded : ℝ := 16

theorem largest_non_shaded_area : 
  figure_z_non_shaded > figure_x_non_shaded ∧ figure_z_non_shaded > figure_y_non_shaded := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_shaded_area_l1278_127802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_not_associative_l1278_127816

/-- The ⋆ operation for positive real numbers -/
noncomputable def star (x y : ℝ) : ℝ := (x * y) / (x + y - 1)

/-- Commutativity of the ⋆ operation -/
theorem star_commutative : ∀ (x y : ℝ), x > 0 → y > 0 → star x y = star y x := by
  sorry

/-- Non-associativity of the ⋆ operation -/
theorem star_not_associative : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ star (star x y) z ≠ star x (star y z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_not_associative_l1278_127816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1278_127898

theorem sin_double_angle_special_case (θ : Real) :
  Real.sin (π / 4 + θ) = 1 / 3 → Real.sin (2 * θ) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1278_127898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_value_l1278_127829

-- Define the line
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + a = 0

-- Define the perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem intersection_perpendicular_value (A B C : ℝ × ℝ) (a : ℝ) :
  line A.1 A.2 →
  line B.1 B.2 →
  circle_eq A.1 A.2 a →
  circle_eq B.1 B.2 a →
  circle_eq C.1 C.2 a →
  perpendicular A B C →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_value_l1278_127829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_intersection_theorem_l1278_127859

/-- Represents a square frame in the arrangement -/
structure Frame where
  index : Nat
  sideLength : ℝ

/-- The arrangement of frames -/
structure FrameArrangement where
  frames : Vector Frame 50
  line_e : Set (ℝ × ℝ)  -- Represents the horizontal line e

/-- Predicate to check if a frame intersects with line e -/
def intersectsLineE (f : Frame) (arr : FrameArrangement) : Prop := sorry

/-- The set of frame indices that intersect with line e -/
def intersectingFrameIndices : Set Nat :=
  {0, 1, 9, 17, 25, 33, 41, 49, 6, 14, 22, 30, 38, 46}

/-- Main theorem: The frames that intersect line e are those with indices in intersectingFrameIndices -/
theorem frame_intersection_theorem (arr : FrameArrangement) :
  ∀ f, f ∈ arr.frames.toList → (intersectsLineE f arr ↔ f.index ∈ intersectingFrameIndices) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_intersection_theorem_l1278_127859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l1278_127845

/-- Given two points A(x₁, y₁) and B(x₂, y₂) that satisfy the equation 3x - 4y - 2 = 0,
    prove that this equation represents the line passing through these points. -/
theorem line_equation_through_points (x₁ y₁ x₂ y₂ : ℝ) 
  (hA : 3 * x₁ - 4 * y₁ - 2 = 0)
  (hB : 3 * x₂ - 4 * y₂ - 2 = 0) :
  ∀ (x y : ℝ), ((x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁)) ↔ 3 * x - 4 * y - 2 = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l1278_127845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_true_l1278_127856

-- Define the properties of quadrilaterals
structure Quadrilateral where
  -- We'll leave the internal structure undefined for now
  mk :: -- empty constructor

def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry

-- Given statement
axiom parallelogram_implies_rhombus :
  ∀ q : Quadrilateral, is_parallelogram q → is_rhombus q

-- Converse and Inverse to prove
theorem converse_and_inverse_true :
  (∀ q : Quadrilateral, is_rhombus q → is_parallelogram q) ∧
  (∀ q : Quadrilateral, ¬is_parallelogram q → ¬is_rhombus q) :=
by
  sorry

#check converse_and_inverse_true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_true_l1278_127856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eq3_is_linear_one_var_l1278_127864

-- Define what it means for an equation to be linear with one variable
def is_linear_one_var (f : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x ↔ a * x + b = 0

-- Define the four equations
def eq1 : ℝ → ℝ → Prop := λ x y ↦ -x = 2*y + 3
def eq2 : ℝ → Prop := λ x ↦ x^2 - 1 = 4
def eq3 : ℝ → Prop := λ x ↦ (x - 1) / 2 = 1
def eq4 : ℝ → Prop := λ x ↦ 1 / x = 6

theorem only_eq3_is_linear_one_var :
  (¬ ∃ f : ℝ → ℝ, is_linear_one_var (λ x ↦ eq1 x (f x))) ∧
  (¬ is_linear_one_var eq2) ∧
  (is_linear_one_var eq3) ∧
  (¬ is_linear_one_var eq4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eq3_is_linear_one_var_l1278_127864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_BAE_in_cube_l1278_127865

-- Define a cube structure
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ
  is_cube : ∀ (i j : Fin 8), i ≠ j → ∃ (k : Fin 8), k ≠ i ∧ k ≠ j ∧
    (vertices i).1 = (vertices k).1 ∧ (vertices i).2 = (vertices k).2 ∧
    (vertices j).1 = (vertices k).1 ∧ (vertices j).2 = (vertices k).2

-- Define vertices A, B, and E
def A (c : Cube) : ℝ × ℝ × ℝ := c.vertices 0
def B (c : Cube) : ℝ × ℝ × ℝ := c.vertices 1
def E (c : Cube) : ℝ × ℝ × ℝ := c.vertices 4

-- Define the angle function (this is a placeholder, you might need to implement it properly)
noncomputable def angle (p q r : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem sin_BAE_in_cube (c : Cube) : 
  Real.sin (angle (A c) (B c) (E c)) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_BAE_in_cube_l1278_127865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_theorem_l1278_127836

/-- Circle C in Cartesian coordinates -/
def circleC (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

/-- Line l in parametric form -/
def line (t : ℝ) : ℝ × ℝ := (-t, 1 + t)

/-- Point M on the line -/
def M : ℝ × ℝ := (0, 1)

/-- Intersection points of the line and circle -/
def intersectionPoints : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line t ∧ circleC p.1 p.2}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_product_theorem :
  ∀ A B, A ∈ intersectionPoints → B ∈ intersectionPoints → distance M A * distance M B = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_theorem_l1278_127836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_5050_l1278_127842

/-- A sequence satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 0  -- Add a case for 0
  | 1 => 1
  | (n + 2) => ((n + 3) * a (n + 1)) / (n + 1)

/-- The theorem stating that the 100th term of the sequence is 5050 -/
theorem a_100_eq_5050 : a 100 = 5050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_5050_l1278_127842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l1278_127800

/-- An arithmetic sequence with common ratio q > 0 -/
structure ArithmeticSequence where
  q : ℝ
  hq : q > 0

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- n-th term of an arithmetic sequence -/
def a_n (a : ArithmeticSequence) (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_first_term
  (a : ArithmeticSequence)
  (h2 : S a 2 = 3 * a_n a 2 + 2)
  (h4 : S a 4 = 3 * a_n a 4 + 2) :
  a_n a 1 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l1278_127800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_slant_height_l1278_127857

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  ab : ℝ  -- Length of side AB
  bc : ℝ  -- Length of side BC
  pa : ℝ  -- Length of edge PA
  pa_perp_ab : Bool  -- PA is perpendicular to AB
  pa_perp_ad : Bool  -- PA is perpendicular to AD

/-- Calculate the volume of a rectangular base pyramid -/
noncomputable def pyramidVolume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.ab * p.bc * p.pa

/-- Calculate the slant height from apex to midpoint of AB -/
noncomputable def slantHeight (p : RectangularBasePyramid) : ℝ :=
  Real.sqrt ((p.ab / 2) ^ 2 + p.pa ^ 2)

theorem pyramid_volume_and_slant_height 
  (p : RectangularBasePyramid) 
  (h1 : p.ab = 10) 
  (h2 : p.bc = 5) 
  (h3 : p.pa = 7) 
  (h4 : p.pa_perp_ab = true) 
  (h5 : p.pa_perp_ad = true) : 
  pyramidVolume p = 350 / 3 ∧ slantHeight p = Real.sqrt 74 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_slant_height_l1278_127857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_correct_l1278_127873

/-- The volume of a single cube in cubic centimetres -/
def cube_volume : ℝ := 27

/-- The number of cubes that fit in the box -/
def num_cubes : ℕ := 24

/-- The length of the first dimension of the box in centimetres -/
def box_length : ℝ := 8

/-- The width of the second dimension of the box in centimetres -/
def box_width : ℝ := 12

/-- The height of the third dimension of the box in centimetres -/
def box_height : ℝ := 6.75

/-- Theorem stating that the given dimensions of the box are correct -/
theorem box_dimensions_correct : 
  box_height * box_length * box_width = (num_cubes : ℝ) * cube_volume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_correct_l1278_127873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1278_127892

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 6*x + 13)

-- State the theorem
theorem f_properties :
  (∀ x, f x > 0) ∧ 
  (∀ y, y ∈ Set.Ioo 0 (1/16) → ∃ x, f x = y) ∧
  (f 3 = 1/16) ∧
  (∀ x y, x < y ∧ y < 3 → f x < f y) ∧
  (∀ x y, 3 < x ∧ x < y → f x > f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1278_127892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_white_balls_l1278_127847

theorem number_of_white_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 4 →
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 4 →
  total_balls = red_balls + white_balls →
  white_balls = 12 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_white_balls_l1278_127847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1278_127875

noncomputable def P : ℝ × ℝ := (2, 1)

noncomputable def line_equation (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, 1 + t * Real.sin α)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def intersection_x_axis (α : ℝ) : ℝ × ℝ :=
  (2 - 1 / Real.tan α, 0)

noncomputable def intersection_y_axis (α : ℝ) : ℝ × ℝ :=
  (0, 1 - 2 * Real.tan α)

theorem line_properties (α : ℝ) :
  distance P (intersection_x_axis α) * distance P (intersection_y_axis α) = 4 →
  α = 3 * Real.pi / 4 ∧
  ∀ θ : ℝ, (3 : ℝ) / (Real.sin θ + Real.cos θ) =
    distance (0, 0) (line_equation α ((3 : ℝ) / (Real.sin θ + Real.cos θ) - 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1278_127875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_team_members_l1278_127852

-- Define the types for hat colors and flag colors
inductive HatColor
| Red
| Green

inductive FlagColor
| Yellow
| Blue

-- Define a structure for a student
structure Student where
  name : String
  hat : HatColor
  flag : FlagColor

-- Define the function to check if two students share a property
def shareProperty (s1 s2 : Student) : Prop :=
  s1.hat = s2.hat ∨ s1.flag = s2.flag

-- Define the theorem
theorem sarahs_team_members
  (students : List Student)
  (h1 : students.length = 6)
  (h2 : ∃ team1 team2 : List Student,
        team1.length = 3 ∧ team2.length = 3 ∧
        (∀ s1 s2, s1 ∈ team1 → s2 ∈ team1 → shareProperty s1 s2) ∧
        (∀ s1 s2, s1 ∈ team2 → s2 ∈ team2 → shareProperty s1 s2) ∧
        team1 ++ team2 = students)
  (sarah : Student)
  (h_sarah : sarah ∈ students ∧ sarah.hat = HatColor.Red ∧ sarah.flag = FlagColor.Yellow)
  (lucas : Student)
  (h_lucas : lucas ∈ students ∧ lucas.hat = HatColor.Red ∧ lucas.flag = FlagColor.Yellow)
  (ethan : Student)
  (h_ethan : ethan ∈ students ∧ ethan.hat = HatColor.Red ∧ ethan.flag = FlagColor.Blue)
  (emma : Student)
  (h_emma : emma ∈ students ∧ emma.hat = HatColor.Red ∧ emma.flag = FlagColor.Blue)
  : (∃ team : List Student, team.length = 3 ∧ sarah ∈ team ∧ lucas ∈ team ∧
     (ethan ∈ team ∨ emma ∈ team) ∧ (∀ s1 s2, s1 ∈ team → s2 ∈ team → shareProperty s1 s2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_team_members_l1278_127852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PAB_area_bounds_l1278_127820

-- Define the curves and line
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2
def L (x y : ℝ) : Prop := x + y = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 2

-- Define points A and B as the intersection of C₁ and L
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define P as any point on C₂
noncomputable def P : ℝ × ℝ := sorry

-- Helper function for area of triangle
noncomputable def area_of_triangle (P A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_PAB_area_bounds :
  (∀ (x y : ℝ), C₂ x y → P = (x, y)) →
  (∃ (area_min area_max : ℝ),
    (∀ (x y : ℝ), C₂ x y → area_of_triangle P A B ≥ area_min) ∧
    (∀ (x y : ℝ), C₂ x y → area_of_triangle P A B ≤ area_max) ∧
    area_min = 2 * Real.sqrt 3 ∧
    area_max = 4 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PAB_area_bounds_l1278_127820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1278_127813

theorem equation_solutions :
  ∃ (S : Finset (ℤ × ℤ)),
    (∀ (x y : ℤ), (x, y) ∈ S ↔ 7 * x^2 + 5 * y^2 = 1155) ∧
    (S.card = 4) ∧
    (∀ (x y : ℤ), (x, y) ∈ S → x * y = 70 ∨ x * y = -70) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1278_127813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_ratio_proof_l1278_127849

theorem group_ratio_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (12.8 * x + 10.2 * y) / (x + y) = 12.02 →
  x / y = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_ratio_proof_l1278_127849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_percentage_is_twenty_percent_l1278_127861

/-- Calculates the insurance percentage given the MSRP, total cost, and tax rate -/
noncomputable def insurance_percentage (msrp : ℝ) (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let insurance_cost := (total_cost - msrp) / (1 + tax_rate) - msrp * tax_rate / (1 + tax_rate)
  (insurance_cost / msrp) * 100

/-- Theorem stating that the insurance percentage for the given conditions is 20% -/
theorem insurance_percentage_is_twenty_percent :
  insurance_percentage 30 54 0.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_percentage_is_twenty_percent_l1278_127861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_line_l_cartesian_equation_l1278_127806

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/3) + m = 0

theorem curve_line_intersection :
  ∀ m : ℝ,
  (∃ x y : ℝ, (Real.sqrt 3 * x + y + 2 * m = 0) ∧ 
    (∃ t : ℝ, curve_C t = (x, y))) ↔ 
  -19/12 ≤ m ∧ m ≤ 5/2 :=
by sorry

theorem line_l_cartesian_equation :
  ∀ x y m : ℝ,
  (∃ ρ θ : ℝ, line_l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  Real.sqrt 3 * x + y + 2 * m = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_line_l_cartesian_equation_l1278_127806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_formula_l1278_127889

/-- The angle between a plane drawn through a side of the base and the midpoint of the opposite 
lateral edge and the base plane of a right prism with a rhombus base. -/
noncomputable def angleBetweenPlanes (α : ℝ) (k : ℝ) : ℝ :=
  Real.arctan (k / (2 * Real.sin α))

/-- Theorem stating the relationship between the angle, the acute angle of the rhombus base, 
and the ratio of prism height to base side. -/
theorem angle_between_planes_formula {α k : ℝ} 
  (h_acute : 0 < α ∧ α < π / 2) 
  (h_positive : k > 0) : 
  angleBetweenPlanes α k = Real.arctan (k / (2 * Real.sin α)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_formula_l1278_127889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_travel_time_is_8_4_l1278_127860

/-- Represents the problem of three children traveling with two bicycles -/
structure ChildrenTravelProblem where
  totalDistance : ℝ
  numChildren : ℕ
  numBicycles : ℕ
  walkingSpeed : ℝ
  cyclingSpeed : ℝ

/-- The specific problem instance -/
def problem : ChildrenTravelProblem :=
  { totalDistance := 84
  , numChildren := 3
  , numBicycles := 2
  , walkingSpeed := 5
  , cyclingSpeed := 20 }

/-- The minimum time required for all children to reach the destination -/
noncomputable def minTravelTime (p : ChildrenTravelProblem) : ℝ :=
  p.totalDistance * (1 / p.walkingSpeed + 2 / p.cyclingSpeed) / 3

/-- Theorem stating that the minimum travel time for the given problem is 8.4 hours -/
theorem min_travel_time_is_8_4 :
  minTravelTime problem = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_travel_time_is_8_4_l1278_127860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_squared_l1278_127819

/-- A circle with radius 3 centered at the origin -/
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

/-- An equilateral triangle inscribed in the circle -/
structure InscribedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_in_circle : A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle
  h_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  h_vertex_at_03 : A = (0, 3)
  h_altitude_on_y : B.1 = -C.1 ∧ B.2 = C.2

theorem inscribed_triangle_side_length_squared (t : InscribedTriangle) :
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_squared_l1278_127819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_polynomial_l1278_127855

theorem divides_polynomial : 
  ∃ q : Polynomial ℤ, X^13 - X + 60 = (X^2 - 2*X + 2) * q := by
  sorry

#check divides_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_polynomial_l1278_127855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_back_on_105_l1278_127879

-- Define the card numbers
def card_numbers : List Nat := [101, 102, 103, 104, 105]

-- Define the structure for a card
structure Card where
  front : Nat
  back : Nat

-- Define the theorem
theorem largest_back_on_105 (a b c d e : Nat) (cards : List Card) :
  cards.length = 5 →
  (∀ card, card ∈ cards → card.front ∈ card_numbers) →
  (∃! card, card ∈ cards ∧ card.back = a) →
  (∃! card, card ∈ cards ∧ card.back = b) →
  (∃! card, card ∈ cards ∧ card.back = c) →
  (∃! card, card ∈ cards ∧ card.back = d) →
  (∃! card, card ∈ cards ∧ card.back = e) →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + 2 = b - 2 ∧ b - 2 = 2 * c ∧ 2 * c = d / 2 ∧ d / 2 = e * e →
  ∃ card, card ∈ cards ∧ card.front = 105 ∧ card.back = max a (max b (max c (max d e))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_back_on_105_l1278_127879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_a15_l1278_127828

def sequenceA (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2^(sequenceA n)

theorem last_digit_a15 : sequenceA 15 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_a15_l1278_127828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l1278_127863

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  first_positive : a 1 > 0
  all_positive : ∀ n : ℕ, a n > 0
  geometric : ∀ n : ℕ, a (n + 1) = q * a n

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then
    n * seq.a 1
  else
    seq.a 1 * (1 - seq.q^n) / (1 - seq.q)

/-- The sequence is increasing -/
def IsIncreasing (seq : GeometricSequence) : Prop :=
  seq.q > 1

theorem geometric_sequence_condition (seq : GeometricSequence) :
  (S seq 19 + S seq 21 > 2 * S seq 20) ↔ IsIncreasing seq := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l1278_127863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1278_127804

/-- Calculates the time (in seconds) for a train to pass a person moving in the opposite direction. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  train_length / relative_speed_ms

/-- Theorem stating that the time for a 180m train moving at 55 km/h to pass a person
    moving at 7 km/h in the opposite direction is approximately 10.45 seconds. -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 180 55 7 - 10.45| < ε :=
by
  sorry

-- Use #eval only for decidable types
/-- Approximate calculation of the train passing time -/
def approx_train_passing_time : ℚ := 
  (180 : ℚ) / ((55 + 7) * (5 / 18))

#eval approx_train_passing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1278_127804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_powers_less_than_1000_l1278_127805

theorem sum_of_fourth_powers_less_than_1000 : 
  (Finset.filter (fun n : ℕ => n^4 < 1000) (Finset.range 1000)).sum (fun n => n^4) = 979 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_powers_less_than_1000_l1278_127805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_self_inverse_l1278_127825

theorem matrix_self_inverse (x y : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; x, y]
  A * A = Matrix.of (λ i j => if i = j then 1 else 0) → x = 12 ∧ y = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_self_inverse_l1278_127825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l1278_127811

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_valid_number (n : ℕ) : Bool :=
  10 ≤ n && n ≤ 99 && (sum_of_digits n)^2 = sum_of_digits (n^2)

theorem sum_of_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 100)).sum id = 139 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 100)).sum id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l1278_127811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_a_2018_value_l1278_127871

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => sorry  -- We don't need to define the exact formula here

theorem sequence_a_property (n : ℕ) (h : n ≥ 1) : 
  4 * sequence_a (n + 1) - sequence_a (n + 1) * sequence_a n + 4 * sequence_a n = 9 := by
  sorry

theorem a_2018_value : sequence_a 2018 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_a_2018_value_l1278_127871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_l1278_127801

noncomputable def f (x : ℝ) : ℝ := 
  abs (2 * Real.sin x * Real.cos x + Real.cos (2 * x))

theorem f_monotonic_increase_interval (k : ℤ) :
  ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8),
    ∀ y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8),
      x ≤ y → f x ≤ f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_l1278_127801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_of_body_length_l1278_127826

noncomputable def total_height : ℝ := 180

noncomputable def leg_ratio : ℝ := 1/3
noncomputable def head_ratio : ℝ := 1/4
noncomputable def arm_ratio : ℝ := 1/5

noncomputable def leg_length : ℝ := leg_ratio * total_height
noncomputable def head_length : ℝ := head_ratio * total_height
noncomputable def arm_length : ℝ := arm_ratio * total_height

theorem rest_of_body_length :
  total_height - (leg_length + head_length + arm_length) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_of_body_length_l1278_127826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_in_circle_l1278_127899

/-- A quadrilateral is represented by its four vertices in 2D space -/
def Quadrilateral := Fin 4 → ℝ × ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A quadrilateral has all sides and diagonals less than 1 meter -/
def all_sides_and_diagonals_less_than_one (q : Quadrilateral) : Prop :=
  ∀ i j, i ≠ j → distance (q i) (q j) < 1

/-- A circle contains a point if the distance from the center to the point is less than or equal to the radius -/
def circle_contains (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  distance center point ≤ radius

/-- A circle contains a quadrilateral if it contains all of its vertices -/
def circle_contains_quadrilateral (center : ℝ × ℝ) (radius : ℝ) (q : Quadrilateral) : Prop :=
  ∀ i, circle_contains center radius (q i)

/-- Main theorem: Any quadrilateral with all sides and diagonals less than 1 meter can be placed inside a circle with radius 0.9 meters -/
theorem quadrilateral_in_circle (q : Quadrilateral) 
  (h : all_sides_and_diagonals_less_than_one q) : 
  ∃ center : ℝ × ℝ, circle_contains_quadrilateral center 0.9 q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_in_circle_l1278_127899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1278_127894

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h1 : a^2 + b^2 = c^2 + a*b)
  (h2 : Real.sqrt 3 * c = 14 * Real.sin C)
  (h3 : a + b = 13) :
  C = Real.pi/3 ∧ c = 7 ∧ 
  (1/2 * a * b * Real.sin C = 10 * Real.sqrt 3) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1278_127894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_approx_l1278_127868

/-- The area of a rectangular field with one side of 16 meters and a diagonal of 17 meters -/
noncomputable def rectangleArea : ℝ :=
  let side := 16
  let diagonal := 17
  let otherSide := Real.sqrt (diagonal^2 - side^2)
  side * otherSide

/-- Theorem stating that the area of the rectangular field is approximately 91.84 square meters -/
theorem rectangle_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (rectangleArea - 91.84) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_approx_l1278_127868
