import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l55_5595

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (4 * x - Real.pi / 4)

theorem amplitude_and_phase_shift :
  ∃ (A φ : ℝ),
    (∀ x, f x = A * Real.sin (4 * (x - φ))) ∧
    A = 3 ∧
    φ = Real.pi / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l55_5595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_m_value_l55_5581

/-- Given two vectors a and b in R³, prove that if they are perpendicular
    and have specific components, then the unknown component of b is -2. -/
theorem perpendicular_vectors_m_value (a b : Fin 3 → ℝ) (m : ℝ) :
  a = ![2, 3, 4] →
  b = ![-1, m, 2] →
  (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2) = 0 →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_m_value_l55_5581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_conditions_l55_5566

-- Define constants m and n
variable (m n : ℝ)

-- Define the function y
def y (x : ℝ) := 2 * x - 13

-- Define the property of direct proportionality
def is_directly_proportional (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * g x

-- Theorem statement
theorem y_satisfies_conditions :
  is_directly_proportional (λ x ↦ y x + m) (λ x ↦ x - n) ∧
  y (-1) = -15 ∧
  y 7 = 1 ∧
  ∃ k b : ℝ, ∀ x, y x = k * x + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_conditions_l55_5566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_harmonic_sum_bound_l55_5513

open Real

noncomputable def f (x : ℝ) := 1 - 1/x - log x

theorem f_nonpositive : ∀ x > 0, f x ≤ 0 := by
  sorry

theorem harmonic_sum_bound (n : ℕ) : 
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℝ)) - log (n : ℝ) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_harmonic_sum_bound_l55_5513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5508

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- State the theorem
theorem f_properties (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f a x > f a y) ∧
  (a > 0 → (∀ x y, x < y ∧ y < Real.log (1/a) → f a x > f a y) ∧
           (∀ x y, Real.log (1/a) < x ∧ x < y → f a x < f a y)) ∧
  (a > 0 → ∀ x, f a x > 2 * Real.log a + 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l55_5573

-- Define the circle O in polar coordinates
def circle_O (ρ θ : ℝ) : Prop :=
  ρ = Real.cos θ + Real.sin θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi/4) = Real.sqrt 2 / 2

-- Define the constraints on ρ and θ
def valid_polar (ρ θ : ℝ) : Prop :=
  ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi

-- Theorem stating the intersection point in polar coordinates
theorem intersection_point :
  ∃ (ρ θ : ℝ), 
    circle_O ρ θ ∧ 
    line_l ρ θ ∧ 
    valid_polar ρ θ ∧ 
    0 < θ ∧ θ < Real.pi ∧
    ρ = 1 ∧ θ = Real.pi/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l55_5573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l55_5530

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.A = Real.pi / 3 ∧ 
  ∀ x, (x = Real.sin t.B + Real.sin t.C) → 
       (Real.sqrt 3 / 2 < x ∧ x ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l55_5530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l55_5538

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = -6*x + 8*y - 18

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-3, 4)

/-- The given point -/
def given_point : ℝ × ℝ := (3, 10)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_from_center_to_point :
  distance circle_center given_point = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l55_5538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_angle_sin_sq_value_sum_of_fraction_parts_value_l55_5565

/-- Two externally tangent circles with radii 1 and 5 -/
structure TangentCircles where
  small_radius : ℝ
  large_radius : ℝ
  small_radius_eq : small_radius = 1
  large_radius_eq : large_radius = 5

/-- The angle formed after rolling the smaller circle around the larger one -/
noncomputable def roll_angle (tc : TangentCircles) : ℝ := 2 * Real.pi

/-- The sine squared of the angle formed by the initial point of tangency,
    the center of the larger circle, and the final position of the initial point of tangency -/
noncomputable def final_angle_sin_sq (tc : TangentCircles) : ℝ :=
  (tc.small_radius / tc.large_radius) ^ 2

/-- Theorem: The sine squared of the final angle is 1/25 -/
theorem final_angle_sin_sq_value (tc : TangentCircles) :
  final_angle_sin_sq tc = 1 / 25 := by
  sorry

/-- The sum of numerator and denominator in the fraction representation of final_angle_sin_sq -/
def sum_of_fraction_parts : ℕ := 26

/-- Theorem: The sum of the numerator and denominator is 26 -/
theorem sum_of_fraction_parts_value :
  sum_of_fraction_parts = 26 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_angle_sin_sq_value_sum_of_fraction_parts_value_l55_5565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_rope_length_approx_23_l55_5521

/-- The new length of a rope given the initial length and additional grazing area -/
noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  Real.sqrt (initial_length ^ 2 + additional_area / Real.pi)

/-- Theorem stating that the new rope length is approximately 23 meters -/
theorem new_rope_length_approx_23 :
  let initial_length := 16
  let additional_area := 858
  abs (new_rope_length initial_length additional_area - 23) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_rope_length_approx_23_l55_5521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_sphere_volume_minimum_l55_5576

theorem box_sphere_volume_minimum (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  (∃ x : ℝ, x > 0 ∧ x < b/2 ∧
   (∀ y : ℝ, y > 0 → y < b/2 →
    (a - 2*x)^2 + (b - 2*x)^2 + x^2 ≤ (a - 2*y)^2 + (b - 2*y)^2 + y^2)) →
  1 < a/b ∧ a/b < 5/4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_sphere_volume_minimum_l55_5576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l55_5564

theorem problem_1 : (2021 - Real.pi)^0 + |Real.sqrt 3 - 1| - (1/2)^(-1 : ℤ) + Real.sqrt 12 = 3 * Real.sqrt 3 - 2 := by
  sorry

theorem problem_2 : (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2) + Real.sqrt 6 * Real.sqrt (2/3) = 3 := by
  sorry

theorem problem_3 : (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - 6 * Real.sqrt (1/3) = 3 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l55_5564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_squared_minus_q_squared_l55_5519

theorem p_squared_minus_q_squared (P Q : ℝ) : 
  P = 3^(2000 : ℤ) + 3^(-(2000 : ℤ)) → 
  Q = 3^(2000 : ℤ) - 3^(-(2000 : ℤ)) → 
  P^2 - Q^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_squared_minus_q_squared_l55_5519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l55_5579

-- Define the function g(x) = x³ - 3x² + 2
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the function f(x) which is symmetric to g(x) about (1/2, 0)
def f (x : ℝ) : ℝ := (x - 1)^3 + 3*(1 - x)^2 - 2

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*(x^2 - 1)

-- Define the tangent line equation
def tangent_line (m : ℝ) (x : ℝ) : ℝ := f m + f_derivative m * (x - m)

theorem range_of_t :
  ∀ t : ℝ,
  (∃! m : ℝ, tangent_line m 1 = t) ↔ 
  (t < -3 ∨ t > -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l55_5579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l55_5560

def b : ℕ → ℕ
  | 0 => 0  -- Add a case for 0
  | 1 => 0  -- Add cases for 1 to 6
  | 2 => 0
  | 3 => 0
  | 4 => 0
  | 5 => 0
  | 6 => 0
  | 7 => 7
  | n+8 => 50 * b (n+7) + 2 * (n+8)

def is_multiple_of_55 (n : ℕ) : Prop := ∃ k, n = 55 * k

theorem least_multiple_of_55 : 
  (∀ n, 7 < n → n < 54 → ¬ is_multiple_of_55 (b n)) ∧ 
  is_multiple_of_55 (b 54) :=
by
  sorry

#eval b 54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l55_5560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_angle_triples_l55_5541

def ε : ℝ := 22.5

def is_valid_angle_triple (α β γ : ℝ) : Prop :=
  ∃ (k m n : ℕ), α = k * ε ∧ β = m * ε ∧ γ = n * ε ∧
  α + β + γ = 180 ∧
  0 < α ∧ α < 180 ∧ 0 < β ∧ β < 180 ∧ 0 < γ ∧ γ < 180

def angle_triples : List (ℝ × ℝ × ℝ) :=
  [(22.5, 22.5, 135), (22.5, 45, 112.5), (22.5, 67.5, 90),
   (45, 45, 90), (45, 67.5, 67.5)]

theorem valid_angle_triples :
  ∀ (t : ℝ × ℝ × ℝ), is_valid_angle_triple t.1 t.2.1 t.2.2 ↔ t ∈ angle_triples :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_angle_triples_l55_5541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_athletes_l55_5583

theorem stratified_sampling_female_athletes 
  (total_athletes : ℕ) 
  (male_athletes : ℕ) 
  (female_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = male_athletes + female_athletes)
  (h2 : total_athletes = 98)
  (h3 : male_athletes = 56)
  (h4 : female_athletes = 42)
  (h5 : sample_size = 28) :
  (female_athletes : ℚ) / (total_athletes : ℚ) * (sample_size : ℚ) = 12 := by
  sorry

#check stratified_sampling_female_athletes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_athletes_l55_5583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_7_l55_5504

/-- The circle with equation x^2 + y^2 - 4x - 4y - 1 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- The line with equation y = x + 2 -/
def line_eq (x y : ℝ) : Prop := y = x + 2

/-- The chord length cut by the circle on the line -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 7

/-- Theorem stating that the chord length cut by the given circle on the given line is 2√7 -/
theorem chord_length_is_2_sqrt_7 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_7_l55_5504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_theorem_l55_5506

/-- Triangle PQR with vertices P(1, 10), Q(4, 0), and R(10, 0) -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- Horizontal line y = t intersecting PQ at V and PR at W -/
structure IntersectionPoints (t : ℝ) (T : Triangle) where
  V : ℝ × ℝ
  W : ℝ × ℝ
  on_horizontal_line : V.2 = t ∧ W.2 = t
  V_on_PQ : ∃ l : ℝ, V = l • T.Q + (1 - l) • T.P
  W_on_PR : ∃ m : ℝ, W = m • T.R + (1 - m) • T.P

/-- Area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem horizontal_line_intersection_theorem (T : Triangle) (t : ℝ) 
  (h : T.P = (1, 10) ∧ T.Q = (4, 0) ∧ T.R = (10, 0)) :
  ∀ (I : IntersectionPoints t T), 
    triangle_area T.P I.V I.W = 18 → t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_theorem_l55_5506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_angle_range_l55_5532

-- Define the function
def f (x : ℝ) := x^2

-- Define the domain
def domain : Set ℝ := { x | -1/2 ≤ x ∧ x ≤ 1/2 }

-- Define the range of the inclination angle
def angle_range : Set ℝ := { α | (0 ≤ α ∧ α ≤ Real.pi/4) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi) }

-- Theorem statement
theorem tangent_inclination_angle_range :
  ∀ x ∈ domain, ∃ α ∈ angle_range,
    Real.tan α = (deriv f) x ∧
    (∀ β, Real.tan β = (deriv f) x → β ∈ angle_range) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_angle_range_l55_5532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l55_5593

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus -/
def LineThroughFocus (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ 
  ∃ (t : ℝ), (1 - t) • Focus + t • A = B

/-- The length of a chord -/
noncomputable def ChordLength (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_chord_length 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Parabola) 
  (hB : B ∈ Parabola) 
  (hLine : LineThroughFocus A B)
  (hSum : A.1 + B.1 = 4/3) :
  ChordLength A B = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l55_5593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l55_5591

-- Define the ellipse and its properties
def Ellipse (F₁ F₂ P : ℝ × ℝ) : Prop :=
  F₁ = (-1, 0) ∧ F₂ = (1, 0) ∧ 
  2 * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 
    Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

-- Define the condition for P being in the second quadrant
def SecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

-- Define the angle condition
noncomputable def AngleCondition (F₁ F₂ P : ℝ × ℝ) : Prop :=
  let v1 := (F₂.1 - F₁.1, F₂.2 - F₁.2)
  let v2 := (P.1 - F₁.1, P.2 - F₁.2)
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / 
    (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))) = 2 * Real.pi / 3

-- Theorem statement
theorem ellipse_properties 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : Ellipse F₁ F₂ P) 
  (h_quadrant : SecondQuadrant P) 
  (h_angle : AngleCondition F₁ F₂ P) :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ Ellipse F₁ F₂ (x, y)) ∧ 
  (1 / 2 * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) * P.2 = 3 * Real.sqrt 3 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l55_5591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_coverage_time_l55_5522

/-- Represents the fraction of the pond covered by lotus leaves after a given number of days -/
noncomputable def lotus_coverage (days : ℕ) : ℝ :=
  (1 / 2) ^ (20 - days)

theorem half_coverage_time : ∃ (d : ℕ), d = 19 ∧ lotus_coverage d = 1 / 2 := by
  use 19
  constructor
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_coverage_time_l55_5522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_dance_pairs_equality_l55_5578

/-- Represents the number of ways to pair r girls with r boys in town A -/
def A (n r : ℕ) : ℕ := (n.choose r) * (n.choose r) * r.factorial

/-- Represents the number of ways to pair r girls with r boys in town B -/
def B : ℕ → ℕ → ℕ
  | 0, _ => 0
  | _, 0 => 1
  | n+1, r+1 => (2 * (n+1) - (r+1)) * B n r + B n (r+1)

/-- 
Theorem stating that the number of ways to pair r girls with r boys 
is the same in both towns for all valid n and r
-/
theorem dance_pairs_equality (n r : ℕ) (h1 : n ≥ 1) (h2 : r ≥ 1) (h3 : r ≤ n) : 
  A n r = B n r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_dance_pairs_equality_l55_5578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l55_5534

noncomputable def clockwise_rotation (angle : ℝ) : ℝ := angle % 360

theorem rotation_equivalence (y : ℝ) : 
  y < 360 →
  clockwise_rotation 690 = (360 - y) →
  y = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l55_5534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l55_5549

/-- Conversion factor from miles to kilometers -/
noncomputable def miles_to_km : ℝ := 1.60934

/-- Length of the street in miles -/
noncomputable def street_length : ℝ := 1.11847

/-- Time taken to cross the street in minutes -/
noncomputable def time_taken : ℝ := 14.5

/-- Conversion factor from minutes to hours -/
noncomputable def minutes_to_hours : ℝ := 1 / 60

/-- Calculates the speed in km/h given distance in miles and time in minutes -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance * miles_to_km) / (time * minutes_to_hours)

/-- Theorem stating that the calculated speed is approximately 7.44 km/h -/
theorem speed_calculation :
  ∃ ε > 0, abs (calculate_speed street_length time_taken - 7.44) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l55_5549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_ratio_l55_5597

theorem polynomial_root_ratio (a b c d e : ℝ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x ∈ ({1, 2, 3, 4} : Set ℝ)) →
  d / e = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_ratio_l55_5597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l55_5537

/-- A pyramid with specific properties -/
structure SpecialPyramid where
  l : ℝ  -- Average length of lateral edge
  base_is_square : True  -- The base is a square
  two_faces_perpendicular : True  -- Two lateral faces are perpendicular to the base
  two_faces_inclined : True  -- Two lateral faces are inclined at 45° to the base

/-- The volume of the special pyramid -/
noncomputable def volume (p : SpecialPyramid) : ℝ :=
  (p.l^3 * Real.sqrt 2) / 12

/-- The total surface area of the special pyramid -/
noncomputable def total_surface_area (p : SpecialPyramid) : ℝ :=
  (p.l^2 * (2 + Real.sqrt 2)) / 2

/-- Theorem stating the volume and surface area of the special pyramid -/
theorem special_pyramid_properties (p : SpecialPyramid) :
  volume p = (p.l^3 * Real.sqrt 2) / 12 ∧
  total_surface_area p = (p.l^2 * (2 + Real.sqrt 2)) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l55_5537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_processing_weight_l55_5598

/-- The weight of a side of beef after processing, given its initial weight and the percentage lost during processing. -/
noncomputable def weight_after_processing (initial_weight : ℝ) (percent_lost : ℝ) : ℝ :=
  initial_weight * (1 - percent_lost / 100)

/-- Theorem stating that a side of beef weighing 400 pounds before processing
    and losing 40 percent of its weight during processing will weigh 240 pounds after processing. -/
theorem beef_processing_weight :
  weight_after_processing 400 40 = 240 := by
  -- Unfold the definition of weight_after_processing
  unfold weight_after_processing
  -- Simplify the arithmetic
  simp [mul_sub, mul_div, mul_one]
  -- Check that 400 * (1 - 40 / 100) = 240
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check weight_after_processing 400 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_processing_weight_l55_5598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5529

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / 2

-- State the theorem
theorem f_properties :
  (∀ α : ℝ, f (Real.sin α + Real.cos α) = Real.sin α * Real.cos α) →
  (∀ x : ℝ, x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) ∧
  f (Real.sin (π / 6)) = -3/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l55_5548

/-- The time taken for a train to cross a pole -/
noncomputable def train_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmh * 1000 / 3600)

/-- Theorem: The time taken for a train to cross a pole is approximately 8.99 seconds -/
theorem train_crossing_time_approx :
  let speed := (60 : ℝ)
  let length := (150 : ℝ)
  abs (train_crossing_time speed length - 8.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l55_5548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_line_segments_l55_5587

/-- A type representing a point on a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line segment between two points. -/
structure LineSegment where
  start : Point
  stop : Point

/-- A predicate that checks if three points are collinear. -/
def collinear (p q r : Point) : Prop := sorry

/-- A function that returns the ceiling of a real number. -/
def ceiling (x : ℝ) : ℤ := sorry

/-- The main theorem stating the minimum number of line segments required. -/
theorem min_line_segments (n : ℕ) (points : Fin n → Point) 
  (h_noncollinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (m : ℕ) (segments : Fin m → LineSegment),
    (∀ (i j : Fin n), ∃ (k : Fin n), 
      (∃ (s1 s2 : Fin m), 
        (segments s1).start = points k ∧ (segments s1).stop = points i ∧
        (segments s2).start = points k ∧ (segments s2).stop = points j)) ∧
    m = ceiling ((3 * n - 2) / 2) ∧
    (∀ (m' : ℕ), m' < m → 
      ¬∃ (segments' : Fin m' → LineSegment),
        (∀ (i j : Fin n), ∃ (k : Fin n), 
          (∃ (s1 s2 : Fin m'), 
            (segments' s1).start = points k ∧ (segments' s1).stop = points i ∧
            (segments' s2).start = points k ∧ (segments' s2).stop = points j))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_line_segments_l55_5587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_nineteen_l55_5592

theorem complex_expression_equals_nineteen :
  (0.027 : ℝ)^(-(1/3) : ℝ) - (-1/7 : ℝ)^(-(2 : ℝ)) + 256^(3/4 : ℝ) - 3^(-(1 : ℝ)) + (Real.sqrt 2 - 1)^(0 : ℝ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_nineteen_l55_5592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5503

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

theorem f_properties :
  (f (Real.pi / 12) = 1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 0) ∧
  (f 0 = 0) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 3 / 2) ∧
  (f (Real.pi / 3) = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l55_5577

noncomputable def f (x : ℝ) : ℝ := x + Real.log x / Real.log 10 - 3

theorem solution_exists_in_interval :
  ∃! x₀ : ℝ, x₀ > 0 ∧ f x₀ = 0 ∧ x₀ ∈ Set.Ioo 2 3 :=
by
  -- Assuming the following:
  have h1 : Continuous f := sorry
  have h2 : StrictMono f := sorry
  have h3 : f 2 < 0 := sorry
  have h4 : f 3 > 0 := sorry
  
  -- Proof sketch
  -- Use the Intermediate Value Theorem to show existence
  -- Use the strict monotonicity to show uniqueness
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l55_5577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_weekly_earnings_approx_7_l55_5545

/-- Lewis's total earnings over the harvest period -/
def total_earnings : ℚ := 133

/-- Number of weeks Lewis worked during the harvest -/
def num_weeks : ℕ := 19

/-- Lewis's weekly earnings -/
noncomputable def weekly_earnings : ℚ := total_earnings / num_weeks

/-- Theorem stating that Lewis's weekly earnings are approximately $7 -/
theorem lewis_weekly_earnings_approx_7 : 
  ∃ ε > 0, abs (weekly_earnings - 7) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_weekly_earnings_approx_7_l55_5545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_car_l55_5547

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h1 : total = 500)
  (h2 : bike_owners = 450)
  (h3 : car_owners = 200)
  (h4 : ∀ a, a < total → (a < bike_owners ∨ a < car_owners)) :
  bike_owners - (bike_owners + car_owners - total) = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_car_l55_5547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_one_sixth_l55_5586

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
def g (x : ℝ) : ℝ := x

-- Define the area function
noncomputable def area_between_curves : ℝ := ∫ x in (0)..(1), (f x - g x)

-- Theorem statement
theorem area_enclosed_is_one_sixth : area_between_curves = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_one_sixth_l55_5586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l55_5509

noncomputable section

theorem cos_beta_value (α β m : ℝ) 
  (h1 : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = m)
  (h2 : π < β ∧ β < 3*π/2) : 
  Real.cos β = -Real.sqrt (1 - m^2) := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l55_5509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_no_purchase_days_l55_5526

/-- Represents the number of days Vasya buys 9 marshmallows -/
def x : ℕ := sorry

/-- Represents the number of days Vasya buys 2 meat pies -/
def y : ℕ := sorry

/-- Represents the number of days Vasya buys 4 marshmallows and 1 meat pie -/
def z : ℕ := sorry

/-- Represents the number of days Vasya buys nothing -/
def w : ℕ := sorry

/-- The total number of school days -/
def total_days : ℕ := 15

/-- The total number of marshmallows bought -/
def total_marshmallows : ℕ := 30

/-- The total number of meat pies bought -/
def total_meat_pies : ℕ := 9

theorem vasya_no_purchase_days :
  x + y + z + w = total_days ∧
  9 * x + 4 * z = total_marshmallows ∧
  2 * y + z = total_meat_pies →
  w = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_no_purchase_days_l55_5526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_sum_l55_5569

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_dot_product_sum (a b c : V) 
  (sum_zero : a + b + c = 0)
  (norm_a : ‖a‖ = 1)
  (norm_b : ‖b‖ = 2)
  (norm_c : ‖c‖ = 2) :
  inner a b + inner b c + inner c a = -(9 : ℝ)/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_sum_l55_5569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l55_5570

/-- A parabola with vertex at the origin and focus at (0,2) -/
structure Parabola where
  vertex : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (0, 2)

/-- A point in the first quadrant -/
structure PointFirstQuadrant where
  x : ℝ
  y : ℝ
  x_pos : 0 < x
  y_pos : 0 < y

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_coordinates (p : Parabola) (P : PointFirstQuadrant) :
  P.x^2 = 8 * P.y ∧ distance (P.x, P.y) p.focus = 50 →
  P.x = 8 * Real.sqrt 6 ∧ P.y = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l55_5570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_exists_no_min_l55_5505

-- Define the domain M
def M : Set ℝ := {x | x < 1 ∨ x > 3}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

-- Theorem statement
theorem f_max_exists_no_min :
  ∃ (max_val : ℝ) (max_x : ℝ), max_x ∈ M ∧ 
    (∀ x ∈ M, f x ≤ max_val) ∧ 
    f max_x = max_val ∧
    ¬∃ (min_val : ℝ), ∀ x ∈ M, min_val ≤ f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_exists_no_min_l55_5505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_construction_l55_5517

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to reflect a point over a line segment
noncomputable def reflect_point (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a function to check if a point is on a circle
def on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem orthocenter_construction (t : Triangle) :
  let O := circumcenter t
  let O_a' := reflect_point O t.B t.C
  let O_b' := reflect_point O t.A t.C
  let O_c' := reflect_point O t.A t.B
  let M := orthocenter t
  (on_circle O 1 t.A ∧ on_circle O 1 t.B ∧ on_circle O 1 t.C) →
  (on_circle O_a' 1 M ∧ on_circle O_b' 1 M ∧ on_circle O_c' 1 M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_construction_l55_5517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_degree_function_integral_equation_l55_5524

/-- A first-degree function is a function of the form f(x) = ax + b where a and b are constants and a ≠ 0 -/
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The theorem stating that if f is a first-degree function satisfying the given equation, then f(x) = x - 1 -/
theorem first_degree_function_integral_equation
  (f : ℝ → ℝ)
  (h1 : FirstDegreeFunction f)
  (h2 : ∀ x, f x = x + 2 * ∫ (t : ℝ) in Set.Icc 0 1, f t) :
  ∀ x, f x = x - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_degree_function_integral_equation_l55_5524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_one_or_four_l55_5567

/-- Given a natural number, return the sum of squares of its digits -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 * d1 + d2 * d2 + d3 * d3

/-- The sequence defined by the problem -/
def a : ℕ → ℕ
  | 0 => 100  -- Arbitrary three-digit number, we choose 100 as the lower bound
  | n + 1 => sumOfSquaresOfDigits (a n)

theorem sequence_contains_one_or_four :
  ∃ k : ℕ, a k = 1 ∨ a k = 4 := by
  sorry

#check sequence_contains_one_or_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_one_or_four_l55_5567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l55_5546

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  D : ℝ × ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.A = Real.pi/3 ∧ t.b = 2 ∧ t.c = 3 ∧
  (1/2 * t.b * t.c * Real.sin t.A = 6 * Real.sqrt 3)

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_properties t) :
  t.a / Real.tan t.A + t.b / Real.sin t.B = Real.sqrt 21 ∧
  ∀ (AD : ℝ), AD ≥ 3 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l55_5546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l55_5511

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- Definition of line l1 -/
def l1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 2 * x - a * y - 1 = 0

/-- Definition of line l2 -/
def l2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x - y = 0

/-- Main theorem: If l1 is parallel to l2, then a = ±√2 -/
theorem parallel_lines_a_value (a : ℝ) :
  (∃ x y : ℝ, l1 a x y ∧ l2 a x y) →
  (∀ x1 y1 x2 y2 : ℝ, l1 a x1 y1 → l2 a x2 y2 → ∃ k : ℝ, y2 - y1 = k * (x2 - x1)) →
  a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l55_5511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l55_5514

theorem tan_period (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, Real.tan (ω * x) = Real.tan (ω * (x + π / 2))) → ω = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l55_5514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_tangent_planes_l55_5542

/-- A 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Vector3D

/-- A circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Vector3D

/-- A right circular cone -/
structure RightCircularCone where
  axis : Line3D
  base : Circle3D

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  normal : Vector3D
  point : Point3D

/-- The first principal plane -/
def firstPrincipalPlane : Plane3D := sorry

/-- The angle between two planes -/
noncomputable def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

/-- Tangent planes to a cone through a point -/
noncomputable def tangent_planes (cone : RightCircularCone) (p : Point3D) : (Plane3D × Plane3D) := sorry

/-- Theorem: Angle of inclination of tangent planes to a right circular cone -/
theorem angle_of_tangent_planes 
  (cone : RightCircularCone) 
  (L : Point3D) 
  (h_axis_perpendicular : cone.axis.direction.x * firstPrincipalPlane.normal.x + 
                          cone.axis.direction.y * firstPrincipalPlane.normal.y + 
                          cone.axis.direction.z * firstPrincipalPlane.normal.z = 0) :
  ∃ φ : ℝ, 
    let (p1, p2) := tangent_planes cone L
    φ = angle_between_planes p1 p2 ∧
    φ = sorry  -- Expression involving trihedral angle theorem
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_tangent_planes_l55_5542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_root_even_coefficient_l55_5525

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (x : ℚ) 
  (h1 : a ≠ 0) 
  (h2 : a * x^2 + b * x + c = 0) : 
  ∃ k ∈ ({a, b, c} : Set ℤ), Even k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_root_even_coefficient_l55_5525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_theorem_l55_5559

/-- Calculates the new alcohol percentage after adding water to a mixture -/
noncomputable def new_alcohol_percentage (original_volume : ℝ) (original_alcohol_percentage : ℝ) (added_water : ℝ) : ℝ :=
  let original_alcohol := original_volume * (original_alcohol_percentage / 100)
  let original_water := original_volume - original_alcohol
  let new_water := original_water + added_water
  let new_volume := original_alcohol + new_water
  (original_alcohol / new_volume) * 100

/-- Theorem: Given a 28-liter mixture with 35% alcohol, after adding 6 liters of water, 
    the new alcohol percentage is approximately 28.82% -/
theorem alcohol_mixture_theorem :
  let result := new_alcohol_percentage 28 35 6
  abs (result - 28.82) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_theorem_l55_5559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_determines_a_l55_5585

-- Define the point P
def P : ℝ × ℝ := (1, 3)

-- Define the line l: 4x + 3y + a = 0
def l (a : ℝ) (x y : ℝ) : Prop := 4 * x + 3 * y + a = 0

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_determines_a :
  ∀ a : ℝ, a > 0 →
  distance_point_to_line P.1 P.2 4 3 a = 3 →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_determines_a_l55_5585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_when_a_gt_3_l55_5523

/-- The function f(x) = x² + 8/x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 8/x

/-- Theorem stating that for a > 3, f(x) = f(a) has three distinct real solutions -/
theorem three_solutions_when_a_gt_3 (a : ℝ) (h : a > 3) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ = f a ∧ f x₂ = f a ∧ f x₃ = f a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_when_a_gt_3_l55_5523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l55_5555

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = 3/5) (h4 : Real.cos (α + β) = 5/13) : Real.cos β = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l55_5555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_edge_approx_7_l55_5551

-- Define the volumes of the original shapes
def cube1_volume : ℝ := 3^3
def cube2_volume : ℝ := 4^3
def cube3_volume : ℝ := 5^3
noncomputable def cylinder_volume : ℝ := Real.pi * 2^2 * 7
def prism_volume : ℝ := 2 * 3 * 6

-- Define the total volume
noncomputable def total_volume : ℝ := cube1_volume + cube2_volume + cube3_volume + cylinder_volume + prism_volume

-- Define the edge of the new cube
noncomputable def new_cube_edge : ℝ := total_volume^(1/3)

-- Theorem statement
theorem new_cube_edge_approx_7 : 
  |new_cube_edge - 7| < 0.01 := by
  sorry

#eval cube1_volume
#eval cube2_volume
#eval cube3_volume
#eval prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_edge_approx_7_l55_5551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_positions_sufficient_less_than_sixteen_positions_sufficient_l55_5556

/-- Represents a circular arrangement of natural numbers -/
structure CircularNumbers where
  numbers : List Nat
  distinct : numbers.Nodup
  max_count : numbers.length ≤ 100

/-- Represents a strategy for choosing positions to query -/
structure QueryStrategy where
  positions : List Nat
  valid : ∀ p ∈ positions, p > 0

/-- Predicate to check if a strategy can determine the exact count -/
def can_determine_count (s : QueryStrategy) (c : CircularNumbers) : Prop :=
  ∀ c' : CircularNumbers, 
    (∀ p ∈ s.positions, c.numbers[p % c.numbers.length]? = c'.numbers[p % c'.numbers.length]?) →
    c.numbers.length = c'.numbers.length

/-- Theorem stating that 17 positions are always sufficient -/
theorem seventeen_positions_sufficient :
  ∃ s : QueryStrategy, s.positions.length = 17 ∧ ∀ c : CircularNumbers, can_determine_count s c :=
sorry

/-- Theorem stating that fewer than 16 positions are sufficient -/
theorem less_than_sixteen_positions_sufficient :
  ∃ s : QueryStrategy, s.positions.length < 16 ∧ ∀ c : CircularNumbers, can_determine_count s c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_positions_sufficient_less_than_sixteen_positions_sufficient_l55_5556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5539

open Real

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := p * x - p / x - 2 * log x

noncomputable def g (x : ℝ) : ℝ := 2 * exp 1 / x

theorem f_properties (p : ℝ) :
  (∀ x > 0, HasDerivAt (f 2) (2 * x - 2) x) ∧
  (∀ x > 0, Monotone (f p) ↔ p ≥ 1) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), f p x > g x) ↔ p > (4 * exp 1) / (exp 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l55_5501

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_specific :
  let ρ : Real := 15
  let θ : Real := 5 * π / 6
  let φ : Real := π / 3
  spherical_to_rectangular ρ θ φ = (-45/4, -15 * Real.sqrt 3 / 4, 15/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l55_5501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_implies_a_b_values_f_upper_bound_implies_ab_max_l55_5515

noncomputable section

variable (a b x : ℝ)

def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (a * x + b) + x^2

def tangent_line (a b : ℝ) (x₀ : ℝ) (x : ℝ) : ℝ :=
  f a b x₀ + (deriv (f a b) x₀) * (x - x₀)

theorem tangent_condition_implies_a_b_values :
  a ≠ 0 → tangent_line a b 1 = id → a = -1 ∧ b = 2 := by sorry

theorem f_upper_bound_implies_ab_max :
  (∀ x, f a b x ≤ x^2 + x) → (a * b ≤ Real.exp 1 / 2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_implies_a_b_values_f_upper_bound_implies_ab_max_l55_5515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l55_5502

/-- The parabola C and hyperbola D in the Cartesian coordinate system -/
structure GeometricSystem where
  C : Set (ℝ × ℝ)  -- Parabola C
  D : Set (ℝ × ℝ)  -- Hyperbola D

/-- The conditions of the problem -/
def ProblemConditions (sys : GeometricSystem) : Prop :=
  -- The vertex of C is the center of D (which is the origin)
  (∀ x y, (x, y) ∈ sys.C → (x = 0 ∧ y = 0 → (0, 0) ∈ sys.D)) ∧
  -- The focus of C is the same as the focus of D
  (∃ f : ℝ × ℝ, (f ∈ sys.C ∧ f ∈ sys.D)) ∧
  -- The equation of D is y²/3 - x² = 1
  (∀ x y, (x, y) ∈ sys.D ↔ y^2/3 - x^2 = 1) ∧
  -- Point P(t,2) (t > 0) is a fixed point on C
  (∃ t : ℝ, t > 0 ∧ (t, 2) ∈ sys.C) ∧
  -- A and B are two moving points on C
  (∀ A B : ℝ × ℝ, A ∈ sys.C ∧ B ∈ sys.C) ∧
  -- PA ⊥ PB
  (∀ A B : ℝ × ℝ, A ∈ sys.C ∧ B ∈ sys.C →
    ∃ t : ℝ, t > 0 ∧ (t, 2) ∈ sys.C ∧
    ((A.1 - t) * (B.1 - t) + (A.2 - 2) * (B.2 - 2) = 0))

/-- The theorem to be proved -/
theorem parabola_and_line_theorem (sys : GeometricSystem) 
  (h : ProblemConditions sys) : 
  (∀ x y, (x, y) ∈ sys.C ↔ x^2 = 8*y) ∧ 
  (∀ A B : ℝ × ℝ, A ∈ sys.C ∧ B ∈ sys.C → 
    ∃ k m : ℝ, (B.2 - A.2) = k * (B.1 - A.1) ∧ 
               A.2 = k * A.1 + m ∧ 
               10 = k * (-4) + m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l55_5502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l55_5533

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b in a real inner product space V, where |a| = 3, |b| = 4, 
    and the cosine of the angle between a and b is 3/4, 
    prove that the projection of a onto b is equal to (9/16)b. -/
theorem projection_theorem (a b : V) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 4)
  (h3 : inner a b / (‖a‖ * ‖b‖) = 3/4) :
  (inner a b / ‖b‖^2) • b = (9/16 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l55_5533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_parabola_vertex_l55_5527

/-- The vertex coordinates of a parabola after shifting -/
noncomputable def shifted_vertex (a b c : ℝ) (h_x h_y : ℝ) : ℝ × ℝ :=
  let original_vertex := (- b / (2 * a), - (b^2 - 4*a*c) / (4*a))
  (original_vertex.1 - h_x, original_vertex.2 + h_y)

/-- Theorem: The vertex of y = x^2 + 2x after shifting 1 unit left and 2 units up is (-2, 1) -/
theorem shifted_parabola_vertex :
  shifted_vertex 1 2 0 1 2 = (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_parabola_vertex_l55_5527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_indecomposable_amount_l55_5554

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  (List.range (n + 1)).reverse.map (λ i => 3^i * 5^(n - i))

/-- Predicate to check if an amount is decomposable using given coin denominations -/
def is_decomposable (amount : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), coeffs.length = n + 1 ∧
    amount = (List.zip coeffs (coin_denominations n)).foldl (λ sum (coeff, denom) => sum + coeff * denom) 0

/-- The main theorem stating the largest indecomposable amount -/
theorem largest_indecomposable_amount (n : ℕ) :
  ¬(is_decomposable (5^(n+1) - 2 * 3^(n+1)) n) ∧
  ∀ m < 5^(n+1) - 2 * 3^(n+1), is_decomposable m n := by
  sorry

#check largest_indecomposable_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_indecomposable_amount_l55_5554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_l55_5552

noncomputable def original_function (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 4)

noncomputable def translated_function (x : ℝ) : ℝ := original_function (x + Real.pi / 2)

noncomputable def final_function (x : ℝ) : ℝ := translated_function (x / 2)

theorem transformation_proof :
  ∀ x : ℝ, final_function x = 2 * Real.sin (2 * x - Real.pi / 4) := by
  intro x
  -- Expand the definitions and simplify
  simp [final_function, translated_function, original_function]
  -- The rest of the proof would go here
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_l55_5552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l55_5520

-- Define the equation
def equation (x p q : ℝ) : Prop := x * abs x + p * x + q = 0

-- Theorem stating the properties of the equation
theorem equation_properties (p q : ℝ) :
  (∃ (S : Finset ℝ), (∀ x ∈ S, equation x p q) ∧ S.card ≤ 3) ∧
  (∃ x : ℝ, equation x p q) ∧
  (∃ p q : ℝ, p^2 - 4*q < 0 ∧ ∃ x : ℝ, equation x p q) ∧
  (∃ p q : ℝ, p < 0 ∧ q < 0 ∧ ¬(∃ (S : Finset ℝ), (∀ x ∈ S, equation x p q) ∧ S.card = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l55_5520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jewel_thief_case_l55_5535

-- Define the suspects
inductive Suspect : Type
| A | B | C | D

-- Define a function to represent whether a suspect is telling the truth
variable (is_truthful : Suspect → Prop)

-- Define a function to represent whether a suspect is the criminal
variable (is_criminal : Suspect → Prop)

-- State the theorem
theorem jewel_thief_case :
  -- Only one suspect is the criminal
  (∃! s : Suspect, is_criminal s) →
  -- Exactly two suspects are telling the truth
  (∃ s1 s2 : Suspect, s1 ≠ s2 ∧ is_truthful s1 ∧ is_truthful s2 ∧
    ∀ s, s ≠ s1 → s ≠ s2 → ¬is_truthful s) →
  -- A's statement
  (is_truthful Suspect.A ↔ (is_criminal Suspect.B ∨ is_criminal Suspect.C ∨ is_criminal Suspect.D)) →
  -- B's statement
  (is_truthful Suspect.B ↔ (¬is_criminal Suspect.B ∧ is_criminal Suspect.C)) →
  -- C's statement
  (is_truthful Suspect.C ↔ (is_criminal Suspect.A ∨ is_criminal Suspect.B)) →
  -- D's statement
  (is_truthful Suspect.D ↔ is_truthful Suspect.B) →
  -- Conclusion: B is the criminal
  is_criminal Suspect.B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jewel_thief_case_l55_5535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_inequality_bound_l55_5516

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/x - Real.log x + 1

-- State the theorems
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = f x → (x = 1 → y = m * (x - 1) + f 1) ∧ 
  (2 * x + y - 4 = 0 ↔ y = m * (x - 1) + f 1) := by
  sorry

theorem inequality_bound (x : ℝ) (h : x > 0) :
  f x < (1 + Real.exp (-1)) / Real.log (x + 1) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_inequality_bound_l55_5516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l55_5575

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

/-- The set B defined by the exponential inequality -/
def B : Set ℝ := {x | Real.exp ((x - 2) * Real.log 2) > 1}

/-- The theorem stating the intersection of A and the complement of B -/
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l55_5575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l55_5582

/-- Parametric equations of line l -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 - 2*t)

/-- Polar equation of circle C -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Length of chord AB -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 5 / 5

theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ₁ : ℝ, circle_C θ₁ = Real.sqrt ((A.1 - 1)^2 + A.2^2)) ∧
    (∃ θ₂ : ℝ, circle_C θ₂ = Real.sqrt ((B.1 - 1)^2 + B.2^2)) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l55_5582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinates_l55_5531

noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I)

theorem z_coordinates : z.re = 2 ∧ z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinates_l55_5531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l55_5500

/-- A power function passing through the point (2,8) -/
noncomputable def f : ℝ → ℝ :=
  fun x => x^(Real.log 8 / Real.log 2)

theorem power_function_through_point : 
  f 2 = 8 ∧ f 3 = 27 :=
by
  constructor
  · -- Prove f 2 = 8
    sorry
  · -- Prove f 3 = 27
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l55_5500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l55_5588

/-- Proves that given a square with side length s satisfying the semicircle equation,
    and a rectangle with breadth 14 cm and perimeter equal to the square's perimeter,
    the length of the rectangle is approximately 5.54 cm. -/
theorem rectangle_length_approximation (s : ℝ) (length : ℝ) : 
  (Real.pi * s / 2 + s = 25.13) →
  (4 * s = 2 * (length + 14)) →
  ∃ ε > 0, |length - 5.54| < ε := by
  sorry

#check rectangle_length_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l55_5588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_products_sum_reciprocal_odd_products_sum_reciprocal_products_6_sum_reciprocal_odd_products_1012_l55_5571

open BigOperators

theorem sum_reciprocal_products (n : ℕ) :
  (∑ k in Finset.range n, 1 / ((k + 1) * (k + 2) : ℚ)) = 1 - 1 / (n + 1 : ℚ) :=
sorry

theorem sum_reciprocal_odd_products (n : ℕ) :
  (∑ k in Finset.range n, 1 / ((2 * k + 1) * (2 * k + 3) : ℚ)) = (n / (2 * n + 1 : ℚ)) / 2 :=
sorry

-- Part 1
theorem sum_reciprocal_products_6 :
  (∑ k in Finset.range 6, 1 / ((k + 1) * (k + 2) : ℚ)) = 6 / 7 :=
sorry

-- Part 2
theorem sum_reciprocal_odd_products_1012 :
  (∑ k in Finset.range 1012, 1 / ((2 * k + 1) * (2 * k + 3) : ℚ)) = 1002 / 2005 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_products_sum_reciprocal_odd_products_sum_reciprocal_products_6_sum_reciprocal_odd_products_1012_l55_5571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_analysis_final_result_l55_5596

-- Define the equations
noncomputable def equation1 (c d x : ℝ) : ℝ := (x + c) * (x + d) * (x + 10) / ((x + 4) ^ 2)
noncomputable def equation2 (c d x : ℝ) : ℝ := (x + 3 * c) * (x + 5) * (x + 8) / ((x + d) * (x + 10))

-- State the theorem
theorem root_analysis (c d : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation1 c d x = 0 ∧ equation1 c d y = 0 ∧ equation1 c d z = 0 ∧
    ∀ w : ℝ, equation1 c d w = 0 → w = x ∨ w = y ∨ w = z) ∧
  (∃! x : ℝ, equation2 c d x = 0) →
  c = 4/3 ∧ d = 8 := by
  sorry

-- Define the result
def result (c d : ℝ) : ℝ := 100 * c + d

-- Theorem for the final result
theorem final_result (c d : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation1 c d x = 0 ∧ equation1 c d y = 0 ∧ equation1 c d z = 0 ∧
    ∀ w : ℝ, equation1 c d w = 0 → w = x ∨ w = y ∨ w = z) ∧
  (∃! x : ℝ, equation2 c d x = 0) →
  result c d = 141.33333333333334 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_analysis_final_result_l55_5596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_part_speed_l55_5528

/-- Calculates the speed for the last part of a trip given the following conditions:
  * The total distance traveled is 800 miles
  * The initial speed is 80 miles/hour for 6 hours
  * The second speed is 60 miles/hour for 4 hours
  * The final part of the trip lasts 2 hours
-/
theorem last_part_speed (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ)
  (second_speed : ℝ) (second_time : ℝ) (final_time : ℝ) :
  total_distance = 800 →
  initial_speed = 80 →
  initial_time = 6 →
  second_speed = 60 →
  second_time = 4 →
  final_time = 2 →
  (total_distance - (initial_speed * initial_time + second_speed * second_time)) / final_time = 40 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

#check last_part_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_part_speed_l55_5528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l55_5584

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l55_5584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l55_5550

noncomputable section

/-- The hyperbola with equation x²/3 - y² = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 3 - p.2^2 = 1}

/-- The foci of the hyperbola -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-2, 0), (2, 0))

/-- The area of the triangle formed by a point and the foci -/
def TriangleArea (p : ℝ × ℝ) : ℝ :=
  abs (p.2 * 4) / 2

/-- The dot product of vectors PF₁ and PF₂ -/
def DotProduct (p : ℝ × ℝ) : ℝ :=
  let f₁ := Foci.1
  let f₂ := Foci.2
  (f₁.1 - p.1) * (f₂.1 - p.1) + (f₁.2 - p.2) * (f₂.2 - p.2)

theorem hyperbola_dot_product :
  ∀ p ∈ Hyperbola, TriangleArea p = 2 → DotProduct p = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l55_5550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_brick_problem_l55_5590

theorem student_brick_problem (a c : ℕ) (b : ℝ) (h : a > 0 ∧ c > 0 ∧ b > 0) :
  let efficiency := c / (a * b)
  let time := a / (c * efficiency)
  time = a^2 * b / c^2 := by
  sorry

#check student_brick_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_brick_problem_l55_5590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_points_l55_5574

/-- A parabola passing through two given points -/
def Parabola (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1^2 - b * p.1 + c}

theorem parabola_through_points (a b c : ℝ) :
  (1, 2) ∈ Parabola a b c ∧ (2, 3) ∈ Parabola a b c →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ ((-1 : ℝ), 5) ∉ Parabola a b c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_points_l55_5574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_l55_5561

/-- The polynomial function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2

/-- Theorem stating that the equation f(x) = 0 has no real solutions -/
theorem no_real_roots : ∀ x : ℝ, f x ≠ 0 := by
  intro x
  -- The proof would go here, but we'll use sorry for now
  sorry

#check no_real_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_l55_5561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l55_5594

-- Define t(x)
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)

-- Define f(x)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

-- Theorem statement
theorem t_of_f_3 : t (f 3) = Real.sqrt 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l55_5594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_distance_l55_5540

/-- Represents the cyclist's journey --/
structure Journey where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The original journey --/
def original : Journey := {
  speed := 0,
  time := 0,
  distance := 0
}

/-- The journey if the cyclist went 1 mph faster --/
def faster : Journey := {
  speed := 0,
  time := 0,
  distance := 0
}

/-- The journey if the cyclist went 1 mph slower --/
def slower : Journey := {
  speed := 0,
  time := 0,
  distance := 0
}

/-- The cyclist travels at a uniform speed --/
axiom uniform_speed : original.distance = original.speed * original.time

/-- If she cycled 1 mph faster, she would complete the journey in 3/4 of the time --/
axiom faster_condition : faster.speed = original.speed + 1 ∧
                         faster.time = (3/4) * original.time ∧
                         faster.distance = original.distance

/-- If she cycled 1 mph slower, she would spend an additional 3 hours on the journey --/
axiom slower_condition : slower.speed = original.speed - 1 ∧
                         slower.time = original.time + 3 ∧
                         slower.distance = original.distance

/-- The original distance cycled is 18 miles --/
theorem original_distance : original.distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_distance_l55_5540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l55_5553

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^3
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → g a x > g a y) →
  (∀ x y : ℝ, x < y → f a x > f a y) ∧
  ¬(∀ x y : ℝ, x < y → f a x > f a y → g a x > g a y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l55_5553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l55_5510

/-- The function f(x) = cos(2x + π/3) is symmetric about the point (-5π/12, 0) -/
theorem function_symmetry (x : ℝ) : 
  Real.cos (2 * (x + 5 * Real.pi / 12) + Real.pi / 3) = Real.cos (2 * ((-5 * Real.pi / 12) - x) + Real.pi / 3) := by
  sorry

#check function_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l55_5510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_equilateral_triangle_l55_5557

/-- The diameter of a circle inscribed in an equilateral triangle with side length 10 units -/
noncomputable def inscribed_circle_diameter (side_length : ℝ) : ℝ :=
  (10 * Real.sqrt 3) / 3

/-- Theorem stating that the diameter of the inscribed circle in an equilateral triangle
    with side length 10 is equal to 10√3/3 -/
theorem inscribed_circle_diameter_equilateral_triangle :
  inscribed_circle_diameter 10 = (10 * Real.sqrt 3) / 3 := by
  -- Unfold the definition of inscribed_circle_diameter
  unfold inscribed_circle_diameter
  -- The equality now holds by reflexivity
  rfl

#check inscribed_circle_diameter_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_equilateral_triangle_l55_5557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_reciprocal_sum_bound_l55_5543

/-- IsSymmetric predicate to represent that a circle is symmetric about a line -/
def IsSymmetric (circle : (ℝ × ℝ) → Prop) (line : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (center : ℝ × ℝ), circle center ∧ line center

/-- Given a circle and a line, if the circle is symmetric about the line, 
    then the sum of reciprocals of the line's coefficients has a lower bound. -/
theorem circle_symmetry_reciprocal_sum_bound 
  (x y : ℝ) (a b : ℝ) 
  (h_circle : x^2 + y^2 + 2*x - 4*y + 1 = 0)
  (h_line : 2*a*x - b*y + 2 = 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_symmetry : IsSymmetric (λ p : ℝ × ℝ => p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0) 
                            (λ p : ℝ × ℝ => 2*a*p.1 - b*p.2 + 2 = 0)) :
  1/a + 4/b ≥ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_reciprocal_sum_bound_l55_5543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5599

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi/3)

theorem f_properties :
  (∀ x, f (x - 2*Real.pi) = f x) ∧ 
  (∀ x, f (8*Real.pi/3 + x) = f (8*Real.pi/3 - x)) ∧
  (f (Real.pi/6 + Real.pi) = 0) ∧
  (¬ ∀ x ∈ Set.Ioo (Real.pi/2) Real.pi, ∀ y ∈ Set.Ioo (Real.pi/2) Real.pi, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l55_5599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l55_5562

-- Define the function f(x) = sin²(2x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

-- State the theorem about the minimum positive period of f(x)
theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l55_5562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l55_5580

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) :
  t.b = Real.sqrt 3 →
  Real.cos (2 * t.B) - Real.cos (2 * t.A) = 2 * Real.sin t.C * (Real.sin t.A - Real.sin t.C) →
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 →
  t.A + t.B + t.C = Real.pi →
  t.B = Real.pi / 3 ∧
  Real.sqrt 3 ≤ 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l55_5580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_division_property_l55_5536

theorem three_digit_division_property (a b c : Nat) : 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 →
  (100 * a + 10 * b + c) / (10 * b + c) = 9 ∧ 
  (100 * a + 10 * b + c) % (10 * b + c) = 8 →
  (100 * a + 10 * b + c = 224) ∨ 
  (100 * a + 10 * b + c = 449) ∨ 
  (100 * a + 10 * b + c = 674) ∨ 
  (100 * a + 10 * b + c = 899) := by
  sorry

#check three_digit_division_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_division_property_l55_5536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l55_5544

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 1 x) * x - (oplus 2 x)

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 6 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ m := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l55_5544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_1_h_is_minimum_of_f_no_m_n_exist_l55_5568

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/9)^x - 2*a*(1/3)^x + 3

def domain : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

noncomputable def h (a : ℝ) : ℝ := 
  if a ≤ 1/3 then 28/9 - 2*a/3
  else if a < 3 then 3 - a^2
  else 12 - 6*a

theorem range_of_f_when_a_is_1 :
  Set.range (fun x => f 1 x) = { y : ℝ | 2 ≤ y ∧ y ≤ 6 } := by
  sorry

theorem h_is_minimum_of_f :
  ∀ a x, x ∈ domain → f a x ≥ h a := by
  sorry

theorem no_m_n_exist :
  ¬ ∃ m n : ℝ, 
    m > n ∧ 
    n > 3 ∧ 
    Set.range (fun a => h a) = { y : ℝ | n^2 ≤ y ∧ y ≤ m^2 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_1_h_is_minimum_of_f_no_m_n_exist_l55_5568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_equals_one_l55_5507

noncomputable def f (x m : ℝ) : ℝ := 
  2 * (Real.sin x ^ 4 + Real.cos x ^ 4) + m * (Real.sin x + Real.cos x) ^ 4

theorem max_value_implies_m_equals_one :
  (∃ (max_val : ℝ), max_val = 5 ∧ 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≤ max_val)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_equals_one_l55_5507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l55_5518

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/3)

/-- The period of the function -/
noncomputable def period : ℝ := 2 * Real.pi

/-- Theorem stating that f is periodic with period 'period' -/
theorem f_period : ∀ x, f (x + period) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l55_5518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l55_5558

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  isTriangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A) :
  t.A = Real.pi / 6 ∧ 
  Set.Icc (-(Real.sqrt 3 + 2) / 2) (Real.sqrt 3 - 1) (Real.cos (5 * Real.pi / 2 - t.B) - 2 * (Real.sin (t.C / 2)) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l55_5558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l55_5563

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → Real.log x > 0) ↔ (x ≤ 1 → Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l55_5563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l55_5512

noncomputable def circle1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 10*y + 9 = 0

noncomputable def circle2 (x y : ℝ) : Prop := x^2 + 8*x + y^2 - 2*y + 16 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem shortest_distance_between_circles :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x2 y2 ∧
    ∀ (x3 y3 x4 y4 : ℝ),
      circle1 x3 y3 → circle2 x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4 ∧
      distance x1 y1 x2 y2 = Real.sqrt 85 - 6 := by
  sorry

#check shortest_distance_between_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l55_5512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l55_5589

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a^x + b

-- State the theorem
theorem function_properties (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a b 0 = -3) (h4 : f a b 2 = 0) :
  a = 2 ∧ b = -4 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 12) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≥ -15/4) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a b x = 12) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a b x = -15/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l55_5589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l55_5572

theorem sin_beta_value (a β : ℝ) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos a = 4/5) (h4 : Real.cos (a + β) = 5/13) : Real.sin β = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l55_5572
