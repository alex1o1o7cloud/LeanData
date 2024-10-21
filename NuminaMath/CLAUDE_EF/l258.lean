import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l258_25850

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x + 13) / 12

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l258_25850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_simplest_l258_25816

/-- Definition of a simplest square root -/
def is_simplest_sqrt (x : ℝ) : Prop :=
  x > 0 ∧ 
  (∀ a b : ℕ, x ≠ (a : ℝ) / b) ∧ 
  (∀ y : ℕ, y > 1 → ¬∃ (z : ℕ), x = ↑(y * y * z))

/-- The given square roots -/
noncomputable def sqrt_options : List ℝ := [Real.sqrt 0.2, Real.sqrt 12, Real.sqrt 3, Real.sqrt 18]

/-- Theorem: √3 is the simplest square root among the given options -/
theorem sqrt_3_is_simplest : 
  ∃! x, x ∈ sqrt_options ∧ is_simplest_sqrt x ∧ x = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_simplest_l258_25816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l258_25885

/-- The time taken for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem stating that the time taken for the given train to cross the bridge is approximately 30 seconds -/
theorem train_bridge_crossing_time :
  let train_length : ℝ := 140
  let train_speed_kmh : ℝ := 45
  let bridge_length : ℝ := 235.03
  let crossing_time := time_to_cross_bridge train_length train_speed_kmh bridge_length
  ∃ ε > 0, |crossing_time - 30| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l258_25885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_angle_parabola_intersection_distance_l258_25858

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the focus and directrix intersection point
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)
def directrix_intersection (p : ℝ) : ℝ × ℝ := (-p/2, 0)

-- Define a line through a point
def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity of two lines
def perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

theorem parabola_tangent_angle (p : ℝ) (h : p > 0) :
  ∀ M : ℝ × ℝ,
  parabola p M.1 M.2 →
  (∃ l : ℝ × ℝ → ℝ × ℝ → Prop, l M (directrix_intersection p) ∧
    (∀ P : ℝ × ℝ, P ≠ M → parabola p P.1 P.2 → ¬l P (directrix_intersection p))) →
  angle M (directrix_intersection p) (focus p) = π/4 := by sorry

theorem parabola_intersection_distance :
  ∀ A B : ℝ × ℝ,
  parabola 6 A.1 A.2 →
  parabola 6 B.1 B.2 →
  (∃ l : ℝ × ℝ → ℝ × ℝ → Prop, l A (directrix_intersection 6) ∧ l B (directrix_intersection 6)) →
  perpendicular A B B (focus 6) →
  distance A (focus 6) - distance B (focus 6) = 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_angle_parabola_intersection_distance_l258_25858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l258_25863

-- Define the circle
def our_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line AB
def lineAB (a b x y : ℝ) : Prop := a * x + (a + b) * y - a * b = 0

-- Define the tangency condition
def isTangent (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), our_circle x y ∧ lineAB a b x y ∧ 
  ∀ (x' y' : ℝ), lineAB a b x' y' → (x' - x)^2 + (y' - y)^2 > 0

-- State the theorem
theorem min_distance_AB (a b : ℝ) (ha : a > 0) (hb : b > 0) (ht : isTangent a b) :
  (a + b)^2 + a^2 ≥ (2 + 2 * Real.sqrt 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l258_25863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_l258_25865

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ (3 : ℝ)^x ∧ (3 : ℝ)^x < 9}
def B : Set ℝ := {x | x ≥ 1}
def C (a : ℝ) : Set ℝ := {x | x + a > 0}

-- Theorem for part (I)
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : B ∩ C a = B → a > -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_l258_25865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_ending_l258_25855

theorem two_zeros_ending (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_ending_l258_25855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_value_triangle_area_l258_25883

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  A := Real.pi / 4
  B := sorry -- Not given, will be determined by other conditions
  C := sorry -- Not given, will be determined by other conditions
  a := 5
  b := sorry -- Not given, will be determined by other conditions
  c := sorry -- Not given, will be determined by other conditions

-- Theorem for part I
theorem tan_C_value (t : Triangle) (h1 : t.A = Real.pi / 4) (h2 : t.c / t.b = 3 * Real.sqrt 2 / 7) :
  Real.tan t.C = 3 / 4 := by
  sorry

-- Theorem for part II
theorem triangle_area (t : Triangle) (h1 : t.A = Real.pi / 4) (h2 : t.c / t.b = 3 * Real.sqrt 2 / 7) (h3 : t.a = 5) :
  (1 / 2) * t.b * t.c * Real.sin t.A = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_value_triangle_area_l258_25883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_divisibility_l258_25886

theorem collinear_points_divisibility (p : ℕ) (hp : Nat.Prime p) 
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ) 
  (h₁ : ∃ k₁ : ℤ, x₁ * y₁ - 1 = k₁ * p) 
  (h₂ : ∃ k₂ : ℤ, x₂ * y₂ - 1 = k₂ * p) 
  (h₃ : ∃ k₃ : ℤ, x₃ * y₃ - 1 = k₃ * p)
  (hcollinear : ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a * x₁ + b * y₁ = c ∧ a * x₂ + b * y₂ = c ∧ a * x₃ + b * y₃ = c) :
  ∃ (i j : Fin 3), i ≠ j ∧ 
    (∃ m : ℤ, (x₁ - x₂) * (if i = 0 ∧ j = 1 then 1 else if i = 0 ∧ j = 2 then -1 else if i = 1 ∧ j = 2 then 1 else 0) = m * p) ∧
    (∃ n : ℤ, (y₁ - y₂) * (if i = 0 ∧ j = 1 then 1 else if i = 0 ∧ j = 2 then -1 else if i = 1 ∧ j = 2 then 1 else 0) = n * p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_divisibility_l258_25886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l258_25899

/-- A line passing through the origin with an inclination angle of 60° -/
noncomputable def line (x : ℝ) : ℝ := Real.sqrt 3 * x

/-- The circle equation x² + y² - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The radius of the circle -/
def radius : ℝ := 2

/-- The length of the chord cut by the line on the circle -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

/-- Theorem stating the radius of the circle and the length of the chord -/
theorem circle_and_line_intersection :
  radius = 2 ∧ chord_length = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l258_25899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l258_25818

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: For a trapezium with parallel sides of 20 cm and 18 cm, and an area of 285 square centimeters, the distance between the parallel sides is 15 cm. -/
theorem trapezium_height_calculation :
  let a : ℝ := 20
  let b : ℝ := 18
  let area : ℝ := 285
  ∃ h : ℝ, trapezium_area a b h = area ∧ h = 15 := by
  sorry

#check trapezium_height_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l258_25818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ab_max_value_is_quarter_l258_25820

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2 : ℝ)^a * (2 : ℝ)^b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ)^x * (2 : ℝ)^y = 2 → x * y ≤ a * b :=
by sorry

theorem max_value_is_quarter (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2 : ℝ)^a * (2 : ℝ)^b = 2) :
  a * b = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ab_max_value_is_quarter_l258_25820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l258_25866

structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

theorem pyramid_volume (P : Pyramid) (h1 : P.base.length = 10)
    (h2 : P.base.width = 5) (h3 : P.height = 8) :
  (1/3 : ℝ) * P.base.area * P.height = 400/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l258_25866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_mixture_impossibility_l258_25872

theorem peanut_mixture_impossibility (virginia_amount spanish_amount : ℝ) 
  (virginia_cost spanish_cost mixture_cost : ℝ) :
  virginia_amount > 0 → spanish_amount < 0 → False := by
  intro h_virginia h_spanish
  -- The proof is impossible because we can't have negative amounts in a mixture
  sorry

#check peanut_mixture_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_mixture_impossibility_l258_25872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l258_25809

def is_valid_number (n : Nat) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (∀ d₁ d₂, d₁ ∈ n.digits 10 ∧ d₂ ∈ n.digits 10 ∧ d₁ ≠ d₂ → d₁ ≠ d₂) ∧  -- all digits are different
  (∀ d, d ∈ n.digits 10 ∧ d ≠ 0 → n % d = 0)  -- divisible by any of its non-zero digits

theorem largest_valid_number : 
  is_valid_number 9864 ∧ (∀ m, is_valid_number m → m ≤ 9864) := by
  sorry

#check largest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l258_25809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_minus_r_l258_25879

theorem max_d_minus_r : ∃ (d r : ℕ), 
  (2017 % d = r) ∧ 
  (1029 % d = r) ∧ 
  (725 % d = r) ∧ 
  (∀ (d' r' : ℕ), (2017 % d' = r') ∧ (1029 % d' = r') ∧ (725 % d' = r') → d' - r' ≤ d - r) ∧
  (d - r = 35) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_minus_r_l258_25879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_measurement_error_l258_25891

/-- Theorem: Error in rectangle area measurement
  If one side of a rectangle is measured 7% in excess and the other side is measured 6% in deficit,
  then the error percent in the calculated area is approximately 6.58%.
-/
theorem rectangle_area_measurement_error :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  let measured_length := L * 1.07
  let measured_width := W * 0.94
  let true_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percent := (calculated_area - true_area) / true_area * 100
  abs (error_percent - 6.58) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_measurement_error_l258_25891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l258_25873

/-- The area of a triangle in polar coordinates -/
noncomputable def triangle_area_polar (r1 r2 : ℝ) (θ : ℝ) : ℝ := 
  (1/2) * r1 * r2 * Real.sin θ

/-- The problem statement -/
theorem area_triangle_AOB : 
  let r_A : ℝ := 2
  let θ_A : ℝ := π/6
  let r_B : ℝ := 4
  let θ_B : ℝ := π/3
  triangle_area_polar r_A r_B (θ_B - θ_A) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l258_25873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l258_25811

theorem z_in_fourth_quadrant :
  ∀ z : ℂ, (1 + 2*Complex.I)*z = 5 → Complex.re z > 0 ∧ Complex.im z < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l258_25811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l258_25852

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Given geometric sequence with sum S_n = 3 × 2^n + a, prove a = -3 -/
theorem geometric_sequence_sum_problem (a : ℝ) :
  (∃ (a₁ q : ℝ) (h : q ≠ 1), ∀ n : ℕ, geometricSum a₁ q n = 3 * 2^n + a) →
  a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l258_25852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25821

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else
  if -1 < x ∧ x < 2 then x^2 else 0

-- State the theorem
theorem f_properties :
  (∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y < 4) ∧
  (∀ x : ℝ, f x = 3 → x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l258_25819

/-- Rectangle XYZW in 2D plane -/
structure Rectangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  W : ℝ × ℝ

/-- Point P in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points in 2D plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem rectangle_point_distance (rect : Rectangle) (P : Point) :
  rect.X = (0, 0) →
  rect.Y = (2, 0) →
  rect.Z = (2, 1) →
  rect.W = (0, 1) →
  let u := distance P rect.X
  let v := distance P rect.Y
  let w := distance P rect.Z
  u^2 + w^2 = v^2 →
  distance P rect.W = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l258_25819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_100th_term_l258_25878

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => my_sequence n + n

theorem my_sequence_100th_term : my_sequence 99 = 4951 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_100th_term_l258_25878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_cost_verify_conditions_at_10km_l258_25889

/-- The distance that minimizes the total cost -/
noncomputable def optimal_distance : ℝ := 5

/-- The constant for the land occupation fee -/
def k₁ : ℝ := 200000

/-- The constant for the freight cost -/
def k₂ : ℝ := 8000

/-- The land occupation fee as a function of distance -/
noncomputable def y₁ (x : ℝ) : ℝ := k₁ / x

/-- The freight cost as a function of distance -/
noncomputable def y₂ (x : ℝ) : ℝ := k₂ * x

/-- The total cost as a function of distance -/
noncomputable def total_cost (x : ℝ) : ℝ := y₁ x + y₂ x

/-- Theorem stating that the optimal distance minimizes the total cost -/
theorem optimal_distance_minimizes_cost :
  ∀ x > 0, total_cost optimal_distance ≤ total_cost x :=
by sorry

/-- Theorem verifying the given conditions at 10 km -/
theorem verify_conditions_at_10km :
  y₁ 10 = 20000 ∧ y₂ 10 = 80000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_cost_verify_conditions_at_10km_l258_25889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l258_25847

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_theorem (a b : V) 
  (ha : ‖a‖ = 6)
  (hb : ‖b‖ = 8)
  (hab : ‖a + b‖ = 11) :
  inner a b / (‖a‖ * ‖b‖) = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l258_25847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_unity_power_fifteen_l258_25806

noncomputable def x : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2

theorem cube_roots_unity_power_fifteen : x^15 - y^15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_unity_power_fifteen_l258_25806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_two_miles_l258_25835

/-- The number of revolutions required for a point on the rim of a wheel to travel a given distance -/
noncomputable def revolutions_required (wheel_diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * wheel_diameter)

/-- Conversion factor from miles to feet -/
def miles_to_feet : ℝ := 5280

theorem wheel_revolutions_two_miles (wheel_diameter : ℝ) 
  (h : wheel_diameter = 8) : 
  revolutions_required wheel_diameter (2 * miles_to_feet) = 1320 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_two_miles_l258_25835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OAB_l258_25830

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the polar equation of circle C
noncomputable def polar_circle_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define point A
noncomputable def point_A : ℝ × ℝ := (1, Real.sqrt 3)

-- Define ray l
def ray_l (α : ℝ) (ρ : ℝ) : Prop := ρ > 0 ∧ ρ = polar_circle_C α

-- Theorem statement
theorem max_area_triangle_OAB :
  ∃ (α : ℝ), 
    let B := (polar_circle_C α * Real.cos α, polar_circle_C α * Real.sin α)
    let area := (1/2) * Real.sqrt (1^2 + (Real.sqrt 3)^2) * polar_circle_C α * Real.sin (π/3 - α)
    (∀ β, area ≥ (1/2) * Real.sqrt (1^2 + (Real.sqrt 3)^2) * polar_circle_C β * Real.sin (π/3 - β)) ∧
    area = 2 + Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OAB_l258_25830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_centered_iff_sqrt_three_half_l258_25832

/-- The function f(x) = x^3 - cx -/
noncomputable def f (c x : ℝ) : ℝ := x^3 - c*x

/-- The first derivative of f -/
noncomputable def f' (c x : ℝ) : ℝ := 3*x^2 - c

/-- The second derivative of f -/
noncomputable def f'' (_ x : ℝ) : ℝ := 6*x

/-- The radius of curvature at a point x -/
noncomputable def R (c x : ℝ) : ℝ := 1 / |f'' c x|

/-- The critical points of f -/
def criticalPoints (c : ℝ) : Set ℝ := {x | f' c x = 0}

/-- Predicate: the circle of curvature at x is centered on the x-axis -/
def circleCenteredOnXAxis (c x : ℝ) : Prop :=
  R c x = |f c x|

theorem curvature_centered_iff_sqrt_three_half (c : ℝ) :
  (c > 0 ∧ ∀ x ∈ criticalPoints c, circleCenteredOnXAxis c x) ↔ c = Real.sqrt 3 / 2 := by
  sorry

#check curvature_centered_iff_sqrt_three_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_centered_iff_sqrt_three_half_l258_25832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_plus_a4_equals_negative_four_l258_25849

open Real

-- Define the interval (-1/2, 1)
def I : Set ℝ := { x | -1/2 < x ∧ x < 1 }

-- Define the power series
noncomputable def power_series (a : ℕ → ℝ) (x : ℝ) : ℝ := ∑' n, a n * x^n

-- State the theorem
theorem a3_plus_a4_equals_negative_four 
  (a : ℕ → ℝ) 
  (h : ∀ x ∈ I, x / (1 + x - 2*x^2) = power_series a x) : 
  a 3 + a 4 = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_plus_a4_equals_negative_four_l258_25849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l258_25853

def is_valid_combination (combo : Finset ℕ) : Prop :=
  combo.card = 6 ∧
  (∀ n, n ∈ combo → 1 ≤ n ∧ n ≤ 56) ∧
  (∃ k : ℕ, (combo.prod id) = 10^k) ∧
  (∃ p q, p ∈ combo ∧ q ∈ combo ∧ Prime p ∧ Prime q ∧ p ≠ 2 ∧ p ≠ 5 ∧ q ≠ 2 ∧ q ≠ 5 ∧ p ≠ q)

theorem lottery_probability (combo : Finset ℕ) :
  is_valid_combination combo → (1 : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l258_25853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l258_25857

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a1_positive : a 1 > 0
  special_sum : a 1 + a 9 = a 4

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 6 = 0 ∧ S seq 11 = 0 ∧ S seq 5 = S seq 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l258_25857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_l258_25888

-- Define the rotation angle in radians
noncomputable def α : Real := Real.pi / 3

-- Define the area of the shaded figure
noncomputable def shadedArea (R : Real) : Real := (2 * Real.pi * R^2) / 3

-- Theorem statement
theorem rotated_semicircle_area (R : Real) (h : R > 0) :
  shadedArea R = (2 * Real.pi * R^2) / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_l258_25888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_proof_l258_25854

theorem largest_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 62) (h2 : Nat.lcm a b = 62 * 11 * 12) :
  max a b = 744 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_proof_l258_25854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_weekly_water_consumption_l258_25884

/-- Represents the amount of water in ounces that Rachel drinks on a given day -/
def water_consumption (day : Nat) : Nat :=
  if day = 0 then 20  -- Sunday
  else if day = 1 then 40  -- Monday
  else if day ≥ 2 ∧ day ≤ 5 then 30  -- Tuesday to Friday
  else 0  -- We don't know the consumption for Saturday

/-- The total amount of water Rachel drinks from Sunday to Thursday -/
def total_sunday_to_thursday : Nat :=
  (List.range 5).foldl (λ acc day => acc + water_consumption day) 0

/-- Theorem stating that Rachel's weekly water consumption is 180 ounces plus what she drinks on Friday and Saturday -/
theorem rachel_weekly_water_consumption (weekend_consumption : Nat) :
  (List.range 7).foldl (λ acc day => acc + water_consumption day) 0 + weekend_consumption = 180 + weekend_consumption := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_weekly_water_consumption_l258_25884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l258_25805

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

-- Define the eccentricity
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

-- Define the points on the ellipse
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the theorem
theorem ellipse_triangle_perimeter 
  (e : Ellipse) 
  (h_ecc : eccentricity e = 1/2) 
  (A : EllipsePoint e) 
  (F₁ F₂ : ℝ × ℝ) 
  (D E : EllipsePoint e) 
  (h_DE_perp : (D.x - F₁.1) * (F₂.1 - A.x) + (D.y - F₁.2) * (F₂.2 - A.y) = 0)
  (h_DE_length : Real.sqrt ((D.x - E.x)^2 + (D.y - E.y)^2) = 6) :
  Real.sqrt ((A.x - D.x)^2 + (A.y - D.y)^2) +
  Real.sqrt ((A.x - E.x)^2 + (A.y - E.y)^2) +
  Real.sqrt ((D.x - E.x)^2 + (D.y - E.y)^2) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l258_25805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l258_25812

theorem angle_in_third_quadrant (α : Real) :
  (Real.tan α < 0 ∧ Real.cos α < 0) → 
  (π < α ∧ α < 3*π/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l258_25812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_cone_base_circumference_eq_10pi_l258_25893

/-- Given a circular piece of paper with radius 6 inches, when a 300° sector is cut out
    and the edges are glued together to form a cone, the circumference of the base of the cone
    is 10π inches. -/
theorem cone_base_circumference (π : ℝ) : ℝ :=
  let circle_radius : ℝ := 6
  let sector_angle : ℝ := 300
  let full_circle_angle : ℝ := 360
  let cone_base_circumference : ℝ := (sector_angle / full_circle_angle) * (2 * π * circle_radius)
  cone_base_circumference

theorem cone_base_circumference_eq_10pi (π : ℝ) :
  cone_base_circumference π = 10 * π := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_cone_base_circumference_eq_10pi_l258_25893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_sum_max_min_value_decreasing_intervals_l258_25895

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + a + 1

-- Theorem for the smallest positive period
theorem smallest_positive_period (a : ℝ) : 
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f a (x + T) = f a x ∧ 
  ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f a (x + T') = f a x) → T ≤ T' := by
  sorry

-- Theorem for the value of a
theorem sum_max_min_value (a : ℝ) : 
  (∃ max min : ℝ, (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → f a x ≤ max) ∧
                  (∃ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 ∧ f a x = max) ∧
                  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → min ≤ f a x) ∧
                  (∃ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 ∧ f a x = min) ∧
                  max + min = 3) →
  a = 0 := by
  sorry

-- Theorem for decreasing intervals
theorem decreasing_intervals : 
  ∀ k : ℤ, ∀ x y : ℝ, 
    k * π + π/6 ≤ x ∧ x < y ∧ y ≤ k * π + 2*π/3 → 
    f 0 y < f 0 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_sum_max_min_value_decreasing_intervals_l258_25895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l258_25898

/-- The time for two trains to cross each other when running in the same direction -/
noncomputable def time_to_cross_same_direction (v1 v2 time_opposite : ℝ) : ℝ :=
  time_opposite * (v1 + v2) / (v1 - v2)

/-- Theorem: The time for two trains of equal length to cross each other when running in the same direction is 50.00000000000001 seconds -/
theorem trains_crossing_time
  (v1 : ℝ) (hv1 : v1 = 60)
  (v2 : ℝ) (hv2 : v2 = 40)
  (time_opposite : ℝ) (hto : time_opposite = 10.000000000000002) :
  time_to_cross_same_direction v1 v2 time_opposite = 50.00000000000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l258_25898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_value_l258_25803

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem abcd_value :
  ∀ (a b c d e f p q : ℕ),
    b > c → c > d → d > a →
    1000 * c + 100 * d + 10 * a + b - (1000 * a + 100 * b + 10 * c + d) = 1000 * p + 100 * q + 10 * e + f →
    is_perfect_square (10 * e + f) →
    ¬(5 ∣ (10 * p + q)) →
    1000 * a + 100 * b + 10 * c + d = 1983 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_value_l258_25803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_yoga_time_l258_25801

/-- Represents Mandy's daily exercise routine -/
structure ExerciseRoutine where
  total_time : ℕ
  bicycle_time : ℕ
  gym_bicycle_ratio : ℚ
  yoga_exercise_ratio : ℚ

/-- Calculates the time spent on yoga given an exercise routine -/
def yoga_time (routine : ExerciseRoutine) : ℕ :=
  let gym_time := (routine.bicycle_time * routine.gym_bicycle_ratio.num) / routine.gym_bicycle_ratio.den
  let exercise_time := gym_time + routine.bicycle_time
  ((exercise_time * routine.yoga_exercise_ratio.num) / routine.yoga_exercise_ratio.den).toNat

/-- Theorem stating that Mandy's yoga time is 20 minutes -/
theorem mandy_yoga_time :
  let routine : ExerciseRoutine := {
    total_time := 100,
    bicycle_time := 18,
    gym_bicycle_ratio := 2 / 3,
    yoga_exercise_ratio := 2 / 3
  }
  yoga_time routine = 20 := by
  sorry

#eval yoga_time {
  total_time := 100,
  bicycle_time := 18,
  gym_bicycle_ratio := 2 / 3,
  yoga_exercise_ratio := 2 / 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_yoga_time_l258_25801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_and_triangle_properties_l258_25892

noncomputable section

open Real

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1/2, 1/2 * sin x + (sqrt 3)/2 * cos x)
def b (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the parallel condition
def parallel (x y : ℝ) : Prop := ∃ (k : ℝ), k * (a x).1 = (b y).1 ∧ k * (a x).2 = (b y).2

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin (x + π/3)

-- Define the properties of triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π

-- State the theorem
theorem vector_and_triangle_properties
  (x : ℝ)
  (h_parallel : parallel x (f x))
  (A B C : ℝ)
  (h_triangle : triangle_ABC A B C)
  (h_f : f (A - π/3) = sqrt 3)
  (h_BC : sqrt 7 = 2 * sin C)
  (h_sinB : sin B = sqrt 21 / 7) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (2 = 2 * sin C * sin B / sin A) ∧
  (3 * sqrt 3 / 2 = sin A * sqrt 7 / (2 * sin C)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_and_triangle_properties_l258_25892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l258_25844

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π / 2 := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l258_25844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_AB_equation_l258_25880

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := 1

-- Define the conditions
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : a - c = Real.sqrt 3 - 1
axiom h4 : b = Real.sqrt 2
axiom h5 : a^2 = b^2 + c^2

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line AB
def line_AB (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the length of segment AB
noncomputable def AB_length (k : ℝ) : ℝ := (4 * Real.sqrt 3 * (k^2 + 1)) / (2 + 3 * k^2)

-- Theorem statements
theorem ellipse_equation : 
  ellipse_eq = fun x y => x^2 / 3 + y^2 / 2 = 1 := by
  sorry

theorem line_AB_equation :
  ∀ k, AB_length k = (3 * Real.sqrt 3) / 2 → 
    (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_AB_equation_l258_25880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l258_25827

-- Define the triangle and its properties
def Triangle (α β γ : Real) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ 0 < β ∧ 0 < γ

-- Define the inequality condition
def InequalityHolds (α β γ : Real) : Prop :=
  10 * Real.sin α * Real.sin β * Real.sin γ - 
  2 * Real.sin (Real.pi/3) * (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2) > 0

-- Main theorem
theorem triangle_angle_inequality 
  (α β γ : Real) 
  (h_triangle : Triangle α β γ) 
  (h_angle_diff : β - γ = Real.pi/6) 
  (h_α_range : 23.0 * Real.pi/180 ≤ α ∧ α ≤ 100.0 * Real.pi/180) :
  InequalityHolds α β γ ↔ 22.1 * Real.pi/180 ≤ α ∧ α ≤ 104.5 * Real.pi/180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l258_25827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_iff_inequality_l258_25826

/-- A triangle can be constructed with sides a, b, c if and only if (a² + b² + c²)² > 2(a⁴ + b⁴ + c⁴) -/
theorem triangle_construction_iff_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c ∧ x + y > z ∧ x + z > y ∧ y + z > x) ↔ 
  (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_iff_inequality_l258_25826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inverse_point_l258_25839

-- Define the exponential function
noncomputable def exp_func (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the inverse function property
def inverse_passes_through (a : ℝ) : Prop :=
  exp_func a 2 = 16

-- Theorem statement
theorem exp_inverse_point (a : ℝ) :
  inverse_passes_through a → a = 4 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inverse_point_l258_25839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximized_at_15_l258_25848

/-- Revenue function for a bookstore selling a novel -/
noncomputable def revenue (p : ℝ) : ℝ :=
  if p ≤ 15 then p * (150 - 6 * p)
  else p * (120 - 4 * p)

/-- The maximum price allowed for the book -/
def max_price : ℝ := 30

theorem revenue_maximized_at_15 :
  ∃ (max_revenue : ℝ), 
    (∀ p, 0 ≤ p ∧ p ≤ max_price → revenue p ≤ max_revenue) ∧
    revenue 15 = max_revenue ∧
    max_revenue = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximized_at_15_l258_25848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l258_25841

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ - Real.cos θ = 1/3) : Real.sin (2 * θ) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l258_25841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_1000_after_transform_l258_25828

/-- Represents the state of the blackboard -/
def Blackboard := List Int

/-- Represents a single pairing operation -/
def Pairing := List (Int × Int)

/-- Applies the transformation step to a pair of numbers -/
def transformPair (p : Int × Int) : (Int × Int) :=
  (p.1 + p.2, p.1 - p.2)

/-- Applies the transformation step to the entire blackboard -/
def transformBlackboard (b : Blackboard) (p : Pairing) : Blackboard :=
  p.map transformPair |>.foldl (fun acc (x, y) => x :: y :: acc) []

/-- Checks if a list of integers contains 1000 consecutive integers -/
def isConsecutive1000 (b : Blackboard) : Prop :=
  ∃ start : Int, ∀ i : Fin 1000, b.contains (start + i.val)

/-- Main theorem: After any number of transformations, 
    the blackboard will not contain 1000 consecutive integers again -/
theorem no_consecutive_1000_after_transform 
  (initial : Blackboard) 
  (h_initial : isConsecutive1000 initial) :
  ∀ (steps : ℕ) (transformations : List Pairing), 
    let final := (transformations.take steps).foldl transformBlackboard initial
    ¬ isConsecutive1000 final :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_1000_after_transform_l258_25828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_chairs_l258_25864

/-- Represents a row of chairs -/
def ChairRow := List Bool

/-- Checks if a chair arrangement is valid (no adjacent occupied chairs) -/
def isValidArrangement (chairs : ChairRow) : Prop :=
  ∀ i, i + 1 < chairs.length → ¬(chairs.get? i = some true ∧ chairs.get? (i+1) = some true)

/-- Counts the number of occupied chairs in a row -/
def countOccupied (chairs : ChairRow) : Nat :=
  chairs.filter id |>.length

/-- Theorem: The maximum number of occupied chairs in a row of 20 chairs is 19 -/
theorem max_occupied_chairs :
  ∃ (arrangement : ChairRow),
    arrangement.length = 20 ∧
    isValidArrangement arrangement ∧
    countOccupied arrangement = 19 ∧
    (∀ (other : ChairRow),
      other.length = 20 →
      isValidArrangement other →
      countOccupied other ≤ 19) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_chairs_l258_25864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_surface_area_in_tetrahedron_l258_25829

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a sphere touches three edges of a regular tetrahedron -/
def Sphere.touchesThreeEdges (s : Sphere) : Prop :=
  sorry

/-- The surface area of a sphere inside a regular tetrahedron -/
def Sphere.surfaceAreaInside (s : Sphere) : ℝ :=
  sorry

/-- Given a regular tetrahedron with edge length a and a sphere touching three edges
    at their ends emanating from one vertex, the area of the spherical surface located
    inside the tetrahedron is equal to (π * a^2 / 6) * (2 * √3 - 3). -/
theorem spherical_surface_area_in_tetrahedron (a : ℝ) (a_pos : a > 0) :
  ∃ (sphere : Sphere), 
    sphere.touchesThreeEdges ∧
    sphere.surfaceAreaInside = (π * a^2 / 6) * (2 * Real.sqrt 3 - 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_surface_area_in_tetrahedron_l258_25829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25823

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 3)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ y, f (y + q) ≠ f y) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 2 * Real.pi / 3 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l258_25887

/-- GeometricSequence represents a geometric sequence with real terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  hq : q ≠ 1
  seq_def : ∀ n, a (n + 1) = q * a n

/-- S_n is the sum of the first n terms of a geometric sequence -/
noncomputable def S_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_40 (g : GeometricSequence) 
  (h10 : S_n g 10 = 10) (h30 : S_n g 30 = 70) : 
  S_n g 40 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l258_25887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l258_25842

/-- Parabola type representing y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Point on a parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

/-- Helper definition: The normal at point A passes through point B -/
def NormalPassesThrough (para : Parabola) (A B : PointOnParabola para) : Prop :=
  (B.y - A.y) * (B.x - A.x) = -para.p^2 / A.y

/-- Theorem: The minimum length of a chord AB on a parabola y^2 = 2px, 
    where the normal at A passes through B, is 3√3p -/
theorem min_chord_length (para : Parabola) 
  (A B : PointOnParabola para) 
  (normal_passes : NormalPassesThrough para A B) : 
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) ≥ 3 * Real.sqrt 3 * para.p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l258_25842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_l258_25875

-- Define the speeds of the trains in mph
noncomputable def speed_A : ℝ := 50
noncomputable def speed_B : ℝ := 80

-- Define the time difference in hours
noncomputable def time_difference : ℝ := 0.5

-- Define the function to calculate the overtaking time in minutes
noncomputable def overtaking_time : ℝ :=
  let distance_A := speed_A * time_difference
  let relative_speed := speed_B - speed_A
  (distance_A / relative_speed) * 60

-- Theorem statement
theorem train_overtake :
  overtaking_time = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_l258_25875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_diagonal_distinct_l258_25870

/-- Represents a symmetric table with distinct elements in each row and column -/
structure SymmetricTable (n : ℕ) where
  table : Fin (2*n + 1) → Fin (2*n + 1) → Fin (2*n + 1)
  distinct_rows : ∀ i, Function.Injective (table i)
  distinct_cols : ∀ j, Function.Injective (λ i ↦ table i j)
  symmetric : ∀ i j, table i j = table j i

/-- Theorem: All elements on the main diagonal of a SymmetricTable are distinct -/
theorem main_diagonal_distinct (n : ℕ) (t : SymmetricTable n) :
  Function.Injective (λ i ↦ t.table i i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_diagonal_distinct_l258_25870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_eq_2036_l258_25867

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ := sorry

/-- The 1000th term of the sequence u_n is 2036 -/
theorem u_1000_eq_2036 : u 1000 = 2036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_eq_2036_l258_25867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_correct_guesses_l258_25836

/-- Represents a deck of cards -/
structure Deck where
  cards : Finset (Fin 4 × Fin 13)
  card_count : cards.card = 52

/-- Represents a strategy for guessing suits -/
def GuessStrategy := Deck → Fin 4

/-- The optimal strategy always guesses a suit with the maximum number of remaining cards -/
def optimal_strategy : GuessStrategy :=
  λ d => (Fin.find (λ s => ∀ t, (d.cards.filter (λ c => c.1 = s)).card ≥ (d.cards.filter (λ c => c.1 = t)).card)).get sorry

/-- The number of correct guesses when using a given strategy -/
def correct_guesses (d : Deck) (strategy : GuessStrategy) : ℕ :=
  sorry

/-- Theorem: The optimal strategy results in at least 13 correct guesses -/
theorem optimal_strategy_correct_guesses (d : Deck) :
  correct_guesses d optimal_strategy ≥ 13 := by
  sorry

#check optimal_strategy_correct_guesses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_correct_guesses_l258_25836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_to_work_time_l258_25897

/-- Calculates the time taken for a one-way trip given the distance and speed -/
noncomputable def time_for_trip (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Represents a round trip with different speeds for each direction -/
structure RoundTrip where
  distance : ℝ
  speed_to : ℝ
  speed_from : ℝ
  total_time : ℝ

/-- Calculates the total time for a round trip -/
noncomputable def total_trip_time (trip : RoundTrip) : ℝ :=
  time_for_trip trip.distance trip.speed_to + time_for_trip trip.distance trip.speed_from

/-- Theorem: Given the conditions of Cole's trip, the time to work is 70 minutes -/
theorem cole_trip_to_work_time :
  ∀ (trip : RoundTrip),
    trip.speed_to = 75 →
    trip.speed_from = 105 →
    trip.total_time = 2 →
    time_for_trip trip.distance trip.speed_to * 60 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_to_work_time_l258_25897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_bounds_l258_25856

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 5
  | (n + 1) => sequence_a n + 1 / sequence_a n

theorem a_1000_bounds : 45 < sequence_a 1000 ∧ sequence_a 1000 < 45.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_bounds_l258_25856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_max_distance_line_properties_l258_25840

-- Define the line l
def line_l (a b x y : ℝ) : Prop := (2*a + b)*x + (a + b)*y + a - b = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, 3)

-- Define point P
def point_p : ℝ × ℝ := (3, 4)

-- Theorem 1: Line l passes through the fixed point for all a and b
theorem line_passes_through_fixed_point (a b : ℝ) :
  line_l a b (fixed_point.1) (fixed_point.2) := by sorry

-- Define the maximized distance line
def max_distance_line (x y : ℝ) : Prop := 5*x + y + 7 = 0

-- Theorem 2: The maximized distance line passes through the fixed point
--            and is perpendicular to the line segment from fixed point to P
theorem max_distance_line_properties :
  max_distance_line (fixed_point.1) (fixed_point.2) ∧
  (fixed_point.2 - point_p.2) / (fixed_point.1 - point_p.1) * 
  (-(5 : ℝ)) = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_max_distance_line_properties_l258_25840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l258_25804

noncomputable def data : List ℝ := [5, 7, 7, 8, 10, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_data :
  standardDeviation data = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l258_25804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_2sqrt3_plus_2_l258_25822

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * t - 2)

-- Define the circle C
def circle_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * Real.cos θ + 3 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, -2)

-- Define the intersection points A and B
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry

-- State the theorem
theorem distance_sum_equals_2sqrt3_plus_2 :
  ∃ (A B : ℝ × ℝ),
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ ρ θ : ℝ, circle_C ρ θ ∧ A.1 = ρ * Real.cos θ ∧ A.2 = ρ * Real.sin θ) ∧
    (∃ ρ θ : ℝ, circle_C ρ θ ∧ B.1 = ρ * Real.cos θ ∧ B.2 = ρ * Real.sin θ) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_2sqrt3_plus_2_l258_25822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l258_25802

/-- Proves that the percentage of discretionary income put into the vacation fund is 30% --/
theorem vacation_fund_percentage (net_salary : ℝ) (discretionary_income : ℝ) 
  (savings_percent : ℝ) (social_percent : ℝ) (gifts_amount : ℝ) :
  net_salary = 3700 →
  discretionary_income = net_salary / 5 →
  savings_percent = 0.20 →
  social_percent = 0.35 →
  gifts_amount = 111 →
  ∃ (vacation_percent : ℝ),
    vacation_percent * discretionary_income + 
    savings_percent * discretionary_income + 
    social_percent * discretionary_income + 
    gifts_amount = discretionary_income ∧
    abs (vacation_percent - 0.30) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l258_25802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l258_25808

noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

theorem trapezium_area_example :
  let a : ℝ := 20 -- length of one parallel side
  let b : ℝ := 10 -- length of the other parallel side
  let h : ℝ := 10 -- distance between parallel sides
  trapezium_area a b h = 150 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l258_25808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_system_days_to_use_50_liters_l258_25813

/-- Represents the water usage of a sprinkler system -/
structure SprinklerSystem where
  morning_usage : ℚ
  evening_usage : ℚ

/-- Calculates the number of days required to use a given amount of water -/
def days_to_use_water (s : SprinklerSystem) (total_water : ℚ) : ℚ :=
  total_water / (s.morning_usage + s.evening_usage)

/-- Theorem: The sprinkler system takes 5 days to use 50 liters of water -/
theorem sprinkler_system_days_to_use_50_liters :
  let s : SprinklerSystem := { morning_usage := 4, evening_usage := 6 }
  days_to_use_water s 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_system_days_to_use_50_liters_l258_25813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotone_decreasing_implies_a_bound_l258_25859

/-- A function f: ℝ → ℝ is monotonically decreasing on an interval I if for all x, y ∈ I,
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ {x y}, x ∈ I → y ∈ I → x < y → f x > f y

/-- The quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem quadratic_monotone_decreasing_implies_a_bound :
  ∀ a : ℝ, MonotonicallyDecreasing (f a) { x : ℝ | x ≤ 4 } → a ≤ -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotone_decreasing_implies_a_bound_l258_25859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cubic_with_three_roots_of_unity_l258_25851

/-- A complex number z is a root of unity if z^n = 1 for some positive integer n -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ+, z ^ (n : ℕ) = 1

/-- The equation z^3 + pz + q = 0 where p and q are integers -/
def cubic_equation (p q : ℤ) (z : ℂ) : Prop :=
  z^3 + (p : ℂ) * z + (q : ℂ) = 0

/-- There exist integers p and q such that the equation z^3 + pz + q = 0 
    has exactly 3 roots of unity as its solutions -/
theorem exists_cubic_with_three_roots_of_unity :
  ∃ p q : ℤ, ∃ z₁ z₂ z₃ : ℂ, 
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
    is_root_of_unity z₁ ∧ is_root_of_unity z₂ ∧ is_root_of_unity z₃ ∧
    cubic_equation p q z₁ ∧ cubic_equation p q z₂ ∧ cubic_equation p q z₃ ∧
    (∀ z : ℂ, is_root_of_unity z ∧ cubic_equation p q z → z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cubic_with_three_roots_of_unity_l258_25851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l258_25810

/-- The ellipse C with equation x^2/3 + y^2 = 1 -/
def ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- A line that intersects the ellipse C -/
structure IntersectingLine where
  slope : ℝ
  intercept : ℝ
  intersects_ellipse : ∃ (x y : ℝ), (x, y) ∈ ellipse_C ∧ y = slope * x + intercept

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (l : IntersectingLine) : ℝ :=
  abs l.intercept / Real.sqrt (1 + l.slope^2)

/-- The area of a triangle given two points on a line and the origin -/
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1/2) * abs (x1 * y2 - x2 * y1)

theorem max_triangle_area :
  ∀ (l : IntersectingLine),
    distance_point_to_line 0 0 l = Real.sqrt 3 / 2 →
    ∃ (x1 y1 x2 y2 : ℝ),
      (x1, y1) ∈ ellipse_C ∧
      (x2, y2) ∈ ellipse_C ∧
      y1 = l.slope * x1 + l.intercept ∧
      y2 = l.slope * x2 + l.intercept ∧
      triangle_area x1 y1 x2 y2 ≤ Real.sqrt 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l258_25810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_increase_approx_26_22_percent_l258_25862

/-- Calculates the percent increase in actual sales between two years --/
noncomputable def percent_increase_in_sales (total_sales_this_year total_sales_last_year : ℝ)
  (tax_rate_this_year tax_rate_last_year : ℝ)
  (discount_this_year discount_last_year : ℝ) : ℝ :=
  let actual_sales_this_year := total_sales_this_year / (1 + tax_rate_this_year - discount_this_year)
  let actual_sales_last_year := total_sales_last_year / (1 + tax_rate_last_year - discount_last_year)
  (actual_sales_this_year - actual_sales_last_year) / actual_sales_last_year * 100

/-- The percent increase in actual sales is approximately 26.22% --/
theorem sales_increase_approx_26_22_percent :
  let total_sales_this_year := (400 : ℝ)
  let total_sales_last_year := (320 : ℝ)
  let tax_rate_this_year := (0.07 : ℝ)
  let tax_rate_last_year := (0.06 : ℝ)
  let discount_this_year := (0.05 : ℝ)
  let discount_last_year := (0.03 : ℝ)
  abs (percent_increase_in_sales total_sales_this_year total_sales_last_year
    tax_rate_this_year tax_rate_last_year discount_this_year discount_last_year - 26.22) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_increase_approx_26_22_percent_l258_25862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l258_25861

/-- A complex number z = x + yi where x and y are positive integers -/
structure PositiveIntegerComplex where
  x : ℕ+
  y : ℕ+

/-- The fourth power of a complex number equals 162 + di for some integer d -/
def satisfiesFourthPower (z : PositiveIntegerComplex) : Prop :=
  ∃ d : ℤ, (z.x.val : ℂ) + (z.y.val : ℂ) * Complex.I ^ 4 = 162 + d * Complex.I

theorem unique_solution :
  ∃! z : PositiveIntegerComplex, satisfiesFourthPower z ∧ z.x = 1 ∧ z.y = 5 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l258_25861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l258_25845

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | n+1 => 1 / (1 - sequenceA n)

theorem sequence_properties :
  (sequenceA 1 = 2) ∧
  (sequenceA 2 = -1) ∧
  (sequenceA 3 = 1/2) ∧
  (∀ n : ℕ, sequenceA n = sequenceA (n + 3)) :=
by
  sorry

#eval sequenceA 1
#eval sequenceA 2
#eval sequenceA 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l258_25845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25837

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 1/(x+1)

-- State the theorem
theorem f_properties :
  ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    (f x ≥ 1 - x + x^2) ∧ (f x > 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l258_25837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosR_l258_25800

theorem right_triangle_cosR (P Q R : ℝ) (h1 : P + Q + R = 180) (h2 : P = 90) (h3 : Real.sin Q = 3/5) : Real.cos R = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosR_l258_25800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_s_value_l258_25846

def loop_iteration (i s : Int) : Int × Int :=
  (i + 1, 2 * s - 1)

def final_state (initial_i initial_s : Int) : Int × Int :=
  let rec loop (i s : Int) (fuel : Nat) : Int × Int :=
    if fuel = 0 then (i, s)
    else if i < 6 then
      let (new_i, new_s) := loop_iteration i s
      loop new_i new_s (fuel - 1)
    else
      (i, s)
  loop initial_i initial_s 5

theorem final_s_value :
  let (_, final_s) := final_state 1 0
  final_s = -31 := by
  sorry

#eval final_state 1 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_s_value_l258_25846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l258_25824

theorem largest_initial_number : ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), 
  (89 + a₁ + a₂ + a₃ + a₄ + a₅ = 100) ∧ 
  (¬(a₁ ∣ 89)) ∧ 
  (¬(a₂ ∣ (89 + a₁))) ∧ 
  (¬(a₃ ∣ (89 + a₁ + a₂))) ∧ 
  (¬(a₄ ∣ (89 + a₁ + a₂ + a₃))) ∧ 
  (¬(a₅ ∣ (89 + a₁ + a₂ + a₃ + a₄))) ∧ 
  (∀ n > 89, ¬∃ (b₁ b₂ b₃ b₄ b₅ : ℕ), 
    (n + b₁ + b₂ + b₃ + b₄ + b₅ = 100) ∧ 
    (¬(b₁ ∣ n)) ∧ 
    (¬(b₂ ∣ (n + b₁))) ∧ 
    (¬(b₃ ∣ (n + b₁ + b₂))) ∧ 
    (¬(b₄ ∣ (n + b₁ + b₂ + b₃))) ∧ 
    (¬(b₅ ∣ (n + b₁ + b₂ + b₃ + b₄)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l258_25824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_distance_theorem_l258_25831

/-- Calculates the distance flown by a jet given flight times and wind speed -/
noncomputable def distance_flown (time_with_wind time_against_wind wind_speed : ℝ) : ℝ :=
  let jet_speed := (time_against_wind + time_with_wind) * wind_speed / (time_against_wind - time_with_wind)
  time_with_wind * (jet_speed + wind_speed)

/-- Theorem stating that under given conditions, the jet flies 2000 miles -/
theorem jet_distance_theorem :
  distance_flown 4 5 50 = 2000 := by
  -- Unfold the definition of distance_flown
  unfold distance_flown
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_distance_theorem_l258_25831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l258_25843

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  (∀ (p q : ℝ), p > 0 → q > 0 → 
    p + (q - 1) = 0 →
    1/x + 2/y ≤ 1/p + 2/q) ∧
  1/x + 2/y = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l258_25843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_sets_bound_l258_25807

theorem weight_sets_bound (weights : Finset ℕ) (h1 : weights.card = 19) 
  (h2 : ∀ w ∈ weights, w ≤ 70) : 
  (Finset.powerset weights).card ≤ 1230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_sets_bound_l258_25807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l258_25825

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.cos θ - ρ * Real.sin θ + 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ) (α₁ α₂ ρ₁ ρ₂ θ₁ θ₂ : ℝ),
    A = curve_C α₁ ∧
    B = curve_C α₂ ∧
    line_l ρ₁ θ₁ ∧
    line_l ρ₂ θ₂ ∧
    A.1 = ρ₁ * Real.cos θ₁ ∧
    A.2 = ρ₁ * Real.sin θ₁ ∧
    B.1 = ρ₂ * Real.cos θ₂ ∧
    B.2 = ρ₂ * Real.sin θ₂ ∧
    1 / distance point_P A + 1 / distance point_P B = 8 * Real.sqrt 5 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l258_25825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_line_segment_l258_25876

-- Define a Euclidean space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- Define two distinct points in the space
variable (p q : E)

-- Hypothesis that the points are distinct
variable (h : p ≠ q)

-- Statement to prove
theorem shortest_path_is_line_segment :
  ∀ (γ : ℝ → E), (γ 0 = p) → (γ 1 = q) → 
  ‖q - p‖ ≤ ∫ (t : ℝ) in (0:ℝ)..(1:ℝ), norm (deriv γ t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_line_segment_l258_25876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l258_25882

/-- A function that checks if a number is a valid three-digit number according to the problem conditions -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧  -- three-digit number
  n % 2 = 0 ∧  -- even number
  (n / 10 % 10 + n % 10 = 14)  -- sum of tens and units digits is 14

/-- The count of valid numbers according to the problem conditions -/
def validNumberCount : ℕ := (Finset.filter (fun n => isValidNumber n) (Finset.range 1000)).card

/-- Theorem stating that there are exactly 36 numbers satisfying the given conditions -/
theorem valid_number_count_is_36 : validNumberCount = 36 := by
  sorry

#eval validNumberCount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l258_25882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_lower_bound_l258_25838

theorem existence_implies_lower_bound (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a ≤ a*x - 3) → a ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_lower_bound_l258_25838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_2_sqrt_41_l258_25869

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (P : ℝ × ℝ × ℝ) (L : ℝ → ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The point P in 3D space -/
def P : ℝ × ℝ × ℝ := (2, 4, 6)

/-- The line L in 3D space -/
def L (t : ℝ) : ℝ × ℝ × ℝ := (8 + 4*t, 9 + 3*t, 9 - 3*t)

theorem distance_point_to_line_is_2_sqrt_41 :
  distance_point_to_line P L = 2 * Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_2_sqrt_41_l258_25869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_B_value_l258_25834

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * Real.sin x ^ 2 - Real.cos x ^ 2 + 3

-- Theorem for the range of f(x)
theorem f_range : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc 0 3 := by sorry

-- Define a triangle
structure Triangle where
  a : Real
  b : Real
  c : Real
  A : Real
  B : Real
  C : Real
  -- Add conditions
  h1 : b / a = Real.sqrt 3
  h2 : Real.sin (2 * A + C) / Real.sin A = 2 + 2 * Real.cos (A + C)
  -- Add triangle axioms
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Theorem for f(B) in the triangle
theorem f_B_value (t : Triangle) : f t.B = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_B_value_l258_25834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructibility_l258_25833

/-- Given parameters for triangle construction -/
structure TriangleParams where
  u : ℝ  -- length of AO
  f_a : ℝ  -- length of angle bisector from A
  d : ℝ  -- difference between sides b and c

/-- Conditions for triangle constructibility -/
def is_constructible (p : TriangleParams) : Prop :=
  0 < p.u ∧ p.u < p.f_a ∧ p.f_a < 2 * p.u ∧
  p.d ≤ (2 * p.u * (p.f_a - p.u)) / (2 * p.u - p.f_a)

/-- Angle bisector property -/
def is_angle_bisector (a b c f_a : ℝ) : Prop :=
  f_a * (b + c) = 2 * a * b * c / (b + c)

/-- Inscribed circle property -/
def is_inscribed_circle (a b c r : ℝ) : Prop :=
  r = (a + b - c) / 2

/-- Theorem stating the conditions for triangle constructibility -/
theorem triangle_constructibility (p : TriangleParams) :
  ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
    b - c = p.d ∧  -- given difference between sides
    ∃ (f_a : ℝ), f_a = p.f_a ∧ is_angle_bisector a b c f_a ∧  -- angle bisector condition
    ∃ (r : ℝ), 0 < r ∧ is_inscribed_circle a b c r ∧ p.u = a - r  -- inscribed circle condition
  ↔ 
  is_constructible p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructibility_l258_25833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_PQ_and_RS_l258_25815

noncomputable def P : ℝ × ℝ × ℝ := (3, -9, 6)
noncomputable def Q : ℝ × ℝ × ℝ := (13, -19, 11)
noncomputable def R : ℝ × ℝ × ℝ := (1, 4, -7)
noncomputable def S : ℝ × ℝ × ℝ := (3, -6, 9)

noncomputable def intersection_point : ℝ × ℝ × ℝ := (-19/3, 10/3, 4/3)

theorem intersection_of_PQ_and_RS :
  ∃ (t s : ℝ),
    (P.1 + t * (Q.1 - P.1) = R.1 + s * (S.1 - R.1)) ∧
    (P.2.1 + t * (Q.2.1 - P.2.1) = R.2.1 + s * (S.2.1 - R.2.1)) ∧
    (P.2.2 + t * (Q.2.2 - P.2.2) = R.2.2 + s * (S.2.2 - R.2.2)) ∧
    (P.1 + t * (Q.1 - P.1), P.2.1 + t * (Q.2.1 - P.2.1), P.2.2 + t * (Q.2.2 - P.2.2)) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_PQ_and_RS_l258_25815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cairo_greatest_increase_l258_25860

/-- Represents a city with its population data --/
structure City where
  name : String
  pop1970 : ℚ
  pop1980 : ℚ

/-- Calculates the percentage increase in population --/
def percentageIncrease (city : City) : ℚ :=
  (city.pop1980 - city.pop1970) / city.pop1970

/-- The list of cities with their population data --/
def cities : List City :=
  [
    { name := "Paris", pop1970 := 15/10, pop1980 := 18/10 },
    { name := "Cairo", pop1970 := 24/10, pop1980 := 33/10 },
    { name := "Lima", pop1970 := 12/10, pop1980 := 156/100 },
    { name := "Tokyo", pop1970 := 86/10, pop1980 := 952/100 },
    { name := "Toronto", pop1970 := 2, pop1980 := 24/10 }
  ]

/-- Theorem stating that Cairo had the greatest percentage increase --/
theorem cairo_greatest_increase :
  ∃ cairo ∈ cities, cairo.name = "Cairo" ∧
  ∀ city ∈ cities, percentageIncrease cairo ≥ percentageIncrease city := by
  sorry

#eval cities.map (fun c => (c.name, percentageIncrease c))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cairo_greatest_increase_l258_25860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_values_theorem_l258_25894

def possible_sum_values (x y : ℝ) : Prop :=
  x = y * (3 - y)^2 ∧ y = x * (3 - x)^2 →
  (x + y) ∈ ({0, 3, 4, 5, 8} : Set ℝ)

theorem sum_values_theorem :
  ∀ x y : ℝ, possible_sum_values x y :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_values_theorem_l258_25894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l258_25814

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l258_25814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_poem_memory_l258_25890

/-- Represents the state of Sally's poem memorization --/
structure PoemMemory where
  total : Nat
  perfect : Nat
  singleStanza : Nat
  mixedUp : Nat

/-- Calculates the number of completely forgotten poems --/
def forgotten (pm : PoemMemory) : Nat :=
  pm.total - (pm.perfect + pm.singleStanza + pm.mixedUp)

/-- Calculates the number of partially remembered or mixed up poems --/
def partialOrMixed (pm : PoemMemory) : Nat :=
  pm.singleStanza + pm.mixedUp

theorem sally_poem_memory :
  let pm : PoemMemory := { total := 20, perfect := 7, singleStanza := 5, mixedUp := 4 }
  forgotten pm = 4 ∧ partialOrMixed pm = 9 := by
  sorry

#eval forgotten { total := 20, perfect := 7, singleStanza := 5, mixedUp := 4 }
#eval partialOrMixed { total := 20, perfect := 7, singleStanza := 5, mixedUp := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_poem_memory_l258_25890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_beneficial_iff_effective_rate_higher_l258_25881

/-- Represents the terms of a discount offer for tea. -/
structure DiscountTerms where
  discountPercentage : ℚ
  discountPeriod : ℕ
  maxDelay : ℕ

/-- Calculates whether a discount is beneficial given the terms and the annual interest rate. -/
def isDiscountBeneficial (terms : DiscountTerms) (annualInterestRate : ℚ) : Bool :=
  let effectiveRate := (terms.discountPercentage / (100 - terms.discountPercentage)) * (365 / (terms.maxDelay - terms.discountPeriod))
  effectiveRate > annualInterestRate

theorem discount_beneficial_iff_effective_rate_higher 
  (terms : DiscountTerms) 
  (annualInterestRate : ℚ) 
  (h1 : terms.discountPercentage > 0)
  (h2 : terms.discountPercentage < 100)
  (h3 : terms.maxDelay > terms.discountPeriod)
  (h4 : annualInterestRate > 0) :
  isDiscountBeneficial terms annualInterestRate ↔ 
  ((terms.discountPercentage / (100 - terms.discountPercentage)) * (365 / (terms.maxDelay - terms.discountPeriod))) > annualInterestRate := by
  sorry

def main : IO Unit := do
  IO.println s!"Tea \"Wuyi Mountain Oolong\": {isDiscountBeneficial ⟨3, 7, 31⟩ (22/100)}"
  IO.println s!"Tea \"Da Hong Pao\": {isDiscountBeneficial ⟨2, 4, 40⟩ (22/100)}"
  IO.println s!"Tea \"Tieguanyin\": {isDiscountBeneficial ⟨5, 10, 35⟩ (22/100)}"
  IO.println s!"Tea \"Pu-erh\": {isDiscountBeneficial ⟨1, 3, 24⟩ (22/100)}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_beneficial_iff_effective_rate_higher_l258_25881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_3333_l258_25874

/-- The sum of the alternating series from 0 to n -/
def alternating_sum : ℕ → ℤ
| 0 => 0
| (n + 1) => alternating_sum n + (if n % 2 = 0 then (n + 1 : ℤ) else -(n + 1 : ℤ))

/-- The last term in the series -/
def last_term : ℕ := 9998

theorem alternating_sum_equals_3333 : alternating_sum last_term = 3333 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_3333_l258_25874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l258_25877

/-- Given a circle of radius r and three points dividing it into arcs with ratio 3:4:5,
    the area of the triangle formed by tangents at these points is r^2 * (3√3 + √3) -/
theorem tangent_triangle_area (r : ℝ) (h : r > 0) :
  let arc_ratios := [3, 4, 5]
  let total_ratio := (arc_ratios.sum)
  let arc_angles := List.map (λ x => (x / total_ratio) * (2 * Real.pi)) arc_ratios
  let triangle_angles := List.map (λ x => Real.pi - x) arc_angles
  let triangle_area := r^2 * (3 * Real.sqrt 3 + Real.sqrt 3)
  triangle_area = r^2 * (3 * Real.sqrt 3 + Real.sqrt 3) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l258_25877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l258_25896

theorem problem_solution (m n : ℤ) 
  (h1 : ((-2 : ℚ) ^ (2 * m) = 2 ^ (6 - m)))
  (h2 : ((-3 : ℚ) ^ n = 3 ^ (4 - n))) : 
  m = 2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l258_25896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_problem_l258_25868

/-- Represents the delivery problem from the ancient Chinese mathematics book --/
theorem delivery_problem (x : ℝ) (h : x > 2) : 
  (800 : ℝ) / (x - 2) = (5 / 2 : ℝ) * ((800 : ℝ) / (x + 1)) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_problem_l258_25868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l258_25817

theorem cube_root_of_27 : (27 : ℝ) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l258_25817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_cube_and_reciprocal_result_l258_25871

/-- Repeatedly apply cube and cubic reciprocal operations n times -/
noncomputable def repeated_cube_and_reciprocal (x : ℝ) (n : ℕ) : ℝ :=
  x ^ ((-9 : ℤ) ^ (3 ^ n))

/-- The result of repeatedly applying cube and cubic reciprocal operations n times -/
theorem repeated_cube_and_reciprocal_result (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  repeated_cube_and_reciprocal x n = x ^ ((-9 : ℤ) ^ (3 ^ n)) := by
  -- Unfold the definition of repeated_cube_and_reciprocal
  unfold repeated_cube_and_reciprocal
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_cube_and_reciprocal_result_l258_25871
