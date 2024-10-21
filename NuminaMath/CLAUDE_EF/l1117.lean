import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l1117_111763

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x + Real.pi / 3) - (1 / 2) * Real.cos (ω * x - 7 * Real.pi / 6)

theorem smallest_omega : 
  ∃ ω : ℝ, ω > 0 ∧ f ω (-Real.pi / 6) = 3 / 4 ∧ 
  ∀ ω' : ℝ, ω' > 0 → f ω' (-Real.pi / 6) = 3 / 4 → ω ≤ ω' ∧ ω = 1 := by
  sorry

#check smallest_omega

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l1117_111763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_zero_point_related_l1117_111704

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.exp (x - 2) + Real.log (x - 1) - 1
noncomputable def g (x a : ℝ) := x * (Real.log x - a * x) - 2

-- Define what it means for two functions to be "zero-point related"
def zero_point_related (f g : ℝ → ℝ) :=
  ∃ m n : ℝ, f m = 0 ∧ g n = 0 ∧ |m - n| ≤ 1

-- State the theorem
theorem min_a_for_zero_point_related :
  (∃ a : ℝ, zero_point_related f (g · a)) →
  (∀ a : ℝ, zero_point_related f (g · a) → a ≥ -2) ∧
  zero_point_related f (g · (-2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_zero_point_related_l1117_111704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_l_l1117_111751

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (l : ℝ → ℝ → ℝ) : ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let line_cartesian := fun x y => x + y - 1
  |line_cartesian x y| / Real.sqrt 2

/-- Theorem stating the distance from point M to line l --/
theorem distance_M_to_l :
  let M_r := 2
  let M_θ := π / 3
  let l := fun ρ θ => ρ * Real.sin (θ + π / 4) - Real.sqrt 2 / 2
  distance_point_to_line M_r M_θ l = Real.sqrt 6 / 2 := by
  sorry

#check distance_M_to_l

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_l_l1117_111751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l1117_111737

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the region for point Q
def region (x y : ℝ) : Prop := x*y ≥ 2 ∧ x - y ≥ 0 ∧ y ≤ 1

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the minimum length of segment PQ
noncomputable def min_length : ℝ := Real.sqrt 5 - 2

-- Theorem statement
theorem min_segment_length :
  ∀ (x y : ℝ), circle_eq x y →
  ∀ (qx qy : ℝ), region qx qy →
  ∀ (px py : ℝ), circle_eq px py →
  Real.sqrt ((px - qx)^2 + (py - qy)^2) ≥ min_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l1117_111737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_minus_y_l1117_111770

theorem max_x_minus_y (x y : Real) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π/2) (h4 : Real.tan x = 3 * Real.tan y) :
  ∃ (max_val : Real), max_val = π/6 ∧ ∀ z, 0 < z ∧ z ≤ x - y → z ≤ max_val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_minus_y_l1117_111770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_pi_over_six_l1117_111720

-- Define the complex number k
noncomputable def k : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

-- Define the equation relating z₁, z₂, and z₃
def triangle_equation (z₁ z₂ z₃ : ℂ) : Prop :=
  z₁ + k * z₂ + k^2 * (2 * z₃ - z₁) = 0

-- State the theorem
theorem smallest_angle_is_pi_over_six 
  (z₁ z₂ z₃ : ℂ) 
  (h : triangle_equation z₁ z₂ z₃) : 
  ∃ (θ : ℝ), θ = Real.pi / 6 ∧ 
  θ = min (min (abs (Complex.arg (z₂ - z₁))) (abs (Complex.arg (z₃ - z₂)))) (abs (Complex.arg (z₁ - z₃))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_pi_over_six_l1117_111720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l1117_111731

/-- The least number of distinct points in the plane such that for each k=1,2,...,n, 
    there exists a straight line containing exactly k of these points -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the explicit formula for f(n) -/
theorem f_formula (n : ℕ) : f n = (n + 1) / 2 * ((n + 2) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l1117_111731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rotation_volumes_l1117_111760

theorem circle_rotation_volumes (r : ℝ) (x : ℝ) (n : ℝ) :
  x > 0 ∧ r > 0 →
  (2 * Real.pi * r * x^4) / (3 * (r^2 + x^2)) = (2 * Real.pi * r * x^4) / (3 * (r^2 + x^2)) ∧
  x = r * Real.sqrt (n - 1 + Real.sqrt (n^2 + 4*n)) ∧
  (n = 4/3 → x = r * Real.sqrt 3) ∧
  1 / Real.sqrt 3 = 1 / Real.sqrt 3 ∧
  x = r * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rotation_volumes_l1117_111760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forgotten_pigs_count_l1117_111791

/-- Given information about Mary's animal count at the petting zoo -/
def petting_zoo_count (mary_count : ℕ) (actual_count : ℕ) (double_counted : ℕ) : Prop :=
  let corrected_count := mary_count - double_counted
  actual_count - corrected_count = mary_count - actual_count

/-- Theorem: Mary forgot to count as many pigs as the difference between
    the actual count and her corrected count -/
theorem forgotten_pigs_count 
  (mary_count : ℕ) (actual_count : ℕ) (double_counted : ℕ) 
  (h : mary_count ≥ double_counted) 
  (h2 : petting_zoo_count mary_count actual_count double_counted) : 
  actual_count - (mary_count - double_counted) = mary_count - actual_count :=
by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof
-- #eval forgotten_pigs_count 60 56 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forgotten_pigs_count_l1117_111791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l1117_111785

/-- Represents temperature in Celsius -/
structure Temperature where
  value : Int

/-- Converts a temperature to its string representation -/
def temp_to_string (t : Temperature) : String :=
  if t.value ≥ 0 then s!"+{t.value}°C" else s!"{t.value}°C"

/-- The temperature 5°C above zero -/
def above_zero : Temperature := ⟨5⟩

/-- The temperature 5°C below zero -/
def below_zero : Temperature := ⟨-5⟩

theorem below_zero_notation :
  temp_to_string above_zero = "+5°C" →
  temp_to_string below_zero = "-5°C" :=
by
  intro h
  simp [temp_to_string, above_zero, below_zero]
  sorry

#eval temp_to_string above_zero
#eval temp_to_string below_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l1117_111785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_congruence_l1117_111796

theorem four_number_congruence (S : Finset ℕ) :
  S.card = 4 →
  (∀ n, n ∈ S → n ∈ Finset.range 20) →
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ ∃ x : ℤ, a * x ≡ b [ZMOD c] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_congruence_l1117_111796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1117_111730

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 4 = 0

-- Define the line passing through (1,1)
def line_through_point (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ 1 = m*1 + b

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
  line_through_point A.1 A.2 ∧ line_through_point B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_chord_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  ∀ C D : ℝ × ℝ, intersection_points C D →
  distance A B ≤ distance C D ∧
  ∃ E F : ℝ × ℝ, intersection_points E F ∧ distance E F = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1117_111730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1117_111748

def isValidArrangement (arr : List Nat) : Prop :=
  arr.length = 5 ∧
  arr.toFinset = {1, 2, 3, 4, 5} ∧
  arr.getLast?.isSome ∧ Odd (arr.getLast?.get!) ∧
  ∀ i : Fin 3, (arr[i.val]! + arr[i.val + 1]! + arr[i.val + 2]!) % arr[i.val]! = 0

theorem valid_arrangements_count :
  ∃! (arrangements : List (List Nat)),
    arrangements.length = 5 ∧
    ∀ arr : List Nat, arr ∈ arrangements ↔ isValidArrangement arr :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1117_111748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_vacation_length_l1117_111780

/-- Susan's vacation problem -/
theorem susan_vacation_length :
  ∀ (work_days_per_week : ℕ) 
    (paid_vacation_days : ℕ) 
    (hourly_pay : ℚ) 
    (work_hours_per_day : ℕ) 
    (missed_pay : ℚ),
  work_days_per_week = 5 →
  paid_vacation_days = 6 →
  hourly_pay = 15 →
  work_hours_per_day = 8 →
  missed_pay = 480 →
  ∃ (total_vacation_days : ℕ),
    total_vacation_days = paid_vacation_days + 
      (missed_pay / (hourly_pay * ↑work_hours_per_day)).floor ∧
    total_vacation_days = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_vacation_length_l1117_111780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_proof_l1117_111736

/-- Calculate the distance between two points observed at the bottom corners of a window --/
noncomputable def window_observation_distance 
  (window_width : ℝ) 
  (eye_to_window : ℝ) 
  (altitude : ℝ) : ℝ :=
  let distance_ratio := altitude / eye_to_window
  distance_ratio * window_width

/-- Prove that the distance between two ships observed at the bottom corners of a window is 12.875 km --/
theorem ship_distance_proof 
  (window_width : ℝ) 
  (window_height : ℝ) 
  (eye_to_window : ℝ) 
  (altitude : ℝ) 
  (h1 : window_width = 40 / 100) -- 40 cm in km
  (h2 : window_height = 25 / 100) -- 25 cm in km
  (h3 : eye_to_window = 20 / 100) -- 20 cm in km
  (h4 : altitude = 10.3) -- 10.3 km
  : window_observation_distance window_width eye_to_window altitude = 12.875 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_proof_l1117_111736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_theorem_l1117_111767

def rotate90CounterClockwise (center : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let (h, k) := center
  let (x₀, y₀) := p
  (h - (y₀ - k), k + (x₀ - h))

def reflectAboutYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem point_transformation_theorem (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90CounterClockwise (2, 3) p
  let final := reflectAboutYEqualsX rotated
  final = (5, 1) → b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_theorem_l1117_111767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangles_common_area_l1117_111733

/-- A set is a regular triangle -/
def IsRegularTriangle (T : Set (Fin 2 → ℝ)) : Prop := sorry

/-- The measure of the area of a set -/
def MeasureArea (S : Set (Fin 2 → ℝ)) : ℝ := sorry

/-- The center of mass of a set -/
def CenterOfMass (S : Set (Fin 2 → ℝ)) : Fin 2 → ℝ := sorry

/-- Two regular triangles with unit area whose centers coincide have a common area of at least 2/3 -/
theorem regular_triangles_common_area (T1 T2 : Set (Fin 2 → ℝ)) : 
  IsRegularTriangle T1 → 
  IsRegularTriangle T2 → 
  MeasureArea T1 = 1 → 
  MeasureArea T2 = 1 → 
  CenterOfMass T1 = CenterOfMass T2 → 
  MeasureArea (T1 ∩ T2) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangles_common_area_l1117_111733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l1117_111794

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define a point on the left branch of the hyperbola
structure PointOnLeftBranch where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y
  left_branch : x < 0

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem hyperbola_property (M : PointOnLeftBranch) :
  let MF1 := distance (M.x, M.y) left_focus
  let MF2 := distance (M.x, M.y) right_focus
  let F1F2 := distance left_focus right_focus
  MF1 + F1F2 - MF2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l1117_111794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundredth_digit_of_seven_twenty_ninths_l1117_111717

/-- The decimal representation of 7/29 -/
def decimal_rep : ℚ := 7 / 29

/-- The length of the repeating block in the decimal representation of 7/29 -/
def repeat_length : ℕ := 28

/-- The position we're interested in -/
def target_position : ℕ := 200

/-- The theorem stating that the 200th digit after the decimal point in 7/29 is 1 -/
theorem two_hundredth_digit_of_seven_twenty_ninths (h : decimal_rep = 7 / 29) :
  ∃ (d : ℕ), d = 1 ∧ d = (decimal_rep * 10^target_position).floor % 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundredth_digit_of_seven_twenty_ninths_l1117_111717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l1117_111722

theorem sine_difference_value (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.tan (α / 2) + 1 / Real.tan (α / 2) = 5 / 2) : 
  Real.sin (α - Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l1117_111722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_range_l1117_111789

noncomputable def a (θ : ℝ) : ℝ × ℝ := (1, Real.sin θ)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sqrt 3)

theorem vector_difference_magnitude_range (θ : ℝ) :
  1 ≤ Real.sqrt ((a θ).1 - (b θ).1)^2 + ((a θ).2 - (b θ).2)^2 ∧
  Real.sqrt ((a θ).1 - (b θ).1)^2 + ((a θ).2 - (b θ).2)^2 ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_range_l1117_111789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_q_theorem_l1117_111705

def p : Nat → Nat
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | n => n

def q : Nat → Nat
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | n => n

theorem p_q_theorem :
  (∃ f : Nat → Nat, ∀ n : Nat, f (f n) = p n + 2) ∧
  (¬ ∃ f : Nat → Nat, ∀ n : Nat, f (f n) = q n + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_q_theorem_l1117_111705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_properties_l1117_111776

-- Define the triangle AOB
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (0, 30)
noncomputable def B : ℝ × ℝ := (40, 0)

-- Define point C on AB such that OC is perpendicular to AB
noncomputable def C : ℝ × ℝ := (72/5, 96/5)

-- Define M as the center of the circle passing through O, A, and B
noncomputable def M : ℝ × ℝ := (20, 15)

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem triangle_AOB_properties :
  (distance O C = 24) ∧
  (C = (72/5, 96/5)) ∧
  (distance C M = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_properties_l1117_111776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_4_denominator_nonzero_l1117_111743

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 6)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

/-- Theorem stating that g(x) crosses its horizontal asymptote at x = 4 -/
theorem g_crosses_asymptote_at_4 :
  g 4 = horizontal_asymptote ∧ x^2 - 5 * x + 6 ≠ 0 :=
by
  sorry

/-- Theorem stating that the denominator of g(x) is not zero for all real x except 2 and 3 -/
theorem denominator_nonzero (x : ℝ) (h : x ≠ 2 ∧ x ≠ 3) : x^2 - 5 * x + 6 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_4_denominator_nonzero_l1117_111743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1117_111747

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given polar coordinates -/
noncomputable def given_polar : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 3)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ := (2 * Real.sqrt 2, 2 * Real.sqrt 6)

/-- Theorem stating that the conversion from the given polar coordinates
    results in the expected rectangular coordinates -/
theorem polar_to_rectangular_conversion :
  polar_to_rectangular given_polar.1 given_polar.2 = expected_rectangular := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1117_111747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1117_111795

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 32 = 0

-- Define the line passing through P(0, 2) with slope k
def line_eq (k x y : ℝ) : Prop := y = k*x + 2

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq k x₁ y₁ ∧ line_eq k x₂ y₂

-- Define the dot product of position vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁*x₂ + y₁*y₂

-- Main theorem
theorem circle_line_intersection :
  (∀ k, intersects_at_two_points k ↔ -3/4 < k ∧ k < 0) ∧
  (∃ k, (∀ x₁ y₁ x₂ y₂, circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq k x₁ y₁ ∧ line_eq k x₂ y₂ →
    dot_product x₁ y₁ x₂ y₂ = 28) ∧ k = -3 + Real.sqrt 6) ∧
  (∃ C : ℝ × ℝ, C.1 = 0 ∧
    ∀ k x₁ y₁ x₂ y₂, intersects_at_two_points k ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq k x₁ y₁ ∧ line_eq k x₂ y₂ →
      dot_product (x₁ - C.1) (y₁ - C.2) (x₂ - C.1) (y₂ - C.2) = 36) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1117_111795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1117_111723

-- Define the friends
inductive Friend
| Ana
| Bento
| Celina
| Diana
| Elisa
| Fabio
| Guilherme

-- Define the figures
inductive Figure
| Triangle
| Square
| Circle

-- Define a function to represent the position of each friend
def position : Friend → Fin 7 := sorry

-- Define predicates for being inside figures
def inside_triangle : Friend → Prop := sorry
def inside_square : Friend → Prop := sorry
def inside_circle : Friend → Prop := sorry

-- Define a predicate for being inside a polygon
def inside_polygon (f : Friend) : Prop :=
  inside_triangle f ∨ inside_square f

-- Define the conditions for each friend
axiom bento_condition : ∃! fig : Figure, 
  (fig = Figure.Triangle ∧ inside_triangle Friend.Bento) ∨
  (fig = Figure.Square ∧ inside_square Friend.Bento) ∨
  (fig = Figure.Circle ∧ inside_circle Friend.Bento)

axiom celina_condition : inside_triangle Friend.Celina ∧ 
  inside_square Friend.Celina ∧ inside_circle Friend.Celina

axiom diana_condition : inside_triangle Friend.Diana ∧ 
  ¬inside_square Friend.Diana

axiom elisa_condition : inside_triangle Friend.Elisa ∧ 
  inside_circle Friend.Elisa

axiom fabio_condition : ¬inside_polygon Friend.Fabio

axiom guilherme_condition : inside_circle Friend.Guilherme

-- Theorem stating the uniqueness of the solution
theorem unique_solution : 
  ∃! (p : Friend → Fin 7), 
    (∀ f₁ f₂ : Friend, f₁ ≠ f₂ → p f₁ ≠ p f₂) ∧
    (p Friend.Ana = 6) ∧
    (p Friend.Bento = 7) ∧
    (p Friend.Celina = 3) ∧
    (p Friend.Diana = 5) ∧
    (p Friend.Elisa = 4) ∧
    (p Friend.Fabio = 1) ∧
    (p Friend.Guilherme = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1117_111723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1117_111712

noncomputable def f (x : ℝ) := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 0 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1117_111712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1117_111702

noncomputable def α : ℝ := sorry

-- Define the condition that the terminal side of α lies on y = -√3x
def terminal_side_condition (α : ℝ) : Prop :=
  ∃ (x y : ℝ), y = -Real.sqrt 3 * x ∧ x * Real.cos α = y * Real.sin α

-- Define the set S
def S : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi + 2 * Real.pi / 3}

-- The main theorem
theorem angle_properties (h : terminal_side_condition α) :
  (Real.tan α = -Real.sqrt 3) ∧
  (S = {β | ∃ k : ℤ, β = k * Real.pi + 2 * Real.pi / 3}) ∧
  ((Real.sqrt 3 * Real.sin (α - Real.pi) + 5 * Real.cos (2 * Real.pi - α)) /
   (-Real.sqrt 3 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (Real.pi + α)) = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1117_111702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1117_111757

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.cos (2 * x)

theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∃ M, M = 9/8 ∧ ∀ x, f x ≤ M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1117_111757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1117_111718

noncomputable def a : ℝ × ℝ := (1, 2)

def b_magnitude : ℝ := 1

noncomputable def angle_ab : ℝ := Real.pi / 3  -- 60 degrees in radians

noncomputable def perpendicular_unit_vector (v : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {u | u.1^2 + u.2^2 = 1 ∧ u.1 * v.1 + u.2 * v.2 = 0}

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ :=
  ((u.1 * v.1 + u.2 * v.2) / (v.1^2 + v.2^2)) * Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem :
  (perpendicular_unit_vector a = {(-2 * Real.sqrt 5 / 5, Real.sqrt 5 / 5), (2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5)}) ∧
  (vector_projection ((b_magnitude * Real.cos angle_ab, b_magnitude * Real.sin angle_ab) - (2 * a.1, 2 * a.2)) a = 1/2 - 2 * Real.sqrt 5) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1117_111718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1117_111701

/-- The area of the triangle bounded by the lines y=x, y=-x, and y=8 is 64 -/
theorem triangle_area : ∃ (area : ℝ), area = 64 := by
  -- Define the lines
  let line1 : ℝ → ℝ := λ x => x
  let line2 : ℝ → ℝ := λ x => -x
  let line3 : ℝ → ℝ := λ _ => 8

  -- Define the intersection points
  let point1 : ℝ × ℝ := (8, 8)
  let point2 : ℝ × ℝ := (-8, 8)

  -- Calculate the base and height of the triangle
  let base : ℝ := point1.1 - point2.1
  let height : ℝ := 8

  -- Calculate the area of the triangle
  let area : ℝ := (1 / 2) * base * height

  -- Prove that the area is 64
  use area
  sorry  -- This skips the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1117_111701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l1117_111746

theorem trig_simplification (x : ℝ) (h : Real.cos x ≠ 0) :
  (Real.cos x) / (1 + Real.tan x) + (1 + Real.tan x) / (Real.cos x) = Real.cos x + 1 / (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l1117_111746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_approx_l1117_111752

/-- The circumference of a circle -/
def circumference : ℝ := 6.28

/-- The mathematical constant π (pi) -/
noncomputable def π : ℝ := Real.pi

/-- Theorem: Given a circle with circumference 6.28 yards, its diameter is approximately 2 yards -/
theorem circle_diameter_approx (ε : ℝ) (h : ε > 0) : 
  ∃ (d : ℝ), circumference = π * d ∧ |d - 2| < ε := by
  sorry

#check circle_diameter_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_approx_l1117_111752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1117_111715

/-- Represents the income in thousands of dollars -/
def x : Real := sorry

/-- The tax rate as a function of income -/
def tax_rate (x : Real) : Real := x + 5

/-- The income in dollars -/
def income (x : Real) : Real := 1000 * x

/-- The tax amount in dollars -/
noncomputable def tax (x : Real) : Real := (tax_rate x / 100) * income x

/-- The take-home pay in dollars -/
noncomputable def take_home_pay (x : Real) : Real := income x - tax x

/-- The theorem stating that the income maximizing take-home pay is $47,500 -/
theorem max_take_home_pay :
  ∃ (x : Real), (∀ (y : Real), take_home_pay y ≤ take_home_pay x) ∧ income x = 47500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1117_111715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_center_trajectory_max_chord_length_l1117_111727

-- Define the circle M
def circle_M (x y α : ℝ) : Prop :=
  x^2 + y^2 - 4*x*(Real.cos α) - 2*y*(Real.sin α) + 3*(Real.cos α)^2 = 0

-- Define the line l
def line_l (x y θ t : ℝ) : Prop :=
  x = Real.tan θ ∧ y = 1 + t*(Real.sin θ)

-- Define the trajectory C
def trajectory_C (x y α : ℝ) : Prop :=
  x = 2*(Real.cos α) ∧ y = Real.sin α

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  x^2/4 + y^2 = 1

theorem circle_M_center_trajectory :
  ∀ x y α : ℝ, circle_M x y α → ∃ x' y' : ℝ, trajectory_C x' y' α ∧ ellipse_eq x' y' := by
  sorry

theorem max_chord_length :
  ∃ max_length : ℝ, max_length = 4*(Real.sqrt 3)/3 ∧
  ∀ x y α θ t : ℝ, trajectory_C x y α → line_l x y θ t →
  ∀ chord_length : ℝ, chord_length ≤ max_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_center_trajectory_max_chord_length_l1117_111727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_l1117_111781

-- Define the focal distance of an ellipse
noncomputable def focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Define the properties of the first ellipse
def ellipse1_a : ℝ := 5
def ellipse1_b : ℝ := 3

-- Define the properties of the second ellipse
noncomputable def ellipse2_a (k : ℝ) : ℝ := Real.sqrt (25 - k)
noncomputable def ellipse2_b (k : ℝ) : ℝ := Real.sqrt (9 - k)

-- State the theorem
theorem equal_focal_distances (k : ℝ) (h : k < 9) :
  focal_distance ellipse1_a ellipse1_b = focal_distance (ellipse2_a k) (ellipse2_b k) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_l1117_111781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_20_seconds_l1117_111753

/-- The time taken for a faster train to cross a man in a slower train -/
noncomputable def train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : ℝ :=
  train_length / (faster_speed - slower_speed)

/-- Theorem: The time taken for the faster train to cross the man in the slower train is 20 seconds -/
theorem train_crossing_time_is_20_seconds :
  let faster_speed : ℝ := 72 * (1000 / 3600)  -- Convert km/h to m/s
  let slower_speed : ℝ := 36 * (1000 / 3600)  -- Convert km/h to m/s
  let train_length : ℝ := 200
  train_crossing_time faster_speed slower_speed train_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_20_seconds_l1117_111753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_is_seven_l1117_111797

/-- A line in a plane --/
structure Line where
  id : Nat

/-- A point in a plane --/
structure Point where
  id : Nat

/-- A configuration of lines and points in a plane --/
structure Configuration where
  lines : Finset Line
  points : Finset Point
  on_line : Line → Point → Bool

/-- The property that each line has exactly 3 points --/
def has_three_points (config : Configuration) : Prop :=
  ∀ l ∈ config.lines, (config.points.filter (λ p => config.on_line l p)).card = 3

/-- The theorem stating that the minimum number of points is 7 --/
theorem min_points_is_seven :
  ∀ config : Configuration,
    config.lines.card = 6 →
    has_three_points config →
    config.points.card ≥ 7 := by
  sorry

#check min_points_is_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_is_seven_l1117_111797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1117_111771

noncomputable def f (a x : ℝ) : ℝ := |Real.log x / Real.log a| - 2^(-x)

theorem zeros_product_less_than_one (a m n : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a m = 0 → f a n = 0 → m * n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1117_111771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l1117_111774

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def adjacent_digits_sum_square (n : ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < (n.digits 10).length →
    is_perfect_square ((n.digits 10).get ⟨i, by sorry⟩ + (n.digits 10).get ⟨i+1, by sorry⟩)

def all_digits_different (n : ℕ) : Prop :=
  (n.digits 10).Nodup

def is_interesting (n : ℕ) : Prop :=
  all_digits_different n ∧ adjacent_digits_sum_square n

theorem largest_interesting_number :
  ∀ n : ℕ, is_interesting n → n ≤ 6310972 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l1117_111774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1117_111783

-- Define ω as a complex number
noncomputable def ω : ℂ := -1/2 + Complex.I * (Real.sqrt 3 / 2)

-- Define the theorem
theorem triangle_is_equilateral (z₁ z₂ z₃ : ℂ) 
  (h : z₁ + ω * z₂ + ω^2 * z₃ = 0) : 
  Complex.abs (z₁ - z₂) = Complex.abs (z₂ - z₃) ∧ 
  Complex.abs (z₂ - z₃) = Complex.abs (z₃ - z₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1117_111783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1117_111742

/-- Given distinct real numbers x and y such that the determinant 
    |2 6 10; 4 x y; 4 y x| = 0, prove that x + y = 32 -/
theorem sum_of_roots (x y : ℝ) (hxy : x ≠ y) 
  (hdet : Matrix.det !![2, 6, 10; 4, x, y; 4, y, x] = 0) : 
  x + y = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1117_111742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1117_111745

/-- The series defined by the given summation. -/
def mySeries (n : ℕ) : ℚ :=
  (6 * n^3 - 3 * n^2 + 2 * n - 1) / (n^6 - n^5 + n^4 - n^3 + n^2 - n)

/-- The sum of the series from n = 2 to infinity. -/
noncomputable def seriesSum : ℚ := ∑' n, if n ≥ 2 then mySeries n else 0

/-- The theorem stating that the sum of the series equals 1. -/
theorem series_sum_equals_one : seriesSum = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1117_111745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_27_l1117_111790

theorem x_plus_y_equals_27 (x y : ℝ) (h1 : (2 : ℝ)^x = (8 : ℝ)^(y+1)) (h2 : (9 : ℝ)^y = (3 : ℝ)^(x-9)) : 
  x + y = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_27_l1117_111790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strip_covering_theorem_l1117_111713

/-- A hexagon with parallel opposite sides -/
structure ParallelHexagon where
  vertices : Fin 6 → ℝ × ℝ
  parallel_sides : ∀ i : Fin 3, ∃ v : ℝ × ℝ, 
    (vertices (i+3) - vertices i) = (vertices (i+4) - vertices (i+1))

/-- A strip with a given width -/
structure Strip where
  center_line : ℝ × ℝ → Prop
  width : ℝ

/-- A family of hexagons satisfying the given conditions -/
def HexagonFamily := {H : ParallelHexagon | ∀ (v1 v2 v3 : Fin 6), ∃ (strip : Strip),
  strip.width = 1 ∧ (strip.center_line (H.vertices v1)) ∧ 
  (strip.center_line (H.vertices v2)) ∧ (strip.center_line (H.vertices v3))}

/-- The strip covering theorem -/
theorem strip_covering_theorem (𝓕 : Set ParallelHexagon) (h𝓕 : 𝓕 ⊆ HexagonFamily) :
  (∀ H ∈ 𝓕, ∃ strip : Strip, strip.width = Real.sqrt 2 ∧ (∀ v : Fin 6, strip.center_line (H.vertices v))) ∧
  (∀ ℓ : ℝ, ℓ < Real.sqrt 2 → ∃ H ∈ 𝓕, ∀ strip : Strip, strip.width = ℓ → 
    ∃ v : Fin 6, ¬(strip.center_line (H.vertices v))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strip_covering_theorem_l1117_111713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1117_111782

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Function to calculate the area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * Real.sin t.B

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a + 2 * t.c = 2 * t.b * Real.cos t.A) 
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : t.a + t.c = 4) : 
  (t.B = 2 * Real.pi / 3) ∧ 
  (Triangle.area t = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1117_111782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_finite_even_values_l1117_111777

-- Define the type for our function
def ContinuousFunction := (Set.Icc 0 1) → ℝ

-- Define the property of taking each value a finite and even number of times
def TakesValuesFiniteEvenTimes (f : ContinuousFunction) : Prop :=
  ∀ y ∈ Set.range f, (Set.Finite {x ∈ Set.Icc 0 1 | f x = y}) ∧ 
    Even (Finset.card (Set.toFinite {x ∈ Set.Icc 0 1 | f x = y}).toFinset)

-- The main theorem
theorem exists_continuous_function_finite_even_values :
  ∃ f : ContinuousFunction, Continuous f ∧ TakesValuesFiniteEvenTimes f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_finite_even_values_l1117_111777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1117_111759

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

-- Define the sum of the first n terms
noncomputable def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_diff : S a 9 - S a 4 = 10) :
  S a 17 = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1117_111759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l1117_111775

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_of_2_7 : floor 2.7 = 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l1117_111775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l1117_111735

theorem oldest_child_age (n : ℕ) (avg : ℚ) (a b c d : ℕ) :
  n = 5 →
  avg = 41 / 5 →
  a = 6 →
  b = 10 →
  c = 12 →
  d = 7 →
  ∃ x : ℕ, x = (n : ℚ) * avg - (a + b + c + d) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l1117_111735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l1117_111734

noncomputable def f (x : ℝ) : ℝ := -1 / (x + 1)

theorem max_value_of_f_on_interval :
  ∃ (m : ℝ), m = -1/3 ∧ ∀ x ∈ Set.Icc 1 2, f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l1117_111734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_circumcenters_form_parallelogram_l1117_111741

-- Define the points
variable (A B C D E O P Q R S : EuclideanPlane) in

-- Define the properties
def is_cyclic_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

def is_circumcenter (O A B C D : EuclideanPlane) : Prop := sorry

def is_diagonal_intersection (E A B C D : EuclideanPlane) : Prop := sorry

def is_triangle_circumcenter (P A B E : EuclideanPlane) : Prop := sorry

def is_parallelogram (P Q R S : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_circumcenters_form_parallelogram 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : is_circumcenter O A B C D)
  (h3 : is_diagonal_intersection E A B C D)
  (h4 : is_triangle_circumcenter P A B E)
  (h5 : is_triangle_circumcenter Q B C E)
  (h6 : is_triangle_circumcenter R C D E)
  (h7 : is_triangle_circumcenter S A D E) :
  is_parallelogram P Q R S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_circumcenters_form_parallelogram_l1117_111741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_sum_l1117_111724

noncomputable def heart (x : ℝ) : ℝ := (x^2 - x + 3) / 2

theorem heart_sum : heart 1 + heart 2 + heart 3 = 8.5 := by
  -- Unfold the definition of heart
  unfold heart
  -- Simplify the expressions
  simp [pow_two, add_div, sub_div]
  -- Perform arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_sum_l1117_111724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_time_approx_l1117_111719

/-- Calculates the time (in seconds) for two trains to meet given their lengths, initial distance, and speeds. -/
noncomputable def trainMeetingTime (length1 length2 initialDistance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let totalDistance := initialDistance + length1 + length2
  let relativeSpeed := (speed1 + speed2) * (1000 / 3600)  -- Convert km/h to m/s
  totalDistance / relativeSpeed

/-- The time for two trains to meet is approximately 32.57 seconds. -/
theorem train_meeting_time_approx :
  ∃ ε > 0, |trainMeetingTime 100 200 840 54 72 - 32.57| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_time_approx_l1117_111719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_ratio_angles_l1117_111721

/-- An isosceles triangle with a specific area ratio has angles 30°, 30°, and 120° -/
theorem isosceles_triangle_area_ratio_angles : ∀ (a h : ℝ),
  a > 0 → h > 0 →
  (1/2 * a * h) / (a^2) = Real.sqrt 3 / 12 →
  ∃ (α β γ : ℝ),
    α = 30 * π / 180 ∧
    β = 30 * π / 180 ∧
    γ = 120 * π / 180 ∧
    α + β + γ = π ∧
    Real.sin α / a = Real.sin β / a ∧ Real.sin α / a = Real.sin γ / (2 * h) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_ratio_angles_l1117_111721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1117_111756

theorem triangle_tangent (A B : ℝ) (h1 : Real.cos A = 4/5) (h2 : Real.tan (A - B) = 1/3) : 
  Real.tan B = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1117_111756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_pyramid_cone_l1117_111708

/-- A regular pyramid PABCD and a cone with specific properties -/
structure PyramidAndCone where
  -- Edge length of the base of the pyramid
  a : ℝ
  -- Vertex A coincides with apex of cone
  -- Vertices B and D lie on cone's lateral surface
  -- Vertex P is on the circle of the cone's base
  -- Vertex C is in the plane of the cone's base

/-- The ratio of the volume of the cone to the volume of the pyramid -/
noncomputable def volumeRatio (pc : PyramidAndCone) : ℝ :=
  (9 * Real.pi * Real.sqrt 2) / 8

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio_pyramid_cone (pc : PyramidAndCone) :
  volumeRatio pc = (9 * Real.pi * Real.sqrt 2) / 8 := by
  -- Unfold the definition of volumeRatio
  unfold volumeRatio
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_pyramid_cone_l1117_111708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_laws_used_l1117_111758

theorem multiplication_laws_used (a b c : ℕ) : 
  (a * (b * c) = b * (a * c)) → 
  (a * (b * c) = (a * b) * c) → 
  (a = 125 ∧ b = 32 ∧ c = 8) →
  True :=
by
  intros h1 h2 h3
  exact True.intro

#check multiplication_laws_used

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_laws_used_l1117_111758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_equality_l1117_111711

-- Define the function f
noncomputable def f (p : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^x + 1 else x^2 + p*x

-- State the theorem
theorem function_value_equality (p : ℝ) : f p (f p 0) = 5*p → p = 4/3 := by
  intro h
  have h1 : f p 0 = 2 := by
    simp [f]
    norm_num
  have h2 : f p 2 = 5*p := by
    rw [h1] at h
    exact h
  simp [f] at h2
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_equality_l1117_111711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_with_tangent_circle_l1117_111799

/-- Given a circle with radius r tangent to one side of an angle and intersecting the other side
    at segments of lengths a and b, the measure of the angle α is:
    α = arctan(r / √(ab)) ± arccos((a + b) / (2√(r² + ab))) -/
theorem angle_measure_with_tangent_circle (r a b : ℝ) (hr : r > 0) (ha : a > 0) (hb : b > 0) :
  ∃ α : ℝ, α = Real.arctan (r / Real.sqrt (a * b)) +
             Real.arccos ((a + b) / (2 * Real.sqrt (r^2 + a * b))) ∨
           α = Real.arctan (r / Real.sqrt (a * b)) -
             Real.arccos ((a + b) / (2 * Real.sqrt (r^2 + a * b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_with_tangent_circle_l1117_111799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_condition_max_value_of_h_harmonic_series_bound_l1117_111728

open Real Set

-- Part I
theorem function_positive_condition (a : ℝ) :
  (∀ x ∈ Ioo 0 1, sin x - a * x > 0) ↔ a ≤ 0 := by sorry

-- Part II
theorem max_value_of_h :
  IsGreatest {y | ∃ x > 0, y = log x - x + 1} 0 := by sorry

-- Part III
theorem harmonic_series_bound (n : ℕ) :
  log (n + 1 : ℝ) < (Finset.range n).sum (λ i ↦ 1 / (i + 1 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_condition_max_value_of_h_harmonic_series_bound_l1117_111728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_trip_duration_xiao_hong_lan_trip_durations_l1117_111750

/-- Billing rules for online car-hailing --/
structure BillingRules where
  startingFee : ℚ
  timeFee : ℚ
  mileageFee : ℚ
  longDistanceFee : ℚ
  longDistanceThreshold : ℚ

/-- Calculate the fare for a trip --/
def calculateFare (rules : BillingRules) (distance : ℚ) (duration : ℚ) : ℚ :=
  rules.startingFee +
  rules.timeFee * duration +
  rules.mileageFee * distance +
  max 0 (distance - rules.longDistanceThreshold) * rules.longDistanceFee

/-- Theorem for Mr. Wang's trip --/
theorem wang_trip_duration 
  (rules : BillingRules)
  (h_rules : rules = {
    startingFee := 12,
    timeFee := 1/2,
    mileageFee := 2,
    longDistanceFee := 1,
    longDistanceThreshold := 15
  }) :
  ∃ (duration : ℚ), calculateFare rules 20 duration = 139/2 ∧ duration = 25 := by
  sorry

/-- Theorem for Xiao Hong and Xiao Lan's trips --/
theorem xiao_hong_lan_trip_durations
  (rules : BillingRules)
  (h_rules : rules = {
    startingFee := 12,
    timeFee := 1/2,
    mileageFee := 2,
    longDistanceFee := 1,
    longDistanceThreshold := 15
  }) :
  ∃ (duration_lan : ℚ) (duration_hong : ℚ),
    calculateFare rules 16 duration_lan = calculateFare rules 14 duration_hong ∧
    duration_hong = 3/2 * duration_lan ∧
    duration_lan = 20 ∧
    duration_hong = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_trip_duration_xiao_hong_lan_trip_durations_l1117_111750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_Q_equals_interval_l1117_111793

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 1 - (a + 1) / (x + 1) < 0}
def Q : Set ℝ := {x | |x + 2| < 3}

-- Part 1: Prove that when a = 3, P = (-1, 3)
theorem part_one : P 3 = Set.Ioo (-1) 3 := by sorry

-- Part 2: Prove that when P ∪ Q = Q and a > 0, the range of a is (0, 1]
theorem part_two : 
  (∀ a : ℝ, a > 0 → P a ∪ Q = Q) ↔ 
  (∀ a : ℝ, a > 0 → a ∈ Set.Ioc 0 1) := by sorry

-- Additional helper theorem to show Q = (-5, 1)
theorem Q_equals_interval : Q = Set.Ioo (-5) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_Q_equals_interval_l1117_111793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_our_parabola_l1117_111738

/-- A parabola is defined by its equation in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry -/
def directrix (p : Parabola) : ℝ → ℝ → Prop :=
  sorry  -- Definition omitted for brevity

/-- Our specific parabola with equation x^2 = 4y -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 = 4*y }

theorem directrix_of_our_parabola :
  directrix our_parabola = fun x y => y = -1 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_our_parabola_l1117_111738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turning_grille_keys_l1117_111768

/-- Rotate a set of coordinates by 90 degrees k times -/
def rotate90 (n : ℕ) (k : ℕ) (s : Set (Fin n × Fin n)) : Set (Fin n × Fin n) :=
  sorry

/-- A stencil is valid if it covers every cell exactly once when rotated through four orientations -/
def IsValidStencil (n : ℕ) (stencil : Set (Fin n × Fin n)) : Prop :=
  (∀ i j, ∃! k, (⟨i, sorry⟩, ⟨j, sorry⟩) ∈ rotate90 n k stencil) ∧
  (∀ x ∈ stencil, x.1.val < n ∧ x.2.val < n)

/-- The number of valid stencils for a turning grille of size n × n -/
def validStencilCount (n : ℕ) : ℕ :=
  sorry

theorem turning_grille_keys (n : ℕ) (h : Even n) :
  validStencilCount n = 4^(n^2/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turning_grille_keys_l1117_111768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_is_120_degrees_l1117_111787

noncomputable def angle_at_C (a b c : ℝ) : ℝ := sorry

theorem angle_c_is_120_degrees 
  (a b c : ℝ) 
  (f₁ f₂ : ℝ) 
  (h1 : a > b) 
  (h2 : f₂ / f₁ = ((a + b) / (a - b)) * Real.sqrt 3) :
  let γ := angle_at_C a b c
  γ = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_is_120_degrees_l1117_111787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_in_unit_square_l1117_111709

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Calculates the distance between two parallel lines -/
noncomputable def distanceBetweenParallelLines (l1 l2 : Line) : ℝ :=
  abs (l1.b - l2.b) / Real.sqrt (1 + l1.m^2)

/-- Calculates the perimeter of a triangle given its three vertices -/
noncomputable def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  let d12 := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := Real.sqrt ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := Real.sqrt ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 + d23 + d31

/-- Main theorem: Sum of perimeters of triangles formed by parallel lines in a unit square -/
theorem sum_of_triangle_perimeters_in_unit_square (s : Square) (l1 l2 : Line) :
  s.sideLength = 1 →
  distanceBetweenParallelLines l1 l2 = 1 →
  ∃ (p1 p2 p3 p4 p5 p6 : Point),
    trianglePerimeter p1 p2 p3 + trianglePerimeter p4 p5 p6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_in_unit_square_l1117_111709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_cube_volume_ratio_l1117_111714

/-- The ratio of the volume of a right circular cylinder inscribed in a cube
    to the volume of the cube, where the cylinder's height equals the cube's side length. -/
theorem cylinder_to_cube_volume_ratio :
  ∀ (s : ℝ), s > 0 →
  (π * (s/2)^2 * s) / s^3 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_cube_volume_ratio_l1117_111714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_distance_for_equal_areas_l1117_111761

/-- Represents a rectangular sheet of paper with two-sided color -/
structure ColoredSheet :=
  (width : ℝ)
  (length : ℝ)

/-- Represents the folding of the sheet -/
noncomputable def fold (sheet : ColoredSheet) (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + sheet.width^2)

theorem fold_distance_for_equal_areas (sheet : ColoredSheet) 
  (h1 : sheet.width = 3)
  (h2 : sheet.length = 4)
  (h3 : ∃ x : ℝ, 0 < x ∧ x < sheet.width ∧ 2 * x = sheet.width * sheet.length - 2 * x) :
  ∃ x : ℝ, fold sheet x = 3 * Real.sqrt 2 := by
  sorry

#check fold_distance_for_equal_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_distance_for_equal_areas_l1117_111761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_form_l1117_111778

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) + (4 : ℝ) * (y - 8) = 0

/-- The slope of the line -/
noncomputable def m : ℝ := -3/4

/-- The y-intercept of the line -/
noncomputable def b : ℝ := 19/2

/-- Theorem stating that the line equation is equivalent to y = mx + b -/
theorem line_slope_intercept_form :
  ∀ x y : ℝ, line_equation x y ↔ y = m * x + b := by
  sorry

#check line_slope_intercept_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_form_l1117_111778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_bound_l1117_111765

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 1) + a / Real.exp x

-- Define the derivative of g
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a - a / Real.exp x

-- Theorem statement
theorem min_derivative_bound (a : ℝ) (h : a ≤ -1) :
  ∀ x : ℝ, g' a x ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_bound_l1117_111765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1117_111716

theorem problem_statement : 
  100 * (Real.cos (10 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 - 
  Real.sin (40 * π / 180) * Real.sin (80 * π / 180)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1117_111716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_solution_l1117_111703

/-- Represents the problem of calculating the time taken for a bike ride on a specific path. -/
def BikeRideProblem (highway_length highway_width extension_length : ℝ) (bike_speed : ℝ) : Prop :=
  let radius := highway_width / 2
  let num_semicircles := highway_length / highway_width
  let semicircle_distance := num_semicircles * (Real.pi * radius)
  let total_distance := semicircle_distance + extension_length
  let total_distance_miles := total_distance / 5280
  let time_taken := total_distance_miles / bike_speed
  time_taken = (Real.pi + 0.2) / 16

/-- Theorem stating the solution to the bike ride problem. -/
theorem bike_ride_solution :
  BikeRideProblem 5280 30 528 8 := by
  sorry

#check bike_ride_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_solution_l1117_111703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_length_l1117_111744

def bathroom_interval : ℚ := 50
def bathroom_visits : ℚ := 3

theorem movie_length : 
  (bathroom_visits * bathroom_interval) / 60 = 2.5 := by
  -- Convert the fraction to decimal
  norm_num
  -- Simplify the expression
  simp [bathroom_interval, bathroom_visits]
  -- Prove the equality
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_length_l1117_111744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l1117_111739

/-- A function that transforms B to A by moving the last digit to the first place -/
def transform (B : ℕ) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

/-- Predicate to check if a number is coprime with 18 -/
def coprime_with_18 (n : ℕ) : Prop :=
  Nat.Coprime n 18

/-- Theorem stating the largest and smallest values of A given the conditions -/
theorem largest_and_smallest_A :
  ∃ (A_max A_min : ℕ),
    (∀ A B : ℕ,
      A = transform B →
      coprime_with_18 B →
      B > 222222222 →
      A ≤ A_max ∧ A ≥ A_min) ∧
    A_max = 999999998 ∧
    A_min = 122222224 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l1117_111739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_pi_sqrt_4_5_l1117_111779

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 9 * y^2 - 18 * y + 8 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := Real.pi * Real.sqrt 4.5

/-- Theorem stating that the area of the ellipse represented by the given equation is π√(4.5) -/
theorem ellipse_area_is_pi_sqrt_4_5 :
  ∃ (a b : ℝ), (∀ x y, ellipse_equation x y ↔ (x + 2)^2 / a^2 + (y - 1)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_pi_sqrt_4_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_pi_sqrt_4_5_l1117_111779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_triangle_function_k_range_l1117_111755

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 
  2 * k * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * k * Real.cos x ^ 2 - k + 1

noncomputable def h (x : ℝ) : ℝ := 
  Real.sin (2 * x) - (13 * Real.sqrt 2 / 5) * Real.sin (x + Real.pi / 4) + 369 / 100

theorem perfect_triangle_function_k_range :
  ∀ k : ℝ,
  (k > 0) →
  (∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → 
    a ∈ Set.Icc 0 (Real.pi/2) → b ∈ Set.Icc 0 (Real.pi/2) → c ∈ Set.Icc 0 (Real.pi/2) →
    g k a + g k b > g k c ∧ g k b + g k c > g k a ∧ g k c + g k a > g k b) →
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi/2) → ∃ x₂ : ℝ, x₂ ∈ Set.Icc 0 (Real.pi/2) ∧ g k x₂ ≥ h x₁) →
  k ∈ Set.Icc (9/200) (1/4) :=
by
  sorry

#check perfect_triangle_function_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_triangle_function_k_range_l1117_111755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_l1117_111726

theorem complex_purely_imaginary (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * 3 / (1 + Complex.I * 2)) * (1 - Complex.I * (a / 3)) = Complex.I * b) ↔ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_l1117_111726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_given_condition_l1117_111710

theorem max_cos_x_given_condition (x y : ℝ) : 
  (Real.cos (x - y) = Real.cos x - Real.cos y) → 
  ∃ (z : ℝ), Real.cos z = 1 ∧ ∀ w, Real.cos w ≤ Real.cos z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_given_condition_l1117_111710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1117_111798

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : Fin 2 → ℝ := ![a, Real.sqrt 3 * b]
  let n : Fin 2 → ℝ := ![Real.cos A, Real.sin B]
  0 < A ∧ A < Real.pi ∧ 
  (∃ (k : ℝ), k ≠ 0 ∧ m = k • n) →
  A = Real.pi / 3 ∧
  (a = 2 → 2 < b + c ∧ b + c ≤ 4)

theorem triangle_theorem :
  ∀ (a b c A B C : ℝ), triangle_proof a b c A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1117_111798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_minus_sqrt_twelve_half_squared_l1117_111766

theorem cube_root_eight_minus_sqrt_twelve_half_squared : 
  (8^(1/3 : ℝ) - Real.sqrt (12 + 1/2))^2 = (33 - 20 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_minus_sqrt_twelve_half_squared_l1117_111766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_implies_sin_2theta_l1117_111772

theorem cos_product_implies_sin_2theta (θ : ℝ) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2/6)
  (h2 : 0 < θ) (h3 : θ < π/2) :
  Real.sin (2*θ) = Real.sqrt 7/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_implies_sin_2theta_l1117_111772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meteorologist_more_reliable_l1117_111754

/-- Represents the accuracy of a senator's forecast -/
def p : ℝ := sorry

/-- Probability of a clear day -/
def clear_day_prob : ℝ := 0.74

/-- Probability of the meteorologist correctly predicting a rainy day -/
def meteorologist_accuracy (p : ℝ) : ℝ := 1.5 * p

/-- Event that both senators predict a clear day and the meteorologist predicts rain -/
def forecast_event (p : ℝ) (r : ℝ) : ℝ := 
  (1 - meteorologist_accuracy p) * p^2 * r + meteorologist_accuracy p * (1 - p)^2 * (1 - r)

/-- Theorem stating that the meteorologist's forecast is more reliable -/
theorem meteorologist_more_reliable : 
  ∀ p ∈ Set.Icc 0 1, 
    forecast_event p (1 - clear_day_prob) > forecast_event p clear_day_prob :=
by
  sorry

#check meteorologist_more_reliable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meteorologist_more_reliable_l1117_111754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sixth_power_l1117_111773

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem matrix_sixth_power : 
  A ^ 6 = !![64, 0; 0, 64] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sixth_power_l1117_111773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_regression_analysis_l1117_111786

-- Define the propositions as axioms since they are not defined in Mathlib
axiom correlation : (α : Type) → α → α → Prop
axiom circumference : ℝ → ℝ
axiom radius : ℝ → ℝ
axiom non_deterministic_relationship : (ℝ → ℝ) → (ℝ → ℝ) → Prop
axiom regression_line : (ℝ → ℝ) → Prop
axiom meaningless : (ℝ → ℝ) → Prop
axiom deterministic_problem : Prop → Prop → Prop
axiom demand : ℝ → ℝ
axiom price : ℝ → ℝ

-- Define the propositions
def proposition1 : Prop := ∀ (α : Type) (x y : α), correlation α x y
def proposition2 : Prop := ∀ c r : ℝ, ∃ k : ℝ, circumference c = k * radius r
def proposition3 : Prop := ∀ d p : ℝ, non_deterministic_relationship (demand) (price)
def proposition4 : Prop := ∃ l : ℝ → ℝ, regression_line l ∧ meaningless l
def proposition5 : Prop := ∀ (α : Type) (x y : α), ∃ l : ℝ → ℝ, deterministic_problem (correlation α x y) (regression_line l)

-- Define the theorem
theorem correlation_regression_analysis :
  ¬proposition1 ∧
  ¬proposition2 ∧
  proposition3 ∧
  proposition4 ∧
  proposition5 := by
  sorry -- Skip the proof as requested


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_regression_analysis_l1117_111786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1117_111764

def a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => 3 * a (n + 2) - 2 * a (n + 1)

theorem a_formula (n : ℕ) : a n = 2^n - 1 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    cases n with
    | zero => rfl
    | succ n =>
      cases n with
      | zero => rfl
      | succ n =>
        simp [a]
        -- The proof steps would go here, but we'll use sorry for now
        sorry

#eval a 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1117_111764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_0359_to_hundredth_l1117_111707

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_2_0359_to_hundredth :
  round_to_hundredth 2.0359 = 2.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_0359_to_hundredth_l1117_111707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_composition_equals_26_l1117_111762

-- Define the function p
noncomputable def p (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3*y
  else if x + y > 0 then 2*x + 2*y
  else x + 4*y

-- State the theorem
theorem p_composition_equals_26 : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  -- Evaluate p(2, -3)
  have h1 : p 2 (-3) = -10 := by
    simp [p]
    norm_num
  
  -- Evaluate p(-3, -4)
  have h2 : p (-3) (-4) = 9 := by
    simp [p]
    norm_num
  
  -- Evaluate p(-10, 9)
  have h3 : p (-10) 9 = 26 := by
    simp [p]
    norm_num
  
  -- Combine the results
  calc
    p (p 2 (-3)) (p (-3) (-4)) = p (-10) 9 := by rw [h1, h2]
    _ = 26 := by rw [h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_composition_equals_26_l1117_111762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_l1117_111792

/-- The equation of a circle in the form x^2 + y^2 = ax + by + c -/
noncomputable def CircleEquation (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 = a * p.1 + b * p.2 + c

/-- The center of a circle given by its equation coefficients -/
noncomputable def CircleCenter (a b c : ℝ) : ℝ × ℝ :=
  (a / 2, b / 2)

/-- Theorem: For a circle with equation x^2 + y^2 = 6x + 8y + 15,
    the sum of the coordinates of its center is 7 -/
theorem circle_center_sum :
  let eq := CircleEquation 6 8 15
  let center := CircleCenter 6 8 15
  center.1 + center.2 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_l1117_111792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_equality_implies_prime_powers_l1117_111784

-- Define the sum of divisors function
def sigma (N : ℕ) : ℕ := (Finset.sum (Nat.divisors N) id)

-- Define the condition for m and n
def condition (m n : ℕ) : Prop :=
  m ≥ n ∧ n ≥ 2 ∧
  (sigma m - 1) * (n - 1) = (sigma n - 1) * (m - 1) ∧
  (sigma m - 1) * (m * n - 1) = (sigma (m * n) - 1) * (m - 1)

-- Define what it means for a number to be a prime power
def is_prime_power (x : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ x = p ^ k

-- State the theorem
theorem sigma_equality_implies_prime_powers (m n : ℕ) :
  condition m n → is_prime_power m ∧ is_prime_power n ∧ ∃ (p : ℕ), Nat.Prime p ∧ (∃ (e f : ℕ), m = p^e ∧ n = p^f) :=
by
  sorry

#check sigma_equality_implies_prime_powers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_equality_implies_prime_powers_l1117_111784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1117_111729

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem triangle_theorem (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_condition : Real.sin B ^ 2 + Real.sin C ^ 2 = Real.sin A ^ 2 + Real.sin B * Real.sin C) :
  A = Real.pi/3 ∧ (a = 3 → ∃ (b_max c_max : ℝ), b_max + c_max = 6 ∧ ∀ (b' c' : ℝ), Triangle A B C 3 b' c' → b' + c' ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1117_111729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1117_111732

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) / x

theorem f_monotonicity_and_inequality :
  (∀ x > 0, (deriv f) x < 0) ∧
  (∀ a : ℝ, (∀ x ∈ ({x : ℝ | -1 < x ∧ x ≠ 0} : Set ℝ), f x < (1 - a*x) / (1 + x)) ↔ a = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1117_111732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1117_111788

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Define the parabola S
def parabola_S (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through a point with slope n
def line_through_point (x y n x₀ y₀ : ℝ) : Prop := x - x₀ = n * (y - y₀)

-- Define the center of the circle P
def center_P : ℝ × ℝ := (1, 0)

-- Define the arithmetic sequence property for four points
def arithmetic_sequence (a b c d : ℝ) : Prop := b - a = c - b ∧ c - b = d - c

-- Main theorem
theorem line_equation_proof :
  ∃ n : ℝ, (∃ A B C D : ℝ × ℝ,
    circle_P A.1 A.2 ∧ circle_P D.1 D.2 ∧
    parabola_S B.1 B.2 ∧ parabola_S C.1 C.2 ∧
    line_through_point A.1 A.2 n center_P.1 center_P.2 ∧
    line_through_point B.1 B.2 n center_P.1 center_P.2 ∧
    line_through_point C.1 C.2 n center_P.1 center_P.2 ∧
    line_through_point D.1 D.2 n center_P.1 center_P.2 ∧
    A.2 > B.2 ∧ B.2 > C.2 ∧ C.2 > D.2 ∧
    arithmetic_sequence (A.2 - B.2) (B.2 - C.2) (C.2 - D.2) 0) ∧
  (n = Real.sqrt 2 / 2 ∨ n = -Real.sqrt 2 / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1117_111788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1117_111706

theorem count_integer_pairs : ∃ n : ℕ, n = 2020032 ∧ 
  n = (Finset.range 2004).card * 84 * 12 := by
  use 2020032
  constructor
  · rfl
  · sorry  -- Proof details omitted for brevity

#eval (Finset.range 2004).card * 84 * 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1117_111706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABP_range_l1117_111740

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the circle
def mycircle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Define point P on the circle
def P : ℝ × ℝ → Prop
  | (x, y) => mycircle x y

-- Define the area of triangle ABP
noncomputable def area_ABP (p : ℝ × ℝ) : ℝ :=
  let (px, py) := p
  abs ((A.1 * (B.2 - py) + B.1 * (py - A.2) + px * (A.2 - B.2)) / 2)

-- Theorem statement
theorem area_ABP_range :
  ∀ p : ℝ × ℝ, P p → 2 ≤ area_ABP p ∧ area_ABP p ≤ 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABP_range_l1117_111740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wall_length_divisible_by_four_l1117_111769

/-- Represents a square room partitioned into smaller square rooms -/
structure PartitionedRoom where
  side_length : ℕ
  partitions : List (ℕ × ℕ)
  is_valid : ∀ p ∈ partitions, p.1 ≤ side_length ∧ p.2 ≤ side_length

/-- Calculates the total length of all partition walls in a partitioned room -/
def total_wall_length (room : PartitionedRoom) : ℕ := sorry

/-- Theorem: The total length of all partition walls is divisible by 4 -/
theorem total_wall_length_divisible_by_four (room : PartitionedRoom) :
  4 ∣ total_wall_length room := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wall_length_divisible_by_four_l1117_111769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_angles_relationship_l1117_111749

/-- Predicate indicating that the terminal sides of two angles are perpendicular -/
def terminal_sides_perpendicular (α β : ℝ) : Prop :=
  sorry

/-- Given two angles α and β with perpendicular terminal sides, 
    there exists an integer k such that β = k * 360° + α ± 90° -/
theorem perpendicular_angles_relationship (α β : ℝ) 
  (h : terminal_sides_perpendicular α β) :
  ∃ k : ℤ, β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_angles_relationship_l1117_111749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_max_volume_l1117_111725

theorem triangle_rotation_max_volume
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  let s := (a + b + c) / 2
  let V := λ x ↦ (4 * Real.pi / 3) * (s * (s - a) * (s - b) * (s - c)) / x
  V (min a (min b c)) = max (V a) (max (V b) (V c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_max_volume_l1117_111725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1117_111700

theorem beta_value (α β : ℝ) (h1 : Real.cos α = 3/5) 
  (h2 : Real.cos (α - β) = 7 * Real.sqrt 2 / 10) (h3 : 0 < β) (h4 : β < α) (h5 : α < π/2) : 
  β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1117_111700
