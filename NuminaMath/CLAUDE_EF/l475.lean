import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sides_visibility_probability_l475_47532

/-- The radius of a circle concentric with and outside a regular hexagon,
    such that the probability of seeing exactly two entire sides of the hexagon
    from a random point on the circle is 1/4 -/
noncomputable def radius_for_two_sides_visibility (side_length : ℝ) : ℝ :=
  (2 * side_length * Real.sqrt 3) / Real.sqrt (2 + Real.sqrt 2)

/-- The probability of seeing exactly two entire sides of a regular hexagon
    from a random point on a concentric circle -/
noncomputable def probability_of_seeing_two_sides (p : ℝ × ℝ) (r : ℝ) (side_length : ℝ) : ℝ :=
  sorry

/-- Predicate to check if a point is on a circle with given radius -/
def on_circle (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1)^2 + (p.2)^2 = r^2

theorem two_sides_visibility_probability (r : ℝ) (side_length : ℝ) :
  side_length = 3 →
  (∀ p : ℝ × ℝ, on_circle p r → 
    (probability_of_seeing_two_sides p r side_length = 1/4) ↔ 
    r = radius_for_two_sides_visibility side_length) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sides_visibility_probability_l475_47532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integers_average_l475_47572

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (w + x + y + z : ℚ) / 4 = 5 →
  max a b - min a b ≤ max w x - min w x ∧
  max a b - min a b ≤ max w y - min w y ∧
  max a b - min a b ≤ max w z - min w z ∧
  max a c - min a c ≤ max w x - min w x ∧
  max a c - min a c ≤ max w y - min w y ∧
  max a c - min a c ≤ max w z - min w z ∧
  max a d - min a d ≤ max w x - min w x ∧
  max a d - min a d ≤ max w y - min w y ∧
  max a d - min a d ≤ max w z - min w z ∧
  max b c - min b c ≤ max w x - min w x ∧
  max b c - min b c ≤ max w y - min w y ∧
  max b c - min b c ≤ max w z - min w z ∧
  max b d - min b d ≤ max w x - min w x ∧
  max b d - min b d ≤ max w y - min w y ∧
  max b d - min b d ≤ max w z - min w z ∧
  max c d - min c d ≤ max w x - min w x ∧
  max c d - min c d ≤ max w y - min w y ∧
  max c d - min c d ≤ max w z - min w z →
  ((max (min a b) (min (min a c) (min a d)) + 
    min (max a b) (max (max a c) (max a d))) : ℚ) / 2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integers_average_l475_47572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l475_47509

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem stating that g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l475_47509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_square_equal_area_l475_47514

theorem equilateral_triangle_square_equal_area (s h : ℝ) : 
  s > 0 → 
  ((Real.sqrt 3 / 4) * s^2 = s^2) → 
  (h = (Real.sqrt 3 / 2) * s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_square_equal_area_l475_47514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l475_47503

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α = 2/3)
  (h2 : Real.cos β = -3/4)
  (h3 : α ∈ Set.Ioo (π/2) π)
  (h4 : β ∈ Set.Ioo π (3*π/2)) :
  Real.cos (α - β) = (3*Real.sqrt 5 - 2*Real.sqrt 7) / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l475_47503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l475_47577

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/2) :
  a = 1/2 ∧ Set.range (f a) = Set.Ioo 0 2 := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l475_47577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_b_equals_two_common_tangent_analysis_l475_47565

noncomputable section

def f (a b x : ℝ) : ℝ := a * (x - 1 / x) - b * Real.log x
def g (x : ℝ) : ℝ := x^2

theorem tangent_perpendicular_implies_b_equals_two :
  ∀ b : ℝ, (∀ x : ℝ, x > 0 → DifferentiableAt ℝ (f 1 b) x) →
  (deriv (f 1 b) 1 = 0) → b = 2 := by sorry

theorem common_tangent_analysis :
  ∀ a : ℝ,
  (a ≤ 0 → ¬∃ x : ℝ, x > 0 ∧ f a 2 x = g x ∧ deriv (f a 2) x = deriv g x) ∧
  (a > 0 → ∃! n : ℕ, n = 2 ∧ 
    ∃ s : Finset ℝ, s.card = n ∧ 
    ∀ a' ∈ s, ∃ x : ℝ, x > 0 ∧ f a' 2 x = g x ∧ deriv (f a' 2) x = deriv g x) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_b_equals_two_common_tangent_analysis_l475_47565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_half_l475_47520

noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

theorem inverse_of_log_half :
  Function.LeftInverse f log_half ∧ Function.RightInverse f log_half := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_half_l475_47520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l475_47521

-- Define the triangle BCD
structure Triangle where
  B : Point
  C : Point
  D : Point

-- Define the angles
def angle_ABD : ℝ := 120
def angle_BCD : ℝ := 54

-- Define the properties
def ABC_is_straight_line : Prop := sorry
def angle_ABD_is_exterior_angle (triangle : Triangle) : Prop := sorry
def angle_CBD (triangle : Triangle) : ℝ := sorry

-- Define the theorem
theorem exterior_angle_theorem (triangle : Triangle) :
  ABC_is_straight_line →
  angle_ABD_is_exterior_angle triangle →
  angle_CBD triangle = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l475_47521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_line_line_equation_l475_47560

/-- A line passing through the origin -/
structure LineThroughOrigin where
  slope : ℝ

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : LineThroughOrigin) : ℝ :=
  let (x, y) := p
  abs (l.slope * x - y) / Real.sqrt (1 + l.slope ^ 2)

theorem equal_distance_line (l : LineThroughOrigin) :
  distanceToLine (1, 4) l = distanceToLine (3, 2) l →
  (l.slope = -1 ∨ l.slope = 3/2) := by
  sorry

theorem line_equation (l : LineThroughOrigin) :
  (l.slope = -1 ∨ l.slope = 3/2) →
  (∀ x y : ℝ, y = -x ∨ 2*y = 3*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_line_line_equation_l475_47560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_cone_apex_angle_l475_47518

/-- Represents a cone with its lateral surface unfolded into a semicircle -/
structure SemicircleCone where
  radius : ℝ
  lateral_surface_is_semicircle : True

/-- The apex angle of a cone in radians -/
noncomputable def apex_angle (cone : SemicircleCone) : ℝ := Real.pi / 3

theorem semicircle_cone_apex_angle (cone : SemicircleCone) :
  apex_angle cone = Real.pi / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_cone_apex_angle_l475_47518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_percentage_in_seawater_l475_47558

/-- The percentage of salt in seawater, given the volume of seawater and the volume of salt obtained after evaporation. -/
noncomputable def salt_percentage (seawater_volume : ℝ) (salt_volume : ℝ) : ℝ :=
  (salt_volume / seawater_volume) * 100

/-- Theorem stating that the percentage of salt in seawater is 20% when 2 liters of seawater yields 400 ml of salt. -/
theorem salt_percentage_in_seawater :
  let seawater_volume : ℝ := 2000  -- 2 liters in milliliters
  let salt_volume : ℝ := 400       -- 400 ml of salt
  salt_percentage seawater_volume salt_volume = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_percentage_in_seawater_l475_47558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_numbers_l475_47547

theorem equality_of_numbers (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a + b + 1 / (a * b) = c + d + 1 / (c * d))
  (h2 : 1 / a + 1 / b + a * b = 1 / c + 1 / d + c * d) :
  ∃ (x y : ℝ), x ∈ ({a, b, c, d} : Set ℝ) ∧ y ∈ ({a, b, c, d} : Set ℝ) ∧ x ≠ y ∧ x = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_numbers_l475_47547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_value_l475_47557

/-- The coefficient of x^2 in the expansion of (x^2-3x+2)^4 -/
def coefficient_x_squared : ℕ := 248

/-- Theorem stating that the coefficient of x^2 in the expansion of (x^2-3x+2)^4 is 248 -/
theorem coefficient_x_squared_value : coefficient_x_squared = 248 := by
  rfl

#eval coefficient_x_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_value_l475_47557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_mars_axis_ratio_l475_47527

/-- Kepler's third law constant -/
noncomputable def keplerConstant (a : ℝ) (T : ℝ) : ℝ := a^3 / T^2

/-- Orbital period of Earth -/
def earthPeriod : ℝ := 5

/-- Orbital period of Mars -/
def marsPeriod : ℝ := 9

/-- Semi-major axis of Earth's orbit -/
noncomputable def earthAxis : ℝ := sorry

/-- Semi-major axis of Mars' orbit -/
noncomputable def marsAxis : ℝ := sorry

/-- Kepler's third law holds for both Earth and Mars -/
axiom kepler_law : keplerConstant earthAxis earthPeriod = keplerConstant marsAxis marsPeriod

/-- The ratio of Earth's semi-major axis to Mars' semi-major axis -/
noncomputable def axisRatio : ℝ := earthAxis / marsAxis

theorem earth_mars_axis_ratio : axisRatio = (25 / 81) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_mars_axis_ratio_l475_47527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l475_47567

/-- The percentage increase in weekly earnings -/
noncomputable def percentage_increase (initial_earnings new_earnings : ℝ) : ℝ :=
  (new_earnings - initial_earnings) / initial_earnings * 100

/-- Theorem: John's percentage increase in weekly earnings is 100% -/
theorem johns_earnings_increase (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 120) :
  percentage_increase initial_earnings new_earnings = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l475_47567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_after_fall_initial_velocity_for_melting_l475_47584

-- Constants
noncomputable def initial_temp : ℝ := 20
noncomputable def fall_height : ℝ := 100
noncomputable def specific_heat : ℝ := 0.0315
noncomputable def mech_equiv_heat : ℝ := 425
noncomputable def melting_point : ℝ := 335
noncomputable def gravity : ℝ := 9.80

-- Theorems to prove
theorem temp_after_fall :
  ∃ (temp : ℝ), abs (temp - (initial_temp + gravity * fall_height / (mech_equiv_heat * specific_heat))) < 0.1 := by
  sorry

theorem initial_velocity_for_melting :
  ∃ (v : ℝ), abs (v - Real.sqrt (2 * ((melting_point - initial_temp) * mech_equiv_heat * specific_heat - gravity * fall_height))) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_after_fall_initial_velocity_for_melting_l475_47584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l475_47581

-- Define the polynomial P(x)
def P (a : ℕ → ℤ) (n : ℕ) (x : ℤ) : ℤ :=
  (Finset.range (n + 1)).sum (λ i ↦ a i * x^(n - i))

-- State the theorem
theorem no_integer_roots (a : ℕ → ℤ) (n : ℕ) :
  (P a n 0 % 2 = 1) →  -- P(0) is odd
  (P a n 1 % 2 = 1) →  -- P(1) is odd
  ∀ k : ℤ, P a n k ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l475_47581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_a_is_one_condition_for_inequality_l475_47592

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x

-- Part 1: Extreme value when a = 1
theorem extreme_value_when_a_is_one :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
    (∀ (x : ℝ), x > 0 → f 1 x₀ ≤ f 1 x) ∧
    f 1 x₀ = 1/2 ∧ x₀ = 1 := by
  sorry

-- Part 2: Condition for f(x) ≥ x
theorem condition_for_inequality :
  ∀ (a : ℝ), (∀ (x : ℝ), x > 0 → f a x ≥ x) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_a_is_one_condition_for_inequality_l475_47592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sudoku_bottom_row_l475_47559

/-- Represents a 6x6 Sudoku-like grid --/
def Grid := Matrix (Fin 6) (Fin 6) ℕ

/-- Checks if a number appears exactly once in each row --/
def validRows (g : Grid) : Prop :=
  ∀ i : Fin 6, (Finset.univ.image (g i)).card = 6

/-- Checks if a number appears exactly once in each column --/
def validColumns (g : Grid) : Prop :=
  ∀ j : Fin 6, (Finset.univ.image (fun i => g i j)).card = 6

/-- Represents the 6 regions of the grid --/
def Regions : List (Finset (Fin 6 × Fin 6)) := sorry

/-- Checks if a number appears exactly once in each region --/
def validRegions (g : Grid) : Prop :=
  ∀ r ∈ Regions, (r.image (fun p => g p.1 p.2)).card = 6

/-- The main theorem stating that the first four numbers in the bottom row are 2, 4, 1, 3 --/
theorem sudoku_bottom_row (g : Grid) 
  (hrows : validRows g) 
  (hcols : validColumns g) 
  (hregs : validRegions g) :
  g 5 0 = 2 ∧ g 5 1 = 4 ∧ g 5 2 = 1 ∧ g 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sudoku_bottom_row_l475_47559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_product_bound_l475_47524

noncomputable def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 4 - (1/4)^x

noncomputable def f₂ (x : ℝ) : ℝ := Real.log x / Real.log (1/4) - (1/4)^x

theorem zero_points_product_bound (x₁ x₂ : ℝ) 
  (h₁ : f₁ x₁ = 0) (h₂ : f₂ x₂ = 0) : 0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_product_bound_l475_47524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_specific_meeting_time_l475_47587

/-- Two people traveling towards each other on a route will meet at a specific time -/
theorem meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance > 0) (h2 : speed1 > 0) (h3 : speed2 > 0) :
  (distance / (speed1 + speed2)) * (speed1 + speed2) = distance :=
by sorry

/-- The specific case of two people meeting on a 600 km route -/
theorem specific_meeting_time :
  (600 : ℝ) / (70 + 80) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_specific_meeting_time_l475_47587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_equals_double_side_length_l475_47516

/-- Helper function to calculate the altitude of an isosceles triangle -/
noncomputable def triangle_altitude (base side : ℝ) : ℝ :=
  Real.sqrt (side^2 - (base/2)^2)

/-- Given a square with side length s and an isosceles triangle with base s and two equal sides √2s,
    if the area of the triangle equals the area of the square,
    then the altitude of the triangle to the base is 2s -/
theorem triangle_altitude_equals_double_side_length (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_base := s
  let triangle_side := Real.sqrt 2 * s
  let triangle_area := (1/2) * triangle_base * (triangle_altitude triangle_base triangle_side)
  triangle_area = square_area →
  triangle_altitude triangle_base triangle_side = 2 * s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_equals_double_side_length_l475_47516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1215_l475_47508

theorem divisors_of_1215 : (Finset.filter (λ x : ℕ => 1215 % x = 0) (Finset.range 1216)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1215_l475_47508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_oclock_angle_l475_47563

/-- Represents a clock with hour and minute hands -/
structure Clock where
  hours : ℕ
  minutes : ℕ

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- Calculate the angle of the hour hand from 12 o'clock position -/
def hour_hand_angle (c : Clock) : ℕ :=
  (c.hours * full_circle) / clock_hours

/-- Calculate the angle of the minute hand from 12 o'clock position -/
def minute_hand_angle (c : Clock) : ℕ :=
  (c.minutes * full_circle) / 60

/-- Calculate the smaller angle between hour and minute hands -/
def smaller_angle (c : Clock) : ℕ :=
  min (Int.natAbs (hour_hand_angle c - minute_hand_angle c)) 
      (full_circle - Int.natAbs (hour_hand_angle c - minute_hand_angle c))

/-- Theorem: The smaller angle between hour and minute hands at 8:00 is 120° -/
theorem eight_oclock_angle : 
  ∀ c : Clock, c.hours = 8 ∧ c.minutes = 0 → smaller_angle c = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_oclock_angle_l475_47563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l475_47555

theorem tan_one_condition (x : ℝ) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 4) → Real.tan x = 1 ∧
  ¬(Real.tan x = 1 → ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l475_47555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_and_single_point_l475_47598

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ _ => 0  -- Add a case for 0 to satisfy the missing case error
| 1 => λ x => Real.sqrt (2 - x)
| (n + 1) => λ x => f n (Real.sqrt ((n + 2)^2 - x))

-- Define the domain of a function
def domain (f : ℝ → ℝ) := {x : ℝ | ∃ y, f x = y}

-- State the theorem
theorem largest_nonempty_domain_and_single_point :
  (∃ N : ℕ, N > 0 ∧
    (∀ n > N, domain (f n) = ∅) ∧
    (domain (f N) ≠ ∅) ∧
    (∃ c : ℝ, domain (f N) = {c})) ∧
  (let N := 2; let c := 9;
    N > 0 ∧
    (∀ n > N, domain (f n) = ∅) ∧
    (domain (f N) ≠ ∅) ∧
    domain (f N) = {c}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_and_single_point_l475_47598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_independent_of_r_l475_47528

noncomputable def area_quadrilateral (a b c d : ℂ) : ℝ := sorry

theorem quadrilateral_area_independent_of_r (r : ℝ) (hr : r ≠ 0) : 
  let roots := {z : ℂ | r^4 * z^4 + (10*r^6 - 2*r^2) * z^2 - 16*r^5 * z + (9*r^8 + 10*r^4 + 1) = 0}
  ∃ (a b c d : ℂ), roots = {a, b, c, d} ∧ 
  area_quadrilateral a b c d = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_independent_of_r_l475_47528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_20_percent_l475_47504

/-- Represents the markup and pricing structure of a store selling turtleneck sweaters -/
structure SweaterPricing where
  C : ℝ  -- Cost price
  M : ℝ  -- Initial markup percentage

/-- Calculates the final selling price after all markups and discounts -/
noncomputable def finalPrice (p : SweaterPricing) : ℝ :=
  p.C * (1 + p.M / 100) * 1.25 * 0.80

/-- Theorem stating that the initial markup percentage is 20% -/
theorem initial_markup_is_20_percent (p : SweaterPricing) :
  finalPrice p = 1.20 * p.C → p.M = 20 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_20_percent_l475_47504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l475_47525

/-- Given two squares, one with perimeter 8 cm and another with area 64 cm²,
    prove that the distance between points A and B is √136 cm. -/
theorem distance_between_points (small_square_perimeter : ℝ) (large_square_area : ℝ)
  (h1 : small_square_perimeter = 8)
  (h2 : large_square_area = 64) :
  let small_side := small_square_perimeter / 4
  let large_side := Real.sqrt large_square_area
  let horizontal_distance := small_side + large_side
  let vertical_distance := large_side - small_side
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = Real.sqrt 136 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l475_47525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_reaches_corner_l475_47580

/-- Represents a grid with dimensions m and n -/
structure Grid where
  m : Nat
  n : Nat

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Checks if a position is a corner of the grid -/
def is_corner (g : Grid) (p : Position) : Prop :=
  (p.x = 0 ∨ p.x = g.m) ∧ (p.y = 0 ∨ p.y = g.n)

/-- Represents the diagonal path on the grid -/
def diagonal_path (g : Grid) : Set Position :=
  sorry

/-- Theorem stating that the diagonal path will reach a corner -/
theorem diagonal_path_reaches_corner (g : Grid) (start : Position) :
  g.m = 101 ∧ g.n = 200 ∧ is_corner g start →
  ∃ (end_pos : Position), end_pos ∈ diagonal_path g ∧ is_corner g end_pos ∧ end_pos ≠ start :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_reaches_corner_l475_47580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l475_47549

/-- The curve C' in polar coordinates -/
def curve_C' (ρ θ : ℝ) : Prop :=
  ρ^2 * (1 + 3 * Real.sin θ^2) = 4

/-- Two points A and B on the curve C' -/
structure PointOnC' where
  ρ : ℝ
  θ : ℝ
  on_curve : curve_C' ρ θ

/-- The distance from O to a line passing through two points -/
noncomputable def distance_to_line (A B : PointOnC') : ℝ :=
  (A.ρ * B.ρ) / Real.sqrt (A.ρ^2 + B.ρ^2)

/-- The theorem to be proved -/
theorem distance_is_constant (A B : PointOnC') 
  (perpendicular : A.θ = B.θ - π/2 ∨ A.θ = B.θ + π/2) :
  distance_to_line A B = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l475_47549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l475_47562

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x + 1)^4
def equation2 (x y : ℂ) : Prop := x * y + y = 5

-- Define a solution as a pair satisfying both equations
def is_solution (p : ℂ × ℂ) : Prop :=
  equation1 p.fst p.snd ∧ equation2 p.fst p.snd

-- State the theorem
theorem system_solutions :
  ∃ (solutions : Finset (ℂ × ℂ)),
    (∀ p, p ∈ solutions ↔ is_solution p) ∧
    solutions.card = 5 ∧
    (∃! p, p ∈ solutions ∧ p.fst.im = 0 ∧ p.snd.im = 0) ∧
    (∃ p₁ p₂ p₃ p₄, p₁ ∈ solutions ∧ p₂ ∈ solutions ∧ p₃ ∈ solutions ∧ p₄ ∈ solutions ∧
      p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
      p₁.fst.im ≠ 0 ∧ p₂.fst.im ≠ 0 ∧ p₃.fst.im ≠ 0 ∧ p₄.fst.im ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l475_47562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_v4_l475_47582

def recurrence_sequence (v : ℕ → ℚ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem find_v4 (v : ℕ → ℚ) (h1 : recurrence_sequence v) (h2 : v 2 = 6) (h3 : v 5 = 58) :
  v 4 = 24.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_v4_l475_47582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_walking_problem_l475_47554

/-- Given two friends walking from opposite ends of a 50 km trail, 
    with one friend's speed 35% faster than the other, 
    prove that the distance walked by the faster friend when they meet 
    is approximately 28.72 km. -/
theorem friend_walking_problem (v : ℝ) (h : v > 0) : 
  let trail_length : ℝ := 50
  let speed_ratio : ℝ := 1.35
  let d_P : ℝ := (speed_ratio * v * trail_length) / (v + speed_ratio * v)
  ‖d_P - 28.72‖ < 0.01 := by
  sorry

#eval (1.35 * 50) / 2.35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_walking_problem_l475_47554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_cost_increase_is_three_percent_l475_47579

/-- Represents the percentage increase in employment costs during 1993 -/
noncomputable def employment_cost_increase : ℝ := 0.035

/-- Represents the percentage increase in fringe-benefit costs during 1993 -/
noncomputable def fringe_benefit_increase : ℝ := 0.055

/-- Represents the fraction of employment costs that were fringe-benefits at the start of 1993 -/
noncomputable def fringe_benefit_fraction : ℝ := 0.20

/-- Calculates the percentage increase in salary costs during 1993 -/
noncomputable def salary_cost_increase : ℝ :=
  (employment_cost_increase - fringe_benefit_fraction * fringe_benefit_increase) /
  (1 - fringe_benefit_fraction)

theorem salary_cost_increase_is_three_percent :
  salary_cost_increase = 0.03 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_cost_increase_is_three_percent_l475_47579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_greater_than_three_l475_47564

theorem inequality_holds_iff_a_greater_than_three (a : ℝ) :
  (∀ θ : ℝ, θ ∈ Set.Icc 0 (π / 2) →
    Real.sin (2 * θ) - (2 * Real.sqrt 2 + Real.sqrt 2 * a) * Real.sin (θ + π / 4) -
    2 * Real.sqrt 2 / Real.cos (θ - π / 4) > -3 - 2 * a) ↔
  a > 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_greater_than_three_l475_47564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_properties_l475_47586

/-- Represents a point on the ellipse x²/9 + y² = 1 above the x-axis -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 9 + y^2 = 1
  above_x_axis : y > 0

/-- The perpendicular bisector equation for two points on the ellipse -/
def perpendicular_bisector (A B : EllipsePoint) : ℝ → ℝ → Prop :=
  λ x y ↦ 9*x - 2*y - 8 = 0

/-- The y-intercept of the line through two points -/
noncomputable def y_intercept (A B : EllipsePoint) : ℝ :=
  sorry  -- Definition of y-intercept

theorem ellipse_points_properties 
  (A B : EllipsePoint) 
  (sum_x : A.x + B.x = 2) 
  (sum_y : A.y + B.y = 1) : 
  (perpendicular_bisector A B = λ x y ↦ 9*x - 2*y - 8 = 0) ∧ 
  (∀ C D : EllipsePoint, y_intercept C D ≥ 2/3) ∧
  (∃ E F : EllipsePoint, y_intercept E F = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_properties_l475_47586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subway_takers_exists_max_subway_takers_l475_47505

-- Define the total number of employees
def total_employees : ℕ := 48

-- Define the function to calculate subway takers
def subway_takers (part_time full_time : ℕ) : ℕ :=
  (((1 : ℚ) / 3 * part_time).floor + ((1 : ℚ) / 4 * full_time).floor).toNat

-- Theorem statement
theorem max_subway_takers :
  ∀ (part_time full_time : ℕ),
    part_time + full_time = total_employees →
    part_time > 0 →
    full_time > 0 →
    subway_takers part_time full_time ≤ 15 :=
by sorry

-- The maximum number of subway takers
def max_subway : ℕ := 15

-- Theorem that there exists a configuration achieving the maximum
theorem exists_max_subway_takers :
  ∃ (part_time full_time : ℕ),
    part_time + full_time = total_employees ∧
    part_time > 0 ∧
    full_time > 0 ∧
    subway_takers part_time full_time = max_subway :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subway_takers_exists_max_subway_takers_l475_47505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_sum_zero_l475_47576

def letter_value (n : ℕ) : ℤ :=
  match n % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | _ => -2

def letter_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'l' => 12
  | 'g' => 7
  | 'e' => 5
  | 'b' => 2
  | 'r' => 18
  | _ => 0

def word_value (word : String) : ℤ :=
  word.data.map (fun c => letter_value (letter_position c)) |>.sum

theorem algebra_sum_zero : word_value "algebra" = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_sum_zero_l475_47576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l475_47550

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions of the problem
def ValidTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧  -- Angles are positive
  t.A + t.B + t.C = Real.pi ∧    -- Sum of angles is π (180°)
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c    -- Sides are positive

-- Theorem statement
theorem side_b_range (t : Triangle) 
  (valid : ValidTriangle t) 
  (angle_B : t.B = Real.pi / 3)  -- B = 60°
  (side_sum : t.a + t.c = 1) :   -- a + c = 1
  0.5 ≤ t.b ∧ t.b < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l475_47550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_nine_greater_than_factorial_eight_l475_47517

theorem divisors_of_factorial_nine_greater_than_factorial_eight :
  (Finset.filter (fun d => d > Nat.factorial 8 ∧ Nat.factorial 9 % d = 0) 
    (Finset.range (Nat.factorial 9 + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_nine_greater_than_factorial_eight_l475_47517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l475_47542

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_evaluation :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6.2 : ℝ) = 16.2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l475_47542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l475_47553

-- Define the triangles and their properties
structure Triangle :=
  (P Q R : ℝ)

def similar (t1 t2 : Triangle) : Prop :=
  t1.P / t2.P = t1.Q / t2.Q ∧ t1.Q / t2.Q = t1.R / t2.R

-- Define the theorem
theorem similar_triangles_side_length 
  (PQR STU : Triangle)
  (h_similar : similar PQR STU)
  (h_QR : PQR.R = 30)
  (h_TU : STU.R = 10)  -- Changed from STU.U to STU.R
  (h_PQ : PQR.Q = 18) :
  STU.P = 6 :=  -- Changed from STU.T to STU.P
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l475_47553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_2008_l475_47538

-- Define the hyperbola C
def C (x y : ℝ) : Prop := y^2 - x^2 = 1

-- Define the sequence of points on the x-axis
def P : ℕ → ℝ → ℝ
| 0, x₀ => x₀
| n+1, x₀ => sorry  -- Actual implementation would go here

-- Define the property that P₀ = P₂₀₀₈
def periodic_sequence (x₀ : ℝ) : Prop := P 0 x₀ = P 2008 x₀

-- Define N as the number of starting positions satisfying the property
noncomputable def N : ℕ := (Set.range periodic_sequence).ncard

-- Theorem statement
theorem N_mod_2008 : N % 2008 = 254 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_2008_l475_47538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l475_47583

/-- A circle C with center at polar coordinates (2, π/4) passing through the pole has the polar equation ρ = 2√2 (sin θ + cos θ) -/
theorem circle_polar_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (ρ θ : ℝ) :
  (center = (2, π/4)) →
  ((0, 0) ∈ C) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - Real.sqrt 2)^2 + (y - Real.sqrt 2)^2 = 4) →
  ((ρ * Real.cos θ, ρ * Real.sin θ) ∈ C ↔ ρ = 2 * Real.sqrt 2 * (Real.sin θ + Real.cos θ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l475_47583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l475_47571

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 4 = 0

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_from_circle_center_to_line :
  ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧ 
  (∀ (x y : ℝ), circle_eq x y → (x - x₀)^2 + (y - y₀)^2 ≤ (x - 1)^2 + (y - 2)^2) ∧
  distance_point_to_line x₀ y₀ 3 4 4 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l475_47571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l475_47500

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- The focus of the parabola y^2 = 8x -/
def focus : Point := ⟨2, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (x : ℝ) : ℝ :=
  |p.x - x|

theorem parabola_focus_distance (M : Point) (h1 : M ∈ Parabola) 
    (h2 : distanceToVerticalLine M (-3) = 6) : 
  distance M focus = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l475_47500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l475_47541

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1/2) * x + m else x - Real.log x

theorem f_monotone_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l475_47541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l475_47593

/-- Calculates the speed of a vehicle in km/h given the wheel radius and revolutions per minute -/
noncomputable def calculate_speed (radius : ℝ) (rpm : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let distance_per_minute := circumference * rpm / 100000  -- Convert cm to km
  distance_per_minute * 60  -- Convert to km/h

/-- Theorem stating that a vehicle with a wheel radius of 70 cm and 250.22747952684256 rpm 
    travels at approximately 66.04 km/h -/
theorem bus_speed_calculation (ε : ℝ) (h : ε > 0) :
  ∃ (speed : ℝ), abs (speed - calculate_speed 70 250.22747952684256) < ε ∧ 
                  abs (speed - 66.04) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l475_47593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l475_47535

/-- Definition of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 64 = 1

/-- Definition of the foci coordinates -/
def foci : ℝ × ℝ := (10, 0)

/-- Distance from a point to a focus -/
noncomputable def dist_to_focus (x y : ℝ) (focus : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2)

/-- Theorem statement -/
theorem hyperbola_focus_distance (x y : ℝ) :
  is_on_hyperbola x y →
  (dist_to_focus x y foci = 15 ∨ dist_to_focus x y (-foci) = 15) →
  (dist_to_focus x y foci = 27 ∨ dist_to_focus x y (-foci) = 27) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l475_47535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l475_47573

noncomputable def f (ω : ℝ) (b : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x - Real.pi / 6) + b

noncomputable def g (x : ℝ) : ℝ := f 2 (-1/2) (x - Real.pi / 12)

theorem function_range_theorem (ω : ℝ) (b : ℝ) (m : ℝ) :
  (ω > 0) →
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → f ω b x ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 3) → g x - 3 ≤ m ∧ m ≤ g x + 3) →
  (m ∈ Set.Icc (-2) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l475_47573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_exponents_l475_47544

open Complex

theorem sum_of_complex_exponents :
  let z₁ := exp (11 * π * I / 60)
  let z₂ := exp (23 * π * I / 60)
  let z₃ := exp (35 * π * I / 60)
  let z₄ := exp (47 * π * I / 60)
  let z₅ := exp (59 * π * I / 60)
  arg (z₁ + z₂ + z₃ + z₄ + z₅) = 7 * π / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_exponents_l475_47544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l475_47551

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x

-- Define the point of tangency
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem tangent_triangle_area :
  let tangent_line (x : ℝ) := -x - 1
  let x_intercept := -1
  let y_intercept := -1
  (1/2 : ℝ) * (x_intercept - 0) * (y_intercept - 0) = 1/2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l475_47551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_polar_equation_rho_2_l475_47552

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  /-- The equation of the circle in polar coordinates -/
  equation : ℝ → ℝ
  /-- The area of the circle -/
  area : ℝ

/-- The area of a circle with polar coordinate equation ρ = 2 is 4π -/
theorem area_of_circle_polar_equation_rho_2 :
  ∀ (c : PolarCircle), (∀ θ, c.equation θ = 2) → c.area = 4 * Real.pi :=
by
  intro c h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_polar_equation_rho_2_l475_47552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_identity_l475_47506

/-- Double factorial of an odd number -/
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (n + 2) * double_factorial n

theorem factorial_identity (n : ℕ) :
  Nat.factorial (2 * n) / Nat.factorial n = 2^n * double_factorial (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_identity_l475_47506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l475_47533

theorem circle_center_and_radius : 
  let equation := fun (x y : ℝ) => x^2 + y^2 + 3*x - 2*y - 1
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-3/2, 1) ∧ 
    radius = Real.sqrt 17 / 2 ∧
    ∀ (x y : ℝ), equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l475_47533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_relationship_l475_47519

-- Define a, b, and c as definite integrals
noncomputable def a : ℝ := ∫ x in (0 : ℝ)..1, x
noncomputable def b : ℝ := ∫ x in (0 : ℝ)..1, x^2
noncomputable def c : ℝ := ∫ x in (0 : ℝ)..1, Real.sqrt x

-- Theorem statement
theorem integral_relationship : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_relationship_l475_47519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_problem_l475_47512

theorem systematic_sampling_problem (total_students sample_size : ℕ) 
  (h1 : total_students = 600) 
  (h2 : sample_size = 60) 
  (h3 : 6 ∈ (Finset.range total_students).filter (λ x => x % 10 = 6)) 
  (h4 : 16 ∈ (Finset.range total_students).filter (λ x => x % 10 = 6)) 
  (h5 : 26 ∈ (Finset.range total_students).filter (λ x => x % 10 = 6)) : 
  36 ∈ (Finset.range total_students).filter (λ x => x % 10 = 6) :=
by
  sorry

#check systematic_sampling_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_problem_l475_47512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l475_47561

/-- The distance between two points in polar coordinates --/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

/-- Theorem: The distance between P(1, π/6) and Q(2, π/2) in polar coordinates is √3 --/
theorem distance_P_Q : polar_distance 1 (π/6) 2 (π/2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l475_47561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_12_seconds_l475_47595

/-- Represents the length of each train in meters -/
noncomputable def train_length : ℝ := 250

/-- Represents the speed of the faster train in km/hr -/
noncomputable def faster_train_speed : ℝ := 45

/-- Represents the speed of the slower train in km/hr -/
noncomputable def slower_train_speed : ℝ := 30

/-- Converts km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * (5/18)

/-- Calculates the relative speed of the trains in m/s -/
noncomputable def relative_speed : ℝ := km_per_hr_to_m_per_s (faster_train_speed + slower_train_speed)

/-- Calculates the time taken for the slower train to pass the driver of the faster train -/
noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem time_to_pass_approx_12_seconds :
  abs (time_to_pass - 12) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_12_seconds_l475_47595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_max_ratio_l475_47539

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the foci of the hyperbola
noncomputable def hyperbola_foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (-c, 0, c, 0)

-- Define a point on the right branch of the hyperbola
def right_branch_point (a b : ℝ) (x y : ℝ) : Prop :=
  hyperbola a b x y ∧ x > 0

-- Define the distance ratio
noncomputable def distance_ratio (a b x y : ℝ) : ℝ :=
  let (f1x, f1y, f2x, f2y) := hyperbola_foci a b
  let pf2 := Real.sqrt ((x - f2x)^2 + (y - f2y)^2)
  let pf1 := Real.sqrt ((x - f1x)^2 + (y - f1y)^2)
  pf2 / (pf1^2)

-- State the theorem
theorem hyperbola_ellipse_max_ratio 
  (a b : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  ellipse x y →
  right_branch_point a b x y →
  (∀ (x' y' : ℝ), right_branch_point a b x' y' → 
    distance_ratio a b x y ≥ distance_ratio a b x' y') →
  distance_ratio a b x y = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_max_ratio_l475_47539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l475_47556

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℚ
  alcoholPercentage : ℚ

/-- Calculates the new alcohol concentration when mixing two vessels -/
def mixConcentration (v1 v2 : Vessel) (totalVolume : ℚ) : ℚ :=
  let totalAlcohol := v1.capacity * v1.alcoholPercentage / 100 + v2.capacity * v2.alcoholPercentage / 100
  (totalAlcohol / totalVolume) * 100

/-- Theorem: The concentration of the mixture is 46.25% -/
theorem mixture_concentration :
  let vessel1 : Vessel := { capacity := 2, alcoholPercentage := 35 }
  let vessel2 : Vessel := { capacity := 6, alcoholPercentage := 50 }
  let totalVolume : ℚ := 8
  mixConcentration vessel1 vessel2 totalVolume = 185/4 := by
  sorry

#eval mixConcentration { capacity := 2, alcoholPercentage := 35 } { capacity := 6, alcoholPercentage := 50 } 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l475_47556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l475_47575

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = 2 → b = 2 * Real.sqrt 3 → A = π / 6 → 
  (B = π / 3 ∨ B = 2 * π / 3) ∧ 
  (Real.sin B = Real.sqrt 3 / 2) ∧
  (A + B ≤ π) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l475_47575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l475_47585

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 10 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 25

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3 * x + 4 * y + 10| / Real.sqrt (3^2 + 4^2)

-- Theorem statement
theorem line_intersects_circle_not_through_center :
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  (x, y) ≠ center ∧
  distance_to_line (center.1) (center.2) < 5 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l475_47585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_B_and_D_know_their_grades_l475_47513

-- Define the grade type
inductive Grade
| Excellent
| Good

-- Define the student type
inductive Student
| A
| B
| C
| D

-- Function to get a student's grade
def getGrade : Student → Grade := sorry

-- Function to check if a student can see another student's grade
def canSee : Student → Student → Prop := sorry

-- Theorem statement
theorem both_B_and_D_know_their_grades :
  -- Two students have Excellent grades and two have Good grades
  (∃ (s1 s2 s3 s4 : Student), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    getGrade s1 = Grade.Excellent ∧ getGrade s2 = Grade.Excellent ∧
    getGrade s3 = Grade.Good ∧ getGrade s4 = Grade.Good) →
  -- A sees B and C's grades
  (canSee Student.A Student.B ∧ canSee Student.A Student.C) →
  -- B sees C's grade
  canSee Student.B Student.C →
  -- D sees A's grade
  canSee Student.D Student.A →
  -- A doesn't know his own grade after seeing B and C's grades
  (∀ (g : Grade), ¬(getGrade Student.A = g ↔ 
    (getGrade Student.B = Grade.Excellent ∧ getGrade Student.C = Grade.Good) ∨
    (getGrade Student.B = Grade.Good ∧ getGrade Student.C = Grade.Excellent))) →
  -- B knows his own grade
  (∃ (g : Grade), getGrade Student.B = g) ∧
  -- D knows his own grade
  (∃ (g : Grade), getGrade Student.D = g) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_B_and_D_know_their_grades_l475_47513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_same_sphere_l475_47530

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- A predicate to check if a point is on or inside a sphere -/
def isOnOrInside (p : Point3D) (s : Sphere) : Prop := sorry

/-- A predicate to check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- A function to create a sphere passing through four points -/
noncomputable def sphereThroughFourPoints (p1 p2 p3 p4 : Point3D) : Sphere := sorry

/-- The main theorem -/
theorem all_points_on_same_sphere 
  (n : ℕ) 
  (points : Fin n → Point3D) 
  (h_n : n ≥ 5) 
  (h_not_coplanar : ∀ (i j k l : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    ¬ areCoplanar (points i) (points j) (points k) (points l))
  (h_sphere_property : ∀ (i j k l : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    let s := sphereThroughFourPoints (points i) (points j) (points k) (points l)
    ∀ (m : Fin n), m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ m ≠ l → isOnOrInside (points m) s) :
  ∃ (s : Sphere), ∀ (i : Fin n), isOnOrInside (points i) s ∧ 
    ∃ (j k l m : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ k ≠ l ∧ k ≠ m ∧ l ≠ m ∧
    s = sphereThroughFourPoints (points i) (points j) (points k) (points l) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_same_sphere_l475_47530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_boys_attending_l475_47515

-- Define the schools
structure School where
  name : String
  total_students : ℕ
  boy_ratio : ℚ
  girl_ratio : ℚ
  attendance_rate : ℚ

-- Define the two schools
def riverbank : School := {
  name := "Riverbank Middle School",
  total_students := 300,
  boy_ratio := 3/5,
  girl_ratio := 2/5,
  attendance_rate := 4/5
}

def brookside : School := {
  name := "Brookside Middle School",
  total_students := 240,
  boy_ratio := 2/5,
  girl_ratio := 3/5,
  attendance_rate := 3/4
}

-- Function to calculate the number of boys attending from a school
def boys_attending (s : School) : ℚ :=
  ↑s.total_students * s.boy_ratio * s.attendance_rate

-- Function to calculate the total number of students attending from a school
def total_attending (s : School) : ℚ :=
  ↑s.total_students * s.attendance_rate

-- Theorem statement
theorem fraction_of_boys_attending :
  (boys_attending riverbank + boys_attending brookside) / 
  (total_attending riverbank + total_attending brookside) = 18/35 := by
  -- Proof steps would go here
  sorry

-- Evaluate the result
#eval (boys_attending riverbank + boys_attending brookside) / 
      (total_attending riverbank + total_attending brookside)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_boys_attending_l475_47515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_iff_f_is_odd_l475_47590

noncomputable section

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ln(a + 2/(x-1)) where a is a real number -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (a + 2 / (x - 1))

/-- Theorem stating that a = 1 is a necessary and sufficient condition for f to be an odd function -/
theorem a_eq_one_iff_f_is_odd (a : ℝ) :
  a = 1 ↔ IsOdd (f a) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_iff_f_is_odd_l475_47590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_properties_l475_47599

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

-- State the theorem
theorem f_satisfies_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ ≠ x₂ →
    (f x₁ - f x₂) / (x₁ - x₂) < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_properties_l475_47599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_f_of_1_l475_47545

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 + a^2 * x^2 + a * x + b

theorem extreme_value_implies_f_of_1 (a b : ℝ) :
  f a b (-1) = -7/12 ∧ 
  (deriv (f a b)) (-1) = 0 →
  f a b 1 = 25/12 ∨ f a b 1 = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_f_of_1_l475_47545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_speed_is_5_l475_47568

/-- Emily's speed in miles per hour -/
noncomputable def emily_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem: Emily's speed is 5 miles per hour -/
theorem emily_speed_is_5 : emily_speed 10 2 = 5 := by
  -- Unfold the definition of emily_speed
  unfold emily_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_speed_is_5_l475_47568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_21_l475_47543

/-- The sequence defined by a_n + b_n -/
def seq : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | n + 1 => seq n + 2

/-- Theorem: The 11th term of the sequence is 21 -/
theorem eleventh_term_is_21 : seq 10 = 21 := by
  -- We use 10 here because Lean uses 0-based indexing
  sorry

#eval seq 10  -- This will evaluate the 11th term (0-based index 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_21_l475_47543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_XYZ_l475_47574

-- Define the triangle XYZ
variable (X Y Z : ℝ × ℝ)

-- Define points G, H, I on the sides of the triangle
noncomputable def G (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * Y.1 + (2/5) * Z.1, (3/5) * Y.2 + (2/5) * Z.2)
noncomputable def H (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * Z.1 + (2/5) * X.1, (3/5) * Z.2 + (2/5) * X.2)
noncomputable def I (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * X.1 + (2/5) * Y.1, (3/5) * X.2 + (2/5) * Y.2)

-- Define the intersection points P, Q, R
noncomputable def P (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * (G X Y Z).1 + (2/5) * X.1, (3/5) * (G X Y Z).2 + (2/5) * X.2)
noncomputable def Q (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * (H X Y Z).1 + (2/5) * Y.1, (3/5) * (H X Y Z).2 + (2/5) * Y.2)
noncomputable def R (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * (I X Y Z).1 + (2/5) * Z.1, (3/5) * (I X Y Z).2 + (2/5) * Z.2)

-- Function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem area_ratio_PQR_XYZ (X Y Z : ℝ × ℝ) :
  triangleArea (P X Y Z) (Q X Y Z) (R X Y Z) / triangleArea X Y Z = 12/125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_XYZ_l475_47574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l475_47507

open Real

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t * cos (π/4), t * sin (π/4))

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 8 * cos θ / (1 - cos θ ^ 2)

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem area_of_triangle : ∃ A B : ℝ × ℝ,
  (∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂) ∧
  curve_C_cartesian A.1 A.2 ∧
  curve_C_cartesian B.1 B.2 ∧
  let O : ℝ × ℝ := (0, 0)
  let area := abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2)) / 2
  area = 2 * sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l475_47507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l475_47548

/-- Represents the score distribution of students in an exam -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score100 : ℝ
  sum_to_one : score60 + score75 + score85 + score90 + score100 = 1

/-- Calculates the mean score given a score distribution -/
def calculateMean (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 100 * d.score100

/-- Calculates the median score given a score distribution -/
noncomputable def calculateMedian (d : ScoreDistribution) : ℝ :=
  if d.score60 + d.score75 < 0.5 ∧ d.score60 + d.score75 + d.score85 ≥ 0.5 then 85
  else if d.score60 + d.score75 + d.score85 < 0.5 ∧ d.score60 + d.score75 + d.score85 + d.score90 ≥ 0.5 then 90
  else 100

/-- The main theorem stating the difference between median and mean for the given distribution -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.15)
  (h2 : d.score75 = 0.20)
  (h3 : d.score85 = 0.25)
  (h4 : d.score90 = 0.30) :
  calculateMedian d - calculateMean d = 2.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l475_47548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l475_47502

theorem smallest_number_divisible (n : ℕ) : n = 1012 ↔ 
  (∀ m : ℕ, m < n → ¬(∀ d ∈ ({12, 16, 18, 21, 28} : Finset ℕ), (m - 4) % d = 0)) ∧ 
  (∀ d ∈ ({12, 16, 18, 21, 28} : Finset ℕ), (n - 4) % d = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l475_47502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l475_47536

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) 
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ (t : ℝ × ℝ) => (a * t.1, b * t.2)))
  (h_foci : c > 0 ∧ c^2 = a^2 - b^2)
  (h_equilateral : ∃ A B : ℝ × ℝ, 
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧ 
    A.1 = -c ∧ 
    B.1 = -c ∧
    Real.sqrt ((A.1 + c)^2 + A.2^2) = Real.sqrt ((A.1 - c)^2 + A.2^2) ∧
    Real.sqrt ((B.1 + c)^2 + B.2^2) = Real.sqrt ((B.1 - c)^2 + B.2^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt ((A.1 - c)^2 + A.2^2)) :
  c / a = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l475_47536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_normal_line_l475_47566

noncomputable def cosh_curve (x : ℝ) : ℝ := Real.cosh x
noncomputable def sinh_curve (x : ℝ) : ℝ := Real.sinh x

-- Define what it means for a line to be normal to a curve at a point
def is_normal_to_curve (m : ℝ) (x₀ : ℝ) (f : ℝ → ℝ) : Prop :=
  m = -1 / (deriv f x₀)

-- Theorem statement
theorem no_common_normal_line :
  ¬ ∃ (m : ℝ) (x₁ x₂ : ℝ), 
    x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    is_normal_to_curve m x₁ cosh_curve ∧
    is_normal_to_curve m x₂ sinh_curve :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_normal_line_l475_47566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_line_circle_relationship_l475_47570

theorem line_circle_disjoint (m : ℝ) (h : m > 0) :
  (1 + m) / Real.sqrt 2 > Real.sqrt m := by
  sorry

theorem line_circle_relationship (m : ℝ) (h : m ≥ 0) :
  (∀ x y : ℝ, x + y + 1 + m = 0 → x^2 + y^2 ≠ m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_line_circle_relationship_l475_47570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_neg_five_l475_47588

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else 3 * x - 18

theorem solutions_of_f_eq_neg_five :
  {x : ℝ | f x = -5} = {-3, 13/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_neg_five_l475_47588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_time_formula_l475_47523

/-- Represents a loom with a specific weaving rate -/
structure Loom where
  rate : ℝ  -- weaving rate in meters per second

/-- Calculates the time needed to weave a given length of cloth -/
noncomputable def weaving_time (loom : Loom) (length : ℝ) : ℝ :=
  length / loom.rate

/-- Theorem stating the weaving time formula for a loom -/
theorem loom_weaving_time_formula 
  (loom : Loom) 
  (h1 : loom.rate = 0.126) 
  (h2 : weaving_time loom 15 = 119.04761904761905) :
  ∀ L : ℝ, weaving_time loom L = L / 0.126 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_time_formula_l475_47523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_120_degree_inclination_l475_47531

structure Line2D where
  slope : ℝ
  angle_of_inclination : ℝ

theorem slope_of_line_with_120_degree_inclination :
  ∀ (line : Line2D),
  line.angle_of_inclination = 120 * Real.pi / 180 →
  line.slope = -Real.sqrt 3 :=
by
  intro line h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_120_degree_inclination_l475_47531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_l475_47511

noncomputable section

-- Define the ellipse
def ellipse (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the line (in polar form)
def line (ρ θ : ℝ) : Prop := 2 * ρ * Real.cos (θ + Real.pi / 3) = 3 * Real.sqrt 6

-- Define the distance function from a point to the line
def distance (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  |x - Real.sqrt 3 * y - 3 * Real.sqrt 6| / 2

-- Statement of the theorem
theorem ellipse_line_distance :
  (∀ θ : ℝ, distance (ellipse θ) ≤ 2 * Real.sqrt 6) ∧
  (∃ θ : ℝ, distance (ellipse θ) = 2 * Real.sqrt 6) ∧
  (∀ θ : ℝ, distance (ellipse θ) ≥ Real.sqrt 6) ∧
  (∃ θ : ℝ, distance (ellipse θ) = Real.sqrt 6) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_l475_47511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_intersection_l475_47578

def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 4}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {2, 3}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {0, 1, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_intersection_l475_47578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_speed_l475_47597

/-- The position function of the particle -/
noncomputable def s (t : ℝ) : ℝ := 3 + t^2

/-- The average speed of the particle over a time interval -/
noncomputable def average_speed (t₁ t₂ : ℝ) : ℝ := (s t₂ - s t₁) / (t₂ - t₁)

/-- Theorem: The average speed of the particle during the time interval [2, 2.1] is 4.1 -/
theorem particle_average_speed : average_speed 2 2.1 = 4.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_speed_l475_47597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_value_l475_47596

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℚ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_function_a_value 
  (f : QuadraticFunction) 
  (h1 : f.eval (-2) = 3)
  (h2 : f.eval 1 = 6)
  (h3 : f.a = 1 ∨ f.a = 1/3)
  : f.a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_value_l475_47596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l475_47529

/-- Represents the voltage of the battery in volts. -/
noncomputable def voltage : ℝ := 48

/-- Represents the resistance in ohms. -/
noncomputable def resistance : ℝ := 12

/-- Calculates the current in amperes given the resistance. -/
noncomputable def current (r : ℝ) : ℝ := voltage / r

/-- Theorem stating that the current is 4A when the resistance is 12Ω. -/
theorem current_for_given_resistance :
  current resistance = 4 := by
  -- Unfold the definitions and simplify
  unfold current resistance voltage
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l475_47529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_evaluation_l475_47537

theorem function_evaluation (f : ℝ → ℝ) (α : ℝ) :
  (f = λ x ↦ Real.sin (x + π/6)) →
  (Real.sin α = 3/5) →
  (π/2 < α) →
  (α < π) →
  f (α + π/12) = -(Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_evaluation_l475_47537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_l475_47589

/-- The function f(x) = x^2 ln(x) - a(x^2 - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log x - a * (x^2 - 1)

/-- Theorem: If f(x) ≥ 0 for all x in (0, 1], then a ≥ 1/2 -/
theorem f_nonnegative_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≥ 0) → a ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_l475_47589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_10_pow_41_l475_47534

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition function
def satisfies_equation (x y : ℕ) : Prop :=
  (Real.sqrt (log10 (x : ℝ)) + Real.sqrt (log10 (y : ℝ)) + 
   log10 (Real.sqrt (x : ℝ)) + log10 (Real.sqrt (y : ℝ)) = 50) ∧
  (∃ (a b c d : ℤ), 
    Real.sqrt (log10 (x : ℝ)) = a ∧
    Real.sqrt (log10 (y : ℝ)) = b ∧
    log10 (Real.sqrt (x : ℝ)) = c ∧
    log10 (Real.sqrt (y : ℝ)) = d)

theorem product_equals_10_pow_41 (x y : ℕ) (h : satisfies_equation x y) :
  x * y = 10^41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_10_pow_41_l475_47534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivier_winning_strategy_l475_47501

/-- Represents the state of a square in the game -/
inductive SquareState
| Empty : SquareState
| Brown : SquareState
| Orange : SquareState

/-- Represents the game board -/
def GameBoard := List SquareState

/-- Checks if a move is valid according to the game rules -/
def isValidMove (board : GameBoard) (position : Nat) (color : SquareState) : Prop :=
  position < board.length ∧
  board.get? position = some SquareState.Empty ∧
  (position = 0 ∨ board.get? (position - 1) ≠ some color) ∧
  (position = board.length - 1 ∨ board.get? (position + 1) ≠ some color)

/-- Represents a winning strategy for a player -/
def hasWinningStrategy (player : Nat) (n : Nat) : Prop :=
  ∀ (board : GameBoard),
    board.length = n →
    (player = 0 ∧ ∃ (move : Nat), isValidMove board move SquareState.Brown) ∨
    (player = 1 ∧ ∀ (move : Nat), isValidMove board move SquareState.Brown →
      ∃ (counterMove : Nat), isValidMove (board.set move SquareState.Brown) counterMove SquareState.Orange)

/-- The main theorem stating that Olivier (player 1) has a winning strategy for N ≥ 2 -/
theorem olivier_winning_strategy (n : Nat) (h : n ≥ 2) :
  hasWinningStrategy 1 n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivier_winning_strategy_l475_47501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrangular_prism_volume_32cm_l475_47522

/-- The volume of a quadrangular prism with square base and sides -/
noncomputable def quadrangular_prism_volume (base_perimeter : ℝ) : ℝ :=
  let side_length := base_perimeter / 4
  side_length ^ 3

theorem quadrangular_prism_volume_32cm (base_perimeter : ℝ) 
  (h : base_perimeter = 32) : quadrangular_prism_volume base_perimeter = 512 := by
  -- Unfold the definition of quadrangular_prism_volume
  unfold quadrangular_prism_volume
  -- Substitute the value of base_perimeter
  rw [h]
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrangular_prism_volume_32cm_l475_47522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_nails_equivalent_to_line_two_points_determine_line_l475_47569

/-- A point in a plane -/
structure Point where
  x : Real
  y : Real

/-- A piece of wood fixed to a wall with two nails -/
structure WoodPiece where
  nail1 : Point
  nail2 : Point

/-- A straight line determined by two points -/
structure Line where
  point1 : Point
  point2 : Point

/-- The theorem stating that a piece of wood fixed with two nails is equivalent to a line determined by two points -/
theorem wood_nails_equivalent_to_line : 
  ∀ (w : WoodPiece), ∃ (l : Line), l.point1 = w.nail1 ∧ l.point2 = w.nail2 ∧
  ∀ (l' : Line), l'.point1 = w.nail1 ∧ l'.point2 = w.nail2 → l' = l := by
  sorry

/-- The theorem stating that two points determine a unique line -/
theorem two_points_determine_line :
  ∀ (p1 p2 : Point), ∃! (l : Line), l.point1 = p1 ∧ l.point2 = p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_nails_equivalent_to_line_two_points_determine_line_l475_47569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l475_47546

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x₃ : ℝ, x₃ ∈ Set.Icc 0 Real.pi ∧ f x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ∧
  (∀ x : ℝ, f (Real.pi/12 + x) = f (Real.pi/12 - x)) ∧
  (∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ 4 * x + 2*Real.pi/3 ∧ f x₀ = 4 * x₀ + 2*Real.pi/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l475_47546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_years_minimizes_cost_l475_47591

/-- Represents the cost structure of owning and maintaining a car over time. -/
structure CarCost where
  initialCost : ℝ
  annualFixedExpenses : ℝ
  firstYearMaintenance : ℝ
  annualMaintenanceIncrease : ℝ

/-- Calculates the average annual cost of owning the car for a given number of years. -/
noncomputable def averageAnnualCost (c : CarCost) (years : ℝ) : ℝ :=
  (c.initialCost + c.annualFixedExpenses * years + 
   (c.firstYearMaintenance + (years - 1) * c.annualMaintenanceIncrease / 2) * years) / years

/-- Theorem stating that 10 years minimizes the average annual cost for the given car cost structure. -/
theorem ten_years_minimizes_cost (c : CarCost)
  (h1 : c.initialCost = 100000)
  (h2 : c.annualFixedExpenses = 15000)
  (h3 : c.firstYearMaintenance = 1000)
  (h4 : c.annualMaintenanceIncrease = 2000) :
  ∀ y : ℝ, y > 0 → averageAnnualCost c 10 ≤ averageAnnualCost c y := by
  sorry

#check ten_years_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_years_minimizes_cost_l475_47591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_speed_approximately_5_11_l475_47540

/-- The speed of person A in km/h -/
noncomputable def speed_A : ℝ := 4

/-- The time B starts walking after A in hours -/
noncomputable def time_delay : ℝ := 0.5

/-- The time it takes B to overtake A in hours -/
noncomputable def time_overtake : ℝ := 1 + 48 / 60

/-- The speed of person B in km/h -/
noncomputable def speed_B : ℝ := (speed_A * time_delay + speed_A * time_overtake) / time_overtake

theorem b_speed_approximately_5_11 :
  |speed_B - 5.11| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_speed_approximately_5_11_l475_47540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l475_47510

/-- Given two real numbers p and q, their arithmetic mean is (p + q) / 2 -/
noncomputable def arithmetic_mean (p q : ℝ) : ℝ := (p + q) / 2

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : arithmetic_mean p q = 10)
  (h2 : arithmetic_mean q r = 22)
  (h3 : r - p = 24) :
  arithmetic_mean p q = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l475_47510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l475_47526

-- Define the set T
def T : Set (Real × Real × Real) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 2}

-- Define the support relation
def supports (x y z a b c : Real) : Prop :=
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

-- Define the set S
def S : Set (Real × Real × Real) :=
  {p ∈ T | supports p.1 p.2.1 p.2.2 1 (2/3) (1/3)}

-- Define the area function (noncomputable as it's not constructive)
noncomputable def area (s : Set (Real × Real × Real)) : Real := sorry

-- State the theorem
theorem area_ratio_S_T : area S / area T = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l475_47526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_for_digit_product_equality_l475_47594

/-- Represents an n-digit integer formed by consecutive digits starting from a given digit -/
def consecutive_digit_integer (n : ℕ) (start : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * 10 + (start + i)) 0

/-- Represents a 2n-digit integer where each digit is the sum of two given digits -/
def sum_digit_integer (n : ℕ) (a b : ℕ) : ℕ :=
  ((a + b) * (10^(2*n) - 1)) / 9

theorem smallest_sum_for_digit_product_equality :
  ∃ (a b : ℕ) (n : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n > 0 ∧
    sum_digit_integer n a b = consecutive_digit_integer n a * consecutive_digit_integer n b ∧
    (∀ (a' b' : ℕ) (n' : ℕ),
      a' ≠ 0 → b' ≠ 0 → a' < 10 → b' < 10 → n' > 0 →
      sum_digit_integer n' a' b' = consecutive_digit_integer n' a' * consecutive_digit_integer n' b' →
      a + b ≤ a' + b') ∧
    a + b = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_for_digit_product_equality_l475_47594
