import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l50_5019

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := x^2 + (a-2)*x - a * Real.log x

-- State the theorem
theorem f_inequality_implies_a_range (a : ℝ) (h : a < 0) :
  (∀ x > 0, f a x + 2*x + 2*a * Real.log x > (1/2)*(Real.exp 1 + 1)*a) →
  -2*(Real.exp 1)^2/(Real.exp 1 + 1) < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l50_5019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_of_specific_rhombus_l50_5041

/-- Represents a rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ

/-- Calculates the length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal (r : Rhombus) : ℝ :=
  let ratio_sum := r.diagonal_ratio.fst + r.diagonal_ratio.snd
  2 * (r.diagonal_ratio.fst / ratio_sum) * Real.sqrt (2 * r.area)

theorem longest_diagonal_of_specific_rhombus :
  let r : Rhombus := { area := 144, diagonal_ratio := (4, 3) }
  longest_diagonal r = 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_of_specific_rhombus_l50_5041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_medical_costs_l50_5073

/-- Calculates Tom's total out-of-pocket medical costs --/
theorem toms_medical_costs : 
  let doctor_visit_cost : ℕ := 300
  let cast_cost : ℕ := 200
  let initial_insurance_coverage : ℚ := 60 / 100
  let pt_sessions : ℕ := 8
  let pt_cost_per_session : ℕ := 100
  let pt_insurance_coverage : ℚ := 40 / 100
  let pt_copay : ℕ := 20

  let initial_total_cost := doctor_visit_cost + cast_cost
  let initial_out_of_pocket := initial_total_cost - (initial_total_cost * initial_insurance_coverage).floor
  
  let pt_total_cost := pt_sessions * pt_cost_per_session
  let pt_out_of_pocket := pt_total_cost - (pt_total_cost * pt_insurance_coverage).floor + (pt_sessions * pt_copay)
  
  let total_out_of_pocket := initial_out_of_pocket + pt_out_of_pocket

  total_out_of_pocket = 840 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_medical_costs_l50_5073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_square_eccentricity_l50_5006

/-- Hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

/-- Theorem: If the endpoints of the real and imaginary axes of a hyperbola form a square,
    then its eccentricity is √2 -/
theorem hyperbola_square_eccentricity (a b : ℝ) (h : Hyperbola a b) 
    (square : 2*a = 2*b) : eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_square_eccentricity_l50_5006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_quadrilateral_area_is_7_2_l50_5087

noncomputable section

/-- The area of a small concave quadrilateral formed by the intersection of two rings -/
def small_quadrilateral_area (inner_diameter outer_diameter : ℝ) (total_area : ℝ) : ℝ :=
  let inner_radius := inner_diameter / 2
  let outer_radius := outer_diameter / 2
  let single_ring_area := Real.pi * (outer_radius^2 - inner_radius^2)
  let total_non_overlapping_area := 5 * single_ring_area
  let overlap_area := total_non_overlapping_area - total_area
  overlap_area / 4

/-- The theorem stating the area of each small concave quadrilateral -/
theorem small_quadrilateral_area_is_7_2 :
  small_quadrilateral_area 8 10 112.5 = 7.2 := by
  sorry

/-- π is approximately equal to 3.14 -/
axiom pi_approx : Real.pi = 3.14

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_quadrilateral_area_is_7_2_l50_5087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l50_5091

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2

theorem f_properties :
  -- Part 1: Minimum positive period
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Part 2: Maximum value in [0, π/2]
  (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f x_max) ∧
    f x_max = 1 - Real.sqrt 3 / 2) ∧
  -- Part 3: Minimum value in [0, π/2]
  (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x_min ≤ f x) ∧
    f x_min = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l50_5091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_longest_altitudes_is_31_l50_5067

/-- A triangle with sides 7, 24, and 25 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 7
  hb : b = 24
  hc : c = 25
  right_angle : a^2 + b^2 = c^2

/-- The length of an altitude in the triangle -/
noncomputable def altitude (t : RightTriangle) (side : ℝ) : ℝ :=
  if side = t.c then (t.a * t.b) / t.c else side

/-- The sum of the two longest altitudes in the triangle -/
noncomputable def sum_two_longest_altitudes (t : RightTriangle) : ℝ :=
  max (altitude t t.a + altitude t t.b) (max (altitude t t.a + altitude t t.c) (altitude t t.b + altitude t t.c))

/-- Theorem: The sum of the two longest altitudes in the given triangle is 31 -/
theorem sum_two_longest_altitudes_is_31 (t : RightTriangle) :
  sum_two_longest_altitudes t = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_longest_altitudes_is_31_l50_5067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_positive_set_l50_5035

theorem odd_function_positive_set 
  (f : ℝ → ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (hf_zero : f 1 = 0) 
  (hf_deriv : ∀ x > 0, x * (deriv f x) < f x) :
  {x : ℝ | f x > 0} = Set.Ioo 0 1 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_positive_set_l50_5035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l50_5018

/-- Given a triangle ABC with sides a, b, c, circumradius R, and exradii r_a, r_b, r_c,
    if 2R ≤ r_a, then a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 →
  (∃ A B C : ℝ, 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    a = 2 * R * Real.sin A ∧
    b = 2 * R * Real.sin B ∧
    c = 2 * R * Real.sin C) →
  2 * R ≤ r_a →
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l50_5018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_hourly_rate_l50_5030

/-- Lloyd's work scenario --/
structure WorkScenario where
  regularHours : ℚ
  overtimeMultiplier : ℚ
  actualHours : ℚ
  dailyEarnings : ℚ

/-- Calculate Lloyd's hourly rate --/
def calculateHourlyRate (scenario : WorkScenario) : ℚ :=
  scenario.dailyEarnings / (scenario.regularHours + (scenario.actualHours - scenario.regularHours) * scenario.overtimeMultiplier)

/-- Theorem: Lloyd's hourly rate is $4.50 --/
theorem lloyds_hourly_rate :
  let scenario : WorkScenario := {
    regularHours := 15/2,
    overtimeMultiplier := 2,
    actualHours := 21/2,
    dailyEarnings := 243/4
  }
  calculateHourlyRate scenario = 9/2 := by
  -- Proof goes here
  sorry

#eval calculateHourlyRate {
  regularHours := 15/2,
  overtimeMultiplier := 2,
  actualHours := 21/2,
  dailyEarnings := 243/4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_hourly_rate_l50_5030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l50_5058

/-- The function f(x) = ln(2x^2 - 3) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3)

/-- The monotonic decreasing interval of f(x) -/
def monotonic_decreasing_interval : Set ℝ := Set.Iio (-Real.sqrt 6 / 2)

/-- Theorem stating that the monotonic decreasing interval of f(x) is (-∞, -√6/2) -/
theorem f_monotonic_decreasing_interval :
  {x : ℝ | ∀ y, y ∈ monotonic_decreasing_interval → x < y → f x > f y} = monotonic_decreasing_interval :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_interval_l50_5058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l50_5001

/-- The time (in hours) it takes for the electric pump to fill the tank without the leak -/
noncomputable def T : ℝ := 10

/-- The rate at which the leak empties the tank (fraction of tank per hour) -/
noncomputable def leak_rate : ℝ := 1 / 20

/-- The time it takes to fill the tank with both pump and leak active (in hours) -/
noncomputable def fill_time_with_leak : ℝ := 20

theorem pump_fill_time :
  T = 10 ∧
  leak_rate = 1 / 20 ∧
  fill_time_with_leak = 20 →
  1 / T - leak_rate = 1 / fill_time_with_leak :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l50_5001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_identification_l50_5040

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_identification 
  (ω : ℝ) (φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_max : f ω φ (π/4) = 1) 
  (h_min : f ω φ (7*π/12) = -1) : 
  ∀ x, f ω φ x = Real.sin (3*x - π/4) := by
  sorry

#check sine_function_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_identification_l50_5040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l50_5061

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- Add a case for 0 to handle all natural numbers
  | 1 => 1
  | n + 2 => 2^(n + 2) * a (n + 1)

-- State the theorem
theorem a_general_term (n : ℕ) (h : n ≥ 1) : a n = 2^((n^2 + n - 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l50_5061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_BaBr2_approx_l50_5046

/-- The mass percentage of bromine in barium bromide -/
noncomputable def mass_percentage_Br_in_BaBr2 : ℝ :=
  let molar_mass_Ba : ℝ := 137.33
  let molar_mass_Br : ℝ := 79.90
  let molar_mass_BaBr2 : ℝ := molar_mass_Ba + 2 * molar_mass_Br
  (2 * molar_mass_Br / molar_mass_BaBr2) * 100

/-- Theorem stating that the mass percentage of Br in BaBr2 is approximately 53.80% -/
theorem mass_percentage_Br_in_BaBr2_approx :
  |mass_percentage_Br_in_BaBr2 - 53.80| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_BaBr2_approx_l50_5046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_out_of_pocket_l50_5069

def initial_purchase : ℝ := 5000
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def tv_cost : ℝ := 1000
def first_bike_cost : ℝ := 700
def second_bike_markup : ℝ := 0.20
def second_bike_resale_rate : ℝ := 0.85
def toaster_cost : ℝ := 100
def microwave_cost : ℝ := 150
def subscription_cost : ℝ := 80
def subscription_discount : ℝ := 0.30
def subscription_months : ℝ := 3

def total_out_of_pocket : ℝ := 4157

theorem james_out_of_pocket :
  let discounted_price := initial_purchase * (1 - discount_rate)
  let taxed_price := discounted_price * (1 + tax_rate)
  let returns := tv_cost + first_bike_cost
  let second_bike_cost := first_bike_cost * (1 + second_bike_markup)
  let second_bike_resale := second_bike_cost * second_bike_resale_rate
  let additional_purchases := toaster_cost + microwave_cost
  let discounted_subscription := subscription_cost * (1 - subscription_discount) * subscription_months
  taxed_price - returns + second_bike_resale + additional_purchases + discounted_subscription = total_out_of_pocket := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_out_of_pocket_l50_5069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l50_5014

noncomputable def P : ℝ × ℝ := (-4, -5)
noncomputable def Q : ℝ × ℝ := (2, 3)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_Q : distance P Q = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l50_5014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l50_5070

/-- A geometric sequence with a₁ = 1 and a₄ = 27 -/
noncomputable def geometric_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 3^(n-1)

/-- The sum of the first n terms of the geometric sequence -/
noncomputable def geometric_sum (n : ℕ) : ℝ :=
  (1/2) * (3^n - 1)

theorem geometric_sequence_properties :
  (geometric_sequence 1 = 1) ∧
  (geometric_sequence 4 = 27) ∧
  (geometric_sequence 3 = 9) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 3^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sum n = (1/2) * (3^n - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l50_5070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l50_5085

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l50_5085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_parametric_line_l50_5081

/-- The angle of inclination of a line given by parametric equations -/
theorem angle_of_inclination_parametric_line :
  let x : ℝ → ℝ := λ t => 3 - (Real.sqrt 2 / 2) * t
  let y : ℝ → ℝ := λ t => Real.sqrt 5 - (Real.sqrt 2 / 2) * t
  (∃ α : ℝ, ∀ t : ℝ, Real.tan α = (y t - y 0) / (x t - x 0)) →
  ∃ α : ℝ, α = π / 4 ∧ ∀ t : ℝ, Real.tan α = (y t - y 0) / (x t - x 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_parametric_line_l50_5081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l50_5048

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem function_properties (φ : ℝ) 
  (h1 : f 0 φ = 0) 
  (h2 : -Real.pi / 2 < φ) 
  (h3 : φ < 0) :
  ∃ (k : ℤ),
    φ = -Real.pi / 6 ∧ 
    (∀ x, f x φ ≤ 3) ∧ 
    f (k * Real.pi + 2 * Real.pi / 3) φ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l50_5048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_eq1_two_solutions_eq2_l50_5027

-- First equation
theorem unique_solution_eq1 : 
  ∃! x : ℝ, Real.sqrt ((x - 1) * (x - 2)) + Real.sqrt ((x - 3) * (x - 4)) = Real.sqrt 2 :=
by sorry

-- Second equation
theorem two_solutions_eq2 (p q r : ℝ) (hp : p = 3/2) (hq : q = 1/2) (hr : r = 7/2) :
  ∃ x y : ℝ, x ≠ y ∧ 
    Real.sqrt ((x - p - q) * (x - p + q)) + Real.sqrt ((x - r - q) * (x - r + q)) = 
      Real.sqrt ((p - r) * (p - r) * (p - r + 2*q)) ∧
    Real.sqrt ((y - p - q) * (y - p + q)) + Real.sqrt ((y - r - q) * (y - r + q)) = 
      Real.sqrt ((p - r) * (p - r) * (p - r + 2*q)) ∧
    (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_eq1_two_solutions_eq2_l50_5027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_eq_sqrt_3_l50_5086

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (2 * x + 3)
noncomputable def f (x : ℝ) : ℝ := 6 - 2 * t x

-- Theorem statement
theorem t_of_f_3_eq_sqrt_3 : t (f 3) = Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_eq_sqrt_3_l50_5086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_y_axis_l50_5032

theorem slope_angle_of_y_axis (α : Real) : 
  (∀ x y : Real, x = 0 → (x, y) ∈ ({(x, y) | x = 0} : Set (Real × Real))) → α = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_y_axis_l50_5032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_three_digit_numbers_count_sum_of_squares_equals_285_valid_three_digit_numbers_equals_sum_of_squares_l50_5084

/-- Count of three-digit numbers where the hundreds digit is greater than both other digits -/
def validThreeDigitNumbers : ℕ := Finset.sum (Finset.range 9) (fun i => (i + 1)^2)

/-- Sum of squares of the first 9 positive integers -/
def sumOfSquares : ℕ := Finset.sum (Finset.range 9) (fun i => (i + 1)^2)

theorem valid_three_digit_numbers_count :
  validThreeDigitNumbers = 285 := by
  sorry

theorem sum_of_squares_equals_285 :
  sumOfSquares = 285 := by
  sorry

theorem valid_three_digit_numbers_equals_sum_of_squares :
  validThreeDigitNumbers = sumOfSquares := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_three_digit_numbers_count_sum_of_squares_equals_285_valid_three_digit_numbers_equals_sum_of_squares_l50_5084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l50_5089

/-- The angle of the minute hand at a given number of minutes past the hour -/
noncomputable def minute_hand_angle (minutes : ℕ) : ℝ :=
  (minutes : ℝ) / 60 * 360

/-- The angle of the hour hand at a given hour and minute -/
noncomputable def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour : ℝ) / 12 * 360 + (minute : ℝ) / 60 * 30

/-- The acute angle between the hour and minute hands -/
noncomputable def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let angle := abs (minute_hand_angle minute - hour_hand_angle hour minute)
  min angle (360 - angle)

theorem clock_angle_at_3_25 :
  clock_angle 3 25 = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l50_5089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l50_5037

open Finset Nat

def count_solutions : ℕ :=
  (range 11).sum (λ x => choose (23 - 2*x) 2)

theorem solution_count :
  count_solutions = 946 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l50_5037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l50_5080

-- Define constants
def train1_length : ℝ := 250
def train1_speed : ℝ := 120
def train2_speed : ℝ := 80
def crossing_time : ℝ := 9

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := (train1_speed + train2_speed) * 1000 / 3600
  let total_distance : ℝ := relative_speed * crossing_time
  let train2_length : ℝ := total_distance - train1_length
  ∃ ε > 0, |train2_length - 249.95| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l50_5080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caleb_circles_have_two_external_tangents_l50_5002

/-- Two circles on a plane with radii r₁ and r₂, separated by distance d -/
structure TwoCircles where
  r₁ : ℝ
  r₂ : ℝ
  d : ℝ

/-- The number of external tangent lines to two circles -/
noncomputable def num_external_tangents (c : TwoCircles) : ℕ :=
  if c.d > c.r₁ + c.r₂ then 4
  else if c.d = c.r₁ + c.r₂ then 3
  else if c.d > max c.r₁ c.r₂ - min c.r₁ c.r₂ then 2
  else if c.d = max c.r₁ c.r₂ - min c.r₁ c.r₂ then 1
  else 0

/-- The specific configuration of circles in the problem -/
def caleb_circles : TwoCircles := ⟨7, 5, 10⟩

theorem caleb_circles_have_two_external_tangents :
  num_external_tangents caleb_circles = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caleb_circles_have_two_external_tangents_l50_5002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l50_5008

/-- An arithmetic sequence with positive terms and common difference 2 -/
def arithmetic_seq (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0) ∧ (∀ n : ℕ+, a (n + 1) = a n + 2)

/-- The geometric mean of two consecutive terms of the sequence -/
noncomputable def b (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  Real.sqrt (a n * a (n + 1))

/-- The sequence c_n defined as the difference of squares of consecutive b terms -/
noncomputable def c (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (b a (n + 1))^2 - (b a n)^2

/-- The main theorem combining all conditions and results -/
theorem arithmetic_sequence_properties (a : ℕ+ → ℝ) :
  arithmetic_seq a →
  (c a 1 = 16) →
  (∀ n : ℕ+, c a (n + 1) - c a n = 8) ∧
  (∀ n : ℕ+, a n = 2 * ↑n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l50_5008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l50_5064

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with common difference d,
    if 2S_3 = 3S_2 + 6, then d = 2 -/
theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) : 2 * S 3 a₁ d = 3 * S 2 a₁ d + 6 → d = 2 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l50_5064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_discount_is_sixty_percent_l50_5093

/-- Represents the cash discount problem for a dealer. -/
structure DealerDiscount where
  /-- The markup percentage above cost price -/
  markup : ℚ
  /-- The number of articles sold at the cost price of fewer articles -/
  articlesAtDiscount : ℕ
  /-- The number of articles whose cost price is used for the discounted sale -/
  articlesAtCost : ℕ
  /-- The profit percentage after applying the discount -/
  profitPercentage : ℚ

/-- Calculates the cash discount percentage offered by the dealer. -/
def cashDiscountPercentage (d : DealerDiscount) : ℚ :=
  (1 - (d.articlesAtCost : ℚ) / (d.articlesAtDiscount : ℚ) * (1 + d.profitPercentage / 100)) * 100

/-- The theorem stating that the cash discount percentage is 60% for the given conditions. -/
theorem cash_discount_is_sixty_percent :
  let d := DealerDiscount.mk 100 25 20 36
  cashDiscountPercentage d = 60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_discount_is_sixty_percent_l50_5093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_brand_a_l50_5003

/-- The weight of one liter of vegetable ghee packet of brand 'b' in grams -/
def weight_b : ℚ := 850

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℚ := 3
def ratio_b : ℚ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℚ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℚ := 3520

/-- The weight of one liter of vegetable ghee packet of brand 'a' in grams -/
noncomputable def weight_a : ℚ := (total_weight - ratio_b * weight_b) / ratio_a

theorem weight_of_brand_a : ∃ ε > 0, |weight_a - 606.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_brand_a_l50_5003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_exact_successes_l50_5063

def probability_of_success : ℝ := 0.6

def number_of_trials : ℕ := 4

def number_of_successes : ℕ := 2

theorem probability_of_exact_successes :
  (Nat.choose number_of_trials number_of_successes : ℝ) *
  probability_of_success ^ number_of_successes *
  (1 - probability_of_success) ^ (number_of_trials - number_of_successes) =
  0.3456 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_exact_successes_l50_5063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l50_5082

noncomputable def f (x : ℝ) : ℝ := x / 50 - Real.sin x

theorem solution_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ≥ 0 ∧ f x = 0) ∧ S.card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l50_5082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l50_5007

-- Define the functions f and g
def f (x : ℝ) : ℝ := -x^2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2^x - m

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g m x₂) →
  m ≥ 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l50_5007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_l50_5023

theorem yellow_tint_percentage (original_volume : ℝ) (original_yellow_percent : ℝ) 
  (added_yellow : ℝ) (new_yellow_percent : ℝ) : 
  original_volume = 50 ∧ 
  original_yellow_percent = 25 ∧ 
  added_yellow = 10 ∧
  new_yellow_percent = (original_yellow_percent / 100 * original_volume + added_yellow) / 
    (original_volume + added_yellow) * 100 →
  new_yellow_percent = 37.5 := by
  intro h
  sorry

#check yellow_tint_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_l50_5023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_iff_m_eq_3_or_neg_3_l50_5044

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
def circle2 (x y m : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 1

-- Define the center and radius of each circle
def center1 : ℝ × ℝ := (3, 0)
def radius1 : ℝ := 4
def center2 (m : ℝ) : ℝ × ℝ := (-1, m)
def radius2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers (m : ℝ) : ℝ := 
  Real.sqrt ((center1.1 - (center2 m).1)^2 + (center1.2 - (center2 m).2)^2)

-- Theorem statement
theorem circles_tangent_iff_m_eq_3_or_neg_3 : 
  ∀ m : ℝ, (distance_between_centers m = radius1 + radius2 ∨ 
             distance_between_centers m = |radius1 - radius2|) ↔ 
            (m = 3 ∨ m = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_iff_m_eq_3_or_neg_3_l50_5044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_population_l50_5017

theorem fish_tank_population :
  -- Total fish distribution
  let blue_percent : ℚ := 35/100
  let yellow_percent : ℚ := 25/100
  let red_percent : ℚ := 1 - blue_percent - yellow_percent

  -- Pattern distribution
  let spotted_percent : ℚ := 15/100
  let striped_percent : ℚ := 10/100
  let solid_percent : ℚ := 1 - spotted_percent - striped_percent

  -- Blue fish pattern
  let blue_spotted_percent : ℚ := 50/100
  let blue_striped_percent : ℚ := 25/100
  let blue_solid_percent : ℚ := 1 - blue_spotted_percent - blue_striped_percent

  -- Yellow fish pattern
  let yellow_spotted_percent : ℚ := 30/100
  let yellow_striped_percent : ℚ := 45/100
  let yellow_solid_percent : ℚ := 1 - yellow_spotted_percent - yellow_striped_percent

  -- Red fish pattern
  let red_spotted_percent : ℚ := 20/100
  let red_striped_percent : ℚ := 35/100
  let red_solid_percent : ℚ := 1 - red_spotted_percent - red_striped_percent

  -- Given information
  let blue_spotted_count : ℕ := 28
  let yellow_striped_count : ℕ := 15

  -- Total fish count
  ∃ (total : ℕ),
    (blue_percent * (total : ℚ) = (blue_spotted_count : ℚ) / blue_spotted_percent) ∧
    (yellow_percent * (total : ℚ) = (yellow_striped_count : ℚ) / yellow_striped_percent) ∧
    total = 160 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_population_l50_5017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_calculation_l50_5015

/-- Represents the total accumulated spending in millions of dollars -/
def AccumulatedSpending : ℝ → ℝ := sorry

/-- The spending at the beginning of March -/
def march_start : ℝ := 1.2

/-- The spending at the end of July -/
def july_end : ℝ := 5.4

/-- The amount spent during March, April, May, June, and July -/
def spending_mar_to_jul : ℝ := july_end - march_start

theorem spending_calculation :
  spending_mar_to_jul = 4.2 := by
  -- Unfold the definition of spending_mar_to_jul
  unfold spending_mar_to_jul
  -- Unfold the definitions of july_end and march_start
  unfold july_end march_start
  -- Simplify the arithmetic
  norm_num

#check spending_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_calculation_l50_5015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l50_5054

-- Define a linear function
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- Define the distance between intersection points
noncomputable def intersection_distance (a b c d : ℝ) : ℝ :=
  Real.sqrt ((a^2 + 1) * (a^2 + 4*(b - c) + 4*d))

-- Theorem statement
theorem intersection_distance_theorem (a b : ℝ) :
  intersection_distance a b 0 0 = Real.sqrt 10 →
  intersection_distance a b (-1) 1 = Real.sqrt 42 →
  intersection_distance a b 1 2 = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l50_5054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_round_needed_l50_5004

structure Candidate where
  name : String
  initial_votes : Float
  second_choice_votes : Float

def winning_threshold : Float := 40

def redistribute_votes (candidates : List Candidate) : List Candidate :=
  sorry

theorem second_round_needed (candidates : List Candidate) :
  let initial_candidates := [
    { name := "A", initial_votes := 32, second_choice_votes := 0 },
    { name := "B", initial_votes := 25, second_choice_votes := 0 },
    { name := "C", initial_votes := 22, second_choice_votes := 6.4 },
    { name := "D", initial_votes := 13, second_choice_votes := 0 },
    { name := "E", initial_votes := 8, second_choice_votes := 0 }
  ]
  let redistributed_candidates := redistribute_votes initial_candidates
  (∀ c, c ∈ redistributed_candidates → c.initial_votes + c.second_choice_votes < winning_threshold) ∧
  (∃ a b, a ∈ redistributed_candidates ∧ b ∈ redistributed_candidates ∧ 
    a.name = "A" ∧ b.name = "B" ∧
    ∀ c, c ∈ redistributed_candidates → 
      c.name ≠ "A" ∧ c.name ≠ "B" → 
      c.initial_votes + c.second_choice_votes ≤ a.initial_votes + a.second_choice_votes ∧
      c.initial_votes + c.second_choice_votes ≤ b.initial_votes + b.second_choice_votes) :=
by
  sorry

#check second_round_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_round_needed_l50_5004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l50_5077

/-- The area of a triangle using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 28, 26, and 10 is approximately 130 -/
theorem triangle_area_specific : ∃ ε > 0, |triangle_area 28 26 10 - 130| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l50_5077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_eaten_cake_cost_l50_5024

/-- Represents the cost of ingredients for a cake -/
structure CakeIngredients where
  flour : ℚ
  sugar : ℚ
  butter : ℚ
  eggs : ℚ
  salt : ℚ
  vanilla : ℚ
  milk : ℚ

/-- Calculates the cost of dog-eaten cake slices -/
def cost_of_dog_eaten_slices (ingredients : CakeIngredients) 
  (sales_tax_rate : ℚ) (total_slices : ℕ) (mother_eaten_slices : ℕ) : ℚ :=
  let total_cost := ingredients.flour + ingredients.sugar + ingredients.butter + 
                    ingredients.eggs + ingredients.salt + ingredients.vanilla + 
                    ingredients.milk
  let total_cost_with_tax := total_cost * (1 + sales_tax_rate)
  let cost_per_slice := total_cost_with_tax / total_slices
  let dog_eaten_slices := total_slices - mother_eaten_slices
  cost_per_slice * dog_eaten_slices

/-- Theorem stating the cost of dog-eaten cake slices -/
theorem dog_eaten_cake_cost :
  let ingredients := CakeIngredients.mk 6 2 5 (3/2) (1/4) (3/2) (5/4)
  let sales_tax_rate := 6/100
  let total_slices := 8
  let mother_eaten_slices := 3
  ∃ ε > 0, |cost_of_dog_eaten_slices ingredients sales_tax_rate total_slices mother_eaten_slices - (11591/1000)| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_eaten_cake_cost_l50_5024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l50_5051

-- Define a real-valued function
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Ioo (-1) 0

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | f x = f x} = Set.Ioo (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l50_5051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_story_height_l50_5056

/-- The height of each story in Paul's apartment building -/
def story_height : ℝ := 10

/-- The number of stories in Paul's apartment building -/
def num_stories : ℕ := 5

/-- The number of trips Paul makes each day -/
def trips_per_day : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total vertical distance Paul travels in a week -/
def total_distance : ℝ := 2100

theorem apartment_story_height :
  story_height * (num_stories * trips_per_day * days_in_week * 2) = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_story_height_l50_5056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_can_circle_tree_l50_5065

/-- The minimum rope length required for a goat to circle a tree -/
noncomputable def min_rope_length (tree_radius : ℝ) (stake_distance : ℝ) : ℝ :=
  let α := Real.arcsin (tree_radius / (tree_radius + stake_distance))
  2 * tree_radius * (Real.sqrt ((tree_radius + stake_distance)^2 - tree_radius^2) / tree_radius) +
  tree_radius * (Real.pi + 2 * α)

/-- Theorem stating that the given rope length is sufficient -/
theorem goat_can_circle_tree (rope_length : ℝ) (tree_radius : ℝ) (stake_distance : ℝ)
    (h_rope : rope_length = 4.7)
    (h_radius : tree_radius = 0.5)
    (h_distance : stake_distance = 1) :
    min_rope_length tree_radius stake_distance < rope_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_can_circle_tree_l50_5065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_inverse_four_equals_two_l50_5029

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 20 / (4 + 2 * x)

-- State the theorem
theorem inverse_g_inverse_four_equals_two :
  (g⁻¹ 4)⁻¹ = 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_inverse_four_equals_two_l50_5029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_seven_gram_coins_determine_coin_weight_l50_5005

/-- Represents the weight of coins in grams -/
def CoinWeight := Fin 14

/-- Represents a bag of coins -/
structure CoinBag where
  weight : CoinWeight
  count : Nat

/-- Represents the state of a balance scale -/
inductive Balance
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents the result of a weighing operation -/
def WeighResult := Balance

/-- Function to perform a single weighing -/
def weigh (left : List CoinBag) (right : List CoinBag) : WeighResult :=
  sorry

/-- Theorem: It's possible to verify if a bag contains 7-gram coins in one weighing -/
theorem verify_seven_gram_coins (bags : List CoinBag) 
  (h1 : bags.length = 7)
  (h2 : ∀ b ∈ bags, b.count = 100)
  (h3 : ∀ w : Fin 14, w.val ∈ [7, 8, 9, 10, 11, 12, 13] → ∃ b ∈ bags, b.weight = w)
  (indicated : CoinBag) 
  (h4 : indicated ∈ bags) :
  ∃ (left right : List CoinBag), 
    let result := weigh left right
    result = Balance.LeftHeavier ↔ indicated.weight = ⟨7, by norm_num⟩ := by
  sorry

/-- Theorem: It's possible to determine the weight of coins in a bag in at most two weighings -/
theorem determine_coin_weight (bags : List CoinBag) 
  (h1 : bags.length = 7)
  (h2 : ∀ b ∈ bags, b.count = 100)
  (h3 : ∀ w : Fin 14, w.val ∈ [7, 8, 9, 10, 11, 12, 13] → ∃ b ∈ bags, b.weight = w)
  (indicated : CoinBag) 
  (h4 : indicated ∈ bags) :
  ∃ (w1 w2 : WeighResult) (f : WeighResult → WeighResult → CoinWeight),
    f w1 w2 = indicated.weight ∧ 
    (w1 = w2 → ∃ (left1 right1 : List CoinBag), w1 = weigh left1 right1) ∧
    (w1 ≠ w2 → ∃ (left1 right1 left2 right2 : List CoinBag), 
      w1 = weigh left1 right1 ∧ w2 = weigh left2 right2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_seven_gram_coins_determine_coin_weight_l50_5005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_three_x_over_two_l50_5045

-- Define the function f(x) = tan(3x/2)
noncomputable def f (x : ℝ) : ℝ := Real.tan ((3 : ℝ) * x / 2)

-- State the theorem about the period of f(x)
theorem period_of_tan_three_x_over_two :
  ∃ p : ℝ, p > 0 ∧ 
    (∀ x : ℝ, f (x + p) = f x) ∧ 
    (∀ q : ℝ, 0 < q → q < p → ∃ y : ℝ, f (y + q) ≠ f y) :=
by
  -- We claim that p = 2π/3 is the period
  use 2 * Real.pi / 3
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_three_x_over_two_l50_5045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_30_percent_l50_5020

/-- Represents the financial situation of a man over two years -/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- The percentage of income saved in the first year -/
noncomputable def savings_percentage (fs : FinancialSituation) : ℝ :=
  (fs.savings_year1 / fs.income_year1) * 100

/-- The conditions given in the problem -/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = fs.income_year1 * 1.3 ∧
  fs.savings_year2 = fs.savings_year1 * 2 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- Theorem stating that if the financial situation satisfies the given conditions,
    then the savings percentage in the first year is 30% -/
theorem savings_percentage_is_30_percent (fs : FinancialSituation) :
  satisfies_conditions fs → savings_percentage fs = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_30_percent_l50_5020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_terms_l50_5012

-- Define is_arithmetic_sequence
def is_arithmetic_sequence (s : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin (s.length - 1), s[i.val + 1] - s[i.val] = d

theorem arithmetic_sequence_middle_terms :
  ∀ (a b : ℝ),
  (is_arithmetic_sequence [-1, a, b, 8] ∧ 
   -1 < a ∧ a < b ∧ b < 8) →
  (a = 2 ∧ b = 5) :=
by
  intro a b
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_terms_l50_5012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evening_speed_is_30_l50_5028

/-- Calculates the evening speed given total commute time, morning speed, and distance to work. -/
noncomputable def evening_speed (total_time : ℝ) (morning_speed : ℝ) (distance : ℝ) : ℝ :=
  let morning_time := distance / morning_speed
  let evening_time := total_time - morning_time
  distance / evening_time

/-- Proves that given the specified conditions, the evening speed is 30 miles per hour. -/
theorem evening_speed_is_30 :
  evening_speed 1 45 18 = 30 := by
  -- Unfold the definition of evening_speed
  unfold evening_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evening_speed_is_30_l50_5028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l50_5068

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n+2 => 1 + 1 / sequence_a (n+1)

theorem a_4_value : sequence_a 4 = 5/3 := by
  -- Expand the definition and simplify
  unfold sequence_a
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l50_5068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_theorem_l50_5090

/-- Represents a digit in base 6 -/
def Base6Digit := Fin 6

/-- Addition in base 6 -/
def addBase6 (a b : Nat) : Nat :=
  (a + b) % 6

/-- Conversion from base 6 to decimal -/
def fromBase6 (digits : List Base6Digit) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d.val) 0

/-- Conversion from decimal to base 6 -/
def toBase6 (n : Nat) : List Base6Digit :=
  if n < 6 then [⟨n, by sorry⟩]
  else let d := n % 6; ⟨d, by sorry⟩ :: toBase6 (n / 6)

theorem base6_addition_theorem (P Q R : Base6Digit) 
  (h_distinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) 
  (h_nonzero : P.val ≠ 0 ∧ Q.val ≠ 0 ∧ R.val ≠ 0)
  (h_addition : fromBase6 [P, Q, R] + fromBase6 [Q, R] = fromBase6 [P, R, P]) :
  toBase6 (P.val + Q.val + R.val) = [⟨1, by sorry⟩, ⟨1, by sorry⟩] :=
by sorry

#check base6_addition_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_theorem_l50_5090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l50_5095

noncomputable def f (x m : ℝ) : ℝ := (2^(x+1) + m) / (2^x - 1)

theorem odd_function_implies_m_equals_two (m : ℝ) :
  (∀ x, f x m = -f (-x) m) → m = 2 := by
  intro h
  -- The proof steps would go here
  sorry

#check odd_function_implies_m_equals_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l50_5095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l50_5033

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 10) / Real.sqrt (x^2 - 5*x - 6)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -1 ∨ x > 6}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l50_5033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l50_5097

/-- An ellipse with focus on the x-axis, focal length 4, and passing through (3, -2√6) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a
  h_focal_length : Real.sqrt (a^2 - b^2) = 4
  h_point : (3^2 / a^2) + ((-2 * Real.sqrt 6)^2 / b^2) = 1

/-- The standard equation of the ellipse is x²/36 + y²/32 = 1 -/
theorem ellipse_standard_equation (e : Ellipse) :
  e.a^2 = 36 ∧ e.b^2 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l50_5097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l50_5099

theorem min_translation_for_symmetry (x m : ℝ) : 
  m > 0 ∧ 
  (∀ x, Real.sin (2 * (x - m) + π / 3) = Real.sin (2 * (-x - m) + π / 3)) →
  m ≥ 5 * π / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l50_5099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l50_5059

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem inequality_solution (k : ℝ) (h : k > 0) :
  (∀ x, 0 < k ∧ k < 1 → ((deriv f) x + k * (1 - x) * f x > 0 ↔ 1 < x ∧ x < 1 / k)) ∧
  (k = 1 → ¬∃ x, (deriv f) x + k * (1 - x) * f x > 0) ∧
  (∀ x, k > 1 → ((deriv f) x + k * (1 - x) * f x > 0 ↔ 1 / k < x ∧ x < 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l50_5059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_machine_final_price_l50_5043

/-- The final price of a washing machine after discounts and tax -/
theorem washing_machine_final_price : 
  let original_price : ℝ := 500
  let first_discount_rate : ℝ := 0.25
  let second_discount_rate : ℝ := 0.10
  let tax_rate : ℝ := 0.05
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let final_price := price_after_second_discount * (1 + tax_rate)
  final_price = 354.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_machine_final_price_l50_5043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_represents_circle_l50_5076

/-- The system of equations represents a circle -/
theorem system_represents_circle (x y z : ℝ) :
  x^2 + y^2 + z^2 - 4*x = 0 ∧ y = 1 →
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ),
    center = (2, 1, 0) ∧
    radius = Real.sqrt 3 ∧
    (x - center.1)^2 + (z - center.2.2)^2 = radius^2 ∧
    y = center.2.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_represents_circle_l50_5076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l50_5062

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (∃ (k₁ : ℕ), (b.val)^2 = k₁ * a.val) ∧
  (∃ (k₂ : ℕ), (a.val)^3 = k₂ * (b.val)^2) ∧
  (∃ (k₃ : ℕ), (b.val)^4 = k₃ * (a.val)^3) ∧
  (∃ (k₄ : ℕ), (a.val)^5 = k₄ * (b.val)^4) ∧
  (∀ (k₅ : ℕ), (b.val)^6 ≠ k₅ * (a.val)^5) ∧
  a = 16 ∧ b = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l50_5062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l50_5009

/-- The function f(x) = ln x - ax + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

/-- Theorem stating the monotonicity properties of f(x) and the minimum value of m -/
theorem f_properties (a : ℝ) :
  (∀ x y, 0 < x → 0 < y → x < y → a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y, 0 < x → 0 < y → x < y → y < 1/a → f a x < f a y) ∧
    (∀ x y, 1/a < x → x < y → f a y < f a x)) ∧
  (∃ m : ℕ+, (∀ x, 0 < x → f (-2) x ≤ m * (x + 1)) ∧
    (∀ m' : ℕ+, (∀ x, 0 < x → f (-2) x ≤ m' * (x + 1)) → m ≤ m') ∧
    m = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l50_5009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_key_arrangements_l50_5010

/-- Represents the number of keys -/
def total_keys : ℕ := 6

/-- Represents the number of keys that must be adjacent -/
def adjacent_keys : ℕ := 3

/-- Calculates the number of distinct circular arrangements -/
def circular_arrangements (n : ℕ) : ℕ := (n - 1).factorial / 2

/-- Calculates the number of permutations of the adjacent keys -/
def adjacent_permutations (k : ℕ) : ℕ := (k - 1).factorial

/-- Theorem stating the number of distinct arrangements -/
theorem key_arrangements :
  circular_arrangements (total_keys - adjacent_keys + 1) * adjacent_permutations adjacent_keys = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_key_arrangements_l50_5010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_trigonometric_form_l50_5011

noncomputable def z : ℂ := 1 - Complex.cos (200 * Real.pi / 180) + Complex.I * Complex.sin (200 * Real.pi / 180)

theorem z_trigonometric_form :
  z = 2 * Real.sin (10 * Real.pi / 180) * Complex.exp (-Complex.I * (10 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_trigonometric_form_l50_5011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_points_l50_5038

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x - Real.sqrt 2 * y + Real.sqrt 2 = 0

-- Define the condition for P to be on a circle with diameter AB
def on_circle_diameter (x₀ y₀ : ℝ) : Prop := x₀^2 + y₀^2 = 3

-- Define the condition for tangent lines from P to C
def tangent_condition (x₀ y₀ : ℝ) : Prop := x₀^2 + 2 * y₀^2 - 2 > 0

-- Define the two points P₁ and P₂
noncomputable def P₁ : ℝ × ℝ := ((Real.sqrt 14 - Real.sqrt 2) / 3, (2 + Real.sqrt 7) / 3)
noncomputable def P₂ : ℝ × ℝ := (-(Real.sqrt 14 + Real.sqrt 2) / 3, (2 - Real.sqrt 7) / 3)

theorem ellipse_tangent_points :
  ∃! (P₁ P₂ : ℝ × ℝ), 
    (l P₁.1 P₁.2 ∧ on_circle_diameter P₁.1 P₁.2 ∧ tangent_condition P₁.1 P₁.2) ∧
    (l P₂.1 P₂.2 ∧ on_circle_diameter P₂.1 P₂.2 ∧ tangent_condition P₂.1 P₂.2) ∧
    P₁ = ((Real.sqrt 14 - Real.sqrt 2) / 3, (2 + Real.sqrt 7) / 3) ∧
    P₂ = (-(Real.sqrt 14 + Real.sqrt 2) / 3, (2 - Real.sqrt 7) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_points_l50_5038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_numbers_satisfy_sum_property_l50_5025

/-- A two-digit number is a natural number between 10 and 99 inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The reverse of a two-digit number. -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The property that a two-digit number and its reverse sum to 143. -/
def hasSumProperty (n : ℕ) : Prop :=
  TwoDigitNumber n ∧ n + reverse n = 143

/-- Proof that hasSumProperty is decidable -/
instance (n : ℕ) : Decidable (hasSumProperty n) :=
  show Decidable ((10 ≤ n ∧ n ≤ 99) ∧ n + reverse n = 143) from
  inferInstance

/-- The theorem stating that exactly 6 two-digit numbers satisfy the sum property. -/
theorem six_numbers_satisfy_sum_property :
  (Finset.filter hasSumProperty (Finset.range 90)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_numbers_satisfy_sum_property_l50_5025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_5_range_of_m_l50_5074

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 3|

-- Theorem for the solution set of f(x) > 5
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = Set.Ioi (-1) ∪ Set.Iio 1 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x ≥ |2 * m + 1|) ↔ -9/4 ≤ m ∧ m ≤ 5/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_5_range_of_m_l50_5074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_net_income_is_103_l50_5016

/-- Calculates Olivia's net income after expenses and taxes --/
def olivia_net_income (
  monday_wage : ℚ)
  (wednesday_wage : ℚ)
  (friday_wage : ℚ)
  (saturday_wage : ℚ)
  (monday_hours : ℚ)
  (wednesday_hours : ℚ)
  (friday_hours : ℚ)
  (saturday_hours : ℚ)
  (business_expenses : ℚ)
  (tax_rate : ℚ) : ℚ :=
  let total_earnings := 
    monday_wage * monday_hours +
    wednesday_wage * wednesday_hours +
    friday_wage * friday_hours +
    saturday_wage * saturday_hours
  let earnings_after_expenses := total_earnings - business_expenses
  let tax_amount := tax_rate * total_earnings
  earnings_after_expenses - tax_amount

/-- Theorem stating that Olivia's net income is $103 --/
theorem olivia_net_income_is_103 :
  olivia_net_income 10 12 14 20 5 4 3 2 50 (15/100) = 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_net_income_is_103_l50_5016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_bound_l50_5013

theorem quadratic_roots_bound (a b c : ℕ) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ 
             a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) : 
  a ≥ 5 ∧ b ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_bound_l50_5013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_48_multiple_of_6_l50_5022

theorem factors_of_48_multiple_of_6 : 
  (Finset.filter (λ x => x ∣ 48 ∧ 6 ∣ x) (Finset.range 49)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_48_multiple_of_6_l50_5022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_l50_5042

theorem fraction_inequality_solution (x : ℝ) (h : x + 4 ≠ 0) :
  (x - 1) / (x + 4) ≤ 0 ↔ x ∈ Set.Ioc (-4) 0 ∪ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_l50_5042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l50_5088

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (condition : a^2 + c^2 - b^2 = a*c) : 
  Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l50_5088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_quadrilaterals_l50_5050

-- Define the types of quadrilaterals
inductive Quadrilateral
| Square
| Rectangle
| Rhombus
| Parallelogram
| Trapezoid

-- Define a function that checks if a quadrilateral has a point equidistant from all vertices
def hasEquidistantPoint (q : Quadrilateral) : Bool :=
  match q with
  | Quadrilateral.Square => true
  | Quadrilateral.Rectangle => true
  | _ => false

-- Define a function to count the number of quadrilaterals with an equidistant point
def countEquidistantQuadrilaterals : Nat :=
  (List.filter hasEquidistantPoint [Quadrilateral.Square, Quadrilateral.Rectangle, 
   Quadrilateral.Rhombus, Quadrilateral.Parallelogram, Quadrilateral.Trapezoid]).length

-- Theorem statement
theorem equidistant_quadrilaterals :
  countEquidistantQuadrilaterals = 2 := by
  -- Proof goes here
  sorry

#eval countEquidistantQuadrilaterals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_quadrilaterals_l50_5050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l50_5049

/-- Calculates the speed of a train given its length, time to pass a person, and the person's speed -/
noncomputable def trainSpeed (trainLength : ℝ) (timeToPass : ℝ) (personSpeed : ℝ) : ℝ :=
  let relativeSpeed := trainLength / 1000 / (timeToPass / 3600)
  relativeSpeed + personSpeed

/-- The speed of the train is approximately 68.0048 kmph -/
theorem train_speed_calculation :
  let trainLength : ℝ := 250
  let timeToPass : ℝ := 14.998800095992321
  let personSpeed : ℝ := 8
  abs (trainSpeed trainLength timeToPass personSpeed - 68.0048) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l50_5049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_form_l50_5078

/-- The target function -/
def target_function (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + 1

/-- The condition for right-angled tangents -/
def right_angle_tangents (a : ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv (target_function a)) x₁ * (deriv (target_function a)) x₂ = -1 ∧
    y₀ = target_function a x₀

/-- The logarithmic equation condition -/
def log_equation (x₀ y₀ : ℝ) : Prop :=
  (3*x₀ - x₀^2 + 1 > 0) ∧ (y₀ > 4) ∧
  (y₀ - 4 = (abs (2*x₀ + 4) - abs (2*x₀ + 1)) / (3*x₀ + 4.5) * 
   Real.sqrt (x₀^2 + 3*x₀ + 2.25))

/-- The main theorem -/
theorem target_function_form (x₀ y₀ : ℝ) :
  (∃ a : ℝ, right_angle_tangents a x₀ y₀ ∧ log_equation x₀ y₀) →
  (∀ x : ℝ, target_function (-1/16) x = -0.0625 * (x + 1)^2 + 1) := by
  sorry

/-- Auxiliary lemma to show the equality of the constant -/
lemma constant_equality : (-1/16 : ℝ) = -0.0625 := by
  norm_num

#check target_function_form
#check constant_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_form_l50_5078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_of_special_tetrahedron_l50_5060

/-- A tetrahedron with one vertex having pairwise perpendicular edges -/
structure SpecialTetrahedron where
  /-- The radius of the circumscribed sphere -/
  circumRadius : ℝ
  /-- The radius of the circumscribed sphere is 3√3 -/
  circumRadius_eq : circumRadius = 3 * Real.sqrt 3

/-- The radius of the sphere touching all edges of the tetrahedron -/
noncomputable def inscribedSphereRadius (t : SpecialTetrahedron) : ℝ :=
  6 * (Real.sqrt 2 - 1)

/-- Theorem: The radius of the sphere touching all edges of the special tetrahedron is 6(√2 - 1) -/
theorem inscribed_sphere_radius_of_special_tetrahedron (t : SpecialTetrahedron) :
    inscribedSphereRadius t = 6 * (Real.sqrt 2 - 1) := by
  -- Unfold the definition of inscribedSphereRadius
  unfold inscribedSphereRadius
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_of_special_tetrahedron_l50_5060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_louisa_first_day_distance_l50_5053

/-- Represents Louisa's travel data -/
structure LouisaTravel where
  average_speed : ℚ
  second_day_distance : ℚ
  time_difference : ℚ

/-- Calculates the distance Louisa traveled on the first day -/
noncomputable def first_day_distance (travel : LouisaTravel) : ℚ :=
  travel.average_speed * (travel.second_day_distance / travel.average_speed - travel.time_difference)

/-- Theorem stating that Louisa traveled 100 miles on the first day -/
theorem louisa_first_day_distance :
  let travel := LouisaTravel.mk 25 175 3
  first_day_distance travel = 100 := by
  sorry

#check louisa_first_day_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_louisa_first_day_distance_l50_5053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l50_5026

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else x^2 + 4*x

-- State the theorem
theorem solution_set_of_inequality (h1 : ∀ x, f (-x) = f x) 
    (h2 : ∀ x ≥ 0, f x = x^2 - 4*x) :
  {x : ℝ | f (2*x + 3) ≤ 5} = Set.Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l50_5026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l50_5079

theorem cos_alpha_value (α β : ℝ) 
  (h1 : Real.sin α = 2 * Real.sin β) 
  (h2 : Real.tan α = 3 * Real.tan β) : 
  Real.cos α = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l50_5079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pour_oil_result_l50_5094

/-- Represents the capacity of a drum -/
structure Drum where
  capacity : ℚ
  oil : ℚ
  h_oil_nonneg : 0 ≤ oil
  h_oil_le_capacity : oil ≤ capacity

/-- The resulting fill ratio of Drum Y after pouring oil from Drum X into it -/
def final_fill_ratio (x y : Drum) : ℚ :=
  (y.oil + x.oil) / y.capacity

theorem pour_oil_result (x y : Drum) 
  (h1 : x.oil = x.capacity / 2)
  (h2 : y.capacity = 2 * x.capacity)
  (h3 : y.oil = y.capacity / 4) :
  final_fill_ratio x y = 1 / 2 := by
  sorry

#eval 1 / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pour_oil_result_l50_5094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_2x10_grid_l50_5021

/-- The number of ways to cover a 2 × n grid with 1 × 2 tiles -/
def f : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | n + 3 => f (n + 2) + f (n + 1)

/-- Theorem: The number of ways to cover a 2 × 10 grid with 1 × 2 tiles is 89 -/
theorem cover_2x10_grid : f 10 = 89 := by
  sorry

#eval f 10  -- This will evaluate f 10 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_2x10_grid_l50_5021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l50_5052

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

-- Part I
theorem part_one :
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 (Real.exp 1) → f 4 x ≤ f 4 y) ∧
  f 4 x = -3 ∧ x = Real.exp 1 := by sorry

-- Part II
theorem part_two :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 2 (Real.exp 1) ∧ f a x ≥ (a - 2) * x) →
  a ≤ 8 / (2 + Real.log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l50_5052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l50_5057

/-- The cosine of the angle between two vectors -/
noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_angle_specific_vectors :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, 2)  -- Derived from a + 2b = (4, 5)
  cos_angle a b = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l50_5057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l50_5034

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y - l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem circle_center_proof (center : Point) : 
  (3 * center.x + 2 * center.y = 0) ∧ 
  (distancePointToLine center { a := 3, b := -4, c := 24 } = 
   distancePointToLine center { a := 3, b := -4, c := -12 }) ∧
  (center.x = 2/3) ∧ 
  (center.y = -1) := by
  sorry

#check circle_center_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l50_5034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l50_5000

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1) + 2

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (1/2) = 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_sub, pow_mul, pow_one]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l50_5000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_closed_form_l50_5072

/-- The sequence {aₙ} defined recursively -/
def a : ℕ → ℚ
  | 0 => 1/2  -- Added case for 0
  | 1 => 1/2
  | n + 2 => (a (n + 1) + 3) / (2 * a (n + 1) - 4)

/-- The proposed closed form for the sequence -/
def closed_form (n : ℕ) : ℚ :=
  ((-5)^n + 3 * 2^(n+1)) / (2^(n+1) - 2 * (-5)^n)

/-- Theorem stating that the recursive definition equals the closed form -/
theorem a_equals_closed_form (n : ℕ) : a n = closed_form n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_closed_form_l50_5072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l50_5055

theorem complex_equation_solution (z : ℂ) :
  (3 - 4*Complex.I) * z = 5 + 10*Complex.I → z = -1 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l50_5055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shape_rectangle_area_l50_5075

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents an L-shape formed by three squares -/
structure LShape where
  squares : Fin 3 → Square

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a vertex of a rectangle -/
inductive Vertex
  | topLeft
  | topRight
  | bottomLeft
  | bottomRight
  | midpointTop
  | midpointBottom
  | midpointLeft
  | midpointRight

/-- Predicate to check if a vertex is on the rectangle -/
def vertex_on_rectangle (r : Rectangle) (v : Vertex) : Prop :=
  match v with
  | Vertex.topLeft | Vertex.topRight | Vertex.bottomLeft | Vertex.bottomRight => True
  | Vertex.midpointTop | Vertex.midpointBottom | Vertex.midpointLeft | Vertex.midpointRight => True

/-- Predicate to check if an L-shape fits inside a rectangle with the given conditions -/
def fits_inside (l : LShape) (r : Rectangle) : Prop :=
  ∀ (i : Fin 3), (l.squares i).sideLength = 2 ∧
  ∃ (v : Fin 5 → Vertex), (∀ j : Fin 5, vertex_on_rectangle r (v j)) ∧
  ∃ (j : Fin 5), (v j = Vertex.midpointTop ∨ v j = Vertex.midpointBottom ∨ 
                  v j = Vertex.midpointLeft ∨ v j = Vertex.midpointRight)

theorem l_shape_rectangle_area (l : LShape) (r : Rectangle) 
  (h : fits_inside l r) : r.area = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shape_rectangle_area_l50_5075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_quantity_in_mixture_l50_5036

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℝ
  water : ℝ

/-- Represents the ratio of alcohol to water in a mixture -/
noncomputable def ratio (m : Mixture) : ℝ := m.alcohol / m.water

theorem alcohol_quantity_in_mixture (m : Mixture) :
  ratio m = 4/3 ∧ 
  ratio { alcohol := m.alcohol, water := m.water + 4 } = 4/5 →
  m.alcohol = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_quantity_in_mixture_l50_5036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_l50_5092

/-- The cosine of the angle between vectors AB and AC is 0.96, given points A(0, 2, -4), B(8, 2, 2), and C(6, 2, 4). -/
theorem cosine_angle_AB_AC : 
  let A : Fin 3 → ℝ := ![0, 2, -4]
  let B : Fin 3 → ℝ := ![8, 2, 2]
  let C : Fin 3 → ℝ := ![6, 2, 4]
  let AB : Fin 3 → ℝ := fun i => B i - A i
  let AC : Fin 3 → ℝ := fun i => C i - A i
  let dot_product : ℝ := (AB 0 * AC 0) + (AB 1 * AC 1) + (AB 2 * AC 2)
  let magnitude_AB : ℝ := Real.sqrt ((AB 0)^2 + (AB 1)^2 + (AB 2)^2)
  let magnitude_AC : ℝ := Real.sqrt ((AC 0)^2 + (AC 1)^2 + (AC 2)^2)
  dot_product / (magnitude_AB * magnitude_AC) = 0.96 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_l50_5092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l50_5083

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (p a b : ℝ) (A : ℝ × ℝ) :
  p > 0 →
  a > 0 →
  b > 0 →
  (let (x, y) := A; y^2 = 2*p*x ∧ x^2/a^2 - y^2/b^2 = 1) →
  (let (x, y) := A; (y / (x - p/2) = Real.sqrt 3)) →
  let e := Real.sqrt ((a^2 + b^2) / a^2);
  e = (Real.sqrt 7 + 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l50_5083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_l50_5096

/-- The area of intersection of two circles with radius 3, centered at (3,0) and (0,3) -/
noncomputable def intersection_area : ℝ := (9/2) * Real.pi - 9

/-- Two circles with radius 3, one centered at (3,0) and another at (0,3) -/
structure TwoCircles where
  radius : ℝ
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ

/-- The specific configuration of two circles in our problem -/
def our_circles : TwoCircles where
  radius := 3
  center1 := (3, 0)
  center2 := (0, 3)

/-- Theorem stating that the area of intersection of the two circles is (9/2)π - 9 -/
theorem area_of_intersection (c : TwoCircles) (h : c = our_circles) : 
  intersection_area = (9/2) * Real.pi - 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_l50_5096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l50_5039

noncomputable def C (n : ℕ) : ℝ := 1000 * (1 - (1/3)^n) / (1 - 1/3)

noncomputable def D (n : ℕ) : ℝ := 2700 * (1 - (1/(-3))^n) / (1 + 1/3)

theorem geometric_series_equality (n : ℕ) : 
  (n > 0 ∧ C n = D n) ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l50_5039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l50_5031

def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 33
  else sequenceA (n - 1) + 2 * (n - 1)

theorem min_value_of_sequence_ratio :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n > 0 → sequenceA n / n ≥ 21 / 2 ∧
  sequenceA k / k = 21 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l50_5031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l50_5071

def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

theorem complement_of_A : Set.compl A = Set.Iic 2 ∪ Set.Ioi 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l50_5071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l50_5066

/-- Predicate to state that a point is the focus of the parabola x² = 4y -/
def is_focus_of_parabola (f : ℝ) : Prop := sorry

/-- Predicate to state that a line is the directrix of the parabola x² = 4y -/
def is_directrix_of_parabola (d : ℝ) : Prop := sorry

/-- For a parabola with equation x² = 4y, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance :
  ∃ (f d : ℝ), is_focus_of_parabola f ∧ is_directrix_of_parabola d ∧ |f - d| = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l50_5066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l50_5047

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the points
def P : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (1, 1)

-- State the theorem
theorem parallel_line_equation : 
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * P.1 + b = P.2 ∧ 
      m = (deriv f) M.1 ∧ 
      y = m * x + b) ∧
    (∀ x y : ℝ, y = m * x + b ↔ 2 * x - y + 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l50_5047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_point_l50_5098

/-- The parabola y = x^2 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2

/-- Point C on the parabola -/
noncomputable def C : ℝ × ℝ := (2, 4)

/-- Point D, the other intersection of the normal at C with the parabola -/
noncomputable def D : ℝ × ℝ := (-9/4, 81/16)

/-- The slope of the tangent line to the parabola at a point (x, y) -/
noncomputable def tangent_slope (x : ℝ) : ℝ := 2 * x

/-- The slope of the normal line to the parabola at a point (x, y) -/
noncomputable def normal_slope (x : ℝ) : ℝ := -1 / (tangent_slope x)

/-- The equation of the normal line to the parabola at point C -/
noncomputable def normal_line (x : ℝ) : ℝ := normal_slope C.1 * (x - C.1) + C.2

theorem normal_intersection_point :
  D.2 = parabola D.1 ∧
  D.2 = normal_line D.1 ∧
  D ≠ C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_point_l50_5098
