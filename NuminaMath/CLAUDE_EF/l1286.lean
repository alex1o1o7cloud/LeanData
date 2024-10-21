import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_12m_squared_l1286_128650

/-- Given an odd integer m with exactly 11 positive divisors, 
    the number of positive divisors of 12m^2 is 126. -/
theorem divisors_of_12m_squared (m : ℕ) 
  (h_odd : Odd m) 
  (h_divisors : (Nat.divisors m).card = 11) : 
  (Nat.divisors (12 * m^2)).card = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_12m_squared_l1286_128650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_values_with_connection_one_l1286_128640

def connection (a b : ℕ) : ℚ :=
  (Nat.lcm a b : ℚ) / (a * b : ℚ)

theorem count_values_with_connection_one : 
  (Finset.filter (fun y => y < 20 ∧ connection y 6 = 1) (Finset.range 20)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_values_with_connection_one_l1286_128640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l1286_128670

theorem condition_relationship :
  (∀ a : ℝ, (|a| ≤ 1 → a ≤ 1)) ∧ (∃ a : ℝ, (a ≤ 1 ∧ ¬(|a| ≤ 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l1286_128670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_quadrant_l1286_128610

noncomputable def terminal_side_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (Real.pi/2 < (α/2) % (2*Real.pi) ∧ (α/2) % (2*Real.pi) < Real.pi) ∨
  (3*Real.pi/2 < (α/2) % (2*Real.pi) ∧ (α/2) % (2*Real.pi) < 2*Real.pi)

theorem angle_half_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α < 0) 
  (h2 : Real.sin α * Real.tan α > 0) : 
  terminal_side_in_second_or_fourth_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_quadrant_l1286_128610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_two_l1286_128653

theorem log_sum_equals_two : Real.log 2 + (Real.sqrt 2 - 1)^0 + Real.log 5 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_two_l1286_128653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l1286_128612

/-- Given a circular sector with perimeter 144 cm and radius 28.000000000000004 cm,
    its central angle is approximately 180.21 degrees. -/
theorem sector_central_angle (perimeter : ℝ) (radius : ℝ) (angle : ℝ) : 
  perimeter = 144 →
  radius = 28.000000000000004 →
  abs (angle - 180.21) < 0.01 →
  perimeter = 2 * radius + (angle / 360) * 2 * Real.pi * radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l1286_128612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_in_position_correct_l1286_128627

/-- The probability of at least one digit appearing in its own position in a 9-digit number formed by 9 independent draws with replacement from the set {1, 2, 3, 4, 5, 6, 7, 8, 9} -/
noncomputable def prob_at_least_one_in_position : ℝ :=
  1 - (8/9)^9

/-- The set of possible digits -/
def digit_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The number of digits -/
def n : ℕ := 9

/-- The probability of not drawing the corresponding digit in one draw -/
noncomputable def prob_not_in_position : ℝ := 8/9

theorem prob_at_least_one_in_position_correct :
  prob_at_least_one_in_position = 1 - (prob_not_in_position ^ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_in_position_correct_l1286_128627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_pharmacies_l1286_128683

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a pharmacy on the grid -/
structure Pharmacy where
  location : GridPoint

/-- The grid size -/
def gridSize : Nat := 9

/-- The number of pharmacies -/
def numPharmacies : Nat := 12

/-- The maximum distance a pharmacy can cover -/
def maxCoverDistance : Nat := 3

/-- Function to check if a point is within the coverage of a pharmacy -/
def isWithinCoverage (point : GridPoint) (pharmacy : Pharmacy) : Bool :=
  let dx := Int.natAbs (point.x - pharmacy.location.x)
  let dy := Int.natAbs (point.y - pharmacy.location.y)
  dx ≤ maxCoverDistance ∧ dy ≤ maxCoverDistance

/-- Theorem stating that 12 pharmacies are not enough to cover the entire grid -/
theorem insufficient_pharmacies :
  ∀ (pharmacies : List Pharmacy),
    pharmacies.length = numPharmacies →
    ∃ (point : GridPoint),
      point.x < gridSize ∧ point.y < gridSize ∧
      ∀ (pharmacy : Pharmacy),
        pharmacy ∈ pharmacies →
        ¬ isWithinCoverage point pharmacy :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_pharmacies_l1286_128683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_calculation_l1286_128698

/-- Calculates the monthly rent per square meter required to recoup a construction investment --/
noncomputable def monthly_rent_per_sqm (investment : ℝ) (service_life : ℕ) (annual_interest_rate : ℝ) : ℝ :=
  let annual_rent := (investment * annual_interest_rate * (1 + annual_interest_rate) ^ service_life) /
                     ((1 + annual_interest_rate) ^ service_life - 1)
  annual_rent / 12

/-- The monthly rent per square meter is approximately 1.14 yuan --/
theorem monthly_rent_calculation :
  let investment := (250 : ℝ)
  let service_life := (50 : ℕ)
  let annual_interest_rate := (0.05 : ℝ)
  abs (monthly_rent_per_sqm investment service_life annual_interest_rate - 1.14) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_calculation_l1286_128698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_eq_one_l1286_128664

/-- A function f with the given properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b / x + 3

/-- Theorem stating that if f(4) = 5, then f(-4) = 1 -/
theorem f_neg_four_eq_one (a b : ℝ) :
  f a b 4 = 5 → f a b (-4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_eq_one_l1286_128664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1286_128687

/-- The differential equation y'' + 3y' + 2y = x sin x -/
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = x * Real.sin x

/-- The general solution of the differential equation -/
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.exp (-x) + C₂ * Real.exp (-2*x) + 
  (-3/10*x + 17/50) * Real.cos x + (1/10*x + 3/25) * Real.sin x

/-- Theorem stating that the general_solution satisfies the differential equation -/
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1286_128687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_fourth_quadrant_l1286_128624

theorem tangent_fourth_quadrant (α : Real) :
  (α > -π ∧ α < -π/2) →  -- α is in the fourth quadrant
  Real.sin (π/2 + α) = 4/5 →
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_fourth_quadrant_l1286_128624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l1286_128678

/-- The time when the first candle is three times the height of the second candle -/
noncomputable def candle_time_ratio : ℚ := 40 / 11

theorem candle_height_ratio :
  let initial_height : ℚ := 1
  let burn_time_1 : ℚ := 5
  let burn_time_2 : ℚ := 4
  let burn_rate_1 : ℚ := initial_height / burn_time_1
  let burn_rate_2 : ℚ := initial_height / burn_time_2
  let height_1 (t : ℚ) : ℚ := initial_height - burn_rate_1 * t
  let height_2 (t : ℚ) : ℚ := initial_height - burn_rate_2 * t
  ∃ t : ℚ, t = candle_time_ratio ∧ height_1 t = 3 * height_2 t :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l1286_128678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l1286_128673

theorem quadratic_inequality_solution_range (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x, a*(x^2 + 1) + b*(x - 1) + c < 2*a*x ↔ x < 0 ∨ x > 3) :=
by
  intro h
  sorry  -- The detailed proof would go here

#check quadratic_inequality_solution_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l1286_128673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_six_heads_correct_l1286_128667

/-- The probability of getting exactly 6 heads when flipping a fair coin 8 times -/
def prob_six_heads_in_eight_flips : ℚ := 7/64

/-- The number of coin flips -/
def num_flips : ℕ := 8

/-- The number of heads we're looking for -/
def num_heads : ℕ := 6

theorem prob_six_heads_correct :
  prob_six_heads_in_eight_flips = (Nat.choose num_flips num_heads : ℚ) / 2^num_flips :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_six_heads_correct_l1286_128667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kendra_minivans_l1286_128676

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := sorry

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := sorry

/-- The total number of minivans Kendra saw -/
def total_minivans : ℕ := sorry

theorem kendra_minivans :
  evening_minivans = 1 →
  total_minivans = 5 →
  afternoon_minivans + evening_minivans = total_minivans →
  afternoon_minivans = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kendra_minivans_l1286_128676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_colored_unit_triangle_l1286_128647

/-- Represents a point in the triangle -/
structure Point where
  n : ℕ
  m : ℕ

/-- Represents a color -/
inductive Color
  | Blue
  | White
  | Red

/-- Represents the set S of points inside the triangle -/
def S (N : ℕ) : Set Point :=
  {p : Point | p.n ≤ N ∧ p.m ≤ N ∧ p.n + p.m ≤ N}

/-- Coloring function for points in S -/
def coloring (N : ℕ) : S N → Color := sorry

/-- Predicate for a valid coloring according to the rules -/
def valid_coloring (N : ℕ) (c : S N → Color) : Prop :=
  (∀ p : S N, p.val.m = 0 → c p ≠ Color.Blue) ∧
  (∀ p : S N, p.val.n = 0 → c p ≠ Color.White) ∧
  (∀ p : S N, p.val.n + p.val.m = N → c p ≠ Color.Red)

/-- Predicate for an equilateral triangle with side length 1 -/
def unit_triangle (N : ℕ) (p q r : S N) : Prop := sorry

/-- Main theorem -/
theorem exists_colored_unit_triangle (N : ℕ) :
  ∃ (p q r : S N),
    unit_triangle N p q r ∧
    coloring N p = Color.Blue ∧
    coloring N q = Color.White ∧
    coloring N r = Color.Red :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_colored_unit_triangle_l1286_128647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l1286_128652

noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

def largeSide : ℝ := 10

def smallSide : ℝ := 5

noncomputable def largeTriangleArea : ℝ := equilateralTriangleArea largeSide

noncomputable def smallTriangleArea : ℝ := equilateralTriangleArea smallSide

noncomputable def trapezoidArea : ℝ := largeTriangleArea - smallTriangleArea

noncomputable def areaRatio : ℝ := smallTriangleArea / trapezoidArea

theorem triangle_trapezoid_area_ratio :
  areaRatio = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l1286_128652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_power_inverse_l1286_128645

noncomputable def complex_number : ℂ := Complex.mk (Real.sqrt 2 / 2) (-Real.sqrt 2 / 2)

theorem complex_fourth_power_inverse (z : ℂ) (h : z = complex_number) : 
  1 / z^4 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_power_inverse_l1286_128645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l1286_128694

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 294) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ surface_area = 6 * edge_length^2 ∧ edge_length = 7 := by
  use 7
  constructor
  · exact lt_trans zero_lt_one (by norm_num)
  constructor
  · rw [h]
    ring
  · rfl

#check cube_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l1286_128694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_times_b_is_integer_l1286_128638

/-- Defines the sequence a_n as the sum of k^n / k! from k = 0 to infinity -/
noncomputable def a (n : ℕ) : ℝ := ∑' k : ℕ, (k : ℝ) ^ n / k.factorial

/-- Defines the sequence b_n as the sum of (-1)^k * k^n / k! from k = 0 to infinity -/
noncomputable def b (n : ℕ) : ℝ := ∑' k : ℕ, (-1 : ℝ) ^ k * (k : ℝ) ^ n / k.factorial

/-- Theorem stating that a_n * b_n is an integer for all n ≥ 1 -/
theorem a_times_b_is_integer (n : ℕ) (h : n ≥ 1) : ∃ m : ℤ, a n * b n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_times_b_is_integer_l1286_128638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l1286_128641

noncomputable def Triangle := Real × Real × Real

noncomputable def circumradius (t : Triangle) : Real := sorry
noncomputable def inradius (t : Triangle) : Real := sorry
noncomputable def longestSide (t : Triangle) : Real := sorry
noncomputable def shortestAltitude (t : Triangle) : Real := sorry

theorem triangle_radius_inequality (ABC : Triangle) 
  (R : Real) (r : Real) (a : Real) (h : Real) 
  (h_R : R = circumradius ABC)
  (h_r : r = inradius ABC)
  (h_a : a = longestSide ABC)
  (h_h : h = shortestAltitude ABC) :
  R / r > a / h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l1286_128641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_two_l1286_128681

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (2*x - 5) / (x - 6)

-- State the theorem
theorem inverse_g_undefined_at_two : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 2, f (g x) = x ∧ g (f x) = x) ∧ 
  ¬∃ y, f 2 = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_two_l1286_128681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l1286_128661

-- Define the type for dice outcomes
def DieOutcome := Fin 6

-- Define the sample space
def SampleSpace := DieOutcome × DieOutcome

-- Define event A: red die is a multiple of 3
def EventA (outcome : SampleSpace) : Prop :=
  outcome.1.val + 1 = 3 ∨ outcome.1.val + 1 = 6  -- Accounting for 0-based indexing

-- Define event B: sum of dice is greater than 8
def EventB (outcome : SampleSpace) : Prop :=
  outcome.1.val + outcome.2.val + 2 > 8  -- +2 because of 0-based indexing

-- Define the probability measure
noncomputable def P : Set SampleSpace → ℝ := sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {o : SampleSpace | EventB o ∧ EventA o} / P {o : SampleSpace | EventA o} = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l1286_128661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_relation_l1286_128660

theorem median_mean_relation (n : ℚ) : 
  let s : Finset ℚ := {n, n + 5, n + 7, n + 10, n + 20}
  n + 7 = 10 →
  (Finset.sum s id) / (Finset.card s : ℚ) = 57 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_relation_l1286_128660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_line_l1286_128690

/-- The circle in the problem -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

/-- The line passing through (2,1) -/
def my_line (x y : ℝ) : Prop := 3*x - y - 5 = 0

/-- The point through which the line passes -/
def my_point : ℝ × ℝ := (2, 1)

theorem longest_chord_line :
  ∀ (x y : ℝ), my_line x y →
  (∀ (a b : ℝ), my_circle a b → (a - x)^2 + (b - y)^2 ≤ (2 - 1)^2 + (1 - (-2))^2) ∧
  my_line my_point.1 my_point.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_line_l1286_128690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1286_128675

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the symmetry condition
def symmetric_to_exp (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = Real.exp (-x)

-- Theorem statement
theorem function_symmetry :
  symmetric_to_exp f → ∀ x : ℝ, f x = Real.exp (-x) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1286_128675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_inequality_l1286_128619

noncomputable def f (x : ℝ) := (Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)

theorem f_range_and_inequality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < f x ∧ f x ≤ 8 ∧ Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_inequality_l1286_128619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_forming_triangle_with_area_5_l1286_128607

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.a * l2.b = l2.a * l1.b

/-- The area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangle_area (l : Line) : ℝ := abs (l.c^2 / (2 * l.a * l.b))

/-- The main theorem -/
theorem line_forming_triangle_with_area_5 (C : ℝ) :
  let l := Line.mk 2 5 C
  let l_given := Line.mk 2 5 (-1)
  parallel l l_given ∧ triangle_area l = 5 ↔ C = 10 ∨ C = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_forming_triangle_with_area_5_l1286_128607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1286_128626

/-- The function f(x) with the given properties -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.log (Real.sin (ω * x + Real.pi / 6))

/-- The theorem stating the decreasing interval of f(x) -/
theorem decreasing_interval_of_f (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∃ (a b : ℝ), a = Real.pi / 6 ∧ b = 5 * Real.pi / 12 ∧
  ∀ x y, x ∈ Set.Icc 0 Real.pi → y ∈ Set.Icc 0 Real.pi → 
  x ∈ Set.Ico a b → y ∈ Set.Ico a b → x < y → f ω x > f ω y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1286_128626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_specific_dilation_l1286_128655

/-- A dilation of the plane that maps a circle to another circle -/
structure MyDilation where
  originalCenter : ℝ × ℝ
  originalRadius : ℝ
  dilatedCenter : ℝ × ℝ
  dilatedRadius : ℝ

/-- The distance the origin moves under a dilation -/
noncomputable def originMovement (d : MyDilation) : ℝ := sorry

/-- The theorem stating the distance the origin moves under the specific dilation -/
theorem origin_movement_specific_dilation :
  ∃ (d : MyDilation), 
    d.originalCenter = (3, 3) ∧
    d.originalRadius = 3 ∧
    d.dilatedCenter = (9, 12) ∧
    d.dilatedRadius = 6 ∧
    originMovement d = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_specific_dilation_l1286_128655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_sum_l1286_128604

noncomputable section

open Real

theorem triangle_tan_sum (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  b - c = (1/3) * a →
  Real.sin B = 2 * Real.sin A →
  Real.sin A > 0 →
  Real.sin B > 0 →
  Real.sin C > 0 →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  Real.tan (B + C) = (11/7) * Real.tan A := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_sum_l1286_128604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l1286_128639

theorem quarter_circle_area : 
  ∀ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) →
    radius = 1 →
    (∫ (x : ℝ) in Set.Icc 0 1, Real.sqrt (1 - (x - 1)^2)) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l1286_128639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_inventory_and_pricing_l1286_128613

-- Define variables
variable (x : ℚ)
variable (m : ℚ)

-- Define the conditions
def type_a_count (x : ℚ) : ℚ := 2 * x
def type_b_count (x : ℚ) : ℚ := x
def type_a_total_cost : ℚ := 9200
def type_b_total_cost : ℚ := 6400
def type_b_unit_cost (x : ℚ) : ℚ := type_b_total_cost / x
def type_a_unit_cost (x : ℚ) : ℚ := type_b_unit_cost x - 30

-- Define the profit requirement
def min_profit : ℚ := 10920

-- Theorem statement
theorem shirt_inventory_and_pricing :
  (type_a_count 60 = 120 ∧ type_b_count 60 = 60) ∧
  (∀ m' : ℚ, m' ≥ 70 → 
    (type_a_unit_cost 60 * (1 + m' / 100) * type_a_count 60 + 
     type_b_unit_cost 60 * (1 + m' / 100) * type_b_count 60) ≥ 
    (type_a_total_cost + type_b_total_cost + min_profit)) ∧
  (∀ m' : ℚ, m' < 70 → 
    (type_a_unit_cost 60 * (1 + m' / 100) * type_a_count 60 + 
     type_b_unit_cost 60 * (1 + m' / 100) * type_b_count 60) < 
    (type_a_total_cost + type_b_total_cost + min_profit)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_inventory_and_pricing_l1286_128613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l1286_128689

theorem silverware_probability : 
  (Nat.choose 5 2 * Nat.choose 7 1 * Nat.choose 6 1 : ℚ) / Nat.choose 18 4 = 7 / 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l1286_128689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_distance_calculation_l1286_128614

/-- Calculates the distance traveled in a uniformly accelerated motion -/
noncomputable def takeoffRunDistance (time : ℝ) (finalVelocity : ℝ) : ℝ :=
  let acceleration := finalVelocity / time
  (1/2) * acceleration * time^2

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmhToMs (v : ℝ) : ℝ :=
  v * 1000 / 3600

theorem takeoff_run_distance_calculation :
  let time := (15 : ℝ)
  let finalVelocity := kmhToMs 100
  ⌊takeoffRunDistance time finalVelocity⌋ = 208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_distance_calculation_l1286_128614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1286_128633

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 2)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (xa ya xb yb : ℝ),
    curve_C xa ya ∧
    line_l xb yb ∧
    (∀ (x'a y'a x'b y'b : ℝ),
      curve_C x'a y'a →
      line_l x'b y'b →
      distance xa ya xb yb + distance xb yb (point_P.1) (point_P.2) ≤
      distance x'a y'a x'b y'b + distance x'b y'b (point_P.1) (point_P.2)) ∧
    distance xa ya xb yb + distance xb yb (point_P.1) (point_P.2) = Real.sqrt 37 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1286_128633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_function_theorem_l1286_128602

-- Define the solution set O
noncomputable def O : Set ℝ := {x : ℝ | x > 2}

-- Define x₀
def x₀ : ℝ := 2

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

-- Theorem statement
theorem inequality_and_function_theorem :
  (O = {x : ℝ | |x + 3| - 2*x - 1 < 0}) ∧
  (∀ m : ℝ, m > 0 → (∃ x : ℝ, f m x = 0) ↔ m = 1) :=
by
  sorry

#check inequality_and_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_function_theorem_l1286_128602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equality_l1286_128677

theorem root_product_equality : (27 : Real)^(1/3) * (81 : Real)^(1/4) * (64 : Real)^(1/6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equality_l1286_128677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_sum_mod_3_l1286_128665

def initial_board : List Nat := List.range' 1 25

def board_sum (board : List Nat) : Nat :=
  board.sum

def replace_numbers (board : List Nat) (a b c : Nat) : List Nat :=
  let new_number := a^3 + b^3 + c^3
  (board.filter (λ x => x ≠ a ∧ x ≠ b ∧ x ≠ c)) ++ [new_number]

theorem board_sum_mod_3 (board : List Nat) :
  (∀ (a b c : Nat), a ∈ board → b ∈ board → c ∈ board → a ≠ b → b ≠ c → a ≠ c →
    board_sum (replace_numbers board a b c) % 3 = board_sum board % 3) →
  board_sum board % 3 = 1 →
  ¬(∃ (final_board : List Nat), board_sum final_board = 2013^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_sum_mod_3_l1286_128665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_derangement_9_l1286_128611

open Nat

/-- The number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of derangements of n elements -/
def derangements' (n : ℕ) : ℕ := permutations n - 1

theorem probability_of_derangement_9 :
  (derangements' 9 : ℚ) / (permutations 9 : ℚ) = 362879 / 362880 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_derangement_9_l1286_128611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l1286_128668

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l1286_128668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_in_acres_l1286_128605

/-- Represents the dimensions of a trapezoid in centimeters -/
structure TrapezoidDimensions where
  bottom : ℝ
  top : ℝ
  height : ℝ

/-- Converts centimeters to miles -/
noncomputable def cmToMiles : ℝ → ℝ := (· / 2)

/-- Converts square miles to acres -/
noncomputable def sqMilesToAcres : ℝ → ℝ := (· * 640)

/-- Calculates the area of a trapezoid in square centimeters -/
noncomputable def trapezoidArea (d : TrapezoidDimensions) : ℝ :=
  (d.bottom + d.top) * d.height / 2

theorem plot_area_in_acres 
  (dims : TrapezoidDimensions)
  (h_bottom : dims.bottom = 15)
  (h_top : dims.top = 10)
  (h_height : dims.height = 10) :
  sqMilesToAcres ((cmToMiles (trapezoidArea dims))^2) = 320000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_in_acres_l1286_128605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1286_128648

noncomputable section

def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := x / (Real.exp x)

theorem f_derivative : 
  deriv f = λ x => (1 - x) / (Real.exp x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1286_128648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheesecake_factory_work_days_l1286_128642

noncomputable def hourly_rate : ℝ := 10
noncomputable def hours_per_day : ℝ := 10
noncomputable def weeks : ℝ := 4
noncomputable def total_savings : ℝ := 3000

noncomputable def robby_savings_rate : ℝ := 2/5
noncomputable def jaylen_savings_rate : ℝ := 3/5
noncomputable def miranda_savings_rate : ℝ := 1/2

def days_worked (d : ℝ) : Prop :=
  robby_savings_rate * hourly_rate * hours_per_day * d * weeks +
  jaylen_savings_rate * hourly_rate * hours_per_day * d * weeks +
  miranda_savings_rate * hourly_rate * hours_per_day * d * weeks = total_savings

theorem cheesecake_factory_work_days : 
  ∃ (d : ℝ), days_worked d ∧ d = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheesecake_factory_work_days_l1286_128642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_minimizes_squared_differences_l1286_128636

/-- Represents a data point with actual and predicted values -/
structure DataPoint where
  actual : ℝ
  predicted : ℝ

/-- The objective function of the least squares method -/
def leastSquaresObjective (data : List DataPoint) : ℝ :=
  (data.map fun point => (point.actual - point.predicted) ^ 2).sum

/-- The method of least squares minimizes the sum of squared differences -/
theorem least_squares_minimizes_squared_differences 
  (data : List DataPoint) (other_method : List DataPoint → ℝ) :
  leastSquaresObjective data ≤ other_method data := by
  sorry

#check least_squares_minimizes_squared_differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_minimizes_squared_differences_l1286_128636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1286_128685

noncomputable def f (x : ℝ) : ℝ := Real.log (x + (1 + x^3) ^ (1/3))

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1286_128685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_symmetric_l1286_128659

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.sin x ^ 2

theorem f_increasing_and_symmetric (a : ℝ) (x₁ : ℝ) (y₁ : ℝ) :
  (∀ x ∈ Set.Icc a (π / 16), ∀ y ∈ Set.Icc a (π / 16), x < y → f x < f y) →
  a ∈ Set.Icc (-π / 8) (π / 16) ∧
  (∀ x : ℝ, f (2 * x₁ - x) = f x) ∧
  x₁ ∈ Set.Icc (-π / 4) (π / 4) →
  a ∈ Set.Ioc (-π / 8) (π / 16) ∧ x₁ = π / 8 ∧ y₁ = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_symmetric_l1286_128659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l1286_128695

/-- Definition of an ellipse with parameters a and b -/
noncomputable def Ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (a^2) + y^2 / (b^2) = 1}

/-- Definition of eccentricity for an ellipse -/
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (min a b / max a b)^2)

/-- Theorem: For an ellipse with equation x^2/4 + y^2/m = 1, 
    where m > 0, m ≠ 4, and eccentricity = 1/2, 
    the only possible values for m are 3 and 16/3 -/
theorem ellipse_eccentricity_m_values (m : ℝ) :
  m > 0 ∧ m ≠ 4 ∧ 
  (∃ (a b : ℝ), Ellipse a b = Ellipse 2 (Real.sqrt m) ∧ Eccentricity a b = 1/2) →
  m = 3 ∨ m = 16/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l1286_128695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l1286_128646

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3*x - 2) / (x + 4)

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (4*x + 2) / (3 - x)

-- Theorem statement
theorem inverse_function_and_ratio :
  (∀ x, g (g_inv x) = x) ∧ 
  (∀ x, g_inv (g x) = x) ∧
  (4 : ℝ) / (-1 : ℝ) = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l1286_128646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1286_128649

def line (x b : ℝ) : ℝ := x + b

noncomputable def curve (x : ℝ) : ℝ := 3 - Real.sqrt (4 * x - x^2)

theorem range_of_b (b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   line x₁ b = curve x₁ ∧ 
   line x₂ b = curve x₂) →
  (1 - Real.sqrt 29) / 2 < b ∧ b < (1 + Real.sqrt 29) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1286_128649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1286_128688

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

-- Theorem statement
theorem f_properties :
  -- Domain of f
  (∀ x : ℝ, f x ≠ 0 ↔ x ≠ 1 ∧ x ≠ -1) ∧
  -- f is even
  (∀ x : ℝ, f (-x) = f x) ∧
  -- f is decreasing on (1, +∞)
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1286_128688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l1286_128635

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f_deriv (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

theorem max_value_on_interval (a b : ℝ) : 
  (f_deriv a 1 = -1) →  -- Condition for tangent line slope
  (f a b 1 = 2) →       -- Condition for point on tangent line
  (∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 5 ∧ f a b x = 58/3) ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 5 → f a b y ≤ 58/3) := by
  sorry

#check max_value_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l1286_128635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_2_l1286_128608

-- Define the piecewise linear function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 then 2*x + 4
  else if x ≤ 0 then -0.5*x
  else if x ≤ 2 then 2*x - 1
  else 0.5*x + 2

-- Theorem statement
theorem sum_of_x_coordinates_where_g_equals_2 :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ g x₁ = 2 ∧ g x₂ = 2 ∧ x₁ + x₂ = 0 ∧
  ∀ x, g x = 2 → x = x₁ ∨ x = x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_2_l1286_128608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_probability_l1286_128686

/-- Represents the number of representatives -/
def num_representatives : ℕ := 9

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of representatives per country -/
def reps_per_country : ℕ := 3

/-- Represents the number of chairs at the round table -/
def num_chairs : ℕ := 9

/-- Calculates the total number of possible seating arrangements -/
def total_arrangements : ℕ := Nat.factorial (num_representatives - 1) / (Nat.factorial reps_per_country ^ num_countries)

/-- Calculates the number of favorable seating arrangements -/
def favorable_arrangements : ℕ := total_arrangements - 450

/-- The probability of each representative having at least one representative 
    from another country sitting next to them -/
noncomputable def probability : ℚ := (favorable_arrangements : ℚ) / total_arrangements

theorem seating_probability : probability = 41 / 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_probability_l1286_128686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_with_equal_digit_sum_l1286_128600

def digit_sum (n : ℕ) : ℕ := sorry

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_numbers_with_equal_digit_sum :
  ∃ (A : Finset ℕ), (∀ a ∈ A, is_three_digit a ∧ digit_sum a = digit_sum (2 * a)) ∧ Finset.card A = 80 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_with_equal_digit_sum_l1286_128600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1286_128631

/-- The surface area of a sphere circumscribing a right circular cone with three mutually perpendicular side edges of length √3 is equal to 9π. -/
theorem circumscribed_sphere_surface_area 
  (cone : Real → Real → Real → Real) 
  (side_edge_length : cone 0 0 0 = Real.sqrt 3)
  (perpendicular_edges : ∀ i j, i ≠ j → (cone i 0 0) * (cone j 0 0) = 0) :
  4 * Real.pi * (3/2)^2 = 9 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1286_128631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_nonnegative_iff_x_in_interval_l1286_128651

/-- The function representing the given expression -/
noncomputable def f (x : ℝ) : ℝ := (x - 10*x^2 + 25*x^3) / (8 - x^3)

/-- Theorem stating that the expression is nonnegative if and only if x is in [0,2) -/
theorem expression_nonnegative_iff_x_in_interval :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_nonnegative_iff_x_in_interval_l1286_128651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_trisection_l1286_128603

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = c^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = a^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = b^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the length of a vector
noncomputable def VectorLength (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Define the dot product of two vectors
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the trisection point
def TrisectionPoint (A B D : ℝ × ℝ) : Prop :=
  D.1 = (2 * B.1 + A.1) / 3 ∧ D.2 = (2 * B.2 + A.2) / 3

-- Theorem statement
theorem dot_product_trisection (A B C D : ℝ × ℝ) :
  Triangle A B C →
  RightAngle A C B →
  VectorLength (C.1 - B.1, C.2 - B.2) = 3 →
  TrisectionPoint A B D →
  DotProduct (C.1 - B.1, C.2 - B.2) (C.1 - D.1, C.2 - D.2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_trisection_l1286_128603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_distance_l1286_128697

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line
def line (k x y : ℝ) : Prop := 4*k*x - 4*y - k = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem parabola_chord_distance (k x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  line k x1 y1 ∧ line k x2 y2 ∧
  distance x1 y1 x2 y2 = 4 →
  distance ((x1 + x2) / 2) ((y1 + y2) / 2) (-1/2) 0 = 9/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_distance_l1286_128697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_a_solution_b_l1286_128644

-- Case (a)
def is_solution_a (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → f (x * y * z) = f x + f y + f z

theorem solution_a (f : ℝ → ℝ) (hf : is_solution_a f) :
  ∃ c : ℝ, ∀ x : ℝ, x > 0 → f x = c * Real.log x :=
sorry

-- Case (b)
def is_solution_b (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ 1 < a^3 ∧ a^3 < b ∧
  ∀ (x y z : ℝ), a < x → x < b → a < y → y < b → a < z → z < b →
    f (x * y * z) = f x + f y + f z

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (Real.exp x)

theorem solution_b (f : ℝ → ℝ) (a b : ℝ) (hf : is_solution_b f a b) :
  ∃ c : ℝ, ∀ x : ℝ, Real.log a < x → x < Real.log b → F f x = c * x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_a_solution_b_l1286_128644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_history_textbook_pages_l1286_128628

def history_pages : ℕ → Prop := λ h => True
def geography_pages : ℕ → ℕ → Prop := λ h g => g = h + 70
def math_pages : ℕ → ℕ → Prop := λ h m => m = (2 * h + 70) / 2
def science_pages : ℕ → ℕ → Prop := λ h s => s = 2 * h
def total_pages : ℕ → ℕ → ℕ → ℕ → ℕ → Prop := λ h g m s t => h + g + m + s = t

axiom geography_relation (h : ℕ) : geography_pages h (h + 70)
axiom math_relation (h : ℕ) : math_pages h ((2 * h + 70) / 2)
axiom science_relation (h : ℕ) : science_pages h (2 * h)
axiom total_sum (h g m s : ℕ) : total_pages h g m s 905

theorem history_textbook_pages :
  ∃ h : ℕ, history_pages h ∧
    geography_pages h (h + 70) ∧
    math_pages h ((2 * h + 70) / 2) ∧
    science_pages h (2 * h) ∧
    h + (h + 70) + ((2 * h + 70) / 2) + (2 * h) = 905 ∧
    h = 160 :=
by
  use 160
  apply And.intro
  · trivial
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · norm_num
  · rfl

#check history_textbook_pages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_history_textbook_pages_l1286_128628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1286_128692

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * log x + (a - 1/2) * x^2 + 2*(1-a)*x + a

/-- Theorem stating that f(x) has exactly 3 zeros when a < -2 -/
theorem f_has_three_zeros (a : ℝ) (h : a < -2) :
  ∃ (x₁ x₂ x₃ : ℝ), (0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃) ∧ 
    (f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ∧
    (∀ x > 0, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by
  sorry

#check f_has_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1286_128692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_equation_l1286_128658

/-- Represents the average monthly growth rate for May and June -/
def x : ℝ := sorry

/-- Revenue in March (in 10,000 yuan) -/
def march_revenue : ℝ := 15

/-- Percentage decrease in April -/
def april_decrease : ℝ := 0.1

/-- Revenue in June (in 10,000 yuan) -/
def june_revenue : ℝ := 20

/-- The equation representing the relationship between March and June revenue -/
theorem revenue_equation : march_revenue * (1 - april_decrease) * (1 + x)^2 = june_revenue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_equation_l1286_128658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l1286_128637

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x =>
  if x > 0 then -x * (1 + x) else x * (x - 1)

-- Theorem statement
theorem f_is_odd_and_correct :
  odd_function f ∧ (∀ x < 0, f x = x * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l1286_128637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_condition_l1286_128629

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (1 - 3*x)) / (2*x)

-- State the theorem
theorem f_well_defined_condition (x : ℝ) :
  (∃ y, f x = y) ↔ (x ≤ 1/3 ∧ x ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_condition_l1286_128629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_i_power_2015_l1286_128622

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property of i
theorem i_squared : i^2 = -1 := Complex.I_sq

-- State the theorem
theorem i_power_2015 : i^2015 = -i := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_i_power_2015_l1286_128622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_win_probability_l1286_128669

theorem carl_win_probability (carl_prob dana_prob : ℚ) : 
  carl_prob = 2/7 →
  dana_prob = 3/8 →
  (carl_prob * (1 - dana_prob) / (1 - (1 - carl_prob) * (1 - dana_prob))) = 10/31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_win_probability_l1286_128669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_when_perimeter_equals_area_l1286_128654

/-- An isosceles right triangle inscribed in a circle -/
structure IsoscelesRightTriangleInCircle where
  /-- The radius of the circumscribed circle -/
  r : ℝ
  /-- The radius is positive -/
  r_pos : r > 0

/-- The perimeter of the isosceles right triangle -/
noncomputable def perimeter (t : IsoscelesRightTriangleInCircle) : ℝ :=
  t.r * (1 + Real.sqrt 2)

/-- The area of the circumscribed circle -/
noncomputable def circleArea (t : IsoscelesRightTriangleInCircle) : ℝ :=
  Real.pi * t.r^2

/-- 
Theorem: If the perimeter of an isosceles right triangle (in inches) equals 
the area of its circumscribed circle (in square inches), then the radius of 
the circle (in inches) is (1 + √2) / π.
-/
theorem radius_when_perimeter_equals_area 
  (t : IsoscelesRightTriangleInCircle) 
  (h : perimeter t = circleArea t) : 
  t.r = (1 + Real.sqrt 2) / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_when_perimeter_equals_area_l1286_128654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_theorem_l1286_128634

/-- Parabola represented by its equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Given a parabola, calculate its vertex -/
noncomputable def vertex (p : Parabola) : Point :=
  { x := -p.b / (2 * p.a), y := p.c - p.b^2 / (4 * p.a) }

/-- Given a parabola, calculate its focus -/
noncomputable def focus (p : Parabola) : Point :=
  let v := vertex p
  { x := v.x, y := v.y + 1 / (4 * p.a) }

/-- Theorem stating the ratio of distances between foci and vertices of two related parabolas -/
theorem parabola_ratio_theorem (P : Parabola) (R : Parabola)
    (h1 : P.a = 4 ∧ P.b = 0 ∧ P.c = 0)  -- P is y = 4x²
    (h2 : R.a = 8 ∧ R.b = 0 ∧ R.c = 1/8)  -- R is y = 8x² + 1/8
    (h3 : ∀ (C D : Point), C.y = 4 * C.x^2 → D.y = 4 * D.x^2 → 
          (C.y / C.x) * (D.y / D.x) = -1 →  -- perpendicular condition
          R.a = 2 * P.a ∧ R.c = 1 / (8 * P.a)) :  -- relation between P and R
  distance (focus P) (focus R) / distance (vertex P) (vertex R) = -27/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_theorem_l1286_128634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1286_128682

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x - 2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the perpendicular line through (√3, 0)
def perp_line (x y : ℝ) : Prop := y = -(1 / Real.sqrt 3) * (x - Real.sqrt 3)

-- Theorem statement
theorem intersection_distance_difference :
  ∀ D E : ℝ × ℝ,
  line_l (2 * Real.sqrt 3) 4 →
  line_l (Real.sqrt 3) 1 →
  curve_C D.1 D.2 →
  curve_C E.1 E.2 →
  perp_line D.1 D.2 →
  perp_line E.1 E.2 →
  D.2 > 0 →
  E.2 < 0 →
  (1 / Real.sqrt ((D.1 - Real.sqrt 3)^2 + D.2^2)) -
  (1 / Real.sqrt ((E.1 - Real.sqrt 3)^2 + E.2^2)) = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1286_128682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_little_ming_walking_speed_l1286_128617

noncomputable section

/-- The distance from home to bus station in km -/
def distance : ℝ := 1.2

/-- The time saved by cycling in hours -/
def time_saved : ℝ := 12 / 60

/-- The ratio of cycling speed to walking speed -/
def speed_ratio : ℝ := 2.5

/-- Little Ming's average walking speed in km/h -/
def walking_speed : ℝ := 3.6

theorem little_ming_walking_speed :
  (distance / walking_speed) - (distance / (speed_ratio * walking_speed)) = time_saved := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_little_ming_walking_speed_l1286_128617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_ellipse_equations_l1286_128699

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, imaginary axis length 2, and focal distance 2√3,
    prove its asymptotes and associated ellipse equations. -/
theorem hyperbola_and_ellipse_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  b = 1 →
  (∃ c : ℝ, c^2 - b^2 = a^2 ∧ c = Real.sqrt 3) →
  (∀ x y : ℝ, y = Real.sqrt 2 / 2 * x ∨ y = -(Real.sqrt 2 / 2 * x)) ∧
  (∀ x y : ℝ, x^2 / 3 + y^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_ellipse_equations_l1286_128699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_domain_l1286_128616

-- Define the function f(x) = e^x - x - 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

-- Theorem statement
theorem f_positive_on_domain :
  ∀ x : ℝ, x ≠ 0 → f x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_domain_l1286_128616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l1286_128696

/-- First parabola equation -/
def parabola1 (X Y : ℝ) : Prop := Y = X^2 + X - 41

/-- Second parabola equation -/
def parabola2 (X Y : ℝ) : Prop := X = Y^2 + Y - 40

/-- Circle equation -/
def circleEq (X Y : ℝ) : Prop := X^2 + Y^2 = 81

theorem intersection_points_on_circle :
  ∀ X Y : ℝ, parabola1 X Y ∧ parabola2 X Y → circleEq X Y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l1286_128696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_to_equilateral_triangle_area_l1286_128601

theorem polygon_to_equilateral_triangle_area (r s : ℝ) (h1 : r > 0) (h2 : s > 0) :
  let polygon_area := r * s
  let equilateral_triangle_side := Real.sqrt ((2 * s / 3) * (2 * r * Real.sqrt 3))
  let equilateral_triangle_area := equilateral_triangle_side^2 * Real.sqrt 3 / 4
  polygon_area = equilateral_triangle_area := by
  sorry

#check polygon_to_equilateral_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_to_equilateral_triangle_area_l1286_128601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_14_probability_l1286_128621

/-- Represents the process of selecting balls and moving them through bags -/
structure BallSelection where
  numBags : ℕ 
  numBalls : ℕ 
  finalBallDistribution : Fin numBags → Fin numBalls

/-- The specific ball selection process described in the problem -/
def franklinBallSelection : BallSelection where
  numBags := 4
  numBalls := 15
  finalBallDistribution := sorry

/-- The probability of a specific ball being in one of the final bags -/
def probabilityInFinalBags (b : BallSelection) (ballNum : Fin b.numBalls) : ℚ :=
  sorry

theorem ball_14_probability :
  probabilityInFinalBags franklinBallSelection ⟨14, sorry⟩ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_14_probability_l1286_128621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_median_equal_mean_l1286_128618

noncomputable def number_set (x : ℝ) : Finset ℝ := {1, 3, 5, 14, x}

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem no_median_equal_mean :
  ∀ x : ℝ, ¬(is_median (number_set x) ((1 + 3 + 5 + 14 + x) / 5)) :=
by sorry

#check no_median_equal_mean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_median_equal_mean_l1286_128618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_preserves_divisibility_by_seven_initial_number_divisible_by_seven_target_number_not_divisible_by_seven_cannot_obtain_target_number_l1286_128615

/-- The operation described in the problem -/
def boardOperation (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let remainingNumber := n / 10
  remainingNumber + 5 * lastDigit

/-- The property that the operation preserves divisibility by 7 -/
theorem operation_preserves_divisibility_by_seven (n : ℕ) :
  n % 7 = 0 → (boardOperation n) % 7 = 0 :=
by sorry

/-- The initial number 7^1998 is divisible by 7 -/
theorem initial_number_divisible_by_seven :
  (7^1998) % 7 = 0 :=
by sorry

/-- 1998^7 is not divisible by 7 -/
theorem target_number_not_divisible_by_seven :
  (1998^7) % 7 ≠ 0 :=
by sorry

/-- Main theorem: It's impossible to obtain 1998^7 from 7^1998 using the board operation -/
theorem cannot_obtain_target_number :
  ∀ k : ℕ, (Nat.iterate boardOperation k (7^1998)) ≠ 1998^7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_preserves_divisibility_by_seven_initial_number_divisible_by_seven_target_number_not_divisible_by_seven_cannot_obtain_target_number_l1286_128615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cut_length_30x30_225_l1286_128623

/-- The maximum length of cuts for a square board --/
noncomputable def max_cut_length (side_length : ℕ) (num_parts : ℕ) : ℝ :=
  let total_area : ℝ := (side_length ^ 2 : ℝ)
  let part_area : ℝ := total_area / num_parts
  let max_part_perimeter : ℝ := if part_area = 4 then 10 else 8
  let total_perimeter : ℝ := (num_parts : ℝ) * max_part_perimeter
  let board_perimeter : ℝ := (4 * side_length : ℝ)
  (total_perimeter - board_perimeter) / 2

/-- Theorem stating the maximum cut length for a 30x30 board cut into 225 parts --/
theorem max_cut_length_30x30_225 :
  max_cut_length 30 225 = 1065 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cut_length_30x30_225_l1286_128623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_max_min_values_part2_inequality_l1286_128662

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

-- Part 1
theorem part1_max_min_values :
  let f1 := f (-1) 3
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f1 x ≤ 2) ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f1 x ≥ 2 - Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f1 x = 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f1 x = 2 - Real.log 2) :=
by sorry

-- Part 2
theorem part2_inequality (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, f a b x ≥ f a b 1) :
  Real.log a < -2 * b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_max_min_values_part2_inequality_l1286_128662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1286_128609

/-- The y-intercept of a line is the point where it crosses the y-axis (x = 0) -/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ :=
  (0, -c / b)

/-- A line is defined by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem y_intercept_of_line (l : Line) (h : l.b ≠ 0) :
  y_intercept l.a l.b (-l.c) = (0, -7) ↔ l.a = 7 ∧ l.b = -3 ∧ l.c = -21 := by
  sorry

#check y_intercept_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1286_128609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equiv_l1286_128620

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_equiv (x : ℝ) :
  4 * (floor x)^2 - 12 * (floor x) + 5 ≤ 0 ↔ 1 ≤ x ∧ x < 3 :=
by
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equiv_l1286_128620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_of_subsets_l1286_128691

def A : Finset ℚ := {1/2, 1/7, 1/11, 1/13, 1/15, 1/32}

theorem sum_of_products_of_subsets :
  (Finset.powerset A).sum (λ s => if s.card > 0 then s.prod id else 0) - 1 = 14/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_of_subsets_l1286_128691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_centroid_is_hyperbola_and_locus_l1286_128656

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  O : Point
  A : Point
  B : Point

-- Define the centroid
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.O.x + t.A.x + t.B.x) / 3,
    y := (t.O.y + t.A.y + t.B.y) / 3 }

-- Define a conic (abstractly)
structure Conic where
  passesThrough : Point → Prop
  center : Point

-- Define a hyperbola (abstractly)
structure Hyperbola extends Conic

-- Define the Steiner inellipse (abstractly)
noncomputable def steinerInellipse (t : Triangle) : Conic :=
  { passesThrough := λ _ => true,  -- Placeholder
    center := { x := 0, y := 0 } }  -- Placeholder

-- The main theorem
theorem conic_through_centroid_is_hyperbola_and_locus (t : Triangle) :
  let G := centroid t
  ∀ c : Conic,
    (c.passesThrough t.O ∧ c.passesThrough t.A ∧ c.passesThrough t.B ∧ c.passesThrough G) →
    (∃ h : Hyperbola, c = h.toConic) ∧
    (∃ steiner : Conic, steiner = steinerInellipse t ∧ c.center = steiner.center) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_centroid_is_hyperbola_and_locus_l1286_128656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1286_128672

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - (3*x^2 + 2*a*x + b)

theorem problem_solution (a b : ℝ) 
  (h1 : ∀ x, g a b x = -g a b (-x)) :
  (∀ x, f a b x = x^3 + 3*x^2) ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, g 3 0 x ≥ g 3 0 y) ∧
  g 3 0 3 = 9 ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, g 3 0 x ≤ g 3 0 y) ∧
  g 3 0 (Real.sqrt 2) = -4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1286_128672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_76_to_nearest_tenth_l1286_128632

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  (⌊x * 10 + 0.5⌋ : ℝ) / 10

/-- The standard rounding rule: round up if the fractional part is ≥ 0.5 -/
axiom roundingRule (x : ℝ) : 
  (x - ⌊x⌋ ≥ 0.5) → ⌈x⌉ = ⌊x⌋ + 1

theorem round_4_76_to_nearest_tenth :
  roundToNearestTenth 4.76 = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_76_to_nearest_tenth_l1286_128632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_numbers_bound_l1286_128663

/-- A positive integer is nice if it can be expressed as a² + b³ for some non-negative integers a and b. -/
def IsNice (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^3

/-- The set of nice numbers not exceeding 1,000,000. -/
def NiceSet : Set ℕ :=
  {n : ℕ | n ≤ 1000000 ∧ IsNice n}

/-- The number of nice integers not exceeding 1,000,000 is between 10,000 and 100,000. -/
theorem nice_numbers_bound : 
  ∃ s : Finset ℕ, (∀ n ∈ s, n ∈ NiceSet) ∧ 
    10000 ≤ s.card ∧ s.card ≤ 100000 :=
by
  sorry

#check nice_numbers_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_numbers_bound_l1286_128663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1286_128666

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 2 => sequence_a (n + 1) + 1 / sequence_a (n + 1)

theorem sequence_a_properties :
  (∀ n ≥ 2, sequence_a n ≥ Real.sqrt (2 * n)) ∧
  (∀ C : ℝ, ∃ n : ℕ, sequence_a n ≥ Real.sqrt (2 * n + C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1286_128666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l1286_128679

/-- The number of blocks Youseff lives from his office -/
def blocks : ℕ := 21

/-- The time it takes Youseff to walk to work in minutes -/
def walk_time : ℝ := blocks

/-- The time it takes Youseff to ride his bike to work in minutes -/
noncomputable def bike_time : ℝ := (1/3) * blocks

/-- The statement that it takes Youseff 14 minutes more to walk than to ride his bike -/
axiom time_difference : walk_time = bike_time + 14

/-- Theorem: Youseff lives 21 blocks from his office -/
theorem youseff_distance : blocks = 21 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l1286_128679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_coin_A_given_result_l1286_128643

/-- Probability of getting heads for each coin -/
def prob_heads (coin : Fin 3) : ℚ :=
  match coin with
  | 0 => 1/3  -- Coin A
  | 1 => 1/2  -- Coin B
  | 2 => 2/3  -- Coin C

/-- Probability of selecting each coin -/
def prob_select_coin : ℚ := 1/3

/-- Number of flips -/
def num_flips : ℕ := 4

/-- Number of heads observed -/
def num_heads : ℕ := 3

theorem prob_coin_A_given_result :
  let prob_result_given_coin (coin : Fin 3) := 
    (num_flips.choose num_heads : ℚ) * 
    (prob_heads coin) ^ num_heads * 
    (1 - prob_heads coin) ^ (num_flips - num_heads)
  let prob_result := (Finset.univ.sum fun coin => prob_result_given_coin coin * prob_select_coin)
  let prob_coin_A_given_result := 
    (prob_result_given_coin 0 * prob_select_coin) / prob_result
  prob_coin_A_given_result = 32/241 := by
  sorry

#eval (32 : ℕ) + (241 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_coin_A_given_result_l1286_128643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_functions_l1286_128630

def is_valid_function (f : ℕ → ℤ) : Prop :=
  (∀ x, x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : Set ℕ)) ∧
  (f 1 = 1) ∧
  (∀ x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} : Set ℕ), |f (x + 1) - f x| = 1) ∧
  (∃ r : ℚ, r ≠ 1 ∧ f 6 = f 1 * r ∧ f 12 = f 6 * r)

-- We need to make this noncomputable because we're working with functions
-- from ℕ to ℤ, which don't automatically have a fintype instance
noncomputable def valid_functions : Set (ℕ → ℤ) :=
  { f | is_valid_function f }

-- We use noncomputable instance to create a fintype instance for our set
noncomputable instance : Fintype valid_functions :=
  sorry

theorem count_valid_functions :
  Fintype.card valid_functions = 155 := by
  sorry

#check count_valid_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_functions_l1286_128630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l1286_128625

theorem four_integer_product (m n p q : ℕ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l1286_128625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_nonnegative_f_plus_m_m_for_minimum_f_equals_six_l1286_128657

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (x^2 + 3) / (x - m)

-- Part 1: Range of m such that f(x) + m ≥ 0
theorem range_of_m_for_nonnegative_f_plus_m :
  ∀ m : ℝ, (∀ x : ℝ, x > m → f x m + m ≥ 0) ↔ m ≥ -2 * Real.sqrt 15 / 5 :=
by sorry

-- Part 2: Value of m for minimum f(x) = 6
theorem m_for_minimum_f_equals_six :
  ∃! m : ℝ, (∀ x : ℝ, x > m → f x m ≥ 6) ∧ (∃ x : ℝ, x > m ∧ f x m = 6) ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_nonnegative_f_plus_m_m_for_minimum_f_equals_six_l1286_128657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l1286_128684

/-- Represents a polynomial function -/
def MyPolynomial (α : Type*) [Ring α] := List α

/-- Counts the number of operations in Horner's method -/
def hornerOperations (p : MyPolynomial ℝ) : ℕ × ℕ := sorry

/-- The specific polynomial f(x) = 3x^3 + 2x^2 + 4x + 6 -/
def f : MyPolynomial ℝ := [3, 2, 4, 6]

theorem horner_operations_for_f :
  hornerOperations f = (3, 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l1286_128684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_pharmacies_not_enough_l1286_128671

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 10
  y : Fin 10

/-- Represents a pharmacy on the grid -/
def Pharmacy := GridPoint

/-- The distance between two points on the grid -/
def gridDistance (p1 p2 : GridPoint) : ℕ :=
  (Int.natAbs (p1.x.val - p2.x.val) + Int.natAbs (p1.y.val - p2.y.val))

/-- A point is within walking distance of a pharmacy if the grid distance is <= 3 -/
def isWithinWalkingDistance (point : GridPoint) (pharmacy : Pharmacy) : Prop :=
  gridDistance point pharmacy ≤ 3

/-- A street segment is a pair of adjacent points on the grid -/
structure StreetSegment where
  start : GridPoint
  stop : GridPoint
  adjacent : gridDistance start stop = 1

/-- A pharmacy covers a street segment if either endpoint is within walking distance -/
def pharmacyCoversSegment (pharmacy : Pharmacy) (segment : StreetSegment) : Prop :=
  isWithinWalkingDistance segment.start pharmacy ∨ isWithinWalkingDistance segment.stop pharmacy

/-- The main theorem: 12 pharmacies are not enough to cover all street segments -/
theorem twelve_pharmacies_not_enough :
  ∀ (pharmacies : Finset Pharmacy),
    pharmacies.card = 12 →
    ∃ (segment : StreetSegment),
      ∀ (pharmacy : Pharmacy),
        pharmacy ∈ pharmacies →
        ¬ pharmacyCoversSegment pharmacy segment :=
by
  sorry

#check twelve_pharmacies_not_enough

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_pharmacies_not_enough_l1286_128671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l1286_128674

/-- The x-coordinate of the intersection point of two lines -/
noncomputable def intersection_x (m1 b1 a2 b2 c2 : ℝ) : ℝ :=
  (c2 - b1) / (m1 + a2)

theorem intersection_of_lines :
  let line1 : ℝ → ℝ := λ x => 5 * x - 28
  let line2 : ℝ → ℝ → ℝ := λ x y => 3 * x + y
  intersection_x 5 (-28) 3 1 120 = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l1286_128674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_similarity_l1286_128606

-- Define the structures and functions
structure RegularQuadrilateralPyramid where
  base : Set (EuclideanSpace ℝ (Fin 3))
  apex : EuclideanSpace ℝ (Fin 3)
  is_regular : sorry -- Placeholder for regularity condition
  is_pyramid : sorry -- Placeholder for pyramid condition

structure Triangle where
  vertices : Finset (EuclideanSpace ℝ (Fin 3))
  is_triangle : sorry -- Placeholder for triangle condition

def project_to_base (pyramid : RegularQuadrilateralPyramid) (triangle : Triangle) : Triangle :=
  sorry

def project_to_lateral_face (pyramid : RegularQuadrilateralPyramid) (triangle : Triangle) : Triangle :=
  sorry

def IsSimilar (t1 t2 : Triangle) : Prop :=
  sorry -- Placeholder for similarity condition

-- The main theorem
theorem projection_similarity (pyramid : RegularQuadrilateralPyramid) (original : Triangle) :
  let base_projection := project_to_base pyramid original
  let lateral_projection := project_to_lateral_face pyramid base_projection
  IsSimilar original lateral_projection := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_similarity_l1286_128606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_defg_product_difference_l1286_128680

-- Define a function to convert a list of digits to a number
def toNumber (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

-- Define the theorem
theorem abc_defg_product_difference :
  ∀ (a b c d e f g : Nat),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g →
    1 ≤ a ∧ a ≤ 9 →
    1 ≤ b ∧ b ≤ 9 →
    1 ≤ c ∧ c ≤ 9 →
    1 ≤ d ∧ d ≤ 9 →
    1 ≤ e ∧ e ≤ 9 →
    1 ≤ f ∧ f ≤ 9 →
    1 ≤ g ∧ g ≤ 9 →
    toNumber [a, b, c] + toNumber [d, e, f, g] = 2020 →
    (Nat.max (toNumber [9, 8, 9] * toNumber [1, 0, 3, 1]) 
             (toNumber [9, 1, 1] * toNumber [1, 1, 0, 9]) -
     Nat.min (toNumber [9, 8, 9] * toNumber [1, 0, 3, 1]) 
             (toNumber [9, 1, 1] * toNumber [1, 1, 0, 9])) = 91900 :=
by
  intros a b c d e f g h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- The proof is omitted for brevity
  sorry

-- Example usage to show the theorem is well-formed
#check abc_defg_product_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_defg_product_difference_l1286_128680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_even_l1286_128693

-- Define a real number a > 0
variable (a : ℝ) (ha : a > 0)

-- Define the function f on the real line
variable (f : ℝ → ℝ)

-- Define the function F
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + f (-x)

-- Theorem statement
theorem F_is_even (f : ℝ → ℝ) : ∀ x ∈ Set.Ioo (-a) a, F f x = F f (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_even_l1286_128693
