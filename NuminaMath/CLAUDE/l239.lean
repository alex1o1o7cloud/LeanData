import Mathlib

namespace NUMINAMATH_CALUDE_a_value_is_negative_one_l239_23904

/-- The coefficient of x^2 in the expansion of (1+ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ :=
  (Nat.choose 5 2 : ℝ) + a * (Nat.choose 5 1 : ℝ)

/-- The theorem stating that a = -1 given the coefficient of x^2 is 5 -/
theorem a_value_is_negative_one :
  ∃ a : ℝ, coefficient_x_squared a = 5 ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_a_value_is_negative_one_l239_23904


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l239_23991

/-- A quadratic equation with roots -1 and 3 -/
theorem quadratic_equation_roots (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ↔ (x = -1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l239_23991


namespace NUMINAMATH_CALUDE_new_device_significantly_improved_l239_23941

-- Define the sample means and variances
def x_bar : ℝ := 10
def y_bar : ℝ := 10.3
def s1_squared : ℝ := 0.036
def s2_squared : ℝ := 0.04

-- Define the significant improvement criterion
def significant_improvement (x_bar y_bar s1_squared s2_squared : ℝ) : Prop :=
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / 10)

-- Theorem statement
theorem new_device_significantly_improved :
  significant_improvement x_bar y_bar s1_squared s2_squared := by
  sorry


end NUMINAMATH_CALUDE_new_device_significantly_improved_l239_23941


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l239_23948

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 16) and (-2, -10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 16
  let x₂ : ℝ := -2
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l239_23948


namespace NUMINAMATH_CALUDE_roden_gold_fish_l239_23969

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 22 - 7

/-- Theorem stating that Roden bought 15 gold fish -/
theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_roden_gold_fish_l239_23969


namespace NUMINAMATH_CALUDE_f_even_iff_a_zero_f_min_value_when_x_geq_a_l239_23989

/-- Definition of the function f(x) -/
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem about the evenness of f(x) -/
theorem f_even_iff_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

/-- Theorem about the minimum value of f(x) when x ≥ a -/
theorem f_min_value_when_x_geq_a (a : ℝ) :
  (∀ x ≥ a, f a x ≥ (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) ∧
  (∃ x ≥ a, f a x = (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) := by sorry

end NUMINAMATH_CALUDE_f_even_iff_a_zero_f_min_value_when_x_geq_a_l239_23989


namespace NUMINAMATH_CALUDE_complex_magnitude_l239_23960

theorem complex_magnitude (a : ℝ) :
  (∃ (b : ℝ), (a + I) / (2 - I) = b * I) →
  Complex.abs (1/2 + (a + I) / (2 - I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l239_23960


namespace NUMINAMATH_CALUDE_reverse_digit_increase_l239_23994

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a + b + c = 10 ∧
    b = a + c ∧
    n = 253

theorem reverse_digit_increase (n : ℕ) (h : is_valid_number n) :
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    100 * c + 10 * b + a - n = 99 := by
  sorry

end NUMINAMATH_CALUDE_reverse_digit_increase_l239_23994


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_property_l239_23999

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is inscribed in a circle -/
def Quadrilateral.inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  q.A.onCircle c ∧ q.B.onCircle c ∧ q.C.onCircle c ∧ q.D.onCircle c

/-- Represents the tangent line at a point on a circle -/
def tangentLine (p : Point) (c : Circle) : Line :=
  sorry

/-- Checks if two lines intersect at a point -/
def Line.intersectAt (l1 l2 : Line) (p : Point) : Prop :=
  p.onLine l1 ∧ p.onLine l2

/-- Calculates the distance between two points -/
def Point.dist (p1 p2 : Point) : ℝ :=
  sorry

theorem inscribed_quadrilateral_property (c : Circle) (q : Quadrilateral) (K : Point) :
  q.inscribed c →
  Line.intersectAt (tangentLine q.B c) (tangentLine q.D c) K →
  K.onLine (Line.mk 0 1 0) →  -- Assuming y-axis is the line AC
  q.A.dist q.B * q.C.dist q.D = q.B.dist q.C * q.A.dist q.D ∧
  ∀ (P Q R : Point) (l : Line),
    l.intersectAt (Line.mk 1 0 0) P →  -- Assuming x-axis is the line BA
    l.intersectAt (Line.mk 1 1 0) Q →  -- Assuming y=x is the line BD
    l.intersectAt (Line.mk 0 1 0) R →  -- Assuming y-axis is the line BC
    P.dist Q = Q.dist R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_property_l239_23999


namespace NUMINAMATH_CALUDE_even_sum_implies_one_even_l239_23914

theorem even_sum_implies_one_even (a b c : ℕ) :
  Even (a + b + c) →
  ¬((Odd a ∧ Odd b ∧ Odd c) ∨ 
    (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_implies_one_even_l239_23914


namespace NUMINAMATH_CALUDE_cubic_root_solutions_l239_23981

/-- A rational triple (a, b, c) is a solution if a, b, and c are the roots of the polynomial x^3 + ax^2 + bx + c = 0 -/
def IsSolution (a b c : ℚ) : Prop :=
  ∀ x : ℚ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = a ∨ x = b ∨ x = c)

/-- The only rational triples (a, b, c) that are solutions are (0, 0, 0), (1, -1, -1), and (1, -2, 0) -/
theorem cubic_root_solutions :
  ∀ a b c : ℚ, IsSolution a b c ↔ ((a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, -1, -1) ∨ (a, b, c) = (1, -2, 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_solutions_l239_23981


namespace NUMINAMATH_CALUDE_right_triangle_sine_roots_l239_23927

theorem right_triangle_sine_roots (n : ℤ) (A B C : ℝ) : 
  A + B = π / 2 →
  (∃ (x y : ℝ), x = Real.sin A ∧ y = Real.sin B ∧ 
    (5 * n + 8 : ℝ) * x^2 - (7 * n - 20 : ℝ) * x + 120 = 0 ∧
    (5 * n + 8 : ℝ) * y^2 - (7 * n - 20 : ℝ) * y + 120 = 0) →
  n = 66 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sine_roots_l239_23927


namespace NUMINAMATH_CALUDE_bug_position_after_3000_jumps_l239_23966

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def isOdd : Point → Bool
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Performs a single jump based on the current point -/
def jump : Point → Point
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs multiple jumps -/
def multiJump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (multiJump start n)

theorem bug_position_after_3000_jumps :
  multiJump Point.seven 3000 = Point.two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_3000_jumps_l239_23966


namespace NUMINAMATH_CALUDE_m_range_l239_23903

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∃ x : ℝ, 3^x - m + 1 ≤ 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, 1 < m ∧ m ≤ 2 ↔ q m ∧ ¬(p m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l239_23903


namespace NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l239_23908

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y, f x = y ↔ f (-x) = y :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l239_23908


namespace NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l239_23951

-- Define the propositions p and q as functions of m
def proposition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m_for_p_or_q :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ↔ m < -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l239_23951


namespace NUMINAMATH_CALUDE_exponent_calculation_l239_23978

theorem exponent_calculation : (1 / ((-5^2)^4)) * ((-5)^9) = -5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l239_23978


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l239_23957

/-- Calculates the interval for systematic sampling. -/
def systematicSamplingInterval (totalPopulation sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Proves that for a population of 72 and sample size of 8, the systematic sampling interval is 9. -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l239_23957


namespace NUMINAMATH_CALUDE_distance_ratio_l239_23929

-- Define the total distance and traveled distance
def total_distance : ℝ := 234
def traveled_distance : ℝ := 156

-- Define the theorem
theorem distance_ratio :
  let remaining_distance := total_distance - traveled_distance
  (traveled_distance / remaining_distance) = 2 := by
sorry

end NUMINAMATH_CALUDE_distance_ratio_l239_23929


namespace NUMINAMATH_CALUDE_average_speed_calculation_l239_23959

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def time_day1 : ℕ := 6
def time_day2 : ℕ := 4

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := time_day1 + time_day2
  (total_distance : ℚ) / total_time = 22 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l239_23959


namespace NUMINAMATH_CALUDE_largest_divisible_number_l239_23987

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem largest_divisible_number : 
  (∀ m : ℕ, 5 ≤ m ∧ m ≤ 10 → is_divisible 2520 m) ∧ 
  ¬(∀ m : ℕ, 5 ≤ m ∧ m ≤ 11 → is_divisible 2520 m) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_number_l239_23987


namespace NUMINAMATH_CALUDE_petra_age_l239_23912

theorem petra_age (petra_age mother_age : ℕ) : 
  petra_age + mother_age = 47 →
  mother_age = 2 * petra_age + 14 →
  mother_age = 36 →
  petra_age = 11 :=
by sorry

end NUMINAMATH_CALUDE_petra_age_l239_23912


namespace NUMINAMATH_CALUDE_original_number_l239_23938

theorem original_number (x : ℝ) : x * 1.4 = 1680 → x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l239_23938


namespace NUMINAMATH_CALUDE_sum_of_even_integers_302_to_400_l239_23930

theorem sum_of_even_integers_302_to_400 (sum_first_50 : ℕ) (sum_302_to_400 : ℕ) : 
  sum_first_50 = 2550 → sum_302_to_400 = 17550 → sum_302_to_400 - sum_first_50 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_302_to_400_l239_23930


namespace NUMINAMATH_CALUDE_cricketer_average_last_four_matches_l239_23900

/-- Calculates the average score for the last 4 matches of a cricketer given the average score for all 10 matches and the average score for the first 6 matches. -/
def average_last_four_matches (total_average : ℚ) (first_six_average : ℚ) : ℚ :=
  let total_runs := total_average * 10
  let first_six_runs := first_six_average * 6
  let last_four_runs := total_runs - first_six_runs
  last_four_runs / 4

/-- Theorem stating that given a cricketer with an average score of 38.9 runs for 10 matches
    and an average of 42 runs for the first 6 matches, the average for the last 4 matches is 34.25 runs. -/
theorem cricketer_average_last_four_matches :
  average_last_four_matches (389 / 10) 42 = 34.25 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_last_four_matches_l239_23900


namespace NUMINAMATH_CALUDE_sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l239_23954

theorem sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12 :
  ∀ x : ℝ, Real.sin (3 * x + π / 4) = Real.cos (3 * (x - π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l239_23954


namespace NUMINAMATH_CALUDE_exist_three_numbers_equal_sum_l239_23940

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of three different natural numbers with equal sum of number and its digits -/
theorem exist_three_numbers_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ m + S m = n + S n ∧ n + S n = p + S p :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_equal_sum_l239_23940


namespace NUMINAMATH_CALUDE_total_weight_of_sand_l239_23998

/-- The total weight of sand in two jugs with different capacities and sand densities -/
theorem total_weight_of_sand (jug1_capacity jug2_capacity : ℝ)
  (fill_percentage : ℝ)
  (density1 density2 : ℝ) :
  jug1_capacity = 2 →
  jug2_capacity = 3 →
  fill_percentage = 0.7 →
  density1 = 4 →
  density2 = 5 →
  jug1_capacity * fill_percentage * density1 +
  jug2_capacity * fill_percentage * density2 = 16.1 := by
  sorry

#check total_weight_of_sand

end NUMINAMATH_CALUDE_total_weight_of_sand_l239_23998


namespace NUMINAMATH_CALUDE_coin_probability_l239_23934

theorem coin_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- p is a probability
  (p * (1 - p)^4 = 1/32) →  -- probability of HTTT = 0.03125
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l239_23934


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l239_23933

/-- For a triangle with circumcircle radius R, excircle radius p, and distance d between their centers, d^2 = R^2 + 2Rp. -/
theorem triangle_circle_relation (R p d : ℝ) : d^2 = R^2 + 2*R*p :=
sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l239_23933


namespace NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l239_23917

/-- The time taken to complete a job when three people work together, given their individual completion times. -/
theorem job_completion_time (t1 t2 t3 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : t3 > 0) :
  1 / (1 / t1 + 1 / t2 + 1 / t3) = (t1 * t2 * t3) / (t2 * t3 + t1 * t3 + t1 * t2) :=
by sorry

/-- The specific case of the job completion time for the given problem. -/
theorem specific_job_completion_time :
  1 / (1 / 15 + 1 / 20 + 1 / 25 : ℝ) = 300 / 47 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l239_23917


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l239_23949

theorem complex_magnitude_product : 3 * Complex.abs (1 - 3*I) * Complex.abs (1 + 3*I) = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l239_23949


namespace NUMINAMATH_CALUDE_G_fraction_difference_l239_23937

/-- G is defined as the infinite repeating decimal 0.427427427... -/
def G : ℚ := 427 / 999

/-- Theorem stating that the difference between the denominator and numerator of G is 572 -/
theorem G_fraction_difference : 
  let (n, d) := (Rat.num G, Rat.den G)
  d - n = 572 := by sorry

end NUMINAMATH_CALUDE_G_fraction_difference_l239_23937


namespace NUMINAMATH_CALUDE_temperature_equation_initial_temperature_temperature_increase_l239_23932

/-- Represents the temperature in °C at a given time t in minutes -/
def temperature (t : ℝ) : ℝ := 7 * t + 30

theorem temperature_equation (t : ℝ) (h : t < 10) :
  temperature t = 7 * t + 30 :=
by sorry

theorem initial_temperature :
  temperature 0 = 30 :=
by sorry

theorem temperature_increase (t₁ t₂ : ℝ) (h₁ : t₁ < 10) (h₂ : t₂ < 10) (h₃ : t₁ < t₂) :
  temperature t₂ - temperature t₁ = 7 * (t₂ - t₁) :=
by sorry

end NUMINAMATH_CALUDE_temperature_equation_initial_temperature_temperature_increase_l239_23932


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l239_23972

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [1, 2, 1, 0, 2] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l239_23972


namespace NUMINAMATH_CALUDE_sum_of_squares_130_l239_23936

theorem sum_of_squares_130 : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 0 ∧ 
  b > 0 ∧ 
  a^2 + b^2 = 130 ∧ 
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_130_l239_23936


namespace NUMINAMATH_CALUDE_total_bales_at_end_of_week_l239_23970

def initial_bales : ℕ := 28
def daily_additions : List ℕ := [10, 15, 8, 20, 12, 4, 18]

theorem total_bales_at_end_of_week : 
  initial_bales + daily_additions.sum = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_bales_at_end_of_week_l239_23970


namespace NUMINAMATH_CALUDE_solve_linear_equation_l239_23986

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 5*x = 80 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l239_23986


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l239_23995

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of the polynomial -/
structure PolynomialRoots (w : ℂ) where
  root1 : ℂ := w - Complex.I
  root2 : ℂ := w - 3 * Complex.I
  root3 : ℂ := 2 * w + 2

/-- Theorem statement -/
theorem cubic_polynomial_sum (P : CubicPolynomial) (w : ℂ) 
  (h : ∀ z : ℂ, (z - (w - Complex.I)) * (z - (w - 3 * Complex.I)) * (z - (2 * w + 2)) = 
       z^3 + P.a * z^2 + P.b * z + P.c) :
  P.a + P.b + P.c = 22 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l239_23995


namespace NUMINAMATH_CALUDE_homework_time_calculation_l239_23920

theorem homework_time_calculation (jacob_time greg_time patrick_time : ℕ) : 
  jacob_time = 18 →
  greg_time = jacob_time - 6 →
  patrick_time = 2 * greg_time - 4 →
  jacob_time + greg_time + patrick_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l239_23920


namespace NUMINAMATH_CALUDE_chess_tournament_solution_l239_23931

-- Define the tournament structure
structure ChessTournament where
  grade8_students : ℕ
  grade7_points : ℕ
  grade8_points : ℕ

-- Define the tournament conditions
def valid_tournament (t : ChessTournament) : Prop :=
  t.grade7_points = 8 ∧
  t.grade8_points * t.grade8_students = (t.grade8_students + 2) * (t.grade8_students + 1) / 2 - 8

-- Theorem statement
theorem chess_tournament_solution (t : ChessTournament) :
  valid_tournament t → (t.grade8_students = 7 ∨ t.grade8_students = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_solution_l239_23931


namespace NUMINAMATH_CALUDE_prime_has_property_p_infinitely_many_composite_with_property_p_l239_23916

/-- Definition of property p -/
def has_property_p (n : ℕ) : Prop :=
  ∀ a : ℕ, n ∣ (a^n - 1) → n^2 ∣ (a^n - 1)

/-- Every prime number has property p -/
theorem prime_has_property_p (p : ℕ) (hp : Nat.Prime p) : has_property_p p := by
  sorry

/-- There exist infinitely many composite numbers with property p -/
theorem infinitely_many_composite_with_property_p :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ ¬(Nat.Prime n) ∧ has_property_p n := by
  sorry

end NUMINAMATH_CALUDE_prime_has_property_p_infinitely_many_composite_with_property_p_l239_23916


namespace NUMINAMATH_CALUDE_binary_111100_is_even_l239_23926

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem binary_111100_is_even :
  let binary := [false, false, true, true, true, true]
  is_even (binary_to_decimal binary) := by
  sorry

end NUMINAMATH_CALUDE_binary_111100_is_even_l239_23926


namespace NUMINAMATH_CALUDE_circle_area_not_quadrupled_l239_23913

theorem circle_area_not_quadrupled (r : ℝ) (h : r > 0) : 
  ∃ k : ℝ, k ≠ 4 ∧ π * (r^2)^2 = k * (π * r^2) :=
sorry

end NUMINAMATH_CALUDE_circle_area_not_quadrupled_l239_23913


namespace NUMINAMATH_CALUDE_smallest_n_with_all_digit_sums_l239_23997

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to get all divisors of a number
def divisors (n : ℕ) : Set ℕ := sorry

-- Define a function to get the set of sums of digits of all divisors
def sumsOfDigitsOfDivisors (n : ℕ) : Set ℕ := sorry

-- Main theorem
theorem smallest_n_with_all_digit_sums :
  ∀ n : ℕ, n < 288 →
    ¬(∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors n) ∧
  (∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors 288) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_all_digit_sums_l239_23997


namespace NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l239_23947

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
sorry

end NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l239_23947


namespace NUMINAMATH_CALUDE_binary_to_octal_l239_23996

-- Define the binary number
def binary_number : ℕ := 0b110101

-- Define the octal number
def octal_number : ℕ := 66

-- Theorem stating that the binary number is equal to the octal number
theorem binary_to_octal : binary_number = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_l239_23996


namespace NUMINAMATH_CALUDE_prob_success_constant_l239_23906

/-- Represents the probability of finding the correct key on the kth attempt
    given n total keys. -/
def prob_success (n : ℕ) (k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ n then 1 / n else 0

/-- Theorem stating that the probability of success on any attempt
    is 1/n for any valid k. -/
theorem prob_success_constant (n : ℕ) (k : ℕ) (h1 : n > 0) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  prob_success n k = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_prob_success_constant_l239_23906


namespace NUMINAMATH_CALUDE_cloth_selling_price_l239_23939

/-- Represents the selling price calculation for cloth --/
def total_selling_price (metres : ℕ) (cost_price : ℕ) (loss : ℕ) : ℕ :=
  metres * (cost_price - loss)

/-- Theorem stating the total selling price for the given conditions --/
theorem cloth_selling_price :
  total_selling_price 300 65 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l239_23939


namespace NUMINAMATH_CALUDE_gcd_1343_816_l239_23955

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1343_816_l239_23955


namespace NUMINAMATH_CALUDE_second_meeting_time_l239_23984

-- Define the pool and swimmers
def Pool : Type := Unit
def Swimmer : Type := Unit

-- Define the time to meet in the center
def time_to_center : ℝ := 1.5

-- Define the function to calculate the time for the second meeting
def time_to_second_meeting (p : Pool) (s1 s2 : Swimmer) : ℝ :=
  2 * time_to_center + time_to_center

-- Theorem statement
theorem second_meeting_time (p : Pool) (s1 s2 : Swimmer) :
  time_to_second_meeting p s1 s2 = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_second_meeting_time_l239_23984


namespace NUMINAMATH_CALUDE_truck_weight_l239_23915

/-- Given a truck and trailer with specified weight relationship, prove the truck's weight -/
theorem truck_weight (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end NUMINAMATH_CALUDE_truck_weight_l239_23915


namespace NUMINAMATH_CALUDE_triangle_properties_l239_23923

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 * b * c * Real.sin A = 3 * Real.sin A) →
  (a + b + c = 4 * (Real.sqrt 2 + 1)) →
  (Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 4 ∧ 
   Real.cos A = 1/3 ∧
   Real.cos (2*A - π/3) = (4*Real.sqrt 6 - 7) / 18) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l239_23923


namespace NUMINAMATH_CALUDE_triangle_angle_from_cosine_relation_l239_23907

theorem triangle_angle_from_cosine_relation (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) →
  B = π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_from_cosine_relation_l239_23907


namespace NUMINAMATH_CALUDE_simplify_sqrt_144000_l239_23918

theorem simplify_sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_144000_l239_23918


namespace NUMINAMATH_CALUDE_last_digit_of_2011_powers_l239_23983

theorem last_digit_of_2011_powers : ∃ n : ℕ, (2^2011 + 3^2011) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2011_powers_l239_23983


namespace NUMINAMATH_CALUDE_mini_marshmallows_count_l239_23967

/-- Calculates the number of mini marshmallows used in a recipe --/
def mini_marshmallows_used (total_marshmallows : ℕ) (large_marshmallows : ℕ) : ℕ :=
  total_marshmallows - large_marshmallows

/-- Proves that the number of mini marshmallows used is correct --/
theorem mini_marshmallows_count 
  (total_marshmallows : ℕ) 
  (large_marshmallows : ℕ) 
  (h : large_marshmallows ≤ total_marshmallows) :
  mini_marshmallows_used total_marshmallows large_marshmallows = 
    total_marshmallows - large_marshmallows :=
by
  sorry

#eval mini_marshmallows_used 18 8  -- Should output 10

end NUMINAMATH_CALUDE_mini_marshmallows_count_l239_23967


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l239_23956

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l239_23956


namespace NUMINAMATH_CALUDE_delay_and_wait_l239_23988

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

def addMinutes (t : Time) (m : Nat) : Time := sorry

theorem delay_and_wait (start : Time) (delay : Nat) (wait : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 → 
  delay = 30 → 
  wait = 2500 → 
  (addMinutes (addMinutes start delay) wait).hours = 21 ∧ 
  (addMinutes (addMinutes start delay) wait).minutes = 10 := by
  sorry

end NUMINAMATH_CALUDE_delay_and_wait_l239_23988


namespace NUMINAMATH_CALUDE_age_difference_l239_23952

theorem age_difference (rona_age rachel_age collete_age : ℕ) : 
  rona_age = 8 →
  rachel_age = 2 * rona_age →
  collete_age = rona_age / 2 →
  rachel_age - collete_age = 12 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l239_23952


namespace NUMINAMATH_CALUDE_minimum_properties_l239_23965

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem minimum_properties {x₀ : ℝ} (h₀ : x₀ > 0) 
  (h₁ : ∀ x > 0, f x ≥ f x₀) : 
  f x₀ = x₀ + 1 ∧ f x₀ < 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_properties_l239_23965


namespace NUMINAMATH_CALUDE_veridux_managers_count_l239_23945

/-- Veridux Corporation employee structure -/
structure VeriduxCorp where
  total_employees : ℕ
  female_employees : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- Theorem: The total number of managers at Veridux Corporation is 40 -/
theorem veridux_managers_count (v : VeriduxCorp)
  (h1 : v.total_employees = 250)
  (h2 : v.female_employees = 90)
  (h3 : v.male_associates = 160)
  (h4 : v.female_managers = 40)
  (h5 : v.total_employees = v.female_employees + (v.male_associates + v.female_managers)) :
  v.female_managers + (v.total_employees - v.female_employees - v.male_associates) = 40 := by
  sorry

#check veridux_managers_count

end NUMINAMATH_CALUDE_veridux_managers_count_l239_23945


namespace NUMINAMATH_CALUDE_quadratic_inequalities_condition_l239_23922

theorem quadratic_inequalities_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0) ↔
    (a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_condition_l239_23922


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l239_23963

/-- Given that Lucy originally has $20 and Linda has $10, prove that if Lucy gives $5 to Linda,
    they will have the same amount of money. -/
theorem equal_money_after_transfer (lucy_initial : ℕ) (linda_initial : ℕ) (transfer_amount : ℕ) : 
  lucy_initial = 20 →
  linda_initial = 10 →
  transfer_amount = 5 →
  lucy_initial - transfer_amount = linda_initial + transfer_amount :=
by sorry

end NUMINAMATH_CALUDE_equal_money_after_transfer_l239_23963


namespace NUMINAMATH_CALUDE_jade_transactions_l239_23925

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 16 →
  jade = 82 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l239_23925


namespace NUMINAMATH_CALUDE_siblings_age_sum_l239_23942

theorem siblings_age_sum (ages : Fin 6 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ ages i) 
  (h_mean : (Finset.univ.sum ages) / 6 = 10)
  (h_median : (ages (Fin.mk 2 (by norm_num)) + ages (Fin.mk 3 (by norm_num))) / 2 = 12) :
  ages 0 + ages 5 = 12 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l239_23942


namespace NUMINAMATH_CALUDE_gas_pressure_change_l239_23976

/-- Represents the state of a gas with pressure and volume -/
structure GasState where
  pressure : ℝ
  volume : ℝ

/-- The constant of proportionality for the gas -/
def gasConstant (state : GasState) : ℝ := state.pressure * state.volume

theorem gas_pressure_change 
  (initial : GasState) 
  (final : GasState) 
  (h1 : initial.pressure = 8) 
  (h2 : initial.volume = 3.5)
  (h3 : final.volume = 10.5)
  (h4 : gasConstant initial = gasConstant final) : 
  final.pressure = 8/3 := by
  sorry

#check gas_pressure_change

end NUMINAMATH_CALUDE_gas_pressure_change_l239_23976


namespace NUMINAMATH_CALUDE_larger_number_proof_l239_23961

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 35 → L = 1631 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l239_23961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l239_23950

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive terms
  (∀ k, a (k + 1) = 2 * a k) →  -- Common ratio q = 2
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Condition on a_m and a_n
  (∃ p q : ℕ, 1/p + 4/q ≤ 1/m + 4/n) →  -- Existence of minimum
  1/m + 4/n ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l239_23950


namespace NUMINAMATH_CALUDE_subset_implies_zero_intersection_of_A_and_B_l239_23958

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 2 < x ∧ x < 6}

-- Theorem 1: If {x | ax = 1} is a subset of any set, then a = 0
theorem subset_implies_zero (a : ℝ) (h : ∀ S : Set ℝ, {x | a * x = 1} ⊆ S) : a = 0 := by
  sorry

-- Theorem 2: A ∩ B = {x | 2 < x ∧ x < 4}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_zero_intersection_of_A_and_B_l239_23958


namespace NUMINAMATH_CALUDE_chord_equation_l239_23985

/-- The equation of a line that is a chord of the ellipse x^2 + 4y^2 = 36 and is bisected at (4, 2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Points (x₁, y₁) and (x₂, y₂) lie on the ellipse
    x₁^2 + 4*y₁^2 = 36 ∧ x₂^2 + 4*y₂^2 = 36 ∧
    -- (4, 2) is the midpoint of the chord
    (x₁ + x₂)/2 = 4 ∧ (y₁ + y₂)/2 = 2 ∧
    -- (x, y) is on the line containing the chord
    ∃ (t : ℝ), x = x₁ + t*(x₂ - x₁) ∧ y = y₁ + t*(y₂ - y₁)) →
  x + 2*y - 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_l239_23985


namespace NUMINAMATH_CALUDE_emily_flight_remaining_time_l239_23924

/-- Given a flight duration and a series of activities, calculate the remaining time -/
def remaining_flight_time (flight_duration : ℕ) (tv_episodes : ℕ) (tv_episode_duration : ℕ) 
  (sleep_duration : ℕ) (movies : ℕ) (movie_duration : ℕ) : ℕ :=
  flight_duration - (tv_episodes * tv_episode_duration + sleep_duration + movies * movie_duration)

/-- Theorem: Given Emily's flight and activities, prove that 45 minutes remain -/
theorem emily_flight_remaining_time : 
  remaining_flight_time 600 3 25 270 2 105 = 45 := by
  sorry

end NUMINAMATH_CALUDE_emily_flight_remaining_time_l239_23924


namespace NUMINAMATH_CALUDE_decimal_69_is_234_base5_l239_23911

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base 5 to its decimal representation -/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem decimal_69_is_234_base5 :
  toBase5 69 = [4, 3, 2] ∧ fromBase5 [4, 3, 2] = 69 := by
  sorry

#eval toBase5 69  -- Should output [4, 3, 2]
#eval fromBase5 [4, 3, 2]  -- Should output 69

end NUMINAMATH_CALUDE_decimal_69_is_234_base5_l239_23911


namespace NUMINAMATH_CALUDE_carol_can_invite_198_friends_l239_23944

/-- The number of invitations in each package -/
def invitations_per_pack : ℕ := 18

/-- The number of packs Carol bought -/
def packs_bought : ℕ := 11

/-- The total number of friends Carol can invite -/
def friends_to_invite : ℕ := invitations_per_pack * packs_bought

/-- Theorem stating that Carol can invite 198 friends -/
theorem carol_can_invite_198_friends : friends_to_invite = 198 := by
  sorry

end NUMINAMATH_CALUDE_carol_can_invite_198_friends_l239_23944


namespace NUMINAMATH_CALUDE_equation_solutions_l239_23901

def f (x : ℝ) : ℝ := (x - 4)^6 + (x - 6)^6

theorem equation_solutions :
  {x : ℝ | f x = 64} = {4, 6} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l239_23901


namespace NUMINAMATH_CALUDE_repeating_decimal_sqrt_pairs_l239_23968

def is_valid_pair (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (b * b = 9 * a)

theorem repeating_decimal_sqrt_pairs :
  ∀ a b : Nat, is_valid_pair a b ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 9 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sqrt_pairs_l239_23968


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l239_23974

/-- Given a man's downstream and upstream speeds, and the rate of current,
    calculate the man's rate in still water. -/
theorem mans_rate_in_still_water
  (downstream_speed : ℝ)
  (upstream_speed : ℝ)
  (current_rate : ℝ)
  (h1 : downstream_speed = 45)
  (h2 : upstream_speed = 23)
  (h3 : current_rate = 11) :
  (downstream_speed + upstream_speed) / 2 = 34 := by
sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l239_23974


namespace NUMINAMATH_CALUDE_problem_statement_l239_23975

theorem problem_statement :
  (∀ (x : ℕ), x > 0 → (1/2 : ℝ)^x ≥ (1/3 : ℝ)^x) ∧
  ¬(∃ (x : ℕ), x > 0 ∧ (2 : ℝ)^x + (2 : ℝ)^(1-x) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l239_23975


namespace NUMINAMATH_CALUDE_dog_park_ratio_l239_23905

theorem dog_park_ratio (total : ℕ) (running : ℕ) (doing_nothing : ℕ) 
  (h1 : total = 88)
  (h2 : running = 12)
  (h3 : doing_nothing = 10)
  (h4 : total / 4 = total / 4) : -- This represents that 1/4 of dogs are barking
  (total - running - (total / 4) - doing_nothing) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l239_23905


namespace NUMINAMATH_CALUDE_taxi_fare_fraction_l239_23973

/-- Represents the taxi fare structure and proves the fraction of a mile for each part. -/
theorem taxi_fare_fraction (first_part_cost : ℚ) (additional_part_cost : ℚ) 
  (total_distance : ℚ) (total_cost : ℚ) : 
  first_part_cost = 21/10 →
  additional_part_cost = 2/5 →
  total_distance = 8 →
  total_cost = 177/10 →
  ∃ (part_fraction : ℚ), 
    part_fraction = 7/39 ∧
    first_part_cost + (total_distance - 1) * (additional_part_cost / part_fraction) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_taxi_fare_fraction_l239_23973


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l239_23902

open Set

def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 > 0}
def B : Set ℝ := {x : ℝ | x - 1 < 0}

theorem intersection_of_A_and_B :
  A ∩ B = Iio 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l239_23902


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l239_23935

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let side₁ := R * Real.sqrt 3
  let height := side₁ * (Real.sqrt 3 / 2)
  let side₂ := 2 * height / Real.sqrt 3
  let r := side₂ * Real.sqrt 3 / 6
  r = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l239_23935


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l239_23980

theorem sum_of_fractions_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l239_23980


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l239_23993

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l239_23993


namespace NUMINAMATH_CALUDE_orphanage_children_count_l239_23953

/-- Represents the number of cupcakes in a package -/
inductive PackageSize
| small : PackageSize
| large : PackageSize

/-- Returns the number of cupcakes in a package -/
def packageCupcakes (size : PackageSize) : ℕ :=
  match size with
  | PackageSize.small => 10
  | PackageSize.large => 15

/-- Calculates the total number of cupcakes from a given number of packages -/
def totalCupcakes (size : PackageSize) (numPackages : ℕ) : ℕ :=
  numPackages * packageCupcakes size

/-- Represents Jean's cupcake purchase and distribution plan -/
structure CupcakePlan where
  largePacks : ℕ
  smallPacks : ℕ
  childrenCount : ℕ

/-- Theorem: The number of children in the orphanage equals the total number of cupcakes -/
theorem orphanage_children_count (plan : CupcakePlan)
  (h1 : plan.largePacks = 4)
  (h2 : plan.smallPacks = 4)
  (h3 : plan.childrenCount = totalCupcakes PackageSize.large plan.largePacks + totalCupcakes PackageSize.small plan.smallPacks) :
  plan.childrenCount = 100 := by
  sorry

end NUMINAMATH_CALUDE_orphanage_children_count_l239_23953


namespace NUMINAMATH_CALUDE_shift_left_theorem_l239_23946

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original quadratic function y = x^2 -/
def original : QuadraticFunction := ⟨1, 0, 0⟩

/-- Shifts a quadratic function to the left by h units -/
def shift_left (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  ⟨f.a, f.b + 2 * f.a * h, f.c + f.b * h + f.a * h^2⟩

/-- The shifted quadratic function -/
def shifted : QuadraticFunction := shift_left original 1

theorem shift_left_theorem :
  shifted = ⟨1, 2, 1⟩ := by sorry

end NUMINAMATH_CALUDE_shift_left_theorem_l239_23946


namespace NUMINAMATH_CALUDE_rectangle_from_isosceles_120_l239_23919

/-- An isosceles triangle with a vertex angle of 120 degrees --/
structure IsoscelesTriangle120 where
  -- We represent the triangle by its side lengths
  base : ℝ
  leg : ℝ
  base_positive : 0 < base
  leg_positive : 0 < leg
  vertex_angle : Real.cos (120 * π / 180) = (base^2 - 2 * leg^2) / (2 * leg^2)

/-- A rectangle formed by isosceles triangles --/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangles : List IsoscelesTriangle120
  width_positive : 0 < width
  height_positive : 0 < height

/-- Theorem stating that it's possible to form a rectangle from isosceles triangles with 120° vertex angle --/
theorem rectangle_from_isosceles_120 : 
  ∃ (r : RectangleFromTriangles), r.triangles.length > 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_from_isosceles_120_l239_23919


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l239_23971

theorem unique_prime_with_remainder : ∃! m : ℕ,
  Prime m ∧ 30 < m ∧ m < 50 ∧ m % 12 = 7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l239_23971


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l239_23982

theorem sum_of_squares_zero (x y z : ℝ) 
  (h : x / (y + z) + y / (z + x) + z / (x + y) = 1) :
  x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l239_23982


namespace NUMINAMATH_CALUDE_conference_handshakes_l239_23962

/-- Represents a group of employees at a conference -/
structure EmployeeGroup where
  size : Nat
  has_closed_loop : Bool

/-- Calculates the number of handshakes in the employee group -/
def count_handshakes (group : EmployeeGroup) : Nat :=
  if group.has_closed_loop && group.size ≥ 3 then
    (group.size * (group.size - 3)) / 2
  else
    0

/-- Theorem: In a group of 10 employees with a closed managerial loop,
    where each person shakes hands with everyone except their direct manager
    and direct subordinate, the total number of handshakes is 35 -/
theorem conference_handshakes :
  let group : EmployeeGroup := { size := 10, has_closed_loop := true }
  count_handshakes group = 35 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l239_23962


namespace NUMINAMATH_CALUDE_cost_price_calculation_l239_23928

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 648 →
  profit_percentage = 0.08 →
  selling_price = cost_price * (1 + profit_percentage) →
  cost_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l239_23928


namespace NUMINAMATH_CALUDE_kids_joined_in_l239_23909

theorem kids_joined_in (initial_kids final_kids : ℕ) (h : initial_kids = 14 ∧ final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_kids_joined_in_l239_23909


namespace NUMINAMATH_CALUDE_fan_daily_usage_l239_23990

/-- Calculates the daily usage of an electric fan given its power, monthly energy consumption, and days in a month -/
theorem fan_daily_usage 
  (fan_power : ℝ) 
  (monthly_energy : ℝ) 
  (days_in_month : ℕ) 
  (h1 : fan_power = 75) 
  (h2 : monthly_energy = 18) 
  (h3 : days_in_month = 30) : 
  (monthly_energy * 1000) / (fan_power * days_in_month) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fan_daily_usage_l239_23990


namespace NUMINAMATH_CALUDE_basket_probability_l239_23921

def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem basket_probability : 
  let n : ℕ := 6
  let k : ℕ := 2
  let p : ℝ := 2/3
  binomial_probability n k p = 20/243 := by sorry

end NUMINAMATH_CALUDE_basket_probability_l239_23921


namespace NUMINAMATH_CALUDE_slab_rate_per_square_meter_l239_23964

theorem slab_rate_per_square_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 14437.5) : 
  total_cost / (length * width) = 700 :=
by sorry

end NUMINAMATH_CALUDE_slab_rate_per_square_meter_l239_23964


namespace NUMINAMATH_CALUDE_percentage_difference_l239_23977

theorem percentage_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.30 * y → x - y = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l239_23977


namespace NUMINAMATH_CALUDE_edward_lawn_mowing_earnings_l239_23943

/-- Edward's lawn mowing earnings problem -/
theorem edward_lawn_mowing_earnings 
  (rate : ℕ) -- Rate per lawn mowed
  (total_lawns : ℕ) -- Total number of lawns to mow
  (forgotten_lawns : ℕ) -- Number of lawns forgotten
  (h1 : rate = 4) -- Edward earns 4 dollars for each lawn
  (h2 : total_lawns = 17) -- Edward had 17 lawns to mow
  (h3 : forgotten_lawns = 9) -- Edward forgot to mow 9 lawns
  : (total_lawns - forgotten_lawns) * rate = 32 := by
  sorry

end NUMINAMATH_CALUDE_edward_lawn_mowing_earnings_l239_23943


namespace NUMINAMATH_CALUDE_triangle_inradius_inequality_l239_23992

-- Define a triangle with sides a, b, c and inradius r
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  -- Ensure that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  -- Ensure that r is positive
  positive_inradius : r > 0

-- State the theorem
theorem triangle_inradius_inequality (t : Triangle) :
  1 / t.a^2 + 1 / t.b^2 + 1 / t.c^2 ≤ 1 / (4 * t.r^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_inequality_l239_23992


namespace NUMINAMATH_CALUDE_lawn_width_proof_l239_23910

theorem lawn_width_proof (length : ℝ) (road_width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  length = 110 →
  road_width = 10 →
  total_cost = 4800 →
  cost_per_sqm = 3 →
  ∃ width : ℝ, width = 50 ∧ 
    (road_width * width + road_width * length) * cost_per_sqm = total_cost :=
by sorry

end NUMINAMATH_CALUDE_lawn_width_proof_l239_23910


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l239_23979

theorem quadratic_roots_nature (k : ℂ) (h : k.re = 0 ∧ k.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), 10 * z₁^2 - 5 * z₁ - k = 0 ∧
                 10 * z₂^2 - 5 * z₂ - k = 0 ∧
                 z₁.im = 0 ∧
                 z₂.re = 0 ∧ z₂.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l239_23979
