import Mathlib

namespace NUMINAMATH_CALUDE_last_digit_of_expression_l3798_379863

theorem last_digit_of_expression : ∃ n : ℕ, (287 * 287 + 269 * 269 - 2 * 287 * 269) % 10 = 8 ∧ 10 * n + 8 = 287 * 287 + 269 * 269 - 2 * 287 * 269 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l3798_379863


namespace NUMINAMATH_CALUDE_problem_solution_l3798_379885

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ a b, a * b ≤ 1) ∧
  (∀ a b, 1 / a + 1 / b ≥ 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, |x + m| - |x + 1| ≤ 1 / a + 1 / b) ↔ m ∈ Set.Icc (-1) 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3798_379885


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3798_379889

theorem unique_solution_for_equation (b : ℝ) (hb : b ≠ 0) :
  ∃! a : ℝ, a ≠ 0 ∧ (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3798_379889


namespace NUMINAMATH_CALUDE_smallest_among_given_rationals_l3798_379839

theorem smallest_among_given_rationals :
  let S : Set ℚ := {5, -7, 0, -5/3}
  ∀ x ∈ S, -7 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_rationals_l3798_379839


namespace NUMINAMATH_CALUDE_work_completion_time_l3798_379818

theorem work_completion_time (p_rate q_rate : ℚ) (work_left : ℚ) : 
  p_rate = 1/20 → q_rate = 1/10 → work_left = 7/10 → 
  (p_rate + q_rate) * 2 = 1 - work_left := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3798_379818


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l3798_379808

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + 2*b + c = 0) :
  ∃ (x : ℝ), -3 < x ∧ x < -1/3 ∧ x = c/a :=
sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l3798_379808


namespace NUMINAMATH_CALUDE_binomial_10_2_l3798_379838

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by sorry

end NUMINAMATH_CALUDE_binomial_10_2_l3798_379838


namespace NUMINAMATH_CALUDE_green_ball_probability_l3798_379869

structure Container where
  red : ℕ
  green : ℕ

def X : Container := { red := 5, green := 7 }
def Y : Container := { red := 7, green := 5 }
def Z : Container := { red := 7, green := 5 }

def total_containers : ℕ := 3

def prob_select_container : ℚ := 1 / total_containers

def prob_green (c : Container) : ℚ := c.green / (c.red + c.green)

def total_prob_green : ℚ := 
  prob_select_container * prob_green X + 
  prob_select_container * prob_green Y + 
  prob_select_container * prob_green Z

theorem green_ball_probability : total_prob_green = 17 / 36 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3798_379869


namespace NUMINAMATH_CALUDE_incorrect_to_correct_ratio_l3798_379858

theorem incorrect_to_correct_ratio (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 75 →
  incorrect = 2 * correct →
  total = correct + incorrect →
  (incorrect : ℚ) / (correct : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_to_correct_ratio_l3798_379858


namespace NUMINAMATH_CALUDE_tangent_line_at_point_2_minus_6_l3798_379807

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem tangent_line_at_point_2_minus_6 :
  let P : ℝ × ℝ := (2, -6)
  let tangent_slope : ℝ := f' P.1
  let tangent_equation (x : ℝ) : ℝ := tangent_slope * (x - P.1) + P.2
  (∀ x, tangent_equation x = -3 * x) ∧ f P.1 = P.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_point_2_minus_6_l3798_379807


namespace NUMINAMATH_CALUDE_polynomial_condition_implies_monomial_l3798_379873

/-- A polynomial with nonnegative coefficients and degree ≤ n -/
def NonNegPolynomial (n : ℕ) := {p : Polynomial ℝ // p.degree ≤ n ∧ ∀ i, 0 ≤ p.coeff i}

theorem polynomial_condition_implies_monomial {n : ℕ} (P : NonNegPolynomial n) :
  (∀ x : ℝ, x > 0 → P.val.eval x * P.val.eval (1/x) ≤ (P.val.eval 1)^2) →
  ∃ (k : ℕ) (a : ℝ), k ≤ n ∧ a ≥ 0 ∧ P.val = Polynomial.monomial k a :=
sorry

end NUMINAMATH_CALUDE_polynomial_condition_implies_monomial_l3798_379873


namespace NUMINAMATH_CALUDE_inequality_preservation_l3798_379849

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 5 < b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3798_379849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3798_379810

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_2011 :
  ∃ n : ℕ, arithmeticSequence 1 3 n = 2011 ∧ n = 671 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3798_379810


namespace NUMINAMATH_CALUDE_min_value_f_prime_2_l3798_379886

theorem min_value_f_prime_2 (a : ℝ) (h : a > 0) :
  let f := fun x : ℝ => x^3 + 2*a*x^2 + (1/a)*x
  let f_prime := fun x : ℝ => 3*x^2 + 4*a*x + 1/a
  ∀ x : ℝ, f_prime 2 ≥ 12 + 4*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_prime_2_l3798_379886


namespace NUMINAMATH_CALUDE_complex_sum_real_l3798_379853

theorem complex_sum_real (a : ℝ) : 
  (a / (1 + 2*I) + (1 + 2*I) / 5 : ℂ).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_real_l3798_379853


namespace NUMINAMATH_CALUDE_evaluate_expression_l3798_379897

theorem evaluate_expression (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3798_379897


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l3798_379843

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a d 20 = 17) 
  (h2 : arithmetic_sequence a d 21 = 20) : 
  arithmetic_sequence a d 3 = -34 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l3798_379843


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3798_379829

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def swap_digits (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
    (tens_digit n * ones_digit n = 2 * (tens_digit n + ones_digit n)) ∧
    (n + 9 = 2 * swap_digits n) ∧
    n = 63 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3798_379829


namespace NUMINAMATH_CALUDE_gcd_of_37500_and_61250_l3798_379880

theorem gcd_of_37500_and_61250 : Nat.gcd 37500 61250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_37500_and_61250_l3798_379880


namespace NUMINAMATH_CALUDE_swimmer_problem_l3798_379803

theorem swimmer_problem (swimmer_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  swimmer_speed = 5 →
  downstream_distance = 54 →
  upstream_distance = 6 →
  ∃ (time current_speed : ℝ),
    time > 0 ∧
    current_speed > 0 ∧
    current_speed < swimmer_speed ∧
    time = downstream_distance / (swimmer_speed + current_speed) ∧
    time = upstream_distance / (swimmer_speed - current_speed) ∧
    time = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_problem_l3798_379803


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3798_379850

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2

-- Define the velocity function as the derivative of the position function
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3798_379850


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l3798_379848

theorem trigonometric_equalities : 
  (Real.sqrt 2 / 2) * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180)) = 1/2 ∧
  Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l3798_379848


namespace NUMINAMATH_CALUDE_angle_trig_values_l3798_379828

/-- Given an angle α whose terminal side passes through the point (3,4),
    prove the values of sin α, cos α, and tan α. -/
theorem angle_trig_values (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) →
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_values_l3798_379828


namespace NUMINAMATH_CALUDE_coin_position_determinable_l3798_379802

-- Define the coin values
def left_coin : ℕ := 10
def right_coin : ℕ := 15

-- Define the possible multipliers
def left_multipliers : List ℕ := [4, 10, 12, 26]
def right_multipliers : List ℕ := [7, 13, 21, 35]

-- Define a function to check if a number is even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the possible configurations
structure Configuration :=
  (left_value : ℕ)
  (right_value : ℕ)
  (left_multiplier : ℕ)
  (right_multiplier : ℕ)

-- Define the theorem
theorem coin_position_determinable :
  ∀ (c : Configuration),
  c.left_value ∈ [left_coin, right_coin] ∧
  c.right_value ∈ [left_coin, right_coin] ∧
  c.left_value ≠ c.right_value ∧
  c.left_multiplier ∈ left_multipliers ∧
  c.right_multiplier ∈ right_multipliers →
  (is_even (c.left_value * c.left_multiplier + c.right_value * c.right_multiplier) ↔
   c.right_value = left_coin) :=
by sorry

end NUMINAMATH_CALUDE_coin_position_determinable_l3798_379802


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l3798_379824

/-- A rectangular plot with given conditions --/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  total_fencing_cost : ℝ
  length_width_relation : length = width + 10
  fencing_cost_relation : fencing_rate * (2 * (length + width)) = total_fencing_cost

/-- The perimeter of the rectangular plot is 180 meters --/
theorem rectangular_plot_perimeter (plot : RectangularPlot) 
  (h_rate : plot.fencing_rate = 6.5)
  (h_cost : plot.total_fencing_cost = 1170) : 
  2 * (plot.length + plot.width) = 180 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l3798_379824


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3798_379812

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  n % 17 = 0 ∧ 
  10 ≤ n ∧ 
  n < 100 ∧ 
  ∀ m : ℕ, (m % 17 = 0 ∧ 10 ≤ m ∧ m < 100) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3798_379812


namespace NUMINAMATH_CALUDE_percentage_of_rejected_meters_l3798_379841

def total_meters : ℕ := 100
def rejected_meters : ℕ := 10

theorem percentage_of_rejected_meters :
  (rejected_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_rejected_meters_l3798_379841


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_l3798_379875

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_l3798_379875


namespace NUMINAMATH_CALUDE_tangent_fifteen_degrees_ratio_l3798_379888

theorem tangent_fifteen_degrees_ratio (π : Real) :
  let tan15 := Real.tan (15 * π / 180)
  (1 + tan15) / (1 - tan15) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_fifteen_degrees_ratio_l3798_379888


namespace NUMINAMATH_CALUDE_min_value_theorem_l3798_379859

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧ 
  x^4 * y^3 * z^2 = (1/3456 : ℝ) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3798_379859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3798_379830

/-- Given an arithmetic sequence starting at 200, ending at 0, with a common difference of -5,
    the number of terms in the sequence is 41. -/
theorem arithmetic_sequence_length : 
  let start : ℤ := 200
  let end_val : ℤ := 0
  let diff : ℤ := -5
  let n : ℤ := (start - end_val) / (-diff) + 1
  n = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3798_379830


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l3798_379867

theorem wood_length_after_sawing (original_length sawing_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : sawing_length = 2.3) : 
  original_length - sawing_length = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l3798_379867


namespace NUMINAMATH_CALUDE_general_rule_l3798_379827

theorem general_rule (n : ℕ+) :
  (n + 1 : ℚ) / n + (n + 1 : ℚ) = (n + 2 : ℚ) + 1 / n := by sorry

end NUMINAMATH_CALUDE_general_rule_l3798_379827


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3798_379822

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 30 →
  ∃ (isosceles_perimeter : ℝ),
    isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
    isosceles_perimeter = 70 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3798_379822


namespace NUMINAMATH_CALUDE_triangle_properties_l3798_379896

/-- Given a triangle ABC where b = 2√3 and 2a - c = 2b cos C, prove that B = π/3 and the maximum value of 3a + 2c is 4√19 -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b = 2 * Real.sqrt 3 →
  2 * a - c = 2 * b * Real.cos C →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (B = π / 3 ∧ ∃ (x : ℝ), 3 * a + 2 * c ≤ 4 * Real.sqrt 19 ∧ 
    ∃ (A' B' C' a' b' c' : ℝ), 
      b' = 2 * Real.sqrt 3 ∧
      2 * a' - c' = 2 * b' * Real.cos C' ∧
      0 < A' ∧ A' < π ∧
      0 < B' ∧ B' < π ∧
      0 < C' ∧ C' < π ∧
      A' + B' + C' = π ∧
      3 * a' + 2 * c' = x) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3798_379896


namespace NUMINAMATH_CALUDE_f_min_max_on_interval_l3798_379835

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- State the theorem
theorem f_min_max_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≥ min) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    min = 1 ∧ max = 193 :=
by sorry


end NUMINAMATH_CALUDE_f_min_max_on_interval_l3798_379835


namespace NUMINAMATH_CALUDE_probability_is_one_sixth_l3798_379820

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light
    during a randomly chosen five-second interval -/
def probabilityOfColorChange (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let favorableDuration := 15 -- 5 seconds before each color change
  favorableDuration / totalCycleDuration

/-- The specific traffic light cycle from the problem -/
def problemCycle : TrafficLightCycle :=
  { green := 45
  , yellow := 5
  , red := 40 }

theorem probability_is_one_sixth :
  probabilityOfColorChange problemCycle = 1 / 6 := by
  sorry

#eval probabilityOfColorChange problemCycle

end NUMINAMATH_CALUDE_probability_is_one_sixth_l3798_379820


namespace NUMINAMATH_CALUDE_distinct_roots_sum_squares_l3798_379815

theorem distinct_roots_sum_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_squares_l3798_379815


namespace NUMINAMATH_CALUDE_cheese_cost_is_seven_l3798_379893

/-- The cost of a pound of cheese, given Tony's initial amount, beef cost, and remaining amount after purchase. -/
def cheese_cost (initial_amount beef_cost remaining_amount : ℚ) : ℚ :=
  (initial_amount - beef_cost - remaining_amount) / 3

/-- Theorem stating that the cost of a pound of cheese is $7 under the given conditions. -/
theorem cheese_cost_is_seven :
  let initial_amount : ℚ := 87
  let beef_cost : ℚ := 5
  let remaining_amount : ℚ := 61
  cheese_cost initial_amount beef_cost remaining_amount = 7 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_is_seven_l3798_379893


namespace NUMINAMATH_CALUDE_product_of_radicals_l3798_379847

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 98 * q * Real.sqrt (3 * q) := by
  sorry

end NUMINAMATH_CALUDE_product_of_radicals_l3798_379847


namespace NUMINAMATH_CALUDE_matrix_power_zero_l3798_379883

theorem matrix_power_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_zero_l3798_379883


namespace NUMINAMATH_CALUDE_square_sum_product_l3798_379882

theorem square_sum_product (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) :
  a^2 + a*b + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3798_379882


namespace NUMINAMATH_CALUDE_march_first_is_monday_l3798_379854

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given date in March
def marchDayOfWeek (date : Nat) : DayOfWeek := sorry

-- State the theorem
theorem march_first_is_monday : 
  marchDayOfWeek 8 = DayOfWeek.Monday → marchDayOfWeek 1 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_march_first_is_monday_l3798_379854


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_l3798_379845

theorem opposite_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  2023 * a + 2023 * b - 21 / (c * d) = -21 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_l3798_379845


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_to_7_l3798_379878

/-- The sum of the 4th to 7th terms of a geometric sequence with first term 1 and common ratio 3 is 1080 -/
theorem geometric_sequence_sum_4_to_7 :
  let a : ℕ → ℝ := λ n => 1 * (3 : ℝ) ^ (n - 1)
  (a 4) + (a 5) + (a 6) + (a 7) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_4_to_7_l3798_379878


namespace NUMINAMATH_CALUDE_number_problem_l3798_379806

theorem number_problem (x : ℝ) : 
  ((1/5 * 1/4 * x) - (5/100 * x)) + ((1/3 * x) - (1/7 * x)) = (1/10 * x - 12) → 
  x = -132 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3798_379806


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_perimeter_l3798_379811

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Define the given condition
def tan_condition (t : RightTriangle) : Prop :=
  Real.tan t.A + Real.tan t.B + Real.tan t.A * Real.tan t.B = 1

-- Theorem for part 1
theorem angle_C_measure (t : RightTriangle) (h : tan_condition t) : t.C = 135 := by
  sorry

-- Theorem for part 2
theorem triangle_perimeter 
  (t : RightTriangle) 
  (h1 : tan_condition t) 
  (h2 : t.A = 15) 
  (h3 : t.AB = Real.sqrt 2) : 
  t.AB + t.BC + t.AC = (2 + Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_perimeter_l3798_379811


namespace NUMINAMATH_CALUDE_routes_equal_choose_l3798_379804

/-- The number of routes in a 3x2 grid from top-left to bottom-right -/
def num_routes : ℕ := 10

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_2_from_5 : ℕ := Nat.choose 5 2

/-- Theorem stating that the number of routes is equal to choosing 2 from 5 -/
theorem routes_equal_choose :
  num_routes = choose_2_from_5 := by sorry

end NUMINAMATH_CALUDE_routes_equal_choose_l3798_379804


namespace NUMINAMATH_CALUDE_blue_paint_cans_l3798_379879

/-- Given a paint mixture with blue to yellow ratio of 7:3 and 50 total cans, 
    prove that 35 cans contain blue paint. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio yellow_ratio : ℕ) : 
  total_cans = 50 → 
  blue_ratio = 7 → 
  yellow_ratio = 3 → 
  (blue_ratio * total_cans) / (blue_ratio + yellow_ratio) = 35 := by
sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l3798_379879


namespace NUMINAMATH_CALUDE_dice_sum_product_l3798_379864

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l3798_379864


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3798_379813

/-- The interval for systematic sampling -/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The systematic sampling interval for a population of 1200 and sample size of 30 is 40 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3798_379813


namespace NUMINAMATH_CALUDE_octahedron_sphere_probability_l3798_379887

/-- Represents a regular octahedron with inscribed and circumscribed spheres -/
structure OctahedronWithSpheres where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Assumption that the inscribed sphere radius is one-third of the circumscribed sphere radius -/
  h_r_eq : r = R / 3

/-- The probability that a randomly chosen point in the circumscribed sphere
    lies inside one of the nine smaller spheres (one inscribed and eight tangent to faces) -/
theorem octahedron_sphere_probability (o : OctahedronWithSpheres) :
  (volume_ratio : ℝ) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_octahedron_sphere_probability_l3798_379887


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3798_379817

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 13 ∧ c = 7 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3798_379817


namespace NUMINAMATH_CALUDE_pennies_found_l3798_379890

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of quarters found -/
def num_quarters : ℕ := 12

/-- The total value in cents -/
def total_value : ℕ := 307

/-- The number of pennies found -/
def num_pennies : ℕ := (total_value - num_quarters * quarter_value) / penny_value

theorem pennies_found : num_pennies = 7 := by
  sorry

end NUMINAMATH_CALUDE_pennies_found_l3798_379890


namespace NUMINAMATH_CALUDE_sequence_a_10_l3798_379868

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m + n) = a m * a n

theorem sequence_a_10 (a : ℕ+ → ℝ) (h1 : sequence_property a) (h2 : a 3 = 8) :
  a 10 = 1024 := by sorry

end NUMINAMATH_CALUDE_sequence_a_10_l3798_379868


namespace NUMINAMATH_CALUDE_brenda_final_lead_l3798_379814

theorem brenda_final_lead (initial_lead : ℕ) (brenda_play : ℕ) (david_play : ℕ) : 
  initial_lead = 22 → brenda_play = 15 → david_play = 32 → 
  initial_lead + brenda_play - david_play = 5 := by
  sorry

end NUMINAMATH_CALUDE_brenda_final_lead_l3798_379814


namespace NUMINAMATH_CALUDE_custom_op_example_l3798_379821

-- Define the custom operation ⊗
def custom_op (a b : ℤ) : ℤ := 2 * a - b

-- Theorem statement
theorem custom_op_example : custom_op 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3798_379821


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3798_379870

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 8) 
  (h2 : x * y = -6) : 
  9 * x^2 + 16 * y^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3798_379870


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3798_379836

-- Define the type for planes
variable {Plane : Type}

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (α β γ : Plane) 
  (h1 : parallel γ α) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3798_379836


namespace NUMINAMATH_CALUDE_total_cost_per_pineapple_l3798_379844

/-- The cost per pineapple including shipping -/
def cost_per_pineapple (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) : ℚ :=
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples

/-- Theorem: The total cost per pineapple is $3.00 -/
theorem total_cost_per_pineapple :
  cost_per_pineapple (25/20) 12 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_per_pineapple_l3798_379844


namespace NUMINAMATH_CALUDE_miles_per_day_l3798_379862

theorem miles_per_day (weekly_goal : ℕ) (days_run : ℕ) (miles_left : ℕ) : 
  weekly_goal = 24 → 
  days_run = 6 → 
  miles_left = 6 → 
  (weekly_goal - miles_left) / days_run = 3 := by
sorry

end NUMINAMATH_CALUDE_miles_per_day_l3798_379862


namespace NUMINAMATH_CALUDE_max_angles_less_than_108_is_4_l3798_379892

/-- The maximum number of angles less than 108° in a convex polygon -/
def max_angles_less_than_108 (n : ℕ) : ℕ := 4

/-- Theorem stating that the maximum number of angles less than 108° in a convex n-gon is 4 -/
theorem max_angles_less_than_108_is_4 (n : ℕ) (h : n ≥ 3) :
  max_angles_less_than_108 n = 4 := by sorry

end NUMINAMATH_CALUDE_max_angles_less_than_108_is_4_l3798_379892


namespace NUMINAMATH_CALUDE_A_work_days_l3798_379816

/-- Represents the total work to be done -/
def W : ℝ := 1

/-- The number of days B alone can finish the work -/
def B_days : ℝ := 6

/-- The number of days A worked alone before B joined -/
def A_solo_days : ℝ := 3

/-- The number of days A and B worked together -/
def AB_days : ℝ := 3

/-- The number of days A can finish the work alone -/
def A_days : ℝ := 12

theorem A_work_days : 
  W = A_solo_days * (W / A_days) + AB_days * (W / A_days + W / B_days) → A_days = 12 := by
  sorry

end NUMINAMATH_CALUDE_A_work_days_l3798_379816


namespace NUMINAMATH_CALUDE_binary_1011011_eq_91_l3798_379837

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011011₂ -/
def binary_1011011 : List Bool := [true, true, false, true, true, false, true]

/-- Theorem: The decimal equivalent of 1011011₂ is 91 -/
theorem binary_1011011_eq_91 : binary_to_decimal binary_1011011 = 91 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011011_eq_91_l3798_379837


namespace NUMINAMATH_CALUDE_expression_equality_l3798_379855

theorem expression_equality : (3 + 2)^127 + 3 * (2^126 + 3^126) = 5^127 + 3 * 2^126 + 3 * 3^126 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3798_379855


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l3798_379840

theorem log_expression_equals_two :
  Real.log 4 + Real.log 5 * Real.log 20 + (Real.log 5)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l3798_379840


namespace NUMINAMATH_CALUDE_hospital_staff_count_l3798_379832

theorem hospital_staff_count (total : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (h1 : total = 250) 
  (h2 : ratio_doctors = 2) 
  (h3 : ratio_nurses = 3) : 
  (ratio_nurses * total) / (ratio_doctors + ratio_nurses) = 150 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l3798_379832


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l3798_379826

theorem floor_sqrt_equality (n : ℕ) : ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l3798_379826


namespace NUMINAMATH_CALUDE_point_on_line_l3798_379823

/-- Given six points on a line and a point P satisfying certain conditions, prove OP -/
theorem point_on_line (a b c d e : ℝ) : 
  ∀ (O A B C D E P : ℝ), 
    O < A ∧ A < B ∧ B < C ∧ C < D ∧ D < E ∧   -- Points in order
    A - O = a ∧                               -- Distance OA
    B - O = b ∧                               -- Distance OB
    C - O = c ∧                               -- Distance OC
    D - O = d ∧                               -- Distance OE
    E - O = e ∧                               -- Distance OE
    C ≤ P ∧ P ≤ D ∧                           -- P between C and D
    (A - P) * (P - D) = (C - P) * (P - E) →   -- AP:PE = CP:PD
    P - O = (c * e - a * d) / (a - c + e - d) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l3798_379823


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3798_379877

theorem arithmetic_calculations :
  ((54 + 38) * 15 = 1380) ∧
  (1500 - 32 * 45 = 60) ∧
  (157 * (70 / 35) = 314) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3798_379877


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3798_379857

/-- Calculates the total amount after simple interest --/
def simpleInterestTotal (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the initial conditions, the total amount after 7 years is $850 --/
theorem simple_interest_problem (initialSum : ℝ) (totalAfter2Years : ℝ) :
  initialSum = 500 →
  totalAfter2Years = 600 →
  ∃ (rate : ℝ),
    simpleInterestTotal initialSum rate 2 = totalAfter2Years ∧
    simpleInterestTotal initialSum rate 7 = 850 := by
  sorry

/-- The solution to the problem --/
def solution : ℝ := 850

end NUMINAMATH_CALUDE_simple_interest_problem_l3798_379857


namespace NUMINAMATH_CALUDE_sum_of_f_zero_and_f_neg_two_l3798_379866

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = f (x + y)

theorem sum_of_f_zero_and_f_neg_two (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 2 = 4) : 
  f 0 + f (-2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_zero_and_f_neg_two_l3798_379866


namespace NUMINAMATH_CALUDE_hiding_ways_correct_l3798_379851

/-- The number of ways to hide 3 people in 6 cabinets with at most 2 people per cabinet -/
def hidingWays : ℕ := 210

/-- The number of people to be hidden -/
def numPeople : ℕ := 3

/-- The number of available cabinets -/
def numCabinets : ℕ := 6

/-- The maximum number of people that can be hidden in a single cabinet -/
def maxPerCabinet : ℕ := 2

theorem hiding_ways_correct :
  hidingWays = 
    (numCabinets * (numCabinets - 1) * (numCabinets - 2)) + 
    (Nat.choose numPeople 2 * Nat.choose numCabinets 1 * Nat.choose (numCabinets - 1) 1) := by
  sorry

#check hiding_ways_correct

end NUMINAMATH_CALUDE_hiding_ways_correct_l3798_379851


namespace NUMINAMATH_CALUDE_max_value_z_l3798_379895

theorem max_value_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y ≤ 2) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3*x - y ∧ z ≤ 6 ∧ ∃ (x' y' : ℝ), x' - y' ≥ 0 ∧ x' + y' ≤ 2 ∧ y' ≥ 0 ∧ 3*x' - y' = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l3798_379895


namespace NUMINAMATH_CALUDE_car_travel_time_l3798_379819

/-- Given a car's initial travel and additional distance, calculate the total travel time. -/
theorem car_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) 
  (h1 : initial_distance = 180) 
  (h2 : initial_time = 4)
  (h3 : additional_distance = 135) :
  let speed := initial_distance / initial_time
  let additional_time := additional_distance / speed
  initial_time + additional_time = 7 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_l3798_379819


namespace NUMINAMATH_CALUDE_meaningful_expression_l3798_379898

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1)^(0 : ℕ) / Real.sqrt (x + 2)) ↔ x > -2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3798_379898


namespace NUMINAMATH_CALUDE_fourteen_own_all_pets_l3798_379805

/-- The number of people who own all three types of pets (cats, dogs, and rabbits) -/
def people_with_all_pets (total : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (rabbit_owners : ℕ) (two_pet_owners : ℕ) : ℕ :=
  cat_owners + dog_owners + rabbit_owners - two_pet_owners - total

/-- Theorem stating that given the conditions in the problem, 14 people own all three types of pets -/
theorem fourteen_own_all_pets :
  people_with_all_pets 60 30 40 16 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_own_all_pets_l3798_379805


namespace NUMINAMATH_CALUDE_root_product_equation_l3798_379876

theorem root_product_equation (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - p*(a^2 + 1/b^2) + r = 0) →
  ((b^2 + 1/a^2)^2 - p*(b^2 + 1/a^2) + r = 0) →
  r = 100/9 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l3798_379876


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l3798_379860

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

theorem sandy_clothes_cost :
  shorts_cost + shirt_cost + jacket_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l3798_379860


namespace NUMINAMATH_CALUDE_num_small_triangles_odd_num_small_triangles_formula_l3798_379831

/-- A triangle with interior points and connections -/
structure TriangleWithPoints where
  n : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points (including vertices) are collinear
  max_connections : Bool  -- points are connected to maximize small triangles
  no_intersections : Bool  -- resulting segments do not intersect

/-- The number of small triangles formed in a TriangleWithPoints -/
def num_small_triangles (t : TriangleWithPoints) : ℕ := 2 * t.n + 1

/-- Theorem stating that the number of small triangles is odd -/
theorem num_small_triangles_odd (t : TriangleWithPoints) : 
  Odd (num_small_triangles t) := by
  sorry

/-- Theorem stating that the number of small triangles is 2n + 1 -/
theorem num_small_triangles_formula (t : TriangleWithPoints) : 
  num_small_triangles t = 2 * t.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_num_small_triangles_odd_num_small_triangles_formula_l3798_379831


namespace NUMINAMATH_CALUDE_carnival_tickets_billy_carnival_tickets_l3798_379891

/-- Calculate the total number of tickets used at a carnival --/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides ferris_wheel_cost bumper_car_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost =
  (ferris_wheel_rides * ferris_wheel_cost) + (bumper_car_rides * bumper_car_cost) := by
  sorry

/-- Billy's carnival ticket usage --/
theorem billy_carnival_tickets :
  let ferris_wheel_rides : ℕ := 7
  let bumper_car_rides : ℕ := 3
  let ferris_wheel_cost : ℕ := 6
  let bumper_car_cost : ℕ := 4
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_billy_carnival_tickets_l3798_379891


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l3798_379842

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : Marathon :=
  { miles := 26, yards := 395 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y } : TotalDistance
    ) = 645 ∧
    m * yards_per_mile + y = 
      number_of_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l3798_379842


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l3798_379833

theorem concert_ticket_sales : ∀ (adult_tickets : ℕ) (senior_tickets : ℕ),
  -- Total tickets sold is 120
  adult_tickets + senior_tickets + adult_tickets = 120 →
  -- Total revenue is $1100
  12 * adult_tickets + 10 * senior_tickets + 6 * adult_tickets = 1100 →
  -- The number of senior tickets is 20
  senior_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l3798_379833


namespace NUMINAMATH_CALUDE_max_profit_is_120_l3798_379894

/-- Profit function for location A -/
def L₁ (x : ℕ) : ℤ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℕ) : ℤ := 2*x

/-- Total profit function -/
def L (x : ℕ) : ℤ := L₁ x + L₂ (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x ≤ 15 ∧ L x = 120 ∧ ∀ y : ℕ, y ≤ 15 → L y ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l3798_379894


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l3798_379884

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) / 2 = (((p : ℚ) / 2 - 5) / p) * 100 → p / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l3798_379884


namespace NUMINAMATH_CALUDE_problem_statement_l3798_379801

theorem problem_statement (x y : ℝ) (h1 : 2*x + 5*y = 10) (h2 : x*y = -10) :
  4*x^2 + 25*y^2 = 300 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3798_379801


namespace NUMINAMATH_CALUDE_fifty_percent_x_equals_690_l3798_379834

theorem fifty_percent_x_equals_690 : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_x_equals_690_l3798_379834


namespace NUMINAMATH_CALUDE_simplify_expression_l3798_379861

theorem simplify_expression : 3 * Real.sqrt 48 - 6 * Real.sqrt (1/3) + (Real.sqrt 3 - 1)^2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3798_379861


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3798_379856

/-- Given a rectangle intersecting a circle with specific chord properties,
    prove the ratio of their areas. -/
theorem rectangle_circle_area_ratio :
  ∀ (r : ℝ) (x y : ℝ),
    r > 0 →                 -- radius is positive
    x > 0 →                 -- shorter side is positive
    y > 0 →                 -- longer side is positive
    y = r →                 -- longer side equals radius (chord property)
    x = r / 2 →             -- shorter side equals half radius (chord property)
    y = 2 * x →             -- longer side is twice the shorter side
    (x * y) / (π * r^2) = 1 / (2 * π) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3798_379856


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3798_379825

theorem geometric_sequence_sum (a₁ : ℝ) (r : ℝ) :
  a₁ = 3125 →
  r = 1/5 →
  (a₁ * r^5 = 1) →
  (a₁ * r^3 + a₁ * r^4 = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3798_379825


namespace NUMINAMATH_CALUDE_initial_deer_families_l3798_379874

/-- The number of deer families that stayed in the area -/
def families_stayed : ℕ := 45

/-- The number of deer families that moved out of the area -/
def families_moved_out : ℕ := 34

/-- The initial number of deer families in the area -/
def initial_families : ℕ := families_stayed + families_moved_out

theorem initial_deer_families : initial_families = 79 := by
  sorry

end NUMINAMATH_CALUDE_initial_deer_families_l3798_379874


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3798_379881

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The length of each lateral side -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  inscribed_radius : ℝ

/-- The area of an isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  2 * t.lateral_side * t.inscribed_radius

/-- Theorem: The area of an isosceles trapezoid with lateral side length 9 and an inscribed circle of radius 4 is 72 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨9, 4⟩
  area t = 72 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3798_379881


namespace NUMINAMATH_CALUDE_exists_n_with_uniform_200th_digit_distribution_l3798_379800

def digit_at_position (x : ℝ) (pos : ℕ) : ℕ := sorry

def count_occurrences (digit : ℕ) (numbers : List ℝ) (pos : ℕ) : ℕ := sorry

theorem exists_n_with_uniform_200th_digit_distribution :
  ∃ (n : ℕ+),
    ∀ (digit : Fin 10),
      count_occurrences digit.val
        (List.map (λ k => Real.sqrt (n.val + k)) (List.range 1000))
        200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_uniform_200th_digit_distribution_l3798_379800


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3798_379872

theorem quadratic_equation_solution (k : ℚ) : 
  (∀ x : ℚ, k * x^2 + 8 * x + 15 = 0 ↔ (x = -3 ∨ x = -5/2)) → k = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3798_379872


namespace NUMINAMATH_CALUDE_symmetry_of_point_l3798_379809

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to y-axis -/
def symmetricToYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨6, 4⟩
  let B : Point := symmetricToYAxis A
  B = ⟨-6, 4⟩ := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l3798_379809


namespace NUMINAMATH_CALUDE_max_value_S_l3798_379899

/-- The maximum value of S given the conditions -/
theorem max_value_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≥ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_S_l3798_379899


namespace NUMINAMATH_CALUDE_solution_value_l3798_379865

theorem solution_value (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3798_379865


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3798_379871

/-- Given a quadratic equation x^2 - 4x + m = 0 with one root x₁ = 1, 
    prove that the other root x₂ = 3 -/
theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3798_379871


namespace NUMINAMATH_CALUDE_f_prime_at_two_l3798_379846

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_prime_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  ((deriv (f a b)) 1 = 0) →
  ((deriv (f a b)) 2 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_f_prime_at_two_l3798_379846


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3798_379852

theorem binomial_expansion_coefficient (a : ℝ) (b : ℝ) :
  (∃ x, (1 + a * x)^5 = 1 + 10 * x + b * x^2 + x^3 * (1 + a * x)^2) →
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3798_379852
