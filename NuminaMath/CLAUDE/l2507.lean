import Mathlib

namespace NUMINAMATH_CALUDE_base_eight_representation_l2507_250758

-- Define the representation function
def represent (base : ℕ) (n : ℕ) : ℕ := 
  3 * base^4 + 0 * base^3 + 4 * base^2 + 0 * base + 7

-- Define the theorem
theorem base_eight_representation : 
  ∃ (base : ℕ), base > 1 ∧ represent base 12551 = 30407 ∧ base = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_representation_l2507_250758


namespace NUMINAMATH_CALUDE_inequality_and_uniqueness_l2507_250778

theorem inequality_and_uniqueness 
  (a b c d : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (pos_d : 0 < d) 
  (sum_eq : a + b = 4) 
  (prod_eq : c * d = 4) : 
  (a * b ≤ c + d) ∧ 
  (a * b = c + d → 
    ∀ (a' b' c' d' : ℝ), 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
      a' + b' = 4 ∧ c' * d' = 4 ∧ 
      a' * b' = c' + d' → 
      a' = a ∧ b' = b ∧ c' = c ∧ d' = d) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_uniqueness_l2507_250778


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l2507_250750

/-- Conversion of flood damage from Canadian to American dollars -/
theorem flood_damage_conversion (damage_cad : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_cad = 50000000)
  (h2 : exchange_rate = 1.25)
  : damage_cad / exchange_rate = 40000000 := by
  sorry

end NUMINAMATH_CALUDE_flood_damage_conversion_l2507_250750


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_base_6_number_l2507_250779

/-- Represents the number 100111001 in base 6 -/
def base_6_number : ℕ := 6^8 + 6^5 + 6^4 + 6^3 + 6 + 1

/-- The largest prime divisor of base_6_number -/
def largest_prime_divisor : ℕ := 43

theorem largest_prime_divisor_of_base_6_number :
  (∀ p : ℕ, Prime p → p ∣ base_6_number → p ≤ largest_prime_divisor) ∧
  (Prime largest_prime_divisor ∧ largest_prime_divisor ∣ base_6_number) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_base_6_number_l2507_250779


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l2507_250724

/-- For a cone with an equilateral triangle as its axial section, 
    the angle of the sector formed by unfolding its lateral surface is π radians. -/
theorem cone_lateral_surface_angle (R r : ℝ) (α : ℝ) : 
  R > 0 ∧ r > 0 ∧ R = 2 * r → α = π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l2507_250724


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2507_250792

def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.45
def prob_rain_sunday : ℝ := 0.55

theorem weekend_rain_probability : 
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_weekend := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  1 - prob_no_rain_weekend = 0.82675 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2507_250792


namespace NUMINAMATH_CALUDE_right_triangle_tangent_midpoint_l2507_250743

theorem right_triangle_tangent_midpoint (n : ℕ) (a h : ℝ) (α : ℝ) :
  n > 1 →
  Odd n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_midpoint_l2507_250743


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l2507_250703

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (1 / x + 1 / y = 1) ∧
    (2 / x + 3 / y = 4) ∧
    (x = -1) ∧
    (y = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l2507_250703


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2507_250786

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 100) ∧ 
  (n % 6 = 4) ∧ 
  (∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n) ∧
  (n = 94) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2507_250786


namespace NUMINAMATH_CALUDE_tangent_line_t_value_l2507_250710

/-- A line in polar coordinates defined by ρcosθ = t, where t > 0 -/
structure PolarLine where
  t : ℝ
  t_pos : t > 0

/-- A curve in polar coordinates defined by ρ = 2sinθ -/
def PolarCurve : Type := Unit

/-- Predicate to check if a line is tangent to the curve -/
def is_tangent (l : PolarLine) (c : PolarCurve) : Prop := sorry

theorem tangent_line_t_value (l : PolarLine) (c : PolarCurve) :
  is_tangent l c → l.t = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_t_value_l2507_250710


namespace NUMINAMATH_CALUDE_sum_first_20_triangular_numbers_l2507_250717

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ :=
  (List.range n).map triangular_number |>.sum

/-- Theorem: The sum of the first 20 triangular numbers is 1540 -/
theorem sum_first_20_triangular_numbers :
  sum_triangular_numbers 20 = 1540 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_20_triangular_numbers_l2507_250717


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2507_250713

theorem complex_modulus_problem (b : ℝ) (z : ℂ) : 
  z = (b * Complex.I) / (4 + 3 * Complex.I) → 
  Complex.abs z = 5 → 
  b = 25 ∨ b = -25 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2507_250713


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2507_250790

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    a circle Ω with the real axis of C as its diameter,
    and P the intersection point of Ω and the asymptote of C in the first quadrant,
    if the slope of FP (where F is the right focus of C) is -b/a,
    then the equation of the asymptote of C is y = ±x -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 + b^2), 0)
  let Ω := {(x, y) : ℝ × ℝ | x^2 + y^2 = a^2}
  let P := (a / Real.sqrt 2, a / Real.sqrt 2)
  (P.2 - F.2) / (P.1 - F.1) = -b / a →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) : ℝ × ℝ | y = x ∨ y = -x} ↔
    ∃ (t : ℝ), t ≠ 0 ∧ x = a * t ∧ y = b * t :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2507_250790


namespace NUMINAMATH_CALUDE_max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l2507_250798

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0  -- Define a default value for x outside the given ranges

-- Theorem statements
theorem max_receptivity (x : ℝ) :
  (∀ x, f x ≤ 60) ∧
  (f 10 = 60) ∧
  (∀ x, 10 < x → x ≤ 15 → f x = 60) :=
sorry

theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

theorem no_continuous_high_receptivity :
  ¬ ∃ a b : ℝ, b - a = 12 ∧ ∀ x, a ≤ x ∧ x ≤ b → f x ≥ 56 :=
sorry

end NUMINAMATH_CALUDE_max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l2507_250798


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2507_250749

/-- The sum of the infinite geometric series 4/3 - 1/2 + 3/32 - 9/256 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3  -- first term
  let r : ℚ := -3/8 -- common ratio
  let S := Nat → ℚ  -- sequence type
  let series : S := fun n => a * r^n  -- geometric series
  ∑' n, series n = 32/33 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2507_250749


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2507_250728

def a (n : ℕ) : ℝ := 2 * n - 8

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) > a n) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) > a n / n) ∧
  (∃ n : ℕ, (n + 1) * a (n + 1) ≤ n * a n) ∧
  (∃ n : ℕ, a (n + 1)^2 ≤ a n^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2507_250728


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2507_250771

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ C * (a + b + c)) ∧
  (C = 2 / Real.sqrt 3) ∧
  (∀ (D : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ D * (a + b + c)) → D ≤ C) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2507_250771


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2507_250759

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  -b * (2 * a - b) + (a + b)^2 = a^2 + 2 * b^2 := by sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 - x / (2 + x)) / ((x^2 - 4) / (x^2 + 4*x + 4)) = 2 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2507_250759


namespace NUMINAMATH_CALUDE_range_of_a_max_value_of_z_l2507_250719

-- Define the variables
variable (a b z : ℝ)

-- Define the conditions
def condition1 : Prop := 2 * a + b = 9
def condition2 : Prop := |9 - b| + |a| < 3
def condition3 : Prop := a > 0 ∧ b > 0
def condition4 : Prop := z = a^2 * b

-- Theorem for part (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : 
  -1 < a ∧ a < 1 := by sorry

-- Theorem for part (ii)
theorem max_value_of_z (h1 : condition1 a b) (h3 : condition3 a b) (h4 : condition4 a b z) : 
  z ≤ 27 := by sorry

end NUMINAMATH_CALUDE_range_of_a_max_value_of_z_l2507_250719


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2507_250730

/-- If 4x^2 + mxy + y^2 is a perfect square, then m = ±4 -/
theorem perfect_square_condition (x y m : ℝ) : 
  (∃ (k : ℝ), 4*x^2 + m*x*y + y^2 = k^2) → (m = 4 ∨ m = -4) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2507_250730


namespace NUMINAMATH_CALUDE_regression_line_equation_l2507_250706

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the regression line -/
def is_regression_line_equation (line : RegressionLine) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = line.slope * (x - line.point.1) + line.point.2) ∧
  (f line.point.1 = line.point.2)

/-- The theorem stating the equation of the specific regression line -/
theorem regression_line_equation (line : RegressionLine) 
  (h1 : line.slope = 6.5)
  (h2 : line.point = (2, 3)) :
  is_regression_line_equation line (λ x => -10 + 6.5 * x) := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l2507_250706


namespace NUMINAMATH_CALUDE_sequence_equality_l2507_250769

theorem sequence_equality (a : Fin 100 → ℝ) 
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by sorry

end NUMINAMATH_CALUDE_sequence_equality_l2507_250769


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2507_250721

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a * x + y - 1 = 0 ∧ x - y + 3 = 0) → 
   (a * 1 + (-1) * 1 = -1)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2507_250721


namespace NUMINAMATH_CALUDE_vector_angle_condition_l2507_250729

-- Define the vectors a and b as functions of x
def a (x : ℝ) : Fin 2 → ℝ := ![2, x + 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 2, 6]

-- Define the dot product of a and b
def dot_product (x : ℝ) : ℝ := (a x 0) * (b x 0) + (a x 1) * (b x 1)

-- Define the cross product of a and b
def cross_product (x : ℝ) : ℝ := (a x 0) * (b x 1) - (a x 1) * (b x 0)

-- Theorem statement
theorem vector_angle_condition (x : ℝ) :
  (dot_product x > 0 ∧ cross_product x ≠ 0) ↔ (x > -5/4 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_vector_angle_condition_l2507_250729


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2507_250791

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    if a_1, a_3, and a_9 form a geometric sequence, then a_3 / a_1 = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence condition
  (h3 : (a 3 / a 1) = (a 9 / a 3)) -- geometric sequence condition
  : a 3 / a 1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2507_250791


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l2507_250787

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a ≥ 5 → b ≥ 5 → c ≥ 5 →
    isPrime a → isPrime b → isPrime c →
    isScalene a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l2507_250787


namespace NUMINAMATH_CALUDE_solution_difference_l2507_250747

theorem solution_difference (r s : ℝ) : 
  (∀ x, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2507_250747


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2507_250700

theorem power_of_two_equality (a b : ℕ+) (h : 2^(a.val) * 2^(b.val) = 8) : 
  (2^(a.val))^(b.val) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2507_250700


namespace NUMINAMATH_CALUDE_beach_trip_result_l2507_250796

/-- Represents the number of seashells found during a beach trip -/
def beach_trip (days : ℕ) (shells_per_day : ℕ) : ℕ :=
  days * shells_per_day

/-- Proves that a 5-day beach trip with 7 shells found per day results in 35 total shells -/
theorem beach_trip_result : beach_trip 5 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_result_l2507_250796


namespace NUMINAMATH_CALUDE_principal_value_range_of_argument_l2507_250755

theorem principal_value_range_of_argument (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (k : ℕ) (θ : ℝ), k ≤ 1 ∧ 
  Complex.arg z = θ ∧
  k * Real.pi - Real.arccos (-1/2) ≤ θ ∧ 
  θ ≤ k * Real.pi + Real.arccos (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_principal_value_range_of_argument_l2507_250755


namespace NUMINAMATH_CALUDE_difference_numbers_between_500_and_600_l2507_250748

def is_difference_number (n : ℕ) : Prop :=
  n % 7 = 6 ∧ n % 5 = 4

theorem difference_numbers_between_500_and_600 :
  {n : ℕ | 500 < n ∧ n < 600 ∧ is_difference_number n} = {524, 559, 594} := by
  sorry

end NUMINAMATH_CALUDE_difference_numbers_between_500_and_600_l2507_250748


namespace NUMINAMATH_CALUDE_floor_product_equals_49_l2507_250772

theorem floor_product_equals_49 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
sorry

end NUMINAMATH_CALUDE_floor_product_equals_49_l2507_250772


namespace NUMINAMATH_CALUDE_tan_4305_degrees_l2507_250731

theorem tan_4305_degrees : Real.tan (4305 * π / 180) = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4305_degrees_l2507_250731


namespace NUMINAMATH_CALUDE_plane_speed_against_wind_l2507_250711

/-- Calculates the ground speed of a plane flying against a tailwind, given its ground speed with the tailwind and the wind speed. -/
def ground_speed_against_wind (ground_speed_with_wind wind_speed : ℝ) : ℝ :=
  2 * ground_speed_with_wind - 2 * wind_speed - wind_speed

/-- Theorem stating that a plane with a ground speed of 460 mph with a 75 mph tailwind
    will have a ground speed of 310 mph against the same tailwind. -/
theorem plane_speed_against_wind :
  ground_speed_against_wind 460 75 = 310 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_against_wind_l2507_250711


namespace NUMINAMATH_CALUDE_book_pages_theorem_l2507_250715

theorem book_pages_theorem (P : ℕ) 
  (h1 : P / 2 + P / 4 + P / 6 + 20 = P) : P = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l2507_250715


namespace NUMINAMATH_CALUDE_retail_price_decrease_percentage_l2507_250739

/-- Proves that the retail price decrease percentage is equal to 44.000000000000014% 
    given the conditions in the problem. -/
theorem retail_price_decrease_percentage 
  (wholesale_price : ℝ) 
  (retail_price : ℝ) 
  (decrease_percentage : ℝ) : 
  retail_price = wholesale_price * 1.80 →
  retail_price * (1 - decrease_percentage) = 
    (wholesale_price * 1.44000000000000014) * 1.80 →
  decrease_percentage = 0.44000000000000014 := by
sorry

end NUMINAMATH_CALUDE_retail_price_decrease_percentage_l2507_250739


namespace NUMINAMATH_CALUDE_log_inequality_inequality_with_roots_l2507_250797

-- Theorem 1
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 := by
  sorry

-- Theorem 2
theorem inequality_with_roots :
  6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_inequality_with_roots_l2507_250797


namespace NUMINAMATH_CALUDE_negative_x_count_l2507_250788

theorem negative_x_count : 
  ∃ (S : Finset ℤ), 
    (∀ x ∈ S, x < 0 ∧ ∃ n : ℕ+, (x + 196 : ℝ) = n^2) ∧ 
    (∀ x : ℤ, x < 0 → (∃ n : ℕ+, (x + 196 : ℝ) = n^2) → x ∈ S) ∧
    Finset.card S = 13 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_count_l2507_250788


namespace NUMINAMATH_CALUDE_kitchen_hours_theorem_l2507_250753

/-- The minimum number of hours required to produce a given number of large and small cakes -/
def min_hours_required (num_helpers : ℕ) (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) (large_cakes_needed : ℕ) (small_cakes_needed : ℕ) : ℕ :=
  max 
    (large_cakes_needed / (num_helpers * large_cakes_per_hour))
    (small_cakes_needed / (num_helpers * small_cakes_per_hour))

theorem kitchen_hours_theorem :
  min_hours_required 10 2 35 20 700 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_hours_theorem_l2507_250753


namespace NUMINAMATH_CALUDE_saltwater_aquariums_count_l2507_250722

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 2184 / 39

/-- The number of freshwater aquariums Tyler has -/
def freshwater_aquariums : ℕ := 10

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := 2184

theorem saltwater_aquariums_count : saltwater_aquariums = 56 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_aquariums_count_l2507_250722


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2507_250789

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2507_250789


namespace NUMINAMATH_CALUDE_f_range_l2507_250733

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem f_range :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 3) ∧
  (∀ ε > 0, ∃ x, |f x - 3| < ε ∧ f x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_range_l2507_250733


namespace NUMINAMATH_CALUDE_bounds_of_abs_diff_over_sum_l2507_250761

theorem bounds_of_abs_diff_over_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z, z = |x - y| / (|x| + |y|) → m ≤ z ∧ z ≤ M) ∧
    m = 0 ∧ M = 1 ∧ M - m = 1 :=
by sorry

end NUMINAMATH_CALUDE_bounds_of_abs_diff_over_sum_l2507_250761


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2507_250793

theorem quadratic_equation_coefficient :
  ∀ a b c : ℝ,
  (∀ x : ℝ, 2 * x^2 = 9 * x + 8) →
  (a * x^2 + b * x + c = 0 ↔ 2 * x^2 - 9 * x - 8 = 0) →
  a = 2 →
  b = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2507_250793


namespace NUMINAMATH_CALUDE_product_abcd_equals_1280_l2507_250776

theorem product_abcd_equals_1280 
  (a b c d : ℝ) 
  (eq1 : 2*a + 4*b + 6*c + 8*d = 48)
  (eq2 : 4*d + 2*c = 2*b)
  (eq3 : 4*b + 2*c = 2*a)
  (eq4 : c - 2 = d)
  (eq5 : d + b = 10) :
  a * b * c * d = 1280 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_equals_1280_l2507_250776


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_cubic_l2507_250781

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_cubic (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_cubic_l2507_250781


namespace NUMINAMATH_CALUDE_fifteenths_in_fraction_l2507_250702

theorem fifteenths_in_fraction : 
  let whole_number : ℚ := 82
  let fraction : ℚ := 3 / 5
  let divisor : ℚ := 1 / 15
  let multiplier : ℕ := 3
  let subtrahend_whole : ℕ := 42
  let subtrahend_fraction : ℚ := 7 / 10
  
  ((whole_number + fraction) / divisor * multiplier) - 
  (subtrahend_whole + subtrahend_fraction) = 3674.3 := by sorry

end NUMINAMATH_CALUDE_fifteenths_in_fraction_l2507_250702


namespace NUMINAMATH_CALUDE_average_daily_low_temperature_l2507_250701

theorem average_daily_low_temperature (temperatures : List ℝ) : 
  temperatures = [40, 47, 45, 41, 39] → 
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temperature_l2507_250701


namespace NUMINAMATH_CALUDE_meetings_count_is_four_l2507_250752

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the number of meetings is 4 --/
theorem meetings_count_is_four :
  ∀ (p : MeetingProblem),
    p.michael_speed = 6 ∧
    p.pail_distance = 150 ∧
    p.truck_speed = 12 ∧
    p.truck_stop_time = 20 →
    number_of_meetings p = 4 :=
  sorry

end NUMINAMATH_CALUDE_meetings_count_is_four_l2507_250752


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2507_250775

theorem tile_border_ratio (n : ℕ) (s d : ℝ) (h1 : n = 24) 
  (h2 : (n^2 : ℝ) * s^2 * 0.64 = 576 * s^2) : d / s = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2507_250775


namespace NUMINAMATH_CALUDE_f_16_values_l2507_250725

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * f (a^2 + b^2) = 2 * (f a)^2 + 2 * (f b)^2 - f a * f b

theorem f_16_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  {n : ℕ | f 16 = n} = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_f_16_values_l2507_250725


namespace NUMINAMATH_CALUDE_bobby_cars_after_seven_years_l2507_250780

def initial_cars : ℕ := 30

def double (n : ℕ) : ℕ := 2 * n

def donate (n : ℕ) : ℕ := n - (n / 10)

def update_cars (year : ℕ) (cars : ℕ) : ℕ :=
  if year % 2 = 0 ∧ year ≠ 0 then
    donate (double cars)
  else
    double cars

def cars_after_years (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_cars
  | n + 1 => update_cars n (cars_after_years n)

theorem bobby_cars_after_seven_years :
  cars_after_years 7 = 2792 := by sorry

end NUMINAMATH_CALUDE_bobby_cars_after_seven_years_l2507_250780


namespace NUMINAMATH_CALUDE_weight_range_proof_l2507_250723

/-- Given the weights of Tracy, John, and Jake, prove that the range of their weights is 14 kg -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) :
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end NUMINAMATH_CALUDE_weight_range_proof_l2507_250723


namespace NUMINAMATH_CALUDE_basin_capacity_l2507_250794

/-- The capacity of a basin given waterfall flow rate, leak rate, and fill time -/
theorem basin_capacity
  (waterfall_flow : ℝ)  -- Flow rate of the waterfall in gallons per second
  (leak_rate : ℝ)       -- Leak rate of the basin in gallons per second
  (fill_time : ℝ)       -- Time to fill the basin in seconds
  (h1 : waterfall_flow = 24)
  (h2 : leak_rate = 4)
  (h3 : fill_time = 13)
  : (waterfall_flow - leak_rate) * fill_time = 260 :=
by
  sorry

#check basin_capacity

end NUMINAMATH_CALUDE_basin_capacity_l2507_250794


namespace NUMINAMATH_CALUDE_Q_bounds_l2507_250783

/-- The equation of the given curve -/
def curve_equation (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

/-- The expression we want to bound -/
def Q (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

/-- Theorem stating the bounds of Q for points on the curve -/
theorem Q_bounds :
  ∀ x y : ℝ, curve_equation x y → 3 ≤ Q x y ∧ Q x y ≤ 124 :=
by sorry

end NUMINAMATH_CALUDE_Q_bounds_l2507_250783


namespace NUMINAMATH_CALUDE_angle_trigonometric_identity_l2507_250751

theorem angle_trigonometric_identity (α : Real) (m n : Real) : 
  -- Conditions
  α ∈ Set.Icc 0 π ∧ 
  m^2 + n^2 = 1 ∧ 
  n / m = -2 →
  -- Conclusion
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometric_identity_l2507_250751


namespace NUMINAMATH_CALUDE_floor_e_l2507_250745

theorem floor_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_e_l2507_250745


namespace NUMINAMATH_CALUDE_sum_of_zeros_l2507_250744

/-- The parabola after transformations -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 7

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros : ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_l2507_250744


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2507_250760

/-- Given vectors a and b in ℝ², prove that the magnitude of 2a + b equals 4. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : a = (3, -3)) (hb : b = (-2, 6)) : 
  ‖(2 : ℝ) • a + b‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2507_250760


namespace NUMINAMATH_CALUDE_construction_work_proof_l2507_250742

/-- Represents the number of men who dropped out -/
def men_dropped_out : ℕ := 1

theorem construction_work_proof :
  let initial_men : ℕ := 5
  let half_job_days : ℕ := 15
  let full_job_days : ℕ := 30
  let completion_days : ℕ := 25
  (initial_men * full_job_days : ℚ) = ((initial_men - men_dropped_out) * completion_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_construction_work_proof_l2507_250742


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2507_250737

theorem complex_number_modulus (z : ℂ) (h : 1 + z = (1 - z) * Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2507_250737


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2507_250734

def num_letters : Nat := 26
def num_letter_positions : Nat := 4
def num_digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  Nat.choose num_letters 2 *
  Nat.choose num_letter_positions 2 *
  2 *
  (10 * 9 * 8)

theorem license_plate_theorem :
  license_plate_combinations = 2808000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2507_250734


namespace NUMINAMATH_CALUDE_order_of_mnpq_l2507_250782

theorem order_of_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end NUMINAMATH_CALUDE_order_of_mnpq_l2507_250782


namespace NUMINAMATH_CALUDE_invalid_diagonal_sets_l2507_250773

-- Define a function to check if a set of three numbers satisfies the condition
def isValidDiagonalSet (x y z : ℝ) : Prop :=
  x^2 + y^2 ≥ z^2 ∧ x^2 + z^2 ≥ y^2 ∧ y^2 + z^2 ≥ x^2

-- Theorem stating which sets are invalid for external diagonals of a right regular prism
theorem invalid_diagonal_sets :
  (¬ isValidDiagonalSet 3 4 6) ∧
  (¬ isValidDiagonalSet 5 5 8) ∧
  (¬ isValidDiagonalSet 7 8 12) ∧
  (isValidDiagonalSet 6 8 10) ∧
  (isValidDiagonalSet 3 4 5) :=
by sorry

end NUMINAMATH_CALUDE_invalid_diagonal_sets_l2507_250773


namespace NUMINAMATH_CALUDE_sum_equals_140_l2507_250736

theorem sum_equals_140 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (h1 : x^2 + y^2 = 2500) (h2 : z^2 + w^2 = 2500)
  (h3 : x * z = 1200) (h4 : y * w = 1200) : 
  x + y + z + w = 140 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_140_l2507_250736


namespace NUMINAMATH_CALUDE_min_distance_between_lines_l2507_250746

/-- The minimum distance between a point on the line 3x+4y-12=0 and a point on the line 6x+8y+5=0 is 29/10 -/
theorem min_distance_between_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 12 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ (d : ℝ), d = 29/10 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ line1 → q ∈ line2 → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_lines_l2507_250746


namespace NUMINAMATH_CALUDE_inequality_proof_l2507_250707

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hxy : x * y < 0) :
  (b / a + a / b ≥ 2) ∧ (x / y + y / x ≤ -2) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2507_250707


namespace NUMINAMATH_CALUDE_stock_percentage_calculation_l2507_250766

/-- Calculates the percentage of a stock given its yield and quoted price. -/
theorem stock_percentage_calculation (yield : ℝ) (quote : ℝ) :
  yield = 10 →
  quote = 160 →
  let face_value := 100
  let market_price := quote * face_value / 100
  let annual_income := yield * face_value / 100
  let stock_percentage := annual_income / market_price * 100
  stock_percentage = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_calculation_l2507_250766


namespace NUMINAMATH_CALUDE_inscribed_circles_distance_l2507_250768

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 100^2 ∧
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 160^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 200^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define a line perpendicular to another line
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the inscribed circle
def InscribedCircle (C : ℝ × ℝ) (r : ℝ) (A B D : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = r^2

theorem inscribed_circles_distance 
  (X Y Z M N O P : ℝ × ℝ) 
  (C₁ C₂ C₃ : ℝ × ℝ) 
  (r₁ r₂ r₃ : ℝ) :
  Triangle X Y Z →
  RightAngle X Y Z →
  Perpendicular X Z M N →
  Perpendicular X Y O P →
  InscribedCircle C₁ r₁ X Y Z →
  InscribedCircle C₂ r₂ Z M N →
  InscribedCircle C₃ r₃ Y O P →
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 26000 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_distance_l2507_250768


namespace NUMINAMATH_CALUDE_max_area_OAPF_l2507_250741

/-- The equation of ellipse C is (x^2/9) + (y^2/10) = 1 -/
def ellipse_equation (x y : ℝ) : Prop := x^2/9 + y^2/10 = 1

/-- F is the upper focus of ellipse C -/
def F : ℝ × ℝ := (0, 1)

/-- A is the right vertex of ellipse C -/
def A : ℝ × ℝ := (3, 0)

/-- P is a point on ellipse C located in the first quadrant -/
def P : ℝ × ℝ := sorry

/-- The area of quadrilateral OAPF -/
def area_OAPF (P : ℝ × ℝ) : ℝ := sorry

theorem max_area_OAPF :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ P.1 > 0 ∧ P.2 > 0 ∧
  ∀ (Q : ℝ × ℝ), ellipse_equation Q.1 Q.2 → Q.1 > 0 → Q.2 > 0 →
  area_OAPF P ≥ area_OAPF Q ∧
  area_OAPF P = (3 * Real.sqrt 11) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_area_OAPF_l2507_250741


namespace NUMINAMATH_CALUDE_target_hit_probability_l2507_250799

theorem target_hit_probability (p_a p_b p_c : ℚ) 
  (h_a : p_a = 1/2) 
  (h_b : p_b = 1/3) 
  (h_c : p_c = 1/4) : 
  1 - (1 - p_a) * (1 - p_b) * (1 - p_c) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2507_250799


namespace NUMINAMATH_CALUDE_f_monotonicity_l2507_250708

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - 1 / x

theorem f_monotonicity (k : ℝ) :
  (∀ x > 0, HasDerivAt (f k) ((k * x + 1) / (x^2)) x) →
  (k ≥ 0 → ∀ x > 0, (k * x + 1) / (x^2) > 0) ∧
  (k < 0 → (∀ x, 0 < x ∧ x < -1/k → (k * x + 1) / (x^2) > 0) ∧
           (∀ x > -1/k, (k * x + 1) / (x^2) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_l2507_250708


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2507_250720

theorem sum_of_decimals : 
  (5.76 : ℝ) + (4.29 : ℝ) = (10.05 : ℝ) := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2507_250720


namespace NUMINAMATH_CALUDE_spinach_not_music_lover_l2507_250784

-- Define the universe
variable (U : Type)

-- Define predicates
variable (S : U → Prop)  -- x likes spinach
variable (G : U → Prop)  -- x is a pearl diver
variable (Z : U → Prop)  -- x is a music lover

-- State the theorem
theorem spinach_not_music_lover 
  (h1 : ∃ x, S x ∧ ¬G x)
  (h2 : ∀ x, Z x → (G x ∨ ¬S x))
  (h3 : (∀ x, ¬G x → Z x) ∨ (∀ x, G x → ¬Z x))
  : ∀ x, S x → ¬Z x :=
by sorry

end NUMINAMATH_CALUDE_spinach_not_music_lover_l2507_250784


namespace NUMINAMATH_CALUDE_larger_number_proof_l2507_250705

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 40) (diff_eq : x - y = 10) :
  max x y = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2507_250705


namespace NUMINAMATH_CALUDE_bisector_line_equation_chord_length_at_pi_over_4_l2507_250712

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 24 + y^2 / 12 = 1

-- Define point M
def M : ℝ × ℝ := (3, 1)

-- Define a line passing through M
def line_through_M (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points of a line with the ellipse
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_through_M m p.1 p.2}

-- Part I: Equation of line when M bisects AB
theorem bisector_line_equation :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    line_through_M (-3/2) M.1 M.2 :=
sorry

-- Part II: Length of AB when angle of inclination is π/4
theorem chord_length_at_pi_over_4 :
  ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points 1 ∧
    B ∈ intersection_points 1 →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 16/(3*2^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_bisector_line_equation_chord_length_at_pi_over_4_l2507_250712


namespace NUMINAMATH_CALUDE_problem_statement_l2507_250754

theorem problem_statement (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + c + 1 = 0) 
  (h3 : a = 1) : 
  b^2 - 4*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2507_250754


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_today_l2507_250762

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final total number of trees -/
def final_total_trees : ℕ := 100

theorem dogwood_trees_planted_today :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_total_trees :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_today_l2507_250762


namespace NUMINAMATH_CALUDE_candy_ratio_l2507_250785

theorem candy_ratio (kitkat : ℕ) (nerds : ℕ) (lollipops : ℕ) (babyruths : ℕ) (reeses : ℕ) (remaining : ℕ) :
  kitkat = 5 →
  nerds = 8 →
  lollipops = 11 →
  babyruths = 10 →
  reeses = babyruths / 2 →
  remaining = 49 →
  ∃ (hershey : ℕ),
    hershey + kitkat + nerds + (lollipops - 5) + babyruths + reeses = remaining ∧
    hershey / kitkat = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l2507_250785


namespace NUMINAMATH_CALUDE_second_digit_of_n_l2507_250704

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n ∧ 8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102 ∧ 81*n - 102 < 10^102) →
  ∃ k : ℕ, 12 * 10^97 ≤ n ∧ n < 13 * 10^97 ∧ n = 2 * 10^97 + k :=
by sorry

end NUMINAMATH_CALUDE_second_digit_of_n_l2507_250704


namespace NUMINAMATH_CALUDE_multiplication_exercise_l2507_250765

theorem multiplication_exercise (a b : ℕ+) 
  (h1 : (a + 6) * b = 255)  -- Units digit changed from 1 to 7
  (h2 : (a - 10) * b = 335) -- Tens digit changed from 6 to 5
  : a * b = 285 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_exercise_l2507_250765


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l2507_250726

/-- 
Theorem: The smallest value of n for which 5x^2 + nx + 60 can be factored 
as the product of two linear factors with integer coefficients is 56.
-/
theorem smallest_n_for_factorization : 
  (∃ n : ℤ, ∀ m : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d))) ∧
  (∀ n : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d)) 
    → n = 56) :=
sorry


end NUMINAMATH_CALUDE_smallest_n_for_factorization_l2507_250726


namespace NUMINAMATH_CALUDE_height_inscribed_circle_inequality_l2507_250756

/-- For a right triangle, the height dropped to the hypotenuse is at most (1 + √2) times the radius of the inscribed circle. -/
theorem height_inscribed_circle_inequality (a b c h r : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  h = (a * b) / c →  -- Definition of height
  r = (a + b - c) / 2 →  -- Definition of inscribed circle radius
  h ≤ r * (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_height_inscribed_circle_inequality_l2507_250756


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2507_250732

/-- The angle between two vectors in R² -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-1, 2)) :
  angle (a + b) a = π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2507_250732


namespace NUMINAMATH_CALUDE_cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l2507_250795

-- Problem 1
theorem cos_squared_minus_three_sin_cos (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  let P : ℝ × ℝ := (m, 3 * m)
  (Real.cos α)^2 - 3 * (Real.sin α) * (Real.cos α) = -4/5 := by sorry

-- Problem 2
theorem angle_in_second_quadrant (θ : ℝ) (a : ℝ) 
  (h1 : Real.sin θ = (1 - a) / (1 + a))
  (h2 : Real.cos θ = (3 * a - 1) / (1 + a))
  (h3 : 0 < Real.sin θ ∧ Real.cos θ < 0) :
  a = 1/9 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l2507_250795


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2507_250767

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (∀ x ∈ Set.Ioo 2 3, f b c x ≤ 1) →  -- Maximum value of 1 in (2,3]
  (∃ x ∈ Set.Ioo 2 3, f b c x = 1) →  -- Maximum value of 1 is achieved in (2,3]
  (∀ x : ℝ, abs x > 2 → f b c x > 0) →  -- f(x) > 0 when |x| > 2
  (c = 4 → b = -4) ∧  -- Part 1: When c = 4, b = -4
  (Set.Icc (-34/7) (-15/4) = {x | ∃ b c : ℝ, b + 1/c = x}) -- Part 2: Range of b + 1/c
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2507_250767


namespace NUMINAMATH_CALUDE_max_discarded_grapes_l2507_250718

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_l2507_250718


namespace NUMINAMATH_CALUDE_revenue_change_l2507_250727

theorem revenue_change 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : price_increase = 0.75) 
  (h2 : quantity_decrease = 0.45) : 
  let new_price := original_price * (1 + price_increase)
  let new_quantity := original_quantity * (1 - quantity_decrease)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = -0.0375 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l2507_250727


namespace NUMINAMATH_CALUDE_triangle_side_length_l2507_250716

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  (B = Real.pi / 3) →
  (a^2 + c^2 = 3 * a * c) →
  (b = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2507_250716


namespace NUMINAMATH_CALUDE_economy_class_seats_count_l2507_250757

/-- Represents the seating configuration of an airplane -/
structure AirplaneSeating where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ
  first_class_occupied : ℕ
  business_class_occupied : ℕ

/-- Theorem stating the number of economy class seats in the given airplane configuration -/
theorem economy_class_seats_count (a : AirplaneSeating) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.business_class_seats = 30)
  (h3 : a.first_class_occupied = 3)
  (h4 : a.business_class_occupied = 22)
  (h5 : a.first_class_occupied + a.business_class_occupied = a.economy_class_seats / 2)
  : a.economy_class_seats = 50 := by
  sorry

#check economy_class_seats_count

end NUMINAMATH_CALUDE_economy_class_seats_count_l2507_250757


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2507_250777

/-- Calculates the time taken for a monkey to climb a tree given the tree height,
    distance hopped up per hour, and distance slipped back per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let net_distance := hop_distance - slip_distance
  let full_climb_distance := tree_height - hop_distance
  full_climb_distance / net_distance + 1

theorem monkey_climb_theorem :
  monkey_climb_time 19 3 2 = 17 :=
sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2507_250777


namespace NUMINAMATH_CALUDE_clock_right_angle_time_l2507_250770

/-- The time (in minutes) between two consecutive instances of the clock hands forming a right angle after 7 PM -/
def time_between_right_angles : ℚ := 360 / 11

/-- The angle (in degrees) that the minute hand moves relative to the hour hand between two consecutive right angle formations -/
def relative_angle_change : ℚ := 180

theorem clock_right_angle_time :
  time_between_right_angles = 360 / 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_right_angle_time_l2507_250770


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2507_250740

theorem absolute_value_equation_solutions (m n k : ℝ) : 
  (∀ x : ℝ, |2*x - 3| + m ≠ 0) →
  (∃! x : ℝ, |3*x - 4| + n = 0) →
  (∃ x y : ℝ, x ≠ y ∧ |4*x - 5| + k = 0 ∧ |4*y - 5| + k = 0) →
  m > n ∧ n > k :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2507_250740


namespace NUMINAMATH_CALUDE_floor_width_is_twenty_l2507_250709

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem floor_width_is_twenty
  (floor : FloorWithRug)
  (h1 : floor.length = 25)
  (h2 : floor.strip_width = 4)
  (h3 : floor.rug_area = 204)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.width = 20 := by
  sorry

#check floor_width_is_twenty

end NUMINAMATH_CALUDE_floor_width_is_twenty_l2507_250709


namespace NUMINAMATH_CALUDE_stratified_sampling_and_probability_l2507_250763

-- Define the total number of students
def total_students : ℕ := 350

-- Define the number of students excellent in Chinese
def excellent_chinese : ℕ := 200

-- Define the number of students excellent in English
def excellent_english : ℕ := 150

-- Define the probability of being excellent in both subjects
def prob_both_excellent : ℚ := 1 / 6

-- Define the number of students selected for the sample
def sample_size : ℕ := 6

-- Define the function to calculate the number of students in each category
def calculate_category_sizes : ℕ × ℕ × ℕ := sorry

-- Define the function to calculate the probability of selecting two students with excellent Chinese scores
def calculate_probability : ℚ := sorry

-- Theorem statement
theorem stratified_sampling_and_probability :
  let (a, b, c) := calculate_category_sizes
  (a = 3 ∧ b = 2 ∧ c = 1) ∧ calculate_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_and_probability_l2507_250763


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l2507_250735

theorem product_remainder_mod_five : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l2507_250735


namespace NUMINAMATH_CALUDE_power_three_mod_thirteen_l2507_250714

theorem power_three_mod_thirteen : 3^21 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_thirteen_l2507_250714


namespace NUMINAMATH_CALUDE_community_average_age_l2507_250774

/-- Given a community with a ratio of women to men of 13:10, where the average age of women
    is 36 years and the average age of men is 31 years, prove that the average age of the
    community is 33 19/23 years. -/
theorem community_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 13 / 10)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 31) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 33 + 19 / 23 :=
by sorry

end NUMINAMATH_CALUDE_community_average_age_l2507_250774


namespace NUMINAMATH_CALUDE_algebra_test_average_l2507_250738

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 32 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 82 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2507_250738


namespace NUMINAMATH_CALUDE_mod_equiv_unique_solution_l2507_250764

theorem mod_equiv_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2357 ≡ n [ZMOD 9] :=
by sorry

end NUMINAMATH_CALUDE_mod_equiv_unique_solution_l2507_250764
