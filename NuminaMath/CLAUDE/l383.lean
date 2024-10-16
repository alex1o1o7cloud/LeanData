import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l383_38351

theorem inequality_and_equality_conditions 
  (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l383_38351


namespace NUMINAMATH_CALUDE_fish_crate_weight_l383_38375

theorem fish_crate_weight (total_weight : ℝ) (cost_per_crate : ℝ) (total_cost : ℝ)
  (h1 : total_weight = 540)
  (h2 : cost_per_crate = 1.5)
  (h3 : total_cost = 27) :
  total_weight / (total_cost / cost_per_crate) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_crate_weight_l383_38375


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l383_38350

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l383_38350


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l383_38343

/-- Given a parabola with equation x^2 = ay and directrix y = -2, prove that a = 8 -/
theorem parabola_directrix_parameter (x y a : ℝ) : 
  (∀ x y, x^2 = a * y) →  -- Parabola equation
  (∃ p, p = -2 ∧ ∀ x, x^2 = a * p) →  -- Directrix equation
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l383_38343


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_factor_of_1000_l383_38396

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_factor_of_1000 :
  ∃ n : ℕ, is_mersenne_prime n ∧ n < 500 ∧ n ∣ 1000 ∧
  ∀ m : ℕ, is_mersenne_prime m ∧ m < 500 ∧ m ∣ 1000 → m ≤ n :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_factor_of_1000_l383_38396


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l383_38308

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  if city.num_streets ≤ 1 then 0
  else (city.num_streets - 1) * (city.num_streets - 2) / 2

/-- Theorem: A city with 10 straight streets where no two are parallel has 45 intersections -/
theorem ten_streets_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel_streets = true →
  max_intersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l383_38308


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l383_38345

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 40) (h3 : x - y = 10) :
  (7 : ℝ) * (375 / 7) = k := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l383_38345


namespace NUMINAMATH_CALUDE_samson_sandwiches_l383_38339

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

theorem samson_sandwiches (monday_lunch : ℕ) (monday_dinner : ℕ) (monday_total : ℕ) :
  monday_lunch = 3 →
  monday_dinner = 2 * monday_lunch →
  monday_total = monday_lunch + monday_dinner →
  monday_total = tuesday_breakfast + 8 →
  tuesday_breakfast = 1 := by sorry

end NUMINAMATH_CALUDE_samson_sandwiches_l383_38339


namespace NUMINAMATH_CALUDE_monotonic_interval_implies_a_bound_l383_38398

open Real

theorem monotonic_interval_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, (fun x => 1/x + 2*a*x - 2) > 0) →
  a > -1/2 := by
sorry

end NUMINAMATH_CALUDE_monotonic_interval_implies_a_bound_l383_38398


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l383_38304

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The set of three-digit numbers -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_7_heavy : 
  ∃ (n : ℕ), n ∈ three_digit_numbers ∧ is_7_heavy n ∧ 
  ∀ (m : ℕ), m ∈ three_digit_numbers → is_7_heavy m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l383_38304


namespace NUMINAMATH_CALUDE_binomial_150_150_l383_38369

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l383_38369


namespace NUMINAMATH_CALUDE_m_range_l383_38347

-- Define the set M
def M : Set ℝ := {x | 3 * x^2 - 5 * x - 2 ≤ 0}

-- Define the set N
def N (m : ℝ) : Set ℝ := {m, m + 1}

-- Theorem statement
theorem m_range (m : ℝ) :
  M ∪ N m = M → m ∈ Set.Icc (-1/3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l383_38347


namespace NUMINAMATH_CALUDE_license_plate_count_l383_38358

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 6

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of positions where the two-letter word can be placed -/
def word_positions : ℕ := total_chars - num_plate_letters + 1

/-- The number of positions for the fixed digit 7 -/
def fixed_digit_positions : ℕ := total_chars - 1

theorem license_plate_count :
  (fixed_digit_positions) * (num_letters ^ num_plate_letters) * (num_digits ^ (num_plate_digits - 1)) = 47320000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l383_38358


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l383_38397

theorem geometric_to_arithmetic_sequence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q) ∧
  a + b + c = 19 ∧
  b - a = (c - 1) - b :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l383_38397


namespace NUMINAMATH_CALUDE_even_function_property_l383_38316

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_prop : ∀ x, f (x + 2) = x * f x) : 
  f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_property_l383_38316


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l383_38353

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ / a₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l383_38353


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l383_38327

theorem two_digit_integer_problem (n : ℕ) :
  n ≥ 10 ∧ n ≤ 99 →
  (60 + n) / 2 = 60 + n / 100 →
  min 60 n = 59 := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l383_38327


namespace NUMINAMATH_CALUDE_ratio_of_recurring_decimals_l383_38357

/-- The value of the repeating decimal 0.848484... -/
def recurring_84 : ℚ := 84 / 99

/-- The value of the repeating decimal 0.212121... -/
def recurring_21 : ℚ := 21 / 99

/-- Theorem stating that the ratio of the two repeating decimals is equal to 4 -/
theorem ratio_of_recurring_decimals : recurring_84 / recurring_21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_recurring_decimals_l383_38357


namespace NUMINAMATH_CALUDE_tangent_unique_tangent_values_l383_38338

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ k : ℝ,
  (1 : ℝ)^3 + a * 1 + b = 3 ∧  -- The point (1, 3) is on the curve
  k * 1 + 1 = 3 ∧              -- The point (1, 3) is on the line
  3 * (1 : ℝ)^2 + a = k        -- The slope of the curve at x = 1 equals the slope of the line

/-- The values of a and b for which the line is tangent to the curve at (1, 3) are unique -/
theorem tangent_unique : ∃! (a b : ℝ), is_tangent a b :=
sorry

/-- The unique values of a and b for which the line is tangent to the curve at (1, 3) are -1 and 3 respectively -/
theorem tangent_values : ∃! (a b : ℝ), is_tangent a b ∧ a = -1 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_unique_tangent_values_l383_38338


namespace NUMINAMATH_CALUDE_two_envelopes_require_fee_l383_38377

-- Define the envelope structure
structure Envelope where
  name : String
  length : ℚ
  height : ℚ

-- Define the condition for additional fee
def requiresAdditionalFee (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.5 || ratio > 2.8

-- Define the list of envelopes
def envelopes : List Envelope := [
  ⟨"E", 7, 5⟩,
  ⟨"F", 10, 4⟩,
  ⟨"G", 5, 5⟩,
  ⟨"H", 14, 5⟩
]

-- Theorem statement
theorem two_envelopes_require_fee :
  (envelopes.filter requiresAdditionalFee).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_envelopes_require_fee_l383_38377


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l383_38384

theorem max_value_sum_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ((x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 ↔ x = y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l383_38384


namespace NUMINAMATH_CALUDE_bowl_water_problem_l383_38334

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) :
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end NUMINAMATH_CALUDE_bowl_water_problem_l383_38334


namespace NUMINAMATH_CALUDE_complex_cube_sum_div_product_l383_38317

theorem complex_cube_sum_div_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_div_product_l383_38317


namespace NUMINAMATH_CALUDE_julie_monthly_salary_l383_38399

/-- Calculates the monthly salary for a worker given their hourly rate, hours per day,
    days per week, and number of missed days in a month. -/
def monthly_salary (hourly_rate : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (missed_days : ℕ) : ℚ :=
  let daily_earnings := hourly_rate * hours_per_day
  let weekly_earnings := daily_earnings * days_per_week
  let monthly_earnings := weekly_earnings * 4
  monthly_earnings - (daily_earnings * missed_days)

/-- Proves that Julie's monthly salary after missing a day of work is $920. -/
theorem julie_monthly_salary :
  monthly_salary 5 8 6 1 = 920 := by
  sorry

end NUMINAMATH_CALUDE_julie_monthly_salary_l383_38399


namespace NUMINAMATH_CALUDE_midpoint_fraction_l383_38393

theorem midpoint_fraction : 
  let a := 3/4
  let b := 5/6
  (a + b) / 2 = 19/24 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l383_38393


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l383_38301

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating that one cubic foot is equal to 1728 cubic inches -/
theorem cubic_foot_to_cubic_inches : (1 : ℝ)^3 * feet_to_inches^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l383_38301


namespace NUMINAMATH_CALUDE_sara_golf_balls_l383_38374

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Sara has -/
def sara_dozens : ℕ := 16

/-- The total number of golf balls Sara has -/
def sara_total : ℕ := sara_dozens * dozen

theorem sara_golf_balls : sara_total = 192 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l383_38374


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l383_38319

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n ≥ 3) → ((n - 2) * 180 = 900) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l383_38319


namespace NUMINAMATH_CALUDE_balloon_distribution_l383_38386

theorem balloon_distribution (total_balloons : ℕ) (friends : ℕ) 
  (h1 : total_balloons = 235) (h2 : friends = 10) : 
  total_balloons % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l383_38386


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l383_38321

theorem hyperbola_m_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m - y^2 / (2*m - 1) = 1) → 
  (0 < m ∧ m < 1/2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l383_38321


namespace NUMINAMATH_CALUDE_floor_of_neg_two_point_seven_l383_38359

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_of_neg_two_point_seven :
  floor (-2.7) = -3 := by sorry

end NUMINAMATH_CALUDE_floor_of_neg_two_point_seven_l383_38359


namespace NUMINAMATH_CALUDE_two_digit_number_between_30_and_40_with_units_digit_2_l383_38387

theorem two_digit_number_between_30_and_40_with_units_digit_2 (n : ℕ) :
  (n ≥ 30 ∧ n < 40) →  -- two-digit number between 30 and 40
  (n % 10 = 2) →       -- units digit is 2
  n = 32 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_between_30_and_40_with_units_digit_2_l383_38387


namespace NUMINAMATH_CALUDE_max_value_fraction_l383_38320

theorem max_value_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z : ℚ) ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l383_38320


namespace NUMINAMATH_CALUDE_min_value_expression_l383_38324

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 12) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 13 ∧
  ∃ y : ℝ, y > 1 ∧ (y + 12) / Real.sqrt (y - 1) = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l383_38324


namespace NUMINAMATH_CALUDE_range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l383_38390

-- Define the propositions
def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- State the theorems
theorem range_of_m_when_S_true :
  ∀ m : ℝ, S m → m < 0 ∨ m ≥ 1 :=
sorry

theorem range_of_m_when_p_or_q_and_not_q :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l383_38390


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l383_38313

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 18 * x - 90 = x * (x + 10)) → x ≥ -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l383_38313


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l383_38355

/-- Given a cone whose lateral surface area is equal to the area of a semicircle with area 2π,
    prove that the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (cone : Real → Real → Real) 
  (lateral_surface_area : Real) (semicircle_area : Real) :
  lateral_surface_area = semicircle_area →
  semicircle_area = 2 * Real.pi →
  (∃ (r h : Real), cone r h = (1/3) * Real.pi * r^2 * h ∧ 
                   cone r h = (Real.sqrt 3 / 3) * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l383_38355


namespace NUMINAMATH_CALUDE_salt_solution_problem_l383_38346

/-- Proves the initial water mass and percentage increase given final conditions --/
theorem salt_solution_problem (final_mass : ℝ) (final_concentration : ℝ) 
  (h_final_mass : final_mass = 850)
  (h_final_concentration : final_concentration = 0.36) : 
  ∃ (initial_mass : ℝ) (percentage_increase : ℝ),
    initial_mass = 544 ∧ 
    percentage_increase = 25 ∧
    final_mass = initial_mass * (1 + percentage_increase / 100)^2 ∧
    final_concentration = 1 - (initial_mass / final_mass) :=
by
  sorry


end NUMINAMATH_CALUDE_salt_solution_problem_l383_38346


namespace NUMINAMATH_CALUDE_gcd_228_1995_l383_38326

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l383_38326


namespace NUMINAMATH_CALUDE_least_value_of_x_l383_38389

theorem least_value_of_x (x p : ℕ) : 
  x > 0 → 
  Nat.Prime p → 
  ∃ q, Nat.Prime q ∧ q % 2 = 1 ∧ x / (9 * p) = q →
  x ≥ 81 :=
sorry

end NUMINAMATH_CALUDE_least_value_of_x_l383_38389


namespace NUMINAMATH_CALUDE_sock_pair_selection_l383_38337

def total_socks : ℕ := 20
def white_socks : ℕ := 6
def brown_socks : ℕ := 7
def blue_socks : ℕ := 3
def red_socks : ℕ := 4

theorem sock_pair_selection :
  (Nat.choose white_socks 2) +
  (Nat.choose brown_socks 2) +
  (Nat.choose blue_socks 2) +
  (red_socks * white_socks) +
  (red_socks * brown_socks) +
  (red_socks * blue_socks) +
  (Nat.choose red_socks 2) = 109 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_selection_l383_38337


namespace NUMINAMATH_CALUDE_is_min_point_l383_38330

/-- The function representing the translated graph -/
def f (x : ℝ) : ℝ := |x - 4| - 3

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -3)

/-- Theorem stating that min_point is the minimum of the function f -/
theorem is_min_point :
  ∀ x : ℝ, f x ≥ f min_point.fst ∧ f min_point.fst = min_point.snd := by
  sorry

end NUMINAMATH_CALUDE_is_min_point_l383_38330


namespace NUMINAMATH_CALUDE_line_y_intercept_l383_38349

/-- A straight line in the xy-plane passing through (100, 1000) with slope 9.9 has y-intercept 10 -/
theorem line_y_intercept :
  ∀ (f : ℝ → ℝ),
  (∀ x y, f x = 9.9 * x + y) →  -- Line equation with slope 9.9 and y-intercept y
  f 100 = 1000 →               -- Line passes through (100, 1000)
  f 0 = 10 :=                  -- y-intercept is 10
by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l383_38349


namespace NUMINAMATH_CALUDE_xiaoming_pencil_theorem_l383_38303

/-- Represents the number of pencils and amount spent in Xiaoming's purchases -/
structure PencilPurchase where
  x : ℕ  -- number of pencils in first purchase
  y : ℕ  -- amount spent in first purchase in yuan

/-- Determines if a PencilPurchase satisfies the problem conditions -/
def satisfiesConditions (p : PencilPurchase) : Prop :=
  ∃ (price : ℚ), 
    price = p.y / p.x ∧  -- initial price per pencil
    (4 : ℚ) / 5 * price * (p.x + 10) = 4  -- condition after price drop

/-- The theorem stating the possible total numbers of pencils bought -/
theorem xiaoming_pencil_theorem (p : PencilPurchase) :
  satisfiesConditions p → (p.x + (p.x + 10) = 40 ∨ p.x + (p.x + 10) = 90) :=
by
  sorry

#check xiaoming_pencil_theorem

end NUMINAMATH_CALUDE_xiaoming_pencil_theorem_l383_38303


namespace NUMINAMATH_CALUDE_claire_earnings_l383_38372

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def price_per_red_rose : ℚ := 3/4

def total_roses : ℕ := total_flowers - tulips
def red_roses : ℕ := total_roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings :
  (red_roses_to_sell : ℚ) * price_per_red_rose = 75 := by
  sorry

end NUMINAMATH_CALUDE_claire_earnings_l383_38372


namespace NUMINAMATH_CALUDE_total_age_difference_l383_38302

-- Define the ages of A, B, and C as natural numbers
variable (A B C : ℕ)

-- Define the condition that C is 15 years younger than A
def age_difference : Prop := C = A - 15

-- Define the difference in total ages
def age_sum_difference : ℕ := (A + B) - (B + C)

-- Theorem statement
theorem total_age_difference (h : age_difference A C) : age_sum_difference A B C = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_age_difference_l383_38302


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_l383_38342

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
theorem prob_sum_greater_than_four :
  (1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_l383_38342


namespace NUMINAMATH_CALUDE_novels_difference_l383_38388

def jordan_novels : ℕ := 120

def alexandre_novels : ℕ := jordan_novels / 10

theorem novels_difference : jordan_novels - alexandre_novels = 108 := by
  sorry

end NUMINAMATH_CALUDE_novels_difference_l383_38388


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_l383_38335

theorem abs_sum_inequality_solution_set :
  {x : ℝ | |x - 1| + |x| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_l383_38335


namespace NUMINAMATH_CALUDE_jeff_initial_pencils_l383_38364

theorem jeff_initial_pencils (J : ℝ) : 
  J > 0 →
  (0.7 * J + 0.25 * (2 * J) = 360) →
  J = 300 := by
sorry

end NUMINAMATH_CALUDE_jeff_initial_pencils_l383_38364


namespace NUMINAMATH_CALUDE_mary_milk_weight_l383_38360

/-- Proves that the weight of milk Mary bought is 6 pounds -/
theorem mary_milk_weight (bag_capacity : ℕ) (green_beans_weight : ℕ) (remaining_capacity : ℕ) : 
  bag_capacity = 20 →
  green_beans_weight = 4 →
  remaining_capacity = 2 →
  6 = bag_capacity - remaining_capacity - (green_beans_weight + 2 * green_beans_weight) :=
by sorry

end NUMINAMATH_CALUDE_mary_milk_weight_l383_38360


namespace NUMINAMATH_CALUDE_original_cost_price_l383_38382

/-- Given an article with a 15% markup and 20% discount, 
    prove that the original cost is 540 when sold at 496.80 --/
theorem original_cost_price (marked_up_price : ℝ) (selling_price : ℝ) : 
  marked_up_price = 1.15 * 540 ∧ 
  selling_price = 0.8 * marked_up_price ∧
  selling_price = 496.80 → 
  540 = (496.80 : ℝ) / 0.92 := by
sorry

#eval (496.80 : Float) / 0.92

end NUMINAMATH_CALUDE_original_cost_price_l383_38382


namespace NUMINAMATH_CALUDE_correct_calculation_l383_38362

theorem correct_calculation (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l383_38362


namespace NUMINAMATH_CALUDE_geese_ducks_difference_l383_38315

def geese : ℝ := 58.0
def ducks : ℝ := 37.0

theorem geese_ducks_difference : geese - ducks = 21.0 := by
  sorry

end NUMINAMATH_CALUDE_geese_ducks_difference_l383_38315


namespace NUMINAMATH_CALUDE_uma_income_l383_38341

theorem uma_income (uma_income bala_income uma_expenditure bala_expenditure : ℚ)
  (income_ratio : uma_income / bala_income = 4 / 3)
  (expenditure_ratio : uma_expenditure / bala_expenditure = 3 / 2)
  (uma_savings : uma_income - uma_expenditure = 5000)
  (bala_savings : bala_income - bala_expenditure = 5000) :
  uma_income = 20000 := by
  sorry

end NUMINAMATH_CALUDE_uma_income_l383_38341


namespace NUMINAMATH_CALUDE_juliet_age_l383_38309

theorem juliet_age (maggie ralph juliet : ℕ) 
  (h1 : juliet = maggie + 3)
  (h2 : juliet = ralph - 2)
  (h3 : maggie + ralph = 19) :
  juliet = 10 := by
  sorry

end NUMINAMATH_CALUDE_juliet_age_l383_38309


namespace NUMINAMATH_CALUDE_domain_intersection_l383_38365

def A : Set ℝ := {x : ℝ | x > -1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem domain_intersection : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_domain_intersection_l383_38365


namespace NUMINAMATH_CALUDE_geometry_theorem_l383_38371

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- Define the non-coincidence of lines and planes
variable (non_coincident_lines : Line → Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_non_coincident_lines : non_coincident_lines m n l)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (∀ (l m : Line) (α β : Plane),
    line_perpendicular_to_plane l α →
    line_perpendicular_to_plane m β →
    parallel l m →
    plane_parallel α β) ∧
  (∀ (m n : Line) (α β : Plane),
    plane_perpendicular α β →
    plane_intersection α β = m →
    line_in_plane n β →
    perpendicular n m →
    line_perpendicular_to_plane n α) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l383_38371


namespace NUMINAMATH_CALUDE_store_discount_income_increase_l383_38394

theorem store_discount_income_increase 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.25) 
  (h3 : original_price > 0) 
  (h4 : original_sales > 0) : 
  let new_price := original_price * (1 - discount_rate)
  let new_sales := original_sales * (1 + sales_increase_rate)
  let original_income := original_price * original_sales
  let new_income := new_price * new_sales
  (new_income - original_income) / original_income = 0.125 := by
sorry

end NUMINAMATH_CALUDE_store_discount_income_increase_l383_38394


namespace NUMINAMATH_CALUDE_simple_interest_problem_l383_38336

/-- Given a principal P and an interest rate R, if increasing the rate by 15%
    results in $300 more interest over 10 years, then P must equal $200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) : 
  (P * (R + 15) * 10 / 100 = P * R * 10 / 100 + 300) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l383_38336


namespace NUMINAMATH_CALUDE_solve_equations_l383_38361

theorem solve_equations :
  (∀ x : ℝ, (x - 2)^2 - 1 = 0 ↔ x = 3 ∨ x = 1) ∧
  (∀ x : ℝ, 3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, 2*x^2 + 4*x - 5 = 0 ↔ x = -1 + Real.sqrt 14 / 2 ∨ x = -1 - Real.sqrt 14 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l383_38361


namespace NUMINAMATH_CALUDE_edward_escape_problem_l383_38314

/-- The problem of Edward escaping from prison and being hit by an arrow. -/
theorem edward_escape_problem (initial_distance : ℝ) (arrow_initial_velocity : ℝ) 
  (edward_acceleration : ℝ) (arrow_deceleration : ℝ) :
  initial_distance = 1875 →
  arrow_initial_velocity = 100 →
  edward_acceleration = 1 →
  arrow_deceleration = 1 →
  ∃ t : ℝ, t > 0 ∧ 
    (-1/2 * arrow_deceleration * t^2 + arrow_initial_velocity * t) = 
    (1/2 * edward_acceleration * t^2 + initial_distance) ∧
    (arrow_initial_velocity - arrow_deceleration * t) = 75 :=
by sorry

end NUMINAMATH_CALUDE_edward_escape_problem_l383_38314


namespace NUMINAMATH_CALUDE_max_cross_section_area_l383_38363

/-- A right rectangular prism with a square base and varying height -/
structure Prism where
  base_length : ℝ
  height_a : ℝ
  height_b : ℝ
  height_c : ℝ
  height_d : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The cross-section area formed by the intersection of a prism and a plane -/
def cross_section_area (p : Prism) (pl : Plane) : ℝ := sorry

/-- The theorem stating that the maximal area of the cross-section is 110 -/
theorem max_cross_section_area (p : Prism) (pl : Plane) :
  p.base_length = 8 ∧
  p.height_a = 3 ∧ p.height_b = 2 ∧ p.height_c = 4 ∧ p.height_d = 1 ∧
  pl.a = 3 ∧ pl.b = -5 ∧ pl.c = 3 ∧ pl.d = 24 →
  cross_section_area p pl = 110 := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l383_38363


namespace NUMINAMATH_CALUDE_alien_mineral_conversion_l383_38300

/-- Converts a three-digit number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The base 7 number 365₇ is equal to 194 in base 10 -/
theorem alien_mineral_conversion :
  base7ToBase10 3 6 5 = 194 := by
  sorry

end NUMINAMATH_CALUDE_alien_mineral_conversion_l383_38300


namespace NUMINAMATH_CALUDE_expand_expression_l383_38376

theorem expand_expression (x y : ℝ) : (10 * x - 6 * y + 9) * (3 * y) = 30 * x * y - 18 * y^2 + 27 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l383_38376


namespace NUMINAMATH_CALUDE_product_347_6_base9_l383_38333

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Theorem: The product of 347₉ and 6₉ in base 9 is 2316₉ --/
theorem product_347_6_base9 :
  base10ToBase9 (base9ToBase10 [7, 4, 3] * base9ToBase10 [6]) = [6, 1, 3, 2] := by
  sorry

end NUMINAMATH_CALUDE_product_347_6_base9_l383_38333


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l383_38311

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l383_38311


namespace NUMINAMATH_CALUDE_lucy_bought_six_fifty_cent_items_l383_38348

/-- Represents the number of items Lucy bought at each price point -/
structure PurchasedItems where
  fifty_cent : ℕ
  one_fifty : ℕ
  three_dollar : ℕ

/-- The total number of items Lucy bought -/
def total_items : ℕ := 30

/-- The total purchase price in cents -/
def total_price : ℕ := 4500

/-- Theorem stating that Lucy bought 6 items at 50 cents -/
theorem lucy_bought_six_fifty_cent_items :
  ∃ (items : PurchasedItems),
    items.fifty_cent + items.one_fifty + items.three_dollar = total_items ∧
    50 * items.fifty_cent + 150 * items.one_fifty + 300 * items.three_dollar = total_price ∧
    items.fifty_cent = 6 := by
  sorry

end NUMINAMATH_CALUDE_lucy_bought_six_fifty_cent_items_l383_38348


namespace NUMINAMATH_CALUDE_equation_solutions_l383_38368

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧ 
  (∃ x : ℝ, (x + 3)^2 = (2*x - 1)*(x + 3) ↔ x = -3 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l383_38368


namespace NUMINAMATH_CALUDE_negation_equivalence_l383_38378

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l383_38378


namespace NUMINAMATH_CALUDE_max_rectangle_area_l383_38352

/-- The maximum area of a rectangle with perimeter 12 is 9 -/
theorem max_rectangle_area : 
  let perimeter : ℝ := 12
  let area (x : ℝ) : ℝ := x * (perimeter / 2 - x)
  ∀ x, 0 < x → x < perimeter / 2 → area x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l383_38352


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_48_l383_38325

theorem complex_expression_equals_negative_48 : 
  ((-1/2 * (1/100))^5 * (2/3 * (2/100))^4 * (-3/4 * (3/100))^3 * (4/5 * (4/100))^2 * (-5/6 * (5/100))) * (10^30) = -48 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_48_l383_38325


namespace NUMINAMATH_CALUDE_range_of_f_l383_38305

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l383_38305


namespace NUMINAMATH_CALUDE_patricia_books_l383_38373

def book_tournament (candice amanda kara patricia : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  candice = 18

theorem patricia_books :
  ∀ candice amanda kara patricia : ℕ,
    book_tournament candice amanda kara patricia →
    patricia = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_patricia_books_l383_38373


namespace NUMINAMATH_CALUDE_no_common_solutions_l383_38391

theorem no_common_solutions : 
  ¬∃ x : ℝ, (|x - 10| = |x + 3| ∧ 2 * x + 6 = 18) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solutions_l383_38391


namespace NUMINAMATH_CALUDE_det_A_equals_two_l383_38395

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, 2; -3, d] →
  A + A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l383_38395


namespace NUMINAMATH_CALUDE_selection_methods_equality_l383_38306

def num_male_students : ℕ := 20
def num_female_students : ℕ := 30
def total_students : ℕ := num_male_students + num_female_students
def num_selected : ℕ := 4

theorem selection_methods_equality :
  (Nat.choose total_students num_selected - Nat.choose num_male_students num_selected - Nat.choose num_female_students num_selected) =
  (Nat.choose num_male_students 1 * Nat.choose num_female_students 3 +
   Nat.choose num_male_students 2 * Nat.choose num_female_students 2 +
   Nat.choose num_male_students 3 * Nat.choose num_female_students 1) :=
by sorry

end NUMINAMATH_CALUDE_selection_methods_equality_l383_38306


namespace NUMINAMATH_CALUDE_sum_of_naturals_l383_38312

theorem sum_of_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_naturals_l383_38312


namespace NUMINAMATH_CALUDE_total_snow_volume_l383_38328

/-- Calculates the total volume of snow on two sidewalk sections -/
theorem total_snow_volume 
  (length1 width1 depth1 : ℝ)
  (length2 width2 depth2 : ℝ)
  (h1 : length1 = 30)
  (h2 : width1 = 3)
  (h3 : depth1 = 1)
  (h4 : length2 = 15)
  (h5 : width2 = 2)
  (h6 : depth2 = 1/2) :
  length1 * width1 * depth1 + length2 * width2 * depth2 = 105 := by
sorry

end NUMINAMATH_CALUDE_total_snow_volume_l383_38328


namespace NUMINAMATH_CALUDE_abc_remainder_mod_7_l383_38367

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 3)
  (h2 : (2*a + b + 3*c) % 7 = 2)
  (h3 : (3*a + 2*b + c) % 7 = 1) :
  (a * b * c) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_7_l383_38367


namespace NUMINAMATH_CALUDE_fifth_term_is_67_l383_38344

def sequence_condition (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) = (s n + s (n + 2)) / 3

theorem fifth_term_is_67 (s : ℕ → ℕ) :
  sequence_condition s →
  s 1 = 3 →
  s 4 = 27 →
  s 5 = 67 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_67_l383_38344


namespace NUMINAMATH_CALUDE_hyperbola_equation_l383_38322

/-- Given a hyperbola passing through the point (2√2, 1) with one asymptote equation y = 1/2x,
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k ∧ (2 * Real.sqrt 2)^2 / 4 - 1^2 = k) ∧
  (∃ m : ℝ, y = 1/2 * x + m) →
  x^2 / 4 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l383_38322


namespace NUMINAMATH_CALUDE_inheritance_calculation_l383_38340

-- Define the original inheritance amount
def original_inheritance : ℝ := 45500

-- Define the federal tax rate
def federal_tax_rate : ℝ := 0.25

-- Define the state tax rate
def state_tax_rate : ℝ := 0.15

-- Define the total tax paid
def total_tax_paid : ℝ := 16500

-- Theorem statement
theorem inheritance_calculation :
  let remaining_after_federal := original_inheritance * (1 - federal_tax_rate)
  let state_tax := remaining_after_federal * state_tax_rate
  let total_tax := original_inheritance * federal_tax_rate + state_tax
  total_tax = total_tax_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l383_38340


namespace NUMINAMATH_CALUDE_circle_equation_l383_38329

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define the line x-y-1=0
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_equation :
  (point_A ∈ circle_C) ∧
  (point_B ∈ line) ∧
  (∃ (t : ℝ), ∀ (p : ℝ × ℝ), p ∈ circle_C → (p.1 - point_B.1) * 1 + (p.2 - point_B.2) * (-1) = t * ((p.1 - point_B.1)^2 + (p.2 - point_B.2)^2)) →
  ∀ (x y : ℝ), (x, y) ∈ circle_C ↔ (x - 3)^2 + y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l383_38329


namespace NUMINAMATH_CALUDE_line_relationship_l383_38307

-- Define the concept of lines in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real implementation, 
  -- we might represent a line using a point and a direction vector.
  id : ℕ

-- Define the relationships between lines
def are_skew (l1 l2 : Line3D) : Prop := sorry
def are_parallel (l1 l2 : Line3D) : Prop := sorry
def are_intersecting (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationship (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel a c) : 
  are_intersecting b c ∨ are_skew b c := by sorry

end NUMINAMATH_CALUDE_line_relationship_l383_38307


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l383_38381

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -(1/3) * x^3 - x^2 - (2/3) * x - 3
  (q 1 = -5) ∧ (q 2 = -8) ∧ (q 3 = -17) ∧ (q 4 = -34) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l383_38381


namespace NUMINAMATH_CALUDE_term_properties_l383_38356

-- Define a structure for a monomial term
structure Monomial where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define the monomial -1/3 * x * y^2
def term : Monomial := {
  coefficient := -1/3,
  x_power := 1,
  y_power := 2
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m.coefficient

-- Define the degree of a monomial
def degree (m : Monomial) : ℕ := m.x_power + m.y_power

-- Theorem stating the coefficient and degree of the term
theorem term_properties :
  coefficient term = -1/3 ∧ degree term = 3 := by
  sorry


end NUMINAMATH_CALUDE_term_properties_l383_38356


namespace NUMINAMATH_CALUDE_lakota_spending_l383_38370

/-- The price of a new compact disk -/
def new_cd_price : ℚ := 17.99

/-- The price of a used compact disk -/
def used_cd_price : ℚ := 9.99

/-- The number of new CDs Lakota bought -/
def lakota_new_cds : ℕ := 6

/-- The number of used CDs Lakota bought -/
def lakota_used_cds : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used_cds : ℕ := 8

/-- The total amount Mackenzie spent -/
def mackenzie_total : ℚ := 133.89

theorem lakota_spending :
  (lakota_new_cds : ℚ) * new_cd_price + (lakota_used_cds : ℚ) * used_cd_price = 127.92 :=
by sorry

end NUMINAMATH_CALUDE_lakota_spending_l383_38370


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_94_l383_38379

-- Define the polynomials
def p (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 + 5 * x + 6
def q (x : ℝ) : ℝ := 7 * x^2 + 8 * x + 9

-- Theorem statement
theorem coefficient_of_x_cubed_is_94 :
  ∃ a b c d e : ℝ, p * q = (λ x => a * x^5 + b * x^4 + 94 * x^3 + c * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_94_l383_38379


namespace NUMINAMATH_CALUDE_smaller_number_of_sum_and_product_l383_38332

theorem smaller_number_of_sum_and_product (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_of_sum_and_product_l383_38332


namespace NUMINAMATH_CALUDE_triangle_side_c_l383_38392

theorem triangle_side_c (a b c : ℝ) (S : ℝ) (B : ℝ) :
  B = π / 4 →  -- 45° in radians
  a = 4 →
  S = 16 * Real.sqrt 2 →
  S = 1 / 2 * a * c * Real.sin B →
  c = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_c_l383_38392


namespace NUMINAMATH_CALUDE_f_sum_equals_six_l383_38385

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_six_l383_38385


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l383_38331

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = c and C = π/5, then B = 3π/10 -/
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l383_38331


namespace NUMINAMATH_CALUDE_phone_rep_work_hours_l383_38354

theorem phone_rep_work_hours 
  (num_reps : ℕ) 
  (num_days : ℕ) 
  (hourly_rate : ℚ) 
  (total_pay : ℚ) 
  (h1 : num_reps = 50)
  (h2 : num_days = 5)
  (h3 : hourly_rate = 14)
  (h4 : total_pay = 28000) :
  (total_pay / hourly_rate) / (num_reps * num_days) = 8 := by
sorry

end NUMINAMATH_CALUDE_phone_rep_work_hours_l383_38354


namespace NUMINAMATH_CALUDE_polygonal_number_formula_l383_38366

def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (n^2 + n) / 2
  | 4 => n^2
  | 5 => (3*n^2 - n) / 2
  | 6 => 2*n^2 - n
  | _ => 0

theorem polygonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n :=
by sorry

end NUMINAMATH_CALUDE_polygonal_number_formula_l383_38366


namespace NUMINAMATH_CALUDE_circle_symmetry_minimum_l383_38318

/-- Given a circle x^2 + y^2 + 2x - 4y + 1 = 0 symmetric with respect to the line 2ax - by + 2 = 0,
    where a > 0 and b > 0, the minimum value of 4/a + 1/b is 9. -/
theorem circle_symmetry_minimum (a b : ℝ) : a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  (∀ t : ℝ, 4/a + 1/b ≥ t) →
  t = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_minimum_l383_38318


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l383_38380

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^504 * k = 14^504 - 8^252 ∧ 
  ∀ (m : ℕ), 2^m * k = 14^504 - 8^252 → m ≤ 504 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l383_38380


namespace NUMINAMATH_CALUDE_truncated_cube_volume_ratio_l383_38383

/-- A convex polyhedron with specific properties -/
structure TruncatedCube where
  /-- The polyhedron has 6 square faces -/
  square_faces : Nat
  /-- The polyhedron has 8 equilateral triangle faces -/
  triangle_faces : Nat
  /-- Each edge is shared between one triangle and one square -/
  shared_edges : Bool
  /-- All dihedral angles between triangles and squares are equal -/
  equal_dihedral_angles : Bool
  /-- The polyhedron can be circumscribed by a sphere -/
  circumscribable : Bool
  /-- Properties of the truncated cube -/
  h_square_faces : square_faces = 6
  h_triangle_faces : triangle_faces = 8
  h_shared_edges : shared_edges = true
  h_equal_dihedral_angles : equal_dihedral_angles = true
  h_circumscribable : circumscribable = true

/-- The theorem stating the ratio of squared volumes -/
theorem truncated_cube_volume_ratio (tc : TruncatedCube) :
  ∃ (v_polyhedron v_sphere : ℝ),
    v_polyhedron > 0 ∧ v_sphere > 0 ∧
    (v_polyhedron / v_sphere)^2 = 25 / (8 * Real.pi^2) :=
sorry

end NUMINAMATH_CALUDE_truncated_cube_volume_ratio_l383_38383


namespace NUMINAMATH_CALUDE_largest_solution_sum_l383_38323

noncomputable def f (x : ℝ) : ℝ := 
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20)

theorem largest_solution_sum (n : ℝ) (p q r : ℕ+) :
  (∀ x : ℝ, f x = x^2 - 13*x - 6 → x ≤ n) ∧
  f n = n^2 - 13*n - 6 ∧
  n = p + Real.sqrt (q + Real.sqrt r) →
  p + q + r = 309 := by sorry

end NUMINAMATH_CALUDE_largest_solution_sum_l383_38323


namespace NUMINAMATH_CALUDE_u_value_l383_38310

/-- A line passing through points (2, 8), (4, 14), (6, 20), and (18, u) -/
structure Line where
  -- Define the slope of the line
  slope : ℝ
  -- Define the y-intercept of the line
  intercept : ℝ
  -- Ensure the line passes through (2, 8)
  point1 : 8 = slope * 2 + intercept
  -- Ensure the line passes through (4, 14)
  point2 : 14 = slope * 4 + intercept
  -- Ensure the line passes through (6, 20)
  point3 : 20 = slope * 6 + intercept

/-- The u-coordinate of the point (18, u) on the line -/
def u (l : Line) : ℝ := l.slope * 18 + l.intercept

/-- Theorem stating that u = 56 for the given line -/
theorem u_value (l : Line) : u l = 56 := by
  sorry

end NUMINAMATH_CALUDE_u_value_l383_38310
