import Mathlib

namespace NUMINAMATH_CALUDE_ascent_speed_l312_31209

/-- 
Given a round trip journey with:
- Total time of 8 hours
- Ascent time of 5 hours
- Descent time of 3 hours
- Average speed for the entire journey of 3 km/h
Prove that the average speed during the ascent is 2.4 km/h
-/
theorem ascent_speed (total_time : ℝ) (ascent_time : ℝ) (descent_time : ℝ) (avg_speed : ℝ) :
  total_time = 8 →
  ascent_time = 5 →
  descent_time = 3 →
  avg_speed = 3 →
  (avg_speed * total_time / 2) / ascent_time = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ascent_speed_l312_31209


namespace NUMINAMATH_CALUDE_even_sine_function_l312_31249

theorem even_sine_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (∀ x, f (-x) = f x) →
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_even_sine_function_l312_31249


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_2_5_7_l312_31287

theorem least_three_digit_multiple_of_2_5_7 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 140 → ¬(2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_2_5_7_l312_31287


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l312_31220

theorem circular_seating_arrangement (n : ℕ) (h1 : n ≤ 6) (h2 : Nat.factorial (n - 1) = 144) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l312_31220


namespace NUMINAMATH_CALUDE_line_parameterization_l312_31263

/-- Given a line y = 3x + 2 parameterized as (x, y) = (5, r) + t(m, 6),
    prove that r = 17 and m = 2 -/
theorem line_parameterization (r m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, 
    (x = 5 + t * m ∧ y = r + t * 6) → y = 3 * x + 2) →
  r = 17 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l312_31263


namespace NUMINAMATH_CALUDE_two_cones_cost_l312_31280

/-- The cost of a single ice cream cone in cents -/
def single_cone_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_cones_cost : single_cone_cost * num_cones = 198 := by
  sorry

end NUMINAMATH_CALUDE_two_cones_cost_l312_31280


namespace NUMINAMATH_CALUDE_equation_solution_l312_31203

theorem equation_solution :
  ∀ x : ℚ,
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l312_31203


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l312_31251

theorem unique_solution_floor_equation :
  ∃! b : ℝ, b + ⌊b⌋ = 22.6 ∧ b = 11.6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l312_31251


namespace NUMINAMATH_CALUDE_cosine_sum_equals_one_l312_31239

theorem cosine_sum_equals_one (α β γ : Real) 
  (sum_eq_pi : α + β + γ = Real.pi)
  (tan_sum_eq_one : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_equals_one_l312_31239


namespace NUMINAMATH_CALUDE_super_lucky_years_l312_31276

def is_super_lucky_year (Y : ℕ) : Prop :=
  ∃ (m d : ℕ), 
    1 ≤ m ∧ m ≤ 12 ∧
    1 ≤ d ∧ d ≤ 31 ∧
    m + d = 24 ∧
    m * d = 2 * ((Y % 100) / 10 + Y % 10)

theorem super_lucky_years : 
  is_super_lucky_year 2076 ∧ 
  is_super_lucky_year 2084 ∧ 
  ¬is_super_lucky_year 2070 ∧ 
  ¬is_super_lucky_year 2081 ∧ 
  ¬is_super_lucky_year 2092 :=
sorry

end NUMINAMATH_CALUDE_super_lucky_years_l312_31276


namespace NUMINAMATH_CALUDE_x_power_8000_minus_inverse_l312_31267

theorem x_power_8000_minus_inverse (x : ℂ) : 
  x - 1/x = 2*Complex.I → x^8000 - 1/x^8000 = 0 := by sorry

end NUMINAMATH_CALUDE_x_power_8000_minus_inverse_l312_31267


namespace NUMINAMATH_CALUDE_new_person_weight_is_97_l312_31211

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 97 kg -/
theorem new_person_weight_is_97 :
  weight_of_new_person 8 4 65 = 97 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_97_l312_31211


namespace NUMINAMATH_CALUDE_system_solution_l312_31223

-- Define the system of equations
def system_equations (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

-- Theorem statement
theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), system_equations a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧ 
    x₁ = 1 / |a₁ - a₄| ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / |a₁ - a₄| :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l312_31223


namespace NUMINAMATH_CALUDE_girls_in_school_l312_31245

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ girls = boys - 10) :
  ∃ (school_girls : ℕ), school_girls = 760 ∧ 
    school_girls * sample_size = total_students * 95 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l312_31245


namespace NUMINAMATH_CALUDE_factorial_division_l312_31225

theorem factorial_division : Nat.factorial 6 / Nat.factorial (6 - 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l312_31225


namespace NUMINAMATH_CALUDE_total_cost_proof_l312_31259

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_rate : ℝ := 0.25
def num_sets : ℕ := 3

def total_cost : ℝ := num_sets * ((hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * (1 - discount_rate))

theorem total_cost_proof : total_cost = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l312_31259


namespace NUMINAMATH_CALUDE_x_minus_y_equals_60_l312_31244

theorem x_minus_y_equals_60 (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_60_l312_31244


namespace NUMINAMATH_CALUDE_waiter_tables_theorem_l312_31275

/-- Given the initial number of customers, the number of customers who left,
    and the number of people at each remaining table, calculate the number of
    tables with customers remaining. -/
def remaining_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) : ℕ :=
  (initial_customers - customers_left) / people_per_table

theorem waiter_tables_theorem (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ)
    (h1 : initial_customers ≥ customers_left)
    (h2 : people_per_table > 0)
    (h3 : (initial_customers - customers_left) % people_per_table = 0) :
    remaining_tables initial_customers customers_left people_per_table =
    (initial_customers - customers_left) / people_per_table :=
  by sorry

end NUMINAMATH_CALUDE_waiter_tables_theorem_l312_31275


namespace NUMINAMATH_CALUDE_expression_value_l312_31253

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- absolute value of m is 4
  : a + b - (c * d) ^ 2021 - 3 * m = -13 ∨ a + b - (c * d) ^ 2021 - 3 * m = 11 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l312_31253


namespace NUMINAMATH_CALUDE_triangle_value_l312_31289

theorem triangle_value (triangle : ℝ) :
  (∀ x : ℝ, (x - 5) * (x + triangle) = x^2 + 2*x - 35) →
  triangle = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l312_31289


namespace NUMINAMATH_CALUDE_complex_equation_solution_l312_31241

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = Complex.abs (2 * i)) : z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l312_31241


namespace NUMINAMATH_CALUDE_problem_solution_l312_31294

theorem problem_solution (a b : ℝ) 
  (h1 : 2 + a = 5 - b) 
  (h2 : 5 + b = 8 + a) : 
  2 - a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l312_31294


namespace NUMINAMATH_CALUDE_pizza_slice_count_l312_31229

/-- Given a number of pizzas and slices per pizza, calculates the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Proves that 21 pizzas with 8 slices each results in 168 total slices -/
theorem pizza_slice_count : total_slices 21 8 = 168 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_count_l312_31229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l312_31221

theorem arithmetic_sequence_common_difference (d : ℕ+) : 
  (∃ n : ℕ, 1 + (n - 1) * d.val = 81) → d ≠ 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l312_31221


namespace NUMINAMATH_CALUDE_civil_service_exam_probability_l312_31224

theorem civil_service_exam_probability 
  (pass_rate_written : ℝ) 
  (pass_rate_overall : ℝ) 
  (h1 : pass_rate_written = 0.2) 
  (h2 : pass_rate_overall = 0.04) :
  pass_rate_overall / pass_rate_written = 0.2 :=
sorry

end NUMINAMATH_CALUDE_civil_service_exam_probability_l312_31224


namespace NUMINAMATH_CALUDE_d_value_l312_31264

-- Define the function f(x) = x⋅(4x-3)
def f (x : ℝ) : ℝ := x * (4 * x - 3)

-- Define the interval (-9/4, 3/2)
def interval : Set ℝ := { x | -9/4 < x ∧ x < 3/2 }

-- State the theorem
theorem d_value : 
  ∃ d : ℝ, (∀ x : ℝ, f x < d ↔ x ∈ interval) → d = 27/2 := by sorry

end NUMINAMATH_CALUDE_d_value_l312_31264


namespace NUMINAMATH_CALUDE_problem_solution_l312_31238

def f (k : ℝ) (x : ℝ) := k - |x - 3|

theorem problem_solution (k : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 1 = {x | f k (x + 3) ≥ 0})
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  k = 1 ∧ 1/9 * a + 2/9 * b + 3/9 * c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l312_31238


namespace NUMINAMATH_CALUDE_percentage_absent_l312_31243

/-- Given a class of 50 students with 45 present, prove that 10% are absent. -/
theorem percentage_absent (total : ℕ) (present : ℕ) (h1 : total = 50) (h2 : present = 45) :
  (total - present) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_absent_l312_31243


namespace NUMINAMATH_CALUDE_milk_sales_l312_31201

theorem milk_sales : 
  let morning_packets : ℕ := 150
  let morning_250ml : ℕ := 60
  let morning_300ml : ℕ := 40
  let morning_350ml : ℕ := morning_packets - morning_250ml - morning_300ml
  let evening_packets : ℕ := 100
  let evening_400ml : ℕ := evening_packets / 2
  let evening_500ml : ℕ := evening_packets / 4
  let evening_450ml : ℕ := evening_packets - evening_400ml - evening_500ml
  let ml_per_ounce : ℕ := 30
  let remaining_ml : ℕ := 42000
  let total_ml : ℕ := 
    morning_250ml * 250 + morning_300ml * 300 + morning_350ml * 350 +
    evening_400ml * 400 + evening_500ml * 500 + evening_450ml * 450
  let sold_ml : ℕ := total_ml - remaining_ml
  let sold_ounces : ℚ := sold_ml / ml_per_ounce
  sold_ounces = 1541.67 := by
    sorry

end NUMINAMATH_CALUDE_milk_sales_l312_31201


namespace NUMINAMATH_CALUDE_gcd_193116_127413_properties_l312_31284

theorem gcd_193116_127413_properties :
  let g := Nat.gcd 193116 127413
  ∃ (g : ℕ),
    g = 3 ∧
    g ∣ 3 ∧
    ¬(2 ∣ g) ∧
    ¬(9 ∣ g) ∧
    ¬(11 ∣ g) ∧
    ¬(33 ∣ g) ∧
    ¬(99 ∣ g) := by
  sorry

end NUMINAMATH_CALUDE_gcd_193116_127413_properties_l312_31284


namespace NUMINAMATH_CALUDE_geometric_sum_15_l312_31277

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_15_l312_31277


namespace NUMINAMATH_CALUDE_power_of_two_pairs_l312_31283

theorem power_of_two_pairs (a b : ℕ) (h1 : a ≠ b) (h2 : ∃ k : ℕ, a + b = 2^k) (h3 : ∃ m : ℕ, a * b + 1 = 2^m) :
  (∃ k : ℕ, k ≥ 1 ∧ ((a = 1 ∧ b = 2^k - 1) ∨ (a = 2^k - 1 ∧ b = 1))) ∨
  (∃ k : ℕ, k ≥ 2 ∧ ((a = 2^k - 1 ∧ b = 2^k + 1) ∨ (a = 2^k + 1 ∧ b = 2^k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_pairs_l312_31283


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_70_factorial_l312_31295

-- Define 70!
def factorial_70 : ℕ := Nat.factorial 70

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial :
  last_two_nonzero_digits factorial_70 = 48 := by
  sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_70_factorial_l312_31295


namespace NUMINAMATH_CALUDE_magic_square_difference_l312_31216

/-- Represents a 3x3 magic square with some given values -/
structure MagicSquare where
  x : ℝ
  y : ℝ
  isValid : x - 2 = 2*y + y ∧ x - 2 = -2 + y + 6

/-- Proves that in the given magic square, y - x = -6 -/
theorem magic_square_difference (ms : MagicSquare) : ms.y - ms.x = -6 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_difference_l312_31216


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l312_31235

/-- Given that (2x-1)^5 + (x+2)^4 = a + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove that |a| + |a₂| + |a₄| = 30 -/
theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (2*x - 1)^5 + (x + 2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  |a| + |a₂| + |a₄| = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l312_31235


namespace NUMINAMATH_CALUDE_total_tickets_is_900_l312_31208

/-- Represents the total number of tickets sold at a movie theater. -/
def total_tickets (adult_price child_price : ℕ) (total_revenue child_tickets : ℕ) : ℕ :=
  let adult_tickets := (total_revenue - child_price * child_tickets) / adult_price
  adult_tickets + child_tickets

/-- Theorem stating that the total number of tickets sold is 900. -/
theorem total_tickets_is_900 :
  total_tickets 7 4 5100 400 = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_900_l312_31208


namespace NUMINAMATH_CALUDE_additional_sugar_needed_l312_31255

/-- The amount of sugar needed for a cake recipe -/
def recipe_sugar : ℕ := 14

/-- The amount of sugar already added to the cake -/
def sugar_added : ℕ := 2

/-- The additional amount of sugar needed -/
def additional_sugar : ℕ := recipe_sugar - sugar_added

theorem additional_sugar_needed : additional_sugar = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_sugar_needed_l312_31255


namespace NUMINAMATH_CALUDE_sum_exponents_15_factorial_l312_31233

/-- The largest perfect square that divides n! -/
def largestPerfectSquareDivisor (n : ℕ) : ℕ := sorry

/-- The sum of the exponents of the prime factors of the square root of a number -/
def sumExponentsOfSquareRoot (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the exponents of the prime factors of the square root
    of the largest perfect square that divides 15! is equal to 10 -/
theorem sum_exponents_15_factorial :
  sumExponentsOfSquareRoot (largestPerfectSquareDivisor 15) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_exponents_15_factorial_l312_31233


namespace NUMINAMATH_CALUDE_rhombus_area_l312_31242

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 8√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l312_31242


namespace NUMINAMATH_CALUDE_prism_volume_l312_31226

/-- Given a right rectangular prism with face areas 30 cm², 50 cm², and 75 cm², 
    its volume is 335 cm³. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 335 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l312_31226


namespace NUMINAMATH_CALUDE_min_draws_for_eighteen_balls_l312_31296

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_draws_for_eighteen_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 23)
  (h_yellow : counts.yellow = 21)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 14)
  (h_black : counts.black = 12) :
  minDrawsForColor counts 18 = 95 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_eighteen_balls_l312_31296


namespace NUMINAMATH_CALUDE_quadratic_solution_l312_31299

theorem quadratic_solution (b : ℝ) : 
  ((-10 : ℝ)^2 + b * (-10) - 30 = 0) → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l312_31299


namespace NUMINAMATH_CALUDE_expand_expression_l312_31261

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l312_31261


namespace NUMINAMATH_CALUDE_max_distance_for_specific_bicycle_l312_31266

/-- Represents a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with tire swapping -/
def max_distance (b : Bicycle) : ℝ :=
  sorry

/-- Theorem stating the maximum distance for a specific bicycle -/
theorem max_distance_for_specific_bicycle :
  let b : Bicycle := { front_tire_life := 5000, rear_tire_life := 3000 }
  max_distance b = 3750 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_specific_bicycle_l312_31266


namespace NUMINAMATH_CALUDE_remainder_of_three_to_500_mod_17_l312_31288

theorem remainder_of_three_to_500_mod_17 : 3^500 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_three_to_500_mod_17_l312_31288


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l312_31212

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 3 → b = 27 → c^2 = a * b → c = 9 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_segment_l312_31212


namespace NUMINAMATH_CALUDE_complex_norm_problem_l312_31237

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = Real.sqrt (225 / 7) :=
sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l312_31237


namespace NUMINAMATH_CALUDE_product_of_sines_equality_l312_31271

theorem product_of_sines_equality : 
  (1 + Real.sin (π/12)) * (1 + Real.sin (5*π/12)) * (1 + Real.sin (7*π/12)) * (1 + Real.sin (11*π/12)) = 
  (1 + Real.sin (π/12))^2 * (1 + Real.sin (5*π/12))^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_sines_equality_l312_31271


namespace NUMINAMATH_CALUDE_base_seven_54321_to_decimal_l312_31279

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_54321_to_decimal :
  base_seven_to_decimal [1, 2, 3, 4, 5] = 13539 :=
by sorry

end NUMINAMATH_CALUDE_base_seven_54321_to_decimal_l312_31279


namespace NUMINAMATH_CALUDE_cheryl_unused_material_l312_31228

-- Define the amount of material Cheryl bought of each type
def material1 : ℚ := 3 / 8
def material2 : ℚ := 1 / 3

-- Define the total amount of material Cheryl bought
def total_bought : ℚ := material1 + material2

-- Define the amount of material Cheryl used
def material_used : ℚ := 0.33333333333333326

-- Define the amount of material left unused
def material_left : ℚ := total_bought - material_used

-- Theorem statement
theorem cheryl_unused_material : material_left = 0.375 := by sorry

end NUMINAMATH_CALUDE_cheryl_unused_material_l312_31228


namespace NUMINAMATH_CALUDE_parabola_focus_point_slope_l312_31227

/-- The slope of a line between the focus of a parabola and a point on the parabola -/
theorem parabola_focus_point_slope (x y : ℝ) :
  y^2 = 4*x →  -- parabola equation
  x > 0 →  -- point is in the fourth quadrant
  y < 0 →  -- point is in the fourth quadrant
  x + 1 = 5 →  -- distance from point to directrix is 5
  (y - 0) / (x - 1) = -4/3 :=  -- slope of line AF
by sorry

end NUMINAMATH_CALUDE_parabola_focus_point_slope_l312_31227


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l312_31258

def bacterial_population (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_exceeding_500 :
  ∃ n : ℕ, bacterial_population n > 500 ∧ ∀ m : ℕ, m < n → bacterial_population m ≤ 500 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l312_31258


namespace NUMINAMATH_CALUDE_ellipse_equation_and_max_area_l312_31293

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci where
  left : Point
  right : Point

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

theorem ellipse_equation_and_max_area 
  (C : Ellipse) 
  (P : Point)
  (F : Foci)
  (h_P_on_C : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_P_coords : P.x = 1 ∧ P.y = Real.sqrt 2 / 2)
  (h_PF_sum : distance P F.left + distance P F.right = 2 * Real.sqrt 2) :
  (∃ (a b : ℝ), C.a = a ∧ C.b = b ∧ a^2 = 2 ∧ b^2 = 1) ∧
  (∃ (max_area : ℝ), 
    (∀ (Q : Point) (h_Q_on_C : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1),
      abs (P.x * Q.y - P.y * Q.x) / 2 ≤ max_area) ∧
    max_area = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_max_area_l312_31293


namespace NUMINAMATH_CALUDE_sock_pair_probability_l312_31256

def total_socks : ℕ := 40
def white_socks : ℕ := 10
def red_socks : ℕ := 12
def black_socks : ℕ := 18
def drawn_socks : ℕ := 3

theorem sock_pair_probability :
  let total_ways := Nat.choose total_socks drawn_socks
  let all_different := white_socks * red_socks * black_socks
  let at_least_one_pair := total_ways - all_different
  (at_least_one_pair : ℚ) / total_ways = 193 / 247 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_probability_l312_31256


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l312_31202

theorem multiply_and_add_equality : 52 * 46 + 104 * 52 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l312_31202


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_three_l312_31297

/-- The function f(x) = (x^3 + x^2 + 1) / (x - 3) has a vertical asymptote at x = 3 -/
theorem vertical_asymptote_at_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^3 + x^2 + 1) / (x - 3)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ → δ < ε → |f (3 + δ)| > (1 / δ) ∧ |f (3 - δ)| > (1 / δ) :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_three_l312_31297


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l312_31206

theorem prime_squared_minus_one_divisible_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l312_31206


namespace NUMINAMATH_CALUDE_number_of_balls_correct_l312_31292

/-- The number of balls in a box, which is as much greater than 40 as it is less than 60. -/
def number_of_balls : ℕ := 50

/-- The condition that the number of balls is as much greater than 40 as it is less than 60. -/
def ball_condition (x : ℕ) : Prop := x - 40 = 60 - x

theorem number_of_balls_correct : ball_condition number_of_balls := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_correct_l312_31292


namespace NUMINAMATH_CALUDE_beijing_olympics_village_area_notation_l312_31250

/-- Expresses 38.66 million in scientific notation -/
theorem beijing_olympics_village_area_notation :
  (38.66 * 1000000 : ℝ) = 3.866 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_beijing_olympics_village_area_notation_l312_31250


namespace NUMINAMATH_CALUDE_square_5_on_top_l312_31282

/-- Represents a square on the paper grid -/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the paper grid -/
def Grid := List Square

/-- Defines the initial configuration of the grid -/
def initialGrid : Grid :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].map
    (fun n => ⟨n, (n - 1) / 5 + 1, (n - 1) % 5 + 1⟩)

/-- Performs a folding operation on the grid -/
def fold (g : Grid) (foldType : String) : Grid := sorry

/-- Theorem stating that after all folding operations, square 5 is on top -/
theorem square_5_on_top (g : Grid) (h : g = initialGrid) :
  (fold (fold (fold (fold g "left_third") "right_third") "bottom_half") "top_half").head?.map Square.number = some 5 := by sorry

end NUMINAMATH_CALUDE_square_5_on_top_l312_31282


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l312_31281

theorem absolute_value_inequality_solution :
  {x : ℤ | |7 * x - 5| ≤ 9} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l312_31281


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l312_31214

theorem geometric_sequence_problem (a b c : ℝ) :
  (∀ q : ℝ, 1 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 4) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l312_31214


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l312_31257

/-- Given two lines passing through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₁ b₁ ∧ 
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₂ b₂ := by
  sorry

#check line_through_coefficient_points

end NUMINAMATH_CALUDE_line_through_coefficient_points_l312_31257


namespace NUMINAMATH_CALUDE_luke_received_21_dollars_l312_31268

/-- Calculates the amount of money Luke received from his mom -/
def money_from_mom (initial amount_spent final : ℕ) : ℕ :=
  final - (initial - amount_spent)

/-- Proves that Luke received 21 dollars from his mom -/
theorem luke_received_21_dollars :
  money_from_mom 48 11 58 = 21 := by
  sorry

end NUMINAMATH_CALUDE_luke_received_21_dollars_l312_31268


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l312_31218

def numbers : List ℕ := [38, 114, 152, 95]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l312_31218


namespace NUMINAMATH_CALUDE_M_subset_N_l312_31254

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 3 + 1 / 6}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 6 + 1 / 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l312_31254


namespace NUMINAMATH_CALUDE_unique_prime_twice_squares_l312_31217

theorem unique_prime_twice_squares : 
  ∀ p : ℕ, 
    Prime p → 
    (∃ x : ℕ, p + 1 = 2 * x^2) → 
    (∃ y : ℕ, p^2 + 1 = 2 * y^2) → 
    p = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_twice_squares_l312_31217


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l312_31213

theorem quadratic_solution_difference_squared : 
  ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 3*Φ + 1 = 0 → 
  φ^2 - 3*φ + 1 = 0 → 
  (Φ - φ)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l312_31213


namespace NUMINAMATH_CALUDE_student_committee_size_l312_31269

theorem student_committee_size (ways : ℕ) (h : ways = 30) : 
  (∃ n : ℕ, n * (n - 1) = ways) → 
  (∃! n : ℕ, n > 0 ∧ n * (n - 1) = ways) ∧ 
  (∃ n : ℕ, n > 0 ∧ n * (n - 1) = ways ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_student_committee_size_l312_31269


namespace NUMINAMATH_CALUDE_smallest_m_proof_l312_31200

/-- The smallest positive integer m such that 15m - 3 is divisible by 11 -/
def smallest_m : ℕ := 9

theorem smallest_m_proof :
  smallest_m = 9 ∧
  ∀ k : ℕ, k > 0 → (15 * k - 3) % 11 = 0 → k ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_proof_l312_31200


namespace NUMINAMATH_CALUDE_total_nails_eq_252_l312_31291

/-- The number of nails/claws/toes to be cut -/
def total_nails : ℕ :=
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let dog_nails := dogs * 4 * 4
  let parrot_claws := (parrots - 1) * 2 * 3 + 1 * 2 * 4
  let cat_toes := cats * (2 * 5 + 2 * 4)
  let rabbit_nails := rabbits * (2 * 5 + 3 + 4)
  dog_nails + parrot_claws + cat_toes + rabbit_nails

/-- Theorem stating that the total number of nails/claws/toes to be cut is 252 -/
theorem total_nails_eq_252 : total_nails = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_eq_252_l312_31291


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l312_31252

-- Problem 1
theorem factorization_problem1 (m : ℝ) : 
  m * (m - 5) - 2 * (5 - m)^2 = -(m - 5) * (m - 10) := by sorry

-- Problem 2
theorem factorization_problem2 (x : ℝ) : 
  -4 * x^3 + 8 * x^2 - 4 * x = -4 * x * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l312_31252


namespace NUMINAMATH_CALUDE_karls_total_income_l312_31236

/-- Represents the prices of items in Karl's store -/
structure Prices where
  tshirt : ℚ
  pants : ℚ
  skirt : ℚ
  refurbished_tshirt : ℚ

/-- Represents the quantities of items sold -/
structure QuantitiesSold where
  tshirt : ℕ
  pants : ℕ
  skirt : ℕ
  refurbished_tshirt : ℕ

/-- Calculates the total income given prices and quantities sold -/
def totalIncome (prices : Prices) (quantities : QuantitiesSold) : ℚ :=
  prices.tshirt * quantities.tshirt +
  prices.pants * quantities.pants +
  prices.skirt * quantities.skirt +
  prices.refurbished_tshirt * quantities.refurbished_tshirt

/-- Theorem stating that Karl's total income is $53 -/
theorem karls_total_income :
  let prices : Prices := {
    tshirt := 5,
    pants := 4,
    skirt := 6,
    refurbished_tshirt := 5/2
  }
  let quantities : QuantitiesSold := {
    tshirt := 2,
    pants := 1,
    skirt := 4,
    refurbished_tshirt := 6
  }
  totalIncome prices quantities = 53 := by
  sorry


end NUMINAMATH_CALUDE_karls_total_income_l312_31236


namespace NUMINAMATH_CALUDE_lychee_harvest_l312_31204

theorem lychee_harvest (initial : ℕ) : 
  (initial / 2 : ℚ) * (2 / 5 : ℚ) = 100 → initial = 500 := by
  sorry

end NUMINAMATH_CALUDE_lychee_harvest_l312_31204


namespace NUMINAMATH_CALUDE_third_place_prize_l312_31260

def prize_distribution (total_people : ℕ) (contribution : ℕ) (first_place_percentage : ℚ) : ℚ :=
  let total_pot : ℚ := (total_people * contribution : ℚ)
  let first_place_prize : ℚ := total_pot * first_place_percentage
  let remaining : ℚ := total_pot - first_place_prize
  remaining / 2

theorem third_place_prize :
  prize_distribution 8 5 (4/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_place_prize_l312_31260


namespace NUMINAMATH_CALUDE_derivative_of_f_l312_31285

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 10) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - Real.log 10 * (Real.log x / Real.log 10)) / (x^2 * Real.log 10) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l312_31285


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l312_31219

theorem exponent_equation_solution :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = 9^(y - 1) ↔ y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l312_31219


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l312_31290

/-- A line y = 3x + d is tangent to the parabola y² = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + d → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l312_31290


namespace NUMINAMATH_CALUDE_lcm_36_105_l312_31232

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l312_31232


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l312_31278

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

theorem binary_253_ones_minus_zeros :
  let bin_253 := binary_representation 253
  let x := count_zeros bin_253
  let y := count_ones bin_253
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l312_31278


namespace NUMINAMATH_CALUDE_triangle_area_l312_31205

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(5, 10) is 24 square units. -/
theorem triangle_area : let A : ℝ × ℝ := (2, 2)
                        let B : ℝ × ℝ := (8, 2)
                        let C : ℝ × ℝ := (5, 10)
                        (1/2 : ℝ) * |((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))| = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l312_31205


namespace NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l312_31215

theorem tan_eleven_pi_sixths : Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l312_31215


namespace NUMINAMATH_CALUDE_max_stickers_l312_31274

theorem max_stickers (class_size : ℕ) (mean_stickers : ℕ) (max_stickers : ℕ) : 
  class_size = 25 →
  mean_stickers = 4 →
  max_stickers = class_size * mean_stickers - (class_size - 1) →
  max_stickers = 76 :=
by
  sorry

#check max_stickers

end NUMINAMATH_CALUDE_max_stickers_l312_31274


namespace NUMINAMATH_CALUDE_rectangle_max_area_l312_31247

/-- A rectangle with perimeter 40 and area 100 has sides of length 10 -/
theorem rectangle_max_area (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧  -- x and y are positive (implicitly defining a rectangle)
  2 * (x + y) = 40 ∧  -- perimeter is 40
  x * y = 100  -- area is 100
  → x = 10 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l312_31247


namespace NUMINAMATH_CALUDE_laptop_cost_proof_l312_31231

theorem laptop_cost_proof (x y : ℝ) (h1 : y = 3 * x) (h2 : x + y = 2000) : x = 500 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_proof_l312_31231


namespace NUMINAMATH_CALUDE_expression_value_l312_31230

theorem expression_value : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l312_31230


namespace NUMINAMATH_CALUDE_monogram_count_l312_31207

/-- The number of letters in the alphabet before 'M' -/
def letters_before_m : Nat := 12

/-- The number of letters in the alphabet after 'M' -/
def letters_after_m : Nat := 13

/-- A monogram is valid if it satisfies the given conditions -/
def is_valid_monogram (f m l : Char) : Prop :=
  f < m ∧ m < l ∧ f ≠ m ∧ m ≠ l ∧ f ≠ l ∧ m = 'M'

/-- The total number of valid monograms -/
def total_valid_monograms : Nat := letters_before_m * letters_after_m

theorem monogram_count :
  total_valid_monograms = 156 :=
sorry

end NUMINAMATH_CALUDE_monogram_count_l312_31207


namespace NUMINAMATH_CALUDE_hyperbola_equation_l312_31265

-- Define the hyperbola
def Hyperbola (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- State the theorem
theorem hyperbola_equation :
  ∀ (a b c : ℝ),
    -- Conditions
    (2 * a = 8) →  -- Distance between vertices
    (c / a = 5 / 4) →  -- Eccentricity
    (c^2 = a^2 + b^2) →  -- Relation between a, b, and c
    -- Conclusion
    Hyperbola a b c = Hyperbola 4 3 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l312_31265


namespace NUMINAMATH_CALUDE_distinct_outfits_l312_31286

theorem distinct_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (hats : ℕ) 
  (h_shirts : shirts = 7)
  (h_pants : pants = 5)
  (h_ties : ties = 6)
  (h_hats : hats = 4) :
  shirts * pants * (ties + 1) * (hats + 1) = 1225 := by
  sorry

end NUMINAMATH_CALUDE_distinct_outfits_l312_31286


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l312_31240

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (187 * π / 180) * Real.cos (52 * π / 180) +
  Real.cos (7 * π / 180) * Real.sin (52 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l312_31240


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l312_31234

/-- The probability of having at least one boy and one girl in a family of four children,
    given that the birth of a boy or a girl is equally likely. -/
theorem prob_at_least_one_boy_one_girl : 
  let p_boy : ℚ := 1/2  -- Probability of having a boy
  let p_girl : ℚ := 1/2  -- Probability of having a girl
  let n : ℕ := 4  -- Number of children
  1 - (p_boy ^ n + p_girl ^ n) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l312_31234


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l312_31262

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 * lg 5 + (lg 5)^2 + lg 2 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l312_31262


namespace NUMINAMATH_CALUDE_wheat_D_tallest_and_neatest_l312_31272

-- Define the wheat types
inductive WheatType
| A
| B
| C
| D

-- Define a function for average height
def averageHeight (t : WheatType) : ℝ :=
  match t with
  | .A => 13
  | .B => 15
  | .C => 13
  | .D => 15

-- Define a function for variance
def variance (t : WheatType) : ℝ :=
  match t with
  | .A => 3.6
  | .B => 6.3
  | .C => 6.3
  | .D => 3.6

-- Define a predicate for tallness
def isTaller (t1 t2 : WheatType) : Prop :=
  averageHeight t1 > averageHeight t2

-- Define a predicate for neatness (lower variance means neater)
def isNeater (t1 t2 : WheatType) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem wheat_D_tallest_and_neatest :
  ∀ t : WheatType, t ≠ WheatType.D →
    (isTaller WheatType.D t ∨ averageHeight WheatType.D = averageHeight t) ∧
    (isNeater WheatType.D t ∨ variance WheatType.D = variance t) :=
by sorry

end NUMINAMATH_CALUDE_wheat_D_tallest_and_neatest_l312_31272


namespace NUMINAMATH_CALUDE_peanut_butter_cans_l312_31210

theorem peanut_butter_cans (n : ℕ) (initial_avg_price remaining_avg_price returned_avg_price : ℚ)
  (h1 : initial_avg_price = 365/10)
  (h2 : remaining_avg_price = 30/1)
  (h3 : returned_avg_price = 495/10)
  (h4 : n * initial_avg_price = (n - 2) * remaining_avg_price + 2 * returned_avg_price) :
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_cans_l312_31210


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l312_31248

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ ∧
  z = z :=
by
  sorry

#check rectangular_to_cylindrical

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l312_31248


namespace NUMINAMATH_CALUDE_fraction_reciprocal_sum_ge_two_l312_31222

theorem fraction_reciprocal_sum_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b + b / a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_sum_ge_two_l312_31222


namespace NUMINAMATH_CALUDE_circle_angle_sum_l312_31298

theorem circle_angle_sum (a b : ℝ) : 
  a + b + 110 + 60 = 360 → a + b = 190 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_sum_l312_31298


namespace NUMINAMATH_CALUDE_max_sum_product_l312_31270

theorem max_sum_product (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_200 : a + b + c + d = 200) : 
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_sum_product_l312_31270


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l312_31246

theorem geometric_mean_minimum (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (hgm : z^2 = x*y) : 
  (Real.log z)/(4*Real.log x) + (Real.log z)/(Real.log y) ≥ 9/8 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l312_31246


namespace NUMINAMATH_CALUDE_circle_equation_correct_l312_31273

/-- Represents a circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- The circle passes through a point (x, y) -/
def Circle.passesThrough (c : Circle) (x y : ℝ) : Prop :=
  c.equation x y

/-- The center of the circle lies on the line x + y = 0 -/
def Circle.centerOnLine (c : Circle) : Prop :=
  c.a + c.b = 0

theorem circle_equation_correct :
  ∃ (c : Circle), 
    c.equation x y ↔ (x + 3)^2 + (y - 3)^2 = 10 ∧
    c.centerOnLine ∧
    c.passesThrough 0 2 ∧
    c.passesThrough (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l312_31273
