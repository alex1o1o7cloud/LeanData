import Mathlib

namespace NUMINAMATH_CALUDE_investment_problem_l2482_248254

/-- The investment problem -/
theorem investment_problem (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 16000)
  (hc : c_investment = 20000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 86400) :
  ∃ a_investment : ℕ, 
    a_investment * total_profit = 
      (total_profit - b_investment * total_profit / (a_investment + b_investment + c_investment) - 
       c_investment * total_profit / (a_investment + b_investment + c_investment)) :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l2482_248254


namespace NUMINAMATH_CALUDE_total_distance_meters_l2482_248202

def distance_feet : ℝ := 30
def feet_to_meters : ℝ := 0.3048
def num_trips : ℕ := 4

theorem total_distance_meters : 
  distance_feet * feet_to_meters * (num_trips : ℝ) = 36.576 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_meters_l2482_248202


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2482_248252

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2482_248252


namespace NUMINAMATH_CALUDE_stating_roper_lawn_cutting_l2482_248251

/-- Represents the number of times Mr. Roper cuts his lawn in different periods --/
structure LawnCutting where
  summer_months : ℕ  -- Number of months from April to September
  winter_months : ℕ  -- Number of months from October to March
  summer_cuts : ℕ    -- Number of cuts per month in summer
  average_cuts : ℕ   -- Average number of cuts per month over a year
  total_months : ℕ   -- Total number of months in a year

/-- 
Theorem stating that given the conditions, 
Mr. Roper cuts his lawn 3 times a month from October to March 
-/
theorem roper_lawn_cutting (l : LawnCutting) 
  (h1 : l.summer_months = 6)
  (h2 : l.winter_months = 6)
  (h3 : l.summer_cuts = 15)
  (h4 : l.average_cuts = 9)
  (h5 : l.total_months = 12) :
  (l.total_months * l.average_cuts - l.summer_months * l.summer_cuts) / l.winter_months = 3 := by
  sorry

#check roper_lawn_cutting

end NUMINAMATH_CALUDE_stating_roper_lawn_cutting_l2482_248251


namespace NUMINAMATH_CALUDE_program_output_equals_b_l2482_248272

def program (a b : ℕ) : ℕ :=
  if a > b then a else b

theorem program_output_equals_b :
  let a : ℕ := 2
  let b : ℕ := 3
  program a b = b := by sorry

end NUMINAMATH_CALUDE_program_output_equals_b_l2482_248272


namespace NUMINAMATH_CALUDE_sequence_inequality_l2482_248222

theorem sequence_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∀ n : ℕ, (a^n / (n : ℝ)^b) < (a^(n+1) / ((n+1) : ℝ)^b) :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2482_248222


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2482_248259

def P (x : ℝ) : ℝ := 5*x^3 - 12*x^2 + 6*x - 15

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), P = λ x => q x * (x - 3) + 30 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2482_248259


namespace NUMINAMATH_CALUDE_base_n_not_prime_l2482_248206

/-- For a positive integer n ≥ 2, 2002_n represents the number in base n notation -/
def base_n (n : ℕ) : ℕ := 2 * n^3 + 2

/-- A number is prime if it has exactly two distinct positive divisors -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ (∀ m : ℕ, m > 0 → m < p → p % m ≠ 0)

theorem base_n_not_prime (n : ℕ) (h : n ≥ 2) : ¬ (is_prime (base_n n)) := by
  sorry

end NUMINAMATH_CALUDE_base_n_not_prime_l2482_248206


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2482_248201

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
    factors with integer coefficients if and only if a = 34 -/
theorem quadratic_factorization (a : ℤ) : 
  (∃ (m n p q : ℤ), 15 * X^2 + a * X + 15 = (m * X + n) * (p * X + q)) ↔ a = 34 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2482_248201


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_lcm_sum_l2482_248274

theorem two_numbers_with_sum_and_gcd_lcm_sum (a b : ℕ) : 
  a + b = 60 ∧ 
  Nat.gcd a b + Nat.lcm a b = 84 → 
  (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_lcm_sum_l2482_248274


namespace NUMINAMATH_CALUDE_compare_fractions_l2482_248244

theorem compare_fractions (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l2482_248244


namespace NUMINAMATH_CALUDE_textbook_cost_ratio_l2482_248291

/-- The ratio of the cost of bookstore textbooks to online ordered books -/
theorem textbook_cost_ratio : 
  ∀ (sale_price online_price bookstore_price total_price : ℕ),
  sale_price = 5 * 10 →
  online_price = 40 →
  total_price = 210 →
  bookstore_price = total_price - sale_price - online_price →
  ∃ (k : ℕ), bookstore_price = k * online_price →
  (bookstore_price : ℚ) / online_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_textbook_cost_ratio_l2482_248291


namespace NUMINAMATH_CALUDE_inequality_solution_l2482_248281

def solution_set (a : ℝ) : Set ℝ :=
  if a < 1 then {x | x < a ∨ x > 1}
  else if a = 1 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1) * x + a > 0} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2482_248281


namespace NUMINAMATH_CALUDE_least_possible_a_2000_l2482_248293

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ m n, m ∣ n → m < n → a m ∣ a n ∧ a m < a n

theorem least_possible_a_2000 (a : ℕ → ℕ) (h : sequence_property a) : a 2000 ≥ 128 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_a_2000_l2482_248293


namespace NUMINAMATH_CALUDE_b_invests_after_six_months_l2482_248292

/-- A partnership with three investors A, B, and C -/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.total_gain = 24000 ∧ 
  p.a_share = 8000 ∧ 
  0 < p.x ∧ 
  0 < p.m ∧ 
  p.m < 12

/-- The theorem stating that B invests after 6 months -/
theorem b_invests_after_six_months (p : Partnership) 
  (h : partnership_conditions p) : p.m = 6 := by
  sorry


end NUMINAMATH_CALUDE_b_invests_after_six_months_l2482_248292


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l2482_248216

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 + 6 = x*y*z) :
  (x^3 + y^3 + z^3 - 3*x*y*z) / (x*y*z) = 5 - 30 / (x*y*z) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l2482_248216


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l2482_248282

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  -- The hyperbola passes through (5, -3)
  point1 : (5 : ℝ)^2 - (-3 : ℝ)^2 * a = b
  -- The hyperbola passes through (3, 0)
  point2 : (3 : ℝ)^2 = b
  -- The hyperbola passes through (s, -1)
  point3 : s^2 - (-1 : ℝ)^2 * a = b
  -- Ensure a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- s is a real number
  s : ℝ

/-- The theorem stating the value of s^2 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 873/81 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l2482_248282


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l2482_248296

theorem contractor_engagement_days
  (daily_wage : ℕ)
  (daily_fine : ℚ)
  (total_pay : ℕ)
  (absent_days : ℕ)
  (h_daily_wage : daily_wage = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_pay : total_pay = 425)
  (h_absent_days : absent_days = 10) :
  ∃ (work_days : ℕ),
    (daily_wage : ℚ) * work_days - daily_fine * absent_days = total_pay ∧
    work_days + absent_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l2482_248296


namespace NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l2482_248219

theorem two_fifths_of_n_is_80 (n : ℚ) : n = 5 / 6 * 240 → 2 / 5 * n = 80 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l2482_248219


namespace NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l2482_248257

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (x : ℤ), (x + 1)^2 - x^2 = 2000 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l2482_248257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2482_248279

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_ninth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_diff : a 3 - a 2 = -2) 
  (h_seventh : a 7 = -2) : 
  a 9 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2482_248279


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2482_248280

-- Define the shape
structure CubeShape :=
  (base_length : ℕ)
  (base_width : ℕ)
  (column_height : ℕ)

-- Define the properties of our specific shape
def our_shape : CubeShape :=
  { base_length := 3
  , base_width := 3
  , column_height := 3 }

-- Calculate the volume of the shape
def volume (shape : CubeShape) : ℕ :=
  shape.base_length * shape.base_width + shape.column_height - 1

-- Calculate the surface area of the shape
def surface_area (shape : CubeShape) : ℕ :=
  let base_area := 2 * shape.base_length * shape.base_width
  let side_area := 2 * shape.base_length * shape.column_height + 2 * shape.base_width * shape.column_height
  let column_area := 4 * (shape.column_height - 1)
  base_area + side_area + column_area - (shape.base_length * shape.base_width - 1)

-- Theorem: The ratio of volume to surface area is 9:40
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  shape = our_shape → volume shape * 40 = surface_area shape * 9 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2482_248280


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2482_248205

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, 2*x + 3*y = 7 ∧ x = -2*y + 3 ∧ x = 5 ∧ y = -1) ∧
  -- Second system
  (∃ x y : ℝ, 5*x + y = 4 ∧ 2*x - 3*y - 5 = 0 ∧ x = 1 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l2482_248205


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l2482_248288

/-- The ratio of distances covered by two ferries --/
theorem ferry_distance_ratio :
  let v_p : ℝ := 6  -- Speed of ferry P in km/h
  let t_p : ℝ := 3  -- Time taken by ferry P in hours
  let v_q : ℝ := v_p + 3  -- Speed of ferry Q in km/h
  let t_q : ℝ := t_p + 1  -- Time taken by ferry Q in hours
  let d_p : ℝ := v_p * t_p  -- Distance covered by ferry P
  let d_q : ℝ := v_q * t_q  -- Distance covered by ferry Q
  d_q / d_p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l2482_248288


namespace NUMINAMATH_CALUDE_point_trajectory_l2482_248246

-- Define the condition for point M(x,y)
def point_condition (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt (x^2 + (y-2)^2) = 8

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 16 = 1

-- Theorem statement
theorem point_trajectory : ∀ x y : ℝ, point_condition x y → trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l2482_248246


namespace NUMINAMATH_CALUDE_safety_cost_per_mile_approx_l2482_248248

/-- Safety Rent-a-Car's daily rate -/
def safety_daily_rate : ℝ := 21.95

/-- City Rentals' daily rate -/
def city_daily_rate : ℝ := 18.95

/-- City Rentals' cost per mile -/
def city_cost_per_mile : ℝ := 0.21

/-- Number of miles for equal cost -/
def equal_cost_miles : ℝ := 150.0

/-- Safety Rent-a-Car's cost per mile -/
noncomputable def safety_cost_per_mile : ℝ := 
  (city_daily_rate + city_cost_per_mile * equal_cost_miles - safety_daily_rate) / equal_cost_miles

theorem safety_cost_per_mile_approx :
  ∃ ε > 0, abs (safety_cost_per_mile - 0.177) < ε ∧ ε < 0.001 :=
sorry

end NUMINAMATH_CALUDE_safety_cost_per_mile_approx_l2482_248248


namespace NUMINAMATH_CALUDE_standard_deviation_measures_stability_l2482_248232

/-- A measure of stability for a set of numbers -/
def stability_measure (data : List ℝ) : ℝ := sorry

/-- Standard deviation of a list of real numbers -/
def standard_deviation (data : List ℝ) : ℝ := sorry

/-- Theorem stating that the standard deviation is a valid measure of stability for crop yields -/
theorem standard_deviation_measures_stability 
  (n : ℕ) 
  (yields : List ℝ) 
  (h1 : yields.length = n) 
  (h2 : n > 0) :
  stability_measure yields = standard_deviation yields := by sorry

end NUMINAMATH_CALUDE_standard_deviation_measures_stability_l2482_248232


namespace NUMINAMATH_CALUDE_coolant_replacement_l2482_248209

/-- Calculates the amount of original coolant left in a car's cooling system after partial replacement. -/
theorem coolant_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 19) 
  (h2 : initial_concentration = 0.3)
  (h3 : replacement_concentration = 0.8)
  (h4 : final_concentration = 0.5) : 
  initial_volume - (final_concentration * initial_volume - initial_concentration * initial_volume) / 
  (replacement_concentration - initial_concentration) = 11.4 := by
sorry

end NUMINAMATH_CALUDE_coolant_replacement_l2482_248209


namespace NUMINAMATH_CALUDE_solution_set_of_increasing_function_l2482_248262

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the solution set of |f(x+1)| < 1 is (-1, 2). -/
theorem solution_set_of_increasing_function (f : ℝ → ℝ) 
  (h_incr : StrictMono f) (h_f0 : f 0 = -1) (h_f3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_increasing_function_l2482_248262


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l2482_248260

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem units_digit_of_sum_factorials :
  (sum_factorials 100) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l2482_248260


namespace NUMINAMATH_CALUDE_always_odd_l2482_248263

theorem always_odd (k : ℤ) : Odd (2007 + 2 * k^2) := by sorry

end NUMINAMATH_CALUDE_always_odd_l2482_248263


namespace NUMINAMATH_CALUDE_sum_of_ones_and_twos_2020_l2482_248295

theorem sum_of_ones_and_twos_2020 :
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 5 * p.2 = 2020) (Finset.product (Finset.range 505) (Finset.range 404))).card = 102 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ones_and_twos_2020_l2482_248295


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2482_248290

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2482_248290


namespace NUMINAMATH_CALUDE_inequality_contradiction_l2482_248230

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * c * d < a * b * (c + d) ∧ (a + b) * (c + d) < a * b + c * d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l2482_248230


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l2482_248258

theorem spinster_cat_ratio : 
  ∀ (x : ℚ), 
    (22 : ℚ) / x = 7 →  -- ratio of spinsters to cats is x:7
    x = 22 + 55 →      -- there are 55 more cats than spinsters
    (2 : ℚ) / 7 = 22 / x -- the ratio of spinsters to cats is 2:7
  := by sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l2482_248258


namespace NUMINAMATH_CALUDE_polygon_with_72_degree_exterior_angles_has_5_sides_l2482_248243

/-- A polygon with exterior angles each measuring 72° has 5 sides -/
theorem polygon_with_72_degree_exterior_angles_has_5_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 72 →
    n * exterior_angle = 360 →
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_72_degree_exterior_angles_has_5_sides_l2482_248243


namespace NUMINAMATH_CALUDE_negation_equivalence_l2482_248214

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teacher : U → Prop)
variable (ExcellentAtMath : U → Prop)
variable (PoorAtMath : U → Prop)

-- Define the statements
def AllTeachersExcellent : Prop := ∀ x, Teacher x → ExcellentAtMath x
def AtLeastOneTeacherPoor : Prop := ∃ x, Teacher x ∧ PoorAtMath x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneTeacherPoor U Teacher PoorAtMath ↔ ¬(AllTeachersExcellent U Teacher ExcellentAtMath) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2482_248214


namespace NUMINAMATH_CALUDE_binomial_18_12_l2482_248208

theorem binomial_18_12 (h1 : Nat.choose 17 10 = 19448)
                        (h2 : Nat.choose 17 11 = 12376)
                        (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 18 12 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_12_l2482_248208


namespace NUMINAMATH_CALUDE_ball_attendees_l2482_248287

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l2482_248287


namespace NUMINAMATH_CALUDE_remainder_proof_l2482_248285

/-- The largest integer n such that 5^n divides 12^2015 + 13^2015 -/
def n : ℕ := 3

/-- The theorem statement -/
theorem remainder_proof :
  (12^2015 + 13^2015) / 5^n % 1000 = 625 :=
sorry

end NUMINAMATH_CALUDE_remainder_proof_l2482_248285


namespace NUMINAMATH_CALUDE_tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l2482_248276

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- Define the interval [0, 4]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Theorem for the tangent line equation at x = -1
theorem tangent_line_at_neg_one :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 4 * x - y + 4 = 0 :=
sorry

-- Theorem for the maximum value of f(x) on the interval [0, 4]
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 45 :=
sorry

-- Theorem for the minimum value of f(x) on the interval [0, 4]
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l2482_248276


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l2482_248255

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents a ball with its color and number -/
structure Ball :=
  (color : BallColor)
  (number : Nat)

/-- The box of balls -/
def box : List Ball := [
  ⟨BallColor.Red, 1⟩, ⟨BallColor.Red, 2⟩, ⟨BallColor.Red, 3⟩, ⟨BallColor.Red, 4⟩,
  ⟨BallColor.White, 3⟩, ⟨BallColor.White, 4⟩
]

/-- The number of balls to draw -/
def drawCount : Nat := 3

/-- Calculates the probability of drawing 3 balls with maximum number 3 -/
def probMaxThree : ℚ := 1 / 5

/-- Calculates the mathematical expectation of the maximum number among red balls drawn -/
def expectationMaxRed : ℚ := 13 / 4

theorem ball_drawing_probabilities :
  (probMaxThree = 1 / 5) ∧
  (expectationMaxRed = 13 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ball_drawing_probabilities_l2482_248255


namespace NUMINAMATH_CALUDE_only_ShouZhuDaiTu_describes_random_event_l2482_248278

-- Define the type for idioms
inductive Idiom
  | HaiKuShiLan
  | ShouZhuDaiTu
  | HuaBingChongJi
  | GuaShuDiLuo

-- Define a property for describing a random event
def describesRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.ShouZhuDaiTu => True
  | _ => False

-- Theorem statement
theorem only_ShouZhuDaiTu_describes_random_event :
  ∀ i : Idiom, describesRandomEvent i ↔ i = Idiom.ShouZhuDaiTu :=
by
  sorry


end NUMINAMATH_CALUDE_only_ShouZhuDaiTu_describes_random_event_l2482_248278


namespace NUMINAMATH_CALUDE_disjunction_true_false_l2482_248266

theorem disjunction_true_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_false_l2482_248266


namespace NUMINAMATH_CALUDE_journey_distance_total_distance_value_l2482_248231

/-- Represents the total distance travelled by a family in a journey -/
def total_distance : ℝ := sorry

/-- The total travel time in hours -/
def total_time : ℝ := 18

/-- The speed for the first third of the journey in km/h -/
def speed1 : ℝ := 35

/-- The speed for the second third of the journey in km/h -/
def speed2 : ℝ := 40

/-- The speed for the last third of the journey in km/h -/
def speed3 : ℝ := 45

/-- Theorem stating the relationship between distance, time, and speeds -/
theorem journey_distance : 
  (total_distance / 3) / speed1 + 
  (total_distance / 3) / speed2 + 
  (total_distance / 3) / speed3 = total_time :=
by sorry

/-- Theorem stating that the total distance is approximately 712.46 km -/
theorem total_distance_value : 
  ∃ ε > 0, abs (total_distance - 712.46) < ε :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_total_distance_value_l2482_248231


namespace NUMINAMATH_CALUDE_least_possible_b_in_right_triangle_l2482_248235

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate to check if a number is in the Fibonacci sequence
def is_fibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem least_possible_b_in_right_triangle :
  ∀ a b : ℕ,
  a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
  a > b →  -- a is greater than b
  is_fibonacci a →  -- a is in the Fibonacci sequence
  is_fibonacci b →  -- b is in the Fibonacci sequence
  b ≥ 1 →  -- b is at least 1 (as it's an angle)
  ∀ c : ℕ, (c < b ∧ is_fibonacci c) → c < 1 :=
by sorry

#check least_possible_b_in_right_triangle

end NUMINAMATH_CALUDE_least_possible_b_in_right_triangle_l2482_248235


namespace NUMINAMATH_CALUDE_streetlight_purchase_l2482_248273

theorem streetlight_purchase (squares : Nat) (lights_per_square : Nat) (repair_lights : Nat) (bought_lights : Nat) : 
  squares = 15 → 
  lights_per_square = 12 → 
  repair_lights = 35 → 
  bought_lights = 200 → 
  squares * lights_per_square + repair_lights - bought_lights = 15 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_purchase_l2482_248273


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2482_248238

/-- Parabola 1 equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- Parabola 2 equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-4, 37), (3/2, 9.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, (parabola1 p.1 = parabola2 p.1) ↔ p ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2482_248238


namespace NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l2482_248223

/-- Given a basket of peaches, prove that the total number of green and yellow peaches is 20. -/
theorem green_and_yellow_peaches_count (yellow_peaches green_peaches : ℕ) 
  (h1 : yellow_peaches = 14)
  (h2 : green_peaches = 6) : 
  yellow_peaches + green_peaches = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l2482_248223


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2482_248267

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - x^2 / 25 = 1

/-- The asymptote equation -/
def asymptote (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given equations are the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), hyperbola x y → 
      (asymptote m b x y ∨ asymptote (-m) b x y)) ∧
    m = 4/5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2482_248267


namespace NUMINAMATH_CALUDE_partner_calculation_l2482_248236

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l2482_248236


namespace NUMINAMATH_CALUDE_cubic_identity_l2482_248227

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2482_248227


namespace NUMINAMATH_CALUDE_charles_vowel_learning_time_l2482_248240

/-- The number of days Charles takes to learn one alphabet. -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet. -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels. -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end NUMINAMATH_CALUDE_charles_vowel_learning_time_l2482_248240


namespace NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2482_248298

theorem decimal_digits_of_fraction : ∃ (n : ℚ), 
  n = (5^7 : ℚ) / ((10^5 : ℚ) * 125) ∧ 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (m : ℕ), n = (m : ℚ) / (10^d : ℚ) ∧ 
   m % 10 ≠ 0 ∧ 
   (∀ (k : ℕ), k > d → (m * 10^(k-d)) % 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2482_248298


namespace NUMINAMATH_CALUDE_largest_angle_after_change_l2482_248294

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the initial conditions
def initial_triangle : Triangle :=
  { D := 60, E := 60, F := 60 }

-- Define the angle decrease
def angle_decrease : ℝ := 20

-- Theorem statement
theorem largest_angle_after_change (t : Triangle) :
  t = initial_triangle →
  ∃ (new_t : Triangle),
    new_t.D = t.D - angle_decrease ∧
    new_t.D + new_t.E + new_t.F = 180 ∧
    new_t.E = new_t.F ∧
    max new_t.D (max new_t.E new_t.F) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_after_change_l2482_248294


namespace NUMINAMATH_CALUDE_five_colored_flags_count_l2482_248297

/-- The number of different colors available -/
def num_colors : ℕ := 11

/-- The number of stripes in the flag -/
def num_stripes : ℕ := 5

/-- The number of ways to choose and arrange colors for the flag -/
def num_flags : ℕ := (num_colors.choose num_stripes) * num_stripes.factorial

/-- Theorem stating the number of different five-colored flags -/
theorem five_colored_flags_count : num_flags = 55440 := by
  sorry

end NUMINAMATH_CALUDE_five_colored_flags_count_l2482_248297


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l2482_248271

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y : ℝ), 5 * x^2 + k * y^2 = 5 ↔
      (x^2 / a^2 + y^2 / b^2 = 1 ∧
       c^2 = a^2 - b^2 ∧
       2^2 = c^2)) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l2482_248271


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2482_248277

theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n)
  (h_third_term : a 3 = 9)
  (h_seventh_term : a 7 = 1) :
  a 5 = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2482_248277


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2482_248207

def batsman_average (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem batsman_average_increase
  (prev_total : ℕ)
  (new_score : ℕ)
  (innings : ℕ)
  (avg_increase : ℚ)
  (h1 : innings = 17)
  (h2 : new_score = 88)
  (h3 : avg_increase = 3)
  (h4 : batsman_average (prev_total + new_score) innings - batsman_average prev_total (innings - 1) = avg_increase) :
  batsman_average (prev_total + new_score) innings = 40 :=
by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l2482_248207


namespace NUMINAMATH_CALUDE_intersection_problem_l2482_248213

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_problem (a : ℝ) :
  (a = 1/2 → A a ∩ B = {x | 0 < x ∧ x < 1}) ∧
  (A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_problem_l2482_248213


namespace NUMINAMATH_CALUDE_equation_solution_l2482_248269

theorem equation_solution : ∃ X : ℝ, 
  (1.5 * ((X * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
  (abs (X - 3.6000000000000005) < 1e-10) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2482_248269


namespace NUMINAMATH_CALUDE_roses_unchanged_l2482_248249

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- The initial state of the flower vase -/
def initial_vase : FlowerVase := { roses := 13, orchids := 84 }

/-- The final state of the flower vase -/
def final_vase : FlowerVase := { roses := 13, orchids := 91 }

/-- Theorem stating that the number of roses remains unchanged -/
theorem roses_unchanged (initial : FlowerVase) (final : FlowerVase) 
  (h_initial : initial = initial_vase) 
  (h_final_orchids : final.orchids = 91) :
  final.roses = initial.roses := by sorry

end NUMINAMATH_CALUDE_roses_unchanged_l2482_248249


namespace NUMINAMATH_CALUDE_probability_is_175_323_l2482_248270

-- Define the number of black and white balls
def black_balls : ℕ := 10
def white_balls : ℕ := 9
def total_balls : ℕ := black_balls + white_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_at_least_two_black : ℚ :=
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 +
   Nat.choose black_balls 3) /
  Nat.choose total_balls drawn_balls

-- Theorem statement
theorem probability_is_175_323 :
  probability_at_least_two_black = 175 / 323 :=
by sorry

end NUMINAMATH_CALUDE_probability_is_175_323_l2482_248270


namespace NUMINAMATH_CALUDE_factor_twoOnesWithZeros_l2482_248217

/-- Creates a number with two ones and n zeros between them -/
def twoOnesWithZeros (n : ℕ) : ℕ :=
  10^(n + 1) + 1

/-- The other factor in the decomposition -/
def otherFactor (k : ℕ) : ℕ :=
  (10^(3*k + 3) - 1) / 9999

theorem factor_twoOnesWithZeros (k : ℕ) :
  ∃ (m : ℕ), twoOnesWithZeros (3*k + 2) = 73 * 137 * m :=
sorry

end NUMINAMATH_CALUDE_factor_twoOnesWithZeros_l2482_248217


namespace NUMINAMATH_CALUDE_complement_of_union_l2482_248265

def U : Set ℤ := {x | 0 < x ∧ x ≤ 8}
def M : Set ℤ := {1, 3, 5, 7}
def N : Set ℤ := {5, 6, 7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2482_248265


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2482_248283

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2482_248283


namespace NUMINAMATH_CALUDE_susie_pizza_sales_l2482_248253

/-- Represents the pizza sales scenario --/
structure PizzaSales where
  slice_price : ℕ
  whole_price : ℕ
  slices_sold : ℕ
  total_earnings : ℕ

/-- Calculates the number of whole pizzas sold --/
def whole_pizzas_sold (s : PizzaSales) : ℕ :=
  (s.total_earnings - s.slice_price * s.slices_sold) / s.whole_price

/-- Theorem stating that under the given conditions, 3 whole pizzas were sold --/
theorem susie_pizza_sales :
  let s : PizzaSales := {
    slice_price := 3,
    whole_price := 15,
    slices_sold := 24,
    total_earnings := 117
  }
  whole_pizzas_sold s = 3 := by sorry

end NUMINAMATH_CALUDE_susie_pizza_sales_l2482_248253


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l2482_248210

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numFridays = 5

/-- Function to determine the day of the week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l2482_248210


namespace NUMINAMATH_CALUDE_samsung_tv_cost_l2482_248204

/-- The cost of a Samsung TV based on Latia's work hours and wages -/
theorem samsung_tv_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks : ℕ) (additional_hours : ℕ) : 
  hourly_wage = 10 →
  weekly_hours = 30 →
  weeks = 4 →
  additional_hours = 50 →
  hourly_wage * (weekly_hours * weeks + additional_hours) = 1700 := by
sorry

end NUMINAMATH_CALUDE_samsung_tv_cost_l2482_248204


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l2482_248212

/-- The reflection of a line about the x-axis -/
def reflect_about_x_axis (line : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  λ x y => line x (-y)

/-- The original line -/
def original_line : ℝ → ℝ → Prop :=
  λ x y => x - y + 1 = 0

/-- The reflected line -/
def reflected_line : ℝ → ℝ → Prop :=
  λ x y => x + y + 1 = 0

theorem reflection_about_x_axis :
  reflect_about_x_axis original_line = reflected_line := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l2482_248212


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l2482_248268

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l2482_248268


namespace NUMINAMATH_CALUDE_rod_string_equilibrium_theorem_l2482_248226

/-- Represents the equilibrium conditions for a rod and string system --/
def rod_string_equilibrium (a b : ℝ) (θ : ℝ) : Prop :=
  (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) ∧ a > 0 ∧ b > 0

/-- Theorem stating the equilibrium conditions for the rod and string system --/
theorem rod_string_equilibrium_theorem (a b : ℝ) (θ : ℝ) :
  a > 0 → b > 0 → rod_string_equilibrium a b θ ↔
    (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) :=
by sorry

end NUMINAMATH_CALUDE_rod_string_equilibrium_theorem_l2482_248226


namespace NUMINAMATH_CALUDE_tires_in_parking_lot_parking_lot_tire_count_l2482_248229

/-- The number of tires in a parking lot with four-wheel drive cars and spare tires -/
theorem tires_in_parking_lot (num_cars : ℕ) (wheels_per_car : ℕ) (has_spare : Bool) : ℕ :=
  let regular_tires := num_cars * wheels_per_car
  let spare_tires := if has_spare then num_cars else 0
  regular_tires + spare_tires

/-- Proof that there are 150 tires in the parking lot with 30 four-wheel drive cars and spare tires -/
theorem parking_lot_tire_count :
  tires_in_parking_lot 30 4 true = 150 := by
  sorry

end NUMINAMATH_CALUDE_tires_in_parking_lot_parking_lot_tire_count_l2482_248229


namespace NUMINAMATH_CALUDE_eleven_only_divisor_l2482_248233

theorem eleven_only_divisor : ∃! n : ℕ, 
  (∃ k : ℕ, n = (10^k - 1) / 9) ∧ 
  (∃ m : ℕ, (10^m + 1) % n = 0) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_eleven_only_divisor_l2482_248233


namespace NUMINAMATH_CALUDE_trig_ratio_sum_l2482_248225

theorem trig_ratio_sum (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 3)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_sum_l2482_248225


namespace NUMINAMATH_CALUDE_b_95_mod_49_l2482_248241

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_95_mod_49 : b 95 ≡ 28 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l2482_248241


namespace NUMINAMATH_CALUDE_coin_toss_and_match_probability_l2482_248228

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Head
| Tail

/-- Represents the weather condition during a match -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the result of a match -/
inductive MatchResult
| Draw
| NotDraw

/-- Represents a football match with its associated coin toss, weather, and result -/
structure Match where
  toss : CoinToss
  weather : Weather
  result : MatchResult

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draw_on_heads : ℕ := 7
def rainy_on_tails : ℕ := 4

/-- The main theorem to prove -/
theorem coin_toss_and_match_probability :
  (coin_tosses - heads_count = 14) ∧
  (∀ m : Match, m.toss = CoinToss.Head → m.result = MatchResult.Draw → 
               m.toss = CoinToss.Tail → m.weather = Weather.Rainy → False) :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_and_match_probability_l2482_248228


namespace NUMINAMATH_CALUDE_sector_central_angle_l2482_248239

/-- Given a sector with radius 8 and area 32, prove that its central angle in radians is 1 -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 8) (h2 : area = 32) :
  let α := 2 * area / (r * r)
  α = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2482_248239


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2482_248289

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its eccentricity is √3 and the directrix of the parabola y² = 12x
    passes through one of its foci, then the equation of the hyperbola is
    x²/3 - y²/6 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c / a = Real.sqrt 3 ∧ c = 3) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2482_248289


namespace NUMINAMATH_CALUDE_planes_parallel_if_skew_lines_parallel_l2482_248242

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contains : Plane → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_skew_lines_parallel
  (α β : Plane) (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : lineParallelPlane a β)
  (h4 : lineParallelPlane b α)
  (h5 : skew a b) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_skew_lines_parallel_l2482_248242


namespace NUMINAMATH_CALUDE_mallory_journey_expenses_l2482_248220

theorem mallory_journey_expenses :
  let initial_fuel_cost : ℚ := 45
  let miles_per_tank : ℚ := 500
  let total_miles : ℚ := 2000
  let food_cost_ratio : ℚ := 3/5
  let hotel_nights : ℕ := 3
  let hotel_cost_per_night : ℚ := 80
  let fuel_cost_increase : ℚ := 5

  let num_refills : ℕ := (total_miles / miles_per_tank).ceil.toNat
  let fuel_costs : List ℚ := List.range num_refills |>.map (λ i => initial_fuel_cost + i * fuel_cost_increase)
  let total_fuel_cost : ℚ := fuel_costs.sum
  let food_cost : ℚ := food_cost_ratio * total_fuel_cost
  let hotel_cost : ℚ := hotel_nights * hotel_cost_per_night
  let total_expenses : ℚ := total_fuel_cost + food_cost + hotel_cost

  total_expenses = 576
  := by sorry

end NUMINAMATH_CALUDE_mallory_journey_expenses_l2482_248220


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2482_248299

theorem regular_polygon_sides (b : ℕ) (h : b ≥ 3) : (180 * (b - 2) = 1080) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2482_248299


namespace NUMINAMATH_CALUDE_cube_difference_l2482_248250

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : 
  a^3 - b^3 = 353.5 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l2482_248250


namespace NUMINAMATH_CALUDE_max_value_inequality_l2482_248234

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * x * z + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2482_248234


namespace NUMINAMATH_CALUDE_quadratic_function_and_tangent_line_l2482_248237

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A point x is a zero of function f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- A line y = kx + m is tangent to the graph of f if there exists exactly one point
    where the line touches the graph of f -/
def IsTangent (f : ℝ → ℝ) (k m : ℝ) : Prop :=
  ∃! x, f x = k * x + m

theorem quadratic_function_and_tangent_line 
  (f : ℝ → ℝ) (b c k m : ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c)
  (h2 : IsEven f)
  (h3 : IsZero f 1)
  (h4 : k > 0)
  (h5 : IsTangent f k m) :
  (∀ x, f x = x^2 - 1) ∧ 
  (∀ k m, k > 0 → IsTangent f k m → m * k ≤ -4) ∧
  (∃ k m, k > 0 ∧ IsTangent f k m ∧ m * k = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_tangent_line_l2482_248237


namespace NUMINAMATH_CALUDE_student_grouping_l2482_248284

theorem student_grouping (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 30) (h2 : students_per_group = 5) :
  total_students / students_per_group = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_grouping_l2482_248284


namespace NUMINAMATH_CALUDE_multiplications_in_thirty_minutes_l2482_248203

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 20000

/-- Represents the number of minutes we want to calculate for -/
def minutes : ℕ := 30

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem stating that the computer can perform 36,000,000 multiplications in 30 minutes -/
theorem multiplications_in_thirty_minutes :
  multiplications_per_second * minutes * seconds_per_minute = 36000000 := by
  sorry

end NUMINAMATH_CALUDE_multiplications_in_thirty_minutes_l2482_248203


namespace NUMINAMATH_CALUDE_chris_winning_configurations_l2482_248215

/-- Modified nim-value for a single wall in the brick removal game -/
def modified_nim_value (n : ℕ) : ℕ := sorry

/-- Nim-sum of a list of natural numbers -/
def nim_sum (l : List ℕ) : ℕ := sorry

/-- Represents a game configuration as a list of wall sizes -/
def GameConfig := List ℕ

/-- Determines if Chris (second player) can guarantee a win for a given game configuration -/
def chris_wins (config : GameConfig) : Prop :=
  nim_sum (config.map modified_nim_value) = 0

theorem chris_winning_configurations :
  ∀ config : GameConfig,
    (chris_wins config ↔ 
      (config = [7, 5, 2] ∨ config = [7, 5, 3])) :=
by sorry

end NUMINAMATH_CALUDE_chris_winning_configurations_l2482_248215


namespace NUMINAMATH_CALUDE_unique_solution_l2482_248200

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  x^2*y + x*y^2 + 3*x + 3*y + 24 = 0

def equation2 (x y : ℝ) : Prop :=
  x^3*y - x*y^3 + 3*x^2 - 3*y^2 - 48 = 0

-- Theorem stating that (-3, -1) is the unique solution
theorem unique_solution :
  (∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2) ∧
  (equation1 (-3) (-1) ∧ equation2 (-3) (-1)) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2482_248200


namespace NUMINAMATH_CALUDE_pencil_price_l2482_248275

theorem pencil_price (num_pens num_pencils total_spent pen_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_spent = 450)
  (h4 : pen_avg_price = 10) :
  (total_spent - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l2482_248275


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l2482_248261

/-- Given 4 non-negative integers representing ages, if their mean is 8 and
    their median is 5, then the sum of the smallest and largest of these
    integers is 22. -/
theorem cousins_ages_sum (a b c d : ℕ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d →  -- Sorted in ascending order
  (a + b + c + d) / 4 = 8 →  -- Mean is 8
  (b + c) / 2 = 5 →  -- Median is 5
  a + d = 22 := by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l2482_248261


namespace NUMINAMATH_CALUDE_prob_third_grade_parent_is_three_fifths_l2482_248286

/-- Represents the number of parents in each grade's committee -/
structure ParentCommittee where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Represents the number of parents sampled from each grade -/
structure SampledParents where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Calculates the total number of parents in all committees -/
def totalParents (pc : ParentCommittee) : Nat :=
  pc.grade1 + pc.grade2 + pc.grade3

/-- Calculates the stratified sample for each grade -/
def calculateSample (pc : ParentCommittee) (totalSample : Nat) : SampledParents :=
  let ratio := totalSample / (totalParents pc)
  { grade1 := pc.grade1 * ratio
  , grade2 := pc.grade2 * ratio
  , grade3 := pc.grade3 * ratio }

/-- Calculates the probability of selecting at least one third-grade parent -/
def probThirdGradeParent (sp : SampledParents) : Rat :=
  let totalCombinations := (sp.grade1 + sp.grade2 + sp.grade3).choose 2
  let favorableCombinations := sp.grade3 * (sp.grade1 + sp.grade2) + sp.grade3.choose 2
  favorableCombinations / totalCombinations

theorem prob_third_grade_parent_is_three_fifths 
  (pc : ParentCommittee) 
  (h1 : pc.grade1 = 54)
  (h2 : pc.grade2 = 18)
  (h3 : pc.grade3 = 36)
  (totalSample : Nat)
  (h4 : totalSample = 6) :
  probThirdGradeParent (calculateSample pc totalSample) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_grade_parent_is_three_fifths_l2482_248286


namespace NUMINAMATH_CALUDE_todd_ingredient_cost_l2482_248218

/-- Represents the financial details of Todd's snow-cone business --/
structure SnowConeBusiness where
  borrowed : ℝ
  repay : ℝ
  snowConesSold : ℕ
  pricePerSnowCone : ℝ
  remainingAfterRepay : ℝ

/-- Calculates the amount spent on ingredients for the snow-cone business --/
def ingredientCost (business : SnowConeBusiness) : ℝ :=
  business.borrowed + business.snowConesSold * business.pricePerSnowCone - business.repay - business.remainingAfterRepay

/-- Theorem stating that Todd spent $25 on ingredients --/
theorem todd_ingredient_cost :
  let business : SnowConeBusiness := {
    borrowed := 100,
    repay := 110,
    snowConesSold := 200,
    pricePerSnowCone := 0.75,
    remainingAfterRepay := 65
  }
  ingredientCost business = 25 := by
  sorry


end NUMINAMATH_CALUDE_todd_ingredient_cost_l2482_248218


namespace NUMINAMATH_CALUDE_noemi_initial_amount_l2482_248264

/-- Calculates the initial amount of money Noemi had before gambling --/
def initial_amount (roulette_loss blackjack_loss poker_loss baccarat_loss purse_left : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + poker_loss + baccarat_loss + purse_left

/-- Proves that Noemi's initial amount is correct given her losses and remaining money --/
theorem noemi_initial_amount : 
  initial_amount 600 800 400 700 1500 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_amount_l2482_248264


namespace NUMINAMATH_CALUDE_connor_date_expense_l2482_248245

/-- The total amount Connor spends on his movie date -/
def connor_total_spent (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price + cup_price

/-- Theorem: Connor spends $49.00 on his movie date -/
theorem connor_date_expense :
  connor_total_spent 14 11 2.5 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_connor_date_expense_l2482_248245


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l2482_248211

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 25*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l2482_248211


namespace NUMINAMATH_CALUDE_escalator_length_is_210_l2482_248224

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Theorem stating that under the given conditions, the escalator length is 210 feet. -/
theorem escalator_length_is_210 :
  escalator_length 12 2 15 = 210 := by
  sorry

#eval escalator_length 12 2 15

end NUMINAMATH_CALUDE_escalator_length_is_210_l2482_248224


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_66_l2482_248247

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold_is_66 : 
  (books_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 66 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sold_is_66_l2482_248247


namespace NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l2482_248221

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 ≤ 5) ∨
               (5 = y - 6 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 6 ∧ 5 ≤ x + 3)}

/-- The three rays that make up set S -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 11 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≥ 2}

/-- The common point of the three rays -/
def commonPoint : ℝ × ℝ := (2, 11)

theorem S_is_three_rays_with_common_point :
  S = ray1 ∪ ray2 ∪ ray3 ∧
  ray1 ∩ ray2 ∩ ray3 = {commonPoint} :=
by sorry

end NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l2482_248221


namespace NUMINAMATH_CALUDE_first_five_terms_of_sequence_l2482_248256

def a (n : ℕ) : ℤ := (-1: ℤ)^n + n

theorem first_five_terms_of_sequence :
  (a 1 = 0) ∧ (a 2 = 3) ∧ (a 3 = 2) ∧ (a 4 = 5) ∧ (a 5 = 4) :=
by sorry

end NUMINAMATH_CALUDE_first_five_terms_of_sequence_l2482_248256
