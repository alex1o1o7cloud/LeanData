import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1940_194097

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(p + 1)
  let c := (p^2 + 2*p - 3) / 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = Real.sqrt (2*p + 1 - p^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1940_194097


namespace NUMINAMATH_CALUDE_dress_making_hours_l1940_194053

/-- Calculates the total hours required to make dresses given the available fabric, fabric per dress, and time per dress. -/
def total_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (time_per_dress : ℕ) : ℕ :=
  (total_fabric / fabric_per_dress) * time_per_dress

/-- Proves that given 56 square meters of fabric, where each dress requires 4 square meters of fabric
    and 3 hours to make, the total number of hours required to make all possible dresses is 42 hours. -/
theorem dress_making_hours : total_hours 56 4 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_dress_making_hours_l1940_194053


namespace NUMINAMATH_CALUDE_compare_negative_roots_l1940_194038

theorem compare_negative_roots : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_roots_l1940_194038


namespace NUMINAMATH_CALUDE_x_equals_y_when_q_is_seven_l1940_194094

theorem x_equals_y_when_q_is_seven :
  ∀ (q : ℤ), 
  let x := 55 + 2 * q
  let y := 4 * q + 41
  q = 7 → x = y :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_when_q_is_seven_l1940_194094


namespace NUMINAMATH_CALUDE_age_of_a_l1940_194006

/-- Given the ages of four people a, b, c, and d, prove that the age of a is 11 years. -/
theorem age_of_a (A B C D : ℕ) : 
  A + B + C + D = 76 →
  ∃ (k : ℕ), A - 3 = k ∧ B - 3 = 2 * k ∧ C - 3 = 3 * k →
  ∃ (m : ℕ), A - 5 = 3 * m ∧ D - 5 = 4 * m ∧ B - 5 = 5 * m →
  A = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_of_a_l1940_194006


namespace NUMINAMATH_CALUDE_percentage_decrease_l1940_194096

theorem percentage_decrease (x y z : ℝ) 
  (h1 : x = 1.2 * y) 
  (h2 : x = 0.6 * z) : 
  y = 0.5 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1940_194096


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1940_194044

theorem simplify_trig_expression : 
  Real.sqrt (1 + Real.sin 10) + Real.sqrt (1 - Real.sin 10) = -2 * Real.sin 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1940_194044


namespace NUMINAMATH_CALUDE_smaller_number_l1940_194074

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 10) : 
  min x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_l1940_194074


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1940_194009

theorem tangent_line_to_circle (c : ℝ) : 
  (c > 0) → 
  (∀ x y : ℝ, x^2 + y^2 = 8 → (x + y = c → (x - 0)^2 + (y - 0)^2 = 8)) → 
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1940_194009


namespace NUMINAMATH_CALUDE_harry_worked_36_hours_l1940_194037

/-- Payment structure for Harry and James -/
structure PaymentStructure where
  base_rate : ℝ
  harry_base_hours : ℕ := 30
  harry_overtime_rate : ℝ := 2
  james_base_hours : ℕ := 40
  james_overtime_rate : ℝ := 1.5

/-- Calculate pay for a given number of hours worked -/
def calculate_pay (ps : PaymentStructure) (base_hours : ℕ) (overtime_rate : ℝ) (hours_worked : ℕ) : ℝ :=
  if hours_worked ≤ base_hours then
    ps.base_rate * hours_worked
  else
    ps.base_rate * base_hours + ps.base_rate * overtime_rate * (hours_worked - base_hours)

/-- Theorem stating that Harry worked 36 hours if paid the same as James who worked 41 hours -/
theorem harry_worked_36_hours (ps : PaymentStructure) :
  calculate_pay ps ps.james_base_hours ps.james_overtime_rate 41 =
  calculate_pay ps ps.harry_base_hours ps.harry_overtime_rate 36 :=
sorry

end NUMINAMATH_CALUDE_harry_worked_36_hours_l1940_194037


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1940_194003

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1940_194003


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1940_194095

theorem gcd_of_three_numbers : Nat.gcd 18222 (Nat.gcd 24546 66364) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1940_194095


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l1940_194001

/-- Two points on a parabola with tangents intersecting at 45° -/
structure ParabolaPoints where
  a : ℝ
  b : ℝ

/-- The parabola y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- Tangent slope at a point on the parabola -/
def tangentSlope (x : ℝ) : ℝ := 8 * x

/-- Condition for tangents intersecting at 45° -/
def tangentAngle45 (p : ParabolaPoints) : Prop :=
  |((tangentSlope p.a - tangentSlope p.b) / (1 + tangentSlope p.a * tangentSlope p.b))| = 1

/-- Y-coordinate of the intersection point of tangents -/
def intersectionY (p : ParabolaPoints) : ℝ := 4 * p.a * p.b

theorem parabola_tangent_intersection
  (p : ParabolaPoints)
  (h1 : parabola p.a = 4 * p.a^2)
  (h2 : parabola p.b = 4 * p.b^2)
  (h3 : tangentAngle45 p) :
  intersectionY p = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l1940_194001


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l1940_194057

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l1940_194057


namespace NUMINAMATH_CALUDE_lunch_cost_l1940_194076

def weekly_savings : ℝ := 50
def total_weeks : ℕ := 5
def final_savings : ℝ := 135

theorem lunch_cost (lunch_frequency : ℕ) (lunch_cost : ℝ) : 
  lunch_frequency = 2 ∧ 
  lunch_cost * (total_weeks / lunch_frequency) = weekly_savings * total_weeks - final_savings →
  lunch_cost = 57.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_l1940_194076


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equation_l1940_194022

theorem sum_of_reciprocal_equation (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4)
  (eq2 : 1 / x - 1 / y = -8) :
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equation_l1940_194022


namespace NUMINAMATH_CALUDE_inequality_proof_l1940_194004

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 < 4*b) (h2 : c^2 < 4*d) : 
  ((a + c)/2)^2 < 4*((b + d)/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1940_194004


namespace NUMINAMATH_CALUDE_partner_A_investment_l1940_194039

/-- Calculates the investment of partner A in a business partnership --/
theorem partner_A_investment
  (b_investment : ℕ)
  (c_investment : ℕ)
  (total_profit : ℕ)
  (a_profit_share : ℕ)
  (h1 : b_investment = 4200)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12200)
  (h4 : a_profit_share = 3660) :
  ∃ a_investment : ℕ,
    a_investment = 6725 ∧
    a_investment * total_profit = a_profit_share * (a_investment + b_investment + c_investment) :=
by
  sorry


end NUMINAMATH_CALUDE_partner_A_investment_l1940_194039


namespace NUMINAMATH_CALUDE_order_of_numbers_l1940_194068

theorem order_of_numbers : 7^(3/10) > (3/10)^7 ∧ (3/10)^7 > Real.log (3/10) := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1940_194068


namespace NUMINAMATH_CALUDE_certain_number_proof_l1940_194048

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x > 0.25 * 16 + 2) → (0.15 * x = 6) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1940_194048


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1940_194060

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the condition p
def condition_p (m : ℝ) : Prop :=
  ∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

-- Define the condition q
def condition_q (m : ℝ) : Prop := m ≥ -4/3

-- Theorem statement
theorem sufficient_not_necessary :
  (∃ m, condition_p m ∧ ¬condition_q m) ∧
  (∀ m, condition_p m → condition_q m) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1940_194060


namespace NUMINAMATH_CALUDE_f_at_4_l1940_194085

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f : List ℤ := [11, -9, 7, -5, 3, 1]

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : horner f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l1940_194085


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l1940_194091

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l1940_194091


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l1940_194040

-- Define the repeating decimals
def repeating_038 : ℚ := 38 / 999
def repeating_4 : ℚ := 4 / 9

-- State the theorem
theorem product_of_repeating_decimals :
  repeating_038 * repeating_4 = 152 / 8991 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l1940_194040


namespace NUMINAMATH_CALUDE_yolanda_free_throws_l1940_194007

/-- Calculates the average number of free throws per game given the total points,
    number of games, and average two-point and three-point baskets per game. -/
def avg_free_throws (total_points : ℕ) (num_games : ℕ) 
                    (avg_two_point : ℕ) (avg_three_point : ℕ) : ℕ :=
  let avg_points_per_game := total_points / num_games
  let points_from_two_point := avg_two_point * 2
  let points_from_three_point := avg_three_point * 3
  avg_points_per_game - (points_from_two_point + points_from_three_point)

theorem yolanda_free_throws : 
  avg_free_throws 345 15 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_free_throws_l1940_194007


namespace NUMINAMATH_CALUDE_flight_distance_difference_l1940_194042

def beka_flights : List Nat := [425, 320, 387]
def jackson_flights : List Nat := [250, 170, 353, 201]

theorem flight_distance_difference :
  (List.sum beka_flights) - (List.sum jackson_flights) = 158 := by
  sorry

end NUMINAMATH_CALUDE_flight_distance_difference_l1940_194042


namespace NUMINAMATH_CALUDE_sequence_increasing_l1940_194005

/-- Given positive real numbers a, b, c, and a natural number n,
    prove that a_n < a_{n+1} where a_n = (a*n)/(b*n + c) -/
theorem sequence_increasing (a b c : ℝ) (n : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    let a_n := (a * n) / (b * n + c)
    let a_n_plus_1 := (a * (n + 1)) / (b * (n + 1) + c)
    a_n < a_n_plus_1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l1940_194005


namespace NUMINAMATH_CALUDE_range_of_S_3_l1940_194045

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem range_of_S_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a2 : a 2 = 2) :
  ∀ x : ℝ, (x ∈ Set.Iic (-2) ∪ Set.Ici 6) ↔ ∃ q : ℝ, q ≠ 0 ∧ S_3 a = x :=
sorry

end NUMINAMATH_CALUDE_range_of_S_3_l1940_194045


namespace NUMINAMATH_CALUDE_rotate_point_A_l1940_194026

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_point_A :
  let A : ℝ × ℝ := (-4, 1)
  rotate90Clockwise A = (1, 4) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_l1940_194026


namespace NUMINAMATH_CALUDE_tv_production_average_l1940_194050

/-- Calculates the daily average TV production for a month given the production rates for two periods. -/
def monthly_average (first_period_days : ℕ) (first_period_avg : ℚ) 
                    (second_period_days : ℕ) (second_period_avg : ℚ) : ℚ :=
  (first_period_days * first_period_avg + second_period_days * second_period_avg) / 
  (first_period_days + second_period_days)

/-- Theorem stating that given the specific production rates, the monthly average is 58 TVs/day. -/
theorem tv_production_average : 
  monthly_average 25 63 5 33 = 58 := by sorry

end NUMINAMATH_CALUDE_tv_production_average_l1940_194050


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_expected_worth_is_zero_l1940_194065

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : ℝ :=
  let p_heads : ℝ := 2/3
  let p_tails : ℝ := 1/3
  let gain_heads : ℝ := 5
  let loss_tails : ℝ := 10
  p_heads * gain_heads + p_tails * (-loss_tails)

/-- Proof that the expected worth of the unfair coin flip is 0 -/
theorem expected_worth_is_zero : expected_worth_unfair_coin = 0 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_expected_worth_is_zero_l1940_194065


namespace NUMINAMATH_CALUDE_park_outer_diameter_l1940_194086

/-- Represents the structure of a circular park with concentric regions -/
structure CircularPark where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 48 feet -/
theorem park_outer_diameter :
  let park : CircularPark := {
    fountain_diameter := 10,
    garden_width := 12,
    inner_path_width := 3,
    outer_path_width := 4
  }
  outer_boundary_diameter park = 48 := by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l1940_194086


namespace NUMINAMATH_CALUDE_minimum_score_for_raised_average_l1940_194083

def current_scores : List ℝ := [88, 92, 75, 85, 80]
def raise_average : ℝ := 5

theorem minimum_score_for_raised_average 
  (scores : List ℝ) 
  (raise : ℝ) 
  (h1 : scores = current_scores) 
  (h2 : raise = raise_average) : 
  (scores.sum + (scores.length + 1) * (scores.sum / scores.length + raise) - scores.sum) = 114 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_for_raised_average_l1940_194083


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1940_194067

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (5 * Real.sqrt 6) = 30 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1940_194067


namespace NUMINAMATH_CALUDE_prime_factorization_2020_2021_l1940_194072

theorem prime_factorization_2020_2021 :
  (2020 = 2^2 * 5 * 101) ∧ (2021 = 43 * 47) := by sorry

end NUMINAMATH_CALUDE_prime_factorization_2020_2021_l1940_194072


namespace NUMINAMATH_CALUDE_marksmen_hit_probability_l1940_194036

theorem marksmen_hit_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.6) (h2 : p2 = 0.7) (h3 : p3 = 0.75) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.97 := by
  sorry

end NUMINAMATH_CALUDE_marksmen_hit_probability_l1940_194036


namespace NUMINAMATH_CALUDE_singer_hourly_rate_l1940_194018

/-- Given a singer hired for 3 hours, with a 20% tip, and a total payment of $54, 
    the hourly rate for the singer is $15. -/
theorem singer_hourly_rate (hours : ℕ) (tip_percentage : ℚ) (total_payment : ℚ) :
  hours = 3 →
  tip_percentage = 1/5 →
  total_payment = 54 →
  ∃ (hourly_rate : ℚ), 
    hourly_rate * hours * (1 + tip_percentage) = total_payment ∧
    hourly_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_singer_hourly_rate_l1940_194018


namespace NUMINAMATH_CALUDE_complex_division_l1940_194000

theorem complex_division (i : ℂ) (h : i * i = -1) : (4 * i) / (1 + i) = 2 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1940_194000


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_for_problem_config_l1940_194010

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration described in the problem -/
def problemConfig : IntersectingCones :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  }

theorem max_sphere_radius_squared_for_problem_config :
  maxSphereRadiusSquared problemConfig = 4176 / 841 :=
sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_for_problem_config_l1940_194010


namespace NUMINAMATH_CALUDE_total_supervisors_is_21_l1940_194008

/-- The number of buses used for the field trip. -/
def num_buses : ℕ := 7

/-- The number of adult supervisors per bus. -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors for the field trip. -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

/-- Theorem stating that the total number of supervisors is 21. -/
theorem total_supervisors_is_21 : total_supervisors = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_is_21_l1940_194008


namespace NUMINAMATH_CALUDE_sqrt_3_expression_simplification_l1940_194016

theorem sqrt_3_expression_simplification :
  Real.sqrt 3 * (Real.sqrt 3 - 2) - Real.sqrt 12 / Real.sqrt 3 + |2 - Real.sqrt 3| = 3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_expression_simplification_l1940_194016


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l1940_194079

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y : ℝ, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l1940_194079


namespace NUMINAMATH_CALUDE_routes_in_3x3_grid_l1940_194030

/-- The number of different routes in a 3x3 grid from top-left to bottom-right -/
def numRoutes : ℕ := 20

/-- The size of the grid -/
def gridSize : ℕ := 3

/-- The total number of moves required to reach the destination -/
def totalMoves : ℕ := gridSize * 2

/-- The number of moves in one direction (either right or down) -/
def movesInOneDirection : ℕ := gridSize

theorem routes_in_3x3_grid :
  numRoutes = Nat.choose totalMoves movesInOneDirection := by sorry

end NUMINAMATH_CALUDE_routes_in_3x3_grid_l1940_194030


namespace NUMINAMATH_CALUDE_cosine_equality_degrees_l1940_194047

theorem cosine_equality_degrees (n : ℤ) : ∃ n : ℤ, 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) ∧ n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_degrees_l1940_194047


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1940_194049

theorem boys_to_girls_ratio 
  (total_students : ℕ) 
  (girls : ℕ) 
  (h1 : total_students = 546) 
  (h2 : girls = 210) : 
  (total_students - girls) / girls = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1940_194049


namespace NUMINAMATH_CALUDE_worker_savings_fraction_l1940_194031

/-- A worker saves a constant fraction of her constant monthly take-home pay. -/
structure Worker where
  /-- Monthly take-home pay -/
  P : ℝ
  /-- Fraction of monthly take-home pay saved -/
  f : ℝ
  /-- Monthly take-home pay is positive -/
  P_pos : P > 0
  /-- Savings fraction is between 0 and 1 -/
  f_range : 0 ≤ f ∧ f ≤ 1

/-- The theorem stating that if a worker's yearly savings equals 8 times
    her monthly non-savings, then she saves 2/5 of her income. -/
theorem worker_savings_fraction (w : Worker) 
    (h : 12 * w.f * w.P = 8 * (1 - w.f) * w.P) : 
    w.f = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_fraction_l1940_194031


namespace NUMINAMATH_CALUDE_trapezoid_median_equals_nine_inches_l1940_194058

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 18 inches, then the median of the trapezoid is 9 inches. -/
theorem trapezoid_median_equals_nine_inches 
  (triangle_area trapezoid_area : ℝ) 
  (triangle_altitude trapezoid_altitude : ℝ) 
  (triangle_base : ℝ) 
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_altitude = trapezoid_altitude →
  triangle_base = 18 →
  triangle_area = (1/2) * triangle_base * triangle_altitude →
  trapezoid_area = trapezoid_median * trapezoid_altitude →
  trapezoid_median = 9 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_equals_nine_inches_l1940_194058


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l1940_194093

def f (x : ℝ) := x^2 - 2*x - 10

theorem fixed_points_of_f :
  ∀ x : ℝ, f x = x ↔ x = -2 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l1940_194093


namespace NUMINAMATH_CALUDE_total_distance_calculation_l1940_194013

/-- Calculates the total distance driven by Darius, Julia, and Thomas in miles and kilometers -/
def total_distance (darius_miles : ℝ) (julia_miles : ℝ) (thomas_miles : ℝ) (detour_miles : ℝ) (km_per_mile : ℝ) : ℝ × ℝ :=
  let darius_total := darius_miles * 2 + detour_miles
  let julia_total := julia_miles * 2 + detour_miles
  let thomas_total := thomas_miles * 2
  let total_miles := darius_total + julia_total + thomas_total
  let total_km := total_miles * km_per_mile
  (total_miles, total_km)

theorem total_distance_calculation :
  total_distance 679 998 1205 120 1.60934 = (6004, 9665.73616) := by
  sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l1940_194013


namespace NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1940_194011

/-- Given a book with 354 pages, if a person reads 63 pages on day one,
    twice that amount on day two, and 10 more pages than day two on day three,
    then the number of pages read on day four is 29. -/
theorem pages_read_on_fourth_day
  (total_pages : ℕ)
  (pages_day_one : ℕ)
  (h1 : total_pages = 354)
  (h2 : pages_day_one = 63)
  : total_pages - pages_day_one - (2 * pages_day_one) - (2 * pages_day_one + 10) = 29 := by
  sorry

#check pages_read_on_fourth_day

end NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1940_194011


namespace NUMINAMATH_CALUDE_arflaser_wavelength_scientific_notation_l1940_194051

theorem arflaser_wavelength_scientific_notation :
  ∀ (wavelength : ℝ),
  wavelength = 0.000000193 →
  ∃ (a : ℝ) (n : ℤ),
    wavelength = a * (10 : ℝ) ^ n ∧
    1 ≤ a ∧ a < 10 ∧
    a = 1.93 ∧ n = -7 :=
by sorry

end NUMINAMATH_CALUDE_arflaser_wavelength_scientific_notation_l1940_194051


namespace NUMINAMATH_CALUDE_rs_length_l1940_194046

/-- Triangle ABC with altitude CH, inscribed circles tangent points R and S --/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  M : ℝ × ℝ
  -- CH is altitude
  altitude : (C.1 - H.1) * (A.1 - B.1) + (C.2 - H.2) * (A.2 - B.2) = 0
  -- R and S are on CH
  r_on_ch : ∃ t : ℝ, R = (1 - t) • C + t • H
  s_on_ch : ∃ t : ℝ, S = (1 - t) • C + t • H
  -- M is midpoint of AB
  midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Given side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 21
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 29

/-- The length of RS in the special triangle is 4 --/
theorem rs_length (t : SpecialTriangle) : Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_rs_length_l1940_194046


namespace NUMINAMATH_CALUDE_proposition_false_negation_true_l1940_194019

theorem proposition_false_negation_true :
  (¬ (∀ x y : ℝ, x + y > 0 → x > 0 ∧ y > 0)) ∧
  (∃ x y : ℝ, x + y > 0 ∧ (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_false_negation_true_l1940_194019


namespace NUMINAMATH_CALUDE_dividend_calculation_l1940_194029

theorem dividend_calculation (quotient divisor k : ℕ) 
  (h1 : quotient = 4)
  (h2 : divisor = k)
  (h3 : k = 4) :
  quotient * divisor = 16 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1940_194029


namespace NUMINAMATH_CALUDE_distance_difference_after_three_hours_l1940_194012

/-- Represents a cyclist with a constant cycling rate -/
structure Cyclist where
  name : String
  rate : ℝ  -- cycling rate in miles per hour

/-- Calculates the distance traveled by a cyclist in a given time -/
def distanceTraveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.rate * time

/-- Proves that the difference in distance traveled between Carlos and Diana after 3 hours is 15 miles -/
theorem distance_difference_after_three_hours 
  (carlos : Cyclist)
  (diana : Cyclist)
  (h1 : carlos.rate = 20)
  (h2 : diana.rate = 15)
  : distanceTraveled carlos 3 - distanceTraveled diana 3 = 15 := by
  sorry

#check distance_difference_after_three_hours

end NUMINAMATH_CALUDE_distance_difference_after_three_hours_l1940_194012


namespace NUMINAMATH_CALUDE_problem_1_l1940_194027

theorem problem_1 (x : ℝ) : x^4 * x^3 * x - (x^4)^2 + (-2*x)^3 = -8*x^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1940_194027


namespace NUMINAMATH_CALUDE_skaters_practice_hours_l1940_194043

/-- Represents the practice hours for each skater -/
structure SkaterHours where
  hannah_weekend : ℕ
  hannah_weekday : ℕ
  sarah_weekday : ℕ
  sarah_weekend : ℕ
  emma_weekday : ℕ
  emma_weekend : ℕ

/-- Calculates the total practice hours for all skaters -/
def total_practice_hours (hours : SkaterHours) : ℕ :=
  hours.hannah_weekend + hours.hannah_weekday +
  hours.sarah_weekday + hours.sarah_weekend +
  hours.emma_weekday + hours.emma_weekend

/-- Theorem stating the total practice hours for the skaters -/
theorem skaters_practice_hours :
  ∃ (hours : SkaterHours),
    hours.hannah_weekend = 8 ∧
    hours.hannah_weekday = hours.hannah_weekend + 17 ∧
    hours.sarah_weekday = 12 ∧
    hours.sarah_weekend = 6 ∧
    hours.emma_weekday = 2 * hours.sarah_weekday ∧
    hours.emma_weekend = hours.sarah_weekend + 5 ∧
    total_practice_hours hours = 86 := by
  sorry

end NUMINAMATH_CALUDE_skaters_practice_hours_l1940_194043


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_representation_l1940_194059

theorem divisibility_implies_sum_representation (n k : ℕ) 
  (h1 : n > 20) 
  (h2 : k > 1) 
  (h3 : k^2 ∣ n) : 
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_representation_l1940_194059


namespace NUMINAMATH_CALUDE_locus_equation_l1940_194080

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₃ -/
def locus_of_centers (C₁ C₃ : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ r : ℝ,
    (Circle.mk p r).radius + C₁.radius = Real.sqrt ((p.1 - C₁.center.1)^2 + (p.2 - C₁.center.2)^2) ∧
    C₃.radius - (Circle.mk p r).radius = Real.sqrt ((p.1 - C₃.center.1)^2 + (p.2 - C₃.center.2)^2)}

theorem locus_equation (C₁ C₃ : Circle)
  (h₁ : C₁.center = (0, 0) ∧ C₁.radius = 2)
  (h₃ : C₃.center = (3, 0) ∧ C₃.radius = 5) :
  locus_of_centers C₁ C₃ = {p : ℝ × ℝ | p.1^2 + 7*p.2^2 - 34*p.1 - 57 = 0} :=
sorry

end NUMINAMATH_CALUDE_locus_equation_l1940_194080


namespace NUMINAMATH_CALUDE_cookie_count_l1940_194063

/-- The number of edible cookies at the end of Alice and Bob's baking session -/
def total_edible_cookies (alice_initial : ℕ) (bob_initial : ℕ) (thrown_away : ℕ) (alice_additional : ℕ) (bob_additional : ℕ) : ℕ :=
  alice_initial + bob_initial - thrown_away + alice_additional + bob_additional

/-- Theorem stating the total number of edible cookies at the end -/
theorem cookie_count : total_edible_cookies 74 7 29 5 36 = 93 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l1940_194063


namespace NUMINAMATH_CALUDE_dorchester_washed_16_puppies_l1940_194054

/-- Represents the number of puppies washed by Dorchester on Wednesday -/
def puppies_washed : ℕ := sorry

/-- Dorchester's daily base pay in cents -/
def daily_base_pay : ℕ := 4000

/-- Pay per puppy washed in cents -/
def pay_per_puppy : ℕ := 225

/-- Total earnings on Wednesday in cents -/
def total_earnings : ℕ := 7600

/-- Theorem stating that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_washed_16_puppies : 
  puppies_washed = 16 ∧
  total_earnings = daily_base_pay + puppies_washed * pay_per_puppy :=
sorry

end NUMINAMATH_CALUDE_dorchester_washed_16_puppies_l1940_194054


namespace NUMINAMATH_CALUDE_delegate_grouping_exists_l1940_194041

/-- Represents a delegate with their country and seating position. -/
structure Delegate where
  country : Fin 50
  position : Fin 100

/-- Represents a seating arrangement of delegates around a circular table. -/
def SeatingArrangement := Fin 100 → Delegate

/-- Represents a grouping of delegates. -/
def Grouping := Delegate → Bool

/-- Checks if a delegate has at most one neighbor in the same group. -/
def atMostOneNeighborInGroup (s : SeatingArrangement) (g : Grouping) (d : Delegate) : Prop :=
  let leftNeighbor := s ((d.position - 1 + 100) % 100)
  let rightNeighbor := s ((d.position + 1) % 100)
  ¬(g leftNeighbor ∧ g rightNeighbor ∧ g d = g leftNeighbor ∧ g d = g rightNeighbor)

/-- Main theorem statement -/
theorem delegate_grouping_exists (s : SeatingArrangement) :
  ∃ g : Grouping,
    (∀ c : Fin 50, ∃! d : Delegate, g d = true ∧ d.country = c) ∧
    (∀ c : Fin 50, ∃! d : Delegate, g d = false ∧ d.country = c) ∧
    (∀ d : Delegate, atMostOneNeighborInGroup s g d) :=
  sorry

end NUMINAMATH_CALUDE_delegate_grouping_exists_l1940_194041


namespace NUMINAMATH_CALUDE_train_passing_time_l1940_194023

/-- Proves that a train passing a platform in given time and speed will pass a stationary point in approximately 20 seconds -/
theorem train_passing_time (platform_length : ℝ) (platform_passing_time : ℝ) (train_speed_kmh : ℝ) 
  (h1 : platform_length = 360.0288)
  (h2 : platform_passing_time = 44)
  (h3 : train_speed_kmh = 54) : 
  ∃ (time : ℝ), abs (time - 20) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1940_194023


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1940_194075

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 1) → 
  (1^2 - 2*1 + m = 1) → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1940_194075


namespace NUMINAMATH_CALUDE_regular_tetrahedron_vertices_and_edges_l1940_194033

/-- A regular tetrahedron is a regular triangular pyramid -/
structure RegularTetrahedron where
  is_regular_triangular_pyramid : Bool

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- Theorem stating that a regular tetrahedron has 4 vertices and 6 edges -/
theorem regular_tetrahedron_vertices_and_edges (t : RegularTetrahedron) :
  num_vertices t = 4 ∧ num_edges t = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_vertices_and_edges_l1940_194033


namespace NUMINAMATH_CALUDE_sum_of_interchanged_digits_divisible_by_11_l1940_194032

theorem sum_of_interchanged_digits_divisible_by_11 (a b : ℕ) 
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : ℕ, (10 * a + b) + (10 * b + a) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interchanged_digits_divisible_by_11_l1940_194032


namespace NUMINAMATH_CALUDE_missing_number_exists_l1940_194099

theorem missing_number_exists : ∃ x : ℝ, (1 / ((1 / 0.03) + (1 / x))) = 0.02775 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_exists_l1940_194099


namespace NUMINAMATH_CALUDE_ted_peeling_time_l1940_194024

/-- The time it takes Julie to peel potatoes individually (in hours) -/
def julie_time : ℝ := 10

/-- The time Julie and Ted work together (in hours) -/
def together_time : ℝ := 4

/-- The time it takes Julie to complete the task after Ted leaves (in hours) -/
def julie_remaining_time : ℝ := 0.9999999999999998

/-- The time it takes Ted to peel potatoes individually (in hours) -/
def ted_time : ℝ := 8

/-- Theorem stating that given the conditions, Ted's individual time to peel potatoes is 8 hours -/
theorem ted_peeling_time :
  (together_time * (1 / julie_time + 1 / ted_time)) + (julie_remaining_time * (1 / julie_time)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ted_peeling_time_l1940_194024


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l1940_194098

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 21 → (c + d) / 2 = 43.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l1940_194098


namespace NUMINAMATH_CALUDE_jan_miles_difference_l1940_194077

-- Define variables
variable (t s : ℝ)

-- Define distances
def ian_distance : ℝ := s * t
def han_distance : ℝ := (s + 10) * (t + 2)
def jan_distance : ℝ := (s + 15) * (t + 3)

-- State the theorem
theorem jan_miles_difference :
  han_distance t s = ian_distance t s + 120 →
  jan_distance t s = ian_distance t s + 195 := by
  sorry

end NUMINAMATH_CALUDE_jan_miles_difference_l1940_194077


namespace NUMINAMATH_CALUDE_altitude_length_of_triangle_on_square_diagonal_l1940_194070

/-- Given a square with side length s and a triangle constructed on its diagonal
    with an area half that of the square, the length of the altitude drawn to the
    diagonal is s√2/2. -/
theorem altitude_length_of_triangle_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let triangle_area := square_area / 2
  let altitude := triangle_area * 2 / diagonal
  altitude = s * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_altitude_length_of_triangle_on_square_diagonal_l1940_194070


namespace NUMINAMATH_CALUDE_harmonic_quadratic_radical_simplification_l1940_194055

theorem harmonic_quadratic_radical_simplification :
  ∃ (x y : ℕ+), (x + y : ℝ) = 11 ∧ (x * y : ℝ) = 28 →
  Real.sqrt (11 + 2 * Real.sqrt 28) = 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_quadratic_radical_simplification_l1940_194055


namespace NUMINAMATH_CALUDE_min_common_roots_quadratic_trinomials_l1940_194025

theorem min_common_roots_quadratic_trinomials 
  (n : ℕ) 
  (f : Fin n → ℝ → ℝ) 
  (h1 : n = 1004)
  (h2 : ∀ i : Fin n, ∃ a b c : ℝ, ∀ x, f i x = x^2 + a*x + b)
  (h3 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 2007 → ∃ i : Fin n, ∃ x : ℝ, f i x = 0 ∧ x = k)
  : (∀ i j : Fin n, i ≠ j → ∀ x : ℝ, f i x ≠ f j x) :=
sorry

end NUMINAMATH_CALUDE_min_common_roots_quadratic_trinomials_l1940_194025


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1940_194017

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1940_194017


namespace NUMINAMATH_CALUDE_negative_product_plus_two_l1940_194078

theorem negative_product_plus_two :
  ∀ (a b : ℤ), a = -2 → b = -3 → a * b + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_plus_two_l1940_194078


namespace NUMINAMATH_CALUDE_touching_circles_perimeter_l1940_194087

/-- Given a circle with center O and radius R, and two smaller circles with centers O₁ and O₂
    that touch each other and internally touch the larger circle,
    the perimeter of triangle OO₁O₂ is 2R. -/
theorem touching_circles_perimeter (O O₁ O₂ : ℝ × ℝ) (R : ℝ) :
  (∃ R₁ R₂ : ℝ, 
    R₁ > 0 ∧ R₂ > 0 ∧
    dist O O₁ = R - R₁ ∧
    dist O O₂ = R - R₂ ∧
    dist O₁ O₂ = R₁ + R₂) →
  dist O O₁ + dist O O₂ + dist O₁ O₂ = 2 * R :=
by sorry


end NUMINAMATH_CALUDE_touching_circles_perimeter_l1940_194087


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l1940_194069

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the condition for the distance between foci
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

-- Define the range of n
def n_range (n : ℝ) : Prop :=
  -1 < n ∧ n < 3

-- Theorem statement
theorem hyperbola_n_range :
  ∀ m n : ℝ,
  (∃ x y : ℝ, hyperbola_equation x y m n) →
  foci_distance m n →
  n_range n :=
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l1940_194069


namespace NUMINAMATH_CALUDE_two_prime_factors_phi_tau_equality_l1940_194015

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number has exactly two distinct prime factors -/
def has_two_distinct_prime_factors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem two_prime_factors_phi_tau_equality (n : ℕ) :
  has_two_distinct_prime_factors n ∧ phi (tau n) = tau (phi n) ↔
  ∃ (t r : ℕ), r.Prime ∧ t > 0 ∧ n = 2^(t-1) * 3^(r-1) :=
sorry

end NUMINAMATH_CALUDE_two_prime_factors_phi_tau_equality_l1940_194015


namespace NUMINAMATH_CALUDE_unrestricted_arrangements_count_restricted_arrangements_count_l1940_194064

/-- Represents the number of singers in the chorus -/
def total_singers : ℕ := 8

/-- Represents the number of female singers -/
def female_singers : ℕ := 6

/-- Represents the number of male singers -/
def male_singers : ℕ := 2

/-- Represents the number of people per row -/
def people_per_row : ℕ := 4

/-- Represents the number of rows -/
def num_rows : ℕ := 2

/-- Calculates the number of arrangements with no restrictions -/
def unrestricted_arrangements : ℕ := Nat.factorial total_singers

/-- Calculates the number of arrangements with lead singer in front and male singers in back -/
def restricted_arrangements : ℕ :=
  (Nat.choose (female_singers - 1) (people_per_row - 1)) *
  (Nat.factorial people_per_row) *
  (Nat.factorial people_per_row)

/-- Theorem stating the number of unrestricted arrangements -/
theorem unrestricted_arrangements_count :
  unrestricted_arrangements = 40320 := by sorry

/-- Theorem stating the number of restricted arrangements -/
theorem restricted_arrangements_count :
  restricted_arrangements = 5760 := by sorry

end NUMINAMATH_CALUDE_unrestricted_arrangements_count_restricted_arrangements_count_l1940_194064


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_20_l1940_194092

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers that sum to 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! s : ConsecutiveSet, sum_consecutive s = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_20_l1940_194092


namespace NUMINAMATH_CALUDE_thirtieth_term_is_351_l1940_194052

/-- Arithmetic sequence with first term 3 and common difference 12 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 12

/-- The 30th term of the arithmetic sequence is 351 -/
theorem thirtieth_term_is_351 : arithmeticSequence 30 = 351 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_351_l1940_194052


namespace NUMINAMATH_CALUDE_tangent_through_origin_l1940_194089

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_through_origin (x₀ : ℝ) (h₁ : x₀ > 0) :
  (∃ k : ℝ, k * x₀ = f x₀ ∧ ∀ x : ℝ, f x₀ + k * (x - x₀) = k * x) →
  x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l1940_194089


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1940_194066

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 7 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1940_194066


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1940_194088

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1940_194088


namespace NUMINAMATH_CALUDE_inequality_proof_l1940_194062

theorem inequality_proof (n : ℕ) (a : ℝ) (hn : n > 1) (ha : 0 < a ∧ a < 1) :
  1 + a < (1 + a / n)^n ∧ (1 + a / n)^n < (1 + a / (n + 1))^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1940_194062


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1940_194002

theorem unique_solution_power_equation :
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1940_194002


namespace NUMINAMATH_CALUDE_dhoni_spending_l1940_194084

theorem dhoni_spending (total_earnings : ℝ) (rent_percent dishwasher_percent leftover_percent : ℝ) :
  rent_percent = 25 →
  leftover_percent = 52.5 →
  dishwasher_percent = 100 - rent_percent - leftover_percent →
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dhoni_spending_l1940_194084


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1940_194071

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1940_194071


namespace NUMINAMATH_CALUDE_existence_of_m_l1940_194056

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem existence_of_m :
  ∃ m : ℕ, 19 ≤ m ∧ m ≤ 60 ∧ 
  x m ≤ 4 + 1 / 2^10 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 4 + 1 / 2^10 :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l1940_194056


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1940_194035

theorem shaded_area_calculation (π : ℝ) (h : π > 0) : 
  let square_side : ℝ := 8
  let quarter_circle_radius : ℝ := 0.6 * square_side
  let square_area : ℝ := square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2
  square_area - quarter_circles_area = 64 - 23.04 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1940_194035


namespace NUMINAMATH_CALUDE_order_of_expressions_l1940_194028

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(3/2)
  let b : ℝ := Real.log π
  let c : ℝ := Real.log (3/2) / Real.log (1/2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l1940_194028


namespace NUMINAMATH_CALUDE_physics_marks_l1940_194034

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 70)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 110 := by sorry

end NUMINAMATH_CALUDE_physics_marks_l1940_194034


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1940_194020

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) :
  a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1940_194020


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l1940_194090

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 5 / 3
  | 1 => 30 / 7
  | 2 => 180 / 49
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, ∀ n : ℕ, n > 0 → geometric_series (n + 1) = r * geometric_series n :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l1940_194090


namespace NUMINAMATH_CALUDE_ellipse_intersection_property_l1940_194081

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- Definition of the line l passing through P(x₁, y₁) -/
def line_l (x₁ y₁ x y : ℝ) : Prop :=
  4 * x₁ * x + 9 * y₁ * y = 36

/-- Theorem statement -/
theorem ellipse_intersection_property :
  ∀ (x₁ y₁ : ℝ),
  is_on_ellipse x₁ y₁ →
  ∃ (M_x M_y M'_x M'_y : ℝ),
    line_l x₁ y₁ M_x M_y ∧
    line_l x₁ y₁ M'_x M'_y ∧
    M_x = 3 ∧
    M'_x = -3 ∧
    (M_y^2 + 9) * (M'_y^2 + 9) = 36 ∧
    ∀ (N_x N_y N'_x N'_y : ℝ),
      line_l x₁ y₁ N_x N_y →
      line_l x₁ y₁ N'_x N'_y →
      N_x = 3 →
      N'_x = -3 →
      6 * (|N_y| + |N'_y|) ≥ 72 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_property_l1940_194081


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1940_194061

/-- The quadratic function f(x) = -2(x+3)^2 - 5 has vertex at (-3, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * (x + 3)^2 - 5
  (∀ x, f x ≤ f (-3)) ∧ f (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1940_194061


namespace NUMINAMATH_CALUDE_prob_at_least_one_3_l1940_194073

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling a 3 on a single fair die -/
def probThree : ℚ := 1 / numSides

/-- The probability of not rolling a 3 on a single fair die -/
def probNotThree : ℚ := 1 - probThree

/-- The probability of rolling at least one 3 when two fair dice are rolled -/
def probAtLeastOne3 : ℚ := 1 - probNotThree * probNotThree

theorem prob_at_least_one_3 : probAtLeastOne3 = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_3_l1940_194073


namespace NUMINAMATH_CALUDE_catch_time_correct_l1940_194014

/-- Represents the pursuit scenario between a smuggler and coast guard -/
structure Pursuit where
  initial_distance : ℝ
  initial_smuggler_speed : ℝ
  initial_coast_guard_speed : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time when the coast guard catches the smuggler -/
def catch_time (p : Pursuit) : ℝ :=
  sorry

/-- Theorem stating that the coast guard catches the smuggler after 6 hours and 36 minutes -/
theorem catch_time_correct (p : Pursuit) : 
  p.initial_distance = 15 ∧ 
  p.initial_smuggler_speed = 13 ∧ 
  p.initial_coast_guard_speed = 15 ∧
  p.speed_change_time = 3 ∧
  p.new_speed_ratio = 18/15 →
  catch_time p = 6 + 36/60 := by
  sorry

end NUMINAMATH_CALUDE_catch_time_correct_l1940_194014


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1940_194082

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1940_194082


namespace NUMINAMATH_CALUDE_sum_multiple_of_three_l1940_194021

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end NUMINAMATH_CALUDE_sum_multiple_of_three_l1940_194021
