import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3890_389099

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b^2 = 75/4 for the ellipse -/
theorem ellipse_hyperbola_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/64 - y^2/36 = 1/16) →
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 64 - 36) →
  b^2 = 75/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3890_389099


namespace NUMINAMATH_CALUDE_percentage_calculation_l3890_389081

theorem percentage_calculation (initial_amount : ℝ) : 
  initial_amount = 1200 →
  (((initial_amount * 0.60) * 0.30) * 2) / 3 = 144 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3890_389081


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3890_389079

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3890_389079


namespace NUMINAMATH_CALUDE_unique_triangle_configuration_l3890_389004

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  a : Stick
  b : Stick
  c : Stick
  valid : a.length + b.length > c.length ∧
          a.length + c.length > b.length ∧
          b.length + c.length > a.length

/-- A configuration of 15 sticks forming 5 triangles -/
structure Configuration where
  sticks : Fin 15 → Stick
  triangles : Fin 5 → Triangle
  uses_all_sticks : ∀ s : Fin 15, ∃ t : Fin 5, (triangles t).a = sticks s ∨
                                               (triangles t).b = sticks s ∨
                                               (triangles t).c = sticks s

/-- Theorem stating that there's only one way to form 5 triangles from 15 sticks -/
theorem unique_triangle_configuration (c1 c2 : Configuration) : c1 = c2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_configuration_l3890_389004


namespace NUMINAMATH_CALUDE_red_cars_count_l3890_389003

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 70 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 26 := by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l3890_389003


namespace NUMINAMATH_CALUDE_problem_solution_l3890_389041

theorem problem_solution : ∀ x y : ℝ,
  x = 88 * (1 + 0.3) →
  y = x * (1 - 0.15) →
  y = 97.24 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3890_389041


namespace NUMINAMATH_CALUDE_f_composition_eq_inverse_e_l3890_389007

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_eq_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_eq_inverse_e_l3890_389007


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3890_389025

/-- Given a rectangle with perimeter 80 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is sqrt(46400)/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 80) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 46400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3890_389025


namespace NUMINAMATH_CALUDE_equal_points_per_round_l3890_389064

-- Define the total points and number of rounds
def total_points : ℕ := 300
def num_rounds : ℕ := 5

-- Define the points per round
def points_per_round : ℕ := total_points / num_rounds

-- Theorem to prove
theorem equal_points_per_round :
  (total_points = num_rounds * points_per_round) ∧ (points_per_round = 60) := by
  sorry

end NUMINAMATH_CALUDE_equal_points_per_round_l3890_389064


namespace NUMINAMATH_CALUDE_collinear_points_k_value_unique_k_value_l3890_389053

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₂) / (x₃ - x₂)

/-- Theorem: If the points (2,-3), (4,3), and (5, k/2) are collinear, then k = 12. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 2 (-3) 4 3 5 (k/2) → k = 12 :=
by
  sorry

/-- Corollary: The only value of k that makes the points (2,-3), (4,3), and (5, k/2) collinear is 12. -/
theorem unique_k_value :
  ∃! k : ℝ, collinear 2 (-3) 4 3 5 (k/2) :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_unique_k_value_l3890_389053


namespace NUMINAMATH_CALUDE_units_digit_of_a_l3890_389070

theorem units_digit_of_a (a : ℕ) : a = 2003^2004 - 2004^2003 → a % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_a_l3890_389070


namespace NUMINAMATH_CALUDE_problem_solution_l3890_389091

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2021)
  (h2 : x + 2021 * Real.cos y = 2020)
  (h3 : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2020 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3890_389091


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3890_389009

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 757 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3890_389009


namespace NUMINAMATH_CALUDE_inequality_to_interval_l3890_389028

theorem inequality_to_interval : 
  {x : ℝ | -8 ≤ x ∧ x < 15} = Set.Icc (-8) 15 := by sorry

end NUMINAMATH_CALUDE_inequality_to_interval_l3890_389028


namespace NUMINAMATH_CALUDE_third_side_length_l3890_389074

/-- A triangle with known perimeter and two side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter : ℝ
  perimeter_eq : side1 + side2 + side3 = perimeter

/-- The theorem stating that for a triangle with two sides 7 and 15, and perimeter 32, the third side is 10 -/
theorem third_side_length (t : Triangle) 
    (h1 : t.side1 = 7)
    (h2 : t.side2 = 15)
    (h3 : t.perimeter = 32) : 
  t.side3 = 10 := by
  sorry


end NUMINAMATH_CALUDE_third_side_length_l3890_389074


namespace NUMINAMATH_CALUDE_average_speed_of_trip_l3890_389050

/-- Proves that the average speed of a trip is 16 km/h given the specified conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
    (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
    (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
    total_distance / (first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_of_trip_l3890_389050


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3890_389040

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : ∃ i, t.angles i = 80) :
  (t.angles 0 = 80 ∨ t.angles 0 = 20) ∨
  (t.angles 1 = 80 ∨ t.angles 1 = 20) ∨
  (t.angles 2 = 80 ∨ t.angles 2 = 20) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3890_389040


namespace NUMINAMATH_CALUDE_positive_integer_power_equality_l3890_389018

theorem positive_integer_power_equality (a b : ℕ+) :
  a ^ b.val = b ^ (a.val ^ 2) ↔ (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_power_equality_l3890_389018


namespace NUMINAMATH_CALUDE_dvd_shipping_cost_percentage_l3890_389072

/-- Given Mike's DVD cost, Steve's DVD cost as twice Mike's, and Steve's total cost,
    prove that the shipping cost percentage of Steve's DVD price is 80% -/
theorem dvd_shipping_cost_percentage
  (mike_cost : ℝ)
  (steve_dvd_cost : ℝ)
  (steve_total_cost : ℝ)
  (h1 : mike_cost = 5)
  (h2 : steve_dvd_cost = 2 * mike_cost)
  (h3 : steve_total_cost = 18) :
  (steve_total_cost - steve_dvd_cost) / steve_dvd_cost * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dvd_shipping_cost_percentage_l3890_389072


namespace NUMINAMATH_CALUDE_box_of_balls_l3890_389087

theorem box_of_balls (x : ℕ) : 
  (25 - x = 30 - 25) → x = 20 := by sorry

end NUMINAMATH_CALUDE_box_of_balls_l3890_389087


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3890_389094

/-- The number of points in each row of the grid -/
def rows : ℕ := 3

/-- The number of points in each column of the grid -/
def columns : ℕ := 4

/-- The total number of points in the grid -/
def total_points : ℕ := rows * columns

/-- The number of degenerate cases (collinear points) -/
def degenerate_cases : ℕ := rows + columns + 2

theorem distinct_triangles_in_grid : 
  (total_points.choose 3) - degenerate_cases = 76 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3890_389094


namespace NUMINAMATH_CALUDE_circle_center_from_diameter_endpoints_l3890_389093

/-- The center of a circle given the endpoints of its diameter -/
theorem circle_center_from_diameter_endpoints (x₁ y₁ x₂ y₂ : ℝ) :
  let endpoint1 : ℝ × ℝ := (x₁, y₁)
  let endpoint2 : ℝ × ℝ := (x₂, y₂)
  let center : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  endpoint1 = (2, -3) → endpoint2 = (8, 9) → center = (5, 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_from_diameter_endpoints_l3890_389093


namespace NUMINAMATH_CALUDE_fixed_internet_charge_is_4_l3890_389011

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem fixed_internet_charge_is_4
  (january : MonthlyBill)
  (february : MonthlyBill)
  (h1 : totalBill january = 40)
  (h2 : totalBill february = 76)
  (h3 : february.callCharge = 2 * january.callCharge)
  (h4 : january.internetCharge = february.internetCharge) :
  january.internetCharge = 4 := by
  sorry

#check fixed_internet_charge_is_4

end NUMINAMATH_CALUDE_fixed_internet_charge_is_4_l3890_389011


namespace NUMINAMATH_CALUDE_rectangleWithHoleAreaTheorem_l3890_389042

/-- The area of a rectangle with a hole, given the dimensions of both rectangles -/
def rectangleWithHoleArea (x : ℝ) : ℝ :=
  let largeRectLength := x + 7
  let largeRectWidth := x + 5
  let holeLength := 2*x - 3
  let holeWidth := x - 2
  (largeRectLength * largeRectWidth) - (holeLength * holeWidth)

/-- Theorem stating that the area of the rectangle with a hole is equal to -x^2 + 19x + 29 -/
theorem rectangleWithHoleAreaTheorem (x : ℝ) :
  rectangleWithHoleArea x = -x^2 + 19*x + 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangleWithHoleAreaTheorem_l3890_389042


namespace NUMINAMATH_CALUDE_irene_worked_50_hours_l3890_389063

/-- Calculates the total hours worked given the regular hours, overtime hours, regular pay, overtime pay rate, and total income. -/
def total_hours_worked (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ) : ℕ :=
  regular_hours + (total_income - regular_pay) / overtime_rate

/-- Proves that given the problem conditions, Irene worked 50 hours. -/
theorem irene_worked_50_hours (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ)
  (h1 : regular_hours = 40)
  (h2 : regular_pay = 500)
  (h3 : overtime_rate = 20)
  (h4 : total_income = 700) :
  total_hours_worked regular_hours regular_pay overtime_rate total_income = 50 := by
  sorry

#eval total_hours_worked 40 500 20 700

end NUMINAMATH_CALUDE_irene_worked_50_hours_l3890_389063


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_for_a_squared_9_l3890_389038

theorem a_equals_3_sufficient_not_necessary_for_a_squared_9 :
  (∀ a : ℝ, a = 3 → a^2 = 9) ∧
  (∃ a : ℝ, a ≠ 3 ∧ a^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_for_a_squared_9_l3890_389038


namespace NUMINAMATH_CALUDE_lottery_probability_l3890_389057

def powerball_count : ℕ := 30
def luckyball_count : ℕ := 50
def luckyball_draw : ℕ := 6

theorem lottery_probability :
  (1 : ℚ) / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 476721000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3890_389057


namespace NUMINAMATH_CALUDE_jaces_remaining_money_l3890_389085

/-- Proves that Jace's remaining money after transactions is correct -/
theorem jaces_remaining_money
  (earnings : ℚ)
  (debt : ℚ)
  (neighbor_percentage : ℚ)
  (exchange_rate : ℚ)
  (h1 : earnings = 1500)
  (h2 : debt = 358)
  (h3 : neighbor_percentage = 1/4)
  (h4 : exchange_rate = 121/100) :
  earnings - debt - (earnings - debt) * neighbor_percentage = 8565/10 :=
by sorry


end NUMINAMATH_CALUDE_jaces_remaining_money_l3890_389085


namespace NUMINAMATH_CALUDE_candy_distribution_l3890_389044

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 → num_students = 43 → 
  pieces_per_student * num_students = total_candy →
  pieces_per_student = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3890_389044


namespace NUMINAMATH_CALUDE_money_left_calculation_l3890_389088

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let medium_pizza_cost := 3 * q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is 50 - 15q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 15 * q := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l3890_389088


namespace NUMINAMATH_CALUDE_expression_simplification_l3890_389047

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3890_389047


namespace NUMINAMATH_CALUDE_if_statement_properties_l3890_389080

-- Define the structure of an IF statement
structure IfStatement where
  has_else : Bool
  has_end_if : Bool

-- Define what makes an IF statement valid
def is_valid_if_statement (stmt : IfStatement) : Prop :=
  stmt.has_end_if ∧ (stmt.has_else ∨ ¬stmt.has_else)

-- Theorem statement
theorem if_statement_properties :
  ∀ (stmt : IfStatement),
    is_valid_if_statement stmt →
    (stmt.has_else ∨ ¬stmt.has_else) ∧ stmt.has_end_if :=
by sorry

end NUMINAMATH_CALUDE_if_statement_properties_l3890_389080


namespace NUMINAMATH_CALUDE_aluminum_decoration_problem_l3890_389075

def available_lengths : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (· ∈ available_lengths) ∧ combination.sum = 50

theorem aluminum_decoration_problem :
  ∀ combination : List ℕ,
    is_valid_combination combination ↔
      combination = [19, 19, 12] ∨ combination = [19, 19] :=
by sorry

end NUMINAMATH_CALUDE_aluminum_decoration_problem_l3890_389075


namespace NUMINAMATH_CALUDE_integral_x_cos_x_plus_cube_root_x_squared_l3890_389078

open Real
open MeasureTheory
open Interval

theorem integral_x_cos_x_plus_cube_root_x_squared : 
  ∫ x in (-1)..1, (x * cos x + (x^2)^(1/3)) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_cos_x_plus_cube_root_x_squared_l3890_389078


namespace NUMINAMATH_CALUDE_mary_sheep_theorem_l3890_389046

def initial_sheep : ℕ := 1500

def sister_percentage : ℚ := 1/4
def brother_percentage : ℚ := 3/10
def cousin_fraction : ℚ := 1/7

def remaining_sheep : ℕ := 676

theorem mary_sheep_theorem :
  let sheep_after_sister := initial_sheep - ⌊initial_sheep * sister_percentage⌋
  let sheep_after_brother := sheep_after_sister - ⌊sheep_after_sister * brother_percentage⌋
  let sheep_after_cousin := sheep_after_brother - ⌊sheep_after_brother * cousin_fraction⌋
  sheep_after_cousin = remaining_sheep := by sorry

end NUMINAMATH_CALUDE_mary_sheep_theorem_l3890_389046


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l3890_389030

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 5) (hb : 5 < b ∧ b < 12) :
  -10 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l3890_389030


namespace NUMINAMATH_CALUDE_intersection_distance_l3890_389076

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 12) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 12) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_l3890_389076


namespace NUMINAMATH_CALUDE_soccer_ball_min_cost_l3890_389005

/-- Represents the purchase of soccer balls -/
structure SoccerBallPurchase where
  brand_a_price : ℕ
  brand_b_price : ℕ
  total_balls : ℕ
  min_brand_a : ℕ
  max_cost : ℕ

/-- Calculates the total cost for a given number of brand A balls -/
def total_cost (p : SoccerBallPurchase) (brand_a_count : ℕ) : ℕ :=
  p.brand_a_price * brand_a_count + p.brand_b_price * (p.total_balls - brand_a_count)

/-- Theorem stating the minimum cost of the soccer ball purchase -/
theorem soccer_ball_min_cost (p : SoccerBallPurchase)
  (h1 : p.brand_a_price = p.brand_b_price + 10)
  (h2 : 2 * p.brand_a_price + 3 * p.brand_b_price = 220)
  (h3 : p.total_balls = 60)
  (h4 : p.min_brand_a = 43)
  (h5 : p.max_cost = 2850) :
  ∃ (m : ℕ), m ≥ p.min_brand_a ∧ m ≤ p.total_balls ∧
    total_cost p m ≤ p.max_cost ∧
    ∀ (n : ℕ), n ≥ p.min_brand_a → n ≤ p.total_balls →
      total_cost p n ≤ p.max_cost → total_cost p m ≤ total_cost p n :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_min_cost_l3890_389005


namespace NUMINAMATH_CALUDE_petes_age_proof_l3890_389077

/-- Pete's current age -/
def petes_age : ℕ := 35

/-- Pete's son's current age -/
def sons_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 4

theorem petes_age_proof :
  petes_age = 35 ∧
  sons_age = 9 ∧
  petes_age + years_later = 3 * (sons_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_petes_age_proof_l3890_389077


namespace NUMINAMATH_CALUDE_harvest_earnings_problem_harvest_duration_l3890_389006

/-- Represents the harvest earnings problem --/
theorem harvest_earnings_problem (total_earnings : ℕ) (initial_earnings : ℕ) 
  (weekly_increase : ℕ) (weekly_deduction : ℕ) (weeks : ℕ) : Prop :=
  total_earnings = 1216 ∧
  initial_earnings = 16 ∧
  weekly_increase = 8 ∧
  weekly_deduction = 12 ∧
  weeks = 17 →
  total_earnings = (weeks * (2 * initial_earnings + (weeks - 1) * weekly_increase)) / 2 - 
    weeks * weekly_deduction

/-- Proves that the harvest lasted 17 weeks --/
theorem harvest_duration : 
  ∃ (total_earnings initial_earnings weekly_increase weekly_deduction weeks : ℕ),
  harvest_earnings_problem total_earnings initial_earnings weekly_increase weekly_deduction weeks :=
by
  sorry

end NUMINAMATH_CALUDE_harvest_earnings_problem_harvest_duration_l3890_389006


namespace NUMINAMATH_CALUDE_student_arrangement_equality_l3890_389019

/-- The number of ways to arrange k items out of n items -/
def arrange (n k : ℕ) : ℕ := sorry

theorem student_arrangement_equality (n : ℕ) :
  arrange (2*n) (2*n) = arrange (2*n) n * arrange n n := by sorry

end NUMINAMATH_CALUDE_student_arrangement_equality_l3890_389019


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3890_389097

theorem inequality_equivalence (x : ℝ) : x + 1 > 3 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3890_389097


namespace NUMINAMATH_CALUDE_tickets_purchased_l3890_389035

theorem tickets_purchased (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (money_left : ℕ) :
  olivia_money = 112 →
  nigel_money = 139 →
  ticket_cost = 28 →
  money_left = 83 →
  (olivia_money + nigel_money - money_left) / ticket_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_tickets_purchased_l3890_389035


namespace NUMINAMATH_CALUDE_probability_of_one_red_ball_l3890_389068

/-- The probability of drawing exactly one red ball from a bag containing 2 yellow balls, 3 red balls, and 5 white balls is 3/10. -/
theorem probability_of_one_red_ball (yellow_balls red_balls white_balls : ℕ) 
  (h_yellow : yellow_balls = 2)
  (h_red : red_balls = 3)
  (h_white : white_balls = 5) : 
  (red_balls : ℚ) / (yellow_balls + red_balls + white_balls) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_red_ball_l3890_389068


namespace NUMINAMATH_CALUDE_chick_count_product_l3890_389090

/-- Represents the state of chicks in a nest for a given week -/
structure ChickState :=
  (open_beak : ℕ)
  (growing_feathers : ℕ)

/-- The chick lifecycle in the nest -/
def chick_lifecycle : Prop :=
  ∃ (last_week this_week : ChickState),
    last_week.open_beak = 20 ∧
    last_week.growing_feathers = 14 ∧
    this_week.open_beak = 15 ∧
    this_week.growing_feathers = 11

/-- The theorem to be proved -/
theorem chick_count_product :
  chick_lifecycle →
  ∃ (two_weeks_ago next_week : ℕ),
    two_weeks_ago = 11 ∧
    next_week = 15 ∧
    two_weeks_ago * next_week = 165 :=
by
  sorry


end NUMINAMATH_CALUDE_chick_count_product_l3890_389090


namespace NUMINAMATH_CALUDE_range_of_expression_l3890_389000

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3890_389000


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_example_l3890_389032

/-- Given a tree with squirrels and nuts, calculate the difference between their quantities -/
def squirrel_nut_difference (num_squirrels num_nuts : ℕ) : ℤ :=
  (num_squirrels : ℤ) - (num_nuts : ℤ)

/-- Theorem: In a tree with 4 squirrels and 2 nuts, the difference between
    the number of squirrels and nuts is 2 -/
theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_example_l3890_389032


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3890_389022

theorem product_of_polynomials (p x y : ℝ) : 
  (2 * p^2 - 5 * p + x) * (5 * p^2 + y * p - 10) = 10 * p^4 + 5 * p^3 - 65 * p^2 + 40 * p + 40 →
  x + y = -6.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3890_389022


namespace NUMINAMATH_CALUDE_t_shape_perimeter_specific_l3890_389002

/-- The perimeter of a T-shape formed by two rectangles --/
def t_shape_perimeter (length width overlap : ℝ) : ℝ :=
  2 * (2 * (length + width)) - 2 * overlap

/-- Theorem: The perimeter of a T-shape formed by two 3x5 inch rectangles with a 1.5 inch overlap is 29 inches --/
theorem t_shape_perimeter_specific : t_shape_perimeter 5 3 1.5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_specific_l3890_389002


namespace NUMINAMATH_CALUDE_steiner_ellipses_equations_l3890_389083

/-- Barycentric coordinates in a triangle -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Circumscribed Steiner Ellipse equation -/
def circumscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  p.β * p.γ + p.α * p.γ + p.α * p.β = 0

/-- Inscribed Steiner Ellipse equation -/
def inscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2

/-- Theorem stating the equations of Steiner ellipses in barycentric coordinates -/
theorem steiner_ellipses_equations (p : BarycentricCoord) :
  (circumscribedSteinerEllipse p ↔ p.β * p.γ + p.α * p.γ + p.α * p.β = 0) ∧
  (inscribedSteinerEllipse p ↔ 2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2) :=
by sorry

end NUMINAMATH_CALUDE_steiner_ellipses_equations_l3890_389083


namespace NUMINAMATH_CALUDE_notebook_purchase_solution_l3890_389082

/-- Notebook types -/
inductive NotebookType
| A
| B
| C

/-- Represents the notebook purchase problem -/
structure NotebookPurchase where
  totalNotebooks : ℕ
  priceA : ℕ
  priceB : ℕ
  priceC : ℕ
  totalCostI : ℕ
  totalCostII : ℕ

/-- Represents the solution for part I -/
structure SolutionI where
  numA : ℕ
  numB : ℕ

/-- Represents the solution for part II -/
structure SolutionII where
  numA : ℕ

/-- The given notebook purchase problem -/
def problem : NotebookPurchase :=
  { totalNotebooks := 30
  , priceA := 11
  , priceB := 9
  , priceC := 6
  , totalCostI := 288
  , totalCostII := 188
  }

/-- Checks if the solution for part I is correct -/
def checkSolutionI (p : NotebookPurchase) (s : SolutionI) : Prop :=
  s.numA + s.numB = p.totalNotebooks ∧
  s.numA * p.priceA + s.numB * p.priceB = p.totalCostI

/-- Checks if the solution for part II is correct -/
def checkSolutionII (p : NotebookPurchase) (s : SolutionII) : Prop :=
  ∃ (numB numC : ℕ), 
    s.numA + numB + numC = p.totalNotebooks ∧
    s.numA * p.priceA + numB * p.priceB + numC * p.priceC = p.totalCostII

/-- The main theorem to prove -/
theorem notebook_purchase_solution :
  checkSolutionI problem { numA := 9, numB := 21 } ∧
  checkSolutionII problem { numA := 1 } :=
sorry


end NUMINAMATH_CALUDE_notebook_purchase_solution_l3890_389082


namespace NUMINAMATH_CALUDE_bus_children_difference_l3890_389073

/-- Given the initial number of children on a bus, the number of children who got on,
    and the final number of children on the bus, this theorem proves that
    2 more children got on than got off. -/
theorem bus_children_difference (initial : ℕ) (got_on : ℕ) (final : ℕ)
    (h1 : initial = 28)
    (h2 : got_on = 82)
    (h3 : final = 30)
    (h4 : final = initial + got_on - (initial + got_on - final)) :
  got_on - (initial + got_on - final) = 2 :=
by sorry

end NUMINAMATH_CALUDE_bus_children_difference_l3890_389073


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l3890_389084

/-- A function g satisfying the given properties -/
def g : Set ℝ → (ℝ → ℝ) → Prop :=
  λ D f => ∀ x ∈ D, (1 / x) ∈ D ∧ f x + f (1 / x) = x + 2

/-- The theorem stating that {-1, 1} is the largest possible domain for g -/
theorem largest_domain_of_g :
  ∀ D : Set ℝ, ∀ f : ℝ → ℝ, g D f → D ⊆ {-1, 1} :=
sorry

end NUMINAMATH_CALUDE_largest_domain_of_g_l3890_389084


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l3890_389061

theorem set_equality_implies_values (x y : ℝ) : 
  ({1, x, y} : Set ℝ) = {x, x^2, x*y} → x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l3890_389061


namespace NUMINAMATH_CALUDE_pizza_combinations_l3890_389010

theorem pizza_combinations (n_toppings : ℕ) (n_crusts : ℕ) (k_toppings : ℕ) : 
  n_toppings = 8 → n_crusts = 2 → k_toppings = 5 → 
  (Nat.choose n_toppings k_toppings) * n_crusts = 112 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3890_389010


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3890_389021

theorem sin_2alpha_value (α : Real) (h : Real.cos (α + Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3890_389021


namespace NUMINAMATH_CALUDE_number_divided_by_quarter_l3890_389034

theorem number_divided_by_quarter : ∀ x : ℝ, x / 0.25 = 400 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_quarter_l3890_389034


namespace NUMINAMATH_CALUDE_other_sales_percentage_l3890_389033

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- The percentage of sales that were not notebooks or markers -/
def other_sales : ℝ := total_sales - (notebook_sales + marker_sales)

theorem other_sales_percentage : other_sales = 32 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l3890_389033


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l3890_389065

theorem twins_age_product_difference : 
  ∀ (current_age : ℕ), 
    current_age = 6 → 
    (current_age + 1) * (current_age + 1) - current_age * current_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l3890_389065


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l3890_389086

/-- The trajectory of point G -/
def trajectory (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The condition that the product of slopes of GE and FG is -4 -/
def slope_condition (x y : ℝ) : Prop := 
  y ≠ 0 → (y / (x - 1)) * (y / (x + 1)) = -4

/-- The line passing through (0, -1) with slope k -/
def line (k x : ℝ) : ℝ := k * x - 1

/-- The x-coordinates of the intersection points sum to 8 -/
def intersection_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    trajectory x₁ (line k x₁) ∧ 
    trajectory x₂ (line k x₂) ∧ 
    x₁ + x₂ = 8

theorem trajectory_and_intersection :
  (∀ x y : ℝ, slope_condition x y → trajectory x y) ∧
  (∀ k : ℝ, intersection_condition k → k = 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l3890_389086


namespace NUMINAMATH_CALUDE_water_depth_relationship_l3890_389008

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real
  horizontalWaterDepth : Real

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20) 
  (d : tank.baseDiameter = 5) 
  (w : tank.horizontalWaterDepth = 2) : 
  ∃ ε > 0, abs (verticalWaterDepth tank - 0.9) < ε :=
sorry

end NUMINAMATH_CALUDE_water_depth_relationship_l3890_389008


namespace NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l3890_389055

/-- The amount of additional money Albert needs to buy his art supplies -/
def additional_money_needed (paintbrush_cost paint_cost easel_cost current_money : ℚ) : ℚ :=
  paintbrush_cost + paint_cost + easel_cost - current_money

/-- Theorem stating that Albert needs $12 more -/
theorem albert_needs_twelve_dollars :
  additional_money_needed 1.50 4.35 12.65 6.50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l3890_389055


namespace NUMINAMATH_CALUDE_handshakes_15_couples_l3890_389092

/-- The number of handshakes in a gathering of married couples -/
def num_handshakes (n : ℕ) : ℕ :=
  (n * 2 * (n * 2 - 2)) / 2 - n

/-- Theorem: In a gathering of 15 married couples, the total number of handshakes is 405 -/
theorem handshakes_15_couples :
  num_handshakes 15 = 405 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_15_couples_l3890_389092


namespace NUMINAMATH_CALUDE_sin_seventeen_pi_quarters_l3890_389089

theorem sin_seventeen_pi_quarters : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seventeen_pi_quarters_l3890_389089


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l3890_389029

theorem power_mod_thirteen : 777^777 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l3890_389029


namespace NUMINAMATH_CALUDE_inequality_solution_l3890_389049

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 1) / (x - 3)^2 ≥ 8 ↔ 3 < x ∧ x ≤ 24/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3890_389049


namespace NUMINAMATH_CALUDE_smallest_number_l3890_389060

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3890_389060


namespace NUMINAMATH_CALUDE_triangle_conditions_equivalence_l3890_389096

theorem triangle_conditions_equivalence (x : ℝ) :
  (∀ (BC AC AB : ℝ),
    BC = x + 11 ∧ AC = x + 6 ∧ AB = 3*x + 2 →
    AB + AC > BC ∧ AB + BC > AC ∧ AC + BC > AB ∧
    BC > AB ∧ BC > AC) ↔
  (1 < x ∧ x < 4.5) :=
sorry

end NUMINAMATH_CALUDE_triangle_conditions_equivalence_l3890_389096


namespace NUMINAMATH_CALUDE_residue_of_negative_998_mod_28_l3890_389001

theorem residue_of_negative_998_mod_28 :
  ∃ (q : ℤ), -998 = 28 * q + 10 ∧ (0 ≤ 10) ∧ (10 < 28) := by sorry

end NUMINAMATH_CALUDE_residue_of_negative_998_mod_28_l3890_389001


namespace NUMINAMATH_CALUDE_set_operations_l3890_389066

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def C : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1,2,3,4,5}) ∧
  ((U \ B) ∪ (U \ C) = {1,2,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3890_389066


namespace NUMINAMATH_CALUDE_greg_trousers_count_l3890_389062

/-- The cost of a shirt -/
def shirtCost : ℝ := sorry

/-- The cost of a pair of trousers -/
def trousersCost : ℝ := sorry

/-- The cost of a tie -/
def tieCost : ℝ := sorry

/-- The number of trousers Greg bought in the first scenario -/
def firstScenarioTrousers : ℕ := sorry

theorem greg_trousers_count :
  (6 * shirtCost + firstScenarioTrousers * trousersCost + 2 * tieCost = 80) ∧
  (4 * shirtCost + 2 * trousersCost + 2 * tieCost = 140) ∧
  (5 * shirtCost + 3 * trousersCost + 2 * tieCost = 110) →
  firstScenarioTrousers = 4 := by
  sorry

end NUMINAMATH_CALUDE_greg_trousers_count_l3890_389062


namespace NUMINAMATH_CALUDE_course_choice_theorem_l3890_389095

/-- The number of ways to choose courses for 5 students -/
def course_choice_ways : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of courses -/
def num_courses : ℕ := 2

/-- The minimum number of students required for each course -/
def min_students_per_course : ℕ := 2

theorem course_choice_theorem :
  ∀ (ways : ℕ),
  ways = course_choice_ways →
  ways = (num_students.choose min_students_per_course) * num_courses.factorial :=
by sorry

end NUMINAMATH_CALUDE_course_choice_theorem_l3890_389095


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l3890_389016

theorem cousins_ages_sum : ∃ (a b c d : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 20 ∧ c * d = 21) ∨ (a * c = 20 ∧ b * d = 21) ∨ 
   (a * d = 20 ∧ b * c = 21) ∨ (b * c = 20 ∧ a * d = 21) ∨ 
   (b * d = 20 ∧ a * c = 21) ∧ (c * d = 20 ∧ a * b = 21)) ∧
  (a + b + c + d = 19) :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l3890_389016


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3890_389012

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if f(-1) is between -1 and 2,
    and f(1) is between 2 and 4, then f(-2) is between -1 and 10. -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  ((-1 : ℝ) ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  ((-1 : ℝ) ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3890_389012


namespace NUMINAMATH_CALUDE_no_intersection_at_vertex_l3890_389052

/-- The line equation y = x + b -/
def line (x b : ℝ) : ℝ := x + b

/-- The parabola equation y = x^2 + b^2 + 1 -/
def parabola (x b : ℝ) : ℝ := x^2 + b^2 + 1

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 0

/-- Theorem: There are no real values of b for which the line y = x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 -/
theorem no_intersection_at_vertex :
  ¬∃ b : ℝ, line vertex_x b = parabola vertex_x b := by sorry

end NUMINAMATH_CALUDE_no_intersection_at_vertex_l3890_389052


namespace NUMINAMATH_CALUDE_smallest_b_value_l3890_389015

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3890_389015


namespace NUMINAMATH_CALUDE_existence_of_d_l3890_389059

theorem existence_of_d : ∃ d : ℝ,
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * (n : ℝ)^2 + 20 * (n : ℝ) - 67 = 0) ∧
  (4 * (d - ⌊d⌋)^2 - 15 * (d - ⌊d⌋) + 5 = 0) ∧
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = -8.63 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_d_l3890_389059


namespace NUMINAMATH_CALUDE_number_of_molecules_value_l3890_389045

/-- The number of molecules in a given substance -/
def number_of_molecules : ℕ := 3 * 10^26

/-- Theorem stating that the number of molecules is 3 · 10^26 -/
theorem number_of_molecules_value : number_of_molecules = 3 * 10^26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_molecules_value_l3890_389045


namespace NUMINAMATH_CALUDE_roots_cubic_reciprocal_sum_l3890_389026

/-- Given a quadratic equation px^2 + qx + m = 0 with roots r and s,
    prove that 1/r^3 + 1/s^3 = (-q^3 + 3qm) / m^3 -/
theorem roots_cubic_reciprocal_sum (p q m : ℝ) (hp : p ≠ 0) (hm : m ≠ 0) :
  ∃ (r s : ℝ), (p * r^2 + q * r + m = 0) ∧ 
               (p * s^2 + q * s + m = 0) ∧ 
               (1 / r^3 + 1 / s^3 = (-q^3 + 3*q*m) / m^3) := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_reciprocal_sum_l3890_389026


namespace NUMINAMATH_CALUDE_cube_of_integer_l3890_389013

theorem cube_of_integer (n p : ℕ+) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3)
  (h_div1 : n ∣ (p - 3)) (h_div2 : p ∣ ((n + 1)^3 - 1)) :
  p * n + 1 = (n + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_integer_l3890_389013


namespace NUMINAMATH_CALUDE_cosine_function_triangle_constraint_l3890_389098

open Real

theorem cosine_function_triangle_constraint (ω : ℝ) : 
  ω > 0 →
  let f : ℝ → ℝ := λ x => cos (ω * x)
  let A : ℝ × ℝ := (2 * π / ω, 1)
  let B : ℝ × ℝ := (π / ω, -1)
  let O : ℝ × ℝ := (0, 0)
  (∀ x > 0, x < 2 * π / ω → f x ≤ 1) →
  (∀ x > 0, x < π / ω → f x ≥ -1) →
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) > 0 →
  (O.1 - A.1) * (B.1 - A.1) + (O.2 - A.2) * (B.2 - A.2) > 0 →
  (O.1 - B.1) * (A.1 - B.1) + (O.2 - B.2) * (A.2 - B.2) > 0 →
  sqrt 2 * π / 2 < ω ∧ ω < sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_triangle_constraint_l3890_389098


namespace NUMINAMATH_CALUDE_gamma_interval_for_f_l3890_389051

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

def is_gamma_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (1 / n) (1 / m), ∃ x ∈ Set.Icc m n, f x = y

theorem gamma_interval_for_f :
  let m : ℝ := 1
  let n : ℝ := (1 + Real.sqrt 5) / 2
  m < n ∧ 
  Set.Icc m n ⊆ Set.Ioi 1 ∧ 
  is_gamma_interval f m n := by sorry

end NUMINAMATH_CALUDE_gamma_interval_for_f_l3890_389051


namespace NUMINAMATH_CALUDE_smallest_q_is_31_l3890_389024

theorem smallest_q_is_31 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = 15 * p + 1) :
  q ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_q_is_31_l3890_389024


namespace NUMINAMATH_CALUDE_largest_tile_size_is_correct_courtyard_largest_tile_size_l3890_389039

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

theorem largest_tile_size_is_correct (length width : ℕ) (h1 : length > 0) (h2 : width > 0) :
  let tile_size := largest_tile_size length width
  ∀ n : ℕ, n > tile_size → ¬(n ∣ length ∧ n ∣ width) :=
by sorry

theorem courtyard_largest_tile_size :
  largest_tile_size 378 595 = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_tile_size_is_correct_courtyard_largest_tile_size_l3890_389039


namespace NUMINAMATH_CALUDE_fraction_expression_value_l3890_389048

theorem fraction_expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 18 / 7 + x / (2 * q + p) = 3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_value_l3890_389048


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l3890_389067

-- Define the properties of function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 6) = f x) ∧
  (f 1 = 1)

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : 
  f 2015 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l3890_389067


namespace NUMINAMATH_CALUDE_loads_required_l3890_389056

def washing_machine_capacity : ℕ := 9
def total_clothing : ℕ := 27

theorem loads_required : (total_clothing + washing_machine_capacity - 1) / washing_machine_capacity = 3 := by
  sorry

end NUMINAMATH_CALUDE_loads_required_l3890_389056


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3890_389027

-- Define the parabolas
def P₁ (x y : ℝ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℝ) : Prop := x = y^2 + 95/8

-- Define the common tangent line
def CommonTangent (a b c : ℕ) (x y : ℝ) : Prop :=
  (a : ℝ) * x + (b : ℝ) * y = c

-- Main theorem
theorem common_tangent_sum :
  ∃ (a b c : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (m : ℚ), ∀ (x y : ℝ),
      CommonTangent a b c x y → y = m * x + (c / b : ℝ)) ∧
    (∀ (x y : ℝ),
      (P₁ x y → ∃ (x₀ y₀ : ℝ), P₁ x₀ y₀ ∧ CommonTangent a b c x₀ y₀) ∧
      (P₂ x y → ∃ (x₀ y₀ : ℝ), P₂ x₀ y₀ ∧ CommonTangent a b c x₀ y₀)) ∧
    a + b + c = 59 := by
  sorry


end NUMINAMATH_CALUDE_common_tangent_sum_l3890_389027


namespace NUMINAMATH_CALUDE_jan_extra_miles_l3890_389020

theorem jan_extra_miles (t s : ℝ) 
  (ian_distance : ℝ → ℝ → ℝ)
  (han_distance : ℝ → ℝ → ℝ)
  (jan_distance : ℝ → ℝ → ℝ)
  (h1 : ian_distance t s = s * t)
  (h2 : han_distance t s = (s + 10) * (t + 2))
  (h3 : han_distance t s = ian_distance t s + 100)
  (h4 : jan_distance t s = (s + 15) * (t + 3)) :
  jan_distance t s - ian_distance t s = 165 := by
sorry

end NUMINAMATH_CALUDE_jan_extra_miles_l3890_389020


namespace NUMINAMATH_CALUDE_mcnugget_theorem_l3890_389017

/-- Represents the possible package sizes for Chicken McNuggets -/
def nugget_sizes : List ℕ := [6, 9, 20]

/-- Checks if a number can be expressed as a combination of nugget sizes -/
def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

/-- The largest number that cannot be ordered -/
def largest_unorderable : ℕ := 43

/-- Main theorem: 43 is the largest number that cannot be ordered -/
theorem mcnugget_theorem :
  (∀ m > largest_unorderable, is_orderable m) ∧
  ¬(is_orderable largest_unorderable) :=
sorry

end NUMINAMATH_CALUDE_mcnugget_theorem_l3890_389017


namespace NUMINAMATH_CALUDE_action_movies_rented_l3890_389036

theorem action_movies_rented (a : ℝ) : 
  let total_movies := 10 * a / 0.64
  let comedy_movies := 10 * a
  let non_comedy_movies := total_movies - comedy_movies
  let drama_movies := 5 * (non_comedy_movies / 6)
  let action_movies := non_comedy_movies / 6
  action_movies = 0.9375 * a := by
sorry

end NUMINAMATH_CALUDE_action_movies_rented_l3890_389036


namespace NUMINAMATH_CALUDE_square_area_16m_l3890_389069

/-- The area of a square with side length 16 meters is 256 square meters. -/
theorem square_area_16m (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_area_16m_l3890_389069


namespace NUMINAMATH_CALUDE_rays_initial_cents_l3890_389054

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The amount of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of nickels Peter receives -/
def nickels_to_peter : ℕ := cents_to_peter / nickel_value

/-- The amount of cents Ray gives to Randi -/
def cents_to_randi : ℕ := 2 * cents_to_peter

/-- The number of nickels Randi receives -/
def nickels_to_randi : ℕ := cents_to_randi / nickel_value

/-- The difference in nickels between Randi and Peter -/
def nickel_difference : ℕ := 6

theorem rays_initial_cents :
  nickels_to_randi = nickels_to_peter + nickel_difference →
  cents_to_peter + cents_to_randi = 90 := by
  sorry

end NUMINAMATH_CALUDE_rays_initial_cents_l3890_389054


namespace NUMINAMATH_CALUDE_soaking_solution_l3890_389043

/-- Represents the time needed to soak clothes for each type of stain -/
structure SoakingTime where
  grass : ℕ
  marinara : ℕ

/-- Conditions for the soaking problem -/
def soaking_problem (t : SoakingTime) : Prop :=
  t.marinara = t.grass + 7 ∧ 
  3 * t.grass + t.marinara = 19

/-- Theorem stating the solution to the soaking problem -/
theorem soaking_solution :
  ∃ (t : SoakingTime), soaking_problem t ∧ t.grass = 3 := by
  sorry

end NUMINAMATH_CALUDE_soaking_solution_l3890_389043


namespace NUMINAMATH_CALUDE_school_students_count_l3890_389058

theorem school_students_count (boys girls : ℕ) 
  (h1 : 2 * boys / 3 + 3 * girls / 4 = 550)
  (h2 : 3 * girls / 4 = 150) : 
  boys + girls = 800 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l3890_389058


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3890_389031

/-- A quadratic function with vertex at (-3, 2) passing through (2, -43) has a = -9/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) → 
  (a * 2^2 + b * 2 + c = -43) →
  a = -9/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3890_389031


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3890_389037

theorem fraction_to_decimal : (17 : ℚ) / (2^2 * 5^4) = (68 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3890_389037


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l3890_389071

/-- The number of ways to assign volunteers to areas --/
def assignmentCount (volunteers : ℕ) (areas : ℕ) : ℕ :=
  areas^volunteers - areas * (areas - 1)^volunteers + areas * (areas - 2)^volunteers

/-- Theorem stating that the number of ways to assign 5 volunteers to 3 areas,
    with at least one volunteer in each area, is equal to 150 --/
theorem volunteer_assignment_count :
  assignmentCount 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l3890_389071


namespace NUMINAMATH_CALUDE_no_complex_numbers_satisfying_condition_l3890_389014

theorem no_complex_numbers_satisfying_condition : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (k • a + l • b + m • c) > 1 / h) :=
by sorry

end NUMINAMATH_CALUDE_no_complex_numbers_satisfying_condition_l3890_389014


namespace NUMINAMATH_CALUDE_total_amount_is_2500_l3890_389023

/-- Proves that the total amount of money divided into two parts is 2500,
    given the conditions from the original problem. -/
theorem total_amount_is_2500 
  (total : ℝ) 
  (part1 : ℝ) 
  (part2 : ℝ) 
  (h1 : total = part1 + part2)
  (h2 : part1 = 1000)
  (h3 : 0.05 * part1 + 0.06 * part2 = 140) :
  total = 2500 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_2500_l3890_389023
