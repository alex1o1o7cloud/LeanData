import Mathlib

namespace NUMINAMATH_CALUDE_max_cookies_eaten_l1435_143593

/-- Given 36 cookies shared among three siblings, where one sibling eats twice as many as another,
    and the third eats the same as the second, the maximum number of cookies the second sibling
    could have eaten is 9. -/
theorem max_cookies_eaten (total_cookies : ℕ) (andy bella charlie : ℕ) : 
  total_cookies = 36 →
  bella = 2 * andy →
  charlie = andy →
  total_cookies = andy + bella + charlie →
  andy ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l1435_143593


namespace NUMINAMATH_CALUDE_women_group_size_l1435_143596

/-- The number of women in the first group -/
def first_group_size : ℕ := 6

/-- The length of cloth colored by the first group -/
def first_group_cloth_length : ℕ := 180

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of women in the second group -/
def second_group_size : ℕ := 5

/-- The length of cloth colored by the second group -/
def second_group_cloth_length : ℕ := 200

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 4

theorem women_group_size :
  first_group_size * second_group_cloth_length * first_group_days =
  second_group_size * first_group_cloth_length * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_women_group_size_l1435_143596


namespace NUMINAMATH_CALUDE_kanul_cash_percentage_l1435_143510

def total_amount : ℝ := 5555.56
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 2000

theorem kanul_cash_percentage :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_kanul_cash_percentage_l1435_143510


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1435_143530

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of functions satisfying the functional equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1435_143530


namespace NUMINAMATH_CALUDE_ladder_matches_l1435_143561

/-- Represents the number of matches needed for a ladder with a given number of steps. -/
def matches_for_ladder (steps : ℕ) : ℕ :=
  6 * steps

theorem ladder_matches :
  matches_for_ladder 3 = 18 →
  matches_for_ladder 25 = 150 :=
by sorry

end NUMINAMATH_CALUDE_ladder_matches_l1435_143561


namespace NUMINAMATH_CALUDE_company_workforce_l1435_143551

theorem company_workforce (initial_workforce : ℕ) : 
  (initial_workforce * 60 = initial_workforce * 100 * 3 / 5) →
  ((initial_workforce * 60 : ℕ) = ((initial_workforce + 28) * 55 : ℕ)) →
  (initial_workforce + 28 = 336) := by
  sorry

end NUMINAMATH_CALUDE_company_workforce_l1435_143551


namespace NUMINAMATH_CALUDE_train_speed_ratio_l1435_143577

/-- Prove that the ratio of the speeds of two trains is 2:1 given specific conditions --/
theorem train_speed_ratio :
  let train_length : ℝ := 150  -- Length of each train in meters
  let crossing_time : ℝ := 8   -- Time taken to cross in seconds
  let faster_speed : ℝ := 90   -- Speed of faster train in km/h

  let total_distance : ℝ := 2 * train_length
  let relative_speed : ℝ := total_distance / crossing_time
  let faster_speed_ms : ℝ := faster_speed * 1000 / 3600
  let slower_speed_ms : ℝ := relative_speed - faster_speed_ms

  (faster_speed_ms / slower_speed_ms : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l1435_143577


namespace NUMINAMATH_CALUDE_profit_for_five_yuan_reduction_optimal_price_reduction_l1435_143581

/-- Represents the product details and sales dynamics -/
structure ProductSales where
  cost : ℕ  -- Cost per unit in yuan
  originalPrice : ℕ  -- Original selling price per unit in yuan
  initialSales : ℕ  -- Initial sales volume
  salesIncrease : ℕ  -- Increase in sales for every 1 yuan price reduction

/-- Calculates the profit for a given price reduction -/
def calculateProfit (p : ProductSales) (priceReduction : ℕ) : ℕ :=
  let newPrice := p.originalPrice - priceReduction
  let newSales := p.initialSales + p.salesIncrease * priceReduction
  (newPrice - p.cost) * newSales

/-- Theorem for the profit calculation with a 5 yuan price reduction -/
theorem profit_for_five_yuan_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  calculateProfit p 5 = 2700 := by sorry

/-- Theorem for the optimal price reduction to achieve 2860 yuan profit -/
theorem optimal_price_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  ∃ (x : ℕ), calculateProfit p x = 2860 ∧ 
    ∀ (y : ℕ), calculateProfit p y = 2860 → x ≤ y := by sorry

end NUMINAMATH_CALUDE_profit_for_five_yuan_reduction_optimal_price_reduction_l1435_143581


namespace NUMINAMATH_CALUDE_bottles_bought_l1435_143527

theorem bottles_bought (initial bottles_drunk final : ℕ) : 
  initial = 42 → bottles_drunk = 25 → final = 47 → 
  final - (initial - bottles_drunk) = 30 := by sorry

end NUMINAMATH_CALUDE_bottles_bought_l1435_143527


namespace NUMINAMATH_CALUDE_average_breadth_is_18_l1435_143518

/-- Represents a trapezoidal plot with equal diagonal distances -/
structure TrapezoidalPlot where
  averageBreadth : ℝ
  maximumLength : ℝ
  area : ℝ

/-- The conditions of the problem -/
def PlotConditions (plot : TrapezoidalPlot) : Prop :=
  plot.area = 23 * plot.averageBreadth ∧
  plot.maximumLength - plot.averageBreadth = 10 ∧
  plot.area = (1/2) * (plot.maximumLength + plot.averageBreadth) * plot.averageBreadth

/-- The theorem to be proved -/
theorem average_breadth_is_18 (plot : TrapezoidalPlot) 
  (h : PlotConditions plot) : plot.averageBreadth = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_breadth_is_18_l1435_143518


namespace NUMINAMATH_CALUDE_five_objects_three_containers_l1435_143586

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def num_ways (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to put 5 distinguishable objects into 3 distinguishable containers is 3^5 -/
theorem five_objects_three_containers : num_ways 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_objects_three_containers_l1435_143586


namespace NUMINAMATH_CALUDE_min_distance_PQ_l1435_143537

def f (x : ℝ) : ℝ := x^2 - 2*x

def distance_squared (x : ℝ) : ℝ := (x - 4)^2 + (f x + 1)^2

theorem min_distance_PQ :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x : ℝ), Real.sqrt (distance_squared x) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_PQ_l1435_143537


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l1435_143558

theorem smallest_n_inequality (x y z : ℝ) :
  (∃ (n : ℕ), ∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) ∧
  (∀ (n : ℕ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) → n ≥ 3) ∧
  ((x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l1435_143558


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l1435_143595

/-- Given that point P (x₀, y₀) and point A (1, 2) are on opposite sides of the line l: 3x + 2y - 8 = 0,
    prove that 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) < 0 → 3*x₀ + 2*y₀ > 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l1435_143595


namespace NUMINAMATH_CALUDE_fruit_buckets_l1435_143526

theorem fruit_buckets (bucketA bucketB bucketC : ℕ) : 
  bucketA = bucketB + 4 →
  bucketB = bucketC + 3 →
  bucketA + bucketB + bucketC = 37 →
  bucketC = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_buckets_l1435_143526


namespace NUMINAMATH_CALUDE_positive_real_solution_l1435_143529

theorem positive_real_solution (x : ℝ) : 
  x > 0 → x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 → 
  15 * x^2 + 32 * x - 256 = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l1435_143529


namespace NUMINAMATH_CALUDE_store_traffic_proof_l1435_143528

/-- The number of people who entered the store in the first hour -/
def first_hour_entries : ℕ := 94

/-- The number of people who entered the store in the second hour -/
def second_hour_entries : ℕ := 18

/-- The number of people who left the store in the second hour -/
def second_hour_exits : ℕ := 9

/-- The number of people in the store after two hours -/
def final_count : ℕ := 76

/-- The number of people who left during the first hour -/
def first_hour_exits : ℕ := 27

theorem store_traffic_proof :
  first_hour_entries - first_hour_exits + second_hour_entries - second_hour_exits = final_count :=
by sorry

end NUMINAMATH_CALUDE_store_traffic_proof_l1435_143528


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l1435_143509

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l1435_143509


namespace NUMINAMATH_CALUDE_rectangle_area_l1435_143556

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width →
  2 * length + 2 * width = 200 →
  length * width = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1435_143556


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1435_143570

theorem binomial_expansion_coefficient (a : ℝ) : 
  (6 : ℕ) * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1435_143570


namespace NUMINAMATH_CALUDE_xy_system_solution_l1435_143535

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) :
  x^2 + y^2 = 8044 / 169 := by
sorry

end NUMINAMATH_CALUDE_xy_system_solution_l1435_143535


namespace NUMINAMATH_CALUDE_jessica_roses_cut_l1435_143502

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := 99

theorem jessica_roses_cut :
  let initial_roses : ℕ := 17
  let roses_thrown : ℕ := 8
  let roses_now : ℕ := 42
  let roses_given : ℕ := 6
  (initial_roses - roses_thrown + roses_cut / 3 = roses_now) ∧
  (roses_cut / 3 + roses_given = roses_now - initial_roses + roses_thrown + roses_given) →
  roses_cut = 99 := by
sorry

end NUMINAMATH_CALUDE_jessica_roses_cut_l1435_143502


namespace NUMINAMATH_CALUDE_deal_or_no_deal_boxes_l1435_143575

theorem deal_or_no_deal_boxes (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  high_value_boxes = 7 →
  (high_value_boxes : ℚ) / ((total_boxes - eliminated_boxes) : ℚ) ≥ 2 / 3 →
  eliminated_boxes ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_boxes_l1435_143575


namespace NUMINAMATH_CALUDE_lawrence_county_houses_l1435_143543

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_houses_l1435_143543


namespace NUMINAMATH_CALUDE_min_value_constraint_l1435_143541

theorem min_value_constraint (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_constraint : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀^3 * y₀^2 * z₀ = 1 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_constraint_l1435_143541


namespace NUMINAMATH_CALUDE_square_of_zero_is_not_positive_l1435_143503

theorem square_of_zero_is_not_positive : ¬ (∀ x : ℕ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_zero_is_not_positive_l1435_143503


namespace NUMINAMATH_CALUDE_intersection_point_on_circle_l1435_143576

theorem intersection_point_on_circle (m : ℝ) :
  ∃ (x y r : ℝ),
    r > 0 ∧
    m * x + y + 2 * m = 0 ∧
    x - m * y + 2 * m = 0 ∧
    (x - 2)^2 + (y - 4)^2 = r^2 →
    2 * Real.sqrt 2 ≤ r ∧ r ≤ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_circle_l1435_143576


namespace NUMINAMATH_CALUDE_max_product_under_constraints_l1435_143508

theorem max_product_under_constraints (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : 10 * x + 15 * y = 150) (h2 : x^2 + y^2 ≤ 100) :
  x * y ≤ 37.5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
  10 * x₀ + 15 * y₀ = 150 ∧ x₀^2 + y₀^2 ≤ 100 ∧ x₀ * y₀ = 37.5 :=
sorry

end NUMINAMATH_CALUDE_max_product_under_constraints_l1435_143508


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l1435_143500

theorem least_positive_linear_combination (x y z : ℤ) : 
  ∃ (a b c : ℤ), 24*a + 20*b + 12*c = 4 ∧ 
  (∀ (x y z : ℤ), 24*x + 20*y + 12*z = 0 ∨ |24*x + 20*y + 12*z| ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l1435_143500


namespace NUMINAMATH_CALUDE_percentage_difference_l1435_143512

theorem percentage_difference (x y : ℝ) (h : x = 0.5 * y) : y = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1435_143512


namespace NUMINAMATH_CALUDE_school_dinosaur_cost_l1435_143566

def dinosaur_model_cost : ℕ := 100

def kindergarten_models : ℕ := 2
def elementary_models : ℕ := 2 * kindergarten_models
def high_school_models : ℕ := 3 * kindergarten_models

def total_models : ℕ := kindergarten_models + elementary_models + high_school_models

def discount_rate : ℚ :=
  if total_models > 10 then 1/10
  else if total_models > 5 then 1/20
  else 0

def discounted_price : ℚ := dinosaur_model_cost * (1 - discount_rate)

def total_cost : ℚ := total_models * discounted_price

theorem school_dinosaur_cost : total_cost = 1080 := by
  sorry

end NUMINAMATH_CALUDE_school_dinosaur_cost_l1435_143566


namespace NUMINAMATH_CALUDE_range_of_power_function_l1435_143506

theorem range_of_power_function (k c : ℝ) (h_k : k > 0) :
  Set.range (fun x => x^k + c) = Set.Ici (1 + c) := by sorry

end NUMINAMATH_CALUDE_range_of_power_function_l1435_143506


namespace NUMINAMATH_CALUDE_correct_division_result_l1435_143559

theorem correct_division_result (wrong_divisor correct_divisor student_answer : ℕ) 
  (h1 : wrong_divisor = 840)
  (h2 : correct_divisor = 420)
  (h3 : student_answer = 36) :
  (wrong_divisor * student_answer) / correct_divisor = 72 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l1435_143559


namespace NUMINAMATH_CALUDE_rising_number_Q_l1435_143580

/-- Definition of a rising number -/
def is_rising_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ 
  a < b ∧ b < c ∧ c < d ∧ a + d = b + c

/-- Function F as defined in the problem -/
def F (m : ℕ) : ℚ :=
  let m' := 1000*(m/10%10) + 100*(m/100%10) + 10*(m/1000) + (m%10)
  (m' - m) / 99

/-- Main theorem -/
theorem rising_number_Q (P Q : ℕ) (x y z t : ℕ) : 
  is_rising_number P ∧ 
  is_rising_number Q ∧
  P = 1000 + 100*x + 10*y + z ∧
  Q = 1000*x + 100*t + 60 + z ∧
  ∃ (k : ℤ), F P + F Q = k * 7 →
  Q = 3467 := by sorry

end NUMINAMATH_CALUDE_rising_number_Q_l1435_143580


namespace NUMINAMATH_CALUDE_max_value_expression_l1435_143599

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + 4 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1435_143599


namespace NUMINAMATH_CALUDE_train_travel_time_l1435_143590

theorem train_travel_time (initial_time : ℝ) (increase1 increase2 increase3 : ℝ) :
  initial_time = 19.5 ∧ 
  increase1 = 0.3 ∧ 
  increase2 = 0.25 ∧ 
  increase3 = 0.2 → 
  initial_time / ((1 + increase1) * (1 + increase2) * (1 + increase3)) = 10 := by
sorry

end NUMINAMATH_CALUDE_train_travel_time_l1435_143590


namespace NUMINAMATH_CALUDE_star_value_l1435_143579

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  star a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1435_143579


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l1435_143592

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  |x₁ - x₂| = 1 →
  |p - q| = 1 →
  (∃ (p' q' x₁' x₂' : ℤ), p = p' ∧ q = q' ∧ x₁ = x₁' ∧ x₂ = x₂') := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l1435_143592


namespace NUMINAMATH_CALUDE_min_distance_squared_l1435_143520

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)² + (b-d)² is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1435_143520


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1435_143516

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: The number of distinct diagonals in a convex dodecagon is 54 -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1435_143516


namespace NUMINAMATH_CALUDE_angle_four_value_l1435_143594

/-- Given an isosceles triangle and some angle relationships, prove that angle 4 is 37.5 degrees -/
theorem angle_four_value (angle1 angle2 angle3 angle4 angle5 x y : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle3 + angle4 + angle5 = 180 →
  angle1 = 45 + x →
  angle3 = 30 + y →
  x = 2 * y →
  angle4 = 37.5 := by
sorry

end NUMINAMATH_CALUDE_angle_four_value_l1435_143594


namespace NUMINAMATH_CALUDE_balloon_difference_l1435_143504

-- Define the initial conditions
def allan_initial : ℕ := 6
def jake_initial : ℕ := 2
def jake_bought : ℕ := 3
def allan_bought : ℕ := 4
def claire_from_jake : ℕ := 2
def claire_from_allan : ℕ := 3

-- Theorem statement
theorem balloon_difference :
  (allan_initial + allan_bought - claire_from_allan) -
  (jake_initial + jake_bought - claire_from_jake) = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l1435_143504


namespace NUMINAMATH_CALUDE_elephant_entry_rate_utopia_park_elephant_rate_l1435_143572

/-- Calculates the rate at which new elephants entered Utopia National Park --/
theorem elephant_entry_rate (initial_elephants : ℕ) (exodus_rate : ℕ) (exodus_duration : ℕ) 
  (entry_duration : ℕ) (final_elephants : ℕ) : ℕ :=
  let elephants_left := exodus_rate * exodus_duration
  let elephants_after_exodus := initial_elephants - elephants_left
  let new_elephants := final_elephants - elephants_after_exodus
  new_elephants / entry_duration

/-- Proves that the rate of new elephants entering the park is 1500 per hour --/
theorem utopia_park_elephant_rate : 
  elephant_entry_rate 30000 2880 4 7 28980 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_elephant_entry_rate_utopia_park_elephant_rate_l1435_143572


namespace NUMINAMATH_CALUDE_independent_events_probability_l1435_143505

theorem independent_events_probability (a b : Set α) (p : Set α → ℚ) :
  (p a = 4/7) → (p b = 2/5) → (∀ x y, p (x ∩ y) = p x * p y) → p (a ∩ b) = 8/35 := by
  sorry

end NUMINAMATH_CALUDE_independent_events_probability_l1435_143505


namespace NUMINAMATH_CALUDE_balloons_in_park_l1435_143521

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 4

/-- The total number of balloons Allan and Jake have in the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l1435_143521


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1435_143513

theorem polynomial_divisibility (k n : ℕ+) 
  (h : (k : ℝ) + 1 ≤ Real.sqrt ((n : ℝ) + 1 / Real.log (n + 1))) :
  ∃ (P : Polynomial ℤ), 
    (∀ i, P.coeff i ∈ ({0, 1, -1} : Set ℤ)) ∧ 
    P.degree = n ∧ 
    (X - 1 : Polynomial ℤ)^(k : ℕ) ∣ P :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1435_143513


namespace NUMINAMATH_CALUDE_irrational_sqrt_7_and_others_rational_l1435_143560

theorem irrational_sqrt_7_and_others_rational : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 7 = (a : ℝ) / (b : ℝ)) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (4 : ℝ) / 3 = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℝ) / (b : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_7_and_others_rational_l1435_143560


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_two_l1435_143531

theorem sum_of_roots_eq_two :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 5) - 19
  (∃ a b : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_two_l1435_143531


namespace NUMINAMATH_CALUDE_incorrect_statement_l1435_143548

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1435_143548


namespace NUMINAMATH_CALUDE_tyson_basketball_score_l1435_143547

theorem tyson_basketball_score (three_pointers : ℕ) (one_pointers : ℕ) (total_score : ℕ) :
  three_pointers = 15 →
  one_pointers = 6 →
  total_score = 75 →
  ∃ (two_pointers : ℕ), two_pointers = 12 ∧ 
    3 * three_pointers + 2 * two_pointers + one_pointers = total_score :=
by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_score_l1435_143547


namespace NUMINAMATH_CALUDE_complement_of_union_l1435_143533

open Set

theorem complement_of_union (U M N : Set ℕ) : 
  U = {1, 2, 3, 4} →
  M = {1, 2} →
  N = {2, 3} →
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1435_143533


namespace NUMINAMATH_CALUDE_value_of_y_l1435_143591

theorem value_of_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1435_143591


namespace NUMINAMATH_CALUDE_algorithm_correctness_l1435_143540

def sum_2i (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

theorem algorithm_correctness :
  (sum_2i 3 = 12) ∧
  (∀ m : ℕ, sum_2i m = 30 → m ≥ 5) ∧
  (sum_2i 5 = 30) := by
sorry

end NUMINAMATH_CALUDE_algorithm_correctness_l1435_143540


namespace NUMINAMATH_CALUDE_same_number_of_atoms_l1435_143525

/-- The number of atoms in a mole of a substance -/
def atoms_per_mole (substance : String) : ℕ :=
  match substance with
  | "H₃PO₄" => 8
  | "H₂O₂" => 4
  | _ => 0

/-- The number of moles of a substance -/
def moles (substance : String) : ℚ :=
  match substance with
  | "H₃PO₄" => 1/5
  | "H₂O₂" => 2/5
  | _ => 0

/-- The total number of atoms in a given amount of a substance -/
def total_atoms (substance : String) : ℚ :=
  (moles substance) * (atoms_per_mole substance)

theorem same_number_of_atoms : total_atoms "H₃PO₄" = total_atoms "H₂O₂" := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_atoms_l1435_143525


namespace NUMINAMATH_CALUDE_rope_cutting_l1435_143588

/-- Proves that a 200-meter rope cut into equal parts, with half given away and the rest subdivided,
    results in 25-meter pieces if and only if it was initially cut into 8 parts. -/
theorem rope_cutting (total_length : ℕ) (final_piece_length : ℕ) (initial_parts : ℕ) : 
  total_length = 200 ∧ 
  final_piece_length = 25 ∧
  (initial_parts : ℚ) * final_piece_length = total_length ∧
  (initial_parts / 2 : ℚ) * 2 * final_piece_length = total_length →
  initial_parts = 8 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l1435_143588


namespace NUMINAMATH_CALUDE_r_profit_share_l1435_143549

/-- Represents a partner in the business partnership --/
inductive Partner
| P
| Q
| R

/-- Represents the initial share ratio of each partner --/
def initial_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- The number of months after which P withdraws half of their capital --/
def withdrawal_month : Nat := 2

/-- The total number of months for the profit calculation --/
def total_months : Nat := 12

/-- The total profit to be divided --/
def total_profit : ℚ := 378

/-- Calculates the effective share ratio for a partner over the entire period --/
def effective_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => (initial_share_ratio Partner.P * withdrawal_month + initial_share_ratio Partner.P / 2 * (total_months - withdrawal_month)) / total_months
  | _ => initial_share_ratio p

/-- Calculates a partner's share of the profit --/
def profit_share (p : Partner) : ℚ :=
  (effective_share_ratio p / (effective_share_ratio Partner.P + effective_share_ratio Partner.Q + effective_share_ratio Partner.R)) * total_profit

/-- The main theorem stating R's share of the profit --/
theorem r_profit_share : profit_share Partner.R = 108 := by
  sorry


end NUMINAMATH_CALUDE_r_profit_share_l1435_143549


namespace NUMINAMATH_CALUDE_polynomial_sum_l1435_143562

theorem polynomial_sum (m : ℝ) : (m^2 + m) + (-3*m) = m^2 - 2*m := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1435_143562


namespace NUMINAMATH_CALUDE_fabian_shopping_cost_l1435_143563

/-- Calculates the total cost of Fabian's shopping --/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Proves that the total cost of Fabian's shopping is $16 --/
theorem fabian_shopping_cost :
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fabian_shopping_cost_l1435_143563


namespace NUMINAMATH_CALUDE_incorrect_transformation_l1435_143574

theorem incorrect_transformation :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b c : ℝ, c ≠ 0 → a / c = b / c → a = b) ∧
  (∀ a b c : ℝ, a = b → a / (c^2 + 1) = b / (c^2 + 1)) ∧
  ¬(∀ a b c : ℝ, a * c = b * c → a = b) := by
sorry


end NUMINAMATH_CALUDE_incorrect_transformation_l1435_143574


namespace NUMINAMATH_CALUDE_sample_capacity_l1435_143539

/-- Given a sample divided into groups, prove that the sample capacity is 320
    when a certain group has a frequency of 40 and a rate of 0.125. -/
theorem sample_capacity (frequency : ℕ) (rate : ℝ) (n : ℕ) 
  (h1 : frequency = 40)
  (h2 : rate = 0.125)
  (h3 : (rate : ℝ) * n = frequency) : 
  n = 320 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l1435_143539


namespace NUMINAMATH_CALUDE_chord_length_intercepted_by_line_l1435_143598

/-- The chord length intercepted by a line on a circle -/
theorem chord_length_intercepted_by_line (x y : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 4}
  let line : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ p q : ℝ × ℝ, p ∈ circle ∧ q ∈ circle ∧ p ∈ line ∧ q ∈ line ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) :=
by
  sorry


end NUMINAMATH_CALUDE_chord_length_intercepted_by_line_l1435_143598


namespace NUMINAMATH_CALUDE_fraction_multiplication_and_addition_l1435_143552

theorem fraction_multiplication_and_addition : (2 : ℚ) / 9 * 5 / 11 + 1 / 3 = 43 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_and_addition_l1435_143552


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l1435_143501

/-- The number of faces in a prism with a given number of edges -/
def prism_faces (edges : ℕ) : ℕ :=
  2 + edges / 3

/-- Theorem: A prism with 21 edges has 9 faces -/
theorem prism_with_21_edges_has_9_faces :
  prism_faces 21 = 9 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l1435_143501


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1435_143571

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

-- Theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1435_143571


namespace NUMINAMATH_CALUDE_valera_car_position_l1435_143587

/-- Represents a train with a fixed number of cars -/
structure Train :=
  (num_cars : ℕ)

/-- Represents the meeting of two trains -/
structure TrainMeeting :=
  (train1 : Train)
  (train2 : Train)
  (total_passing_time : ℕ)
  (sasha_passing_time : ℕ)
  (sasha_car : ℕ)

/-- Theorem stating the position of Valera's car -/
theorem valera_car_position
  (meeting : TrainMeeting)
  (h1 : meeting.train1.num_cars = 15)
  (h2 : meeting.train2.num_cars = 15)
  (h3 : meeting.total_passing_time = 60)
  (h4 : meeting.sasha_passing_time = 28)
  (h5 : meeting.sasha_car = 3) :
  ∃ (valera_car : ℕ), valera_car = 12 :=
by sorry

end NUMINAMATH_CALUDE_valera_car_position_l1435_143587


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_five_half_l1435_143532

theorem abs_ratio_eq_sqrt_five_half (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 18*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_five_half_l1435_143532


namespace NUMINAMATH_CALUDE_baba_yaga_students_l1435_143546

theorem baba_yaga_students (total : ℕ) (boys girls : ℕ) : 
  total = 33 →
  boys + girls = total →
  22 = (2 * total) / 3 := by
  sorry

end NUMINAMATH_CALUDE_baba_yaga_students_l1435_143546


namespace NUMINAMATH_CALUDE_wrok_represents_5167_l1435_143573

/-- Represents a mapping from characters to digits -/
def CodeMapping : Type := Char → Nat

/-- The code "GREAT WORK" represents digits 0-8 respectively -/
def great_work_code (mapping : CodeMapping) : Prop :=
  mapping 'G' = 0 ∧
  mapping 'R' = 1 ∧
  mapping 'E' = 2 ∧
  mapping 'A' = 3 ∧
  mapping 'T' = 4 ∧
  mapping 'W' = 5 ∧
  mapping 'O' = 6 ∧
  mapping 'R' = 1 ∧
  mapping 'K' = 7

/-- The code word "WROK" represents a 4-digit number -/
def wrok_code (mapping : CodeMapping) : Nat :=
  mapping 'W' * 1000 + mapping 'R' * 100 + mapping 'O' * 10 + mapping 'K'

theorem wrok_represents_5167 (mapping : CodeMapping) :
  great_work_code mapping → wrok_code mapping = 5167 := by
  sorry

end NUMINAMATH_CALUDE_wrok_represents_5167_l1435_143573


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_is_three_twentythirds_l1435_143507

/-- Represents the cost of ingredients relative to raisins -/
structure RelativeCost where
  raisins : ℚ := 1
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Represents the composition of the mixture in pounds -/
structure MixtureComposition where
  raisins : ℚ := 3
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Calculates the fraction of total cost attributed to raisins -/
def raisin_cost_fraction (rc : RelativeCost) (mc : MixtureComposition) : ℚ :=
  (mc.raisins * rc.raisins) / 
  (mc.raisins * rc.raisins + mc.nuts * rc.nuts + mc.dried_berries * rc.dried_berries)

theorem raisin_cost_fraction_is_three_twentythirds 
  (rc : RelativeCost) (mc : MixtureComposition) : 
  raisin_cost_fraction rc mc = 3 / 23 := by
  sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_is_three_twentythirds_l1435_143507


namespace NUMINAMATH_CALUDE_l_shaped_grid_squares_l1435_143514

/-- Represents a modified L-shaped grid -/
structure LShapedGrid :=
  (size : Nat)
  (missing_size : Nat)
  (missing_row : Nat)
  (missing_col : Nat)

/-- Counts the number of squares in the L-shaped grid -/
def count_squares (grid : LShapedGrid) : Nat :=
  sorry

/-- The main theorem stating that the number of squares in the specific L-shaped grid is 61 -/
theorem l_shaped_grid_squares :
  let grid : LShapedGrid := {
    size := 6,
    missing_size := 2,
    missing_row := 5,
    missing_col := 1
  }
  count_squares grid = 61 := by sorry

end NUMINAMATH_CALUDE_l_shaped_grid_squares_l1435_143514


namespace NUMINAMATH_CALUDE_hyperbola_min_eccentricity_l1435_143568

/-- Given an ellipse and a hyperbola with coinciding foci, and a line intersecting
    the right branch of the hyperbola, when the eccentricity of the hyperbola is minimized,
    the equation of the hyperbola is x^2/5 - y^2/4 = 1 -/
theorem hyperbola_min_eccentricity 
  (ellipse : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (line : ℝ → ℝ → Prop)
  (h_ellipse : ∀ x y, ellipse x y ↔ x^2/16 + y^2/7 = 1)
  (h_hyperbola : ∀ a b x y, a > b ∧ b > 0 → (hyperbola a b x y ↔ x^2/a^2 - y^2/b^2 = 1))
  (h_foci : ∀ a b, hyperbola a b (-3) 0 ∧ hyperbola a b 3 0)
  (h_line : ∀ x y, line x y ↔ x - y = 1)
  (h_intersect : ∃ x y, hyperbola a b x y ∧ line x y ∧ x > 0)
  (h_min_eccentricity : ∀ a' b', (∃ x y, hyperbola a' b' x y ∧ line x y) → 
    (a^2 - b^2)/(a^2) ≤ (a'^2 - b'^2)/(a'^2)) :
  hyperbola 5 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_min_eccentricity_l1435_143568


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l1435_143515

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l1435_143515


namespace NUMINAMATH_CALUDE_harkamal_payment_l1435_143553

/-- Calculates the final amount paid after discount and tax --/
def calculate_final_amount (fruits : List (String × ℕ × ℕ)) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := (fruits.map (λ (_, quantity, price) => quantity * price)).sum
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

/-- Theorem stating the final amount Harkamal paid --/
theorem harkamal_payment : 
  let fruits := [
    ("Grapes", 8, 70),
    ("Mangoes", 9, 55),
    ("Apples", 4, 40),
    ("Oranges", 6, 30),
    ("Pineapples", 2, 90),
    ("Cherries", 5, 100)
  ]
  let discount_rate : ℚ := 5 / 100
  let tax_rate : ℚ := 10 / 100
  calculate_final_amount fruits discount_rate tax_rate = 2168375 / 1000 := by
  sorry

#eval calculate_final_amount [
  ("Grapes", 8, 70),
  ("Mangoes", 9, 55),
  ("Apples", 4, 40),
  ("Oranges", 6, 30),
  ("Pineapples", 2, 90),
  ("Cherries", 5, 100)
] (5 / 100) (10 / 100)

end NUMINAMATH_CALUDE_harkamal_payment_l1435_143553


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l1435_143585

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : boat_distance boat_speed stream_speed = 11) :
  boat_distance boat_speed (-stream_speed) = 5 := by
  sorry

#check boat_upstream_distance

end NUMINAMATH_CALUDE_boat_upstream_distance_l1435_143585


namespace NUMINAMATH_CALUDE_total_gifts_received_l1435_143569

def gifts_from_emilio : ℕ := 11
def gifts_from_jorge : ℕ := 6
def gifts_from_pedro : ℕ := 4

theorem total_gifts_received : 
  gifts_from_emilio + gifts_from_jorge + gifts_from_pedro = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_gifts_received_l1435_143569


namespace NUMINAMATH_CALUDE_x_value_from_ratio_l1435_143519

theorem x_value_from_ratio (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3) →
  x = (y^3 + 2*y - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_ratio_l1435_143519


namespace NUMINAMATH_CALUDE_triangles_drawn_l1435_143542

theorem triangles_drawn (squares pentagons total_lines : ℕ) 
  (h_squares : squares = 8)
  (h_pentagons : pentagons = 4)
  (h_total_lines : total_lines = 88) :
  ∃ (triangles : ℕ), 
    3 * triangles + 4 * squares + 5 * pentagons = total_lines ∧ 
    triangles = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangles_drawn_l1435_143542


namespace NUMINAMATH_CALUDE_unequal_probabilities_after_adding_balls_l1435_143550

/-- Represents the contents of the bag -/
structure BagContents where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing a specific color ball -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.red : ℚ)

/-- The initial contents of the bag -/
def initialBag : BagContents := { white := 1, red := 2 }

/-- The bag after adding 1 white ball and 2 red balls -/
def updatedBag : BagContents := { white := initialBag.white + 1, red := initialBag.red + 2 }

theorem unequal_probabilities_after_adding_balls :
  probability updatedBag updatedBag.white ≠ probability updatedBag updatedBag.red := by
  sorry

end NUMINAMATH_CALUDE_unequal_probabilities_after_adding_balls_l1435_143550


namespace NUMINAMATH_CALUDE_remaining_backpack_price_l1435_143567

-- Define the problem parameters
def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_sold : ℕ := 10
def dept_store_price : ℕ := 25
def total_profit : ℕ := 442

-- Define the theorem
theorem remaining_backpack_price :
  let remaining_backpacks := total_backpacks - (swap_meet_sold + dept_store_sold)
  let swap_meet_revenue := swap_meet_sold * swap_meet_price
  let dept_store_revenue := dept_store_sold * dept_store_price
  let total_revenue := total_cost + total_profit
  let remaining_revenue := total_revenue - (swap_meet_revenue + dept_store_revenue)
  remaining_revenue / remaining_backpacks = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_backpack_price_l1435_143567


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l1435_143536

theorem sum_of_odd_numbers (N : ℕ) : 
  991 + 993 + 995 + 997 + 999 = 5000 - N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l1435_143536


namespace NUMINAMATH_CALUDE_adams_to_ricks_ratio_l1435_143557

/-- Represents the cost of lunch for each person -/
structure LunchCost where
  adam : ℚ
  rick : ℚ
  jose : ℚ

/-- The conditions of the lunch scenario -/
def lunch_scenario (cost : LunchCost) : Prop :=
  cost.rick = cost.jose ∧ 
  cost.jose = 45 ∧
  cost.adam + cost.rick + cost.jose = 120

/-- The theorem stating the ratio of Adam's lunch cost to Rick's lunch cost -/
theorem adams_to_ricks_ratio (cost : LunchCost) :
  lunch_scenario cost → cost.adam / cost.rick = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adams_to_ricks_ratio_l1435_143557


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1435_143544

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬((a - b) * a^2 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1435_143544


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l1435_143582

/-- The trajectory of point P given fixed points A and B -/
def trajectory (x y : ℝ) : Prop :=
  x^2 + y^2/2 = 1 ∧ x ≠ 1 ∧ x ≠ -1

/-- The line intersecting the trajectory -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 1

/-- Theorem stating the properties of the trajectory and intersection -/
theorem trajectory_and_intersection :
  ∀ (x y : ℝ),
  (∀ (x' y' : ℝ), (y' / (x' + 1)) * (y' / (x' - 1)) = -2 → trajectory x' y') ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    trajectory x1 y1 ∧ trajectory x2 y2 ∧
    intersecting_line x1 y1 ∧ intersecting_line x2 y2 ∧
    ((x1 - x2)^2 + (y1 - y2)^2)^(1/2 : ℝ) = 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l1435_143582


namespace NUMINAMATH_CALUDE_square_of_product_l1435_143597

theorem square_of_product (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l1435_143597


namespace NUMINAMATH_CALUDE_pipe_a_fill_time_l1435_143584

/-- Represents the time (in minutes) it takes for Pipe A to fill the tank alone -/
def pipe_a_time : ℝ := 21

/-- Represents how many times faster Pipe B is compared to Pipe A -/
def pipe_b_speed_ratio : ℝ := 6

/-- Represents the time (in minutes) it takes for both pipes to fill the tank together -/
def combined_time : ℝ := 3

/-- Proves that the time taken by Pipe A to fill the tank alone is 21 minutes -/
theorem pipe_a_fill_time :
  (1 / pipe_a_time + pipe_b_speed_ratio / pipe_a_time) * combined_time = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_a_fill_time_l1435_143584


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1435_143538

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 0)

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry


end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1435_143538


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1435_143524

theorem sufficient_not_necessary (x y a m : ℝ) :
  (∀ x y a m : ℝ, (|x - a| < m ∧ |y - a| < m) → |x - y| < 2*m) ∧
  (∃ x y a m : ℝ, |x - y| < 2*m ∧ ¬(|x - a| < m ∧ |y - a| < m)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1435_143524


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l1435_143589

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The number of yellow bouncy ball packs Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of green bouncy ball packs Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of green bouncy ball packs Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℝ := yellow_packs * balls_per_pack + green_packs_bought * balls_per_pack - green_packs_given * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l1435_143589


namespace NUMINAMATH_CALUDE_car_distance_in_30_minutes_l1435_143511

-- Define the train's speed in miles per hour
def train_speed : ℚ := 100

-- Define the car's speed as a fraction of the train's speed
def car_speed : ℚ := (2/3) * train_speed

-- Define the time in hours (30 minutes = 1/2 hour)
def time : ℚ := 1/2

-- Theorem statement
theorem car_distance_in_30_minutes :
  car_speed * time = 100/3 := by sorry

end NUMINAMATH_CALUDE_car_distance_in_30_minutes_l1435_143511


namespace NUMINAMATH_CALUDE_inequality_preservation_l1435_143534

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 3 < b - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1435_143534


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1435_143578

theorem greatest_divisor_with_remainders :
  ∃ (n : ℕ), n > 0 ∧
  1255 % n = 8 ∧
  1490 % n = 11 ∧
  ∀ (m : ℕ), m > n → (1255 % m ≠ 8 ∨ 1490 % m ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1435_143578


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1435_143522

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 9 * (a + b) →
  (10 * a + b) + (10 * b + a) = 11 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1435_143522


namespace NUMINAMATH_CALUDE_circle_center_sum_l1435_143583

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 14*y - 11 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h + 14*k - 11)) →
  h + k = -4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1435_143583


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1435_143565

/-- The sum of 123.45 and 678.90 is equal to 802.35 -/
theorem sum_of_decimals : (123.45 : ℝ) + 678.90 = 802.35 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1435_143565


namespace NUMINAMATH_CALUDE_shekars_average_marks_l1435_143555

/-- Calculates the average marks given scores in five subjects -/
def averageMarks (math science socialStudies english biology : ℕ) : ℚ :=
  (math + science + socialStudies + english + biology : ℚ) / 5

/-- Theorem stating that Shekar's average marks are 75 -/
theorem shekars_average_marks :
  averageMarks 76 65 82 67 85 = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekars_average_marks_l1435_143555


namespace NUMINAMATH_CALUDE_chord_length_squared_l1435_143545

/-- Two circles with given properties and intersecting chords --/
structure CircleConfiguration where
  -- First circle radius
  r1 : ℝ
  -- Second circle radius
  r2 : ℝ
  -- Distance between circle centers
  d : ℝ
  -- Length of chord QP
  x : ℝ
  -- Ensure the configuration is valid
  h1 : r1 = 10
  h2 : r2 = 7
  h3 : d = 15
  -- QP = PR = PS = PT
  h4 : ∀ (chord : ℝ), chord = x → (chord = QP ∨ chord = PR ∨ chord = PS ∨ chord = PT)

/-- The theorem stating that the square of QP's length is 265 --/
theorem chord_length_squared (config : CircleConfiguration) : config.x^2 = 265 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l1435_143545


namespace NUMINAMATH_CALUDE_abc_inequality_l1435_143554

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) = 2) :
  a * b + b * c + c * a ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1435_143554


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1435_143517

/-- Given a geometric series with first term a and common ratio r,
    prove that if the sum of the series is 20 and the sum of terms
    with odd powers of r is 8, then r = √(11/12) -/
theorem geometric_series_ratio (a r : ℝ) (h₁ : a ≠ 0) (h₂ : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = Real.sqrt (11/12) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1435_143517


namespace NUMINAMATH_CALUDE_storks_count_l1435_143523

theorem storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_total : ℕ) : 
  initial_birds = 6 → additional_birds = 4 → final_total = 10 →
  final_total = initial_birds + additional_birds →
  0 = final_total - (initial_birds + additional_birds) :=
by sorry

end NUMINAMATH_CALUDE_storks_count_l1435_143523


namespace NUMINAMATH_CALUDE_computer_operations_l1435_143564

theorem computer_operations (additions_per_second multiplications_per_second : ℕ) 
  (h1 : additions_per_second = 12000)
  (h2 : multiplications_per_second = 8000) :
  (additions_per_second + multiplications_per_second) * (30 * 60) = 36000000 := by
  sorry

#check computer_operations

end NUMINAMATH_CALUDE_computer_operations_l1435_143564
