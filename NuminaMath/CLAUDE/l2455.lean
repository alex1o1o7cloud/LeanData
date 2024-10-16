import Mathlib

namespace NUMINAMATH_CALUDE_measure_S_eq_one_l2455_245523

open MeasureTheory

/-- The set of times where car A has completed twice as many laps as car B -/
def S (α : ℝ) : Set ℝ :=
  {t : ℝ | t ≥ α ∧ ⌊t⌋ = 2 * ⌊t - α⌋}

/-- The theorem stating that the measure of S is 1 -/
theorem measure_S_eq_one (α : ℝ) (hα : α > 0) :
  volume (S α) = 1 := by sorry

end NUMINAMATH_CALUDE_measure_S_eq_one_l2455_245523


namespace NUMINAMATH_CALUDE_flashlight_distance_ratio_l2455_245531

/-- Proves that the ratio of Freddie's flashlight distance to Veronica's is 3:1 --/
theorem flashlight_distance_ratio :
  ∀ (V F : ℕ),
  V = 1000 →
  F > V →
  ∃ (D : ℕ), D = 5 * F - 2000 →
  D = V + 12000 →
  F / V = 3 :=
by sorry

end NUMINAMATH_CALUDE_flashlight_distance_ratio_l2455_245531


namespace NUMINAMATH_CALUDE_least_n_multiple_of_1000_l2455_245530

theorem least_n_multiple_of_1000 : ∃ (n : ℕ), n > 0 ∧ n = 797 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1000 ∣ (2^m + 5^m - m))) ∧ 
  (1000 ∣ (2^n + 5^n - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_n_multiple_of_1000_l2455_245530


namespace NUMINAMATH_CALUDE_quadratic_properties_l2455_245504

def f (x : ℝ) := x^2 + 6*x + 5

theorem quadratic_properties :
  (f 0 = 5) ∧
  (∃ v : ℝ × ℝ, v = (-3, -4) ∧ ∀ x : ℝ, f x ≥ f v.1) ∧
  (∀ x : ℝ, f (x + (-3)) = f ((-3) - x)) ∧
  (∀ p : ℝ, f p ≠ -p^2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2455_245504


namespace NUMINAMATH_CALUDE_value_of_a_l2455_245597

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1)

-- State the theorem
theorem value_of_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2455_245597


namespace NUMINAMATH_CALUDE_f_max_value_l2455_245500

/-- The quadratic function f(x) = 10x - 2x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

/-- The maximum value of f(x) is 12.5 -/
theorem f_max_value : ∃ (M : ℝ), M = 12.5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2455_245500


namespace NUMINAMATH_CALUDE_rectangle_area_l2455_245596

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 3 * r.breadth) 
  (h2 : r.perimeter = 88) : 
  r.length * r.breadth = 363 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l2455_245596


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2455_245546

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (f = fun x ↦ x^2 + 5*x + 6) →
  (a > 0) →
  (b > 0) →
  (∀ x, |x + 1| < b → |f x + 3| < a) ↔ (a > 11/4 ∧ b > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2455_245546


namespace NUMINAMATH_CALUDE_pool_perimeter_is_20_l2455_245586

/-- Represents the dimensions and constraints of a rectangular pool in a garden --/
structure PoolInGarden where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ
  pool_length : ℝ := garden_length - 2 * walkway_width
  pool_width : ℝ := garden_width - 2 * walkway_width

/-- Calculates the perimeter of the pool --/
def pool_perimeter (p : PoolInGarden) : ℝ :=
  2 * (p.pool_length + p.pool_width)

/-- Theorem: The perimeter of the pool is 20 meters --/
theorem pool_perimeter_is_20 (p : PoolInGarden) 
    (h1 : p.garden_length = 8)
    (h2 : p.garden_width = 6)
    (h3 : p.pool_area = 24)
    (h4 : p.pool_length * p.pool_width = p.pool_area) : 
  pool_perimeter p = 20 := by
  sorry

#check pool_perimeter_is_20

end NUMINAMATH_CALUDE_pool_perimeter_is_20_l2455_245586


namespace NUMINAMATH_CALUDE_max_percentage_both_services_l2455_245551

theorem max_percentage_both_services (internet_percentage : Real) (snack_percentage : Real) :
  internet_percentage = 0.4 →
  snack_percentage = 0.7 →
  ∃ (both_percentage : Real),
    both_percentage ≤ internet_percentage ∧
    both_percentage ≤ snack_percentage ∧
    ∀ (x : Real),
      x ≤ internet_percentage ∧
      x ≤ snack_percentage →
      x ≤ both_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_percentage_both_services_l2455_245551


namespace NUMINAMATH_CALUDE_candy_spent_approx_11_l2455_245594

/-- The amount John spent at the supermarket -/
def total_spent : ℚ := 29.999999999999996

/-- The fraction of money spent on fresh fruits and vegetables -/
def fruits_veg_fraction : ℚ := 1 / 5

/-- The fraction of money spent on meat products -/
def meat_fraction : ℚ := 1 / 3

/-- The fraction of money spent on bakery products -/
def bakery_fraction : ℚ := 1 / 10

/-- The fraction of money spent on candy -/
def candy_fraction : ℚ := 1 - (fruits_veg_fraction + meat_fraction + bakery_fraction)

/-- The amount spent on candy -/
def candy_spent : ℚ := candy_fraction * total_spent

theorem candy_spent_approx_11 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ |candy_spent - 11| < ε :=
sorry

end NUMINAMATH_CALUDE_candy_spent_approx_11_l2455_245594


namespace NUMINAMATH_CALUDE_two_digit_even_multiple_of_seven_perfect_square_digit_product_l2455_245589

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem two_digit_even_multiple_of_seven_perfect_square_digit_product :
  {n : ℕ | is_two_digit n ∧ 
           n % 2 = 0 ∧ 
           n % 7 = 0 ∧ 
           is_perfect_square (digit_product n)} = {14, 28, 70} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_even_multiple_of_seven_perfect_square_digit_product_l2455_245589


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2455_245507

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2455_245507


namespace NUMINAMATH_CALUDE_rice_purchase_problem_l2455_245506

/-- The problem of determining the amount of rice bought given the prices and quantities of different grains --/
theorem rice_purchase_problem (rice_price corn_price beans_price : ℚ)
  (total_weight total_cost : ℚ) (beans_weight : ℚ) :
  rice_price = 75 / 100 →
  corn_price = 110 / 100 →
  beans_price = 55 / 100 →
  total_weight = 36 →
  total_cost = 2835 / 100 →
  beans_weight = 8 →
  ∃ (rice_weight : ℚ), 
    (rice_weight + (total_weight - rice_weight - beans_weight) + beans_weight = total_weight) ∧
    (rice_price * rice_weight + corn_price * (total_weight - rice_weight - beans_weight) + beans_price * beans_weight = total_cost) ∧
    (abs (rice_weight - 196 / 10) < 1 / 10) :=
by sorry

end NUMINAMATH_CALUDE_rice_purchase_problem_l2455_245506


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2455_245502

/-- A line passing through point A(2, 1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(2, 1) -/
  passes_through_A : m * 2 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of the line is either x - 2y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2455_245502


namespace NUMINAMATH_CALUDE_triangle_sides_gp_implies_altitudes_gp_l2455_245581

/-- Theorem: If the sides of a triangle form a geometric progression, 
    then its altitudes also form a geometric progression. -/
theorem triangle_sides_gp_implies_altitudes_gp 
  (a q : ℝ) 
  (h_positive : a > 0 ∧ q > 0) 
  (h_sides : ∃ (s₁ s₂ s₃ : ℝ), s₁ = a ∧ s₂ = a*q ∧ s₃ = a*q^2) :
  ∃ (h₁ h₂ h₃ : ℝ), h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ 
    h₂ / h₁ = 1/q ∧ h₃ / h₂ = 1/q :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_gp_implies_altitudes_gp_l2455_245581


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l2455_245574

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 - years_since_built →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary = 95 := by
  sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l2455_245574


namespace NUMINAMATH_CALUDE_distribute_negation_l2455_245579

theorem distribute_negation (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distribute_negation_l2455_245579


namespace NUMINAMATH_CALUDE_line_passes_through_point_and_trisection_l2455_245517

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the trisection points of a line segment -/
def trisectionPoints (p1 p2 : Point) : (Point × Point) :=
  let dx := (p2.x - p1.x) / 3
  let dy := (p2.y - p1.y) / 3
  ( Point.mk (p1.x + dx) (p1.y + dy),
    Point.mk (p1.x + 2*dx) (p1.y + 2*dy) )

theorem line_passes_through_point_and_trisection :
  let p := Point.mk 2 3
  let p1 := Point.mk 1 2
  let p2 := Point.mk 6 0
  let l := Line.mk 3 1 (-9)
  let (t1, t2) := trisectionPoints p1 p2
  pointOnLine p l ∧ (pointOnLine t1 l ∨ pointOnLine t2 l) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_and_trisection_l2455_245517


namespace NUMINAMATH_CALUDE_last_year_winner_ounces_l2455_245525

/-- The amount of ounces in each hamburger -/
def hamburger_ounces : ℕ := 4

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat : ℕ := 22

/-- Theorem: The amount of ounces eaten by last year's winner is 88 -/
theorem last_year_winner_ounces : 
  hamburger_ounces * hamburgers_to_beat - hamburger_ounces = 88 := by
  sorry

end NUMINAMATH_CALUDE_last_year_winner_ounces_l2455_245525


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2455_245538

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2455_245538


namespace NUMINAMATH_CALUDE_parallel_condition_l2455_245582

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line: ax + 2y - 1 = 0 -/
def line1 (a x y : ℝ) : Prop :=
  a * x + 2 * y - 1 = 0

/-- The second line: x + 2y + 4 = 0 -/
def line2 (x y : ℝ) : Prop :=
  x + 2 * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, are_parallel a 2 (-1) 1 2 4) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_condition_l2455_245582


namespace NUMINAMATH_CALUDE_sine_sum_inequality_l2455_245522

theorem sine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  Real.sin x + Real.sin y + Real.sin z ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_inequality_l2455_245522


namespace NUMINAMATH_CALUDE_first_bear_price_correct_l2455_245529

/-- The price of the first bear in a sequence of bear prices -/
def first_bear_price : ℚ := 57 / 2

/-- The number of bears purchased -/
def num_bears : ℕ := 101

/-- The discount applied to each bear after the first -/
def discount : ℚ := 1 / 2

/-- The total cost of all bears -/
def total_cost : ℚ := 354

/-- Theorem stating that the first bear price is correct given the conditions -/
theorem first_bear_price_correct :
  (num_bears : ℚ) / 2 * (2 * first_bear_price - (num_bears - 1) * discount) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_bear_price_correct_l2455_245529


namespace NUMINAMATH_CALUDE_number_of_lineups_l2455_245520

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of regular players in the starting lineup
def regular_players : ℕ := 11

-- Define the number of goalies in the starting lineup
def goalies : ℕ := 1

-- Theorem stating the number of different starting lineups
theorem number_of_lineups : 
  (total_players.choose goalies) * ((total_players - goalies).choose regular_players) = 222768 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lineups_l2455_245520


namespace NUMINAMATH_CALUDE_complex_number_problem_l2455_245509

/-- Given a complex number z satisfying z = i(2-z), prove that z = 1 + i and |z-(2-i)| = √5 -/
theorem complex_number_problem (z : ℂ) (h : z = Complex.I * (2 - z)) : 
  z = 1 + Complex.I ∧ Complex.abs (z - (2 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2455_245509


namespace NUMINAMATH_CALUDE_transaction_gain_l2455_245570

/-- Calculates the simple interest for a given principal, rate, and time --/
def simpleInterest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (principal : ℚ) * rate * (time : ℚ) / 100

/-- Calculates the annual gain from borrowing and lending money --/
def annualGain (principal : ℕ) (borrowRate : ℚ) (lendRate : ℚ) (time : ℕ) : ℚ :=
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / (time : ℚ)

theorem transaction_gain (principal : ℕ) (borrowRate lendRate : ℚ) (time : ℕ) :
  principal = 8000 →
  borrowRate = 4 →
  lendRate = 6 →
  time = 2 →
  annualGain principal borrowRate lendRate time = 800 := by
  sorry

end NUMINAMATH_CALUDE_transaction_gain_l2455_245570


namespace NUMINAMATH_CALUDE_max_a_value_exists_max_a_l2455_245524

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 := by
  sorry

theorem exists_max_a : ∃ a b : ℕ, 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120 ∧ a = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_exists_max_a_l2455_245524


namespace NUMINAMATH_CALUDE_sister_amount_calculation_l2455_245571

-- Define the amounts received from each source
def aunt_amount : ℝ := 9
def uncle_amount : ℝ := 9
def friends_amounts : List ℝ := [22, 23, 22, 22]

-- Define the mean of all amounts
def total_mean : ℝ := 16.3

-- Define the number of sources (including sister)
def num_sources : ℕ := 7

-- Theorem to prove
theorem sister_amount_calculation :
  let total_known := aunt_amount + uncle_amount + friends_amounts.sum
  let sister_amount := total_mean * num_sources - total_known
  sister_amount = 7.1 := by sorry

end NUMINAMATH_CALUDE_sister_amount_calculation_l2455_245571


namespace NUMINAMATH_CALUDE_distance_between_foci_l2455_245559

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem: The distance between the foci is 2√29
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2455_245559


namespace NUMINAMATH_CALUDE_only_clock_hands_rotate_l2455_245558

-- Define the concept of rotation
def is_rotation (motion : String) : Prop := 
  motion = "movement around a fixed point"

-- Define the given examples
def clock_hands : String := "movement of the hands of a clock"
def car_on_road : String := "car driving on a straight road"
def bottles_on_belt : String := "bottled beverages moving on a conveyor belt"
def soccer_ball : String := "soccer ball flying into the goal"

-- Theorem to prove
theorem only_clock_hands_rotate :
  is_rotation clock_hands ∧
  ¬is_rotation car_on_road ∧
  ¬is_rotation bottles_on_belt ∧
  ¬is_rotation soccer_ball :=
by sorry


end NUMINAMATH_CALUDE_only_clock_hands_rotate_l2455_245558


namespace NUMINAMATH_CALUDE_price_increase_calculation_l2455_245532

/-- Represents the ticket pricing model for an airline -/
structure TicketPricing where
  basePrice : ℝ
  daysBeforeDeparture : ℕ
  dailyIncreaseRate : ℝ

/-- Calculates the price increase for buying a ticket one day later -/
def priceIncrease (pricing : TicketPricing) : ℝ :=
  pricing.basePrice * pricing.dailyIncreaseRate

/-- Theorem: The price increase for buying a ticket one day later is $52.50 -/
theorem price_increase_calculation (pricing : TicketPricing)
  (h1 : pricing.basePrice = 1050)
  (h2 : pricing.daysBeforeDeparture = 14)
  (h3 : pricing.dailyIncreaseRate = 0.05) :
  priceIncrease pricing = 52.50 := by
  sorry

#eval priceIncrease { basePrice := 1050, daysBeforeDeparture := 14, dailyIncreaseRate := 0.05 }

end NUMINAMATH_CALUDE_price_increase_calculation_l2455_245532


namespace NUMINAMATH_CALUDE_correct_average_calculation_l2455_245562

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 20 ∧ incorrect_num = 26 ∧ correct_num = 86 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l2455_245562


namespace NUMINAMATH_CALUDE_gcd_117_182_l2455_245539

theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

end NUMINAMATH_CALUDE_gcd_117_182_l2455_245539


namespace NUMINAMATH_CALUDE_watch_cost_price_l2455_245578

theorem watch_cost_price (cp : ℝ) : 
  (0.9 * cp = cp - 0.1 * cp) →
  (1.04 * cp = cp + 0.04 * cp) →
  (1.04 * cp = 0.9 * cp + 200) →
  cp = 1428.57 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2455_245578


namespace NUMINAMATH_CALUDE_license_plate_difference_l2455_245555

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible New York license plates -/
def new_york_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of license plates between California and New York -/
theorem license_plate_difference :
  california_plates - new_york_plates = 28121600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2455_245555


namespace NUMINAMATH_CALUDE_min_discount_factor_l2455_245568

def cost_price : ℝ := 800
def marked_price : ℝ := 1200
def min_profit_margin : ℝ := 0.2

theorem min_discount_factor (x : ℝ) : 
  (cost_price * (1 + min_profit_margin) = marked_price * x) → x = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_min_discount_factor_l2455_245568


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2455_245553

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_geometric : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2455_245553


namespace NUMINAMATH_CALUDE_jacks_estimate_is_larger_l2455_245567

theorem jacks_estimate_is_larger (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (ha : a > 0) (hb : b > 0) : 
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_CALUDE_jacks_estimate_is_larger_l2455_245567


namespace NUMINAMATH_CALUDE_age_sum_theorem_l2455_245536

theorem age_sum_theorem (f d : ℕ) (h1 : f * d = 238) (h2 : (f + 4) * (d + 4) = 378) :
  f + d = 31 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l2455_245536


namespace NUMINAMATH_CALUDE_solution_is_two_intersecting_lines_l2455_245593

/-- The set of points (x, y) satisfying the equation (x+3y)³ = x³ + 9y³ -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 3 * p.2)^3 = p.1^3 + 9 * p.2^3}

/-- A line in ℝ² defined by a*x + b*y + c = 0 -/
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem solution_is_two_intersecting_lines :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ ≠ 0 ∨ b₁ ≠ 0) ∧ 
    (a₂ ≠ 0 ∨ b₂ ≠ 0) ∧
    (a₁ * b₂ ≠ a₂ * b₁) ∧
    S = Line a₁ b₁ c₁ ∪ Line a₂ b₂ c₂ :=
  sorry

end NUMINAMATH_CALUDE_solution_is_two_intersecting_lines_l2455_245593


namespace NUMINAMATH_CALUDE_min_m_value_l2455_245549

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(2+x) = f(2-x) for all x
    and f is monotonically increasing on [m, +∞), then the minimum value of m is 2. -/
theorem min_m_value (f : ℝ → ℝ) (a : ℝ) (m : ℝ) :
  (∀ x, f x = 2^(|x - a|)) →
  (∀ x, f (2 + x) = f (2 - x)) →
  (∀ x y, m ≤ x → x < y → f x ≤ f y) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_m_value_l2455_245549


namespace NUMINAMATH_CALUDE_precision_of_4_028e5_l2455_245547

/-- Represents the precision of a number in scientific notation -/
inductive Precision
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Determines the precision of a number in scientific notation -/
def precision_of (n : ℝ) (e : ℤ) : Precision :=
  match e with
  | 0 => Precision.Ones
  | 1 => Precision.Tens
  | 2 => Precision.Hundreds
  | 3 => Precision.Thousands
  | _ => Precision.Ones  -- Default case

theorem precision_of_4_028e5 :
  precision_of 4.028 5 = Precision.Tens :=
sorry

end NUMINAMATH_CALUDE_precision_of_4_028e5_l2455_245547


namespace NUMINAMATH_CALUDE_machine_work_time_l2455_245519

/-- Given machines A, B, and C, where B takes 3 hours and C takes 6 hours to complete a job,
    and all three machines together take 4/3 hours, prove that A takes 4 hours alone. -/
theorem machine_work_time (time_B time_C time_ABC : ℝ) (time_A : ℝ) : 
  time_B = 3 → 
  time_C = 6 → 
  time_ABC = 4/3 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_A = 4 := by
sorry

end NUMINAMATH_CALUDE_machine_work_time_l2455_245519


namespace NUMINAMATH_CALUDE_seafood_price_proof_l2455_245533

/-- The regular price of seafood given the sale price and discount -/
def regular_price (sale_price : ℚ) (discount_percent : ℚ) : ℚ :=
  sale_price / (1 - discount_percent)

/-- The price for a given weight of seafood at the regular price -/
def price_for_weight (price_per_unit : ℚ) (weight : ℚ) : ℚ :=
  price_per_unit * weight

theorem seafood_price_proof :
  let sale_price_per_pack : ℚ := 4
  let pack_weight : ℚ := 3/4
  let discount_percent : ℚ := 3/4
  let target_weight : ℚ := 3/2

  let regular_price_per_pack := regular_price sale_price_per_pack discount_percent
  let regular_price_per_pound := regular_price_per_pack / pack_weight
  
  price_for_weight regular_price_per_pound target_weight = 32 := by
  sorry

end NUMINAMATH_CALUDE_seafood_price_proof_l2455_245533


namespace NUMINAMATH_CALUDE_problem_solution_l2455_245542

theorem problem_solution (x y : ℝ) (h1 : x = 3) (h2 : y = 3) : 
  x - y^((x - y) / 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2455_245542


namespace NUMINAMATH_CALUDE_exponential_function_inequality_l2455_245543

theorem exponential_function_inequality (m n : ℝ) : 
  let a : ℝ := (Real.sqrt 5 - Real.sqrt 2) / 2
  let f : ℝ → ℝ := fun x ↦ a^x
  0 < a ∧ a < 1 → f m > f n → m < n := by sorry

end NUMINAMATH_CALUDE_exponential_function_inequality_l2455_245543


namespace NUMINAMATH_CALUDE_expression_value_l2455_245511

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  5*a + 5*b - 2 + 6*c*d - 7*m = -24 ∨ 5*a + 5*b - 2 + 6*c*d - 7*m = 32 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2455_245511


namespace NUMINAMATH_CALUDE_age_difference_proof_l2455_245590

def age_difference_in_decades (x y z : ℕ) : ℚ :=
  (x - z : ℚ) / 10

theorem age_difference_proof (x y z : ℕ) 
  (h : x + y = y + z + 15) : 
  age_difference_in_decades x y z = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2455_245590


namespace NUMINAMATH_CALUDE_k_at_neg_one_eq_64_l2455_245563

/-- The polynomial h(x) -/
def h (p : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 + 3*x + 20

/-- The polynomial k(x) -/
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 - q*x^2 + 50*x + r

/-- Theorem stating that k(-1) = 64 given the conditions -/
theorem k_at_neg_one_eq_64 (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →
  k q r (-1) = 64 :=
by sorry

end NUMINAMATH_CALUDE_k_at_neg_one_eq_64_l2455_245563


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solutions_for_2891_l2455_245577

def cubic_equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem cubic_equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ x y : ℤ, cubic_equation x y n) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ, 
    cubic_equation x₁ y₁ n ∧ 
    cubic_equation x₂ y₂ n ∧ 
    cubic_equation x₃ y₃ n ∧ 
    (x₁, y₁) ≠ (x₂, y₂) ∧ 
    (x₁, y₁) ≠ (x₃, y₃) ∧ 
    (x₂, y₂) ≠ (x₃, y₃)) :=
sorry

theorem no_solutions_for_2891 :
  ¬ ∃ x y : ℤ, cubic_equation x y 2891 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solutions_for_2891_l2455_245577


namespace NUMINAMATH_CALUDE_ae_length_l2455_245503

/-- Triangle ABC and ADE share vertex A and angle A --/
structure NestedTriangles where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  k : ℝ
  area_proportion : AB * AC = k * AD * AE

/-- The specific nested triangles in the problem --/
def problem_triangles : NestedTriangles where
  AB := 5
  AC := 7
  AD := 2
  AE := 17.5
  k := 1
  area_proportion := by sorry

theorem ae_length (t : NestedTriangles) (h1 : t.AB = 5) (h2 : t.AC = 7) (h3 : t.AD = 2) (h4 : t.k = 1) :
  t.AE = 17.5 := by
  sorry

#check ae_length problem_triangles

end NUMINAMATH_CALUDE_ae_length_l2455_245503


namespace NUMINAMATH_CALUDE_twenty_percent_of_twentyfive_percent_is_five_percent_l2455_245572

theorem twenty_percent_of_twentyfive_percent_is_five_percent :
  (20 / 100) * (25 / 100) = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_of_twentyfive_percent_is_five_percent_l2455_245572


namespace NUMINAMATH_CALUDE_water_depth_calculation_l2455_245534

-- Define the heights of Ron and Dean
def ron_height : ℝ := 13
def dean_height : ℝ := ron_height + 4

-- Define the maximum depth at high tide
def max_depth : ℝ := 15 * dean_height

-- Define the current tide percentage and current percentage
def tide_percentage : ℝ := 0.75
def current_percentage : ℝ := 0.20

-- Theorem statement
theorem water_depth_calculation :
  let current_tide_depth := tide_percentage * max_depth
  let additional_depth := current_percentage * current_tide_depth
  current_tide_depth + additional_depth = 229.5 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l2455_245534


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l2455_245598

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (∃ (x : ℝ), x^2 + p*x + q = 0) →  -- Quadratic equation has real solutions
  (x₁^2 + p*x₁ + q = 0) →           -- x₁ is a solution
  (x₂^2 + p*x₂ + q = 0) →           -- x₂ is a solution
  (x₁ ≠ x₂) →                       -- Solutions are distinct
  |x₁ - x₂| = 1 →                   -- Absolute difference of solutions is 1
  |p - q| = 1 →                     -- Absolute difference of p and q is 1
  (∃ (m n k l : ℤ), (↑m : ℝ) = p ∧ (↑n : ℝ) = q ∧ (↑k : ℝ) = x₁ ∧ (↑l : ℝ) = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l2455_245598


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2455_245501

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  ∀ x y z w : ℝ, x * y * z * w = 8 → 
  ∀ p q r s : ℝ, p * q * r * s = 16 →
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 ∧
  (∃ x y z w p q r s : ℝ, x * y * z * w = 8 ∧ p * q * r * s = 16 ∧
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 = 32) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2455_245501


namespace NUMINAMATH_CALUDE_divisibility_by_240_l2455_245554

theorem divisibility_by_240 (a b c d : ℕ) : 
  240 ∣ (a^(4*b+d) - a^(4*c+d)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l2455_245554


namespace NUMINAMATH_CALUDE_viola_count_l2455_245550

theorem viola_count (cellos : ℕ) (pairs : ℕ) (prob : ℚ) (violas : ℕ) : 
  cellos = 800 → 
  pairs = 100 → 
  prob = 100 / (800 * violas) → 
  prob = 0.00020833333333333335 → 
  violas = 600 := by
  sorry

end NUMINAMATH_CALUDE_viola_count_l2455_245550


namespace NUMINAMATH_CALUDE_tens_digit_equals_number_of_tens_l2455_245521

theorem tens_digit_equals_number_of_tens (n : ℕ) (h : 10 ≤ n ∧ n ≤ 999) : 
  (n / 10) % 10 = n / 10 - (n / 100) * 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_equals_number_of_tens_l2455_245521


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_5_7_l2455_245545

/-- Definition of our series of pairs -/
def pair_series : ℕ → ℕ × ℕ
| n => sorry

/-- The sum of the components of the nth pair -/
def pair_sum (n : ℕ) : ℕ := (pair_series n).1 + (pair_series n).2

/-- The 60th pair in the series -/
def sixtieth_pair : ℕ × ℕ := pair_series 60

/-- Theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_5_7 : sixtieth_pair = (5, 7) := by sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_5_7_l2455_245545


namespace NUMINAMATH_CALUDE_jackies_tree_climbing_l2455_245541

theorem jackies_tree_climbing (h : ℝ) : 
  (1000 + 500 + 500 + h) / 4 = 800 → h - 1000 = 200 := by sorry

end NUMINAMATH_CALUDE_jackies_tree_climbing_l2455_245541


namespace NUMINAMATH_CALUDE_sum_of_sequence_l2455_245566

def arithmetic_sequence : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]

theorem sum_of_sequence : 
  2 * (arithmetic_sequence.sum) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequence_l2455_245566


namespace NUMINAMATH_CALUDE_product_ends_with_three_zeros_l2455_245556

theorem product_ends_with_three_zeros :
  ∃ n : ℕ, 350 * 60 = n * 1000 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_product_ends_with_three_zeros_l2455_245556


namespace NUMINAMATH_CALUDE_triangle_heights_inscribed_circle_inequality_l2455_245595

/-- Given a triangle with heights h₁ and h₂, and an inscribed circle with radius r,
    prove that 1/(2r) < 1/h₁ + 1/h₂ < 1/r. -/
theorem triangle_heights_inscribed_circle_inequality 
  (h₁ h₂ r : ℝ) 
  (h₁_pos : 0 < h₁) 
  (h₂_pos : 0 < h₂) 
  (r_pos : 0 < r) : 
  1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r := by
  sorry

end NUMINAMATH_CALUDE_triangle_heights_inscribed_circle_inequality_l2455_245595


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2455_245591

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 3/5)
  (h2 : α > Real.pi/2 ∧ α < Real.pi) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2455_245591


namespace NUMINAMATH_CALUDE_point_2_4_in_first_quadrant_l2455_245557

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem stating that the point (2,4) is in the first quadrant -/
theorem point_2_4_in_first_quadrant :
  let p : Point := ⟨2, 4⟩
  is_first_quadrant p := by
  sorry


end NUMINAMATH_CALUDE_point_2_4_in_first_quadrant_l2455_245557


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2455_245564

theorem fraction_sum_inequality (x y z : ℝ) (n : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hn : n > 0) : 
  x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z) ≤ 3 / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2455_245564


namespace NUMINAMATH_CALUDE_octal_calculation_l2455_245552

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Addition of two octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction of two octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

/-- Theorem: 24₈ + 53₈ - 17₈ = 60₈ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 24) (to_octal 53)) (to_octal 17) = to_octal 60 :=
by sorry

end NUMINAMATH_CALUDE_octal_calculation_l2455_245552


namespace NUMINAMATH_CALUDE_subset_implies_m_geq_two_l2455_245592

def set_A (m : ℝ) : Set ℝ := {x | x ≤ m}
def set_B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

theorem subset_implies_m_geq_two (m : ℝ) :
  set_B ⊆ set_A m → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_geq_two_l2455_245592


namespace NUMINAMATH_CALUDE_det_A_l2455_245561

/-- The matrix A as described in the problem -/
def A (n : ℕ) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => 1 / (min i.val j.val + 1 : ℚ)

/-- The theorem stating the determinant of matrix A -/
theorem det_A (n : ℕ) : 
  Matrix.det (A n) = (-1 : ℚ)^(n-1) / ((Nat.factorial (n-1)) * (Nat.factorial n)) := by
  sorry

end NUMINAMATH_CALUDE_det_A_l2455_245561


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2016m_45000n_l2455_245505

theorem smallest_positive_integer_2016m_45000n :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (m n : ℤ), k = 2016 * m + 45000 * n) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2016 * x + 45000 * y) → j ≥ k) ∧
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2016m_45000n_l2455_245505


namespace NUMINAMATH_CALUDE_gas_tank_cost_l2455_245575

theorem gas_tank_cost (initial_fullness : ℚ) (after_adding_fullness : ℚ) 
  (added_amount : ℚ) (gas_price : ℚ) : 
  initial_fullness = 1/8 →
  after_adding_fullness = 3/4 →
  added_amount = 30 →
  gas_price = 138/100 →
  (1 - after_adding_fullness) * 
    (added_amount / (after_adding_fullness - initial_fullness)) * 
    gas_price = 1656/100 := by
  sorry

#eval (1 : ℚ) - 3/4  -- Expected: 1/4
#eval 30 / (3/4 - 1/8)  -- Expected: 48
#eval 1/4 * 48  -- Expected: 12
#eval 12 * 138/100  -- Expected: 16.56

end NUMINAMATH_CALUDE_gas_tank_cost_l2455_245575


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_l2455_245540

theorem unique_divisor_with_remainder (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ (145 % d = 4) → d = 47 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_l2455_245540


namespace NUMINAMATH_CALUDE_F15_triangles_l2455_245510

/-- The number of triangles in figure n of the sequence -/
def T (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else T (n - 1) + 3 * n + 3

/-- The sequence of figures satisfies the given construction rules -/
axiom construction_rule (n : ℕ) : n ≥ 2 → T n = T (n - 1) + 3 * n + 3

/-- F₂ has 7 triangles -/
axiom F2_triangles : T 2 = 7

/-- The number of triangles in F₁₅ is 400 -/
theorem F15_triangles : T 15 = 400 := by sorry

end NUMINAMATH_CALUDE_F15_triangles_l2455_245510


namespace NUMINAMATH_CALUDE_algebraic_identity_l2455_245518

theorem algebraic_identity (a b : ℝ) : 3 * a^2 * b - 2 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2455_245518


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2455_245588

/-- The distance traveled by a car under specific conditions -/
theorem car_distance_theorem (actual_speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  actual_speed = 20 →
  speed_increase = 10 →
  time_decrease = 0.5 →
  ∃ (distance : ℝ),
    distance = actual_speed * (distance / actual_speed) ∧
    distance = (actual_speed + speed_increase) * (distance / actual_speed - time_decrease) ∧
    distance = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2455_245588


namespace NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l2455_245515

-- Define the coefficient of determination
def coefficient_of_determination : ℝ → ℝ := sorry

-- Define a normal distribution
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define systematic sampling
def systematic_sampling (start interval n : ℕ) : List ℕ := sorry

-- Define a proposition
structure Proposition :=
  (statement : Prop)
  (is_correct : Bool)

-- Define our three propositions
def proposition1 : Proposition :=
  ⟨ ∀ (R : ℝ), R < 0 → coefficient_of_determination R > coefficient_of_determination (-R), false ⟩

def proposition2 : Proposition :=
  let ξ := normal_distribution 2 1
  ⟨ ξ 4 = 0.79 → ξ (-2) = 0.21, false ⟩

def proposition3 : Proposition :=
  ⟨ systematic_sampling 5 11 5 = [5, 16, 27, 38, 49] → 60 ∈ systematic_sampling 5 11 5, false ⟩

-- Theorem to prove
theorem number_of_correct_propositions_is_zero :
  (proposition1.is_correct = false) ∧
  (proposition2.is_correct = false) ∧
  (proposition3.is_correct = false) := by
  sorry

end NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l2455_245515


namespace NUMINAMATH_CALUDE_alligator_count_theorem_l2455_245585

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : ℕ :=
  samara_count + friend_count * friend_average

/-- Theorem stating the total number of alligators seen -/
theorem alligator_count_theorem :
  total_alligators 20 3 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_alligator_count_theorem_l2455_245585


namespace NUMINAMATH_CALUDE_fairy_tale_book_weighs_1_1_kg_l2455_245508

/-- The weight of the fairy tale book in kilograms -/
def fairy_tale_book_weight : ℝ := sorry

/-- The total weight on the other side of the scale in kilograms -/
def other_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- The scale is level, so the weights on both sides are equal -/
axiom scale_balance : fairy_tale_book_weight = other_side_weight

/-- Theorem: The fairy tale book weighs 1.1 kg -/
theorem fairy_tale_book_weighs_1_1_kg : fairy_tale_book_weight = 1.1 := by sorry

end NUMINAMATH_CALUDE_fairy_tale_book_weighs_1_1_kg_l2455_245508


namespace NUMINAMATH_CALUDE_hens_and_cows_l2455_245580

theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 48 →
  total_feet = 140 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 26 := by
sorry

end NUMINAMATH_CALUDE_hens_and_cows_l2455_245580


namespace NUMINAMATH_CALUDE_club_women_count_l2455_245528

/-- Proves the number of women in a club given certain conditions -/
theorem club_women_count (total : ℕ) (attendees : ℕ) (men : ℕ) (women : ℕ) :
  total = 30 →
  attendees = 18 →
  men + women = total →
  men + (women / 3) = attendees →
  women = 18 := by
  sorry

end NUMINAMATH_CALUDE_club_women_count_l2455_245528


namespace NUMINAMATH_CALUDE_garage_sale_books_sold_l2455_245526

def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

theorem garage_sale_books_sold :
  let initial_books : ℕ := 108
  let remaining_books : ℕ := 66
  books_sold initial_books remaining_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_books_sold_l2455_245526


namespace NUMINAMATH_CALUDE_angle_relation_l2455_245513

theorem angle_relation (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  Real.tan α + Real.tan β = 1 / Real.cos α →
  2 * β + α = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l2455_245513


namespace NUMINAMATH_CALUDE_higher_selling_price_l2455_245560

/-- Given an article with a cost price, a lower selling price, and a higher selling price that yields 5% more gain, calculate the higher selling price. -/
theorem higher_selling_price (cost_price lower_price : ℕ) (higher_price : ℕ) : 
  cost_price = 200 →
  lower_price = 340 →
  (higher_price - cost_price) = (lower_price - cost_price) * 105 / 100 →
  higher_price = 347 :=
by sorry

end NUMINAMATH_CALUDE_higher_selling_price_l2455_245560


namespace NUMINAMATH_CALUDE_ring_toss_game_l2455_245548

/-- The ring toss game problem -/
theorem ring_toss_game (total_amount : ℕ) (daily_revenue : ℕ) (second_period : ℕ) : 
  total_amount = 186 → 
  daily_revenue = 6 →
  second_period = 16 →
  ∃ (first_period : ℕ), first_period * daily_revenue + second_period * (total_amount - first_period * daily_revenue) / second_period = total_amount ∧ 
                         first_period = 20 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_game_l2455_245548


namespace NUMINAMATH_CALUDE_beef_jerky_ratio_l2455_245527

/-- Proves that the ratio of beef jerky pieces Janette gives to her brother
    to the pieces she keeps for herself is 1:1 --/
theorem beef_jerky_ratio (days : ℕ) (initial_pieces : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ)
  (pieces_left : ℕ) :
  days = 5 →
  initial_pieces = 40 →
  breakfast = 1 →
  lunch = 1 →
  dinner = 2 →
  pieces_left = 10 →
  let daily_consumption := breakfast + lunch + dinner
  let total_consumption := daily_consumption * days
  let remaining_after_trip := initial_pieces - total_consumption
  let given_to_brother := remaining_after_trip - pieces_left
  (given_to_brother : ℚ) / pieces_left = 1 := by
sorry

end NUMINAMATH_CALUDE_beef_jerky_ratio_l2455_245527


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_sqrt3_l2455_245516

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral formed by intersecting a plane with a rectangular prism -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the area of the quadrilateral ABCD -/
def quadrilateralArea (quad : Quadrilateral) : ℝ := sorry

/-- Theorem: The area of quadrilateral ABCD is √3 -/
theorem quadrilateral_area_is_sqrt3 (prism : RectangularPrism) (quad : Quadrilateral) :
  prism.length = 2 ∧ prism.width = 1 ∧ prism.height = 1 →
  (quad.A.x = 0 ∧ quad.A.y = 0 ∧ quad.A.z = 0) →
  (quad.C.x = 2 ∧ quad.C.y = 1 ∧ quad.C.z = 1) →
  (quad.B.x = 1 ∧ quad.B.y = 0.5 ∧ quad.B.z = 1) →
  (quad.D.x = 1 ∧ quad.D.y = 1 ∧ quad.D.z = 0.5) →
  quadrilateralArea quad = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_sqrt3_l2455_245516


namespace NUMINAMATH_CALUDE_family_size_l2455_245587

/-- Given a family where one side has 10 members and the other side is 30% larger,
    the total number of family members is 23. -/
theorem family_size (fathers_side : ℕ) (mothers_side : ℕ) : 
  fathers_side = 10 →
  mothers_side = fathers_side + (fathers_side * 3 / 10) →
  fathers_side + mothers_side = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_family_size_l2455_245587


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l2455_245514

/-- Given a car's travel data, prove its speed in the first hour -/
theorem car_speed_first_hour 
  (speed_second_hour : ℝ) 
  (average_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_second_hour = 40)
  (h2 : average_speed = 60)
  (h3 : total_time = 2) :
  let total_distance := average_speed * total_time
  let speed_first_hour := 2 * total_distance / total_time - speed_second_hour
  speed_first_hour = 80 := by
sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l2455_245514


namespace NUMINAMATH_CALUDE_basketball_card_cost_l2455_245573

/-- The cost of one deck of basketball cards -/
def cost_of_deck (mary_total rose_total shoe_cost : ℕ) : ℕ :=
  (rose_total - shoe_cost) / 2

theorem basketball_card_cost :
  ∀ (mary_total rose_total shoe_cost : ℕ),
    mary_total = rose_total →
    mary_total = 200 →
    shoe_cost = 150 →
    cost_of_deck mary_total rose_total shoe_cost = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_card_cost_l2455_245573


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2455_245599

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 2 * Real.sin A + 3 * Real.cos B = 4) 
  (h2 : 3 * Real.sin B + 2 * Real.cos A = Real.sqrt 3) 
  (h3 : A + B + C = Real.pi) : C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2455_245599


namespace NUMINAMATH_CALUDE_sqrt_equation_roots_l2455_245583

theorem sqrt_equation_roots :
  ∃! (x : ℝ), x > 15 ∧ Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 ∧
  ∃ (y : ℝ), -15 < y ∧ y < -10 ∧ 
    (Real.sqrt (y + 15) - 7 / Real.sqrt (y + 15) = 6 → False) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_roots_l2455_245583


namespace NUMINAMATH_CALUDE_expression_value_l2455_245512

theorem expression_value (x y z : ℤ) (hx : x = -5) (hy : y = 8) (hz : z = 3) :
  2 * (x - y)^2 - x^3 * y + z^4 * y^2 - x^2 * z^3 = 5847 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2455_245512


namespace NUMINAMATH_CALUDE_exists_abs_le_neg_l2455_245537

theorem exists_abs_le_neg : ∃ a : ℝ, |a| ≤ -a := by sorry

end NUMINAMATH_CALUDE_exists_abs_le_neg_l2455_245537


namespace NUMINAMATH_CALUDE_q_investment_l2455_245565

/-- Represents the investment of two people in a shop --/
structure Investment where
  p : ℕ  -- Amount invested by P
  q : ℕ  -- Amount invested by Q
  ratio_p : ℕ  -- Profit ratio for P
  ratio_q : ℕ  -- Profit ratio for Q

/-- Theorem: Given the conditions, Q's investment is 60000 --/
theorem q_investment (i : Investment) 
  (h1 : i.p = 40000)  -- P invested 40000
  (h2 : i.ratio_p = 2)  -- P's profit ratio is 2
  (h3 : i.ratio_q = 3)  -- Q's profit ratio is 3
  : i.q = 60000 := by
  sorry

#check q_investment

end NUMINAMATH_CALUDE_q_investment_l2455_245565


namespace NUMINAMATH_CALUDE_alice_cannot_arrive_before_bob_l2455_245544

/-- Proves that Alice cannot arrive before Bob given the conditions --/
theorem alice_cannot_arrive_before_bob :
  let distance : ℝ := 120  -- Distance between cities in miles
  let bob_speed : ℝ := 40  -- Bob's speed in miles per hour
  let alice_speed : ℝ := 48  -- Alice's speed in miles per hour
  let bob_head_start : ℝ := 0.5  -- Bob's head start in hours

  let bob_initial_distance : ℝ := bob_speed * bob_head_start
  let bob_remaining_distance : ℝ := distance - bob_initial_distance
  let bob_remaining_time : ℝ := bob_remaining_distance / bob_speed
  let alice_total_time : ℝ := distance / alice_speed

  alice_total_time ≥ bob_remaining_time :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_alice_cannot_arrive_before_bob_l2455_245544


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2455_245584

theorem complex_arithmetic_equality : 
  (1000 + 15 + 314) * (201 + 360 + 110) + (1000 - 201 - 360 - 110) * (15 + 314) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2455_245584


namespace NUMINAMATH_CALUDE_f_of_g_eight_l2455_245569

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_eight : f (g 8) = 211 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_eight_l2455_245569


namespace NUMINAMATH_CALUDE_sum_increase_by_three_percent_l2455_245576

theorem sum_increase_by_three_percent : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  (1.01 * x + 1.04 * y) = 1.03 * (x + y) := by
sorry

end NUMINAMATH_CALUDE_sum_increase_by_three_percent_l2455_245576


namespace NUMINAMATH_CALUDE_bird_count_proof_l2455_245535

/-- The number of storks on the fence -/
def num_storks : ℕ := 6

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 3

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 2

theorem bird_count_proof :
  initial_birds = 2 ∧
  num_storks = (initial_birds + additional_birds) + 1 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_proof_l2455_245535
