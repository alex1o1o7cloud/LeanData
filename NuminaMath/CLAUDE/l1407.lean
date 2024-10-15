import Mathlib

namespace NUMINAMATH_CALUDE_y_increase_proof_l1407_140711

/-- Represents a line in the Cartesian plane -/
structure Line where
  slope : ℝ

/-- Calculates the change in y given a change in x for a line -/
def Line.deltaY (l : Line) (deltaX : ℝ) : ℝ :=
  l.slope * deltaX

theorem y_increase_proof (l : Line) (h : l.deltaY 4 = 6) :
  l.deltaY 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_increase_proof_l1407_140711


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_a_and_b_l1407_140765

theorem arithmetic_mean_of_a_and_b (a b : ℝ) : 
  a = Real.sqrt 3 + Real.sqrt 2 → 
  b = Real.sqrt 3 - Real.sqrt 2 → 
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_a_and_b_l1407_140765


namespace NUMINAMATH_CALUDE_bruce_shopping_result_l1407_140705

def bruce_shopping (initial_amount : ℚ) (shirt_price : ℚ) (shirt_count : ℕ) 
  (pants_price : ℚ) (sock_price : ℚ) (sock_count : ℕ) (belt_price : ℚ) 
  (belt_discount : ℚ) (total_discount : ℚ) : ℚ :=
  let shirt_total := shirt_price * shirt_count
  let sock_total := sock_price * sock_count
  let discounted_belt_price := belt_price * (1 - belt_discount)
  let subtotal := shirt_total + pants_price + sock_total + discounted_belt_price
  let final_total := subtotal * (1 - total_discount)
  initial_amount - final_total

theorem bruce_shopping_result : 
  bruce_shopping 71 5 5 26 3 2 12 0.25 0.1 = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_shopping_result_l1407_140705


namespace NUMINAMATH_CALUDE_projectile_collision_time_l1407_140741

-- Define the parameters
def initial_distance : ℝ := 1386 -- km
def speed1 : ℝ := 445 -- km/h
def speed2 : ℝ := 545 -- km/h

-- Define the theorem
theorem projectile_collision_time :
  let relative_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / relative_speed
  let time_minutes : ℝ := time_hours * 60
  ∃ ε > 0, |time_minutes - 84| < ε :=
sorry

end NUMINAMATH_CALUDE_projectile_collision_time_l1407_140741


namespace NUMINAMATH_CALUDE_rational_square_difference_l1407_140726

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end NUMINAMATH_CALUDE_rational_square_difference_l1407_140726


namespace NUMINAMATH_CALUDE_relationship_holds_l1407_140715

/-- A function representing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 + 3*x + 1

/-- The set of x values given in the problem -/
def X : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of y values given in the problem -/
def Y : Finset ℕ := {5, 11, 19, 29, 41}

/-- Theorem stating that the function f correctly relates all given x and y values -/
theorem relationship_holds : ∀ x ∈ X, f x ∈ Y :=
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l1407_140715


namespace NUMINAMATH_CALUDE_gcd_117_182_l1407_140724

theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

end NUMINAMATH_CALUDE_gcd_117_182_l1407_140724


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_l1407_140784

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem largest_three_digit_divisible_by_6 :
  ∀ n : ℕ, is_three_digit n → divisible_by n 6 → n ≤ 996 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_l1407_140784


namespace NUMINAMATH_CALUDE_discount_effect_l1407_140728

theorem discount_effect (P N : ℝ) (h_pos_P : P > 0) (h_pos_N : N > 0) :
  let D : ℝ := 10
  let new_price : ℝ := (1 - D / 100) * P
  let new_quantity : ℝ := 1.25 * N
  let old_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity / N = 1.25) ∧ (new_income / old_income = 1.125) :=
sorry

end NUMINAMATH_CALUDE_discount_effect_l1407_140728


namespace NUMINAMATH_CALUDE_system_solution_l1407_140762

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 5) : x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1407_140762


namespace NUMINAMATH_CALUDE_gross_profit_calculation_l1407_140709

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 44 ∧ gross_profit_percentage = 1.2 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    sales_price = cost + gross_profit ∧
    gross_profit = gross_profit_percentage * cost ∧
    gross_profit = 24 := by
  sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l1407_140709


namespace NUMINAMATH_CALUDE_power_equality_l1407_140732

theorem power_equality (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1407_140732


namespace NUMINAMATH_CALUDE_bottle_caps_eaten_l1407_140713

theorem bottle_caps_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 65 → final = 61 → initial - final = eaten → eaten = 4 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_eaten_l1407_140713


namespace NUMINAMATH_CALUDE_pentagon_area_sqrt_sum_m_n_l1407_140759

/-- A pentagon constructed from 11 line segments of length 2 --/
structure Pentagon where
  /-- The number of line segments --/
  num_segments : ℕ
  /-- The length of each segment --/
  segment_length : ℝ
  /-- Assertion that the pentagon is constructed from 11 segments of length 2 --/
  h_segments : num_segments = 11 ∧ segment_length = 2

/-- The area of the pentagon --/
noncomputable def area (p : Pentagon) : ℝ := sorry

/-- Theorem stating that the area of the pentagon can be expressed as √11 + √12 --/
theorem pentagon_area_sqrt (p : Pentagon) : 
  area p = Real.sqrt 11 + Real.sqrt 12 := by sorry

/-- Corollary showing that m + n = 23 --/
theorem sum_m_n (p : Pentagon) : 
  ∃ (m n : ℕ), (m > 0 ∧ n > 0) ∧ area p = Real.sqrt m + Real.sqrt n ∧ m + n = 23 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sqrt_sum_m_n_l1407_140759


namespace NUMINAMATH_CALUDE_amanda_pay_calculation_l1407_140777

/-- Calculates the amount Amanda receives if she doesn't finish her sales report --/
theorem amanda_pay_calculation (hourly_rate : ℝ) (hours_worked : ℝ) (withholding_percentage : ℝ) : 
  hourly_rate = 50 →
  hours_worked = 10 →
  withholding_percentage = 0.2 →
  hourly_rate * hours_worked * (1 - withholding_percentage) = 400 := by
sorry

end NUMINAMATH_CALUDE_amanda_pay_calculation_l1407_140777


namespace NUMINAMATH_CALUDE_at_most_two_distinct_values_l1407_140783

theorem at_most_two_distinct_values (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (sum_squares_eq : a^2 + b^2 = c^2 + d^2) : 
  ∃ (x y : ℝ), (a = x ∨ a = y) ∧ (b = x ∨ b = y) ∧ (c = x ∨ c = y) ∧ (d = x ∨ d = y) :=
by sorry

end NUMINAMATH_CALUDE_at_most_two_distinct_values_l1407_140783


namespace NUMINAMATH_CALUDE_kittens_problem_l1407_140701

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_sara - remaining

theorem kittens_problem (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) 
  (h1 : initial = 18) 
  (h2 : to_sara = 6) 
  (h3 : remaining = 9) : 
  kittens_to_jessica initial to_sara remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_kittens_problem_l1407_140701


namespace NUMINAMATH_CALUDE_bullet_evaluation_l1407_140729

-- Define the bullet operation
def bullet (a b : ℤ) : ℤ := 10 * a - b

-- State the theorem
theorem bullet_evaluation :
  bullet (bullet (bullet 2 0) 1) 3 = 1987 := by
  sorry

end NUMINAMATH_CALUDE_bullet_evaluation_l1407_140729


namespace NUMINAMATH_CALUDE_toys_per_day_l1407_140714

/-- Given a factory that produces toys, this theorem proves the number of toys produced each day. -/
theorem toys_per_day 
  (total_toys : ℕ)           -- Total number of toys produced per week
  (work_days : ℕ)            -- Number of work days per week
  (h1 : total_toys = 4560)   -- The factory produces 4560 toys per week
  (h2 : work_days = 4)       -- Workers work 4 days a week
  (h3 : total_toys % work_days = 0)  -- The number of toys produced is the same each day
  : total_toys / work_days = 1140 := by
  sorry

#check toys_per_day

end NUMINAMATH_CALUDE_toys_per_day_l1407_140714


namespace NUMINAMATH_CALUDE_cubic_factorization_l1407_140708

theorem cubic_factorization (x y : ℝ) : x^3 - x*y^2 = x*(x-y)*(x+y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1407_140708


namespace NUMINAMATH_CALUDE_jermaine_earnings_difference_l1407_140716

def total_earnings : ℕ := 90
def terrence_earnings : ℕ := 30
def emilee_earnings : ℕ := 25

theorem jermaine_earnings_difference : 
  ∃ (jermaine_earnings : ℕ), 
    jermaine_earnings > terrence_earnings ∧
    jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings ∧
    jermaine_earnings - terrence_earnings = 5 :=
by sorry

end NUMINAMATH_CALUDE_jermaine_earnings_difference_l1407_140716


namespace NUMINAMATH_CALUDE_bob_painting_fraction_l1407_140720

-- Define the time it takes Bob to paint a whole house
def full_painting_time : ℕ := 60

-- Define the time we want to calculate the fraction for
def partial_painting_time : ℕ := 15

-- Theorem statement
theorem bob_painting_fraction :
  (partial_painting_time : ℚ) / full_painting_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_painting_fraction_l1407_140720


namespace NUMINAMATH_CALUDE_symmetry_probability_l1407_140781

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a square with a grid of points -/
structure GridSquare where
  size : Nat
  points : List GridPoint

/-- Checks if a line through two points is a symmetry line for the square -/
def isSymmetryLine (square : GridSquare) (p q : GridPoint) : Bool :=
  sorry

/-- Counts the number of points that form symmetry lines with a given point -/
def countSymmetryPoints (square : GridSquare) (p : GridPoint) : Nat :=
  sorry

theorem symmetry_probability (square : GridSquare) (p : GridPoint) :
  square.size = 7 ∧
  square.points.length = 49 ∧
  p = ⟨3, 4⟩ →
  (countSymmetryPoints square p : Rat) / (square.points.length - 1 : Rat) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_probability_l1407_140781


namespace NUMINAMATH_CALUDE_four_common_tangents_min_area_PAOB_l1407_140776

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the moving circle C
def circle_C (x y k : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line
def line (x y : ℝ) : Prop := x + y = 4

-- Theorem 1: Four common tangents condition
theorem four_common_tangents (k : ℝ) :
  (∀ x y, circle_O x y → ∀ x' y', circle_C x' y' k → 
    ∃! t1 t2 t3 t4 : ℝ × ℝ, 
      (circle_O t1.1 t1.2 ∧ circle_C t1.1 t1.2 k) ∧
      (circle_O t2.1 t2.2 ∧ circle_C t2.1 t2.2 k) ∧
      (circle_O t3.1 t3.2 ∧ circle_C t3.1 t3.2 k) ∧
      (circle_O t4.1 t4.2 ∧ circle_C t4.1 t4.2 k)) ↔
  abs k > 2 := by sorry

-- Theorem 2: Minimum area of quadrilateral PAOB
theorem min_area_PAOB :
  ∃ min_area : ℝ, 
    min_area = 4 ∧
    ∀ P A B O : ℝ × ℝ,
      line P.1 P.2 →
      circle_O A.1 A.2 →
      circle_O B.1 B.2 →
      O = (0, 0) →
      (∀ x y, (x - P.1) * (A.1 - P.1) + (y - P.2) * (A.2 - P.2) = 0 → ¬ circle_O x y) →
      (∀ x y, (x - P.1) * (B.1 - P.1) + (y - P.2) * (B.2 - P.2) = 0 → ¬ circle_O x y) →
      let area := abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))
      area ≥ min_area := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_min_area_PAOB_l1407_140776


namespace NUMINAMATH_CALUDE_craig_and_mother_age_difference_l1407_140773

/-- Craig and his mother's ages problem -/
theorem craig_and_mother_age_difference :
  ∀ (craig_age mother_age : ℕ),
    craig_age + mother_age = 56 →
    craig_age = 16 →
    mother_age - craig_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_craig_and_mother_age_difference_l1407_140773


namespace NUMINAMATH_CALUDE_soda_cost_l1407_140787

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l1407_140787


namespace NUMINAMATH_CALUDE_bus_trip_distance_l1407_140761

/-- The distance of a bus trip in miles. -/
def trip_distance : ℝ := 280

/-- The actual average speed of the bus in miles per hour. -/
def actual_speed : ℝ := 35

/-- The increased speed of the bus in miles per hour. -/
def increased_speed : ℝ := 40

/-- Theorem stating that the trip distance is 280 miles given the conditions. -/
theorem bus_trip_distance :
  (trip_distance / actual_speed - trip_distance / increased_speed = 1) →
  trip_distance = 280 := by
  sorry


end NUMINAMATH_CALUDE_bus_trip_distance_l1407_140761


namespace NUMINAMATH_CALUDE_greatest_gcd_with_linear_combination_l1407_140752

theorem greatest_gcd_with_linear_combination (m n : ℕ) : 
  Nat.gcd m n = 1 → 
  (∃ (a b : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) = a ∧ 
                a ≤ b ∧ 
                ∀ (c : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) ≤ c → c ≤ b) ∧
  3999999 = Nat.gcd (m + 2000 * n) (n + 2000 * m) := by
  sorry

end NUMINAMATH_CALUDE_greatest_gcd_with_linear_combination_l1407_140752


namespace NUMINAMATH_CALUDE_nell_ace_cards_l1407_140796

/-- The number of baseball cards Nell has now -/
def baseball_cards : ℕ := 178

/-- The difference between baseball cards and Ace cards Nell has now -/
def difference : ℕ := 123

/-- Theorem: The number of Ace cards Nell has now is 55 -/
theorem nell_ace_cards : 
  ∃ (ace_cards : ℕ), ace_cards = baseball_cards - difference ∧ ace_cards = 55 := by
  sorry

end NUMINAMATH_CALUDE_nell_ace_cards_l1407_140796


namespace NUMINAMATH_CALUDE_expression_evaluation_l1407_140735

theorem expression_evaluation (a x : ℝ) (h1 : a = x^2) (h2 : a = Real.sqrt 2) :
  4 * a^3 / (x^4 + a^4) + 1 / (a + x) + 2 * a / (x^2 + a^2) + 1 / (a - x) = 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1407_140735


namespace NUMINAMATH_CALUDE_product_digit_sum_l1407_140747

/-- Represents a 101-digit number that repeats a 3-digit pattern -/
def RepeatingNumber (a b c : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Returns the units digit of a number -/
def unitsDigit (n : Nat) : Nat :=
  n % 10

/-- Returns the thousands digit of a number -/
def thousandsDigit (n : Nat) : Nat :=
  (n / 1000) % 10

/-- The main theorem -/
theorem product_digit_sum :
  let n1 := RepeatingNumber 6 0 6
  let n2 := RepeatingNumber 7 0 7
  let product := n1 * n2
  (thousandsDigit product) + (unitsDigit product) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1407_140747


namespace NUMINAMATH_CALUDE_restaurant_expenditure_l1407_140782

theorem restaurant_expenditure (num_people : ℕ) (regular_cost : ℚ) (num_regular : ℕ) (extra_cost : ℚ) :
  num_people = 7 →
  regular_cost = 11 →
  num_regular = 6 →
  extra_cost = 6 →
  let total_regular := num_regular * regular_cost
  let average := (total_regular + (total_regular + extra_cost) / num_people) / num_people
  let total_cost := total_regular + (average + extra_cost)
  total_cost = 84 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_expenditure_l1407_140782


namespace NUMINAMATH_CALUDE_square_root_sum_l1407_140788

theorem square_root_sum (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l1407_140788


namespace NUMINAMATH_CALUDE_coin_combination_difference_l1407_140725

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | five : Coin
  | ten : Coin
  | twenty : Coin

/-- The value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.five => 5
  | Coin.ten => 10
  | Coin.twenty => 20

/-- A combination of coins -/
def CoinCombination := List Coin

/-- The total value of a coin combination in cents -/
def combinationValue (combo : CoinCombination) : Nat :=
  combo.map coinValue |>.sum

/-- Predicate for valid coin combinations that sum to 30 cents -/
def isValidCombination (combo : CoinCombination) : Prop :=
  combinationValue combo = 30

/-- The number of coins in a combination -/
def coinCount (combo : CoinCombination) : Nat :=
  combo.length

theorem coin_combination_difference :
  ∃ (minCombo maxCombo : CoinCombination),
    isValidCombination minCombo ∧
    isValidCombination maxCombo ∧
    (∀ c : CoinCombination, isValidCombination c → 
      coinCount c ≥ coinCount minCombo ∧
      coinCount c ≤ coinCount maxCombo) ∧
    coinCount maxCombo - coinCount minCombo = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l1407_140725


namespace NUMINAMATH_CALUDE_product_positive_l1407_140768

theorem product_positive (x y z t : ℝ) 
  (h1 : x > y^3) 
  (h2 : y > z^3) 
  (h3 : z > t^3) 
  (h4 : t > x^3) : 
  x * y * z * t > 0 := by
sorry

end NUMINAMATH_CALUDE_product_positive_l1407_140768


namespace NUMINAMATH_CALUDE_line_through_point_equation_line_with_slope_equation_l1407_140737

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to calculate the area of a triangle formed by a line and coordinate axes
def triangleArea (l : Line) : ℝ :=
  sorry

-- Function to check if a line passes through a point
def linePassesPoint (l : Line) (p : Point) : Prop :=
  sorry

-- Theorem for condition 1
theorem line_through_point_equation (l : Line) (A : Point) :
  triangleArea l = 3 ∧ linePassesPoint l A ∧ A.x = -3 ∧ A.y = 4 →
  (∃ a b c, a * l.slope + b = 0 ∧ a = 2 ∧ b = 3 ∧ c = -6) ∨
  (∃ a b c, a * l.slope + b = 0 ∧ a = 8 ∧ b = 3 ∧ c = 12) :=
sorry

-- Theorem for condition 2
theorem line_with_slope_equation (l : Line) :
  triangleArea l = 3 ∧ l.slope = 1/6 →
  (∃ b, l.intercept = b ∧ (b = 1 ∨ b = -1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_equation_line_with_slope_equation_l1407_140737


namespace NUMINAMATH_CALUDE_arrangement_count_l1407_140770

/-- The number of ways to arrange 4 boys and 3 girls in a row -/
def total_arrangements : ℕ := Nat.factorial 7

/-- The number of ways to arrange 4 boys and 3 girls where all 3 girls are adjacent -/
def three_girls_adjacent : ℕ := Nat.factorial 5 * Nat.factorial 3

/-- The number of ways to arrange 4 boys and 3 girls where exactly 2 girls are adjacent -/
def two_girls_adjacent : ℕ := Nat.factorial 6 * Nat.factorial 2 * 3

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := two_girls_adjacent - three_girls_adjacent

theorem arrangement_count : valid_arrangements = 3600 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1407_140770


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1407_140755

-- Define the swimming speed in still water
def still_water_speed : ℝ := 6

-- Define the function for downstream speed
def downstream_speed (stream_speed : ℝ) : ℝ := still_water_speed + stream_speed

-- Define the function for upstream speed
def upstream_speed (stream_speed : ℝ) : ℝ := still_water_speed - stream_speed

-- Theorem statement
theorem stream_speed_calculation :
  ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    downstream_speed stream_speed / upstream_speed stream_speed = 2 ∧
    stream_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l1407_140755


namespace NUMINAMATH_CALUDE_solve_books_problem_l1407_140758

def books_problem (initial_books : ℕ) (new_books : ℕ) : Prop :=
  let after_nephew := initial_books - (initial_books / 4)
  let after_library := after_nephew - (after_nephew / 5)
  let after_neighbor := after_library - (after_library / 6)
  let final_books := after_neighbor + new_books
  final_books = 68

theorem solve_books_problem :
  books_problem 120 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_books_problem_l1407_140758


namespace NUMINAMATH_CALUDE_initial_machines_count_l1407_140753

/-- The number of shirts produced by a group of machines -/
def shirts_produced (num_machines : ℕ) (time : ℕ) : ℕ := sorry

/-- The production rate of a single machine in shirts per minute -/
def machine_rate : ℚ := sorry

/-- The total production rate of all machines in shirts per minute -/
def total_rate : ℕ := 32

theorem initial_machines_count :
  ∃ (n : ℕ), 
    shirts_produced 8 10 = 160 ∧
    (n : ℚ) * machine_rate = total_rate ∧
    n = 16 :=
sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1407_140753


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l1407_140767

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x ↦ 9*x^2 - 45*x + 50
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l1407_140767


namespace NUMINAMATH_CALUDE_age_half_in_ten_years_l1407_140798

def mother_age : ℕ := 50

def person_age : ℕ := (2 * mother_age) / 5

def years_until_half (y : ℕ) : Prop :=
  2 * (person_age + y) = mother_age + y

theorem age_half_in_ten_years :
  ∃ y : ℕ, years_until_half y ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_age_half_in_ten_years_l1407_140798


namespace NUMINAMATH_CALUDE_tens_digit_of_7_power_2011_l1407_140707

theorem tens_digit_of_7_power_2011 : 7^2011 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_power_2011_l1407_140707


namespace NUMINAMATH_CALUDE_function_properties_l1407_140740

def is_additive (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties (f : ℝ → ℝ) 
  (h_additive : is_additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (f (-3) = 6 ∧ f 3 = -6) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1407_140740


namespace NUMINAMATH_CALUDE_david_subtraction_l1407_140793

theorem david_subtraction (n : ℕ) (h : n = 40) : n^2 - 79 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_david_subtraction_l1407_140793


namespace NUMINAMATH_CALUDE_cube_root_cube_equality_l1407_140779

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_cube_equality_l1407_140779


namespace NUMINAMATH_CALUDE_factors_of_1320_l1407_140744

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 has exactly 24 distinct, positive factors -/
theorem factors_of_1320 : num_factors_1320 = 24 := by sorry

end NUMINAMATH_CALUDE_factors_of_1320_l1407_140744


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1407_140745

def M : Set ℝ := {x | 0 < x ∧ x < 8}
def N : Set ℝ := {x | ∃ n : ℕ, x = 2 * n + 1}

theorem set_intersection_problem : M ∩ N = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1407_140745


namespace NUMINAMATH_CALUDE_children_who_got_off_bus_l1407_140763

/-- Proves that 10 children got off the bus given the initial, final, and additional children counts -/
theorem children_who_got_off_bus 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (final_children : ℕ) 
  (h1 : initial_children = 21)
  (h2 : children_who_got_on = 5)
  (h3 : final_children = 16) :
  initial_children - final_children + children_who_got_on = 10 :=
by sorry

end NUMINAMATH_CALUDE_children_who_got_off_bus_l1407_140763


namespace NUMINAMATH_CALUDE_number_categorization_l1407_140700

def numbers : List ℚ := [7, -3.14, -5, 1/8, 0, -7/4, -4/5]

def is_positive_rational (x : ℚ) : Prop := x > 0

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ↑(⌊x⌋)

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = ↑n

theorem number_categorization :
  (∀ x ∈ numbers, is_positive_rational x ↔ x ∈ [7, 1/8]) ∧
  (∀ x ∈ numbers, is_negative_fraction x ↔ x ∈ [-3.14, -7/4, -4/5]) ∧
  (∀ x ∈ numbers, is_integer x ↔ x ∈ [7, -5, 0]) :=
sorry

end NUMINAMATH_CALUDE_number_categorization_l1407_140700


namespace NUMINAMATH_CALUDE_concert_theorem_l1407_140751

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  elsa : ℕ

/-- The conditions of the problem -/
def concert_conditions (s : SongCounts) : Prop :=
  s.hanna = 9 ∧ 
  s.mary = 3 ∧ 
  s.alina + s.tina = 16 ∧
  s.hanna > s.alina ∧ s.hanna > s.tina ∧ s.hanna > s.elsa ∧
  s.alina > s.mary ∧ s.tina > s.mary ∧ s.elsa > s.mary

/-- The total number of songs sung -/
def total_songs (s : SongCounts) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.elsa) / 4

/-- The main theorem: given the conditions, the total number of songs is 8 -/
theorem concert_theorem (s : SongCounts) : 
  concert_conditions s → total_songs s = 8 := by
  sorry

end NUMINAMATH_CALUDE_concert_theorem_l1407_140751


namespace NUMINAMATH_CALUDE_workers_in_first_group_l1407_140717

/-- The number of workers in the first group -/
def W : ℕ := 360

/-- The time taken by the first group to build the wall -/
def T1 : ℕ := 48

/-- The number of workers in the second group -/
def T2 : ℕ := 24

/-- The time taken by the second group to build the wall -/
def W2 : ℕ := 30

/-- Theorem stating that W is the correct number of workers in the first group -/
theorem workers_in_first_group :
  W * T1 = T2 * W2 := by sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l1407_140717


namespace NUMINAMATH_CALUDE_ellipse_equation_l1407_140754

/-- Represents an ellipse with focus on the x-axis -/
structure Ellipse where
  /-- Distance from the right focus to the short axis endpoint -/
  short_axis_dist : ℝ
  /-- Distance from the right focus to the left vertex -/
  left_vertex_dist : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_equation (e : Ellipse) (h1 : e.short_axis_dist = 2) (h2 : e.left_vertex_dist = 3) :
  ∀ x y : ℝ, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1407_140754


namespace NUMINAMATH_CALUDE_x_range_theorem_l1407_140702

theorem x_range_theorem (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ x : ℝ, (1/a) + (4/b) ≥ |2*x - 1| - |x + 1|) :
  ∀ x : ℝ, -7 ≤ x ∧ x ≤ 11 := by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1407_140702


namespace NUMINAMATH_CALUDE_max_weekly_profit_l1407_140743

/-- Represents the weekly sales profit as a function of the price increase -/
def weekly_profit (x : ℝ) : ℝ := -10 * x^2 + 100 * x + 6000

/-- Represents the number of items sold per week as a function of the price increase -/
def items_sold (x : ℝ) : ℝ := 300 - 10 * x

theorem max_weekly_profit :
  ∀ x : ℝ, x ≤ 20 → weekly_profit x ≤ 6250 ∧
  ∃ x₀ : ℝ, x₀ ≤ 20 ∧ weekly_profit x₀ = 6250 :=
sorry

end NUMINAMATH_CALUDE_max_weekly_profit_l1407_140743


namespace NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l1407_140723

/-- The volume of space inside a sphere but outside an inscribed right cylinder -/
theorem sphere_minus_cylinder_volume (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  (4/3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) = 
  (288 - 64 * Real.sqrt 5) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l1407_140723


namespace NUMINAMATH_CALUDE_f_derivative_sum_l1407_140790

def f (x : ℝ) := x^4 + x - 1

theorem f_derivative_sum : (deriv f 1) + (deriv f (-1)) = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l1407_140790


namespace NUMINAMATH_CALUDE_fruit_basket_total_cost_l1407_140721

/-- Represents the cost of a fruit basket -/
def fruit_basket_cost (banana_price : ℚ) (apple_price : ℚ) (strawberry_price : ℚ) 
  (avocado_price : ℚ) (grape_price : ℚ) : ℚ :=
  4 * banana_price + 3 * apple_price + 24 * strawberry_price / 12 + 
  2 * avocado_price + 2 * grape_price

/-- Theorem stating the total cost of the fruit basket -/
theorem fruit_basket_total_cost : 
  fruit_basket_cost 1 2 (4/12) 3 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_cost_l1407_140721


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l1407_140749

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) :
  (n > 2) →
  (exterior_angle > 0) →
  (exterior_angle < 180) →
  (n * exterior_angle = 360) →
  (exterior_angle = 60) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l1407_140749


namespace NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l1407_140719

-- Define the quadratic function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_symmetry_implies_ordering (b c : ℝ) :
  (∀ x : ℝ, f b c (1 + x) = f b c (1 - x)) →
  f b c 4 > f b c 2 ∧ f b c 2 > f b c 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l1407_140719


namespace NUMINAMATH_CALUDE_dots_per_blouse_is_twenty_l1407_140727

/-- The number of dots on each blouse -/
def dots_per_blouse (total_dye : ℕ) (num_blouses : ℕ) (dye_per_dot : ℕ) : ℕ :=
  (total_dye / num_blouses) / dye_per_dot

/-- Theorem stating that the number of dots per blouse is 20 -/
theorem dots_per_blouse_is_twenty :
  dots_per_blouse (50 * 400) 100 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dots_per_blouse_is_twenty_l1407_140727


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1407_140792

theorem max_value_on_circle : 
  ∃ (M : ℝ), M = 8 ∧ 
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 1 → |3*x + 4*y - 3| ≤ M) ∧
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ |3*x + 4*y - 3| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1407_140792


namespace NUMINAMATH_CALUDE_min_value_theorem_l1407_140778

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1407_140778


namespace NUMINAMATH_CALUDE_det_max_value_l1407_140797

open Real Matrix

theorem det_max_value (θ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1, 1 + sin φ, 1; 1, 1, 1 + cos φ]) → det A ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l1407_140797


namespace NUMINAMATH_CALUDE_joan_flour_cups_l1407_140704

theorem joan_flour_cups (total : ℕ) (remaining : ℕ) (already_added : ℕ) : 
  total = 7 → remaining = 4 → already_added = total - remaining → already_added = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_flour_cups_l1407_140704


namespace NUMINAMATH_CALUDE_collinear_points_d_values_l1407_140712

/-- Four points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Define the four points -/
def p1 (a : ℝ) : Point3D := ⟨2, 0, a⟩
def p2 (b : ℝ) : Point3D := ⟨b, 2, 0⟩
def p3 (c : ℝ) : Point3D := ⟨0, c, 2⟩
def p4 (d : ℝ) : Point3D := ⟨8*d, 8*d, -2*d⟩

/-- Define collinearity for four points -/
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), (q.x - p.x, q.y - p.y, q.z - p.z) = t₁ • (r.x - p.x, r.y - p.y, r.z - p.z)
                 ∧ (q.x - p.x, q.y - p.y, q.z - p.z) = t₂ • (s.x - p.x, s.y - p.y, s.z - p.z)
                 ∧ (r.x - p.x, r.y - p.y, r.z - p.z) = t₃ • (s.x - p.x, s.y - p.y, s.z - p.z)

/-- The main theorem -/
theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (p1 a) (p2 b) (p3 c) (p4 d) → d = 1/8 ∨ d = -1/32 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_d_values_l1407_140712


namespace NUMINAMATH_CALUDE_trajectory_of_M_l1407_140799

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the slope sum condition
def slope_sum_condition (x y : ℝ) : Prop :=
  y / (x + 2) + y / (x - 2) = 2

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x * y - x^2 + 4 = 0

-- Theorem statement
theorem trajectory_of_M (x y : ℝ) (h1 : y ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  slope_sum_condition x y → trajectory_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l1407_140799


namespace NUMINAMATH_CALUDE_power_multiplication_addition_l1407_140748

theorem power_multiplication_addition : 2^4 * 3^2 * 5^2 + 7^3 = 3943 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_addition_l1407_140748


namespace NUMINAMATH_CALUDE_card_distribution_correct_l1407_140766

/-- The total number of cards to be distributed -/
def total_cards : ℕ := 363

/-- The ratio of cards Xiaoming gets to Xiaohua's cards -/
def ratio_xiaoming_xiaohua : ℚ := 7 / 6

/-- The ratio of cards Xiaogang gets to Xiaoming's cards -/
def ratio_xiaogang_xiaoming : ℚ := 8 / 5

/-- The number of cards Xiaoming receives -/
def xiaoming_cards : ℕ := 105

/-- The number of cards Xiaohua receives -/
def xiaohua_cards : ℕ := 90

/-- The number of cards Xiaogang receives -/
def xiaogang_cards : ℕ := 168

theorem card_distribution_correct :
  (xiaoming_cards + xiaohua_cards + xiaogang_cards = total_cards) ∧
  (xiaoming_cards : ℚ) / xiaohua_cards = ratio_xiaoming_xiaohua ∧
  (xiaogang_cards : ℚ) / xiaoming_cards = ratio_xiaogang_xiaoming :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_correct_l1407_140766


namespace NUMINAMATH_CALUDE_linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l1407_140764

-- Define a linear function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define a constant derivative
def has_constant_derivative (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x, deriv f x = c

theorem linear_implies_constant_derivative :
  ∀ f : ℝ → ℝ, is_linear f → has_constant_derivative f :=
sorry

theorem constant_derivative_not_sufficient_for_linear :
  ∃ f : ℝ → ℝ, has_constant_derivative f ∧ ¬is_linear f :=
sorry

end NUMINAMATH_CALUDE_linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l1407_140764


namespace NUMINAMATH_CALUDE_star_operation_result_l1407_140734

def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_result :
  let M : Set ℕ := {1, 2, 3, 4, 5}
  let P : Set ℕ := {2, 3, 6}
  star P M = {6} := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l1407_140734


namespace NUMINAMATH_CALUDE_average_of_five_l1407_140769

/-- Given five real numbers x₁, x₂, x₃, x₄, x₅, if the average of x₁ and x₂ is 2
    and the average of x₃, x₄, and x₅ is 4, then the average of all five numbers is 3.2. -/
theorem average_of_five (x₁ x₂ x₃ x₄ x₅ : ℝ) 
    (h₁ : (x₁ + x₂) / 2 = 2)
    (h₂ : (x₃ + x₄ + x₅) / 3 = 4) :
    (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_l1407_140769


namespace NUMINAMATH_CALUDE_weighted_average_is_correct_l1407_140703

/-- Represents the number of pens sold for each type -/
def pens_sold : Fin 2 → ℕ
  | 0 => 100  -- Type A
  | 1 => 200  -- Type B

/-- Represents the number of pens gained for each type -/
def pens_gained : Fin 2 → ℕ
  | 0 => 30   -- Type A
  | 1 => 40   -- Type B

/-- Calculates the gain percentage for each pen type -/
def gain_percentage (i : Fin 2) : ℚ :=
  (pens_gained i : ℚ) / (pens_sold i : ℚ) * 100

/-- Calculates the weighted average of gain percentages -/
def weighted_average : ℚ :=
  (gain_percentage 0 * pens_sold 0 + gain_percentage 1 * pens_sold 1) / (pens_sold 0 + pens_sold 1)

theorem weighted_average_is_correct :
  weighted_average = 7000 / 300 :=
sorry

end NUMINAMATH_CALUDE_weighted_average_is_correct_l1407_140703


namespace NUMINAMATH_CALUDE_optimal_purchase_solution_max_basketballs_part2_l1407_140722

def basketball_price : ℕ := 100
def soccer_ball_price : ℕ := 80
def total_budget : ℕ := 5600
def total_items : ℕ := 60

theorem optimal_purchase_solution :
  ∃! (basketballs soccer_balls : ℕ),
    basketballs + soccer_balls = total_items ∧
    basketball_price * basketballs + soccer_ball_price * soccer_balls = total_budget ∧
    basketballs = 40 ∧
    soccer_balls = 20 :=
by sorry

theorem max_basketballs_part2 (new_budget : ℕ) (new_total_items : ℕ)
  (h1 : new_budget = 6890) (h2 : new_total_items = 80) :
  ∃ (max_basketballs : ℕ),
    max_basketballs ≤ new_total_items ∧
    basketball_price * max_basketballs + soccer_ball_price * (new_total_items - max_basketballs) ≤ new_budget ∧
    ∀ (basketballs : ℕ),
      basketballs ≤ new_total_items →
      basketball_price * basketballs + soccer_ball_price * (new_total_items - basketballs) ≤ new_budget →
      basketballs ≤ max_basketballs ∧
    max_basketballs = 24 :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_solution_max_basketballs_part2_l1407_140722


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l1407_140795

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 118) 
  (edge_sum : 4 * (a + b + c) = 52) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l1407_140795


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1407_140789

theorem negation_of_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1407_140789


namespace NUMINAMATH_CALUDE_symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l1407_140710

-- Define the curve C
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Symmetry about y=0
theorem symmetry_y_axis (x y : ℝ) : C x y ↔ C x (-y) := by sorry

-- Symmetry about x=0
theorem symmetry_x_axis (x y : ℝ) : C x y ↔ C (-x) y := by sorry

-- Symmetry about (0,0)
theorem symmetry_origin (x y : ℝ) : C x y ↔ C (-x) (-y) := by sorry

-- Define the area of C
noncomputable def area_C : ℝ := sorry

-- Area of C is greater than π
theorem area_greater_than_pi : area_C > π := by sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l1407_140710


namespace NUMINAMATH_CALUDE_function_transformation_l1407_140746

open Real

theorem function_transformation (x : ℝ) :
  let f (x : ℝ) := sin (2 * x + π / 3)
  let g (x : ℝ) := 2 * f (x - π / 6)
  g x = 2 * sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1407_140746


namespace NUMINAMATH_CALUDE_exactly_one_prop_true_l1407_140771

-- Define a type for lines
structure Line where
  -- Add necessary fields for a line

-- Define what it means for two lines to form equal angles with a third line
def form_equal_angles (l1 l2 l3 : Line) : Prop := sorry

-- Define what it means for a line to be perpendicular to another line
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the three propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, form_equal_angles l1 l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 ∧ perpendicular l2 l3 → parallel l1 l2
def prop3 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 ∧ parallel l2 l3 → parallel l1 l2

-- Theorem stating that exactly one proposition is true
theorem exactly_one_prop_true : (prop1 ∧ ¬prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_prop_true_l1407_140771


namespace NUMINAMATH_CALUDE_oranges_left_uneaten_l1407_140791

theorem oranges_left_uneaten (total : Nat) (ripe_fraction : Rat) (ripe_eaten_fraction : Rat) (unripe_eaten_fraction : Rat) :
  total = 96 →
  ripe_fraction = 1/2 →
  ripe_eaten_fraction = 1/4 →
  unripe_eaten_fraction = 1/8 →
  total - (ripe_fraction * total * ripe_eaten_fraction + (1 - ripe_fraction) * total * unripe_eaten_fraction) = 78 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_uneaten_l1407_140791


namespace NUMINAMATH_CALUDE_circle_m_range_l1407_140738

/-- A circle in the xy-plane can be represented by the equation x² + y² + dx + ey + f = 0,
    where d, e, and f are real constants, and d² + e² - 4f > 0 -/
def is_circle (d e f : ℝ) : Prop := d^2 + e^2 - 4*f > 0

/-- The equation x² + y² - 2x - 4y + m = 0 represents a circle -/
def represents_circle (m : ℝ) : Prop := is_circle (-2) (-4) m

theorem circle_m_range :
  ∀ m : ℝ, represents_circle m → m < 5 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l1407_140738


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1407_140731

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 15 * x + b * y + c * z = 0)
  (eq2 : a * x + 25 * y + c * z = 0)
  (eq3 : a * x + b * y + 45 * z = 0)
  (ha : a ≠ 15)
  (hb : b ≠ 25)
  (hx : x ≠ 0) :
  a / (a - 15) + b / (b - 25) + c / (c - 45) = 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1407_140731


namespace NUMINAMATH_CALUDE_shortest_side_is_thirteen_l1407_140742

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the first segment of the divided side -/
  segment1 : ℝ
  /-- The length of the second segment of the divided side -/
  segment2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: radius is positive -/
  radius_pos : radius > 0
  /-- Condition: segments are positive -/
  segment1_pos : segment1 > 0
  segment2_pos : segment2 > 0
  /-- Condition: shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: The shortest side of the triangle is 13 units -/
theorem shortest_side_is_thirteen (t : TriangleWithInscribedCircle) 
    (h1 : t.radius = 4)
    (h2 : t.segment1 = 6)
    (h3 : t.segment2 = 8) :
    t.shortest_side = 13 :=
  sorry


end NUMINAMATH_CALUDE_shortest_side_is_thirteen_l1407_140742


namespace NUMINAMATH_CALUDE_alberts_cabbage_patch_l1407_140733

/-- Albert's cabbage patch problem -/
theorem alberts_cabbage_patch (rows : ℕ) (heads_per_row : ℕ) 
  (h1 : rows = 12) (h2 : heads_per_row = 15) : 
  rows * heads_per_row = 180 := by
  sorry

end NUMINAMATH_CALUDE_alberts_cabbage_patch_l1407_140733


namespace NUMINAMATH_CALUDE_sandy_fish_problem_l1407_140730

/-- The number of pet fish Sandy has after buying more -/
def sandys_final_fish_count (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Sandy's final fish count is 32 given the initial conditions -/
theorem sandy_fish_problem :
  sandys_final_fish_count 26 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_problem_l1407_140730


namespace NUMINAMATH_CALUDE_min_airlines_needed_l1407_140775

/-- Represents the number of towns -/
def num_towns : ℕ := 21

/-- Represents the size of the group of towns served by each airline -/
def group_size : ℕ := 5

/-- Calculates the total number of pairs of towns -/
def total_pairs : ℕ := num_towns.choose 2

/-- Calculates the number of pairs served by each airline -/
def pairs_per_airline : ℕ := group_size.choose 2

/-- Theorem stating the minimum number of airlines needed -/
theorem min_airlines_needed : 
  ∃ (n : ℕ), n * pairs_per_airline ≥ total_pairs ∧ 
  ∀ (m : ℕ), m * pairs_per_airline ≥ total_pairs → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_airlines_needed_l1407_140775


namespace NUMINAMATH_CALUDE_max_nickels_in_jar_l1407_140736

theorem max_nickels_in_jar (total_nickels : ℕ) (jar_score : ℕ) (ground_score : ℕ) (final_score : ℕ) :
  total_nickels = 40 →
  jar_score = 5 →
  ground_score = 2 →
  final_score = 88 →
  ∃ (jar_nickels ground_nickels : ℕ),
    jar_nickels + ground_nickels = total_nickels ∧
    jar_score * jar_nickels - ground_score * ground_nickels = final_score ∧
    jar_nickels ≤ 24 ∧
    (∀ (x : ℕ), x > 24 →
      ¬(∃ (y : ℕ), x + y = total_nickels ∧
        jar_score * x - ground_score * y = final_score)) :=
by sorry

end NUMINAMATH_CALUDE_max_nickels_in_jar_l1407_140736


namespace NUMINAMATH_CALUDE_angle_decomposition_negative_495_decomposition_l1407_140786

theorem angle_decomposition (angle : ℤ) : ∃ (k : ℤ) (θ : ℤ), 
  angle = k * 360 + θ ∧ -180 < θ ∧ θ ≤ 180 :=
by sorry

theorem negative_495_decomposition : 
  ∃ (k : ℤ), -495 = k * 360 + (-135) ∧ -180 < -135 ∧ -135 ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_decomposition_negative_495_decomposition_l1407_140786


namespace NUMINAMATH_CALUDE_john_juice_bottles_l1407_140760

/-- The number of fluid ounces John needs -/
def required_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 150

/-- The number of fluid ounces in 1 liter -/
def oz_per_liter : ℝ := 34

/-- The number of milliliters in 1 liter -/
def ml_per_liter : ℝ := 1000

/-- The smallest number of bottles John should buy -/
def min_bottles : ℕ := 12

theorem john_juice_bottles : 
  ∃ (n : ℕ), n = min_bottles ∧ 
  (n : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter ≥ required_oz ∧
  ∀ (m : ℕ), m < n → (m : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter < required_oz :=
by sorry

end NUMINAMATH_CALUDE_john_juice_bottles_l1407_140760


namespace NUMINAMATH_CALUDE_min_value_C_over_D_l1407_140780

theorem min_value_C_over_D (C D y : ℝ) (hC : C > 0) (hD : D > 0) (hy : y > 0)
  (hCy : y^3 + 1/y^3 = C) (hDy : y - 1/y = D) :
  C / D ≥ 6 ∧ ∃ y > 0, y^3 + 1/y^3 = C ∧ y - 1/y = D ∧ C / D = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_C_over_D_l1407_140780


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1407_140774

/-- A sequence a : ℕ → ℝ is geometric if there exists a non-zero real number r 
    such that for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
  (h2 : a 3 * a 5 = 64) : a 4 = 8 ∨ a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1407_140774


namespace NUMINAMATH_CALUDE_flute_cost_l1407_140785

/-- The cost of a flute given the total spent and costs of other items --/
theorem flute_cost (total_spent music_stand_cost song_book_cost : ℚ) :
  total_spent = 158.35 →
  music_stand_cost = 8.89 →
  song_book_cost = 7 →
  total_spent - (music_stand_cost + song_book_cost) = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_l1407_140785


namespace NUMINAMATH_CALUDE_min_max_values_l1407_140750

theorem min_max_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → Real.sqrt x + Real.sqrt y ≥ Real.sqrt a + Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l1407_140750


namespace NUMINAMATH_CALUDE_angle_measure_l1407_140772

theorem angle_measure (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 :=     -- Measure of angle C is 150 degrees
by sorry

end NUMINAMATH_CALUDE_angle_measure_l1407_140772


namespace NUMINAMATH_CALUDE_difference_not_one_l1407_140794

theorem difference_not_one (a b : ℝ) (h : a^2 - b^2 + 2*a - 4*b - 3 ≠ 0) : a - b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_not_one_l1407_140794


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l1407_140756

/-- The sum of all three-digit numbers -/
def sum_three_digit : ℕ := 494550

/-- Predicate to check if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_sum_theorem (x y : ℕ) :
  is_three_digit x ∧ is_three_digit y ∧ 
  sum_three_digit - x - y = 600 * x →
  x = 823 ∧ y = 527 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l1407_140756


namespace NUMINAMATH_CALUDE_bus_capacity_is_198_l1407_140739

/-- Represents the capacity of a double-decker bus -/
def BusCapacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := (15 - 3) * 3
  let lower_back := 11
  let lower_standing := 12
  let upper_left := 20 * 2
  let upper_right_regular := (18 - 5) * 2
  let upper_right_reserved := 5 * 4
  let upper_standing := 8
  lower_left + lower_right + lower_back + lower_standing +
  upper_left + upper_right_regular + upper_right_reserved + upper_standing

/-- Theorem stating that the bus capacity is 198 people -/
theorem bus_capacity_is_198 : BusCapacity = 198 := by
  sorry

#eval BusCapacity

end NUMINAMATH_CALUDE_bus_capacity_is_198_l1407_140739


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l1407_140718

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 4 * x^2 - 6 * x = 2 * x * (x - 3) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l1407_140718


namespace NUMINAMATH_CALUDE_parabola_tangent_line_existence_l1407_140757

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the isosceles right triangle condition
def isosceles_right_triangle (F₁ F₂ F : ℝ × ℝ) : Prop :=
  (F₁.1 = -1 ∧ F₁.2 = 0) ∧ (F₂.1 = 1 ∧ F₂.2 = 0) ∧ (F.1 = 0 ∧ F.2 = 1)

-- Define the line passing through E(-2, 0)
def line_through_E (x y : ℝ) : Prop := y = (1/2) * (x + 2)

-- Define the perpendicular tangent lines condition
def perpendicular_tangents (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 = -4

-- Main theorem
theorem parabola_tangent_line_existence :
  ∃ (A B : ℝ × ℝ),
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line_through_E A.1 A.2 ∧
    line_through_E B.1 B.2 ∧
    perpendicular_tangents A B :=
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_existence_l1407_140757


namespace NUMINAMATH_CALUDE_extreme_values_when_a_is_two_unique_zero_range_of_a_l1407_140706

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Theorem for part 1
theorem extreme_values_when_a_is_two :
  let f := f 2
  ∃ (x_max x_min : ℝ), 
    (∀ x, f x ≤ f x_max) ∧
    (∀ x, f x ≥ f x_min) ∧
    f x_max = 1 ∧
    f x_min = 0 :=
sorry

-- Theorem for part 2
theorem unique_zero_range_of_a :
  ∀ a : ℝ, 
    (∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ 
    (a = 2 ∨ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_is_two_unique_zero_range_of_a_l1407_140706
