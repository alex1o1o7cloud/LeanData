import Mathlib

namespace NUMINAMATH_CALUDE_juan_peter_speed_difference_l3787_378794

/-- The speed difference between Juan and Peter -/
def speed_difference (juan_speed peter_speed : ℝ) : ℝ :=
  juan_speed - peter_speed

/-- The total distance traveled by Juan and Peter -/
def total_distance (juan_speed peter_speed : ℝ) (time : ℝ) : ℝ :=
  (juan_speed + peter_speed) * time

theorem juan_peter_speed_difference :
  ∃ (juan_speed : ℝ),
    speed_difference juan_speed 5.0 = 3 ∧
    total_distance juan_speed 5.0 1.5 = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_juan_peter_speed_difference_l3787_378794


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3787_378721

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f a x ≥ 5) ∧ (∃ x, f a x = 5) → a = -6 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3787_378721


namespace NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l3787_378714

def is_in_range (n : ℕ) : Prop := 7 ≤ n ∧ n ≤ 49

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def remainder_3_mod_5 (n : ℕ) : Prop := n % 5 = 3

-- We don't need to define primality as it's already in Mathlib

theorem no_numbers_satisfying_conditions :
  ¬∃ n : ℕ, is_in_range n ∧ divisible_by_6 n ∧ remainder_3_mod_5 n ∧ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l3787_378714


namespace NUMINAMATH_CALUDE_kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3787_378750

/-- The cost difference between the first and third present given Kyle's purchases. -/
theorem kyles_presents_cost_difference : ℕ → Prop :=
  fun difference =>
    ∀ (cost_1 cost_2 cost_3 : ℕ),
      cost_1 = 18 →
      cost_2 = cost_1 + 7 →
      cost_3 < cost_1 →
      cost_1 + cost_2 + cost_3 = 50 →
      difference = cost_1 - cost_3

/-- The cost difference between the first and third present is 11. -/
theorem kyles_presents_cost_difference_is_11 : kyles_presents_cost_difference 11 := by
  sorry

end NUMINAMATH_CALUDE_kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3787_378750


namespace NUMINAMATH_CALUDE_sequence_property_l3787_378761

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |> List.sum

theorem sequence_property (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - 4) →
  (∀ n : ℕ, n > 0 → a n = 2^(n+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3787_378761


namespace NUMINAMATH_CALUDE_train_crossing_time_l3787_378737

/-- Given a train crossing two platforms, calculate the time to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (h1 : train_length = 270) 
  (h2 : platform1_length = 120) 
  (h3 : platform2_length = 250) 
  (h4 : time1 = 15) : 
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3787_378737


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achievable_l3787_378786

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by sorry

theorem equality_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achievable_l3787_378786


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l3787_378792

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 475 →
  total_boxes = 19 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l3787_378792


namespace NUMINAMATH_CALUDE_reunion_handshakes_l3787_378763

theorem reunion_handshakes (n : ℕ) : n > 0 → (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l3787_378763


namespace NUMINAMATH_CALUDE_max_product_l3787_378711

def digits : List Nat := [3, 5, 7, 8, 9]

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat := (three_digit a b c) * (two_digit d e)

theorem max_product :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    product a b c d e ≤ product 9 7 5 8 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3787_378711


namespace NUMINAMATH_CALUDE_odd_monotonic_function_conditions_l3787_378787

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem odd_monotonic_function_conditions (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) →  -- f is an odd function
  (∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → f a b c x ≤ f a b c y) →  -- f is monotonic on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_odd_monotonic_function_conditions_l3787_378787


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3787_378783

theorem movie_of_the_year_fraction (total_members : ℕ) (min_appearances : ℚ) : 
  total_members = 795 → min_appearances = 198.75 → min_appearances / total_members = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3787_378783


namespace NUMINAMATH_CALUDE_total_broken_marbles_l3787_378732

def marble_set_1 : ℕ := 50
def marble_set_2 : ℕ := 60
def broken_percent_1 : ℚ := 10 / 100
def broken_percent_2 : ℚ := 20 / 100

theorem total_broken_marbles :
  ⌊marble_set_1 * broken_percent_1⌋ + ⌊marble_set_2 * broken_percent_2⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_broken_marbles_l3787_378732


namespace NUMINAMATH_CALUDE_westeros_max_cursed_roads_l3787_378704

/-- A graph representing the Westeros Empire -/
structure WesterosGraph where
  /-- The number of cities (vertices) in the graph -/
  num_cities : Nat
  /-- The number of roads (edges) in the graph -/
  num_roads : Nat
  /-- The graph is initially connected -/
  is_connected : Bool
  /-- The number of kingdoms formed after cursing some roads -/
  num_kingdoms : Nat

/-- The maximum number of roads that can be cursed -/
def max_cursed_roads (g : WesterosGraph) : Nat :=
  g.num_roads - (g.num_cities - g.num_kingdoms)

/-- Theorem stating the maximum number of roads that can be cursed -/
theorem westeros_max_cursed_roads (g : WesterosGraph) 
  (h1 : g.num_cities = 1000)
  (h2 : g.num_roads = 2017)
  (h3 : g.is_connected = true)
  (h4 : g.num_kingdoms = 7) :
  max_cursed_roads g = 1024 := by
  sorry

#eval max_cursed_roads { num_cities := 1000, num_roads := 2017, is_connected := true, num_kingdoms := 7 }

end NUMINAMATH_CALUDE_westeros_max_cursed_roads_l3787_378704


namespace NUMINAMATH_CALUDE_no_solution_for_f_l3787_378780

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := (sumOfDigits n.val) * n.val

/-- 3-adic valuation of a natural number -/
def threeAdicVal (n : ℕ) : ℕ := sorry

/-- Main theorem: There is no positive integer n such that f(n) = 19091997 -/
theorem no_solution_for_f :
  ∀ n : ℕ+, f n ≠ 19091997 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_f_l3787_378780


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3787_378751

/-- Given a function f : ℝ → ℝ that satisfies the functional equation
    3f(x) + 2f(1-x) = 4x for all x, prove that f(x) = 4x - 8/5 for all x. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x, 3 * f x + 2 * f (1 - x) = 4 * x) :
  ∀ x, f x = 4 * x - 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3787_378751


namespace NUMINAMATH_CALUDE_tailor_buttons_count_l3787_378703

/-- The number of buttons purchased by a tailor -/
theorem tailor_buttons_count : 
  let green : ℕ := 90
  let yellow : ℕ := green + 10
  let blue : ℕ := green - 5
  let red : ℕ := 2 * (yellow + blue)
  green + yellow + blue + red = 645 := by sorry

end NUMINAMATH_CALUDE_tailor_buttons_count_l3787_378703


namespace NUMINAMATH_CALUDE_mans_downstream_speed_l3787_378747

/-- Proves that given a man's upstream speed of 30 kmph and still water speed of 35 kmph, his downstream speed is 40 kmph. -/
theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 30) 
  (h2 : still_water_speed = 35) : 
  still_water_speed + (still_water_speed - upstream_speed) = 40 := by
  sorry

#check mans_downstream_speed

end NUMINAMATH_CALUDE_mans_downstream_speed_l3787_378747


namespace NUMINAMATH_CALUDE_complex_modulus_l3787_378797

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I^3) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l3787_378797


namespace NUMINAMATH_CALUDE_chime_1500_date_l3787_378779

/-- Represents a date with year, month, and day. -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute. -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Represents the chiming pattern of the clock. -/
def chime_pattern (hour : Nat) (minute : Nat) : Nat :=
  if minute == 0 then hour
  else if minute == 15 || minute == 30 then 1
  else 0

/-- Calculates the number of chimes from a given start date and time to an end date and time. -/
def count_chimes (start_date : Date) (start_time : Time) (end_date : Date) (end_time : Time) : Nat :=
  sorry

/-- The theorem to be proved. -/
theorem chime_1500_date :
  let start_date := Date.mk 2003 2 28
  let start_time := Time.mk 18 30
  let end_date := Date.mk 2003 3 13
  count_chimes start_date start_time end_date (Time.mk 23 59) ≥ 1500 ∧
  count_chimes start_date start_time end_date (Time.mk 0 0) < 1500 :=
sorry

end NUMINAMATH_CALUDE_chime_1500_date_l3787_378779


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l3787_378715

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin n → Bool

/-- Checks if a pair of numbers sum to a perfect square -/
def IsPerfectSquareSum (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + b = k * k

/-- The main theorem statement -/
theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), ∃ (i j : Fin n), i ≠ j ∧ p i = p j ∧ IsPerfectSquareSum (i.val + 1) (j.val + 1) :=
sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l3787_378715


namespace NUMINAMATH_CALUDE_trapezoid_in_isosceles_triangle_l3787_378772

/-- An isosceles triangle with a trapezoid inscribed within it. -/
structure IsoscelesTriangleWithTrapezoid where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- The distance from the apex to point D on side AB -/
  x : ℝ
  /-- The perimeter of the inscribed trapezoid -/
  trapezoidPerimeter : ℝ

/-- Theorem stating the condition for the inscribed trapezoid in an isosceles triangle -/
theorem trapezoid_in_isosceles_triangle 
    (t : IsoscelesTriangleWithTrapezoid) 
    (h1 : t.base = 12) 
    (h2 : t.side = 18) 
    (h3 : t.trapezoidPerimeter = 40) : 
    t.x = 6 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_in_isosceles_triangle_l3787_378772


namespace NUMINAMATH_CALUDE_base4_representation_has_four_digits_l3787_378765

/-- Converts a natural number from decimal to base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 75

/-- Theorem stating that the base 4 representation of 75 has four digits -/
theorem base4_representation_has_four_digits :
  (toBase4 decimalNumber).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_base4_representation_has_four_digits_l3787_378765


namespace NUMINAMATH_CALUDE_tower_of_hanoi_correct_l3787_378767

/-- Minimum number of moves required to solve the Tower of Hanoi problem with n discs -/
def tower_of_hanoi (n : ℕ) : ℕ :=
  2^n - 1

/-- Theorem: The minimum number of moves for the Tower of Hanoi problem with n discs is 2^n - 1 -/
theorem tower_of_hanoi_correct (n : ℕ) : tower_of_hanoi n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_tower_of_hanoi_correct_l3787_378767


namespace NUMINAMATH_CALUDE_zero_most_frequent_units_digit_l3787_378771

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 9

-- Function to calculate the units digit of a sum
def unitsDigitOfSum (a b : ℕ) : ℕ := (a + b) % 10

-- Function to count occurrences of a specific units digit
def countOccurrences (digit : ℕ) : ℕ :=
  numbers.card * numbers.card

-- Theorem stating that 0 is the most frequent units digit
theorem zero_most_frequent_units_digit :
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 →
    countOccurrences 0 > countOccurrences d :=
sorry

end NUMINAMATH_CALUDE_zero_most_frequent_units_digit_l3787_378771


namespace NUMINAMATH_CALUDE_lauren_subscription_rate_l3787_378728

/-- Represents Lauren's earnings from her social media channel -/
structure Earnings where
  commercialRate : ℚ  -- Rate per commercial view
  commercialViews : ℕ -- Number of commercial views
  subscriptions : ℕ   -- Number of subscriptions
  totalRevenue : ℚ    -- Total revenue
  subscriptionRate : ℚ -- Rate per subscription

/-- Theorem stating that Lauren's subscription rate is $1 -/
theorem lauren_subscription_rate 
  (e : Earnings) 
  (h1 : e.commercialRate = 1/2)      -- $0.50 per commercial view
  (h2 : e.commercialViews = 100)     -- 100 commercial views
  (h3 : e.subscriptions = 27)        -- 27 subscriptions
  (h4 : e.totalRevenue = 77)         -- Total revenue is $77
  : e.subscriptionRate = 1 := by
  sorry


end NUMINAMATH_CALUDE_lauren_subscription_rate_l3787_378728


namespace NUMINAMATH_CALUDE_three_digit_base_problem_l3787_378759

theorem three_digit_base_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1993 ∧
    x + y + z = 22 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_base_problem_l3787_378759


namespace NUMINAMATH_CALUDE_triangle_properties_l3787_378777

/-- Triangle ABC with vertices A(3,0), B(4,6), and C(0,8) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from point B to side AC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  fun p => 2 * p.1 - p.2 - 6 = 0

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := 13

theorem triangle_properties :
  let t : Triangle := { A := (3, 0), B := (4, 6), C := (0, 8) }
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 6 = 0) ∧
  area t = 13 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3787_378777


namespace NUMINAMATH_CALUDE_xiaoming_mother_expenses_l3787_378753

/-- Represents a financial transaction with an amount in Yuan -/
structure Transaction where
  amount : Int

/-- Calculates the net result of a list of transactions -/
def netResult (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

theorem xiaoming_mother_expenses : 
  let transactions : List Transaction := [
    { amount := 42 },   -- Transfer from Hong
    { amount := -30 },  -- Paying phone bill
    { amount := -51 }   -- Scan QR code for payment
  ]
  netResult transactions = -39 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_mother_expenses_l3787_378753


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_neg_two_l3787_378708

theorem expression_zero_iff_x_neg_two (x : ℝ) :
  (x^2 - 4) / (4*x - 8) = 0 ↔ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_neg_two_l3787_378708


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3787_378756

theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    a + b + c = 180 →        -- Sum of angles in a triangle is 180°
    ∃ (x : ℝ), 
      a = 3*x ∧ b = 4*x ∧ c = 5*x →  -- Angles are in ratio 3:4:5
      max a (max b c) = 75 :=  -- The largest angle is 75°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3787_378756


namespace NUMINAMATH_CALUDE_janice_stairs_walked_l3787_378700

/-- The number of flights of stairs to Janice's office -/
def flights_to_office : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_to_office * times_up + flights_to_office * times_down

theorem janice_stairs_walked : total_flights = 24 := by
  sorry

end NUMINAMATH_CALUDE_janice_stairs_walked_l3787_378700


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3787_378781

/-- Given an equilateral triangle where the area is numerically twice the length of one of its sides,
    the perimeter of the triangle is 8√3 units. -/
theorem equilateral_triangle_perimeter : ∀ s : ℝ,
  s > 0 →  -- side length is positive
  (s^2 * Real.sqrt 3) / 4 = 2 * s →  -- area is twice the side length
  3 * s = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3787_378781


namespace NUMINAMATH_CALUDE_theater_eye_colors_l3787_378742

theorem theater_eye_colors (total : ℕ) (blue : ℕ) (brown : ℕ) (black : ℕ) (green : ℕ)
  (h_total : total = 100)
  (h_blue : blue = 19)
  (h_brown : brown = total / 2)
  (h_black : black = total / 4)
  (h_green : green = total - (blue + brown + black)) :
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_theater_eye_colors_l3787_378742


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3787_378723

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : Real.sin x ^ 4 / a ^ 2 + Real.cos x ^ 4 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) :
  Real.sin x ^ 2008 / a ^ 2006 + Real.cos x ^ 2008 / b ^ 2006 = 1 / (a ^ 2 + b ^ 2) ^ 1003 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3787_378723


namespace NUMINAMATH_CALUDE_seating_position_indeterminable_l3787_378782

/-- Represents a seat number as a pair of integers -/
def SeatNumber := ℤ × ℤ

/-- Represents a seating position as a row and column -/
structure SeatingPosition where
  row : ℤ
  column : ℤ

/-- Function that attempts to determine the seating position from a seat number -/
noncomputable def determineSeatingPosition (seatNumber : SeatNumber) : Option SeatingPosition :=
  sorry

/-- Theorem stating that it's not possible to determine the seating position
    from the seat number (2, 4) without additional information -/
theorem seating_position_indeterminable :
  ∀ (f : SeatNumber → Option SeatingPosition),
    ∃ (p1 p2 : SeatingPosition), p1 ≠ p2 ∧
      (f (2, 4) = some p1 ∨ f (2, 4) = some p2 ∨ f (2, 4) = none) :=
by
  sorry

end NUMINAMATH_CALUDE_seating_position_indeterminable_l3787_378782


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_eighth_l3787_378725

theorem halfway_between_one_third_and_one_eighth :
  (1 / 3 : ℚ) / 2 + (1 / 8 : ℚ) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_eighth_l3787_378725


namespace NUMINAMATH_CALUDE_square_field_area_l3787_378748

/-- The area of a square field with side length 10 meters is 100 square meters. -/
theorem square_field_area :
  let side_length : ℝ := 10
  let area : ℝ := side_length * side_length
  area = 100 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3787_378748


namespace NUMINAMATH_CALUDE_incorrect_inequality_implication_l3787_378733

theorem incorrect_inequality_implication : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_implication_l3787_378733


namespace NUMINAMATH_CALUDE_inequality_proof_l3787_378762

theorem inequality_proof :
  (∀ x : ℝ, |x - 1| + |x - 2| ≥ 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) →
    a + 2 * b + 3 * c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3787_378762


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3787_378795

theorem smallest_angle_in_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 3) (h3 : c = 2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = Real.arccos (7/8) ∧ C ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧ C ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3787_378795


namespace NUMINAMATH_CALUDE_intersection_dot_product_l3787_378702

/-- Given a line Ax + By + C = 0 intersecting the circle x^2 + y^2 = 9 at points P and Q,
    where A^2, C^2, and B^2 form an arithmetic sequence, prove that OP · PQ = -1 -/
theorem intersection_dot_product 
  (A B C : ℝ) 
  (P Q : ℝ × ℝ) 
  (h_line : ∀ x y, A * x + B * y + C = 0 ↔ (x, y) = P ∨ (x, y) = Q)
  (h_circle : P.1^2 + P.2^2 = 9 ∧ Q.1^2 + Q.2^2 = 9)
  (h_arithmetic : 2 * C^2 = A^2 + B^2)
  (h_distinct : P ≠ Q) :
  (P.1 * (Q.1 - P.1) + P.2 * (Q.2 - P.2) : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l3787_378702


namespace NUMINAMATH_CALUDE_correct_average_marks_l3787_378773

/-- Proves that the correct average marks for a class of 50 students is 82.8,
    given an initial average of 85 and three incorrectly recorded marks. -/
theorem correct_average_marks
  (num_students : ℕ)
  (initial_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ)
  (h_num_students : num_students = 50)
  (h_initial_average : initial_average = 85)
  (h_incorrect1 : incorrect_mark1 = 95)
  (h_incorrect2 : incorrect_mark2 = 78)
  (h_incorrect3 : incorrect_mark3 = 120)
  (h_correct1 : correct_mark1 = 45)
  (h_correct2 : correct_mark2 = 58)
  (h_correct3 : correct_mark3 = 80) :
  (num_students : ℚ) * initial_average - (incorrect_mark1 - correct_mark1 + incorrect_mark2 - correct_mark2 + incorrect_mark3 - correct_mark3 : ℚ) / num_students = 82.8 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l3787_378773


namespace NUMINAMATH_CALUDE_unique_pell_solution_l3787_378717

def isPellSolution (x y : ℕ+) : Prop :=
  (x : ℤ)^2 - 2003 * (y : ℤ)^2 = 1

def isFundamentalSolution (x₀ y₀ : ℕ+) : Prop :=
  isPellSolution x₀ y₀ ∧ ∀ x y : ℕ+, isPellSolution x y → x₀ ≤ x ∧ y₀ ≤ y

def allPrimeFactorsDivide (x x₀ : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

theorem unique_pell_solution (x₀ y₀ x y : ℕ+) :
  isFundamentalSolution x₀ y₀ →
  isPellSolution x y →
  allPrimeFactorsDivide x x₀ →
  x = x₀ ∧ y = y₀ := by
  sorry

end NUMINAMATH_CALUDE_unique_pell_solution_l3787_378717


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l3787_378713

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The conditions of Bertha's family -/
def bertha_conditions : BerthaFamily where
  daughters := 8
  granddaughters := 32
  total_descendants := 40
  daughters_with_children := 8

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children 
  (family : BerthaFamily) 
  (h1 : family.daughters = bertha_conditions.daughters)
  (h2 : family.total_descendants = bertha_conditions.total_descendants)
  (h3 : family.granddaughters = family.total_descendants - family.daughters)
  (h4 : family.daughters_with_children * 4 = family.granddaughters)
  (h5 : family.daughters_with_children ≤ family.daughters) :
  family.total_descendants - family.daughters_with_children = 32 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l3787_378713


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l3787_378739

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ :=
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℕ :=
  (n / 1000) - (n % 10)

theorem four_digit_number_theorem (n : ℕ) :
  is_valid_four_digit_number n ∧
  digit_sum n = 16 ∧
  middle_digits_sum n = 10 ∧
  thousands_minus_units n = 2 ∧
  n % 11 = 0 →
  n = 4642 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l3787_378739


namespace NUMINAMATH_CALUDE_range_of_a_l3787_378705

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3787_378705


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3787_378727

/-- Given two parallel 2D vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3787_378727


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3787_378790

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root
    if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3787_378790


namespace NUMINAMATH_CALUDE_jake_not_drop_coffee_l3787_378764

/-- The probability of Jake tripping over his dog in the morning -/
def prob_trip : ℝ := 0.4

/-- The probability of Jake dropping his coffee when he trips -/
def prob_drop_given_trip : ℝ := 0.25

/-- The probability of Jake not dropping his coffee in the morning -/
def prob_not_drop : ℝ := 1 - prob_trip * prob_drop_given_trip

theorem jake_not_drop_coffee : prob_not_drop = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_jake_not_drop_coffee_l3787_378764


namespace NUMINAMATH_CALUDE_xy_value_l3787_378734

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3787_378734


namespace NUMINAMATH_CALUDE_fraction_simplification_l3787_378758

theorem fraction_simplification :
  (16 : ℚ) / 54 * 27 / 8 * 64 / 81 = 64 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3787_378758


namespace NUMINAMATH_CALUDE_prime_divisibility_equivalence_l3787_378731

theorem prime_divisibility_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℤ, ∃ d₁ : ℤ, x^2 - x + 3 = d₁ * p) ↔ 
  (∃ y : ℤ, ∃ d₂ : ℤ, y^2 - y + 25 = d₂ * p) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_equivalence_l3787_378731


namespace NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l3787_378744

/-- Given a line segment AB extended to P such that AP:PB = 10:3,
    prove that the position vector of P can be expressed as P = -3/7*A + 10/7*B,
    where A and B are the position vectors of points A and B respectively. -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 10 / 3) -- AP:PB = 10:3
  : ∃ (t u : ℝ), P = t • A + u • B ∧ t = -3/7 ∧ u = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l3787_378744


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_2500_l3787_378709

/-- The multiple of 18 closest to 2500 is 2502 -/
theorem closest_multiple_of_18_to_2500 :
  ∀ n : ℤ, 18 ∣ n → |n - 2500| ≥ |2502 - 2500| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_2500_l3787_378709


namespace NUMINAMATH_CALUDE_chicken_flock_ratio_l3787_378724

/-- Chicken flock problem -/
theorem chicken_flock_ratio : 
  ∀ (susie_rir susie_gc britney_gc britney_total : ℕ),
  susie_rir = 11 →
  susie_gc = 6 →
  britney_gc = susie_gc / 2 →
  britney_total = (susie_rir + susie_gc) + 8 →
  ∃ (britney_rir : ℕ),
    britney_rir + britney_gc = britney_total ∧
    britney_rir = 2 * susie_rir :=
by sorry

end NUMINAMATH_CALUDE_chicken_flock_ratio_l3787_378724


namespace NUMINAMATH_CALUDE_kitchen_module_cost_is_20000_l3787_378788

/-- Represents the cost of a modular home construction --/
structure ModularHomeCost where
  totalSize : Nat
  kitchenSize : Nat
  bathroomSize : Nat
  bathroomCost : Nat
  otherCost : Nat
  kitchenCount : Nat
  bathroomCount : Nat
  totalCost : Nat

/-- Calculates the cost of the kitchen module --/
def kitchenModuleCost (home : ModularHomeCost) : Nat :=
  let otherSize := home.totalSize - home.kitchenSize * home.kitchenCount - home.bathroomSize * home.bathroomCount
  let otherTotalCost := otherSize * home.otherCost
  let bathroomTotalCost := home.bathroomCost * home.bathroomCount
  home.totalCost - otherTotalCost - bathroomTotalCost

/-- Theorem: The kitchen module costs $20,000 --/
theorem kitchen_module_cost_is_20000 (home : ModularHomeCost) 
  (h1 : home.totalSize = 2000)
  (h2 : home.kitchenSize = 400)
  (h3 : home.bathroomSize = 150)
  (h4 : home.bathroomCost = 12000)
  (h5 : home.otherCost = 100)
  (h6 : home.kitchenCount = 1)
  (h7 : home.bathroomCount = 2)
  (h8 : home.totalCost = 174000) :
  kitchenModuleCost home = 20000 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_module_cost_is_20000_l3787_378788


namespace NUMINAMATH_CALUDE_second_grade_selection_l3787_378776

/-- Represents a stratified sampling scenario in a school -/
structure SchoolSampling where
  first_grade : ℕ
  second_grade : ℕ
  total_selected : ℕ
  first_grade_selected : ℕ

/-- Calculates the number of students selected from the second grade -/
def second_grade_selected (s : SchoolSampling) : ℕ :=
  s.total_selected - s.first_grade_selected

/-- Theorem stating that in the given scenario, 18 students are selected from the second grade -/
theorem second_grade_selection (s : SchoolSampling) 
  (h1 : s.first_grade = 400)
  (h2 : s.second_grade = 360)
  (h3 : s.total_selected = 56)
  (h4 : s.first_grade_selected = 20) :
  second_grade_selected s = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_selection_l3787_378776


namespace NUMINAMATH_CALUDE_proportion_check_l3787_378752

/-- A set of four positive real numbers forms a proportion if the product of the first and last
    numbers equals the product of the middle two numbers. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

theorem proportion_check :
  IsProportional 5 15 3 9 ∧
  ¬IsProportional 4 5 6 7 ∧
  ¬IsProportional 3 4 5 8 ∧
  ¬IsProportional 8 4 1 3 :=
by sorry

end NUMINAMATH_CALUDE_proportion_check_l3787_378752


namespace NUMINAMATH_CALUDE_age_ratio_in_one_year_l3787_378757

/-- Mike's current age -/
def m : ℕ := sorry

/-- Sarah's current age -/
def s : ℕ := sorry

/-- The condition that 3 years ago, Mike was twice as old as Sarah -/
axiom three_years_ago : m - 3 = 2 * (s - 3)

/-- The condition that 5 years ago, Mike was three times as old as Sarah -/
axiom five_years_ago : m - 5 = 3 * (s - 5)

/-- The number of years until the ratio of their ages is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- The theorem stating that the number of years until the ratio of their ages is 3:2 is 1 -/
theorem age_ratio_in_one_year : 
  years_until_ratio = 1 ∧ (m + years_until_ratio) / (s + years_until_ratio) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_in_one_year_l3787_378757


namespace NUMINAMATH_CALUDE_nth_derivative_reciprocal_polynomial_l3787_378754

theorem nth_derivative_reciprocal_polynomial (k n : ℕ) (h : k > 0) :
  let f : ℝ → ℝ := λ x => 1 / (x^k - 1)
  let nth_derivative := (deriv^[n] f)
  ∃ P : ℝ → ℝ, (∀ x, nth_derivative x = P x / (x^k - 1)^(n + 1)) ∧
                P 1 = (-1)^n * n.factorial * k^n :=
by
  sorry

end NUMINAMATH_CALUDE_nth_derivative_reciprocal_polynomial_l3787_378754


namespace NUMINAMATH_CALUDE_harry_travel_time_l3787_378701

/-- Calculates the total travel time for Harry's journey --/
def total_travel_time (initial_bus_time remaining_bus_time : ℕ) : ℕ :=
  let bus_time := initial_bus_time + remaining_bus_time
  let walk_time := bus_time / 2
  bus_time + walk_time

/-- Proves that Harry's total travel time is 60 minutes --/
theorem harry_travel_time :
  total_travel_time 15 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_harry_travel_time_l3787_378701


namespace NUMINAMATH_CALUDE_count_numeric_hex_500_l3787_378796

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal digit is numeric (0-9) --/
def is_numeric_hex_digit (d : Fin 16) : Bool :=
  d.val < 10

/-- Checks if a hexadecimal number contains only numeric digits --/
def has_only_numeric_digits (h : List (Fin 16)) : Bool :=
  h.all is_numeric_hex_digit

/-- Counts numbers with only numeric hexadecimal digits up to n --/
def count_numeric_hex (n : ℕ+) : ℕ :=
  (List.range n).filter (fun i => has_only_numeric_digits (to_hex ⟨i + 1, by sorry⟩)) |>.length

theorem count_numeric_hex_500 : count_numeric_hex 500 = 199 :=
  sorry

end NUMINAMATH_CALUDE_count_numeric_hex_500_l3787_378796


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3787_378720

theorem arithmetic_expression_equality : 9 - 3 / (1 / 3) + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3787_378720


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3787_378798

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3787_378798


namespace NUMINAMATH_CALUDE_four_folds_result_l3787_378755

/-- Represents a square piece of paper. -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents a fold on the paper. -/
inductive Fold
  | Diagonal
  | Perpendicular

/-- Represents the pattern of creases on the unfolded paper. -/
structure CreasePattern :=
  (folds : List Fold)
  (is_symmetrical : Bool)
  (center_at_mean : Bool)

/-- Function to perform a single fold. -/
def fold (s : Square) : CreasePattern :=
  sorry

/-- Function to perform four folds. -/
def four_folds (s : Square) : CreasePattern :=
  sorry

/-- Theorem stating the result of folding a square paper four times. -/
theorem four_folds_result (s : Square) :
  let pattern := four_folds s
  pattern.is_symmetrical ∧ 
  pattern.center_at_mean ∧ 
  (∃ (d p : Fold), d = Fold.Diagonal ∧ p = Fold.Perpendicular ∧ d ∈ pattern.folds ∧ p ∈ pattern.folds) :=
by sorry

end NUMINAMATH_CALUDE_four_folds_result_l3787_378755


namespace NUMINAMATH_CALUDE_art_of_passing_through_walls_l3787_378770

theorem art_of_passing_through_walls (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * 8 / n)) ↔ n = 63 :=
sorry

end NUMINAMATH_CALUDE_art_of_passing_through_walls_l3787_378770


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3787_378789

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 17 players, where each player plays twice against every other player, the total number of games played is 272. -/
theorem chess_tournament_games :
  tournament_games 17 = 272 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3787_378789


namespace NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_l3787_378785

-- Define the properties of the triangles
def isIsosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the theorem
theorem isosceles_triangles_same_perimeter 
  (c d : ℝ) 
  (h1 : isIsosceles 7 7 10) 
  (h2 : isIsosceles c c d) 
  (h3 : c ≠ d) 
  (h4 : perimeter 7 7 10 = 24) 
  (h5 : perimeter c c d = 24) :
  d = 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_l3787_378785


namespace NUMINAMATH_CALUDE_quadratic_and_line_properties_l3787_378799

/-- Given a quadratic equation with two equal real roots, prove the value of m and the quadrants through which the corresponding line passes -/
theorem quadratic_and_line_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 + (2*m + 1)*x + m^2 + 2 = 0 → (∃! r : ℝ, x = r)) →
  (m = 7/4 ∧ 
   ∀ x y : ℝ, y = (2*m - 3)*x - 4*m + 6 →
   (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (2*m - 3)*x₁ - 4*m + 6) ∧
   (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ < 0 ∧ y₂ = (2*m - 3)*x₂ - 4*m + 6) ∧
   (∃ x₃ y₃ : ℝ, x₃ > 0 ∧ y₃ < 0 ∧ y₃ = (2*m - 3)*x₃ - 4*m + 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_and_line_properties_l3787_378799


namespace NUMINAMATH_CALUDE_systematic_sampling_l3787_378735

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  groups : Nat
  first_group_number : Nat
  sixteenth_group_number : Nat

/-- The systematic sampling theorem -/
theorem systematic_sampling
  (s : SystematicSample)
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.groups = 20)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_l3787_378735


namespace NUMINAMATH_CALUDE_system_solution_1_l3787_378736

theorem system_solution_1 (x y : ℝ) :
  x + y = 10^20 ∧ x - y = 10^19 → x = 55 * 10^18 ∧ y = 45 * 10^18 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_1_l3787_378736


namespace NUMINAMATH_CALUDE_problem_solution_l3787_378746

theorem problem_solution : 
  (-24 / (1/2 - 1/6 + 1/3) = -36) ∧ 
  (-1^3 - |(-9)| + 3 + 6 * (-1/3)^2 = -19/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3787_378746


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3787_378749

theorem quadratic_equation_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) :
  6 * a^2 + 9 * a - 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3787_378749


namespace NUMINAMATH_CALUDE_dragon_eventual_defeat_l3787_378745

/-- Represents the probabilities of head growth after each cut -/
structure HeadGrowthProbabilities where
  two_heads : ℝ
  one_head : ℝ
  no_heads : ℝ

/-- The probability of eventually defeating the dragon -/
def defeat_probability (probs : HeadGrowthProbabilities) : ℝ :=
  sorry

/-- The theorem stating that the dragon will eventually be defeated -/
theorem dragon_eventual_defeat (probs : HeadGrowthProbabilities) 
  (h1 : probs.two_heads = 1/4)
  (h2 : probs.one_head = 1/3)
  (h3 : probs.no_heads = 5/12)
  (h4 : probs.two_heads + probs.one_head + probs.no_heads = 1) :
  defeat_probability probs = 1 := by
  sorry

end NUMINAMATH_CALUDE_dragon_eventual_defeat_l3787_378745


namespace NUMINAMATH_CALUDE_abs_negative_two_l3787_378793

theorem abs_negative_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l3787_378793


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3787_378719

theorem systematic_sampling_interval 
  (population : ℕ) 
  (sample_size : ℕ) 
  (h1 : population = 800) 
  (h2 : sample_size = 40) 
  (h3 : population > 0) 
  (h4 : sample_size > 0) :
  population / sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3787_378719


namespace NUMINAMATH_CALUDE_opposite_numbers_l3787_378774

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l3787_378774


namespace NUMINAMATH_CALUDE_profit_is_three_l3787_378741

/-- Calculates the profit from selling apples and oranges -/
def calculate_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                     (orange_buy_price : ℚ) (orange_sell_price : ℚ)
                     (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := (apple_sell_price - apple_buy_price) * apples_sold
  let orange_profit := (orange_sell_price - orange_buy_price) * oranges_sold
  apple_profit + orange_profit

/-- Proves that the profit from selling 5 apples and 5 oranges is $3 -/
theorem profit_is_three :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  calculate_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_three_l3787_378741


namespace NUMINAMATH_CALUDE_ourSystem_is_linear_l3787_378726

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := fun x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 5 }
  eq2 := { a := 0, b := 1, c := 2 }

/-- Predicate to check if a system is linear -/
def isLinearSystem (system : SystemOfTwoEquations) : Prop :=
  system.eq1.a ≠ 0 ∨ system.eq1.b ≠ 0 ∧
  system.eq2.a ≠ 0 ∨ system.eq2.b ≠ 0

theorem ourSystem_is_linear : isLinearSystem ourSystem := by
  sorry

end NUMINAMATH_CALUDE_ourSystem_is_linear_l3787_378726


namespace NUMINAMATH_CALUDE_unique_non_negative_twelve_quotient_l3787_378706

def pairs : List (Int × Int) := [(24, -2), (-36, 3), (144, 12), (-48, 4), (72, -6)]

theorem unique_non_negative_twelve_quotient :
  ∃! p : Int × Int, p ∈ pairs ∧ p.1 / p.2 ≠ -12 :=
by sorry

end NUMINAMATH_CALUDE_unique_non_negative_twelve_quotient_l3787_378706


namespace NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l3787_378778

/-- The quadratic equation in question -/
def quadratic (a x : ℝ) : ℝ := x^2 + 2*a*x + 2*a^2 + 4*a + 3

/-- The sum of squares of roots of the quadratic equation -/
def sumOfSquaresOfRoots (a : ℝ) : ℝ := -8*a - 6

/-- The theorem stating the maximum sum of squares of roots and when it occurs -/
theorem max_sum_of_squares_of_roots :
  (∃ (a : ℝ), ∀ (b : ℝ), sumOfSquaresOfRoots b ≤ sumOfSquaresOfRoots a) ∧
  (sumOfSquaresOfRoots (-3) = 18) := by
  sorry

#check max_sum_of_squares_of_roots

end NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l3787_378778


namespace NUMINAMATH_CALUDE_total_vegetarian_consumers_is_33_l3787_378707

/-- Represents the dietary information of a family -/
structure DietaryInfo where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both : ℕ
  gluten_free : ℕ
  vegan : ℕ
  non_veg_gluten_free : ℕ
  veg_gluten_free : ℕ
  both_gluten_free : ℕ
  vegan_strict_veg : ℕ
  vegan_non_veg : ℕ

/-- Calculates the total number of people consuming vegetarian dishes -/
def total_vegetarian_consumers (info : DietaryInfo) : ℕ :=
  info.only_vegetarian + info.both + info.vegan_non_veg

/-- The main theorem stating that the total number of vegetarian consumers is 33 -/
theorem total_vegetarian_consumers_is_33 (info : DietaryInfo) 
  (h1 : info.only_vegetarian = 19)
  (h2 : info.only_non_vegetarian = 9)
  (h3 : info.both = 12)
  (h4 : info.gluten_free = 6)
  (h5 : info.vegan = 5)
  (h6 : info.non_veg_gluten_free = 2)
  (h7 : info.veg_gluten_free = 3)
  (h8 : info.both_gluten_free = 1)
  (h9 : info.vegan_strict_veg = 3)
  (h10 : info.vegan_non_veg = 2) :
  total_vegetarian_consumers info = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_consumers_is_33_l3787_378707


namespace NUMINAMATH_CALUDE_sauce_per_burger_is_quarter_cup_l3787_378712

/-- The amount of barbecue sauce per burger -/
def sauce_per_burger (total_sauce : ℚ) (sauce_per_sandwich : ℚ) (num_sandwiches : ℕ) (num_burgers : ℕ) : ℚ :=
  (total_sauce - sauce_per_sandwich * num_sandwiches) / num_burgers

/-- Theorem stating that the amount of sauce per burger is 1/4 cup -/
theorem sauce_per_burger_is_quarter_cup :
  sauce_per_burger 5 (1/6) 18 8 = 1/4 := by sorry

end NUMINAMATH_CALUDE_sauce_per_burger_is_quarter_cup_l3787_378712


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l3787_378730

/-- The weight that a toy bridge must support -/
theorem toy_bridge_weight (full_cans : Nat) (soda_per_can : Nat) (empty_can_weight : Nat) (additional_empty_cans : Nat) : 
  full_cans * (soda_per_can + empty_can_weight) + additional_empty_cans * empty_can_weight = 88 :=
by
  sorry

#check toy_bridge_weight 6 12 2 2

end NUMINAMATH_CALUDE_toy_bridge_weight_l3787_378730


namespace NUMINAMATH_CALUDE_marathon_checkpoints_l3787_378775

/-- Represents a circular marathon with checkpoints -/
structure Marathon where
  total_distance : ℕ
  checkpoint_spacing : ℕ
  distance_to_first : ℕ
  distance_from_last : ℕ

/-- Calculates the number of checkpoints in a marathon -/
def num_checkpoints (m : Marathon) : ℕ :=
  (m.total_distance - m.distance_to_first - m.distance_from_last) / m.checkpoint_spacing + 1

/-- Theorem stating that a marathon with given specifications has 5 checkpoints -/
theorem marathon_checkpoints :
  ∃ (m : Marathon),
    m.total_distance = 26 ∧
    m.checkpoint_spacing = 6 ∧
    m.distance_to_first = 1 ∧
    m.distance_from_last = 1 ∧
    num_checkpoints m = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_marathon_checkpoints_l3787_378775


namespace NUMINAMATH_CALUDE_spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l3787_378722

theorem spring_ice_cream_percentage : ℝ → Prop :=
  fun spring_percentage =>
    (spring_percentage + 30 + 25 + 20 = 100) →
    spring_percentage = 25

-- The proof is omitted
theorem spring_ice_cream_percentage_proof : spring_ice_cream_percentage 25 := by
  sorry

end NUMINAMATH_CALUDE_spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l3787_378722


namespace NUMINAMATH_CALUDE_subset_union_equality_l3787_378718

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end NUMINAMATH_CALUDE_subset_union_equality_l3787_378718


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l3787_378740

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (5 - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 :=
by sorry

theorem five_is_largest_integer :
  ∃ x : ℤ, x = 5 ∧
    ((x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3) ∧
    (∀ y : ℤ, y > x → (y - 1 : ℚ) / 4 - 3 / 7 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l3787_378740


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3787_378743

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2 ≥ 100 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 10)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3787_378743


namespace NUMINAMATH_CALUDE_power_of_i_sum_l3787_378760

theorem power_of_i_sum : ∃ (i : ℂ), i^2 = -1 ∧ i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l3787_378760


namespace NUMINAMATH_CALUDE_triangle_perimeter_increase_l3787_378716

/-- Given three equilateral triangles where each subsequent triangle has sides 200% of the previous,
    prove that the percent increase in perimeter from the first to the third triangle is 300%. -/
theorem triangle_perimeter_increase (side_length : ℝ) (side_length_positive : side_length > 0) :
  let first_perimeter := 3 * side_length
  let third_perimeter := 3 * (4 * side_length)
  let percent_increase := (third_perimeter - first_perimeter) / first_perimeter * 100
  percent_increase = 300 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_increase_l3787_378716


namespace NUMINAMATH_CALUDE_lcm_1188_924_l3787_378766

theorem lcm_1188_924 : Nat.lcm 1188 924 = 8316 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1188_924_l3787_378766


namespace NUMINAMATH_CALUDE_expression_evaluation_l3787_378784

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  let f := (((x + 2)^2 * (x^2 - 2*x + 4)^2) / (x^3 + 8)^2)^2 *
            (((x - 2)^2 * (x^2 + 2*x + 4)^2) / (x^3 - 8)^2)^2
  f = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3787_378784


namespace NUMINAMATH_CALUDE_roots_on_circle_l3787_378729

theorem roots_on_circle : ∃ (r : ℝ), r = 1 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z - 1)^3 = 8*z^3 → Complex.abs (z + 1/3) = r := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3787_378729


namespace NUMINAMATH_CALUDE_art_club_theorem_l3787_378769

/-- Represents the number of artworks created by the art club over three school years. -/
def artworks_three_years (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  let artworks_q1 := students_q1 * artworks_per_student
  let artworks_q2_q3 := students_q2_q3 * artworks_per_student
  let artworks_q4_q5 := students_q4_q5 * artworks_per_student
  let artworks_per_year := artworks_q1 + 2 * artworks_q2_q3 + 2 * artworks_q4_q5
  artworks_per_year * years

/-- Represents the number of artworks created in each quarter for the entire club. -/
def artworks_per_quarter (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) : List ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  [students_q1 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q4_q5 * artworks_per_student,
   students_q4_q5 * artworks_per_student]

theorem art_club_theorem :
  artworks_three_years 30 3 4 6 5 3 = 1386 ∧
  artworks_per_quarter 30 3 4 6 = [90, 102, 102, 84, 84] := by
  sorry

end NUMINAMATH_CALUDE_art_club_theorem_l3787_378769


namespace NUMINAMATH_CALUDE_sum_of_remainders_l3787_378768

theorem sum_of_remainders (d e f : ℕ+) 
  (hd : d ≡ 19 [ZMOD 53])
  (he : e ≡ 33 [ZMOD 53])
  (hf : f ≡ 14 [ZMOD 53]) :
  (d + e + f : ℤ) ≡ 13 [ZMOD 53] := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_l3787_378768


namespace NUMINAMATH_CALUDE_cupcake_problem_l3787_378710

theorem cupcake_problem (total_girls : ℕ) (avg_cupcakes : ℚ) (max_cupcakes : ℕ) (no_cupcake_girls : ℕ) :
  total_girls = 12 →
  avg_cupcakes = 3/2 →
  max_cupcakes = 2 →
  no_cupcake_girls = 2 →
  ∃ (two_cupcake_girls : ℕ),
    two_cupcake_girls = 8 ∧
    two_cupcake_girls + no_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = total_girls ∧
    2 * two_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = (avg_cupcakes * total_girls).num :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_problem_l3787_378710


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3787_378738

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from (4, 0) to its asymptote is √2,
    then its eccentricity is (2√14)/7 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b / Real.sqrt (a^2 + b^2) = Real.sqrt 2) →
  (Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 14 / 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3787_378738


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3787_378791

theorem polynomial_factorization (t : ℝ) :
  ∃ (a b c d : ℝ), ∀ (x : ℝ),
    x^4 + t*x^2 + 1 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3787_378791
