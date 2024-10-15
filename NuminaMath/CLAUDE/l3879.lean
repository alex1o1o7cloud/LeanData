import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3879_387964

theorem sqrt_expression_equality : Real.sqrt 3 * Real.sqrt 2 - Real.sqrt 2 + Real.sqrt 8 = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3879_387964


namespace NUMINAMATH_CALUDE_inequality_proof_l3879_387968

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) ≥ 3/2) ∧ 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) = 3/2 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3879_387968


namespace NUMINAMATH_CALUDE_intersection_point_l3879_387943

def f (x : ℝ) : ℝ := 4 * x - 2

theorem intersection_point :
  ∃ (x : ℝ), f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3879_387943


namespace NUMINAMATH_CALUDE_base_8_4512_equals_2378_l3879_387944

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 : 
  base_8_to_10 [2, 1, 5, 4] = 2378 := by sorry

end NUMINAMATH_CALUDE_base_8_4512_equals_2378_l3879_387944


namespace NUMINAMATH_CALUDE_feeding_to_total_ratio_l3879_387927

/-- Represents the time Larry spends on his dog in minutes -/
structure DogTime where
  walking_playing : ℕ  -- Time spent walking and playing (in minutes)
  total : ℕ           -- Total time spent on the dog (in minutes)

/-- The ratio of feeding time to total time is 1:6 -/
theorem feeding_to_total_ratio (t : DogTime) 
  (h1 : t.walking_playing = 30 * 2)
  (h2 : t.total = 72) : 
  (t.total - t.walking_playing) * 6 = t.total :=
by sorry

end NUMINAMATH_CALUDE_feeding_to_total_ratio_l3879_387927


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l3879_387911

theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l3879_387911


namespace NUMINAMATH_CALUDE_abs_sum_equality_l3879_387986

theorem abs_sum_equality (a b c : ℤ) (h : |a - b| + |c - a| = 1) :
  |a - c| + |c - b| + |b - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_equality_l3879_387986


namespace NUMINAMATH_CALUDE_job_completion_time_l3879_387923

theorem job_completion_time (total_work : ℝ) (time_together time_person2 : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_person2 > 0)
  (h3 : total_work > 0)
  (h4 : total_work / time_together = total_work / time_person2 + total_work / (24 : ℝ)) :
  total_work / (total_work / time_together - total_work / time_person2) = 24 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3879_387923


namespace NUMINAMATH_CALUDE_no_gcd_solution_l3879_387961

theorem no_gcd_solution : ¬∃ (a b c : ℕ), 
  (Nat.gcd a b = Nat.factorial 30 + 111) ∧ 
  (Nat.gcd b c = Nat.factorial 40 + 234) ∧ 
  (Nat.gcd c a = Nat.factorial 50 + 666) := by
sorry

end NUMINAMATH_CALUDE_no_gcd_solution_l3879_387961


namespace NUMINAMATH_CALUDE_sequence_p_bounded_l3879_387938

def isPrime (n : ℕ) : Prop := sorry

def largestPrimeDivisor (n : ℕ) : ℕ := sorry

def sequenceP : ℕ → ℕ
  | 0 => 2  -- Assuming the sequence starts with 2
  | 1 => 3  -- Assuming the second prime is 3
  | (n + 2) => largestPrimeDivisor (sequenceP (n + 1) + sequenceP n + 2008)

theorem sequence_p_bounded :
  ∃ (M : ℕ), ∀ (n : ℕ), sequenceP n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_p_bounded_l3879_387938


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3879_387990

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3879_387990


namespace NUMINAMATH_CALUDE_odd_floor_time_building_floor_time_l3879_387984

theorem odd_floor_time (total_floors : ℕ) (even_floor_time : ℕ) (total_time : ℕ) : ℕ :=
  let odd_floors := (total_floors + 1) / 2
  let even_floors := total_floors / 2
  let even_total_time := even_floors * even_floor_time
  let odd_total_time := total_time - even_total_time
  odd_total_time / odd_floors

/-- 
Given a building with 10 floors, where:
- It takes 15 seconds to reach each even-numbered floor
- It takes 120 seconds (2 minutes) to reach the 10th floor
Prove that it takes 9 seconds to reach each odd-numbered floor
-/
theorem building_floor_time : odd_floor_time 10 15 120 = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_floor_time_building_floor_time_l3879_387984


namespace NUMINAMATH_CALUDE_complex_number_property_l3879_387997

theorem complex_number_property (w : ℂ) (h : w + 1 / w = 2 * Real.cos (π / 4)) :
  w^12 + 1 / w^12 = -2 := by sorry

end NUMINAMATH_CALUDE_complex_number_property_l3879_387997


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3879_387941

theorem ping_pong_rackets_sold (total_sales : ℝ) (avg_price : ℝ) (h1 : total_sales = 735) (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3879_387941


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l3879_387908

def average_salary_jan_to_apr : ℕ := 8000
def salary_jan : ℕ := 6100
def salary_may : ℕ := 6500
def target_average : ℕ := 8100

theorem average_salary_feb_to_may :
  (4 * average_salary_jan_to_apr - salary_jan + salary_may) / 4 = target_average :=
sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l3879_387908


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3879_387983

def N : ℕ := 34 * 34 * 63 * 270

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors N) * 14 = sum_of_even_divisors N := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3879_387983


namespace NUMINAMATH_CALUDE_truck_profit_analysis_l3879_387959

def initial_cost : ℕ := 490000
def first_year_expense : ℕ := 60000
def annual_expense_increase : ℕ := 20000
def annual_income : ℕ := 250000

def profit_function (n : ℕ) : ℤ := -n^2 + 20*n - 49

def option1_sell_price : ℕ := 40000
def option2_sell_price : ℕ := 130000

theorem truck_profit_analysis :
  -- 1. Profit function
  (∀ n : ℕ, profit_function n = annual_income * n - (first_year_expense * n + (n * (n - 1) / 2) * annual_expense_increase) - initial_cost) ∧
  -- 2. Profit exceeds 150,000 in 5th year
  (profit_function 5 > 150 ∧ ∀ k < 5, profit_function k ≤ 150) ∧
  -- 3. Maximum profit at n = 10
  (∀ n : ℕ, profit_function n ≤ profit_function 10) ∧
  -- 4. Maximum average annual profit at n = 7
  (∀ n : ℕ, n ≠ 0 → profit_function n / n ≤ profit_function 7 / 7) ∧
  -- 5. Both options yield 550,000 total profit
  (profit_function 10 + option1_sell_price = 550000 ∧
   profit_function 7 + option2_sell_price = 550000) ∧
  -- 6. Option 2 is more time-efficient
  (7 < 10) :=
by sorry

end NUMINAMATH_CALUDE_truck_profit_analysis_l3879_387959


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3879_387960

theorem sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ a b : ℝ, a > b ∧ b > 0 → (b + m) / (a + m) > b / a) ∧
  (∃ a b : ℝ, (b + m) / (a + m) > b / a ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3879_387960


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l3879_387945

/-- The equation x^3 + y^3 + z^3 - 3xyz = 2003 has only three integer solutions. -/
theorem cube_sum_minus_product_eq_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(667, 668, 668), (668, 667, 668), (668, 668, 667)} := by
  sorry

#check cube_sum_minus_product_eq_2003

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l3879_387945


namespace NUMINAMATH_CALUDE_train_length_problem_l3879_387919

theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 36 →
  ∃ (train_length : ℝ), 
    train_length = 50 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_problem_l3879_387919


namespace NUMINAMATH_CALUDE_square_difference_l3879_387999

theorem square_difference (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3879_387999


namespace NUMINAMATH_CALUDE_mother_carrots_count_l3879_387917

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

/-- The number of carrots Haley's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - haley_carrots

theorem mother_carrots_count : mother_carrots = 38 := by
  sorry

end NUMINAMATH_CALUDE_mother_carrots_count_l3879_387917


namespace NUMINAMATH_CALUDE_cathy_worked_180_hours_l3879_387937

/-- Calculates the total hours worked by Cathy over 2 months, given the following conditions:
  * Normal work schedule is 20 hours per week
  * There are 4 weeks in a month
  * The job lasts for 2 months
  * Cathy covers an additional week of shifts (20 hours) due to Chris's illness
-/
def cathys_total_hours (hours_per_week : ℕ) (weeks_per_month : ℕ) (months : ℕ) (extra_week_hours : ℕ) : ℕ :=
  hours_per_week * weeks_per_month * months + extra_week_hours

/-- Proves that Cathy worked 180 hours during the 2 months -/
theorem cathy_worked_180_hours :
  cathys_total_hours 20 4 2 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cathy_worked_180_hours_l3879_387937


namespace NUMINAMATH_CALUDE_unique_three_digit_number_exists_l3879_387994

/-- Represents a 3-digit number abc as 100a + 10b + c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the number acb obtained by swapping the last two digits of abc -/
def swap_last_two_digits (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem unique_three_digit_number_exists :
  ∃! (a b c : ℕ),
    (100 ≤ three_digit_number a b c) ∧
    (three_digit_number a b c ≤ 999) ∧
    (1730 ≤ three_digit_number a b c + swap_last_two_digits a b c) ∧
    (three_digit_number a b c + swap_last_two_digits a b c ≤ 1739) ∧
    (three_digit_number a b c = 832) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_exists_l3879_387994


namespace NUMINAMATH_CALUDE_damage_cost_is_1450_l3879_387978

/-- Calculates the total cost of damages caused by Jack --/
def total_damage_cost (num_tires : ℕ) (cost_per_tire : ℕ) (window_cost : ℕ) : ℕ :=
  num_tires * cost_per_tire + window_cost

/-- Proves that the total cost of damages is $1450 --/
theorem damage_cost_is_1450 :
  total_damage_cost 3 250 700 = 1450 :=
by sorry

end NUMINAMATH_CALUDE_damage_cost_is_1450_l3879_387978


namespace NUMINAMATH_CALUDE_definite_integral_evaluation_l3879_387963

theorem definite_integral_evaluation :
  ∫ x in (1 : ℝ)..3, (2 * x - 1 / (x^2)) = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_evaluation_l3879_387963


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3879_387905

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 8 → 
  flavors * (toppings.choose 3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3879_387905


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3879_387956

theorem shaded_area_calculation (S T : ℝ) : 
  (16 / S = 4) → 
  (S / T = 4) → 
  (S^2 + 16 * T^2 = 32) := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3879_387956


namespace NUMINAMATH_CALUDE_construction_cost_l3879_387928

/-- The cost of hiring builders to construct houses -/
theorem construction_cost
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (pay_per_day : ℕ)
  (num_builders : ℕ)
  (num_houses : ℕ)
  (floors_per_house : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : pay_per_day = 100)
  (h4 : num_builders = 6)
  (h5 : num_houses = 5)
  (h6 : floors_per_house = 6) :
  (num_houses * floors_per_house * days_per_floor * pay_per_day * num_builders) / builders_per_floor = 270000 :=
by sorry

end NUMINAMATH_CALUDE_construction_cost_l3879_387928


namespace NUMINAMATH_CALUDE_min_value_theorem_l3879_387914

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) ∧
  4/a + 1/(b-1) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3879_387914


namespace NUMINAMATH_CALUDE_age_problem_l3879_387936

theorem age_problem :
  ∃ (x y z w v : ℕ),
    x + y + z = 74 ∧
    x = 7 * w ∧
    y = 2 * w + 2 * v ∧
    z = 2 * w + 3 * v ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ v > 0 ∧
    x = 28 ∧ y = 20 ∧ z = 26 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3879_387936


namespace NUMINAMATH_CALUDE_square_difference_representation_l3879_387940

theorem square_difference_representation (n : ℕ) :
  (∃ (a b : ℤ), n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_square_difference_representation_l3879_387940


namespace NUMINAMATH_CALUDE_chocolate_count_l3879_387979

/-- Represents the total number of chocolates in the jar -/
def total_chocolates : ℕ := 50

/-- Represents the number of chocolates that are not hazelnut -/
def not_hazelnut : ℕ := 12

/-- Represents the number of chocolates that are not liquor -/
def not_liquor : ℕ := 18

/-- Represents the number of chocolates that are not milk -/
def not_milk : ℕ := 20

/-- Theorem stating that the total number of chocolates is 50 -/
theorem chocolate_count :
  total_chocolates = 50 ∧
  not_hazelnut = 12 ∧
  not_liquor = 18 ∧
  not_milk = 20 ∧
  (total_chocolates - not_hazelnut) + (total_chocolates - not_liquor) + (total_chocolates - not_milk) = 2 * total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_count_l3879_387979


namespace NUMINAMATH_CALUDE_min_c_value_l3879_387952

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  ∀ c' : ℕ, (0 < c' ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < c' ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - c'|) → c' ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3879_387952


namespace NUMINAMATH_CALUDE_binary_10111_is_23_l3879_387930

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Nat := [1, 0, 1, 1, 1]

/-- Theorem stating that the decimal representation of (10111)₂ is 23 -/
theorem binary_10111_is_23 : binary_to_decimal binary_number = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_is_23_l3879_387930


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3879_387967

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 2 = 0) → 
  (b^3 - 2*b + 2 = 0) → 
  (c^3 - 2*c + 2 = 0) → 
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -1) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3879_387967


namespace NUMINAMATH_CALUDE_quadrant_line_conditions_l3879_387974

/-- A line passing through the first, third, and fourth quadrants -/
structure QuadrantLine where
  k : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem stating the conditions on k and b for a line passing through the first, third, and fourth quadrants -/
theorem quadrant_line_conditions (l : QuadrantLine) : l.k > 0 ∧ l.b < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_line_conditions_l3879_387974


namespace NUMINAMATH_CALUDE_school_trip_equation_correct_l3879_387953

/-- Represents the scenario of a school trip to Shaoshan -/
structure SchoolTrip where
  distance : ℝ
  delay : ℝ
  speedRatio : ℝ

/-- The equation representing the travel times for bus and car -/
def travelTimeEquation (trip : SchoolTrip) (x : ℝ) : Prop :=
  trip.distance / x = trip.distance / (trip.speedRatio * x) + trip.delay

/-- Theorem stating that the given equation correctly represents the scenario -/
theorem school_trip_equation_correct (x : ℝ) : 
  let trip : SchoolTrip := { 
    distance := 50,
    delay := 1/6,
    speedRatio := 1.2
  }
  travelTimeEquation trip x :=
by sorry

end NUMINAMATH_CALUDE_school_trip_equation_correct_l3879_387953


namespace NUMINAMATH_CALUDE_existence_of_abcd_l3879_387957

theorem existence_of_abcd (n : ℕ) (h : n > 1) : ∃ (a b c d : ℕ),
  a = 3*n - 1 ∧
  b = n + 1 ∧
  c = 3*n + 1 ∧
  d = n - 1 ∧
  a + b = 4*n ∧
  c + d = 4*n ∧
  a * b - c * d = 4*n :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_abcd_l3879_387957


namespace NUMINAMATH_CALUDE_speed_calculation_l3879_387996

/-- Proves that given a distance of 600 meters and a time of 5 minutes, the speed is 7.2 km/hour -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 5) :
  (distance / 1000) / (time / 60) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l3879_387996


namespace NUMINAMATH_CALUDE_michaels_additional_money_michael_needs_additional_money_l3879_387949

/-- Calculates the additional money Michael needs to buy all items for Mother's Day. -/
theorem michaels_additional_money (michael_money : ℝ) 
  (cake_price discount_cake : ℝ) (bouquet_price tax_bouquet : ℝ) 
  (balloons_price : ℝ) (perfume_price_gbp discount_perfume gbp_to_usd : ℝ) 
  (album_price_eur tax_album eur_to_usd : ℝ) : ℝ :=
  let cake_cost := cake_price * (1 - discount_cake)
  let bouquet_cost := bouquet_price * (1 + tax_bouquet)
  let balloons_cost := balloons_price
  let perfume_cost := perfume_price_gbp * (1 - discount_perfume) * gbp_to_usd
  let album_cost := album_price_eur * (1 + tax_album) * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost + album_cost
  total_cost - michael_money

/-- Proves that Michael needs an additional $78.90 to buy all items. -/
theorem michael_needs_additional_money :
  michaels_additional_money 50 20 0.1 36 0.05 5 30 0.15 1.4 25 0.08 1.2 = 78.9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_additional_money_michael_needs_additional_money_l3879_387949


namespace NUMINAMATH_CALUDE_miller_rabin_correct_for_primes_l3879_387951

/-- Miller-Rabin primality test function -/
def miller_rabin (n : ℕ) : Bool := sorry

/-- Definition of primality -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem miller_rabin_correct_for_primes (n : ℕ) (h : is_prime n) : 
  miller_rabin n = true := by sorry

end NUMINAMATH_CALUDE_miller_rabin_correct_for_primes_l3879_387951


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3879_387948

/-- Proves that the cost price is 1250 given the markup percentage and selling price -/
theorem cost_price_calculation (markup_percentage : ℝ) (selling_price : ℝ) : 
  markup_percentage = 60 →
  selling_price = 2000 →
  (100 + markup_percentage) / 100 * (selling_price / ((100 + markup_percentage) / 100)) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3879_387948


namespace NUMINAMATH_CALUDE_tan_negative_two_implies_fraction_l3879_387991

theorem tan_negative_two_implies_fraction (θ : Real) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_two_implies_fraction_l3879_387991


namespace NUMINAMATH_CALUDE_cos_4theta_value_l3879_387915

/-- If e^(iθ) = (3 - i√2) / 4, then cos 4θ = 121/256 -/
theorem cos_4theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 - Complex.I * Real.sqrt 2) / 4) : 
  Real.cos (4 * θ) = 121 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_value_l3879_387915


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l3879_387954

/-- Represents the number of bottles of each juice type -/
structure JuiceBottles where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- Represents the cost in cents of each juice type -/
structure JuiceCosts where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- The main theorem to prove -/
theorem orange_juice_bottles (b : JuiceBottles) (c : JuiceCosts) : 
  c.orange = 70 ∧ 
  c.apple = 60 ∧ 
  c.grape = 80 ∧ 
  b.orange + b.apple + b.grape = 100 ∧ 
  c.orange * b.orange + c.apple * b.apple + c.grape * b.grape = 7250 ∧
  b.apple = b.grape ∧
  b.orange = 2 * b.apple →
  b.orange = 50 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l3879_387954


namespace NUMINAMATH_CALUDE_solution_set_F_max_value_F_inequality_holds_l3879_387975

-- Define the function F(x) = |x + 2| - 3|x|
def F (x : ℝ) : ℝ := |x + 2| - 3 * |x|

-- Theorem 1: The solution set of F(x) ≥ 0 is {x | -1/2 ≤ x ≤ 1}
theorem solution_set_F : 
  {x : ℝ | F x ≥ 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: The maximum value of F(x) is 2
theorem max_value_F : 
  ∃ (x : ℝ), F x = 2 ∧ ∀ (y : ℝ), F y ≤ 2 := by sorry

-- Corollary: The inequality F(x) ≥ a holds for all a ∈ (-∞, 2]
theorem inequality_holds :
  ∀ (a : ℝ), a ≤ 2 → ∃ (x : ℝ), F x ≥ a := by sorry

end NUMINAMATH_CALUDE_solution_set_F_max_value_F_inequality_holds_l3879_387975


namespace NUMINAMATH_CALUDE_at_least_n_minus_two_have_real_root_l3879_387932

/-- A linear function of the form ax + b where a ≠ 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The product of all LinearFunctions except the i-th one -/
def productExcept (funcs : List LinearFunction) (i : Nat) : LinearFunction → LinearFunction :=
  sorry

/-- The polynomial formed by the sum of the product of n-1 functions and the remaining function -/
def formPolynomial (funcs : List LinearFunction) (i : Nat) : LinearFunction :=
  sorry

/-- A function has a real root if there exists a real number x such that f(x) = 0 -/
def hasRealRoot (f : LinearFunction) : Prop :=
  ∃ x : ℝ, f.a * x + f.b = 0

/-- The main theorem -/
theorem at_least_n_minus_two_have_real_root (funcs : List LinearFunction) :
  funcs.length ≥ 3 →
  ∃ (roots : List LinearFunction),
    roots.length ≥ funcs.length - 2 ∧
    ∀ f ∈ roots, ∃ i, f = formPolynomial funcs i ∧ hasRealRoot f :=
  sorry

end NUMINAMATH_CALUDE_at_least_n_minus_two_have_real_root_l3879_387932


namespace NUMINAMATH_CALUDE_f_has_local_minimum_in_interval_l3879_387950

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem f_has_local_minimum_in_interval :
  ∃ x₀ : ℝ, 1/2 < x₀ ∧ x₀ < 1 ∧ IsLocalMin f x₀ := by sorry

end NUMINAMATH_CALUDE_f_has_local_minimum_in_interval_l3879_387950


namespace NUMINAMATH_CALUDE_equation_solution_l3879_387924

theorem equation_solution :
  ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3879_387924


namespace NUMINAMATH_CALUDE_ratio_and_linear_equation_l3879_387942

theorem ratio_and_linear_equation (c d : ℝ) : 
  c / d = 4 → c = 20 - 6 * d → d = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_and_linear_equation_l3879_387942


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3879_387920

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3879_387920


namespace NUMINAMATH_CALUDE_tournament_ceremony_theorem_l3879_387955

def tournament_ceremony_length (initial_players : Nat) (initial_ceremony_length : Nat) (ceremony_increase : Nat) : Nat :=
  let rounds := Nat.log2 initial_players
  let ceremony_lengths := List.range rounds |>.map (λ i => initial_ceremony_length + i * ceremony_increase)
  let winners_per_round := List.range rounds |>.map (λ i => initial_players / (2^(i+1)))
  List.sum (List.zipWith (·*·) ceremony_lengths winners_per_round)

theorem tournament_ceremony_theorem :
  tournament_ceremony_length 16 10 10 = 260 := by
  sorry

end NUMINAMATH_CALUDE_tournament_ceremony_theorem_l3879_387955


namespace NUMINAMATH_CALUDE_expenditure_ratio_l3879_387918

/-- Given a person's income and savings pattern over two years, prove the ratio of total expenditure to first year expenditure --/
theorem expenditure_ratio (income : ℝ) (h1 : income > 0) : 
  let first_year_savings := 0.25 * income
  let first_year_expenditure := income - first_year_savings
  let second_year_income := 1.25 * income
  let second_year_savings := 2 * first_year_savings
  let second_year_expenditure := second_year_income - second_year_savings
  let total_expenditure := first_year_expenditure + second_year_expenditure
  (total_expenditure / first_year_expenditure) = 2 := by
  sorry


end NUMINAMATH_CALUDE_expenditure_ratio_l3879_387918


namespace NUMINAMATH_CALUDE_triangle_sides_l3879_387907

theorem triangle_sides (average_length : ℝ) (perimeter : ℝ) (n : ℕ) :
  average_length = 12 →
  perimeter = 36 →
  average_length * n = perimeter →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_l3879_387907


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3879_387966

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence with positive terms, 
    if a₂ = 1 - a₁ and a₄ = 9 - a₃, then a₄ + a₅ = 27 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.a 2 = 1 - seq.a 1)
  (h2 : seq.a 4 = 9 - seq.a 3) :
  seq.a 4 + seq.a 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3879_387966


namespace NUMINAMATH_CALUDE_ellipse_C_equation_l3879_387931

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 5 - y^2 / 4 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci and vertices of ellipse C
def ellipse_C_foci (x y : ℝ) : Prop := (x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0)
def ellipse_C_vertices (x y : ℝ) : Prop := (x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0)

-- Theorem statement
theorem ellipse_C_equation :
  (∀ x y : ℝ, hyperbola x y → 
    ((x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0) → ellipse_C_vertices x y) ∧
    ((x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0) → ellipse_C_foci x y)) →
  (∀ x y : ℝ, ellipse_C_foci x y ∨ ellipse_C_vertices x y → ellipse_C x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_l3879_387931


namespace NUMINAMATH_CALUDE_dusty_cake_purchase_l3879_387988

/-- The number of double layer cake slices Dusty bought -/
def double_layer_slices : ℕ := sorry

/-- The price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- The price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- The number of single layer cake slices Dusty bought -/
def single_layer_bought : ℕ := 7

/-- The amount Dusty paid in dollars -/
def amount_paid : ℕ := 100

/-- The change Dusty received in dollars -/
def change_received : ℕ := 37

theorem dusty_cake_purchase : 
  double_layer_slices = 5 ∧
  amount_paid = 
    single_layer_price * single_layer_bought + 
    double_layer_price * double_layer_slices + 
    change_received :=
by sorry

end NUMINAMATH_CALUDE_dusty_cake_purchase_l3879_387988


namespace NUMINAMATH_CALUDE_certain_number_proof_l3879_387971

theorem certain_number_proof (x : ℕ) : x > 72 ∧ x ∣ (72 * 14) ∧ (∀ y : ℕ, y > 72 ∧ y ∣ (72 * 14) → x ≤ y) → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3879_387971


namespace NUMINAMATH_CALUDE_debby_water_bottles_l3879_387926

/-- Given the initial number of water bottles, daily consumption, and remaining bottles,
    calculate the number of days Debby drank water bottles. -/
theorem debby_water_bottles (initial : ℕ) (daily : ℕ) (remaining : ℕ) :
  initial = 264 →
  daily = 15 →
  remaining = 99 →
  (initial - remaining) / daily = 11 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l3879_387926


namespace NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3879_387998

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : IntegerParallelogram :=
  { v1 := (2, 3)
    v2 := (5, 7)
    v3 := (11, 7)
    v4 := (8, 3) }

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  sorry

/-- Calculate the area of the parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) :
  p = specificParallelogram → perimeter p + area p = 46 :=
sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3879_387998


namespace NUMINAMATH_CALUDE_radical_product_simplification_l3879_387933

theorem radical_product_simplification (p : ℝ) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p^2) * Real.sqrt (30 * p^5) = 30 * p^5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l3879_387933


namespace NUMINAMATH_CALUDE_breakfast_customers_count_l3879_387939

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- Theorem stating that the number of customers during breakfast on Friday is 73 -/
theorem breakfast_customers_count : 
  breakfast_customers = 
    saturday_prediction / 2 - (lunch_customers + dinner_customers) :=
by
  sorry

#check breakfast_customers_count

end NUMINAMATH_CALUDE_breakfast_customers_count_l3879_387939


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3879_387934

theorem expand_and_simplify (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3879_387934


namespace NUMINAMATH_CALUDE_runner_picture_probability_l3879_387970

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the state of the race at a given time -/
def RaceState :=
  ℕ  -- time in seconds

/-- Represents the camera setup -/
structure Camera where
  coverageFraction : ℚ
  centerPosition : ℚ  -- fraction of track from start line

/-- Calculate the position of a runner at a given time -/
def runnerPosition (r : Runner) (t : ℕ) : ℚ :=
  sorry

/-- Check if a runner is in the camera's view -/
def isInPicture (r : Runner) (t : ℕ) (c : Camera) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem runner_picture_probability :
  let alice : Runner := ⟨"Alice", 120, true⟩
  let ben : Runner := ⟨"Ben", 100, false⟩
  let camera : Camera := ⟨1/3, 0⟩
  let raceTime : ℕ := 900
  let totalOverlapTime : ℚ := 40/3
  (totalOverlapTime / 60 : ℚ) = 1333/6000 := by
  sorry

end NUMINAMATH_CALUDE_runner_picture_probability_l3879_387970


namespace NUMINAMATH_CALUDE_division_problem_l3879_387969

theorem division_problem : (96 : ℚ) / ((8 : ℚ) / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3879_387969


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3879_387922

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x ≠ 1 ∧ x ≠ 2) ↔ x^2 - 3*x + 2 = 0) ↔
  (x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3879_387922


namespace NUMINAMATH_CALUDE_union_equals_real_when_m_is_one_sufficient_necessary_condition_l3879_387995

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - m - 1) ≥ 0}

theorem union_equals_real_when_m_is_one :
  A ∪ B 1 = Set.univ := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≤ -2 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_equals_real_when_m_is_one_sufficient_necessary_condition_l3879_387995


namespace NUMINAMATH_CALUDE_pocket_balls_theorem_l3879_387976

/-- Represents the number of balls in each pocket -/
def pocket_balls : List Nat := [2, 4, 5]

/-- The total number of ways to take a ball from any pocket -/
def total_ways_one_ball : Nat := pocket_balls.sum

/-- The total number of ways to take one ball from each pocket -/
def total_ways_three_balls : Nat := pocket_balls.prod

theorem pocket_balls_theorem :
  total_ways_one_ball = 11 ∧ total_ways_three_balls = 40 := by
  sorry

end NUMINAMATH_CALUDE_pocket_balls_theorem_l3879_387976


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_l3879_387982

theorem quadratic_roots_integer (p : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x + p + 4 = 0 ∧ y^2 + p*y + p + 4 = 0) →
  p = 8 ∨ p = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_l3879_387982


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3879_387972

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 12 + a 13 = 24) : 
  a 7 = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3879_387972


namespace NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l3879_387947

theorem six_digit_divisibility_by_seven (a b c d e f : Nat) 
  (h1 : a ≥ 1 ∧ a ≤ 9)  -- Ensure it's a six-digit number
  (h2 : b ≥ 0 ∧ b ≤ 9)
  (h3 : c ≥ 0 ∧ c ≤ 9)
  (h4 : d ≥ 0 ∧ d ≤ 9)
  (h5 : e ≥ 0 ∧ e ≤ 9)
  (h6 : f ≥ 0 ∧ f ≤ 9)
  (h7 : (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≡ 0 [MOD 7]) :
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ≡ 0 [MOD 7] := by
  sorry

#check six_digit_divisibility_by_seven

end NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l3879_387947


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3879_387909

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3879_387909


namespace NUMINAMATH_CALUDE_trig_inequality_l3879_387912

open Real

theorem trig_inequality (a b c d : ℝ) : 
  a = sin (sin (2009 * π / 180)) →
  b = sin (cos (2009 * π / 180)) →
  c = cos (sin (2009 * π / 180)) →
  d = cos (cos (2009 * π / 180)) →
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l3879_387912


namespace NUMINAMATH_CALUDE_every_multiple_of_2_is_even_is_universal_l3879_387958

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be a multiple of 2
def MultipleOf2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what a universal proposition is
def UniversalProposition (P : ℤ → Prop) : Prop :=
  ∀ x : ℤ, P x

-- Statement to prove
theorem every_multiple_of_2_is_even_is_universal :
  UniversalProposition (λ n : ℤ => MultipleOf2 n → IsEven n) :=
sorry

end NUMINAMATH_CALUDE_every_multiple_of_2_is_even_is_universal_l3879_387958


namespace NUMINAMATH_CALUDE_additional_driving_hours_l3879_387916

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ := 2

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The total number of hours Carl drives in two weeks after promotion -/
def total_hours_two_weeks : ℝ := 40

/-- The number of weeks in the given period -/
def num_weeks : ℝ := 2

theorem additional_driving_hours :
  let hours_before := hours_per_day * days_per_week
  let hours_after := total_hours_two_weeks / num_weeks
  hours_after - hours_before = 6 := by sorry

end NUMINAMATH_CALUDE_additional_driving_hours_l3879_387916


namespace NUMINAMATH_CALUDE_equation_solution_l3879_387993

theorem equation_solution : 
  ∃ x : ℝ, (3 * x^2 + 6 = |(-25 + x)|) ∧ 
  (x = (-1 + Real.sqrt 229) / 6 ∨ x = (-1 - Real.sqrt 229) / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3879_387993


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3879_387973

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 5 ∧
  n % 11 = 5 ∧
  n % 13 = 5 ∧
  n % 17 = 5 ∧
  n % 23 = 5 ∧
  n % 19 = 0 ∧
  ∀ m : ℕ, m > 0 →
    m % 7 = 5 →
    m % 11 = 5 →
    m % 13 = 5 →
    m % 17 = 5 →
    m % 23 = 5 →
    m % 19 = 0 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3879_387973


namespace NUMINAMATH_CALUDE_tangent_inequality_tan_pi_12_l3879_387989

theorem tangent_inequality (a : Fin 13 → ℝ) (h : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ 0 < (a i - a j) / (1 + a i * a j) ∧ 
    (a i - a j) / (1 + a i * a j) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

theorem tan_pi_12 : Real.tan (π / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_inequality_tan_pi_12_l3879_387989


namespace NUMINAMATH_CALUDE_perimeter_of_square_with_semicircular_arcs_l3879_387929

/-- The perimeter of a region bounded by four semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_of_square_with_semicircular_arcs (π : ℝ) (h : π > 0) : 
  let side_length : ℝ := 1 / π
  let semicircle_length : ℝ := π * side_length / 2
  let num_semicircles : ℕ := 4
  num_semicircles * semicircle_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_with_semicircular_arcs_l3879_387929


namespace NUMINAMATH_CALUDE_expression_evaluation_l3879_387910

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z) :
  (x^3 - 1/x^3) * (z^3 + 1/z^3) = x^6 - 1/x^6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3879_387910


namespace NUMINAMATH_CALUDE_pat_stickers_l3879_387987

theorem pat_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l3879_387987


namespace NUMINAMATH_CALUDE_tangency_condition_l3879_387962

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 4

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x', y') = (x, y)

/-- The theorem statement -/
theorem tangency_condition (m : ℝ) :
  are_tangent m ↔ m = 8 + 4 * Real.sqrt 3 ∨ m = 8 - 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tangency_condition_l3879_387962


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3879_387902

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- The given circle equation -/
def givenCircle : Circle :=
  { equation := fun x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem circle_center_and_radius :
  Circle.center givenCircle = (1, 0) ∧ Circle.radius givenCircle = 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3879_387902


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3879_387921

theorem sin_2theta_value (θ : Real) (h : (Real.sqrt 2 * Real.cos (2 * θ)) / Real.cos (π / 4 + θ) = Real.sqrt 3 * Real.sin (2 * θ)) : 
  Real.sin (2 * θ) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3879_387921


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3879_387906

theorem quadratic_roots_sum (p : ℝ) : 
  (∃ x y : ℝ, x * y = 9 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) →
  (∃ x y : ℝ, x + y = 7 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3879_387906


namespace NUMINAMATH_CALUDE_line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l3879_387900

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (angle : Line → Line → ℝ)
variable (angle_line_plane : Line → Plane → ℝ)
variable (dihedral_angle : Plane → Plane → ℝ)

-- Theorem statements
theorem line_relationships (l1 l2 : Line) : 
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ intersect l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ skew l1 l2) ∧ 
  ¬(intersect l1 l2 ∧ skew l1 l2) := sorry

theorem skew_lines_angle_range (l1 l2 : Line) (h : skew l1 l2) : 
  0 < angle l1 l2 ∧ angle l1 l2 ≤ Real.pi / 2 := sorry

theorem line_plane_angle_range (l : Line) (p : Plane) : 
  0 ≤ angle_line_plane l p ∧ angle_line_plane l p ≤ Real.pi / 2 := sorry

theorem dihedral_angle_range (p1 p2 : Plane) : 
  0 ≤ dihedral_angle p1 p2 ∧ dihedral_angle p1 p2 ≤ Real.pi := sorry

theorem parallel_line_plane (a b : Line) (α : Plane) 
  (h1 : ¬contained_in a α) (h2 : contained_in b α) (h3 : parallel a b) : 
  parallel_plane a α := sorry

theorem parallel_planes (a b : Line) (α β : Plane) (P : Point)
  (h1 : contained_in a β) (h2 : contained_in b β) 
  (h3 : intersect a b) (h4 : ¬parallel_plane a α) (h5 : ¬parallel_plane b α) : 
  planes_parallel α β := sorry

theorem perpendicular_line_plane (a b l : Line) (α : Plane) (A : Point)
  (h1 : contained_in a α) (h2 : contained_in b α) 
  (h3 : intersect a b) (h4 : perpendicular l a) (h5 : perpendicular l b) : 
  perpendicular_plane l α := sorry

theorem perpendicular_planes (l : Line) (α β : Plane)
  (h1 : perpendicular_plane l α) (h2 : contained_in l β) : 
  planes_perpendicular α β := sorry

end NUMINAMATH_CALUDE_line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l3879_387900


namespace NUMINAMATH_CALUDE_no_integer_roots_l3879_387977

theorem no_integer_roots : ∀ (x : ℤ), x^3 - 3*x^2 - 10*x + 20 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3879_387977


namespace NUMINAMATH_CALUDE_even_function_monotonicity_l3879_387981

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define monotonically decreasing in an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

-- Define monotonically increasing in an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Theorem statement
theorem even_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_decreasing : monotone_decreasing_on f (-2) (-1)) :
  monotone_increasing_on f 1 2 ∧ 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f 1 ≤ f x) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ f 2) :=
by sorry

end NUMINAMATH_CALUDE_even_function_monotonicity_l3879_387981


namespace NUMINAMATH_CALUDE_triangle_area_l3879_387985

/-- Given a triangle with sides AC, BC, and BD, prove that its area is 14 -/
theorem triangle_area (AC BC BD : ℝ) (h1 : AC = 4) (h2 : BC = 3) (h3 : BD = 10) :
  (1 / 2 : ℝ) * (BD - BC) * AC = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3879_387985


namespace NUMINAMATH_CALUDE_problem_solution_l3879_387913

theorem problem_solution (a b m n : ℝ) : 
  (a = -(-(3 : ℝ))) → 
  (b = (-((1 : ℝ)/(2 : ℝ)))⁻¹) → 
  (|m - a| + |n + b| = 0) → 
  (m = 3 ∧ n = -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3879_387913


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l3879_387992

/-- The value of m for which the parabola y = x^2 + 4 is tangent to the hyperbola y^2 - mx^2 = 4 -/
def tangency_value : ℝ := 8

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 4

/-- Theorem stating that the parabola is tangent to the hyperbola if and only if m = 8 -/
theorem parabola_tangent_hyperbola :
  ∀ (m : ℝ), (∃ (x : ℝ), hyperbola m x (parabola x) ∧
    ∀ (x' : ℝ), x' ≠ x → ¬(hyperbola m x' (parabola x'))) ↔ m = tangency_value :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l3879_387992


namespace NUMINAMATH_CALUDE_ice_pop_cost_l3879_387980

theorem ice_pop_cost (ice_pop_price : ℝ) (pencil_price : ℝ) (ice_pops_sold : ℕ) (pencils_bought : ℕ) : 
  ice_pop_price = 1.50 →
  pencil_price = 1.80 →
  ice_pops_sold = 300 →
  pencils_bought = 100 →
  ice_pops_sold * ice_pop_price = pencils_bought * pencil_price →
  ice_pop_price - (ice_pops_sold * ice_pop_price - pencils_bought * pencil_price) / ice_pops_sold = 0.90 := by
sorry

end NUMINAMATH_CALUDE_ice_pop_cost_l3879_387980


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3879_387901

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + 6 = (x - 2)*(x + n)) → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3879_387901


namespace NUMINAMATH_CALUDE_pushup_difference_l3879_387903

-- Define the number of push-ups for Zachary and the total
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- Define David's push-ups
def david_pushups : ℕ := total_pushups - zachary_pushups

-- State the theorem
theorem pushup_difference :
  david_pushups > zachary_pushups ∧
  david_pushups - zachary_pushups = 58 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l3879_387903


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3879_387925

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 → 
  area = side * side → 
  perimeter = 4 * side → 
  perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3879_387925


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_sum_of_digits_l3879_387904

theorem bernardo_silvia_game (N : ℕ) : N = 38 ↔ 
  (27 * N + 900 < 2000) ∧ 
  (27 * N + 900 ≥ 1925) ∧ 
  (∀ k : ℕ, k < N → (27 * k + 900 < 1925 ∨ 27 * k + 900 ≥ 2000)) :=
sorry

theorem sum_of_digits (N : ℕ) : N = 38 → (N % 10 + N / 10) = 11 :=
sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_sum_of_digits_l3879_387904


namespace NUMINAMATH_CALUDE_specific_path_count_l3879_387946

/-- The number of paths on a grid with given dimensions and constraints -/
def numPaths (width height diagonalSteps : ℕ) : ℕ :=
  Nat.choose (width + height - diagonalSteps) diagonalSteps *
  Nat.choose (width + height - 2 * diagonalSteps) height

/-- Theorem stating the number of paths for the specific problem -/
theorem specific_path_count :
  numPaths 7 6 2 = 6930 := by
  sorry

end NUMINAMATH_CALUDE_specific_path_count_l3879_387946


namespace NUMINAMATH_CALUDE_school_trip_buses_l3879_387965

/-- The number of buses needed for a school trip -/
def buses_needed (students : ℕ) (seats_per_bus : ℕ) : ℕ :=
  (students + seats_per_bus - 1) / seats_per_bus

/-- Proof that 5 buses are needed for 45 students with 9 seats per bus -/
theorem school_trip_buses :
  buses_needed 45 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_buses_l3879_387965


namespace NUMINAMATH_CALUDE_average_age_proof_l3879_387935

def average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) : ℚ :=
  let total_age := initial_people * initial_average
  let remaining_age := total_age - (leaving_age1 + leaving_age2)
  let remaining_people := initial_people - 2
  remaining_age / remaining_people

theorem average_age_proof :
  average_age_after_leaving 7 28 22 25 = 29.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l3879_387935
