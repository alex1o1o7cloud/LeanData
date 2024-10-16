import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l1554_155462

theorem problem_statement (x y : ℝ) (h : 3 * x^2 + 3 * y^2 - 2 * x * y = 5) :
  (x + y ≥ -Real.sqrt 5) ∧
  (x^2 + y^2 ≥ 5/4) ∧
  (x - y/3 ≥ -Real.sqrt 15 / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1554_155462


namespace NUMINAMATH_CALUDE_no_real_roots_l1554_155427

theorem no_real_roots (a : ℝ) : (∀ x : ℝ, |x| ≠ a * x + 1) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1554_155427


namespace NUMINAMATH_CALUDE_rate_of_profit_l1554_155495

/-- Calculate the rate of profit given the cost price and selling price -/
theorem rate_of_profit (cost_price selling_price : ℕ) : 
  cost_price = 50 → selling_price = 60 → 
  (selling_price - cost_price) * 100 / cost_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_profit_l1554_155495


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1554_155494

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * 9^x - 3^x + a^2 - a - 3 > 0) → 
  (a > 2 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1554_155494


namespace NUMINAMATH_CALUDE_duration_is_twelve_hours_l1554_155451

/-- Calculates the duration of a population change period given birth rate, death rate, and total net increase -/
def calculate_duration (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let net_rate_per_second := (birth_rate - death_rate) / 2
  let duration_seconds := net_increase / net_rate_per_second
  duration_seconds / 3600

/-- Theorem stating that given the specified birth rate, death rate, and net increase, the duration is 12 hours -/
theorem duration_is_twelve_hours :
  calculate_duration (7 : ℚ) (3 : ℚ) 172800 = 12 := by
  sorry

#eval calculate_duration (7 : ℚ) (3 : ℚ) 172800

end NUMINAMATH_CALUDE_duration_is_twelve_hours_l1554_155451


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1554_155444

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1554_155444


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l1554_155443

theorem tan_sum_product_equals_one :
  ∀ (a b : ℝ),
  a + b = Real.pi / 4 →
  Real.tan (Real.pi / 4) = 1 →
  Real.tan a + Real.tan b + Real.tan a * Real.tan b = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l1554_155443


namespace NUMINAMATH_CALUDE_max_value_sqrt7_plus_2xy_l1554_155475

theorem max_value_sqrt7_plus_2xy (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) 
  (h3 : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  ∃ (M : ℝ), M = 16 ∧ ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32 → Real.sqrt 7*(x + 2*y) + 2*x*y ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt7_plus_2xy_l1554_155475


namespace NUMINAMATH_CALUDE_initial_fee_is_correct_l1554_155465

/-- The initial fee for a taxi trip -/
def initial_fee : ℝ := 2.25

/-- The charge per 2/5 of a mile -/
def charge_per_two_fifths_mile : ℝ := 0.15

/-- The length of the trip in miles -/
def trip_length : ℝ := 3.6

/-- The total charge for the trip -/
def total_charge : ℝ := 3.60

/-- Theorem stating that the initial fee is correct given the conditions -/
theorem initial_fee_is_correct :
  initial_fee + (trip_length * (charge_per_two_fifths_mile * 5 / 2)) = total_charge :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_is_correct_l1554_155465


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l1554_155440

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance (c : Cyclist) (time : ℝ) : ℝ := c.speed * time

theorem cyclist_distance_difference 
  (clara : Cyclist) 
  (david : Cyclist) 
  (h1 : clara.speed = 14.4) 
  (h2 : david.speed = 10.8) 
  (time : ℝ) 
  (h3 : time = 5) : 
  distance clara time - distance david time = 18 := by
sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l1554_155440


namespace NUMINAMATH_CALUDE_remainder_problem_l1554_155407

theorem remainder_problem (n : ℤ) : n % 5 = 3 → (4 * n + 6) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1554_155407


namespace NUMINAMATH_CALUDE_certain_number_problem_l1554_155473

theorem certain_number_problem (x : ℝ) : x * 11 = 99 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1554_155473


namespace NUMINAMATH_CALUDE_shortest_distance_is_eight_fifths_l1554_155467

/-- Square ABCD with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 2)

/-- Circular arc with center A from B to D -/
structure CircularArc (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = sq.A ∧ start_point = sq.B ∧ end_point = sq.D)

/-- Semicircle with center at midpoint of CD, from C to D -/
structure Semicircle (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = ((sq.C.1 + sq.D.1) / 2, (sq.C.2 + sq.D.2) / 2) ∧ 
              start_point = sq.C ∧ end_point = sq.D)

/-- Intersection point of the circular arc and semicircle -/
def intersectionPoint (sq : Square) (arc : CircularArc sq) (semi : Semicircle sq) : ℝ × ℝ := sorry

/-- Shortest distance from a point to a line segment -/
def shortestDistance (point : ℝ × ℝ) (segment_start : ℝ × ℝ) (segment_end : ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The shortest distance from the intersection point to AD is 8/5 -/
theorem shortest_distance_is_eight_fifths (sq : Square) 
  (arc : CircularArc sq) (semi : Semicircle sq) :
  shortestDistance (intersectionPoint sq arc semi) sq.A sq.D = 8/5 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_is_eight_fifths_l1554_155467


namespace NUMINAMATH_CALUDE_eight_towns_distances_l1554_155458

/-- The number of unique distances needed to connect n towns -/
def uniqueDistances (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: For 8 towns, the number of unique distances is 28 -/
theorem eight_towns_distances : uniqueDistances 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_towns_distances_l1554_155458


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1554_155483

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 4

/-- 
The number of ways to distribute indistinguishable objects into distinguishable categories
n: number of objects
k: number of categories
-/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of distinct ice cream flavors that can be created -/
def total_flavors : ℕ := stars_and_bars num_scoops num_flavors

theorem ice_cream_flavors_count : total_flavors = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1554_155483


namespace NUMINAMATH_CALUDE_fraction_equality_l1554_155433

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1554_155433


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1554_155459

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℤ) - 5 = 11^y → x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1554_155459


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l1554_155497

def carrots_to_pounds (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℚ :=
  (bed1 + bed2 + bed3) / carrots_per_pound

theorem kelly_carrot_harvest (bed1 bed2 bed3 carrots_per_pound : ℕ) :
  carrots_to_pounds bed1 bed2 bed3 carrots_per_pound =
  (bed1 + bed2 + bed3) / carrots_per_pound :=
by
  sorry

#eval carrots_to_pounds 55 101 78 6

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l1554_155497


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l1554_155448

/-- The cost per trip to an amusement park given the following conditions:
  - Two season passes are purchased
  - Each pass costs 100 units of currency
  - One person uses their pass 35 times
  - Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) : 
  pass_cost = 100 ∧ 
  trips_person1 = 35 ∧ 
  trips_person2 = 15 → 
  (2 * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l1554_155448


namespace NUMINAMATH_CALUDE_fourth_equation_in_sequence_l1554_155470

/-- Given a sequence of equations, prove that the fourth equation follows the pattern. -/
theorem fourth_equation_in_sequence : 
  (3^2 + 4^2 = 5^2) → 
  (10^2 + 11^2 + 12^2 = 13^2 + 14^2) → 
  (21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2) → 
  (36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2) := by
  sorry

end NUMINAMATH_CALUDE_fourth_equation_in_sequence_l1554_155470


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1554_155431

theorem banana_orange_equivalence (banana_value orange_value : ℚ) : 
  (3 / 4 : ℚ) * 12 * banana_value = 6 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 4 * orange_value := by
sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1554_155431


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_l1554_155482

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop :=
  n % m = 0

theorem least_four_digit_divisible :
  ∀ n : ℕ,
    1000 ≤ n                          -- four-digit number
    → n < 10000                       -- four-digit number
    → has_different_digits n          -- all digits are different
    → is_divisible_by n 1             -- divisible by 1
    → is_divisible_by n 2             -- divisible by 2
    → is_divisible_by n 4             -- divisible by 4
    → is_divisible_by n 8             -- divisible by 8
    → 1248 ≤ n                        -- 1248 is the least such number
  := by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_l1554_155482


namespace NUMINAMATH_CALUDE_investment_amounts_l1554_155412

/-- Represents the investment and profit ratios for three investors over two years -/
structure InvestmentData where
  p_investment_year1 : ℚ
  p_investment_year2 : ℚ
  profit_ratio_year1 : Fin 3 → ℚ
  profit_ratio_year2 : Fin 3 → ℚ

/-- Calculates the investments of q and r based on the given data -/
def calculate_investments (data : InvestmentData) : ℚ × ℚ :=
  let q_investment := (data.profit_ratio_year1 1 / data.profit_ratio_year1 0) * data.p_investment_year1
  let r_investment := (data.profit_ratio_year1 2 / data.profit_ratio_year1 0) * data.p_investment_year1
  (q_investment, r_investment)

/-- The main theorem stating the investment amounts for q and r -/
theorem investment_amounts (data : InvestmentData)
  (h1 : data.p_investment_year1 = 52000)
  (h2 : data.p_investment_year2 = 62400)
  (h3 : data.profit_ratio_year1 = ![4, 5, 6])
  (h4 : data.profit_ratio_year2 = ![3, 4, 5])
  (h5 : data.p_investment_year2 = data.p_investment_year1 * (1 + 1/5)) :
  calculate_investments data = (65000, 78000) := by
  sorry

end NUMINAMATH_CALUDE_investment_amounts_l1554_155412


namespace NUMINAMATH_CALUDE_trailingZeros_30_factorial_l1554_155478

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ := 
  let factN := factorial n
  (Nat.digits 10 factN).reverse.takeWhile (·= 0) |>.length

theorem trailingZeros_30_factorial : trailingZeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_trailingZeros_30_factorial_l1554_155478


namespace NUMINAMATH_CALUDE_unique_linear_equation_solution_l1554_155464

/-- A linear equation of the form y = kx + b -/
structure LinearEquation where
  k : ℝ
  b : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the linear equation -/
def satisfiesEquation (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.k * p.x + eq.b

theorem unique_linear_equation_solution (eq : LinearEquation) :
  satisfiesEquation ⟨1, 1⟩ eq → satisfiesEquation ⟨2, 3⟩ eq → eq.k = 2 ∧ eq.b = -1 := by
  sorry

#check unique_linear_equation_solution

end NUMINAMATH_CALUDE_unique_linear_equation_solution_l1554_155464


namespace NUMINAMATH_CALUDE_second_bus_ride_time_l1554_155441

theorem second_bus_ride_time (waiting_time first_bus_time : ℕ) 
  (h1 : waiting_time = 12)
  (h2 : first_bus_time = 30)
  (h3 : ∀ x, x = (waiting_time + first_bus_time) / 2 → x = 21) :
  ∃ second_bus_time : ℕ, second_bus_time = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_second_bus_ride_time_l1554_155441


namespace NUMINAMATH_CALUDE_quadratic_function_conditions_l1554_155418

/-- A quadratic function passing through (1, -4) with vertex at (-1, 0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ (∀ x : ℝ, f x ≥ f (-1)) ∧ (f (-1) = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_conditions_l1554_155418


namespace NUMINAMATH_CALUDE_expression_simplification_l1554_155460

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1554_155460


namespace NUMINAMATH_CALUDE_lcm_factor_not_unique_l1554_155425

/-- Given two positive integers with HCF 52 and larger number 624, 
    the other factor of their LCM cannot be uniquely determined. -/
theorem lcm_factor_not_unique (A B : ℕ+) : 
  (Nat.gcd A B = 52) → 
  (max A B = 624) → 
  ∃ (y : ℕ+), y ≠ 1 ∧ 
    ∃ (lcm : ℕ+), lcm = Nat.lcm A B ∧ lcm = 624 * y :=
by sorry

end NUMINAMATH_CALUDE_lcm_factor_not_unique_l1554_155425


namespace NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l1554_155429

/-- Calculates the time needed to read a book given reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

/-- Proves that Gwendolyn will take 225 hours to read the book -/
theorem gwendolyn_reading_time : 
  reading_time_calculation 200 30 15 100 = 225 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l1554_155429


namespace NUMINAMATH_CALUDE_special_function_property_l1554_155453

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 1) / f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1554_155453


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1554_155493

theorem square_circle_union_area (s : Real) (r : Real) : 
  s = 12 → r = 12 → (s ^ 2 + π * r ^ 2 - s ^ 2) = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1554_155493


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_range_l1554_155405

theorem sqrt_2x_plus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_range_l1554_155405


namespace NUMINAMATH_CALUDE_power_of_prime_squared_minus_one_l1554_155410

theorem power_of_prime_squared_minus_one (n : ℕ) : 
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ n^2 - 1 = p^k) ↔ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_prime_squared_minus_one_l1554_155410


namespace NUMINAMATH_CALUDE_sin_max_min_difference_l1554_155442

theorem sin_max_min_difference (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 9 → f x = 2 * Real.sin (π * x / 6 - π / 3)) →
  (⨆ x ∈ Set.Icc 0 9, f x) - (⨅ x ∈ Set.Icc 0 9, f x) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_max_min_difference_l1554_155442


namespace NUMINAMATH_CALUDE_sam_spent_three_dimes_per_candy_bar_l1554_155426

/-- Represents the number of cents in a dime -/
def dime_value : ℕ := 10

/-- Represents the number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- Represents the initial number of dimes Sam has -/
def initial_dimes : ℕ := 19

/-- Represents the initial number of quarters Sam has -/
def initial_quarters : ℕ := 6

/-- Represents the number of candy bars Sam buys -/
def candy_bars : ℕ := 4

/-- Represents the number of lollipops Sam buys -/
def lollipops : ℕ := 1

/-- Represents the amount of money Sam has left after purchases (in cents) -/
def money_left : ℕ := 195

/-- Proves that Sam spent 3 dimes on each candy bar -/
theorem sam_spent_three_dimes_per_candy_bar :
  ∃ (dimes_per_candy : ℕ),
    dimes_per_candy * candy_bars * dime_value + 
    lollipops * quarter_value + 
    money_left = 
    initial_dimes * dime_value + 
    initial_quarters * quarter_value ∧
    dimes_per_candy = 3 := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_three_dimes_per_candy_bar_l1554_155426


namespace NUMINAMATH_CALUDE_barry_pretzels_l1554_155432

/-- Given the following conditions about pretzel purchases:
  - Angie bought three times as many pretzels as Shelly
  - Shelly bought half as many pretzels as Barry
  - Angie bought 18 pretzels
Prove that Barry bought 12 pretzels. -/
theorem barry_pretzels (angie shelly barry : ℕ) 
  (h1 : angie = 3 * shelly) 
  (h2 : shelly = barry / 2) 
  (h3 : angie = 18) : 
  barry = 12 := by
  sorry

end NUMINAMATH_CALUDE_barry_pretzels_l1554_155432


namespace NUMINAMATH_CALUDE_real_part_of_z_l1554_155489

theorem real_part_of_z (z : ℂ) (h : z * (2 - Complex.I) = 18 + 11 * Complex.I) :
  z.re = 5 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1554_155489


namespace NUMINAMATH_CALUDE_remaining_area_calculation_l1554_155420

theorem remaining_area_calculation (large_square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : large_square_side = 9)
  (h2 : small_square1_side = 4)
  (h3 : small_square2_side = 2)
  (h4 : small_square1_side ^ 2 + small_square2_side ^ 2 ≤ large_square_side ^ 2) :
  large_square_side ^ 2 - (small_square1_side ^ 2 + small_square2_side ^ 2) = 61 := by
sorry


end NUMINAMATH_CALUDE_remaining_area_calculation_l1554_155420


namespace NUMINAMATH_CALUDE_fraction_of_juices_consumed_l1554_155408

/-- Represents the fraction of juices consumed at a summer picnic -/
theorem fraction_of_juices_consumed (total_people : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (soda_drinkers : ℚ) (water_drinkers : ℚ) (total_recyclables : ℕ) :
  total_people = 90 →
  soda_cans = 50 →
  water_bottles = 50 →
  juice_bottles = 50 →
  soda_drinkers = 1/2 →
  water_drinkers = 1/3 →
  total_recyclables = 115 →
  (juice_bottles - (total_recyclables - (soda_drinkers * total_people + water_drinkers * total_people))) / juice_bottles = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_juices_consumed_l1554_155408


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l1554_155437

theorem mod_eight_equivalence : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 ≡ n [ZMOD 8] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l1554_155437


namespace NUMINAMATH_CALUDE_min_trig_fraction_l1554_155424

theorem min_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (17/8) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_trig_fraction_l1554_155424


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l1554_155476

/-- Given the following conditions:
    - Nancy has 40 metal beads and 20 more pearl beads than metal beads
    - Rose has 20 crystal beads and twice as many stone beads as crystal beads
    - They can make 20 bracelets
    Prove that the number of beads in each bracelet is 8. -/
theorem beads_per_bracelet :
  let nancy_metal : ℕ := 40
  let nancy_pearl : ℕ := nancy_metal + 20
  let rose_crystal : ℕ := 20
  let rose_stone : ℕ := 2 * rose_crystal
  let total_bracelets : ℕ := 20
  let total_beads : ℕ := nancy_metal + nancy_pearl + rose_crystal + rose_stone
  (total_beads / total_bracelets : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l1554_155476


namespace NUMINAMATH_CALUDE_cubic_complex_roots_l1554_155401

/-- A cubic polynomial with integer coefficients and leading coefficient 1 -/
def CubicPolynomial (a b c : ℤ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- The property that a polynomial has exactly one real integer root -/
def HasOneRealIntegerRoot (P : ℝ → ℝ) : Prop :=
  ∃! (p : ℤ), P p = 0 ∧ ∀ (x : ℝ), P x = 0 → x = p

theorem cubic_complex_roots (a b c : ℤ) :
  let P := CubicPolynomial a b c
  HasOneRealIntegerRoot P →
  (∃ (z : ℂ), P z.re = 0 ∧ (z = -1 + Complex.I * Real.sqrt 3 ∨
                            z = -1 + Complex.I * 2 ∨
                            z = -1 + Complex.I * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_complex_roots_l1554_155401


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l1554_155428

theorem cone_vertex_angle (r l : ℝ) (h : r > 0) (h2 : l > 0) : 
  (π * r * l) / (π * r^2) = 2 → 2 * Real.arcsin (r / l) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l1554_155428


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1554_155455

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter : ℝ := 60
  let pentagon_perimeter : ℝ := 60
  let triangle_side : ℝ := triangle_perimeter / 3
  let pentagon_side : ℝ := pentagon_perimeter / 5
  triangle_side / pentagon_side = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1554_155455


namespace NUMINAMATH_CALUDE_room_expansion_theorem_l1554_155477

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a room --/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: If increasing the length and breadth of a rectangular room by y feet
    results in a perimeter increase of 16 feet, then y must equal 4 feet. --/
theorem room_expansion_theorem (r : Room) (y : ℝ) 
    (h : perimeter { length := r.length + y, breadth := r.breadth + y } - perimeter r = 16) : 
  y = 4 := by
  sorry

end NUMINAMATH_CALUDE_room_expansion_theorem_l1554_155477


namespace NUMINAMATH_CALUDE_sarah_and_tom_ages_l1554_155484

/-- Given the age relationship between Sarah and Tom, prove their current ages sum to 33 -/
theorem sarah_and_tom_ages : ∃ (s t : ℕ),
  (s = t + 7) ∧                   -- Sarah is seven years older than Tom
  (s + 10 = 3 * (t - 3)) ∧        -- Ten years from now, Sarah will be three times as old as Tom was three years ago
  (s + t = 33)                    -- The sum of their current ages is 33
:= by sorry

end NUMINAMATH_CALUDE_sarah_and_tom_ages_l1554_155484


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l1554_155447

/-- Given that the cost price of 75 articles after a 5% discount
    equals the selling price of 60 articles before a 12% sales tax,
    prove that the percent profit is 25%. -/
theorem percent_profit_calculation (CP : ℝ) (SP : ℝ) :
  75 * CP * (1 - 0.05) = 60 * SP →
  (SP - CP * (1 - 0.05)) / (CP * (1 - 0.05)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l1554_155447


namespace NUMINAMATH_CALUDE_prob_same_team_is_one_third_l1554_155400

/-- The number of teams -/
def num_teams : ℕ := 3

/-- The probability of two students choosing the same team -/
def prob_same_team : ℚ := 1 / 3

/-- Theorem: The probability of two students independently and randomly choosing the same team out of three teams is 1/3 -/
theorem prob_same_team_is_one_third :
  prob_same_team = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_team_is_one_third_l1554_155400


namespace NUMINAMATH_CALUDE_fibonacci_lucas_relation_l1554_155456

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define Lucas sequence
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

-- State the theorem
theorem fibonacci_lucas_relation (n p : ℕ) :
  ((lucas n : ℝ) + Real.sqrt 5 * (fib n : ℝ)) / 2 ^ p =
  (lucas (n * p) + Real.sqrt 5 * fib (n * p)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_lucas_relation_l1554_155456


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1554_155416

theorem fraction_equivalence (x b : ℝ) : 
  (x + 2*b) / (x + 3*b) = 2/3 ↔ x = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1554_155416


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1554_155472

theorem unique_positive_solution (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (6 * y) * Real.sqrt (18 * y) * Real.sqrt (9 * y) = 27) : 
  y = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1554_155472


namespace NUMINAMATH_CALUDE_elias_bananas_l1554_155411

def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem elias_bananas : 
  let initial := 12
  let remaining := 11
  bananas_eaten initial remaining = 1 := by
sorry

end NUMINAMATH_CALUDE_elias_bananas_l1554_155411


namespace NUMINAMATH_CALUDE_dessert_distribution_l1554_155415

/-- Proves that given 14 mini-cupcakes, 12 donut holes, and 13 students,
    if each student receives the same amount, then each student gets 2 desserts. -/
theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 := by
  sorry

end NUMINAMATH_CALUDE_dessert_distribution_l1554_155415


namespace NUMINAMATH_CALUDE_wine_consumption_problem_l1554_155450

/-- Represents the wine consumption problem from the Ming Dynasty's "The Great Compendium of Mathematics" -/
theorem wine_consumption_problem (x y : ℚ) : 
  (x + y = 19 ∧ 3 * x + (1/3) * y = 33) ↔ 
  (x ≥ 0 ∧ y ≥ 0 ∧ 
   ∃ (good_wine weak_wine guests : ℕ),
     good_wine = x ∧
     weak_wine = y ∧
     guests = 33 ∧
     good_wine + weak_wine = 19 ∧
     (3 * good_wine + (weak_wine / 3 : ℚ)) = guests) :=
by sorry

end NUMINAMATH_CALUDE_wine_consumption_problem_l1554_155450


namespace NUMINAMATH_CALUDE_impossible_same_color_l1554_155438

/-- Represents the number of chips of each color -/
structure ChipState :=
  (blue : Nat)
  (red : Nat)
  (yellow : Nat)

/-- Represents a single recoloring step -/
inductive RecolorStep
  | BlueRedToYellow
  | RedYellowToBlue
  | BlueYellowToRed

/-- The initial state of chips -/
def initialState : ChipState :=
  { blue := 2008, red := 2009, yellow := 2010 }

/-- Applies a recoloring step to a given state -/
def applyStep (state : ChipState) (step : RecolorStep) : ChipState :=
  match step with
  | RecolorStep.BlueRedToYellow => 
      { blue := state.blue - 1, red := state.red - 1, yellow := state.yellow + 2 }
  | RecolorStep.RedYellowToBlue => 
      { blue := state.blue + 2, red := state.red - 1, yellow := state.yellow - 1 }
  | RecolorStep.BlueYellowToRed => 
      { blue := state.blue - 1, red := state.red + 2, yellow := state.yellow - 1 }

/-- Represents a sequence of recoloring steps -/
def RecolorSequence := List RecolorStep

/-- Applies a sequence of recoloring steps to the initial state -/
def applySequence (seq : RecolorSequence) : ChipState :=
  seq.foldl applyStep initialState

/-- Checks if all chips are of the same color -/
def allSameColor (state : ChipState) : Bool :=
  (state.blue = 0 && state.red = 0) ||
  (state.blue = 0 && state.yellow = 0) ||
  (state.red = 0 && state.yellow = 0)

/-- The main theorem: It's impossible to make all chips the same color -/
theorem impossible_same_color : ∀ (seq : RecolorSequence), ¬(allSameColor (applySequence seq)) := by
  sorry

end NUMINAMATH_CALUDE_impossible_same_color_l1554_155438


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_19_l1554_155488

theorem modular_inverse_17_mod_19 :
  ∃ x : ℕ, x ≤ 18 ∧ (17 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_19_l1554_155488


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_set_l1554_155498

theorem range_of_a_for_nonempty_set (A : Set ℝ) (h_nonempty : A.Nonempty) :
  (∃ a : ℝ, A = {x : ℝ | a * x = 1}) → (∃ a : ℝ, a ≠ 0 ∧ A = {x : ℝ | a * x = 1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_set_l1554_155498


namespace NUMINAMATH_CALUDE_zoo_visitors_l1554_155413

theorem zoo_visitors (total_people : ℕ) (adult_price child_price : ℚ) (total_bill : ℚ) :
  total_people = 201 ∧ 
  adult_price = 8 ∧ 
  child_price = 4 ∧ 
  total_bill = 964 →
  ∃ (adults children : ℕ), 
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_bill ∧
    children = 161 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1554_155413


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l1554_155422

/-- For any positive integer n, there exists a positive integer m such that
    for all k in the range 0 ≤ k < n, m + k is not an integer power of a prime number. -/
theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ m : ℕ+, ∀ k : ℕ, k < n → ¬∃ (p : ℕ) (e : ℕ), Prime p ∧ (m + k : ℕ) = p ^ e :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l1554_155422


namespace NUMINAMATH_CALUDE_slower_pipe_filling_time_l1554_155471

/-- Proves that given two pipes with specified relative speeds and combined filling time,
    the slower pipe takes 180 minutes to fill the tank alone. -/
theorem slower_pipe_filling_time 
  (fast_pipe_speed : ℝ) 
  (slow_pipe_speed : ℝ) 
  (combined_time : ℝ) 
  (h1 : fast_pipe_speed = 4 * slow_pipe_speed) 
  (h2 : combined_time = 36) 
  (h3 : (fast_pipe_speed + slow_pipe_speed) * combined_time = 1) : 
  1 / slow_pipe_speed = 180 := by
  sorry

end NUMINAMATH_CALUDE_slower_pipe_filling_time_l1554_155471


namespace NUMINAMATH_CALUDE_lindas_broken_eggs_l1554_155423

/-- The number of eggs Linda broke -/
def broken_eggs (initial_white : ℕ) (initial_brown : ℕ) (total_after : ℕ) : ℕ :=
  initial_white + initial_brown - total_after

theorem lindas_broken_eggs :
  let initial_brown := 5
  let initial_white := 3 * initial_brown
  let total_after := 12
  broken_eggs initial_white initial_brown total_after = 8 := by
  sorry

#eval broken_eggs (3 * 5) 5 12  -- Should output 8

end NUMINAMATH_CALUDE_lindas_broken_eggs_l1554_155423


namespace NUMINAMATH_CALUDE_smaller_package_size_l1554_155496

/-- The number of notebooks in a large package -/
def large_package : ℕ := 7

/-- The total number of notebooks Wilson bought -/
def total_notebooks : ℕ := 69

/-- The number of large packages Wilson bought -/
def large_packages_bought : ℕ := 7

/-- The number of notebooks in the smaller package -/
def small_package : ℕ := 5

/-- Theorem stating that the smaller package contains 5 notebooks -/
theorem smaller_package_size :
  ∃ (n : ℕ), 
    n * small_package + large_packages_bought * large_package = total_notebooks ∧
    n > 0 ∧
    small_package < large_package ∧
    small_package ∣ (total_notebooks - large_packages_bought * large_package) :=
by sorry

end NUMINAMATH_CALUDE_smaller_package_size_l1554_155496


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1554_155419

open Real

theorem sum_of_x_values (x : ℝ) : 
  (0 < x) → 
  (x < 180) → 
  (sin (2 * x * π / 180))^3 + (sin (6 * x * π / 180))^3 = 
    8 * (sin (3 * x * π / 180))^3 * (sin (x * π / 180))^3 → 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (0 < x₁) ∧ (x₁ < 180) ∧
    (0 < x₂) ∧ (x₂ < 180) ∧
    (0 < x₃) ∧ (x₃ < 180) ∧
    (sin (2 * x₁ * π / 180))^3 + (sin (6 * x₁ * π / 180))^3 = 
      8 * (sin (3 * x₁ * π / 180))^3 * (sin (x₁ * π / 180))^3 ∧
    (sin (2 * x₂ * π / 180))^3 + (sin (6 * x₂ * π / 180))^3 = 
      8 * (sin (3 * x₂ * π / 180))^3 * (sin (x₂ * π / 180))^3 ∧
    (sin (2 * x₃ * π / 180))^3 + (sin (6 * x₃ * π / 180))^3 = 
      8 * (sin (3 * x₃ * π / 180))^3 * (sin (x₃ * π / 180))^3 ∧
    x₁ + x₂ + x₃ = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1554_155419


namespace NUMINAMATH_CALUDE_carson_gold_stars_l1554_155445

/-- Proves that Carson earned 6 gold stars yesterday -/
theorem carson_gold_stars :
  ∀ (yesterday today total : ℕ),
    today = 9 →
    total = 15 →
    total = yesterday + today →
    yesterday = 6 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l1554_155445


namespace NUMINAMATH_CALUDE_valid_call_start_l1554_155469

/-- Represents a time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- The time difference between Beijing and Moscow in hours -/
def time_difference : Nat := 5

/-- Checks if a given time is within the allowed call window -/
def is_valid_call_time (t : Time) : Prop :=
  9 ≤ t.hour ∧ t.hour < 17

/-- Converts Beijing time to Moscow time -/
def beijing_to_moscow (t : Time) : Time :=
  let new_hour := (t.hour - time_difference + 24) % 24
  { hour := new_hour, minute := t.minute, 
    h_valid := by sorry,
    m_valid := t.m_valid }

/-- The proposed call start time in Beijing -/
def call_start_beijing : Time :=
  { hour := 15, minute := 0, h_valid := by sorry, m_valid := by sorry }

/-- Theorem stating that the proposed call start time is valid -/
theorem valid_call_start : 
  is_valid_call_time call_start_beijing ∧ 
  is_valid_call_time (beijing_to_moscow call_start_beijing) :=
by sorry


end NUMINAMATH_CALUDE_valid_call_start_l1554_155469


namespace NUMINAMATH_CALUDE_line_segment_ratios_l1554_155436

/-- Given four points X, Y, Z, W on a straight line in that order,
    with XY = 3, YZ = 4, and XW = 20, prove that
    the ratio of XZ to YW is 7/16 and the ratio of YZ to XW is 1/5. -/
theorem line_segment_ratios
  (X Y Z W : ℝ)  -- Points represented as real numbers
  (h_order : X < Y ∧ Y < Z ∧ Z < W)  -- Order of points
  (h_xy : Y - X = 3)  -- XY = 3
  (h_yz : Z - Y = 4)  -- YZ = 4
  (h_xw : W - X = 20)  -- XW = 20
  : (Z - X) / (W - Y) = 7 / 16 ∧ (Z - Y) / (W - X) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_ratios_l1554_155436


namespace NUMINAMATH_CALUDE_least_number_divisibility_l1554_155430

theorem least_number_divisibility (n : ℕ) : n = 215988 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 72 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 108 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 125 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    (n + 12) = 48 * k₁ ∧
    (n + 12) = 64 * k₂ ∧
    (n + 12) = 72 * k₃ ∧
    (n + 12) = 108 * k₄ ∧
    (n + 12) = 125 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l1554_155430


namespace NUMINAMATH_CALUDE_fifth_month_sales_l1554_155466

def sales_problem (month1 month2 month3 month4 month6 : ℕ) (target_average : ℕ) : ℕ :=
  6 * target_average - (month1 + month2 + month3 + month4 + month6)

theorem fifth_month_sales :
  sales_problem 6635 6927 6855 7230 4791 6500 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l1554_155466


namespace NUMINAMATH_CALUDE_dot_product_equals_25_l1554_155481

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_25 (b : ℝ × ℝ) 
  (h : a - (1/5 : ℝ) • b = (-2, 1)) : 
  a • b = 25 := by sorry

end NUMINAMATH_CALUDE_dot_product_equals_25_l1554_155481


namespace NUMINAMATH_CALUDE_number_of_possible_sets_l1554_155449

theorem number_of_possible_sets (A : Set ℤ) : 
  (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (S : Finset (Set ℤ)), (∀ X ∈ S, X ∪ {-1, 1} = {-1, 0, 1}) ∧ S.card = 4 ∧ 
    ∀ Y, Y ∪ {-1, 1} = {-1, 0, 1} → Y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sets_l1554_155449


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l1554_155485

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 7 ≤ 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x ≤ 7} := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  (∀ m : ℝ, (B m).Nonempty → (B m ⊆ A ∧ B m ≠ A)) ↔ 2 ≤ m ∧ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l1554_155485


namespace NUMINAMATH_CALUDE_triangle_special_angle_l1554_155468

theorem triangle_special_angle (a b c : ℝ) (h : (a + 2*b + c)*(a + b - c - 2) = 4*a*b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l1554_155468


namespace NUMINAMATH_CALUDE_state_quarters_fraction_l1554_155434

theorem state_quarters_fraction :
  ∀ (total_quarters : ℕ) (states_in_decade : ℕ),
    total_quarters = 18 →
    states_in_decade = 5 →
    (states_in_decade : ℚ) / (total_quarters : ℚ) = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_state_quarters_fraction_l1554_155434


namespace NUMINAMATH_CALUDE_kareems_son_age_l1554_155439

theorem kareems_son_age (kareem_age : ℕ) (son_age : ℕ) : 
  kareem_age = 42 →
  kareem_age = 3 * son_age →
  (kareem_age + 10) + (son_age + 10) = 76 →
  son_age = 14 := by
sorry

end NUMINAMATH_CALUDE_kareems_son_age_l1554_155439


namespace NUMINAMATH_CALUDE_school_age_ratio_l1554_155406

theorem school_age_ratio :
  ∀ (total below_eight eight above_eight : ℕ),
    total = 125 →
    below_eight = total / 5 →
    eight = 60 →
    total = below_eight + eight + above_eight →
    (above_eight : ℚ) / (eight : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_school_age_ratio_l1554_155406


namespace NUMINAMATH_CALUDE_min_n_for_monochromatic_sum_l1554_155402

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ a b c d, a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n →
    (coloring a = coloring b ∧ coloring b = coloring c ∧ coloring c = coloring d) →
    a + b + c ≠ d

theorem min_n_for_monochromatic_sum : 
  (∀ coloring : ℕ → Bool, ¬(is_valid_coloring 11 coloring)) ∧
  (∃ coloring : ℕ → Bool, is_valid_coloring 10 coloring) :=
sorry

end NUMINAMATH_CALUDE_min_n_for_monochromatic_sum_l1554_155402


namespace NUMINAMATH_CALUDE_butterfly_price_is_three_l1554_155461

/-- Given information about John's butterfly business -/
structure ButterflyBusiness where
  jars : ℕ
  caterpillars_per_jar : ℕ
  success_rate : ℚ
  total_revenue : ℚ

/-- Calculate the price per butterfly -/
def price_per_butterfly (b : ButterflyBusiness) : ℚ :=
  b.total_revenue / (b.jars * b.caterpillars_per_jar * b.success_rate)

/-- Theorem: The price per butterfly is $3 -/
theorem butterfly_price_is_three (b : ButterflyBusiness) 
  (h1 : b.jars = 4)
  (h2 : b.caterpillars_per_jar = 10)
  (h3 : b.success_rate = 3/5)
  (h4 : b.total_revenue = 72) :
  price_per_butterfly b = 3 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_price_is_three_l1554_155461


namespace NUMINAMATH_CALUDE_ticket_identification_operations_l1554_155486

/-- The maximum ticket number --/
def max_ticket : Nat := 30

/-- The number of operations needed to identify all ticket numbers --/
def num_operations : Nat := 5

/-- Function to calculate the number of binary digits needed to represent a number --/
def binary_digits (n : Nat) : Nat :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem ticket_identification_operations :
  binary_digits max_ticket = num_operations :=
by sorry

end NUMINAMATH_CALUDE_ticket_identification_operations_l1554_155486


namespace NUMINAMATH_CALUDE_min_value_of_f_l1554_155479

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1554_155479


namespace NUMINAMATH_CALUDE_mrs_a_speed_l1554_155492

/-- Proves that Mrs. A's speed is 10 kmph given the problem conditions --/
theorem mrs_a_speed (initial_distance : ℝ) (mr_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mr_a_speed = 30)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  let time := bee_distance / bee_speed
  let mr_a_distance := mr_a_speed * time
  let mrs_a_distance := initial_distance - mr_a_distance
  mrs_a_distance / time = 10 := by
  sorry

#check mrs_a_speed

end NUMINAMATH_CALUDE_mrs_a_speed_l1554_155492


namespace NUMINAMATH_CALUDE_a_range_l1554_155421

theorem a_range (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  a ∈ Set.Ici 4 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l1554_155421


namespace NUMINAMATH_CALUDE_chocolates_per_first_year_student_l1554_155480

theorem chocolates_per_first_year_student :
  ∀ (total_students first_year_students second_year_students total_chocolates leftover_chocolates : ℕ),
    total_students = 24 →
    total_students = first_year_students + second_year_students →
    second_year_students = 2 * first_year_students →
    total_chocolates = 50 →
    leftover_chocolates = 2 →
    (total_chocolates - leftover_chocolates) / first_year_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_per_first_year_student_l1554_155480


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1554_155487

theorem infinite_solutions_condition (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1554_155487


namespace NUMINAMATH_CALUDE_brahmagupta_theorem_l1554_155474

/-- An inscribed quadrilateral with side lengths a, b, c, d and diagonals p, q -/
structure InscribedQuadrilateral (a b c d p q : ℝ) : Prop where
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < p ∧ 0 < q
  inscribed : ∃ (r : ℝ), 0 < r ∧ a + c = b + d -- Condition for inscribability

/-- Brahmagupta's theorem for inscribed quadrilaterals -/
theorem brahmagupta_theorem {a b c d p q : ℝ} (quad : InscribedQuadrilateral a b c d p q) :
  p^2 + q^2 = a^2 + b^2 + c^2 + d^2 ∧ 2*p*q = a^2 + c^2 - b^2 - d^2 := by
  sorry

#check brahmagupta_theorem

end NUMINAMATH_CALUDE_brahmagupta_theorem_l1554_155474


namespace NUMINAMATH_CALUDE_jack_morning_emails_l1554_155435

theorem jack_morning_emails (afternoon_emails : ℕ) (morning_afternoon_difference : ℕ) : 
  afternoon_emails = 2 → 
  morning_afternoon_difference = 4 →
  afternoon_emails + morning_afternoon_difference = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l1554_155435


namespace NUMINAMATH_CALUDE_subtraction_result_l1554_155403

theorem subtraction_result (chosen_number : ℕ) : 
  chosen_number = 127 → (2 * chosen_number) - 152 = 102 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1554_155403


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1554_155417

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    and a point P on its right branch satisfying |PF₁| = 4|PF₂|, 
    prove that the eccentricity e is in the range (1, 5/3] -/
theorem hyperbola_eccentricity_range (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) 
  (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)  -- P is on the hyperbola
  (h₄ : P.1 > 0)  -- P is on the right branch
  (h₅ : ‖P - F₁‖ = 4 * ‖P - F₂‖)  -- |PF₁| = 4|PF₂|
  (h₆ : F₁.1 < 0 ∧ F₂.1 > 0)  -- F₁ is left focus, F₂ is right focus
  (h₇ : ‖F₁ - F₂‖ = 2 * (a^2 + b^2).sqrt)  -- distance between foci
  : 1 < (a^2 + b^2).sqrt / a ∧ (a^2 + b^2).sqrt / a ≤ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1554_155417


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l1554_155499

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l1554_155499


namespace NUMINAMATH_CALUDE_acid_dilution_l1554_155454

/-- Proves that adding 80/3 ounces of pure water to 40 ounces of a 25% acid solution
    results in a 15% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) : 
    initial_volume = 40 →
    initial_concentration = 0.25 →
    added_water = 80 / 3 →
    final_concentration = 0.15 →
    (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l1554_155454


namespace NUMINAMATH_CALUDE_f_range_on_interval_l1554_155491

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x) - 4 * (Real.cos x) ^ 2 + 2

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (3 * Real.pi / 4) Real.pi,
    ∃ y ∈ Set.Icc (-2 * Real.sqrt 2) (-2),
      f x = y ∧
      ∀ z, f x = z → z ∈ Set.Icc (-2 * Real.sqrt 2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l1554_155491


namespace NUMINAMATH_CALUDE_orange_problem_l1554_155414

theorem orange_problem (initial_oranges : ℕ) : 
  (initial_oranges : ℚ) * (3/4) * (4/7) - 4 = 32 → initial_oranges = 84 := by
  sorry

end NUMINAMATH_CALUDE_orange_problem_l1554_155414


namespace NUMINAMATH_CALUDE_solve_equation_l1554_155457

theorem solve_equation (x n : ℚ) (h1 : n * (x - 3) = 15) (h2 : x = 12) : n = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1554_155457


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1554_155446

theorem sum_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1554_155446


namespace NUMINAMATH_CALUDE_transistor_count_2005_l1554_155452

/-- Calculates the number of transistors in a CPU after applying Moore's law and an additional growth law over a specified time period. -/
def transistor_count (initial_count : ℕ) (years : ℕ) : ℕ :=
  let doubling_cycles := years / 2
  let tripling_cycles := years / 6
  initial_count * 2^doubling_cycles + initial_count * 3^tripling_cycles

/-- Theorem stating that the number of transistors in a CPU in 2005 is 68,500,000,
    given an initial count of 500,000 in 1990 and the application of Moore's law
    and an additional growth law. -/
theorem transistor_count_2005 :
  transistor_count 500000 15 = 68500000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2005_l1554_155452


namespace NUMINAMATH_CALUDE_rook_placements_on_chessboard_l1554_155404

/-- The number of ways to place n rooks on an n×n chessboard so that no two rooks 
    are in the same row or column -/
def valid_rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- The size of the chessboard -/
def board_size : ℕ := 8

theorem rook_placements_on_chessboard : 
  valid_rook_placements board_size = 40320 := by
  sorry

#eval valid_rook_placements board_size

end NUMINAMATH_CALUDE_rook_placements_on_chessboard_l1554_155404


namespace NUMINAMATH_CALUDE_weightlifting_time_l1554_155490

/-- Represents Kyle's basketball practice schedule --/
structure BasketballPractice where
  total_time : ℝ
  shooting_time : ℝ
  running_time : ℝ
  weightlifting_time : ℝ
  stretching_time : ℝ
  dribbling_time : ℝ
  defense_time : ℝ

/-- Kyle's basketball practice schedule satisfies the given conditions --/
def valid_practice (p : BasketballPractice) : Prop :=
  p.total_time = 2 ∧
  p.shooting_time = (1/3) * p.total_time ∧
  p.running_time = 2 * p.weightlifting_time ∧
  p.stretching_time = p.weightlifting_time ∧
  p.dribbling_time = (1/6) * p.total_time ∧
  p.defense_time = (1/12) * p.total_time ∧
  p.total_time = p.shooting_time + p.running_time + p.weightlifting_time + 
                 p.stretching_time + p.dribbling_time + p.defense_time

/-- Theorem: Kyle spends 5/12 hours lifting weights --/
theorem weightlifting_time (p : BasketballPractice) 
  (h : valid_practice p) : p.weightlifting_time = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_time_l1554_155490


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1554_155463

/-- An isosceles triangle with specific heights -/
structure IsoscelesTriangle where
  -- The height drawn to the base
  baseHeight : ℝ
  -- The height drawn to one of the equal sides
  sideHeight : ℝ
  -- Assumption that the triangle is isosceles
  isIsosceles : True

/-- The base of the triangle -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  7.5

/-- Theorem stating that for an isosceles triangle with given heights, the base length is 7.5 -/
theorem isosceles_triangle_base_length 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.baseHeight = 5) 
  (h2 : triangle.sideHeight = 6) : 
  baseLength triangle = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1554_155463


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1554_155409

theorem arithmetic_expression_equality : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1554_155409
