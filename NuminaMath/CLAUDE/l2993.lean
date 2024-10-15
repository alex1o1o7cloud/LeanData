import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2993_299385

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2993_299385


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l2993_299352

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l2993_299352


namespace NUMINAMATH_CALUDE_third_group_data_points_l2993_299392

/-- Given a sample divided into 5 groups with specific conditions, prove the number of data points in the third group --/
theorem third_group_data_points
  (total_groups : ℕ)
  (group_123_sum : ℕ)
  (group_345_sum : ℕ)
  (group_3_frequency : ℚ)
  (h1 : total_groups = 5)
  (h2 : group_123_sum = 160)
  (h3 : group_345_sum = 260)
  (h4 : group_3_frequency = 1/5) :
  ∃ (group_3 : ℕ), 
    group_3 = 70 ∧ 
    (group_3 : ℚ) / (group_123_sum + group_345_sum - group_3) = group_3_frequency :=
by sorry

end NUMINAMATH_CALUDE_third_group_data_points_l2993_299392


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l2993_299308

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧ digits.toFinset.card = 5

def divisible_by_digits_except_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5 → n % d = 0

theorem least_five_digit_divisible_by_digits :
  ∀ n : ℕ,
    is_five_digit n ∧
    all_digits_different n ∧
    divisible_by_digits_except_five n →
    12376 ≤ n :=
by sorry

theorem twelve_three_seven_six_satisfies_conditions :
  is_five_digit 12376 ∧
  all_digits_different 12376 ∧
  divisible_by_digits_except_five 12376 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l2993_299308


namespace NUMINAMATH_CALUDE_quadratic_sum_l2993_299336

/-- For the quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2993_299336


namespace NUMINAMATH_CALUDE_complex_magnitude_real_part_l2993_299384

theorem complex_magnitude_real_part (t : ℝ) : 
  t > 0 → Complex.abs (9 + t * Complex.I) = 15 → Complex.re (9 + t * Complex.I) = 9 → t = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_real_part_l2993_299384


namespace NUMINAMATH_CALUDE_quotient_problem_l2993_299351

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 166)
    (h2 : divisor = 20)
    (h3 : remainder = 6)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l2993_299351


namespace NUMINAMATH_CALUDE_cafe_tables_l2993_299370

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of tables needed given the number of people and people per table -/
def tablesNeeded (people : ℕ) (peoplePerTable : ℕ) : ℕ := 
  (people + peoplePerTable - 1) / peoplePerTable

theorem cafe_tables : 
  let totalPeople : ℕ := base7ToBase10 310
  let peoplePerTable : ℕ := 3
  tablesNeeded totalPeople peoplePerTable = 52 := by sorry

end NUMINAMATH_CALUDE_cafe_tables_l2993_299370


namespace NUMINAMATH_CALUDE_developed_countries_modern_pattern_l2993_299380

/-- Represents different types of countries --/
inductive CountryType
| Developed
| Developing

/-- Represents different population growth patterns --/
inductive GrowthPattern
| Traditional
| Modern

/-- Represents the growth rate of a country --/
structure GrowthRate where
  rate : ℝ

/-- A country with its properties --/
structure Country where
  type : CountryType
  growthPattern : GrowthPattern
  growthRate : GrowthRate
  hasImplementedFamilyPlanning : Bool

/-- Axiom: Developed countries have slow growth rates --/
axiom developed_country_slow_growth (c : Country) :
  c.type = CountryType.Developed → c.growthRate.rate ≤ 0

/-- Axiom: Developing countries have faster growth rates --/
axiom developing_country_faster_growth (c : Country) :
  c.type = CountryType.Developing → c.growthRate.rate > 0

/-- Axiom: Most developing countries are in the traditional growth pattern --/
axiom most_developing_traditional (c : Country) :
  c.type = CountryType.Developing → c.growthPattern = GrowthPattern.Traditional

/-- Axiom: Countries with family planning are in the modern growth pattern --/
axiom family_planning_modern_pattern (c : Country) :
  c.hasImplementedFamilyPlanning → c.growthPattern = GrowthPattern.Modern

/-- Theorem: Developed countries are in the modern population growth pattern --/
theorem developed_countries_modern_pattern (c : Country) :
  c.type = CountryType.Developed → c.growthPattern = GrowthPattern.Modern := by
  sorry

end NUMINAMATH_CALUDE_developed_countries_modern_pattern_l2993_299380


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2993_299363

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p) * Real.sqrt (8 * p^5) = 20 * p^4 * Real.sqrt (6 * p) := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2993_299363


namespace NUMINAMATH_CALUDE_x_y_negative_l2993_299309

theorem x_y_negative (x y : ℝ) (h1 : x - y > 2*x) (h2 : x + y < 0) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l2993_299309


namespace NUMINAMATH_CALUDE_room_width_calculation_l2993_299330

/-- Given a rectangular room with length 5 meters, prove that its width is 4.75 meters
    when the cost of paving is 900 per square meter and the total cost is 21375. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5 →
  cost_per_sqm = 900 →
  total_cost = 21375 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

#eval (21375 / 900) / 5

end NUMINAMATH_CALUDE_room_width_calculation_l2993_299330


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2993_299354

noncomputable def i : ℂ := Complex.I

theorem modulus_of_complex_fraction :
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2993_299354


namespace NUMINAMATH_CALUDE_yasmin_bank_account_l2993_299357

/-- Yasmin's bank account problem -/
theorem yasmin_bank_account (deposit : ℕ) (new_balance : ℕ) (initial_balance : ℕ) : 
  deposit = 50 →
  4 * deposit = new_balance →
  initial_balance = new_balance - deposit →
  initial_balance = 150 := by
  sorry

end NUMINAMATH_CALUDE_yasmin_bank_account_l2993_299357


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l2993_299394

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_a_values (a : ℝ) :
  (∃ x y : ℝ, y = a * x - 2 ∧ 3 * x - (a + 2) * y + 1 = 0) →
  (∀ x y : ℝ, y = a * x - 2 ↔ 3 * x - (a + 2) * y + 1 = 0) →
  a = 1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l2993_299394


namespace NUMINAMATH_CALUDE_maoming_population_scientific_notation_l2993_299301

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Maoming city in millions -/
def maoming_population : ℝ := 6.8

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem maoming_population_scientific_notation :
  to_scientific_notation maoming_population = ScientificNotation.mk 6.8 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_maoming_population_scientific_notation_l2993_299301


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l2993_299396

theorem isosceles_right_triangle (A B C : ℝ) (a b c : ℝ) : 
  (Real.sin (A - B))^2 + (Real.cos C)^2 = 0 → 
  (A = B ∧ C = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l2993_299396


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l2993_299393

theorem circle_intersection_radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) → 
  1 < r ∧ r < 11 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l2993_299393


namespace NUMINAMATH_CALUDE_robie_second_purchase_l2993_299381

/-- The number of bags of chocolates Robie bought the second time -/
def second_purchase (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Robie bought 3 bags of chocolates the second time -/
theorem robie_second_purchase :
  second_purchase 3 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_robie_second_purchase_l2993_299381


namespace NUMINAMATH_CALUDE_class_average_weight_l2993_299342

theorem class_average_weight (n1 : ℕ) (n2 : ℕ) (w1 : ℝ) (w2 : ℝ) (h1 : n1 = 22) (h2 : n2 = 8) (h3 : w1 = 50.25) (h4 : w2 = 45.15) :
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l2993_299342


namespace NUMINAMATH_CALUDE_singer_songs_released_l2993_299310

/-- Given a singer's work schedule and total time spent, calculate the number of songs released --/
theorem singer_songs_released 
  (hours_per_day : ℕ) 
  (days_per_song : ℕ) 
  (total_hours : ℕ) 
  (h1 : hours_per_day = 10)
  (h2 : days_per_song = 10)
  (h3 : total_hours = 300) :
  total_hours / (hours_per_day * days_per_song) = 3 := by
  sorry

end NUMINAMATH_CALUDE_singer_songs_released_l2993_299310


namespace NUMINAMATH_CALUDE_g_zero_value_l2993_299361

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 2

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -6

-- Theorem to prove
theorem g_zero_value : g.eval 0 = -3 := by sorry

end NUMINAMATH_CALUDE_g_zero_value_l2993_299361


namespace NUMINAMATH_CALUDE_derivative_sqrt_derivative_log2_l2993_299346

-- Define the derivative of square root
theorem derivative_sqrt (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

-- Define the derivative of log base 2
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_derivative_log2_l2993_299346


namespace NUMINAMATH_CALUDE_expression_equals_one_l2993_299320

theorem expression_equals_one (x : ℝ) 
  (h1 : x^4 + 2*x + 2 ≠ 0) 
  (h2 : x^4 - 2*x + 2 ≠ 0) : 
  ((((x+2)^3 * (x^3-2*x+2)^3) / (x^4+2*x+2)^3)^3 * 
   (((x-2)^3 * (x^3+2*x+2)^3) / (x^4-2*x+2)^3)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2993_299320


namespace NUMINAMATH_CALUDE_random_walk_2d_properties_l2993_299397

-- Define the random walk on a 2D grid
def RandomWalk2D := ℕ × ℕ → ℝ

-- Probability of reaching a specific x-coordinate
def prob_reach_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Expected y-coordinate when reaching a specific x-coordinate
def expected_y_at_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Theorem statement
theorem random_walk_2d_properties (walk : RandomWalk2D) :
  (∀ x : ℕ, prob_reach_x walk x = 1) ∧
  (∀ n : ℕ, expected_y_at_x walk n = n) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_2d_properties_l2993_299397


namespace NUMINAMATH_CALUDE_tan_678_degrees_equals_138_l2993_299347

theorem tan_678_degrees_equals_138 :
  ∃ (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (678 * π / 180) ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_tan_678_degrees_equals_138_l2993_299347


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_two_l2993_299355

theorem sqrt_equality_implies_one_two :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (18 + 8 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_two_l2993_299355


namespace NUMINAMATH_CALUDE_profit_increase_l2993_299333

theorem profit_increase (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := 1.20 * march_profit
  let may_profit := 0.80 * april_profit
  let june_profit := 1.50 * may_profit
  (june_profit - march_profit) / march_profit * 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l2993_299333


namespace NUMINAMATH_CALUDE_isabel_math_pages_l2993_299368

/-- The number of pages of math homework Isabel had -/
def math_pages : ℕ := sorry

/-- The number of pages of reading homework Isabel had -/
def reading_pages : ℕ := 4

/-- The number of problems per page -/
def problems_per_page : ℕ := 5

/-- The total number of problems Isabel had to complete -/
def total_problems : ℕ := 30

/-- Theorem stating that Isabel had 2 pages of math homework -/
theorem isabel_math_pages : math_pages = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_math_pages_l2993_299368


namespace NUMINAMATH_CALUDE_notebook_profit_l2993_299325

/-- Calculates the profit from selling notebooks -/
def calculate_profit (
  num_notebooks : ℕ
  ) (purchase_price : ℚ)
    (sell_price : ℚ) : ℚ :=
  num_notebooks * sell_price - num_notebooks * purchase_price

/-- Proves that the profit from selling 1200 notebooks, 
    purchased at 4 for $5 and sold at 5 for $8, is $420 -/
theorem notebook_profit : 
  calculate_profit 1200 (5/4) (8/5) = 420 := by
  sorry

end NUMINAMATH_CALUDE_notebook_profit_l2993_299325


namespace NUMINAMATH_CALUDE_quadratic_roots_identity_l2993_299311

theorem quadratic_roots_identity (a b c : ℝ) : 
  (∃ x y : ℝ, x = Real.sin (42 * π / 180) ∧ y = Real.sin (48 * π / 180) ∧ 
    (∀ z : ℝ, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y)) →
  b^2 = a^2 + 2*a*c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_identity_l2993_299311


namespace NUMINAMATH_CALUDE_same_color_probability_l2993_299387

/-- The probability of drawing two balls of the same color from a bag containing 3 white balls
    and 2 black balls when 2 balls are randomly drawn at the same time. -/
theorem same_color_probability (total : ℕ) (white : ℕ) (black : ℕ) :
  total = 5 →
  white = 3 →
  black = 2 →
  (Nat.choose white 2 + Nat.choose black 2) / Nat.choose total 2 = 2 / 5 := by
  sorry

#eval Nat.choose 5 2  -- Total number of ways to draw 2 balls from 5
#eval Nat.choose 3 2  -- Number of ways to draw 2 white balls
#eval Nat.choose 2 2  -- Number of ways to draw 2 black balls

end NUMINAMATH_CALUDE_same_color_probability_l2993_299387


namespace NUMINAMATH_CALUDE_price_increase_l2993_299315

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + 30 / 100) = 1 + 62.5 / 100 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l2993_299315


namespace NUMINAMATH_CALUDE_certain_number_proof_l2993_299360

theorem certain_number_proof (N : ℝ) : 
  (1/2)^22 * N^11 = 1/(18^22) → N = 81 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2993_299360


namespace NUMINAMATH_CALUDE_language_letters_l2993_299372

theorem language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 129 → n = 65 := by
  sorry

end NUMINAMATH_CALUDE_language_letters_l2993_299372


namespace NUMINAMATH_CALUDE_expected_heads_is_75_l2993_299356

/-- The number of coins -/
def num_coins : ℕ := 80

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip of a fair coin -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads at least once in up to four flips -/
def p_heads_in_four_flips : ℚ := 1 - (1 - p_heads)^max_flips

/-- The expected number of coins showing heads after all tosses -/
def expected_heads : ℚ := num_coins * p_heads_in_four_flips

theorem expected_heads_is_75 : expected_heads = 75 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_75_l2993_299356


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l2993_299324

/-- Given a conic section with equation x²/m + y² = 1 and eccentricity √7, prove that m = -6 -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2 = 1) →  -- Condition 1: Conic section equation
  (∃ (e : ℝ), e = Real.sqrt 7 ∧ e^2 = (1 - m)/1) →  -- Condition 2: Eccentricity
  m = -6 := by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l2993_299324


namespace NUMINAMATH_CALUDE_central_angle_values_l2993_299358

/-- A circular sector with perimeter p and area a -/
structure CircularSector where
  p : ℝ  -- perimeter
  a : ℝ  -- area
  h_p_pos : p > 0
  h_a_pos : a > 0

/-- The central angle (in radians) of a circular sector -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.p ∧ 1/2 * r^2 * θ = s.a}

/-- Theorem: For a circular sector with perimeter 6 and area 2, 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_p : s.p = 6) (h_a : s.a = 2) : 
  central_angle s = {1, 4} := by sorry

end NUMINAMATH_CALUDE_central_angle_values_l2993_299358


namespace NUMINAMATH_CALUDE_hotel_payment_ratio_l2993_299334

/-- Given a hotel with operations expenses and a loss, compute the ratio of total payments to operations cost -/
theorem hotel_payment_ratio (operations_cost loss : ℚ) 
  (h1 : operations_cost = 100)
  (h2 : loss = 25) :
  (operations_cost - loss) / operations_cost = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hotel_payment_ratio_l2993_299334


namespace NUMINAMATH_CALUDE_old_socks_thrown_away_l2993_299338

def initial_socks : ℕ := 11
def new_socks : ℕ := 26
def final_socks : ℕ := 33

theorem old_socks_thrown_away : 
  initial_socks + new_socks - final_socks = 4 := by
  sorry

end NUMINAMATH_CALUDE_old_socks_thrown_away_l2993_299338


namespace NUMINAMATH_CALUDE_polynomial_root_behavior_l2993_299328

def Q (x : ℝ) : ℝ := x^6 - 6*x^5 + 10*x^4 - x^3 - x + 12

theorem polynomial_root_behavior :
  (∀ x < 0, Q x ≠ 0) ∧ (∃ x > 0, Q x = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_behavior_l2993_299328


namespace NUMINAMATH_CALUDE_path_length_squares_l2993_299362

/-- Given a line PQ of length 24 cm divided into six equal parts, with squares drawn on each part,
    the path following three sides of each square from P to Q is 72 cm long. -/
theorem path_length_squares (PQ : ℝ) (num_parts : ℕ) : 
  PQ = 24 →
  num_parts = 6 →
  (num_parts : ℝ) * (3 * (PQ / num_parts)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_path_length_squares_l2993_299362


namespace NUMINAMATH_CALUDE_train_passing_time_l2993_299321

/-- Proves the time it takes for a train to pass a stationary point given its speed and time to cross a platform of known length -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) : 
  train_speed_kmph = 72 → 
  platform_length = 260 → 
  platform_crossing_time = 30 → 
  (platform_length + (train_speed_kmph * 1000 / 3600 * platform_crossing_time)) / (train_speed_kmph * 1000 / 3600) = 17 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2993_299321


namespace NUMINAMATH_CALUDE_matrix_commute_l2993_299369

theorem matrix_commute (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![5, 1; -2, 4]) : 
  D * C = !![5, 1; -2, 4] := by sorry

end NUMINAMATH_CALUDE_matrix_commute_l2993_299369


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2993_299365

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 2 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/9 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2993_299365


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2993_299391

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (81/16 : ℚ) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2993_299391


namespace NUMINAMATH_CALUDE_sum_first_third_is_five_l2993_299337

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 2
  inverse_sum : 1 / a 1 + 1 / a 3 = 5 / 4

/-- The sum of the first and third terms of the geometric sequence is 5 -/
theorem sum_first_third_is_five (seq : GeometricSequence) : seq.a 1 + seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_third_is_five_l2993_299337


namespace NUMINAMATH_CALUDE_fourth_seat_is_19_l2993_299339

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 3 → ℕ
  hclass_size : class_size = 52
  hsample_size : sample_size = 4
  hknown_seats : known_seats = ![6, 32, 45]

/-- The step size in systematic sampling -/
def step_size (s : SystematicSample) : ℕ := s.class_size / s.sample_size

/-- The first seat number in the systematic sample -/
def first_seat (s : SystematicSample) : ℕ := 19

/-- Theorem stating that the fourth seat in the systematic sample is 19 -/
theorem fourth_seat_is_19 (s : SystematicSample) : first_seat s = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_seat_is_19_l2993_299339


namespace NUMINAMATH_CALUDE_pressure_volume_relation_l2993_299316

-- Define the constants for the problem
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3
def final_volume : ℝ := 6

-- Define the theorem
theorem pressure_volume_relation :
  ∀ (p1 p2 v1 v2 : ℝ),
    p1 > 0 → p2 > 0 → v1 > 0 → v2 > 0 →
    p1 = initial_pressure →
    v1 = initial_volume →
    v2 = final_volume →
    (p1 * v1 = p2 * v2) →
    p2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pressure_volume_relation_l2993_299316


namespace NUMINAMATH_CALUDE_child_ticket_price_l2993_299386

theorem child_ticket_price
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : total_tickets = 900)
  (h3 : total_revenue = 5100)
  (h4 : child_tickets = 400)
  : ∃ (child_price : ℕ),
    child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧
    child_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_l2993_299386


namespace NUMINAMATH_CALUDE_apple_orange_cost_l2993_299343

/-- The cost of oranges and apples in two scenarios -/
theorem apple_orange_cost (orange_cost apple_cost : ℝ) : 
  orange_cost = 29 →
  apple_cost = 29 →
  6 * orange_cost + 8 * apple_cost = 419 →
  5 * orange_cost + 7 * apple_cost = 488 →
  8 = ⌊(419 - 6 * orange_cost) / apple_cost⌋ := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_cost_l2993_299343


namespace NUMINAMATH_CALUDE_age_ratio_after_15_years_l2993_299340

/-- Represents the ages of a father and his children -/
structure FamilyAges where
  fatherAge : ℕ
  childrenAgesSum : ℕ

/-- Theorem about the ratio of ages after 15 years -/
theorem age_ratio_after_15_years (family : FamilyAges) 
  (h1 : family.fatherAge = family.childrenAgesSum)
  (h2 : family.fatherAge = 75) :
  (family.childrenAgesSum + 5 * 15) / (family.fatherAge + 15) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_15_years_l2993_299340


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2993_299323

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

theorem sufficient_not_necessary_condition (m : ℝ) :
  (m < 1 → ∃ x, f m x = 0) ∧
  ¬(∀ m, (∃ x, f m x = 0) → m < 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2993_299323


namespace NUMINAMATH_CALUDE_robin_candy_count_l2993_299300

/-- Given Robin's initial candy count, the number she ate, and the number her sister gave her, 
    her final candy count is equal to 37. -/
theorem robin_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) 
    (h1 : initial = 23) 
    (h2 : eaten = 7) 
    (h3 : received = 21) : 
  initial - eaten + received = 37 := by
  sorry

end NUMINAMATH_CALUDE_robin_candy_count_l2993_299300


namespace NUMINAMATH_CALUDE_odd_painted_faces_count_l2993_299326

/-- Represents a cube with its number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Represents the block of cubes -/
def Block := List Cube

/-- Creates a 6x6x1 block of painted cubes -/
def create_block : Block :=
  sorry

/-- Counts the number of cubes with an odd number of painted faces -/
def count_odd_painted (block : Block) : Nat :=
  sorry

/-- Theorem stating that the number of cubes with an odd number of painted faces is 16 -/
theorem odd_painted_faces_count (block : Block) : 
  block = create_block → count_odd_painted block = 16 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_count_l2993_299326


namespace NUMINAMATH_CALUDE_unit_digit_of_product_is_zero_l2993_299305

/-- Get the unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given numbers -/
def productOfNumbers : ℕ := 785846 * 1086432 * 4582735 * 9783284 * 5167953 * 3821759 * 7594683

theorem unit_digit_of_product_is_zero :
  unitDigit productOfNumbers = 0 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_is_zero_l2993_299305


namespace NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degrees_l2993_299329

theorem scalene_triangle_with_double_angle_and_36_degrees :
  ∀ (x y z : ℝ),
  0 < x ∧ 0 < y ∧ 0 < z →  -- angles are positive
  x < y ∧ y < z →  -- scalene triangle condition
  x + y + z = 180 →  -- sum of angles in a triangle
  (x = 36 ∨ y = 36 ∨ z = 36) →  -- one angle is 36°
  (x = 2*y ∨ y = 2*x ∨ y = 2*z ∨ z = 2*x ∨ z = 2*y) →  -- one angle is double another
  ((x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126)) := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degrees_l2993_299329


namespace NUMINAMATH_CALUDE_min_p_plus_q_min_p_plus_q_value_l2993_299399

theorem min_p_plus_q (p q : ℕ+) (h : 90 * p = q^3) : 
  ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q' :=
by sorry

theorem min_p_plus_q_value (p q : ℕ+) (h : 90 * p = q^3) 
  (h_min : ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q') : 
  p + q = 330 :=
by sorry

end NUMINAMATH_CALUDE_min_p_plus_q_min_p_plus_q_value_l2993_299399


namespace NUMINAMATH_CALUDE_g_ge_f_implies_t_range_l2993_299303

noncomputable def g (x : ℝ) : ℝ := Real.log x + 3 / (4 * x) - (1 / 4) * x - 1

def f (t x : ℝ) : ℝ := x^2 - 2 * t * x + 4

theorem g_ge_f_implies_t_range (t : ℝ) :
  (∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, g x1 ≥ f t x2) →
  t ≥ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_g_ge_f_implies_t_range_l2993_299303


namespace NUMINAMATH_CALUDE_northern_walks_of_length_6_l2993_299375

/-- A northern walk is a path on a grid with the following properties:
  1. It starts at the origin.
  2. Each step is 1 unit north, east, or west.
  3. It never revisits a point.
  4. It has a specified length. -/
def NorthernWalk (length : ℕ) : Type := Unit

/-- Count the number of northern walks of a given length. -/
def countNorthernWalks (length : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 239 northern walks of length 6. -/
theorem northern_walks_of_length_6 : countNorthernWalks 6 = 239 := by sorry

end NUMINAMATH_CALUDE_northern_walks_of_length_6_l2993_299375


namespace NUMINAMATH_CALUDE_event_probability_l2993_299322

theorem event_probability (P_A_and_B P_A_or_B P_B : ℝ) 
  (h1 : P_A_and_B = 0.25)
  (h2 : P_A_or_B = 0.8)
  (h3 : P_B = 0.65) :
  ∃ P_A : ℝ, P_A = 0.4 ∧ P_A_or_B = P_A + P_B - P_A_and_B := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2993_299322


namespace NUMINAMATH_CALUDE_no_valid_reassignment_l2993_299388

/-- Represents a seating arrangement in a classroom -/
structure Classroom :=
  (rows : Nat)
  (cols : Nat)
  (students : Nat)
  (center_empty : Bool)

/-- Checks if a reassignment is possible given the classroom setup -/
def reassignment_possible (c : Classroom) : Prop :=
  c.rows = 5 ∧ c.cols = 7 ∧ c.students = 34 ∧ c.center_empty = true →
  ∃ (new_arrangement : Fin c.students → Fin (c.rows * c.cols)),
    ∀ i : Fin c.students,
      let old_pos := i.val
      let new_pos := (new_arrangement i).val
      (new_pos ≠ old_pos) ∧
      ((new_pos = old_pos + 1 ∨ new_pos = old_pos - 1) ∨
       (new_pos = old_pos + c.cols ∨ new_pos = old_pos - c.cols))

theorem no_valid_reassignment (c : Classroom) :
  ¬(reassignment_possible c) :=
sorry

end NUMINAMATH_CALUDE_no_valid_reassignment_l2993_299388


namespace NUMINAMATH_CALUDE_five_zero_points_l2993_299341

open Set
open Real

noncomputable def f (x : ℝ) := Real.sin (π * Real.cos x)

theorem five_zero_points :
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
  (∀ x ∈ s, x ∈ Icc 0 (2 * π) ∧ f x = 0) ∧
  (∀ x ∈ Icc 0 (2 * π), f x = 0 → x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_five_zero_points_l2993_299341


namespace NUMINAMATH_CALUDE_random_event_last_third_probability_l2993_299312

/-- The probability of a random event occurring in the last third of a given time interval is 1/3 -/
theorem random_event_last_third_probability (total_interval : ℝ) (h : total_interval > 0) :
  let last_third := total_interval / 3
  (last_third / total_interval) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_random_event_last_third_probability_l2993_299312


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l2993_299304

theorem cubic_equation_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l2993_299304


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_coverage_l2993_299353

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesNeeded (region : Dimensions) (tile : Dimensions) : ℕ :=
  (area region + area tile - 1) / area tile

theorem min_tiles_for_floor_coverage :
  let tile := Dimensions.mk 2 6
  let region := Dimensions.mk (feetToInches 3) (feetToInches 4)
  tilesNeeded region tile = 144 := by
    sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_coverage_l2993_299353


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l2993_299382

def inequality (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

def solution_set (x : ℝ) : Prop :=
  x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l2993_299382


namespace NUMINAMATH_CALUDE_exists_valid_coloring_for_all_k_l2993_299395

/-- A point on an infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black squares on an infinite white grid -/
def BlackSquares := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) -/
structure GridLine where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The number of black squares on a given line -/
def blackSquaresOnLine (blacks : BlackSquares) (line : GridLine) : ℕ :=
  sorry

/-- A valid coloring of the grid for a given k -/
def validColoring (k : ℕ) (blacks : BlackSquares) : Prop :=
  (blacks.Nonempty) ∧
  (∀ line : GridLine, blackSquaresOnLine blacks line = k ∨ blackSquaresOnLine blacks line = 0)

/-- The main theorem: for any positive k, there exists a valid coloring -/
theorem exists_valid_coloring_for_all_k :
  ∀ k : ℕ, k > 0 → ∃ blacks : BlackSquares, validColoring k blacks :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_for_all_k_l2993_299395


namespace NUMINAMATH_CALUDE_square_value_l2993_299398

theorem square_value (x : ℚ) : 
  10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1 → x = 28 := by
sorry

end NUMINAMATH_CALUDE_square_value_l2993_299398


namespace NUMINAMATH_CALUDE_two_player_three_point_probability_l2993_299377

/-- The probability that at least one of two players makes both of their two three-point shots -/
theorem two_player_three_point_probability (p_a p_b : ℝ) 
  (h_a : p_a = 0.4) (h_b : p_b = 0.5) : 
  1 - (1 - p_a^2) * (1 - p_b^2) = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_two_player_three_point_probability_l2993_299377


namespace NUMINAMATH_CALUDE_only_point_0_neg2_satisfies_l2993_299332

def point_satisfies_inequalities (x y : ℝ) : Prop :=
  x + y - 1 < 0 ∧ x - y + 1 > 0

theorem only_point_0_neg2_satisfies : 
  ¬(point_satisfies_inequalities 0 2) ∧
  ¬(point_satisfies_inequalities (-2) 0) ∧
  point_satisfies_inequalities 0 (-2) ∧
  ¬(point_satisfies_inequalities 2 0) :=
sorry

end NUMINAMATH_CALUDE_only_point_0_neg2_satisfies_l2993_299332


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l2993_299383

/-- The total number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem stating that Melanie's total dimes is the sum of her initial dimes and those received from her parents -/
theorem melanie_dimes_count (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_dimes initial from_dad from_mom = initial + from_dad + from_mom :=
by
  sorry

#eval total_dimes 19 39 25

end NUMINAMATH_CALUDE_melanie_dimes_count_l2993_299383


namespace NUMINAMATH_CALUDE_complex_roots_count_l2993_299376

theorem complex_roots_count (z : ℂ) : 
  let θ := Complex.arg z
  (Complex.abs z = 1) →
  ((z ^ (7 * 6 * 5 * 4 * 3 * 2 * 1) - z ^ (6 * 5 * 4 * 3 * 2 * 1)).im = 0) →
  ((z ^ (6 * 5 * 4 * 3 * 2 * 1) - z ^ (5 * 4 * 3 * 2 * 1)).im = 0) →
  (0 ≤ θ) →
  (θ < 2 * Real.pi) →
  (Real.cos (4320 * θ) = 0 ∨ Real.sin (3360 * θ) = 0) →
  (Real.cos (420 * θ) = 0 ∨ Real.sin (300 * θ) = 0) →
  Nat := by sorry

end NUMINAMATH_CALUDE_complex_roots_count_l2993_299376


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2993_299313

theorem min_value_of_expression (x : ℝ) :
  ∃ (min_val : ℝ), min_val = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2993_299313


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2993_299366

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : ∃ (d : ℝ),
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 10*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 8*x + y^2 - 2*y + 16 = 0}
  d = Real.sqrt 85 - 6 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    d ≤ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2993_299366


namespace NUMINAMATH_CALUDE_total_canoes_april_l2993_299331

def canoe_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else initial * (3^months - 1) / 2

theorem total_canoes_april : canoe_production 5 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_april_l2993_299331


namespace NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l2993_299348

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ
  edges : ℕ

/-- Theorem: A prism with 5 faces has 9 edges. -/
theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : p.edges = 9 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l2993_299348


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2993_299335

/-- The quadratic function f(c) = 3/4*c^2 - 6c + 4 is minimized when c = 4 -/
theorem quadratic_minimum : ∃ (c : ℝ), ∀ (x : ℝ), (3/4 : ℝ) * c^2 - 6*c + 4 ≤ (3/4 : ℝ) * x^2 - 6*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2993_299335


namespace NUMINAMATH_CALUDE_teddy_cats_count_l2993_299367

/-- Prove that Teddy has 8 cats given the conditions of the problem -/
theorem teddy_cats_count :
  -- Teddy's dogs
  let teddy_dogs : ℕ := 7
  -- Ben's dogs relative to Teddy's
  let ben_dogs : ℕ := teddy_dogs + 9
  -- Dave's dogs relative to Teddy's
  let dave_dogs : ℕ := teddy_dogs - 5
  -- Dave's cats relative to Teddy's
  let dave_cats (teddy_cats : ℕ) : ℕ := teddy_cats + 13
  -- Total pets
  let total_pets : ℕ := 54
  -- The number of Teddy's cats that satisfies all conditions
  ∃ (teddy_cats : ℕ),
    teddy_dogs + ben_dogs + dave_dogs + teddy_cats + dave_cats teddy_cats = total_pets ∧
    teddy_cats = 8 := by
  sorry

end NUMINAMATH_CALUDE_teddy_cats_count_l2993_299367


namespace NUMINAMATH_CALUDE_five_people_handshakes_l2993_299302

/-- The number of handshakes in a group of n people where each person
    shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 5 people, where each person shakes hands with
    every other person exactly once, the total number of handshakes is 10 -/
theorem five_people_handshakes :
  handshakes 5 = 10 := by
  sorry

#eval handshakes 5  -- To verify the result

end NUMINAMATH_CALUDE_five_people_handshakes_l2993_299302


namespace NUMINAMATH_CALUDE_isabel_games_l2993_299307

/-- The number of DS games Isabel had initially -/
def initial_games : ℕ := 90

/-- The number of DS games Isabel gave away -/
def games_given_away : ℕ := 87

/-- The number of DS games Isabel has left -/
def games_left : ℕ := 3

/-- Theorem stating that the initial number of games is equal to the sum of games given away and games left -/
theorem isabel_games : initial_games = games_given_away + games_left := by
  sorry

end NUMINAMATH_CALUDE_isabel_games_l2993_299307


namespace NUMINAMATH_CALUDE_root_ratio_implies_k_value_l2993_299359

theorem root_ratio_implies_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 10*x + k = 0 ∧ 
   y^2 + 10*y + k = 0 ∧
   x / y = 3) →
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_implies_k_value_l2993_299359


namespace NUMINAMATH_CALUDE_jungkook_weight_proof_l2993_299306

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

/-- Jungkook's base weight in kilograms -/
def base_weight_kg : ℕ := 54

/-- Additional weight in grams -/
def additional_weight_g : ℕ := 154

/-- Jungkook's total weight in grams -/
def jungkook_weight_g : ℕ := base_weight_kg * kg_to_g + additional_weight_g

theorem jungkook_weight_proof : jungkook_weight_g = 54154 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_weight_proof_l2993_299306


namespace NUMINAMATH_CALUDE_inequality_proof_l2993_299374

theorem inequality_proof (a d b c : ℝ) 
  (h1 : a ≥ 0) (h2 : d ≥ 0) (h3 : b > 0) (h4 : c > 0) (h5 : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (b + a)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2993_299374


namespace NUMINAMATH_CALUDE_cone_no_rectangular_front_view_l2993_299345

-- Define the types of solids
inductive Solid
  | Cube
  | RegularTriangularPrism
  | Cylinder
  | Cone

-- Define a property for having a rectangular front view
def has_rectangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cube => True
  | Solid.RegularTriangularPrism => True
  | Solid.Cylinder => True
  | Solid.Cone => False

-- Theorem statement
theorem cone_no_rectangular_front_view :
  ∀ s : Solid, ¬(has_rectangular_front_view s) ↔ s = Solid.Cone :=
sorry

end NUMINAMATH_CALUDE_cone_no_rectangular_front_view_l2993_299345


namespace NUMINAMATH_CALUDE_product_base_conversion_l2993_299318

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat := sorry

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat := sorry

theorem product_base_conversion :
  let binary := [true, true, false, true]  -- 1101 in base 2
  let ternary := [2, 0, 2]  -- 202 in base 3
  (base2To10 binary) * (base3To10 ternary) = 260 := by sorry

end NUMINAMATH_CALUDE_product_base_conversion_l2993_299318


namespace NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2993_299379

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 4)
  (h_sum_y : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2993_299379


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2993_299344

/-- Given that x₁ and x₂ are real roots of the equation x² - (k-2)x + (k² + 3k + 5) = 0,
    where k is a real number, prove that the maximum value of x₁² + x₂² is 18. -/
theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) 
    (h₁ : x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0)
    (h₂ : x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0)
    (h₃ : x₁ ≠ x₂) : 
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2993_299344


namespace NUMINAMATH_CALUDE_order_of_zeros_and_roots_l2993_299378

def f (x m n : ℝ) : ℝ := 2 * (x - m) * (x - n) - 7

theorem order_of_zeros_and_roots (m n α β : ℝ) 
  (h1 : m < n) 
  (h2 : α < β) 
  (h3 : f α m n = 0)
  (h4 : f β m n = 0) :
  α < m ∧ m < n ∧ n < β := by sorry

end NUMINAMATH_CALUDE_order_of_zeros_and_roots_l2993_299378


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2993_299371

/-- Given two vectors a and b in ℝ², prove that if (a - b) is perpendicular
    to (m * a + b), then m = 1/4. -/
theorem perpendicular_vectors_m_value 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) 
  (hb : b = (1, -1)) 
  (h_perp : (a.1 - b.1) * (m * a.1 + b.1) + (a.2 - b.2) * (m * a.2 + b.2) = 0) : 
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2993_299371


namespace NUMINAMATH_CALUDE_prob_three_odd_less_than_eighth_l2993_299319

def total_integers : ℕ := 2020
def odd_integers : ℕ := total_integers / 2

theorem prob_three_odd_less_than_eighth :
  let p := (odd_integers : ℚ) / total_integers *
           ((odd_integers - 1) : ℚ) / (total_integers - 1) *
           ((odd_integers - 2) : ℚ) / (total_integers - 2)
  p < 1 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_three_odd_less_than_eighth_l2993_299319


namespace NUMINAMATH_CALUDE_function_describes_relationship_l2993_299390

-- Define the set of x values
def X : Set ℕ := {1, 2, 3, 4, 5}

-- Define the function f
def f (x : ℕ) : ℕ := x^2

-- Define the set of points (x, y)
def points : Set (ℕ × ℕ) := {(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)}

-- Theorem statement
theorem function_describes_relationship :
  ∀ (x : ℕ), x ∈ X → (x, f x) ∈ points := by
  sorry

end NUMINAMATH_CALUDE_function_describes_relationship_l2993_299390


namespace NUMINAMATH_CALUDE_unicorn_journey_flowers_l2993_299350

/-- The number of flowers that bloom when unicorns walk across a forest -/
def flowers_bloomed (num_unicorns : ℕ) (distance_km : ℕ) (step_length_m : ℕ) (flowers_per_step : ℕ) : ℕ :=
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step

/-- Proof that 6 unicorns walking 9 km with 3m steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers : flowers_bloomed 6 9 3 4 = 72000 := by
  sorry

#eval flowers_bloomed 6 9 3 4

end NUMINAMATH_CALUDE_unicorn_journey_flowers_l2993_299350


namespace NUMINAMATH_CALUDE_maria_chairs_l2993_299373

/-- The number of chairs Maria bought -/
def num_chairs : ℕ := 2

/-- The number of tables Maria bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 32

theorem maria_chairs :
  num_chairs * time_per_furniture + num_tables * time_per_furniture = total_time :=
by sorry

end NUMINAMATH_CALUDE_maria_chairs_l2993_299373


namespace NUMINAMATH_CALUDE_smallest_product_l2993_299317

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l2993_299317


namespace NUMINAMATH_CALUDE_integer_power_sum_l2993_299349

theorem integer_power_sum (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l2993_299349


namespace NUMINAMATH_CALUDE_product_of_solutions_l2993_299327

theorem product_of_solutions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1905) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1910)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1905) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1910)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1905) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1910) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/191 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2993_299327


namespace NUMINAMATH_CALUDE_latest_time_60_degrees_l2993_299314

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 40

-- State the theorem
theorem latest_time_60_degrees :
  ∃ t : ℝ, t ≤ 12 ∧ t ≥ 0 ∧ temperature t = 60 ∧
  ∀ s : ℝ, s > t ∧ s ≥ 0 → temperature s ≠ 60 :=
by sorry

end NUMINAMATH_CALUDE_latest_time_60_degrees_l2993_299314


namespace NUMINAMATH_CALUDE_b_oxen_count_l2993_299364

/-- Represents the number of oxen-months for a person's contribution -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents A's contribution in oxen-months -/
def a_contribution : ℕ := oxen_months 10 7

/-- Represents C's contribution in oxen-months -/
def c_contribution : ℕ := oxen_months 15 3

/-- Represents C's share of the rent -/
def c_share : ℕ := 45

/-- Represents the number of months B's oxen grazed -/
def b_months : ℕ := 5

/-- Theorem stating that B put 12 oxen for grazing -/
theorem b_oxen_count : 
  ∃ (b_oxen : ℕ), 
    b_oxen = 12 ∧ 
    (c_share : ℚ) / total_rent = 
      (c_contribution : ℚ) / (a_contribution + oxen_months b_oxen b_months + c_contribution) :=
sorry

end NUMINAMATH_CALUDE_b_oxen_count_l2993_299364


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2993_299389

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2993_299389
