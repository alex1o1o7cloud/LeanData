import Mathlib

namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_60_kmph_l1417_141710

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- The speed of the train is 60 kmph given the specified conditions -/
theorem train_speed_is_60_kmph : 
  train_speed 55 3 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_60_kmph_l1417_141710


namespace NUMINAMATH_CALUDE_square_side_length_l1417_141755

/-- Given a rectangle with sides 9 cm and 16 cm and a square with the same area,
    prove that the side length of the square is 12 cm. -/
theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) :
  rectangle_width = 9 →
  rectangle_length = 16 →
  rectangle_width * rectangle_length = square_side * square_side →
  square_side = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1417_141755


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1417_141762

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 13*x^3) 
  (h3 : a - b = 2*x) : 
  (a = x + (Real.sqrt 66 * x) / 6) ∨ (a = x - (Real.sqrt 66 * x) / 6) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1417_141762


namespace NUMINAMATH_CALUDE_perfect_squares_difference_99_l1417_141720

theorem perfect_squares_difference_99 :
  ∃! (l : List ℕ), 
    (∀ x ∈ l, ∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) ∧ 
    (∀ x : ℕ, (∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) → x ∈ l) ∧
    l.length = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_99_l1417_141720


namespace NUMINAMATH_CALUDE_saree_price_calculation_l1417_141779

/-- Calculates the final price after applying multiple discounts and a tax rate -/
def finalPrice (originalPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := discounts.foldl (fun price discount => price * (1 - discount)) originalPrice
  discountedPrice * (1 + taxRate)

/-- Theorem: The final price of a 510 Rs item after specific discounts and tax is approximately 302.13 Rs -/
theorem saree_price_calculation :
  let originalPrice : ℝ := 510
  let discounts : List ℝ := [0.12, 0.15, 0.20, 0.10]
  let taxRate : ℝ := 0.10
  abs (finalPrice originalPrice discounts taxRate - 302.13) < 0.01 := by
  sorry

#eval finalPrice 510 [0.12, 0.15, 0.20, 0.10] 0.10

end NUMINAMATH_CALUDE_saree_price_calculation_l1417_141779


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1417_141752

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- State the theorem
theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x^2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1417_141752


namespace NUMINAMATH_CALUDE_max_ab_max_expression_min_sum_l1417_141701

-- Define the conditions
def is_valid_pair (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Theorem 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h : is_valid_pair a b) :
  a * b ≤ 1/4 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ a₀ * b₀ = 1/4 :=
sorry

-- Theorem 2: Maximum value of 4a - 1/(4b)
theorem max_expression (a b : ℝ) (h : is_valid_pair a b) :
  4*a - 1/(4*b) ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 4*a₀ - 1/(4*b₀) = 2 :=
sorry

-- Theorem 3: Minimum value of 1/a + 2/b
theorem min_sum (a b : ℝ) (h : is_valid_pair a b) :
  1/a + 2/b ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 1/a₀ + 2/b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_max_expression_min_sum_l1417_141701


namespace NUMINAMATH_CALUDE_magazine_circulation_ratio_l1417_141738

/-- Given a magazine's circulation data, proves the ratio of circulation in 1961 to total circulation from 1961-1970 -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h1 : avg_circulation > 0) -- Assumption that average circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_magazine_circulation_ratio_l1417_141738


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l1417_141700

theorem hexagon_largest_angle (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 2160 / 11 := by
sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l1417_141700


namespace NUMINAMATH_CALUDE_two_car_efficiency_l1417_141787

/-- Two-car family fuel efficiency problem -/
theorem two_car_efficiency (mpg1 : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (gallons1 : ℝ) :
  mpg1 = 25 →
  total_miles = 1825 →
  total_gallons = 55 →
  gallons1 = 30 →
  (total_miles - mpg1 * gallons1) / (total_gallons - gallons1) = 43 := by
sorry

end NUMINAMATH_CALUDE_two_car_efficiency_l1417_141787


namespace NUMINAMATH_CALUDE_die_roll_prime_probability_l1417_141743

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (x y : ℕ) : Prop := is_prime (x + y)

def count_prime_sums : ℕ := 22

def total_outcomes : ℕ := 48

theorem die_roll_prime_probability :
  (count_prime_sums : ℚ) / total_outcomes = 11 / 24 := by sorry

end NUMINAMATH_CALUDE_die_roll_prime_probability_l1417_141743


namespace NUMINAMATH_CALUDE_correct_annual_take_home_pay_l1417_141741

def annual_take_home_pay (hourly_rate : ℝ) (regular_hours_per_week : ℝ) (weeks_per_year : ℝ)
  (overtime_hours_per_quarter : ℝ) (overtime_rate_multiplier : ℝ)
  (federal_tax_rate_1 : ℝ) (federal_tax_threshold_1 : ℝ)
  (federal_tax_rate_2 : ℝ) (federal_tax_threshold_2 : ℝ)
  (state_tax_rate : ℝ) (unemployment_insurance_rate : ℝ)
  (unemployment_insurance_threshold : ℝ) (social_security_rate : ℝ)
  (social_security_threshold : ℝ) : ℝ :=
  sorry

theorem correct_annual_take_home_pay :
  annual_take_home_pay 15 40 52 20 1.5 0.1 10000 0.12 30000 0.05 0.01 7000 0.062 142800 = 25474 :=
by sorry

end NUMINAMATH_CALUDE_correct_annual_take_home_pay_l1417_141741


namespace NUMINAMATH_CALUDE_expression_value_l1417_141702

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1417_141702


namespace NUMINAMATH_CALUDE_checkers_games_theorem_l1417_141723

theorem checkers_games_theorem (games_friend1 games_friend2 : ℕ) 
  (h1 : games_friend1 = 25) 
  (h2 : games_friend2 = 17) : 
  (∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 34) ∧ 
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 35) ∧
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 56) :=
by sorry

#check checkers_games_theorem

end NUMINAMATH_CALUDE_checkers_games_theorem_l1417_141723


namespace NUMINAMATH_CALUDE_sum_product_reciprocal_sum_squared_inequality_l1417_141739

theorem sum_product_reciprocal_sum_squared_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a*b + b*c + c*a) * (1/(a+b)^2 + 1/(b+c)^2 + 1/(c+a)^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_reciprocal_sum_squared_inequality_l1417_141739


namespace NUMINAMATH_CALUDE_new_average_production_theorem_l1417_141784

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageDailyProduction (n : ℕ) (pastAverage : ℚ) (todayProduction : ℚ) : ℚ :=
  ((n : ℚ) * pastAverage + todayProduction) / ((n : ℚ) + 1)

theorem new_average_production_theorem :
  let n : ℕ := 8
  let pastAverage : ℚ := 50
  let todayProduction : ℚ := 95
  newAverageDailyProduction n pastAverage todayProduction = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_theorem_l1417_141784


namespace NUMINAMATH_CALUDE_sum_in_base7_l1417_141707

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of 666₇, 66₇, and 6₇ in base 7 is 1400₇ -/
theorem sum_in_base7 : 
  base10ToBase7 (base7ToBase10 666 + base7ToBase10 66 + base7ToBase10 6) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l1417_141707


namespace NUMINAMATH_CALUDE_fraction_difference_2023_2022_l1417_141783

theorem fraction_difference_2023_2022 : 
  (2023 : ℚ) / 2022 - 2022 / 2023 = 4045 / (2022 * 2023) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_2023_2022_l1417_141783


namespace NUMINAMATH_CALUDE_unique_line_through_point_l1417_141722

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_line_through_point :
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    is_prime b ∧ 
    (6 : ℚ) / a + (5 : ℚ) / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_through_point_l1417_141722


namespace NUMINAMATH_CALUDE_dice_sum_product_l1417_141776

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 120 →
  a + b + c + d ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_product_l1417_141776


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1417_141758

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 2 ∧ 
   x₁^2 + 2*p*x₁ + q = 0 ∧ x₂^2 + 2*p*x₂ + q = 0) ↔ 
  (q > 0 ∧ p < -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1417_141758


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1417_141754

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1417_141754


namespace NUMINAMATH_CALUDE_eighth_number_with_digit_sum_13_l1417_141725

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns whether a natural number has a digit sum of 13 -/
def has_digit_sum_13 (n : ℕ) : Prop := digit_sum n = 13

/-- A function that returns the nth positive integer with digit sum 13 -/
def nth_digit_sum_13 (n : ℕ) : ℕ := sorry

theorem eighth_number_with_digit_sum_13 : nth_digit_sum_13 8 = 148 := by sorry

end NUMINAMATH_CALUDE_eighth_number_with_digit_sum_13_l1417_141725


namespace NUMINAMATH_CALUDE_shopping_cost_theorem_l1417_141747

/-- Calculates the total cost of Fabian's shopping trip --/
def calculate_shopping_cost (
  apple_price : ℝ)
  (walnut_price : ℝ)
  (orange_price : ℝ)
  (pasta_price : ℝ)
  (sugar_discount : ℝ)
  (orange_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let apple_cost := 5 * apple_price
  let sugar_cost := 3 * (apple_price - sugar_discount)
  let walnut_cost := 0.5 * walnut_price
  let orange_cost := 2 * orange_price * (1 - orange_discount)
  let pasta_cost := 3 * pasta_price
  let total_before_tax := apple_cost + sugar_cost + walnut_cost + orange_cost + pasta_cost
  total_before_tax * (1 + sales_tax)

/-- The theorem stating the total cost of Fabian's shopping --/
theorem shopping_cost_theorem :
  calculate_shopping_cost 2 6 3 1.5 1 0.1 0.05 = 27.20 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_theorem_l1417_141747


namespace NUMINAMATH_CALUDE_students_per_table_unchanged_l1417_141711

/-- Proves that the number of students per table remains the same when evenly dividing the total number of students across all tables. -/
theorem students_per_table_unchanged 
  (initial_students_per_table : ℝ) 
  (num_tables : ℝ) 
  (h1 : initial_students_per_table = 6.0)
  (h2 : num_tables = 34.0) :
  let total_students := initial_students_per_table * num_tables
  total_students / num_tables = initial_students_per_table := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_unchanged_l1417_141711


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l1417_141721

theorem exterior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_exterior_angles : ℝ),
  n = 8 → 
  sum_exterior_angles = 360 →
  (sum_exterior_angles / n : ℝ) = 45 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l1417_141721


namespace NUMINAMATH_CALUDE_unique_intersection_k_values_l1417_141781

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the theorem
theorem unique_intersection_k_values :
  ∃! z, equation1 z ∧ equation2 z k → k = 0.631 ∨ k = 25.369 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_k_values_l1417_141781


namespace NUMINAMATH_CALUDE_compare_polynomial_expressions_l1417_141771

theorem compare_polynomial_expressions {a b c : ℝ} (h1 : a > b) (h2 : b > c) :
  a^2*b + b^2*c + c^2*a > a*b^2 + b*c^2 + c*a^2 := by
  sorry

end NUMINAMATH_CALUDE_compare_polynomial_expressions_l1417_141771


namespace NUMINAMATH_CALUDE_reef_age_conversion_l1417_141780

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

theorem reef_age_conversion :
  octal_to_decimal 367 = 247 := by
  sorry

end NUMINAMATH_CALUDE_reef_age_conversion_l1417_141780


namespace NUMINAMATH_CALUDE_mistaken_calculation_system_l1417_141718

theorem mistaken_calculation_system (x y : ℝ) : 
  (5/4 * x = 4/5 * x + 36) ∧ 
  (7/3 * y = 3/7 * y + 28) → 
  x = 80 ∧ y = 14.7 := by
sorry

end NUMINAMATH_CALUDE_mistaken_calculation_system_l1417_141718


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l1417_141740

/-- The initial number of bottle caps Joshua had -/
def initial_caps : ℕ := 40

/-- The number of bottle caps Joshua bought -/
def bought_caps : ℕ := 7

/-- The total number of bottle caps Joshua has after buying more -/
def total_caps : ℕ := 47

theorem joshua_bottle_caps : initial_caps + bought_caps = total_caps := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l1417_141740


namespace NUMINAMATH_CALUDE_base_number_proof_l1417_141753

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^18) 
  (h2 : n = 17) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l1417_141753


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1417_141729

theorem quadratic_roots_property (a b : ℝ) : 
  a ≠ b ∧ 
  a^2 + 3*a - 5 = 0 ∧ 
  b^2 + 3*b - 5 = 0 → 
  a^2 + 3*a*b + a - 2*b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1417_141729


namespace NUMINAMATH_CALUDE_finite_solutions_of_system_l1417_141727

theorem finite_solutions_of_system (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
    x * y + z * w = a ∧ x * z + y * w = b → (x, y, z, w) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_of_system_l1417_141727


namespace NUMINAMATH_CALUDE_trig_identity_l1417_141717

theorem trig_identity (α : ℝ) : 
  Real.sin (9 * α) + Real.sin (10 * α) + Real.sin (11 * α) + Real.sin (12 * α) = 
  4 * Real.cos (α / 2) * Real.cos α * Real.sin ((21 * α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1417_141717


namespace NUMINAMATH_CALUDE_median_in_middle_interval_l1417_141793

/-- Represents the intervals of scores -/
inductive ScoreInterval
| I60to64
| I65to69
| I70to74
| I75to79
| I80to84

/-- The total number of students -/
def totalStudents : ℕ := 100

/-- The number of intervals -/
def numIntervals : ℕ := 5

/-- The number of students in each interval -/
def studentsPerInterval : ℕ := totalStudents / numIntervals

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the middle interval -/
theorem median_in_middle_interval :
  medianPosition > 2 * studentsPerInterval ∧
  medianPosition ≤ 3 * studentsPerInterval :=
sorry

end NUMINAMATH_CALUDE_median_in_middle_interval_l1417_141793


namespace NUMINAMATH_CALUDE_passing_marks_l1417_141795

theorem passing_marks (T : ℝ) 
  (h1 : 0.3 * T + 60 = 0.4 * T) 
  (h2 : 0.5 * T = 0.4 * T + 40) : 
  0.4 * T = 240 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_l1417_141795


namespace NUMINAMATH_CALUDE_octagon_area_l1417_141708

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 1024 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l1417_141708


namespace NUMINAMATH_CALUDE_cos_two_alpha_l1417_141772

theorem cos_two_alpha (α : Real) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l1417_141772


namespace NUMINAMATH_CALUDE_hcf_problem_l1417_141798

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.lcm a b = 1485) :
  Nat.gcd a b = 12 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l1417_141798


namespace NUMINAMATH_CALUDE_car_speed_percentage_increase_l1417_141735

/-- Proves that given two cars driving toward each other, with the first car traveling at 100 km/h,
    a distance of 720 km between them, and meeting after 4 hours, the percentage increase in the
    speed of the first car compared to the second car is 25%. -/
theorem car_speed_percentage_increase
  (speed_first : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : speed_first = 100)
  (h2 : distance = 720)
  (h3 : time = 4)
  (h4 : speed_first * time + (distance / time) * time = distance) :
  (speed_first - (distance / time)) / (distance / time) * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_percentage_increase_l1417_141735


namespace NUMINAMATH_CALUDE_no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l1417_141744

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Theorem 1: When m = 1, the equation has no real roots
theorem no_real_roots_when_m_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Theorem 2: Range of m when roots are in specified intervals
theorem m_range_for_specified_root_intervals :
  (∃ x y : ℝ, x ∈ Set.Ioo (-1) 0 ∧ y ∈ Set.Ioo 1 2 ∧ f m x = 0 ∧ f m y = 0) ↔
  m ∈ Set.Ioo (-5/6) (-1/2) := by sorry

end NUMINAMATH_CALUDE_no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l1417_141744


namespace NUMINAMATH_CALUDE_right_triangle_area_l1417_141763

/-- The area of a right triangle with base 12 and height 15 is 90 -/
theorem right_triangle_area : ∀ (base height area : ℝ),
  base = 12 →
  height = 15 →
  area = (1/2) * base * height →
  area = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1417_141763


namespace NUMINAMATH_CALUDE_circle_area_decrease_l1417_141799

theorem circle_area_decrease (r : ℝ) (h : r > 0) : 
  let r' := 0.8 * r
  let A := π * r^2
  let A' := π * r'^2
  (A - A') / A = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l1417_141799


namespace NUMINAMATH_CALUDE_three_roots_implies_a_range_l1417_141714

theorem three_roots_implies_a_range (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x : ℝ, x^2 = a * Real.exp x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  0 < a ∧ a < 4 / Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_three_roots_implies_a_range_l1417_141714


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l1417_141765

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) → 
  -3 ≤ m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l1417_141765


namespace NUMINAMATH_CALUDE_pythagorean_triple_square_l1417_141767

theorem pythagorean_triple_square (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  ∃ m n : ℤ, (1/2 : ℚ) * ((c : ℚ) - (a : ℚ)) * ((c : ℚ) - (b : ℚ)) = (n^2 * (m - n)^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_square_l1417_141767


namespace NUMINAMATH_CALUDE_paper_recycling_trees_saved_l1417_141703

theorem paper_recycling_trees_saved 
  (trees_per_tonne : ℕ) 
  (schools : ℕ) 
  (paper_per_school : ℚ) 
  (h1 : trees_per_tonne = 24)
  (h2 : schools = 4)
  (h3 : paper_per_school = 3/4) : 
  ↑schools * paper_per_school * trees_per_tonne = 72 := by
  sorry

end NUMINAMATH_CALUDE_paper_recycling_trees_saved_l1417_141703


namespace NUMINAMATH_CALUDE_largest_five_digit_integer_l1417_141778

def digit_product (n : ℕ) : ℕ := 
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem largest_five_digit_integer : 
  ∀ n : ℕ, 
    n ≤ 99999 ∧ 
    n ≥ 10000 ∧ 
    digit_product n = 40320 ∧ 
    digit_sum n < 35 → 
    n ≤ 98764 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_integer_l1417_141778


namespace NUMINAMATH_CALUDE_two_hundred_squared_minus_399_is_composite_l1417_141785

theorem two_hundred_squared_minus_399_is_composite : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 200^2 - 399 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_two_hundred_squared_minus_399_is_composite_l1417_141785


namespace NUMINAMATH_CALUDE_mandy_med_school_acceptances_l1417_141756

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 96)
  (h2 : applied_fraction = 5 / 8)
  (h3 : accepted_fraction = 3 / 5)
  : ℕ :=
by
  sorry

end NUMINAMATH_CALUDE_mandy_med_school_acceptances_l1417_141756


namespace NUMINAMATH_CALUDE_permutations_with_fixed_front_five_people_one_fixed_front_l1417_141797

/-- The number of ways to arrange n people in a line. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person always at the front. -/
def permutationsWithFixed (n : ℕ) : ℕ := permutations (n - 1)

theorem permutations_with_fixed_front (n : ℕ) (h : n > 1) :
  permutationsWithFixed n = Nat.factorial (n - 1) := by
  sorry

/-- There are 5 people, and we want to arrange them with one specific person at the front. -/
theorem five_people_one_fixed_front :
  permutationsWithFixed 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_with_fixed_front_five_people_one_fixed_front_l1417_141797


namespace NUMINAMATH_CALUDE_will_toy_purchase_l1417_141766

def max_toys_purchasable (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : ℕ :=
  ((initial_amount - game_cost) / toy_cost)

theorem will_toy_purchase : max_toys_purchasable 57 27 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_will_toy_purchase_l1417_141766


namespace NUMINAMATH_CALUDE_nth_prime_greater_than_3n_l1417_141782

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem nth_prime_greater_than_3n (n : ℕ) (h : n > 12) : nth_prime n > 3 * n := by
  sorry

end NUMINAMATH_CALUDE_nth_prime_greater_than_3n_l1417_141782


namespace NUMINAMATH_CALUDE_f_of_3_eq_3_l1417_141730

/-- The exponent in the function definition -/
def n : ℕ := 2008

/-- The function f(x) is defined implicitly by this equation -/
def f_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (x^(3^n - 1) - 1) * f x = 
    (List.range n).foldl (λ acc i => acc * (x^(3^i) + 1)) (x + 1) + (x^2 - 1) - 1

/-- The theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 (f : ℝ → ℝ) (hf : ∀ x, f_equation f x) : f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_3_l1417_141730


namespace NUMINAMATH_CALUDE_no_real_solutions_l1417_141792

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - 9 * z^2 = -5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1417_141792


namespace NUMINAMATH_CALUDE_anna_bills_count_l1417_141786

theorem anna_bills_count (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : 
  five_dollar_bills = 4 → ten_dollar_bills = 8 → five_dollar_bills + ten_dollar_bills = 12 := by
  sorry

end NUMINAMATH_CALUDE_anna_bills_count_l1417_141786


namespace NUMINAMATH_CALUDE_sarah_birthday_next_monday_l1417_141791

def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def days_since_reference_date (year month day : ℕ) : ℕ :=
  sorry

def day_of_week (year month day : ℕ) : ℕ :=
  (days_since_reference_date year month day) % 7

theorem sarah_birthday_next_monday (start_year : ℕ) (start_day_of_week : ℕ) :
  start_year = 2017 →
  start_day_of_week = 5 →
  day_of_week 2025 6 16 = 1 →
  ∀ y : ℕ, start_year < y → y < 2025 → day_of_week y 6 16 ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_sarah_birthday_next_monday_l1417_141791


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1417_141745

def shoe_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 5
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value ≥ shoe_cost ∧
    ∀ m : ℕ, m < n → (m : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value < shoe_cost :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1417_141745


namespace NUMINAMATH_CALUDE_point_on_axes_l1417_141768

theorem point_on_axes (a : ℝ) :
  let P : ℝ × ℝ := (2*a - 1, a + 2)
  (P.1 = 0 ∨ P.2 = 0) → (P = (-5, 0) ∨ P = (0, 2.5)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_axes_l1417_141768


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1417_141704

theorem arithmetic_computation : 1 + (6 * 2 - 3 + 5) * 4 / 2 = 29 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1417_141704


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l1417_141757

theorem no_solution_to_inequalities :
  ¬ ∃ x : ℝ, (4 * x + 2 < (x + 3)^2) ∧ ((x + 3)^2 < 8 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l1417_141757


namespace NUMINAMATH_CALUDE_midpoint_linear_combination_l1417_141760

/-- Given two points A and B in the plane, prove that if C is their midpoint,
    then a specific linear combination of C's coordinates equals -21. -/
theorem midpoint_linear_combination (A B : ℝ × ℝ) (h : A = (20, 9) ∧ B = (4, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 6 * C.2 = -21 := by
  sorry

#check midpoint_linear_combination

end NUMINAMATH_CALUDE_midpoint_linear_combination_l1417_141760


namespace NUMINAMATH_CALUDE_carpet_width_l1417_141789

/-- Given a rectangular carpet with length 9 feet that covers 20% of a 180 square feet living room floor, prove that the width of the carpet is 4 feet. -/
theorem carpet_width (carpet_length : ℝ) (room_area : ℝ) (coverage_percent : ℝ) :
  carpet_length = 9 →
  room_area = 180 →
  coverage_percent = 20 →
  (coverage_percent / 100) * room_area / carpet_length = 4 := by
sorry

end NUMINAMATH_CALUDE_carpet_width_l1417_141789


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1417_141788

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1417_141788


namespace NUMINAMATH_CALUDE_opposite_of_nine_l1417_141742

theorem opposite_of_nine : -(9 : ℤ) = -9 := by sorry

end NUMINAMATH_CALUDE_opposite_of_nine_l1417_141742


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_union_B_C_implies_a_bound_l1417_141746

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem for part (2)
theorem union_B_C_implies_a_bound (a : ℝ) :
  B ∪ C a = C a → a > -4 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_union_B_C_implies_a_bound_l1417_141746


namespace NUMINAMATH_CALUDE_intersection_points_range_l1417_141796

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = g b x₁ ∧ f x₂ = g b x₂ ∧ f x₃ = g b x₃

-- State the theorem
theorem intersection_points_range :
  ∀ b : ℝ, has_three_distinct_intersections b ↔ -1 < b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l1417_141796


namespace NUMINAMATH_CALUDE_prop_logic_l1417_141733

theorem prop_logic (p q : Prop) (h1 : ¬p) (h2 : p ∨ q) : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_l1417_141733


namespace NUMINAMATH_CALUDE_tree_height_problem_l1417_141769

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 24 →  -- The taller tree is 24 feet higher
  h₁ / h₂ = 5 / 7 →  -- The ratio of heights is 5:7
  h₂ = 84 := by
sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1417_141769


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l1417_141759

def standard_polar_form (r : ℝ) (θ : ℝ) : Prop :=
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem polar_coordinate_equivalence :
  ∀ (r₁ r₂ θ₁ θ₂ : ℝ),
  r₁ = -3 ∧ θ₁ = 5 * Real.pi / 6 →
  r₂ = 3 ∧ θ₂ = 11 * Real.pi / 6 →
  standard_polar_form r₂ θ₂ →
  (r₁ * (Real.cos θ₁), r₁ * (Real.sin θ₁)) = (r₂ * (Real.cos θ₂), r₂ * (Real.sin θ₂)) :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l1417_141759


namespace NUMINAMATH_CALUDE_exactly_three_even_dice_probability_l1417_141775

def num_sides : ℕ := 12
def num_dice : ℕ := 4
def num_even_sides : ℕ := 6

def prob_even_on_one_die : ℚ := num_even_sides / num_sides

theorem exactly_three_even_dice_probability :
  (num_dice.choose 3) * (prob_even_on_one_die ^ 3) * ((1 - prob_even_on_one_die) ^ (num_dice - 3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_even_dice_probability_l1417_141775


namespace NUMINAMATH_CALUDE_alice_bob_race_difference_l1417_141724

/-- The difference in finish times between two runners in a race. -/
def finish_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the difference in finish times for Alice and Bob in a 12-mile race. -/
theorem alice_bob_race_difference :
  finish_time_difference 5 7 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_race_difference_l1417_141724


namespace NUMINAMATH_CALUDE_same_day_of_week_l1417_141734

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : Nat) (dayNumber : Nat) : DayOfWeek := sorry

theorem same_day_of_week (year : Nat) :
  dayOfWeek year 15 = DayOfWeek.Monday →
  dayOfWeek year 197 = DayOfWeek.Monday :=
by
  sorry

end NUMINAMATH_CALUDE_same_day_of_week_l1417_141734


namespace NUMINAMATH_CALUDE_diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l1417_141719

/-- Calculates the concentration of a diluted vinegar solution -/
theorem diluted_vinegar_concentration 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let vinegar_amount := original_volume * (original_concentration / 100)
  let total_volume := original_volume + water_added
  let diluted_concentration := (vinegar_amount / total_volume) * 100
  diluted_concentration

/-- Proves that the diluted vinegar concentration is approximately 7% -/
theorem diluted_vinegar_concentration_proof 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) 
  (h1 : original_volume = 12) 
  (h2 : original_concentration = 36.166666666666664) 
  (h3 : water_added = 50) :
  ∃ ε > 0, |diluted_vinegar_concentration original_volume original_concentration water_added - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l1417_141719


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l1417_141751

theorem right_triangle_squares_area (x : ℝ) :
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let total_area := triangle_area + square1_area + square2_area
  (total_area = 1000) → (x = 10 * Real.sqrt 31 / 31) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l1417_141751


namespace NUMINAMATH_CALUDE_m_greater_than_n_l1417_141728

theorem m_greater_than_n (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l1417_141728


namespace NUMINAMATH_CALUDE_longer_subsegment_length_l1417_141713

/-- Triangle with sides in ratio 3:4:5 -/
structure Triangle :=
  (a b c : ℝ)
  (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)

/-- Angle bisector theorem -/
axiom angle_bisector_theorem {t : Triangle} (d : ℝ) :
  d / (t.c - d) = t.a / t.b

/-- Main theorem -/
theorem longer_subsegment_length (t : Triangle) (h : t.c = 15) :
  let d := t.c * (t.a / (t.a + t.b))
  d = 75 / 8 := by sorry

end NUMINAMATH_CALUDE_longer_subsegment_length_l1417_141713


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1417_141712

/-- Line represented by slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region of interest in the first quadrant -/
def Region (p q r : Line) : Set Point :=
  {pt : Point | 0 ≤ pt.x ∧ 0 ≤ pt.y ∧ 
                pt.y ≤ p.slope * pt.x + p.intercept ∧
                r.intercept < pt.y ∧ 
                q.slope * pt.x + q.intercept < pt.y ∧ 
                pt.y < p.slope * pt.x + p.intercept}

/-- The area of the region of interest -/
noncomputable def areaOfRegion (p q r : Line) : ℝ := sorry

/-- The total area under line p and above x-axis in the first quadrant -/
noncomputable def totalArea (p : Line) : ℝ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_one_third 
  (p : Line) 
  (q : Line) 
  (r : Line) 
  (hp : p.slope = -2 ∧ p.intercept = 8) 
  (hq : q.slope = -3 ∧ q.intercept = 8) 
  (hr : r.slope = 0 ∧ r.intercept = 4) : 
  areaOfRegion p q r / totalArea p = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1417_141712


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l1417_141761

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) :
  a + b > 4 := by
sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l1417_141761


namespace NUMINAMATH_CALUDE_kevin_food_spending_l1417_141774

def total_budget : ℕ := 20
def samuel_ticket : ℕ := 14
def samuel_total : ℕ := 20
def kevin_ticket : ℕ := 14
def kevin_drinks : ℕ := 2

theorem kevin_food_spending :
  ∃ (kevin_food : ℕ),
    kevin_food = total_budget - (kevin_ticket + kevin_drinks) ∧
    kevin_food = 4 :=
by sorry

end NUMINAMATH_CALUDE_kevin_food_spending_l1417_141774


namespace NUMINAMATH_CALUDE_incorrect_conjunction_falsehood_l1417_141705

theorem incorrect_conjunction_falsehood : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_conjunction_falsehood_l1417_141705


namespace NUMINAMATH_CALUDE_employee_payment_l1417_141726

/-- Given two employees X and Y with a total payment of 550 units,
    where X is paid 120% of Y's payment, prove that Y is paid 250 units. -/
theorem employee_payment (x y : ℝ) 
  (total : x + y = 550)
  (ratio : x = 1.2 * y) : 
  y = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l1417_141726


namespace NUMINAMATH_CALUDE_add_248_64_l1417_141794

theorem add_248_64 : 248 + 64 = 312 := by
  sorry

end NUMINAMATH_CALUDE_add_248_64_l1417_141794


namespace NUMINAMATH_CALUDE_smallest_value_l1417_141764

def Q (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem smallest_value (x₁ x₂ x₃ : ℝ) (hzeros : Q x₁ = 0 ∧ Q x₂ = 0 ∧ Q x₃ = 0) :
  min (min (Q (-1)) (1 + (-3) + (-9) + 2)) (min (x₁ * x₂ * x₃) (Q 1)) = x₁ * x₂ * x₃ :=
sorry

end NUMINAMATH_CALUDE_smallest_value_l1417_141764


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1417_141736

/-- Given two vectors a and b in ℝ², if a + x*b is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
  (ha : a = (3, 4))
  (hb : b = (2, -1))
  (h_perp : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1417_141736


namespace NUMINAMATH_CALUDE_jellybean_count_l1417_141748

/-- The number of blue jellybeans in a jar -/
def blue_jellybeans (total purple orange red : ℕ) : ℕ :=
  total - (purple + orange + red)

/-- Theorem: In a jar with 200 total jellybeans, 26 purple, 40 orange, and 120 red jellybeans,
    there are 14 blue jellybeans. -/
theorem jellybean_count : blue_jellybeans 200 26 40 120 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1417_141748


namespace NUMINAMATH_CALUDE_rectangular_box_problem_l1417_141732

theorem rectangular_box_problem (m n r : ℕ) (hm : m > 0) (hn : n > 0) (hr : r > 0)
  (h_order : m ≤ n ∧ n ≤ r) (h_equation : (m-2)*(n-2)*(r-2) + 4*((m-2) + (n-2) + (r-2)) - 
  2*((m-2)*(n-2) + (m-2)*(r-2) + (n-2)*(r-2)) = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_problem_l1417_141732


namespace NUMINAMATH_CALUDE_adult_admission_price_l1417_141716

/-- Proves that the adult admission price was 60 cents given the conditions -/
theorem adult_admission_price (total_attendance : ℕ) (child_ticket_price : ℕ) 
  (children_attended : ℕ) (total_revenue : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_adult_admission_price_l1417_141716


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1417_141749

/-- The custom operation ⊗ -/
def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y z : ℝ) : 
  otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x^2 + 2*x*z - y^2 - 2*z*y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1417_141749


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1417_141709

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * (3 : ℝ)^x + (3 : ℝ)^(-x) = 3) ↔ 
  a ∈ Set.Iic (0 : ℝ) ∪ {9/4} :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1417_141709


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1417_141706

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - 21*x^3 + 8*x^2 - 17*x + 12 = (x - 3)*(x^4 + 3*x^3 - 12*x^2 - 28*x - 101) + (-201) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1417_141706


namespace NUMINAMATH_CALUDE_max_knights_between_knights_max_knights_between_knights_proof_l1417_141773

theorem max_knights_between_knights (total_knights : ℕ) (total_samurais : ℕ) 
  (knights_with_samurai_right : ℕ) (max_knights_between_knights : ℕ) : Prop :=
  total_knights = 40 →
  total_samurais = 10 →
  knights_with_samurai_right = 7 →
  max_knights_between_knights = 32 →
  max_knights_between_knights = total_knights - (knights_with_samurai_right + 1)

-- The proof would go here, but we're skipping it as per instructions
theorem max_knights_between_knights_proof : 
  max_knights_between_knights 40 10 7 32 := by sorry

end NUMINAMATH_CALUDE_max_knights_between_knights_max_knights_between_knights_proof_l1417_141773


namespace NUMINAMATH_CALUDE_solve_equation_l1417_141715

theorem solve_equation (x : ℝ) : 
  5 * x^(1/3) - 3 * (x / x^(2/3)) = 9 + x^(1/3) ↔ x = 729 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1417_141715


namespace NUMINAMATH_CALUDE_det_E_l1417_141737

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- Theorem: The determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by sorry

end NUMINAMATH_CALUDE_det_E_l1417_141737


namespace NUMINAMATH_CALUDE_candy_solution_l1417_141770

/-- Represents the candy distribution problem --/
def candy_problem (billy_initial caleb_initial andy_initial : ℕ)
                  (new_candies billy_new caleb_new : ℕ) : Prop :=
  let billy_total := billy_initial + billy_new
  let caleb_total := caleb_initial + caleb_new
  let andy_new := new_candies - billy_new - caleb_new
  let andy_total := andy_initial + andy_new
  andy_total - caleb_total = 4

/-- Theorem stating the solution to the candy problem --/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_solution_l1417_141770


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1417_141750

theorem number_puzzle_solution : ∃ x : ℝ, 12 * x = x + 198 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1417_141750


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l1417_141790

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  : ∃ (num_items : ℕ),
    let total_length := num_items * desk_length + num_items * bookcase_length
    wall_length - total_length = 1 ∧
    ∀ (n : ℕ), n * desk_length + n * bookcase_length ≤ wall_length → n ≤ num_items :=
by sorry

end NUMINAMATH_CALUDE_library_wall_leftover_space_l1417_141790


namespace NUMINAMATH_CALUDE_inequality_proof_l1417_141731

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1417_141731


namespace NUMINAMATH_CALUDE_triangle_side_length_l1417_141777

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →  -- Convert 45° to radians
  C = 105 * π / 180 →  -- Convert 105° to radians
  b = Real.sqrt 2 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Law of sines
  c * Real.sin B = b * Real.sin C →  -- Law of sines
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1417_141777
