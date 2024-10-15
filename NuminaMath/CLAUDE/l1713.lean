import Mathlib

namespace NUMINAMATH_CALUDE_system_of_equations_l1713_171301

theorem system_of_equations (a : ℝ) :
  let x := 2 * a + 3
  let y := -a - 2
  (x > 0 ∧ y ≥ 0) →
  ((-3 < a ∧ a ≤ -2) ∧
   (a = -5/3 → x = y) ∧
   (a = -2 → x + y = 5 + a)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_l1713_171301


namespace NUMINAMATH_CALUDE_log_equation_solution_l1713_171368

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 3 + Real.log x / Real.log (1/3) = 6 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1713_171368


namespace NUMINAMATH_CALUDE_sum_of_squares_l1713_171332

theorem sum_of_squares (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) : 
  x^2 + y^2 = 2424 / 49 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1713_171332


namespace NUMINAMATH_CALUDE_unique_half_value_l1713_171315

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + 2 * f x

/-- The theorem stating that f(1/2) has only one possible value, which is 1 -/
theorem unique_half_value (f : ℝ → ℝ) (hf : special_function f) : 
  ∃! v : ℝ, f (1/2) = v ∧ v = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_half_value_l1713_171315


namespace NUMINAMATH_CALUDE_eighth_fibonacci_term_l1713_171335

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_term_l1713_171335


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_three_equals_three_l1713_171310

theorem absolute_value_of_negative_three_equals_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_three_equals_three_l1713_171310


namespace NUMINAMATH_CALUDE_min_value_z_l1713_171349

/-- The function z(x) = 5x^2 + 10x + 20 has a minimum value of 15 -/
theorem min_value_z (x : ℝ) : ∀ y : ℝ, 5 * x^2 + 10 * x + 20 ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l1713_171349


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_time_jimmy_total_time_l1713_171333

/-- The sum of an arithmetic sequence with 5 terms, first term 20, and common difference 5 -/
def arithmetic_sum : ℕ := by sorry

/-- The number of flights Jimmy climbs -/
def num_flights : ℕ := 5

/-- The time taken to climb the first flight -/
def first_flight_time : ℕ := 20

/-- The increase in time for each subsequent flight -/
def time_increase : ℕ := 5

theorem jimmy_stair_climbing_time :
  arithmetic_sum = num_flights * (2 * first_flight_time + (num_flights - 1) * time_increase) / 2 :=
by sorry

theorem jimmy_total_time : arithmetic_sum = 150 := by sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_time_jimmy_total_time_l1713_171333


namespace NUMINAMATH_CALUDE_defective_product_scenarios_l1713_171389

theorem defective_product_scenarios 
  (total_products : Nat) 
  (defective_products : Nat) 
  (good_products : Nat) 
  (h1 : total_products = 10)
  (h2 : defective_products = 4)
  (h3 : good_products = 6)
  (h4 : total_products = defective_products + good_products) :
  (Nat.choose good_products 1) * (Nat.choose defective_products 1) * (Nat.factorial 4) = 
  (number_of_scenarios : Nat) := by
  sorry

end NUMINAMATH_CALUDE_defective_product_scenarios_l1713_171389


namespace NUMINAMATH_CALUDE_shopping_price_difference_l1713_171374

/-- Proves that the difference between shoe price and bag price is $17 --/
theorem shopping_price_difference 
  (initial_amount : ℕ) 
  (shoe_price : ℕ) 
  (remaining_amount : ℕ) 
  (bag_price : ℕ) 
  (lunch_price : ℕ) 
  (h1 : initial_amount = 158)
  (h2 : shoe_price = 45)
  (h3 : remaining_amount = 78)
  (h4 : lunch_price = bag_price / 4)
  (h5 : initial_amount = shoe_price + bag_price + lunch_price + remaining_amount) :
  shoe_price - bag_price = 17 := by
  sorry

end NUMINAMATH_CALUDE_shopping_price_difference_l1713_171374


namespace NUMINAMATH_CALUDE_kishore_savings_l1713_171386

def monthly_salary (expenses : ℕ) (savings_rate : ℚ) : ℚ :=
  expenses / (1 - savings_rate)

def savings (salary : ℚ) (savings_rate : ℚ) : ℚ :=
  salary * savings_rate

theorem kishore_savings (expenses : ℕ) (savings_rate : ℚ) :
  expenses = 18000 →
  savings_rate = 1/10 →
  savings (monthly_salary expenses savings_rate) savings_rate = 2000 := by
sorry

end NUMINAMATH_CALUDE_kishore_savings_l1713_171386


namespace NUMINAMATH_CALUDE_eight_books_distribution_l1713_171385

/-- The number of ways to distribute indistinguishable books between two locations --/
def distribute_books (total : ℕ) : ℕ := 
  if total ≥ 2 then total - 1 else 0

/-- Theorem: Distributing 8 indistinguishable books between two locations, 
    with at least one book in each location, results in 7 different ways --/
theorem eight_books_distribution : distribute_books 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_books_distribution_l1713_171385


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l1713_171342

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 3707 ∧ 
  ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l1713_171342


namespace NUMINAMATH_CALUDE_inequality_solution_l1713_171339

theorem inequality_solution (x : ℝ) : 2 ≤ (3*x)/(3*x-7) ∧ (3*x)/(3*x-7) < 6 ↔ 7/3 < x ∧ x < 42/15 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1713_171339


namespace NUMINAMATH_CALUDE_parabola_roots_and_point_below_axis_l1713_171308

/-- A parabola with a point below the x-axis has two distinct real roots, and the x-coordinate of the point is between these roots. -/
theorem parabola_roots_and_point_below_axis 
  (p q x₀ : ℝ) 
  (h_below : x₀^2 + p*x₀ + q < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (x₁ < x₀) ∧ 
    (x₀ < x₂) ∧ 
    (x₁ ≠ x₂) := by
  sorry

end NUMINAMATH_CALUDE_parabola_roots_and_point_below_axis_l1713_171308


namespace NUMINAMATH_CALUDE_remainder_theorem_l1713_171323

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 7

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  q D E F 2 = 5 → q D E F (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1713_171323


namespace NUMINAMATH_CALUDE_b_joined_after_ten_months_l1713_171344

/-- Represents the business scenario --/
structure Business where
  a_investment : ℕ
  b_investment : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which B joined the business --/
def months_before_b_joined (b : Business) : ℕ :=
  b.total_duration - (b.a_investment * b.total_duration * b.profit_ratio_b) / 
    (b.b_investment * b.profit_ratio_a)

/-- Theorem stating that B joined after 10 months --/
theorem b_joined_after_ten_months (b : Business) 
  (h1 : b.a_investment = 3500)
  (h2 : b.b_investment = 31500)
  (h3 : b.profit_ratio_a = 2)
  (h4 : b.profit_ratio_b = 3)
  (h5 : b.total_duration = 12) :
  months_before_b_joined b = 10 := by
  sorry

end NUMINAMATH_CALUDE_b_joined_after_ten_months_l1713_171344


namespace NUMINAMATH_CALUDE_unique_pairs_satisfying_W_l1713_171365

def W (x : ℕ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

theorem unique_pairs_satisfying_W :
  ∀ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_pairs_satisfying_W_l1713_171365


namespace NUMINAMATH_CALUDE_power_division_rule_l1713_171340

theorem power_division_rule (m : ℝ) : m^7 / m^3 = m^4 := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l1713_171340


namespace NUMINAMATH_CALUDE_banana_orange_relation_bananas_to_oranges_l1713_171354

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The given relationship between bananas and oranges -/
theorem banana_orange_relation : (3/4 : ℚ) * 16 * banana_value = 12 := by sorry

/-- Theorem to prove: If 3/4 of 16 bananas are worth 12 oranges, 
    then 1/3 of 9 bananas are worth 3 oranges -/
theorem bananas_to_oranges : 
  ((1/3 : ℚ) * 9 * banana_value = 3) := by sorry

end NUMINAMATH_CALUDE_banana_orange_relation_bananas_to_oranges_l1713_171354


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1713_171307

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x + 7 > 3*x + 2 ∧ 2*x - 2 < 2*m) ↔ x < 5) →
  m ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1713_171307


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1713_171388

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 3) - 2
  f (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1713_171388


namespace NUMINAMATH_CALUDE_largest_negative_integer_l1713_171313

theorem largest_negative_integer : ∀ n : ℤ, n < 0 → n ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l1713_171313


namespace NUMINAMATH_CALUDE_min_value_expression_l1713_171318

theorem min_value_expression (x : ℝ) (h : x > 0) : 
  4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1713_171318


namespace NUMINAMATH_CALUDE_hall_breadth_calculation_l1713_171326

/-- Proves that given a hall of length 36 meters, paved with 1350 stones each measuring 8 dm by 5 dm, the breadth of the hall is 15 meters. -/
theorem hall_breadth_calculation (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.8 →
  stone_width = 0.5 →
  num_stones = 1350 →
  (num_stones * stone_length * stone_width) / hall_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_hall_breadth_calculation_l1713_171326


namespace NUMINAMATH_CALUDE_a_6_value_l1713_171375

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_6_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 = 1 ∧ a 8 = 3) ∨ (a 4 = 3 ∧ a 8 = 1) →
  a 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l1713_171375


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1713_171348

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_square_sum (n : ℕ) : ℕ := (n / 10)^2 + (n % 10)^2

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
            n / digit_sum n = 3 ∧
            n % digit_sum n = 7 ∧
            digit_square_sum n - digit_product n = n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1713_171348


namespace NUMINAMATH_CALUDE_oprah_car_giveaway_l1713_171395

theorem oprah_car_giveaway (initial_cars final_cars years : ℕ) 
  (h1 : initial_cars = 3500)
  (h2 : final_cars = 500)
  (h3 : years = 60) :
  (initial_cars - final_cars) / years = 50 :=
by sorry

end NUMINAMATH_CALUDE_oprah_car_giveaway_l1713_171395


namespace NUMINAMATH_CALUDE_distinct_sums_count_l1713_171399

def bag_X : Finset ℕ := {2, 5, 7}
def bag_Y : Finset ℕ := {1, 4, 8}

theorem distinct_sums_count : 
  Finset.card ((bag_X.product bag_Y).image (fun p => p.1 + p.2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l1713_171399


namespace NUMINAMATH_CALUDE_diagonal_cut_square_area_l1713_171390

/-- A square cut along its diagonal with specific translations of one half -/
structure DiagonalCutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The first translation distance -/
  trans1 : ℝ
  /-- The second translation distance -/
  trans2 : ℝ
  /-- The first translation is 3 units -/
  h_trans1 : trans1 = 3
  /-- The second translation is 5 units -/
  h_trans2 : trans2 = 5
  /-- The overlapping areas after each translation are equal -/
  h_equal_overlap : trans1 * trans2 = trans2 * (side - trans1 - trans2)

/-- The theorem stating that under the given conditions, the square's area is 121 -/
theorem diagonal_cut_square_area (s : DiagonalCutSquare) : s.side^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cut_square_area_l1713_171390


namespace NUMINAMATH_CALUDE_equation_solution_l1713_171319

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1713_171319


namespace NUMINAMATH_CALUDE_inequality_range_l1713_171304

theorem inequality_range (P : ℝ) (h : 0 ≤ P ∧ P ≤ 4) :
  (∀ x : ℝ, x^2 + P*x > 4*x + P - 3) ↔ (∀ x : ℝ, x < -1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1713_171304


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l1713_171392

theorem decimal_to_percentage (x : ℝ) : x = 1.20 → (x * 100 : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l1713_171392


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1713_171367

theorem consecutive_numbers_sum (x : ℕ) : 
  x * (x + 1) = 12650 → x + (x + 1) = 225 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1713_171367


namespace NUMINAMATH_CALUDE_triangle_area_l1713_171305

/-- The area of a triangle with one side of length 12 cm and an adjacent angle of 30° is 36 square centimeters. -/
theorem triangle_area (BC : ℝ) (angle_C : ℝ) : 
  BC = 12 → angle_C = 30 * (π / 180) → 
  (1/2) * BC * (BC * Real.sin angle_C) = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1713_171305


namespace NUMINAMATH_CALUDE_integers_abs_leq_three_l1713_171369

theorem integers_abs_leq_three :
  {x : ℤ | |x| ≤ 3} = {-3, -2, -1, 0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_integers_abs_leq_three_l1713_171369


namespace NUMINAMATH_CALUDE_solution_value_l1713_171355

theorem solution_value (x y m : ℝ) : 
  x = 2 ∧ y = -1 ∧ 2*x - 3*y = m → m = 7 := by sorry

end NUMINAMATH_CALUDE_solution_value_l1713_171355


namespace NUMINAMATH_CALUDE_angle_C_measure_l1713_171396

theorem angle_C_measure (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
  (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1713_171396


namespace NUMINAMATH_CALUDE_x_less_than_negative_one_sufficient_not_necessary_l1713_171327

theorem x_less_than_negative_one_sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_negative_one_sufficient_not_necessary_l1713_171327


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1713_171356

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1713_171356


namespace NUMINAMATH_CALUDE_total_shingle_area_l1713_171346

/-- Calculate the total square footage of shingles required for a house with a main roof and a porch roof. -/
theorem total_shingle_area (main_roof_base main_roof_height porch_roof_length porch_roof_upper_base porch_roof_lower_base porch_roof_height : ℝ) : 
  main_roof_base = 20.5 →
  main_roof_height = 25 →
  porch_roof_length = 6 →
  porch_roof_upper_base = 2.5 →
  porch_roof_lower_base = 4.5 →
  porch_roof_height = 3 →
  (main_roof_base * main_roof_height + (porch_roof_upper_base + porch_roof_lower_base) * porch_roof_height * 2) = 554.5 := by
  sorry

#check total_shingle_area

end NUMINAMATH_CALUDE_total_shingle_area_l1713_171346


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l1713_171350

theorem sum_of_roots_equals_seven : 
  ∀ (x y : ℝ), x^2 - 7*x + 12 = 0 ∧ y^2 - 7*y + 12 = 0 ∧ x ≠ y → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l1713_171350


namespace NUMINAMATH_CALUDE_store_discount_l1713_171362

theorem store_discount (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.4
  let second_discount := 0.1
  let claimed_discount := 0.5
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  let actual_discount := 1 - (final_price / original_price)
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_l1713_171362


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1713_171363

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1713_171363


namespace NUMINAMATH_CALUDE_optimal_strategy_with_budget_optimal_strategy_without_budget_l1713_171371

/-- Revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - (x₁ + x₂)

/-- Theorem for part 1 -/
theorem optimal_strategy_with_budget :
  ∀ x₁ x₂ : ℝ, x₁ + x₂ = 5 → profit x₁ x₂ ≤ profit 2 3 :=
sorry

/-- Theorem for part 2 -/
theorem optimal_strategy_without_budget :
  ∀ x₁ x₂ : ℝ, profit x₁ x₂ ≤ profit 3 5 :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_with_budget_optimal_strategy_without_budget_l1713_171371


namespace NUMINAMATH_CALUDE_age_ratio_l1713_171353

theorem age_ratio (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 → 
  years_ago = 5 → 
  (current_age : ℚ) / ((current_age - years_ago) : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l1713_171353


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l1713_171381

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 1106 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l1713_171381


namespace NUMINAMATH_CALUDE_integral_root_iff_odd_l1713_171324

theorem integral_root_iff_odd (n : ℕ) :
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_integral_root_iff_odd_l1713_171324


namespace NUMINAMATH_CALUDE_boys_on_playground_l1713_171312

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end NUMINAMATH_CALUDE_boys_on_playground_l1713_171312


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l1713_171351

theorem inverse_proportion_percentage_change (x y a b : ℝ) (k : ℝ) : 
  x > 0 → y > 0 → 
  (x * y = k) → 
  ((1 + a / 100) * x) * ((1 - b / 100) * y) = k → 
  b = |100 * a / (100 + a)| := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l1713_171351


namespace NUMINAMATH_CALUDE_total_discount_percentage_l1713_171329

-- Define the discounts
def initial_discount : ℝ := 0.3
def clearance_discount : ℝ := 0.2

-- Theorem statement
theorem total_discount_percentage : 
  (1 - (1 - initial_discount) * (1 - clearance_discount)) * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_percentage_l1713_171329


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1713_171311

/-- Two lines intersect in the first quadrant if and only if k is in the open interval (-2/3, 2) -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + k + 2 ∧ y = -2 * x + 4) ↔ 
  -2/3 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1713_171311


namespace NUMINAMATH_CALUDE_words_per_page_smaller_type_l1713_171360

/-- Calculates words per page in smaller type given article details -/
def wordsPerPageSmallerType (totalWords : ℕ) (totalPages : ℕ) (smallerTypePages : ℕ) (wordsPerPageLargerType : ℕ) : ℕ :=
  let largerTypePages := totalPages - smallerTypePages
  let wordsInLargerType := largerTypePages * wordsPerPageLargerType
  let wordsInSmallerType := totalWords - wordsInLargerType
  wordsInSmallerType / smallerTypePages

/-- Proves that words per page in smaller type is 2400 for given article details -/
theorem words_per_page_smaller_type :
  wordsPerPageSmallerType 48000 21 17 1800 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_smaller_type_l1713_171360


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l1713_171376

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  positive_a : 0 < a
  positive_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the plane -/
structure Point (α : Type*) where
  x : α
  y : α

/-- The area of a triangle given three points -/
def triangle_area (A B C : Point ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with specific properties is √2 -/
theorem hyperbola_eccentricity_sqrt_two 
  (a b c : ℝ) (h : Hyperbola a b) 
  (M N : Point ℝ) (A : Point ℝ) :
  (∃ F₁ F₂ : Point ℝ, 
    -- M and N are on the asymptote
    -- MF₁NF₂ is a rectangle
    -- A is a vertex of the hyperbola
    triangle_area A M N = (1/2) * c^2) →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l1713_171376


namespace NUMINAMATH_CALUDE_coin_set_existence_l1713_171358

def is_valid_coin_set (weights : List Nat) : Prop :=
  ∀ k, k ∈ weights → 
    ∃ (A B : List Nat), 
      A ∪ B = weights.erase k ∧ 
      A.sum = B.sum

theorem coin_set_existence (n : Nat) : 
  (∃ weights : List Nat, 
    weights.length = n ∧ 
    weights.Nodup ∧
    is_valid_coin_set weights) ↔ 
  (Odd n ∧ n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_coin_set_existence_l1713_171358


namespace NUMINAMATH_CALUDE_multiple_of_six_as_sum_of_four_cubes_l1713_171379

theorem multiple_of_six_as_sum_of_four_cubes (k : ℤ) :
  ∃ (a b c d : ℤ), 6 * k = a^3 + b^3 + c^3 + d^3 :=
sorry

end NUMINAMATH_CALUDE_multiple_of_six_as_sum_of_four_cubes_l1713_171379


namespace NUMINAMATH_CALUDE_solve_windows_problem_l1713_171334

def windows_problem (installed : ℕ) (hours_per_window : ℕ) (remaining_hours : ℕ) : Prop :=
  let remaining := remaining_hours / hours_per_window
  installed + remaining = 9

theorem solve_windows_problem :
  windows_problem 6 6 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_windows_problem_l1713_171334


namespace NUMINAMATH_CALUDE_reciprocal_sum_and_product_l1713_171366

theorem reciprocal_sum_and_product (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 12) (h4 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 ∧ 1 / (x * y) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_and_product_l1713_171366


namespace NUMINAMATH_CALUDE_complex_power_sum_l1713_171303

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 * z^99 * z^100 + z^101 + z^102 + z^103 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1713_171303


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1713_171322

theorem simplify_fraction_product : (222 : ℚ) / 999 * 111 = 74 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1713_171322


namespace NUMINAMATH_CALUDE_captain_smollett_problem_l1713_171378

theorem captain_smollett_problem :
  ∃! (a c l : ℕ), 
    0 < a ∧ a < 100 ∧
    c > 3 ∧
    l > 0 ∧
    a * c * l = 32118 ∧
    a = 53 ∧ c = 6 ∧ l = 101 := by
  sorry

end NUMINAMATH_CALUDE_captain_smollett_problem_l1713_171378


namespace NUMINAMATH_CALUDE_min_value_theorem_l1713_171331

noncomputable section

variables (a m n : ℝ)

-- Define the function f
def f (x : ℝ) := a^(x - 1) - 2

-- State the conditions
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
axiom m_pos : m > 0
axiom n_pos : n > 0

-- Define the fixed point A
def A : ℝ × ℝ := (1, -1)

-- State that A lies on the line mx - ny - 1 = 0
axiom A_on_line : m * A.1 - n * A.2 - 1 = 0

-- State the theorem to be proved
theorem min_value_theorem : 
  (∀ x : ℝ, f x = f (A.1) → x = A.1) → 
  (∃ (m' n' : ℝ), m' > 0 ∧ n' > 0 ∧ m' * A.1 - n' * A.2 - 1 = 0 ∧ 1/m' + 2/n' < 1/m + 2/n) → 
  1/m + 2/n ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_value_theorem_l1713_171331


namespace NUMINAMATH_CALUDE_system_solution_fractional_solution_l1713_171316

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  3 * x + y = 7 ∧ 2 * x - y = 3

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ -1 ∧ x ≠ 1 ∧ 1 / (x + 1) = 1 / (x^2 - 1)

-- Theorem for the system of equations
theorem system_solution :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 1 := by
  sorry

-- Theorem for the fractional equation
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_fractional_solution_l1713_171316


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_120_l1713_171383

theorem product_of_consecutive_integers_120 : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (a * b = 120) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (c * d * e = 120) ∧ 
    (a + b + c + d + e = 36) :=
by sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_120_l1713_171383


namespace NUMINAMATH_CALUDE_custom_polynomial_value_l1713_171347

/-- Custom multiplication operation -/
def star_mult (x y : ℕ) : ℕ := (x + 1) * (y + 1)

/-- Custom squaring operation -/
def star_square (x : ℕ) : ℕ := star_mult x x

/-- The main theorem to prove -/
theorem custom_polynomial_value :
  3 * (star_square 2) - 2 * 2 + 1 = 32 := by sorry

end NUMINAMATH_CALUDE_custom_polynomial_value_l1713_171347


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l1713_171306

/-- 
Given a quadratic function y = ax² + 6ax - 5 where a > 0, 
and points A(-4, y₁), B(-3, y₂), and C(1, y₃) on this function's graph,
prove that y₂ < y₁ < y₃.
-/
theorem quadratic_point_ordering (a y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (hA : y₁ = a * (-4)^2 + 6 * a * (-4) - 5)
  (hB : y₂ = a * (-3)^2 + 6 * a * (-3) - 5)
  (hC : y₃ = a * 1^2 + 6 * a * 1 - 5) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l1713_171306


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1713_171359

/-- Proves that the speed of a boat in still water is 20 km/hr -/
theorem boat_speed_in_still_water : 
  ∀ (x : ℝ), 
    (5 : ℝ) = 5 → -- Rate of current is 5 km/hr
    ((x + 5) * (21 / 60) = (35 / 4 : ℝ)) → -- Distance travelled downstream in 21 minutes is 8.75 km
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1713_171359


namespace NUMINAMATH_CALUDE_common_zero_condition_l1713_171337

/-- The first polynomial -/
def P (k : ℝ) (x : ℝ) : ℝ := 1988 * x^2 + k * x + 8891

/-- The second polynomial -/
def Q (k : ℝ) (x : ℝ) : ℝ := 8891 * x^2 + k * x + 1988

/-- Theorem stating the condition for common zeros -/
theorem common_zero_condition (k : ℝ) :
  (∃ x : ℝ, P k x = 0 ∧ Q k x = 0) ↔ (k = 10879 ∨ k = -10879) := by sorry

end NUMINAMATH_CALUDE_common_zero_condition_l1713_171337


namespace NUMINAMATH_CALUDE_probability_wait_two_minutes_expected_wait_time_l1713_171343

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def suitcase_interval : ℕ := 2  -- seconds

-- Part a
theorem probability_wait_two_minutes :
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases) =
  ↑(Nat.choose 59 9) / ↑(Nat.choose total_suitcases business_suitcases) := by sorry

-- Part b
theorem expected_wait_time :
  (4020 : ℚ) / 11 = 2 * (business_suitcases * (total_suitcases + 1) / (business_suitcases + 1)) := by sorry

end NUMINAMATH_CALUDE_probability_wait_two_minutes_expected_wait_time_l1713_171343


namespace NUMINAMATH_CALUDE_circumradius_area_ratio_not_always_equal_l1713_171314

/-- Isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- Given two isosceles triangles with distinct sides, prove that the ratio of their circumradii
is not always equal to the ratio of their areas -/
theorem circumradius_area_ratio_not_always_equal
  (I II : IsoscelesTriangle)
  (h_distinct_base : I.base ≠ II.base)
  (h_distinct_side : I.side ≠ II.side) :
  ¬ ∀ (I II : IsoscelesTriangle),
    I.circumradius / II.circumradius = I.area / II.area :=
sorry

end NUMINAMATH_CALUDE_circumradius_area_ratio_not_always_equal_l1713_171314


namespace NUMINAMATH_CALUDE_train_length_l1713_171330

/-- The length of a train that crosses a platform of equal length in one minute at 54 km/hr -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 54 → -- speed in km/hr
  time = 1 / 60 → -- time in hours (1 minute = 1/60 hour)
  length = speed * time / 2 → -- distance formula, divided by 2 due to equal lengths
  length = 450 / 1000 -- length in km (450m = 0.45km)
  := by sorry

end NUMINAMATH_CALUDE_train_length_l1713_171330


namespace NUMINAMATH_CALUDE_average_difference_l1713_171398

def num_students : ℕ := 120
def num_teachers : ℕ := 5
def class_sizes : List ℕ := [40, 30, 20, 15, 15]

def t : ℚ := (num_students : ℚ) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -3.92 := by sorry

end NUMINAMATH_CALUDE_average_difference_l1713_171398


namespace NUMINAMATH_CALUDE_black_friday_sales_l1713_171361

/-- Calculates the number of televisions sold after a given number of years,
    given an initial sale and yearly increase. -/
def televisionsSold (initialSale : ℕ) (yearlyIncrease : ℕ) (years : ℕ) : ℕ :=
  initialSale + yearlyIncrease * years

/-- Theorem stating that given an initial sale of 327 televisions and
    an increase of 50 televisions per year, the number of televisions
    sold after 3 years will be 477. -/
theorem black_friday_sales : televisionsSold 327 50 3 = 477 := by
  sorry

end NUMINAMATH_CALUDE_black_friday_sales_l1713_171361


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l1713_171325

structure Triangle where
  -- Define a triangle structure
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the different sets of data
def ratioSideToAngleBisector (t : Triangle) : ℝ := sorry
def ratiosOfAngleBisectors (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def midpointsOfSides (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry
def twoSidesAndOppositeAngle (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def ratioOfTwoAngles (t : Triangle) : ℝ := sorry

-- Define what it means for a set of data to uniquely determine a triangle
def uniquelyDetermines (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

theorem triangle_shape_determination :
  (¬ uniquelyDetermines ratioSideToAngleBisector) ∧
  (uniquelyDetermines ratiosOfAngleBisectors) ∧
  (¬ uniquelyDetermines midpointsOfSides) ∧
  (uniquelyDetermines twoSidesAndOppositeAngle) ∧
  (uniquelyDetermines ratioOfTwoAngles) := by sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l1713_171325


namespace NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l1713_171357

/-- The measure of the largest angle in a quadrilateral with angles in the ratio 3:4:5:6 -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  (a + b + c + d = 360) →  -- Sum of angles in a quadrilateral is 360°
  (∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k) →  -- Angles are in the ratio 3:4:5:6
  max a (max b (max c d)) = 120  -- The largest angle is 120°
:= by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l1713_171357


namespace NUMINAMATH_CALUDE_harriet_drive_time_l1713_171373

theorem harriet_drive_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 100)
  (h3 : return_speed = 150) :
  let distance := (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)
  let outbound_time := distance / outbound_speed
  outbound_time * 60 = 180 := by
sorry

end NUMINAMATH_CALUDE_harriet_drive_time_l1713_171373


namespace NUMINAMATH_CALUDE_average_score_l1713_171317

def scores : List ℕ := [65, 67, 76, 82, 85]

theorem average_score : (scores.sum / scores.length : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_score_l1713_171317


namespace NUMINAMATH_CALUDE_union_of_positive_and_square_ge_self_is_reals_l1713_171364

open Set

theorem union_of_positive_and_square_ge_self_is_reals :
  let M : Set ℝ := {x | x > 0}
  let N : Set ℝ := {x | x^2 ≥ x}
  M ∪ N = univ := by sorry

end NUMINAMATH_CALUDE_union_of_positive_and_square_ge_self_is_reals_l1713_171364


namespace NUMINAMATH_CALUDE_int_poly5_root_count_l1713_171394

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPoly5 where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The set of possible numbers of integer roots for an IntPoly5 -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 4, 5}

/-- The number of integer roots (counting multiplicity) of an IntPoly5 -/
def numIntegerRoots (p : IntPoly5) : ℕ := sorry

/-- Theorem stating that the number of integer roots of an IntPoly5 is in the set of possible root counts -/
theorem int_poly5_root_count (p : IntPoly5) : numIntegerRoots p ∈ possibleRootCounts := by sorry

end NUMINAMATH_CALUDE_int_poly5_root_count_l1713_171394


namespace NUMINAMATH_CALUDE_fraction_problem_l1713_171328

theorem fraction_problem : ∃ x : ℝ, x * (5/9) * (1/2) = 0.11111111111111112 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1713_171328


namespace NUMINAMATH_CALUDE_triangle_property_l1713_171377

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.a^2 + t.b^2 = 2018 * t.c^2) :
  (2 * Real.sin t.A * Real.sin t.B * Real.cos t.C) / (1 - Real.cos t.C ^ 2) = 2017 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1713_171377


namespace NUMINAMATH_CALUDE_total_answer_key_ways_l1713_171341

/-- Represents a sequence of true-false answers -/
def TFSequence := List Bool

/-- Represents a sequence of multiple-choice answers -/
def MCSequence := List Nat

/-- Checks if a TFSequence is valid (no more than 3 consecutive true or false answers) -/
def isValidTFSequence (seq : TFSequence) : Bool :=
  sorry

/-- Checks if a MCSequence is valid (no consecutive answers are the same) -/
def isValidMCSequence (seq : MCSequence) : Bool :=
  sorry

/-- Counts the number of valid TFSequences of length 10 -/
def countValidTFSequences : Nat :=
  sorry

/-- Counts the number of valid MCSequences of length 5 with 6 choices each -/
def countValidMCSequences : Nat :=
  sorry

/-- The main theorem stating the total number of ways to write the answer key -/
theorem total_answer_key_ways :
  (countValidTFSequences * countValidMCSequences) =
  (countValidTFSequences * 3750) :=
by
  sorry

end NUMINAMATH_CALUDE_total_answer_key_ways_l1713_171341


namespace NUMINAMATH_CALUDE_x_percent_of_2x_is_10_l1713_171302

theorem x_percent_of_2x_is_10 (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * (2 * x) = 10) : 
  x = 10 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_x_percent_of_2x_is_10_l1713_171302


namespace NUMINAMATH_CALUDE_max_mn_for_exponential_intersection_max_mn_achieved_l1713_171300

/-- The maximum value of mn for a line mx + ny = 1 that intersects
    the graph of y = a^(x-1) at a fixed point, where a > 0 and a ≠ 1 -/
theorem max_mn_for_exponential_intersection (a : ℝ) (m n : ℝ) 
  (ha : a > 0) (ha_ne_one : a ≠ 1) : 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) → m*n ≤ 1/4 := by
  sorry

/-- The maximum value of mn is achieved when m = n = 1/2 -/
theorem max_mn_achieved (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (m n : ℝ), m*n = 1/4 ∧ 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_mn_for_exponential_intersection_max_mn_achieved_l1713_171300


namespace NUMINAMATH_CALUDE_range_of_a_l1713_171380

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1713_171380


namespace NUMINAMATH_CALUDE_trivia_team_total_score_l1713_171397

def trivia_team_points : Prop :=
  let total_members : ℕ := 12
  let absent_members : ℕ := 4
  let present_members : ℕ := total_members - absent_members
  let scores : List ℕ := [8, 12, 9, 5, 10, 7, 14, 11]
  scores.length = present_members ∧ scores.sum = 76

theorem trivia_team_total_score : trivia_team_points := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_total_score_l1713_171397


namespace NUMINAMATH_CALUDE_gwen_spent_nothing_l1713_171336

/-- Represents the amount of money Gwen received from her mom -/
def mom_money : ℤ := 8

/-- Represents the amount of money Gwen received from her dad -/
def dad_money : ℤ := 5

/-- Represents the difference in money Gwen has from her mom compared to her dad after spending -/
def difference_after_spending : ℤ := 3

/-- Represents the amount of money Gwen spent -/
def money_spent : ℤ := 0

theorem gwen_spent_nothing :
  (mom_money - money_spent) - (dad_money - money_spent) = difference_after_spending :=
sorry

end NUMINAMATH_CALUDE_gwen_spent_nothing_l1713_171336


namespace NUMINAMATH_CALUDE_gear_rotation_l1713_171345

/-- Represents a gear in the system -/
structure Gear where
  angle : Real

/-- Represents a system of two meshed gears -/
structure GearSystem where
  left : Gear
  right : Gear

/-- Rotates the left gear by a given angle -/
def rotateLeft (system : GearSystem) (θ : Real) : GearSystem :=
  { left := { angle := system.left.angle + θ },
    right := { angle := system.right.angle - θ } }

/-- Theorem stating that rotating the left gear by θ results in the right gear rotating by -θ -/
theorem gear_rotation (system : GearSystem) (θ : Real) :
  (rotateLeft system θ).right.angle = system.right.angle - θ :=
by sorry

end NUMINAMATH_CALUDE_gear_rotation_l1713_171345


namespace NUMINAMATH_CALUDE_smallest_integer_l1713_171382

theorem smallest_integer (a b : ℕ+) (ha : a = 60) (h : Nat.lcm a b / Nat.gcd a b = 44) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), (Nat.lcm a n / Nat.gcd a n = 44) → m ≤ n ∧ m = 165 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l1713_171382


namespace NUMINAMATH_CALUDE_bert_pencil_usage_l1713_171387

/-- The number of days it takes to use up a pencil given the total words per pencil and words per puzzle -/
def days_to_use_pencil (total_words_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  total_words_per_pencil / words_per_puzzle

/-- Theorem stating that it takes Bert 14 days to use up a pencil -/
theorem bert_pencil_usage : days_to_use_pencil 1050 75 = 14 := by
  sorry

#eval days_to_use_pencil 1050 75

end NUMINAMATH_CALUDE_bert_pencil_usage_l1713_171387


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l1713_171352

theorem farm_corn_cobs (field1_rows field1_cobs_per_row : ℕ)
                       (field2_rows field2_cobs_per_row : ℕ)
                       (field3_rows field3_cobs_per_row : ℕ)
                       (field4_rows field4_cobs_per_row : ℕ)
                       (h1 : field1_rows = 13 ∧ field1_cobs_per_row = 8)
                       (h2 : field2_rows = 16 ∧ field2_cobs_per_row = 12)
                       (h3 : field3_rows = 9 ∧ field3_cobs_per_row = 10)
                       (h4 : field4_rows = 20 ∧ field4_cobs_per_row = 6) :
  field1_rows * field1_cobs_per_row +
  field2_rows * field2_cobs_per_row +
  field3_rows * field3_cobs_per_row +
  field4_rows * field4_cobs_per_row = 506 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l1713_171352


namespace NUMINAMATH_CALUDE_dice_coloring_probability_l1713_171391

/-- Represents the number of faces on a die -/
def numFaces : ℕ := 6

/-- Represents the number of color options for each face -/
def numColors : ℕ := 3

/-- Represents a coloring of a die -/
def DieColoring := Fin numFaces → Fin numColors

/-- Represents whether two die colorings are equivalent under rotation -/
def areEquivalentUnderRotation (d1 d2 : DieColoring) : Prop :=
  ∃ (rotation : Equiv.Perm (Fin numFaces)), ∀ i, d1 i = d2 (rotation i)

/-- The total number of ways to color two dice -/
def totalColorings : ℕ := numColors^numFaces * numColors^numFaces

/-- The number of ways to color two dice that are equivalent under rotation -/
def equivalentColorings : ℕ := 8425

theorem dice_coloring_probability :
  (equivalentColorings : ℚ) / totalColorings = 8425 / 531441 := by sorry

end NUMINAMATH_CALUDE_dice_coloring_probability_l1713_171391


namespace NUMINAMATH_CALUDE_sum_lent_calculation_l1713_171320

-- Define the interest rate and time period
def interest_rate : ℚ := 3 / 100
def time_period : ℕ := 3

-- Define the theorem
theorem sum_lent_calculation (P : ℚ) : 
  P * interest_rate * time_period = P - 1820 → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_calculation_l1713_171320


namespace NUMINAMATH_CALUDE_part_one_part_two_l1713_171384

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the set A as a function of m
def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

-- Part 1
theorem part_one (m : ℝ) : (Aᶜ m = {1, 2}) → m = -3 := by sorry

-- Part 2
theorem part_two (m : ℝ) : (∃! x, x ∈ A m) → m = 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1713_171384


namespace NUMINAMATH_CALUDE_jakes_weight_l1713_171393

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 →
  jake_weight = 108 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l1713_171393


namespace NUMINAMATH_CALUDE_one_sheet_removal_median_l1713_171338

/-- Represents a collection of notes with pages and sheets -/
structure Notes where
  total_pages : ℕ
  total_sheets : ℕ
  last_sheet_pages : ℕ
  mk_notes_valid : total_pages = 2 * (total_sheets - 1) + last_sheet_pages

/-- Calculates the median page number after removing sheets -/
def median_after_removal (notes : Notes) (sheets_removed : ℕ) : ℕ :=
  (notes.total_pages - 2 * sheets_removed + 1) / 2

/-- Theorem stating that removing one sheet results in a median of 36 -/
theorem one_sheet_removal_median (notes : Notes)
  (h1 : notes.total_pages = 65)
  (h2 : notes.total_sheets = 33)
  (h3 : notes.last_sheet_pages = 1) :
  median_after_removal notes 1 = 36 := by
  sorry

#check one_sheet_removal_median

end NUMINAMATH_CALUDE_one_sheet_removal_median_l1713_171338


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1713_171321

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (a 2 + a 4 + a 6 = 3) →       -- Given condition
  (a 5 + a 7 + a 9 = 24) :=     -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1713_171321


namespace NUMINAMATH_CALUDE_walker_speed_l1713_171309

-- Define the speed of person B
def speed_B : ℝ := 3

-- Define the number of crossings
def num_crossings : ℕ := 5

-- Define the time period in hours
def time_period : ℝ := 1

-- Theorem statement
theorem walker_speed (speed_A : ℝ) : 
  (num_crossings : ℝ) / (speed_A + speed_B) = time_period → 
  speed_A = 2 := by
sorry

end NUMINAMATH_CALUDE_walker_speed_l1713_171309


namespace NUMINAMATH_CALUDE_quadratic_root_bound_l1713_171372

theorem quadratic_root_bound (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hx : a * x^2 + b * x + c = 0) : 
  |x| ≤ (2 * |a * c| + b^2) / (|a * b|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_bound_l1713_171372


namespace NUMINAMATH_CALUDE_projection_implies_y_value_l1713_171370

/-- Given two vectors v and w in ℝ², prove that if the projection of v onto w
    is [-8, -12], then the y-coordinate of v must be -56/3. -/
theorem projection_implies_y_value (v w : ℝ × ℝ) (y : ℝ) 
    (h1 : v = (2, y))
    (h2 : w = (4, 6))
    (h3 : (v • w / (w • w)) • w = (-8, -12)) :
  y = -56/3 := by
  sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l1713_171370
