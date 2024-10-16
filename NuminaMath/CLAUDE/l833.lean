import Mathlib

namespace NUMINAMATH_CALUDE_compound_interest_calculation_l833_83383

theorem compound_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℕ) (final_amount : ℝ) : 
  principal = 8000 →
  rate = 0.05 →
  time = 2 →
  final_amount = 8820 →
  final_amount = principal * (1 + rate) ^ time :=
sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l833_83383


namespace NUMINAMATH_CALUDE_honda_cars_count_l833_83322

-- Define the total number of cars
def total_cars : ℕ := 9000

-- Define the percentage of red Honda cars
def red_honda_percentage : ℚ := 90 / 100

-- Define the percentage of total red cars
def total_red_percentage : ℚ := 60 / 100

-- Define the percentage of red non-Honda cars
def red_non_honda_percentage : ℚ := 225 / 1000

-- Theorem statement
theorem honda_cars_count (honda_cars : ℕ) :
  (honda_cars : ℚ) * red_honda_percentage + 
  ((total_cars - honda_cars) : ℚ) * red_non_honda_percentage = 
  (total_cars : ℚ) * total_red_percentage →
  honda_cars = 5000 := by
sorry

end NUMINAMATH_CALUDE_honda_cars_count_l833_83322


namespace NUMINAMATH_CALUDE_two_pairs_exist_l833_83388

/-- A function that checks if a number consists of three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- The main theorem stating the existence of two distinct pairs of numbers
    satisfying the given conditions -/
theorem two_pairs_exist : ∃ (a b c d : ℕ),
  has_three_identical_digits (a * b) ∧
  has_three_identical_digits (a + b) ∧
  has_three_identical_digits (c * d) ∧
  has_three_identical_digits (c + d) ∧
  (a ≠ c ∨ b ≠ d) :=
sorry

end NUMINAMATH_CALUDE_two_pairs_exist_l833_83388


namespace NUMINAMATH_CALUDE_probability_both_days_is_correct_l833_83352

def num_students : ℕ := 5
def num_days : ℕ := 2

def probability_both_days : ℚ :=
  1 - (2 : ℚ) / (2 ^ num_students : ℚ)

theorem probability_both_days_is_correct :
  probability_both_days = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_days_is_correct_l833_83352


namespace NUMINAMATH_CALUDE_textbook_transfer_l833_83339

theorem textbook_transfer (initial_a initial_b transfer : ℕ) 
  (h1 : initial_a = 200)
  (h2 : initial_b = 200)
  (h3 : transfer = 40) :
  (initial_b + transfer) = (initial_a - transfer) * 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_textbook_transfer_l833_83339


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l833_83398

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l833_83398


namespace NUMINAMATH_CALUDE_contact_lenses_sales_l833_83335

theorem contact_lenses_sales (soft_price hard_price : ℕ) 
  (soft_hard_difference total_sales : ℕ) : 
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_difference = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ), 
    soft = hard + soft_hard_difference ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 :=
by sorry

end NUMINAMATH_CALUDE_contact_lenses_sales_l833_83335


namespace NUMINAMATH_CALUDE_total_items_in_jar_l833_83344

-- Define the number of candies
def total_candies : ℕ := 3409
def chocolate_candies : ℕ := 1462
def gummy_candies : ℕ := 1947

-- Define the number of secret eggs
def total_eggs : ℕ := 145
def eggs_with_one_prize : ℕ := 98
def eggs_with_two_prizes : ℕ := 38
def eggs_with_three_prizes : ℕ := 9

-- Theorem to prove
theorem total_items_in_jar : 
  total_candies + 
  (eggs_with_one_prize * 1 + eggs_with_two_prizes * 2 + eggs_with_three_prizes * 3) = 3610 := by
  sorry

end NUMINAMATH_CALUDE_total_items_in_jar_l833_83344


namespace NUMINAMATH_CALUDE_cubic_feet_to_cubic_inches_l833_83394

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume in cubic feet
def cubic_feet : ℕ := 4

-- Theorem statement
theorem cubic_feet_to_cubic_inches :
  cubic_feet * (inches_per_foot ^ 3) = 6912 := by
  sorry


end NUMINAMATH_CALUDE_cubic_feet_to_cubic_inches_l833_83394


namespace NUMINAMATH_CALUDE_existence_of_solution_l833_83331

theorem existence_of_solution : ∃ (a b : ℕ), 
  a > 1 ∧ b > 1 ∧ a^13 * b^31 = 6^2015 ∧ a = 2^155 ∧ b = 3^65 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l833_83331


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l833_83345

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 13*r + 12 = 0 ∧
    r^3 + b*r^2 + 17*r + 15 = 0 ∧
    s^3 + a*s^2 + 13*s + 12 = 0 ∧
    s^3 + b*s^2 + 17*s + 15 = 0) →
  a = 0 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l833_83345


namespace NUMINAMATH_CALUDE_f_plus_a_over_e_positive_sum_less_than_two_l833_83354

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem f_plus_a_over_e_positive (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, f a x + a / Real.exp 1 > 0 := by sorry

theorem sum_less_than_two (x₁ x₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : f (-1/2) x₁ = f (-1/2) x₂) :
  x₁ + x₂ < 2 := by sorry

end

end NUMINAMATH_CALUDE_f_plus_a_over_e_positive_sum_less_than_two_l833_83354


namespace NUMINAMATH_CALUDE_tangent_line_equation_l833_83391

/-- The function f(x) = x³ - 2x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at P -/
def m : ℝ := f' P.1

/-- Theorem: The equation of the tangent line to y = f(x) at P(1, 2) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - P.1) + P.2 ↔ x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l833_83391


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l833_83318

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l833_83318


namespace NUMINAMATH_CALUDE_profit_180_greater_than_170_l833_83365

/-- Sales data for 20 days -/
def sales_data : List (ℕ × ℕ) := [(150, 3), (160, 4), (170, 6), (180, 5), (190, 1), (200, 1)]

/-- Total number of days -/
def total_days : ℕ := 20

/-- Purchase price in yuan per kg -/
def purchase_price : ℚ := 6

/-- Selling price in yuan per kg -/
def selling_price : ℚ := 10

/-- Return price in yuan per kg -/
def return_price : ℚ := 4

/-- Calculate expected profit for a given purchase amount -/
def expected_profit (purchase_amount : ℕ) : ℚ :=
  sorry

/-- Theorem: Expected profit from 180 kg purchase is greater than 170 kg purchase -/
theorem profit_180_greater_than_170 :
  expected_profit 180 > expected_profit 170 :=
sorry

end NUMINAMATH_CALUDE_profit_180_greater_than_170_l833_83365


namespace NUMINAMATH_CALUDE_dane_daughters_flowers_l833_83358

def flowers_per_basket (initial_flowers_per_daughter : ℕ) (daughters : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (baskets : ℕ) : ℕ :=
  ((initial_flowers_per_daughter * daughters + new_flowers) - dead_flowers) / baskets

theorem dane_daughters_flowers :
  flowers_per_basket 5 2 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dane_daughters_flowers_l833_83358


namespace NUMINAMATH_CALUDE_coins_equal_dollar_l833_83399

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The theorem stating that the sum of the coins equals 100% of a dollar -/
theorem coins_equal_dollar :
  (nickel_value + 2 * dime_value + quarter_value + half_dollar_value) / cents_per_dollar * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_equal_dollar_l833_83399


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l833_83310

/-- Proves that the annual interest rate is 0.1 given the initial investment,
    final amount, and time period. -/
theorem interest_rate_calculation (initial_investment : ℝ) (final_amount : ℝ) (years : ℕ) :
  initial_investment = 3000 →
  final_amount = 3630.0000000000005 →
  years = 2 →
  ∃ (r : ℝ), r = 0.1 ∧ final_amount = initial_investment * (1 + r) ^ years :=
by sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l833_83310


namespace NUMINAMATH_CALUDE_exradii_product_bound_l833_83326

/-- For any triangle with side lengths a, b, c and exradii r_a, r_b, r_c,
    the product of the exradii does not exceed (3√3/8) times the product of the side lengths. -/
theorem exradii_product_bound (a b c r_a r_b r_c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_exradii : r_a * (b + c - a) = r_b * (c + a - b) ∧ 
               r_b * (c + a - b) = r_c * (a + b - c)) : 
  r_a * r_b * r_c ≤ (3 * Real.sqrt 3 / 8) * a * b * c := by
  sorry

#check exradii_product_bound

end NUMINAMATH_CALUDE_exradii_product_bound_l833_83326


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l833_83336

def base7_to_base10 (n : ℕ) : ℕ := sorry

def diamonds : ℕ := 6352
def ancient_coins : ℕ := 3206
def silver : ℕ := 156

theorem pirate_treasure_sum :
  base7_to_base10 diamonds + base7_to_base10 ancient_coins + base7_to_base10 silver = 3465 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_sum_l833_83336


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l833_83337

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 10*x - 3 = 0 ↔ (x + 5)^2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l833_83337


namespace NUMINAMATH_CALUDE_complex_equation_solution_l833_83347

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 11 + 7 * Complex.I) : 
  z = 7 - 11 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l833_83347


namespace NUMINAMATH_CALUDE_triangle_properties_l833_83374

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (5 * 2, 12 * 2, 13 * 2)) 
  (h_perimeter : a + b + c = 60) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 > 100) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l833_83374


namespace NUMINAMATH_CALUDE_max_weighings_for_15_coins_l833_83300

/-- Represents a coin which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing -/
inductive WeighingResult
| left_heavier : WeighingResult
| right_heavier : WeighingResult
| equal : WeighingResult

/-- A function that simulates weighing two groups of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighingResult := sorry

/-- A function that finds the counterfeit coin -/
def find_counterfeit (coins : List Coin) : Nat → Option Coin := sorry

theorem max_weighings_for_15_coins :
  ∀ (coins : List Coin),
    coins.length = 15 →
    (∃! c, c ∈ coins ∧ c = Coin.counterfeit) →
    ∃ n, n ≤ 3 ∧ (find_counterfeit coins n).isSome ∧
        ∀ m, m < n → (find_counterfeit coins m).isNone := by sorry

#check max_weighings_for_15_coins

end NUMINAMATH_CALUDE_max_weighings_for_15_coins_l833_83300


namespace NUMINAMATH_CALUDE_equation_transformation_l833_83360

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^3 + x^2 - 6*x + 1 = 0 ↔ x*(x^2*y - 6) + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_transformation_l833_83360


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l833_83333

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → s^2 = s * Real.sqrt 2 → 4 * s = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l833_83333


namespace NUMINAMATH_CALUDE_bag_problem_l833_83386

theorem bag_problem (total_slips : ℕ) (value1 value2 : ℝ) (expected_value : ℝ) :
  total_slips = 12 →
  value1 = 2 →
  value2 = 7 →
  expected_value = 3.25 →
  ∃ (slips_with_value1 : ℕ),
    slips_with_value1 ≤ total_slips ∧
    (slips_with_value1 : ℝ) / total_slips * value1 +
    ((total_slips - slips_with_value1) : ℝ) / total_slips * value2 = expected_value ∧
    slips_with_value1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l833_83386


namespace NUMINAMATH_CALUDE_semicircles_area_ratio_l833_83316

theorem semicircles_area_ratio (r : ℝ) (hr : r > 0) : 
  let circle_area := π * r^2
  let semicircle1_area := π * (r/2)^2 / 2
  let semicircle2_area := π * (r/3)^2 / 2
  (semicircle1_area + semicircle2_area) / circle_area = 13/72 := by
  sorry

end NUMINAMATH_CALUDE_semicircles_area_ratio_l833_83316


namespace NUMINAMATH_CALUDE_total_donation_is_1570_l833_83357

/-- Represents the donation amounts to different parks -/
structure Donations where
  treetown_and_forest : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ

/-- Calculates the total donation to all three parks -/
def total_donation (d : Donations) : ℝ :=
  d.treetown_and_forest + d.animal_preservation

/-- Theorem stating the total donation to all three parks -/
theorem total_donation_is_1570 (d : Donations) 
  (h1 : d.treetown_and_forest = 570)
  (h2 : d.forest_reserve = d.animal_preservation + 140)
  (h3 : d.treetown_and_forest = d.forest_reserve + d.animal_preservation) : 
  total_donation d = 1570 := by
  sorry

#check total_donation_is_1570

end NUMINAMATH_CALUDE_total_donation_is_1570_l833_83357


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l833_83382

/-- Given a function y = log_a(x + 3) - 1 where a > 0 and a ≠ 1, 
    its graph always passes through a fixed point A.
    If point A lies on the line mx + ny + 2 = 0 where mn > 0,
    then the minimum value of 1/m + 2/n is 4. -/
theorem min_value_sum_reciprocals (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : m > 0) (h4 : n > 0) (h5 : m * n > 0) :
  let f : ℝ → ℝ := λ x => (Real.log x) / (Real.log a) - 1
  let A : ℝ × ℝ := (-2, -1)
  (f (A.1 + 3) = A.2) →
  (m * A.1 + n * A.2 + 2 = 0) →
  (∀ x y, f y = x → m * x + n * y + 2 = 0) →
  (1 / m + 2 / n) ≥ 4 ∧ ∃ m₀ n₀, 1 / m₀ + 2 / n₀ = 4 := by
  sorry


end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l833_83382


namespace NUMINAMATH_CALUDE_local_face_value_difference_l833_83348

/-- The numeral we're working with -/
def numeral : ℕ := 65793

/-- The digit we're focusing on -/
def digit : ℕ := 7

/-- The place value of the digit in the numeral (hundreds) -/
def place_value : ℕ := 100

/-- The local value of the digit in the numeral -/
def local_value : ℕ := digit * place_value

/-- The face value of the digit -/
def face_value : ℕ := digit

/-- Theorem stating the difference between local value and face value -/
theorem local_face_value_difference :
  local_value - face_value = 693 := by sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l833_83348


namespace NUMINAMATH_CALUDE_base4_21012_to_decimal_l833_83361

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21012_to_decimal :
  base4_to_decimal [2, 1, 0, 1, 2] = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_21012_to_decimal_l833_83361


namespace NUMINAMATH_CALUDE_lcm_14_18_20_l833_83317

theorem lcm_14_18_20 : Nat.lcm 14 (Nat.lcm 18 20) = 1260 := by sorry

end NUMINAMATH_CALUDE_lcm_14_18_20_l833_83317


namespace NUMINAMATH_CALUDE_money_at_departure_l833_83355

def money_at_arrival : ℕ := 87
def money_difference : ℕ := 71

theorem money_at_departure : 
  money_at_arrival - money_difference = 16 := by sorry

end NUMINAMATH_CALUDE_money_at_departure_l833_83355


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l833_83334

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) * (2 : ℝ)^2 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l833_83334


namespace NUMINAMATH_CALUDE_john_tax_increase_l833_83305

/-- Calculates the increase in taxes paid given old and new tax rates and incomes -/
def tax_increase (old_rate new_rate : ℚ) (old_income new_income : ℕ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that John's tax increase is $250,000 -/
theorem john_tax_increase :
  let old_rate : ℚ := 1/5
  let new_rate : ℚ := 3/10
  let old_income : ℕ := 1000000
  let new_income : ℕ := 1500000
  tax_increase old_rate new_rate old_income new_income = 250000 := by
  sorry

#eval tax_increase (1/5) (3/10) 1000000 1500000

end NUMINAMATH_CALUDE_john_tax_increase_l833_83305


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l833_83396

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan of brownies -/
def pan : Rectangle := { length := 24, width := 20 }

/-- A single brownie piece -/
def piece : Rectangle := { length := 3, width := 4 }

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (area pan) / (area piece)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l833_83396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l833_83330

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l833_83330


namespace NUMINAMATH_CALUDE_triangle_inequality_l833_83377

theorem triangle_inequality (a : ℝ) : a ≥ 1 →
  (3 * a + (a + 1) ≥ 2) ∧
  (3 * (a - 1) + 2 * a ≥ 2) ∧
  (3 * 1 + 3 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l833_83377


namespace NUMINAMATH_CALUDE_store_comparison_l833_83370

/-- Represents the cost difference between Store B and Store A -/
def cost_difference (x : ℝ) : ℝ := 520 - 2.5 * x

theorem store_comparison (x : ℝ) (h : x > 40) :
  cost_difference x = 520 - 2.5 * x ∧
  cost_difference 80 > 0 :=
sorry

#check store_comparison

end NUMINAMATH_CALUDE_store_comparison_l833_83370


namespace NUMINAMATH_CALUDE_triangle_inequality_l833_83381

theorem triangle_inequality (a b c p q r : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- triangle side lengths are positive
  a + b > c ∧ b + c > a ∧ c + a > b ∧  -- triangle inequality
  p + q + r = 0 →  -- sum of p, q, r is zero
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l833_83381


namespace NUMINAMATH_CALUDE_katie_spent_sixty_dollars_l833_83324

/-- The amount Katie spent on flowers -/
def katies_spending (flower_cost : ℕ) (roses : ℕ) (daisies : ℕ) : ℕ :=
  flower_cost * (roses + daisies)

/-- Theorem: Katie spent 60 dollars on flowers -/
theorem katie_spent_sixty_dollars : katies_spending 6 5 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_katie_spent_sixty_dollars_l833_83324


namespace NUMINAMATH_CALUDE_correct_divisor_l833_83340

theorem correct_divisor (incorrect_result : ℝ) (dividend : ℝ) (h1 : incorrect_result = 204) (h2 : dividend = 30.6) :
  ∃ (correct_divisor : ℝ), 
    dividend / (correct_divisor * 10) = incorrect_result ∧
    correct_divisor = (dividend / incorrect_result) / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l833_83340


namespace NUMINAMATH_CALUDE_min_value_of_f_l833_83304

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ * a₄ - a₂ * a₃ = 1) : 
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₄ - x₂ * x₃ = 1 → 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₁*x₃ + x₂*x₄ ≥ m ∧
    ∃ (y₁ y₂ y₃ y₄ : ℝ), y₁ * y₄ - y₂ * y₃ = 1 ∧
      y₁^2 + y₂^2 + y₃^2 + y₄^2 + y₁*y₃ + y₂*y₄ = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l833_83304


namespace NUMINAMATH_CALUDE_fraction_less_than_two_l833_83369

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_two_l833_83369


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l833_83346

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := (1/2) * a * b
  let c := Real.sqrt (b^2 - a^2)
  let area2 := (1/2) * a * c
  min area1 area2 = 6 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l833_83346


namespace NUMINAMATH_CALUDE_smallest_n_for_odd_ratio_l833_83376

def concatenate_decimal_expansions (k : ℕ) : ℕ :=
  sorry

def X (n : ℕ) : ℕ := concatenate_decimal_expansions n

theorem smallest_n_for_odd_ratio :
  (∀ n : ℕ, n ≥ 2 → n < 5 → ¬(Odd (X n / 1024^n))) ∧
  (Odd (X 5 / 1024^5)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_odd_ratio_l833_83376


namespace NUMINAMATH_CALUDE_largest_initial_number_l833_83313

theorem largest_initial_number : 
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧ 
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 → 
      ¬(∃ (a' b' c' d' e' : ℕ), 
        n + a' + b' + c' + d' + e' = 200 ∧
        a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
        n % a' ≠ 0 ∧ n % b' ≠ 0 ∧ n % c' ≠ 0 ∧ n % d' ≠ 0 ∧ n % e' ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l833_83313


namespace NUMINAMATH_CALUDE_strap_problem_l833_83349

theorem strap_problem (shorter longer : ℝ) 
  (h1 : shorter + longer = 64)
  (h2 : longer = shorter + 48) :
  longer / shorter = 7 := by
  sorry

end NUMINAMATH_CALUDE_strap_problem_l833_83349


namespace NUMINAMATH_CALUDE_constant_calculation_l833_83307

theorem constant_calculation (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 8100) → (∀ m : ℤ, m ≤ 8) → c = 126.5625 := by
sorry

end NUMINAMATH_CALUDE_constant_calculation_l833_83307


namespace NUMINAMATH_CALUDE_M_greater_than_N_l833_83372

theorem M_greater_than_N : ∀ x : ℝ, x^2 + 4*x - 2 > 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l833_83372


namespace NUMINAMATH_CALUDE_three_equidistant_lines_l833_83323

/-- A point in a plane represented by its coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a plane represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if three points are not collinear -/
def nonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- Returns true if a line is equidistant from three points -/
def equidistantLine (l : Line2D) (p1 p2 p3 : Point2D) : Prop :=
  let d1 := |l.a * p1.x + l.b * p1.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d2 := |l.a * p2.x + l.b * p2.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d3 := |l.a * p3.x + l.b * p3.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d1 = d2 ∧ d2 = d3

/-- Main theorem: There are exactly three lines equidistant from three non-collinear points -/
theorem three_equidistant_lines (p1 p2 p3 : Point2D) 
  (h : nonCollinear p1 p2 p3) : 
  ∃! (s : Finset Line2D), s.card = 3 ∧ ∀ l ∈ s, equidistantLine l p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_three_equidistant_lines_l833_83323


namespace NUMINAMATH_CALUDE_weekly_commute_time_l833_83353

def bike_time : ℕ := 30
def bus_time : ℕ := bike_time + 10
def friend_ride_time : ℕ := bike_time / 3

def total_commute_time : ℕ :=
  1 * bike_time + 3 * bus_time + 1 * friend_ride_time

theorem weekly_commute_time : total_commute_time = 160 := by
  sorry

end NUMINAMATH_CALUDE_weekly_commute_time_l833_83353


namespace NUMINAMATH_CALUDE_virginia_average_rainfall_l833_83343

/-- The average rainfall in Virginia over five months --/
def average_rainfall (march april may june july : Float) : Float :=
  (march + april + may + june + july) / 5

/-- Theorem stating that the average rainfall in Virginia is 4 inches --/
theorem virginia_average_rainfall :
  average_rainfall 3.79 4.5 3.95 3.09 4.67 = 4 := by
  sorry

end NUMINAMATH_CALUDE_virginia_average_rainfall_l833_83343


namespace NUMINAMATH_CALUDE_max_sequence_length_l833_83338

/-- A sequence satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) (n m : ℕ) : Prop :=
  (∀ k, k < n → a k ≤ m) ∧
  (∀ k, 1 < k ∧ k < n - 1 → a (k - 1) ≠ a (k + 1)) ∧
  (∀ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < n →
    ¬(a i₁ = a i₃ ∧ a i₁ ≠ a i₂ ∧ a i₂ = a i₄))

/-- The maximum length of a valid sequence -/
def MaxSequenceLength (m : ℕ) : ℕ :=
  4 * m - 2

/-- Theorem: The maximum length of a valid sequence is 4m - 2 -/
theorem max_sequence_length (m : ℕ) (h : m > 0) :
  (∃ a n, n = MaxSequenceLength m ∧ ValidSequence a n m) ∧
  (∀ a n, ValidSequence a n m → n ≤ MaxSequenceLength m) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l833_83338


namespace NUMINAMATH_CALUDE_intersection_point_AB_CD_l833_83359

def A : ℝ × ℝ × ℝ := (8, -9, 5)
def B : ℝ × ℝ × ℝ := (18, -19, 15)
def C : ℝ × ℝ × ℝ := (2, 5, -8)
def D : ℝ × ℝ × ℝ := (4, -3, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem intersection_point_AB_CD :
  line_intersection A B C D = (16, -19, 13) := by sorry

end NUMINAMATH_CALUDE_intersection_point_AB_CD_l833_83359


namespace NUMINAMATH_CALUDE_third_square_properties_l833_83328

/-- Given two squares with perimeters 40 and 32, prove that a third square
    with area equal to the difference of their areas has perimeter 24
    and integer side length. -/
theorem third_square_properties (s₁ s₂ s₃ : ℝ) : 
  (4 * s₁ = 40) →  -- Perimeter of first square
  (4 * s₂ = 32) →  -- Perimeter of second square
  (s₃^2 = s₁^2 - s₂^2) →  -- Area of third square
  (4 * s₃ = 24 ∧ ∃ n : ℕ, s₃ = n) := by
  sorry

#check third_square_properties

end NUMINAMATH_CALUDE_third_square_properties_l833_83328


namespace NUMINAMATH_CALUDE_divisor_sum_relation_l833_83325

theorem divisor_sum_relation (n f g : ℕ) : 
  n > 1 → 
  (∃ d1 d2 : ℕ, d1 ∣ n ∧ d2 ∣ n ∧ d1 ≤ d2 ∧ ∀ d : ℕ, d ∣ n → d = d1 ∨ d ≥ d2 → f = d1 + d2) →
  (∃ d3 d4 : ℕ, d3 ∣ n ∧ d4 ∣ n ∧ d3 ≥ d4 ∧ ∀ d : ℕ, d ∣ n → d = d3 ∨ d ≤ d4 → g = d3 + d4) →
  n = (g * (f - 1)) / f :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_relation_l833_83325


namespace NUMINAMATH_CALUDE_prove_a_equals_two_l833_83342

/-- Given two differentiable functions f and g on ℝ, prove that a = 2 -/
theorem prove_a_equals_two
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_g_nonzero : ∀ x, g x ≠ 0)
  (h_f_def : ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x)
  (h_inequality : ∀ x, f x * (deriv g x) < (deriv f x) * g x)
  (h_sum : f 1 / g 1 + f (-1) / g (-1) = 5) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x :=
sorry

end NUMINAMATH_CALUDE_prove_a_equals_two_l833_83342


namespace NUMINAMATH_CALUDE_peters_socks_l833_83302

theorem peters_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2*x + 3*y + 5*z = 45 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_peters_socks_l833_83302


namespace NUMINAMATH_CALUDE_inverse_function_intersection_l833_83364

def f (x : ℝ) : ℝ := 3 * x^2 - 8

theorem inverse_function_intersection (x : ℝ) : 
  f x = x ↔ x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_intersection_l833_83364


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l833_83320

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 72 -/
def M : ℕ := sorry

theorem sum_of_digits_M :
  is_five_digit M ∧
  digit_product M = 72 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 72 → n ≤ M) →
  digit_sum M = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l833_83320


namespace NUMINAMATH_CALUDE_data_grouping_l833_83378

theorem data_grouping (max min interval : ℕ) (h1 : max = 145) (h2 : min = 50) (h3 : interval = 10) :
  (max - min + interval - 1) / interval = 10 := by
  sorry

end NUMINAMATH_CALUDE_data_grouping_l833_83378


namespace NUMINAMATH_CALUDE_vector_problem_l833_83379

/-- Given two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1/2, 1)
  let c : ℝ × ℝ := (a.1 + 2*b.1, a.2 + 2*b.2)
  let d : ℝ × ℝ := (2*a.1 - b.1, 2*a.2 - b.2)
  are_parallel c d →
  (c.1 - 2*d.1, c.2 - 2*d.2) = (-1, -2) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l833_83379


namespace NUMINAMATH_CALUDE_residual_plot_vertical_axis_l833_83375

/-- A residual plot used in residual analysis. -/
structure ResidualPlot where
  vertical_axis : Set ℝ
  horizontal_axis : Set ℝ

/-- The definition of a residual in the context of residual analysis. -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents the residuals. -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) :
  plot.vertical_axis = Set.range (λ r : Residual => r) :=
sorry

end NUMINAMATH_CALUDE_residual_plot_vertical_axis_l833_83375


namespace NUMINAMATH_CALUDE_kayla_kimiko_age_ratio_l833_83385

/-- Proves that the ratio of Kayla's age to Kimiko's age is 1:2 -/
theorem kayla_kimiko_age_ratio :
  let kimiko_age : ℕ := 26
  let min_driving_age : ℕ := 18
  let years_until_driving : ℕ := 5
  let kayla_age : ℕ := min_driving_age - years_until_driving
  (kayla_age : ℚ) / kimiko_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_kayla_kimiko_age_ratio_l833_83385


namespace NUMINAMATH_CALUDE_gumball_probability_l833_83371

theorem gumball_probability (blue_prob : ℚ) (pink_prob : ℚ) : 
  blue_prob^2 = 9/49 → pink_prob = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l833_83371


namespace NUMINAMATH_CALUDE_division_of_decimals_l833_83314

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l833_83314


namespace NUMINAMATH_CALUDE_polynomial_simplification_l833_83321

theorem polynomial_simplification (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) =
  y^6 - y^5 + 2 * y^4 + y^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l833_83321


namespace NUMINAMATH_CALUDE_sqrt_40_div_sqrt_5_l833_83329

theorem sqrt_40_div_sqrt_5 : Real.sqrt 40 / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40_div_sqrt_5_l833_83329


namespace NUMINAMATH_CALUDE_race_distance_l833_83311

/-- Calculates the total distance of a race given the conditions --/
theorem race_distance (speed_a speed_b : ℝ) (head_start winning_margin : ℝ) : 
  speed_a / speed_b = 5 / 4 →
  head_start = 100 →
  winning_margin = 200 →
  (speed_a * ((head_start + winning_margin) / speed_b)) - head_start = 600 :=
by
  sorry


end NUMINAMATH_CALUDE_race_distance_l833_83311


namespace NUMINAMATH_CALUDE_f_properties_l833_83393

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ x_max : ℝ, x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_max = (2 - Real.sqrt 3) / 2 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x_max) ∧
  (∃ x_min : ℝ, x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_min = -Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x_min ≤ f x) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l833_83393


namespace NUMINAMATH_CALUDE_height_opposite_y_is_8_l833_83389

/-- Regular triangle XYZ with pillars -/
structure Triangle where
  /-- Side length of the triangle -/
  side : ℝ
  /-- Height of pillar at X -/
  height_x : ℝ
  /-- Height of pillar at Y -/
  height_y : ℝ
  /-- Height of pillar at Z -/
  height_z : ℝ

/-- Calculate the height of the pillar opposite to Y -/
def height_opposite_y (t : Triangle) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the height of the pillar opposite Y is 8m -/
theorem height_opposite_y_is_8 (t : Triangle) 
  (h_regular : t.side > 0)
  (h_x : t.height_x = 8)
  (h_y : t.height_y = 5)
  (h_z : t.height_z = 7) : 
  height_opposite_y t = 8 :=
sorry

end NUMINAMATH_CALUDE_height_opposite_y_is_8_l833_83389


namespace NUMINAMATH_CALUDE_digit_doubling_theorem_l833_83384

def sumOfDigits (n : ℕ) : ℕ := sorry

def doubleDigitSum (n : ℕ) : ℕ := 2 * (sumOfDigits n)

def eventuallyOneDigit (n : ℕ) : Prop :=
  ∃ k, ∃ m : ℕ, (m < 10) ∧ (Nat.iterate doubleDigitSum k n = m)

theorem digit_doubling_theorem :
  (∀ n : ℕ, n ≠ 18 → doubleDigitSum n ≠ n) ∧
  (doubleDigitSum 18 = 18) ∧
  (∀ n : ℕ, n ≠ 18 → eventuallyOneDigit n) := by sorry

end NUMINAMATH_CALUDE_digit_doubling_theorem_l833_83384


namespace NUMINAMATH_CALUDE_john_shorter_than_rebeca_l833_83356

def height_difference (john_height lena_height rebeca_height : ℕ) : Prop :=
  (john_height = lena_height + 15) ∧
  (john_height < rebeca_height) ∧
  (john_height = 152) ∧
  (lena_height + rebeca_height = 295)

theorem john_shorter_than_rebeca (john_height lena_height rebeca_height : ℕ) :
  height_difference john_height lena_height rebeca_height →
  rebeca_height - john_height = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_john_shorter_than_rebeca_l833_83356


namespace NUMINAMATH_CALUDE_cubic_function_and_tangent_lines_l833_83368

/-- Given a cubic function f(x) = ax³ + b with a tangent line y = 3x - 1 at x = 1,
    prove that f(x) = x³ + 1 and find the equations of tangent lines passing through (-1, 0) --/
theorem cubic_function_and_tangent_lines 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b)
  (h2 : ∃ c d, ∀ x, (3 : ℝ) * x - 1 = c * (x - 1) + d ∧ f 1 = d ∧ (deriv f) 1 = c) :
  (∀ x, f x = x^3 + 1) ∧ 
  (∃ m₁ m₂ : ℝ, 
    (m₁ = 3 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₁) ∨ 
    (m₂ = 3/4 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₂)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_and_tangent_lines_l833_83368


namespace NUMINAMATH_CALUDE_inequality_holds_only_for_m_equals_negative_four_l833_83315

theorem inequality_holds_only_for_m_equals_negative_four :
  ∀ m : ℝ, (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) ↔ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_only_for_m_equals_negative_four_l833_83315


namespace NUMINAMATH_CALUDE_distance_minus_two_to_three_l833_83351

-- Define the distance function between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_two_to_three : distance (-2) 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_two_to_three_l833_83351


namespace NUMINAMATH_CALUDE_square_divisible_into_2020_elegant_triangles_l833_83397

/-- An elegant triangle is a right-angled triangle where one leg is 10 times longer than the other. -/
def ElegantTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a = 10*b ∨ b = 10*a)

/-- A square can be divided into n identical elegant triangles. -/
def SquareDivisibleIntoElegantTriangles (n : ℕ) : Prop :=
  ∃ (s a b c : ℝ), s > 0 ∧ ElegantTriangle a b c ∧ 
    (n : ℝ) * (1/2 * a * b) = s^2

theorem square_divisible_into_2020_elegant_triangles :
  SquareDivisibleIntoElegantTriangles 2020 := by
  sorry


end NUMINAMATH_CALUDE_square_divisible_into_2020_elegant_triangles_l833_83397


namespace NUMINAMATH_CALUDE_buckets_required_l833_83380

/-- The number of buckets required to fill a tank with the original bucket size,
    given that 62.5 buckets are needed when the bucket capacity is reduced to two-fifths. -/
theorem buckets_required (original_buckets : ℝ) : 
  (62.5 * (2/5) * original_buckets = original_buckets) → original_buckets = 25 := by
  sorry

end NUMINAMATH_CALUDE_buckets_required_l833_83380


namespace NUMINAMATH_CALUDE_students_in_sunghoons_class_l833_83308

theorem students_in_sunghoons_class 
  (jisoo_students : ℕ) 
  (product : ℕ) 
  (h1 : jisoo_students = 36)
  (h2 : jisoo_students * sunghoon_students = product)
  (h3 : product = 1008) : 
  sunghoon_students = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_sunghoons_class_l833_83308


namespace NUMINAMATH_CALUDE_cube_cuboid_volume_ratio_l833_83366

theorem cube_cuboid_volume_ratio :
  let cube_side : ℝ := 1
  let cuboid_width : ℝ := 50 / 100
  let cuboid_length : ℝ := 50 / 100
  let cuboid_height : ℝ := 20 / 100
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume / cuboid_volume = 20 := by
    sorry

end NUMINAMATH_CALUDE_cube_cuboid_volume_ratio_l833_83366


namespace NUMINAMATH_CALUDE_auto_finance_credit_l833_83373

/-- Proves that the credit extended by automobile finance companies is $40 billion given the specified conditions -/
theorem auto_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_companies_fraction : ℝ)
  (h1 : total_credit = 342.857)
  (h2 : auto_credit_percentage = 0.35)
  (h3 : finance_companies_fraction = 1/3) :
  finance_companies_fraction * (auto_credit_percentage * total_credit) = 40 := by
  sorry

end NUMINAMATH_CALUDE_auto_finance_credit_l833_83373


namespace NUMINAMATH_CALUDE_calculation_result_l833_83367

theorem calculation_result : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l833_83367


namespace NUMINAMATH_CALUDE_total_cost_theorem_l833_83362

-- Define the cost of individual items
def eraser_cost : ℝ := sorry
def pen_cost : ℝ := sorry
def marker_cost : ℝ := sorry

-- Define the conditions
axiom condition1 : eraser_cost + 3 * pen_cost + 2 * marker_cost = 240
axiom condition2 : 2 * eraser_cost + 4 * marker_cost + 5 * pen_cost = 440

-- Define the theorem to prove
theorem total_cost_theorem :
  3 * eraser_cost + 4 * pen_cost + 6 * marker_cost = 520 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l833_83362


namespace NUMINAMATH_CALUDE_compare_large_powers_l833_83303

theorem compare_large_powers : 100^100 > 50^50 * 150^50 := by
  sorry

end NUMINAMATH_CALUDE_compare_large_powers_l833_83303


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l833_83350

theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f := fun x => a * Real.sin (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (π / 3) = 3) → a = 3 ∧ c = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l833_83350


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l833_83312

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors Ice-cream-o-rama can create -/
theorem ice_cream_flavors : distribute 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l833_83312


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l833_83392

theorem sqrt_expression_equals_two : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l833_83392


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l833_83341

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ (x - y)^2 = 9) →
  p = Real.sqrt (4*q + 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l833_83341


namespace NUMINAMATH_CALUDE_part_time_employees_l833_83387

theorem part_time_employees (total_employees full_time_employees : ℕ) 
  (h1 : total_employees = 65134)
  (h2 : full_time_employees = 63093)
  (h3 : total_employees ≥ full_time_employees) :
  total_employees - full_time_employees = 2041 := by
  sorry

end NUMINAMATH_CALUDE_part_time_employees_l833_83387


namespace NUMINAMATH_CALUDE_parallel_implies_parallel_to_intersection_l833_83395

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a line lies on a plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns the intersection line of two planes -/
def intersection (p1 p2 : Plane3D) : Line3D :=
  sorry

theorem parallel_implies_parallel_to_intersection
  (a b c : Line3D) (M N : Plane3D)
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : lies_on a M)
  (h3 : lies_on b N)
  (h4 : c = intersection M N)
  (h5 : parallel a b) :
  parallel a c :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_parallel_to_intersection_l833_83395


namespace NUMINAMATH_CALUDE_number_problem_l833_83301

theorem number_problem (x : ℝ) : 0.95 * x - 12 = 178 ↔ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l833_83301


namespace NUMINAMATH_CALUDE_singleton_quadratic_set_l833_83306

theorem singleton_quadratic_set (m : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + m = 0) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_singleton_quadratic_set_l833_83306


namespace NUMINAMATH_CALUDE_general_term_k_n_l833_83390

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  a2_geometric_mean : a 2 ^ 2 = a 1 * a 4
  geometric_subseq : ∀ n, a (3^n) / a (3^(n-1)) = a 3 / a 1

/-- The theorem stating the general term of k_n -/
theorem general_term_k_n (seq : ArithmeticSequence) : 
  ∀ n : ℕ, ∃ k_n : ℕ, seq.a k_n = seq.a 1 * (3 : ℝ)^(n-1) ∧ k_n = 3^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_k_n_l833_83390


namespace NUMINAMATH_CALUDE_angle_complement_l833_83309

theorem angle_complement (A : ℝ) : 
  A = 45 → 90 - A = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_complement_l833_83309


namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l833_83319

-- Define the capabilities of Digital Earth
def can_simulate_environmental_impact : Prop := True
def can_monitor_crop_pests : Prop := True
def can_predict_submerged_areas : Prop := True
def can_simulate_past_environments : Prop := True

-- Define the statement to be proven false
def incorrect_statement : Prop :=
  ∃ (can_predict_future : Prop),
    can_predict_future ∧ ¬can_simulate_past_environments

-- Theorem statement
theorem digital_earth_capabilities :
  can_simulate_environmental_impact →
  can_monitor_crop_pests →
  can_predict_submerged_areas →
  can_simulate_past_environments →
  ¬incorrect_statement :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l833_83319


namespace NUMINAMATH_CALUDE_farmer_land_usage_l833_83363

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0

/-- The amount of land used for corn in acres -/
def corn_land : ℕ := 376

/-- The total amount of land used by the farmer in acres -/
def total_land : ℕ := 1034

/-- Theorem stating that given the land ratio and corn land usage, 
    the total land used by the farmer is 1034 acres -/
theorem farmer_land_usage : 
  (land_ratio 2 : ℚ) / (land_ratio 0 + land_ratio 1 + land_ratio 2 : ℚ) * total_land = corn_land :=
by sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l833_83363


namespace NUMINAMATH_CALUDE_parallelogram_side_lengths_l833_83332

/-- A parallelogram with the given properties has sides of length 4 and 12 -/
theorem parallelogram_side_lengths 
  (perimeter : ℝ) 
  (triangle_perimeter_diff : ℝ) 
  (h_perimeter : perimeter = 32) 
  (h_diff : triangle_perimeter_diff = 8) :
  ∃ (a b : ℝ), a + b = perimeter / 2 ∧ b - a = triangle_perimeter_diff ∧ a = 4 ∧ b = 12 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_side_lengths_l833_83332


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l833_83327

/-- Represents the charge for a cargo based on its weight range -/
def charge (weight : ℕ) : ℚ :=
  if 3 ≤ weight ∧ weight ≤ 5 then 5/2
  else if 6 ≤ weight ∧ weight ≤ 8 then 4
  else if 9 ≤ weight ∧ weight ≤ 12 then 6
  else if 13 ≤ weight ∧ weight ≤ 15 then 8
  else 0

/-- Calculates the daily earnings based on the number of deliveries for each weight -/
def dailyEarnings (deliveries : List (ℕ × ℕ)) : ℚ :=
  deliveries.foldl (fun acc (weight, count) => acc + charge weight * count) 0

/-- Cherry's daily delivery schedule -/
def cherryDeliveries : List (ℕ × ℕ) := [(5, 4), (8, 2), (10, 3), (14, 1)]

/-- Number of days in a week -/
def daysInWeek : ℕ := 7

/-- Theorem stating that Cherry's weekly earnings equal $308 -/
theorem cherry_weekly_earnings : 
  dailyEarnings cherryDeliveries * daysInWeek = 308 := by
  sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l833_83327
