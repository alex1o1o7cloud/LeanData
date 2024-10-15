import Mathlib

namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l3273_327345

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

-- Define the fractions in their respective bases
def fraction1_numerator : List Nat := [4, 6, 2]
def fraction1_denominator : List Nat := [2, 1]
def fraction2_numerator : List Nat := [4, 4, 1]
def fraction2_denominator : List Nat := [3, 3]

-- Define the bases
def base1 : Nat := 8
def base2 : Nat := 4
def base3 : Nat := 5

-- State the theorem
theorem base_conversion_sum_equality :
  (baseToDecimal fraction1_numerator base1 / baseToDecimal fraction1_denominator base2 : ℚ) +
  (baseToDecimal fraction2_numerator base3 / baseToDecimal fraction2_denominator base2 : ℚ) =
  499 / 15 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l3273_327345


namespace NUMINAMATH_CALUDE_total_distance_calculation_l3273_327352

/-- Calculates the total distance traveled given fuel efficiencies and fuel used for different driving conditions -/
theorem total_distance_calculation (city_efficiency highway_efficiency gravel_efficiency : ℝ)
  (city_fuel highway_fuel gravel_fuel : ℝ) : 
  city_efficiency = 15 →
  highway_efficiency = 25 →
  gravel_efficiency = 18 →
  city_fuel = 2.5 →
  highway_fuel = 3.8 →
  gravel_fuel = 1.7 →
  city_efficiency * city_fuel + highway_efficiency * highway_fuel + gravel_efficiency * gravel_fuel = 163.1 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l3273_327352


namespace NUMINAMATH_CALUDE_square_value_l3273_327300

theorem square_value (triangle circle square : ℕ) 
  (h1 : triangle + circle = square) 
  (h2 : triangle + circle + square = 100) : 
  square = 50 := by sorry

end NUMINAMATH_CALUDE_square_value_l3273_327300


namespace NUMINAMATH_CALUDE_cost_price_determination_l3273_327320

theorem cost_price_determination (selling_price_profit selling_price_loss : ℝ) 
  (h : selling_price_profit = 54 ∧ selling_price_loss = 40) :
  ∃ cost_price : ℝ, 
    selling_price_profit - cost_price = cost_price - selling_price_loss ∧ 
    cost_price = 47 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_determination_l3273_327320


namespace NUMINAMATH_CALUDE_g_of_two_equals_fourteen_l3273_327314

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_two_equals_fourteen :
  (∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) →
  g 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_fourteen_l3273_327314


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3273_327311

/-- The area of a square containing four touching circles -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3273_327311


namespace NUMINAMATH_CALUDE_line_points_k_value_l3273_327309

/-- 
Given two points (m, n) and (m + 2, n + k) on the line x = 2y + 5,
prove that k = 0.
-/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + 2 = 2*(n + k) + 5) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3273_327309


namespace NUMINAMATH_CALUDE_garden_ants_approximation_l3273_327343

/-- The number of ants in a rectangular garden --/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * (12 * 12) * ants_per_sq_inch

/-- Theorem stating that the number of ants in the garden is approximately 72 million --/
theorem garden_ants_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
  abs (number_of_ants 200 500 5 - 72000000) < ε :=
sorry

end NUMINAMATH_CALUDE_garden_ants_approximation_l3273_327343


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3273_327350

/-- The equation of the line passing through the points of tangency of the tangents drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  P = (2, 1) →
  (∀ x y, c (x, y) ↔ x^2 + y^2 = 4) →
  ∃ A B : ℝ × ℝ,
    (c A ∧ c B) ∧
    (∀ t : ℝ, c ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2) → t = 0 ∨ t = 1) ∧
    (∀ x y, 2 * x + y - 4 = 0 ↔ ∃ t : ℝ, x = (1 - t) * A.1 + t * B.1 ∧ y = (1 - t) * A.2 + t * B.2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3273_327350


namespace NUMINAMATH_CALUDE_cost_difference_analysis_l3273_327303

/-- Represents the cost difference between option 2 and option 1 -/
def cost_difference (x : ℝ) : ℝ := 54 * x + 9000 - (60 * x + 8800)

/-- Proves that the cost difference is 6x - 200 for x > 20, and positive when x = 30 -/
theorem cost_difference_analysis :
  (∀ x > 20, cost_difference x = 6 * x - 200) ∧
  (cost_difference 30 > 0) := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_analysis_l3273_327303


namespace NUMINAMATH_CALUDE_b_2017_value_l3273_327380

/-- Given sequences a and b with the specified properties, b₂₀₁₇ equals 2016/2017 -/
theorem b_2017_value (a b : ℕ → ℚ) : 
  (b 1 = 0) →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / (n * (n + 1))) →
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) + a (n - 1)) →
  b 2017 = 2016 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_b_2017_value_l3273_327380


namespace NUMINAMATH_CALUDE_other_number_proof_l3273_327398

/-- Given two positive integers with specific LCM, HCF, and one known value, prove the value of the other integer -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.lcm A B = 2310 → 
  Nat.gcd A B = 30 → 
  A = 210 → 
  B = 330 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l3273_327398


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l3273_327313

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l3273_327313


namespace NUMINAMATH_CALUDE_wendy_time_l3273_327321

-- Define the race participants
structure Racer where
  name : String
  time : Real

-- Define the race
def waterslideRace (bonnie wendy : Racer) : Prop :=
  wendy.time + 0.25 = bonnie.time ∧ bonnie.time = 7.80

-- Theorem to prove
theorem wendy_time (bonnie wendy : Racer) :
  waterslideRace bonnie wendy → wendy.time = 7.55 := by
  sorry

end NUMINAMATH_CALUDE_wendy_time_l3273_327321


namespace NUMINAMATH_CALUDE_three_power_plus_five_power_plus_fourteen_equals_factorial_l3273_327324

theorem three_power_plus_five_power_plus_fourteen_equals_factorial :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) := by
  sorry

end NUMINAMATH_CALUDE_three_power_plus_five_power_plus_fourteen_equals_factorial_l3273_327324


namespace NUMINAMATH_CALUDE_system_solution_l3273_327319

theorem system_solution (x y : ℝ) : 
  x^2 = 4*y^2 + 19 ∧ x*y + 2*y^2 = 18 → 
  (x = 55 / Real.sqrt 91 ∨ x = -55 / Real.sqrt 91) ∧
  (y = 18 / Real.sqrt 91 ∨ y = -18 / Real.sqrt 91) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3273_327319


namespace NUMINAMATH_CALUDE_complex_roots_l3273_327388

theorem complex_roots (a' b' c' d' k' : ℂ) 
  (h1 : a' * k' ^ 2 + b' * k' + c' = 0)
  (h2 : b' * k' ^ 2 + c' * k' + d' = 0)
  (h3 : d' = a')
  (h4 : k' ≠ 0) :
  k' = 1 ∨ k' = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k' = (-1 - Complex.I * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_l3273_327388


namespace NUMINAMATH_CALUDE_perimeter_C_is_24_l3273_327302

/-- Represents a polygon in the triangular grid -/
structure Polygon where
  perimeter : ℝ

/-- Represents the triangular grid with four polygons -/
structure TriangularGrid where
  A : Polygon
  B : Polygon
  C : Polygon
  D : Polygon

/-- The perimeter of triangle C in the given triangular grid -/
def perimeter_C (grid : TriangularGrid) : ℝ :=
  -- Definition to be proved
  24

/-- Theorem stating that the perimeter of triangle C is 24 cm -/
theorem perimeter_C_is_24 (grid : TriangularGrid) 
    (h1 : grid.A.perimeter = 56)
    (h2 : grid.B.perimeter = 34)
    (h3 : grid.D.perimeter = 42) :
  perimeter_C grid = 24 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_C_is_24_l3273_327302


namespace NUMINAMATH_CALUDE_mystery_number_proof_l3273_327394

theorem mystery_number_proof : ∃ x : ℕ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l3273_327394


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l3273_327354

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + correct_value - wrong_value) / n

/-- Theorem stating that the corrected mean is 36.14 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let wrong_value : ℚ := 23
  let correct_value : ℚ := 30
  corrected_mean n initial_mean wrong_value correct_value = 36.14 := by
sorry

#eval corrected_mean 50 36 23 30

end NUMINAMATH_CALUDE_corrected_mean_problem_l3273_327354


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l3273_327336

def students : ℕ := 4
def teachers : ℕ := 3

def arrangements_teachers_together : ℕ := 720
def arrangements_teachers_together_students_split : ℕ := 144
def arrangements_teachers_apart : ℕ := 1440

theorem photo_lineup_arrangements :
  (students = 4 ∧ teachers = 3) →
  (arrangements_teachers_together = 720 ∧
   arrangements_teachers_together_students_split = 144 ∧
   arrangements_teachers_apart = 1440) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l3273_327336


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l3273_327386

/-- The probability of arranging n items in a row, where k specific items are placed consecutively. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n

/-- The probability of arranging 12 items in a row, where 4 specific items are placed consecutively, is 1/55. -/
theorem art_arrangement_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_probability_l3273_327386


namespace NUMINAMATH_CALUDE_certain_number_is_five_l3273_327382

theorem certain_number_is_five (n d : ℕ) (h1 : d > 0) (h2 : n % d = 3) (h3 : (n^2) % d = 4) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_five_l3273_327382


namespace NUMINAMATH_CALUDE_op_twice_equals_identity_l3273_327376

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 - y

-- Statement to prove
theorem op_twice_equals_identity (h : ℝ) : op h (op h h) = h := by
  sorry

end NUMINAMATH_CALUDE_op_twice_equals_identity_l3273_327376


namespace NUMINAMATH_CALUDE_hyperbrick_probability_l3273_327363

open Set
open Real
open Finset

-- Define the set of numbers
def S : Finset ℕ := Finset.range 500

-- Define the type for our 9 randomly selected numbers
structure NineNumbers :=
  (numbers : Finset ℕ)
  (size_eq : numbers.card = 9)
  (subset_S : numbers ⊆ S)

-- Define the probability function
def probability (n : NineNumbers) : ℚ :=
  -- Implementation details omitted
  sorry

-- The main theorem
theorem hyperbrick_probability :
  ∀ n : NineNumbers, probability n = 16 / 63 :=
sorry

end NUMINAMATH_CALUDE_hyperbrick_probability_l3273_327363


namespace NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l3273_327371

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 :=
sorry

end NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l3273_327371


namespace NUMINAMATH_CALUDE_p_plus_q_equals_30_l3273_327318

theorem p_plus_q_equals_30 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 35) / (x - 3)) →
  P + Q = 30 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_30_l3273_327318


namespace NUMINAMATH_CALUDE_mistake_in_report_l3273_327383

def reported_numbers : List Nat := [3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]

def num_boys : Nat := 7
def num_girls : Nat := 8

theorem mistake_in_report :
  (List.sum reported_numbers) % 2 = 0 →
  ¬(∃ (boys_sum : Nat), 
    boys_sum * 2 = List.sum reported_numbers ∧
    boys_sum = num_girls * (List.sum reported_numbers / (num_boys + num_girls))) :=
by sorry

end NUMINAMATH_CALUDE_mistake_in_report_l3273_327383


namespace NUMINAMATH_CALUDE_division_remainder_l3273_327349

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  dividend % divisor = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3273_327349


namespace NUMINAMATH_CALUDE_product_equality_l3273_327326

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Proves that if a * (reversed b) = 143, then a * b = 143 -/
theorem product_equality (a b : ℕ) 
  (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 10 ≤ b ∧ b < 100) 
  (h : a * (reverse_digits b) = 143) : 
  a * b = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3273_327326


namespace NUMINAMATH_CALUDE_wifes_raise_is_760_l3273_327361

/-- Calculates the raise amount for Don's wife given the conditions of the problem -/
def wifes_raise (dons_raise : ℚ) (income_difference : ℚ) (raise_percentage : ℚ) : ℚ :=
  let dons_income := dons_raise / raise_percentage
  let wifes_income := dons_income - (income_difference / (1 + raise_percentage))
  wifes_income * raise_percentage

/-- Proves that Don's wife's raise is 760 given the problem conditions -/
theorem wifes_raise_is_760 : 
  wifes_raise 800 540 (8/100) = 760 := by
  sorry

#eval wifes_raise 800 540 (8/100)

end NUMINAMATH_CALUDE_wifes_raise_is_760_l3273_327361


namespace NUMINAMATH_CALUDE_police_text_percentage_l3273_327353

theorem police_text_percentage : 
  ∀ (total_texts grocery_texts response_texts police_texts : ℕ),
    total_texts = 33 →
    grocery_texts = 5 →
    response_texts = 5 * grocery_texts →
    police_texts = total_texts - (grocery_texts + response_texts) →
    (police_texts : ℚ) / (grocery_texts + response_texts : ℚ) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_police_text_percentage_l3273_327353


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l3273_327359

theorem event_ticket_revenue :
  ∀ (full_price : ℚ) (full_count half_count : ℕ),
    full_count + half_count = 180 →
    full_price * full_count + (full_price / 2) * half_count = 2652 →
    full_price * full_count = 984 :=
by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l3273_327359


namespace NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3273_327312

theorem unique_quadratic_trinomial :
  ∃! (a b c : ℝ), 
    (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
    a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3273_327312


namespace NUMINAMATH_CALUDE_exact_money_for_widgets_l3273_327397

/-- If a person can buy exactly 6 items at a certain price, and exactly 8 items if the price is reduced by 10%, then the person has exactly $5 to spend. -/
theorem exact_money_for_widgets (price : ℝ) (money : ℝ) 
  (h1 : money = 6 * price) 
  (h2 : money = 8 * (0.9 * price)) : 
  money = 5 := by sorry

end NUMINAMATH_CALUDE_exact_money_for_widgets_l3273_327397


namespace NUMINAMATH_CALUDE_max_consoles_assembled_l3273_327344

def chipA : ℕ := 467
def chipB : ℕ := 413
def chipC : ℕ := 532
def chipD : ℕ := 356
def chipE : ℕ := 494

def dailyProduction : List ℕ := [chipA, chipB, chipC, chipD, chipE]

theorem max_consoles_assembled (consoles : ℕ) :
  consoles = List.minimum dailyProduction → consoles = 356 := by
  sorry

end NUMINAMATH_CALUDE_max_consoles_assembled_l3273_327344


namespace NUMINAMATH_CALUDE_ants_in_field_approx_50_million_l3273_327358

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a field in square inches -/
def fieldAreaInSquareInches (d : FieldDimensions) : ℝ :=
  d.width * d.length * 144  -- 144 = 12^2, converting square feet to square inches

/-- Calculates the total number of ants in a field -/
def totalAnts (d : FieldDimensions) (antsPerSquareInch : ℝ) : ℝ :=
  fieldAreaInSquareInches d * antsPerSquareInch

/-- Theorem stating that the number of ants in the given field is approximately 50 million -/
theorem ants_in_field_approx_50_million :
  let d : FieldDimensions := { width := 300, length := 400 }
  let antsPerSquareInch : ℝ := 3
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
    abs (totalAnts d antsPerSquareInch - 50000000) < ε := by
  sorry

end NUMINAMATH_CALUDE_ants_in_field_approx_50_million_l3273_327358


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3273_327341

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3273_327341


namespace NUMINAMATH_CALUDE_winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l3273_327333

-- Define a temperature range
structure TemperatureRange where
  min : ℤ
  max : ℤ
  h : min ≤ max

-- Define a predicate for a scenario using negative numbers
def usesNegativeNumbers (range : TemperatureRange) : Prop :=
  range.min < 0

-- Theorem: The given temperature range uses negative numbers
theorem winter_temperature_uses_negative_numbers :
  ∃ (range : TemperatureRange), usesNegativeNumbers range :=
by
  -- The proof would go here
  sorry

-- Example of the temperature range mentioned in the solution
def winter_day_range : TemperatureRange :=
  { min := -2
  , max := 5
  , h := by norm_num }

-- Theorem: The specific winter day range uses negative numbers
theorem specific_winter_day_uses_negative_numbers :
  usesNegativeNumbers winter_day_range :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l3273_327333


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l3273_327356

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 23

/-- The number of magazines on each bookshelf -/
def magazines_per_shelf : ℕ := 61

/-- The total number of books and magazines -/
def total_items : ℕ := 2436

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 29

theorem bryan_bookshelves :
  (books_per_shelf + magazines_per_shelf) * num_bookshelves = total_items :=
sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l3273_327356


namespace NUMINAMATH_CALUDE_f_12_equals_190_l3273_327340

def f (n : ℤ) : ℤ := n^2 + 2*n + 22

theorem f_12_equals_190 : f 12 = 190 := by
  sorry

end NUMINAMATH_CALUDE_f_12_equals_190_l3273_327340


namespace NUMINAMATH_CALUDE_transformation_solvable_l3273_327337

/-- A transformation that replaces two numbers with their product -/
def transformation (numbers : List ℝ) (i j : Nat) : List ℝ :=
  if i < numbers.length ∧ j < numbers.length ∧ i ≠ j then
    let product := numbers[i]! * numbers[j]!
    numbers.set i product |>.set j product
  else
    numbers

/-- Predicate to check if all numbers in the list are the same -/
def allSame (numbers : List ℝ) : Prop :=
  ∀ i j, i < numbers.length → j < numbers.length → numbers[i]! = numbers[j]!

/-- The main theorem stating when the problem is solvable -/
theorem transformation_solvable (n : ℕ) :
  (∃ (numbers : List ℝ) (k : ℕ), numbers.length = n ∧ 
   ∃ (transformations : List (ℕ × ℕ)), 
     allSame (transformations.foldl (λ acc (i, j) => transformation acc i j) numbers)) ↔ 
  (n % 2 = 0 ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_transformation_solvable_l3273_327337


namespace NUMINAMATH_CALUDE_appropriate_presentation_length_l3273_327390

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration : Type := { d : ℝ // 20 ≤ d ∧ d ≤ 40 }

/-- The speaking rate in words per minute -/
def speakingRate : ℝ := 160

/-- Calculates the number of words for a given duration -/
def wordsForDuration (d : PresentationDuration) : ℝ := d.val * speakingRate

/-- Theorem stating that 5000 words is an appropriate length for the presentation -/
theorem appropriate_presentation_length :
  ∃ (d : PresentationDuration), 5000 = wordsForDuration d := by
  sorry

end NUMINAMATH_CALUDE_appropriate_presentation_length_l3273_327390


namespace NUMINAMATH_CALUDE_projection_theorem_l3273_327322

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the projection theorem
theorem projection_theorem (t : Triangle) : 
  t.a = t.b * Real.cos t.C + t.c * Real.cos t.B ∧
  t.b = t.c * Real.cos t.A + t.a * Real.cos t.C ∧
  t.c = t.a * Real.cos t.B + t.b * Real.cos t.A :=
by sorry

end NUMINAMATH_CALUDE_projection_theorem_l3273_327322


namespace NUMINAMATH_CALUDE_aerith_win_probability_l3273_327373

def coin_game (p_heads : ℚ) : ℚ :=
  (1 - p_heads) / (2 - p_heads)

theorem aerith_win_probability :
  let p_heads : ℚ := 4/7
  coin_game p_heads = 7/11 := by sorry

end NUMINAMATH_CALUDE_aerith_win_probability_l3273_327373


namespace NUMINAMATH_CALUDE_volume_of_sphere_with_radius_three_l3273_327327

/-- The volume of a sphere with radius 3 is 36π. -/
theorem volume_of_sphere_with_radius_three : 
  (4 / 3 : ℝ) * Real.pi * 3^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_with_radius_three_l3273_327327


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l3273_327347

theorem subtraction_of_fractions : (8 : ℚ) / 15 - (11 : ℚ) / 20 = -1 / 60 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l3273_327347


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l3273_327368

/-- The area of a circle circumscribed around an isosceles triangle -/
theorem circumscribed_circle_area (base lateral : ℝ) (h_base : base = 24) (h_lateral : lateral = 13) :
  let height : ℝ := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let triangle_area : ℝ := (base * height) / 2
  let radius : ℝ := (base * lateral ^ 2) / (4 * triangle_area)
  let circle_area : ℝ := π * radius ^ 2
  circle_area = 285.61 * π :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l3273_327368


namespace NUMINAMATH_CALUDE_instrument_players_l3273_327375

theorem instrument_players (total_people : ℕ) 
  (fraction_at_least_one : ℚ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : fraction_at_least_one = 2/5) 
  (h3 : prob_exactly_one = 28/100) : 
  ℕ := by
  sorry

#check instrument_players

end NUMINAMATH_CALUDE_instrument_players_l3273_327375


namespace NUMINAMATH_CALUDE_intersection_at_single_point_l3273_327385

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 10

/-- The line equation -/
def line (k : ℝ) : ℝ := k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

/-- Theorem stating the value of k for which the line intersects the parabola at exactly one point -/
theorem intersection_at_single_point :
  ∀ k : ℝ, single_intersection k ↔ k = 34/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_at_single_point_l3273_327385


namespace NUMINAMATH_CALUDE_one_not_in_set_l3273_327310

theorem one_not_in_set : 1 ∉ {x : ℝ | ∃ a : ℕ+, x = -a^2 + 1} := by sorry

end NUMINAMATH_CALUDE_one_not_in_set_l3273_327310


namespace NUMINAMATH_CALUDE_ladies_walking_ratio_l3273_327348

/-- Given two ladies walking in Central Park, prove that the ratio of their distances is 2:1 -/
theorem ladies_walking_ratio :
  ∀ (distance1 distance2 : ℝ),
  distance2 = 4 →
  distance1 + distance2 = 12 →
  distance1 / distance2 = 2 := by
sorry

end NUMINAMATH_CALUDE_ladies_walking_ratio_l3273_327348


namespace NUMINAMATH_CALUDE_hedge_sections_count_l3273_327346

def section_blocks : ℕ := 30
def block_cost : ℚ := 2
def total_cost : ℚ := 480

theorem hedge_sections_count :
  (total_cost / (section_blocks * block_cost) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hedge_sections_count_l3273_327346


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_five_sixths_l3273_327332

theorem greatest_integer_less_than_negative_twenty_five_sixths :
  Int.floor (-25 / 6 : ℚ) = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_five_sixths_l3273_327332


namespace NUMINAMATH_CALUDE_grandmother_five_times_lingling_age_l3273_327301

/-- Represents the current age of Lingling -/
def lingling_age : ℕ := 8

/-- Represents the current age of the grandmother -/
def grandmother_age : ℕ := 60

/-- Represents the number of years after which the grandmother's age will be 5 times Lingling's age -/
def years_until_five_times : ℕ := 5

/-- Proves that after 'years_until_five_times' years, the grandmother's age will be 5 times Lingling's age -/
theorem grandmother_five_times_lingling_age : 
  grandmother_age + years_until_five_times = 5 * (lingling_age + years_until_five_times) := by
  sorry

end NUMINAMATH_CALUDE_grandmother_five_times_lingling_age_l3273_327301


namespace NUMINAMATH_CALUDE_poster_collection_ratio_l3273_327396

theorem poster_collection_ratio : 
  let current_size : ℕ := 22
  let past_size : ℕ := 14
  let gcd := Nat.gcd current_size past_size
  (current_size / gcd) = 11 ∧ (past_size / gcd) = 7 :=
by sorry

end NUMINAMATH_CALUDE_poster_collection_ratio_l3273_327396


namespace NUMINAMATH_CALUDE_piggy_bank_value_l3273_327381

-- Define the number of pennies and dimes in one piggy bank
def pennies_per_bank : ℕ := 100
def dimes_per_bank : ℕ := 50

-- Define the value of pennies and dimes in cents
def penny_value : ℕ := 1
def dime_value : ℕ := 10

-- Define the number of piggy banks
def num_banks : ℕ := 2

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem piggy_bank_value :
  (num_banks * (pennies_per_bank * penny_value + dimes_per_bank * dime_value)) / cents_per_dollar = 12 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_value_l3273_327381


namespace NUMINAMATH_CALUDE_tangent_line_x_ln_x_at_1_l3273_327374

/-- The equation of the tangent line to y = x ln x at x = 1 is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x_at_1 : 
  let f : ℝ → ℝ := λ x => x * Real.log x
  let tangent_line : ℝ → ℝ := λ x => x - 1
  (∀ x, x > 0 → HasDerivAt f (Real.log x + 1) x) ∧ 
  HasDerivAt f 1 1 ∧
  f 1 = 0 →
  ∀ x y, y = tangent_line x ↔ x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_x_ln_x_at_1_l3273_327374


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3273_327342

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 7 * p - 3 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 7 * q - 3 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 7 * r - 3 = 0) → 
  p * q + q * r + r * p = 7/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3273_327342


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l3273_327317

/-- A triangle with integer side lengths and perimeter 24 has maximum side length 11 -/
theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ -- Three different side lengths
  a + b + c = 24 ∧ -- Perimeter is 24
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  c ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l3273_327317


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l3273_327304

theorem fraction_equality_implies_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 2 / (x - 5)) →
  C + D = 32/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l3273_327304


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l3273_327364

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (h1 : total = 108) (h2 : red_fraction = 5/6) :
  total * (1 - red_fraction) = 18 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l3273_327364


namespace NUMINAMATH_CALUDE_water_speed_in_swimming_problem_l3273_327362

/-- Proves that the speed of water is 4 km/h given the conditions of the swimming problem -/
theorem water_speed_in_swimming_problem : 
  ∀ (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ),
    still_water_speed = 8 →
    distance = 8 →
    time = 2 →
    distance = (still_water_speed - water_speed) * time →
    water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_in_swimming_problem_l3273_327362


namespace NUMINAMATH_CALUDE_squared_roots_equation_l3273_327360

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves that
    the equation x^2 - (p^2 - 2q)x + q^2 = 0 has roots that are the squares
    of the roots of the original equation. -/
theorem squared_roots_equation (p q : ℝ) :
  let original_eq (x : ℝ) := x^2 + p*x + q
  let new_eq (x : ℝ) := x^2 - (p^2 - 2*q)*x + q^2
  ∀ (r : ℝ), original_eq r = 0 → new_eq (r^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_squared_roots_equation_l3273_327360


namespace NUMINAMATH_CALUDE_walkway_area_is_416_l3273_327384

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the configuration of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed.length + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed.width + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed.length * g.bed.width
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h_rows : g.rows = 4)
  (h_columns : g.columns = 3)
  (h_bed_length : g.bed.length = 8)
  (h_bed_width : g.bed.width = 3)
  (h_walkway_width : g.walkway_width = 2) :
  walkway_area g = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l3273_327384


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l3273_327391

theorem prime_equation_solutions :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  (p^(2*q) + q^(2*p)) / (p^3 - p*q + q^3) = r →
  ((p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l3273_327391


namespace NUMINAMATH_CALUDE_janet_key_search_time_l3273_327330

/-- The number of minutes Janet spends looking for her keys every day -/
def key_search_time : ℝ := 8

/-- The number of minutes Janet spends complaining after finding her keys -/
def complain_time : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The number of minutes Janet would save per week if she stops losing her keys -/
def time_saved_per_week : ℝ := 77

theorem janet_key_search_time :
  key_search_time = (time_saved_per_week - days_per_week * complain_time) / days_per_week :=
by sorry

end NUMINAMATH_CALUDE_janet_key_search_time_l3273_327330


namespace NUMINAMATH_CALUDE_average_price_per_book_l3273_327335

theorem average_price_per_book (books_shop1 : ℕ) (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) 
  (h1 : books_shop1 = 42)
  (h2 : cost_shop1 = 520)
  (h3 : books_shop2 = 22)
  (h4 : cost_shop2 = 248) :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l3273_327335


namespace NUMINAMATH_CALUDE_pumpkin_spiderweb_ratio_l3273_327370

/-- Represents the Halloween decorations problem --/
def halloween_decorations (total : ℕ) (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget : ℕ) (left_to_put : ℕ) : Prop :=
  ∃ (pumpkins : ℕ),
    total = skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget + left_to_put ∧
    pumpkins = 2 * spiderwebs

/-- The ratio of pumpkins to spiderwebs is 2:1 given the specified conditions --/
theorem pumpkin_spiderweb_ratio :
  halloween_decorations 83 12 4 12 1 20 10 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_spiderweb_ratio_l3273_327370


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3273_327357

variable (a b : ℝ)
variable (x y : ℝ)

theorem simplify_expression_1 : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := by
  sorry

theorem simplify_expression_2 : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3273_327357


namespace NUMINAMATH_CALUDE_adults_who_ate_proof_l3273_327334

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := sorry

/-- The total number of adults in the group -/
def total_adults : ℕ := 55

/-- The total number of children in the group -/
def total_children : ℕ := 70

/-- The meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- The meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be fed with remaining food after some adults eat -/
def remaining_children_fed : ℕ := 72

theorem adults_who_ate_proof :
  adults_who_ate = 14 ∧
  adults_who_ate ≤ total_adults ∧
  (meal_capacity_adults - adults_who_ate) * meal_capacity_children / meal_capacity_adults = remaining_children_fed :=
sorry

end NUMINAMATH_CALUDE_adults_who_ate_proof_l3273_327334


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3273_327328

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = 7 ∧
  ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) →
  ((3/4) * x^2 + 2*x - y^2 ≤ M) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3273_327328


namespace NUMINAMATH_CALUDE_reflected_arcs_area_l3273_327372

/-- The area of the region bounded by 8 reflected arcs in a regular octagon inscribed in a circle -/
theorem reflected_arcs_area (s : ℝ) (h : s = 2) : 
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := π * r^2 / 8
  let triangle_area := s^2 / 4
  let reflected_arc_area := sector_area - triangle_area
  8 * reflected_arc_area = 2 * π * Real.sqrt 2 - 8 := by
  sorry

end NUMINAMATH_CALUDE_reflected_arcs_area_l3273_327372


namespace NUMINAMATH_CALUDE_winning_team_arrangements_winning_team_groupings_winning_team_selections_l3273_327393

/-- A debate team with male and female members -/
structure DebateTeam where
  male_members : ℕ
  female_members : ℕ

/-- The national winning debate team -/
def winning_team : DebateTeam :=
  { male_members := 3, female_members := 5 }

/-- Number of arrangements with male members not adjacent -/
def non_adjacent_arrangements (team : DebateTeam) : ℕ := sorry

/-- Number of ways to divide into pairs for classes -/
def pair_groupings (team : DebateTeam) (num_classes : ℕ) : ℕ := sorry

/-- Number of ways to select debaters with at least one male -/
def debater_selections (team : DebateTeam) (num_debaters : ℕ) : ℕ := sorry

theorem winning_team_arrangements :
  non_adjacent_arrangements winning_team = 14400 := by sorry

theorem winning_team_groupings :
  pair_groupings winning_team 4 = 2520 := by sorry

theorem winning_team_selections :
  debater_selections winning_team 4 = 1560 := by sorry

end NUMINAMATH_CALUDE_winning_team_arrangements_winning_team_groupings_winning_team_selections_l3273_327393


namespace NUMINAMATH_CALUDE_constant_product_on_circle_l3273_327387

theorem constant_product_on_circle (x₀ y₀ : ℝ) :
  x₀ ≠ 0 →
  y₀ ≠ 0 →
  x₀^2 + y₀^2 = 4 →
  |2 + 2*x₀/(y₀-2)| * |2 + 2*y₀/(x₀-2)| = 8 := by
sorry

end NUMINAMATH_CALUDE_constant_product_on_circle_l3273_327387


namespace NUMINAMATH_CALUDE_range_of_a_l3273_327351

theorem range_of_a (x a : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → (|x| ≠ 1)) ∧ 
  (∃ x, |x| ≠ 1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3273_327351


namespace NUMINAMATH_CALUDE_trishas_walk_distance_l3273_327325

/-- The total distance Trisha walked during her vacation in New York City -/
theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_distance_l3273_327325


namespace NUMINAMATH_CALUDE_box_surface_area_l3273_327389

/-- The surface area of a box formed by removing triangles from corners of a rectangle --/
theorem box_surface_area (length width triangle_side : ℕ) : 
  length = 25 →
  width = 40 →
  triangle_side = 4 →
  (length * width) - (4 * (triangle_side * triangle_side / 2)) = 968 :=
by
  sorry


end NUMINAMATH_CALUDE_box_surface_area_l3273_327389


namespace NUMINAMATH_CALUDE_white_balls_count_l3273_327367

theorem white_balls_count (yellow_balls : ℕ) (yellow_prob : ℚ) : 
  yellow_balls = 15 → yellow_prob = 3/4 → 
  ∃ (white_balls : ℕ), (yellow_balls : ℚ) / ((white_balls : ℚ) + yellow_balls) = yellow_prob ∧ white_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3273_327367


namespace NUMINAMATH_CALUDE_square_sum_reciprocals_l3273_327366

theorem square_sum_reciprocals (x y : ℝ) 
  (h : 1 / x - 1 / (2 * y) = 1 / (2 * x + y)) : 
  y^2 / x^2 + x^2 / y^2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocals_l3273_327366


namespace NUMINAMATH_CALUDE_inequality_proof_l3273_327306

theorem inequality_proof (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : x + y = 2 * a) :
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 ∧
  (x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 ↔ x = a ∧ y = a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3273_327306


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3273_327329

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (4 * a 1 - a 3 = a 3 - 2 * a 2) →  -- arithmetic sequence condition
  q = -1 ∨ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3273_327329


namespace NUMINAMATH_CALUDE_stability_comparison_lower_variance_more_stable_shooting_competition_result_l3273_327378

/-- Represents a shooter in the competition -/
structure Shooter where
  name : String
  variance : ℝ

/-- Defines the stability of a shooter based on their variance -/
def moreStable (a b : Shooter) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Shooter) 
  (h : a.variance ≠ b.variance) : 
  moreStable a b ∨ moreStable b a :=
sorry

theorem lower_variance_more_stable (a b : Shooter) 
  (h : a.variance < b.variance) : 
  moreStable a b :=
sorry

theorem shooting_competition_result (a b : Shooter)
  (ha : a.name = "A" ∧ a.variance = 0.25)
  (hb : b.name = "B" ∧ b.variance = 0.12) :
  moreStable b a :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_lower_variance_more_stable_shooting_competition_result_l3273_327378


namespace NUMINAMATH_CALUDE_seven_number_sequence_average_l3273_327377

theorem seven_number_sequence_average (a b c d e f g : ℝ) :
  (a + b + c + d) / 4 = 4 →
  (d + e + f + g) / 4 = 4 →
  d = 11 →
  (a + b + c + d + e + f + g) / 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_seven_number_sequence_average_l3273_327377


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3273_327399

theorem smallest_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
   (∀ a' b' c' d' e' : ℕ+, a' + b' + c' + d' + e' = 2020 →
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) ≥ M) ∧
   M = 674) :=
sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3273_327399


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l3273_327307

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = 1) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l3273_327307


namespace NUMINAMATH_CALUDE_lending_interest_rate_l3273_327315

/-- The interest rate at which a person lends money, given specific borrowing and lending conditions -/
theorem lending_interest_rate (borrowed_amount : ℝ) (borrowing_rate : ℝ) (lending_years : ℝ) (yearly_gain : ℝ) : 
  borrowed_amount = 5000 →
  borrowing_rate = 4 →
  lending_years = 2 →
  yearly_gain = 200 →
  (borrowed_amount * borrowing_rate * lending_years / 100 + 2 * yearly_gain) / (borrowed_amount * lending_years / 100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_l3273_327315


namespace NUMINAMATH_CALUDE_jane_finishing_days_l3273_327323

/-- The number of days Jane needs to finish arranging the remaining vases -/
def days_needed (jane_rate mark_rate mark_days total_vases : ℕ) : ℕ :=
  let combined_rate := jane_rate + mark_rate
  let vases_arranged := combined_rate * mark_days
  let remaining_vases := total_vases - vases_arranged
  (remaining_vases + jane_rate - 1) / jane_rate

theorem jane_finishing_days :
  days_needed 16 20 3 248 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_finishing_days_l3273_327323


namespace NUMINAMATH_CALUDE_minimum_bags_in_warehouse_A_minimum_bags_proof_l3273_327308

theorem minimum_bags_in_warehouse_A : ℕ → ℕ → Prop :=
  fun x y =>
    (∃ k : ℕ, 
      (y + 90 = 2 * (x - 90)) ∧
      (x + k = 6 * (y - k)) ∧
      (x ≥ 139) ∧
      (∀ z : ℕ, z < x → 
        ¬(∃ w k : ℕ, 
          (w + 90 = 2 * (z - 90)) ∧
          (z + k = 6 * (w - k))))) →
    x = 139

-- The proof goes here
theorem minimum_bags_proof : 
  ∃ x y : ℕ, minimum_bags_in_warehouse_A x y :=
sorry

end NUMINAMATH_CALUDE_minimum_bags_in_warehouse_A_minimum_bags_proof_l3273_327308


namespace NUMINAMATH_CALUDE_candy_distribution_l3273_327338

theorem candy_distribution (total_candy : ℕ) (candy_per_friend : ℕ) (h1 : total_candy = 45) (h2 : candy_per_friend = 5) :
  total_candy / candy_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3273_327338


namespace NUMINAMATH_CALUDE_grid_game_winner_l3273_327395

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the game state on a rectangular grid -/
structure GridGame where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- Determines the winner of the game based on the grid dimensions -/
def winner (game : GridGame) : Player :=
  if (game.m + game.n) % 2 = 0 then Player.Second else Player.First

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner (game : GridGame) :
  (winner game = Player.Second ↔ (game.m + game.n) % 2 = 0) ∧
  (winner game = Player.First ↔ (game.m + game.n) % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_grid_game_winner_l3273_327395


namespace NUMINAMATH_CALUDE_difference_of_squares_l3273_327355

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 72) (h2 : a - b = 16) : a^2 - b^2 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3273_327355


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3273_327379

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 8*y - x*y = 0 ∧ x + y = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3273_327379


namespace NUMINAMATH_CALUDE_proposition_truth_l3273_327305

theorem proposition_truth : (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.sin x - x < 0) ∧
  ¬(∃ x₀ ∈ Set.Ioi 0, (2 : ℝ) ^ x₀ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l3273_327305


namespace NUMINAMATH_CALUDE_triangle_proof_l3273_327392

open Real

theorem triangle_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given condition
  a * cos C - c / 2 = b →
  -- Part I
  A = 2 * π / 3 ∧
  -- Part II
  a = 3 →
  -- Perimeter range
  let l := a + b + c
  6 < l ∧ l ≤ 3 + 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l3273_327392


namespace NUMINAMATH_CALUDE_second_number_calculation_l3273_327339

theorem second_number_calculation (A B : ℝ) (h1 : A = 456) (h2 : 0.5 * A = 0.4 * B + 180) : B = 120 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l3273_327339


namespace NUMINAMATH_CALUDE_smallest_tripling_period_l3273_327331

-- Define the annual interest rate
def r : ℝ := 0.3334

-- Define the function that calculates the investment value after n years
def investment_value (n : ℕ) : ℝ := (1 + r) ^ n

-- Theorem statement
theorem smallest_tripling_period :
  ∀ n : ℕ, (investment_value n > 3 ∧ ∀ m : ℕ, m < n → investment_value m ≤ 3) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_tripling_period_l3273_327331


namespace NUMINAMATH_CALUDE_not_all_face_sums_different_not_all_face_sums_different_b_l3273_327369

/-- Represents the possible values that can be assigned to a vertex of the cube -/
inductive VertexValue
  | Zero
  | One

/-- Represents a cube with values assigned to its vertices -/
structure Cube :=
  (vertices : Fin 8 → VertexValue)

/-- Calculates the sum of values on a face of the cube -/
def faceSum (c : Cube) (face : Fin 6) : Nat :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different -/
theorem not_all_face_sums_different (c : Cube) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSum c i ≠ faceSum c j) :=
sorry

/-- Represents the possible values that can be assigned to a vertex of the cube (for part b) -/
inductive VertexValueB
  | NegOne
  | PosOne

/-- Represents a cube with values assigned to its vertices (for part b) -/
structure CubeB :=
  (vertices : Fin 8 → VertexValueB)

/-- Calculates the sum of values on a face of the cube (for part b) -/
def faceSumB (c : CubeB) (face : Fin 6) : Int :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different (for part b) -/
theorem not_all_face_sums_different_b (c : CubeB) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSumB c i ≠ faceSumB c j) :=
sorry

end NUMINAMATH_CALUDE_not_all_face_sums_different_not_all_face_sums_different_b_l3273_327369


namespace NUMINAMATH_CALUDE_tourist_attraction_temperature_difference_l3273_327316

/-- The temperature difference between the highest and lowest temperatures -/
def temperature_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Proof that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem tourist_attraction_temperature_difference :
  temperature_difference 8 (-2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tourist_attraction_temperature_difference_l3273_327316


namespace NUMINAMATH_CALUDE_pencil_average_cost_l3273_327365

/-- The average cost per pencil, including shipping -/
def average_cost (num_pencils : ℕ) (pencil_cost shipping_cost : ℚ) : ℚ :=
  (pencil_cost + shipping_cost) / num_pencils

/-- Theorem stating the average cost per pencil for the given problem -/
theorem pencil_average_cost :
  average_cost 150 (24.75 : ℚ) (8.50 : ℚ) = (33.25 : ℚ) / 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_average_cost_l3273_327365
