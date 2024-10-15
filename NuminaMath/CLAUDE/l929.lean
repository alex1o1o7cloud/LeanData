import Mathlib

namespace NUMINAMATH_CALUDE_min_value_problem_l929_92951

/-- Given positive real numbers a, b, c, and a function f with minimum value 4, 
    prove the sum of a, b, c is 4 and find the minimum value of a quadratic expression. -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l929_92951


namespace NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l929_92937

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l929_92937


namespace NUMINAMATH_CALUDE_thomas_blocks_total_l929_92977

theorem thomas_blocks_total (stack1 stack2 stack3 stack4 stack5 : ℕ) : 
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack5 = 2 * stack2 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_thomas_blocks_total_l929_92977


namespace NUMINAMATH_CALUDE_special_parallelogram_sides_l929_92941

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in radians
  acute_angle : ℝ
  -- The ratio of the parts of the obtuse angle divided by the diagonal
  obtuse_angle_ratio : ℝ
  -- The length of the shorter side
  short_side : ℝ
  -- The length of the longer side
  long_side : ℝ
  -- The perimeter is 90 cm
  perimeter_eq : perimeter = 90
  -- The acute angle is 60 degrees (π/3 radians)
  acute_angle_eq : acute_angle = π / 3
  -- The obtuse angle is divided in a 1:3 ratio
  obtuse_angle_ratio_eq : obtuse_angle_ratio = 1 / 3
  -- The perimeter is the sum of all sides
  perimeter_sum : perimeter = 2 * (short_side + long_side)
  -- The shorter side is half the longer side (derived from the 60° angle)
  side_ratio : short_side = long_side / 2

/-- Theorem: The sides of the special parallelogram are 15 cm and 30 cm -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.short_side = 15 ∧ p.long_side = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_sides_l929_92941


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l929_92936

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l929_92936


namespace NUMINAMATH_CALUDE_absolute_value_problem_l929_92925

theorem absolute_value_problem (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a + b| = 4) :
  ∃ (x : ℝ), |a - b| = x :=
sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l929_92925


namespace NUMINAMATH_CALUDE_simplify_expression_l929_92985

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l929_92985


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_22_l929_92952

/-- The first polynomial -/
def p1 (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

/-- The second polynomial -/
def p2 (x : ℝ) : ℝ := 3*x^3 - 4*x^2 + x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem stating that the coefficient of x^3 in the product is 22 -/
theorem coefficient_x_cubed_is_22 : 
  ∃ (a b c d e : ℝ), product = fun x ↦ 22 * x^3 + a * x^4 + b * x^2 + c * x + d * x^5 + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_22_l929_92952


namespace NUMINAMATH_CALUDE_sale_price_calculation_l929_92997

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price including tax is approximately 677.60 -/
theorem sale_price_calculation :
  let costPrice : ℝ := 545.13
  let profitRate : ℝ := 0.13
  let taxRate : ℝ := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.60) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l929_92997


namespace NUMINAMATH_CALUDE_xiaoming_multiplication_l929_92901

theorem xiaoming_multiplication (a : ℝ) : 
  20.18 * a = 20.18 * (a - 1) + 2270.25 → a = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_multiplication_l929_92901


namespace NUMINAMATH_CALUDE_integral_2x_minus_1_l929_92963

theorem integral_2x_minus_1 : ∫ x in (1:ℝ)..(2:ℝ), 2*x - 1 = 2 := by sorry

end NUMINAMATH_CALUDE_integral_2x_minus_1_l929_92963


namespace NUMINAMATH_CALUDE_eggs_for_bread_l929_92998

/-- The number of dozens of eggs needed given the total weight required, weight per egg, and eggs per dozen -/
def eggs_needed (total_weight : ℚ) (weight_per_egg : ℚ) (eggs_per_dozen : ℕ) : ℚ :=
  (total_weight / weight_per_egg) / eggs_per_dozen

/-- Theorem stating that 8 dozens of eggs are needed for the given conditions -/
theorem eggs_for_bread : eggs_needed 6 (1/16) 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_for_bread_l929_92998


namespace NUMINAMATH_CALUDE_coprime_set_properties_l929_92949

-- Define the set M
def M (a b : ℕ) : Set ℤ :=
  {z : ℤ | ∃ (x y : ℕ), z = a * x + b * y}

-- State the theorem
theorem coprime_set_properties (a b : ℕ) (h : Nat.Coprime a b) :
  -- Part 1: The largest integer not in M is ab - a - b
  (∀ z : ℤ, z ∉ M a b → z ≤ a * b - a - b) ∧
  (a * b - a - b : ℤ) ∉ M a b ∧
  -- Part 2: For any integer n, exactly one of n and (ab - a - b - n) is in M
  (∀ n : ℤ, (n ∈ M a b ↔ (a * b - a - b - n) ∉ M a b)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_set_properties_l929_92949


namespace NUMINAMATH_CALUDE_multiple_of_960_l929_92969

theorem multiple_of_960 (a : ℤ) 
  (h1 : ∃ k : ℤ, a = 10 * k + 4) 
  (h2 : ¬ (∃ m : ℤ, a = 4 * m)) : 
  ∃ n : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * n :=
sorry

end NUMINAMATH_CALUDE_multiple_of_960_l929_92969


namespace NUMINAMATH_CALUDE_f_of_3_equals_5_l929_92919

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_of_3_equals_5 : f 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_5_l929_92919


namespace NUMINAMATH_CALUDE_power_problem_l929_92943

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : 
  a ^ (2 * m + 3 * n) = 72 := by
sorry

end NUMINAMATH_CALUDE_power_problem_l929_92943


namespace NUMINAMATH_CALUDE_power_digits_theorem_l929_92922

/-- The number of digits to the right of the decimal place in a given number -/
def decimalDigits (x : ℝ) : ℕ :=
  sorry

/-- The result of raising a number to a power -/
def powerResult (base : ℝ) (exponent : ℕ) : ℝ :=
  base ^ exponent

theorem power_digits_theorem :
  let base := 10^4 * 3.456789
  decimalDigits (powerResult base 11) = 22 := by
  sorry

end NUMINAMATH_CALUDE_power_digits_theorem_l929_92922


namespace NUMINAMATH_CALUDE_village_panic_percentage_l929_92909

theorem village_panic_percentage (original : ℕ) (final : ℕ) : original = 7800 → final = 5265 → 
  (((original - original / 10) - final) / (original - original / 10) : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_village_panic_percentage_l929_92909


namespace NUMINAMATH_CALUDE_typists_productivity_l929_92907

/-- Given that 25 typists can type 60 letters in 20 minutes, prove that 75 typists 
    working at the same rate can complete 540 letters in 1 hour. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ) 
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 25 →
  letters_base = 60 →
  minutes_base = 20 →
  typists_new = 75 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 540 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l929_92907


namespace NUMINAMATH_CALUDE_sue_answer_formula_l929_92991

/-- Given Ben's initial number, calculate Sue's final answer -/
def sueAnswer (x : ℕ) : ℕ :=
  let benResult := 2 * (2 * x + 1)
  2 * (benResult - 1)

/-- Theorem: Sue's answer is always 4x + 2, where x is Ben's initial number -/
theorem sue_answer_formula (x : ℕ) : sueAnswer x = 4 * x + 2 := by
  sorry

#eval sueAnswer 8  -- Should output 66

end NUMINAMATH_CALUDE_sue_answer_formula_l929_92991


namespace NUMINAMATH_CALUDE_boat_current_speed_l929_92924

theorem boat_current_speed 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = 
    (boat_speed + current_speed) * downstream_time ∧ 
    current_speed = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_speed_l929_92924


namespace NUMINAMATH_CALUDE_x_greater_than_one_l929_92964

theorem x_greater_than_one (x : ℝ) (h : Real.log x > 0) : x > 1 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_one_l929_92964


namespace NUMINAMATH_CALUDE_line_relations_l929_92989

def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

theorem line_relations (m : ℝ) :
  (∀ x y, l1 m x y → l2 m x y → (m - 2 + 3 * m = 0) ↔ m = 1/2) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ m ≠ 3 ∧ m ≠ -3) ↔ m = -1) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ 3 / m = 2 * m / 6) ↔ m = 3) ∧
  (∀ x y, l1 m x y → l2 m x y → (m ≠ 3 ∧ m ≠ -1) ↔ (m ≠ 3 ∧ m ≠ -1)) :=
by sorry

end NUMINAMATH_CALUDE_line_relations_l929_92989


namespace NUMINAMATH_CALUDE_students_taller_than_yoongi_l929_92911

theorem students_taller_than_yoongi 
  (total_students : ℕ) 
  (shorter_than_yoongi : ℕ) 
  (h1 : total_students = 20) 
  (h2 : shorter_than_yoongi = 11) : 
  total_students - shorter_than_yoongi - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_taller_than_yoongi_l929_92911


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l929_92981

open Real

theorem monotone_increasing_condition (b : ℝ) :
  (∃ (a c : ℝ), 1/2 ≤ a ∧ c ≤ 2 ∧ a < c ∧
    ∀ x y, a ≤ x ∧ x < y ∧ y ≤ c →
      exp x * (x^2 - b*x) < exp y * (y^2 - b*y)) →
  b < 8/3 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l929_92981


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l929_92980

/-- The line equation ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l929_92980


namespace NUMINAMATH_CALUDE_paint_cans_used_l929_92938

def initial_capacity : ℕ := 36
def reduced_capacity : ℕ := 28
def lost_cans : ℕ := 4

theorem paint_cans_used : ℕ := by
  -- Prove that the number of cans used to paint 28 rooms is 14
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l929_92938


namespace NUMINAMATH_CALUDE_kate_red_bouncy_balls_l929_92957

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The difference in the number of red and yellow bouncy balls -/
def red_yellow_diff : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

theorem kate_red_bouncy_balls :
  red_packs * balls_per_pack = yellow_packs * balls_per_pack + red_yellow_diff :=
by sorry

end NUMINAMATH_CALUDE_kate_red_bouncy_balls_l929_92957


namespace NUMINAMATH_CALUDE_quadratic_equation_shift_l929_92953

theorem quadratic_equation_shift (a h k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, a * (x - h)^2 + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = 0 ∧ y₂ = 4 ∧ 
   ∀ y : ℝ, a * (y - h - 1)^2 + k = 0 ↔ y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_shift_l929_92953


namespace NUMINAMATH_CALUDE_irreducible_fractions_divisibility_l929_92916

theorem irreducible_fractions_divisibility (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  ∃ k : ℕ, Nat.totient (a^n - 1) = n * k := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_divisibility_l929_92916


namespace NUMINAMATH_CALUDE_short_trees_after_planting_park_short_trees_l929_92910

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of initial and new short trees -/
theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (new_short_trees : ℕ) : 
  total_short_trees initial_short_trees new_short_trees = initial_short_trees + new_short_trees := by
  sorry

/-- Application to the specific problem -/
theorem park_short_trees : total_short_trees 3 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_park_short_trees_l929_92910


namespace NUMINAMATH_CALUDE_base_conversion_2014_to_base_9_l929_92958

theorem base_conversion_2014_to_base_9 :
  2014 = 2 * (9^3) + 6 * (9^2) + 7 * (9^1) + 7 * (9^0) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2014_to_base_9_l929_92958


namespace NUMINAMATH_CALUDE_steves_final_height_l929_92915

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the final height in inches after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_inches_to_inches initial_feet initial_inches + growth

/-- Theorem: Steve's final height is 72 inches -/
theorem steves_final_height :
  final_height 5 6 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_steves_final_height_l929_92915


namespace NUMINAMATH_CALUDE_blankets_per_person_l929_92947

/-- Proves that the number of blankets each person gave on the first day is 2 --/
theorem blankets_per_person (team_size : Nat) (last_day_blankets : Nat) (total_blankets : Nat) :
  team_size = 15 →
  last_day_blankets = 22 →
  total_blankets = 142 →
  ∃ (first_day_blankets : Nat),
    first_day_blankets * team_size + 3 * (first_day_blankets * team_size) + last_day_blankets = total_blankets ∧
    first_day_blankets = 2 := by
  sorry

#check blankets_per_person

end NUMINAMATH_CALUDE_blankets_per_person_l929_92947


namespace NUMINAMATH_CALUDE_floor_sum_example_l929_92920

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l929_92920


namespace NUMINAMATH_CALUDE_prime_divides_product_implies_divides_factor_l929_92983

theorem prime_divides_product_implies_divides_factor 
  (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
  (h_prime : Nat.Prime p) 
  (h_divides_product : p ∣ (Finset.univ.prod a)) : 
  ∃ i, p ∣ a i :=
sorry

end NUMINAMATH_CALUDE_prime_divides_product_implies_divides_factor_l929_92983


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_neg_2x_minus_3_l929_92923

/-- Given a line with equation y = mx + b, this function returns the equation
    of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (m : ℝ) (b : ℝ) : ℝ → ℝ := fun x ↦ -m * x + b

theorem symmetric_line_y_axis_neg_2x_minus_3 :
  symmetricLineYAxis (-2) (-3) = fun x ↦ 2 * x - 3 := by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_neg_2x_minus_3_l929_92923


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l929_92914

/-- Represents the amount of money Chris had before his birthday. -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother. -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle. -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents. -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after receiving all gifts. -/
def total_after_gifts : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday = total_after_gifts - (grandmother_gift + aunt_uncle_gift + parents_gift) :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l929_92914


namespace NUMINAMATH_CALUDE_inequality_group_solution_set_l929_92929

theorem inequality_group_solution_set :
  let S := {x : ℝ | 2 * x + 3 ≥ -1 ∧ 7 - 3 * x > 1}
  S = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_inequality_group_solution_set_l929_92929


namespace NUMINAMATH_CALUDE_cone_radius_is_one_l929_92970

/-- Given a cone whose surface area is 3π and whose lateral surface unfolds into a semicircle,
    prove that the radius of its base is 1. -/
theorem cone_radius_is_one (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  π * l = 2 * π * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 3 * π →  -- surface area is 3π
  r = 1 := by
  sorry

end NUMINAMATH_CALUDE_cone_radius_is_one_l929_92970


namespace NUMINAMATH_CALUDE_rectangle_area_l929_92921

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 160) : ∃ (length width : ℝ),
  length = 4 * width ∧
  2 * (length + width) = perimeter ∧
  length * width = 1024 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l929_92921


namespace NUMINAMATH_CALUDE_blood_expiration_theorem_l929_92960

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the blood expiration time in seconds
def blood_expiration_time : ℕ := factorial 7

-- Define the donation time
def donation_time : ℕ := 18 * 60 * 60  -- 6 PM in seconds

-- Define the expiration datetime
def expiration_datetime : ℕ := donation_time + blood_expiration_time

-- Theorem to prove
theorem blood_expiration_theorem :
  expiration_datetime = 19 * 60 * 60 + 24 * 60 :=  -- 7:24 PM in seconds
by sorry

end NUMINAMATH_CALUDE_blood_expiration_theorem_l929_92960


namespace NUMINAMATH_CALUDE_theater_seats_l929_92965

theorem theater_seats (first_row : ℕ) (last_row : ℕ) (total_seats : ℕ) (num_rows : ℕ) 
  (h1 : first_row = 14)
  (h2 : last_row = 50)
  (h3 : total_seats = 416)
  (h4 : num_rows = 13) :
  ∃ (additional_seats : ℕ), 
    (additional_seats = 3) ∧ 
    (last_row = first_row + (num_rows - 1) * additional_seats) ∧
    (total_seats = (num_rows * (first_row + last_row)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_theater_seats_l929_92965


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l929_92959

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  c = 2 →
  Real.sin C * (Real.cos B - Real.sqrt 3 * Real.sin B) = Real.sin A →
  Real.cos A = 2 * Real.sqrt 2 / 3 →
  -- Conclusions
  C = 5 * π / 6 ∧
  b = (4 * Real.sqrt 2 - 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l929_92959


namespace NUMINAMATH_CALUDE_percentage_of_day_l929_92933

theorem percentage_of_day (hours_in_day : ℝ) (percentage : ℝ) (result : ℝ) : 
  hours_in_day = 24 →
  percentage = 29.166666666666668 →
  result = 7 →
  (percentage / 100) * hours_in_day = result :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_day_l929_92933


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l929_92931

/-- The trajectory of a point P, where the sum of its distances to two fixed points A(-1, 0) and B(1, 0) is constant 2, is the line segment AB. -/
theorem trajectory_is_line_segment (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let dist (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (dist P A + dist P B = 2) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (2*t - 1, 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l929_92931


namespace NUMINAMATH_CALUDE_range_of_a_l929_92956

-- Define the sets A and B
def A : Set ℝ := Set.Ioo 1 4
def B (a : ℝ) : Set ℝ := Set.Ioo (2 * a) (a + 1)

-- State the theorem
theorem range_of_a (a : ℝ) (h : a < 1) :
  B a ⊆ A → 1/2 ≤ a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l929_92956


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l929_92906

/-- An ellipse with foci at (-3, 0) and (3, 0), passing through (0, 3) -/
structure Ellipse where
  /-- The equation of the ellipse in the form (x²/a² + y²/b² = 1) -/
  equation : ℝ → ℝ → Prop
  /-- The foci are at (-3, 0) and (3, 0) -/
  foci : equation (-3) 0 ∧ equation 3 0
  /-- The point (0, 3) is on the ellipse -/
  point : equation 0 3

/-- The standard form of the ellipse equation -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 9 = 1

/-- Theorem: The standard equation of the ellipse is x²/18 + y²/9 = 1 -/
theorem ellipse_standard_equation (e : Ellipse) : e.equation = standard_equation := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l929_92906


namespace NUMINAMATH_CALUDE_square_area_is_25_l929_92903

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareWithYCoords where
  -- The y-coordinates of the vertices
  y1 : ℝ
  y2 : ℝ
  y3 : ℝ
  y4 : ℝ
  -- Ensure the y-coordinates are distinct and in ascending order
  h1 : y1 < y2
  h2 : y2 < y3
  h3 : y3 < y4
  -- Ensure the square property (opposite sides are parallel and equal)
  h4 : y4 - y3 = y2 - y1

/-- The area of a square with specific y-coordinates is 25 -/
theorem square_area_is_25 (s : SquareWithYCoords) (h5 : s.y1 = 2) (h6 : s.y2 = 3) (h7 : s.y3 = 7) (h8 : s.y4 = 8) : 
  (s.y3 - s.y2) * (s.y3 - s.y2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_25_l929_92903


namespace NUMINAMATH_CALUDE_max_profit_at_800_l929_92902

/-- Price function for desk orders -/
def P (x : ℕ) : ℚ :=
  if x ≤ 100 then 80
  else 82 - 0.02 * x

/-- Profit function for desk orders -/
def f (x : ℕ) : ℚ :=
  if x ≤ 100 then 30 * x
  else (32 * x - 0.02 * x^2)

/-- Theorem stating the maximum profit and corresponding order quantity -/
theorem max_profit_at_800 :
  (∀ x : ℕ, 0 < x ∧ x ≤ 1000 → f x ≤ f 800) ∧
  f 800 = 12800 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_800_l929_92902


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l929_92930

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 600
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  (total_alcohol / total_volume) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l929_92930


namespace NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l929_92975

/-- Represents a regular polygon with 6n+1 sides and a coloring of its vertices -/
structure ColoredPolygon (n : ℕ) where
  k : ℕ
  h : ℕ := 6 * n + 1 - k
  k_valid : k ≤ 6 * n + 1

/-- Counts the number of monochromatic isosceles triangles in a colored polygon -/
def monochromaticIsoscelesCount (p : ColoredPolygon n) : ℚ :=
  (1 / 2) * (p.h * (p.h - 1) + p.k * (p.k - 1) - p.k * p.h)

/-- Theorem stating that the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) :
  ∀ p q : ColoredPolygon n, monochromaticIsoscelesCount p = monochromaticIsoscelesCount q :=
sorry

end NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l929_92975


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l929_92978

theorem quadratic_polynomial_problem : ∃ (q : ℝ → ℝ),
  (∀ x, q x = -4.5 * x^2 - 13.5 * x + 81) ∧
  q (-6) = 0 ∧
  q 3 = 0 ∧
  q 4 = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l929_92978


namespace NUMINAMATH_CALUDE_two_non_congruent_triangles_l929_92926

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 7 -/
def triangles_with_perimeter_7 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 7}

/-- The theorem to be proved -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_7 ∧
    t2 ∈ triangles_with_perimeter_7 ∧
    ¬ is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_7 →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end NUMINAMATH_CALUDE_two_non_congruent_triangles_l929_92926


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l929_92986

theorem farmer_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l929_92986


namespace NUMINAMATH_CALUDE_max_profit_is_180_l929_92913

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ
  pricePerBox : ℚ
  transportCost : ℚ

/-- Calculates the profit for a given neighborhood --/
def profit (n : Neighborhood) : ℚ :=
  n.homes * n.boxesPerHome * n.pricePerBox - n.transportCost

/-- Checks if the neighborhood is within the stock limit --/
def withinStockLimit (n : Neighborhood) (stockLimit : ℕ) : Prop :=
  n.homes * n.boxesPerHome ≤ stockLimit

/-- The main theorem stating the maximum profit --/
theorem max_profit_is_180 (stockLimit : ℕ) (A B C D : Neighborhood)
  (hStock : stockLimit = 50)
  (hA : A = { homes := 12, boxesPerHome := 3, pricePerBox := 3, transportCost := 10 })
  (hB : B = { homes := 8, boxesPerHome := 6, pricePerBox := 4, transportCost := 15 })
  (hC : C = { homes := 15, boxesPerHome := 2, pricePerBox := 5/2, transportCost := 5 })
  (hD : D = { homes := 5, boxesPerHome := 8, pricePerBox := 5, transportCost := 20 })
  (hAStock : withinStockLimit A stockLimit)
  (hBStock : withinStockLimit B stockLimit)
  (hCStock : withinStockLimit C stockLimit)
  (hDStock : withinStockLimit D stockLimit) :
  (max (profit A) (max (profit B) (max (profit C) (profit D)))) = 180 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_180_l929_92913


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l929_92972

theorem quadratic_function_theorem (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x) →
  (f 0 = 0) →
  (∀ x, f (x + 1) = f x + x + 1) →
  (a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l929_92972


namespace NUMINAMATH_CALUDE_chocolate_division_l929_92912

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  piles_for_shaina * (total_chocolate / num_piles) = 24 / 7 := by
sorry

end NUMINAMATH_CALUDE_chocolate_division_l929_92912


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l929_92967

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l929_92967


namespace NUMINAMATH_CALUDE_polyhedron_sum_l929_92917

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of pentagonal faces
  euler : V - E + F = 2
  faces : F = 32
  face_types : F = T + P
  vertex_edges : 2 * E = V * (T + P)
  edge_count : 3 * T + 5 * P = 2 * E

/-- Theorem stating that P + T + V = 34 for the given convex polyhedron -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.V = 34 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_sum_l929_92917


namespace NUMINAMATH_CALUDE_certain_number_proof_l929_92908

theorem certain_number_proof (x : ℝ) : x / 14.5 = 179 → x = 2595.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l929_92908


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_school_staff_sampling_l929_92905

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  h_sum : total = subgroups.sum

/-- Represents a sample -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Determines if a sample is representative of a population -/
def is_representative (pop : Population) (samp : Sample) : Prop :=
  samp.method = SamplingMethod.Stratified ∧ pop.subgroups.length > 1

/-- The main theorem stating that stratified sampling is most appropriate for a population with subgroups -/
theorem stratified_sampling_most_appropriate (pop : Population) (samp : Sample) 
    (h_subgroups : pop.subgroups.length > 1) : 
    is_representative pop samp ↔ samp.method = SamplingMethod.Stratified :=
  sorry

/-- The specific instance from the problem -/
def school_staff : Population :=
  { total := 160
  , subgroups := [120, 16, 24]
  , h_sum := by simp }

def staff_sample : Sample :=
  { size := 20
  , method := SamplingMethod.Stratified }

/-- The theorem applied to the specific instance -/
theorem school_staff_sampling : 
    is_representative school_staff staff_sample :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_school_staff_sampling_l929_92905


namespace NUMINAMATH_CALUDE_pi_irrational_less_than_neg_three_l929_92954

theorem pi_irrational_less_than_neg_three : 
  Irrational (-Real.pi) ∧ -Real.pi < -3 := by sorry

end NUMINAMATH_CALUDE_pi_irrational_less_than_neg_three_l929_92954


namespace NUMINAMATH_CALUDE_cos_equation_solution_l929_92966

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos x - 3 * Real.cos (4 * x))^2 = 16 + Real.sin (3 * x)^2 ↔ 
  ∃ k : ℤ, x = π + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l929_92966


namespace NUMINAMATH_CALUDE_girl_squirrel_walnuts_l929_92940

theorem girl_squirrel_walnuts (initial : ℕ) (boy_adds : ℕ) (girl_eats : ℕ) (final : ℕ) :
  initial = 12 →
  boy_adds = 5 →
  girl_eats = 2 →
  final = 20 →
  ∃ girl_brings : ℕ, initial + boy_adds + girl_brings - girl_eats = final ∧ girl_brings = 5 :=
by sorry

end NUMINAMATH_CALUDE_girl_squirrel_walnuts_l929_92940


namespace NUMINAMATH_CALUDE_cubic_sum_fraction_l929_92988

theorem cubic_sum_fraction (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)) = -3 / (2*(x^2 + y^2 + x*y)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_fraction_l929_92988


namespace NUMINAMATH_CALUDE_class_size_l929_92900

theorem class_size (total : ℕ) (brown_eyes : ℕ) (brown_eyes_black_hair : ℕ) : 
  (3 * brown_eyes = 2 * total) →
  (2 * brown_eyes_black_hair = brown_eyes) →
  (brown_eyes_black_hair = 6) →
  total = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l929_92900


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_with_conditions_l929_92946

/-- The greatest four-digit number that is two more than a multiple of 8 and four more than a multiple of 7 -/
def greatest_number : ℕ := 9990

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is two more than a multiple of 8 -/
def is_two_more_than_multiple_of_eight (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k + 2

/-- A number is four more than a multiple of 7 -/
def is_four_more_than_multiple_of_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k + 4

theorem greatest_four_digit_number_with_conditions :
  is_four_digit greatest_number ∧
  is_two_more_than_multiple_of_eight greatest_number ∧
  is_four_more_than_multiple_of_seven greatest_number ∧
  ∀ n : ℕ, is_four_digit n →
    is_two_more_than_multiple_of_eight n →
    is_four_more_than_multiple_of_seven n →
    n ≤ greatest_number :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_with_conditions_l929_92946


namespace NUMINAMATH_CALUDE_angles_on_y_axis_l929_92995

def terminal_side_on_y_axis (θ : Real) : Prop :=
  ∃ n : Int, θ = n * Real.pi + Real.pi / 2

theorem angles_on_y_axis :
  {θ : Real | terminal_side_on_y_axis θ} = {θ : Real | ∃ n : Int, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end NUMINAMATH_CALUDE_angles_on_y_axis_l929_92995


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l929_92996

/-- The number of chairs in the row -/
def totalChairs : ℕ := 10

/-- The set of broken chair numbers -/
def brokenChairs : Finset ℕ := {4, 7}

/-- The set of available chairs -/
def availableChairs : Finset ℕ := Finset.range totalChairs \ brokenChairs

/-- The probability that Mary and James do not sit next to each other -/
def probabilityNotAdjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  let totalWays := Nat.choose availableChairs.card 2
  let adjacentWays := (availableChairs.filter (fun n => n + 1 ∈ availableChairs)).card
  (totalWays - adjacentWays : ℚ) / totalWays = probabilityNotAdjacent :=
sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l929_92996


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l929_92927

/-- A right triangle with side lengths 6, 8, and 10 -/
structure RightTriangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (right_angle : PQ^2 + QR^2 = PR^2)
  (PQ_eq : PQ = 6)
  (QR_eq : QR = 8)
  (PR_eq : PR = 10)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (on_hypotenuse : side_length ≤ t.PR)
  (on_leg1 : side_length ≤ t.PQ)
  (on_leg2 : side_length ≤ t.QR)

/-- The side length of the inscribed square is 3 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l929_92927


namespace NUMINAMATH_CALUDE_alligator_coins_l929_92934

def river_crossing (initial : ℚ) : ℚ := 
  ((((initial * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30)

theorem alligator_coins : 
  ∃ initial : ℚ, river_crossing initial = 10 ∧ initial = 1210 / 81 := by
sorry

end NUMINAMATH_CALUDE_alligator_coins_l929_92934


namespace NUMINAMATH_CALUDE_correct_matching_probability_l929_92994

theorem correct_matching_probability (n : ℕ) (h : n = 4) : 
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l929_92994


namespace NUMINAMATH_CALUDE_football_practice_hours_l929_92984

/-- The number of hours a football team practices daily, given their weekly schedule and total practice time. -/
def daily_practice_hours (total_hours : ℕ) (practice_days : ℕ) : ℚ :=
  total_hours / practice_days

/-- Theorem stating that the daily practice hours is 6, given the conditions of the problem. -/
theorem football_practice_hours :
  let total_week_hours : ℕ := 36
  let days_in_week : ℕ := 7
  let rain_days : ℕ := 1
  let practice_days : ℕ := days_in_week - rain_days
  daily_practice_hours total_week_hours practice_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_hours_l929_92984


namespace NUMINAMATH_CALUDE_dan_onions_l929_92935

/-- The number of onions grown by Nancy, Dan, and Mike -/
structure OnionGrowth where
  nancy : ℕ
  dan : ℕ
  mike : ℕ

/-- The total number of onions grown -/
def total_onions (g : OnionGrowth) : ℕ :=
  g.nancy + g.dan + g.mike

/-- Theorem: Dan grew 9 onions -/
theorem dan_onions :
  ∀ g : OnionGrowth,
    g.nancy = 2 →
    g.mike = 4 →
    total_onions g = 15 →
    g.dan = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_onions_l929_92935


namespace NUMINAMATH_CALUDE_function_form_l929_92979

theorem function_form (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = Real.sin x ^ 2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_function_form_l929_92979


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l929_92955

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 4 * r + 9 = 0 ∧ 
              7 * s^2 + 4 * s + 9 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -4/9 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l929_92955


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l929_92968

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_1000 :
  ∀ m : ℕ, is_mersenne_prime m → m < 1000 → m ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l929_92968


namespace NUMINAMATH_CALUDE_min_m_bound_l929_92962

theorem min_m_bound (a b : ℝ) (h1 : |a - b| ≤ 1) (h2 : |2 * a - 1| ≤ 1) :
  ∃ m : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m) ∧
  (∀ m' : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m') → m ≤ m') ∧
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_min_m_bound_l929_92962


namespace NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l929_92950

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 48 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l929_92950


namespace NUMINAMATH_CALUDE_mooncake_problem_l929_92904

-- Define the types and variables
variable (type_a_cost type_b_cost : ℝ)
variable (total_cost_per_pair : ℝ)
variable (type_a_quantity type_b_quantity : ℕ)
variable (m : ℝ)

-- Define the conditions
def conditions (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) : Prop :=
  type_a_cost = 1200 ∧
  type_b_cost = 600 ∧
  total_cost_per_pair = 9 ∧
  type_a_quantity = 4 * type_b_quantity ∧
  m ≠ 0 ∧
  (type_a_cost / type_a_quantity + type_b_cost / type_b_quantity = total_cost_per_pair) ∧
  (2 * (type_a_quantity - 15 / 2 * m) + (6 - m / 5) * (type_b_quantity + 15 / 2 * m) = 1400 - 2 * m)

-- State the theorem
theorem mooncake_problem (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) :
  conditions type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m →
  type_a_quantity = 400 ∧ type_b_quantity = 100 ∧ m = 8 :=
by sorry

end NUMINAMATH_CALUDE_mooncake_problem_l929_92904


namespace NUMINAMATH_CALUDE_rooms_per_floor_l929_92939

theorem rooms_per_floor (total_earnings : ℕ) (hourly_rate : ℕ) (hours_per_room : ℕ) (num_floors : ℕ)
  (h1 : total_earnings = 3600)
  (h2 : hourly_rate = 15)
  (h3 : hours_per_room = 6)
  (h4 : num_floors = 4) :
  total_earnings / (hourly_rate * num_floors) = 10 := by
  sorry

#check rooms_per_floor

end NUMINAMATH_CALUDE_rooms_per_floor_l929_92939


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l929_92974

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  equation : ℝ → Prop

/-- Checks if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.equation p.x

/-- Checks if a line is parallel to the y-axis -/
def Line.parallelToYAxis (l : Line) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.equation x ↔ x = k

theorem line_through_point_parallel_to_y_axis 
  (A : Point) 
  (h_A : A.x = -3 ∧ A.y = 1) 
  (l : Line) 
  (h_parallel : l.parallelToYAxis) 
  (h_passes : A.liesOn l) : 
  ∀ x : ℝ, l.equation x ↔ x = -3 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l929_92974


namespace NUMINAMATH_CALUDE_rebecca_tips_calculation_l929_92944

/-- Rebecca's hair salon earnings calculation -/
def rebeccaEarnings (haircut_price perm_price dye_price dye_cost : ℕ) 
  (num_haircuts num_perms num_dyes : ℕ) (total_end_day : ℕ) : ℕ :=
  let service_earnings := haircut_price * num_haircuts + perm_price * num_perms + dye_price * num_dyes
  let dye_costs := dye_cost * num_dyes
  let tips := total_end_day - (service_earnings - dye_costs)
  tips

/-- Theorem stating that Rebecca's tips are $50 given the problem conditions -/
theorem rebecca_tips_calculation :
  rebeccaEarnings 30 40 60 10 4 1 2 310 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_tips_calculation_l929_92944


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l929_92976

theorem pictures_in_first_album 
  (total_pictures : ℕ) 
  (num_albums : ℕ) 
  (pics_per_album : ℕ) 
  (h1 : total_pictures = 65)
  (h2 : num_albums = 6)
  (h3 : pics_per_album = 8) :
  total_pictures - (num_albums * pics_per_album) = 17 :=
by sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l929_92976


namespace NUMINAMATH_CALUDE_min_distance_to_line_l929_92990

/-- The minimum distance from a point on y = e^x + x to the line 2x-y-3=0 -/
theorem min_distance_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x + x
  let P : ℝ × ℝ := (0, f 0)
  let d (x y : ℝ) : ℝ := |2*x - y - 3| / Real.sqrt (2^2 + (-1)^2)
  ∀ x : ℝ, d x (f x) ≥ d P.1 P.2 ∧ d P.1 P.2 = 4 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l929_92990


namespace NUMINAMATH_CALUDE_one_female_selection_l929_92945

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students in group A -/
def maleA : ℕ := 5

/-- The number of female students in group A -/
def femaleA : ℕ := 3

/-- The number of male students in group B -/
def maleB : ℕ := 6

/-- The number of female students in group B -/
def femaleB : ℕ := 2

/-- The number of students to be selected from each group -/
def selectPerGroup : ℕ := 2

/-- The total number of ways to select exactly one female student among 4 chosen students -/
theorem one_female_selection : 
  (choose femaleA 1 * choose maleA 1 * choose maleB selectPerGroup) + 
  (choose femaleB 1 * choose maleB 1 * choose maleA selectPerGroup) = 345 := by
  sorry

end NUMINAMATH_CALUDE_one_female_selection_l929_92945


namespace NUMINAMATH_CALUDE_fraction_change_l929_92992

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 - (a * 0.8) / (b * 1.28) / (a / b) = 0.375 := by sorry

end NUMINAMATH_CALUDE_fraction_change_l929_92992


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l929_92999

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l929_92999


namespace NUMINAMATH_CALUDE_time_reduction_percentage_l929_92948

/-- Calculates the time reduction percentage when increasing speed from 60 km/h to 86 km/h for a journey that initially takes 30 minutes. -/
theorem time_reduction_percentage 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (new_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 30) 
  (h3 : new_speed = 86) : 
  ∃ (reduction_percentage : ℝ), 
    (abs (reduction_percentage - 30.23) < 0.01) ∧ 
    (reduction_percentage = (1 - (initial_speed * initial_time) / (new_speed * initial_time)) * 100) :=
by sorry

end NUMINAMATH_CALUDE_time_reduction_percentage_l929_92948


namespace NUMINAMATH_CALUDE_transform_F_coordinates_l929_92932

/-- Reflection over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (1, 0)

theorem transform_F_coordinates :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_transform_F_coordinates_l929_92932


namespace NUMINAMATH_CALUDE_initial_apples_l929_92993

theorem initial_apples (initial : ℝ) (received : ℝ) (total : ℝ) : 
  received = 7.0 → total = 27 → initial + received = total → initial = 20.0 := by
sorry

end NUMINAMATH_CALUDE_initial_apples_l929_92993


namespace NUMINAMATH_CALUDE_factorization_x_squared_plus_5x_l929_92942

theorem factorization_x_squared_plus_5x (x : ℝ) : x^2 + 5*x = x*(x+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_plus_5x_l929_92942


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l929_92973

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  (∃ n : ℕ, a n = 33 ∧ a (n - 2) + a (n - 1) = 51) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l929_92973


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l929_92987

/-- Given a quadratic function f(x) = x² + 2x + c, prove that for points
    A(-3, y₁), B(-2, y₂), and C(2, y₃) on its graph, y₃ > y₁ > y₂ holds. -/
theorem quadratic_point_ordering (c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x + c
  let y₁ : ℝ := f (-3)
  let y₂ : ℝ := f (-2)
  let y₃ : ℝ := f 2
  y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l929_92987


namespace NUMINAMATH_CALUDE_equilateral_triangle_isosceles_points_l929_92918

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : sorry

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def is_inside (P : Point) (triangle : EquilateralTriangle) : Prop := sorry

theorem equilateral_triangle_isosceles_points (ABC : EquilateralTriangle) :
  ∃ (points : Finset Point),
    points.card = 10 ∧
    ∀ P ∈ points,
      is_inside P ABC ∧
      is_isosceles P ABC.B ABC.C ∧
      is_isosceles P ABC.A ABC.B ∧
      is_isosceles P ABC.A ABC.C :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_isosceles_points_l929_92918


namespace NUMINAMATH_CALUDE_bisection_method_step_l929_92961

def f (x : ℝ) := x^5 + 8*x^3 - 1

theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_bisection_method_step_l929_92961


namespace NUMINAMATH_CALUDE_flour_already_put_in_l929_92971

/-- Given a recipe that requires a certain amount of flour and the additional amount needed,
    calculate the amount of flour already put in. -/
theorem flour_already_put_in
  (recipe_requirement : ℕ)  -- Total cups of flour required by the recipe
  (additional_needed : ℕ)   -- Additional cups of flour needed
  (h1 : recipe_requirement = 7)  -- The recipe requires 7 cups of flour
  (h2 : additional_needed = 5)   -- Mary needs to add 5 more cups
  : recipe_requirement - additional_needed = 2 :=
by sorry

end NUMINAMATH_CALUDE_flour_already_put_in_l929_92971


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l929_92928

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l929_92928


namespace NUMINAMATH_CALUDE_intersects_once_impl_a_eq_one_l929_92982

/-- The function f(x) for a given 'a' -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

/-- Predicate to check if f(x) intersects x-axis at exactly one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem: If f(x) intersects x-axis at exactly one point, then a = 1 -/
theorem intersects_once_impl_a_eq_one :
  ∀ a : ℝ, intersects_once a → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersects_once_impl_a_eq_one_l929_92982
