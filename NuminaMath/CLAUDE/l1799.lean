import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1799_179940

theorem quadratic_form_nonnegative (a b c : ℝ) :
  (∀ (f g : ℝ × ℝ), a * (f.1^2 + f.2^2) + b * (f.1 * g.1 + f.2 * g.2) + c * (g.1^2 + g.2^2) ≥ 0) ↔
  (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1799_179940


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1799_179952

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1799_179952


namespace NUMINAMATH_CALUDE_wage_difference_l1799_179925

/-- Proves the wage difference between a manager and a chef given specific wage relationships -/
theorem wage_difference (manager_wage : ℝ) 
  (h1 : manager_wage = 6.50)
  (h2 : ∃ dishwasher_wage : ℝ, dishwasher_wage = manager_wage / 2)
  (h3 : ∃ chef_wage : ℝ, chef_wage = (manager_wage / 2) * 1.20) :
  manager_wage - ((manager_wage / 2) * 1.20) = 2.60 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l1799_179925


namespace NUMINAMATH_CALUDE_chloe_bookcase_problem_l1799_179968

theorem chloe_bookcase_problem :
  let average_books_per_shelf : ℚ := 8.5
  let mystery_shelves : ℕ := 7
  let picture_shelves : ℕ := 5
  let scifi_shelves : ℕ := 3
  let history_shelves : ℕ := 2
  let total_shelves : ℕ := mystery_shelves + picture_shelves + scifi_shelves + history_shelves
  let total_books : ℚ := average_books_per_shelf * total_shelves
  ⌈total_books⌉ = 145 := by
  sorry

#check chloe_bookcase_problem

end NUMINAMATH_CALUDE_chloe_bookcase_problem_l1799_179968


namespace NUMINAMATH_CALUDE_bernardo_always_wins_l1799_179981

def even_set : Finset ℕ := {2, 4, 6, 8, 10}
def odd_set : Finset ℕ := {1, 3, 5, 7, 9}

def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem bernardo_always_wins :
  ∀ (a b c : ℕ) (d e f : ℕ),
    a ∈ even_set → b ∈ even_set → c ∈ even_set →
    d ∈ odd_set → e ∈ odd_set → f ∈ odd_set →
    a ≠ b → b ≠ c → a ≠ c →
    d ≠ e → e ≠ f → d ≠ f →
    a > b → b > c →
    d > e → e > f →
    form_number a b c > form_number d e f :=
by sorry

end NUMINAMATH_CALUDE_bernardo_always_wins_l1799_179981


namespace NUMINAMATH_CALUDE_total_bottles_l1799_179979

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 30) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l1799_179979


namespace NUMINAMATH_CALUDE_custom_op_properties_l1799_179987

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 2 * a * b

-- State the theorem
theorem custom_op_properties :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = 2 * a * b) →
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = custom_op b a) ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → custom_op a (custom_op b c) = custom_op (custom_op a b) c) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/2) = a) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/(2*a)) ≠ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_custom_op_properties_l1799_179987


namespace NUMINAMATH_CALUDE_circle_reflection_and_translation_l1799_179954

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem circle_reflection_and_translation :
  let initial_center : ℝ × ℝ := (-3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_up reflected_center 3
  final_center = (-3, 7) := by sorry

end NUMINAMATH_CALUDE_circle_reflection_and_translation_l1799_179954


namespace NUMINAMATH_CALUDE_total_value_is_71_rupees_l1799_179975

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins and their values -/
def totalValueInRupees (totalCoins : ℕ) (coins20paise : ℕ) : ℚ :=
  let coins25paise := totalCoins - coins20paise
  let value20paise := 20 * coins20paise
  let value25paise := 25 * coins25paise
  (value20paise + value25paise : ℚ) / 100

/-- Theorem stating that the total value of the given coins is 71 rupees -/
theorem total_value_is_71_rupees :
  totalValueInRupees 334 250 = 71 := by
  sorry


end NUMINAMATH_CALUDE_total_value_is_71_rupees_l1799_179975


namespace NUMINAMATH_CALUDE_circle_diameter_l1799_179953

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l1799_179953


namespace NUMINAMATH_CALUDE_sector_angle_l1799_179993

/-- 
Given a circular sector where:
- r is the radius of the sector
- α is the central angle of the sector in radians
- l is the arc length of the sector
- C is the circumference of the sector

Prove that if C = 4r, then α = 2.
-/
theorem sector_angle (r : ℝ) (α : ℝ) (l : ℝ) (C : ℝ) 
  (h1 : C = 4 * r)  -- Circumference is four times the radius
  (h2 : C = 2 * r + l)  -- Circumference formula for a sector
  (h3 : l = α * r)  -- Arc length formula
  : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1799_179993


namespace NUMINAMATH_CALUDE_xy_value_l1799_179985

theorem xy_value (x y : ℝ) (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1799_179985


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1799_179929

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, f 1 + 3 * (x - 1) = 3 * x - 2) : 
  f 1 + (deriv f) 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1799_179929


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l1799_179989

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  elderly_population : ℕ
  young_population : ℕ
  young_sample : ℕ
  (elderly_population_le_total : elderly_population ≤ total_population)
  (young_population_le_total : young_population ≤ total_population)
  (young_sample_le_young_population : young_sample ≤ young_population)

/-- Calculates the number of elderly in the sample based on stratified sampling -/
def elderly_in_sample (s : StratifiedSample) : ℚ :=
  s.elderly_population * (s.young_sample : ℚ) / s.young_population

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSample) 
  (h_total : s.total_population = 430)
  (h_elderly : s.elderly_population = 90)
  (h_young : s.young_population = 160)
  (h_young_sample : s.young_sample = 32) :
  elderly_in_sample s = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l1799_179989


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1799_179955

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1799_179955


namespace NUMINAMATH_CALUDE_store_profit_calculation_l1799_179909

theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.18
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let final_price := new_year_price * (1 - february_discount)
  let profit := final_price - C
  profit = 0.23 * C := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l1799_179909


namespace NUMINAMATH_CALUDE_division_problem_l1799_179965

theorem division_problem (divisor : ℕ) : 
  (265 / divisor = 12) ∧ (265 % divisor = 1) → divisor = 22 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1799_179965


namespace NUMINAMATH_CALUDE_greatest_integer_a_for_quadratic_l1799_179967

theorem greatest_integer_a_for_quadratic : 
  ∃ (a : ℤ), a = 6 ∧ 
  (∀ x : ℝ, x^2 + a*x + 9 ≠ -2) ∧
  (∀ b : ℤ, b > a → ∃ x : ℝ, x^2 + b*x + 9 = -2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_for_quadratic_l1799_179967


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_10_l1799_179924

theorem mean_equality_implies_y_equals_10 : ∀ y : ℝ, 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_10_l1799_179924


namespace NUMINAMATH_CALUDE_greatest_among_given_numbers_l1799_179961

theorem greatest_among_given_numbers :
  let a := (42 : ℚ) * (7 / 11) / 100
  let b := 17 / 23
  let c := (7391 : ℚ) / 10000
  let d := 29 / 47
  b ≥ a ∧ b ≥ c ∧ b ≥ d := by sorry

end NUMINAMATH_CALUDE_greatest_among_given_numbers_l1799_179961


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1799_179935

/-- The curve xy = 2 intersects a circle at four points. Three of these points are given. -/
def intersection_points : Finset (ℚ × ℚ) :=
  {(4, 1/2), (-2, -1), (2/5, 5)}

/-- The fourth intersection point -/
def fourth_point : ℚ × ℚ := (-16/5, -5/8)

/-- All points satisfy the equation xy = 2 -/
def on_curve (p : ℚ × ℚ) : Prop :=
  p.1 * p.2 = 2

theorem fourth_intersection_point :
  (∀ p ∈ intersection_points, on_curve p) →
  on_curve fourth_point →
  ∃ (a b r : ℚ),
    (∀ p ∈ intersection_points, (p.1 - a)^2 + (p.2 - b)^2 = r^2) ∧
    (fourth_point.1 - a)^2 + (fourth_point.2 - b)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l1799_179935


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l1799_179957

theorem cyclists_meeting_time 
  (course_length : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * speed1 + t * speed2 = course_length ∧ t = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l1799_179957


namespace NUMINAMATH_CALUDE_zero_in_M_l1799_179972

def M : Set ℕ := {0, 1, 2}

theorem zero_in_M : 0 ∈ M := by sorry

end NUMINAMATH_CALUDE_zero_in_M_l1799_179972


namespace NUMINAMATH_CALUDE_remaining_money_l1799_179980

def savings : ℕ := 5376
def ticket_cost : ℕ := 1350

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money : 
  octal_to_decimal savings - ticket_cost = 1464 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1799_179980


namespace NUMINAMATH_CALUDE_square_root_problem_l1799_179913

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.64) + (Real.sqrt x / Real.sqrt 0.49) = 3.0892857142857144 →
  x = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1799_179913


namespace NUMINAMATH_CALUDE_history_book_cost_l1799_179966

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books : ℕ) 
  (h1 : total_books = 80) 
  (h2 : math_book_cost = 4) 
  (h3 : total_price = 368) 
  (h4 : math_books = 32) : 
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
sorry

end NUMINAMATH_CALUDE_history_book_cost_l1799_179966


namespace NUMINAMATH_CALUDE_pony_price_is_18_l1799_179988

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.1

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars from purchasing both types of jeans -/
def total_savings : ℝ := 9

/-- Theorem stating that the regular price of Pony jeans is $18 -/
theorem pony_price_is_18 : 
  ∃ (pony_price : ℝ), 
    pony_price = 18 ∧ 
    (fox_quantity * fox_price * (total_discount - pony_discount) + 
     pony_quantity * pony_price * pony_discount = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_18_l1799_179988


namespace NUMINAMATH_CALUDE_total_distance_is_176_l1799_179922

/-- Represents a series of linked rings with specific properties -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  smallestOutsideDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total vertical distance covered by the linked rings -/
def totalVerticalDistance (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total vertical distance is 176 cm for the given conditions -/
theorem total_distance_is_176 (rings : LinkedRings) 
  (h1 : rings.thickness = 2)
  (h2 : rings.topOutsideDiameter = 30)
  (h3 : rings.smallestOutsideDiameter = 10)
  (h4 : rings.diameterDecrease = 2) :
  totalVerticalDistance rings = 176 :=
sorry

end NUMINAMATH_CALUDE_total_distance_is_176_l1799_179922


namespace NUMINAMATH_CALUDE_octal_253_equals_171_l1799_179904

/-- Converts an octal digit to its decimal representation -/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- The octal representation of the number -/
def octal_number : List Nat := [2, 5, 3]

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal_conversion (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * (8 ^ i)) 0

theorem octal_253_equals_171 :
  octal_to_decimal_conversion octal_number = 171 := by
  sorry

end NUMINAMATH_CALUDE_octal_253_equals_171_l1799_179904


namespace NUMINAMATH_CALUDE_soda_sales_difference_l1799_179942

/-- Calculates the difference between evening and morning sales for Remy and Nick's soda business -/
theorem soda_sales_difference (remy_morning : ℕ) (nick_difference : ℕ) (price : ℚ) (evening_sales : ℚ) : 
  remy_morning = 55 →
  nick_difference = 6 →
  price = 1/2 →
  evening_sales = 55 →
  evening_sales - (price * (remy_morning + (remy_morning - nick_difference))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_soda_sales_difference_l1799_179942


namespace NUMINAMATH_CALUDE_probability_one_good_one_inferior_l1799_179928

/-- The probability of drawing one good quality bulb and one inferior quality bulb -/
theorem probability_one_good_one_inferior (total : ℕ) (good : ℕ) (inferior : ℕ) :
  total = 6 →
  good = 4 →
  inferior = 2 →
  (good + inferior : ℚ) / total * inferior / total + inferior / total * good / total = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_good_one_inferior_l1799_179928


namespace NUMINAMATH_CALUDE_range_of_H_l1799_179991

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ 3 ≤ y ∧ y ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_H_l1799_179991


namespace NUMINAMATH_CALUDE_remainder_is_x_squared_l1799_179986

-- Define the polynomials
def f (x : ℝ) := x^1010
def g (x : ℝ) := (x^2 + 1) * (x + 1) * (x - 1)

-- Define the remainder function
noncomputable def remainder (x : ℝ) := f x % g x

-- Theorem statement
theorem remainder_is_x_squared :
  ∀ x : ℝ, remainder x = x^2 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_is_x_squared_l1799_179986


namespace NUMINAMATH_CALUDE_correct_factorization_l1799_179917

theorem correct_factorization (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1799_179917


namespace NUMINAMATH_CALUDE_rectangle_length_equals_two_l1799_179916

theorem rectangle_length_equals_two
  (square_side : ℝ)
  (rectangle_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rectangle_width = 8)
  (h3 : square_side ^ 2 = rectangle_width * rectangle_length) :
  rectangle_length = 2 :=
by
  sorry

#check rectangle_length_equals_two

end NUMINAMATH_CALUDE_rectangle_length_equals_two_l1799_179916


namespace NUMINAMATH_CALUDE_x_power_plus_reciprocal_l1799_179982

theorem x_power_plus_reciprocal (θ : ℝ) (x : ℝ) (n : ℕ+) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = 2 * Real.sin θ) : 
  x^(n : ℝ) + 1 / x^(n : ℝ) = 2 * Real.cos (n * (π / 2 - θ)) := by
  sorry

end NUMINAMATH_CALUDE_x_power_plus_reciprocal_l1799_179982


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1799_179900

theorem equation_has_real_roots (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ x : ℝ, x ≠ 1 ∧ a^2 / x + b^2 / (x - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1799_179900


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1799_179910

theorem chinese_remainder_theorem_example :
  ∀ x : ℤ, x ≡ 9 [ZMOD 17] → x ≡ 5 [ZMOD 11] → x ≡ 60 [ZMOD 187] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1799_179910


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1799_179908

-- Define an H function
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Define a strictly increasing function
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem: A function is an H function if and only if it is strictly increasing
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1799_179908


namespace NUMINAMATH_CALUDE_investment_growth_l1799_179964

-- Define the initial investment
def initial_investment : ℝ := 359

-- Define the interest rate
def interest_rate : ℝ := 0.12

-- Define the number of years
def years : ℕ := 3

-- Define the final amount
def final_amount : ℝ := 504.32

-- Theorem statement
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_amount := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1799_179964


namespace NUMINAMATH_CALUDE_derivative_of_even_is_odd_l1799_179959

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem derivative_of_even_is_odd
  (f : ℝ → ℝ) (hf : IsEven f) (g : ℝ → ℝ) (hg : ∀ x, HasDerivAt f (g x) x) :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_is_odd_l1799_179959


namespace NUMINAMATH_CALUDE_volunteer_selection_schemes_l1799_179914

def num_candidates : ℕ := 5
def num_volunteers : ℕ := 4
def num_jobs : ℕ := 4

def driver_only_volunteer : ℕ := 1
def versatile_volunteers : ℕ := num_candidates - driver_only_volunteer

theorem volunteer_selection_schemes :
  (versatile_volunteers.factorial / (versatile_volunteers - (num_jobs - 1)).factorial) +
  (versatile_volunteers.factorial / (versatile_volunteers - num_jobs).factorial) = 48 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_schemes_l1799_179914


namespace NUMINAMATH_CALUDE_world_cup_2006_group_stage_matches_l1799_179963

/-- The number of matches in a single round-robin tournament with n teams -/
def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of matches in a tournament with groups -/
def total_matches (total_teams : ℕ) (num_groups : ℕ) : ℕ :=
  let teams_per_group := total_teams / num_groups
  num_groups * matches_in_group teams_per_group

theorem world_cup_2006_group_stage_matches :
  total_matches 32 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_2006_group_stage_matches_l1799_179963


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1799_179939

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 4) > 5/x + 21/10) ↔ (-2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1799_179939


namespace NUMINAMATH_CALUDE_susan_spending_ratio_l1799_179948

/-- Proves that the ratio of the amount spent on books to the amount left after buying clothes is 1:2 --/
theorem susan_spending_ratio (
  total_earned : ℝ)
  (spent_on_clothes : ℝ)
  (left_after_books : ℝ)
  (h1 : total_earned = 600)
  (h2 : spent_on_clothes = total_earned / 2)
  (h3 : left_after_books = 150) :
  (total_earned - spent_on_clothes - left_after_books) / (total_earned - spent_on_clothes) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_ratio_l1799_179948


namespace NUMINAMATH_CALUDE_man_coin_value_l1799_179984

/-- Represents the value of a coin in cents -/
def coin_value (is_nickel : Bool) : ℕ :=
  if is_nickel then 5 else 10

/-- Calculates the total value of coins in cents -/
def total_value (total_coins : ℕ) (nickel_count : ℕ) : ℕ :=
  (nickel_count * coin_value true) + ((total_coins - nickel_count) * coin_value false)

theorem man_coin_value :
  total_value 8 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_man_coin_value_l1799_179984


namespace NUMINAMATH_CALUDE_no_right_triangle_with_specific_medians_l1799_179907

theorem no_right_triangle_with_specific_medians : ∀ (m b a_leg b_leg : ℝ),
  a_leg > 0 → b_leg > 0 →
  (∃ (x y : ℝ), y = m * x + b) →  -- hypotenuse parallel to y = mx + b
  (∃ (x y : ℝ), y = 2 * x + 1) →  -- one median on y = 2x + 1
  (∃ (x y : ℝ), y = 5 * x + 2) →  -- another median on y = 5x + 2
  ¬ (
    -- Right triangle condition
    a_leg^2 + b_leg^2 = (a_leg^2 + b_leg^2) ∧
    -- Hypotenuse parallel to y = mx + b
    m = -b_leg / a_leg ∧
    -- One median on y = 2x + 1
    (2 * b_leg / a_leg = 2 ∨ b_leg / (2 * a_leg) = 2) ∧
    -- Another median on y = 5x + 2
    (2 * b_leg / a_leg = 5 ∨ b_leg / (2 * a_leg) = 5)
  ) := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_specific_medians_l1799_179907


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l1799_179974

/-- Given 4 moles of a compound with a total molecular weight of 304 g/mol,
    prove that the molecular weight of 1 mole of the compound is 76 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (total_moles : ℝ) 
  (h1 : total_weight = 304)
  (h2 : total_moles = 4) :
  total_weight / total_moles = 76 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l1799_179974


namespace NUMINAMATH_CALUDE_cos_angle_sum_diff_vectors_l1799_179976

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

theorem cos_angle_sum_diff_vectors :
  let sum := (a.1 + b.1, a.2 + b.2)
  let diff := (a.1 - b.1, a.2 - b.2)
  (sum.1 * diff.1 + sum.2 * diff.2) / 
  (Real.sqrt (sum.1^2 + sum.2^2) * Real.sqrt (diff.1^2 + diff.2^2)) = 
  Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_sum_diff_vectors_l1799_179976


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1799_179973

theorem arithmetic_sequence_count (a₁ a_n d : ℤ) (h1 : a₁ = 156) (h2 : a_n = 36) (h3 : d = -4) :
  (a₁ - a_n) / d + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1799_179973


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l1799_179946

theorem expression_simplification_and_evaluation :
  ∀ a : ℝ, a ≠ 1 ∧ a ≠ 2 ∧ a ≠ 0 →
    (a - 3 + 1 / (a - 1)) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2)) = a / (a - 1) :=
by sorry

theorem expression_evaluation_at_negative_one :
  (-1 - 3 + 1 / (-1 - 1)) / (((-1)^2 - 4) / ((-1)^2 + 2*(-1))) * (1 / (-1 - 2)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l1799_179946


namespace NUMINAMATH_CALUDE_rectangle_area_l1799_179938

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1799_179938


namespace NUMINAMATH_CALUDE_prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l1799_179906

-- Define the number of yellow and white balls
def yellow_balls : ℕ := 3
def white_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing 3 white balls
def prob_three_white : ℚ := 1 / 20

-- Define the probability of drawing 2 yellow and 1 white ball
def prob_two_yellow_one_white : ℚ := 9 / 20

-- Define the number of draws per day
def draws_per_day : ℕ := 100

-- Define the number of days in a month
def days_in_month : ℕ := 30

-- Define the earnings for non-matching draws
def earn_non_matching : ℤ := 1

-- Define the loss for matching draws
def loss_matching : ℤ := 5

-- Theorem for the probability of drawing 3 white balls
theorem prob_three_white_correct :
  prob_three_white = 1 / 20 := by sorry

-- Theorem for the probability of drawing 2 yellow and 1 white ball
theorem prob_two_yellow_one_white_correct :
  prob_two_yellow_one_white = 9 / 20 := by sorry

-- Theorem for the total earnings in a month
theorem total_earnings_correct :
  (draws_per_day * days_in_month * 
   (earn_non_matching * (1 - (prob_three_white + prob_two_yellow_one_white)) - 
    loss_matching * (prob_three_white + prob_two_yellow_one_white))) = 1200 := by sorry

end NUMINAMATH_CALUDE_prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l1799_179906


namespace NUMINAMATH_CALUDE_function_difference_equals_nine_minimum_value_minus_four_l1799_179960

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem 1
theorem function_difference_equals_nine (a : ℝ) :
  f a (a + 1) - f a a = 9 → a = 2 :=
sorry

-- Theorem 2
theorem minimum_value_minus_four (a : ℝ) :
  (∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_function_difference_equals_nine_minimum_value_minus_four_l1799_179960


namespace NUMINAMATH_CALUDE_cookie_sales_value_l1799_179969

theorem cookie_sales_value (total_boxes : ℝ) (plain_boxes : ℝ) 
  (choc_chip_price : ℝ) (plain_price : ℝ) 
  (h1 : total_boxes = 1585) 
  (h2 : plain_boxes = 793.125) 
  (h3 : choc_chip_price = 1.25) 
  (h4 : plain_price = 0.75) : 
  (total_boxes - plain_boxes) * choc_chip_price + plain_boxes * plain_price = 1584.6875 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_value_l1799_179969


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l1799_179970

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 553) : 
  3*x^2*y^2 = 2886 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l1799_179970


namespace NUMINAMATH_CALUDE_prime_congruence_l1799_179923

theorem prime_congruence (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  (∃ x : ℤ, (x^4 + x^3 + x^2 + x + 1) % p = 0) →
  p % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_congruence_l1799_179923


namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l1799_179915

/-- Given the income ratio and savings of Uma and Bala, prove their expenditure ratio -/
theorem expenditure_ratio_proof (uma_income bala_income uma_expenditure bala_expenditure : ℚ) 
  (h1 : uma_income = (8 : ℚ) / 7 * bala_income)
  (h2 : uma_income = 16000)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000) :
  uma_expenditure / bala_expenditure = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_proof_l1799_179915


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1799_179943

theorem circle_area_ratio : 
  let d1 : ℝ := 2  -- diameter of smallest circle
  let d2 : ℝ := 6  -- diameter of middle circle
  let d3 : ℝ := 10 -- diameter of largest circle
  let r1 : ℝ := d1 / 2  -- radius of smallest circle
  let r2 : ℝ := d2 / 2  -- radius of middle circle
  let r3 : ℝ := d3 / 2  -- radius of largest circle
  let area_smallest : ℝ := π * r1^2
  let area_middle : ℝ := π * r2^2
  let area_largest : ℝ := π * r3^2
  let area_green : ℝ := area_largest - area_middle
  let area_red : ℝ := area_smallest
  (area_green / area_red : ℝ) = 16
  := by sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l1799_179943


namespace NUMINAMATH_CALUDE_rectangle_to_rhombus_l1799_179962

-- Define a rectangle
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (width_pos : width > 0)
  (height_pos : height > 0)

-- Define a rhombus
structure Rhombus :=
  (side : ℝ)
  (side_pos : side > 0)

-- Define the theorem
theorem rectangle_to_rhombus (r : Rectangle) : 
  ∃ (rh : Rhombus), r.width * r.height = 4 * rh.side^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_rhombus_l1799_179962


namespace NUMINAMATH_CALUDE_max_tickets_proof_l1799_179918

/-- Represents the maximum number of tickets Jane can buy given the following conditions:
  * Each ticket costs $15
  * Jane has a budget of $180
  * If more than 10 tickets are bought, there's a discount of $2 per ticket
-/
def max_tickets : ℕ := 13

/-- The cost of a ticket without discount -/
def ticket_cost : ℕ := 15

/-- Jane's budget -/
def budget : ℕ := 180

/-- The discount per ticket when buying more than 10 tickets -/
def discount : ℕ := 2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 10

theorem max_tickets_proof :
  (∀ n : ℕ, n ≤ discount_threshold → n * ticket_cost ≤ budget) ∧
  (∀ n : ℕ, n > discount_threshold → n * (ticket_cost - discount) ≤ budget) ∧
  (∀ n : ℕ, n > max_tickets → 
    (if n ≤ discount_threshold then n * ticket_cost > budget
     else n * (ticket_cost - discount) > budget)) :=
sorry

end NUMINAMATH_CALUDE_max_tickets_proof_l1799_179918


namespace NUMINAMATH_CALUDE_discounted_tickets_count_l1799_179996

/-- Proves the number of discounted tickets bought given the problem conditions -/
theorem discounted_tickets_count :
  ∀ (full_price discounted_price : ℚ) 
    (total_tickets : ℕ) 
    (total_spent : ℚ),
  full_price = 2 →
  discounted_price = (8 : ℚ) / 5 →
  total_tickets = 10 →
  total_spent = (92 : ℚ) / 5 →
  ∃ (full_tickets discounted_tickets : ℕ),
    full_tickets + discounted_tickets = total_tickets ∧
    full_price * full_tickets + discounted_price * discounted_tickets = total_spent ∧
    discounted_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_discounted_tickets_count_l1799_179996


namespace NUMINAMATH_CALUDE_fraction_equality_l1799_179931

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (12/5) / (2 * q + p) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1799_179931


namespace NUMINAMATH_CALUDE_sum_even_divisors_1000_l1799_179978

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem sum_even_divisors_1000 : sum_even_divisors 1000 = 2184 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_1000_l1799_179978


namespace NUMINAMATH_CALUDE_distance_traveled_downstream_l1799_179999

/-- The distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 68 km -/
theorem distance_traveled_downstream : 
  distance_downstream 13 4 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_downstream_l1799_179999


namespace NUMINAMATH_CALUDE_triangle_properties_l1799_179911

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (c * Real.sin B = b * Real.cos (C - π/6) ∨
   Real.cos B = (2*a - b) / (2*c) ∨
   (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 3 * a * b) →
  (Real.sin C = Real.sqrt 3 / 2 ∧
   (c = Real.sqrt 3 ∧ a = 3*b → 
    1/2 * a * b * Real.sin C = 9 * Real.sqrt 3 / 28)) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l1799_179911


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l1799_179919

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  Real.log (1 + 27 * x^5) / Real.log (1 + x^2) +
  Real.log (1 + x^2) / Real.log (1 - 2*x^2 + 27*x^4) ≤
  1 + Real.log (1 + 27*x^5) / Real.log (1 - 2*x^2 + 27*x^4)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (x > -Real.rpow 27 (-1/5) ∧ x ≤ -1/3) ∨
  (x > -Real.sqrt (2/27) ∧ x < 0) ∨
  (x > 0 ∧ x < Real.sqrt (2/27)) ∨
  x = 1/3

-- State the theorem
theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l1799_179919


namespace NUMINAMATH_CALUDE_largest_integral_x_l1799_179901

theorem largest_integral_x : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 2/3 ∧ 
  x < 10 ∧
  x = 3 ∧
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 2/3 ∧ y < 10) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l1799_179901


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l1799_179997

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ (a' b' : ℝ), -2 < a' ∧ a' < 1 ∧ 0 < b' ∧ b' < 4 ∧ x = a' - b') ↔ -6 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l1799_179997


namespace NUMINAMATH_CALUDE_original_price_calculation_l1799_179958

def discount_rate : ℝ := 0.2
def discounted_price : ℝ := 56

theorem original_price_calculation :
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_rate) = discounted_price ∧ 
    original_price = 70 :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1799_179958


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1799_179903

theorem mathborough_rainfall_2004 (rainfall_2003 rainfall_2004 : ℕ) 
  (h1 : rainfall_2003 = 45)
  (h2 : rainfall_2004 = rainfall_2003 + 3)
  (h3 : ∃ (high_months low_months : ℕ), 
    high_months = 8 ∧ 
    low_months = 12 - high_months ∧
    (high_months * (rainfall_2004 + 5) + low_months * rainfall_2004 = 616)) : 
  rainfall_2004 * 12 + 8 * 5 = 616 := by
  sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1799_179903


namespace NUMINAMATH_CALUDE_condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l1799_179950

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define functions to calculate side lengths and angles
def side_length (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  let a := side_length t.A t.B
  let b := side_length t.B t.C
  let c := side_length t.C t.A
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem for condition A
theorem condition_A_right_triangle (t : Triangle) :
  side_length t.A t.B = 3 ∧ side_length t.B t.C = 4 ∧ side_length t.C t.A = 5 →
  is_right_triangle t :=
sorry

-- Theorem for condition B
theorem condition_B_right_triangle (t : Triangle) (k : ℝ) :
  side_length t.A t.B = 3*k ∧ side_length t.B t.C = 4*k ∧ side_length t.C t.A = 5*k →
  is_right_triangle t :=
sorry

-- Theorem for condition C
theorem condition_C_not_right_triangle (t : Triangle) :
  ∃ (k : ℝ), angle t.B t.A t.C = 3*k ∧ angle t.C t.B t.A = 4*k ∧ angle t.A t.C t.B = 5*k →
  ¬ is_right_triangle t :=
sorry

-- Theorem for condition D
theorem condition_D_right_triangle (t : Triangle) :
  angle t.B t.A t.C = 40 ∧ angle t.C t.B t.A = 50 →
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l1799_179950


namespace NUMINAMATH_CALUDE_mary_money_left_l1799_179932

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 3 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

/-- Theorem stating that the amount of money Mary has left is 30 - 8p -/
theorem mary_money_left (p : ℝ) : money_left p = 30 - 8 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l1799_179932


namespace NUMINAMATH_CALUDE_lives_per_player_l1799_179977

theorem lives_per_player (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : quitting_players = 5)
  (h3 : total_lives = 15)
  (h4 : initial_players > quitting_players) :
  total_lives / (initial_players - quitting_players) = 5 := by
  sorry

end NUMINAMATH_CALUDE_lives_per_player_l1799_179977


namespace NUMINAMATH_CALUDE_fox_jeans_price_l1799_179949

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

/-- Total savings on 5 pairs of jeans (3 Fox, 2 Pony) in dollars -/
def total_savings : ℝ := 8.91

/-- Sum of discount rates for Fox and Pony jeans as a percentage -/
def total_discount_rate : ℝ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℝ := 10.999999999999996

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

theorem fox_jeans_price : 
  ∃ (fox_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    3 * (fox_price * fox_discount_rate / 100) + 
    2 * (pony_price * pony_discount_rate / 100) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l1799_179949


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l1799_179945

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 5 →
  initial_avg = 30 →
  leaving_age = 18 →
  remaining_people = 4 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l1799_179945


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1799_179956

/-- A quadratic function of the form y = a(x-3)² + c where a < 0 -/
def quadratic_function (a c : ℝ) (h : a < 0) : ℝ → ℝ := 
  fun x => a * (x - 3)^2 + c

theorem quadratic_inequality (a c : ℝ) (h : a < 0) :
  let f := quadratic_function a c h
  let y₁ := f (Real.sqrt 5)
  let y₂ := f 0
  let y₃ := f 4
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1799_179956


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l1799_179905

theorem pencils_in_drawer (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 2)
  (h2 : final_pencils = 5) :
  final_pencils - initial_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l1799_179905


namespace NUMINAMATH_CALUDE_not_prime_sum_minus_one_l1799_179983

theorem not_prime_sum_minus_one (m n : ℤ) 
  (hm : m > 1) 
  (hn : n > 1) 
  (h_divides : (m + n - 1) ∣ (m^2 + n^2 - 1)) : 
  ¬(Nat.Prime (m + n - 1).natAbs) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_minus_one_l1799_179983


namespace NUMINAMATH_CALUDE_sum_of_distances_forms_ellipse_l1799_179951

/-- Definition of an ellipse based on the sum of distances to two foci -/
def is_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0) ∧ a > c ∧ c > 0 ∧
  S = {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x - c)^2 + y^2) + Real.sqrt ((x + c)^2 + y^2) = 2 * a}

/-- Theorem: The set of points satisfying the sum of distances to two foci is an ellipse -/
theorem sum_of_distances_forms_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) 
  (h : is_ellipse F₁ F₂ a S) : 
  ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ S = {p : ℝ × ℝ | let (x, y) := p; x^2 / a'^2 + y^2 / b'^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_forms_ellipse_l1799_179951


namespace NUMINAMATH_CALUDE_reflect_point_5_neg3_l1799_179921

/-- Given a point P in the Cartesian coordinate system, 
    this function returns its coordinates with respect to the y-axis. -/
def reflect_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- The coordinates of P(5,-3) with respect to the y-axis are (-5,-3). -/
theorem reflect_point_5_neg3 : 
  reflect_y_axis 5 (-3) = (-5, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_point_5_neg3_l1799_179921


namespace NUMINAMATH_CALUDE_greatest_power_of_three_dividing_30_factorial_l1799_179927

-- Define v as 30!
def v : ℕ := Nat.factorial 30

-- Define the property that 3^k is a factor of v
def is_factor_of_v (k : ℕ) : Prop := ∃ m : ℕ, v = m * (3^k)

-- Theorem statement
theorem greatest_power_of_three_dividing_30_factorial :
  (∃ k : ℕ, is_factor_of_v k ∧ ∀ j : ℕ, j > k → ¬is_factor_of_v j) ∧
  (∀ k : ℕ, (∃ j : ℕ, j > k ∧ is_factor_of_v j) → k ≤ 14) ∧
  is_factor_of_v 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_dividing_30_factorial_l1799_179927


namespace NUMINAMATH_CALUDE_root_in_smaller_interval_l1799_179998

-- Define the function
def f (x : ℝ) := x^3 - 6*x^2 + 4

-- State the theorem
theorem root_in_smaller_interval :
  (∃ x ∈ Set.Ioo 0 1, f x = 0) →
  (∃ x ∈ Set.Ioo (1/2) 1, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_in_smaller_interval_l1799_179998


namespace NUMINAMATH_CALUDE_volume_of_cut_cone_l1799_179944

/-- The volume of the cone cut to form a frustum, given the frustum's properties -/
theorem volume_of_cut_cone (r R h H : ℝ) : 
  (R = 3 * r) →  -- Area of one base is 9 times the other
  (H = 3 * h) →  -- Height ratio follows from radius ratio
  (π * R^2 * H / 3 - π * r^2 * h / 3 = 52) →  -- Volume of frustum is 52
  (π * r^2 * h / 3 = 54) :=  -- Volume of cut cone is 54
by sorry

end NUMINAMATH_CALUDE_volume_of_cut_cone_l1799_179944


namespace NUMINAMATH_CALUDE_max_table_height_specific_triangle_l1799_179971

/-- Triangle ABC with sides a, b, and c -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The maximum height of a table constructed from a triangle -/
def maxTableHeight {α : Type*} [LinearOrderedField α] (t : Triangle α) : α :=
  sorry

/-- The theorem stating the maximum height of the table -/
theorem max_table_height_specific_triangle :
  let t : Triangle ℝ := ⟨25, 29, 32⟩
  maxTableHeight t = 84 * Real.sqrt 1547 / 57 := by
  sorry

end NUMINAMATH_CALUDE_max_table_height_specific_triangle_l1799_179971


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1799_179947

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / ((x - 2) * (x + 1)) = 0 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1799_179947


namespace NUMINAMATH_CALUDE_operation_2012_equals_55_l1799_179912

def operation_sequence (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case is unreachable, but needed for completeness

theorem operation_2012_equals_55 : operation_sequence 2012 = 55 := by
  sorry

end NUMINAMATH_CALUDE_operation_2012_equals_55_l1799_179912


namespace NUMINAMATH_CALUDE_men_per_table_l1799_179990

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 9 →
  women_per_table = 7 →
  total_customers = 90 →
  ∃ (men_per_table : ℕ), 
    men_per_table * num_tables + women_per_table * num_tables = total_customers ∧
    men_per_table = 3 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l1799_179990


namespace NUMINAMATH_CALUDE_equation_solution_range_l1799_179930

-- Define the set of real numbers greater than 0 and not equal to 1
def A : Set ℝ := {a | a > 0 ∧ a ≠ 1}

-- Define the function representing the equation
def f (a : ℝ) (k : ℝ) (x : ℝ) : Prop :=
  Real.log (x - a * k) / Real.log (Real.sqrt a) = Real.log (x^2 - a^2) / Real.log a

-- Define the set of k values that satisfy the equation for some x
def K (a : ℝ) : Set ℝ := {k | ∃ x, f a k x}

-- Theorem statement
theorem equation_solution_range (a : A) :
  K a = {k | k < -1 ∨ (k > 0 ∧ k < 1)} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1799_179930


namespace NUMINAMATH_CALUDE_max_sin_squared_sum_l1799_179902

theorem max_sin_squared_sum (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / (Real.sin A) = b / (Real.sin B)) →
  (b / (Real.sin B) = c / (Real.sin C)) →
  ((2 * Real.sin A - Real.sin C) / Real.sin C = (a^2 + b^2 - c^2) / (a^2 + c^2 - b^2)) →
  (∃ (x : Real), x = Real.sin A^2 + Real.sin C^2 ∧ ∀ (y : Real), y = Real.sin A^2 + Real.sin C^2 → y ≤ x) →
  (Real.sin A^2 + Real.sin C^2 ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_sin_squared_sum_l1799_179902


namespace NUMINAMATH_CALUDE_discount_theorem_l1799_179936

/-- Calculates the final price and equivalent discount for a given original price and discounts --/
def discount_calculation (original_price : ℝ) (store_discount : ℝ) (vip_discount : ℝ) : ℝ × ℝ :=
  let final_price := original_price * (1 - store_discount) * (1 - vip_discount)
  let equivalent_discount := 1 - (1 - store_discount) * (1 - vip_discount)
  (final_price, equivalent_discount * 100)

theorem discount_theorem :
  discount_calculation 1500 0.8 0.05 = (1140, 76) := by
  sorry

end NUMINAMATH_CALUDE_discount_theorem_l1799_179936


namespace NUMINAMATH_CALUDE_francie_remaining_money_l1799_179926

/-- Calculates the remaining money after Francie's savings and purchases -/
def remaining_money (initial_allowance : ℕ) (initial_weeks : ℕ) 
  (raised_allowance : ℕ) (raised_weeks : ℕ) (video_game_cost : ℕ) : ℕ :=
  let total_savings := initial_allowance * initial_weeks + raised_allowance * raised_weeks
  let after_clothes := total_savings / 2
  after_clothes - video_game_cost

/-- Theorem stating that Francie's remaining money is $3 -/
theorem francie_remaining_money :
  remaining_money 5 8 6 6 35 = 3 := by
  sorry

#eval remaining_money 5 8 6 6 35

end NUMINAMATH_CALUDE_francie_remaining_money_l1799_179926


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1799_179995

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1799_179995


namespace NUMINAMATH_CALUDE_three_fractions_product_one_l1799_179920

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem three_fractions_product_one :
  ∃ (a b c d e f : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_fractions_product_one_l1799_179920


namespace NUMINAMATH_CALUDE_arrival_time_difference_l1799_179992

def distance : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem arrival_time_difference : 
  let jill_time := distance / jill_speed
  let jack_time := distance / jack_speed
  (jack_time - jill_time) * 60 = 5.4 := by sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l1799_179992


namespace NUMINAMATH_CALUDE_distribute_objects_l1799_179937

theorem distribute_objects (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) :
  (Nat.factorial n) / ((Nat.factorial m)^k) = 34650 :=
sorry

end NUMINAMATH_CALUDE_distribute_objects_l1799_179937


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1799_179994

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ x : ℕ, x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1799_179994


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1799_179941

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧
    (∀ x y, x < y ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, Real.log a < x ∧ x < y → f a x < f a y)) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ (∀ z, z ≠ x ∧ z ≠ y → f a z ≠ 0) ↔
    (0 < a ∧ a < 1) ∨ (a > 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1799_179941


namespace NUMINAMATH_CALUDE_february_first_is_monday_l1799_179933

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in February -/
structure FebruaryDay where
  day : Nat
  weekday : Weekday

/-- Defines the properties of February in the given year -/
structure FebruaryProperties where
  days : List FebruaryDay
  monday_count : Nat
  thursday_count : Nat
  first_day : Weekday

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must be a Monday -/
theorem february_first_is_monday
  (feb : FebruaryProperties)
  (h1 : feb.monday_count = 4)
  (h2 : feb.thursday_count = 4)
  : feb.first_day = Weekday.Monday := by
  sorry

end NUMINAMATH_CALUDE_february_first_is_monday_l1799_179933


namespace NUMINAMATH_CALUDE_tax_assignment_correct_l1799_179934

/-- Represents the different types of budget levels in the Russian tax system -/
inductive BudgetLevel
  | Federal
  | Regional

/-- Represents the different types of taxes in the Russian tax system -/
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportFee

/-- Assigns a tax to a budget level -/
def assignTax (tax : TaxType) : BudgetLevel :=
  match tax with
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportFee => BudgetLevel.Regional

/-- Theorem stating the correct assignment of taxes to budget levels -/
theorem tax_assignment_correct :
  (assignTax TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.FederalTax = BudgetLevel.Federal) ∧
  (assignTax TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.TransportFee = BudgetLevel.Regional) :=
by sorry

end NUMINAMATH_CALUDE_tax_assignment_correct_l1799_179934
