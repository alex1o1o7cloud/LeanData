import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2570_257008

theorem equation_solution : ∃ x : ℝ, 6 * x + 12 * x = 558 - 9 * (x - 4) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2570_257008


namespace NUMINAMATH_CALUDE_yard_area_l2570_257024

def yard_length : ℝ := 20
def yard_width : ℝ := 18
def square_cutout_side : ℝ := 4
def rect_cutout_length : ℝ := 2
def rect_cutout_width : ℝ := 5

theorem yard_area : 
  yard_length * yard_width - 
  square_cutout_side * square_cutout_side - 
  rect_cutout_length * rect_cutout_width = 334 := by
sorry

end NUMINAMATH_CALUDE_yard_area_l2570_257024


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_six_l2570_257012

theorem no_solution_iff_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2 * x + m) / (x + 3) ≠ 1) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_six_l2570_257012


namespace NUMINAMATH_CALUDE_tangent_slope_at_origin_l2570_257027

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_at_origin :
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_origin_l2570_257027


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2570_257032

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := -2 + 3*I
  3*a + 4*b = 1 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2570_257032


namespace NUMINAMATH_CALUDE_insulated_cups_problem_l2570_257085

-- Define the cost prices and quantities
def cost_A : ℝ := 110
def cost_B : ℝ := 88
def quantity_A : ℕ := 30
def quantity_B : ℕ := 50

-- Define the selling prices
def sell_A : ℝ := 160
def sell_B : ℝ := 140

-- Define the total number of cups and profit
def total_cups : ℕ := 80
def total_profit : ℝ := 4100

-- Theorem statement
theorem insulated_cups_problem :
  -- Condition 1: 4 type A cups cost the same as 5 type B cups
  4 * cost_A = 5 * cost_B ∧
  -- Condition 2: 3 type A cups cost $154 more than 2 type B cups
  3 * cost_A = 2 * cost_B + 154 ∧
  -- Condition 3: Total cups purchased is 80
  quantity_A + quantity_B = total_cups ∧
  -- Condition 4: Profit calculation
  (sell_A - cost_A) * quantity_A + (sell_B - cost_B) * quantity_B = total_profit :=
by
  sorry


end NUMINAMATH_CALUDE_insulated_cups_problem_l2570_257085


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2570_257052

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (Complex.I * (a - 1) : ℂ).im ≠ 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2570_257052


namespace NUMINAMATH_CALUDE_fib_sum_product_l2570_257021

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: F_{m+n} = F_{m-1} * F_n + F_m * F_{n+1} for all non-negative integers m and n -/
theorem fib_sum_product (m n : ℕ) : fib (m + n) = fib (m - 1) * fib n + fib m * fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_product_l2570_257021


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2570_257070

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2570_257070


namespace NUMINAMATH_CALUDE_prob_exactly_two_of_three_l2570_257006

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two_of_three (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/5) (h_B : p_B = 1/4) (h_C : p_C = 1/3) :
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_two_of_three_l2570_257006


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2570_257044

/-- The area of the shaded region in a square with side length 6 and inscribed circles
    of radius 2√3 at each corner is equal to 36 - 12√3 - 4π. -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ)
  (h_side : square_side = 6)
  (h_radius : circle_radius = 2 * Real.sqrt 3) :
  let total_area := square_side ^ 2
  let triangle_area := 8 * (1 / 2 * (square_side / 2) * circle_radius)
  let sector_area := 4 * (1 / 12 * π * circle_radius ^ 2)
  total_area - triangle_area - sector_area = 36 - 12 * Real.sqrt 3 - 4 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2570_257044


namespace NUMINAMATH_CALUDE_remainder_3_20_mod_11_l2570_257071

theorem remainder_3_20_mod_11 (h : Prime 11) : 3^20 ≡ 1 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_20_mod_11_l2570_257071


namespace NUMINAMATH_CALUDE_quadratic_roots_l2570_257019

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesThrough : f (-3) = 0 ∧ f (-2) = -3 ∧ f 0 = -3

/-- The roots of the quadratic function -/
def roots (qf : QuadraticFunction) : Set ℝ :=
  {x : ℝ | qf.f x = 0}

/-- Theorem stating the roots of the quadratic function -/
theorem quadratic_roots (qf : QuadraticFunction) : roots qf = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2570_257019


namespace NUMINAMATH_CALUDE_egg_production_increase_proof_l2570_257093

/-- The increase in egg production from last year to this year -/
def egg_production_increase (last_year_production this_year_production : ℕ) : ℕ :=
  this_year_production - last_year_production

/-- Theorem stating the increase in egg production -/
theorem egg_production_increase_proof 
  (last_year_production : ℕ) 
  (this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) : 
  egg_production_increase last_year_production this_year_production = 3220 := by
  sorry

end NUMINAMATH_CALUDE_egg_production_increase_proof_l2570_257093


namespace NUMINAMATH_CALUDE_smallest_triple_sum_of_squares_l2570_257078

/-- A function that checks if a number can be expressed as the sum of three squares -/
def isSumOfThreeSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a^2 + b^2 + c^2

/-- A function that counts the number of ways a number can be expressed as the sum of three squares -/
def countSumOfThreeSquares (n : ℕ) : ℕ :=
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => 
    let (a, b, c) := triple
    n = a^2 + b^2 + c^2
  ) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1))))).card

/-- Theorem stating that 110 is the smallest positive integer that can be expressed as the sum of three squares in at least three different ways -/
theorem smallest_triple_sum_of_squares : 
  (∀ m : ℕ, m < 110 → countSumOfThreeSquares m < 3) ∧ 
  countSumOfThreeSquares 110 ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_triple_sum_of_squares_l2570_257078


namespace NUMINAMATH_CALUDE_b_join_time_correct_l2570_257009

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- A's initial investment in Rupees -/
def aInvestment : ℕ := 36000

/-- B's initial investment in Rupees -/
def bInvestment : ℕ := 54000

/-- Profit sharing ratio of A to B -/
def profitRatio : ℚ := 2 / 1

/-- Calculates the time B joined the business in months -/
def bJoinTime : ℕ := monthsInYear - 8

theorem b_join_time_correct :
  (aInvestment * monthsInYear : ℚ) / (bInvestment * bJoinTime) = profitRatio :=
sorry

end NUMINAMATH_CALUDE_b_join_time_correct_l2570_257009


namespace NUMINAMATH_CALUDE_equator_scientific_notation_l2570_257039

/-- The circumference of the equator in meters -/
def equator_circumference : ℕ := 40210000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem equator_scientific_notation :
  to_scientific_notation equator_circumference = ScientificNotation.mk 4.021 7 := by
  sorry

end NUMINAMATH_CALUDE_equator_scientific_notation_l2570_257039


namespace NUMINAMATH_CALUDE_cans_difference_l2570_257082

/-- The number of cans collected by Sarah yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of cans collected by Lara yesterday -/
def lara_yesterday : ℕ := sarah_yesterday + 30

/-- The number of cans collected by Alex yesterday -/
def alex_yesterday : ℕ := 90

/-- The number of cans collected by Sarah today -/
def sarah_today : ℕ := 40

/-- The number of cans collected by Lara today -/
def lara_today : ℕ := 70

/-- The number of cans collected by Alex today -/
def alex_today : ℕ := 55

/-- The total number of cans collected yesterday -/
def total_yesterday : ℕ := sarah_yesterday + lara_yesterday + alex_yesterday

/-- The total number of cans collected today -/
def total_today : ℕ := sarah_today + lara_today + alex_today

theorem cans_difference : total_yesterday - total_today = 55 := by
  sorry

end NUMINAMATH_CALUDE_cans_difference_l2570_257082


namespace NUMINAMATH_CALUDE_fraction_equality_l2570_257034

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2570_257034


namespace NUMINAMATH_CALUDE_keith_total_expenses_l2570_257083

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

theorem keith_total_expenses : 
  speakers_cost + cd_player_cost + tires_cost = 387.85 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_expenses_l2570_257083


namespace NUMINAMATH_CALUDE_cody_book_series_l2570_257015

/-- The number of books in Cody's favorite book series -/
def books_in_series (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  first_week + second_week + (subsequent_weeks * (total_weeks - 2))

/-- Theorem stating the number of books in Cody's series -/
theorem cody_book_series : books_in_series 6 3 9 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cody_book_series_l2570_257015


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2570_257050

theorem complex_modulus_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : x + 3 * i = 2 + y * i) : Complex.abs (x + y * i) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2570_257050


namespace NUMINAMATH_CALUDE_olympic_mascots_arrangement_l2570_257063

/-- The number of possible arrangements of 5 items with specific constraints -/
def num_arrangements : ℕ := 16

/-- The number of ways to choose 1 item from 2 -/
def choose_one_from_two : ℕ := 2

/-- The number of ways to arrange 2 items -/
def arrange_two : ℕ := 2

theorem olympic_mascots_arrangement :
  num_arrangements = 2 * choose_one_from_two * choose_one_from_two * arrange_two :=
sorry

end NUMINAMATH_CALUDE_olympic_mascots_arrangement_l2570_257063


namespace NUMINAMATH_CALUDE_shortest_side_length_l2570_257074

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one side of the triangle -/
  side : ℝ
  /-- The length of the first segment of the side divided by the point of tangency -/
  segment1 : ℝ
  /-- The length of the second segment of the side divided by the point of tangency -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
  (h1 : t.r = 5)
  (h2 : t.segment1 = 7)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), 
    shortest_side = 10 ∧ 
    (∀ (other_side : ℝ), other_side = t.side ∨ other_side ≥ shortest_side) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_length_l2570_257074


namespace NUMINAMATH_CALUDE_students_passed_test_l2570_257080

/-- Represents the result of a proficiency test -/
structure TestResult where
  total_students : ℕ
  passing_score : ℕ
  passed_students : ℕ

/-- The proficiency test result for the university -/
def university_test : TestResult :=
  { total_students := 1000
  , passing_score := 70
  , passed_students := 600 }

/-- Theorem stating the number of students who passed the test -/
theorem students_passed_test : university_test.passed_students = 600 := by
  sorry

#check students_passed_test

end NUMINAMATH_CALUDE_students_passed_test_l2570_257080


namespace NUMINAMATH_CALUDE_total_payment_l2570_257045

/-- The cost of potatoes in yuan per kilogram -/
def potato_cost : ℝ := 1

/-- The cost of celery in yuan per kilogram -/
def celery_cost : ℝ := 0.7

/-- The total cost of buying potatoes and celery -/
def total_cost (a b : ℝ) : ℝ := a * potato_cost + b * celery_cost

theorem total_payment (a b : ℝ) : total_cost a b = a + 0.7 * b := by
  sorry

end NUMINAMATH_CALUDE_total_payment_l2570_257045


namespace NUMINAMATH_CALUDE_draw_three_one_probability_l2570_257081

/-- The probability of drawing exactly 3 balls of one color and 1 of the other color
    from a bin containing 10 black balls and 8 white balls, when 4 balls are drawn at random -/
theorem draw_three_one_probability (black_balls : ℕ) (white_balls : ℕ) (total_draw : ℕ) :
  black_balls = 10 →
  white_balls = 8 →
  total_draw = 4 →
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) /
  Nat.choose (black_balls + white_balls) total_draw = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_draw_three_one_probability_l2570_257081


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l2570_257073

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (n % 18 = 7) ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 18 = 7 → m ≥ n) ∧
    n = 10015 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l2570_257073


namespace NUMINAMATH_CALUDE_basketball_spectators_ratio_l2570_257065

/-- Given the number of spectators at a basketball match, prove the ratio of children to women -/
theorem basketball_spectators_ratio
  (total : ℕ)
  (men : ℕ)
  (children : ℕ)
  (h_total : total = 10000)
  (h_men : men = 7000)
  (h_children : children = 2500)
  : (children : ℚ) / (total - men - children) = 5 := by
  sorry

end NUMINAMATH_CALUDE_basketball_spectators_ratio_l2570_257065


namespace NUMINAMATH_CALUDE_roots_properties_l2570_257084

theorem roots_properties (x₁ x₂ : ℝ) (h : x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) : 
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l2570_257084


namespace NUMINAMATH_CALUDE_complex_fourth_power_equality_implies_ratio_one_l2570_257094

theorem complex_fourth_power_equality_implies_ratio_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_equality_implies_ratio_one_l2570_257094


namespace NUMINAMATH_CALUDE_sandbag_weight_l2570_257031

theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  bag_capacity = 250 →
  fill_percentage = 0.8 →
  weight_increase = 0.4 →
  (bag_capacity * fill_percentage * (1 + weight_increase)) = 280 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_l2570_257031


namespace NUMINAMATH_CALUDE_min_boys_is_two_l2570_257018

/-- Represents the number of apples collected by a boy -/
inductive AppleCount
  | fixed : AppleCount  -- Represents 20 apples
  | percentage : AppleCount  -- Represents 20% of the total

/-- Represents a group of boys collecting apples -/
structure AppleCollection where
  boys : ℕ  -- Number of boys
  fixed_count : ℕ  -- Number of boys collecting fixed amount
  percentage_count : ℕ  -- Number of boys collecting percentage
  total_apples : ℕ  -- Total number of apples collected

/-- Checks if an AppleCollection is valid according to the problem conditions -/
def is_valid_collection (c : AppleCollection) : Prop :=
  c.boys = c.fixed_count + c.percentage_count ∧
  c.fixed_count > 0 ∧
  c.percentage_count > 0 ∧
  c.total_apples = 20 * c.fixed_count + (c.total_apples / 5) * c.percentage_count

/-- The main theorem stating that 2 is the minimum number of boys -/
theorem min_boys_is_two :
  ∀ c : AppleCollection, is_valid_collection c → c.boys ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_boys_is_two_l2570_257018


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2570_257068

theorem min_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                      2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                      (x1 - x2)^2 + (y1 - y2)^2 = 16) →
  (4/a + 1/b ≥ 9 ∧ ∃ a0 b0 : ℝ, a0 > 0 ∧ b0 > 0 ∧ 4/a0 + 1/b0 = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2570_257068


namespace NUMINAMATH_CALUDE_gumdrop_purchase_l2570_257091

theorem gumdrop_purchase (total_cents : ℕ) (cost_per_gumdrop : ℕ) (max_gumdrops : ℕ) : 
  total_cents = 224 → cost_per_gumdrop = 8 → max_gumdrops = total_cents / cost_per_gumdrop → max_gumdrops = 28 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_purchase_l2570_257091


namespace NUMINAMATH_CALUDE_total_clothes_washed_l2570_257025

/-- The total number of clothes washed by Cally, Danny, and Emily -/
theorem total_clothes_washed (
  cally_white_shirts cally_colored_shirts cally_shorts cally_pants cally_jackets : ℕ)
  (danny_white_shirts danny_colored_shirts danny_shorts danny_pants danny_jackets : ℕ)
  (emily_white_shirts emily_colored_shirts emily_shorts emily_pants emily_jackets : ℕ)
  (cally_danny_socks emily_danny_socks : ℕ)
  (h1 : cally_white_shirts = 10)
  (h2 : cally_colored_shirts = 5)
  (h3 : cally_shorts = 7)
  (h4 : cally_pants = 6)
  (h5 : cally_jackets = 3)
  (h6 : danny_white_shirts = 6)
  (h7 : danny_colored_shirts = 8)
  (h8 : danny_shorts = 10)
  (h9 : danny_pants = 6)
  (h10 : danny_jackets = 4)
  (h11 : emily_white_shirts = 8)
  (h12 : emily_colored_shirts = 6)
  (h13 : emily_shorts = 9)
  (h14 : emily_pants = 5)
  (h15 : emily_jackets = 2)
  (h16 : cally_danny_socks = 3)
  (h17 : emily_danny_socks = 2) :
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + cally_jackets +
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants + danny_jackets +
  emily_white_shirts + emily_colored_shirts + emily_shorts + emily_pants + emily_jackets +
  cally_danny_socks + emily_danny_socks = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_washed_l2570_257025


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2570_257020

theorem quadratic_equation_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ + (m - 1) = 0 ∧ x₂^2 - m*x₂ + (m - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2570_257020


namespace NUMINAMATH_CALUDE_conor_eggplants_per_day_l2570_257046

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := sorry

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := 8

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conor_eggplants_per_day :
  eggplants_per_day = 12 :=
by sorry

end NUMINAMATH_CALUDE_conor_eggplants_per_day_l2570_257046


namespace NUMINAMATH_CALUDE_jerry_birthday_money_weighted_mean_l2570_257040

-- Define the exchange rates
def euro_to_usd : ℝ := 1.20
def gbp_to_usd : ℝ := 1.38

-- Define the weighted percentages
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Define the money received from family members in USD
def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9 * euro_to_usd
def sister_gift : ℝ := 7

-- Define the money received from friends in USD
def friends_gifts : List ℝ := [22, 23, 18 * euro_to_usd, 15 * gbp_to_usd, 22]

-- Calculate total family gifts
def family_total : ℝ := aunt_gift + uncle_gift + sister_gift

-- Calculate total friends gifts
def friends_total : ℝ := friends_gifts.sum

-- Define the weighted mean calculation
def weighted_mean : ℝ := family_total * family_weight + friends_total * friends_weight

-- Theorem to prove
theorem jerry_birthday_money_weighted_mean :
  weighted_mean = 76.30 := by sorry

end NUMINAMATH_CALUDE_jerry_birthday_money_weighted_mean_l2570_257040


namespace NUMINAMATH_CALUDE_insurance_compensation_l2570_257011

/-- Insurance compensation calculation --/
theorem insurance_compensation
  (insured_amount : ℝ)
  (deductible_percentage : ℝ)
  (actual_damage : ℝ)
  (h1 : insured_amount = 500000)
  (h2 : deductible_percentage = 0.01)
  (h3 : actual_damage = 4000)
  : min (max (actual_damage - insured_amount * deductible_percentage) 0) insured_amount = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_insurance_compensation_l2570_257011


namespace NUMINAMATH_CALUDE_rajan_investment_is_20000_l2570_257022

/-- Represents the investment scenario with Rajan, Rakesh, and Mukesh --/
structure InvestmentScenario where
  rajan_investment : ℕ
  rakesh_investment : ℕ
  mukesh_investment : ℕ
  total_profit : ℕ
  rajan_profit : ℕ

/-- The investment scenario satisfies the given conditions --/
def satisfies_conditions (scenario : InvestmentScenario) : Prop :=
  scenario.rakesh_investment = 25000 ∧
  scenario.mukesh_investment = 15000 ∧
  scenario.total_profit = 4600 ∧
  scenario.rajan_profit = 2400 ∧
  (scenario.rajan_investment * 12 : ℚ) / 
    (scenario.rajan_investment * 12 + scenario.rakesh_investment * 4 + scenario.mukesh_investment * 8) = 
    (scenario.rajan_profit : ℚ) / scenario.total_profit

/-- Theorem stating that if the scenario satisfies the conditions, Rajan's investment is 20000 --/
theorem rajan_investment_is_20000 (scenario : InvestmentScenario) :
  satisfies_conditions scenario → scenario.rajan_investment = 20000 := by
  sorry

#check rajan_investment_is_20000

end NUMINAMATH_CALUDE_rajan_investment_is_20000_l2570_257022


namespace NUMINAMATH_CALUDE_markup_percentage_is_20_l2570_257002

/-- Calculate the markup percentage given cost price, discount, and profit percentage --/
def calculate_markup_percentage (cost_price discount : ℕ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price + (cost_price * profit_percentage / 100)
  let marked_price := selling_price + discount
  (marked_price - cost_price) / cost_price * 100

/-- Theorem stating that the markup percentage is 20% given the specified conditions --/
theorem markup_percentage_is_20 :
  calculate_markup_percentage 180 50 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_is_20_l2570_257002


namespace NUMINAMATH_CALUDE_dilative_rotation_commutes_l2570_257053

/-- A transformation consisting of a rotation and scaling -/
structure DilativeRotation where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Apply a dilative rotation to a point -/
def applyDilativeRotation (t : DilativeRotation) (p : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Apply a dilative rotation to a triangle -/
def applyDilativeRotationToTriangle (t : DilativeRotation) (tri : Triangle) : Triangle :=
  sorry

/-- Theorem stating that the order of rotation and scaling is interchangeable -/
theorem dilative_rotation_commutes (t : DilativeRotation) (tri : Triangle) :
  let t1 := DilativeRotation.mk t.center t.angle 1
  let t2 := DilativeRotation.mk t.center 0 t.scale
  applyDilativeRotationToTriangle t2 (applyDilativeRotationToTriangle t1 tri) =
  applyDilativeRotationToTriangle t1 (applyDilativeRotationToTriangle t2 tri) :=
  sorry

end NUMINAMATH_CALUDE_dilative_rotation_commutes_l2570_257053


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2570_257047

theorem fraction_multiplication (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (b * c / a^2) * (a / b^2) = c / (a * b) := by
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2570_257047


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l2570_257058

theorem simple_interest_rate_problem (P A T : ℕ) (h1 : P = 25000) (h2 : A = 35500) (h3 : T = 12) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 35 / 10 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l2570_257058


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2570_257090

/-- Given a hyperbola with equation 9y² - 25x² = 169, 
    its asymptotes are given by the equation y = ± (5/3)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 →
  ∃ (k : ℝ), k = 5/3 ∧ (y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2570_257090


namespace NUMINAMATH_CALUDE_greatest_real_part_sixth_power_l2570_257067

theorem greatest_real_part_sixth_power :
  let z₁ : ℂ := -3
  let z₂ : ℂ := -2 * Real.sqrt 2 + (2 * Real.sqrt 2) * I
  let z₃ : ℂ := -Real.sqrt 3 + Real.sqrt 3 * I
  let z₄ : ℂ := -1 + 2 * Real.sqrt 3 * I
  let z₅ : ℂ := 3 * I
  (Complex.re (z₁^6) ≥ Complex.re (z₂^6)) ∧
  (Complex.re (z₁^6) ≥ Complex.re (z₃^6)) ∧
  (Complex.re (z₁^6) ≥ Complex.re (z₄^6)) ∧
  (Complex.re (z₁^6) ≥ Complex.re (z₅^6)) :=
by
  sorry

#check greatest_real_part_sixth_power

end NUMINAMATH_CALUDE_greatest_real_part_sixth_power_l2570_257067


namespace NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l2570_257033

/-- An ellipse with center at the origin and foci on the coordinate axes -/
structure CenteredEllipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The equation of the ellipse -/
def CenteredEllipse.equation (e : CenteredEllipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The ellipse passes through the given points -/
def CenteredEllipse.passes_through (e : CenteredEllipse) : Prop :=
  e.equation 2 (Real.sqrt 2) ∧ e.equation (Real.sqrt 6) 1

/-- The main theorem -/
theorem ellipse_and_circle_theorem (e : CenteredEllipse) 
    (h_passes : e.passes_through) : 
    (e.a^2 = 8 ∧ e.b^2 = 4) ∧
    ∃ (r : ℝ), r^2 = 8/3 ∧
      ∀ (l : ℝ → ℝ → Prop), 
        (∃ (k m : ℝ), ∀ x y, l x y ↔ y = k * x + m) →
        (∃ x y, x^2 + y^2 = r^2 ∧ l x y) →
        ∃ (A B : ℝ × ℝ), 
          A ≠ B ∧
          e.equation A.1 A.2 ∧ 
          e.equation B.1 B.2 ∧
          l A.1 A.2 ∧ 
          l B.1 B.2 ∧
          A.1 * B.1 + A.2 * B.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l2570_257033


namespace NUMINAMATH_CALUDE_range_of_f_l2570_257057

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2570_257057


namespace NUMINAMATH_CALUDE_largest_equal_cost_integer_l2570_257014

/-- Calculates the sum of digits for a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the cost of binary representation -/
def binaryCost (n : ℕ) : ℕ := sorry

/-- Theorem stating that 311 is the largest integer less than 500 with equal costs -/
theorem largest_equal_cost_integer :
  ∀ n : ℕ, n < 500 → n > 311 → sumOfDigits n ≠ binaryCost n ∧
  sumOfDigits 311 = binaryCost 311 := by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_integer_l2570_257014


namespace NUMINAMATH_CALUDE_isosceles_triangle_yw_length_l2570_257038

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  dist t.X t.Z = dist t.Y t.Z

-- Define the point W on XZ
def W (t : Triangle) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem isosceles_triangle_yw_length 
  (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : dist t.X t.Y = 3) 
  (h3 : dist t.X t.Z = 5) 
  (h4 : dist t.Y t.Z = 5) 
  (h5 : dist (W t) t.Z = 2) : 
  dist (W t) t.Y = Real.sqrt 18.5 := 
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_yw_length_l2570_257038


namespace NUMINAMATH_CALUDE_find_number_l2570_257098

theorem find_number (x : ℝ) : ((x - 1.9) * 1.5 + 32) / 2.5 = 20 → x = 13.9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2570_257098


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2570_257048

/-- Represents a bag of marbles with counts for different colors -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the probability of drawing a yellow marble as the last marble
    given the contents of bags A, B, C, and D and the described drawing process -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black
  let totalB := bagB.yellow + bagB.blue
  let totalC := bagC.yellow + bagC.blue
  let totalD := bagD.yellow + bagD.blue
  
  let probWhiteA := bagA.white / totalA
  let probBlackA := bagA.black / totalA
  let probYellowB := bagB.yellow / totalB
  let probBlueC := bagC.blue / totalC
  let probYellowC := bagC.yellow / totalC
  let probYellowD := bagD.yellow / totalD
  
  probWhiteA * probYellowB + 
  probBlackA * probBlueC * probYellowD + 
  probBlackA * probYellowC

theorem yellow_marble_probability :
  let bagA : Bag := { white := 5, black := 6 }
  let bagB : Bag := { yellow := 8, blue := 6 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 1, blue := 4 }
  yellowProbability bagA bagB bagC bagD = 136 / 275 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l2570_257048


namespace NUMINAMATH_CALUDE_sphere_and_cylinder_properties_l2570_257043

/-- Given a sphere with volume 72π cubic inches, prove its surface area and the radius of a cylinder with the same volume and height 4 inches. -/
theorem sphere_and_cylinder_properties :
  ∃ (r : ℝ), 
    (4 / 3 * π * r^3 = 72 * π) ∧ 
    (4 * π * r^2 = 36 * 2^(2/3) * π) ∧
    ∃ (r_cyl : ℝ), 
      (π * r_cyl^2 * 4 = 72 * π) ∧ 
      (r_cyl = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_and_cylinder_properties_l2570_257043


namespace NUMINAMATH_CALUDE_total_length_of_T_l2570_257026

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 2‖ + ‖‖|p.2| - 3‖ - 2‖ = 2}

-- Define the total length of lines in T
def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : total_length T = 128 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l2570_257026


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l2570_257069

theorem unique_triplet_solution :
  ∀ (x y ℓ : ℕ), x^3 + y^3 - 53 = 7^ℓ ↔ x = 3 ∧ y = 3 ∧ ℓ = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l2570_257069


namespace NUMINAMATH_CALUDE_third_player_win_probability_probability_third_player_wins_l2570_257037

/-- The probability of winning for the third player in a four-player 
    coin flipping game where players take turns and the first to flip 
    heads wins. -/
theorem third_player_win_probability : ℝ :=
  2 / 63

/-- The game ends when a player flips heads -/
axiom game_ends_on_heads : Prop

/-- There are four players taking turns -/
axiom four_players : Prop

/-- Players take turns in order -/
axiom turns_in_order : Prop

/-- Each flip has a 1/2 probability of heads -/
axiom fair_coin : Prop

theorem probability_third_player_wins : 
  game_ends_on_heads → four_players → turns_in_order → fair_coin →
  third_player_win_probability = 2 / 63 :=
sorry

end NUMINAMATH_CALUDE_third_player_win_probability_probability_third_player_wins_l2570_257037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2570_257088

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = 10 →
  a 4 = a 3 + 2 →
  a 3 + a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2570_257088


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2570_257054

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_corners rp + num_faces rp = 26 := by
  sorry

#check rectangular_prism_sum

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2570_257054


namespace NUMINAMATH_CALUDE_probability_equals_three_over_646_l2570_257042

-- Define the cube
def cube_side_length : ℕ := 5
def total_cubes : ℕ := cube_side_length ^ 3

-- Define the number of cubes with different numbers of painted faces
def cubes_with_three_painted_faces : ℕ := 1
def cubes_with_one_painted_face : ℕ := 36

-- Define the probability calculation function
def probability_one_three_one_face : ℚ :=
  (cubes_with_three_painted_faces * cubes_with_one_painted_face : ℚ) /
  (total_cubes * (total_cubes - 1) / 2)

-- The theorem to prove
theorem probability_equals_three_over_646 :
  probability_one_three_one_face = 3 / 646 := by
  sorry

end NUMINAMATH_CALUDE_probability_equals_three_over_646_l2570_257042


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2570_257077

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2570_257077


namespace NUMINAMATH_CALUDE_timothy_movie_count_l2570_257056

theorem timothy_movie_count (timothy_prev : ℕ) 
  (h1 : timothy_prev + (timothy_prev + 7) + 2 * (timothy_prev + 7) + timothy_prev / 2 = 129) : 
  timothy_prev = 24 := by
  sorry

end NUMINAMATH_CALUDE_timothy_movie_count_l2570_257056


namespace NUMINAMATH_CALUDE_fraction_product_squares_l2570_257028

theorem fraction_product_squares : 
  (4/5)^2 * (3/7)^2 * (2/3)^2 = 64/1225 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_squares_l2570_257028


namespace NUMINAMATH_CALUDE_store_earnings_theorem_l2570_257023

/-- Represents the earnings from selling bottled drinks in a country store. -/
def store_earnings (cola_price juice_price water_price : ℚ) 
                   (cola_sold juice_sold water_sold : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold

/-- Theorem stating that the store earned $88 from selling bottled drinks. -/
theorem store_earnings_theorem : 
  store_earnings 3 1.5 1 15 12 25 = 88 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_theorem_l2570_257023


namespace NUMINAMATH_CALUDE_initial_books_correct_l2570_257095

/-- The number of books initially in the pile to be put away. -/
def initial_books : ℝ := 46.0

/-- The number of books added by the librarian. -/
def added_books : ℝ := 10.0

/-- The number of books that can fit on each shelf. -/
def books_per_shelf : ℝ := 4.0

/-- The number of shelves needed to arrange all books. -/
def shelves_needed : ℕ := 14

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = (books_per_shelf * shelves_needed : ℝ) - added_books :=
by sorry

end NUMINAMATH_CALUDE_initial_books_correct_l2570_257095


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l2570_257010

-- Define the number of options for each food category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats * (Nat.choose num_vegetables vegetables_to_choose) * num_desserts) = 120 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l2570_257010


namespace NUMINAMATH_CALUDE_distance_proof_l2570_257099

theorem distance_proof (v1 v2 : ℝ) : 
  (5 * v1 + 5 * v2 = 30) →
  (3 * (v1 + 2) + 3 * (v2 + 2) = 30) →
  30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l2570_257099


namespace NUMINAMATH_CALUDE_sine_product_of_half_angles_less_than_quarter_l2570_257000

theorem sine_product_of_half_angles_less_than_quarter (A B C : Real) :
  (A + B + C = Real.pi) →
  (A > 0) → (B > 0) → (C > 0) →
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_of_half_angles_less_than_quarter_l2570_257000


namespace NUMINAMATH_CALUDE_divisibility_relation_l2570_257030

theorem divisibility_relation (p q r s : ℤ) 
  (h_s : s % 5 ≠ 0)
  (h_a : ∃ a : ℤ, (p * a^3 + q * a^2 + r * a + s) % 5 = 0) :
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2570_257030


namespace NUMINAMATH_CALUDE_set_a_values_l2570_257001

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem set_a_values (a : ℝ) : A ∪ B a = A ↔ a = -2 ∨ a ≥ 4 ∨ a < -4 := by
  sorry

end NUMINAMATH_CALUDE_set_a_values_l2570_257001


namespace NUMINAMATH_CALUDE_absolute_difference_equals_one_l2570_257089

theorem absolute_difference_equals_one (x y : ℝ) :
  |x| - |y| = 1 ↔
  ((y = x - 1 ∧ x ≥ 1) ∨
   (y = 1 - x ∧ x ≥ 1) ∨
   (y = -x - 1 ∧ x ≤ -1) ∨
   (y = x + 1 ∧ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_equals_one_l2570_257089


namespace NUMINAMATH_CALUDE_product_11_4_sum_144_l2570_257029

theorem product_11_4_sum_144 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 11^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 144 :=
by sorry

end NUMINAMATH_CALUDE_product_11_4_sum_144_l2570_257029


namespace NUMINAMATH_CALUDE_circulation_ratio_l2570_257086

/-- Represents the circulation of a magazine over time -/
structure MagazineCirculation where
  /-- Circulation in 1962 -/
  C_1962 : ℝ
  /-- Growth rate per year (as a decimal) -/
  r : ℝ
  /-- Average yearly circulation from 1962 to 1970 -/
  A : ℝ

/-- Theorem stating the ratio of circulation in 1961 to total circulation 1961-1970 -/
theorem circulation_ratio (P : MagazineCirculation) :
  /- Circulation in 1961 is 4 times the average from 1962-1970 -/
  (4 * P.A) / 
  /- Total circulation from 1961-1970 -/
  (4 * P.A + 9 * P.A) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_circulation_ratio_l2570_257086


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2570_257075

theorem value_of_a_minus_b (a b c : ℝ) 
  (h1 : a - (b - 2*c) = 19) 
  (h2 : a - b - 2*c = 7) : 
  a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2570_257075


namespace NUMINAMATH_CALUDE_expression_evaluation_l2570_257064

theorem expression_evaluation (x : ℤ) 
  (h1 : 1 - x > (-1 - x) / 2) 
  (h2 : x + 1 > 0) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 0) : 
  (1 + (3*x - 1) / (x + 1)) / (x / (x^2 - 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2570_257064


namespace NUMINAMATH_CALUDE_binomial_coefficients_10_l2570_257004

theorem binomial_coefficients_10 : (Nat.choose 10 10 = 1) ∧ (Nat.choose 10 9 = 10) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficients_10_l2570_257004


namespace NUMINAMATH_CALUDE_characterize_function_l2570_257041

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem characterize_function (f : RealFunction) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_characterize_function_l2570_257041


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2570_257072

def expression (x : ℝ) : ℝ := 6 * (x - 2 * x^3) - 5 * (2 * x^2 - 3 * x^3 + 2 * x^4) + 3 * (3 * x^2 - 2 * x^6)

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ), 
    (∀ x, expression x = a * x + (-1) * x^2 + c * x^3 + d * x^4 + e * x^6 + f) :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2570_257072


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2570_257061

theorem arithmetic_operations : 
  (24 - (-16) + (-25) - 32 = -17) ∧
  ((-1/2) * 2 / 2 * (-1/2) = 1/4) ∧
  (-2^2 * 5 - (-2)^3 * (1/8) + 1 = -18) ∧
  ((-1/4 - 5/6 + 8/9) / (-1/6)^2 + (-2)^2 * (-6) = -31) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2570_257061


namespace NUMINAMATH_CALUDE_yellow_marble_fraction_l2570_257013

theorem yellow_marble_fraction (n : ℝ) (h : n > 0) : 
  let initial_green := (2/3) * n
  let initial_yellow := n - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end NUMINAMATH_CALUDE_yellow_marble_fraction_l2570_257013


namespace NUMINAMATH_CALUDE_total_courses_is_200_l2570_257087

/-- The number of college courses Max attended -/
def max_courses : ℕ := 40

/-- The number of college courses Sid attended relative to Max -/
def sid_multiplier : ℕ := 4

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_multiplier * max_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_courses_is_200_l2570_257087


namespace NUMINAMATH_CALUDE_difference_of_squares_l2570_257007

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2570_257007


namespace NUMINAMATH_CALUDE_symmetrical_lines_and_ellipse_intersection_l2570_257017

/-- Given two lines symmetrical about y = x + 1, prove their slopes multiply to 1 and their intersection points with an ellipse form a line passing through a fixed point. -/
theorem symmetrical_lines_and_ellipse_intersection
  (k : ℝ) (h_k_pos : k > 0) (h_k_neq_one : k ≠ 1)
  (l : Set (ℝ × ℝ)) (l_eq : l = {(x, y) | y = k * x + 1})
  (l₁ : Set (ℝ × ℝ)) (k₁ : ℝ) (l₁_eq : l₁ = {(x, y) | y = k₁ * x + 1})
  (h_symmetry : ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - 1, x + 1) ∈ l₁)
  (E : Set (ℝ × ℝ)) (E_eq : E = {(x, y) | x^2 / 4 + y^2 = 1})
  (A M : ℝ × ℝ) (h_AM : A ∈ E ∧ M ∈ E ∧ A ∈ l ∧ M ∈ l ∧ A ≠ M)
  (N : ℝ × ℝ) (h_AN : A ∈ E ∧ N ∈ E ∧ A ∈ l₁ ∧ N ∈ l₁ ∧ A ≠ N) :
  (k * k₁ = 1) ∧
  (∃ (m b : ℝ), ∀ (x : ℝ), M.2 - N.2 = m * (M.1 - N.1) ∧ N.2 = m * N.1 + b ∧ b = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_lines_and_ellipse_intersection_l2570_257017


namespace NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l2570_257062

-- Define the condition for m
def m_condition (m : ℝ) : Prop := 2 < m ∧ m < 6

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop := m > 2 ∧ m < 6 ∧ m ≠ 4

-- Theorem stating that m_condition is necessary but not sufficient for is_ellipse
theorem m_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → m_condition m) ∧
  ¬(∀ m : ℝ, m_condition m → is_ellipse m) := by sorry

end NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l2570_257062


namespace NUMINAMATH_CALUDE_kerosene_mixture_problem_l2570_257016

theorem kerosene_mixture_problem (first_liquid_percentage : ℝ) 
  (first_liquid_parts : ℝ) (second_liquid_parts : ℝ) (mixture_percentage : ℝ) :
  first_liquid_percentage = 25 →
  first_liquid_parts = 6 →
  second_liquid_parts = 4 →
  mixture_percentage = 27 →
  let total_parts := first_liquid_parts + second_liquid_parts
  let second_liquid_percentage := 
    (mixture_percentage * total_parts - first_liquid_percentage * first_liquid_parts) / second_liquid_parts
  second_liquid_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_mixture_problem_l2570_257016


namespace NUMINAMATH_CALUDE_factorization_equality_l2570_257035

theorem factorization_equality (m n : ℝ) : m^2 * n - 16 * n = n * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2570_257035


namespace NUMINAMATH_CALUDE_triangle_problem_l2570_257096

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (1/2 * b * c * Real.sin A = 10 * Real.sqrt 3) →  -- Area condition
  (a = 7) →  -- Given side length
  (Real.sin A)^2 = (Real.sin B)^2 + (Real.sin C)^2 - Real.sin B * Real.sin C →  -- Given equation
  (A = π/3 ∧ ((b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2570_257096


namespace NUMINAMATH_CALUDE_total_envelopes_l2570_257055

/-- The number of stamps needed for an envelope weighing more than 5 pounds -/
def heavy_envelope_stamps : ℕ := 5

/-- The number of stamps needed for an envelope weighing less than 5 pounds -/
def light_envelope_stamps : ℕ := 2

/-- The total number of stamps Micah bought -/
def total_stamps : ℕ := 52

/-- The number of envelopes weighing less than 5 pounds -/
def light_envelopes : ℕ := 6

/-- Theorem stating the total number of envelopes Micah bought -/
theorem total_envelopes : 
  ∃ (heavy_envelopes : ℕ), 
    light_envelopes * light_envelope_stamps + 
    heavy_envelopes * heavy_envelope_stamps = total_stamps ∧
    light_envelopes + heavy_envelopes = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_envelopes_l2570_257055


namespace NUMINAMATH_CALUDE_freshman_count_l2570_257060

theorem freshman_count (total : ℕ) (f s j r : ℕ) : 
  total = 2158 →
  5 * s = 4 * f →
  8 * s = 7 * j →
  7 * j = 9 * r →
  total = f + s + j + r →
  f = 630 := by
sorry

end NUMINAMATH_CALUDE_freshman_count_l2570_257060


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l2570_257059

/-- The profit percentage of a dishonest dealer who uses a weight of 600 grams per kg while selling at the professed cost price. -/
theorem dishonest_dealer_profit_percentage :
  let actual_weight : ℝ := 600  -- grams
  let claimed_weight : ℝ := 1000  -- grams (1 kg)
  let profit_ratio := (claimed_weight - actual_weight) / actual_weight
  profit_ratio * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l2570_257059


namespace NUMINAMATH_CALUDE_mark_kate_difference_l2570_257079

/-- Represents the project with three workers -/
structure Project where
  kate_hours : ℕ
  pat_hours : ℕ
  mark_hours : ℕ

/-- Conditions of the project -/
def valid_project (p : Project) : Prop :=
  p.pat_hours = 2 * p.kate_hours ∧
  p.mark_hours = p.kate_hours + 6 ∧
  p.kate_hours + p.pat_hours + p.mark_hours = 198

theorem mark_kate_difference (p : Project) (h : valid_project p) :
  p.mark_hours - p.kate_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l2570_257079


namespace NUMINAMATH_CALUDE_jose_age_l2570_257092

theorem jose_age (maria_age jose_age : ℕ) : 
  jose_age = maria_age + 12 →
  maria_age + jose_age = 40 →
  jose_age = 26 := by
sorry

end NUMINAMATH_CALUDE_jose_age_l2570_257092


namespace NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l2570_257097

/-- If positive numbers a, b, c form an arithmetic sequence with non-zero common difference,
    then their reciprocals cannot form an arithmetic sequence. -/
theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h_arith : ∃ d ≠ 0, b - a = d ∧ c - b = d) : 
    ¬∃ k : ℝ, (1 / b - 1 / a = k) ∧ (1 / c - 1 / b = k) := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l2570_257097


namespace NUMINAMATH_CALUDE_nights_with_new_habit_l2570_257003

/-- Represents the number of nights a candle lasts when burned for 1 hour per night -/
def initial_nights_per_candle : ℕ := 8

/-- Represents the number of hours Carmen burns a candle each night after changing her habit -/
def hours_per_night : ℕ := 2

/-- Represents the total number of candles Carmen uses -/
def total_candles : ℕ := 6

/-- Theorem stating the total number of nights Carmen can burn candles with the new habit -/
theorem nights_with_new_habit : 
  (total_candles * initial_nights_per_candle) / hours_per_night = 24 := by
  sorry

end NUMINAMATH_CALUDE_nights_with_new_habit_l2570_257003


namespace NUMINAMATH_CALUDE_rotating_squares_intersection_area_l2570_257051

/-- The area of intersection of two rotating unit squares after 5 minutes -/
theorem rotating_squares_intersection_area : 
  let revolution_rate : ℝ := 2 * Real.pi / 60 -- radians per minute
  let rotation_time : ℝ := 5 -- minutes
  let rotation_angle : ℝ := revolution_rate * rotation_time
  let intersection_area : ℝ := (1 - Real.cos rotation_angle) * (1 - Real.sin rotation_angle)
  intersection_area = (2 - Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_rotating_squares_intersection_area_l2570_257051


namespace NUMINAMATH_CALUDE_no_convex_polygon_with_1974_diagonals_l2570_257076

theorem no_convex_polygon_with_1974_diagonals :
  ¬ ∃ (N : ℕ), N > 0 ∧ N * (N - 3) / 2 = 1974 := by
  sorry

end NUMINAMATH_CALUDE_no_convex_polygon_with_1974_diagonals_l2570_257076


namespace NUMINAMATH_CALUDE_max_value_fraction_l2570_257036

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : b^2 + 2*(a + c)*b - a*c = 0) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → y^2 + 2*(x + z)*y - x*z = 0 → 
  y / (x + z) ≤ b / (a + c) → b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2570_257036


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2570_257005

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2570_257005


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2570_257066

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x + 10) = 90) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2570_257066


namespace NUMINAMATH_CALUDE_centroid_of_V_l2570_257049

-- Define the region V
def V : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a region
def centroid (S : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem centroid_of_V :
  centroid V = (0, 2.31) := by
  sorry

end NUMINAMATH_CALUDE_centroid_of_V_l2570_257049
