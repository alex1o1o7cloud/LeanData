import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_abc_l365_36503

theorem average_marks_abc (M : ℝ) (D : ℝ) :
  -- The average marks of a, b, c is M
  -- When d joins, the average becomes 47
  3 * M + D = 4 * 47 →
  -- The average marks of b, c, d, e is 48
  -- E has 3 more marks than d
  -- The marks of a is 43
  (3 * M - 43) + D + (D + 3) = 4 * 48 →
  -- The average marks of a, b, c is 48
  M = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_abc_l365_36503


namespace NUMINAMATH_CALUDE_distinct_number_probability_l365_36548

-- Define the number of balls of each color and the number to be selected
def total_red_balls : ℕ := 5
def total_black_balls : ℕ := 5
def balls_to_select : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := total_red_balls + total_black_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_number_probability :
  (binomial total_balls balls_to_select : ℚ) ≠ 0 →
  (binomial total_red_balls balls_to_select * 2^balls_to_select : ℚ) /
  (binomial total_balls balls_to_select : ℚ) = 8/21 := by sorry

end NUMINAMATH_CALUDE_distinct_number_probability_l365_36548


namespace NUMINAMATH_CALUDE_stratified_sampling_first_grade_l365_36564

theorem stratified_sampling_first_grade (total_students : ℕ) (sampled_students : ℕ) (first_grade_students : ℕ) :
  total_students = 2400 →
  sampled_students = 100 →
  first_grade_students = 840 →
  (first_grade_students * sampled_students) / total_students = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_first_grade_l365_36564


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l365_36520

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l365_36520


namespace NUMINAMATH_CALUDE_max_trailing_zeros_1003_l365_36542

/-- Three natural numbers whose sum is 1003 -/
def SumTo1003 (a b c : ℕ) : Prop := a + b + c = 1003

/-- The number of trailing zeros in a natural number -/
def TrailingZeros (n : ℕ) : ℕ := sorry

/-- The product of three natural numbers -/
def ProductOfThree (a b c : ℕ) : ℕ := a * b * c

/-- Theorem stating that the maximum number of trailing zeros in the product of three natural numbers summing to 1003 is 7 -/
theorem max_trailing_zeros_1003 :
  ∀ a b c : ℕ, SumTo1003 a b c →
  ∀ n : ℕ, n = TrailingZeros (ProductOfThree a b c) →
  n ≤ 7 ∧ ∃ x y z : ℕ, SumTo1003 x y z ∧ TrailingZeros (ProductOfThree x y z) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_1003_l365_36542


namespace NUMINAMATH_CALUDE_shortest_perpendicular_best_measurement_l365_36510

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a long jump measurement method -/
inductive LongJumpMeasurement
  | Vertical
  | ShortestLineSegment
  | TwoPointLine
  | ShortestPerpendicular

/-- Defines the accuracy of a measurement method -/
def isAccurate (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Defines the consistency of a measurement method -/
def isConsistent (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Theorem: The shortest perpendicular line segment is the most accurate and consistent method for measuring long jump performance -/
theorem shortest_perpendicular_best_measurement :
  ∀ (method : LongJumpMeasurement),
    isAccurate method ∧ isConsistent method ↔ method = LongJumpMeasurement.ShortestPerpendicular :=
by sorry

end NUMINAMATH_CALUDE_shortest_perpendicular_best_measurement_l365_36510


namespace NUMINAMATH_CALUDE_canal_length_scientific_notation_l365_36544

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The length of the Beijing-Hangzhou Grand Canal in meters -/
def canal_length : ℕ := 1790000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem canal_length_scientific_notation :
  to_scientific_notation canal_length = ScientificNotation.mk 1.79 6 :=
sorry

end NUMINAMATH_CALUDE_canal_length_scientific_notation_l365_36544


namespace NUMINAMATH_CALUDE_loss_equals_twenty_pencils_l365_36565

/-- The number of pencils purchased -/
def num_pencils : ℕ := 70

/-- The ratio of cost to selling price for the total purchase -/
def cost_to_sell_ratio : ℚ := 1.2857142857142856

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem loss_equals_twenty_pencils :
  ∀ (cost_per_pencil sell_per_pencil : ℚ),
  cost_per_pencil = cost_to_sell_ratio * sell_per_pencil →
  (num_pencils : ℚ) * (cost_per_pencil - sell_per_pencil) = (loss_in_pencils : ℚ) * sell_per_pencil :=
by sorry

end NUMINAMATH_CALUDE_loss_equals_twenty_pencils_l365_36565


namespace NUMINAMATH_CALUDE_carlsons_original_land_size_l365_36561

/-- Calculates the size of Carlson's original land given the cost and area of new land purchases --/
theorem carlsons_original_land_size
  (cost_land1 : ℝ)
  (cost_land2 : ℝ)
  (cost_per_sqm : ℝ)
  (total_area_after : ℝ)
  (h1 : cost_land1 = 8000)
  (h2 : cost_land2 = 4000)
  (h3 : cost_per_sqm = 20)
  (h4 : total_area_after = 900) :
  total_area_after - (cost_land1 + cost_land2) / cost_per_sqm = 300 :=
by sorry

end NUMINAMATH_CALUDE_carlsons_original_land_size_l365_36561


namespace NUMINAMATH_CALUDE_inequality_proof_l365_36511

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : 1/m < 1/n) : m < 0 ∧ 0 < n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l365_36511


namespace NUMINAMATH_CALUDE_blue_box_contains_70_blueberries_l365_36528

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 30

/-- The increase in difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 100

theorem blue_box_contains_70_blueberries :
  (strawberries - blueberries = total_increase) ∧
  (strawberries = difference_increase) →
  blueberries = 70 := by sorry

end NUMINAMATH_CALUDE_blue_box_contains_70_blueberries_l365_36528


namespace NUMINAMATH_CALUDE_f_minimum_l365_36554

def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem f_minimum :
  (∀ x : ℝ, f x ≥ -9) ∧ (∃ x : ℝ, f x = -9) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_l365_36554


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l365_36515

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l365_36515


namespace NUMINAMATH_CALUDE_expected_attacked_squares_l365_36501

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of rooks placed on the chessboard -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
def expectedAttackedSquares : ℚ := chessboardSquares * (1 - probNotAttacked ^ numberOfRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_l365_36501


namespace NUMINAMATH_CALUDE_selling_price_articles_l365_36571

/-- Proves that if the cost price of 50 articles equals the selling price of N articles,
    and the gain percent is 25%, then N = 40. -/
theorem selling_price_articles (C : ℝ) (N : ℕ) (h1 : N * (C + 0.25 * C) = 50 * C) : N = 40 := by
  sorry

#check selling_price_articles

end NUMINAMATH_CALUDE_selling_price_articles_l365_36571


namespace NUMINAMATH_CALUDE_original_circle_area_l365_36530

/-- Given a circle whose area increases by 8 times and whose circumference
    increases by 50.24 centimeters, prove that its original area is 50.24 square centimeters. -/
theorem original_circle_area (r : ℝ) (h1 : r > 0) : 
  (π * (r + 50.24 / (2 * π))^2 = 9 * π * r^2) ∧ 
  (2 * π * (r + 50.24 / (2 * π)) = 2 * π * r + 50.24) → 
  π * r^2 = 50.24 := by
  sorry

end NUMINAMATH_CALUDE_original_circle_area_l365_36530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l365_36546

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a5 : a 5 = 15) : 
  a 2 + a 4 + a 6 + a 8 = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l365_36546


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l365_36549

open Complex

theorem complex_equation_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1)) ∧
    S.card = 8 ∧
    (∀ z : ℂ, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1) → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l365_36549


namespace NUMINAMATH_CALUDE_inequality_solution_l365_36559

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
axiom f_one_eq_zero : f 1 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f (x - 2) ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l365_36559


namespace NUMINAMATH_CALUDE_rectangular_field_area_l365_36590

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 13) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  a * b = 26 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l365_36590


namespace NUMINAMATH_CALUDE_max_value_range_l365_36595

noncomputable def f (a b x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else b * x - 1

theorem max_value_range (a b : ℝ) :
  ∃ M : ℝ, (∀ x, f a b x ≤ M) ∧ (0 < M) ∧ (M ≤ Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_range_l365_36595


namespace NUMINAMATH_CALUDE_x_value_l365_36596

/-- The value of x is equal to (47% of 1442 - 36% of 1412) + 65 -/
theorem x_value : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l365_36596


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_medians_l365_36584

/-- A right triangle with specific median properties -/
structure RightTriangleWithMedians where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The medians from acute angles are both 6
  median_a : a^2 + (b/2)^2 = 36
  median_b : b^2 + (a/2)^2 = 36
  -- Ensure positivity of sides
  a_pos : a > 0
  b_pos : b > 0

/-- The hypotenuse of the right triangle with the given median properties is 2√57.6 -/
theorem hypotenuse_of_right_triangle_with_medians (t : RightTriangleWithMedians) :
  Real.sqrt ((2*t.a)^2 + (2*t.b)^2) = 2 * Real.sqrt 57.6 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_medians_l365_36584


namespace NUMINAMATH_CALUDE_square_difference_mental_calculation_l365_36576

theorem square_difference (n : ℕ) : 
  ((n + 1) ^ 2 : ℕ) = n ^ 2 + 2 * n + 1 ∧ 
  ((n - 1) ^ 2 : ℕ) = n ^ 2 - 2 * n + 1 := by
  sorry

theorem mental_calculation : 
  (41 ^ 2 : ℕ) = 40 ^ 2 + 81 ∧ 
  (39 ^ 2 : ℕ) = 40 ^ 2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_mental_calculation_l365_36576


namespace NUMINAMATH_CALUDE_hotel_expenditure_l365_36547

theorem hotel_expenditure (total_expenditure : ℕ) 
  (standard_spenders : ℕ) (standard_amount : ℕ) (extra_amount : ℕ) : 
  total_expenditure = 117 →
  standard_spenders = 8 →
  standard_amount = 12 →
  extra_amount = 8 →
  ∃ (n : ℕ), n = 9 ∧ 
    (standard_spenders * standard_amount + 
    (total_expenditure / n + extra_amount) = total_expenditure) :=
by sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l365_36547


namespace NUMINAMATH_CALUDE_tax_ratio_is_300_2001_l365_36566

/-- Represents the lottery winnings and expenses scenario --/
structure LotteryScenario where
  winnings : ℚ
  taxRate : ℚ
  loanRate : ℚ
  savings : ℚ
  investmentRate : ℚ
  funMoney : ℚ

/-- Calculates the tax amount given a lottery scenario --/
def calculateTax (scenario : LotteryScenario) : ℚ :=
  scenario.winnings * scenario.taxRate

/-- Theorem stating that the tax ratio is 300:2001 given the specific scenario --/
theorem tax_ratio_is_300_2001 (scenario : LotteryScenario)
  (h1 : scenario.winnings = 12006)
  (h2 : scenario.loanRate = 1/3)
  (h3 : scenario.savings = 1000)
  (h4 : scenario.investmentRate = 1/5)
  (h5 : scenario.funMoney = 2802)
  (h6 : scenario.winnings * (1 - scenario.taxRate) * (1 - scenario.loanRate) - scenario.savings * (1 + scenario.investmentRate) = 2 * scenario.funMoney) :
  (calculateTax scenario) / scenario.winnings = 300 / 2001 := by
sorry

#eval 300 / 2001

end NUMINAMATH_CALUDE_tax_ratio_is_300_2001_l365_36566


namespace NUMINAMATH_CALUDE_fish_pond_population_l365_36537

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (second_catch : ℚ) →
  (initial_tagged * second_catch : ℚ) / (tagged_in_second : ℚ) = 3200 :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l365_36537


namespace NUMINAMATH_CALUDE_drums_per_day_l365_36519

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  let total_drums : ℕ := 2916
  let total_days : ℕ := 9
  let drums_per_day : ℕ := total_drums / total_days
  drums_per_day = 324 := by sorry

end NUMINAMATH_CALUDE_drums_per_day_l365_36519


namespace NUMINAMATH_CALUDE_problem_statement_l365_36579

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ m = 1/4) ∧
  (∀ x : ℝ, 4/a + 1/b ≥ |2*x - 1| - |x + 2| ↔ -6 ≤ x ∧ x ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l365_36579


namespace NUMINAMATH_CALUDE_quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l365_36597

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by sorry

theorem quadratic_equation_rational_coefficients : ∃ a b c : ℚ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

theorem quadratic_equation_leading_coefficient : ∃ a b c : ℝ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l365_36597


namespace NUMINAMATH_CALUDE_solution_set_inequality_l365_36518

theorem solution_set_inequality (x : ℝ) : 
  (x + 1/2) * (3/2 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l365_36518


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l365_36568

/-- Represents the number of eggs found by each group or individual -/
structure EggCounts where
  kevin : ℕ
  someChildren : ℕ
  george : ℕ
  cheryl : ℕ

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (counts : EggCounts) 
  (h1 : counts.kevin = 5)
  (h2 : counts.george = 9)
  (h3 : counts.cheryl = 56)
  (h4 : counts.cheryl = counts.kevin + counts.someChildren + counts.george + 29) :
  counts.someChildren = 13 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l365_36568


namespace NUMINAMATH_CALUDE_cookie_sale_total_l365_36587

/-- Represents the number of cookies sold for each type -/
structure CookieSales where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Defines the conditions of the cookie sale -/
def cookie_sale_conditions (sales : CookieSales) : Prop :=
  sales.raisin = 42 ∧
  sales.raisin = 6 * sales.oatmeal ∧
  6 * sales.oatmeal = sales.oatmeal + 3 * sales.oatmeal + 2 * sales.oatmeal

theorem cookie_sale_total (sales : CookieSales) :
  cookie_sale_conditions sales →
  sales.raisin + sales.oatmeal + sales.chocolate_chip + sales.peanut_butter = 84 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sale_total_l365_36587


namespace NUMINAMATH_CALUDE_chess_tournament_games_l365_36534

theorem chess_tournament_games (n : ℕ) (h : n = 16) : 
  (n * (n - 1)) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l365_36534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l365_36589

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 4 ∧
  a 2 + a 4 = 4

/-- The 10th term of the arithmetic sequence is -5 -/
theorem arithmetic_sequence_10th_term 
  (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : 
  a 10 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l365_36589


namespace NUMINAMATH_CALUDE_pet_store_kittens_l365_36514

/-- The number of kittens initially at the pet store -/
def initial_kittens : ℕ := 6

/-- The number of puppies initially at the pet store -/
def initial_puppies : ℕ := 7

/-- The number of puppies sold -/
def puppies_sold : ℕ := 2

/-- The number of kittens sold -/
def kittens_sold : ℕ := 3

/-- The number of pets remaining after the sale -/
def remaining_pets : ℕ := 8

theorem pet_store_kittens :
  initial_puppies - puppies_sold + (initial_kittens - kittens_sold) = remaining_pets :=
by sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l365_36514


namespace NUMINAMATH_CALUDE_board_numbers_sum_l365_36578

theorem board_numbers_sum (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a^2 + 2*b*c, b^2 + 2*c*a, c^2 + 2*a*b} → 
  a + b + c = 0 ∨ a + b + c = 1 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_sum_l365_36578


namespace NUMINAMATH_CALUDE_ellen_orange_juice_amount_l365_36502

/-- The amount of orange juice in Ellen's smoothie --/
def orange_juice_amount (strawberries yogurt total : ℚ) : ℚ :=
  total - (strawberries + yogurt)

/-- Theorem: Ellen used 0.2 cups of orange juice in her smoothie --/
theorem ellen_orange_juice_amount :
  orange_juice_amount (2/10) (1/10) (5/10) = 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ellen_orange_juice_amount_l365_36502


namespace NUMINAMATH_CALUDE_g_difference_l365_36551

-- Define the function g
def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

-- Theorem statement
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l365_36551


namespace NUMINAMATH_CALUDE_tv_price_before_tax_l365_36523

theorem tv_price_before_tax (P : ℝ) : P + 0.15 * P = 1955 → P = 1700 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_before_tax_l365_36523


namespace NUMINAMATH_CALUDE_two_integers_sum_squares_product_perfect_square_l365_36553

/-- There exist two integers less than 10 whose sum of squares plus their product is a perfect square. -/
theorem two_integers_sum_squares_product_perfect_square :
  ∃ a b : ℤ, a < 10 ∧ b < 10 ∧ ∃ k : ℤ, a^2 + b^2 + a*b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_squares_product_perfect_square_l365_36553


namespace NUMINAMATH_CALUDE_students_not_enrolled_l365_36529

theorem students_not_enrolled (total : ℕ) (biology_frac : ℚ) (chemistry_frac : ℚ) (physics_frac : ℚ) 
  (h_total : total = 1500)
  (h_biology : biology_frac = 2/5)
  (h_chemistry : chemistry_frac = 3/8)
  (h_physics : physics_frac = 1/10)
  (h_no_overlap : biology_frac + chemistry_frac + physics_frac ≤ 1) :
  total - (⌊biology_frac * total⌋ + ⌊chemistry_frac * total⌋ + ⌊physics_frac * total⌋) = 188 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l365_36529


namespace NUMINAMATH_CALUDE_percentage_of_women_in_parent_group_l365_36593

theorem percentage_of_women_in_parent_group (women_fulltime : Real) 
  (men_fulltime : Real) (total_not_fulltime : Real) :
  women_fulltime = 0.9 →
  men_fulltime = 0.75 →
  total_not_fulltime = 0.19 →
  ∃ (w : Real), w ≥ 0 ∧ w ≤ 1 ∧
    w * (1 - women_fulltime) + (1 - w) * (1 - men_fulltime) = total_not_fulltime ∧
    w = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_parent_group_l365_36593


namespace NUMINAMATH_CALUDE_reflected_ray_tangent_to_circle_l365_36592

-- Define the initial ray of light
def initial_ray (x y : ℝ) : Prop := x + 2*y + 2 + Real.sqrt 5 = 0 ∧ y ≥ 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Define a function to check if a point is on the circle
def on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem reflected_ray_tangent_to_circle :
  ∃ (x y : ℝ), initial_ray x y ∧ 
               x_axis y ∧
               on_circle x y ∧
               ∀ (x' y' : ℝ), on_circle x' y' → 
                 ((x' - x)^2 + (y' - y)^2 ≥ 1 ∨ (x' = x ∧ y' = y)) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_tangent_to_circle_l365_36592


namespace NUMINAMATH_CALUDE_percentage_exceeding_speed_limit_l365_36516

/-- Given a road where:
  * 10% of motorists receive speeding tickets
  * 60% of motorists who exceed the speed limit do not receive tickets
  Prove that 25% of motorists exceed the speed limit -/
theorem percentage_exceeding_speed_limit
  (total_motorists : ℝ)
  (h_positive : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h_ticketed : ticketed_percentage = 0.1)
  (non_ticketed_speeders_percentage : ℝ)
  (h_non_ticketed : non_ticketed_speeders_percentage = 0.6)
  : (ticketed_percentage * total_motorists) / (1 - non_ticketed_speeders_percentage) / total_motorists = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_exceeding_speed_limit_l365_36516


namespace NUMINAMATH_CALUDE_arithmetic_sum_l365_36513

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l365_36513


namespace NUMINAMATH_CALUDE_pet_store_transactions_l365_36540

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : Nat
  kittens : Nat
  rabbits : Nat
  guineaPigs : Nat
  chameleons : Nat

/-- Calculates the total number of pets -/
def totalPets (counts : PetCounts) : Nat :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guineaPigs + counts.chameleons

/-- Represents the sales and returns of pets -/
structure Transactions where
  puppiesSold : Nat
  kittensSold : Nat
  rabbitsSold : Nat
  guineaPigsSold : Nat
  chameleonsSold : Nat
  kittensReturned : Nat
  guineaPigsReturned : Nat
  chameleonsReturned : Nat

/-- Calculates the remaining pets after transactions -/
def remainingPets (initial : PetCounts) (trans : Transactions) : PetCounts :=
  { puppies := initial.puppies - trans.puppiesSold,
    kittens := initial.kittens - trans.kittensSold + trans.kittensReturned,
    rabbits := initial.rabbits - trans.rabbitsSold,
    guineaPigs := initial.guineaPigs - trans.guineaPigsSold + trans.guineaPigsReturned,
    chameleons := initial.chameleons - trans.chameleonsSold + trans.chameleonsReturned }

theorem pet_store_transactions (initial : PetCounts) (trans : Transactions) :
  initial.puppies = 7 ∧
  initial.kittens = 6 ∧
  initial.rabbits = 4 ∧
  initial.guineaPigs = 5 ∧
  initial.chameleons = 3 ∧
  trans.puppiesSold = 2 ∧
  trans.kittensSold = 3 ∧
  trans.rabbitsSold = 3 ∧
  trans.guineaPigsSold = 3 ∧
  trans.chameleonsSold = 0 ∧
  trans.kittensReturned = 1 ∧
  trans.guineaPigsReturned = 1 ∧
  trans.chameleonsReturned = 1 →
  totalPets (remainingPets initial trans) = 15 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_transactions_l365_36540


namespace NUMINAMATH_CALUDE_min_sticks_to_break_12_can_form_square_15_l365_36574

-- Define a function to calculate the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define a function to check if it's possible to form a square without breaking sticks
def can_form_square (n : ℕ) : Bool :=
  sum_to_n n % 4 = 0

-- Define a function to find the minimum number of sticks to break
def min_sticks_to_break (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else if n = 12 then 2
  else sorry  -- We don't have a general formula for other cases

-- Theorem for n = 12
theorem min_sticks_to_break_12 :
  min_sticks_to_break 12 = 2 :=
by sorry

-- Theorem for n = 15
theorem can_form_square_15 :
  can_form_square 15 = true :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_to_break_12_can_form_square_15_l365_36574


namespace NUMINAMATH_CALUDE_min_value_theorem_l365_36531

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (2/a + 3/b) ≥ 8 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l365_36531


namespace NUMINAMATH_CALUDE_pauls_caramel_candy_boxes_l365_36562

/-- Given that Paul bought 6 boxes of chocolate candy, each box has 9 pieces,
    and he had 90 candies in total, prove that he bought 4 boxes of caramel candy. -/
theorem pauls_caramel_candy_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) :
  chocolate_boxes = 6 →
  pieces_per_box = 9 →
  total_candies = 90 →
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_pauls_caramel_candy_boxes_l365_36562


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_28_l365_36575

theorem quadratic_sum_equals_28 (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : b + c = 2) : 
  a^2 + b^2 + c^2 - a*b + b*c + c*a = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_28_l365_36575


namespace NUMINAMATH_CALUDE_skateboard_distance_l365_36526

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The skateboard problem -/
theorem skateboard_distance : arithmeticSequenceSum 8 9 20 = 1870 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l365_36526


namespace NUMINAMATH_CALUDE_num_divisors_360_eq_24_l365_36572

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end NUMINAMATH_CALUDE_num_divisors_360_eq_24_l365_36572


namespace NUMINAMATH_CALUDE_factorization_equality_l365_36598

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l365_36598


namespace NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_l365_36581

theorem regular_polygon_with_160_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- A polygon must have at least 3 sides
  (∀ i : ℕ, i < n → 160 = (n - 2) * 180 / n) →  -- Each interior angle is 160°
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_l365_36581


namespace NUMINAMATH_CALUDE_almond_butter_servings_l365_36517

/-- Represents the number of tablespoons in the container -/
def container_amount : ℚ := 35 + 2/3

/-- Represents the number of tablespoons in one serving -/
def serving_size : ℚ := 2 + 1/2

/-- Represents the number of servings in the container -/
def number_of_servings : ℚ := container_amount / serving_size

theorem almond_butter_servings : 
  ∃ (n : ℕ) (r : ℚ), 0 ≤ r ∧ r < 1 ∧ number_of_servings = n + r ∧ n = 14 ∧ r = 4/15 :=
sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l365_36517


namespace NUMINAMATH_CALUDE_quadratic_sum_l365_36558

/-- A quadratic function f(x) = ax^2 - bx + c passing through (1, -1) with vertex at (-1/2, -1/4) -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x ↦ (a : ℝ) * x^2 - (b : ℝ) * x + (c : ℝ)

theorem quadratic_sum (a b c : ℤ) :
  (QuadraticFunction a b c 1 = -1) →
  (QuadraticFunction a b c (-1/2) = -1/4) →
  a + b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l365_36558


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l365_36599

/-- 
For a rectangle with area 12 and sides of length x and y,
the function relationship between y and x is y = 12/x.
-/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 12) : 
  y = 12 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l365_36599


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l365_36527

theorem cube_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_equality : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l365_36527


namespace NUMINAMATH_CALUDE_inequality_proof_l365_36505

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 : ℝ) / (a^3 + b^3 + c^3) ≤ 1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ∧
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l365_36505


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l365_36506

def expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5

theorem sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l365_36506


namespace NUMINAMATH_CALUDE_regression_maximum_fitting_l365_36500

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the true relationship between x and y --/
def true_relationship : ℝ → ℝ := sorry

/-- Measures the degree of fitting between the regression model and the true relationship --/
def fitting_degree (model : LinearRegression) : ℝ := sorry

/-- The regression equation represents the maximum degree of fitting --/
theorem regression_maximum_fitting (data : List (ℝ × ℝ)) :
  ∃ (model : LinearRegression),
    ∀ (other_model : LinearRegression),
      fitting_degree model ≥ fitting_degree other_model := by
  sorry

end NUMINAMATH_CALUDE_regression_maximum_fitting_l365_36500


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_integer_l365_36507

theorem sum_and_reciprocal_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  (∃ m : ℤ, a^2 + 1/a^2 = m) ∧ (∀ n : ℕ, ∃ l : ℤ, a^n + 1/a^n = l) := by
  sorry


end NUMINAMATH_CALUDE_sum_and_reciprocal_integer_l365_36507


namespace NUMINAMATH_CALUDE_winning_strategy_l365_36524

/-- Represents the colors used in the chocolate table -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a cell in the chocolate table -/
structure Cell :=
  (row : Nat)
  (col : Nat)
  (color : Color)

/-- Represents the chocolate table -/
def ChocolateTable (n : Nat) := Array (Array Cell)

/-- Creates a colored n × n chocolate table -/
def createTable (n : Nat) : ChocolateTable n := sorry

/-- Removes a cell from the chocolate table -/
def removeCell (table : ChocolateTable n) (cell : Cell) : ChocolateTable n := sorry

/-- Checks if a 3 × 1 or 1 × 3 rectangle contains one of each color -/
def validRectangle (rect : Array Cell) : Bool := sorry

/-- Checks if the table can be partitioned into valid rectangles -/
def canPartition (table : ChocolateTable n) : Bool := sorry

theorem winning_strategy 
  (n : Nat) 
  (h1 : n > 3) 
  (h2 : ¬(3 ∣ n)) : 
  ∃ (cell : Cell), cell.color ≠ Color.Red ∧ 
    ¬(canPartition (removeCell (createTable n) cell)) := by sorry

end NUMINAMATH_CALUDE_winning_strategy_l365_36524


namespace NUMINAMATH_CALUDE_anna_score_l365_36525

/-- Calculates the score in a modified contest given the number of correct, incorrect, and unanswered questions -/
def contest_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct : ℚ) + 0 * (incorrect : ℚ) - 0.5 * (unanswered : ℚ)

theorem anna_score :
  let total_questions : ℕ := 30
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 6
  let unanswered_questions : ℕ := 7
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  contest_score correct_answers incorrect_answers unanswered_questions = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_anna_score_l365_36525


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l365_36577

theorem parallel_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), 2*x + (m+1)*y + 4 = 0) ∧ 
  (∃ (x y : ℝ), m*x + 3*y - 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2*x₁ + (m+1)*y₁ + 4 = 0 ∧ m*x₂ + 3*y₂ - 2 = 0 → 
    (2 / (m+1) = m / 3)) →
  m = -3 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l365_36577


namespace NUMINAMATH_CALUDE_prime_n_l365_36582

theorem prime_n (p h n : ℕ) : 
  Nat.Prime p → 
  h < p → 
  n = p * h + 1 → 
  (2^(n-1) - 1) % n = 0 → 
  (2^h - 1) % n ≠ 0 → 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_prime_n_l365_36582


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l365_36539

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream (friday_amount saturday_amount : Real) : Real :=
  friday_amount + saturday_amount

/-- Theorem stating the total amount of ice cream eaten -/
theorem ice_cream_consumption : 
  total_ice_cream 3.25 0.25 = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l365_36539


namespace NUMINAMATH_CALUDE_quadratic_sum_bounds_l365_36535

theorem quadratic_sum_bounds (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 11)
  (eq2 : b^2 + b*c + c^2 = 11) :
  0 ≤ c^2 + c*a + a^2 ∧ c^2 + c*a + a^2 ≤ 44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_bounds_l365_36535


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l365_36543

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 7 = 0 → 
  q^2 - 5*q + 7 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 559 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l365_36543


namespace NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_odd_first_l365_36563

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ Odd (n / 1000)

/-- The theorem stating that 1221 is the smallest four-digit palindrome 
    divisible by 3 with an odd first digit -/
theorem smallest_four_digit_palindrome_div_by_3_odd_first : 
  (∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 ∧ has_odd_first_digit n → n ≥ 1221) ∧
  is_four_digit_palindrome 1221 ∧ 1221 % 3 = 0 ∧ has_odd_first_digit 1221 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_odd_first_l365_36563


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l365_36557

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 18) :
  let r := d / 2
  let m := (2 * r^2)^(1/2)
  m^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l365_36557


namespace NUMINAMATH_CALUDE_g_of_5_equals_22_l365_36512

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x + 2

-- Theorem statement
theorem g_of_5_equals_22 : g 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_22_l365_36512


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l365_36585

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- Another line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Point of intersection of the two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- The problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 5 ∧
  e.y = 5

/-- The area of quadrilateral OBEC -/
def QuadrilateralArea (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : ℝ := 
  sorry  -- The actual calculation would go here

theorem area_of_quadrilateral (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) 
  (h : ProblemSetup l1 l2 e) : QuadrilateralArea l1 l2 e = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l365_36585


namespace NUMINAMATH_CALUDE_trains_crossing_time_l365_36508

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 500 → 
  length2 = 750 → 
  speed1 = 60 → 
  speed2 = 40 → 
  (((length1 + length2) / ((speed1 + speed2) * (5/18))) : ℝ) = 45 := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l365_36508


namespace NUMINAMATH_CALUDE_rectangle_length_l365_36560

/-- The length of a rectangle with width 4 cm and area equal to a square with sides 4 cm -/
theorem rectangle_length (width : ℝ) (square_side : ℝ) (length : ℝ) : 
  width = 4 →
  square_side = 4 →
  length * width = square_side * square_side →
  length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l365_36560


namespace NUMINAMATH_CALUDE_used_car_selection_l365_36588

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  num_clients = 18 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l365_36588


namespace NUMINAMATH_CALUDE_sum_of_three_integers_l365_36591

theorem sum_of_three_integers (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + b + c = 131 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_l365_36591


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l365_36583

/-- A circle with center (-1, 3) that is tangent to the line x - y = 0 -/
def tangentCircle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 8

/-- The line x - y = 0 -/
def tangentLine (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of the circle -/
def circleCenter : ℝ × ℝ := (-1, 3)

theorem circle_tangent_to_line :
  ∃ (x₀ y₀ : ℝ), tangentCircle x₀ y₀ ∧ tangentLine x₀ y₀ ∧
  ∀ (x y : ℝ), tangentCircle x y ∧ tangentLine x y → (x, y) = (x₀, y₀) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l365_36583


namespace NUMINAMATH_CALUDE_probability_two_red_two_green_l365_36552

theorem probability_two_red_two_green (total_red : ℕ) (total_green : ℕ) (drawn : ℕ) : 
  total_red = 10 → total_green = 8 → drawn = 4 →
  (Nat.choose total_red 2 * Nat.choose total_green 2) / Nat.choose (total_red + total_green) drawn = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_two_green_l365_36552


namespace NUMINAMATH_CALUDE_derivative_of_cosine_linear_l365_36570

/-- Given a function f(x) = cos(2x - π/6), its derivative f'(x) = -2sin(2x - π/6) --/
theorem derivative_of_cosine_linear (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x - π / 6)
  deriv f x = -2 * Real.sin (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_cosine_linear_l365_36570


namespace NUMINAMATH_CALUDE_largest_power_l365_36573

theorem largest_power (a b c d e : ℕ) :
  a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 8 ∧ e = 16 →
  c^8 ≥ a^20 ∧ c^8 ≥ b^14 ∧ c^8 ≥ d^5 ∧ c^8 ≥ e^3 :=
by sorry

#check largest_power

end NUMINAMATH_CALUDE_largest_power_l365_36573


namespace NUMINAMATH_CALUDE_gcd_plus_lcm_eq_sum_iff_divides_l365_36536

theorem gcd_plus_lcm_eq_sum_iff_divides (x y : ℕ) :
  (Nat.gcd x y + x * y / Nat.gcd x y = x + y) ↔ (y ∣ x ∨ x ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_gcd_plus_lcm_eq_sum_iff_divides_l365_36536


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_iff_a_geq_four_l365_36538

theorem square_minus_a_nonpositive_iff_a_geq_four :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_iff_a_geq_four_l365_36538


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l365_36569

/-- The line equation y = kx + 2 is tangent to the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if k^2 = 3/4 -/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∃! x y : ℝ, y = k * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) ↔ k^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l365_36569


namespace NUMINAMATH_CALUDE_quadratic_inequality_relationship_l365_36567

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- Define proposition A
def proposition_A (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement
theorem quadratic_inequality_relationship :
  (∀ a : ℝ, proposition_A a → proposition_B a) ∧
  (∃ a : ℝ, proposition_B a ∧ ¬proposition_A a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relationship_l365_36567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l365_36556

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l365_36556


namespace NUMINAMATH_CALUDE_unsupported_attendees_l365_36555

-- Define the total number of attendees
def total_attendance : ℕ := 500

-- Define the percentage of supporters for each team
def team_a_percentage : ℚ := 35 / 100
def team_b_percentage : ℚ := 25 / 100
def team_c_percentage : ℚ := 20 / 100
def team_d_percentage : ℚ := 15 / 100

-- Define the overlap percentages
def team_ab_overlap_percentage : ℚ := 10 / 100
def team_bc_overlap_percentage : ℚ := 5 / 100
def team_cd_overlap_percentage : ℚ := 7 / 100

-- Define the number of people attending for atmosphere
def atmosphere_attendees : ℕ := 30

-- Theorem to prove
theorem unsupported_attendees :
  ∃ (unsupported : ℕ),
    unsupported = total_attendance -
      (((team_a_percentage + team_b_percentage + team_c_percentage + team_d_percentage) * total_attendance).floor -
       ((team_ab_overlap_percentage * team_a_percentage * total_attendance +
         team_bc_overlap_percentage * team_b_percentage * total_attendance +
         team_cd_overlap_percentage * team_c_percentage * total_attendance).floor) +
       atmosphere_attendees) ∧
    unsupported = 26 := by
  sorry

end NUMINAMATH_CALUDE_unsupported_attendees_l365_36555


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l365_36521

theorem imaginary_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i)^2 * (2 + i)
  Complex.im z = 4 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l365_36521


namespace NUMINAMATH_CALUDE_school_buses_l365_36580

def bus_seats (columns : ℕ) (rows : ℕ) : ℕ := columns * rows

def total_capacity (num_buses : ℕ) (seats_per_bus : ℕ) : ℕ := num_buses * seats_per_bus

theorem school_buses (columns : ℕ) (rows : ℕ) (total_students : ℕ) (num_buses : ℕ) :
  columns = 4 →
  rows = 10 →
  total_students = 240 →
  total_capacity num_buses (bus_seats columns rows) = total_students →
  num_buses = 6 := by
  sorry

#check school_buses

end NUMINAMATH_CALUDE_school_buses_l365_36580


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l365_36509

theorem sqrt_meaningful_iff_geq_two (a : ℝ) : ∃ x : ℝ, x^2 = a - 2 ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l365_36509


namespace NUMINAMATH_CALUDE_shoppingMallMethodIsSystematic_l365_36541

/-- Represents a sampling method with a fixed interval and starting point -/
structure SamplingMethod where
  interval : ℕ
  start : ℕ

/-- Defines the characteristics of systematic sampling -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.interval > 0 ∧
  method.start > 0 ∧
  method.start ≤ method.interval

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { interval := 50,
    start := 15 }

/-- Theorem stating that the shopping mall's method is a systematic sampling method -/
theorem shoppingMallMethodIsSystematic :
  isSystematicSampling shoppingMallMethod := by
  sorry

end NUMINAMATH_CALUDE_shoppingMallMethodIsSystematic_l365_36541


namespace NUMINAMATH_CALUDE_fraction_sum_ratio_l365_36522

theorem fraction_sum_ratio : (1 / 3 + 1 / 4) / (1 / 2) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_ratio_l365_36522


namespace NUMINAMATH_CALUDE_t_shirt_cost_is_20_l365_36504

/-- The cost of a single t-shirt -/
def t_shirt_cost : ℝ := sorry

/-- The number of t-shirts bought -/
def num_t_shirts : ℕ := 3

/-- The cost of pants -/
def pants_cost : ℝ := 50

/-- The total amount spent -/
def total_spent : ℝ := 110

theorem t_shirt_cost_is_20 : t_shirt_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_is_20_l365_36504


namespace NUMINAMATH_CALUDE_combination_sum_equality_l365_36586

theorem combination_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + (Finset.range (k + 1)).sum (λ i => (Nat.choose k i) * (Nat.choose n (m - i))) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l365_36586


namespace NUMINAMATH_CALUDE_fruit_salad_price_l365_36545

/-- Represents the cost of the picnic basket items -/
structure PicnicBasket where
  numPeople : Nat
  sandwichPrice : Nat
  sodaPrice : Nat
  snackPrice : Nat
  numSnacks : Nat
  totalCost : Nat

/-- Calculates the cost of fruit salads given the picnic basket information -/
def fruitSaladCost (basket : PicnicBasket) : Nat :=
  basket.totalCost - 
  (basket.numPeople * basket.sandwichPrice + 
   2 * basket.numPeople * basket.sodaPrice + 
   basket.numSnacks * basket.snackPrice)

/-- Theorem stating that the cost of each fruit salad is $3 -/
theorem fruit_salad_price (basket : PicnicBasket) 
  (h1 : basket.numPeople = 4)
  (h2 : basket.sandwichPrice = 5)
  (h3 : basket.sodaPrice = 2)
  (h4 : basket.snackPrice = 4)
  (h5 : basket.numSnacks = 3)
  (h6 : basket.totalCost = 60) :
  fruitSaladCost basket / basket.numPeople = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_price_l365_36545


namespace NUMINAMATH_CALUDE_unique_root_implies_m_equals_one_l365_36532

def f (x : ℝ) := 2 * x^2 + x - 4

theorem unique_root_implies_m_equals_one (m n : ℤ) :
  (n = m + 1) →
  (∃! x : ℝ, m < x ∧ x < n ∧ f x = 0) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_m_equals_one_l365_36532


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l365_36594

theorem min_value_reciprocal_sum (x : ℝ) (h1 : 0 < x) (h2 : x < 3) :
  (1 / x) + (1 / (3 - x)) ≥ 4 / 3 ∧
  ((1 / x) + (1 / (3 - x)) = 4 / 3 ↔ x = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l365_36594


namespace NUMINAMATH_CALUDE_exponential_inequality_l365_36533

theorem exponential_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l365_36533


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l365_36550

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x + 2 * y = 8) ∧ (11 * x - 5 * y = 1) ↔ x = 42/47 ∧ y = 83/47 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l365_36550
