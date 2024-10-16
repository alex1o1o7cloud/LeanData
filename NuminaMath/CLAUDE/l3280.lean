import Mathlib

namespace NUMINAMATH_CALUDE_jason_seashell_count_l3280_328036

def seashell_count (initial : ℕ) (given_tim : ℕ) (given_lily : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial - given_tim - given_lily + found - lost

theorem jason_seashell_count : 
  seashell_count 49 13 7 15 5 = 39 := by sorry

end NUMINAMATH_CALUDE_jason_seashell_count_l3280_328036


namespace NUMINAMATH_CALUDE_ten_men_joined_l3280_328091

/-- Represents the number of men who joined the camp -/
def men_joined : ℕ := sorry

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial duration of the food supply in days -/
def initial_duration : ℕ := 50

/-- The new duration of the food supply after more men join -/
def new_duration : ℕ := 25

/-- The total amount of food available in man-days -/
def total_food : ℕ := initial_men * initial_duration

/-- Theorem stating that 10 men joined the camp -/
theorem ten_men_joined : men_joined = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_men_joined_l3280_328091


namespace NUMINAMATH_CALUDE_sum_of_ages_l3280_328086

/-- Given Bob's and Carol's ages, prove that their sum is 66 years. -/
theorem sum_of_ages (bob_age carol_age : ℕ) : 
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 →
  bob_age = 16 →
  bob_age + carol_age = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3280_328086


namespace NUMINAMATH_CALUDE_problem_solution_l3280_328049

def g (x : ℝ) : ℝ := |x - 1| + |2*x + 4|
def f (a x : ℝ) : ℝ := |x - a| + 2 + a

theorem problem_solution :
  (∀ a : ℝ, ∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) →
  (∀ x : ℝ, x ∈ {x : ℝ | g x < 6} ↔ x ∈ Set.Ioo (-3) 1) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) → a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3280_328049


namespace NUMINAMATH_CALUDE_product_of_first_six_terms_l3280_328058

/-- A geometric sequence with the given property -/
def GeometricSequenceWithProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) * a (2 * n) = 3^n

/-- The theorem to be proved -/
theorem product_of_first_six_terms
  (a : ℕ → ℝ)
  (h : GeometricSequenceWithProperty a) :
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_six_terms_l3280_328058


namespace NUMINAMATH_CALUDE_gcd_multiple_smallest_l3280_328016

/-- Given positive integers m and n with gcd(m,n) = 12, 
    the smallest possible value of gcd(12m,18n) is 72 -/
theorem gcd_multiple_smallest (m n : ℕ+) (h : Nat.gcd m n = 12) :
  ∃ (k : ℕ+), ∀ (x : ℕ+), Nat.gcd (12 * m) (18 * n) ≥ k ∧ 
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 12 ∧ Nat.gcd (12 * m') (18 * n') = k) ∧
  k = 72 := by
  sorry

#check gcd_multiple_smallest

end NUMINAMATH_CALUDE_gcd_multiple_smallest_l3280_328016


namespace NUMINAMATH_CALUDE_expression_evaluation_l3280_328054

theorem expression_evaluation :
  (18^40 : ℕ) / (54^20) * 2^10 = 2^30 * 3^20 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3280_328054


namespace NUMINAMATH_CALUDE_sqrt_sum_ratio_equals_three_l3280_328034

theorem sqrt_sum_ratio_equals_three : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_ratio_equals_three_l3280_328034


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3280_328088

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 1044 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 58 / 100 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3280_328088


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l3280_328020

theorem polynomial_equality_implies_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l3280_328020


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3280_328026

theorem max_product_sum_2000 :
  (∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3280_328026


namespace NUMINAMATH_CALUDE_six_six_six_triangle_l3280_328052

/-- Triangle Inequality Theorem: A set of three positive real numbers can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (6, 6, 6) can form a triangle. -/
theorem six_six_six_triangle : can_form_triangle 6 6 6 := by
  sorry


end NUMINAMATH_CALUDE_six_six_six_triangle_l3280_328052


namespace NUMINAMATH_CALUDE_problem_statement_l3280_328055

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1)
  (h_eq1 : a + 1 / c = 8)
  (h_eq2 : b + 1 / a = 20) :
  c + 1 / b = 10 / 53 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3280_328055


namespace NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l3280_328068

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 > 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l3280_328068


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3280_328025

/-- Represents the selling price of grapes in yuan per kilogram -/
def selling_price : ℝ := 21

/-- Represents the cost price of grapes in yuan per kilogram -/
def cost_price : ℝ := 16

/-- Represents the daily sales volume in kilograms when the price is 26 yuan -/
def base_sales : ℝ := 320

/-- Represents the increase in sales volume for each yuan decrease in price -/
def sales_increase_rate : ℝ := 80

/-- Represents the target daily profit in yuan -/
def target_profit : ℝ := 3600

/-- Calculates the daily sales volume based on the selling price -/
def sales_volume (x : ℝ) : ℝ := base_sales + sales_increase_rate * (26 - x)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

/-- Theorem stating that the chosen selling price satisfies the profit goal and is optimal -/
theorem optimal_selling_price : 
  daily_profit selling_price = target_profit ∧ 
  (∀ y, y < selling_price → daily_profit y < target_profit) :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l3280_328025


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l3280_328014

def sequence_sum (n : ℕ) : ℚ :=
  if n = 0 then 3 / 3
  else if n = 1 then 4 + 1/3 * (sequence_sum 0)
  else (2003 - n + 1) + 1/3 * (sequence_sum (n-1))

theorem sequence_sum_formula : 
  sequence_sum 2000 = 3004.5 - 1 / (2 * 3^1999) := by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l3280_328014


namespace NUMINAMATH_CALUDE_judge_court_cases_judge_court_cases_proof_l3280_328043

theorem judge_court_cases : ℕ → Prop :=
  fun total_cases =>
    let dismissed := 2
    let remaining := total_cases - dismissed
    let innocent := (2 * remaining) / 3
    let delayed := 1
    let guilty := 4
    remaining - innocent - delayed = guilty ∧ total_cases = 17

-- The proof
theorem judge_court_cases_proof : ∃ n : ℕ, judge_court_cases n := by
  sorry

end NUMINAMATH_CALUDE_judge_court_cases_judge_court_cases_proof_l3280_328043


namespace NUMINAMATH_CALUDE_board_number_remainder_l3280_328071

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_board_number_remainder_l3280_328071


namespace NUMINAMATH_CALUDE_sum_between_nine_half_and_ten_l3280_328072

theorem sum_between_nine_half_and_ten : 
  let sum := (29/9 : ℚ) + (11/4 : ℚ) + (81/20 : ℚ)
  (9.5 : ℚ) < sum ∧ sum < (10 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_between_nine_half_and_ten_l3280_328072


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l3280_328030

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number : 
  is_interesting 1800 ∧ ∀ m < 1800, ¬is_interesting m :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l3280_328030


namespace NUMINAMATH_CALUDE_function_minimum_value_l3280_328077

theorem function_minimum_value (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ + a)^2 + (Real.exp x₀ + a / Real.exp 1)^2 ≤ 4 / (Real.exp 2 + 1)) →
  a = (Real.exp 2 - 1) / (Real.exp 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_value_l3280_328077


namespace NUMINAMATH_CALUDE_solution_value_l3280_328038

theorem solution_value (m n : ℝ) : 
  (∀ x, x^2 - m*x + n ≤ 0 ↔ -5 ≤ x ∧ x ≤ 1) →
  ((-5)^2 - m*(-5) + n = 0) →
  (1^2 - m*1 + n = 0) →
  m - n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3280_328038


namespace NUMINAMATH_CALUDE_frequency_third_group_l3280_328093

theorem frequency_third_group (m : ℕ) (h1 : m ≥ 3) : 
  let total_frequency : ℝ := 1
  let third_rectangle_area : ℝ := (1 / 4) * (total_frequency - third_rectangle_area)
  let sample_size : ℕ := 100
  (third_rectangle_area * sample_size : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_frequency_third_group_l3280_328093


namespace NUMINAMATH_CALUDE_inequality_proof_l3280_328057

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3280_328057


namespace NUMINAMATH_CALUDE_fraction_power_product_l3280_328066

theorem fraction_power_product : (1/3)^100 * 3^101 = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3280_328066


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3280_328059

/-- Given a 3-4-5 right triangle, let x be the side length of a square inscribed
    with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x * (7 / 3) = 4) -- Derived from the condition for x
  (hy : y * (37 / 12) = 5) -- Derived from the condition for y
  : x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3280_328059


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3280_328099

/-- Given the cost of pens and pencils, prove the cost of one dozen pens -/
theorem cost_of_dozen_pens 
  (cost_3pens_5pencils : ℕ) 
  (ratio : ℚ) 
  (cost_dozen_pens : ℕ) 
  (h1 : cost_3pens_5pencils = 100)
  (h2 : ratio > 0)
  (h3 : cost_dozen_pens = 300) :
  cost_dozen_pens = 300 := by
  sorry

#check cost_of_dozen_pens

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3280_328099


namespace NUMINAMATH_CALUDE_subtract_repeating_third_from_four_l3280_328012

/-- The repeating decimal 0.3̅ -/
def repeating_third : ℚ := 1/3

/-- Proof that 4 - 0.3̅ = 11/3 -/
theorem subtract_repeating_third_from_four :
  4 - repeating_third = 11/3 := by sorry

end NUMINAMATH_CALUDE_subtract_repeating_third_from_four_l3280_328012


namespace NUMINAMATH_CALUDE_expected_defective_theorem_l3280_328017

/-- The expected number of defective products drawn before a genuine product is drawn -/
def expected_defective_drawn (genuine : ℕ) (defective : ℕ) : ℚ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem expected_defective_theorem :
  expected_defective_drawn 9 3 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_theorem_l3280_328017


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3280_328045

theorem quadratic_equivalence :
  ∀ x : ℝ, (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3280_328045


namespace NUMINAMATH_CALUDE_max_area_of_remaining_rectangle_l3280_328028

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles in the square -/
structure SquareConfiguration where
  sideLength : ℝ
  rect1 : Rectangle
  rect2 : Rectangle
  rectR : Rectangle

/-- The theorem statement -/
theorem max_area_of_remaining_rectangle (config : SquareConfiguration) :
  config.sideLength ≥ 4 →
  config.rect1.width = 2 ∧ config.rect1.height = 4 →
  config.rect2.width = 2 ∧ config.rect2.height = 2 →
  config.rectR.area ≤ config.sideLength^2 - 12 ∧
  (config.sideLength = 4 → config.rectR.area = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_remaining_rectangle_l3280_328028


namespace NUMINAMATH_CALUDE_class_size_problem_l3280_328001

theorem class_size_problem (x : ℕ) : 
  (40 * x + 50 * 90) / (x + 50 : ℝ) = 71.25 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l3280_328001


namespace NUMINAMATH_CALUDE_parabola_line_intersection_bounds_l3280_328023

/-- Parabola P with equation y = 2x^2 -/
def P : ℝ → ℝ := λ x => 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Theorem stating the existence of r and s, and their sum -/
theorem parabola_line_intersection_bounds :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_bounds_l3280_328023


namespace NUMINAMATH_CALUDE_log_five_twelve_l3280_328084

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2 * a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_twelve_l3280_328084


namespace NUMINAMATH_CALUDE_exist_special_numbers_l3280_328002

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two numbers satisfying the given conditions -/
theorem exist_special_numbers : 
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A = 2016 * B ∧ sum_of_digits A = sum_of_digits B / 2016 := by
  sorry

end NUMINAMATH_CALUDE_exist_special_numbers_l3280_328002


namespace NUMINAMATH_CALUDE_tangent_line_and_lower_bound_l3280_328015

noncomputable def f (x : ℝ) := Real.exp x - 3 * x^2 + 4 * x

theorem tangent_line_and_lower_bound :
  (∃ (b : ℝ), ∀ x, (Real.exp 1 - 2) * x + b = f 1 + (Real.exp 1 - 6 + 4) * (x - 1)) ∧
  (∀ x ≥ 1, f x > 3) ∧
  (∃ x₀ ≥ 1, f x₀ < 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_lower_bound_l3280_328015


namespace NUMINAMATH_CALUDE_mission_duration_l3280_328050

theorem mission_duration (planned_duration : ℝ) (overtime_percentage : ℝ) (second_mission_duration : ℝ) : 
  planned_duration = 5 ∧ 
  overtime_percentage = 0.6 ∧ 
  second_mission_duration = 3 → 
  planned_duration * (1 + overtime_percentage) + second_mission_duration = 11 :=
by sorry

end NUMINAMATH_CALUDE_mission_duration_l3280_328050


namespace NUMINAMATH_CALUDE_max_telephones_is_210_quality_rate_at_least_90_percent_l3280_328061

/-- Represents the quality inspection of a batch of telephones. -/
structure TelephoneBatch where
  first_50_high_quality : Nat := 49
  first_50_total : Nat := 50
  subsequent_high_quality : Nat := 7
  subsequent_total : Nat := 8
  quality_threshold : Rat := 9/10

/-- The maximum number of telephones in the batch satisfying the quality conditions. -/
def max_telephones (batch : TelephoneBatch) : Nat :=
  batch.first_50_total + 20 * batch.subsequent_total

/-- Theorem stating that 210 is the maximum number of telephones in the batch. -/
theorem max_telephones_is_210 (batch : TelephoneBatch) :
  max_telephones batch = 210 :=
by sorry

/-- Theorem stating that the quality rate is at least 90% for the maximum batch size. -/
theorem quality_rate_at_least_90_percent (batch : TelephoneBatch) :
  let total := max_telephones batch
  let high_quality := batch.first_50_high_quality + 20 * batch.subsequent_high_quality
  (high_quality : Rat) / total ≥ batch.quality_threshold :=
by sorry

end NUMINAMATH_CALUDE_max_telephones_is_210_quality_rate_at_least_90_percent_l3280_328061


namespace NUMINAMATH_CALUDE_product_of_cosines_l3280_328031

theorem product_of_cosines : 
  (1 + Real.cos (π/8)) * (1 + Real.cos (3*π/8)) * (1 + Real.cos (5*π/8)) * (1 + Real.cos (7*π/8)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l3280_328031


namespace NUMINAMATH_CALUDE_total_pages_in_book_l3280_328029

/-- The number of pages Suzanne read on Monday -/
def monday_pages : ℝ := 15.5

/-- The number of pages Suzanne read on Tuesday -/
def tuesday_pages : ℝ := 1.5 * monday_pages + 16

/-- The total number of pages Suzanne read in two days -/
def total_pages_read : ℝ := monday_pages + tuesday_pages

/-- The theorem stating the total number of pages in the book -/
theorem total_pages_in_book : total_pages_read * 2 = 109.5 := by sorry

end NUMINAMATH_CALUDE_total_pages_in_book_l3280_328029


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3280_328007

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3280_328007


namespace NUMINAMATH_CALUDE_f_formula_f_min_f_max_l3280_328063

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The conditions for the quadratic function -/
axiom f_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0
axiom f_zero : f 0 = 1
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- Theorem: The quadratic function f(x) is x^2 - x + 1 -/
theorem f_formula : ∀ x, f x = x^2 - x + 1 := sorry

/-- Theorem: The minimum value of f(x) on [-1, 1] is 3/4 -/
theorem f_min : Set.Icc (-1 : ℝ) 1 ⊆ f ⁻¹' (Set.Icc (3/4 : ℝ) (f (-1))) := sorry

/-- Theorem: The maximum value of f(x) on [-1, 1] is 3 -/
theorem f_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 3 := sorry

end NUMINAMATH_CALUDE_f_formula_f_min_f_max_l3280_328063


namespace NUMINAMATH_CALUDE_andy_final_position_l3280_328006

-- Define the direction as an enumeration
inductive Direction
  | North
  | West
  | South
  | East

-- Define the position as a pair of integers
def Position := ℤ × ℤ

-- Define the function to get the next direction after turning left
def turn_left (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East
  | Direction.East => Direction.North

-- Define the function to move in a given direction
def move (p : Position) (d : Direction) (distance : ℤ) : Position :=
  match d with
  | Direction.North => (p.1, p.2 + distance)
  | Direction.West => (p.1 - distance, p.2)
  | Direction.South => (p.1, p.2 - distance)
  | Direction.East => (p.1 + distance, p.2)

-- Define the function to perform one step of Andy's movement
def step (p : Position) (d : Direction) (n : ℕ) : Position × Direction :=
  let new_p := move p d (n^2)
  let new_d := turn_left d
  (new_p, new_d)

-- Define the function to perform multiple steps
def multi_step (initial_p : Position) (initial_d : Direction) (steps : ℕ) : Position :=
  if steps = 0 then
    initial_p
  else
    let (p, d) := (List.range steps).foldl
      (fun (acc : Position × Direction) n => step acc.1 acc.2 (n + 1))
      (initial_p, initial_d)
    p

-- Theorem statement
theorem andy_final_position :
  multi_step (10, -10) Direction.North 16 = (154, -138) :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l3280_328006


namespace NUMINAMATH_CALUDE_strawberry_calculation_l3280_328085

/-- Converts kilograms and grams to total grams -/
def to_grams (kg : ℕ) (g : ℕ) : ℕ := kg * 1000 + g

/-- Calculates remaining strawberries in grams -/
def remaining_strawberries (total_kg : ℕ) (total_g : ℕ) (given_kg : ℕ) (given_g : ℕ) : ℕ :=
  to_grams total_kg total_g - to_grams given_kg given_g

theorem strawberry_calculation :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calculation_l3280_328085


namespace NUMINAMATH_CALUDE_prime_factors_power_l3280_328013

/-- Given an expression containing 4^11, 7^5, and 11^x, 
    if the total number of prime factors is 29, then x = 2 -/
theorem prime_factors_power (x : ℕ) : 
  (22 + 5 + x = 29) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_power_l3280_328013


namespace NUMINAMATH_CALUDE_min_absolute_difference_l3280_328003

theorem min_absolute_difference (a b c d : ℝ) 
  (hab : |a - b| = 5)
  (hbc : |b - c| = 8)
  (hcd : |c - d| = 10) :
  ∃ (m : ℝ), (∀ x, |a - d| ≥ x → m ≤ x) ∧ |a - d| ≥ m ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_absolute_difference_l3280_328003


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l3280_328024

/-- Represents the number of pairs for each shoe color --/
structure ShoeInventory where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of picking two shoes of the same color and opposite types --/
def probabilitySameColorOppositeTypes (inventory : ShoeInventory) : Rat :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let prob_black := (2 * inventory.black) / totalShoes * inventory.black / (totalShoes - 1)
  let prob_brown := (2 * inventory.brown) / totalShoes * inventory.brown / (totalShoes - 1)
  let prob_gray := (2 * inventory.gray) / totalShoes * inventory.gray / (totalShoes - 1)
  let prob_red := (2 * inventory.red) / totalShoes * inventory.red / (totalShoes - 1)
  prob_black + prob_brown + prob_gray + prob_red

/-- Sue's shoe inventory --/
def sueInventory : ShoeInventory := ⟨7, 4, 2, 2⟩

theorem sue_shoe_probability :
  probabilitySameColorOppositeTypes sueInventory = 73 / 435 := by
  sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l3280_328024


namespace NUMINAMATH_CALUDE_evaluate_expression_l3280_328096

theorem evaluate_expression (b : ℝ) : 
  let x := 2 * b + 9
  x - 2 * b + 5 = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3280_328096


namespace NUMINAMATH_CALUDE_sibling_pair_probability_l3280_328080

theorem sibling_pair_probability (business_students : ℕ) (law_students : ℕ) (sibling_pairs : ℕ) : 
  business_students = 500 →
  law_students = 800 →
  sibling_pairs = 30 →
  (sibling_pairs : ℚ) / (business_students * law_students) = 0.000075 := by
  sorry

end NUMINAMATH_CALUDE_sibling_pair_probability_l3280_328080


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3280_328098

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x < 2}
  let B : Set ℝ := {x | 3 - 2*x > 0}
  A ∩ B = {x | x < 3/2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3280_328098


namespace NUMINAMATH_CALUDE_cone_base_radius_l3280_328044

/-- 
Given a cone whose lateral surface, when unfolded, is a semicircle with radius 1,
prove that the radius of the base of the cone is 1/2.
-/
theorem cone_base_radius (r : ℝ) : r > 0 → r = 1 → (2 * π * (1 / 2 : ℝ)) = (π * r) → (1 / 2 : ℝ) = r := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3280_328044


namespace NUMINAMATH_CALUDE_connie_markers_total_l3280_328079

theorem connie_markers_total (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ)
  (h_red : red = 5420)
  (h_blue : blue = 3875)
  (h_green : green = 2910)
  (h_yellow : yellow = 6740) :
  red + blue + green + yellow = 18945 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_total_l3280_328079


namespace NUMINAMATH_CALUDE_certain_number_solution_l3280_328035

theorem certain_number_solution : 
  ∃ x : ℝ, (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l3280_328035


namespace NUMINAMATH_CALUDE_painting_selection_theorem_l3280_328027

/-- Number of traditional Chinese paintings -/
def traditional_paintings : ℕ := 5

/-- Number of oil paintings -/
def oil_paintings : ℕ := 2

/-- Number of watercolor paintings -/
def watercolor_paintings : ℕ := 7

/-- Number of ways to choose one painting from each category -/
def one_from_each : ℕ := traditional_paintings * oil_paintings * watercolor_paintings

/-- Number of ways to choose two paintings of different types -/
def two_different_types : ℕ := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem : 
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end NUMINAMATH_CALUDE_painting_selection_theorem_l3280_328027


namespace NUMINAMATH_CALUDE_min_employees_for_given_requirements_l3280_328005

/-- Represents the number of employees needed for each pollution type and their intersections -/
structure PollutionMonitoring where
  water : ℕ
  air : ℕ
  soil : ℕ
  water_air : ℕ
  air_soil : ℕ
  soil_water : ℕ
  all_three : ℕ

/-- Calculates the minimum number of employees needed given the monitoring requirements -/
def min_employees (p : PollutionMonitoring) : ℕ :=
  p.water + p.air + p.soil - p.water_air - p.air_soil - p.soil_water + p.all_three

/-- Theorem stating that given the specific monitoring requirements, 225 employees are needed -/
theorem min_employees_for_given_requirements :
  let p : PollutionMonitoring := {
    water := 115,
    air := 92,
    soil := 60,
    water_air := 32,
    air_soil := 20,
    soil_water := 10,
    all_three := 5
  }
  min_employees p = 225 := by
  sorry


end NUMINAMATH_CALUDE_min_employees_for_given_requirements_l3280_328005


namespace NUMINAMATH_CALUDE_heavens_brother_erasers_l3280_328004

def total_money : ℕ := 100
def sharpener_count : ℕ := 2
def notebook_count : ℕ := 4
def item_price : ℕ := 5
def eraser_price : ℕ := 4
def highlighter_cost : ℕ := 30

theorem heavens_brother_erasers :
  let heaven_spent := sharpener_count * item_price + notebook_count * item_price
  let brother_money := total_money - heaven_spent
  let eraser_money := brother_money - highlighter_cost
  eraser_money / eraser_price = 10 := by sorry

end NUMINAMATH_CALUDE_heavens_brother_erasers_l3280_328004


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3280_328070

def polynomial (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + x^3 - 6) + 5*(2*x^4 - 3*x + 1) - 2*(x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 26 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3280_328070


namespace NUMINAMATH_CALUDE_carolyns_silverware_knives_percentage_l3280_328037

/-- The percentage of knives in Carolyn's silverware after a trade --/
theorem carolyns_silverware_knives_percentage 
  (initial_knives : ℕ) 
  (initial_forks : ℕ) 
  (initial_spoons_multiplier : ℕ) 
  (traded_knives : ℕ) 
  (traded_spoons : ℕ) 
  (h1 : initial_knives = 6)
  (h2 : initial_forks = 12)
  (h3 : initial_spoons_multiplier = 3)
  (h4 : traded_knives = 10)
  (h5 : traded_spoons = 6) :
  let initial_spoons := initial_knives * initial_spoons_multiplier
  let final_knives := initial_knives + traded_knives
  let final_spoons := initial_spoons - traded_spoons
  let total_silverware := final_knives + initial_forks + final_spoons
  (final_knives : ℚ) / (total_silverware : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_carolyns_silverware_knives_percentage_l3280_328037


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3280_328033

def set_A : Set ℝ := {x | 2 * x - 1 ≤ 0}
def set_B : Set ℝ := {x | 1 / x > 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3280_328033


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3280_328022

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*x ∧ 6*s^2 = 2*x) → x = 1/972 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3280_328022


namespace NUMINAMATH_CALUDE_binomial_19_13_l3280_328067

theorem binomial_19_13 : Nat.choose 19 13 = 27132 := by
  -- Given conditions
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  have h3 : Nat.choose 18 12 = 18564 := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_binomial_19_13_l3280_328067


namespace NUMINAMATH_CALUDE_not_right_triangle_l3280_328094

theorem not_right_triangle (a b c : ℕ) (h : a = 7 ∧ b = 9 ∧ c = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3280_328094


namespace NUMINAMATH_CALUDE_brick_length_l3280_328019

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a brick with given dimensions and surface area -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 2) (hSA : SA = 112) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 8 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_l3280_328019


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_values_l3280_328048

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_{2n} is constant -/
def constant_ratio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n / a (2 * n) = c

theorem arithmetic_sequence_constant_ratio_values
  (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : constant_ratio a) :
  ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n / a (2 * n) = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_values_l3280_328048


namespace NUMINAMATH_CALUDE_lemoine_point_minimizes_distance_sum_l3280_328069

/-- Given a triangle ABC, this theorem proves that the sum of squares of distances
    from any point to the sides of the triangle is minimized when the distances
    are proportional to the sides, and this point is the Lemoine point. -/
theorem lemoine_point_minimizes_distance_sum (a b c : ℝ) (S_ABC : ℝ) :
  let f (x y z : ℝ) := x^2 + y^2 + z^2
  ∀ (x y z : ℝ), a * x + b * y + c * z = 2 * S_ABC →
  f x y z ≥ f ((2 * S_ABC * a) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * b) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * c) / (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_lemoine_point_minimizes_distance_sum_l3280_328069


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l3280_328051

/-- The modulus of the complex number z = (1-2i)(3+i) is equal to 5√2 -/
theorem modulus_of_complex_product : 
  let z : ℂ := (1 - 2*I) * (3 + I)
  ‖z‖ = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l3280_328051


namespace NUMINAMATH_CALUDE_rectangle_area_l3280_328009

theorem rectangle_area (a b : ℝ) (h1 : (a + b)^2 = 16) (h2 : (a - b)^2 = 4) : a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3280_328009


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l3280_328042

/-- The ratio of the new circumference to the new diameter when the radius is increased by 2 units -/
theorem circle_ratio_after_increase (r : ℝ) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l3280_328042


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3280_328047

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±x is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 1) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3280_328047


namespace NUMINAMATH_CALUDE_walnutridge_team_count_l3280_328087

/-- The number of different teams that can be formed from a math club -/
def number_of_teams (num_girls num_boys : ℕ) : ℕ :=
  (num_girls.choose 2) * (num_boys.choose 2) * 2

/-- The math club at Walnutridge High School -/
structure MathClub where
  num_girls : ℕ
  num_boys : ℕ

/-- The Walnutridge High School math club has 5 girls and 7 boys -/
def walnutridge_math_club : MathClub := ⟨5, 7⟩

theorem walnutridge_team_count :
  number_of_teams walnutridge_math_club.num_girls walnutridge_math_club.num_boys = 420 := by
  sorry

end NUMINAMATH_CALUDE_walnutridge_team_count_l3280_328087


namespace NUMINAMATH_CALUDE_m_range_l3280_328046

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem m_range :
  (∀ x : ℝ, x ≥ 1 → f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3280_328046


namespace NUMINAMATH_CALUDE_f_inequality_l3280_328039

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
variable (h_deriv : ∀ x, HasDerivAt f (f' x) x)

-- Define the condition that f(x) > f'(x) for all x
variable (h_cond : ∀ x, f x > f' x)

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3280_328039


namespace NUMINAMATH_CALUDE_sarah_walking_speed_l3280_328065

theorem sarah_walking_speed (v : ℝ) : 
  v > 0 → -- v is positive (walking speed)
  (6 / v + 6 / 4 = 3.5) → -- total time equation
  v = 3 := by
sorry

end NUMINAMATH_CALUDE_sarah_walking_speed_l3280_328065


namespace NUMINAMATH_CALUDE_max_product_constrained_l3280_328010

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l3280_328010


namespace NUMINAMATH_CALUDE_pears_given_by_mike_l3280_328011

theorem pears_given_by_mike (initial_pears : ℕ) (pears_given_away : ℕ) (final_pears : ℕ) :
  initial_pears = 46 →
  pears_given_away = 47 →
  final_pears = 11 →
  pears_given_away - initial_pears + final_pears = 12 :=
by sorry

end NUMINAMATH_CALUDE_pears_given_by_mike_l3280_328011


namespace NUMINAMATH_CALUDE_trapezoid_area_l3280_328092

/-- The area of a trapezoid with height h, bases 4h + 2 and 5h is (9h^2 + 2h) / 2 -/
theorem trapezoid_area (h : ℝ) : 
  let base1 := 4 * h + 2
  let base2 := 5 * h
  ((base1 + base2) / 2) * h = (9 * h^2 + 2 * h) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3280_328092


namespace NUMINAMATH_CALUDE_balloon_ratio_l3280_328082

theorem balloon_ratio (sally_balloons fred_balloons : ℕ) 
  (h1 : sally_balloons = 6) 
  (h2 : fred_balloons = 18) : 
  (fred_balloons : ℚ) / sally_balloons = 3 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_l3280_328082


namespace NUMINAMATH_CALUDE_triangle_problem_l3280_328076

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  True

-- Define the theorem
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_a : a = 8)
  (h_bc : b - c = 2)
  (h_cosA : Real.cos A = -1/4) :
  Real.sin B = (3 * Real.sqrt 15) / 16 ∧ 
  Real.cos (2 * A + π/6) = -(7 * Real.sqrt 3) / 16 - (Real.sqrt 15) / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3280_328076


namespace NUMINAMATH_CALUDE_negation_equivalence_l3280_328073

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3280_328073


namespace NUMINAMATH_CALUDE_f_properties_l3280_328078

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (period : ℝ),
    (∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
    (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ = 2) ∧
    (∃ (x₂ : ℝ), x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₂ = -1) ∧
    (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
      f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3280_328078


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3280_328021

/-- 
Given a rectangle where:
- The long sides are three times the length of the short sides
- One short side is 80 feet long
Prove that the perimeter of the rectangle is 640 feet.
-/
theorem rectangle_perimeter (short_side : ℝ) (h1 : short_side = 80) : 
  2 * short_side + 2 * (3 * short_side) = 640 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l3280_328021


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l3280_328041

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percent : ℝ := 50

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket_percent = exceed_limit_percent * (100 - no_ticket_percent) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l3280_328041


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3280_328000

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, k * x + y = k * x + 2 → a = b) →
  (∃ c : ℝ, c^2 = 24 - 16 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3280_328000


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3280_328075

theorem termite_ridden_not_collapsing (total_homes : ℚ) 
  (termite_ridden_ratio : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden_ratio = 1/3 →
  collapsing_ratio = 5/8 →
  termite_ridden_ratio - (termite_ridden_ratio * collapsing_ratio) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3280_328075


namespace NUMINAMATH_CALUDE_gymnasium_doubles_players_l3280_328074

theorem gymnasium_doubles_players (total_tables : ℕ) 
  (h1 : total_tables = 13) 
  (h2 : ∀ x y : ℕ, x + y = total_tables → 4 * x - 2 * y = 4 → 4 * x = 20) :
  ∃ x y : ℕ, x + y = total_tables ∧ 4 * x - 2 * y = 4 ∧ 4 * x = 20 :=
sorry

end NUMINAMATH_CALUDE_gymnasium_doubles_players_l3280_328074


namespace NUMINAMATH_CALUDE_sum_in_range_l3280_328097

theorem sum_in_range : 
  let sum := (17/4 : ℚ) + (11/4 : ℚ) + (57/8 : ℚ)
  14 < sum ∧ sum < 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l3280_328097


namespace NUMINAMATH_CALUDE_twentieth_number_is_381_l3280_328089

/-- The last number of the nth row in the sequence -/
def last_number (n : ℕ) : ℕ := n^2

/-- The 20th number in the 20th row of the sequence -/
def twentieth_number : ℕ := last_number 19 + 20

theorem twentieth_number_is_381 : twentieth_number = 381 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_number_is_381_l3280_328089


namespace NUMINAMATH_CALUDE_andrews_age_l3280_328062

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 10 * andrew_age →
  grandfather_age - andrew_age = 63 →
  andrew_age = 7 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l3280_328062


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3280_328018

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  (x^2 - 26*x + 81 = (x + b)^2 + c) ∧ (b + c = -101) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3280_328018


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3280_328064

theorem circle_area_from_circumference (c : ℝ) (h : c = 36) : 
  (c^2 / (4 * π)) = 324 / π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3280_328064


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3280_328060

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₇ = 3 and a₁₉ = 2011, prove that a₁₃ = 1007 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_7 : a 7 = 3) 
  (h_19 : a 19 = 2011) : 
  a 13 = 1007 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3280_328060


namespace NUMINAMATH_CALUDE_white_balls_count_l3280_328095

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_or_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 8 →
  red = 9 →
  purple = 3 →
  prob_not_red_or_purple = 88/100 →
  total - (green + yellow + red + purple) = 50 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l3280_328095


namespace NUMINAMATH_CALUDE_sara_bouquets_l3280_328090

theorem sara_bouquets (red yellow blue : ℕ) 
  (h_red : red = 42) 
  (h_yellow : yellow = 63) 
  (h_blue : blue = 54) : 
  Nat.gcd red (Nat.gcd yellow blue) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_bouquets_l3280_328090


namespace NUMINAMATH_CALUDE_group_size_problem_l3280_328056

theorem group_size_problem (total_rupees : ℚ) (paise_per_rupee : ℕ) : 
  total_rupees = 92.16 ∧ paise_per_rupee = 100 → 
  ∃ n : ℕ, n * n = total_rupees * paise_per_rupee ∧ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3280_328056


namespace NUMINAMATH_CALUDE_spring_problem_l3280_328032

/-- Represents the length of a spring as a function of mass -/
def spring_length (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem spring_problem (k : ℝ) :
  spring_length k 6 0 = 6 →
  spring_length k 6 4 = 7.2 →
  spring_length k 6 5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_problem_l3280_328032


namespace NUMINAMATH_CALUDE_expression_evaluation_l3280_328083

theorem expression_evaluation : ((-2)^2)^(1^(0^2)) + 3^(0^(1^2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3280_328083


namespace NUMINAMATH_CALUDE_largest_integer_from_averages_l3280_328053

theorem largest_integer_from_averages : 
  ∀ w x y z : ℤ,
  (w + x + y) / 3 = 32 →
  (w + x + z) / 3 = 39 →
  (w + y + z) / 3 = 40 →
  (x + y + z) / 3 = 44 →
  max w (max x (max y z)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_from_averages_l3280_328053


namespace NUMINAMATH_CALUDE_largest_A_k_l3280_328040

-- Define A_k
def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

-- State the theorem
theorem largest_A_k : ∃ (k : ℕ), k = 166 ∧ ∀ (j : ℕ), j ≠ k → j ≤ 1000 → A k ≥ A j := by
  sorry

end NUMINAMATH_CALUDE_largest_A_k_l3280_328040


namespace NUMINAMATH_CALUDE_race_theorem_l3280_328081

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance_run : ℝ → ℝ

/-- The race setup -/
structure Race where
  petya : Runner
  kolya : Runner
  vasya : Runner
  race_distance : ℝ

def Race.valid (r : Race) : Prop :=
  r.race_distance = 100 ∧
  r.petya.speed > 0 ∧ r.kolya.speed > 0 ∧ r.vasya.speed > 0 ∧
  r.petya.distance_run 0 = 0 ∧ r.kolya.distance_run 0 = 0 ∧ r.vasya.distance_run 0 = 0 ∧
  ∀ t, r.petya.distance_run t = r.petya.speed * t ∧
       r.kolya.distance_run t = r.kolya.speed * t ∧
       r.vasya.distance_run t = r.vasya.speed * t

def Race.petya_finishes_first (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance ∧
       r.kolya.distance_run t < r.race_distance ∧
       r.vasya.distance_run t < r.race_distance

def Race.half_distance_condition (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance / 2 ∧
       r.kolya.distance_run t + r.vasya.distance_run t = 85

theorem race_theorem (r : Race) (h_valid : r.valid) (h_first : r.petya_finishes_first)
    (h_half : r.half_distance_condition) :
    ∃ t, r.petya.distance_run t = r.race_distance ∧
         2 * r.race_distance - (r.kolya.distance_run t + r.vasya.distance_run t) = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l3280_328081


namespace NUMINAMATH_CALUDE_subtract_three_five_l3280_328008

theorem subtract_three_five : 3 - 5 = -2 := by sorry

end NUMINAMATH_CALUDE_subtract_three_five_l3280_328008
