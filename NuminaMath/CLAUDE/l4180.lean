import Mathlib

namespace NUMINAMATH_CALUDE_nine_a_value_l4180_418028

theorem nine_a_value (a b : ℚ) (eq1 : 8 * a + 3 * b = 0) (eq2 : a = b - 3) : 9 * a = -81 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_a_value_l4180_418028


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l4180_418066

/-- Represents the number of handshakes in a gymnastics championship. -/
def total_handshakes : ℕ := 456

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts. -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the total number of gymnasts. -/
def total_gymnasts : ℕ := 30

/-- Represents the number of handshakes involving coaches. -/
def coach_handshakes : ℕ := total_handshakes - gymnast_handshakes total_gymnasts

/-- Theorem stating the minimum number of handshakes involving at least one coach. -/
theorem min_coach_handshakes : ∃ (k₁ k₂ : ℕ), k₁ + k₂ = coach_handshakes ∧ min k₁ k₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l4180_418066


namespace NUMINAMATH_CALUDE_remainder_of_198_digit_sequence_l4180_418070

/-- The sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The function that generates the sequence of digits up to the nth digit -/
def sequenceUpTo (n : ℕ) : List ℕ := sorry

/-- Sum of all digits in the sequence up to the nth digit -/
def sumOfSequenceDigits (n : ℕ) : ℕ := 
  (sequenceUpTo n).map sumOfDigits |>.sum

theorem remainder_of_198_digit_sequence : 
  sumOfSequenceDigits 198 % 9 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_of_198_digit_sequence_l4180_418070


namespace NUMINAMATH_CALUDE_even_quadratic_implies_b_zero_l4180_418081

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem even_quadratic_implies_b_zero (b c : ℝ) :
  IsEven (f b c) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_b_zero_l4180_418081


namespace NUMINAMATH_CALUDE_decimal_arithmetic_proof_l4180_418023

theorem decimal_arithmetic_proof : (5.92 + 2.4) - 3.32 = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_proof_l4180_418023


namespace NUMINAMATH_CALUDE_polly_cooking_time_l4180_418036

/-- Represents the cooking times for Polly in a week -/
structure CookingTimes where
  breakfast_time : ℕ  -- Time spent cooking breakfast daily
  lunch_time : ℕ      -- Time spent cooking lunch daily
  dinner_time_short : ℕ  -- Time spent cooking dinner on short days
  dinner_time_long : ℕ   -- Time spent cooking dinner on long days
  short_dinner_days : ℕ  -- Number of days with short dinner cooking time
  long_dinner_days : ℕ   -- Number of days with long dinner cooking time

/-- Calculates the total cooking time for a week given the cooking times -/
def total_cooking_time (times : CookingTimes) : ℕ :=
  7 * (times.breakfast_time + times.lunch_time) +
  times.short_dinner_days * times.dinner_time_short +
  times.long_dinner_days * times.dinner_time_long

/-- Theorem stating that Polly's total cooking time for the week is 305 minutes -/
theorem polly_cooking_time :
  ∀ (times : CookingTimes),
  times.breakfast_time = 20 ∧
  times.lunch_time = 5 ∧
  times.dinner_time_short = 10 ∧
  times.dinner_time_long = 30 ∧
  times.short_dinner_days = 4 ∧
  times.long_dinner_days = 3 →
  total_cooking_time times = 305 := by
  sorry


end NUMINAMATH_CALUDE_polly_cooking_time_l4180_418036


namespace NUMINAMATH_CALUDE_thirteen_to_six_div_three_l4180_418068

theorem thirteen_to_six_div_three (x : ℕ) : 13^6 / 13^3 = 2197 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_to_six_div_three_l4180_418068


namespace NUMINAMATH_CALUDE_inequality_solution_l4180_418058

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom g_never_zero : ∀ x, g x ≠ 0
axiom condition_neg : ∀ x, x < 0 → f x * g x - f x * (deriv g x) > 0
axiom f_3_eq_0 : f 3 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ (0 < x ∧ x < 3)}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x * g x < 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4180_418058


namespace NUMINAMATH_CALUDE_square_of_binomial_l4180_418049

theorem square_of_binomial (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16*x^2 + 40*x + b = (a*x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l4180_418049


namespace NUMINAMATH_CALUDE_constant_product_l4180_418007

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

-- Define the symmetry axis
def symmetry_axis : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - 3 * p.2 + 6 = 0}

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define line m
def line_m : Set (ℝ × ℝ) :=
  {p | p.1 + 2 * p.2 + 2 = 0}

-- Define the theorem
theorem constant_product :
  ∀ l : Set (ℝ × ℝ),
  (P ∈ l) →
  (∃ A B : ℝ × ℝ,
    A ∈ circle_C ∧
    B ∈ line_m ∧
    A ∈ l ∧
    B ∈ l ∧
    (∃ C : ℝ × ℝ, C ∈ circle_C ∧ C ∈ l ∧ A ≠ C ∧ 
      A = ((C.1 + A.1) / 2, (C.2 + A.2) / 2)) →
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 6)) :=
sorry


end NUMINAMATH_CALUDE_constant_product_l4180_418007


namespace NUMINAMATH_CALUDE_extremum_point_implies_inequality_non_negative_function_implies_m_range_l4180_418057

noncomputable section

variable (m : ℝ)
def f (x : ℝ) : ℝ := Real.exp (x + m) - Real.log x

def a : ℝ := Real.exp (1 / Real.exp 1)

theorem extremum_point_implies_inequality :
  (∃ (m : ℝ), f m 1 = 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m 1)) →
  ∀ (x : ℝ), x > 0 → Real.exp x - Real.exp 1 * Real.log x ≥ Real.exp 1 :=
sorry

theorem non_negative_function_implies_m_range :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m x₀)) →
  (∀ (x : ℝ), x > 0 → f m x ≥ 0) →
  m ≥ -a - Real.log a :=
sorry

end

end NUMINAMATH_CALUDE_extremum_point_implies_inequality_non_negative_function_implies_m_range_l4180_418057


namespace NUMINAMATH_CALUDE_income_growth_equation_l4180_418022

theorem income_growth_equation (x : ℝ) : 
  let initial_income : ℝ := 12000
  let final_income : ℝ := 14520
  initial_income * (1 + x)^2 = final_income := by
  sorry

end NUMINAMATH_CALUDE_income_growth_equation_l4180_418022


namespace NUMINAMATH_CALUDE_smallest_candy_count_l4180_418099

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0) → False) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l4180_418099


namespace NUMINAMATH_CALUDE_principal_calculation_l4180_418030

/-- Proves that given the conditions of the problem, the principal must be 2400 --/
theorem principal_calculation (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 1920 → P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l4180_418030


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l4180_418018

-- Define a function to represent mixed numbers
def mixed_number (whole : Int) (numerator : Int) (denominator : Int) : Rat :=
  whole + (numerator : Rat) / (denominator : Rat)

-- Problem 1
theorem problem_one : 
  mixed_number 28 5 7 + mixed_number (-25) (-1) 7 = mixed_number 3 4 7 := by
  sorry

-- Problem 2
theorem problem_two :
  mixed_number (-2022) (-2) 7 + mixed_number (-2023) (-4) 7 + (4046 : Rat) + (-1 : Rat) / 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l4180_418018


namespace NUMINAMATH_CALUDE_jake_weight_proof_l4180_418067

/-- Jake's weight in pounds -/
def jake_weight : ℝ := 230

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 111

/-- Jake's brother's weight in pounds -/
def brother_weight : ℝ := 139

theorem jake_weight_proof :
  -- Condition 1: If Jake loses 8 pounds, he will weigh twice as much as his sister
  jake_weight - 8 = 2 * sister_weight ∧
  -- Condition 2: Jake's brother is currently 6 pounds heavier than twice Jake's weight
  brother_weight = 2 * jake_weight + 6 ∧
  -- Condition 3: Together, all three of them now weigh 480 pounds
  jake_weight + sister_weight + brother_weight = 480 ∧
  -- Condition 4: The brother's weight is 125% of the sister's weight
  brother_weight = 1.25 * sister_weight →
  -- Conclusion: Jake's weight is 230 pounds
  jake_weight = 230 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l4180_418067


namespace NUMINAMATH_CALUDE_problems_completed_is_120_l4180_418075

/-- The number of problems completed given the conditions in the problem -/
def problems_completed (p t : ℕ) : ℕ := p * t

/-- The conditions of the problem -/
def problem_conditions (p t : ℕ) : Prop :=
  p > 15 ∧ t > 0 ∧ p * t = (3 * p - 6) * (t - 3)

/-- The theorem stating that under the given conditions, 120 problems are completed -/
theorem problems_completed_is_120 :
  ∃ p t : ℕ, problem_conditions p t ∧ problems_completed p t = 120 :=
sorry

end NUMINAMATH_CALUDE_problems_completed_is_120_l4180_418075


namespace NUMINAMATH_CALUDE_min_value_at_three_l4180_418098

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 27

/-- The theorem stating that f(x) is minimized when x = 3 -/
theorem min_value_at_three :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_at_three_l4180_418098


namespace NUMINAMATH_CALUDE_roses_given_to_friends_l4180_418038

def total_money : ℕ := 300
def rose_price : ℕ := 2
def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/2

theorem roses_given_to_friends :
  let total_roses := total_money / rose_price
  let jenna_roses := (jenna_fraction * total_roses).floor
  let imma_roses := (imma_fraction * total_roses).floor
  jenna_roses + imma_roses = 125 := by sorry

end NUMINAMATH_CALUDE_roses_given_to_friends_l4180_418038


namespace NUMINAMATH_CALUDE_product_plus_one_l4180_418014

theorem product_plus_one (m n : ℕ) (h : m * n = 121) : (m + 1) * (n + 1) = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_l4180_418014


namespace NUMINAMATH_CALUDE_candy_probability_l4180_418032

def yellow_candies : ℕ := 2
def red_candies : ℕ := 4

def total_candies : ℕ := yellow_candies + red_candies

def favorable_arrangements : ℕ := 1

def total_arrangements : ℕ := Nat.choose total_candies yellow_candies

def probability : ℚ := favorable_arrangements / total_arrangements

theorem candy_probability : probability = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_candy_probability_l4180_418032


namespace NUMINAMATH_CALUDE_solve_apple_problem_l4180_418010

def apple_problem (initial_apples : ℕ) (pears_difference : ℕ) (pears_bought : ℕ) (final_total : ℕ) : Prop :=
  let initial_pears : ℕ := initial_apples + pears_difference
  let new_pears : ℕ := initial_pears + pears_bought
  let apples_sold : ℕ := initial_apples + new_pears - final_total
  apples_sold = 599

theorem solve_apple_problem :
  apple_problem 1238 374 276 2527 :=
by sorry

end NUMINAMATH_CALUDE_solve_apple_problem_l4180_418010


namespace NUMINAMATH_CALUDE_evaluate_expression_l4180_418094

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3/2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4180_418094


namespace NUMINAMATH_CALUDE_second_to_third_quadrant_l4180_418083

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrants
def isSecondQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0
def isThirdQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y < 0

-- Define the transformation from P to Q
def transformPtoQ (p : Point2D) : Point2D :=
  { x := -p.y, y := p.x }

-- Theorem statement
theorem second_to_third_quadrant (a b : ℝ) :
  let p := Point2D.mk a b
  let q := transformPtoQ p
  isSecondQuadrant p → isThirdQuadrant q := by
  sorry

end NUMINAMATH_CALUDE_second_to_third_quadrant_l4180_418083


namespace NUMINAMATH_CALUDE_beidou_chip_scientific_notation_correct_l4180_418015

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The value of the "Fourth Generation Beidou Chip" size in meters -/
def beidou_chip_size : ℝ := 0.000000022

/-- The scientific notation representation of the Beidou chip size -/
def beidou_chip_scientific : ScientificNotation :=
  { coefficient := 2.2
    exponent := -8
    is_valid := by sorry }

theorem beidou_chip_scientific_notation_correct :
  beidou_chip_size = beidou_chip_scientific.coefficient * (10 : ℝ) ^ beidou_chip_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_beidou_chip_scientific_notation_correct_l4180_418015


namespace NUMINAMATH_CALUDE_curve_symmetric_about_y_axis_l4180_418052

theorem curve_symmetric_about_y_axis : ∀ x y : ℝ, x^2 - y^2 = 1 ↔ (-x)^2 - y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_y_axis_l4180_418052


namespace NUMINAMATH_CALUDE_special_function_range_l4180_418009

open Set Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  condition1 : ∀ x, f (-x) / f x = exp (2 * x)
  condition2 : ∀ x, x < 0 → f x + deriv f x > 0

/-- The theorem statement -/
theorem special_function_range (sf : SpecialFunction) :
  {a : ℝ | exp a * sf.f (2 * a + 1) ≥ sf.f (a + 1)} = Icc (-2/3) 0 :=
sorry

end NUMINAMATH_CALUDE_special_function_range_l4180_418009


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4180_418092

-- Problem 1
theorem problem_1 : 42.67 - (12.95 - 7.33) = 37.05 := by sorry

-- Problem 2
theorem problem_2 : (8.4 - 8.4 * (3.12 - 3.7)) / 0.42 = 31.6 := by sorry

-- Problem 3
theorem problem_3 : 5.13 * 0.23 + 8.7 * 0.513 - 5.13 = 0.513 := by sorry

-- Problem 4
theorem problem_4 : 6.66 * 222 + 3.33 * 556 = 3330 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4180_418092


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_constrained_achieved_l4180_418073

theorem max_product_constrained (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → ab ≤ 1/2 := by
  sorry

theorem max_product_constrained_achieved (a b : ℝ) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ ab = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_constrained_achieved_l4180_418073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equality_l4180_418091

/-- Given an arithmetic sequence with 3n terms, prove that t₁ = t₂ -/
theorem arithmetic_sequence_equality (n : ℕ) (s₁ s₂ s₃ : ℝ) 
  (h₁ : s₁ + s₃ = 2 * s₂) -- Property of arithmetic sequence
  (t₁ t₂ : ℝ)
  (h₂ : t₁ = s₂^2 - s₁*s₃)
  (h₃ : t₂ = ((s₁ - s₃) / 2)^2) :
  t₁ = t₂ := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equality_l4180_418091


namespace NUMINAMATH_CALUDE_rachel_tips_l4180_418055

theorem rachel_tips (hourly_wage : ℚ) (people_served : ℕ) (total_made : ℚ) 
  (hw : hourly_wage = 12)
  (ps : people_served = 20)
  (tm : total_made = 37) :
  (total_made - hourly_wage) / people_served = 25 / 20 := by
  sorry

#eval (37 : ℚ) - 12
#eval (25 : ℚ) / 20

end NUMINAMATH_CALUDE_rachel_tips_l4180_418055


namespace NUMINAMATH_CALUDE_b_current_age_l4180_418034

/-- Given two people A and B, proves that B's current age is 39 years
    under the given conditions. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age from 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 :=                  -- B's current age is 39 years
by sorry

end NUMINAMATH_CALUDE_b_current_age_l4180_418034


namespace NUMINAMATH_CALUDE_hundreds_digit_of_expression_l4180_418093

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

-- State the theorem
theorem hundreds_digit_of_expression :
  hundreds_digit ((factorial 17 / 5) - (factorial 10 / 2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_expression_l4180_418093


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l4180_418088

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 7 * a + 2 * b = 52) : 
  a + b = 35 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l4180_418088


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l4180_418077

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 3) = 
  6*x^3 - 16*x^2 + 43*x - 70 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l4180_418077


namespace NUMINAMATH_CALUDE_min_value_of_a_l4180_418079

theorem min_value_of_a (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) →
  a ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l4180_418079


namespace NUMINAMATH_CALUDE_walnut_trees_cut_down_count_l4180_418016

/-- The number of walnut trees cut down in the park --/
def walnut_trees_cut_down (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 13 walnut trees were cut down --/
theorem walnut_trees_cut_down_count : 
  walnut_trees_cut_down 42 29 = 13 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_cut_down_count_l4180_418016


namespace NUMINAMATH_CALUDE_meaningful_range_l4180_418041

def is_meaningful (x : ℝ) : Prop :=
  x + 3 ≥ 0 ∧ x ≠ 1

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -3 ∧ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_range_l4180_418041


namespace NUMINAMATH_CALUDE_prob_different_subjects_is_one_sixth_l4180_418064

/-- The number of subjects available for selection -/
def num_subjects : ℕ := 4

/-- The number of subjects each student selects -/
def subjects_per_student : ℕ := 2

/-- The total number of possible subject selection combinations for one student -/
def total_combinations : ℕ := (num_subjects.choose subjects_per_student)

/-- The total number of possible events (combinations for both students) -/
def total_events : ℕ := total_combinations * total_combinations

/-- The number of events where both students select different subjects -/
def different_subjects_events : ℕ := total_combinations * ((num_subjects - subjects_per_student).choose subjects_per_student)

/-- The probability that the two students select different subjects -/
def prob_different_subjects : ℚ := different_subjects_events / total_events

theorem prob_different_subjects_is_one_sixth : 
  prob_different_subjects = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_different_subjects_is_one_sixth_l4180_418064


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l4180_418087

/-- The equation of a line passing through (0, 2) with slope 2 is y = 2x + 2 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  y - 2 = 2 * (x - 0) → y = 2 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l4180_418087


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4180_418035

theorem sqrt_equation_solution :
  ∀ z : ℝ, (Real.sqrt (3 + z) = 12) ↔ (z = 141) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4180_418035


namespace NUMINAMATH_CALUDE_rotation_composition_implies_triangle_angles_l4180_418054

/-- Represents a rotation in 2D space -/
structure Rotation2D where
  angle : ℝ
  center : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Composition of rotations -/
def compose_rotations (r1 r2 r3 : Rotation2D) : Rotation2D :=
  sorry

/-- Check if a rotation is the identity transformation -/
def is_identity (r : Rotation2D) : Prop :=
  sorry

/-- Get the angle at a vertex of a triangle -/
def angle_at_vertex (t : Triangle) (v : ℝ × ℝ) : ℝ :=
  sorry

theorem rotation_composition_implies_triangle_angles 
  (α β γ : ℝ) (t : Triangle) (r_A r_B r_C : Rotation2D) :
  0 < α ∧ α < π →
  0 < β ∧ β < π →
  0 < γ ∧ γ < π →
  α + β + γ = π →
  r_A.angle = 2 * α →
  r_B.angle = 2 * β →
  r_C.angle = 2 * γ →
  r_A.center = t.A →
  r_B.center = t.B →
  r_C.center = t.C →
  is_identity (compose_rotations r_C r_B r_A) →
  angle_at_vertex t t.A = α ∧
  angle_at_vertex t t.B = β ∧
  angle_at_vertex t t.C = γ :=
by sorry

end NUMINAMATH_CALUDE_rotation_composition_implies_triangle_angles_l4180_418054


namespace NUMINAMATH_CALUDE_counterexample_exists_l4180_418039

theorem counterexample_exists : ∃ (n : ℕ), n ≥ 2 ∧ 
  ∃ (k : ℕ), (2^(2^n) % (2^n - 1) = k) ∧ ¬∃ (m : ℕ), k = 4^m :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4180_418039


namespace NUMINAMATH_CALUDE_congruence_problem_l4180_418045

theorem congruence_problem (x : ℤ) : 
  (3 * x + 7) % 16 = 5 → (4 * x + 3) % 16 = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4180_418045


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l4180_418053

theorem largest_integer_for_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l4180_418053


namespace NUMINAMATH_CALUDE_paths_through_B_and_C_l4180_418078

/-- Represents a point on the square grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a square grid -/
def num_paths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The points on the grid -/
def A : GridPoint := ⟨0, 0⟩
def B : GridPoint := ⟨2, 3⟩
def C : GridPoint := ⟨6, 4⟩
def D : GridPoint := ⟨9, 6⟩

/-- The theorem to be proved -/
theorem paths_through_B_and_C : 
  num_paths A B * num_paths B C * num_paths C D = 500 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_B_and_C_l4180_418078


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l4180_418042

theorem largest_solution_of_equation (x : ℝ) :
  (x / 5 + 1 / (5 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l4180_418042


namespace NUMINAMATH_CALUDE_perfect_square_sum_permutation_l4180_418043

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_permutation (n : ℕ) (p : Fin n → Fin n) : Prop :=
  Function.Bijective p ∧ ∀ i : Fin n, is_perfect_square ((i.val + 1) + (p i).val + 1)

theorem perfect_square_sum_permutation :
  (∃ p : Fin 9 → Fin 9, valid_permutation 9 p) ∧
  (¬ ∃ p : Fin 11 → Fin 11, valid_permutation 11 p) ∧
  (∃ p : Fin 1996 → Fin 1996, valid_permutation 1996 p) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_permutation_l4180_418043


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l4180_418011

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l4180_418011


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4180_418025

theorem trigonometric_simplification :
  (Real.sin (400 * π / 180) * Real.sin (-230 * π / 180)) /
  (Real.cos (850 * π / 180) * Real.tan (-50 * π / 180)) =
  Real.sin (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4180_418025


namespace NUMINAMATH_CALUDE_sum_is_zero_l4180_418008

def circular_sequence (n : ℕ) := Fin n → ℤ

def neighbor_sum_property (s : circular_sequence 14) : Prop :=
  ∀ i : Fin 14, s i = s (i - 1) + s (i + 1)

theorem sum_is_zero (s : circular_sequence 14) 
  (h : neighbor_sum_property s) : 
  (Finset.univ.sum s) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_l4180_418008


namespace NUMINAMATH_CALUDE_complex_addition_proof_l4180_418027

theorem complex_addition_proof : ∃ z : ℂ, 2 * (5 - 3*I) + z = 4 + 11*I :=
by
  use -6 + 17*I
  sorry

end NUMINAMATH_CALUDE_complex_addition_proof_l4180_418027


namespace NUMINAMATH_CALUDE_percentage_difference_l4180_418024

theorem percentage_difference (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwz : w = 1.152 * z) : 
  w = 0.8 * x := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l4180_418024


namespace NUMINAMATH_CALUDE_tangerine_count_l4180_418072

theorem tangerine_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  apples = pears + 21 →
  tangerines = pears + 18 →
  tangerines = 42 := by
sorry

end NUMINAMATH_CALUDE_tangerine_count_l4180_418072


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l4180_418021

/-- Given a circle D with equation x^2 + 4y - 16 = -y^2 + 12x + 16,
    prove that its center (c,d) and radius s satisfy c + d + s = 4 + 6√2 -/
theorem circle_center_radius_sum (x y c d s : ℝ) : 
  (∀ x y, x^2 + 4*y - 16 = -y^2 + 12*x + 16) → 
  ((x - c)^2 + (y - d)^2 = s^2) → 
  c + d + s = 4 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l4180_418021


namespace NUMINAMATH_CALUDE_sliding_window_is_only_translation_l4180_418006

/-- Represents a type of movement --/
inductive Movement
  | PingPongBall
  | SlidingWindow
  | Kite
  | Basketball

/-- Predicate to check if a movement is a translation --/
def isTranslation (m : Movement) : Prop :=
  match m with
  | Movement.SlidingWindow => True
  | _ => False

/-- Theorem stating that only the sliding window movement is a translation --/
theorem sliding_window_is_only_translation :
  ∀ m : Movement, isTranslation m ↔ m = Movement.SlidingWindow :=
sorry

#check sliding_window_is_only_translation

end NUMINAMATH_CALUDE_sliding_window_is_only_translation_l4180_418006


namespace NUMINAMATH_CALUDE_zeros_of_f_l4180_418033

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Theorem statement
theorem zeros_of_f :
  ∃ (a b c : ℝ), (a = -1 ∧ b = 1 ∧ c = 2) ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l4180_418033


namespace NUMINAMATH_CALUDE_mark_fruit_consumption_l4180_418020

/-- Given the total number of fruit pieces, the number kept for next week,
    and the number brought to school on Friday, calculate the number of
    pieces eaten in the first four days. -/
def fruitEatenInFourDays (total : ℕ) (keptForNextWeek : ℕ) (broughtFriday : ℕ) : ℕ :=
  total - keptForNextWeek - broughtFriday

/-- Theorem stating that given 10 pieces of fruit, if 2 are kept for next week
    and 3 are brought to school on Friday, then 5 pieces were eaten in the first four days. -/
theorem mark_fruit_consumption :
  fruitEatenInFourDays 10 2 3 = 5 := by
  sorry

#eval fruitEatenInFourDays 10 2 3

end NUMINAMATH_CALUDE_mark_fruit_consumption_l4180_418020


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l4180_418013

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- The unique solution is x = 33
  use 33
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l4180_418013


namespace NUMINAMATH_CALUDE_chosen_number_proof_l4180_418037

theorem chosen_number_proof : ∃ x : ℝ, (x / 5) - 154 = 6 ∧ x = 800 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l4180_418037


namespace NUMINAMATH_CALUDE_probability_of_selection_for_student_survey_l4180_418084

/-- Represents a simple random sampling without replacement -/
structure SimpleRandomSampling where
  population : ℕ
  sample_size : ℕ
  h_sample_size_le_population : sample_size ≤ population

/-- The probability of a specific item being selected in a simple random sampling without replacement -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population

theorem probability_of_selection_for_student_survey :
  let srs : SimpleRandomSampling := {
    population := 303,
    sample_size := 50,
    h_sample_size_le_population := by sorry
  }
  probability_of_selection srs = 50 / 303 := by sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_student_survey_l4180_418084


namespace NUMINAMATH_CALUDE_sequence_properties_l4180_418051

/-- Given a sequence a_n with the specified properties, prove the geometric sequence property and the sum formula. -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, S (n + 1) + a n = S n + 5 * (4 ^ n)) :
  (∀ n : ℕ, a (n + 1) - 4^(n + 1) = -(a n - 4^n)) ∧ 
  (∀ n : ℕ, S n = (4^(n + 1) / 3) - ((-1)^(n + 1) / 2) - (11 / 6)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4180_418051


namespace NUMINAMATH_CALUDE_function_inequality_l4180_418065

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x) < 0 for all x in ℝ,
    prove that f(m-m^2) / e^(m^2-m+1) > f(1) for all m in ℝ. -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, deriv f x + f x < 0) :
    ∀ m, f (m - m^2) / Real.exp (m^2 - m + 1) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4180_418065


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l4180_418097

theorem opposite_of_negative_six (m : ℤ) : (m + (-6) = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l4180_418097


namespace NUMINAMATH_CALUDE_f_1000_value_l4180_418031

def is_multiplicative_to_additive (f : ℕ+ → ℕ) : Prop :=
  ∀ x y : ℕ+, f (x * y) = f x + f y

theorem f_1000_value
  (f : ℕ+ → ℕ)
  (h_mult_add : is_multiplicative_to_additive f)
  (h_10 : f 10 = 16)
  (h_40 : f 40 = 22) :
  f 1000 = 48 :=
sorry

end NUMINAMATH_CALUDE_f_1000_value_l4180_418031


namespace NUMINAMATH_CALUDE_construction_worker_wage_l4180_418059

/-- Proves that the daily wage of a construction worker is $100 --/
theorem construction_worker_wage :
  ∀ (worker_wage : ℝ),
  (2 * worker_wage) +  -- Two construction workers
  (2 * worker_wage) +  -- Electrician (double the worker's wage)
  (2.5 * worker_wage) = 650 →  -- Plumber (250% of worker's wage)
  worker_wage = 100 := by
sorry

end NUMINAMATH_CALUDE_construction_worker_wage_l4180_418059


namespace NUMINAMATH_CALUDE_jesse_stamp_ratio_l4180_418056

theorem jesse_stamp_ratio :
  let total_stamps : ℕ := 444
  let european_stamps : ℕ := 333
  let asian_stamps : ℕ := total_stamps - european_stamps
  (european_stamps : ℚ) / (asian_stamps : ℚ) = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jesse_stamp_ratio_l4180_418056


namespace NUMINAMATH_CALUDE_coupon_value_is_correct_l4180_418029

/-- Represents the grocery shopping scenario --/
def grocery_shopping (initial_amount milk_price bread_price detergent_price banana_price_per_pound banana_pounds leftover coupon_value : ℚ) : Prop :=
  let milk_discounted := milk_price / 2
  let banana_total := banana_price_per_pound * banana_pounds
  let total_without_coupon := milk_discounted + bread_price + detergent_price + banana_total
  let amount_spent := initial_amount - leftover
  total_without_coupon - amount_spent = coupon_value

/-- Theorem stating that the coupon value is $1.25 --/
theorem coupon_value_is_correct : 
  grocery_shopping 20 4 3.5 10.25 0.75 2 4 1.25 := by sorry

end NUMINAMATH_CALUDE_coupon_value_is_correct_l4180_418029


namespace NUMINAMATH_CALUDE_clock_adjustment_theorem_l4180_418063

/-- Represents the gain of the clock in minutes per day -/
def clock_gain : ℚ := 13/4

/-- Represents the number of days between May 1st 10 A.M. and May 10th 2 P.M. -/
def days : ℚ := 9 + 4/24

/-- Calculates the adjustment needed for the clock -/
def adjustment (gain : ℚ) (time : ℚ) : ℚ := gain * time

/-- Theorem stating that the adjustment is approximately 29.8 minutes -/
theorem clock_adjustment_theorem :
  ∃ ε > 0, abs (adjustment clock_gain days - 29.8) < ε :=
sorry

end NUMINAMATH_CALUDE_clock_adjustment_theorem_l4180_418063


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l4180_418001

theorem price_reduction_sales_increase (price_reduction : Real) 
  (sales_increase : Real) (net_sale_increase : Real) :
  price_reduction = 20 → 
  net_sale_increase = 44 → 
  (1 - price_reduction / 100) * (1 + sales_increase / 100) = 1 + net_sale_increase / 100 →
  sales_increase = 80 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l4180_418001


namespace NUMINAMATH_CALUDE_inverse_g_one_over_120_l4180_418047

noncomputable def g (x : ℝ) : ℝ := (x^5 + 1) / 5

theorem inverse_g_one_over_120 :
  g⁻¹ (1/120) = ((-23/24) : ℝ)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_one_over_120_l4180_418047


namespace NUMINAMATH_CALUDE_root_existence_implies_m_range_l4180_418082

theorem root_existence_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0) → -3 ≤ m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_implies_m_range_l4180_418082


namespace NUMINAMATH_CALUDE_car_journey_equation_l4180_418076

theorem car_journey_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 120
  let slow_car_speed : ℝ := x
  let fast_car_speed : ℝ := 1.5 * x
  let slow_car_delay : ℝ := 1
  let slow_car_travel_time : ℝ := distance / slow_car_speed - slow_car_delay
  let fast_car_travel_time : ℝ := distance / fast_car_speed
  slow_car_travel_time = fast_car_travel_time :=
by sorry

end NUMINAMATH_CALUDE_car_journey_equation_l4180_418076


namespace NUMINAMATH_CALUDE_pen_cost_l4180_418012

theorem pen_cost (pen_cost ink_cost : ℝ) 
  (total_cost : pen_cost + ink_cost = 1.10)
  (price_difference : pen_cost = ink_cost + 1) : 
  pen_cost = 1.05 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l4180_418012


namespace NUMINAMATH_CALUDE_changsha_gdp_scientific_notation_l4180_418019

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The GDP value of Changsha city in 2022 -/
def changsha_gdp : ℕ := 1400000000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem changsha_gdp_scientific_notation :
  to_scientific_notation changsha_gdp =
    ScientificNotation.mk 1.4 12 (by norm_num) (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_changsha_gdp_scientific_notation_l4180_418019


namespace NUMINAMATH_CALUDE_log_equation_solution_l4180_418017

theorem log_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x > 1, 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 10 * (Real.log x)^2 / (Real.log a + Real.log b)) →
  b = a^((5 + Real.sqrt 10) / 3) ∨ b = a^((5 - Real.sqrt 10) / 3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4180_418017


namespace NUMINAMATH_CALUDE_square_area_8m_l4180_418071

theorem square_area_8m (side_length : ℝ) (area : ℝ) : 
  side_length = 8 → area = side_length ^ 2 → area = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_8m_l4180_418071


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l4180_418085

theorem smallest_n_perfect_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 625000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l4180_418085


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_x_cubed_l4180_418048

theorem integral_sqrt_4_minus_x_squared_plus_x_cubed : 
  ∫ x in (-1)..1, (Real.sqrt (4 - x^2) + x^3) = Real.sqrt 3 + (2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_x_cubed_l4180_418048


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l4180_418080

theorem sine_cosine_inequality (c : ℝ) :
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) ↔ c > 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l4180_418080


namespace NUMINAMATH_CALUDE_soccer_ball_seams_soccer_ball_seams_eq_90_l4180_418089

/-- The number of seams needed to make a soccer ball with pentagons and hexagons -/
theorem soccer_ball_seams (num_pentagons num_hexagons : ℕ) 
  (h_pentagons : num_pentagons = 12)
  (h_hexagons : num_hexagons = 20) : ℕ :=
  let total_sides := num_pentagons * 5 + num_hexagons * 6
  total_sides / 2

/-- Proof that a soccer ball with 12 pentagons and 20 hexagons requires 90 seams -/
theorem soccer_ball_seams_eq_90 :
  soccer_ball_seams 12 20 rfl rfl = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_seams_soccer_ball_seams_eq_90_l4180_418089


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4180_418090

theorem sufficient_but_not_necessary (x : ℝ) :
  (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ∧
  ¬(x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4180_418090


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4180_418061

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 1 > 3*x₀) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4180_418061


namespace NUMINAMATH_CALUDE_inequality_theorem_l4180_418086

theorem inequality_theorem (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l4180_418086


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_nine_l4180_418050

theorem difference_of_squares_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_nine_l4180_418050


namespace NUMINAMATH_CALUDE_min_t_value_l4180_418062

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem min_t_value : 
  (∃ (t : ℝ), ∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ (s : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ s) → s ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l4180_418062


namespace NUMINAMATH_CALUDE_triangle_area_l4180_418003

/-- Given a triangle ABC with side lengths b and c, and angle C, prove that its area is √3/4 -/
theorem triangle_area (b c : ℝ) (C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4180_418003


namespace NUMINAMATH_CALUDE_distance_between_vertices_l4180_418000

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 20

-- Define the vertices of the two graphs
def C : ℝ × ℝ := (2, f 2)
def D : ℝ × ℝ := (-3, g (-3))

-- Theorem statement
theorem distance_between_vertices : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l4180_418000


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_to_geometric_l4180_418095

/-- Represents a geometric progression with first term a and common ratio q -/
structure GeometricProgression (α : Type*) [Field α] where
  a : α
  q : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  2 * y = x + z

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y * y = x * z

theorem geometric_to_arithmetic_to_geometric 
  {α : Type*} [Field α] (gp : GeometricProgression α) :
  is_arithmetic_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2) ∧
  is_geometric_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2 + 9) →
  (gp.a = 64 ∧ gp.q = 5/4) ∨ (gp.a = 64/25 ∧ gp.q = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_to_geometric_l4180_418095


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l4180_418040

/-- Two lines in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (l1 l2 : Line) : 
  l1 = Line.mk 2 (-1) → 
  l2 = Line.mk a 1 → 
  parallel l1 l2 → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l4180_418040


namespace NUMINAMATH_CALUDE_expression_simplification_find_k_value_l4180_418074

-- Problem 1: Simplify the expression
theorem expression_simplification (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 :=
by sorry

-- Problem 2: Find the value of k
theorem find_k_value (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k-1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_find_k_value_l4180_418074


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4180_418069

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 10 → b = 10 → c = 5 →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4180_418069


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l4180_418005

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If points A(1, 1), B(-4, 5), and C(x, 13) are collinear, then x = -14 --/
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear 1 1 (-4) 5 x 13 → x = -14 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l4180_418005


namespace NUMINAMATH_CALUDE_num_tilings_div_by_eight_l4180_418044

/-- A tromino is an L-shaped tile covering exactly three cells -/
structure Tromino :=
  (shape : List (Int × Int))
  (shape_size : shape.length = 3)

/-- A tiling of a square grid using trominos -/
def Tiling (n : Nat) := List (List (Option Tromino))

/-- The size of the square grid -/
def gridSize : Nat := 999

/-- The number of distinct tilings of an n x n grid using trominos -/
def numDistinctTilings (n : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct tilings of a 999x999 grid using trominos is divisible by 8 -/
theorem num_tilings_div_by_eight :
  ∃ k : Nat, numDistinctTilings gridSize = 8 * k :=
sorry

end NUMINAMATH_CALUDE_num_tilings_div_by_eight_l4180_418044


namespace NUMINAMATH_CALUDE_faster_train_speed_l4180_418060

/-- The speed of the faster train given two trains crossing each other -/
theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : train_length = 150)
  (h2 : crossing_time = 18)
  (h3 : speed_ratio = 3) : 
  ∃ (v : ℝ), v = 12.5 ∧ v = (2 * train_length) / (crossing_time * (1 + 1 / speed_ratio)) :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l4180_418060


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l4180_418004

/-- Represents the ratio of bleach : detergent : water in a solution -/
structure SolutionRatio :=
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)

/-- Calculates the new ratio after tripling bleach to detergent and halving detergent to water -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := 2 * r.detergent,
    water := 4 * r.water }

/-- Calculates the amount of detergent in the altered solution -/
def detergent_amount (r : SolutionRatio) (water_amount : ℕ) : ℕ :=
  (r.detergent * water_amount) / r.water

theorem detergent_in_altered_solution :
  let original_ratio : SolutionRatio := { bleach := 4, detergent := 40, water := 100 }
  let altered_ratio := alter_ratio original_ratio
  detergent_amount altered_ratio 300 = 60 := by
  sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l4180_418004


namespace NUMINAMATH_CALUDE_least_N_for_prime_condition_l4180_418046

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a multiple of 12 -/
def isMultipleOf12 (n : ℕ) : Prop := sorry

/-- The theorem statement -/
theorem least_N_for_prime_condition : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∀ (n : ℕ), isPrime (1 + N * 2^n) ↔ isMultipleOf12 n) ∧
  (∀ (M : ℕ), M > 0 → M < N → 
    ¬(∀ (n : ℕ), isPrime (1 + M * 2^n) ↔ isMultipleOf12 n)) ∧
  N = 556 := by
  sorry

end NUMINAMATH_CALUDE_least_N_for_prime_condition_l4180_418046


namespace NUMINAMATH_CALUDE_community_service_average_l4180_418026

theorem community_service_average : 
  let participation_numbers : List Nat := [2, 1, 3, 3, 4, 5, 3, 6, 5, 3]
  let num_students : Nat := participation_numbers.length
  let total_participations : Nat := participation_numbers.sum
  (total_participations : ℚ) / num_students = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_community_service_average_l4180_418026


namespace NUMINAMATH_CALUDE_sin_cos_equivalence_l4180_418002

theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x - π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_cos_equivalence_l4180_418002


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l4180_418096

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 2]
def interval : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f_in_interval : 
  ∃ (x : ℝ), x ∈ interval ∧ f x = -2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l4180_418096
