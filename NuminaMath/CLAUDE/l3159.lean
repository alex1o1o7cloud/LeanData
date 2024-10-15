import Mathlib

namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3159_315935

/-- The x-intercept of the line 6x + 7y = 35 is (35/6, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  (6 * x + 7 * y = 35) → (x = 35 / 6 ∧ y = 0) → (6 * (35 / 6) + 7 * 0 = 35) := by
  sorry

#check x_intercept_of_line

end NUMINAMATH_CALUDE_x_intercept_of_line_l3159_315935


namespace NUMINAMATH_CALUDE_first_quarter_homework_points_l3159_315929

theorem first_quarter_homework_points :
  ∀ (homework quiz test : ℕ),
    homework + quiz + test = 265 →
    test = 4 * quiz →
    quiz = homework + 5 →
    homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_homework_points_l3159_315929


namespace NUMINAMATH_CALUDE_youngest_brother_age_l3159_315936

theorem youngest_brother_age (a b c : ℕ) : 
  (a + b + c = 96) → 
  (b = a + 1) → 
  (c = a + 2) → 
  a = 31 := by
sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l3159_315936


namespace NUMINAMATH_CALUDE_circle_satisfies_equation_l3159_315905

/-- A circle passing through two points with its center on a given line -/
structure CircleWithConstraints where
  -- Center of the circle lies on the line x - 2y - 2 = 0
  center : ℝ × ℝ
  center_on_line : center.1 - 2 * center.2 - 2 = 0
  -- Circle passes through points A(0, 4) and B(4, 6)
  passes_through_A : (center.1 - 0)^2 + (center.2 - 4)^2 = (center.1 - 4)^2 + (center.2 - 6)^2

/-- The standard equation of the circle -/
def circle_equation (c : CircleWithConstraints) (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 25

/-- Theorem stating that the circle satisfies the given equation -/
theorem circle_satisfies_equation (c : CircleWithConstraints) :
  ∀ x y, (x - c.center.1)^2 + (y - c.center.2)^2 = (c.center.1 - 0)^2 + (c.center.2 - 4)^2 →
  circle_equation c x y := by
  sorry

end NUMINAMATH_CALUDE_circle_satisfies_equation_l3159_315905


namespace NUMINAMATH_CALUDE_binomial_sum_of_even_coefficients_l3159_315916

theorem binomial_sum_of_even_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_of_even_coefficients_l3159_315916


namespace NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l3159_315978

theorem slope_angle_45_implies_a_equals_1 (a : ℝ) : 
  (∃ (x y : ℝ), a * x + (2 * a - 3) * y = 0 ∧ 
   Real.tan (45 * π / 180) = -(a / (2 * a - 3))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l3159_315978


namespace NUMINAMATH_CALUDE_sandys_marks_per_correct_sum_l3159_315995

/-- Given Sandy's quiz results, calculate the marks for each correct sum -/
theorem sandys_marks_per_correct_sum 
  (total_sums : ℕ) 
  (correct_sums : ℕ) 
  (total_marks : ℤ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : correct_sums = 23) 
  (h3 : total_marks = 55) 
  (h4 : penalty_per_incorrect = 2) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_sums - 
    penalty_per_incorrect * (total_sums - correct_sums) = total_marks ∧ 
    marks_per_correct = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_marks_per_correct_sum_l3159_315995


namespace NUMINAMATH_CALUDE_tangent_slope_angle_is_45_degrees_l3159_315908

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- The point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle_is_45_degrees :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_is_45_degrees_l3159_315908


namespace NUMINAMATH_CALUDE_average_value_function_2x_squared_average_value_function_exponential_l3159_315972

/-- Definition of average value function on [a,b] -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The function f(x) = 2x^2 is an average value function on [-1,1] with average value point 0 -/
theorem average_value_function_2x_squared :
  is_average_value_function (fun x => 2 * x^2) (-1) 1 ∧
  (fun x => 2 * x^2) 0 = ((fun x => 2 * x^2) 1 - (fun x => 2 * x^2) (-1)) / (1 - (-1)) :=
sorry

/-- The function g(x) = -2^(2x+1) + m⋅2^(x+1) + 1 is an average value function on [-1,1]
    if and only if m ∈ (-∞, 13/10) ∪ (17/2, +∞) -/
theorem average_value_function_exponential (m : ℝ) :
  is_average_value_function (fun x => -2^(2*x+1) + m * 2^(x+1) + 1) (-1) 1 ↔
  m < 13/10 ∨ m > 17/2 :=
sorry

end NUMINAMATH_CALUDE_average_value_function_2x_squared_average_value_function_exponential_l3159_315972


namespace NUMINAMATH_CALUDE_problem_statement_l3159_315966

theorem problem_statement (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) : 
  (1/3) * x^8 * y^9 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3159_315966


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l3159_315911

/-- Reverses the digits of a given integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit integer -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72 :
  ∀ p : ℕ,
    isFiveDigit p →
    p % 72 = 0 →
    (reverseDigits p) % 72 = 0 →
    p % 11 = 0 →
    p ≥ 80001 :=
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l3159_315911


namespace NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l3159_315999

theorem odd_coefficients_in_binomial_expansion :
  let coefficients := List.range 9 |>.map (fun k => Nat.choose 8 k)
  (coefficients.filter (fun c => c % 2 = 1)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l3159_315999


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l3159_315942

/-- A line y = -3x + b that does not pass through the first quadrant has b ≤ 0 -/
theorem line_not_in_first_quadrant (b : ℝ) : 
  (∀ x y : ℝ, y = -3 * x + b → ¬(x > 0 ∧ y > 0)) → b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l3159_315942


namespace NUMINAMATH_CALUDE_solve_rope_problem_l3159_315913

def rope_problem (x : ℝ) : Prop :=
  let known_ropes := [8, 20, 7]
  let total_ropes := 6
  let knot_loss := 1.2
  let final_length := 35
  let num_knots := total_ropes - 1
  let total_knot_loss := num_knots * knot_loss
  final_length + total_knot_loss = (known_ropes.sum + 3 * x)

theorem solve_rope_problem :
  ∃ x : ℝ, rope_problem x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_solve_rope_problem_l3159_315913


namespace NUMINAMATH_CALUDE_root_bounds_l3159_315953

theorem root_bounds (b c x : ℝ) (hb : 5.025 ≤ b ∧ b ≤ 5.035) (hc : 1.745 ≤ c ∧ c ≤ 1.755)
  (hx : (3 * x + b) / 4 = (2 * x - 3) / c) :
  7.512 ≤ x ∧ x ≤ 7.618 :=
by sorry

end NUMINAMATH_CALUDE_root_bounds_l3159_315953


namespace NUMINAMATH_CALUDE_denmark_pizza_combinations_l3159_315951

/-- Represents the number of topping combinations for Denmark's pizza order --/
def toppingCombinations (cheeseOptions : Nat) (meatOptions : Nat) (vegetableOptions : Nat) : Nat :=
  let totalCombinations := cheeseOptions * meatOptions * vegetableOptions
  let restrictedCombinations := cheeseOptions * 1 * 1
  totalCombinations - restrictedCombinations

/-- Theorem: Denmark has 57 different topping combinations for his pizza --/
theorem denmark_pizza_combinations :
  toppingCombinations 3 4 5 = 57 := by
  sorry

#eval toppingCombinations 3 4 5

end NUMINAMATH_CALUDE_denmark_pizza_combinations_l3159_315951


namespace NUMINAMATH_CALUDE_scaled_circle_area_l3159_315937

/-- Given a circle with center P(-5, 3) passing through Q(7, -4), 
    when uniformly scaled by a factor of 2 from its center, 
    the area of the resulting circle is 772π. -/
theorem scaled_circle_area : 
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -4)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let scale_factor : ℝ := 2
  let scaled_area := π * (scale_factor * r)^2
  scaled_area = 772 * π :=
by sorry

end NUMINAMATH_CALUDE_scaled_circle_area_l3159_315937


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3159_315930

theorem intersection_of_sets : 
  let M : Set Char := {a, b, c}
  let N : Set Char := {b, c, d}
  M ∩ N = {b, c} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3159_315930


namespace NUMINAMATH_CALUDE_exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l3159_315982

/-- Represents the equation x|x| + px + q = 0 --/
def abs_equation (x p q : ℝ) : Prop :=
  x * abs x + p * x + q = 0

/-- There exists a case where p^2 - 4q < 0 and the equation has real roots --/
theorem exists_roots_when_discriminant_negative :
  ∃ (p q : ℝ), p^2 - 4*q < 0 ∧ (∃ x : ℝ, abs_equation x p q) :=
sorry

/-- There exists a case where p < 0, q > 0, and the equation does not have exactly three real roots --/
theorem not_always_three_roots_when_p_neg_q_pos :
  ∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ ¬(∃! (x y z : ℝ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    abs_equation x p q ∧ abs_equation y p q ∧ abs_equation z p q) :=
sorry

end NUMINAMATH_CALUDE_exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l3159_315982


namespace NUMINAMATH_CALUDE_water_poured_out_l3159_315927

/-- The volume of water poured out from a cylindrical cup -/
theorem water_poured_out (r h : Real) (α β : Real) : 
  r = 4 → 
  h = 8 * Real.sqrt 3 → 
  α = π / 3 → 
  β = π / 6 → 
  let V₁ := π * r^2 * (h - (h - r * Real.tan α))
  let V₂ := π * r^2 * (h / 2)
  V₁ - V₂ = (128 * Real.sqrt 3 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_water_poured_out_l3159_315927


namespace NUMINAMATH_CALUDE_washing_machine_price_difference_l3159_315988

def total_price : ℕ := 7060
def refrigerator_price : ℕ := 4275

theorem washing_machine_price_difference : 
  refrigerator_price - (total_price - refrigerator_price) = 1490 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_price_difference_l3159_315988


namespace NUMINAMATH_CALUDE_cone_central_angle_l3159_315931

/-- Represents a cone with its surface areas and central angle. -/
structure Cone where
  base_area : ℝ
  total_surface_area : ℝ
  lateral_surface_area : ℝ
  central_angle : ℝ

/-- The theorem stating the relationship between the cone's surface areas and its central angle. -/
theorem cone_central_angle (c : Cone) 
  (h1 : c.total_surface_area = 3 * c.base_area)
  (h2 : c.lateral_surface_area = 2 * c.base_area)
  (h3 : c.lateral_surface_area = (c.central_angle / 360) * (2 * π * c.base_area)) :
  c.central_angle = 240 := by
  sorry


end NUMINAMATH_CALUDE_cone_central_angle_l3159_315931


namespace NUMINAMATH_CALUDE_exists_four_digit_with_eleven_multiple_permutation_l3159_315956

/-- A permutation of the digits of a number -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Check if a number is between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_with_eleven_multiple_permutation :
  ∃ n : ℕ, isFourDigit n ∧ ∃ m : ℕ, isDigitPermutation n m ∧ m % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_with_eleven_multiple_permutation_l3159_315956


namespace NUMINAMATH_CALUDE_cheese_weight_l3159_315965

/-- Represents the weight of two pieces of cheese -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- The function that represents taking a bite from the larger piece -/
def take_bite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- The theorem stating the original weight of the cheese -/
theorem cheese_weight (initial : CheesePair) :
  (take_bite (take_bite (take_bite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
sorry

end NUMINAMATH_CALUDE_cheese_weight_l3159_315965


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3159_315928

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3159_315928


namespace NUMINAMATH_CALUDE_cranberry_juice_cost_l3159_315955

/-- The total cost of a can of cranberry juice -/
theorem cranberry_juice_cost (ounces : ℕ) (cost_per_ounce : ℕ) : 
  ounces = 12 → cost_per_ounce = 7 → ounces * cost_per_ounce = 84 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_juice_cost_l3159_315955


namespace NUMINAMATH_CALUDE_company_research_development_l3159_315915

/-- Success probability of Team A -/
def p_a : ℚ := 2/3

/-- Success probability of Team B -/
def p_b : ℚ := 3/5

/-- Profit from successful development of product A (in thousands of dollars) -/
def profit_a : ℕ := 120

/-- Profit from successful development of product B (in thousands of dollars) -/
def profit_b : ℕ := 100

/-- The probability of at least one new product being successfully developed -/
def prob_at_least_one : ℚ := 1 - (1 - p_a) * (1 - p_b)

/-- The expected profit of the company (in thousands of dollars) -/
def expected_profit : ℚ := 
  0 * (1 - p_a) * (1 - p_b) + 
  profit_a * p_a * (1 - p_b) + 
  profit_b * (1 - p_a) * p_b + 
  (profit_a + profit_b) * p_a * p_b

theorem company_research_development :
  (prob_at_least_one = 13/15) ∧ (expected_profit = 140) := by
  sorry

end NUMINAMATH_CALUDE_company_research_development_l3159_315915


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l3159_315901

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 5
  (q 2 = 17) → (q (-2) = 17) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l3159_315901


namespace NUMINAMATH_CALUDE_product_pass_rate_l3159_315925

-- Define the defect rates for each step
variable (a b : ℝ)

-- Assume the defect rates are between 0 and 1
variable (ha : 0 ≤ a ∧ a ≤ 1)
variable (hb : 0 ≤ b ∧ b ≤ 1)

-- Define the pass rate of the product
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem statement
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) :=
by
  sorry

end NUMINAMATH_CALUDE_product_pass_rate_l3159_315925


namespace NUMINAMATH_CALUDE_child_patients_per_hour_l3159_315980

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the cost of an adult office visit in dollars -/
def adult_visit_cost : ℕ := 50

/-- Represents the cost of a child office visit in dollars -/
def child_visit_cost : ℕ := 25

/-- Represents the total revenue for a typical 8-hour day in dollars -/
def total_daily_revenue : ℕ := 2200

/-- Represents the number of hours in a typical workday -/
def hours_per_day : ℕ := 8

/-- 
Proves that the number of child patients seen per hour is 3, 
given the conditions specified in the problem.
-/
theorem child_patients_per_hour : 
  ∃ (c : ℕ), 
    hours_per_day * (adults_per_hour * adult_visit_cost + c * child_visit_cost) = total_daily_revenue ∧
    c = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_patients_per_hour_l3159_315980


namespace NUMINAMATH_CALUDE_original_number_proof_l3159_315903

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3159_315903


namespace NUMINAMATH_CALUDE_adults_trekking_l3159_315979

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  total_children : ℕ
  meal_adults : ℕ
  meal_children : ℕ
  adults_eaten : ℕ
  remaining_children : ℕ

/-- Theorem stating the number of adults who went trekking -/
theorem adults_trekking (group : TrekkingGroup) 
  (h1 : group.total_children = 70)
  (h2 : group.meal_adults = 70)
  (h3 : group.meal_children = 90)
  (h4 : group.adults_eaten = 42)
  (h5 : group.remaining_children = 36) :
  ∃ (adults_trekking : ℕ), adults_trekking = 70 := by
  sorry


end NUMINAMATH_CALUDE_adults_trekking_l3159_315979


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l3159_315985

theorem arithmetic_geometric_mean_difference_bound 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l3159_315985


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l3159_315906

/-- Given a principal amount and an interest rate satisfying the given conditions,
    prove that the interest rate is 10% --/
theorem interest_rate_is_ten_percent (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 2662) :
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l3159_315906


namespace NUMINAMATH_CALUDE_T_properties_l3159_315964

/-- T(N) is the number of arrangements of integers 1 to N satisfying specific conditions. -/
def T (N : ℕ) : ℕ := sorry

/-- v₂(n) is the 2-adic valuation of n. -/
def v₂ (n : ℕ) : ℕ := sorry

theorem T_properties :
  (T 7 = 80) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n - 1)) = 2^n - n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n + 1)) = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l3159_315964


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l3159_315996

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1/2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l3159_315996


namespace NUMINAMATH_CALUDE_stock_value_change_l3159_315976

/-- Calculates the net percentage change in stock value over three years --/
def netPercentageChange (year1Change year2Change year3Change dividend : ℝ) : ℝ :=
  let value1 := (1 + year1Change) * (1 + dividend)
  let value2 := value1 * (1 + year2Change) * (1 + dividend)
  let value3 := value2 * (1 + year3Change) * (1 + dividend)
  (value3 - 1) * 100

/-- The net percentage change in stock value is approximately 17.52% --/
theorem stock_value_change :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  |netPercentageChange (-0.08) 0.10 0.06 0.03 - 17.52| < ε :=
sorry

end NUMINAMATH_CALUDE_stock_value_change_l3159_315976


namespace NUMINAMATH_CALUDE_time_sum_after_advance_l3159_315990

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  h_valid : hours < 12
  m_valid : minutes < 60
  s_valid : seconds < 60

/-- Calculates the time after a given number of hours, minutes, and seconds -/
def advanceTime (start : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Theorem: After 122 hours, 39 minutes, and 44 seconds from midnight, 
    the sum of resulting hours, minutes, and seconds is 85 -/
theorem time_sum_after_advance : 
  let midnight : Time := ⟨0, 0, 0, by simp, by simp, by simp⟩
  let result := advanceTime midnight 122 39 44
  result.hours + result.minutes + result.seconds = 85 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_advance_l3159_315990


namespace NUMINAMATH_CALUDE_rectangle_existence_uniqueness_l3159_315974

theorem rectangle_existence_uniqueness 
  (a b : ℝ) 
  (h_ab : 0 < a ∧ a < b) : 
  ∃! (x y : ℝ), 
    x < a ∧ 
    y < b ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_uniqueness_l3159_315974


namespace NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_l3159_315949

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  c : ℝ  -- length of AB
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem for the first scenario
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.a = 7) (h2 : t.b = 24) : t.c = 25 := by
  sorry

-- Theorem for the second scenario
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = 12) (h2 : t.c = 13) : t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_l3159_315949


namespace NUMINAMATH_CALUDE_binomial_sum_inequality_l3159_315968

theorem binomial_sum_inequality (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_inequality_l3159_315968


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3159_315950

/-- A hyperbola is defined by its equation in the form ax² + by² = c, where a, b, and c are constants and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  opposite_signs : a * b < 0

/-- Two hyperbolas share the same asymptotes if they have the same ratio of coefficients for x² and y². -/
def share_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

/-- A point (x, y) is on a hyperbola if it satisfies the hyperbola's equation. -/
def point_on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  h.a * x^2 + h.b * y^2 = h.c

/-- The main theorem to be proved -/
theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a = 1/4 ∧ h1.b = -1 ∧ h1.c = 1 ∧
  h2.a = -1/16 ∧ h2.b = 1/4 ∧ h2.c = 1 →
  share_asymptotes h1 h2 ∧ point_on_hyperbola h2 2 (Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3159_315950


namespace NUMINAMATH_CALUDE_convenience_store_syrup_cost_l3159_315902

/-- Calculates the weekly syrup cost for a convenience store. -/
def weekly_syrup_cost (soda_sold : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (soda_sold / gallons_per_box) * cost_per_box

/-- Theorem stating the weekly syrup cost for the given conditions. -/
theorem convenience_store_syrup_cost :
  weekly_syrup_cost 180 30 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_syrup_cost_l3159_315902


namespace NUMINAMATH_CALUDE_intersection_with_complement_example_l3159_315947

open Set

theorem intersection_with_complement_example : 
  let U : Set ℕ := {1, 3, 5, 7, 9}
  let A : Set ℕ := {3, 7, 9}
  let B : Set ℕ := {1, 9}
  A ∩ (U \ B) = {3, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_with_complement_example_l3159_315947


namespace NUMINAMATH_CALUDE_candidate_total_score_l3159_315943

/-- Calculates the total score of a candidate based on their written test and interview scores -/
def totalScore (writtenScore : ℝ) (interviewScore : ℝ) : ℝ :=
  0.70 * writtenScore + 0.30 * interviewScore

/-- Theorem stating that the total score of a candidate with given scores is 87 -/
theorem candidate_total_score :
  let writtenScore : ℝ := 90
  let interviewScore : ℝ := 80
  totalScore writtenScore interviewScore = 87 := by
  sorry

#eval totalScore 90 80

end NUMINAMATH_CALUDE_candidate_total_score_l3159_315943


namespace NUMINAMATH_CALUDE_bus_journey_speed_l3159_315932

theorem bus_journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_distance = 250) 
  (h2 : total_time = 5.2) 
  (h3 : first_part_distance = 124) 
  (h4 : second_part_speed = 60) :
  ∃ (first_part_speed : ℝ), 
    first_part_speed = 40 ∧ 
    first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_speed_l3159_315932


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l3159_315969

theorem trigonometric_expressions :
  (2 * Real.sin (30 * π / 180) + 3 * Real.cos (60 * π / 180) - 4 * Real.tan (45 * π / 180) = -3/2) ∧
  (Real.tan (60 * π / 180) - (4 - π)^0 + 2 * Real.cos (30 * π / 180) + (1/4)⁻¹ = 2 * Real.sqrt 3 + 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l3159_315969


namespace NUMINAMATH_CALUDE_calculations_correctness_l3159_315922

theorem calculations_correctness : 
  (-3 - 1 ≠ -2) ∧ 
  ((-3/4) - (3/4) ≠ 0) ∧ 
  (-8 / (-2) ≠ -4) ∧ 
  ((-3)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_calculations_correctness_l3159_315922


namespace NUMINAMATH_CALUDE_pauls_earnings_duration_l3159_315961

/-- Calculates how many weeks Paul's earnings will last given his weekly earnings and expenses. -/
def weeks_earnings_last (lawn_mowing : ℚ) (weed_eating : ℚ) (bush_trimming : ℚ) (fence_painting : ℚ)
                        (food_expense : ℚ) (transportation_expense : ℚ) (entertainment_expense : ℚ) : ℚ :=
  (lawn_mowing + weed_eating + bush_trimming + fence_painting) /
  (food_expense + transportation_expense + entertainment_expense)

/-- Theorem stating that Paul's earnings will last 2.5 weeks given his specific earnings and expenses. -/
theorem pauls_earnings_duration :
  weeks_earnings_last 12 8 5 20 10 5 3 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_earnings_duration_l3159_315961


namespace NUMINAMATH_CALUDE_opposite_hands_count_l3159_315989

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour : ℝ) -- Hour hand position (0 ≤ hour < 12)
  (minute : ℝ) -- Minute hand position (0 ≤ minute < 60)

/-- The speed ratio between minute and hour hands -/
def minute_hour_speed_ratio : ℝ := 12

/-- The angle difference when hands are opposite -/
def opposite_angle_diff : ℝ := 30

/-- Counts the number of times the clock hands are opposite in a 24-hour period -/
def count_opposite_hands (c : Clock) : ℕ := sorry

/-- Theorem stating that the hands are opposite 22 times in a day -/
theorem opposite_hands_count :
  ∀ c : Clock, count_opposite_hands c = 22 := by sorry

end NUMINAMATH_CALUDE_opposite_hands_count_l3159_315989


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l3159_315973

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100

def total_value : ℚ := 380 / 100

theorem max_quarters_sasha :
  ∃ (q : ℕ), 
    (q * quarter_value + q * nickel_value + 2 * q * dime_value ≤ total_value) ∧
    (∀ (n : ℕ), n > q → 
      n * quarter_value + n * nickel_value + 2 * n * dime_value > total_value) ∧
    q = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l3159_315973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3159_315993

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 4 + a 5 + a 6 + a 7 = 450 →
  a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3159_315993


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3159_315934

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- State the theorem
theorem point_on_y_axis (m : ℝ) :
  let p := Point2D.mk (m - 1) (m + 3)
  onYAxis p → p = Point2D.mk 0 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3159_315934


namespace NUMINAMATH_CALUDE_particle_probabilities_l3159_315909

/-- A particle moves on a line with marked points 0, ±1, ±2, ±3, ... 
    Starting at point 0, it moves to n+1 or n-1 with equal probabilities 1/2 -/
def Particle := ℤ

/-- The probability that the particle will be at point 1 at some time -/
def prob_at_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will be at point -1 at some time -/
def prob_at_neg_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will return to point 0 at some time 
    other than the initial starting point -/
def prob_return_to_zero (p : Particle) : ℝ := sorry

/-- The theorem stating that all three probabilities are equal to 1 -/
theorem particle_probabilities (p : Particle) : 
  prob_at_one p = 1 ∧ prob_at_neg_one p = 1 ∧ prob_return_to_zero p = 1 :=
by sorry

end NUMINAMATH_CALUDE_particle_probabilities_l3159_315909


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l3159_315948

-- Define the angle in degrees and minutes
def angle : ℚ := -999 * 360 / 360 - 30 / 60

-- Function to normalize an angle to the range [0, 360)
def normalize_angle (θ : ℚ) : ℚ :=
  θ - 360 * ⌊θ / 360⌋

-- Define the first quadrant
def is_first_quadrant (θ : ℚ) : Prop :=
  0 < normalize_angle θ ∧ normalize_angle θ < 90

-- Theorem statement
theorem angle_in_first_quadrant : is_first_quadrant angle := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l3159_315948


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_curves_l3159_315945

/-- Given two curves in polar coordinates that intersect, 
    prove the equation of the perpendicular bisector of their intersection points. -/
theorem perpendicular_bisector_of_intersecting_curves 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ θ ρ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ)
  (h₂ : ∀ θ ρ, C₂ ρ θ ↔ ρ = 2 * Real.cos θ)
  (A B : ℝ × ℝ)
  (hA : C₁ A.1 A.2 ∧ C₂ A.1 A.2)
  (hB : C₁ B.1 B.2 ∧ C₂ B.1 B.2)
  (hAB : A ≠ B) :
  ∃ (ρ θ : ℝ), ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_curves_l3159_315945


namespace NUMINAMATH_CALUDE_grocer_banana_purchase_l3159_315940

/-- Proves that the grocer purchased 792 pounds of bananas given the conditions -/
theorem grocer_banana_purchase :
  ∀ (pounds : ℝ),
  (pounds / 3 * 0.50 = pounds / 4 * 1.00 - 11.00) →
  pounds = 792 := by
sorry

end NUMINAMATH_CALUDE_grocer_banana_purchase_l3159_315940


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3159_315952

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3159_315952


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l3159_315920

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 7 →
  chord_length = 10 →
  let segment_length := r - 2 * Real.sqrt 6
  ∃ (AK KB : ℝ),
    AK = segment_length ∧
    KB = 2 * r - segment_length ∧
    AK + KB = 2 * r ∧
    AK * KB = (chord_length / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l3159_315920


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l3159_315918

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l3159_315918


namespace NUMINAMATH_CALUDE_sine_function_shifted_symmetric_l3159_315944

/-- Given a function f(x) = sin(ωx + φ), prove that under certain conditions, φ = π/6 -/
theorem sine_function_shifted_symmetric (ω φ : Real) : 
  ω > 0 → 
  0 < φ → 
  φ < Real.pi / 2 → 
  (fun x ↦ Real.sin (ω * x + φ)) 0 = -(fun x ↦ Real.sin (ω * x + φ)) (Real.pi / 2) →
  (∀ x, Real.sin (ω * (x + Real.pi / 12) + φ) = -Real.sin (ω * (-x + Real.pi / 12) + φ)) →
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_shifted_symmetric_l3159_315944


namespace NUMINAMATH_CALUDE_common_tangents_theorem_l3159_315994

/-- Represents the relative position of two circles -/
inductive CirclePosition
  | Outside
  | TouchingExternally
  | Intersecting
  | TouchingInternally
  | Inside
  | Identical
  | OnePoint
  | TwoDistinctPoints
  | TwoCoincidingPoints

/-- Represents the number of common tangents -/
inductive TangentCount
  | Zero
  | One
  | Two
  | Three
  | Four
  | Infinite

/-- Function to determine the number of common tangents based on circle position -/
def commonTangents (position : CirclePosition) : TangentCount :=
  match position with
  | CirclePosition.Outside => TangentCount.Four
  | CirclePosition.TouchingExternally => TangentCount.Three
  | CirclePosition.Intersecting => TangentCount.Two
  | CirclePosition.TouchingInternally => TangentCount.One
  | CirclePosition.Inside => TangentCount.Zero
  | CirclePosition.Identical => TangentCount.Infinite
  | CirclePosition.OnePoint => TangentCount.Two  -- Assuming the point is outside the circle
  | CirclePosition.TwoDistinctPoints => TangentCount.One
  | CirclePosition.TwoCoincidingPoints => TangentCount.Infinite

/-- Theorem stating that the number of common tangents depends on the relative position of circles -/
theorem common_tangents_theorem (position : CirclePosition) :
  (commonTangents position = TangentCount.Zero) ∨
  (commonTangents position = TangentCount.One) ∨
  (commonTangents position = TangentCount.Two) ∨
  (commonTangents position = TangentCount.Three) ∨
  (commonTangents position = TangentCount.Four) ∨
  (commonTangents position = TangentCount.Infinite) :=
by sorry

end NUMINAMATH_CALUDE_common_tangents_theorem_l3159_315994


namespace NUMINAMATH_CALUDE_triangle_construction_uniqueness_l3159_315971

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by its three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- The point where the internal angle bisector from A intersects BC -/
def internalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- The point where the external angle bisector from A intersects BC -/
def externalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- Predicate to check if a point is within the valid region for M -/
def isValidM (M A' A'' : Point) : Prop :=
  sorry

theorem triangle_construction_uniqueness 
  (M A' A'' : Point) 
  (h_valid : isValidM M A' A'') :
  ∃! t : Triangle, 
    orthocenter t = M ∧ 
    internalBisectorIntersection t = A' ∧ 
    externalBisectorIntersection t = A'' :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_uniqueness_l3159_315971


namespace NUMINAMATH_CALUDE_mrs_hilt_spent_74_cents_l3159_315923

/-- Calculates the total amount spent by Mrs. Hilt at the school store -/
def school_store_total (notebook_cost ruler_cost pencil_cost : ℕ) (num_pencils : ℕ) : ℕ :=
  notebook_cost + ruler_cost + (pencil_cost * num_pencils)

/-- Proves that Mrs. Hilt spent 74 cents at the school store -/
theorem mrs_hilt_spent_74_cents :
  school_store_total 35 18 7 3 = 74 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_spent_74_cents_l3159_315923


namespace NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3159_315991

def digits : List Nat := [3, 4, 5, 7, 9]

def isValidNumber (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def validNumbers : List Nat :=
  (List.range 900).filter (fun n => n ≥ 300 ∧ isValidNumber n)

theorem three_digit_numbers_theorem :
  validNumbers.length = 60 ∧ validNumbers.sum = 37296 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3159_315991


namespace NUMINAMATH_CALUDE_dice_probability_relationship_l3159_315983

/-- The probability that the sum of two fair dice does not exceed 5 -/
def p₁ : ℚ := 5/18

/-- The probability that the sum of two fair dice is greater than 5 -/
def p₂ : ℚ := 11/18

/-- The probability that the sum of two fair dice is an even number -/
def p₃ : ℚ := 1/2

/-- Theorem stating the relationship between p₁, p₂, and p₃ -/
theorem dice_probability_relationship : p₁ < p₃ ∧ p₃ < p₂ := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_relationship_l3159_315983


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3159_315970

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3159_315970


namespace NUMINAMATH_CALUDE_erasers_given_to_doris_l3159_315921

def initial_erasers : ℕ := 81
def final_erasers : ℕ := 47

theorem erasers_given_to_doris : initial_erasers - final_erasers = 34 := by
  sorry

end NUMINAMATH_CALUDE_erasers_given_to_doris_l3159_315921


namespace NUMINAMATH_CALUDE_logarithm_calculation_l3159_315963

theorem logarithm_calculation : 
  (Real.log 3 / Real.log (1/9) - (-8)^(2/3)) * (0.125^(1/3)) = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_calculation_l3159_315963


namespace NUMINAMATH_CALUDE_banana_permutations_l3159_315907

theorem banana_permutations : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
sorry

end NUMINAMATH_CALUDE_banana_permutations_l3159_315907


namespace NUMINAMATH_CALUDE_circles_tangent_line_slope_l3159_315997

-- Define the circles and their properties
def Circle := ℝ × ℝ → Prop

-- Define the conditions
def intersect_at_4_9 (C₁ C₂ : Circle) : Prop := 
  C₁ (4, 9) ∧ C₂ (4, 9)

def product_of_radii_85 (C₁ C₂ : Circle) : Prop := 
  ∃ r₁ r₂ : ℝ, r₁ * r₂ = 85

def tangent_to_y_axis (C : Circle) : Prop := 
  ∃ x : ℝ, C (0, x)

def tangent_line (n : ℝ) (C : Circle) : Prop := 
  ∃ x y : ℝ, C (x, y) ∧ y = n * x

-- Main theorem
theorem circles_tangent_line_slope (C₁ C₂ : Circle) (n : ℝ) :
  intersect_at_4_9 C₁ C₂ →
  product_of_radii_85 C₁ C₂ →
  tangent_to_y_axis C₁ →
  tangent_to_y_axis C₂ →
  tangent_line n C₁ →
  tangent_line n C₂ →
  n > 0 →
  ∃ d e f : ℕ,
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ e)) ∧
    Nat.Coprime d f ∧
    n = (d : ℝ) * Real.sqrt e / f ∧
    d + e + f = 243 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_line_slope_l3159_315997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3159_315933

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  x : ℚ
  first_term : ℚ := 3 * x - 4
  second_term : ℚ := 6 * x - 14
  third_term : ℚ := 4 * x + 3
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, nth_term seq n = 3012 ∧ n = 247 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3159_315933


namespace NUMINAMATH_CALUDE_ellipse_properties_l3159_315998

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the triangle AF₁F₂ -/
def triangle_AF1F2 (A F1 F2 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)
  let d1A := Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2)
  let d2A := Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2)
  d12 = 2 * Real.sqrt 2 ∧ d1A = d2A ∧ d1A^2 + d2A^2 = d12^2

/-- Main theorem -/
theorem ellipse_properties
  (a b : ℝ)
  (A F1 F2 : ℝ × ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b)
  (h_triangle : triangle_AF1F2 A F1 F2) :
  (∀ x y, x^2 / 4 + y^2 / 2 = 1 ↔ ellipse_C x y a b) ∧
  (∀ P Q : ℝ × ℝ, P.2 = P.1 + 1 → Q.2 = Q.1 + 1 →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4/3)^2 * 5) ∧
  (¬ ∃ m : ℝ, ∀ P Q : ℝ × ℝ,
    P.2 = P.1 + m → Q.2 = Q.1 + m →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (1/2) * Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * (|m| / Real.sqrt 2) = 4/3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3159_315998


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3159_315959

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 4 * x + 1 > 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3159_315959


namespace NUMINAMATH_CALUDE_harolds_utilities_car_ratio_l3159_315984

/-- Harold's financial situation --/
structure HaroldFinances where
  income : ℕ
  rent : ℕ
  car_payment : ℕ
  groceries : ℕ
  retirement_savings : ℕ
  remaining : ℕ

/-- Calculate the ratio of utilities cost to car payment --/
def utilities_to_car_ratio (h : HaroldFinances) : ℚ :=
  let total_expenses := h.rent + h.car_payment + h.groceries
  let money_before_retirement := h.income - total_expenses
  let utilities := money_before_retirement - h.retirement_savings - h.remaining
  utilities / h.car_payment

/-- Theorem stating the ratio of Harold's utilities cost to his car payment --/
theorem harolds_utilities_car_ratio :
  ∃ h : HaroldFinances,
    h.income = 2500 ∧
    h.rent = 700 ∧
    h.car_payment = 300 ∧
    h.groceries = 50 ∧
    h.retirement_savings = (h.income - h.rent - h.car_payment - h.groceries) / 2 ∧
    h.remaining = 650 ∧
    utilities_to_car_ratio h = 1 / 4 :=
  sorry

end NUMINAMATH_CALUDE_harolds_utilities_car_ratio_l3159_315984


namespace NUMINAMATH_CALUDE_spinner_probability_l3159_315992

theorem spinner_probability : ∀ (p_C : ℚ),
  (1 : ℚ) / 5 + (1 : ℚ) / 3 + p_C + p_C + 2 * p_C = 1 →
  p_C = (7 : ℚ) / 60 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3159_315992


namespace NUMINAMATH_CALUDE_midpoint_is_inferior_exists_n_satisfying_conditions_l3159_315941

/-- Definition of a superior point in the first quadrant -/
def is_superior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b > c / d

/-- Definition of an inferior point in the first quadrant -/
def is_inferior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b < c / d

/-- Theorem: The midpoint of a superior point and an inferior point is inferior to the superior point -/
theorem midpoint_is_inferior (a b c d : ℝ) :
  is_superior_point a b c d →
  is_inferior_point ((a + c) / 2) ((b + d) / 2) a b :=
sorry

/-- Definition of the set of integers from 1 to 2021 -/
def S : Set ℤ := {m | 0 < m ∧ m < 2022}

/-- Theorem: There exists an integer n satisfying the given conditions -/
theorem exists_n_satisfying_conditions :
  ∃ n : ℤ, ∀ m ∈ S,
    (is_inferior_point n (2 * m + 1) 2022 m) ∧
    (is_superior_point n (2 * m + 1) 2023 (m + 1)) :=
sorry

end NUMINAMATH_CALUDE_midpoint_is_inferior_exists_n_satisfying_conditions_l3159_315941


namespace NUMINAMATH_CALUDE_function_property_l3159_315926

/-- Given a function f(x) = 2√3 sin(3ωx + π/3) where ω > 0,
    if f(x+θ) is an even function with a period of 2π,
    then θ = 7π/6 -/
theorem function_property (ω θ : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sqrt 3 * Real.sin (3 * ω * x + π / 3)
  (∀ x, f (x + θ) = f (-x - θ)) ∧  -- f(x+θ) is even
  (∀ x, f (x + θ) = f (x + θ + 2 * π)) →  -- f(x+θ) has period 2π
  θ = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3159_315926


namespace NUMINAMATH_CALUDE_angle_PSU_is_20_degrees_l3159_315914

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the foot of the perpendicular
def foot_of_perpendicular (P S Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define the center of the circumscribed circle
def circumcenter (T P Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define a point on the diameter opposite to another point
def opposite_on_diameter (P U T : ℝ × ℝ) : Prop :=
  sorry

theorem angle_PSU_is_20_degrees 
  (P Q R S T U : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (angle_PRQ : angle_measure P R Q = 60)
  (angle_QRP : angle_measure Q R P = 80)
  (S_perpendicular : foot_of_perpendicular P S Q R)
  (T_circumcenter : circumcenter T P Q R)
  (U_opposite : opposite_on_diameter P U T) :
  angle_measure P S U = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_PSU_is_20_degrees_l3159_315914


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3159_315986

theorem polynomial_factorization (a x : ℝ) : 2*a*x^2 - 12*a*x + 18*a = 2*a*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3159_315986


namespace NUMINAMATH_CALUDE_no_real_roots_l3159_315938

theorem no_real_roots : ∀ x : ℝ, x^2 + 2*x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3159_315938


namespace NUMINAMATH_CALUDE_count_negative_expressions_l3159_315917

theorem count_negative_expressions : 
  let expressions := [-3^2, (-3)^2, -(-3), -|-3|]
  (expressions.filter (· < 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_expressions_l3159_315917


namespace NUMINAMATH_CALUDE_min_value_theorem_l3159_315960

theorem min_value_theorem (x : ℝ) (h : x > 1) : 
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3159_315960


namespace NUMINAMATH_CALUDE_product_equals_243_l3159_315954

theorem product_equals_243 :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l3159_315954


namespace NUMINAMATH_CALUDE_fraction_inequality_l3159_315910

theorem fraction_inequality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (a : ℝ) / b > Real.sqrt 2) :
  (a : ℝ) / b - 1 / (2 * (a : ℝ) * b) > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3159_315910


namespace NUMINAMATH_CALUDE_tank_dimension_l3159_315981

/-- Given a rectangular tank with dimensions 4, x, and 2 feet, 
    if the total cost to cover its surface with insulation at $20 per square foot is $1520, 
    then x = 5 feet. -/
theorem tank_dimension (x : ℝ) : 
  x > 0 →  -- Ensuring positive dimension
  (12 * x + 16) * 20 = 1520 → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_tank_dimension_l3159_315981


namespace NUMINAMATH_CALUDE_derived_sequence_general_term_l3159_315919

/-- An arithmetic sequence {a_n} with specific terms and a derived sequence {b_n} -/
def arithmetic_and_derived_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) ∧  -- arithmetic sequence condition
  (a 2 = 8) ∧
  (a 8 = 26) ∧
  (∀ n : ℕ, b n = a (3^n))  -- definition of b_n

/-- The general term of the derived sequence b_n -/
def b_general_term (n : ℕ) : ℝ := 3 * 3^n + 2

/-- Theorem stating that b_n equals the derived general term -/
theorem derived_sequence_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_and_derived_sequence a b →
  ∀ n : ℕ, b n = b_general_term n :=
by
  sorry

end NUMINAMATH_CALUDE_derived_sequence_general_term_l3159_315919


namespace NUMINAMATH_CALUDE_candy_bar_profit_l3159_315962

/-- Represents the profit calculation for a candy bar sale --/
theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 5 / 6  -- Price per bar when buying
  let sell_price : ℚ := 2 / 3  -- Price per bar when selling
  let cost : ℚ := total_bars * buy_price
  let revenue : ℚ := total_bars * sell_price
  let profit : ℚ := revenue - cost
  profit = -200
:= by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l3159_315962


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l3159_315977

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + m - 2023 = 0 → n^2 + n - 2023 = 0 → m^2 + 2*m + n = 2022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l3159_315977


namespace NUMINAMATH_CALUDE_race_finish_orders_l3159_315904

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

theorem race_finish_orders : number_of_permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l3159_315904


namespace NUMINAMATH_CALUDE_quartic_real_root_condition_l3159_315957

theorem quartic_real_root_condition (p q : ℝ) :
  (∃ x : ℝ, x^4 + p * x^2 + q = 0) →
  p^2 ≥ 4 * q ∧
  ¬(∀ p q : ℝ, p^2 ≥ 4 * q → ∃ x : ℝ, x^4 + p * x^2 + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quartic_real_root_condition_l3159_315957


namespace NUMINAMATH_CALUDE_some_number_value_l3159_315987

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 49 * some_number * 25) :
  some_number = 45 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3159_315987


namespace NUMINAMATH_CALUDE_intersection_equality_l3159_315939

def M : Set ℤ := {-1, 0, 1}
def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3159_315939


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_four_l3159_315975

theorem sum_of_fractions_equals_four (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_eq : a / (b + c + d) = b / (a + c + d) ∧ 
          b / (a + c + d) = c / (a + b + d) ∧ 
          c / (a + b + d) = d / (a + b + c)) : 
  (a + b) / (c + d) + (b + c) / (a + d) + 
  (c + d) / (a + b) + (d + a) / (b + c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_four_l3159_315975


namespace NUMINAMATH_CALUDE_tom_sees_jerry_l3159_315924

/-- Represents the cat-and-mouse chase problem -/
structure ChaseSetup where
  wallSideLength : ℝ
  tomSpeed : ℝ
  jerrySpeed : ℝ
  restTime : ℝ

/-- Calculates the time when Tom first sees Jerry -/
noncomputable def timeToMeet (setup : ChaseSetup) : ℝ :=
  sorry

/-- The main theorem stating when Tom will first see Jerry -/
theorem tom_sees_jerry (setup : ChaseSetup) :
  setup.wallSideLength = 100 ∧
  setup.tomSpeed = 50 ∧
  setup.jerrySpeed = 30 ∧
  setup.restTime = 1 →
  timeToMeet setup = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_sees_jerry_l3159_315924


namespace NUMINAMATH_CALUDE_mother_daughter_ages_l3159_315967

/-- Given a mother and daughter where:
    1. The mother is 27 years older than her daughter.
    2. A year ago, the mother was twice as old as her daughter.
    Prove that the mother is 55 years old and the daughter is 28 years old. -/
theorem mother_daughter_ages (mother_age daughter_age : ℕ) 
  (h1 : mother_age = daughter_age + 27)
  (h2 : mother_age - 1 = 2 * (daughter_age - 1)) :
  mother_age = 55 ∧ daughter_age = 28 := by
sorry

end NUMINAMATH_CALUDE_mother_daughter_ages_l3159_315967


namespace NUMINAMATH_CALUDE_total_soaking_time_with_ink_l3159_315946

/-- Represents the soaking time for different types of stains -/
def SoakingTime : Type := Nat → Nat

/-- Calculates the total soaking time for a piece of clothing -/
def totalSoakingTime (stainCounts : List Nat) (soakingTimes : List Nat) : Nat :=
  List.sum (List.zipWith (· * ·) stainCounts soakingTimes)

/-- Calculates the additional time needed for ink stains -/
def additionalInkTime (inkStainCount : Nat) (extraTimePerInkStain : Nat) : Nat :=
  inkStainCount * extraTimePerInkStain

theorem total_soaking_time_with_ink (shirtStainCounts shirtSoakingTimes
                                     pantsStainCounts pantsSoakingTimes
                                     socksStainCounts socksSoakingTimes : List Nat)
                                    (inkStainCount extraTimePerInkStain : Nat) :
  totalSoakingTime shirtStainCounts shirtSoakingTimes +
  totalSoakingTime pantsStainCounts pantsSoakingTimes +
  totalSoakingTime socksStainCounts socksSoakingTimes +
  additionalInkTime inkStainCount extraTimePerInkStain = 54 :=
by
  sorry

#check total_soaking_time_with_ink

end NUMINAMATH_CALUDE_total_soaking_time_with_ink_l3159_315946


namespace NUMINAMATH_CALUDE_black_region_area_l3159_315900

/-- The area of the black region in a square-within-square configuration -/
theorem black_region_area (larger_side smaller_side : ℝ) (h1 : larger_side = 9) (h2 : smaller_side = 4) :
  larger_side ^ 2 - smaller_side ^ 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_l3159_315900


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l3159_315958

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle :
  let slope := (deriv f) 1
  Real.arctan slope = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l3159_315958


namespace NUMINAMATH_CALUDE_ordering_abc_l3159_315912

theorem ordering_abc (a b c : ℝ) : 
  a = -(5/4) * Real.log (4/5) →
  b = Real.exp (1/4) / 4 →
  c = 1/3 →
  a < b ∧ b < c := by
sorry

end NUMINAMATH_CALUDE_ordering_abc_l3159_315912
