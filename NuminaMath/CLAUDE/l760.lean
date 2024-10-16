import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l760_76056

theorem quadratic_equation_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 3 = 0 ∧ m * y^2 - 2 * y + 3 = 0) ↔ 
  (m < 1/3 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l760_76056


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l760_76044

theorem multiplication_of_powers (a : ℝ) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l760_76044


namespace NUMINAMATH_CALUDE_time_period_is_three_years_l760_76071

/-- Represents the simple interest calculation and conditions --/
def simple_interest_problem (t : ℝ) : Prop :=
  let initial_deposit : ℝ := 9000
  let final_amount : ℝ := 10200
  let higher_rate_amount : ℝ := 10740
  ∃ r : ℝ,
    -- Condition for the original interest rate
    initial_deposit * (1 + r * t / 100) = final_amount ∧
    -- Condition for the interest rate 2% higher
    initial_deposit * (1 + (r + 2) * t / 100) = higher_rate_amount

/-- The theorem stating that the time period is 3 years --/
theorem time_period_is_three_years :
  simple_interest_problem 3 := by sorry

end NUMINAMATH_CALUDE_time_period_is_three_years_l760_76071


namespace NUMINAMATH_CALUDE_least_x_for_divisibility_by_three_l760_76097

theorem least_x_for_divisibility_by_three : 
  (∃ x : ℕ, ∀ n : ℕ, (n * 57) % 3 = 0) ∧ 
  (∀ y : ℕ, y < 0 → ¬(∀ n : ℕ, (n * 57) % 3 = 0)) := by sorry

end NUMINAMATH_CALUDE_least_x_for_divisibility_by_three_l760_76097


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l760_76054

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧ digits.toFinset.card = 5

def divisible_by_digits_except_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5 → n % d = 0

theorem least_five_digit_divisible_by_digits :
  ∀ n : ℕ,
    is_five_digit n ∧
    all_digits_different n ∧
    divisible_by_digits_except_five n →
    12376 ≤ n :=
by sorry

theorem twelve_three_seven_six_satisfies_conditions :
  is_five_digit 12376 ∧
  all_digits_different 12376 ∧
  divisible_by_digits_except_five 12376 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l760_76054


namespace NUMINAMATH_CALUDE_simplify_expression_l760_76035

theorem simplify_expression (x : ℝ) : 2*x - 3*(2-x) + 4*(2+x) - 5*(1-3*x) = 24*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l760_76035


namespace NUMINAMATH_CALUDE_trajectory_equation_l760_76047

/-- Given points A(-1,1) and B(1,-1) symmetrical about the origin,
    prove that a point P(x,y) with x ≠ ±1 satisfies x^2 + 3y^2 = 4
    if the product of slopes of AP and BP is -1/3 -/
theorem trajectory_equation (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3 →
  x^2 + 3*y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l760_76047


namespace NUMINAMATH_CALUDE_min_value_of_expression_l760_76018

theorem min_value_of_expression (x : ℝ) :
  ∃ (min_val : ℝ), min_val = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l760_76018


namespace NUMINAMATH_CALUDE_train_crossing_time_l760_76029

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 150 ∧ 
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l760_76029


namespace NUMINAMATH_CALUDE_fraction_five_seventeenths_repetend_l760_76094

/-- The repetend of a rational number in its decimal representation -/
def repetend (n d : ℕ) : List ℕ := sorry

/-- The length of the repetend of a rational number in its decimal representation -/
def repetendLength (n d : ℕ) : ℕ := sorry

theorem fraction_five_seventeenths_repetend :
  repetend 5 17 = [2, 9, 4, 1, 1, 7, 6] ∧ repetendLength 5 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_seventeenths_repetend_l760_76094


namespace NUMINAMATH_CALUDE_system_solution_l760_76005

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 15 - 2*a)
  (eq2 : a + c = -18 - 3*b)
  (eq3 : a + b = 8 - 4*c)
  (eq4 : a - b + c = 3) :
  3*a + 3*b + 3*c = 24/5 := by sorry

end NUMINAMATH_CALUDE_system_solution_l760_76005


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l760_76036

/-- The perpendicular bisector of a line segment passes through its midpoint and is perpendicular to the segment. -/
structure PerpendicularBisector (p₁ p₂ : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop where
  passes_through_midpoint : l ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  is_perpendicular : True  -- We don't need to express this condition for this problem

/-- The line equation x + y = b -/
def line_equation (b : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = b

/-- The main theorem: if x + y = b is the perpendicular bisector of the line segment
    from (2,5) to (8,11), then b = 13 -/
theorem perpendicular_bisector_value :
  PerpendicularBisector (2, 5) (8, 11) (line_equation b) → b = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_value_l760_76036


namespace NUMINAMATH_CALUDE_gift_distribution_theorem_l760_76059

/-- The number of ways to choose and distribute gifts -/
def giftDistributionWays (totalGifts classmates chosenGifts : ℕ) : ℕ :=
  (totalGifts.choose chosenGifts) * chosenGifts.factorial

/-- Theorem stating that choosing 3 out of 5 gifts and distributing to 3 classmates results in 60 ways -/
theorem gift_distribution_theorem :
  giftDistributionWays 5 3 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_theorem_l760_76059


namespace NUMINAMATH_CALUDE_quadratic_congruence_solutions_l760_76069

theorem quadratic_congruence_solutions (x : ℕ) : 
  (x^2 + x - 6) % 143 = 0 ↔ x ∈ ({2, 41, 101, 140} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solutions_l760_76069


namespace NUMINAMATH_CALUDE_quotient_digits_of_203_div_single_digit_l760_76053

theorem quotient_digits_of_203_div_single_digit :
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 →
  ∃ q : ℕ, 203 / d = q ∧ (100 ≤ q ∧ q ≤ 999 ∨ 10 ≤ q ∧ q ≤ 99) :=
by sorry

end NUMINAMATH_CALUDE_quotient_digits_of_203_div_single_digit_l760_76053


namespace NUMINAMATH_CALUDE_growth_rate_ratio_l760_76006

/-- Given a linear regression equation y = ax + b where a = 4.4,
    prove that the ratio of the growth rate between x and y is 5/22 -/
theorem growth_rate_ratio (a b : ℝ) (h : a = 4.4) :
  (1 / a : ℝ) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_ratio_l760_76006


namespace NUMINAMATH_CALUDE_range_of_function_l760_76049

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, y = x / (1 + x^2)) ↔ -1/2 ≤ y ∧ y ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_function_l760_76049


namespace NUMINAMATH_CALUDE_union_M_complement_N_l760_76095

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def M : Set Nat := {1, 3, 5, 6}
def N : Set Nat := {1, 2, 4, 7, 9}

theorem union_M_complement_N : M ∪ (U \ N) = {1, 3, 5, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_l760_76095


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l760_76038

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2) ∨ 
  (a = 6 ∧ c = 8 ∧ b^2 = c^2 - a^2) ∨
  (b = 6 ∧ c = 8 ∧ a^2 = c^2 - b^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 ∨ b = 10 ∨ b = 2 * Real.sqrt 7 ∨ a = 10 ∨ a = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l760_76038


namespace NUMINAMATH_CALUDE_divisible_by_four_even_equivalence_l760_76025

theorem divisible_by_four_even_equivalence :
  (∀ n : ℤ, 4 ∣ n → Even n) ↔ (∀ n : ℤ, ¬Even n → ¬(4 ∣ n)) := by sorry

end NUMINAMATH_CALUDE_divisible_by_four_even_equivalence_l760_76025


namespace NUMINAMATH_CALUDE_unique_solution_condition_l760_76077

/-- The equation (x + 3) / (mx - 2) = x + 1 has exactly one solution if and only if m = -8 ± 2√15 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 3) / (m * x - 2) = x + 1) ↔ 
  (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l760_76077


namespace NUMINAMATH_CALUDE_storm_rainfall_l760_76052

/-- The total rainfall during a two-hour storm given specific conditions -/
theorem storm_rainfall (first_hour_rain : ℝ) (second_hour_increment : ℝ) : 
  first_hour_rain = 5 →
  second_hour_increment = 7 →
  (first_hour_rain + (2 * first_hour_rain + second_hour_increment)) = 22 := by
sorry

end NUMINAMATH_CALUDE_storm_rainfall_l760_76052


namespace NUMINAMATH_CALUDE_max_rational_products_l760_76057

/-- Represents a table with rational and irrational numbers as labels -/
structure LabeledTable where
  size : ℕ
  rowLabels : Fin size → ℝ
  colLabels : Fin size → ℝ
  distinctLabels : ∀ i j, (rowLabels i = colLabels j) → i = j
  rationalCount : ℕ
  irrationalCount : ℕ
  labelCounts : rationalCount + irrationalCount = size + size

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : LabeledTable) : ℕ :=
  sorry

/-- Theorem stating the maximum number of rational products -/
theorem max_rational_products (t : LabeledTable) : 
  t.size = 50 ∧ t.rationalCount = 50 ∧ t.irrationalCount = 50 → 
  countRationalProducts t ≤ 1275 :=
sorry

end NUMINAMATH_CALUDE_max_rational_products_l760_76057


namespace NUMINAMATH_CALUDE_expand_and_simplify_l760_76016

theorem expand_and_simplify (n b : ℝ) : (n + 2*b)^2 - 4*b^2 = n^2 + 4*n*b := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l760_76016


namespace NUMINAMATH_CALUDE_correct_operation_order_l760_76065

-- Define operation levels
inductive OperationLevel
| FirstLevel
| SecondLevel

-- Define operations
inductive Operation
| Multiplication
| Division
| Subtraction

-- Define the level of each operation
def operationLevel : Operation → OperationLevel
| Operation.Multiplication => OperationLevel.SecondLevel
| Operation.Division => OperationLevel.SecondLevel
| Operation.Subtraction => OperationLevel.FirstLevel

-- Define the rule for operation order
def shouldPerformBefore (op1 op2 : Operation) : Prop :=
  operationLevel op1 = OperationLevel.SecondLevel ∧ 
  operationLevel op2 = OperationLevel.FirstLevel

-- Define the expression
def expression : List Operation :=
  [Operation.Multiplication, Operation.Subtraction, Operation.Division]

-- Theorem to prove
theorem correct_operation_order :
  shouldPerformBefore Operation.Multiplication Operation.Subtraction ∧
  shouldPerformBefore Operation.Division Operation.Subtraction ∧
  (¬ shouldPerformBefore Operation.Multiplication Operation.Division ∨
   ¬ shouldPerformBefore Operation.Division Operation.Multiplication) :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_order_l760_76065


namespace NUMINAMATH_CALUDE_negative_integer_solutions_of_inequality_l760_76004

theorem negative_integer_solutions_of_inequality :
  {x : ℤ | x < 0 ∧ 3 * x + 1 ≥ -5} = {-2, -1} := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_of_inequality_l760_76004


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l760_76063

theorem min_sum_of_squares (a b c d : ℤ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 180 →
  c * d = 104 →
  ∃ (min : ℤ), min = 484 ∧ ∀ (a' b' c' d' : ℤ),
    a' + b' = 18 →
    a' * b' + c' + d' = 85 →
    a' * d' + b' * c' = 180 →
    c' * d' = 104 →
    a' ^ 2 + b' ^ 2 + c' ^ 2 + d' ^ 2 ≥ min :=
by sorry


end NUMINAMATH_CALUDE_min_sum_of_squares_l760_76063


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l760_76072

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l760_76072


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l760_76024

theorem arithmetic_calculation : -8 + (-10) - 3 - (-6) = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l760_76024


namespace NUMINAMATH_CALUDE_curve_tangent_product_l760_76066

/-- Given a curve y = ax³ + bx where the point (2, 2) lies on the curve
    and the slope of the tangent line at this point is 9,
    prove that the product ab equals -3. -/
theorem curve_tangent_product (a b : ℝ) : 
  (2 : ℝ) = a * (2 : ℝ)^3 + b * (2 : ℝ) → -- Point (2, 2) lies on the curve
  (9 : ℝ) = 3 * a * (2 : ℝ)^2 + b →       -- Slope of tangent at (2, 2) is 9
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_curve_tangent_product_l760_76066


namespace NUMINAMATH_CALUDE_two_students_same_school_probability_l760_76084

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The total number of possible outcomes --/
def total_outcomes : ℕ := num_schools ^ num_students

/-- The number of outcomes where exactly two students choose the same school --/
def favorable_outcomes : ℕ := num_students.choose 2 * num_schools * (num_schools - 1)

/-- The probability of exactly two students choosing the same school --/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem two_students_same_school_probability :
  probability = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_two_students_same_school_probability_l760_76084


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l760_76093

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not used in the proof)
def convex (s : Square) : Prop := true
def equiangular (s : Square) : Prop := true
def regular_polygon (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end NUMINAMATH_CALUDE_not_all_squares_congruent_l760_76093


namespace NUMINAMATH_CALUDE_square_in_base_seven_l760_76007

theorem square_in_base_seven :
  ∃ (b : ℕ) (h : b > 6), 
    (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_in_base_seven_l760_76007


namespace NUMINAMATH_CALUDE_quadratic_sum_l760_76042

/-- For the quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l760_76042


namespace NUMINAMATH_CALUDE_largest_m_for_negative_integral_solutions_l760_76039

def is_negative_integer (x : ℝ) : Prop := x < 0 ∧ ∃ n : ℤ, x = n

theorem largest_m_for_negative_integral_solutions :
  ∃ (m : ℝ),
    m = 570 ∧
    (∀ m' : ℝ, m' > m →
      ¬∃ (x y : ℝ),
        10 * x^2 - m' * x + 560 = 0 ∧
        10 * y^2 - m' * y + 560 = 0 ∧
        x ≠ y ∧
        is_negative_integer x ∧
        is_negative_integer y) ∧
    (∃ (x y : ℝ),
      10 * x^2 - m * x + 560 = 0 ∧
      10 * y^2 - m * y + 560 = 0 ∧
      x ≠ y ∧
      is_negative_integer x ∧
      is_negative_integer y) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_negative_integral_solutions_l760_76039


namespace NUMINAMATH_CALUDE_max_area_convex_quadrilateral_l760_76040

/-- A convex quadrilateral with diagonals d₁ and d₂ has an area S. -/
structure ConvexQuadrilateral where
  d₁ : ℝ
  d₂ : ℝ
  S : ℝ
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  S_pos : S > 0
  area_formula : ∃ α : ℝ, 0 ≤ α ∧ α ≤ π ∧ S = (1/2) * d₁ * d₂ * Real.sin α

/-- The maximum area of a convex quadrilateral is half the product of its diagonals. -/
theorem max_area_convex_quadrilateral (q : ConvexQuadrilateral) : 
  q.S ≤ (1/2) * q.d₁ * q.d₂ ∧ ∃ q' : ConvexQuadrilateral, q'.S = (1/2) * q'.d₁ * q'.d₂ := by
  sorry


end NUMINAMATH_CALUDE_max_area_convex_quadrilateral_l760_76040


namespace NUMINAMATH_CALUDE_age_doubling_time_l760_76012

/-- Given the ages of Wesley and Breenah, calculate the number of years until their combined age doubles -/
theorem age_doubling_time (wesley_age breenah_age : ℕ) (h1 : wesley_age = 15) (h2 : breenah_age = 7) 
  (h3 : wesley_age + breenah_age = 22) : 
  (fun n : ℕ => wesley_age + breenah_age + 2 * n = 2 * (wesley_age + breenah_age)) 11 := by
  sorry

end NUMINAMATH_CALUDE_age_doubling_time_l760_76012


namespace NUMINAMATH_CALUDE_fraction_addition_l760_76009

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l760_76009


namespace NUMINAMATH_CALUDE_billys_age_l760_76090

theorem billys_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 64) : B = 48 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l760_76090


namespace NUMINAMATH_CALUDE_pens_left_for_lenny_l760_76000

def total_pens : ℕ := 75 * 15

def friends_percentage : ℚ := 30 / 100
def classmates_percentage : ℚ := 20 / 100
def coworkers_percentage : ℚ := 25 / 100
def neighbors_percentage : ℚ := 15 / 100

def pens_after_friends : ℕ := total_pens - (Nat.floor (friends_percentage * total_pens))
def pens_after_classmates : ℕ := pens_after_friends - (Nat.floor (classmates_percentage * pens_after_friends))
def pens_after_coworkers : ℕ := pens_after_classmates - (Nat.floor (coworkers_percentage * pens_after_classmates))
def pens_after_neighbors : ℕ := pens_after_coworkers - (Nat.floor (neighbors_percentage * pens_after_coworkers))

theorem pens_left_for_lenny : pens_after_neighbors = 403 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_for_lenny_l760_76000


namespace NUMINAMATH_CALUDE_constant_term_expansion_l760_76028

/-- The constant term in the expansion of y^3 * (x + 1/(x^2*y))^n if it exists -/
def constantTerm (n : ℕ+) : ℕ :=
  if n = 9 then 84 else 0

theorem constant_term_expansion (n : ℕ+) :
  (∃ k : ℕ, k ≠ 0 ∧ constantTerm n = k) →
  constantTerm n = 84 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l760_76028


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l760_76099

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

-- Statement to prove
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l760_76099


namespace NUMINAMATH_CALUDE_least_positive_four_digit_octal_l760_76001

/-- The number of digits required to represent a positive integer in a given base -/
def numDigits (n : ℕ+) (base : ℕ) : ℕ :=
  Nat.log base n.val + 1

/-- Checks if a number requires at least four digits in base 8 -/
def requiresFourDigitsOctal (n : ℕ+) : Prop :=
  numDigits n 8 ≥ 4

theorem least_positive_four_digit_octal :
  ∃ (n : ℕ+), requiresFourDigitsOctal n ∧
    ∀ (m : ℕ+), m < n → ¬requiresFourDigitsOctal m ∧
    n = 512 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_four_digit_octal_l760_76001


namespace NUMINAMATH_CALUDE_power_division_rule_l760_76015

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l760_76015


namespace NUMINAMATH_CALUDE_sector_perimeter_l760_76091

/-- The perimeter of a circular sector with a central angle of 60° and a radius of 20 cm -/
theorem sector_perimeter : 
  let r : ℝ := 20
  let central_angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let arc_length : ℝ := r * central_angle
  let straight_sides : ℝ := 2 * r
  straight_sides + arc_length = 40 + 20 * π / 3 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l760_76091


namespace NUMINAMATH_CALUDE_triathlon_problem_l760_76020

/-- Triathlon problem -/
theorem triathlon_problem 
  (swim_distance : ℝ) 
  (cycle_distance : ℝ) 
  (run_distance : ℝ)
  (total_time : ℝ)
  (practice_swim_time : ℝ)
  (practice_cycle_time : ℝ)
  (practice_run_time : ℝ)
  (practice_total_distance : ℝ)
  (h_swim_distance : swim_distance = 1)
  (h_cycle_distance : cycle_distance = 25)
  (h_run_distance : run_distance = 4)
  (h_total_time : total_time = 5/4)
  (h_practice_swim_time : practice_swim_time = 1/16)
  (h_practice_cycle_time : practice_cycle_time = 1/49)
  (h_practice_run_time : practice_run_time = 1/49)
  (h_practice_total_distance : practice_total_distance = 5/4)
  (h_positive_speeds : ∀ v : ℝ, v > 0 → v + 1/v ≥ 2) :
  ∃ (cycle_time cycle_speed : ℝ),
    cycle_time = 5/7 ∧ 
    cycle_speed = 35 ∧
    cycle_distance / cycle_speed = cycle_time ∧
    swim_distance / (swim_distance / practice_swim_time) + 
    cycle_distance / cycle_speed + 
    run_distance / (run_distance / practice_run_time) = total_time ∧
    practice_swim_time * (swim_distance / practice_swim_time) + 
    practice_cycle_time * cycle_speed + 
    practice_run_time * (run_distance / practice_run_time) = practice_total_distance :=
by sorry


end NUMINAMATH_CALUDE_triathlon_problem_l760_76020


namespace NUMINAMATH_CALUDE_binomial_60_3_l760_76076

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l760_76076


namespace NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_three_l760_76013

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a as defined in the problem -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b as defined in the problem -/
def b (m : ℝ) : ℝ × ℝ := (1 + m, 1 - m)

/-- The main theorem: vectors a and b are parallel if and only if m = -3 -/
theorem parallel_iff_m_eq_neg_three :
  ∀ m : ℝ, are_parallel a (b m) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_three_l760_76013


namespace NUMINAMATH_CALUDE_asian_games_touring_routes_l760_76027

theorem asian_games_touring_routes :
  let total_cities : ℕ := 7
  let cities_to_visit : ℕ := 5
  let mandatory_cities : ℕ := 2
  let remaining_cities : ℕ := total_cities - mandatory_cities
  let cities_to_choose : ℕ := cities_to_visit - mandatory_cities
  let gaps : ℕ := cities_to_choose + 1

  (remaining_cities.factorial / (remaining_cities - cities_to_choose).factorial) *
  (gaps.choose mandatory_cities) = 600 :=
by sorry

end NUMINAMATH_CALUDE_asian_games_touring_routes_l760_76027


namespace NUMINAMATH_CALUDE_eight_book_distribution_l760_76048

/-- The number of ways to distribute identical books between a library and being checked out -/
def distribute_books (total : ℕ) : ℕ :=
  if total < 2 then 0 else total - 1

/-- Theorem: For 8 identical books, there are 7 ways to distribute them between a library and being checked out, with at least one book in each location -/
theorem eight_book_distribution :
  distribute_books 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_book_distribution_l760_76048


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l760_76064

theorem smallest_whole_number_above_sum : ⌈(10/3 : ℚ) + (17/4 : ℚ) + (26/5 : ℚ) + (37/6 : ℚ)⌉ = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l760_76064


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l760_76041

theorem sum_of_two_numbers (x y : ℤ) : x + y = 32 ∧ y = -36 → x = 68 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l760_76041


namespace NUMINAMATH_CALUDE_cubic_three_roots_l760_76033

-- Define the cubic function
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + k

-- State the theorem
theorem cubic_three_roots (k : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f k x₁ = 0 ∧ f k x₂ = 0 ∧ f k x₃ = 0 ∧
    ∀ x, f k x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  -5/27 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_l760_76033


namespace NUMINAMATH_CALUDE_not_prime_expression_l760_76074

theorem not_prime_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 6*n^n + 5) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_expression_l760_76074


namespace NUMINAMATH_CALUDE_archimedes_academy_students_l760_76075

/-- The number of distinct students preparing for AMC 8 at Archimedes Academy -/
def distinct_students (algebra_students calculus_students statistics_students overlap : ℕ) : ℕ :=
  algebra_students + calculus_students + statistics_students - overlap

/-- Theorem stating the number of distinct students preparing for AMC 8 at Archimedes Academy -/
theorem archimedes_academy_students :
  distinct_students 13 10 12 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_archimedes_academy_students_l760_76075


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l760_76078

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_3 = 7, S_6 = 63 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 * (1 - 2^3)) / (1 - 2) = 7 →  -- S_3 = 7
  (a 1 * (1 - 2^6)) / (1 - 2) = 63 :=  -- S_6 = 63
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l760_76078


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l760_76008

def x : ℕ := 2^2 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 9^9

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem smallest_factor_for_perfect_square :
  (∀ k : ℕ, k < 105 → ¬is_perfect_square (k * x)) ∧
  is_perfect_square (105 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l760_76008


namespace NUMINAMATH_CALUDE_prob_both_selected_l760_76080

/-- The probability of both brothers being selected in an exam -/
theorem prob_both_selected (p_x p_y : ℚ) (h_x : p_x = 1/5) (h_y : p_y = 2/3) :
  p_x * p_y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l760_76080


namespace NUMINAMATH_CALUDE_smallest_a_value_l760_76083

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (45 * x)) :
  a ≥ 45 ∧ ∃ a₀ : ℝ, a₀ ≥ 0 ∧ (∀ x : ℤ, Real.sin (a₀ * x + b) = Real.sin (45 * x)) ∧ a₀ = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l760_76083


namespace NUMINAMATH_CALUDE_jerry_payment_l760_76014

/-- Calculates the total payment for Jerry's work --/
theorem jerry_payment (painting_time counter_time_multiplier lawn_mowing_time hourly_rate : ℕ) 
  (h1 : counter_time_multiplier = 3)
  (h2 : painting_time = 8)
  (h3 : lawn_mowing_time = 6)
  (h4 : hourly_rate = 15) :
  (painting_time + counter_time_multiplier * painting_time + lawn_mowing_time) * hourly_rate = 570 :=
by sorry

end NUMINAMATH_CALUDE_jerry_payment_l760_76014


namespace NUMINAMATH_CALUDE_not_divisible_by_2310_l760_76062

theorem not_divisible_by_2310 (n : ℕ) (h : n < 2310) : ¬(2310 ∣ n * (2310 - n)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2310_l760_76062


namespace NUMINAMATH_CALUDE_single_point_implies_d_eq_seven_l760_76058

/-- The equation of the graph -/
def equation (x y d : ℝ) : ℝ := 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + d

/-- The graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, equation p.1 p.2 d = 0

/-- Theorem: If the equation represents a graph that consists of a single point, then d = 7 -/
theorem single_point_implies_d_eq_seven :
  ∃ d : ℝ, is_single_point d → d = 7 :=
sorry

end NUMINAMATH_CALUDE_single_point_implies_d_eq_seven_l760_76058


namespace NUMINAMATH_CALUDE_triangle_area_l760_76002

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  B = π / 3 →
  Real.cos A = 11 / 14 →
  let S := (1 / 2) * a * c * Real.sin B
  S = 10 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l760_76002


namespace NUMINAMATH_CALUDE_power_of_ten_zeros_l760_76081

theorem power_of_ten_zeros (n : ℕ) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_zeros_l760_76081


namespace NUMINAMATH_CALUDE_x_y_negative_l760_76055

theorem x_y_negative (x y : ℝ) (h1 : x - y > 2*x) (h2 : x + y < 0) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l760_76055


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_divided_pen_l760_76050

/-- The maximum area of a rectangular pen given a fixed perimeter --/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ) : 
  perimeter = 60 →
  area ≤ 225 ∧ 
  (∃ width height : ℝ, width > 0 ∧ height > 0 ∧ 2 * (width + height) = perimeter ∧ width * height = area) →
  (∀ width height : ℝ, width > 0 → height > 0 → 2 * (width + height) = perimeter → width * height ≤ 225) :=
by sorry

/-- The maximum area remains the same when divided into two equal sections --/
theorem max_area_divided_pen (perimeter : ℝ) (area : ℝ) (width height : ℝ) :
  perimeter = 60 →
  width > 0 →
  height > 0 →
  2 * (width + height) = perimeter →
  width * height = 225 →
  ∃ new_height : ℝ, new_height > 0 ∧ 2 * (width + new_height) = perimeter ∧ width * new_height = 225 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_divided_pen_l760_76050


namespace NUMINAMATH_CALUDE_price_increase_l760_76021

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + 30 / 100) = 1 + 62.5 / 100 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l760_76021


namespace NUMINAMATH_CALUDE_implication_proof_l760_76045

theorem implication_proof (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p → q) → r)) ∧
  ((¬p ∧ q ∧ r) → ((p → q) → r)) := by
  sorry

end NUMINAMATH_CALUDE_implication_proof_l760_76045


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l760_76079

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x^2 - (2 + a) * x + 2 > 0} = {x : ℝ | 2 / a < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l760_76079


namespace NUMINAMATH_CALUDE_slide_wait_time_l760_76061

theorem slide_wait_time (kids_swings : ℕ) (kids_slide : ℕ) (swing_wait_min : ℕ) (time_diff_sec : ℕ) :
  kids_swings = 3 →
  kids_slide = 2 * kids_swings →
  swing_wait_min = 2 →
  (kids_slide * swing_wait_min * 60 + time_diff_sec) - (kids_swings * swing_wait_min * 60) = 270 →
  kids_slide * swing_wait_min * 60 + time_diff_sec = 630 :=
by
  sorry

#check slide_wait_time

end NUMINAMATH_CALUDE_slide_wait_time_l760_76061


namespace NUMINAMATH_CALUDE_cello_practice_time_l760_76092

/-- Given a total practice time of 7.5 hours in a week, with 86 minutes of practice on each of 2 days,
    the remaining practice time on the other days is 278 minutes. -/
theorem cello_practice_time (total_hours : ℝ) (practice_minutes_per_day : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ practice_minutes_per_day = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (practice_minutes_per_day * practice_days : ℕ) = 278 := by
  sorry

end NUMINAMATH_CALUDE_cello_practice_time_l760_76092


namespace NUMINAMATH_CALUDE_three_families_ten_lines_form_150_triangles_l760_76067

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  sorry

/-- Theorem stating that three families of 10 parallel lines form 150 triangles -/
theorem three_families_ten_lines_form_150_triangles :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by sorry

end NUMINAMATH_CALUDE_three_families_ten_lines_form_150_triangles_l760_76067


namespace NUMINAMATH_CALUDE_vector_equation_solution_l760_76087

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) 
  (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l760_76087


namespace NUMINAMATH_CALUDE_square_perimeter_l760_76082

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 / 2 * s = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l760_76082


namespace NUMINAMATH_CALUDE_orthogonal_vectors_magnitude_l760_76023

def vector_a : ℝ × ℝ := (1, -3)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem orthogonal_vectors_magnitude (m : ℝ) 
  (h : vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) : 
  ‖(2 • vector_a - vector_b m)‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_magnitude_l760_76023


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l760_76098

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : 2 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l760_76098


namespace NUMINAMATH_CALUDE_four_objects_three_containers_l760_76017

/-- The number of ways to distribute n distinct objects into k distinct containers --/
def distributionWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct containers is 81 --/
theorem four_objects_three_containers : distributionWays 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_objects_three_containers_l760_76017


namespace NUMINAMATH_CALUDE_shared_foci_implies_a_equals_one_l760_76034

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 4 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 2 = 1

-- Theorem statement
theorem shared_foci_implies_a_equals_one :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, ellipse x y a ↔ hyperbola x y a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_shared_foci_implies_a_equals_one_l760_76034


namespace NUMINAMATH_CALUDE_sum_first_third_is_five_l760_76043

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 2
  inverse_sum : 1 / a 1 + 1 / a 3 = 5 / 4

/-- The sum of the first and third terms of the geometric sequence is 5 -/
theorem sum_first_third_is_five (seq : GeometricSequence) : seq.a 1 + seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_third_is_five_l760_76043


namespace NUMINAMATH_CALUDE_intersection_sum_l760_76011

/-- Given two lines that intersect at (4,3), prove that a + b = 7/4 -/
theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (3/4) * y + a ↔ y = (3/4) * x + b) → 
  (4 = (3/4) * 3 + a ∧ 3 = (3/4) * 4 + b) →
  a + b = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l760_76011


namespace NUMINAMATH_CALUDE_total_processing_fee_l760_76089

-- Define the problem parameters
def total_products : ℕ := 1000
def factory_a_rate : ℕ := 20
def factory_b_rate : ℕ := 25
def factory_a_fee : ℕ := 100
def factory_b_fee : ℕ := 125

-- Define the theorem
theorem total_processing_fee :
  let total_rate := factory_a_rate + factory_b_rate
  let processing_days := total_products / total_rate
  let total_fee := processing_days * (factory_a_fee + factory_b_fee)
  total_fee = 5000 := by sorry

end NUMINAMATH_CALUDE_total_processing_fee_l760_76089


namespace NUMINAMATH_CALUDE_robin_sodas_l760_76096

/-- The number of sodas Robin and her friends drank -/
def sodas_drunk : ℕ := 3

/-- The number of extra sodas Robin had -/
def sodas_extra : ℕ := 8

/-- The total number of sodas Robin bought -/
def total_sodas : ℕ := sodas_drunk + sodas_extra

theorem robin_sodas : total_sodas = 11 := by sorry

end NUMINAMATH_CALUDE_robin_sodas_l760_76096


namespace NUMINAMATH_CALUDE_paths_A_to_C_via_B_l760_76086

/-- The number of paths from A to B -/
def paths_A_to_B : ℕ := Nat.choose 6 2

/-- The number of paths from B to C -/
def paths_B_to_C : ℕ := Nat.choose 6 3

/-- The total number of steps from A to C -/
def total_steps : ℕ := 12

/-- The number of steps from A to B -/
def steps_A_to_B : ℕ := 6

/-- The number of steps from B to C -/
def steps_B_to_C : ℕ := 6

theorem paths_A_to_C_via_B : 
  paths_A_to_B * paths_B_to_C = 300 ∧ 
  steps_A_to_B + steps_B_to_C = total_steps :=
by sorry

end NUMINAMATH_CALUDE_paths_A_to_C_via_B_l760_76086


namespace NUMINAMATH_CALUDE_curve_eccentricity_l760_76032

-- Define the curve in polar coordinates
def polar_curve (ρ : ℝ) (θ : ℝ) : Prop :=
  ρ^2 * Real.cos (2 * θ) = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 2

-- Theorem statement
theorem curve_eccentricity :
  ∃ (e : ℝ), (∀ ρ θ, polar_curve ρ θ → eccentricity e) :=
sorry

end NUMINAMATH_CALUDE_curve_eccentricity_l760_76032


namespace NUMINAMATH_CALUDE_pressure_volume_relation_l760_76022

-- Define the constants for the problem
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3
def final_volume : ℝ := 6

-- Define the theorem
theorem pressure_volume_relation :
  ∀ (p1 p2 v1 v2 : ℝ),
    p1 > 0 → p2 > 0 → v1 > 0 → v2 > 0 →
    p1 = initial_pressure →
    v1 = initial_volume →
    v2 = final_volume →
    (p1 * v1 = p2 * v2) →
    p2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pressure_volume_relation_l760_76022


namespace NUMINAMATH_CALUDE_circle_area_decrease_l760_76031

theorem circle_area_decrease (r : ℝ) (h : r > 0) : 
  let r' := 0.8 * r
  let A := π * r^2
  let A' := π * r'^2
  (A - A') / A = 0.36 := by
sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l760_76031


namespace NUMINAMATH_CALUDE_problem_solving_probability_l760_76088

/-- The probability that Alex, Kyle, and Catherine solve a problem, but not Bella and David -/
theorem problem_solving_probability 
  (p_alex : ℚ) (p_bella : ℚ) (p_kyle : ℚ) (p_david : ℚ) (p_catherine : ℚ)
  (h_alex : p_alex = 1/4)
  (h_bella : p_bella = 3/5)
  (h_kyle : p_kyle = 1/3)
  (h_david : p_david = 2/7)
  (h_catherine : p_catherine = 5/9) :
  p_alex * p_kyle * p_catherine * (1 - p_bella) * (1 - p_david) = 25/378 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l760_76088


namespace NUMINAMATH_CALUDE_b_contribution_is_9000_l760_76037

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_join_month : ℕ
  total_months : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ

/-- Calculates B's contribution to the capital given the partnership details -/
def calculate_b_contribution (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that B's contribution is 9000 rupees given the problem conditions -/
theorem b_contribution_is_9000 :
  let p : Partnership := {
    a_initial_investment := 3500,
    b_join_month := 5,
    total_months := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 9000 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_is_9000_l760_76037


namespace NUMINAMATH_CALUDE_eight_books_three_piles_l760_76073

/-- The number of ways to divide n identical objects into k non-empty groups -/
def divide_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to divide 8 identical books into 3 piles -/
theorem eight_books_three_piles : divide_objects 8 3 = 5 := by sorry

end NUMINAMATH_CALUDE_eight_books_three_piles_l760_76073


namespace NUMINAMATH_CALUDE_student_line_arrangements_l760_76030

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of students who refuse to stand next to each other
def num_refusing_adjacent : ℕ := 2

-- Define the number of students who must stand at an end
def num_at_end : ℕ := 1

-- Function to calculate the number of arrangements
def num_arrangements (n : ℕ) (r : ℕ) (e : ℕ) : ℕ :=
  2 * (n.factorial - (n - r + 1).factorial * r.factorial)

-- Theorem statement
theorem student_line_arrangements :
  num_arrangements num_students num_refusing_adjacent num_at_end = 144 :=
by sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l760_76030


namespace NUMINAMATH_CALUDE_third_side_length_l760_76060

/-- Two similar triangles with given side lengths -/
structure SimilarTriangles where
  -- Larger triangle
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ
  -- Smaller triangle
  d : ℝ
  e : ℝ
  -- Conditions
  ha : a = 16
  hb : b = 20
  hc : c = 24
  hangle : angle = 30 * π / 180
  hd : d = 8
  he : e = 12
  -- Similarity condition
  similar : a / d = b / e

/-- The third side of the smaller triangle is 12 cm -/
theorem third_side_length (t : SimilarTriangles) : t.c / t.d = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l760_76060


namespace NUMINAMATH_CALUDE_hat_problem_l760_76019

/-- Proves that given the conditions of the hat problem, the number of green hats is 30 -/
theorem hat_problem (total_hats : ℕ) (blue_price green_price : ℕ) (total_price : ℕ)
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 540) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 30 :=
by sorry

end NUMINAMATH_CALUDE_hat_problem_l760_76019


namespace NUMINAMATH_CALUDE_investment_return_is_25_percent_l760_76003

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentageReturn (dividendRate : ℚ) (faceValue : ℚ) (purchasePrice : ℚ) : ℚ :=
  (dividendRate * faceValue / purchasePrice) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividendRate : ℚ := 125 / 1000
  let faceValue : ℚ := 40
  let purchasePrice : ℚ := 20
  percentageReturn dividendRate faceValue purchasePrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_is_25_percent_l760_76003


namespace NUMINAMATH_CALUDE_simplify_expression_l760_76026

theorem simplify_expression (a : ℝ) : 3 * a^5 * (4 * a^7) = 12 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l760_76026


namespace NUMINAMATH_CALUDE_point_coordinates_l760_76085

def Point := ℝ × ℝ

def x_coordinate (p : Point) : ℝ := p.1

def distance_to_x_axis (p : Point) : ℝ := |p.2|

theorem point_coordinates (P : Point) 
  (h1 : x_coordinate P = -3)
  (h2 : distance_to_x_axis P = 5) :
  P = (-3, 5) ∨ P = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l760_76085


namespace NUMINAMATH_CALUDE_sphere_volume_derivative_l760_76010

noncomputable section

-- Define the volume function for a sphere
def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

-- Define the surface area function for a sphere
def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- State the theorem
theorem sphere_volume_derivative (R : ℝ) (h : R > 0) :
  deriv sphere_volume R = sphere_surface_area R := by
  sorry

end

end NUMINAMATH_CALUDE_sphere_volume_derivative_l760_76010


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_geq_two_l760_76070

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_inclusion_implies_a_geq_two (a : ℝ) :
  A ⊆ B a → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_geq_two_l760_76070


namespace NUMINAMATH_CALUDE_unique_solution_abc_l760_76051

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Represents a base-7 number with one digit --/
def Base7OneDigit (c : ℕ) : ℕ := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 --/
def Base7TwoDigitWithZero (c : ℕ) : ℕ := 7 * c

theorem unique_solution_abc (A B C : ℕ) :
  (0 < A ∧ A < 7) →
  (0 < B ∧ B < 7) →
  (0 < C ∧ C < 7) →
  Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
  Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
  A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l760_76051


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l760_76046

/-- An arithmetic sequence {a_n} with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 7 = 22)
  (h3 : a 4 + a 10 = 40) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l760_76046


namespace NUMINAMATH_CALUDE_extra_flowers_l760_76068

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l760_76068
