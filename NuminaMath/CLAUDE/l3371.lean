import Mathlib

namespace NUMINAMATH_CALUDE_seedling_count_l3371_337158

theorem seedling_count (packets : ℕ) (seeds_per_packet : ℕ) 
  (h1 : packets = 60) (h2 : seeds_per_packet = 7) :
  packets * seeds_per_packet = 420 := by
  sorry

end NUMINAMATH_CALUDE_seedling_count_l3371_337158


namespace NUMINAMATH_CALUDE_three_X_seven_l3371_337137

/-- Operation X is defined as a X b = b + 10*a - a^2 + 2*a*b -/
def X (a b : ℤ) : ℤ := b + 10*a - a^2 + 2*a*b

/-- The value of 3X7 is 70 -/
theorem three_X_seven : X 3 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_three_X_seven_l3371_337137


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l3371_337170

/-- Given an arithmetic sequence with first term a₁, common difference d, and n-th term aₙ,
    this function calculates the n-th term. -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

#check fifth_term_of_sequence

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l3371_337170


namespace NUMINAMATH_CALUDE_number_of_girls_in_college_l3371_337115

theorem number_of_girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  total_students = 240 → ratio_boys = 5 → ratio_girls = 7 → 
  (ratio_girls * total_students) / (ratio_boys + ratio_girls) = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_college_l3371_337115


namespace NUMINAMATH_CALUDE_xixi_cards_problem_l3371_337199

theorem xixi_cards_problem (x y : ℕ) :
  (x + 3 = 3 * (y - 3) ∧ y + 4 = 4 * (x - 4)) ∨
  (x + 3 = 3 * (y - 3) ∧ x + 5 = 5 * (y - 5)) ∨
  (y + 4 = 4 * (x - 4) ∧ x + 5 = 5 * (y - 5)) →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_xixi_cards_problem_l3371_337199


namespace NUMINAMATH_CALUDE_emily_orange_count_l3371_337179

/-- The number of oranges each person has -/
structure OrangeCount where
  betty : ℕ
  sandra : ℕ
  emily : ℕ

/-- The conditions of the orange distribution problem -/
def orange_distribution (oc : OrangeCount) : Prop :=
  oc.emily = 7 * oc.sandra ∧
  oc.sandra = 3 * oc.betty ∧
  oc.betty = 12

/-- Theorem stating that Emily has 252 oranges given the conditions -/
theorem emily_orange_count (oc : OrangeCount) 
  (h : orange_distribution oc) : oc.emily = 252 := by
  sorry


end NUMINAMATH_CALUDE_emily_orange_count_l3371_337179


namespace NUMINAMATH_CALUDE_find_number_l3371_337177

theorem find_number : ∃ N : ℕ,
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  N / sum = quotient ∧ N % sum = 20 ∧ N = 220020 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3371_337177


namespace NUMINAMATH_CALUDE_difference_of_squares_l3371_337121

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3371_337121


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3371_337187

theorem sine_cosine_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3371_337187


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3371_337125

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3371_337125


namespace NUMINAMATH_CALUDE_max_red_socks_l3371_337101

theorem max_red_socks (a b : ℕ) : 
  a + b ≤ 1991 →
  (a.choose 2 + b.choose 2 : ℚ) / ((a + b).choose 2) = 1/2 →
  a ≤ 990 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l3371_337101


namespace NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3371_337183

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ 
  (dist P Q = s) ∧ (dist Q R = s) ∧ (dist R P = s)

-- Define the perimeter of the triangle
def Perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

-- Define the circumcircle and perpendicular bisectors
def CircumcirclePerpBisectors (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    dist O P = dist O Q ∧ dist O Q = dist O R ∧
    dist O P' = dist O Q' ∧ dist O Q' = dist O R' ∧
    (P'.1 + Q.1) / 2 = P'.1 ∧ (P'.2 + Q.2) / 2 = P'.2 ∧
    (Q'.1 + R.1) / 2 = Q'.1 ∧ (Q'.2 + R.2) / 2 = Q'.2 ∧
    (R'.1 + P.1) / 2 = R'.1 ∧ (R'.2 + P.2) / 2 = R'.2

-- Define the area of a hexagon
def HexagonArea (P Q' R Q' P R' : ℝ × ℝ) : ℝ :=
  sorry  -- Actual calculation of area would go here

-- The main theorem
theorem equilateral_triangle_hexagon_area 
  (P Q R P' Q' R' : ℝ × ℝ) :
  Triangle P Q R →
  Perimeter P Q R = 42 →
  CircumcirclePerpBisectors P Q R P' Q' R' →
  HexagonArea P Q' R Q' P R' = 49 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3371_337183


namespace NUMINAMATH_CALUDE_distinct_collections_proof_l3371_337123

/-- The number of letters in "COMPUTATION" -/
def word_length : ℕ := 11

/-- The number of vowels in "COMPUTATION" -/
def num_vowels : ℕ := 5

/-- The number of consonants in "COMPUTATION" -/
def num_consonants : ℕ := 6

/-- The number of indistinguishable T's in "COMPUTATION" -/
def num_ts : ℕ := 2

/-- The number of vowels removed -/
def vowels_removed : ℕ := 3

/-- The number of consonants removed -/
def consonants_removed : ℕ := 4

/-- The function to calculate the number of distinct possible collections -/
def distinct_collections : ℕ := 110

theorem distinct_collections_proof :
  distinct_collections = 110 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_proof_l3371_337123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3371_337165

theorem arithmetic_sequence_inequality (a b c : ℝ) (h : ∃ d : ℝ, d ≠ 0 ∧ b - a = d ∧ c - b = d) :
  ¬ (∀ a b c : ℝ, a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3371_337165


namespace NUMINAMATH_CALUDE_cost_difference_l3371_337157

def candy_bar_cost : ℝ := 6
def chocolate_cost : ℝ := 3

theorem cost_difference : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l3371_337157


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l3371_337150

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: The equation x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l3371_337150


namespace NUMINAMATH_CALUDE_lunch_total_is_fifteen_l3371_337100

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (friend_spent : ℕ) (difference : ℕ) : ℕ :=
  friend_spent + (friend_spent - difference)

/-- Theorem: The total amount spent on lunch is $15 -/
theorem lunch_total_is_fifteen :
  total_lunch_amount 10 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lunch_total_is_fifteen_l3371_337100


namespace NUMINAMATH_CALUDE_insurance_claims_percentage_l3371_337178

theorem insurance_claims_percentage (jan_claims missy_claims : ℕ) 
  (h1 : jan_claims = 20)
  (h2 : missy_claims = 41)
  (h3 : missy_claims = jan_claims + 15 + (jan_claims * 30 / 100)) :
  ∃ (john_claims : ℕ), 
    missy_claims = john_claims + 15 ∧ 
    john_claims = jan_claims + (jan_claims * 30 / 100) := by
  sorry

end NUMINAMATH_CALUDE_insurance_claims_percentage_l3371_337178


namespace NUMINAMATH_CALUDE_triangle_area_l3371_337172

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3371_337172


namespace NUMINAMATH_CALUDE_square_gt_abs_square_l3371_337159

theorem square_gt_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_abs_square_l3371_337159


namespace NUMINAMATH_CALUDE_unique_common_tangent_common_tangent_segments_bisect_l3371_337168

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 + 2*x
def C₂ (a x : ℝ) : ℝ := -x^2 + a

-- Define the common tangent line
def commonTangent (k b : ℝ) (x : ℝ) : ℝ := k*x + b

-- Define the tangency points
structure TangencyPoint where
  x : ℝ
  y : ℝ

-- Theorem for part 1
theorem unique_common_tangent (a : ℝ) :
  a = -1/2 →
  ∃! (k b : ℝ), ∀ (x : ℝ),
    (C₁ x = commonTangent k b x ∧ C₂ a x = commonTangent k b x) →
    k = 1 ∧ b = -1/4 :=
sorry

-- Theorem for part 2
theorem common_tangent_segments_bisect (a : ℝ) :
  a ≠ -1/2 →
  ∃ (A B C D : TangencyPoint),
    (C₁ A.x = commonTangent k₁ b₁ A.x ∧ C₂ a A.x = commonTangent k₁ b₁ A.x) ∧
    (C₁ B.x = commonTangent k₁ b₁ B.x ∧ C₂ a B.x = commonTangent k₁ b₁ B.x) ∧
    (C₁ C.x = commonTangent k₂ b₂ C.x ∧ C₂ a C.x = commonTangent k₂ b₂ C.x) ∧
    (C₁ D.x = commonTangent k₂ b₂ D.x ∧ C₂ a D.x = commonTangent k₂ b₂ D.x) →
    (A.x + C.x) / 2 = -1/2 ∧ (A.y + C.y) / 2 = (a - 1) / 2 ∧
    (B.x + D.x) / 2 = -1/2 ∧ (B.y + D.y) / 2 = (a - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_common_tangent_common_tangent_segments_bisect_l3371_337168


namespace NUMINAMATH_CALUDE_cousins_distribution_eq_67_l3371_337180

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_67 : cousins_distribution = 67 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_eq_67_l3371_337180


namespace NUMINAMATH_CALUDE_sameColorPairWithBlueCount_l3371_337173

/-- The number of ways to choose a pair of socks of the same color with at least one blue sock -/
def sameColorPairWithBlue (whiteCount brownCount blueCount greenCount : ℕ) : ℕ :=
  Nat.choose blueCount 2

theorem sameColorPairWithBlueCount :
  sameColorPairWithBlue 5 5 5 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sameColorPairWithBlueCount_l3371_337173


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l3371_337120

/-- A cubic polynomial with real coefficients. -/
def CubicPolynomial := ℝ → ℝ

/-- The property that a cubic polynomial satisfies the given conditions. -/
def SatisfiesConditions (g : CubicPolynomial) : Prop :=
  ∃ (a b c d : ℝ), 
    (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧
    (|g (-2)| = 6) ∧ (|g 0| = 6) ∧ (|g 1| = 6) ∧ (|g 4| = 6)

/-- The theorem stating that if a cubic polynomial satisfies the conditions, then |g(-1)| = 27/2. -/
theorem cubic_polynomial_property (g : CubicPolynomial) 
  (h : SatisfiesConditions g) : |g (-1)| = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l3371_337120


namespace NUMINAMATH_CALUDE_max_candies_after_20_hours_l3371_337185

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Next state of candies after one hour -/
def nextState (n : ℕ) : ℕ := n + sumOfDigits n

/-- State of candies after t hours, starting from 1 -/
def candyState (t : ℕ) : ℕ :=
  match t with
  | 0 => 1
  | t + 1 => nextState (candyState t)

theorem max_candies_after_20_hours :
  candyState 20 = 148 := by sorry

end NUMINAMATH_CALUDE_max_candies_after_20_hours_l3371_337185


namespace NUMINAMATH_CALUDE_sum_of_digits_a_l3371_337140

def a : ℕ := 10^101 - 100

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_a : sum_of_digits a = 891 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_a_l3371_337140


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3371_337160

theorem scientific_notation_equivalence : 22000000 = 2.2 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3371_337160


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l3371_337195

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l3371_337195


namespace NUMINAMATH_CALUDE_m_positive_sufficient_not_necessary_for_hyperbola_l3371_337108

-- Define a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ (x y : ℝ), x^2 / m - y^2 / m = 1

-- State the theorem
theorem m_positive_sufficient_not_necessary_for_hyperbola :
  ∃ (m : ℝ), m ≠ 0 ∧
  (∀ (m : ℝ), m > 0 → is_hyperbola m) ∧
  (∃ (m : ℝ), m < 0 ∧ is_hyperbola m) :=
sorry

end NUMINAMATH_CALUDE_m_positive_sufficient_not_necessary_for_hyperbola_l3371_337108


namespace NUMINAMATH_CALUDE_four_Y_three_equals_twentyfive_l3371_337152

-- Define the Y operation
def Y (a b : ℝ) : ℝ := (2 * a^2 - 3 * a * b + b^2)^2

-- Theorem statement
theorem four_Y_three_equals_twentyfive : Y 4 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_twentyfive_l3371_337152


namespace NUMINAMATH_CALUDE_adjacent_integers_product_l3371_337161

theorem adjacent_integers_product (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 740 → (x - 1) * x * (x + 1) = 17550 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_integers_product_l3371_337161


namespace NUMINAMATH_CALUDE_carpet_length_l3371_337127

theorem carpet_length (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_width : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_width = 4 →
  (carpet_coverage * floor_area) / carpet_width = 9 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_l3371_337127


namespace NUMINAMATH_CALUDE_pie_order_cost_l3371_337175

/-- Represents the cost of fruit for Michael's pie order -/
def total_cost (peach_pies apple_pies blueberry_pies : ℕ) 
  (fruit_per_pie : ℕ) 
  (peach_price apple_price blueberry_price : ℚ) : ℚ :=
  (peach_pies * fruit_per_pie : ℚ) * peach_price +
  (apple_pies * fruit_per_pie : ℚ) * apple_price +
  (blueberry_pies * fruit_per_pie : ℚ) * blueberry_price

/-- Theorem stating that the total cost of fruit for Michael's pie order is $51.00 -/
theorem pie_order_cost : 
  total_cost 5 4 3 3 2 1 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_pie_order_cost_l3371_337175


namespace NUMINAMATH_CALUDE_range_of_decreasing_function_l3371_337192

/-- A decreasing function on the real line. -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The range of a function. -/
def Range (f : ℝ → ℝ) : Set ℝ :=
  {y | ∃ x, f x = y}

/-- Theorem: For a decreasing function on the real line, 
    the range of values for a is (0,2]. -/
theorem range_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  Range f = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_decreasing_function_l3371_337192


namespace NUMINAMATH_CALUDE_school_population_theorem_l3371_337174

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  boys + girls = total →
  girls = (boys * total) / 100 →
  boys = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_population_theorem_l3371_337174


namespace NUMINAMATH_CALUDE_employee_salary_problem_l3371_337105

theorem employee_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 15 →
  manager_salary = 4200 →
  avg_increase = 150 →
  (∃ (avg_salary : ℕ),
    num_employees * avg_salary + manager_salary = (num_employees + 1) * (avg_salary + avg_increase) ∧
    avg_salary = 1800) :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l3371_337105


namespace NUMINAMATH_CALUDE_product_equals_specific_number_l3371_337129

theorem product_equals_specific_number : 333333 * (333333 + 1) = 111111222222 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_specific_number_l3371_337129


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3371_337154

-- Define the circle
def circle_center : ℝ × ℝ := (-2, 1)

-- Define the diameter endpoints
def diameter_endpoint_x : ℝ → ℝ × ℝ := λ a => (a, 0)
def diameter_endpoint_y : ℝ → ℝ × ℝ := λ b => (0, b)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y = 0

-- Theorem statement
theorem circle_equation_proof :
  ∃ (a b : ℝ),
    (diameter_endpoint_x a).1 + (diameter_endpoint_y b).1 = 2 * circle_center.1 ∧
    (diameter_endpoint_x a).2 + (diameter_endpoint_y b).2 = 2 * circle_center.2 →
    ∀ (x y : ℝ),
      (x - circle_center.1)^2 + (y - circle_center.2)^2 = 
        ((diameter_endpoint_x a).1 - (diameter_endpoint_y b).1)^2 / 4 +
        ((diameter_endpoint_x a).2 - (diameter_endpoint_y b).2)^2 / 4 →
      circle_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3371_337154


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3371_337138

theorem quadratic_inequality_solution (a b k : ℝ) : 
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ (x < 1 ∨ x > b)) →
  b > 1 →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3371_337138


namespace NUMINAMATH_CALUDE_congruence_solution_l3371_337145

theorem congruence_solution (n : ℕ) : n < 47 → (13 * n) % 47 = 9 % 47 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3371_337145


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3371_337153

/-- A line passing through (3, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, 2)
  point_condition : 2 = m * 3 + b
  -- The x and y intercepts are equal and non-zero
  intercept_condition : ∃ (a : ℝ), a ≠ 0 ∧ a = b ∧ a = -b/m

/-- The equation of the line with equal intercepts passing through (3, 2) is x + y = 5 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  l.m = -1 ∧ l.b = 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3371_337153


namespace NUMINAMATH_CALUDE_horses_meet_after_nine_days_l3371_337143

/-- The distance from Chang'an to Qi in li -/
def total_distance : ℝ := 1125

/-- The distance traveled by the good horse on the first day in li -/
def good_horse_initial : ℝ := 103

/-- The daily increase in distance for the good horse in li -/
def good_horse_increase : ℝ := 13

/-- The distance traveled by the mediocre horse on the first day in li -/
def mediocre_horse_initial : ℝ := 97

/-- The daily decrease in distance for the mediocre horse in li -/
def mediocre_horse_decrease : ℝ := 0.5

/-- The number of days it takes for the horses to meet -/
def meeting_days : ℕ := 9

/-- Theorem stating that the horses meet after 9 days -/
theorem horses_meet_after_nine_days :
  (good_horse_initial * meeting_days + (meeting_days * (meeting_days - 1) / 2) * good_horse_increase +
   mediocre_horse_initial * meeting_days - (meeting_days * (meeting_days - 1) / 2) * mediocre_horse_decrease) =
  2 * total_distance := by
  sorry

#check horses_meet_after_nine_days

end NUMINAMATH_CALUDE_horses_meet_after_nine_days_l3371_337143


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l3371_337130

theorem greatest_power_of_two (n : ℕ) : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (10^1002 - 4^501) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (10^1002 - 4^501) → m ≤ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l3371_337130


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3371_337124

theorem largest_constant_inequality (D : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 ≥ D * (x + y)) ↔ D ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3371_337124


namespace NUMINAMATH_CALUDE_initial_gold_percentage_l3371_337142

/-- Given an alloy weighing 48 ounces, adding 12 ounces of pure gold results in a new alloy that is 40% gold.
    This theorem proves that the initial percentage of gold in the alloy is 25%. -/
theorem initial_gold_percentage (initial_weight : ℝ) (added_gold : ℝ) (final_percentage : ℝ) :
  initial_weight = 48 →
  added_gold = 12 →
  final_percentage = 40 →
  (initial_weight * (25 / 100) + added_gold) / (initial_weight + added_gold) = final_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_gold_percentage_l3371_337142


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3371_337147

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Proves that given the specified conditions, the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry

#eval speed_against_current 15 3.2

end NUMINAMATH_CALUDE_mans_speed_against_current_l3371_337147


namespace NUMINAMATH_CALUDE_min_distance_to_point_l3371_337102

theorem min_distance_to_point (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2 = 0 → x'^2 + (y' - 2)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l3371_337102


namespace NUMINAMATH_CALUDE_odd_function_condition_l3371_337132

/-- The function f(x) defined as (3^(x+1) - 1) / (3^x - 1) + a * (sin x + cos x)^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (3^(x+1) - 1) / (3^x - 1) + a * (Real.sin x + Real.cos x)^2

/-- Theorem stating that for f to be an odd function, a must equal -2 --/
theorem odd_function_condition (a : ℝ) : 
  (∀ x, f a x = -f a (-x)) ↔ a = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l3371_337132


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l3371_337141

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating the bus driver's regular rate given the compensation conditions -/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 17)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 1116)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
        comp.overtimeRate * comp.overtimeHours) : 
  comp.regularRate = 16 := by
  sorry

#check bus_driver_regular_rate

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l3371_337141


namespace NUMINAMATH_CALUDE_friends_game_sales_l3371_337116

/-- The amount of money received by Zachary -/
def zachary_amount : ℚ := 40 * 5

/-- The amount of money received by Jason -/
def jason_amount : ℚ := zachary_amount * (1 + 30 / 100)

/-- The amount of money received by Ryan -/
def ryan_amount : ℚ := jason_amount + 50

/-- The amount of money received by Emily -/
def emily_amount : ℚ := ryan_amount * (1 - 20 / 100)

/-- The amount of money received by Lily -/
def lily_amount : ℚ := emily_amount + 70

/-- The total amount of money received by all five friends -/
def total_amount : ℚ := zachary_amount + jason_amount + ryan_amount + emily_amount + lily_amount

theorem friends_game_sales : total_amount = 1336 := by
  sorry

end NUMINAMATH_CALUDE_friends_game_sales_l3371_337116


namespace NUMINAMATH_CALUDE_gift_wrapping_l3371_337128

theorem gift_wrapping (total_gifts : ℕ) (total_rolls : ℕ) (first_roll_gifts : ℕ) (third_roll_gifts : ℕ) :
  total_gifts = 12 →
  total_rolls = 3 →
  first_roll_gifts = 3 →
  third_roll_gifts = 4 →
  ∃ (second_roll_gifts : ℕ),
    first_roll_gifts + second_roll_gifts + third_roll_gifts = total_gifts ∧
    second_roll_gifts = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l3371_337128


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l3371_337151

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l3371_337151


namespace NUMINAMATH_CALUDE_event_A_is_certain_l3371_337196

/-- The set of card labels -/
def card_labels : Finset ℕ := {1, 2, 3, 4, 5}

/-- The event "The label is less than 6" -/
def event_A (n : ℕ) : Prop := n < 6

/-- Theorem: Event A is a certain event -/
theorem event_A_is_certain : ∀ n ∈ card_labels, event_A n := by
  sorry

end NUMINAMATH_CALUDE_event_A_is_certain_l3371_337196


namespace NUMINAMATH_CALUDE_inequality_range_l3371_337146

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) → 
  (a ≤ -1 ∨ a ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3371_337146


namespace NUMINAMATH_CALUDE_same_solution_value_of_b_l3371_337162

theorem same_solution_value_of_b : ∀ x b : ℝ, 
  (3 * x + 9 = 6) ∧ 
  (5 * b * x - 15 = 5) → 
  b = -4 := by sorry

end NUMINAMATH_CALUDE_same_solution_value_of_b_l3371_337162


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3371_337148

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 100 → initial_money = 250 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l3371_337148


namespace NUMINAMATH_CALUDE_probability_black_not_white_is_three_fifths_l3371_337117

structure Bag where
  total : ℕ
  white : ℕ
  black : ℕ
  red : ℕ

def probability_black_given_not_white (b : Bag) : ℚ :=
  b.black / (b.total - b.white)

theorem probability_black_not_white_is_three_fifths (b : Bag) 
  (h1 : b.total = 10)
  (h2 : b.white = 5)
  (h3 : b.black = 3)
  (h4 : b.red = 2) :
  probability_black_given_not_white b = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_not_white_is_three_fifths_l3371_337117


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l3371_337114

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp : ℕ := 610769

/-- The number of kids from Lawrence county who stay home -/
def kids_at_home : ℕ := 590796

/-- The number of kids from outside the county who attended the camp -/
def outside_kids_at_camp : ℕ := 22

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_at_camp + kids_at_home

theorem lawrence_county_kids_count :
  total_kids = 1201565 :=
sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l3371_337114


namespace NUMINAMATH_CALUDE_seventh_term_is_ten_l3371_337189

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 = 2) ∧ 
  (a 4 + a 5 = 12)

/-- Theorem stating that the 7th term of the arithmetic sequence is 10 -/
theorem seventh_term_is_ten (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_ten_l3371_337189


namespace NUMINAMATH_CALUDE_locus_of_parabola_vertices_l3371_337164

/-- The locus of vertices of parabolas y = x^2 + tx + 1 is y = 1 - x^2 -/
theorem locus_of_parabola_vertices :
  ∀ (t : ℝ), 
  let f (x : ℝ) := x^2 + t*x + 1
  let vertex := (- t/2, f (- t/2))
  (vertex.1)^2 + vertex.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_locus_of_parabola_vertices_l3371_337164


namespace NUMINAMATH_CALUDE_fishing_tournament_result_l3371_337190

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier emily_multiplier : ℕ) 
  (alex_loss emily_loss : ℕ) : ℕ :=
  let alex_initial := alex_multiplier * jacob_initial
  let emily_initial := emily_multiplier * jacob_initial
  let alex_final := alex_initial - alex_loss
  let emily_final := emily_initial - emily_loss
  let target := max alex_final emily_final + 1
  target - jacob_initial

theorem fishing_tournament_result : 
  fishing_tournament 8 7 3 23 10 = 26 := by sorry

end NUMINAMATH_CALUDE_fishing_tournament_result_l3371_337190


namespace NUMINAMATH_CALUDE_contains_2850_of_0_001_l3371_337109

theorem contains_2850_of_0_001 : 2.85 = 2850 * 0.001 := by
  sorry

end NUMINAMATH_CALUDE_contains_2850_of_0_001_l3371_337109


namespace NUMINAMATH_CALUDE_divisors_of_720_l3371_337133

theorem divisors_of_720 : ∃ (n : ℕ), n = 720 ∧ (Finset.card (Finset.filter (λ x => n % x = 0) (Finset.range (n + 1)))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l3371_337133


namespace NUMINAMATH_CALUDE_danny_thrice_jane_age_l3371_337113

theorem danny_thrice_jane_age (danny_age jane_age : ℕ) (h1 : danny_age = 40) (h2 : jane_age = 26) :
  ∃ x : ℕ, x ≤ jane_age ∧ danny_age - x = 3 * (jane_age - x) ∧ x = 19 :=
by sorry

end NUMINAMATH_CALUDE_danny_thrice_jane_age_l3371_337113


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l3371_337112

theorem triangular_array_coin_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 2016 → (N / 10 + N % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l3371_337112


namespace NUMINAMATH_CALUDE_division_chain_l3371_337176

theorem division_chain : (180 / 6) / 3 / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_division_chain_l3371_337176


namespace NUMINAMATH_CALUDE_puppies_sold_calculation_l3371_337135

-- Define the given conditions
def initial_puppies : ℕ := 18
def puppies_per_cage : ℕ := 5
def cages_used : ℕ := 3

-- Define the theorem
theorem puppies_sold_calculation :
  initial_puppies - (cages_used * puppies_per_cage) = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_calculation_l3371_337135


namespace NUMINAMATH_CALUDE_karen_late_start_l3371_337118

/-- Proves that Karen starts the race 4 minutes late given the conditions of the car race. -/
theorem karen_late_start (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_win_margin = 4 →
  (tom_distance / tom_speed * 60 - (tom_distance + karen_win_margin) / karen_speed * 60 : ℝ) = 4 := by
  sorry

#check karen_late_start

end NUMINAMATH_CALUDE_karen_late_start_l3371_337118


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l3371_337156

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l3371_337156


namespace NUMINAMATH_CALUDE_circle_square_area_ratio_l3371_337171

theorem circle_square_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
  r > 0 →
  s > 0 →
  r = s * (Real.sqrt 2) / 2 →
  (π * r^2) / (s^2) = π / 2 := by
sorry

end NUMINAMATH_CALUDE_circle_square_area_ratio_l3371_337171


namespace NUMINAMATH_CALUDE_distribution_ways_4_5_l3371_337169

/-- The number of ways to distribute men and women into groups -/
def distribution_ways (num_men num_women : ℕ) : ℕ :=
  let scenario1 := num_men.choose 1 * num_women.choose 2 * (3 * 2)
  let scenario2 := num_men.choose 2 * num_women.choose 1 * (num_women.choose 2 * 2)
  scenario1 + scenario2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_ways_4_5 :
  distribution_ways 4 5 = 600 := by
  sorry

#eval distribution_ways 4 5

end NUMINAMATH_CALUDE_distribution_ways_4_5_l3371_337169


namespace NUMINAMATH_CALUDE_ten_digit_square_plus_one_has_identical_digits_l3371_337139

-- Define a function to check if a number is ten digits
def isTenDigits (n : ℕ) : Prop := 1000000000 ≤ n ∧ n < 10000000000

-- Define a function to check if a number has at least two identical digits
def hasAtLeastTwoIdenticalDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (n / 10^i % 10 = d) ∧ (n / 10^j % 10 = d))

theorem ten_digit_square_plus_one_has_identical_digits (n : ℕ) :
  isTenDigits (n^2 + 1) → hasAtLeastTwoIdenticalDigits (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ten_digit_square_plus_one_has_identical_digits_l3371_337139


namespace NUMINAMATH_CALUDE_finite_solutions_cube_sum_l3371_337122

theorem finite_solutions_cube_sum (n : ℕ) : 
  Finite {p : ℤ × ℤ | (p.1 ^ 3 + p.2 ^ 3 : ℤ) = n} := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_cube_sum_l3371_337122


namespace NUMINAMATH_CALUDE_additional_wolves_in_pack_l3371_337184

/-- The number of additional wolves in a pack, given hunting conditions -/
def additional_wolves (hunting_wolves : ℕ) (meat_per_day : ℕ) (hunting_period : ℕ) 
                      (meat_per_deer : ℕ) (deer_per_hunter : ℕ) : ℕ :=
  let total_meat := hunting_wolves * deer_per_hunter * meat_per_deer
  let wolves_fed := total_meat / (meat_per_day * hunting_period)
  wolves_fed - hunting_wolves

theorem additional_wolves_in_pack : 
  additional_wolves 4 8 5 200 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_additional_wolves_in_pack_l3371_337184


namespace NUMINAMATH_CALUDE_lower_bound_x_l3371_337107

theorem lower_bound_x (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8)
  (h_diff : ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n = 4) : 3 < x :=
sorry

end NUMINAMATH_CALUDE_lower_bound_x_l3371_337107


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l3371_337194

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def SecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The given point P with coordinates (-1, 2). -/
def P : ℝ × ℝ := (-1, 2)

/-- Theorem: The point P lies in the second quadrant. -/
theorem P_in_second_quadrant : SecondQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_second_quadrant_l3371_337194


namespace NUMINAMATH_CALUDE_monotonicity_interval_min_value_on_interval_max_value_on_interval_l3371_337119

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the interval of monotonicity
theorem monotonicity_interval :
  ∃ (a b : ℝ), a < b ∧ (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨ (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y) :=
sorry

-- Theorem for the minimum value on the interval [-3, 2]
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = -18 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≥ f x :=
sorry

-- Theorem for the maximum value on the interval [-3, 2]
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = 2 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_monotonicity_interval_min_value_on_interval_max_value_on_interval_l3371_337119


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3371_337198

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 7 * (-27 : ℚ) / 3⌉ - ⌊(15 : ℚ) / 7 * ⌈(-27 : ℚ) / 3⌉⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3371_337198


namespace NUMINAMATH_CALUDE_jessicas_allowance_l3371_337193

/-- Jessica's weekly allowance problem -/
theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_allowance_l3371_337193


namespace NUMINAMATH_CALUDE_complex_absolute_value_sum_l3371_337136

theorem complex_absolute_value_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_sum_l3371_337136


namespace NUMINAMATH_CALUDE_james_number_problem_l3371_337103

theorem james_number_problem (x : ℝ) : 3 * ((3 * x + 15) - 5) = 141 → x = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_number_problem_l3371_337103


namespace NUMINAMATH_CALUDE_fundraising_problem_l3371_337186

/-- The number of workers in a fundraising scenario -/
def number_of_workers : ℕ := sorry

/-- The original contribution amount per worker -/
def original_contribution : ℕ := sorry

theorem fundraising_problem :
  (number_of_workers * original_contribution = 300000) ∧
  (number_of_workers * (original_contribution + 50) = 350000) →
  number_of_workers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_problem_l3371_337186


namespace NUMINAMATH_CALUDE_total_whales_count_l3371_337134

/-- The total number of whales observed across three trips -/
def total_whales : ℕ := by sorry

/-- The number of male whales observed in the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales observed in the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales observed in the second trip -/
def second_trip_babies : ℕ := 8

/-- The number of whales in each family group (baby + two parents) -/
def whales_per_family : ℕ := 3

/-- The number of male whales observed in the third trip -/
def third_trip_males : ℕ := first_trip_males / 2

/-- The number of female whales observed in the third trip -/
def third_trip_females : ℕ := first_trip_females

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_count : total_whales = 178 := by sorry

end NUMINAMATH_CALUDE_total_whales_count_l3371_337134


namespace NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l3371_337106

theorem least_prime_factor_of_9_4_minus_9_3 :
  Nat.minFac (9^4 - 9^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l3371_337106


namespace NUMINAMATH_CALUDE_interval_property_l3371_337104

def f (x : ℝ) : ℝ := |x - 1|

theorem interval_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ ≥ f x₂) →
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_interval_property_l3371_337104


namespace NUMINAMATH_CALUDE_largest_b_value_l3371_337163

theorem largest_b_value : ∃ b_max : ℝ,
  (∀ b : ℝ, (3 * b + 7) * (b - 2) = 4 * b → b ≤ b_max) ∧
  ((3 * b_max + 7) * (b_max - 2) = 4 * b_max) ∧
  b_max = 81.5205 / 30 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l3371_337163


namespace NUMINAMATH_CALUDE_total_annual_earnings_l3371_337110

/-- Represents the harvest frequency (in months) and sale price for a fruit type -/
structure FruitInfo where
  harvestFrequency : Nat
  salePrice : Nat

/-- Calculates the annual earnings for a single fruit type -/
def annualEarnings (fruit : FruitInfo) : Nat :=
  (12 / fruit.harvestFrequency) * fruit.salePrice

/-- The farm's fruit information -/
def farmFruits : List FruitInfo := [
  ⟨2, 50⟩,  -- Oranges
  ⟨3, 30⟩,  -- Apples
  ⟨4, 45⟩,  -- Peaches
  ⟨6, 70⟩   -- Blackberries
]

/-- Theorem: The total annual earnings from selling the farm's fruits is $695 -/
theorem total_annual_earnings : 
  (farmFruits.map annualEarnings).sum = 695 := by
  sorry

end NUMINAMATH_CALUDE_total_annual_earnings_l3371_337110


namespace NUMINAMATH_CALUDE_election_vote_difference_l3371_337131

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 10000 → 
  candidate_percentage = 30/100 → 
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
sorry

end NUMINAMATH_CALUDE_election_vote_difference_l3371_337131


namespace NUMINAMATH_CALUDE_baby_births_theorem_l3371_337197

theorem baby_births_theorem (k : ℕ) (x : ℕ → ℕ) 
  (h1 : 1014 < k) (h2 : k ≤ 2014)
  (h3 : x 0 = 0) (h4 : x k = 2014)
  (h5 : ∀ i, i < k → x i < x (i + 1)) :
  ∃ i j, i < j ∧ j ≤ k ∧ x j - x i = 100 := by
sorry

end NUMINAMATH_CALUDE_baby_births_theorem_l3371_337197


namespace NUMINAMATH_CALUDE_function_identity_l3371_337188

def is_positive_integer (n : ℕ) : Prop := n > 0

theorem function_identity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n, is_positive_integer n → is_positive_integer (f n))
  (h2 : ∀ n, is_positive_integer n → f (n + 1) > f (f n)) :
  ∀ n, is_positive_integer n → f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l3371_337188


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3371_337126

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 2, 4]
  A * B = !![17, 1; 16, -12] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3371_337126


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3371_337149

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3371_337149


namespace NUMINAMATH_CALUDE_negative_two_squared_times_negative_one_to_2015_l3371_337167

theorem negative_two_squared_times_negative_one_to_2015 : -2^2 * (-1)^2015 = 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_times_negative_one_to_2015_l3371_337167


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3371_337182

-- Problem 1
theorem problem_1 : -105 - (-112) + 20 + 18 = 45 := by
  sorry

-- Problem 2
theorem problem_2 : 13 + (-22) - 25 - (-18) = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3371_337182


namespace NUMINAMATH_CALUDE_paper_fold_unfold_holes_l3371_337111

/-- Represents a rectangular piece of paper --/
structure Paper where
  height : ℕ
  width : ℕ

/-- Represents the position of a hole on the paper --/
structure HolePosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the paper after folding and hole punching --/
structure FoldedPaper where
  original : Paper
  folds : List (Paper → Paper)
  holePosition : HolePosition

/-- Function to calculate the number and arrangement of holes after unfolding --/
def unfoldAndCount (fp : FoldedPaper) : ℕ × List HolePosition :=
  sorry

/-- Theorem stating the result of folding and unfolding the paper --/
theorem paper_fold_unfold_holes :
  ∀ (fp : FoldedPaper),
    fp.original = Paper.mk 8 12 →
    fp.folds.length = 3 →
    (unfoldAndCount fp).1 = 8 ∧
    (∃ (col1 col2 : ℕ), ∀ (pos : HolePosition),
      pos ∈ (unfoldAndCount fp).2 →
      (pos.x = col1 ∨ pos.x = col2) ∧
      pos.y ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_paper_fold_unfold_holes_l3371_337111


namespace NUMINAMATH_CALUDE_center_is_B_l3371_337166

-- Define the points
variable (A B C D P Q K L : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1 ∧
  B = (1 - t) • A + t • D ∧
  C = (1 - t) • B + t • D

axiom AB_eq_BC : dist A B = dist B C

axiom perp_B : (B.2 - A.2) * (P.1 - B.1) = (B.1 - A.1) * (P.2 - B.2) ∧
               (B.2 - A.2) * (Q.1 - B.1) = (B.1 - A.1) * (Q.2 - B.2)

axiom perp_C : (C.2 - B.2) * (K.1 - C.1) = (C.1 - B.1) * (K.2 - C.2) ∧
               (C.2 - B.2) * (L.1 - C.1) = (C.1 - B.1) * (L.2 - C.2)

axiom on_circle_AD : dist A P + dist P D = dist A D ∧
                     dist A Q + dist Q D = dist A D

axiom on_circle_BD : dist B K + dist K D = dist B D ∧
                     dist B L + dist L D = dist B D

-- State the theorem
theorem center_is_B : 
  dist B P = dist B K ∧ dist B K = dist B L ∧ dist B L = dist B Q :=
sorry

end NUMINAMATH_CALUDE_center_is_B_l3371_337166


namespace NUMINAMATH_CALUDE_jerry_weighted_mean_l3371_337181

/-- Represents different currencies --/
inductive Currency
| USD
| EUR
| GBP
| CAD

/-- Represents a monetary amount with its currency --/
structure Money where
  amount : Float
  currency : Currency

/-- Represents a gift with its source --/
structure Gift where
  amount : Money
  isFamily : Bool

/-- Exchange rates to USD --/
def exchangeRate (c : Currency) : Float :=
  match c with
  | Currency.USD => 1
  | Currency.EUR => 1.20
  | Currency.GBP => 1.38
  | Currency.CAD => 0.82

/-- Convert Money to USD --/
def toUSD (m : Money) : Float :=
  m.amount * exchangeRate m.currency

/-- List of all gifts Jerry received --/
def jerryGifts : List Gift := [
  { amount := { amount := 9.73, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 9.43, currency := Currency.EUR }, isFamily := true },
  { amount := { amount := 22.16, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 23.51, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 18.72, currency := Currency.EUR }, isFamily := false },
  { amount := { amount := 15.53, currency := Currency.GBP }, isFamily := false },
  { amount := { amount := 22.84, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 7.25, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 20.37, currency := Currency.CAD }, isFamily := true }
]

/-- Calculate weighted mean of Jerry's gifts --/
def weightedMean (gifts : List Gift) : Float :=
  let familyGifts := gifts.filter (λ g => g.isFamily)
  let friendGifts := gifts.filter (λ g => ¬g.isFamily)
  let familySum := familyGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  let friendSum := friendGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  familySum * 0.4 + friendSum * 0.6

/-- Theorem: The weighted mean of Jerry's birthday money in USD is $85.4442 --/
theorem jerry_weighted_mean :
  weightedMean jerryGifts = 85.4442 := by
  sorry

end NUMINAMATH_CALUDE_jerry_weighted_mean_l3371_337181


namespace NUMINAMATH_CALUDE_profit_calculation_min_model_A_bicycles_l3371_337155

-- Define the profit functions for models A and B
def profit_A : ℝ := 150
def profit_B : ℝ := 100

-- Define the purchase prices
def price_A : ℝ := 500
def price_B : ℝ := 800

-- Define the total number of bicycles and budget
def total_bicycles : ℕ := 20
def max_budget : ℝ := 13000

-- Theorem for part 1
theorem profit_calculation :
  3 * profit_A + 2 * profit_B = 650 ∧
  profit_A + 2 * profit_B = 350 := by sorry

-- Theorem for part 2
theorem min_model_A_bicycles :
  ∀ m : ℕ,
  (m ≤ total_bicycles ∧ 
   price_A * m + price_B * (total_bicycles - m) ≤ max_budget) →
  m ≥ 10 := by sorry

end NUMINAMATH_CALUDE_profit_calculation_min_model_A_bicycles_l3371_337155


namespace NUMINAMATH_CALUDE_circle_cutting_theorem_l3371_337144

-- Define the circle C1 with center O and radius r
def C1 (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define a point A on the circumference of C1
def A_on_C1 (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2

-- Define the existence of a line that cuts C1 into two parts
def cutting_line_exists (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ (B C : ℝ × ℝ), 
    B ∈ C1 O r ∧ C ∈ C1 O r ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = r^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2

-- Theorem statement
theorem circle_cutting_theorem (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) :
  A_on_C1 O r A → cutting_line_exists O r A := by
  sorry

end NUMINAMATH_CALUDE_circle_cutting_theorem_l3371_337144


namespace NUMINAMATH_CALUDE_expression_equals_19_96_l3371_337191

theorem expression_equals_19_96 : 
  (7 * (19 / 2015) * (6 * (19 / 2016)) - 13 * (1996 / 2015) * (2 * (1997 / 2016)) - 9 * (19 / 2015)) = 19 / 96 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_19_96_l3371_337191
