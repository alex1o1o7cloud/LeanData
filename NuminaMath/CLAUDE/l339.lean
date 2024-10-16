import Mathlib

namespace NUMINAMATH_CALUDE_original_bacteria_count_l339_33980

theorem original_bacteria_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 8917)
  (h2 : increase = 8317)
  (h3 : current = original + increase) :
  original = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_bacteria_count_l339_33980


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l339_33998

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem cross_number_puzzle :
  ∃ (m n : ℕ),
    is_three_digit (3^m) ∧
    is_three_digit (7^n) ∧
    (3^m / 10) % 10 = (7^n / 10) % 10 ∧
    (3^m / 10) % 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l339_33998


namespace NUMINAMATH_CALUDE_negation_of_exp_greater_than_x_l339_33957

theorem negation_of_exp_greater_than_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_exp_greater_than_x_l339_33957


namespace NUMINAMATH_CALUDE_sequence_inequality_l339_33917

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) : 
  a 2 * a 4 ≤ a 3 ^ 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l339_33917


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l339_33927

theorem shirt_cost_theorem (first_shirt_cost second_shirt_cost total_cost : ℕ) : 
  first_shirt_cost = 15 →
  first_shirt_cost = second_shirt_cost + 6 →
  total_cost = first_shirt_cost + second_shirt_cost →
  total_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l339_33927


namespace NUMINAMATH_CALUDE_expression_value_l339_33953

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  ∃ ε > 0, |x + (2 * x^3 / y^2) + (2 * y^3 / x^2) + y - 338| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_l339_33953


namespace NUMINAMATH_CALUDE_problem_solution_l339_33997

/-- Checks if a sequence of binomial coefficients forms an arithmetic sequence -/
def is_arithmetic_sequence (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i < k - 1 → 2 * (n.choose (j + i + 1)) = (n.choose (j + i)) + (n.choose (j + i + 2))

/-- The value of k that satisfies the conditions of the problem -/
def k : ℕ := 4

/-- The condition (a) of the problem -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ∀ j : ℕ, j ≤ n - k + 1 → ¬(is_arithmetic_sequence n j k)

/-- The condition (b) of the problem -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1)

/-- The form of n that satisfies condition (b) for k = 4 -/
def valid_n (m : ℕ) : ℕ := m^2 - 2

theorem problem_solution :
  condition_a k ∧
  condition_b k ∧
  (∀ n : ℕ, n > 0 → (∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1))
                 ↔ (∃ m : ℕ, m ≥ 3 ∧ n = valid_n m)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l339_33997


namespace NUMINAMATH_CALUDE_average_weight_BCDE_l339_33985

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 97.25 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  w_A = 77 →
  (w_A + w_B + w_C) / 3 = 84 →
  (w_A + w_B + w_C + w_D) / 4 = 80 →
  w_E = w_D + 5 →
  (w_B + w_C + w_D + w_E) / 4 = 97.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_BCDE_l339_33985


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l339_33977

theorem remainder_sum_mod_seven : 
  (2 * (4561 + 4562 + 4563 + 4564 + 4565)) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l339_33977


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_210_l339_33947

/-- The number of ways to choose k elements from n elements without replacement and with order -/
def permutations (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The count of four-digit numbers with specific properties -/
def count_special_numbers : ℕ :=
  let digits := 10  -- 0 to 9
  let case1 := permutations 8 2 * permutations 2 2  -- for 0 and 8
  let case2 := permutations 7 1 * permutations 7 1 * permutations 2 2  -- for 1 and 9
  case1 + case2

theorem count_special_numbers_eq_210 :
  count_special_numbers = 210 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_210_l339_33947


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l339_33988

-- Define sets A and B
def A : Set ℝ := Set.univ
def B : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l339_33988


namespace NUMINAMATH_CALUDE_building_height_l339_33964

/-- Prove that given a flagpole of height 18 meters casting a shadow of 45 meters,
    and a building casting a shadow of 70 meters under similar conditions,
    the height of the building is 28 meters. -/
theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_shadow = 70)
  : (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l339_33964


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l339_33930

theorem sum_of_a_and_b (a b : ℚ) : 5 - Real.sqrt 3 * a = 2 * b + Real.sqrt 3 - a → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l339_33930


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l339_33982

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 27 ∧ x - y = 9 → x * y = 162 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l339_33982


namespace NUMINAMATH_CALUDE_impossible_cross_sections_l339_33974

-- Define a cube
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a plane
structure Plane where
  normal_vector : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define possible shapes of cross-sections
inductive CrossSectionShape
  | ObtuseTriangle
  | RightAngledTrapezoid
  | Rhombus
  | RegularPentagon
  | RegularHexagon

-- Function to determine if a shape is possible
def is_possible_cross_section (cube : Cube) (plane : Plane) (shape : CrossSectionShape) : Prop :=
  match shape with
  | CrossSectionShape.ObtuseTriangle => False
  | CrossSectionShape.RightAngledTrapezoid => False
  | CrossSectionShape.Rhombus => True
  | CrossSectionShape.RegularPentagon => False
  | CrossSectionShape.RegularHexagon => True

-- Theorem statement
theorem impossible_cross_sections (cube : Cube) (plane : Plane) :
  ¬(is_possible_cross_section cube plane CrossSectionShape.ObtuseTriangle) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RightAngledTrapezoid) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RegularPentagon) :=
sorry

end NUMINAMATH_CALUDE_impossible_cross_sections_l339_33974


namespace NUMINAMATH_CALUDE_number_problem_l339_33978

theorem number_problem (x : ℝ) : 0.7 * x - 40 = 30 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l339_33978


namespace NUMINAMATH_CALUDE_bill_with_late_charges_l339_33944

/-- The final bill amount after two late charges -/
def final_bill_amount (original_bill : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem stating the final bill amount after specific late charges -/
theorem bill_with_late_charges :
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_bill_with_late_charges_l339_33944


namespace NUMINAMATH_CALUDE_binomial_variance_example_l339_33932

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(10, 0.4) is 2.4 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨10, 0.4, by norm_num⟩
  variance X = 2.4 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l339_33932


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_five_l339_33922

theorem sum_of_solutions_is_five : 
  ∃! (s : ℝ), ∀ (x : ℝ), (x + 25 / x = 10) → (s = x) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_five_l339_33922


namespace NUMINAMATH_CALUDE_max_digit_sum_for_valid_number_l339_33973

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 2000 ∧ n < 3000 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum_for_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
    digit_sum n = 26 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_valid_number_l339_33973


namespace NUMINAMATH_CALUDE_sales_tax_difference_l339_33968

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) 
  (h1 : price = 30)
  (h2 : high_rate = 0.075)
  (h3 : low_rate = 0.07) : 
  price * high_rate - price * low_rate = 0.15 := by
  sorry

#check sales_tax_difference

end NUMINAMATH_CALUDE_sales_tax_difference_l339_33968


namespace NUMINAMATH_CALUDE_area_equality_l339_33991

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A : Point) (B : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def areaTriangle (A : Point) (B : Point) (C : Point) : ℝ := sorry

/-- Main theorem -/
theorem area_equality 
  (C D E F G H J : Point)
  (CDEF : Quadrilateral)
  (h1 : isParallelogram CDEF)
  (h2 : areaQuadrilateral CDEF = 36)
  (h3 : isMidpoint G C D)
  (h4 : isMidpoint H E F) :
  areaTriangle C D J = areaQuadrilateral CDEF :=
sorry

end NUMINAMATH_CALUDE_area_equality_l339_33991


namespace NUMINAMATH_CALUDE_slope_range_l339_33907

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l passing through (0,2) with slope k
def line (x y k : ℝ) : Prop := y = k * x + 2

-- Define the condition for intersection points
def intersects (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k

-- Define the acute angle condition
def acute_angle (k : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂ : ℝ, 
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k → 
  x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem slope_range : 
  ∀ k : ℝ, intersects k ∧ acute_angle k ↔ 
  (k > Real.sqrt 3 / 2 ∧ k < 2) ∨ (k < -Real.sqrt 3 / 2 ∧ k > -2) := by
  sorry

end NUMINAMATH_CALUDE_slope_range_l339_33907


namespace NUMINAMATH_CALUDE_casper_candies_l339_33954

/-- The number of candies Casper originally had -/
def original_candies : ℕ := 176

/-- The number of candies Casper gave to his brother on the first day -/
def candies_to_brother : ℕ := 3

/-- The number of candies Casper gave to his sister on the second day -/
def candies_to_sister : ℕ := 5

/-- The number of candies Casper ate on the third day -/
def final_candies : ℕ := 10

theorem casper_candies :
  let remaining_day1 := original_candies * 3 / 4 - candies_to_brother
  let remaining_day2 := remaining_day1 / 2 - candies_to_sister
  remaining_day2 = final_candies := by sorry

end NUMINAMATH_CALUDE_casper_candies_l339_33954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l339_33986

/-- An increasing arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧  -- Arithmetic sequence
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (a 1 = 1) ∧  -- a_1 = 1
  (a 3 = (a 2)^2 - 4)  -- a_3 = a_2^2 - 4

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) : 
    ∀ n : ℕ, a n = 3 * n - 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l339_33986


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l339_33983

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {-1, 0, 1, 2}

theorem complement_intersection_problem : (I \ A) ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l339_33983


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l339_33940

theorem cubic_expansion_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l339_33940


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l339_33913

theorem simplify_and_ratio (k : ℝ) : ∃ (a b : ℝ), 
  (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a / b = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l339_33913


namespace NUMINAMATH_CALUDE_gracie_number_l339_33921

/-- Represents the counting pattern for a student --/
def student_count (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5*k + 1 ∨ m = 5*k + 2 ∨ m = 5*k + 4 ∨ m = 5*k + 5}

/-- Represents the numbers skipped by a student --/
def student_skip (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5^n * (5*k - 2)}

/-- The set of numbers said by the first n students --/
def numbers_said (n : ℕ) : Set ℕ :=
  if n = 0 then ∅ else (student_count n) ∪ (numbers_said (n-1)) \ (student_skip n)

theorem gracie_number :
  ∃! x, x ∈ {m | 1 ≤ m ∧ m ≤ 2000} \ (numbers_said 7) ∧ x = 1623 :=
sorry

end NUMINAMATH_CALUDE_gracie_number_l339_33921


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l339_33952

/-- The system of equations --/
def system (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 - 2*x - 2*y + 10 = 0 ∧
  x^3 * y - x * y^3 - 2*x^2 + 2*y^2 - 30 = 0

/-- The solution to the system of equations --/
def solution : ℝ × ℝ := (-4, -1)

/-- Theorem stating that the solution satisfies the system of equations --/
theorem solution_satisfies_system :
  let (x, y) := solution
  system x y := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l339_33952


namespace NUMINAMATH_CALUDE_dice_probability_l339_33942

def num_dice : ℕ := 15
def num_ones : ℕ := 3
def prob_one : ℚ := 1/6
def prob_not_one : ℚ := 5/6

theorem dice_probability : 
  (Nat.choose num_dice num_ones : ℚ) * prob_one ^ num_ones * prob_not_one ^ (num_dice - num_ones) = 
  455 * (1/6)^3 * (5/6)^12 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l339_33942


namespace NUMINAMATH_CALUDE_cookie_distribution_l339_33979

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) :
  people = 4 →
  cookies_per_person = 22 →
  total_cookies = people * cookies_per_person →
  total_cookies = 88 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l339_33979


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l339_33972

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 6/11 ∧ ∀ (a b c : ℝ), a + b + c = 1 → 2*a^2 + b^2 + 3*c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l339_33972


namespace NUMINAMATH_CALUDE_train_time_calculation_l339_33938

/-- Proves that the additional time for train-related activities is 15.5 minutes --/
theorem train_time_calculation (distance : ℝ) (walk_speed : ℝ) (train_speed : ℝ) 
  (walk_time_difference : ℝ) :
  distance = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  walk_time_difference = 10 →
  ∃ (x : ℝ), x = 15.5 ∧ 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + walk_time_difference :=
by
  sorry

end NUMINAMATH_CALUDE_train_time_calculation_l339_33938


namespace NUMINAMATH_CALUDE_virus_length_scientific_notation_l339_33951

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_length_scientific_notation :
  toScientificNotation 0.00000032 = ScientificNotation.mk 3.2 (-7) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_virus_length_scientific_notation_l339_33951


namespace NUMINAMATH_CALUDE_herbert_age_next_year_l339_33965

theorem herbert_age_next_year (kris_age : ℕ) (age_difference : ℕ) :
  kris_age = 24 →
  age_difference = 10 →
  kris_age - age_difference + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_herbert_age_next_year_l339_33965


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l339_33984

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l339_33984


namespace NUMINAMATH_CALUDE_polynomial_zeros_product_l339_33900

theorem polynomial_zeros_product (z₁ z₂ : ℂ) : 
  z₁^2 + 6*z₁ + 11 = 0 → 
  z₂^2 + 6*z₂ + 11 = 0 → 
  (1 + z₁^2*z₂)*(1 + z₁*z₂^2) = 1266 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zeros_product_l339_33900


namespace NUMINAMATH_CALUDE_fraction_simplification_l339_33903

theorem fraction_simplification (a b c : ℝ) 
  (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l339_33903


namespace NUMINAMATH_CALUDE_playground_count_l339_33976

theorem playground_count (x : ℤ) : 
  let known_numbers : List ℤ := [12, 1, 12, 7, 3, 8]
  let all_numbers : List ℤ := x :: known_numbers
  (all_numbers.sum / all_numbers.length : ℚ) = 7 → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_playground_count_l339_33976


namespace NUMINAMATH_CALUDE_average_age_of_six_students_l339_33934

theorem average_age_of_six_students 
  (total_students : Nat) 
  (group_of_eight : Nat) 
  (group_of_six : Nat) 
  (fifteenth_student : Nat) 
  (total_average : ℚ) 
  (eight_average : ℚ) 
  (fifteenth_age : Nat) :
  total_students = 15 →
  group_of_eight = 8 →
  group_of_six = 6 →
  fifteenth_student = 1 →
  total_average = 15 →
  eight_average = 14 →
  fifteenth_age = 17 →
  (total_students * total_average - group_of_eight * eight_average - fifteenth_age) / group_of_six = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_six_students_l339_33934


namespace NUMINAMATH_CALUDE_lcm_prime_sum_l339_33945

theorem lcm_prime_sum (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (hlcm : Nat.lcm x (Nat.lcm y z) = 210) (hord : x > y ∧ y > z) : 2 * x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_lcm_prime_sum_l339_33945


namespace NUMINAMATH_CALUDE_trivia_team_total_score_l339_33919

def trivia_team_score (total_members : ℕ) (absent_members : ℕ) (scores : List ℕ) : Prop :=
  total_members = 7 ∧
  absent_members = 2 ∧
  scores = [5, 9, 7, 5, 3] ∧
  scores.length = total_members - absent_members ∧
  scores.sum = 29

theorem trivia_team_total_score :
  ∃ (total_members absent_members : ℕ) (scores : List ℕ),
    trivia_team_score total_members absent_members scores :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_total_score_l339_33919


namespace NUMINAMATH_CALUDE_diamond_is_conditional_l339_33996

/-- Represents shapes in a flowchart --/
inductive FlowchartShape
  | Diamond
  | Rectangle
  | Oval

/-- Represents logical structures in an algorithm --/
inductive LogicalStructure
  | Conditional
  | Loop
  | Sequential

/-- A function that maps flowchart shapes to logical structures --/
def shapeToStructure : FlowchartShape → LogicalStructure
  | FlowchartShape.Diamond => LogicalStructure.Conditional
  | FlowchartShape.Rectangle => LogicalStructure.Sequential
  | FlowchartShape.Oval => LogicalStructure.Sequential

/-- Theorem stating that a diamond shape in a flowchart represents a conditional structure --/
theorem diamond_is_conditional :
  shapeToStructure FlowchartShape.Diamond = LogicalStructure.Conditional :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_is_conditional_l339_33996


namespace NUMINAMATH_CALUDE_handball_tournament_impossibility_l339_33926

structure Tournament :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

def total_games (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

def total_points (t : Tournament) : ℕ :=
  total_games t * (t.points_for_win + t.points_for_loss)

theorem handball_tournament_impossibility 
  (t : Tournament)
  (h1 : t.teams = 14)
  (h2 : t.points_for_win = 2)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0)
  (h5 : ∀ (i j : ℕ), i ≠ j → i < t.teams → j < t.teams → 
       ∃ (pi pj : ℕ), pi ≠ pj ∧ pi ≤ total_points t ∧ pj ≤ total_points t) :
  ¬(∃ (top bottom : Finset ℕ), 
    top.card = 3 ∧ 
    bottom.card = 3 ∧ 
    (∀ i ∈ top, ∀ j ∈ bottom, 
      ∃ (pi pj : ℕ), pi > pj ∧ 
      pi ≤ total_points t ∧ 
      pj ≤ total_points t)) :=
sorry

end NUMINAMATH_CALUDE_handball_tournament_impossibility_l339_33926


namespace NUMINAMATH_CALUDE_p_6_l339_33918

/-- A monic quartic polynomial with specific values at x = 1, 2, 3, 4 -/
def p (x : ℝ) : ℝ :=
  x^4 + a*x^3 + b*x^2 + c*x + d
  where
    a : ℝ := sorry
    b : ℝ := sorry
    c : ℝ := sorry
    d : ℝ := sorry

/-- The polynomial satisfies the given conditions -/
axiom p_1 : p 1 = 3
axiom p_2 : p 2 = 7
axiom p_3 : p 3 = 13
axiom p_4 : p 4 = 21

/-- The theorem to be proved -/
theorem p_6 : p 6 = 158 := by sorry

end NUMINAMATH_CALUDE_p_6_l339_33918


namespace NUMINAMATH_CALUDE_josh_doug_money_ratio_l339_33931

/-- Proves that the ratio of Josh's money to Doug's money is 3:4 given the problem conditions -/
theorem josh_doug_money_ratio :
  ∀ (josh doug brad : ℕ),
  josh + doug + brad = 68 →
  josh = 2 * brad →
  doug = 32 →
  (josh : ℚ) / doug = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_josh_doug_money_ratio_l339_33931


namespace NUMINAMATH_CALUDE_triangle_theorem_l339_33904

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  (b^2 + c^2 = a * (Real.sqrt 3 / 3 * b * c + a)) →
  (∃ (S : ℝ), (a = 2 * Real.sqrt 3 * Real.cos A) ∧
              (A = π / 3 ∧ B = π / 6 → S = Real.sqrt 3 / 2) ∧
              S = (1 / 2) * a * b * Real.sin C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l339_33904


namespace NUMINAMATH_CALUDE_square_field_area_l339_33949

theorem square_field_area (wire_length : ℝ) (wire_rounds : ℕ) (field_area : ℝ) : 
  wire_length = 7348 →
  wire_rounds = 11 →
  wire_length = 4 * wire_rounds * Real.sqrt field_area →
  field_area = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l339_33949


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l339_33916

-- Define the circular sector
def circular_sector (R : ℝ) (θ : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 ≤ R^2 ∧ 0 ≤ x ∧ y ≤ x * Real.tan θ}

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (R : ℝ) (θ : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - (R - r))^2 + (y - r)^2 = r^2}

-- Theorem statement
theorem inscribed_circle_radius :
  ∀ (R : ℝ), R > 0 →
  ∃ (r : ℝ), r > 0 ∧
  inscribed_circle r R (π/6) ⊆ circular_sector R (π/6) ∧
  r = 2 := by
sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l339_33916


namespace NUMINAMATH_CALUDE_population_approximation_l339_33924

def initial_population : ℝ := 14999.999999999998
def first_year_change : ℝ := 0.12
def second_year_change : ℝ := 0.12

def population_after_two_years : ℝ :=
  initial_population * (1 + first_year_change) * (1 - second_year_change)

theorem population_approximation :
  ∃ ε > 0, |population_after_two_years - 14784| < ε :=
sorry

end NUMINAMATH_CALUDE_population_approximation_l339_33924


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l339_33911

/-- Given two partners p and q, proves that if the ratio of their profits is 7:10,
    p invests for 7 months, and q invests for 14 months,
    then the ratio of their investments is 7:5. -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) :
  (p * 7) / (q * 14) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l339_33911


namespace NUMINAMATH_CALUDE_gcf_40_56_l339_33920

theorem gcf_40_56 : Nat.gcd 40 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_40_56_l339_33920


namespace NUMINAMATH_CALUDE_product_equality_l339_33967

theorem product_equality : 72519 * 31415.927 = 2277666538.233 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l339_33967


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l339_33969

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ m : ℤ, 5 * (n^2 + 2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l339_33969


namespace NUMINAMATH_CALUDE_journey_speed_fraction_l339_33925

/-- Proves that if a person travels part of a journey at 5 mph and the rest at 15 mph,
    with an average speed of 10 mph for the entire journey,
    then the fraction of time spent traveling at 15 mph is 1/2. -/
theorem journey_speed_fraction (t₅ t₁₅ : ℝ) (h₁ : t₅ > 0) (h₂ : t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 →
  t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_fraction_l339_33925


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l339_33912

theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * x + Real.pi / 2) - 1 = Real.sin (2 * (x + Real.pi / 4)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l339_33912


namespace NUMINAMATH_CALUDE_geometric_progression_identity_l339_33939

/-- If a, b, c form a geometric progression, then (a+b+c)(a-b+c) = a^2 + b^2 + c^2 -/
theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a*c) :
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_identity_l339_33939


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l339_33992

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sqrt 2 * Real.cos (2 * α) = Real.sin (α + π/4)) : 
  Real.sin (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l339_33992


namespace NUMINAMATH_CALUDE_f_difference_bound_l339_33950

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_difference_bound (a x : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * |a| + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_bound_l339_33950


namespace NUMINAMATH_CALUDE_calen_excess_pencils_l339_33955

/-- The number of pencils each person has -/
structure PencilCount where
  calen : ℕ
  caleb : ℕ
  candy : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCount) : Prop :=
  p.calen > p.caleb ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.candy = 9 ∧
  p.calen - 10 = 10

/-- The theorem to prove -/
theorem calen_excess_pencils (p : PencilCount) :
  pencil_problem p → p.calen - p.caleb = 5 := by
  sorry

end NUMINAMATH_CALUDE_calen_excess_pencils_l339_33955


namespace NUMINAMATH_CALUDE_f_minimum_and_range_l339_33914

/-- The function f(x) = |2x+1| + |2x-1| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

theorem f_minimum_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), |2*a + b| + |a| - 1/2 * |a + b| * f x ≥ 0) →
    x ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_and_range_l339_33914


namespace NUMINAMATH_CALUDE_square_decreasing_on_negative_l339_33935

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem square_decreasing_on_negative : 
  ∀ x y : ℝ, x < y → y < 0 → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_square_decreasing_on_negative_l339_33935


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_unique_l339_33975

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_unique
  (q : ℝ → ℝ)
  (h_monic : ∃ a b c : ℝ, q = MonicCubicPolynomial a b c)
  (h_root : q (2 - 3*I) = 0)
  (h_value : q 1 = 26) :
  q = MonicCubicPolynomial (-2.4) 6.6 20.8 := by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_unique_l339_33975


namespace NUMINAMATH_CALUDE_rachel_winter_clothing_boxes_l339_33941

theorem rachel_winter_clothing_boxes : 
  let scarves_per_box : ℕ := 3
  let mittens_per_box : ℕ := 4
  let total_pieces : ℕ := 49
  let pieces_per_box : ℕ := scarves_per_box + mittens_per_box
  let num_boxes : ℕ := total_pieces / pieces_per_box
  num_boxes = 7 := by
sorry

end NUMINAMATH_CALUDE_rachel_winter_clothing_boxes_l339_33941


namespace NUMINAMATH_CALUDE_smallest_M_for_Q_less_than_three_fourths_l339_33995

def is_multiple_of_six (M : ℕ) : Prop := ∃ k : ℕ, M = 6 * k

def Q (M : ℕ) : ℚ := (⌈(2 / 3 : ℚ) * M + 1⌉ : ℚ) / (M + 1 : ℚ)

theorem smallest_M_for_Q_less_than_three_fourths :
  ∀ M : ℕ, is_multiple_of_six M → (Q M < 3 / 4 → M ≥ 6) ∧ (Q 6 < 3 / 4) := by sorry

end NUMINAMATH_CALUDE_smallest_M_for_Q_less_than_three_fourths_l339_33995


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l339_33937

theorem smallest_solution_quadratic (y : ℝ) :
  (6 * y^2 - 41 * y + 55 = 0) → (y ≥ 2.5) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l339_33937


namespace NUMINAMATH_CALUDE_expression_value_l339_33999

theorem expression_value : 
  let a : ℤ := 2025
  let b : ℤ := a + 1
  let k : ℤ := 1
  (a^3 - 2*k*a^2*b + 3*k*a*b^2 - b^3 + k) / (a*b) = 2025 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l339_33999


namespace NUMINAMATH_CALUDE_monthly_rate_is_42_l339_33961

/-- The monthly parking rate that satisfies the given conditions -/
def monthly_rate : ℚ :=
  let weekly_rate : ℚ := 10
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let yearly_savings : ℚ := 16
  (weekly_rate * weeks_per_year - yearly_savings) / months_per_year

/-- Proof that the monthly parking rate is $42 -/
theorem monthly_rate_is_42 : monthly_rate = 42 := by
  sorry

end NUMINAMATH_CALUDE_monthly_rate_is_42_l339_33961


namespace NUMINAMATH_CALUDE_system_of_equations_l339_33966

theorem system_of_equations (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  2*x + 2*y + 2*z = 7/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l339_33966


namespace NUMINAMATH_CALUDE_twenty_loaves_slices_thirty_loaves_not_enough_l339_33933

-- Define the number of slices per loaf
def slices_per_loaf : ℕ := 12

-- Define the function to calculate total slices
def total_slices (loaves : ℕ) : ℕ := slices_per_loaf * loaves

-- Theorem 1
theorem twenty_loaves_slices : total_slices 20 = 240 := by sorry

-- Theorem 2
theorem thirty_loaves_not_enough (children : ℕ) (h : children = 385) : 
  total_slices 30 < children := by sorry

end NUMINAMATH_CALUDE_twenty_loaves_slices_thirty_loaves_not_enough_l339_33933


namespace NUMINAMATH_CALUDE_symmetric_complex_number_l339_33963

theorem symmetric_complex_number : ∀ z : ℂ, 
  (z.re = (-1 : ℝ) ∧ z.im = (1 : ℝ)) ↔ 
  (z.re = (2 / (Complex.I - 1)).re ∧ z.im = -(2 / (Complex.I - 1)).im) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_complex_number_l339_33963


namespace NUMINAMATH_CALUDE_juan_friends_seating_l339_33901

theorem juan_friends_seating (n : ℕ) : n = 5 :=
  by
    -- Define the conditions
    have juan_fixed : True := True.intro
    have jamal_next_to_juan : ℕ := 2
    have total_arrangements : ℕ := 48

    -- State the relationship between n and the conditions
    have seating_equation : jamal_next_to_juan * Nat.factorial (n - 1) = total_arrangements := by sorry

    -- Prove that n = 5 satisfies the equation
    sorry

end NUMINAMATH_CALUDE_juan_friends_seating_l339_33901


namespace NUMINAMATH_CALUDE_bounce_count_correct_l339_33906

/-- The smallest positive integer k such that 800 * (0.4^k) < 5 -/
def bounce_count : ℕ := 6

/-- The initial height of the ball in feet -/
def initial_height : ℝ := 800

/-- The ratio of the height after each bounce to the previous height -/
def bounce_ratio : ℝ := 0.4

/-- The target height in feet -/
def target_height : ℝ := 5

theorem bounce_count_correct : 
  (∀ k : ℕ, k < bounce_count → initial_height * (bounce_ratio ^ k) ≥ target_height) ∧
  initial_height * (bounce_ratio ^ bounce_count) < target_height :=
sorry

end NUMINAMATH_CALUDE_bounce_count_correct_l339_33906


namespace NUMINAMATH_CALUDE_parabola_coef_sum_l339_33970

/-- A parabola with equation x = ay^2 + by + c passing through points (6, -3) and (4, -1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : 6 = a * (-3)^2 + b * (-3) + c
  point2 : 4 = a * (-1)^2 + b * (-1) + c

/-- The sum of coefficients a, b, and c of the parabola equals -2 -/
theorem parabola_coef_sum (p : Parabola) : p.a + p.b + p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coef_sum_l339_33970


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l339_33990

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 = 2 + 6*y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l339_33990


namespace NUMINAMATH_CALUDE_max_sum_of_three_naturals_l339_33959

theorem max_sum_of_three_naturals (a b c : ℕ) (h1 : a + b = 1014) (h2 : c - b = 497) (h3 : a > b) :
  (∀ a' b' c' : ℕ, a' + b' = 1014 → c' - b' = 497 → a' > b' → a' + b' + c' ≤ a + b + c) ∧
  a + b + c = 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_naturals_l339_33959


namespace NUMINAMATH_CALUDE_inequality_proof_l339_33936

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 14) : 
  a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l339_33936


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l339_33923

def is_valid_digit_set (a b c : ℕ) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def sum_of_numbers (a b c : ℕ) : ℕ :=
  100 * (a + b + c) + 10 * (a + b + c) + (a + b + c)

theorem three_digit_sum_theorem :
  ∀ a b c : ℕ,
    is_valid_digit_set a b c →
    sum_of_numbers a b c = 1221 →
    ((a = 1 ∧ b = 1 ∧ c = 9) ∨
     (a = 2 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 3 ∧ c = 5) ∨
     (a = 4 ∧ b = 4 ∧ c = 3) ∨
     (a = 5 ∧ b = 5 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l339_33923


namespace NUMINAMATH_CALUDE_second_quadrant_m_range_l339_33994

theorem second_quadrant_m_range (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_m_range_l339_33994


namespace NUMINAMATH_CALUDE_income_calculation_l339_33946

/-- Proves that given an income to expenditure ratio of 7:6 and savings of 3000,
    the income is 21000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 6 = expenditure * 7 →
  income - expenditure = savings →
  savings = 3000 →
  income = 21000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l339_33946


namespace NUMINAMATH_CALUDE_bank_deposit_theorem_l339_33908

/-- Calculates the actual amount of principal and interest after one year,
    given an initial deposit, annual interest rate, and interest tax rate. -/
def actual_amount (initial_deposit : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  initial_deposit + (1 - tax_rate) * interest_rate * initial_deposit

theorem bank_deposit_theorem (x : ℝ) :
  actual_amount x 0.0225 0.2 = (0.8 * 0.0225 * x + x) := by sorry

end NUMINAMATH_CALUDE_bank_deposit_theorem_l339_33908


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l339_33989

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = Real.sqrt (19 / 8) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l339_33989


namespace NUMINAMATH_CALUDE_math_books_in_same_box_l339_33981

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 5, 7]

def probability_all_math_in_same_box : ℚ := 25 / 242

theorem math_books_in_same_box :
  let total_arrangements := (total_textbooks.choose box_capacities[0]!) *
    ((total_textbooks - box_capacities[0]!).choose box_capacities[1]!) *
    ((total_textbooks - box_capacities[0]! - box_capacities[1]!).choose box_capacities[2]!)
  let favorable_outcomes := 
    (total_textbooks - math_textbooks).choose box_capacities[0]! +
    ((total_textbooks - math_textbooks).choose (box_capacities[1]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[1]!).choose box_capacities[0]!) +
    ((total_textbooks - math_textbooks).choose (box_capacities[2]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[2]!).choose box_capacities[0]!)
  probability_all_math_in_same_box = favorable_outcomes / total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_math_books_in_same_box_l339_33981


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l339_33956

theorem polynomial_value_constraint (a b c : ℤ) : 
  (b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) → 
  (b + c + a ≠ 2009) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l339_33956


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l339_33958

theorem imaginary_part_of_one_minus_i :
  Complex.im (1 - Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l339_33958


namespace NUMINAMATH_CALUDE_chess_tournament_games_l339_33928

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 18 players, where each player plays twice with every other player, the total number of games played is 612 -/
theorem chess_tournament_games :
  tournament_games 18 * 2 = 612 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l339_33928


namespace NUMINAMATH_CALUDE_salad_cost_l339_33909

/-- The cost of the salad given breakfast and lunch costs -/
theorem salad_cost (muffin_cost coffee_cost soup_cost lemonade_cost : ℝ)
  (h1 : muffin_cost = 2)
  (h2 : coffee_cost = 4)
  (h3 : soup_cost = 3)
  (h4 : lemonade_cost = 0.75)
  (h5 : muffin_cost + coffee_cost + 3 = soup_cost + lemonade_cost + (muffin_cost + coffee_cost)) :
  soup_cost + lemonade_cost + 3 - (soup_cost + lemonade_cost) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_salad_cost_l339_33909


namespace NUMINAMATH_CALUDE_sphere_area_ratio_l339_33987

/-- The area of a region on a sphere is proportional to the square of its radius -/
axiom area_proportional_to_radius_squared {r₁ r₂ A₁ A₂ : ℝ} (h : r₁ > 0 ∧ r₂ > 0) :
  A₂ / A₁ = (r₂ / r₁) ^ 2

/-- Given two concentric spheres with radii 4 cm and 6 cm, if a region on the smaller sphere
    has an area of 37 square cm, then the corresponding region on the larger sphere
    has an area of 83.25 square cm -/
theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 6) (hA₁ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 ∧ A₂ / A₁ = (r₂ / r₁) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_ratio_l339_33987


namespace NUMINAMATH_CALUDE_valid_scheduling_orders_l339_33960

def number_of_lecturers : ℕ := 7

def number_of_dependencies : ℕ := 2

theorem valid_scheduling_orders :
  (number_of_lecturers.factorial / 2^number_of_dependencies : ℕ) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_valid_scheduling_orders_l339_33960


namespace NUMINAMATH_CALUDE_unique_solution_equation_l339_33929

theorem unique_solution_equation : 
  ∃! (x y z : ℕ+), 1 + 2^(x.val) + 3^(y.val) = z.val^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l339_33929


namespace NUMINAMATH_CALUDE_wilted_flowers_calculation_l339_33910

/-- 
Given:
- initial_flowers: The initial number of flowers picked
- flowers_per_bouquet: The number of flowers in each bouquet
- bouquets_made: The number of bouquets that could be made after some flowers wilted

Prove:
The number of wilted flowers is equal to the initial number of flowers minus
the product of the number of bouquets made and the number of flowers per bouquet.
-/
theorem wilted_flowers_calculation (initial_flowers flowers_per_bouquet bouquets_made : ℕ) :
  initial_flowers - (bouquets_made * flowers_per_bouquet) = 
  initial_flowers - bouquets_made * flowers_per_bouquet :=
by sorry

end NUMINAMATH_CALUDE_wilted_flowers_calculation_l339_33910


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l339_33943

/-- Given a hyperbola with equation x^2 - 4y^2 = -1, its asymptotes are x ± 2y = 0 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 - 4*y^2 = -1 →
  ∃ (k : ℝ), (x + 2*y = 0 ∧ x - 2*y = 0) ∨ (2*y + x = 0 ∧ 2*y - x = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l339_33943


namespace NUMINAMATH_CALUDE_existence_of_coefficients_l339_33915

/-- Two polynomials with coefficients A, B, C, D -/
def poly1 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 + 4*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 1
def poly2 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 - 4*x^5 + A*x^4 - B*x^3 + C*x^2 - D*x + 1

/-- The product of the two polynomials -/
def product (A B C D : ℝ) (x : ℝ) : ℝ := (poly1 A B C D x) * (poly2 A B C D x)

/-- Theorem stating the existence of coefficients satisfying the conditions -/
theorem existence_of_coefficients : ∃ (A B C D : ℝ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) ∧ 
  (∃ (k : ℝ), ∀ (x : ℝ), product A B C D x = x^12 + k*x^6 + 1) ∧
  (∃ (b c : ℝ), ∀ (x : ℝ), 
    poly1 A B C D x = (x^3 + 2*x^2 + b*x + c)^2 ∧
    poly2 A B C D x = (x^3 - 2*x^2 + b*x - c)^2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_coefficients_l339_33915


namespace NUMINAMATH_CALUDE_complex_division_l339_33962

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I → z₂ = 1 - I → z₁ / z₂ = I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l339_33962


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_to_circle_l339_33905

/-- The value of m for which the asymptotes of the hyperbola y² - x²/m² = 1 
    are tangent to the circle x² + y² - 4y + 3 = 0, given m > 0 -/
theorem hyperbola_asymptote_tangent_to_circle (m : ℝ) 
  (hm : m > 0)
  (h_hyperbola : ∀ x y : ℝ, y^2 - x^2/m^2 = 1 → 
    (∃ k : ℝ, y = k*x/m ∨ y = -k*x/m))
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → 
    (x - 0)^2 + (y - 2)^2 = 1)
  (h_tangent : ∀ x y : ℝ, (y = x/m ∨ y = -x/m) → 
    ((0 - x)^2 + (2 - y)^2 = 1)) :
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_to_circle_l339_33905


namespace NUMINAMATH_CALUDE_avg_problem_l339_33948

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Theorem: The average of [2 4], [6 2], and [3 3] is 10/3 -/
theorem avg_problem : avg3 (avg2 2 4) (avg2 6 2) (avg2 3 3) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_avg_problem_l339_33948


namespace NUMINAMATH_CALUDE_dividend_calculation_l339_33902

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 3 → 
  dividend = divisor * quotient + remainder →
  dividend = 165 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l339_33902


namespace NUMINAMATH_CALUDE_smallest_perimeter_after_folding_l339_33971

theorem smallest_perimeter_after_folding (l w : ℝ) (hl : l = 17 / 2) (hw : w = 11) : 
  let original_perimeter := 2 * l + 2 * w
  let folded_perimeter1 := 2 * l + 2 * (w / 4)
  let folded_perimeter2 := 2 * (l / 2) + 2 * (w / 2)
  min folded_perimeter1 folded_perimeter2 = 39 / 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_after_folding_l339_33971


namespace NUMINAMATH_CALUDE_grade12_sample_size_l339_33993

/-- Represents the number of grade 12 students in a stratified sample -/
def grade12InSample (totalStudents gradeStudents sampleSize : ℕ) : ℚ :=
  (sampleSize : ℚ) * (gradeStudents : ℚ) / (totalStudents : ℚ)

/-- Theorem: The number of grade 12 students in the sample is 140 -/
theorem grade12_sample_size :
  grade12InSample 2000 700 400 = 140 := by sorry

end NUMINAMATH_CALUDE_grade12_sample_size_l339_33993
