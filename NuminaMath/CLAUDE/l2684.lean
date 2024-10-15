import Mathlib

namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2684_268414

/-- A configuration of tangents to a circle -/
structure TangentConfiguration where
  r : ℝ  -- radius of the circle
  AB : ℝ  -- length of tangent AB
  CD : ℝ  -- length of tangent CD
  EF : ℝ  -- length of EF

/-- The theorem stating the radius of the circle given the tangent configuration -/
theorem tangent_circle_radius (config : TangentConfiguration) 
  (h1 : config.AB = 12)
  (h2 : config.CD = 20)
  (h3 : config.EF = 8) :
  config.r = 6 := by
  sorry

#check tangent_circle_radius

end NUMINAMATH_CALUDE_tangent_circle_radius_l2684_268414


namespace NUMINAMATH_CALUDE_units_digit_sum_cubes_l2684_268497

theorem units_digit_sum_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_cubes_l2684_268497


namespace NUMINAMATH_CALUDE_normal_vector_of_l_l2684_268420

/-- Definition of the line l: 2x - 3y + 4 = 0 -/
def l (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

/-- Definition of a normal vector to a line -/
def is_normal_vector (v : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = (k * 2, k * (-3))

/-- Theorem: (4, -6) is a normal vector to the line l -/
theorem normal_vector_of_l : is_normal_vector (4, -6) l := by
  sorry

end NUMINAMATH_CALUDE_normal_vector_of_l_l2684_268420


namespace NUMINAMATH_CALUDE_balance_sheet_equation_l2684_268429

/-- Given the equation 4m - t = 8000 where m = 4 and t = 4 + 100i, prove that t = -7988 - 100i. -/
theorem balance_sheet_equation (m t : ℂ) (h1 : 4 * m - t = 8000) (h2 : m = 4) (h3 : t = 4 + 100 * Complex.I) : 
  t = -7988 - 100 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_balance_sheet_equation_l2684_268429


namespace NUMINAMATH_CALUDE_range_of_sum_l2684_268434

theorem range_of_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2684_268434


namespace NUMINAMATH_CALUDE_exponential_simplification_l2684_268495

theorem exponential_simplification : 3 * ((-5)^2)^(3/4) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_exponential_simplification_l2684_268495


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l2684_268432

theorem evaluate_complex_expression :
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (5 - 2 * Real.sqrt 6)
  M = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l2684_268432


namespace NUMINAMATH_CALUDE_sequence_properties_l2684_268462

def a (n : ℕ) : ℤ := n^2 - 7*n + 6

theorem sequence_properties :
  (a 4 = -6) ∧
  (a 16 = 150) ∧
  (∀ n : ℕ, n ≥ 7 → a n > 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2684_268462


namespace NUMINAMATH_CALUDE_symmetric_points_product_l2684_268400

/-- Given two points A and B symmetric about the origin, prove their coordinates' product -/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l2684_268400


namespace NUMINAMATH_CALUDE_jerry_butterflies_left_l2684_268411

/-- Given that Jerry had 93 butterflies initially and released 11 butterflies,
    prove that he now has 82 butterflies left. -/
theorem jerry_butterflies_left (initial : ℕ) (released : ℕ) (left : ℕ) : 
  initial = 93 → released = 11 → left = initial - released → left = 82 := by
  sorry

end NUMINAMATH_CALUDE_jerry_butterflies_left_l2684_268411


namespace NUMINAMATH_CALUDE_smallest_number_with_two_thirds_prob_l2684_268441

/-- The smallest number that can be drawn in the lottery -/
def minNumber : ℕ := 1

/-- The largest number in the first range of the lottery -/
def rangeEnd : ℕ := 15

/-- The probability of drawing a number between minNumber and rangeEnd, inclusive -/
def probFirstRange : ℚ := 1/3

/-- The probability of drawing a number less than or equal to rangeEnd -/
def probUpToRangeEnd : ℚ := 2/3

/-- The probability of drawing a number larger than the target number -/
def probLargerThanTarget : ℚ := 2/3

theorem smallest_number_with_two_thirds_prob :
  ∃ N : ℕ, N = rangeEnd + 1 ∧
    (∀ k : ℕ, k > N → (probFirstRange + (k - rangeEnd : ℚ) * probFirstRange = probLargerThanTarget)) ∧
    (∀ m : ℕ, m < N → (probFirstRange + (m - rangeEnd : ℚ) * probFirstRange < probLargerThanTarget)) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_two_thirds_prob_l2684_268441


namespace NUMINAMATH_CALUDE_minimum_triangle_area_l2684_268402

/-- Given a line passing through (2, 1) and intersecting the positive x and y axes at points A and B
    respectively, with O as the origin, the minimum area of triangle AOB is 4. -/
theorem minimum_triangle_area (k : ℝ) (h : k < 0) :
  let xA := 2 - 1 / k
  let yB := 1 - 2 * k
  let area := (1 / 2) * xA * yB
  4 ≤ area :=
by sorry

end NUMINAMATH_CALUDE_minimum_triangle_area_l2684_268402


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2684_268453

/-- Given a triangle with sides satisfying specific conditions, prove its perimeter. -/
theorem triangle_perimeter (a b : ℝ) : 
  let side1 := a + b
  let side2 := side1 + (a + 2)
  let side3 := side2 - 3
  side1 + side2 + side3 = 5*a + 3*b + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2684_268453


namespace NUMINAMATH_CALUDE_find_A_value_l2684_268422

theorem find_A_value (A : Nat) : A < 10 → (691 - (A * 100 + 87) = 4) → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l2684_268422


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l2684_268470

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l2684_268470


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2684_268458

/-- Proves that for a co-ed softball team with 6 more women than men and 24 total players, 
    the ratio of men to women is 3:5 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2684_268458


namespace NUMINAMATH_CALUDE_cot_15_plus_tan_45_l2684_268468

theorem cot_15_plus_tan_45 : Real.cos (15 * π / 180) / Real.sin (15 * π / 180) + Real.tan (45 * π / 180) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_15_plus_tan_45_l2684_268468


namespace NUMINAMATH_CALUDE_min_value_expression_l2684_268423

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (a + 1/b - 1009) + (b + 1/a) * (b + 1/a - 1009) ≥ -509004.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2684_268423


namespace NUMINAMATH_CALUDE_profit_growth_rate_l2684_268483

theorem profit_growth_rate (initial_profit target_profit : ℝ) (growth_rate : ℝ) (months : ℕ) :
  initial_profit * (1 + growth_rate / 100) ^ months = target_profit →
  growth_rate = 25 :=
by
  intro h
  -- Proof goes here
  sorry

#check profit_growth_rate 1.6 2.5 25 2

end NUMINAMATH_CALUDE_profit_growth_rate_l2684_268483


namespace NUMINAMATH_CALUDE_infinite_set_equal_digit_sum_l2684_268404

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains zero in its decimal notation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_set_equal_digit_sum (k : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ t ∈ S, ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end NUMINAMATH_CALUDE_infinite_set_equal_digit_sum_l2684_268404


namespace NUMINAMATH_CALUDE_quarter_angle_tangent_line_through_point_line_with_y_intercept_l2684_268454

-- Define the original line
def original_line (x y : ℝ) : Prop := y = -Real.sqrt 3 * x + 1

-- Define the angle that is one fourth of the slope angle
def quarter_angle : ℝ := 30

-- Theorem 1: The tangent of the quarter angle is √3/3
theorem quarter_angle_tangent :
  Real.tan (quarter_angle * π / 180) = Real.sqrt 3 / 3 := by sorry

-- Theorem 2: Equation of the line passing through (√3, -1)
theorem line_through_point (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 6 = 0) ↔
  (y + 1 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3)) := by sorry

-- Theorem 3: Equation of the line with y-intercept -5
theorem line_with_y_intercept (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 15 = 0) ↔
  (y = (Real.sqrt 3 / 3) * x - 5) := by sorry

end NUMINAMATH_CALUDE_quarter_angle_tangent_line_through_point_line_with_y_intercept_l2684_268454


namespace NUMINAMATH_CALUDE_work_completion_time_l2684_268478

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 30) →
  (1 / x + 1 / y = 1 / 18) →
  y = 45 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2684_268478


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l2684_268464

/-- The number to be multiplied -/
def n : ℕ := 54

/-- The incorrect multiplier -/
def incorrect_multiplier : ℚ := 2.35

/-- The difference between correct and incorrect results -/
def difference : ℚ := 1.8

/-- The two-digit number formed by the repeating digits -/
def repeating_digits : ℕ := 35

/-- The correct multiplier as a rational number -/
def correct_multiplier : ℚ := 2 + repeating_digits / 99

theorem repeating_decimal_problem :
  n * correct_multiplier - n * incorrect_multiplier = difference :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l2684_268464


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l2684_268430

-- Statement 1
theorem contrapositive_equivalence (x : ℝ) :
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Statement 2
theorem contrapositive_true (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) := by sorry

-- Statement 3
theorem negation_equivalence :
  (¬∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

-- Statement 4
theorem inequality_condition (a : ℝ) :
  (∀ x, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) → a > 2 := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l2684_268430


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2684_268485

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  let vertices_sum := a + b + c + d
  let midpoints_sum := (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2
  midpoints_sum = vertices_sum :=
by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2684_268485


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2684_268482

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * (x - h)^2 + b * (x - h) + d
  ∃ x y : ℝ, f x = g x ∧ f x = y ∧ x = (d - c) / b ∧ y = a * ((d - c) / b)^2 + d :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2684_268482


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2684_268489

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, (5/2 * x^2 + 17*x + c = 0) ↔ (x = (-17 + Real.sqrt 23) / 5 ∨ x = (-17 - Real.sqrt 23) / 5)) 
  → c = 26.6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2684_268489


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2684_268436

theorem hot_dogs_remainder : 25197629 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2684_268436


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2684_268426

/-- A quadratic function passing through (-2, 1) with exactly one root -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g derived from f -/
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

/-- The theorem stating the properties of f and g -/
theorem quadratic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ f a b (-2) = 1) →
  (∃! x : ℝ, f a b x = 0) →
  (∀ x : ℝ, f a b x = (x + 1)^2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → Monotone (g a b k)) ↔ k ≤ 0 ∨ k ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2684_268426


namespace NUMINAMATH_CALUDE_range_of_a_l2684_268488

def prop_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2684_268488


namespace NUMINAMATH_CALUDE_f_min_at_three_l2684_268475

/-- The quadratic function f(x) = x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f x ≥ f 3 := by
sorry

end NUMINAMATH_CALUDE_f_min_at_three_l2684_268475


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l2684_268449

theorem no_multiple_of_five (C : ℕ) : 
  (100 ≤ 100 + 10 * C + 4) ∧ (100 + 10 * C + 4 < 1000) ∧ (C < 10) →
  ¬(∃ k : ℕ, 100 + 10 * C + 4 = 5 * k) := by
sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l2684_268449


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2684_268410

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (5 * x - 1 > 3 * (x + 1) ∧ (1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2684_268410


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_than_perfect_square_l2684_268460

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_8_less_than_perfect_square :
  ∃ (n : ℕ), is_prime n ∧ (∃ (m : ℕ), is_perfect_square m ∧ n = m - 8) ∧
  (∀ (k : ℕ), k < n → ¬(is_prime k ∧ ∃ (m : ℕ), is_perfect_square m ∧ k = m - 8)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_than_perfect_square_l2684_268460


namespace NUMINAMATH_CALUDE_a_in_range_l2684_268445

/-- The function f(x) = -x^2 - 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*a*x

/-- The maximum value of f(x) on [0, 1] is a^2 -/
def max_value (a : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f a x = a^2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a y ≤ a^2

/-- If the maximum value of f(x) on [0, 1] is a^2, then a is in [-1, 0] -/
theorem a_in_range (a : ℝ) (h : max_value a) : a ∈ Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_l2684_268445


namespace NUMINAMATH_CALUDE_meal_serving_problem_l2684_268443

/-- The number of derangements of n elements -/
def subfactorial (n : ℕ) : ℕ := sorry

/-- The number of ways to serve meals to exactly three people correctly -/
def waysToServeThreeCorrectly (totalPeople : ℕ) (mealTypes : ℕ) (peoplePerMeal : ℕ) : ℕ :=
  Nat.choose totalPeople 3 * subfactorial (totalPeople - 3)

theorem meal_serving_problem :
  waysToServeThreeCorrectly 15 3 5 = 80157776755 := by sorry

end NUMINAMATH_CALUDE_meal_serving_problem_l2684_268443


namespace NUMINAMATH_CALUDE_dance_steps_proof_l2684_268474

def total_steps (jason_steps nancy_ratio : ℕ) : ℕ :=
  jason_steps + nancy_ratio * jason_steps

theorem dance_steps_proof (jason_steps nancy_ratio : ℕ) 
  (h1 : jason_steps = 8) 
  (h2 : nancy_ratio = 3) : 
  total_steps jason_steps nancy_ratio = 32 := by
  sorry

end NUMINAMATH_CALUDE_dance_steps_proof_l2684_268474


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2684_268424

/-- The number of four-digit odd numbers divisible by 3 -/
def A : ℕ := sorry

/-- The number of four-digit multiples of 7 -/
def B : ℕ := sorry

/-- The sum of A and B is 2786 -/
theorem sum_of_A_and_B : A + B = 2786 := by sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2684_268424


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2684_268428

theorem r_value_when_n_is_3 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^n - 1) 
  (h2 : r = 3^s + s) 
  (h3 : n = 3) : 
  r = 3^26 + 26 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2684_268428


namespace NUMINAMATH_CALUDE_james_golden_retrievers_l2684_268444

/-- Represents the number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  pitbulls : Nat
  golden_retrievers : Nat

/-- Represents the number of pups each breed has -/
structure PupCounts where
  husky_pups : Nat
  pitbull_pups : Nat
  golden_retriever_pups : Nat

/-- The problem statement -/
theorem james_golden_retrievers (dogs : DogCounts) (pups : PupCounts) : 
  dogs.huskies = 5 →
  dogs.pitbulls = 2 →
  pups.husky_pups = 3 →
  pups.pitbull_pups = 3 →
  pups.golden_retriever_pups = pups.husky_pups + 2 →
  dogs.huskies * pups.husky_pups + 
  dogs.pitbulls * pups.pitbull_pups + 
  dogs.golden_retrievers * pups.golden_retriever_pups = 
  (dogs.huskies + dogs.pitbulls + dogs.golden_retrievers) + 30 →
  dogs.golden_retrievers = 4 := by
sorry

end NUMINAMATH_CALUDE_james_golden_retrievers_l2684_268444


namespace NUMINAMATH_CALUDE_marias_initial_savings_l2684_268496

def sweater_price : ℕ := 30
def scarf_price : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def remaining_money : ℕ := 200

theorem marias_initial_savings :
  (sweater_price * num_sweaters + scarf_price * num_scarves + remaining_money) = 500 := by
  sorry

end NUMINAMATH_CALUDE_marias_initial_savings_l2684_268496


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_l2684_268450

theorem quadratic_roots_integer (b c : ℤ) (k : ℤ) (h : b^2 - 4*c = k^2) :
  ∃ x1 x2 : ℤ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_l2684_268450


namespace NUMINAMATH_CALUDE_max_product_value_l2684_268479

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value (hf : Set.range f = Set.Icc (-3) 5) 
                          (hh : Set.range h = Set.Icc 0 4) : 
  ∃ x y : ℝ, f x * h y ≤ 20 ∧ ∃ a b : ℝ, f a * h b = 20 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_max_product_value_l2684_268479


namespace NUMINAMATH_CALUDE_product_lower_bound_l2684_268494

theorem product_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ (7 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_lower_bound_l2684_268494


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_equation_l2684_268440

theorem solution_set_absolute_value_equation (x : ℝ) :
  |1 - x| + |2*x - 1| = |3*x - 2| ↔ x ≤ 1/2 ∨ x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_equation_l2684_268440


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2684_268469

theorem inequality_solution_set (x : ℝ) : 4 * x < 3 * x + 2 ↔ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2684_268469


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2684_268456

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  x^2 + 4*y^2 + 2*x*y ≥ 3/4 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + 4*y^2 + 2*x*y = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2684_268456


namespace NUMINAMATH_CALUDE_min_distance_to_perpendicular_bisector_l2684_268425

open Complex

theorem min_distance_to_perpendicular_bisector (z : ℂ) :
  (abs z = abs (z + 2 + 2*I)) →
  (∃ (min_val : ℝ), ∀ (w : ℂ), abs w = abs (w + 2 + 2*I) → abs (w - 1 + I) ≥ min_val) ∧
  (∃ (z₀ : ℂ), abs z₀ = abs (z₀ + 2 + 2*I) ∧ abs (z₀ - 1 + I) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_perpendicular_bisector_l2684_268425


namespace NUMINAMATH_CALUDE_teacher_age_l2684_268401

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students : ℝ) * student_avg_age + 36 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l2684_268401


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l2684_268416

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (P : ℝ × ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_hyperbola : (P.1^2 / 16) - (P.2^2 / 9) = 1) 
  (h_on_hyperbola : P ∈ {p : ℝ × ℝ | (p.1^2 / 16) - (p.2^2 / 9) = 1}) 
  (h_focal_points : F₁ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1} ∧ 
                    F₂ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1}) 
  (h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) : 
  |P.2| = 9/5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l2684_268416


namespace NUMINAMATH_CALUDE_probability_one_red_one_white_l2684_268466

/-- The probability of selecting one red ball and one white ball from a bag -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 3 →
  white_balls = 2 →
  (red_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose 2 = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_red_one_white_l2684_268466


namespace NUMINAMATH_CALUDE_sweets_problem_l2684_268499

theorem sweets_problem (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  ∃ total_sweets : ℕ,
    total_sweets = num_children * sweets_per_child / (1 - remaining_fraction) ∧
    total_sweets = 288 := by
  sorry

end NUMINAMATH_CALUDE_sweets_problem_l2684_268499


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2684_268467

/-- The manufacturing cost of a shoe given transportation cost, selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (profit_margin : ℚ) :
  transportation_cost = 5 →
  selling_price = 234 →
  profit_margin = 1/5 →
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 190 :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2684_268467


namespace NUMINAMATH_CALUDE_book_count_theorem_l2684_268442

/-- Represents the book collection of a person -/
structure BookCollection where
  initial : ℕ
  bought : ℕ
  lost : ℕ
  borrowed : ℕ

/-- Calculates the current number of books in a collection -/
def current_books (collection : BookCollection) : ℕ :=
  collection.initial - collection.lost

/-- Calculates the future number of books in a collection -/
def future_books (collection : BookCollection) : ℕ :=
  current_books collection + collection.bought + collection.borrowed

/-- Jason's book collection -/
def jason : BookCollection :=
  { initial := 18, bought := 8, lost := 0, borrowed := 0 }

/-- Mary's book collection -/
def mary : BookCollection :=
  { initial := 42, bought := 0, lost := 6, borrowed := 5 }

theorem book_count_theorem :
  (current_books jason + current_books mary = 54) ∧
  (future_books jason + future_books mary = 67) := by
  sorry

end NUMINAMATH_CALUDE_book_count_theorem_l2684_268442


namespace NUMINAMATH_CALUDE_sum_not_ending_in_seven_l2684_268448

theorem sum_not_ending_in_seven (n : ℕ) : ¬ (∃ k : ℕ, n * (n + 1) / 2 = 10 * k + 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_ending_in_seven_l2684_268448


namespace NUMINAMATH_CALUDE_equation_solution_l2684_268427

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The theorem stating that functions satisfying the equation are either the identity function or the absolute value function -/
theorem equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end NUMINAMATH_CALUDE_equation_solution_l2684_268427


namespace NUMINAMATH_CALUDE_no_perfect_square_2007_plus_4n_l2684_268472

theorem no_perfect_square_2007_plus_4n :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), 2007 + 4^n = k^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_2007_plus_4n_l2684_268472


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2684_268415

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

/-- Theorem stating that (2,3) is the only self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 3 ∧
  ∃ a b : ℝ, a ≠ b ∧ 
    x a = x b ∧ y a = y b ∧
    x a = p.1 ∧ y a = p.2 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2684_268415


namespace NUMINAMATH_CALUDE_num_ways_eq_1716_l2684_268486

/-- The number of distinct ways to choose 8 non-negative integers that sum to 6 -/
def num_ways : ℕ := Nat.choose 13 7

theorem num_ways_eq_1716 : num_ways = 1716 := by sorry

end NUMINAMATH_CALUDE_num_ways_eq_1716_l2684_268486


namespace NUMINAMATH_CALUDE_willy_crayon_count_l2684_268463

/-- Given that Lucy has 3,971 crayons and Willy has 1,121 more crayons than Lucy,
    prove that Willy has 5,092 crayons. -/
theorem willy_crayon_count (lucy_crayons : ℕ) (willy_extra_crayons : ℕ) 
    (h1 : lucy_crayons = 3971)
    (h2 : willy_extra_crayons = 1121) :
    lucy_crayons + willy_extra_crayons = 5092 := by
  sorry

end NUMINAMATH_CALUDE_willy_crayon_count_l2684_268463


namespace NUMINAMATH_CALUDE_min_value_theorem_l2684_268438

/-- Given positive real numbers a, b, c, and a function f with minimum value 2,
    prove that a + b + c = 2 and the minimum value of 1/a + 1/b + 1/c is 9/2 -/
theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 2) :
  (a + b + c = 2) ∧ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → 1/x + 1/y + 1/z ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2684_268438


namespace NUMINAMATH_CALUDE_stamp_solution_l2684_268435

def stamp_problem (one_cent two_cent five_cent eight_cent : ℕ) : Prop :=
  two_cent = (3 * one_cent) / 4 ∧
  five_cent = (3 * two_cent) / 4 ∧
  eight_cent = 5 ∧
  one_cent * 1 + two_cent * 2 + five_cent * 5 + eight_cent * 8 = 100000

theorem stamp_solution :
  ∃ (one_cent two_cent five_cent eight_cent : ℕ),
    stamp_problem one_cent two_cent five_cent eight_cent ∧
    one_cent = 18816 ∧
    two_cent = 14112 ∧
    five_cent = 10584 ∧
    eight_cent = 5 :=
  sorry

end NUMINAMATH_CALUDE_stamp_solution_l2684_268435


namespace NUMINAMATH_CALUDE_fraction_change_l2684_268405

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_l2684_268405


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2684_268459

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (x + 1/2 + Real.sqrt (x + 1/4)) + x = a) ↔ a ≥ 1/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2684_268459


namespace NUMINAMATH_CALUDE_points_collinear_opposite_collinear_k_l2684_268493

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-zero vectors a and b
variable (a b : V)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hnc : ¬ ∃ (r : ℝ), a = r • b)

-- Define vectors AB, BC, and CD
def AB : V := a + b
def BC : V := 2 • a + 8 • b
def CD : V := 3 • (a - b)

-- Define collinearity
def collinear (u v : V) : Prop := ∃ (r : ℝ), u = r • v

-- Theorem 1: Points A, B, D are collinear
theorem points_collinear : 
  ∃ (r : ℝ), AB a b = r • (AB a b + BC a b + CD a b) :=
sorry

-- Theorem 2: Value of k for opposite collinearity
theorem opposite_collinear_k : 
  ∃ (k : ℝ), k = -1 ∧ 
  (∃ (r : ℝ), r < 0 ∧ k • a + b = r • (a + k • b)) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_opposite_collinear_k_l2684_268493


namespace NUMINAMATH_CALUDE_oh_squared_equals_526_l2684_268447

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter O and orthocenter H
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumradius R
def circumradius (t : Triangle) : ℝ := sorry

-- Define side lengths a, b, c
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

theorem oh_squared_equals_526 (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let (a, b, c) := side_lengths t
  R = 8 →
  2 * a^2 + b^2 + c^2 = 50 →
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = 526 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_equals_526_l2684_268447


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l2684_268484

/-- Calculates the total cost for Travel Agency A -/
def costA (fullPrice : ℕ) (numStudents : ℕ) : ℕ :=
  fullPrice + numStudents * (fullPrice / 2)

/-- Calculates the total cost for Travel Agency B -/
def costB (fullPrice : ℕ) (numPeople : ℕ) : ℕ :=
  numPeople * (fullPrice * 60 / 100)

theorem travel_agency_comparison (fullPrice : ℕ) :
  (fullPrice = 240) →
  (costA fullPrice 5 < costB fullPrice 6) ∧
  (costB fullPrice 3 < costA fullPrice 2) := by
  sorry


end NUMINAMATH_CALUDE_travel_agency_comparison_l2684_268484


namespace NUMINAMATH_CALUDE_circle_radius_in_rectangle_l2684_268465

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure Circle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 3 -/
theorem circle_radius_in_rectangle (rect : Rectangle) (circ : Circle rect) :
  circ.passes_through_vertices ∧ circ.tangent_to_opposite_side → circ.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_rectangle_l2684_268465


namespace NUMINAMATH_CALUDE_max_A_at_375_l2684_268455

def A (k : ℕ) : ℝ := (Nat.choose 1500 k) * (0.3 ^ k)

theorem max_A_at_375 : 
  ∀ k : ℕ, k ≤ 1500 → A k ≤ A 375 :=
by sorry

end NUMINAMATH_CALUDE_max_A_at_375_l2684_268455


namespace NUMINAMATH_CALUDE_total_stamps_l2684_268446

theorem total_stamps (stamps_AJ : ℕ) (stamps_KJ : ℕ) (stamps_CJ : ℕ) : 
  stamps_AJ = 370 →
  stamps_KJ = stamps_AJ / 2 →
  stamps_CJ = 2 * stamps_KJ + 5 →
  stamps_AJ + stamps_KJ + stamps_CJ = 930 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l2684_268446


namespace NUMINAMATH_CALUDE_range_of_a_l2684_268437

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2684_268437


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2684_268473

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  11 ∣ x →
  (∀ s : ℕ, Prime s → s ∣ x → s = 11 ∨ s = 7 ∨ s = 101) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2684_268473


namespace NUMINAMATH_CALUDE_chase_blue_jays_count_l2684_268476

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := 5

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

/-- The percentage more birds Gabrielle saw compared to Chase -/
def percentage_difference : ℚ := 1/5

theorem chase_blue_jays_count :
  (gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays : ℚ) =
  (chase_robins + chase_cardinals + chase_blue_jays : ℚ) * (1 + percentage_difference) :=
by sorry

end NUMINAMATH_CALUDE_chase_blue_jays_count_l2684_268476


namespace NUMINAMATH_CALUDE_red_envelope_prob_is_one_third_l2684_268407

def red_envelope_prob : ℚ :=
  let total_people : ℕ := 4
  let one_yuan_envelopes : ℕ := 3
  let five_yuan_envelopes : ℕ := 1
  let total_events : ℕ := Nat.choose total_people 2
  let favorable_events : ℕ := one_yuan_envelopes * five_yuan_envelopes
  favorable_events / total_events

theorem red_envelope_prob_is_one_third : 
  red_envelope_prob = 1/3 := by sorry

end NUMINAMATH_CALUDE_red_envelope_prob_is_one_third_l2684_268407


namespace NUMINAMATH_CALUDE_men_entered_room_l2684_268419

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered : ℕ) (h1 : initial_men * 5 = initial_women * 4) 
  (h2 : initial_men + men_entered = 14) 
  (h3 : 2 * (initial_women - 3) = 24) : men_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_men_entered_room_l2684_268419


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2684_268406

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop where
  ratio_eq : (dist P Q) / (dist X Y) = (dist Q R) / (dist Y Z)

theorem similar_triangles_side_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z) 
  (h_PQ : dist P Q = 9)
  (h_QR : dist Q R = 15)
  (h_YZ : dist Y Z = 30) :
  dist X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2684_268406


namespace NUMINAMATH_CALUDE_binomial_not_divisible_l2684_268433

theorem binomial_not_divisible (P n k : ℕ) (hP : P > 1) :
  ∃ i ∈ Finset.range (k + 1), ¬(P ∣ Nat.choose (n + i) k) := by
  sorry

end NUMINAMATH_CALUDE_binomial_not_divisible_l2684_268433


namespace NUMINAMATH_CALUDE_susanna_purchase_l2684_268461

/-- The cost of each item in pounds and pence -/
structure ItemCost where
  pounds : ℕ
  pence : Fin 100
  pence_eq : pence = 99

/-- The total amount spent by Susanna in pence -/
def total_spent : ℕ := 65 * 100 + 76

/-- The number of items Susanna bought -/
def items_bought : ℕ := 24

theorem susanna_purchase :
  ∀ (cost : ItemCost),
  (cost.pounds * 100 + cost.pence) * items_bought = total_spent :=
sorry

end NUMINAMATH_CALUDE_susanna_purchase_l2684_268461


namespace NUMINAMATH_CALUDE_problem_2011_l2684_268471

theorem problem_2011 : (2011^2 - 2011) / 2011 = 2010 := by sorry

end NUMINAMATH_CALUDE_problem_2011_l2684_268471


namespace NUMINAMATH_CALUDE_total_amount_is_175_l2684_268403

/-- Represents the share of each person in rupees -/
structure Shares :=
  (first : ℝ)
  (second : ℝ)
  (third : ℝ)

/-- Calculates the total amount given the shares -/
def total_amount (s : Shares) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: The total amount is 175 rupees -/
theorem total_amount_is_175 :
  ∃ (s : Shares),
    s.second = 45 ∧
    s.second = 0.45 * s.first ∧
    s.third = 0.30 * s.first ∧
    total_amount s = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_175_l2684_268403


namespace NUMINAMATH_CALUDE_prob_yellow_twice_is_one_ninth_l2684_268418

/-- A fair 12-sided die with 4 yellow faces -/
structure YellowDie :=
  (sides : ℕ)
  (yellow_faces : ℕ)
  (is_fair : sides = 12)
  (yellow_count : yellow_faces = 4)

/-- The probability of rolling yellow twice with a YellowDie -/
def prob_yellow_twice (d : YellowDie) : ℚ :=
  (d.yellow_faces : ℚ) / (d.sides : ℚ) * (d.yellow_faces : ℚ) / (d.sides : ℚ)

/-- Theorem: The probability of rolling yellow twice with a YellowDie is 1/9 -/
theorem prob_yellow_twice_is_one_ninth (d : YellowDie) :
  prob_yellow_twice d = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_yellow_twice_is_one_ninth_l2684_268418


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2684_268413

/-- Given a line L1 with equation 4x + 5y - 8 = 0 and a point A(3,2),
    the line L2 passing through A and perpendicular to L1 has equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y - 8 = 0
  let A : ℝ × ℝ := (3, 2)
  let L2 : ℝ → ℝ → Prop := λ x y => 4 * y - 5 * x + 7 = 0
  (∀ x y, L2 x y ↔ (y - A.2) = -(4/5) * (x - A.1)) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 4/5) →
  L2 A.1 A.2 ∧ ∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = -5/4 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2684_268413


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l2684_268457

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 → (3 * i - 3 * i⁻¹)⁻¹ = -i / 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l2684_268457


namespace NUMINAMATH_CALUDE_fourth_rectangle_perimeter_is_10_l2684_268421

/-- The perimeter of the fourth rectangle in a large rectangle cut into four smaller ones --/
def fourth_rectangle_perimeter (p1 p2 p3 : ℕ) : ℕ :=
  p1 + p2 - p3

/-- Theorem stating that the perimeter of the fourth rectangle is 10 --/
theorem fourth_rectangle_perimeter_is_10 :
  fourth_rectangle_perimeter 16 18 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_perimeter_is_10_l2684_268421


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2684_268491

/-- The number of distinct arrangements of beads on a bracelet -/
def distinct_bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet is 2520 -/
theorem eight_bead_bracelet_arrangements :
  distinct_bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2684_268491


namespace NUMINAMATH_CALUDE_sara_apples_l2684_268412

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) : 
  total = 240 →
  ali_factor = 7 →
  total = sara_apples + ali_factor * sara_apples →
  sara_apples = 30 := by
sorry

end NUMINAMATH_CALUDE_sara_apples_l2684_268412


namespace NUMINAMATH_CALUDE_waiter_tips_theorem_l2684_268409

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips_theorem :
  total_tips 10 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_theorem_l2684_268409


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l2684_268492

/-- Represents the tank filling problem -/
structure TankProblem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ
  cycle_duration : ℕ
  total_time : ℕ

/-- The specific tank problem instance -/
def tankInstance : TankProblem :=
  { capacity := 850,
    pipeA_rate := 40,
    pipeB_rate := 30,
    pipeC_rate := 20,
    cycle_duration := 3,
    total_time := 51 }

/-- Calculates the net amount filled in one cycle -/
def netFillPerCycle (t : TankProblem) : ℕ :=
  t.pipeA_rate + t.pipeB_rate - t.pipeC_rate

/-- Theorem stating that the given tank instance has the correct capacity -/
theorem tank_capacity_correct (t : TankProblem) : 
  t = tankInstance → 
  t.capacity = (t.total_time / t.cycle_duration) * netFillPerCycle t :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l2684_268492


namespace NUMINAMATH_CALUDE_cookie_box_cost_l2684_268477

def bracelet_cost : ℝ := 1
def bracelet_price : ℝ := 1.5
def num_bracelets : ℕ := 12
def money_left : ℝ := 3

theorem cookie_box_cost :
  let profit_per_bracelet := bracelet_price - bracelet_cost
  let total_profit := (num_bracelets : ℝ) * profit_per_bracelet
  total_profit - money_left = 3 := by sorry

end NUMINAMATH_CALUDE_cookie_box_cost_l2684_268477


namespace NUMINAMATH_CALUDE_expression_equalities_l2684_268480

theorem expression_equalities :
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
sorry

end NUMINAMATH_CALUDE_expression_equalities_l2684_268480


namespace NUMINAMATH_CALUDE_minimum_percentage_bad_work_l2684_268490

theorem minimum_percentage_bad_work (total_works : ℝ) (h_total_positive : total_works > 0) :
  let bad_works := 0.2 * total_works
  let good_works := 0.8 * total_works
  let misclassified_good := 0.1 * good_works
  let misclassified_bad := 0.1 * bad_works
  let rechecked_works := bad_works - misclassified_bad + misclassified_good
  let actual_bad_rechecked := bad_works - misclassified_bad
  ⌊(actual_bad_rechecked / rechecked_works * 100)⌋ = 69 :=
by sorry

end NUMINAMATH_CALUDE_minimum_percentage_bad_work_l2684_268490


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_a_2n_specific_form_l2684_268417

/-- The number of natural numbers with digit sum n, using only digits 1, 3, and 4 -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: a₂ₙ is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by
  sorry

/-- The specific form of a₂ₙ as (aₙ + aₙ₋₂)² -/
theorem a_2n_specific_form (n : ℕ) : a (2 * n) = (a n + a (n - 2)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_a_2n_specific_form_l2684_268417


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l2684_268498

theorem min_sum_box_dimensions :
  ∀ (l w h : ℕ+),
    l * w * h = 2002 →
    ∀ (a b c : ℕ+),
      a * b * c = 2002 →
      l + w + h ≤ a + b + c →
      l + w + h = 38 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l2684_268498


namespace NUMINAMATH_CALUDE_bug_distance_is_28_l2684_268481

def bug_crawl (start end1 end2 end3 : Int) : Int :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_distance_is_28 :
  bug_crawl 3 (-4) 8 (-1) = 28 := by
  sorry

end NUMINAMATH_CALUDE_bug_distance_is_28_l2684_268481


namespace NUMINAMATH_CALUDE_grape_candy_count_l2684_268451

/-- Represents the number of cherry candies -/
def cherry : ℕ := sorry

/-- Represents the number of grape candies -/
def grape : ℕ := 3 * cherry

/-- Represents the number of apple candies -/
def apple : ℕ := 2 * grape

/-- The cost of each candy in cents -/
def cost_per_candy : ℕ := 250

/-- The total cost of all candies in cents -/
def total_cost : ℕ := 20000

theorem grape_candy_count :
  grape = 24 ∧
  cherry + grape + apple = total_cost / cost_per_candy :=
sorry

end NUMINAMATH_CALUDE_grape_candy_count_l2684_268451


namespace NUMINAMATH_CALUDE_kelly_games_left_l2684_268439

theorem kelly_games_left (initial_games : ℝ) (games_given_away : ℕ) : 
  initial_games = 121.0 →
  games_given_away = 99 →
  initial_games - games_given_away = 22.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l2684_268439


namespace NUMINAMATH_CALUDE_twice_probability_possible_l2684_268408

/-- Represents the schedule of trains in one direction -/
structure TrainSchedule :=
  (interval : ℝ)
  (offset : ℝ)

/-- Represents the metro system with two directions -/
structure MetroSystem :=
  (direction1 : TrainSchedule)
  (direction2 : TrainSchedule)

/-- Calculates the probability of taking a train in a given direction -/
def probability_of_taking_train (metro : MetroSystem) (direction : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the probability of taking one train can be twice the other -/
theorem twice_probability_possible (metro : MetroSystem) :
  ∃ (direction1 direction2 : ℕ),
    direction1 ≠ direction2 ∧
    probability_of_taking_train metro direction1 = 2 * probability_of_taking_train metro direction2 :=
  sorry

end NUMINAMATH_CALUDE_twice_probability_possible_l2684_268408


namespace NUMINAMATH_CALUDE_sum_of_digits_square_count_l2684_268487

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The count of valid numbers -/
def K : ℕ := sorry

theorem sum_of_digits_square_count :
  K % 1000 = 632 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_count_l2684_268487


namespace NUMINAMATH_CALUDE_percentage_of_science_students_l2684_268431

theorem percentage_of_science_students (total_boys : ℕ) (school_A_percentage : ℚ) (non_science_boys : ℕ) : 
  total_boys = 250 →
  school_A_percentage = 1/5 →
  non_science_boys = 35 →
  (((school_A_percentage * total_boys) - non_science_boys) / (school_A_percentage * total_boys)) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_science_students_l2684_268431


namespace NUMINAMATH_CALUDE_function_value_at_three_l2684_268452

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 3 * f (1 - x) = 4 * x^2

theorem function_value_at_three 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : 
  f 3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l2684_268452
