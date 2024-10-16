import Mathlib

namespace NUMINAMATH_CALUDE_positive_real_inequality_l3938_393864

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^4 * b^b * c^c ≥ min a (min b c) * min b (min a c) * min c (min a b) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3938_393864


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3938_393830

theorem power_of_product_equals_product_of_powers (a b : ℝ) : 
  (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3938_393830


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3938_393845

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3938_393845


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l3938_393809

theorem recreation_spending_comparison (last_week_wages : ℝ) : 
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.70 * last_week_wages
  let this_week_recreation := 0.20 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l3938_393809


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l3938_393833

def birds_on_fence (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

theorem total_birds_on_fence :
  birds_on_fence 12 8 = 20 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l3938_393833


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_2014_l3938_393852

theorem largest_prime_divisor_of_2014 (p : ℕ) (hp : Nat.Prime p) :
  (2020 % p = 6) → p ≤ 53 ∧ ∃ (q : ℕ), Nat.Prime q ∧ q ∣ 2014 ∧ q = 53 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_2014_l3938_393852


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3938_393841

/-- Checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments --/
def set_A : List ℝ := [2, 5, 7]
def set_B : List ℝ := [4, 4, 8]
def set_C : List ℝ := [4, 5, 6]
def set_D : List ℝ := [4, 5, 10]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3938_393841


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3938_393806

theorem quadratic_roots_properties : ∃ (a b : ℝ), 
  (a^2 + a - 2023 = 0) ∧ 
  (b^2 + b - 2023 = 0) ∧ 
  (a * b = -2023) ∧ 
  (a^2 - b = 2024) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3938_393806


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3938_393851

/-- The sum of the first n terms of the arithmetic progression -/
def S (n : ℕ) : ℤ := 2 * n + 3 * n^3

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℤ := 9 * r^2 - 9 * r + 5

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3938_393851


namespace NUMINAMATH_CALUDE_pi_estimation_l3938_393839

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l3938_393839


namespace NUMINAMATH_CALUDE_vector_BC_coordinates_l3938_393823

theorem vector_BC_coordinates :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (3, 2)
  let AC : ℝ × ℝ := (4, 3)
  let BC : ℝ × ℝ := (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2)
  BC = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_BC_coordinates_l3938_393823


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l3938_393863

theorem min_distance_to_circle (x y : ℝ) : 
  (x - 2)^2 + (y - 1)^2 = 1 → x^2 + y^2 ≥ 6 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l3938_393863


namespace NUMINAMATH_CALUDE_fault_line_movement_l3938_393828

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  past_year : ℝ
  year_before : ℝ

/-- Calculates the total movement of a fault line over two years -/
def total_movement (f : FaultLineMovement) : ℝ :=
  f.past_year + f.year_before

/-- Theorem: The total movement of the fault line is 6.50 inches -/
theorem fault_line_movement :
  let f : FaultLineMovement := { past_year := 1.25, year_before := 5.25 }
  total_movement f = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l3938_393828


namespace NUMINAMATH_CALUDE_cereal_eating_time_l3938_393885

theorem cereal_eating_time (fat_rate mr_thin_rate total_cereal : ℚ) :
  fat_rate = 1 / 15 →
  mr_thin_rate = 1 / 45 →
  total_cereal = 4 →
  (total_cereal / (fat_rate + mr_thin_rate) = 45) :=
by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l3938_393885


namespace NUMINAMATH_CALUDE_binomial_coefficient_not_always_divisible_l3938_393861

theorem binomial_coefficient_not_always_divisible :
  ∃ k : ℕ+, ∀ n : ℕ, n > 1 → ∃ i : ℕ, 1 ≤ i ∧ i ≤ n - 1 ∧ ¬(k : ℕ) ∣ Nat.choose n i := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_not_always_divisible_l3938_393861


namespace NUMINAMATH_CALUDE_sum_of_squared_pairs_l3938_393815

theorem sum_of_squared_pairs (p q r : ℝ) : 
  (p^3 - 18*p^2 + 25*p - 6 = 0) →
  (q^3 - 18*q^2 + 25*q - 6 = 0) →
  (r^3 - 18*r^2 + 25*r - 6 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 598 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_pairs_l3938_393815


namespace NUMINAMATH_CALUDE_f_is_power_function_l3938_393876

/-- Definition of a power function -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

/-- The function y = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: f is a power function -/
theorem f_is_power_function : is_power_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_power_function_l3938_393876


namespace NUMINAMATH_CALUDE_equality_condition_l3938_393875

theorem equality_condition (x : ℝ) (h1 : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l3938_393875


namespace NUMINAMATH_CALUDE_simplify_expression_l3938_393897

theorem simplify_expression (y : ℝ) : 7*y - 3 + 2*y + 15 = 9*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3938_393897


namespace NUMINAMATH_CALUDE_john_purchase_profit_l3938_393838

/-- Represents the purchase and sale of items with profit or loss -/
theorem john_purchase_profit (x : ℝ) : 
  let grinder_purchase := 15000
  let grinder_loss_percent := 0.04
  let mobile_profit_percent := 0.15
  let total_profit := 600
  let grinder_sale := grinder_purchase * (1 - grinder_loss_percent)
  let mobile_sale := x * (1 + mobile_profit_percent)
  (mobile_sale - x) - (grinder_purchase - grinder_sale) = total_profit →
  x = 8000 := by
sorry

end NUMINAMATH_CALUDE_john_purchase_profit_l3938_393838


namespace NUMINAMATH_CALUDE_count_between_multiples_l3938_393822

def multiples_of_4 : List Nat := List.filter (fun n => n % 4 = 0) (List.range 100)

def fifth_from_left : Nat := multiples_of_4[4]

def eighth_from_right : Nat := multiples_of_4[multiples_of_4.length - 8]

theorem count_between_multiples :
  (List.filter (fun n => n > fifth_from_left ∧ n < eighth_from_right) multiples_of_4).length = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_between_multiples_l3938_393822


namespace NUMINAMATH_CALUDE_small_jar_capacity_l3938_393893

theorem small_jar_capacity 
  (total_jars : ℕ) 
  (large_jar_capacity : ℕ) 
  (total_capacity : ℕ) 
  (small_jars : ℕ) 
  (h1 : total_jars = 100)
  (h2 : large_jar_capacity = 5)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - (total_jars - small_jars) * large_jar_capacity) / small_jars = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_jar_capacity_l3938_393893


namespace NUMINAMATH_CALUDE_f_increasing_interval_f_not_increasing_below_one_l3938_393814

/-- The function f(x) = |x-1| + |x+1| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

/-- The interval of increase for f(x) -/
def interval_of_increase : Set ℝ := { x | x ≥ 1 }

/-- Theorem stating that the interval of increase for f(x) is [1, +∞) -/
theorem f_increasing_interval :
  ∀ x y, x ∈ interval_of_increase → y ∈ interval_of_increase → x < y → f x < f y :=
by sorry

/-- Theorem stating that f(x) is not increasing for x < 1 -/
theorem f_not_increasing_below_one :
  ∃ x y, x < 1 ∧ y < 1 ∧ x < y ∧ f x ≥ f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_interval_f_not_increasing_below_one_l3938_393814


namespace NUMINAMATH_CALUDE_john_travel_solution_l3938_393872

/-- Represents the problem of calculating the distance John travels -/
def john_travel_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (total_distance : ℝ) (total_time : ℝ),
    initial_speed * initial_time = initial_speed ∧
    total_distance = initial_speed * (total_time + late_time / 60) ∧
    total_distance = initial_speed * initial_time + 
      (initial_speed + speed_increase) * (total_time - initial_time - early_time / 60) ∧
    total_distance = 123.4375

/-- The theorem stating that the solution to John's travel problem exists -/
theorem john_travel_solution : 
  john_travel_problem 25 20 1 1.5 0.25 := by sorry

end NUMINAMATH_CALUDE_john_travel_solution_l3938_393872


namespace NUMINAMATH_CALUDE_probability_integer_exponent_x_l3938_393867

theorem probability_integer_exponent_x (a : ℝ) (x : ℝ) :
  let expansion := (x - a / Real.sqrt x) ^ 5
  let total_terms := 6
  let integer_exponent_terms := 3
  (integer_exponent_terms : ℚ) / total_terms = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_probability_integer_exponent_x_l3938_393867


namespace NUMINAMATH_CALUDE_max_sum_scores_max_sum_scores_achievable_l3938_393824

/-- Represents the scoring system for an exam -/
structure ExamScoring where
  m : ℕ             -- number of questions
  n : ℕ             -- number of students
  x : Fin m → ℕ     -- number of students who answered each question incorrectly
  h_m : m ≥ 2
  h_n : n ≥ 2
  h_x : ∀ k, x k ≤ n

/-- The score of a student -/
def student_score (E : ExamScoring) : ℕ → ℕ := sorry

/-- The highest score in the exam -/
def max_score (E : ExamScoring) : ℕ := sorry

/-- The lowest score in the exam -/
def min_score (E : ExamScoring) : ℕ := sorry

/-- Theorem: The maximum possible sum of the highest and lowest scores is m(n-1) -/
theorem max_sum_scores (E : ExamScoring) : 
  max_score E + min_score E ≤ E.m * (E.n - 1) :=
sorry

/-- Theorem: The maximum sum of scores is achievable -/
theorem max_sum_scores_achievable (m n : ℕ) (h_m : m ≥ 2) (h_n : n ≥ 2) : 
  ∃ E : ExamScoring, E.m = m ∧ E.n = n ∧ max_score E + min_score E = m * (n - 1) :=
sorry

end NUMINAMATH_CALUDE_max_sum_scores_max_sum_scores_achievable_l3938_393824


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l3938_393848

theorem polynomial_non_negative (x : ℝ) : x^12 - x^7 - x^5 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l3938_393848


namespace NUMINAMATH_CALUDE_corrected_mean_l3938_393817

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ initial_mean = 32 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n * initial_mean - incorrect_value + correct_value) / n = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3938_393817


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_property_l3938_393859

/-- Given a cubic polynomial x^3 + ax^2 + bx + 16a where a and b are nonzero integers,
    if two of its roots coincide and all three roots are integers,
    then |ab| = 2496 -/
theorem cubic_polynomial_root_property (a b : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∃ r s : ℤ, (X - r)^2 * (X - s) = X^3 + a*X^2 + b*X + 16*a) →
  |a * b| = 2496 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_property_l3938_393859


namespace NUMINAMATH_CALUDE_net_pay_calculation_l3938_393865

/-- Calculate net pay given gross pay and tax paid -/
def netPay (grossPay : ℕ) (taxPaid : ℕ) : ℕ :=
  grossPay - taxPaid

theorem net_pay_calculation (grossPay : ℕ) (taxPaid : ℕ) 
  (h1 : grossPay = 450)
  (h2 : taxPaid = 135) :
  netPay grossPay taxPaid = 315 := by
  sorry

end NUMINAMATH_CALUDE_net_pay_calculation_l3938_393865


namespace NUMINAMATH_CALUDE_percentage_problem_l3938_393883

theorem percentage_problem : 
  ∃ x : ℝ, (0.62 * 150 - x / 100 * 250 = 43) ∧ (x = 20) := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3938_393883


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l3938_393812

/-- Given a 1 gallon container of 75% alcohol solution, if 0.4 gallon is drained off and
    replaced with x% alcohol solution to produce a 1 gallon 65% alcohol solution,
    then x = 50%. -/
theorem alcohol_mixture_problem (x : ℝ) : 
  (0.75 * (1 - 0.4) + 0.4 * (x / 100) = 0.65) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l3938_393812


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3938_393899

/-- A geometric sequence with first term 3 and sum of first, third, and fifth terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = 3 * q ^ (n - 1)) ∧
  a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the sequence is 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3938_393899


namespace NUMINAMATH_CALUDE_problem_statement_l3938_393803

theorem problem_statement : ∃ x : ℝ, x * (1/2)^2 = 2^3 ∧ x = 32 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3938_393803


namespace NUMINAMATH_CALUDE_tower_house_block_difference_l3938_393849

def blocks_for_tower : ℕ := 50
def blocks_for_house : ℕ := 20

theorem tower_house_block_difference :
  blocks_for_tower - blocks_for_house = 30 :=
by sorry

end NUMINAMATH_CALUDE_tower_house_block_difference_l3938_393849


namespace NUMINAMATH_CALUDE_min_sum_of_product_l3938_393829

theorem min_sum_of_product (a b : ℤ) (h : a * b = 196) : 
  ∀ x y : ℤ, x * y = 196 → a + b ≤ x + y ∧ ∃ a b : ℤ, a * b = 196 ∧ a + b = -197 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l3938_393829


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3938_393844

theorem functional_equation_solution (f : ℝ × ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x, y) + f (y, z) + f (z, x) = 0) :
  ∃ g : ℝ → ℝ, ∀ x y : ℝ, f (x, y) = g x - g y := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3938_393844


namespace NUMINAMATH_CALUDE_f_max_value_l3938_393858

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem stating that the maximum value of f is -3
theorem f_max_value : ∃ (M : ℝ), M = -3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3938_393858


namespace NUMINAMATH_CALUDE_matthews_crayons_count_l3938_393846

/-- The number of crayons Annie starts with -/
def initial_crayons : ℕ := 4

/-- The number of crayons Annie ends with -/
def final_crayons : ℕ := 40

/-- The number of crayons Matthew gave to Annie -/
def matthews_crayons : ℕ := final_crayons - initial_crayons

theorem matthews_crayons_count : matthews_crayons = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crayons_count_l3938_393846


namespace NUMINAMATH_CALUDE_problem_factory_daily_production_l3938_393866

/-- A factory that produces toys -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  daily_production : ℕ
  h1 : weekly_production = working_days * daily_production

/-- The specific toy factory in the problem -/
def problem_factory : ToyFactory where
  weekly_production := 8000
  working_days := 4
  daily_production := 2000
  h1 := rfl

/-- Theorem stating that the daily production of the problem factory is 2000 toys -/
theorem problem_factory_daily_production :
  problem_factory.daily_production = 2000 := by sorry

end NUMINAMATH_CALUDE_problem_factory_daily_production_l3938_393866


namespace NUMINAMATH_CALUDE_cashier_bills_l3938_393810

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 126 → total_value = 840 → ∃ (five_dollar_bills ten_dollar_bills : ℕ),
    five_dollar_bills + ten_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    five_dollar_bills = 84 := by
  sorry

end NUMINAMATH_CALUDE_cashier_bills_l3938_393810


namespace NUMINAMATH_CALUDE_maya_shoe_probability_l3938_393886

/-- Represents the number of pairs for each shoe color --/
structure ShoePairs where
  black : Nat
  brown : Nat
  grey : Nat
  white : Nat

/-- Calculates the probability of picking two shoes of the same color,
    one left and one right, given a distribution of shoe pairs --/
def samePairColorProbability (pairs : ShoePairs) : Rat :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.grey + pairs.white)
  let numerator := pairs.black * pairs.black + pairs.brown * pairs.brown +
                   pairs.grey * pairs.grey + pairs.white * pairs.white
  numerator / (totalShoes * (totalShoes - 1))

/-- Maya's shoe collection --/
def mayasShoes : ShoePairs := ⟨8, 4, 3, 1⟩

theorem maya_shoe_probability :
  samePairColorProbability mayasShoes = 45 / 248 := by
  sorry

end NUMINAMATH_CALUDE_maya_shoe_probability_l3938_393886


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_relation_l3938_393802

/-- Given a square with perimeter 40 and a larger equilateral triangle with 
    perimeter a + b√p (where p is prime), prove that if a = 30, b = 10, 
    and p = 3, then 7a + 5b + 3p = 269. -/
theorem square_triangle_perimeter_relation 
  (square_perimeter : ℝ) 
  (a b : ℝ) 
  (p : ℕ) 
  (h1 : square_perimeter = 40)
  (h2 : Nat.Prime p)
  (h3 : a = 30)
  (h4 : b = 10)
  (h5 : p = 3)
  : 7 * a + 5 * b + 3 * ↑p = 269 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_relation_l3938_393802


namespace NUMINAMATH_CALUDE_smallest_block_with_231_hidden_l3938_393818

/-- Represents a rectangular block made of unit cubes -/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of hidden cubes when three faces are visible -/
def hiddenCubes (block : RectangularBlock) : ℕ :=
  (block.length - 1) * (block.width - 1) * (block.height - 1)

/-- Calculates the total number of cubes in the block -/
def totalCubes (block : RectangularBlock) : ℕ :=
  block.length * block.width * block.height

/-- Theorem: The smallest possible number of cubes in a block with 231 hidden cubes is 384 -/
theorem smallest_block_with_231_hidden : 
  ∃ (block : RectangularBlock), 
    hiddenCubes block = 231 ∧ 
    totalCubes block = 384 ∧ 
    (∀ (other : RectangularBlock), hiddenCubes other = 231 → totalCubes other ≥ 384) := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_with_231_hidden_l3938_393818


namespace NUMINAMATH_CALUDE_distance_between_squares_l3938_393816

/-- Given two squares where the smaller square has a perimeter of 8 cm and the larger square has an area of 36 cm², 
    prove that the distance between opposite corners of the two squares is approximately 8.9 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ) 
  (h1 : small_square_perimeter = 8) 
  (h2 : large_square_area = 36) : 
  ∃ (distance : ℝ), abs (distance - Real.sqrt 80) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_squares_l3938_393816


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l3938_393877

theorem waiter_income_fraction (salary : ℚ) (salary_positive : salary > 0) : 
  let tips := (7 / 3) * salary
  let bonuses := (2 / 5) * salary
  let total_income := salary + tips + bonuses
  tips / total_income = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l3938_393877


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3938_393837

theorem polynomial_factorization (m : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3938_393837


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3938_393860

open Set

universe u

def U : Set (Fin 6) := {1,2,3,4,5,6}
def A : Set (Fin 6) := {2,4,6}
def B : Set (Fin 6) := {1,2,3,5}

theorem intersection_with_complement : A ∩ (U \ B) = {4,6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3938_393860


namespace NUMINAMATH_CALUDE_fraction_change_l3938_393819

theorem fraction_change (original_fraction : ℚ) 
  (numerator_increase : ℚ) (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  numerator_increase = 12/100 →
  new_fraction = 6/7 →
  (1 + numerator_increase) * original_fraction / (1 - denominator_decrease/100) = new_fraction →
  denominator_decrease = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l3938_393819


namespace NUMINAMATH_CALUDE_committee_formation_possibilities_l3938_393832

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of members in the club --/
def club_size : ℕ := 25

/-- The size of the executive committee --/
def committee_size : ℕ := 4

/-- Theorem stating that choosing 4 people from 25 results in 12650 possibilities --/
theorem committee_formation_possibilities :
  choose club_size committee_size = 12650 := by sorry

end NUMINAMATH_CALUDE_committee_formation_possibilities_l3938_393832


namespace NUMINAMATH_CALUDE_no_three_prime_roots_in_geometric_progression_l3938_393878

theorem no_three_prime_roots_in_geometric_progression :
  ¬∃ (p₁ p₂ p₃ : ℕ) (n₁ n₂ n₃ : ℤ) (a r : ℝ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ ∧
    n₁ ≠ n₂ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₃ ∧
    a > 0 ∧ r > 0 ∧
    a * r^n₁ = Real.sqrt p₁ ∧
    a * r^n₂ = Real.sqrt p₂ ∧
    a * r^n₃ = Real.sqrt p₃ :=
by sorry

end NUMINAMATH_CALUDE_no_three_prime_roots_in_geometric_progression_l3938_393878


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3938_393898

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3938_393898


namespace NUMINAMATH_CALUDE_equation_solutions_l3938_393807

theorem equation_solutions : 
  ∀ m n : ℕ, 20^m - 10*m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3938_393807


namespace NUMINAMATH_CALUDE_stratified_sampling_car_models_l3938_393831

/-- Represents the number of units to sample from a stratum in stratified sampling -/
def stratified_sample_size (stratum_size : ℕ) (total_population : ℕ) (total_sample : ℕ) : ℕ :=
  (stratum_size * total_sample) / total_population

/-- Theorem stating the correct sample sizes for the given problem -/
theorem stratified_sampling_car_models :
  let model1_size : ℕ := 1200
  let model2_size : ℕ := 6000
  let model3_size : ℕ := 2000
  let total_population : ℕ := model1_size + model2_size + model3_size
  let total_sample : ℕ := 46
  stratified_sample_size model1_size total_population total_sample = 6 ∧
  stratified_sample_size model2_size total_population total_sample = 30 ∧
  stratified_sample_size model3_size total_population total_sample = 10 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_car_models_l3938_393831


namespace NUMINAMATH_CALUDE_root_of_equation_l3938_393896

theorem root_of_equation (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  ∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_equation_l3938_393896


namespace NUMINAMATH_CALUDE_summer_course_duration_l3938_393854

/-- The number of days required for a summer course with the given conditions. -/
def summer_course_days (n k : ℕ) : ℕ :=
  (n.choose 2) / (k.choose 2)

/-- Theorem stating the number of days for the summer course. -/
theorem summer_course_duration :
  summer_course_days 15 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_summer_course_duration_l3938_393854


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l3938_393804

theorem consecutive_squares_sum (n : ℕ) (h : 2 * n + 1 = 144169^2) :
  ∃ (a : ℕ), a^2 + (a + 1)^2 = n + 1 :=
sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l3938_393804


namespace NUMINAMATH_CALUDE_average_students_is_fifty_l3938_393891

/-- Represents a teacher's teaching data over multiple years -/
structure TeacherData where
  total_years : Nat
  first_year_students : Nat
  total_students : Nat

/-- Calculates the average number of students taught per year, excluding the first year -/
def averageStudentsPerYear (data : TeacherData) : Nat :=
  (data.total_students - data.first_year_students) / (data.total_years - 1)

/-- Theorem stating that for the given conditions, the average number of students per year (excluding the first year) is 50 -/
theorem average_students_is_fifty :
  let data : TeacherData := {
    total_years := 10,
    first_year_students := 40,
    total_students := 490
  }
  averageStudentsPerYear data = 50 := by
  sorry

#eval averageStudentsPerYear {
  total_years := 10,
  first_year_students := 40,
  total_students := 490
}

end NUMINAMATH_CALUDE_average_students_is_fifty_l3938_393891


namespace NUMINAMATH_CALUDE_square_ratio_bounds_l3938_393808

theorem square_ratio_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ∃ (m M : ℝ), 
    (0 ≤ m) ∧ 
    (M ≤ 1) ∧ 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → m ≤ ((|z + w| / (|z| + |w|))^2) ∧ ((|z + w| / (|z| + |w|))^2) ≤ M) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ((|a + b| / (|a| + |b|))^2) = m) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ ((|c + d| / (|c| + |d|))^2) = M) ∧
    (M - m = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_bounds_l3938_393808


namespace NUMINAMATH_CALUDE_eighth_prime_is_19_l3938_393887

/-- Natural numbers are non-negative integers -/
def NaturalNumber (n : ℕ) : Prop := True

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d > 0 → d < p → p % d ≠ 0

/-- The nth prime number -/
def NthPrime (n : ℕ) : ℕ :=
  sorry

theorem eighth_prime_is_19 : NthPrime 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_eighth_prime_is_19_l3938_393887


namespace NUMINAMATH_CALUDE_peaches_in_one_basket_l3938_393855

/-- The number of peaches in a basket -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: The total number of peaches in 1 basket is 7 -/
theorem peaches_in_one_basket :
  total_peaches 4 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_peaches_in_one_basket_l3938_393855


namespace NUMINAMATH_CALUDE_refrigerator_price_calculation_l3938_393840

def refrigerator_purchase_price (labelled_price : ℝ) (discount_rate : ℝ) (additional_costs : ℝ) (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  (1 - discount_rate) * labelled_price + additional_costs

theorem refrigerator_price_calculation :
  let labelled_price : ℝ := 18400 / 1.15
  let discount_rate : ℝ := 0.20
  let additional_costs : ℝ := 125 + 250
  let selling_price : ℝ := 18400
  let profit_rate : ℝ := 0.15
  refrigerator_purchase_price labelled_price discount_rate additional_costs selling_price profit_rate = 13175 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_calculation_l3938_393840


namespace NUMINAMATH_CALUDE_positive_expression_l3938_393873

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3938_393873


namespace NUMINAMATH_CALUDE_set_equals_interval_l3938_393835

-- Define the set S as {x | x > 0 and x ≠ 2}
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 2}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ioo 0 2 ∪ Set.Ioi 2

-- Theorem stating the equivalence of the set and the interval representation
theorem set_equals_interval : S = intervalRep := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l3938_393835


namespace NUMINAMATH_CALUDE_original_calculation_result_l3938_393857

theorem original_calculation_result (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_calculation_result_l3938_393857


namespace NUMINAMATH_CALUDE_triangle_side_length_l3938_393834

theorem triangle_side_length (b c : ℝ) (cosA : ℝ) (h1 : b = 3) (h2 : c = 5) (h3 : cosA = -1/2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 - 2*b*c*cosA ∧ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3938_393834


namespace NUMINAMATH_CALUDE_sum_and_double_l3938_393853

theorem sum_and_double : (142 + 29 + 26 + 14) * 2 = 422 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l3938_393853


namespace NUMINAMATH_CALUDE_f_behavior_at_infinity_l3938_393890

def f (x : ℝ) := -3 * x^4 + 4 * x^2 + 5

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x < M) :=
sorry

end NUMINAMATH_CALUDE_f_behavior_at_infinity_l3938_393890


namespace NUMINAMATH_CALUDE_beef_order_weight_l3938_393801

def steak_weight : ℚ := 12
def num_steaks : ℕ := 20
def ounces_per_pound : ℕ := 16

theorem beef_order_weight :
  (steak_weight * num_steaks) / ounces_per_pound = 15 := by
  sorry

end NUMINAMATH_CALUDE_beef_order_weight_l3938_393801


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l3938_393892

theorem polygon_interior_exterior_angle_relation (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l3938_393892


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3938_393895

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - x + m) * (x - 8) = a * x^3 + b * x^2 + c) → m = -8 :=
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3938_393895


namespace NUMINAMATH_CALUDE_double_line_chart_capabilities_l3938_393827

/-- Represents a data set -/
structure DataSet where
  values : List ℝ

/-- Represents a double line chart -/
structure DoubleLineChart where
  dataset1 : DataSet
  dataset2 : DataSet

/-- Function to calculate changes in a dataset -/
def calculateChanges (ds : DataSet) : List ℝ := sorry

/-- Function to analyze differences between two datasets -/
def analyzeDifferences (ds1 ds2 : DataSet) : List ℝ := sorry

theorem double_line_chart_capabilities (dlc : DoubleLineChart) :
  (∃ (changes1 changes2 : List ℝ), 
     changes1 = calculateChanges dlc.dataset1 ∧ 
     changes2 = calculateChanges dlc.dataset2) ∧
  (∃ (differences : List ℝ), 
     differences = analyzeDifferences dlc.dataset1 dlc.dataset2) := by
  sorry

end NUMINAMATH_CALUDE_double_line_chart_capabilities_l3938_393827


namespace NUMINAMATH_CALUDE_root_in_interval_l3938_393856

theorem root_in_interval (a : ℤ) :
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3938_393856


namespace NUMINAMATH_CALUDE_train_speed_l3938_393825

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250) (h2 : time = 12) :
  ∃ (speed : ℝ), abs (speed - length / time) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3938_393825


namespace NUMINAMATH_CALUDE_square_of_105_l3938_393847

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l3938_393847


namespace NUMINAMATH_CALUDE_green_hats_count_l3938_393842

theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 :=
by sorry

end NUMINAMATH_CALUDE_green_hats_count_l3938_393842


namespace NUMINAMATH_CALUDE_multiple_without_zero_digit_l3938_393821

theorem multiple_without_zero_digit (n : ℕ) (hn : n % 10 ≠ 0) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ ∀ d : ℕ, d < 10 → (m / 10^d) % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_without_zero_digit_l3938_393821


namespace NUMINAMATH_CALUDE_negation_equivalence_l3938_393850

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3938_393850


namespace NUMINAMATH_CALUDE_blue_parrots_count_l3938_393813

/-- The number of blue parrots on Bird Island --/
def blue_parrots : ℕ := 38

/-- The total number of parrots on Bird Island after new arrivals --/
def total_parrots : ℕ := 150

/-- The fraction of red parrots --/
def red_fraction : ℚ := 1/2

/-- The fraction of green parrots --/
def green_fraction : ℚ := 1/4

/-- The number of new parrots that arrived --/
def new_parrots : ℕ := 30

theorem blue_parrots_count :
  blue_parrots = total_parrots - (red_fraction * total_parrots).floor - (green_fraction * total_parrots).floor :=
by sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l3938_393813


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3938_393869

theorem reciprocal_sum_of_quadratic_roots (α' β' : ℝ) : 
  (∃ x y : ℝ, 7 * x^2 + 4 * x + 9 = 0 ∧ 7 * y^2 + 4 * y + 9 = 0 ∧ α' = 1/x ∧ β' = 1/y) →
  α' + β' = -4/9 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3938_393869


namespace NUMINAMATH_CALUDE_trajectory_and_point_existence_l3938_393800

-- Define the plane and points
variable (x y : ℝ)
def F : ℝ × ℝ := (1, 0)
def S : ℝ × ℝ := (x, y)

-- Define the distance ratio condition
def distance_ratio (S : ℝ × ℝ) : Prop :=
  Real.sqrt ((S.1 - F.1)^2 + S.2^2) / |S.1 - 2| = Real.sqrt 2 / 2

-- Define the trajectory equation
def trajectory_equation (S : ℝ × ℝ) : Prop :=
  S.1^2 / 2 + S.2^2 = 1

-- Define the line l (not perpendicular to x-axis)
variable (k : ℝ)
def line_l (x : ℝ) : ℝ := k * (x - 1)

-- Define points P and Q on the intersection of line_l and trajectory
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define point M
variable (m : ℝ)
def M : ℝ × ℝ := (m, 0)

-- Define the dot product condition
def dot_product_condition (M P Q : ℝ × ℝ) : Prop :=
  let MP := (P.1 - M.1, P.2 - M.2)
  let MQ := (Q.1 - M.1, Q.2 - M.2)
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  (MP.1 + MQ.1) * PQ.1 + (MP.2 + MQ.2) * PQ.2 = 0

-- Main theorem
theorem trajectory_and_point_existence :
  ∀ S, distance_ratio S →
    (trajectory_equation S ∧
     ∃ m, 0 ≤ m ∧ m < 1/2 ∧
       ∀ k ≠ 0, dot_product_condition (M m) P Q) := by sorry

end NUMINAMATH_CALUDE_trajectory_and_point_existence_l3938_393800


namespace NUMINAMATH_CALUDE_area_of_three_semicircle_intersection_l3938_393888

/-- The area of intersection of three semicircles forming a square -/
theorem area_of_three_semicircle_intersection (r : ℝ) (h : r = 2) : 
  let square_side := 2 * r
  let square_area := square_side ^ 2
  square_area = 16 := by sorry

end NUMINAMATH_CALUDE_area_of_three_semicircle_intersection_l3938_393888


namespace NUMINAMATH_CALUDE_power_of_81_l3938_393820

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l3938_393820


namespace NUMINAMATH_CALUDE_product_equals_fraction_l3938_393811

/-- The repeating decimal 0.1357̄ as a rational number -/
def s : ℚ := 1357 / 9999

/-- The product of 0.1357̄ and 7 -/
def product : ℚ := 7 * s

theorem product_equals_fraction : product = 9499 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l3938_393811


namespace NUMINAMATH_CALUDE_remainder_mod_six_l3938_393868

theorem remainder_mod_six (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_six_l3938_393868


namespace NUMINAMATH_CALUDE_max_y_value_l3938_393871

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : x * y * (x + y) = x - y) : 
  y ≤ 1/3 ∧ ∃ (y0 : ℝ), y0 * x * (x + y0) = x - y0 ∧ y0 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3938_393871


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_arc_angles_l3938_393881

theorem circumscribed_quadrilateral_arc_angles (a b c d : ℝ) :
  let x := (b + c + d) / 2
  let y := (a + c + d) / 2
  let z := (a + b + d) / 2
  let t := (a + b + c) / 2
  a + b + c + d = 360 →
  x + y + z + t = 540 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_arc_angles_l3938_393881


namespace NUMINAMATH_CALUDE_binary_5_is_5_l3938_393862

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_5 : List Bool := [true, false, true]

theorem binary_5_is_5 : binary_to_decimal binary_5 = 5 := by sorry

end NUMINAMATH_CALUDE_binary_5_is_5_l3938_393862


namespace NUMINAMATH_CALUDE_flower_purchase_analysis_l3938_393874

/-- Represents the number and cost of different flower types --/
structure FlowerPurchase where
  roses : ℕ
  lilies : ℕ
  sunflowers : ℕ
  daisies : ℕ
  rose_cost : ℚ
  lily_cost : ℚ
  sunflower_cost : ℚ
  daisy_cost : ℚ

/-- Calculates the total cost of the flower purchase --/
def total_cost (purchase : FlowerPurchase) : ℚ :=
  purchase.roses * purchase.rose_cost +
  purchase.lilies * purchase.lily_cost +
  purchase.sunflowers * purchase.sunflower_cost +
  purchase.daisies * purchase.daisy_cost

/-- Calculates the total number of flowers --/
def total_flowers (purchase : FlowerPurchase) : ℕ :=
  purchase.roses + purchase.lilies + purchase.sunflowers + purchase.daisies

/-- Calculates the percentage of a specific flower type --/
def flower_percentage (count : ℕ) (total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

/-- Theorem stating the total cost and percentages of flowers --/
theorem flower_purchase_analysis (purchase : FlowerPurchase)
  (h1 : purchase.roses = 50)
  (h2 : purchase.lilies = 40)
  (h3 : purchase.sunflowers = 30)
  (h4 : purchase.daisies = 20)
  (h5 : purchase.rose_cost = 2)
  (h6 : purchase.lily_cost = 3/2)
  (h7 : purchase.sunflower_cost = 1)
  (h8 : purchase.daisy_cost = 3/4) :
  total_cost purchase = 205 ∧
  flower_percentage purchase.roses (total_flowers purchase) = 35.71 ∧
  flower_percentage purchase.lilies (total_flowers purchase) = 28.57 ∧
  flower_percentage purchase.sunflowers (total_flowers purchase) = 21.43 ∧
  flower_percentage purchase.daisies (total_flowers purchase) = 14.29 := by
  sorry

end NUMINAMATH_CALUDE_flower_purchase_analysis_l3938_393874


namespace NUMINAMATH_CALUDE_vector_sum_equals_expected_l3938_393879

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-3, 4]

-- Define the sum of the vectors
def sum_ab : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]

-- Theorem statement
theorem vector_sum_equals_expected : sum_ab = ![-1, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equals_expected_l3938_393879


namespace NUMINAMATH_CALUDE_main_diagonal_equals_anti_diagonal_l3938_393805

/-- Represents a square board with side length 2^n -/
structure Board (n : ℕ) where
  size : ℕ := 2^n
  elements : Fin (size * size) → ℕ

/-- Defines the initial arrangement of numbers on the board -/
def initial_board (n : ℕ) : Board n where
  elements := λ i => i.val + 1

/-- Defines the anti-diagonal of a board -/
def anti_diagonal (b : Board n) : List ℕ :=
  List.range b.size |>.map (λ i => b.elements ⟨i + (b.size - 1 - i) * b.size, sorry⟩)

/-- Represents a transformation on the board -/
def transform (b : Board n) : Board n :=
  sorry

/-- Theorem: After transformations, the main diagonal equals the original anti-diagonal -/
theorem main_diagonal_equals_anti_diagonal (n : ℕ) :
  let final_board := (transform^[n] (initial_board n))
  List.range (2^n) |>.map (λ i => final_board.elements ⟨i + i * (2^n), sorry⟩) =
  anti_diagonal (initial_board n) := by
  sorry

end NUMINAMATH_CALUDE_main_diagonal_equals_anti_diagonal_l3938_393805


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3938_393836

/-- Given two vectors a and b in a plane with an angle of 30° between them,
    |a| = √3, and |b| = 2, prove that |a + 2b| = √31 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 30 * π / 180
  (norm a = Real.sqrt 3) →
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = norm a * norm b * Real.cos angle) →
  norm (a + 2 • b) = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3938_393836


namespace NUMINAMATH_CALUDE_expand_expression_l3938_393826

theorem expand_expression (x : ℝ) : (17*x + 18 + 5)*3*x = 51*x^2 + 69*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3938_393826


namespace NUMINAMATH_CALUDE_triangle_lattice_points_l3938_393884

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (t : Triangle) : ℚ := sorry

/-- Counts the number of lattice points on the boundary of a triangle -/
def boundaryPoints (t : Triangle) : ℕ := sorry

/-- Counts the total number of lattice points in and on a triangle -/
def totalLatticePoints (t : Triangle) : ℕ := sorry

theorem triangle_lattice_points :
  ∀ t : Triangle,
    t.p1 = ⟨5, 0⟩ →
    t.p2 = ⟨25, 0⟩ →
    triangleArea t = 200 →
    totalLatticePoints t = 221 := by
  sorry

end NUMINAMATH_CALUDE_triangle_lattice_points_l3938_393884


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3938_393870

theorem repeating_decimal_to_fraction :
  ∃ (y : ℚ), y = 0.37 + (46 / 99) / 100 ∧ y = 3709 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3938_393870


namespace NUMINAMATH_CALUDE_physics_chemistry_average_l3938_393843

theorem physics_chemistry_average (physics chemistry math : ℝ) 
  (h1 : (physics + chemistry + math) / 3 = 80)
  (h2 : (physics + math) / 2 = 90)
  (h3 : physics = 80) :
  (physics + chemistry) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_physics_chemistry_average_l3938_393843


namespace NUMINAMATH_CALUDE_number_of_tests_l3938_393889

/-- Proves the number of initial tests given average scores and lowest score -/
theorem number_of_tests
  (initial_average : ℝ)
  (lowest_score : ℝ)
  (new_average : ℝ)
  (h1 : initial_average = 90)
  (h2 : lowest_score = 75)
  (h3 : new_average = 95)
  (h4 : ∀ n : ℕ, n > 1 → (n * initial_average - lowest_score) / (n - 1) = new_average) :
  ∃ n : ℕ, n = 4 ∧ n > 1 := by
  sorry


end NUMINAMATH_CALUDE_number_of_tests_l3938_393889


namespace NUMINAMATH_CALUDE_pedal_triangles_existence_and_angles_l3938_393880

/-- A triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The pedal triangle of a given triangle -/
structure PedalTriangle where
  original : Triangle
  pedal : Triangle

/-- The theorem statement -/
theorem pedal_triangles_existence_and_angles 
  (T : Triangle) 
  (h1 : T.angle1 = 24) 
  (h2 : T.angle2 = 60) 
  (h3 : T.angle3 = 96) : 
  ∃! (pedals : Finset PedalTriangle), 
    Finset.card pedals = 4 ∧ 
    ∀ P ∈ pedals, 
      (P.pedal.angle1 = 102 ∧ 
       P.pedal.angle2 = 30 ∧ 
       P.pedal.angle3 = 48) := by
  sorry


end NUMINAMATH_CALUDE_pedal_triangles_existence_and_angles_l3938_393880


namespace NUMINAMATH_CALUDE_inequalities_hold_l3938_393894

-- Define the points and lengths
variable (A B C D : ℝ) -- Representing points as real numbers for simplicity
variable (x y z : ℝ)

-- Define the conditions
axiom distinct_points : A < B ∧ B < C ∧ C < D
axiom length_AB : x = B - A
axiom length_AC : y = C - A
axiom length_AD : z = D - A
axiom positive_area : x > 0 ∧ (y - x) > 0 ∧ (z - y) > 0

-- Define the triangle inequality conditions
axiom triangle_inequality1 : x + (y - x) > z - y
axiom triangle_inequality2 : (y - x) + (z - y) > x
axiom triangle_inequality3 : x + (z - y) > y - x

-- State the theorem to be proved
theorem inequalities_hold : x < z / 2 ∧ y < x + z / 2 := by sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3938_393894


namespace NUMINAMATH_CALUDE_school_dinner_theatre_tickets_l3938_393882

theorem school_dinner_theatre_tickets (child_price adult_price total_tickets total_revenue : ℕ) 
  (h1 : child_price = 6)
  (h2 : adult_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (children adults : ℕ),
    children + adults = total_tickets ∧
    child_price * children + adult_price * adults = total_revenue ∧
    children = 50 := by
  sorry

end NUMINAMATH_CALUDE_school_dinner_theatre_tickets_l3938_393882
