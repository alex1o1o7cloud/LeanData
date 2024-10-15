import Mathlib

namespace NUMINAMATH_CALUDE_distinct_integer_roots_l2871_287126

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  (a = 4.5 ∨ a = 1 ∨ a = -9 ∨ a = -12.5) :=
sorry

end NUMINAMATH_CALUDE_distinct_integer_roots_l2871_287126


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l2871_287123

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l2871_287123


namespace NUMINAMATH_CALUDE_mooney_ate_four_brownies_l2871_287134

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The number of brownies Mother initially made -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mother made the next morning -/
def new_batch : ℕ := 2 * dozen

/-- The total number of brownies after adding the new batch -/
def final_count : ℕ := 36

/-- Theorem: Mooney ate 4 brownies -/
theorem mooney_ate_four_brownies :
  initial_brownies - father_ate - (final_count - new_batch) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mooney_ate_four_brownies_l2871_287134


namespace NUMINAMATH_CALUDE_salary_percentage_is_120_percent_l2871_287186

/-- The percentage of one employee's salary compared to another -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  ((total_salary - n_salary) / n_salary) * 100

/-- Proof that the salary percentage is 120% given the conditions -/
theorem salary_percentage_is_120_percent 
  (total_salary : ℚ) 
  (n_salary : ℚ) 
  (h1 : total_salary = 594) 
  (h2 : n_salary = 270) : 
  salary_percentage total_salary n_salary = 120 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_is_120_percent_l2871_287186


namespace NUMINAMATH_CALUDE_completing_square_quadratic_equation_l2871_287155

theorem completing_square_quadratic_equation :
  ∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_equation_l2871_287155


namespace NUMINAMATH_CALUDE_geometric_sequence_178th_term_l2871_287169

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_178th_term :
  let a₁ := 5
  let a₂ := -20
  let r := a₂ / a₁
  geometric_sequence a₁ r 178 = -5 * 4^177 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_178th_term_l2871_287169


namespace NUMINAMATH_CALUDE_gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l2871_287108

/-- The nth triangular number -/
def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

/-- Theorem: The greatest common divisor of 6Tn and n+1 is at most 3 -/
theorem gcd_6Tn_nplus1_le_3 (n : ℕ+) :
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

/-- Theorem: There exists an n such that the greatest common divisor of 6Tn and n+1 is exactly 3 -/
theorem exists_gcd_6Tn_nplus1_eq_3 :
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l2871_287108


namespace NUMINAMATH_CALUDE_two_tails_one_head_prob_l2871_287122

/-- Represents a biased coin with probabilities for heads and tails -/
structure BiasedCoin where
  probHeads : ℝ
  probTails : ℝ
  prob_sum : probHeads + probTails = 1

/-- Calculates the probability of getting exactly two tails followed by one head within 5 flips -/
def prob_two_tails_one_head (c : BiasedCoin) : ℝ :=
  3 * (c.probTails * c.probTails * c.probTails * c.probHeads)

/-- The main theorem to be proved -/
theorem two_tails_one_head_prob :
  let c : BiasedCoin := ⟨0.3, 0.7, by norm_num⟩
  prob_two_tails_one_head c = 0.3087 := by
  sorry


end NUMINAMATH_CALUDE_two_tails_one_head_prob_l2871_287122


namespace NUMINAMATH_CALUDE_trapezoid_area_is_correct_l2871_287100

/-- The area of a trapezoid bounded by y = x, y = 10, y = 5, and the y-axis -/
def trapezoidArea : ℝ := 37.5

/-- The line y = x -/
def lineYeqX (x : ℝ) : ℝ := x

/-- The line y = 10 -/
def lineY10 (x : ℝ) : ℝ := 10

/-- The line y = 5 -/
def lineY5 (x : ℝ) : ℝ := 5

/-- The y-axis (x = 0) -/
def yAxis : Set ℝ := {x | x = 0}

theorem trapezoid_area_is_correct :
  trapezoidArea = 37.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_correct_l2871_287100


namespace NUMINAMATH_CALUDE_speed_difference_20_l2871_287165

/-- The speed equation for a subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem stating that the speed difference between 5 and 3 seconds is 20 km/h -/
theorem speed_difference_20 : speed 5 - speed 3 = 20 := by sorry

end NUMINAMATH_CALUDE_speed_difference_20_l2871_287165


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2871_287183

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2871_287183


namespace NUMINAMATH_CALUDE_product_of_three_integers_l2871_287173

theorem product_of_three_integers (A B C : ℤ) 
  (sum_eq : A + B + C = 33)
  (largest_eq : C = 3 * B)
  (smallest_eq : A = C - 23) :
  A * B * C = 192 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l2871_287173


namespace NUMINAMATH_CALUDE_motel_total_rent_l2871_287103

/-- Represents the total rent charged by a motel on a Saturday night. -/
def total_rent (r40 r60 : ℕ) : ℕ := 40 * r40 + 60 * r60

/-- The condition that changing 10 rooms from $60 to $40 reduces the total rent by 50%. -/
def rent_reduction_condition (r40 r60 : ℕ) : Prop :=
  total_rent (r40 + 10) (r60 - 10) = (total_rent r40 r60) / 2

/-- The theorem stating that the total rent charged by the motel is $800. -/
theorem motel_total_rent :
  ∃ (r40 r60 : ℕ), rent_reduction_condition r40 r60 ∧ total_rent r40 r60 = 800 :=
sorry

end NUMINAMATH_CALUDE_motel_total_rent_l2871_287103


namespace NUMINAMATH_CALUDE_family_race_problem_l2871_287156

/-- Represents the driving data for the family race -/
structure DrivingData where
  cory_time : ℝ
  cory_speed : ℝ
  mira_time : ℝ
  mira_speed : ℝ
  tia_time : ℝ
  tia_speed : ℝ

/-- The theorem statement for the family race problem -/
theorem family_race_problem (data : DrivingData) 
  (h1 : data.mira_time = data.cory_time + 3)
  (h2 : data.mira_speed = data.cory_speed + 8)
  (h3 : data.mira_speed * data.mira_time = data.cory_speed * data.cory_time + 120)
  (h4 : data.tia_time = data.cory_time + 4)
  (h5 : data.tia_speed = data.cory_speed + 12) :
  data.tia_speed * data.tia_time - data.cory_speed * data.cory_time = 192 := by
  sorry

end NUMINAMATH_CALUDE_family_race_problem_l2871_287156


namespace NUMINAMATH_CALUDE_golden_rectangle_perimeter_l2871_287136

/-- A golden rectangle is a rectangle where the ratio of its width to its length is (√5 - 1) / 2 -/
def is_golden_rectangle (width length : ℝ) : Prop :=
  width / length = (Real.sqrt 5 - 1) / 2

/-- The perimeter of a rectangle given its width and length -/
def rectangle_perimeter (width length : ℝ) : ℝ :=
  2 * (width + length)

theorem golden_rectangle_perimeter :
  ∀ width length : ℝ,
  is_golden_rectangle width length →
  (width = Real.sqrt 5 - 1 ∨ length = Real.sqrt 5 - 1) →
  rectangle_perimeter width length = 4 ∨ rectangle_perimeter width length = 2 * Real.sqrt 5 + 2 :=
by sorry

end NUMINAMATH_CALUDE_golden_rectangle_perimeter_l2871_287136


namespace NUMINAMATH_CALUDE_yogurt_milk_calculation_l2871_287118

/-- The cost of milk per liter in dollars -/
def milk_cost : ℚ := 3/2

/-- The cost of fruit per kilogram in dollars -/
def fruit_cost : ℚ := 2

/-- The amount of fruit needed for one batch of yogurt in kilograms -/
def fruit_per_batch : ℚ := 3

/-- The total cost to produce three batches of yogurt in dollars -/
def total_cost_three_batches : ℚ := 63

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℚ := 10

theorem yogurt_milk_calculation :
  milk_per_batch * milk_cost * 3 + fruit_per_batch * fruit_cost * 3 = total_cost_three_batches :=
sorry

end NUMINAMATH_CALUDE_yogurt_milk_calculation_l2871_287118


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2871_287124

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 160)
  (h3 : correct_answers = 44)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2871_287124


namespace NUMINAMATH_CALUDE_derivative_equals_negative_function_l2871_287147

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- State the theorem
theorem derivative_equals_negative_function (x₀ : ℝ) :
  x₀ ≠ 0 → -- Ensure x₀ is not zero to avoid division by zero
  (deriv f) x₀ = -f x₀ →
  x₀ = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_derivative_equals_negative_function_l2871_287147


namespace NUMINAMATH_CALUDE_speed_conversion_proof_l2871_287182

/-- Converts a speed from meters per second to kilometers per hour. -/
def convert_mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

/-- Proves that converting 17/36 m/s to km/h results in 1.7 km/h. -/
theorem speed_conversion_proof :
  convert_mps_to_kmh (17/36) = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_proof_l2871_287182


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2871_287189

theorem integer_solutions_count (m : ℤ) : 
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x ∈ s, x - m < 0 ∧ 5 - 2*x ≤ 1) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2871_287189


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l2871_287107

theorem sum_of_quadratic_roots (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 0) → 
  (∃ r s : ℝ, (2 * r^2 - 8 * r - 10 = 0) ∧ 
              (2 * s^2 - 8 * s - 10 = 0) ∧ 
              (r + s = 4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l2871_287107


namespace NUMINAMATH_CALUDE_fraction_equality_l2871_287172

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) :
  (x + y) / (x - y) = -1001 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2871_287172


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2871_287120

theorem decimal_to_fraction (n d : ℕ) (h : n = 16) :
  (n : ℚ) / d = 32 / 100 → d = 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2871_287120


namespace NUMINAMATH_CALUDE_katie_marbles_l2871_287125

/-- The number of pink marbles Katie has -/
def pink_marbles : ℕ := 13

/-- The number of orange marbles Katie has -/
def orange_marbles : ℕ := pink_marbles - 9

/-- The number of purple marbles Katie has -/
def purple_marbles : ℕ := 4 * orange_marbles

/-- The total number of marbles Katie has -/
def total_marbles : ℕ := 33

theorem katie_marbles : 
  pink_marbles + orange_marbles + purple_marbles = total_marbles ∧ 
  orange_marbles = pink_marbles - 9 ∧
  purple_marbles = 4 * orange_marbles ∧
  pink_marbles = 13 := by sorry

end NUMINAMATH_CALUDE_katie_marbles_l2871_287125


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2871_287105

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2871_287105


namespace NUMINAMATH_CALUDE_all_graphs_different_l2871_287131

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x - 1
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x + 1)
def eq3 (x y : ℝ) : Prop := (x + 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq_a eq_b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq_a x y ↔ eq_b x y

-- Theorem stating that all equations have different graphs
theorem all_graphs_different :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l2871_287131


namespace NUMINAMATH_CALUDE_peter_total_spending_l2871_287197

/-- The cost of one shirt in dollars -/
def shirt_cost : ℚ := 10

/-- The cost of one pair of pants in dollars -/
def pants_cost : ℚ := 6

/-- The number of shirts Peter bought -/
def peter_shirts : ℕ := 5

/-- The number of pairs of pants Peter bought -/
def peter_pants : ℕ := 2

/-- The number of shirts Jessica bought -/
def jessica_shirts : ℕ := 2

/-- The total cost of Jessica's purchase in dollars -/
def jessica_total : ℚ := 20

theorem peter_total_spending :
  peter_shirts * shirt_cost + peter_pants * pants_cost = 62 :=
sorry

end NUMINAMATH_CALUDE_peter_total_spending_l2871_287197


namespace NUMINAMATH_CALUDE_binomial_15_12_times_3_l2871_287157

theorem binomial_15_12_times_3 : 3 * (Nat.choose 15 12) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_times_3_l2871_287157


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2871_287119

/-- A quadratic function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a * x + b

/-- The composition of f with itself -/
def f_comp (a b x : ℝ) : ℝ := f a b (f a b x)

/-- Theorem: If f(f(x)) = 0 has four distinct real solutions and
    the sum of two of these solutions is -1, then b ≤ -1/4 -/
theorem quadratic_inequality (a b : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f_comp a b w = 0 ∧ f_comp a b x = 0 ∧ f_comp a b y = 0 ∧ f_comp a b z = 0) →
  (∃ p q : ℝ, f_comp a b p = 0 ∧ f_comp a b q = 0 ∧ p + q = -1) →
  b ≤ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2871_287119


namespace NUMINAMATH_CALUDE_zero_in_interval_l2871_287102

noncomputable def f (x : ℝ) : ℝ := 2^x - 6 - Real.log x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2871_287102


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2871_287143

/-- An arithmetic sequence with first term 2 and last term 3 -/
def is_arithmetic_sequence (x y : ℝ) : Prop :=
  x - 2 = 3 - y ∧ y - x = 3 - y

/-- A geometric sequence with first term 2 and last term 3 -/
def is_geometric_sequence (m n : ℝ) : Prop :=
  m / 2 = 3 / n ∧ n / m = 3 / n

theorem arithmetic_geometric_sum (x y m n : ℝ) 
  (h1 : is_arithmetic_sequence x y) 
  (h2 : is_geometric_sequence m n) : 
  x + y + m * n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2871_287143


namespace NUMINAMATH_CALUDE_factory_produces_4000_candies_l2871_287160

/-- Represents a candy factory with its production rate and work schedule. -/
structure CandyFactory where
  production_rate : ℕ  -- candies per hour
  work_hours_per_day : ℕ
  work_days : ℕ

/-- Calculates the total number of candies produced by a factory. -/
def total_candies_produced (factory : CandyFactory) : ℕ :=
  factory.production_rate * factory.work_hours_per_day * factory.work_days

/-- Theorem stating that a factory with the given parameters produces 4000 candies. -/
theorem factory_produces_4000_candies 
  (factory : CandyFactory) 
  (h1 : factory.production_rate = 50)
  (h2 : factory.work_hours_per_day = 10)
  (h3 : factory.work_days = 8) : 
  total_candies_produced factory = 4000 := by
  sorry

#eval total_candies_produced { production_rate := 50, work_hours_per_day := 10, work_days := 8 }

end NUMINAMATH_CALUDE_factory_produces_4000_candies_l2871_287160


namespace NUMINAMATH_CALUDE_expand_polynomial_l2871_287142

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2871_287142


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l2871_287140

theorem rectangle_to_square_cut (width : ℕ) (height : ℕ) : 
  width = 4 ∧ height = 9 → 
  ∃ (s : ℕ) (w1 w2 h1 h2 : ℕ),
    s * s = width * height ∧
    w1 + w2 = width ∧
    h1 = height ∧ h2 = height ∧
    (w1 * h1 + w2 * h2 = s * s) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l2871_287140


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l2871_287192

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l2871_287192


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2871_287111

/-- Given a quadratic equation x^2 - 2mx + 4 = 0 where m is a real number,
    if both of its real roots are greater than 1, then m is in the range [2, 5/2). -/
theorem quadratic_roots_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 4 = 0 → x > 1) → 
  m ∈ Set.Icc 2 (5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2871_287111


namespace NUMINAMATH_CALUDE_shell_difference_l2871_287185

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) :
  broken_shells / 2 - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end NUMINAMATH_CALUDE_shell_difference_l2871_287185


namespace NUMINAMATH_CALUDE_expression_evaluation_l2871_287178

theorem expression_evaluation :
  (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2871_287178


namespace NUMINAMATH_CALUDE_h_of_negative_one_l2871_287171

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x)^2 - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_negative_one : h (-1) = 298 := by
  sorry

end NUMINAMATH_CALUDE_h_of_negative_one_l2871_287171


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2871_287176

/-- Given a quadrilateral ABED where ABE and BED are right triangles sharing base BE,
    with AB = 15, BE = 20, and ED = 25, prove that the area of ABED is 400. -/
theorem area_of_quadrilateral (A B E D : ℝ × ℝ) : 
  let triangle_area (a b : ℝ) := (a * b) / 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15^2 →  -- AB = 15
  (B.1 - E.1)^2 + (B.2 - E.2)^2 = 20^2 →  -- BE = 20
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = 25^2 →  -- ED = 25
  (A.1 - E.1) * (B.2 - E.2) = (A.2 - E.2) * (B.1 - E.1) →  -- ABE is right-angled at B
  (B.1 - E.1) * (D.2 - E.2) = (B.2 - E.2) * (D.1 - E.1) →  -- BED is right-angled at E
  triangle_area 15 20 + triangle_area 20 25 = 400 := by
    sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2871_287176


namespace NUMINAMATH_CALUDE_exponent_simplification_l2871_287181

theorem exponent_simplification :
  (5^6 * 5^9 * 5) / 5^3 = 5^13 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2871_287181


namespace NUMINAMATH_CALUDE_unique_sum_of_three_squares_l2871_287112

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_squares (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧ a + b + c = 100

def distinct_combinations (a b c : ℕ) : Prop :=
  sum_of_three_squares a b c ∧ 
  (a ≤ b ∧ b ≤ c)

theorem unique_sum_of_three_squares : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_combinations abc.1 abc.2.1 abc.2.2 :=
sorry

end NUMINAMATH_CALUDE_unique_sum_of_three_squares_l2871_287112


namespace NUMINAMATH_CALUDE_reggie_bought_five_books_l2871_287198

/-- The number of books Reggie bought -/
def number_of_books (initial_amount remaining_amount cost_per_book : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / cost_per_book

/-- Theorem: Reggie bought 5 books -/
theorem reggie_bought_five_books :
  number_of_books 48 38 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_reggie_bought_five_books_l2871_287198


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2871_287164

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2871_287164


namespace NUMINAMATH_CALUDE_jessie_current_weight_l2871_287158

def jessie_weight_problem (initial_weight lost_weight : ℕ) : Prop :=
  initial_weight = 69 ∧ lost_weight = 35 →
  initial_weight - lost_weight = 34

theorem jessie_current_weight : jessie_weight_problem 69 35 := by
  sorry

end NUMINAMATH_CALUDE_jessie_current_weight_l2871_287158


namespace NUMINAMATH_CALUDE_mountain_dew_to_coke_ratio_l2871_287190

/-- Represents the composition of a drink -/
structure DrinkComposition where
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ

/-- Proves that the ratio of Mountain Dew to Coke is 3:2 given the conditions -/
theorem mountain_dew_to_coke_ratio 
  (drink : DrinkComposition)
  (coke_sprite_ratio : drink.coke = 2 * drink.sprite)
  (coke_amount : drink.coke = 6)
  (total_amount : drink.coke + drink.sprite + drink.mountainDew = 18) :
  drink.mountainDew / drink.coke = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mountain_dew_to_coke_ratio_l2871_287190


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2871_287110

theorem consecutive_integers_sum (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2871_287110


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l2871_287148

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (lst : List Nat) : ℚ :=
  (lst.map (λ x => (1 : ℚ) / x)).sum / lst.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l2871_287148


namespace NUMINAMATH_CALUDE_angle_C_is_30_l2871_287121

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.A + t.B + t.C = 180

-- Theorem: If the sum of angles A and B in triangle ABC is 150°, then angle C is 30°
theorem angle_C_is_30 (t : Triangle) (h : t.A + t.B = 150) : t.C = 30 := by
  sorry


end NUMINAMATH_CALUDE_angle_C_is_30_l2871_287121


namespace NUMINAMATH_CALUDE_tammy_running_schedule_l2871_287152

/-- Calculates the number of loops per day given weekly distance goal, track length, and days per week -/
def loops_per_day (weekly_goal : ℕ) (track_length : ℕ) (days_per_week : ℕ) : ℕ :=
  (weekly_goal / track_length) / days_per_week

theorem tammy_running_schedule :
  loops_per_day 3500 50 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tammy_running_schedule_l2871_287152


namespace NUMINAMATH_CALUDE_disinfectant_sales_problem_l2871_287153

-- Define the range of x
def valid_x (x : ℤ) : Prop := 8 ≤ x ∧ x ≤ 15

-- Define the linear function
def y (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def w (x : ℤ) : ℤ := (x - 8) * (-5 * x + 150)

theorem disinfectant_sales_problem :
  (∀ x : ℤ, valid_x x → 
    (x = 9 → y x = 105) ∧ 
    (x = 11 → y x = 95) ∧ 
    (x = 13 → y x = 85)) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 425 ∧ x = 13) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 525 ∧ x = 15 ∧ ∀ x' : ℤ, valid_x x' → w x' ≤ w x) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_sales_problem_l2871_287153


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_division_l2871_287187

/-- Given a triangle with sides a, b, and c, where c is the longest side,
    and a circle touching sides a and b with its center on side c,
    prove that the center divides c into segments of length x and y. -/
theorem circle_tangent_triangle_division (a b c x y : ℝ) : 
  a = 12 → b = 15 → c = 18 → c > a ∧ c > b →
  x + y = c → x / y = a / b →
  x = 8 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_division_l2871_287187


namespace NUMINAMATH_CALUDE_max_value_implies_a_values_l2871_287149

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- State the theorem
theorem max_value_implies_a_values (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = 2 ∨ a = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_values_l2871_287149


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2871_287199

theorem cubic_equation_root (h : ℝ) : 
  (2 : ℝ)^3 + h * 2 + 10 = 0 → h = -9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2871_287199


namespace NUMINAMATH_CALUDE_depak_money_problem_l2871_287150

theorem depak_money_problem :
  ∀ x : ℕ, 
    (x + 1) % 6 = 0 ∧ 
    x % 6 ≠ 0 ∧
    ∀ y : ℕ, y > x → (y + 1) % 6 ≠ 0 ∨ y % 6 = 0
    → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_depak_money_problem_l2871_287150


namespace NUMINAMATH_CALUDE_brownie_count_l2871_287115

def tray_length : ℕ := 24
def tray_width : ℕ := 15
def brownie_side : ℕ := 3

theorem brownie_count : 
  (tray_length * tray_width) / (brownie_side * brownie_side) = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_count_l2871_287115


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2871_287179

theorem sqrt_equation_solution :
  ∃ t : ℝ, t = 3.7 ∧ Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2871_287179


namespace NUMINAMATH_CALUDE_S_equality_l2871_287144

/-- S_k(n) function (not defined, assumed to exist) -/
noncomputable def S_k (k n : ℕ) : ℕ := sorry

/-- The sum S as defined in the problem -/
noncomputable def S (n k : ℕ) : ℚ :=
  (Finset.range ((k + 1) / 2)).sum (λ i =>
    Nat.choose (k + 1) (2 * i + 1) * S_k (k - 2 * i) n)

/-- Theorem stating the equality to be proved -/
theorem S_equality (n k : ℕ) :
  S n k = ((n + 1)^(k + 1) + n^(k + 1) - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_S_equality_l2871_287144


namespace NUMINAMATH_CALUDE_sphere_radius_equals_eight_l2871_287162

-- Define constants for the cylinder dimensions
def cylinder_height : ℝ := 16
def cylinder_diameter : ℝ := 16

-- Define the theorem
theorem sphere_radius_equals_eight :
  ∀ r : ℝ,
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 8 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_equals_eight_l2871_287162


namespace NUMINAMATH_CALUDE_power_of_power_three_l2871_287159

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2871_287159


namespace NUMINAMATH_CALUDE_f_greater_than_three_f_inequality_solution_range_l2871_287130

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Theorem 1: f(x) > 3 iff x > 0
theorem f_greater_than_three (x : ℝ) : f x > 3 ↔ x > 0 := by sorry

-- Theorem 2: f(x) + 1 ≤ 4^a - 5×2^a has a solution iff a ∈ (-∞,0] ∪ [2,+∞)
theorem f_inequality_solution_range (a : ℝ) : 
  (∃ x, f x + 1 ≤ 4^a - 5*2^a) ↔ (a ≤ 0 ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_f_greater_than_three_f_inequality_solution_range_l2871_287130


namespace NUMINAMATH_CALUDE_saucers_per_pitcher_l2871_287167

/-- The weight of a cup -/
def cup_weight : ℝ := sorry

/-- The weight of a pitcher -/
def pitcher_weight : ℝ := sorry

/-- The weight of a saucer -/
def saucer_weight : ℝ := sorry

/-- Two cups and two pitchers weigh the same as 14 saucers -/
axiom weight_equation : 2 * cup_weight + 2 * pitcher_weight = 14 * saucer_weight

/-- One pitcher weighs the same as one cup and one saucer -/
axiom pitcher_cup_saucer : pitcher_weight = cup_weight + saucer_weight

/-- The number of saucers that balance with a pitcher is 4 -/
theorem saucers_per_pitcher : pitcher_weight = 4 * saucer_weight := by sorry

end NUMINAMATH_CALUDE_saucers_per_pitcher_l2871_287167


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2871_287139

def complex_condition (z : ℂ) : Prop := z * (1 + Complex.I) = 2 * Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : complex_condition z) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2871_287139


namespace NUMINAMATH_CALUDE_womens_average_age_l2871_287127

theorem womens_average_age 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (age1 age2 : ℕ) 
  (new_avg_increase : ℝ) :
  n = 8 →
  age1 = 20 →
  age2 = 28 →
  new_avg_increase = 2 →
  (n * initial_avg - (age1 + age2 : ℝ) + 2 * ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2)) / n = initial_avg + new_avg_increase →
  ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2) / 2 = 32 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l2871_287127


namespace NUMINAMATH_CALUDE_incandescent_bulbs_count_l2871_287184

/-- Represents the waterfall system and power consumption --/
structure WaterfallSystem where
  water_flow : ℝ  -- m³/s
  waterfall_height : ℝ  -- m
  turbine_efficiency : ℝ
  dynamo_efficiency : ℝ
  transmission_efficiency : ℝ
  num_motors : ℕ
  power_per_motor : ℝ  -- horsepower
  motor_efficiency : ℝ
  num_arc_lamps : ℕ
  arc_lamp_voltage : ℝ  -- V
  arc_lamp_current : ℝ  -- A
  incandescent_bulb_power : ℝ  -- W

/-- Calculates the number of incandescent bulbs that can be powered --/
def calculate_incandescent_bulbs (system : WaterfallSystem) : ℕ :=
  sorry

/-- Theorem stating the number of incandescent bulbs that can be powered --/
theorem incandescent_bulbs_count (system : WaterfallSystem) 
  (h1 : system.water_flow = 8)
  (h2 : system.waterfall_height = 5)
  (h3 : system.turbine_efficiency = 0.8)
  (h4 : system.dynamo_efficiency = 0.9)
  (h5 : system.transmission_efficiency = 0.95)
  (h6 : system.num_motors = 5)
  (h7 : system.power_per_motor = 10)
  (h8 : system.motor_efficiency = 0.85)
  (h9 : system.num_arc_lamps = 24)
  (h10 : system.arc_lamp_voltage = 40)
  (h11 : system.arc_lamp_current = 10)
  (h12 : system.incandescent_bulb_power = 55) :
  calculate_incandescent_bulbs system = 3920 :=
sorry

end NUMINAMATH_CALUDE_incandescent_bulbs_count_l2871_287184


namespace NUMINAMATH_CALUDE_three_digit_subtraction_l2871_287151

/-- Represents a three-digit number with digits a, b, c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

theorem three_digit_subtraction
  (n₁ n₂ : ThreeDigitNumber)
  (h_reverse : n₂.a = n₁.c ∧ n₂.b = n₁.b ∧ n₂.c = n₁.a)
  (h_result_units : (n₁.toNat - n₂.toNat) % 10 = 2)
  (h_result_tens : ((n₁.toNat - n₂.toNat) / 10) % 10 = 9)
  (h_borrow : n₁.c < n₂.c) :
  n₁.a = 9 ∧ n₁.b = 9 ∧ n₁.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_l2871_287151


namespace NUMINAMATH_CALUDE_linear_equation_implies_m_eq_neg_three_l2871_287166

/-- Given that the equation (|m|-3)x^2 + (-m+3)x - 4 = 0 is linear in x with respect to m, prove that m = -3 -/
theorem linear_equation_implies_m_eq_neg_three (m : ℝ) 
  (h1 : ∀ x, (|m| - 3) * x^2 + (-m + 3) * x - 4 = 0 → (|m| - 3 = 0 ∧ -m + 3 ≠ 0)) : 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_implies_m_eq_neg_three_l2871_287166


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2871_287128

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < π / 2 ∧
    (∃ (r : ℝ),
      arcsin (sin α) * r = arcsin (sin (3 * α)) ∧
      arcsin (sin (3 * α)) * r = arcsin (sin (5 * α)) ∧
      arcsin (sin (5 * α)) * r = arcsin (sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < π / 2 ∧
      (∃ (r' : ℝ),
        arcsin (sin α') * r' = arcsin (sin (3 * α')) ∧
        arcsin (sin (3 * α')) * r' = arcsin (sin (5 * α')) ∧
        arcsin (sin (5 * α')) * r' = arcsin (sin (t' * α'))) →
      t ≤ t') ∧
    t = 9 + 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2871_287128


namespace NUMINAMATH_CALUDE_sequence_term_equals_three_l2871_287113

def a (n : ℝ) : ℝ := n^2 - 8*n + 15

theorem sequence_term_equals_three :
  ∃! (s : Set ℝ), s = {n : ℝ | a n = 3} ∧ s = {2, 6} :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_equals_three_l2871_287113


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2871_287193

theorem negation_of_proposition (P : (ℝ → Prop)) :
  (¬ (∀ x : ℝ, x > 0 → P x)) ↔ (∃ x : ℝ, x > 0 ∧ ¬(P x)) :=
by sorry

-- Define the specific proposition
def Q (x : ℝ) : Prop := x^2 + 2*x - 3 ≥ 0

theorem negation_of_specific_proposition :
  (¬ (∀ x : ℝ, x > 0 → Q x)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2871_287193


namespace NUMINAMATH_CALUDE_equal_domain_function_iff_a_range_l2871_287174

/-- A function that maps a set onto itself --/
def EqualDomainFunction (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ x ∈ A, f x ∈ A ∧ ∀ y ∈ A, ∃ x ∈ A, f x = y

/-- The quadratic function f(x) = a(x-1)^2 - 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 2

theorem equal_domain_function_iff_a_range :
  ∀ a < 0, (∃ m n : ℝ, m < n ∧ EqualDomainFunction (f a) (Set.Icc m n)) ↔ -1/12 < a ∧ a < 0 := by
  sorry

#check equal_domain_function_iff_a_range

end NUMINAMATH_CALUDE_equal_domain_function_iff_a_range_l2871_287174


namespace NUMINAMATH_CALUDE_triangle_theorem_l2871_287133

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the condition for an acute-angled triangle
def isAcuteAngled (t : Triangle) : Prop := sorry

-- Define the point P on AC
def pointOnAC (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define the condition 2AP = BC
def conditionAP (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define points X and Y symmetric to P with respect to A and C
def symmetricPoints (t : Triangle) (P X Y : ℝ × ℝ) : Prop := sorry

-- Define the condition BX = BY
def equalDistances (t : Triangle) (X Y : ℝ × ℝ) : Prop := sorry

-- Define the angle BCA
def angleBCA (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_theorem (t : Triangle) (P X Y : ℝ × ℝ) :
  isAcuteAngled t →
  pointOnAC t P →
  conditionAP t P →
  symmetricPoints t P X Y →
  equalDistances t X Y →
  angleBCA t = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2871_287133


namespace NUMINAMATH_CALUDE_mirror_area_l2871_287180

/-- Given a rectangular mirror centered within two frames, where the outermost frame measures
    100 cm by 140 cm, and both frames have a width of 15 cm on each side, the area of the mirror
    is 3200 cm². -/
theorem mirror_area (outer_length outer_width frame_width : ℕ) 
  (h1 : outer_length = 100)
  (h2 : outer_width = 140)
  (h3 : frame_width = 15) : 
  (outer_length - 2 * frame_width - 2 * frame_width) * 
  (outer_width - 2 * frame_width - 2 * frame_width) = 3200 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l2871_287180


namespace NUMINAMATH_CALUDE_random_walk_properties_l2871_287154

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- Number of right steps
  b : ℕ  -- Number of left steps
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Main theorem about the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry


end NUMINAMATH_CALUDE_random_walk_properties_l2871_287154


namespace NUMINAMATH_CALUDE_first_number_proof_l2871_287145

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℤ, x = 2 * k + 7) ∧ 
  (∃ l : ℤ, 2037 = 2 * l + 5) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ m : ℤ, y = 2 * m + 7)) → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_first_number_proof_l2871_287145


namespace NUMINAMATH_CALUDE_macaron_ratio_l2871_287161

theorem macaron_ratio (mitch joshua miles renz : ℕ) : 
  mitch = 20 →
  joshua = mitch + 6 →
  (∃ k : ℚ, joshua = k * miles) →
  renz = (3 * miles) / 4 - 1 →
  mitch + joshua + miles + renz = 68 * 2 →
  joshua * 2 = miles * 1 := by
  sorry

end NUMINAMATH_CALUDE_macaron_ratio_l2871_287161


namespace NUMINAMATH_CALUDE_mexican_restaurant_bill_solution_l2871_287114

/-- Represents the cost of items at a Mexican restaurant -/
structure MexicanRestaurantCosts where
  T : ℝ  -- Cost of a taco
  E : ℝ  -- Cost of an enchilada
  B : ℝ  -- Cost of a burrito

/-- The bills for three friends at a Mexican restaurant -/
def friend_bills (c : MexicanRestaurantCosts) : Prop :=
  2 * c.T + 3 * c.E = 7.80 ∧
  3 * c.T + 5 * c.E = 12.70 ∧
  4 * c.T + 2 * c.E + c.B = 15.40

/-- The theorem stating the unique solution for the Mexican restaurant bill problem -/
theorem mexican_restaurant_bill_solution :
  ∃! c : MexicanRestaurantCosts, friend_bills c ∧ c.T = 0.90 ∧ c.E = 2.00 ∧ c.B = 7.80 :=
by sorry

end NUMINAMATH_CALUDE_mexican_restaurant_bill_solution_l2871_287114


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_6_l2871_287175

/-- The equation of a hyperbola in its general form -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 432 * y - 1017 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 6)

/-- Theorem: The center of the given hyperbola is (3, 6) -/
theorem hyperbola_center_is_3_6 :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ h k : ℝ, h = hyperbola_center.1 ∧ k = hyperbola_center.2 ∧
  ∀ t : ℝ, hyperbola_equation (t + h) (t + k) ↔ hyperbola_equation (t + x) (t + y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_6_l2871_287175


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2871_287104

theorem multiplication_table_odd_fraction :
  let n : ℕ := 15
  let total_products : ℕ := (n + 1) * (n + 1)
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  odd_products / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2871_287104


namespace NUMINAMATH_CALUDE_relative_prime_linear_forms_l2871_287141

theorem relative_prime_linear_forms (a b : ℤ) : 
  ∃ c d : ℤ, ∀ n : ℤ, Int.gcd (a * n + c) (b * n + d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_relative_prime_linear_forms_l2871_287141


namespace NUMINAMATH_CALUDE_tom_seashells_l2871_287109

theorem tom_seashells (yesterday : ℕ) (today : ℕ) 
  (h1 : yesterday = 7) (h2 : today = 4) : 
  yesterday + today = 11 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l2871_287109


namespace NUMINAMATH_CALUDE_max_product_sum_l2871_287188

theorem max_product_sum (a b M : ℝ) : 
  a > 0 → b > 0 → (a + b = M) → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_l2871_287188


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2871_287195

theorem min_value_sum_squares (x y z : ℝ) (h : 4*x + 3*y + 12*z = 1) :
  x^2 + y^2 + z^2 ≥ 1/169 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2871_287195


namespace NUMINAMATH_CALUDE_alpha_values_l2871_287138

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^3 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 8 ∨ α = -Complex.I * Real.sqrt 8 :=
sorry

end NUMINAMATH_CALUDE_alpha_values_l2871_287138


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l2871_287168

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leapYearFrequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def maxLeapYears : ℕ := period / leapYearFrequency

/-- Theorem: The maximum number of leap years in a 200-year period
    with leap years occurring every five years is 40 -/
theorem max_leap_years_in_period :
  maxLeapYears = 40 := by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l2871_287168


namespace NUMINAMATH_CALUDE_sqrt_equation_proof_l2871_287116

theorem sqrt_equation_proof (y : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt y) + (Real.sqrt 1.00 / Real.sqrt 0.49) = 2.650793650793651 → 
  y = 0.81 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_proof_l2871_287116


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2871_287146

def U : Set ℝ := {x | Real.exp x > 1}
def A : Set ℝ := {x | x > 1}

theorem complement_of_A_in_U : Set.compl A ∩ U = Set.Ioo 0 1 ∪ Set.singleton 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2871_287146


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l2871_287117

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = -20) ∨ (x = 2 ∧ y = 24))) → 
  c = -6 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l2871_287117


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l2871_287163

theorem bicycle_price_problem (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (final_price : ℝ) :
  profit_a_to_b = 0.25 →
  profit_b_to_c = 0.5 →
  final_price = 225 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = final_price ∧
    cost_price_a = 120 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l2871_287163


namespace NUMINAMATH_CALUDE_sector_central_angle_l2871_287101

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians. -/
theorem sector_central_angle (r θ : ℝ) : 
  (2 * r + r * θ = 4) →  -- perimeter condition
  ((1 / 2) * r^2 * θ = 1) →  -- area condition
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2871_287101


namespace NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l2871_287129

/-- The number of burrs on Andrew's dog -/
def num_burrs : ℕ := 12

/-- The ratio of ticks to burrs on Andrew's dog -/
def tick_to_burr_ratio : ℕ := 6

/-- The total number of foreign objects (burrs and ticks) on Andrew's dog -/
def total_foreign_objects : ℕ := num_burrs + num_burrs * tick_to_burr_ratio

theorem andrews_dog_foreign_objects :
  total_foreign_objects = 84 :=
by sorry

end NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l2871_287129


namespace NUMINAMATH_CALUDE_diamond_inequality_l2871_287194

def diamond (x y : ℝ) : ℝ := |x^2 - y^2|

theorem diamond_inequality : ∃ x y : ℝ, diamond (x + y) (x - y) ≠ diamond x y := by
  sorry

end NUMINAMATH_CALUDE_diamond_inequality_l2871_287194


namespace NUMINAMATH_CALUDE_complex_root_of_unity_sum_l2871_287132

theorem complex_root_of_unity_sum (ω : ℂ) : 
  ω = -1/2 + (Complex.I * Real.sqrt 3) / 2 → ω^4 + ω^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_of_unity_sum_l2871_287132


namespace NUMINAMATH_CALUDE_intersection_points_sum_l2871_287137

/-- The quadratic function f(x) = (x+2)(x-4) -/
def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

/-- The function g(x) = -f(x) -/
def g (x : ℝ) : ℝ := -f x

/-- The function h(x) = f(-x) -/
def h (x : ℝ) : ℝ := f (-x)

/-- The number of intersection points between y=f(x) and y=g(x) -/
def a : ℕ := 2

/-- The number of intersection points between y=f(x) and y=h(x) -/
def b : ℕ := 1

theorem intersection_points_sum : 10 * a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l2871_287137


namespace NUMINAMATH_CALUDE_log_sum_simplification_l2871_287135

theorem log_sum_simplification :
  1 / (Real.log 2 / Real.log 15 + 1) + 
  1 / (Real.log 3 / Real.log 10 + 1) + 
  1 / (Real.log 5 / Real.log 6 + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l2871_287135


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l2871_287106

theorem carousel_horse_ratio :
  ∀ (blue purple green gold : ℕ),
    blue = 3 →
    purple = 3 * blue →
    gold = green / 6 →
    blue + purple + green + gold = 33 →
    (green : ℚ) / purple = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l2871_287106


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2871_287177

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2871_287177


namespace NUMINAMATH_CALUDE_equation_solutions_l2871_287196

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 - 9 = 0 ↔ x = 5/2 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2871_287196


namespace NUMINAMATH_CALUDE_pentagon_to_squares_ratio_l2871_287191

-- Define the square structure
structure Square :=
  (side : ℝ)

-- Define the pentagon structure
structure Pentagon :=
  (area : ℝ)

-- Define the theorem
theorem pentagon_to_squares_ratio 
  (s : Square) 
  (p : Pentagon) 
  (h1 : s.side > 0)
  (h2 : p.area = s.side * s.side)
  : p.area / (3 * s.side * s.side) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_pentagon_to_squares_ratio_l2871_287191


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l2871_287170

/-- Represents a 5x5 grid of dots -/
def Grid : Type := Unit

/-- The number of dots in the grid -/
def num_dots : ℕ := 25

/-- The number of ways to choose 4 collinear dots from the grid -/
def num_collinear_sets : ℕ := 54

/-- The total number of ways to choose 4 dots from the grid -/
def total_combinations : ℕ := 12650

/-- The probability of selecting 4 collinear dots when choosing 4 dots at random -/
def collinear_probability (g : Grid) : ℚ := 6 / 1415

theorem collinear_dots_probability (g : Grid) : 
  collinear_probability g = num_collinear_sets / total_combinations :=
by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l2871_287170
