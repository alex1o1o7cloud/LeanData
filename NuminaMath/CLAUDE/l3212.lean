import Mathlib

namespace NUMINAMATH_CALUDE_abs_sum_minimum_l3212_321290

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l3212_321290


namespace NUMINAMATH_CALUDE_q_is_false_l3212_321247

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l3212_321247


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l3212_321212

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c : ℕ)^3 - a^3 - b^3 - c^3 = 180 → a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l3212_321212


namespace NUMINAMATH_CALUDE_dinner_bill_split_l3212_321256

theorem dinner_bill_split (total_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (tax_percent : ℝ) :
  total_bill = 425 →
  num_people = 15 →
  tip_percent = 0.18 →
  tax_percent = 0.08 →
  (total_bill * (1 + tip_percent + tax_percent)) / num_people = 35.70 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_split_l3212_321256


namespace NUMINAMATH_CALUDE_distinct_centroids_count_l3212_321216

/-- Represents a point on the perimeter of the square -/
structure PerimeterPoint where
  x : Fin 11
  y : Fin 11
  on_perimeter : (x = 0 ∨ x = 10) ∨ (y = 0 ∨ y = 10)

/-- The set of 40 equally spaced points on the square's perimeter -/
def perimeterPoints : Finset PerimeterPoint :=
  sorry

/-- Represents the centroid of a triangle -/
structure Centroid where
  x : Rat
  y : Rat
  inside_square : 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10

/-- Function to calculate the centroid given three points -/
def calculateCentroid (p q r : PerimeterPoint) : Centroid :=
  sorry

/-- The set of all possible centroids -/
def allCentroids : Finset Centroid :=
  sorry

/-- Main theorem: The number of distinct centroids is 841 -/
theorem distinct_centroids_count : Finset.card allCentroids = 841 :=
  sorry

end NUMINAMATH_CALUDE_distinct_centroids_count_l3212_321216


namespace NUMINAMATH_CALUDE_mn_positive_necessary_mn_positive_not_sufficient_l3212_321201

/-- Definition of an ellipse equation -/
def is_ellipse_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ m ≠ n ∧ m > 0 ∧ n > 0

/-- The condition mn > 0 is necessary for the equation to represent an ellipse -/
theorem mn_positive_necessary (m n : ℝ) :
  is_ellipse_equation m n → m * n > 0 :=
sorry

/-- The condition mn > 0 is not sufficient for the equation to represent an ellipse -/
theorem mn_positive_not_sufficient :
  ∃ (m n : ℝ), m * n > 0 ∧ ¬(is_ellipse_equation m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_mn_positive_not_sufficient_l3212_321201


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l3212_321237

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

theorem smallest_interesting_number : 
  (is_interesting 1800) ∧ (∀ m : ℕ, m < 1800 → ¬(is_interesting m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l3212_321237


namespace NUMINAMATH_CALUDE_log_sum_simplification_l3212_321265

theorem log_sum_simplification :
  1 / (Real.log 2 / Real.log 15 + 1) + 
  1 / (Real.log 3 / Real.log 10 + 1) + 
  1 / (Real.log 5 / Real.log 6 + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l3212_321265


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3212_321238

def f (x : ℝ) : ℝ := x^3 + 4*x - 5

theorem f_derivative_at_2 : (deriv f) 2 = 16 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3212_321238


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3212_321236

theorem second_discount_percentage 
  (initial_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) :
  initial_price = 1000 →
  first_discount = 15 →
  final_price = 830 →
  initial_price * (1 - first_discount / 100) * (1 - x / 100) = final_price :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3212_321236


namespace NUMINAMATH_CALUDE_completing_square_quadratic_equation_l3212_321245

theorem completing_square_quadratic_equation :
  ∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_equation_l3212_321245


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l3212_321280

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n →
    (is_divisible_by n 6 → n ≥ 1002) ∧
    (is_divisible_by 1002 6) ∧
    is_four_digit 1002 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l3212_321280


namespace NUMINAMATH_CALUDE_no_real_roots_l3212_321298

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3212_321298


namespace NUMINAMATH_CALUDE_wall_decoration_thumbtack_fraction_l3212_321297

theorem wall_decoration_thumbtack_fraction :
  let total_decorations : ℕ := 50 * 3 / 2
  let nailed_decorations : ℕ := 50
  let remaining_decorations : ℕ := total_decorations - nailed_decorations
  let sticky_strip_decorations : ℕ := 15
  let thumbtack_decorations : ℕ := remaining_decorations - sticky_strip_decorations
  (thumbtack_decorations : ℚ) / remaining_decorations = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_wall_decoration_thumbtack_fraction_l3212_321297


namespace NUMINAMATH_CALUDE_proposition_truth_values_l3212_321241

theorem proposition_truth_values (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∧ (¬q)) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l3212_321241


namespace NUMINAMATH_CALUDE_greatest_negative_value_x_minus_y_l3212_321221

theorem greatest_negative_value_x_minus_y :
  ∃ (x y : ℝ), 
    (Real.sin x + Real.sin y) * (Real.cos x - Real.cos y) = 1/2 + Real.sin (x - y) * Real.cos (x + y) ∧
    x - y = -π/6 ∧
    ∀ (a b : ℝ), 
      (Real.sin a + Real.sin b) * (Real.cos a - Real.cos b) = 1/2 + Real.sin (a - b) * Real.cos (a + b) →
      a - b < 0 →
      a - b ≤ -π/6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_negative_value_x_minus_y_l3212_321221


namespace NUMINAMATH_CALUDE_reggie_bought_five_books_l3212_321268

/-- The number of books Reggie bought -/
def number_of_books (initial_amount remaining_amount cost_per_book : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / cost_per_book

/-- Theorem: Reggie bought 5 books -/
theorem reggie_bought_five_books :
  number_of_books 48 38 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_reggie_bought_five_books_l3212_321268


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3212_321211

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 4 + α) = Real.sqrt 2 / 3) : 
  Real.sin (2 * α) / (1 - Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3212_321211


namespace NUMINAMATH_CALUDE_family_race_problem_l3212_321246

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

end NUMINAMATH_CALUDE_family_race_problem_l3212_321246


namespace NUMINAMATH_CALUDE_root_zero_implies_a_half_l3212_321282

theorem root_zero_implies_a_half (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + 2*a - 1 = 0 ∧ x = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_zero_implies_a_half_l3212_321282


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3212_321249

/-- Given the following conditions:
    - 16 chapatis, each costing Rs. 6
    - 5 plates of rice, each costing Rs. 45
    - 7 plates of mixed vegetable, each costing Rs. 70
    - 6 ice-cream cups
    - Total amount paid: Rs. 931
    Prove that the cost of each ice-cream cup is Rs. 20. -/
theorem ice_cream_cost (chapati_count : ℕ) (chapati_cost : ℕ)
                       (rice_count : ℕ) (rice_cost : ℕ)
                       (veg_count : ℕ) (veg_cost : ℕ)
                       (ice_cream_count : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  chapati_cost = 6 →
  rice_count = 5 →
  rice_cost = 45 →
  veg_count = 7 →
  veg_cost = 70 →
  ice_cream_count = 6 →
  total_paid = 931 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + veg_count * veg_cost)) / ice_cream_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3212_321249


namespace NUMINAMATH_CALUDE_eighth_term_is_one_l3212_321271

-- Define the sequence a_n
def a (n : ℕ+) : ℤ := (-1) ^ n.val

-- Theorem statement
theorem eighth_term_is_one : a 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_one_l3212_321271


namespace NUMINAMATH_CALUDE_store_comparison_l3212_321224

/-- Represents the cost function for store A -/
def cost_A (x : ℕ) : ℝ :=
  if x = 0 then 0 else 140 * x + 60

/-- Represents the cost function for store B -/
def cost_B (x : ℕ) : ℝ := 150 * x

theorem store_comparison (x : ℕ) (h : x ≥ 1) :
  (cost_A x = 140 * x + 60) ∧
  (cost_B x = 150 * x) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y < 6 → cost_A y < cost_B y) ∧
  (∀ z : ℕ, z > 6 → cost_A z > cost_B z) :=
by sorry

end NUMINAMATH_CALUDE_store_comparison_l3212_321224


namespace NUMINAMATH_CALUDE_factorization_equality_l3212_321267

theorem factorization_equality (x : ℝ) :
  (x - 1)^4 + x * (2*x + 1) * (2*x - 1) + 5*x = (x^2 + 3 + 2*Real.sqrt 2) * (x^2 + 3 - 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3212_321267


namespace NUMINAMATH_CALUDE_range_of_a_l3212_321206

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, x^a ≥ 2 * Real.exp (2*x) * f a x + Real.exp (2*x)) →
  a ≤ 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3212_321206


namespace NUMINAMATH_CALUDE_j_value_at_one_l3212_321253

theorem j_value_at_one (p q r : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 + p*x^2 + 2*x + 20 = 0) ∧
    (y^3 + p*y^2 + 2*y + 20 = 0) ∧
    (z^3 + p*z^2 + 2*z + 20 = 0)) →
  (∀ x : ℝ, x^3 + p*x^2 + 2*x + 20 = 0 → x^4 + 2*x^3 + q*x^2 + 150*x + r = 0) →
  1^4 + 2*1^3 + q*1^2 + 150*1 + r = -13755 :=
by sorry

end NUMINAMATH_CALUDE_j_value_at_one_l3212_321253


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3212_321242

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sum_of_binary_digits 300 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3212_321242


namespace NUMINAMATH_CALUDE_final_candy_count_l3212_321222

-- Define the variables
def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def received_candy : ℕ := 40

-- State the theorem
theorem final_candy_count :
  initial_candy - eaten_candy + received_candy = 62 := by
  sorry

end NUMINAMATH_CALUDE_final_candy_count_l3212_321222


namespace NUMINAMATH_CALUDE_range_of_m_l3212_321275

/-- The curve equation -/
def curve (x y m : ℝ) : Prop := x^2 + y^2 + y + m = 0

/-- The symmetry line equation -/
def symmetry_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

/-- Predicate for having four common tangents -/
def has_four_common_tangents (m : ℝ) : Prop := sorry

/-- Theorem stating the range of m -/
theorem range_of_m : 
  ∀ m : ℝ, (∀ x y : ℝ, curve x y m → ∃ x' y' : ℝ, symmetry_line x' y' ∧ has_four_common_tangents m) 
  ↔ -11/20 < m ∧ m < 1/4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3212_321275


namespace NUMINAMATH_CALUDE_shared_fixed_points_l3212_321286

/-- A function that represents f(x) = x^2 - 2 --/
def f (x : ℝ) : ℝ := x^2 - 2

/-- A function that represents g(x) = 2x^2 - c --/
def g (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - c

/-- The theorem stating the conditions for shared fixed points --/
theorem shared_fixed_points (c : ℝ) : 
  (c = 3 ∨ c = 6) ↔ ∃ x : ℝ, (f x = x ∧ g c x = x) :=
sorry

end NUMINAMATH_CALUDE_shared_fixed_points_l3212_321286


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3212_321231

theorem factor_of_polynomial (x : ℝ) :
  (x^4 - 4*x^2 + 16) = (x^2 - 4*x + 4) * (x^2 + 2*x + 4) := by
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3212_321231


namespace NUMINAMATH_CALUDE_difference_sum_of_T_l3212_321225

def T : Finset ℕ := Finset.range 9

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if 3^j > 3^i then 3^j - 3^i else 0))

theorem difference_sum_of_T : difference_sum T = 69022 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_T_l3212_321225


namespace NUMINAMATH_CALUDE_simplify_expression_l3212_321295

theorem simplify_expression (y : ℝ) : 5*y + 7*y - 3*y = 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3212_321295


namespace NUMINAMATH_CALUDE_average_score_is_correct_total_students_is_correct_l3212_321230

/-- Calculates the average score given a list of (score, number of students) pairs -/
def averageScore (scores : List (ℚ × ℕ)) : ℚ :=
  let totalScore := scores.foldl (fun acc (score, count) => acc + score * count) 0
  let totalStudents := scores.foldl (fun acc (_, count) => acc + count) 0
  totalScore / totalStudents

/-- The given score distribution -/
def scoreDistribution : List (ℚ × ℕ) :=
  [(100, 10), (95, 20), (85, 40), (70, 40), (60, 20), (55, 10), (45, 10)]

/-- The total number of students -/
def totalStudents : ℕ := 150

/-- Theorem stating that the average score is 75.33 (11300/150) -/
theorem average_score_is_correct :
  averageScore scoreDistribution = 11300 / 150 := by
  sorry

/-- Theorem verifying the total number of students -/
theorem total_students_is_correct :
  (scoreDistribution.foldl (fun acc (_, count) => acc + count) 0) = totalStudents := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_correct_total_students_is_correct_l3212_321230


namespace NUMINAMATH_CALUDE_peter_total_spending_l3212_321262

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

end NUMINAMATH_CALUDE_peter_total_spending_l3212_321262


namespace NUMINAMATH_CALUDE_employee_salary_problem_l3212_321208

/-- Proves that given the conditions of the problem, employee N's salary is $265 per week -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) : 
  total_salary = 583 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 265 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l3212_321208


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l3212_321276

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

end NUMINAMATH_CALUDE_area_of_quadrilateral_l3212_321276


namespace NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l3212_321233

theorem cube_sum_over_product_is_18 
  (x y z : ℂ) 
  (nonzero_x : x ≠ 0) 
  (nonzero_y : y ≠ 0) 
  (nonzero_z : z ≠ 0) 
  (sum_30 : x + y + z = 30) 
  (squared_diff_sum : (x - y)^2 + (x - z)^2 + (y - z)^2 + x*y*z = 2*x*y*z) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l3212_321233


namespace NUMINAMATH_CALUDE_intercept_sum_modulo_40_l3212_321213

/-- 
Given the congruence 5x ≡ 3y - 2 (mod 40), this theorem proves that 
the sum of the x-intercept and y-intercept is 38, where both intercepts 
are non-negative integers less than 40.
-/
theorem intercept_sum_modulo_40 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 40 ∧ y₀ < 40 ∧ 
  (5 * x₀) % 40 = (3 * 0 - 2) % 40 ∧
  (5 * 0) % 40 = (3 * y₀ - 2) % 40 ∧
  x₀ + y₀ = 38 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_modulo_40_l3212_321213


namespace NUMINAMATH_CALUDE_seventh_term_value_l3212_321203

/-- The general term of the series at position n -/
def seriesTerm (n : ℕ) (a : ℝ) : ℝ := (-2)^n * a^(2*n - 1)

/-- The 7th term of the series -/
def seventhTerm (a : ℝ) : ℝ := seriesTerm 7 a

theorem seventh_term_value (a : ℝ) : seventhTerm a = -128 * a^13 := by sorry

end NUMINAMATH_CALUDE_seventh_term_value_l3212_321203


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3212_321259

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  full_days : Nat        -- Number of days working 8 hours
  partial_days : Nat     -- Number of days working 6 hours
  weekly_earnings : Nat  -- Total earnings per week in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.full_days + 6 * schedule.partial_days)

/-- Theorem: Sheila's hourly wage is $6 -/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    full_days := 3,
    partial_days := 2,
    weekly_earnings := 216
  }
  hourly_wage schedule = 6 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l3212_321259


namespace NUMINAMATH_CALUDE_new_person_weight_l3212_321266

theorem new_person_weight (W : ℝ) (new_weight : ℝ) :
  (W + new_weight - 25) / 12 = W / 12 + 3 →
  new_weight = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3212_321266


namespace NUMINAMATH_CALUDE_range_of_m_l3212_321281

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(-x^2 - x) > (1/2 : ℝ)^(2*x^2 - m*x + m + 4)) ↔ 
  (-3 < m ∧ m < 5) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3212_321281


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3212_321260

theorem min_value_sum_squares (x y z : ℝ) (h : 4*x + 3*y + 12*z = 1) :
  x^2 + y^2 + z^2 ≥ 1/169 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3212_321260


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3212_321226

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 60 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (h : ∃ x y, hyperbola_equation x y) : ℝ :=
  1

theorem hyperbola_vertex_distance :
  ∀ h : ∃ x y, hyperbola_equation x y,
  vertex_distance h = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3212_321226


namespace NUMINAMATH_CALUDE_parallelepiped_count_l3212_321251

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a set of four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate to check if four points are coplanar -/
def areCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * points.p1.x + b * points.p1.y + c * points.p1.z + d = 0 ∧
    a * points.p2.x + b * points.p2.y + c * points.p2.z + d = 0 ∧
    a * points.p3.x + b * points.p3.y + c * points.p3.z + d = 0 ∧
    a * points.p4.x + b * points.p4.y + c * points.p4.z + d = 0

/-- Function to count the number of distinct parallelepipeds -/
def countParallelepipeds (points : FourPoints) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem parallelepiped_count (points : FourPoints) 
  (h : ¬ areCoplanar points) : countParallelepipeds points = 29 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_count_l3212_321251


namespace NUMINAMATH_CALUDE_complex_multiplication_l3212_321263

theorem complex_multiplication :
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 2 - 3 * Complex.I
  z₁ * z₂ = 7 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3212_321263


namespace NUMINAMATH_CALUDE_mary_found_two_seashells_l3212_321240

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 7 - 5

/-- The number of seashells Keith found -/
def keith_seashells : ℕ := 5

/-- The total number of seashells Mary and Keith found together -/
def total_seashells : ℕ := 7

theorem mary_found_two_seashells :
  mary_seashells = 2 :=
sorry

end NUMINAMATH_CALUDE_mary_found_two_seashells_l3212_321240


namespace NUMINAMATH_CALUDE_odd_prime_gcd_sum_and_fraction_l3212_321257

theorem odd_prime_gcd_sum_and_fraction (p a b : ℕ) : 
  Nat.Prime p → p % 2 = 1 → Nat.Coprime a b → 
  Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := by
sorry

end NUMINAMATH_CALUDE_odd_prime_gcd_sum_and_fraction_l3212_321257


namespace NUMINAMATH_CALUDE_singing_competition_average_age_l3212_321243

theorem singing_competition_average_age 
  (num_females : Nat) 
  (num_males : Nat)
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 25 →
  avg_age_males = 40 →
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
sorry

end NUMINAMATH_CALUDE_singing_competition_average_age_l3212_321243


namespace NUMINAMATH_CALUDE_common_chord_length_l3212_321255

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define the common chord line
def common_chord_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    common_chord_line A.1 A.2 ∧ common_chord_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3212_321255


namespace NUMINAMATH_CALUDE_gasohol_calculation_l3212_321227

/-- The amount of gasohol initially in the tank (in liters) -/
def initial_gasohol : ℝ := 27

/-- The fraction of ethanol in the initial mixture -/
def initial_ethanol_fraction : ℝ := 0.05

/-- The fraction of ethanol in the desired mixture -/
def desired_ethanol_fraction : ℝ := 0.10

/-- The amount of pure ethanol added (in liters) -/
def added_ethanol : ℝ := 1.5

theorem gasohol_calculation :
  initial_gasohol * initial_ethanol_fraction + added_ethanol =
  desired_ethanol_fraction * (initial_gasohol + added_ethanol) := by
  sorry

end NUMINAMATH_CALUDE_gasohol_calculation_l3212_321227


namespace NUMINAMATH_CALUDE_average_difference_l3212_321264

theorem average_difference (x y z w : ℝ) : 
  (x + y + z) / 3 = (y + z + w) / 3 + 10 → w = x - 30 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3212_321264


namespace NUMINAMATH_CALUDE_eulers_criterion_l3212_321288

theorem eulers_criterion (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_p : p > 2) (h_a : 1 ≤ a ∧ a ≤ p - 1) :
  (∃ x : Nat, x ^ 2 % p = a % p) ↔ a ^ ((p - 1) / 2) % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_eulers_criterion_l3212_321288


namespace NUMINAMATH_CALUDE_polynomial_square_prime_values_l3212_321289

def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

theorem polynomial_square_prime_values :
  {n : ℤ | ∃ (p : ℕ), Prime p ∧ (P n)^2 = p^2} = {-3, -1, 0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_prime_values_l3212_321289


namespace NUMINAMATH_CALUDE_expression_evaluation_l3212_321291

theorem expression_evaluation :
  (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3212_321291


namespace NUMINAMATH_CALUDE_race_distance_l3212_321218

/-- Represents the race scenario with given conditions -/
structure RaceScenario where
  distance : ℝ
  timeA : ℝ
  startAdvantage1 : ℝ
  timeDifference : ℝ
  startAdvantage2 : ℝ

/-- Defines the conditions of the race -/
def raceConditions : RaceScenario → Prop
  | ⟨d, t, s1, dt, s2⟩ => 
    t = 77.5 ∧ 
    s1 = 25 ∧ 
    dt = 10 ∧ 
    s2 = 45 ∧ 
    d / t = (d - s1) / (t + dt) ∧ 
    d / t = (d - s2) / t

/-- Theorem stating that the race distance is 218.75 meters -/
theorem race_distance (scenario : RaceScenario) 
  (h : raceConditions scenario) : scenario.distance = 218.75 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l3212_321218


namespace NUMINAMATH_CALUDE_arithmetic_verification_l3212_321217

theorem arithmetic_verification (A B C M N P : ℝ) : 
  (A - B = C → C + B = A ∧ A - C = B) ∧ 
  (M * N = P → P / N = M ∧ P / M = N) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_verification_l3212_321217


namespace NUMINAMATH_CALUDE_complex_multiplication_l3212_321229

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (3 - 4*i) * (-6 + 2*i) = -10 + 30*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3212_321229


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3212_321273

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (α + Real.pi / 6) ^ 2 + Real.sin α * Real.cos (α + Real.pi / 6) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3212_321273


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l3212_321272

theorem coefficient_x2y2_in_expansion : ℕ → Prop :=
  fun n => (Finset.sum (Finset.range 4) fun i =>
    (Finset.sum (Finset.range 5) fun j =>
      if i + j = 4 then
        (Nat.choose 3 i) * (Nat.choose 4 j)
      else
        0)) = n

theorem expansion_coefficient_is_18 : coefficient_x2y2_in_expansion 18 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l3212_321272


namespace NUMINAMATH_CALUDE_max_product_sum_l3212_321277

theorem max_product_sum (a b M : ℝ) : 
  a > 0 → b > 0 → (a + b = M) → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_l3212_321277


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3212_321252

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3212_321252


namespace NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l3212_321278

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ n : ℤ, n < 0 ∧ n ≥ -9999 ∧ n ≡ 1 [ZMOD 37] → n ≥ -1034 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l3212_321278


namespace NUMINAMATH_CALUDE_current_gas_in_car_l3212_321209

/-- Represents the fuel efficiency of a car in miles per gallon -/
def fuel_efficiency : ℝ := 20

/-- Represents the total distance to be traveled in miles -/
def total_distance : ℝ := 1200

/-- Represents the additional gallons of gas needed for the trip -/
def additional_gas_needed : ℝ := 52

/-- Theorem stating the current amount of gas in the car -/
theorem current_gas_in_car : 
  (total_distance / fuel_efficiency) - additional_gas_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_current_gas_in_car_l3212_321209


namespace NUMINAMATH_CALUDE_onion_bag_weight_l3212_321258

/-- Proves that the weight of each bag of onions is 50 kgs given the specified conditions -/
theorem onion_bag_weight 
  (bags_per_trip : ℕ) 
  (num_trips : ℕ) 
  (total_weight : ℕ) 
  (h1 : bags_per_trip = 10)
  (h2 : num_trips = 20)
  (h3 : total_weight = 10000) :
  total_weight / (bags_per_trip * num_trips) = 50 := by
  sorry

end NUMINAMATH_CALUDE_onion_bag_weight_l3212_321258


namespace NUMINAMATH_CALUDE_circle_area_not_tripled_l3212_321207

/-- Tripling the radius of a circle does not triple its area -/
theorem circle_area_not_tripled (r : ℝ) (h : r > 0) : π * (3 * r)^2 ≠ 3 * (π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_not_tripled_l3212_321207


namespace NUMINAMATH_CALUDE_allowance_spent_at_toy_store_l3212_321248

theorem allowance_spent_at_toy_store 
  (total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (remaining_after_toy_store : ℚ)
  (h1 : total_allowance = 9/4)  -- $2.25 as a fraction
  (h2 : arcade_fraction = 3/5)
  (h3 : remaining_after_toy_store = 3/5)  -- $0.60 as a fraction
  : (total_allowance - arcade_fraction * total_allowance - remaining_after_toy_store) / 
    (total_allowance - arcade_fraction * total_allowance) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allowance_spent_at_toy_store_l3212_321248


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3212_321294

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100 →
  a 4 + a 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3212_321294


namespace NUMINAMATH_CALUDE_rowing_speeds_theorem_l3212_321285

/-- Represents the rowing speeds and wind effects for a man rowing a boat -/
structure RowingScenario where
  normalSpeedWith : ℝ      -- Normal speed with the stream
  normalSpeedAgainst : ℝ   -- Normal speed against the stream
  windSpeedReduction : ℝ   -- Wind speed reduction percentage against the stream
  windSpeedIncrease : ℝ    -- Wind speed increase percentage with the stream

/-- Calculates the effective rowing speed against the stream -/
def effectiveSpeedAgainst (scenario : RowingScenario) : ℝ :=
  scenario.normalSpeedAgainst * (1 - scenario.windSpeedReduction)

/-- Calculates the effective rowing speed with the stream -/
def effectiveSpeedWith (scenario : RowingScenario) : ℝ :=
  scenario.normalSpeedWith * (1 + scenario.windSpeedIncrease)

/-- Theorem stating the effective rowing speeds for the given scenario -/
theorem rowing_speeds_theorem (scenario : RowingScenario) 
  (h1 : scenario.normalSpeedWith = 8)
  (h2 : scenario.normalSpeedAgainst = 4)
  (h3 : scenario.windSpeedReduction = 0.2)
  (h4 : scenario.windSpeedIncrease = 0.1) :
  effectiveSpeedAgainst scenario = 3.2 ∧ effectiveSpeedWith scenario = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speeds_theorem_l3212_321285


namespace NUMINAMATH_CALUDE_cost_of_700_pieces_l3212_321299

/-- The cost function for gum pieces -/
def gum_cost (pieces : ℕ) : ℚ :=
  if pieces ≤ 500 then
    pieces / 100
  else
    5 + (pieces - 500) * 8 / 1000

/-- Theorem stating the cost of 700 pieces of gum -/
theorem cost_of_700_pieces : gum_cost 700 = 33/5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_700_pieces_l3212_321299


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_24_l3212_321215

theorem largest_four_digit_congruent_to_17_mod_24 : ∃ n : ℕ, 
  (n ≡ 17 [ZMOD 24]) ∧ 
  (n < 10000) ∧ 
  (1000 ≤ n) ∧ 
  (∀ m : ℕ, (m ≡ 17 [ZMOD 24]) → (1000 ≤ m) → (m < 10000) → m ≤ n) ∧ 
  n = 9977 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_24_l3212_321215


namespace NUMINAMATH_CALUDE_bicycle_has_four_wheels_l3212_321219

-- Define the universe of objects
variable (Object : Type)

-- Define predicates
variable (isCar : Object → Prop)
variable (hasFourWheels : Object → Prop)

-- Define a specific object
variable (bicycle : Object)

-- Theorem statement
theorem bicycle_has_four_wheels 
  (all_cars_have_four_wheels : ∀ x, isCar x → hasFourWheels x)
  (bicycle_is_car : isCar bicycle) :
  hasFourWheels bicycle :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_has_four_wheels_l3212_321219


namespace NUMINAMATH_CALUDE_total_owed_is_790_l3212_321234

/-- Calculates the total amount owed for three overdue bills -/
def total_amount_owed (bill1_principal : ℝ) (bill1_interest_rate : ℝ) (bill1_months : ℕ)
                      (bill2_principal : ℝ) (bill2_late_fee : ℝ) (bill2_months : ℕ)
                      (bill3_first_month_fee : ℝ) (bill3_months : ℕ) : ℝ :=
  let bill1_total := bill1_principal * (1 + bill1_interest_rate * bill1_months)
  let bill2_total := bill2_principal + bill2_late_fee * bill2_months
  let bill3_total := bill3_first_month_fee * (1 + (bill3_months - 1) * 2)
  bill1_total + bill2_total + bill3_total

/-- Theorem stating the total amount owed is $790 given the specific bill conditions -/
theorem total_owed_is_790 :
  total_amount_owed 200 0.1 2 130 50 6 40 2 = 790 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_is_790_l3212_321234


namespace NUMINAMATH_CALUDE_binomial_square_last_term_l3212_321250

theorem binomial_square_last_term (a b : ℝ) :
  ∃ x y : ℝ, x^2 - 10*x*y + 25*y^2 = (x + y)^2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_square_last_term_l3212_321250


namespace NUMINAMATH_CALUDE_percentage_problem_l3212_321254

theorem percentage_problem (x : ℝ) : 
  (0.15 * 25) + (x / 100 * 45) = 9.15 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3212_321254


namespace NUMINAMATH_CALUDE_tank_fill_level_l3212_321228

theorem tank_fill_level (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) 
  (h1 : tank_capacity = 42)
  (h2 : added_amount = 7)
  (h3 : final_fraction = 9/10)
  (h4 : (final_fraction * tank_capacity) = (added_amount + (initial_fraction * tank_capacity))) :
  initial_fraction = 733/1000 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_level_l3212_321228


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_third_l3212_321274

/-- A cubic function f(x) = ax^3 - x^2 + x - 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- f is monotonically increasing if its derivative is non-negative for all x -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x : ℝ, f_deriv a x ≥ 0

theorem monotone_increasing_implies_a_geq_one_third :
  ∀ a : ℝ, is_monotone_increasing a → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_third_l3212_321274


namespace NUMINAMATH_CALUDE_ellipse_equation_triangle_area_line_equation_l3212_321284

/-- An ellipse passing through (-1, -1) with semi-focal distance c = √2b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 1 / a^2 + 1 / b^2 = 1
  h4 : 2 * b^2 = (a^2 - b^2)

/-- Two points on the ellipse intersected by perpendicular lines through (-1, -1) -/
structure IntersectionPoints (e : Ellipse) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  h1 : M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1
  h2 : N.1^2 / e.a^2 + N.2^2 / e.b^2 = 1
  h3 : (M.1 + 1) * (N.1 + 1) + (M.2 + 1) * (N.2 + 1) = 0

theorem ellipse_equation (e : Ellipse) : e.a^2 = 4 ∧ e.b^2 = 4/3 :=
sorry

theorem triangle_area (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 = 0 ∧ p.N.1 = 1 ∧ p.N.2 = 1) : 
  abs ((p.M.1 + 1) * (p.N.2 + 1) - (p.N.1 + 1) * (p.M.2 + 1)) / 2 = 2 :=
sorry

theorem line_equation (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 + p.N.2 = 0) :
  (p.M.2 = -p.M.1 ∧ p.N.2 = -p.N.1) ∨ 
  (p.M.1 + p.M.2 = 0 ∧ p.N.1 + p.N.2 = 0) ∨ 
  (p.M.1 = -1/2 ∧ p.N.1 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_triangle_area_line_equation_l3212_321284


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3212_321270

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d : ℚ), ax^4 + bx^3 + 40*x^2 - 20*x + 10 = (5*x^2 - 3*x + 2)*(c*x^2 + d*x + 5)) :
  a = 25/4 ∧ b = -65/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3212_321270


namespace NUMINAMATH_CALUDE_probability_of_event_l3212_321296

noncomputable def x : ℝ := sorry

-- x is uniformly distributed between 200 and 300
axiom x_range : 200 ≤ x ∧ x < 300

-- Floor of square root of 2x is 25
axiom floor_sqrt_2x : ⌊Real.sqrt (2 * x)⌋ = 25

-- Define the event that floor of square root of x is 17
def event : Prop := ⌊Real.sqrt x⌋ = 17

-- Define the probability measure
noncomputable def P : Set ℝ → ℝ := sorry

-- Theorem statement
theorem probability_of_event :
  P {y : ℝ | 200 ≤ y ∧ y < 300 ∧ ⌊Real.sqrt (2 * y)⌋ = 25 ∧ ⌊Real.sqrt y⌋ = 17} / 
  P {y : ℝ | 200 ≤ y ∧ y < 300} = 23 / 200 := by sorry

end NUMINAMATH_CALUDE_probability_of_event_l3212_321296


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l3212_321214

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/3, 29/3, 8/3) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, -1, 5)
  let C : ℝ × ℝ × ℝ := (1, 5, 2)
  orthocenter A B C = (5/3, 29/3, 8/3) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l3212_321214


namespace NUMINAMATH_CALUDE_tina_fruit_difference_l3212_321235

/-- Calculates the difference between remaining tangerines and oranges in Tina's bag --/
def remaining_difference (initial_oranges initial_tangerines removed_oranges removed_tangerines : ℕ) : ℕ :=
  (initial_tangerines - removed_tangerines) - (initial_oranges - removed_oranges)

/-- Proves that the difference between remaining tangerines and oranges is 4 --/
theorem tina_fruit_difference :
  remaining_difference 5 17 2 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_fruit_difference_l3212_321235


namespace NUMINAMATH_CALUDE_binomial_coefficient_28_5_l3212_321220

theorem binomial_coefficient_28_5 (h1 : Nat.choose 26 3 = 2600)
                                  (h2 : Nat.choose 26 4 = 14950)
                                  (h3 : Nat.choose 26 5 = 65780) :
  Nat.choose 28 5 = 98280 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_28_5_l3212_321220


namespace NUMINAMATH_CALUDE_probability_at_least_four_successes_l3212_321279

theorem probability_at_least_four_successes (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 3/5) :
  let binomial := fun (k : ℕ) => n.choose k * p^k * (1 - p)^(n - k)
  binomial 4 + binomial 5 = 1053/3125 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_four_successes_l3212_321279


namespace NUMINAMATH_CALUDE_basketball_pricing_solution_l3212_321244

/-- Represents the cost and pricing of basketballs --/
structure BasketballPricing where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for remaining A brand basketballs

/-- Theorem stating the correct pricing and discount for the basketball problem --/
theorem basketball_pricing_solution (p : BasketballPricing) : 
  (40 * p.cost_a + 40 * p.cost_b = 7200) →
  (50 * p.cost_a + 30 * p.cost_b = 7400) →
  (p.price_a = 140) →
  (p.markup_b = 0.3) →
  (40 * (p.price_a - p.cost_a) + 10 * (p.price_a * (1 - p.discount_a / 100) - p.cost_a) + 30 * p.cost_b * p.markup_b = 2440) →
  (p.cost_a = 100 ∧ p.cost_b = 80 ∧ p.discount_a = 20) := by
  sorry

end NUMINAMATH_CALUDE_basketball_pricing_solution_l3212_321244


namespace NUMINAMATH_CALUDE_triangle_area_l3212_321210

/-- The area of a triangle with sides 9, 40, and 41 is 180 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 →
  (a * a + b * b = c * c) → (1/2 : ℝ) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3212_321210


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l3212_321204

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ l > 0) 
  (h_diagonal : a^2 + b^2 + c^2 = l^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l3212_321204


namespace NUMINAMATH_CALUDE_equation_solutions_l3212_321261

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 - 9 = 0 ↔ x = 5/2 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3212_321261


namespace NUMINAMATH_CALUDE_newborn_count_l3212_321293

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 → 
  toddlers = 6 → 
  total_children = toddlers + 5 * toddlers + (total_children - toddlers - 5 * toddlers) → 
  (total_children - toddlers - 5 * toddlers) = 4 := by
sorry

end NUMINAMATH_CALUDE_newborn_count_l3212_321293


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l3212_321223

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4

/-- The area of the triangle formed by the parabola and the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 4)^(3/2)

/-- Theorem stating the relationship between r and the triangle area -/
theorem triangle_area_bounds (r : ℝ) :
  (16 ≤ triangleArea r ∧ triangleArea r ≤ 128) ↔ (8/3 ≤ r ∧ r ≤ 52/3) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l3212_321223


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3212_321269

theorem min_value_of_expression (c d : ℤ) (h : c^2 > d^2) :
  (((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) : ℚ) ≥ 2 ∧
  ∃ (c d : ℤ), c^2 > d^2 ∧ ((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3212_321269


namespace NUMINAMATH_CALUDE_t_values_l3212_321205

theorem t_values (t : ℝ) : 
  let M : Set ℝ := {1, 3, t}
  let N : Set ℝ := {t^2 - t + 1}
  (M ∪ N = M) → (t = 0 ∨ t = 2 ∨ t = -1) := by
sorry

end NUMINAMATH_CALUDE_t_values_l3212_321205


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_6_l3212_321283

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

end NUMINAMATH_CALUDE_hyperbola_center_is_3_6_l3212_321283


namespace NUMINAMATH_CALUDE_largest_square_side_length_is_correct_l3212_321232

/-- The side length of the largest square that can be formed using given sets of sticks -/
def largest_square_side_length : ℕ := 13

/-- Set of sticks for the first side -/
def side1 : List ℕ := [4, 4, 2, 3]

/-- Set of sticks for the second side -/
def side2 : List ℕ := [4, 4, 3, 1, 1]

/-- Set of sticks for the third side -/
def side3 : List ℕ := [4, 3, 3, 2, 1]

/-- Set of sticks for the fourth side -/
def side4 : List ℕ := [3, 3, 3, 2, 2]

/-- Theorem stating that the largest square side length is correct -/
theorem largest_square_side_length_is_correct :
  largest_square_side_length = side1.sum ∧
  largest_square_side_length = side2.sum ∧
  largest_square_side_length = side3.sum ∧
  largest_square_side_length = side4.sum :=
by sorry

end NUMINAMATH_CALUDE_largest_square_side_length_is_correct_l3212_321232


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3212_321292

theorem sqrt_equation_solution :
  ∃ t : ℝ, t = 3.7 ∧ Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3212_321292


namespace NUMINAMATH_CALUDE_range_of_fraction_l3212_321202

theorem range_of_fraction (x y : ℝ) (h : (x - 1)^2 + y^2 = 1) :
  ∃ (k : ℝ), y / (x + 1) = k ∧ -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3212_321202


namespace NUMINAMATH_CALUDE_division_remainder_l3212_321239

theorem division_remainder (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R) ∧ 
  (∃ Q : ℕ, N = 39 * Q + 15) → 
  ∃ Q' : ℕ, N = 44 * Q' + 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3212_321239


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l3212_321200

/-- Given a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between its asymptotes is 45°, then a/b = √2 + 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (2 * (b/a) / (1 - (b/a)^2)) = π/4) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l3212_321200


namespace NUMINAMATH_CALUDE_popcorn_shrimp_orders_l3212_321287

theorem popcorn_shrimp_orders (catfish_price popcorn_price : ℚ)
  (total_orders : ℕ) (total_amount : ℚ)
  (h1 : catfish_price = 6)
  (h2 : popcorn_price = (7/2))
  (h3 : total_orders = 26)
  (h4 : total_amount = (267/2)) :
  ∃ (catfish_orders popcorn_orders : ℕ),
    catfish_orders + popcorn_orders = total_orders ∧
    catfish_price * catfish_orders + popcorn_price * popcorn_orders = total_amount ∧
    popcorn_orders = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_shrimp_orders_l3212_321287
