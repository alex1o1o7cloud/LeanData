import Mathlib

namespace NUMINAMATH_CALUDE_ab_value_l792_79210

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l792_79210


namespace NUMINAMATH_CALUDE_lucy_liam_family_theorem_l792_79218

/-- Represents a family with siblings -/
structure Family where
  girls : Nat
  boys : Nat

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : Nat × Nat :=
  (f.girls, f.boys - 1)

/-- The main theorem about Lucy and Liam's family -/
theorem lucy_liam_family_theorem : 
  ∀ (f : Family), 
  f.girls = 5 → f.boys = 7 → 
  let (s, b) := sibling_count f
  s * b = 25 := by
  sorry

#check lucy_liam_family_theorem

end NUMINAMATH_CALUDE_lucy_liam_family_theorem_l792_79218


namespace NUMINAMATH_CALUDE_max_interval_length_l792_79214

theorem max_interval_length (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
  (b - a) ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_interval_length_l792_79214


namespace NUMINAMATH_CALUDE_kannon_fruit_consumption_l792_79226

/-- Represents the number of fruits Kannon ate last night -/
structure LastNightFruits where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ
  strawberries : ℕ
  kiwis : ℕ

/-- Represents the number of fruits Kannon will eat today -/
structure TodayFruits where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ
  strawberries : ℕ
  kiwis : ℕ

/-- Calculates the total number of fruits eaten over two days -/
def totalFruits (last : LastNightFruits) (today : TodayFruits) : ℕ :=
  last.apples + last.bananas + last.oranges + last.strawberries + last.kiwis +
  today.apples + today.bananas + today.oranges + today.strawberries + today.kiwis

/-- Theorem stating that the total number of fruits eaten is 54 -/
theorem kannon_fruit_consumption :
  ∀ (last : LastNightFruits) (today : TodayFruits),
  last.apples = 3 ∧ last.bananas = 1 ∧ last.oranges = 4 ∧ last.strawberries = 2 ∧ last.kiwis = 3 →
  today.apples = last.apples + 4 →
  today.bananas = 10 * last.bananas →
  today.oranges = 2 * today.apples →
  today.strawberries = (3 * last.oranges) / 2 →
  today.kiwis = today.bananas - 3 →
  totalFruits last today = 54 := by
  sorry


end NUMINAMATH_CALUDE_kannon_fruit_consumption_l792_79226


namespace NUMINAMATH_CALUDE_line_equation_theorem_l792_79251

-- Define the line l
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ
  xIntercept : ℝ
  yIntercept : ℝ

-- Define the conditions
def lineConditions (l : Line) : Prop :=
  l.passesThrough = (2, 3) ∧
  l.slope = Real.tan (2 * Real.pi / 3) ∧
  l.xIntercept + l.yIntercept = 0

-- Define the possible equations of the line
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  (3 * x - 2 * y = 0) ∨ (x - y + 1 = 0)

-- The theorem to prove
theorem line_equation_theorem (l : Line) :
  lineConditions l → ∀ x y, lineEquation l x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l792_79251


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_forty_l792_79202

theorem prime_square_minus_one_divisible_by_forty (p : ℕ) 
  (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) : 
  40 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_forty_l792_79202


namespace NUMINAMATH_CALUDE_range_of_m_l792_79223

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l792_79223


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l792_79291

theorem quadratic_roots_problem (c d : ℝ) (r s : ℝ) : 
  c^2 - 5*c + 3 = 0 →
  d^2 - 5*d + 3 = 0 →
  (c + 2/d)^2 - r*(c + 2/d) + s = 0 →
  (d + 2/c)^2 - r*(d + 2/c) + s = 0 →
  s = 25/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l792_79291


namespace NUMINAMATH_CALUDE_f_one_zero_implies_a_gt_one_l792_79246

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem f_one_zero_implies_a_gt_one (a : ℝ) :
  (∃! x, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

#check f_one_zero_implies_a_gt_one

end NUMINAMATH_CALUDE_f_one_zero_implies_a_gt_one_l792_79246


namespace NUMINAMATH_CALUDE_toothpick_pattern_sum_l792_79220

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem toothpick_pattern_sum :
  arithmeticSum 6 5 150 = 56775 := by sorry

end NUMINAMATH_CALUDE_toothpick_pattern_sum_l792_79220


namespace NUMINAMATH_CALUDE_chunks_for_two_dozen_bananas_l792_79201

/-- The number of chunks needed to purchase a given number of bananas -/
def chunks_needed (bananas : ℚ) : ℚ :=
  (bananas * 3 * 8) / (7 * 5)

theorem chunks_for_two_dozen_bananas :
  chunks_needed 24 = 576 / 35 := by
  sorry

end NUMINAMATH_CALUDE_chunks_for_two_dozen_bananas_l792_79201


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l792_79203

/-- Hyperbola M with equation x^2 - y^2/b^2 = 1 -/
def hyperbola_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 1

/-- Line l with slope 1 passing through the left vertex (-1, 0) -/
def line_l (x y : ℝ) : Prop :=
  y = x + 1

/-- Asymptotes of hyperbola M -/
def asymptotes_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 0

/-- Point A is the left vertex of hyperbola M -/
def point_A : ℝ × ℝ :=
  (-1, 0)

/-- Point B is the intersection of line l and one asymptote -/
def point_B (b : ℝ) : ℝ × ℝ :=
  sorry

/-- Point C is the intersection of line l and the other asymptote -/
def point_C (b : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The eccentricity of hyperbola M -/
def eccentricity (b : ℝ) : ℝ :=
  sorry

theorem hyperbola_eccentricity (b : ℝ) :
  hyperbola_M b (point_A.1) (point_A.2) →
  line_l (point_B b).1 (point_B b).2 →
  line_l (point_C b).1 (point_C b).2 →
  asymptotes_M b (point_B b).1 (point_B b).2 →
  asymptotes_M b (point_C b).1 (point_C b).2 →
  distance point_A (point_B b) = distance (point_B b) (point_C b) →
  eccentricity b = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l792_79203


namespace NUMINAMATH_CALUDE_arrangement_theorem_l792_79260

def num_girls : ℕ := 3
def num_boys : ℕ := 5

def arrangements_girls_together : ℕ := 4320
def arrangements_girls_separate : ℕ := 14400

theorem arrangement_theorem :
  (num_girls = 3 ∧ num_boys = 5) →
  (arrangements_girls_together = 4320 ∧ arrangements_girls_separate = 14400) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l792_79260


namespace NUMINAMATH_CALUDE_probability_no_adjacent_standing_l792_79245

/-- The number of valid arrangements for n people in a circle where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The total number of possible outcomes when n people flip coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

theorem probability_no_adjacent_standing (n : ℕ) : 
  n = 8 → (validArrangements n : ℚ) / totalOutcomes n = 47 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_standing_l792_79245


namespace NUMINAMATH_CALUDE_problem_statement_l792_79230

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 174 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l792_79230


namespace NUMINAMATH_CALUDE_missing_number_proof_l792_79274

theorem missing_number_proof : ∃ x : ℤ, |7 - 8 * (3 - x)| - |5 - 11| = 73 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l792_79274


namespace NUMINAMATH_CALUDE_solution_of_equation_l792_79244

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) : Prop := 2 + i * x = -2 - 2 * i * x

-- State the theorem
theorem solution_of_equation :
  ∃ (x : ℂ), equation x ∧ x = (4 * i) / 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l792_79244


namespace NUMINAMATH_CALUDE_lcm_of_coprime_product_l792_79284

theorem lcm_of_coprime_product (a b : ℕ+) (h_coprime : Nat.Coprime a b) (h_product : a * b = 117) :
  Nat.lcm a b = 117 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_coprime_product_l792_79284


namespace NUMINAMATH_CALUDE_triangle_shape_l792_79256

/-- Given a triangle ABC, prove that it is a right isosceles triangle under certain conditions. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (Real.log a - Real.log c = Real.log (Real.sin B)) →
  (Real.log (Real.sin B) = -Real.log (Real.sqrt 2)) →
  (0 < B) →
  (B < π / 2) →
  (A + B + C = π) →
  (a * Real.sin C = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin B) →
  (A = π / 4 ∧ B = π / 4 ∧ C = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l792_79256


namespace NUMINAMATH_CALUDE_x_equation_result_l792_79225

theorem x_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 3*x^5 + x^2 = -5*x + 4*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_result_l792_79225


namespace NUMINAMATH_CALUDE_fourth_cubed_decimal_l792_79228

theorem fourth_cubed_decimal : (1/4)^3 = 0.015625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_cubed_decimal_l792_79228


namespace NUMINAMATH_CALUDE_percent_division_multiplication_equality_l792_79272

theorem percent_division_multiplication_equality : 
  (30 / 100 : ℚ) / (1 + 2 / 5) * (1 / 3 + 1 / 7) = 5 / 49 := by sorry

end NUMINAMATH_CALUDE_percent_division_multiplication_equality_l792_79272


namespace NUMINAMATH_CALUDE_loan_interest_difference_l792_79208

/-- Proves that for a loan of 2000 at 3% simple interest for 3 years, 
    the difference between the principal and the interest is 1940 -/
theorem loan_interest_difference : 
  let principal : ℚ := 2000
  let rate : ℚ := 3 / 100
  let time : ℚ := 3
  let interest := principal * rate * time
  principal - interest = 1940 := by sorry

end NUMINAMATH_CALUDE_loan_interest_difference_l792_79208


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_l792_79298

/-- In a rectangular solid, if one of its diagonals forms angles α, β, and γ 
    with the three edges emanating from one of its vertices, 
    then cos²α + cos²β + cos²γ = 1 -/
theorem rectangular_solid_diagonal_angles (α β γ : Real) 
  (hα : α = angle_between_diagonal_and_edge1)
  (hβ : β = angle_between_diagonal_and_edge2)
  (hγ : γ = angle_between_diagonal_and_edge3)
  (h_rectangular_solid : is_rectangular_solid) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_l792_79298


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l792_79296

/-- Given a line with slope -5 passing through (4, 2), prove that m + b = 17 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -5 → 
  2 = m * 4 + b → 
  m + b = 17 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l792_79296


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l792_79217

/-- Defines whether an equation represents an ellipse -/
def IsEllipse (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

/-- The condition on m -/
def Condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, IsEllipse m → Condition m) ∧
  (∃ m : ℝ, Condition m ∧ ¬IsEllipse m) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l792_79217


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l792_79279

theorem last_three_digits_of_7_to_103 : 7^103 ≡ 614 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l792_79279


namespace NUMINAMATH_CALUDE_bug_total_distance_l792_79278

def bug_path : List ℤ := [4, -3, 6, 2]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

theorem bug_total_distance :
  (List.zip bug_path bug_path.tail).foldl (λ acc (a, b) => acc + distance a b) 0 = 20 :=
by sorry

end NUMINAMATH_CALUDE_bug_total_distance_l792_79278


namespace NUMINAMATH_CALUDE_eugene_initial_pencils_l792_79250

/-- The number of pencils Eugene initially had -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Eugene gave to Joyce -/
def pencils_given : ℕ := 6

/-- The number of pencils Eugene has left -/
def pencils_left : ℕ := 45

/-- Theorem: Eugene initially had 51 pencils -/
theorem eugene_initial_pencils :
  initial_pencils = pencils_given + pencils_left ∧ initial_pencils = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_initial_pencils_l792_79250


namespace NUMINAMATH_CALUDE_quadratic_one_solution_quadratic_one_solution_positive_l792_79258

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution -/
theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution is 36 -/
theorem quadratic_one_solution_positive (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ m > 0 → m = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_quadratic_one_solution_positive_l792_79258


namespace NUMINAMATH_CALUDE_factor_expression_l792_79267

theorem factor_expression (b : ℝ) : 29*b^2 + 87*b = 29*b*(b+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l792_79267


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l792_79268

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 32) 
  (h_a6 : a 6 = -1) : 
  q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l792_79268


namespace NUMINAMATH_CALUDE_average_income_q_and_r_l792_79237

/-- Given the average monthly incomes of P and Q, P and R, and P's income,
    prove that the average monthly income of Q and R is 6250. -/
theorem average_income_q_and_r (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (p + r) / 2 = 5200 →
  p = 4000 →
  (q + r) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_q_and_r_l792_79237


namespace NUMINAMATH_CALUDE_average_running_time_l792_79235

theorem average_running_time (sixth_grade_time seventh_grade_time eighth_grade_time : ℝ)
  (sixth_to_eighth_ratio sixth_to_seventh_ratio : ℝ) :
  sixth_grade_time = 10 →
  seventh_grade_time = 18 →
  eighth_grade_time = 14 →
  sixth_to_eighth_ratio = 3 →
  sixth_to_seventh_ratio = 3/2 →
  let e := 1  -- Assuming 1 eighth grader for simplicity
  let sixth_count := e * sixth_to_eighth_ratio
  let seventh_count := sixth_count / sixth_to_seventh_ratio
  let eighth_count := e
  let total_time := sixth_grade_time * sixth_count + 
                    seventh_grade_time * seventh_count + 
                    eighth_grade_time * eighth_count
  let total_students := sixth_count + seventh_count + eighth_count
  total_time / total_students = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_average_running_time_l792_79235


namespace NUMINAMATH_CALUDE_total_elixir_ways_l792_79252

/-- The number of ways to prepare magical dust -/
def total_magical_dust_ways : ℕ := 4

/-- The number of elixirs made from fairy dust -/
def fairy_dust_elixirs : ℕ := 3

/-- The number of elixirs made from elf dust -/
def elf_dust_elixirs : ℕ := 4

/-- The number of ways to prepare fairy dust -/
def fairy_dust_ways : ℕ := 2

/-- The number of ways to prepare elf dust -/
def elf_dust_ways : ℕ := 2

/-- Theorem: The total number of ways to prepare all the elixirs is 14 -/
theorem total_elixir_ways : 
  fairy_dust_ways * fairy_dust_elixirs + elf_dust_ways * elf_dust_elixirs = 14 :=
by sorry

end NUMINAMATH_CALUDE_total_elixir_ways_l792_79252


namespace NUMINAMATH_CALUDE_solve_exam_problem_l792_79286

def exam_problem (exam_A_total exam_B_total exam_A_wrong exam_B_correct_diff : ℕ) : Prop :=
  let exam_A_correct := exam_A_total - exam_A_wrong
  let exam_B_correct := exam_A_correct + exam_B_correct_diff
  let exam_B_wrong := exam_B_total - exam_B_correct
  exam_A_wrong + exam_B_wrong = 9

theorem solve_exam_problem :
  exam_problem 12 15 4 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exam_problem_l792_79286


namespace NUMINAMATH_CALUDE_parallel_line_distance_l792_79233

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between two adjacent parallel lines -/
  line_distance : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 30 -/
  chord1_eq : chord1 = 30
  /-- The second chord has length 40 -/
  chord2_eq : chord2 = 40
  /-- The third chord has length 30 -/
  chord3_eq : chord3 = 30

/-- Theorem: The distance between two adjacent parallel lines is 2√30 -/
theorem parallel_line_distance (c : CircleWithParallelLines) :
  c.line_distance = 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_distance_l792_79233


namespace NUMINAMATH_CALUDE_expand_and_simplify_l792_79288

theorem expand_and_simplify (y : ℝ) : -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l792_79288


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l792_79262

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x + 1)) = (2 / (x - 1))
def equation2 (x : ℝ) : Prop := (2 * x + 9) / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ -1 ∧ x ≠ 1 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 3) := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l792_79262


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l792_79205

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (697 * n ≡ 1421 * n [ZMOD 36]) ∧ 
  ∀ (m : ℕ), m > 0 → (697 * m ≡ 1421 * m [ZMOD 36]) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l792_79205


namespace NUMINAMATH_CALUDE_dividend_divisor_problem_l792_79259

theorem dividend_divisor_problem (a b : ℕ+) : 
  (a : ℚ) / b = (b : ℚ) + (a : ℚ) / 10 → a = 5 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_dividend_divisor_problem_l792_79259


namespace NUMINAMATH_CALUDE_fruit_juice_volume_l792_79224

/-- Proves that the volume of fruit juice in Carrie's punch is 40 oz -/
theorem fruit_juice_volume (total_punch : ℕ) (mountain_dew : ℕ) (ice : ℕ) :
  total_punch = 140 ∧ mountain_dew = 72 ∧ ice = 28 →
  ∃ (fruit_juice : ℕ), total_punch = mountain_dew + ice + fruit_juice ∧ fruit_juice = 40 := by
sorry

end NUMINAMATH_CALUDE_fruit_juice_volume_l792_79224


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_roots_l792_79241

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) : 
  let discriminant := b^2 - 4*a*c
  discriminant > 0 → ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by
  sorry

theorem specific_quadratic_roots : 
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - 6 = 0 ∧ y^2 - 2*y - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_roots_l792_79241


namespace NUMINAMATH_CALUDE_distance_between_vertices_l792_79257

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y + 2| = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y ≥ -2

def parabola2 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y < -2

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 1)
def vertex2 : ℝ × ℝ := (0, -3)

-- Theorem statement
theorem distance_between_vertices :
  ∃ (v1 v2 : ℝ × ℝ),
    (∀ x y, parabola1 x y → (x, y) = v1 ∨ y > v1.2) ∧
    (∀ x y, parabola2 x y → (x, y) = v2 ∨ y < v2.2) ∧
    ‖v1 - v2‖ = 4 :=
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l792_79257


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l792_79261

/-- The line equation (m-1)x + (2m-1)y = m-5 always passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l792_79261


namespace NUMINAMATH_CALUDE_total_book_pages_l792_79236

def book_pages : ℕ → ℕ
| 1 => 25
| 2 => 2 * book_pages 1
| 3 => 2 * book_pages 2
| 4 => 10
| _ => 0

def pages_written : ℕ := book_pages 1 + book_pages 2 + book_pages 3 + book_pages 4

def remaining_pages : ℕ := 315

theorem total_book_pages : pages_written + remaining_pages = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_book_pages_l792_79236


namespace NUMINAMATH_CALUDE_non_trivial_solution_exists_l792_79238

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) (hp : Nat.Prime p) :
  ∃ x y z : ℤ, (x, y, z) ≠ (0, 0, 0) ∧ (a * x^2 + b * y^2 + c * z^2) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_non_trivial_solution_exists_l792_79238


namespace NUMINAMATH_CALUDE_illegal_parking_percentage_l792_79299

theorem illegal_parking_percentage
  (total_cars : ℝ)
  (towed_percentage : ℝ)
  (not_towed_percentage : ℝ)
  (h1 : towed_percentage = 0.02)
  (h2 : not_towed_percentage = 0.80)
  (h3 : total_cars > 0) :
  let towed_cars := towed_percentage * total_cars
  let illegally_parked_cars := towed_cars / (1 - not_towed_percentage)
  illegally_parked_cars / total_cars = 0.10 := by
sorry

end NUMINAMATH_CALUDE_illegal_parking_percentage_l792_79299


namespace NUMINAMATH_CALUDE_distance_to_FA_l792_79269

/-- RegularHexagon represents a regular hexagon with a point inside -/
structure RegularHexagon where
  -- Point inside the hexagon
  P : Point
  -- Distances from P to each side
  dist_AB : ℝ
  dist_BC : ℝ
  dist_CD : ℝ
  dist_DE : ℝ
  dist_EF : ℝ
  dist_FA : ℝ

/-- Theorem stating the distance from P to FA in the given hexagon -/
theorem distance_to_FA (h : RegularHexagon)
  (h_AB : h.dist_AB = 1)
  (h_BC : h.dist_BC = 2)
  (h_CD : h.dist_CD = 5)
  (h_DE : h.dist_DE = 7)
  (h_EF : h.dist_EF = 6)
  : h.dist_FA = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_FA_l792_79269


namespace NUMINAMATH_CALUDE_sum_a_d_l792_79243

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 5) 
  (h2 : b + c = 6) 
  (h3 : c + d = 3) : 
  a + d = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l792_79243


namespace NUMINAMATH_CALUDE_circle_tangency_distance_l792_79200

theorem circle_tangency_distance (r_O r_O' d_external : ℝ) : 
  r_O = 5 → 
  d_external = 9 → 
  r_O + r_O' = d_external → 
  |r_O' - r_O| = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_distance_l792_79200


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l792_79265

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) :
  (y = 4 * x - 1) ∧
  (y = -3 * x + 9) ∧
  (y = 2 * x + k) →
  k = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l792_79265


namespace NUMINAMATH_CALUDE_statement_1_statement_4_l792_79277

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the axioms
variable (m n : Line)
variable (α β : Plane)
variable (h_different_lines : m ≠ n)
variable (h_non_coincident_planes : α ≠ β)

-- Statement 1
theorem statement_1 : 
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

-- Statement 4
theorem statement_4 :
  perpendicular m α → line_parallel_plane m β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_l792_79277


namespace NUMINAMATH_CALUDE_hyperbola_properties_l792_79273

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def distance_to_asymptote (h : Hyperbola) (l : Line) : ℝ := sorry

def standard_equation (h : Hyperbola) : Prop :=
  h.a = 1 ∧ h.b = 2

def slope_ratio (h : Hyperbola) (l : Line) : ℝ := sorry

def fixed_point_exists (h : Hyperbola) : Prop :=
  ∃ (G : Point), G.x = 1 ∧ G.y = 0 ∧
  ∀ (l : Line), slope_ratio h l = -1/3 →
  ∃ (H : Point), (H.x - G.x)^2 + (H.y - G.y)^2 = 1

theorem hyperbola_properties (h : Hyperbola) 
  (asymptote : Line)
  (h_asymptote : asymptote.m = 2 ∧ asymptote.c = 0)
  (h_distance : distance_to_asymptote h asymptote = 2) :
  standard_equation h ∧ fixed_point_exists h := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l792_79273


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l792_79294

theorem trader_gain_percentage : 
  ∀ (cost_per_pen : ℝ), cost_per_pen > 0 →
  (19 * cost_per_pen) / (95 * cost_per_pen) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l792_79294


namespace NUMINAMATH_CALUDE_john_earnings_is_80_l792_79234

/-- Calculates the amount of money John makes repairing cars --/
def john_earnings (total_cars : ℕ) (standard_repair_time : ℕ) (longer_repair_percentage : ℚ) (hourly_rate : ℚ) : ℚ :=
  let standard_cars := 3
  let longer_cars := total_cars - standard_cars
  let standard_time := standard_cars * standard_repair_time
  let longer_time := longer_cars * (standard_repair_time * (1 + longer_repair_percentage))
  let total_time := standard_time + longer_time
  let total_hours := total_time / 60
  total_hours * hourly_rate

/-- Theorem stating that John makes $80 repairing cars --/
theorem john_earnings_is_80 :
  john_earnings 5 40 (1/2) 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_is_80_l792_79234


namespace NUMINAMATH_CALUDE_three_solutions_l792_79290

/-- A structure representing a solution to the equation AB = B^V --/
structure Solution :=
  (a b v : Nat)
  (h1 : a ≠ b ∧ a ≠ v ∧ b ≠ v)
  (h2 : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ v > 0 ∧ v < 10)
  (h3 : 10 * a + b = b^v)

/-- The set of all valid solutions --/
def allSolutions : Set Solution := {s | s.a * 10 + s.b ≥ 10 ∧ s.a * 10 + s.b < 100}

/-- The theorem stating that there are exactly three solutions --/
theorem three_solutions :
  ∃ (s1 s2 s3 : Solution),
    s1 ∈ allSolutions ∧ 
    s2 ∈ allSolutions ∧ 
    s3 ∈ allSolutions ∧
    s1.a = 3 ∧ s1.b = 2 ∧ s1.v = 5 ∧
    s2.a = 3 ∧ s2.b = 6 ∧ s2.v = 2 ∧
    s3.a = 6 ∧ s3.b = 4 ∧ s3.v = 3 ∧
    ∀ (s : Solution), s ∈ allSolutions → s = s1 ∨ s = s2 ∨ s = s3 :=
  sorry


end NUMINAMATH_CALUDE_three_solutions_l792_79290


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l792_79240

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l792_79240


namespace NUMINAMATH_CALUDE_cosine_roots_condition_l792_79254

theorem cosine_roots_condition (p q r : ℝ) : 
  (∃ a b c : ℝ, 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a^3 + p*a^2 + q*a + r = 0) ∧
    (b^3 + p*b^2 + q*b + r = 0) ∧
    (c^3 + p*c^2 + q*c + r = 0) ∧
    (∃ α β γ : ℝ, 
      α + β + γ = Real.pi ∧
      a = Real.cos α ∧
      b = Real.cos β ∧
      c = Real.cos γ)) →
  p^2 = 2*q + 2*r + 1 :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_condition_l792_79254


namespace NUMINAMATH_CALUDE_prob_white_given_red_is_two_ninths_l792_79213

/-- The number of red balls in the box -/
def num_red : ℕ := 3

/-- The number of white balls in the box -/
def num_white : ℕ := 2

/-- The number of black balls in the box -/
def num_black : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := num_red + num_white + num_black

/-- The probability of picking a white ball on the second draw given that the first ball picked is red -/
def prob_white_given_red : ℚ := num_white / (total_balls - 1)

theorem prob_white_given_red_is_two_ninths :
  prob_white_given_red = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_given_red_is_two_ninths_l792_79213


namespace NUMINAMATH_CALUDE_smallest_k_proof_l792_79232

def is_perfect_cube (m : ℤ) : Prop := ∃ n : ℤ, m = n^3

def smallest_k : ℕ := 60

theorem smallest_k_proof : 
  (∀ k : ℕ, k < smallest_k → ¬ is_perfect_cube (2^4 * 3^2 * 5^5 * k)) ∧ 
  is_perfect_cube (2^4 * 3^2 * 5^5 * smallest_k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_proof_l792_79232


namespace NUMINAMATH_CALUDE_train_length_l792_79271

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, abs (length - 74.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l792_79271


namespace NUMINAMATH_CALUDE_cube_root_sum_equation_l792_79255

theorem cube_root_sum_equation (y : ℝ) (hy : y > 0) 
  (h : Real.rpow (2 - y^3) (1/3) + Real.rpow (2 + y^3) (1/3) = 2) : 
  y^6 = 116/27 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equation_l792_79255


namespace NUMINAMATH_CALUDE_tangent_line_equation_l792_79239

def parabola (x : ℝ) : ℝ := x^2 + x + 1

theorem tangent_line_equation :
  let f := parabola
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m := (deriv f) x₀
  ∀ x y, y - y₀ = m * (x - x₀) ↔ x - y + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l792_79239


namespace NUMINAMATH_CALUDE_balloon_arrangements_l792_79231

-- Define the word length and repeated letter counts
def word_length : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Theorem statement
theorem balloon_arrangements : 
  (Nat.factorial word_length) / (Nat.factorial l_count * Nat.factorial o_count) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l792_79231


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l792_79207

theorem sqrt_equation_solution (a x : ℝ) (ha : 0 < a ∧ a ≤ 1) (hx : x ≥ 1) :
  Real.sqrt (x + Real.sqrt x) - Real.sqrt (x - Real.sqrt x) = (a + 1) * Real.sqrt (x / (x + Real.sqrt x)) →
  x = ((a^2 + 1) / (2*a))^2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l792_79207


namespace NUMINAMATH_CALUDE_system_solution_l792_79293

theorem system_solution (x y k : ℝ) 
  (eq1 : x - y = k + 2)
  (eq2 : x + 3*y = k)
  (eq3 : x + y = 2) :
  k = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l792_79293


namespace NUMINAMATH_CALUDE_infinitely_many_skew_lines_l792_79211

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if a line is within a plane -/
def within_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_skew_lines 
  (l : Line3D) (α : Plane3D) 
  (h1 : intersects l α) 
  (h2 : ¬perpendicular l α) :
  ∃ S : Set Line3D, (∀ l' ∈ S, within_plane l' α ∧ skew l l') ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_skew_lines_l792_79211


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l792_79280

/-- Given that x varies inversely as square of y, prove that x = 1/9 when y = 6,
    given that y = 2 when x = 1 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
  y = 6 → x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l792_79280


namespace NUMINAMATH_CALUDE_product_inequality_l792_79295

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l792_79295


namespace NUMINAMATH_CALUDE_simplify_square_root_l792_79215

theorem simplify_square_root (x y : ℝ) (h1 : x * y < 0) (h2 : -y / x^2 ≥ 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) :=
sorry

end NUMINAMATH_CALUDE_simplify_square_root_l792_79215


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l792_79289

theorem sqrt_50_between_consecutive_integers : 
  ∃ n : ℕ, n > 0 ∧ n < Real.sqrt 50 ∧ Real.sqrt 50 < n + 1 ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l792_79289


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l792_79282

/-- The number of flips performed -/
def num_flips : ℕ := 10

/-- The number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- The total number of distinct sequences possible -/
def total_sequences : ℕ := outcomes_per_flip ^ num_flips

theorem coin_flip_sequences :
  total_sequences = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l792_79282


namespace NUMINAMATH_CALUDE_parabola_intersection_l792_79219

theorem parabola_intersection (m : ℝ) : 
  (m > 0) →
  (∃! x : ℝ, -1 < x ∧ x < 4 ∧ -x^2 + 4*x - 2 + m = 0) →
  (2 ≤ m ∧ m < 7) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l792_79219


namespace NUMINAMATH_CALUDE_min_daily_expense_l792_79253

/-- Represents the daily transport capacity of a truck --/
def DailyCapacity (capacity : ℕ) (trips : ℕ) : ℕ := capacity * trips

/-- Represents the total daily capacity of a fleet of trucks --/
def FleetCapacity (trucks : ℕ) (dailyCapacity : ℕ) : ℕ := trucks * dailyCapacity

/-- Represents the daily cost for a fleet of trucks --/
def FleetCost (trucks : ℕ) (cost : ℕ) : ℕ := trucks * cost

/-- The minimum daily expense problem --/
theorem min_daily_expense :
  let typeA_capacity : ℕ := 6
  let typeA_trips : ℕ := 4
  let typeA_available : ℕ := 8
  let typeA_cost : ℕ := 320
  let typeB_capacity : ℕ := 10
  let typeB_trips : ℕ := 3
  let typeB_available : ℕ := 4
  let typeB_cost : ℕ := 504
  let daily_requirement : ℕ := 180
  let typeA_daily_capacity := DailyCapacity typeA_capacity typeA_trips
  let typeB_daily_capacity := DailyCapacity typeB_capacity typeB_trips
  ∀ x y : ℕ,
    x ≤ typeA_available →
    y ≤ typeB_available →
    FleetCapacity x typeA_daily_capacity + FleetCapacity y typeB_daily_capacity ≥ daily_requirement →
    FleetCost x typeA_cost + FleetCost y typeB_cost ≥ FleetCost typeA_available typeA_cost :=
by sorry

end NUMINAMATH_CALUDE_min_daily_expense_l792_79253


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l792_79209

/-- A quadratic function with vertex (2, 9) and one x-intercept at (3, 0) has its other x-intercept at x = 1 -/
theorem quadratic_other_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 9 - a * (x - 2)^2) →  -- vertex form with vertex (2, 9)
  a * 3^2 + b * 3 + c = 0 →                         -- x-intercept at (3, 0)
  a * 1^2 + b * 1 + c = 0 :=                        -- other x-intercept at (1, 0)
by sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l792_79209


namespace NUMINAMATH_CALUDE_circle_x_intersection_l792_79281

theorem circle_x_intersection (x : ℝ) : 
  let center_x := (-2 + 6) / 2
  let center_y := (1 + 9) / 2
  let radius := Real.sqrt (((-2 - center_x)^2 + (1 - center_y)^2) : ℝ)
  (x - center_x)^2 + (0 - center_y)^2 = radius^2 →
  x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_intersection_l792_79281


namespace NUMINAMATH_CALUDE_intersection_sum_l792_79216

theorem intersection_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → x = 8 ∧ y = 14) →
  b + m = -63/4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l792_79216


namespace NUMINAMATH_CALUDE_tangent_line_equation_l792_79242

/-- The equation of the tangent line to y = xe^(2x-1) at (1, e) is 3ex - y - 2e = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (2 * x - 1)) → -- Given curve equation
  (3 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0) ↔ -- Tangent line equation
  (y - Real.exp 1 = 3 * Real.exp 1 * (x - 1)) -- Point-slope form at (1, e)
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l792_79242


namespace NUMINAMATH_CALUDE_emily_scores_mean_l792_79292

def emily_scores : List ℕ := [84, 90, 93, 85, 91, 87]

theorem emily_scores_mean : 
  (emily_scores.sum : ℚ) / emily_scores.length = 530 / 6 := by
sorry

end NUMINAMATH_CALUDE_emily_scores_mean_l792_79292


namespace NUMINAMATH_CALUDE_distinct_intersection_points_l792_79212

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A point where at least two lines intersect -/
structure IntersectionPoint :=
  (lines : Finset Line)

/-- The set of all lines in the plane -/
def all_lines : Finset Line := sorry

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem distinct_intersection_points :
  (∀ l ∈ all_lines, ∀ l' ∈ all_lines, l ≠ l' → l.id ≠ l'.id) →  -- lines are distinct
  (Finset.card all_lines = 5) →  -- there are five lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≥ 2) →  -- each intersection point has at least two lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≤ 3) →  -- no more than three lines intersect at a point
  Finset.card intersection_points = 10 :=  -- there are 10 distinct intersection points
by sorry

end NUMINAMATH_CALUDE_distinct_intersection_points_l792_79212


namespace NUMINAMATH_CALUDE_min_sum_of_log_arithmetic_sequence_l792_79248

theorem min_sum_of_log_arithmetic_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x + Real.log y) / 2 = 2) : 
  (∀ a b : ℝ, a > 1 → b > 1 → (Real.log a + Real.log b) / 2 = 2 → x + y ≤ a + b) ∧ 
  ∃ a b : ℝ, a > 1 ∧ b > 1 ∧ (Real.log a + Real.log b) / 2 = 2 ∧ a + b = 200 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_log_arithmetic_sequence_l792_79248


namespace NUMINAMATH_CALUDE_tan_P_is_two_one_l792_79264

/-- Represents a right triangle PQR with altitude QS --/
structure RightTrianglePQR where
  -- Side lengths
  PQ : ℕ
  QR : ℕ
  PR : ℕ
  PS : ℕ
  -- PR = 3^5
  h_PR : PR = 3^5
  -- PS = 3^3
  h_PS : PS = 3^3
  -- Right angle at Q
  h_right_angle : PQ^2 + QR^2 = PR^2
  -- Altitude property
  h_altitude : PQ * PS = PR * QS

/-- The main theorem --/
theorem tan_P_is_two_one (t : RightTrianglePQR) : 
  (t.QR : ℚ) / t.PQ = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_P_is_two_one_l792_79264


namespace NUMINAMATH_CALUDE_constant_term_expansion_l792_79287

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f x = (x + 2) * (1/x - a*x)^7 ∧ 
   ∃ (g : ℝ → ℝ), (∀ x ≠ 0, f x = g x) ∧ g 0 = -280) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l792_79287


namespace NUMINAMATH_CALUDE_root_condition_implies_k_value_l792_79247

theorem root_condition_implies_k_value (a b c k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁^2 - (b+1)*x₁) / ((a+1)*x₁ - c) = (k-2)/(k+2) ∧
    (x₂^2 - (b+1)*x₂) / ((a+1)*x₂ - c) = (k-2)/(k+2) ∧
    x₁ = -x₂ ∧ x₁ ≠ 0) →
  k = (-2*(b-a))/(b+a+2) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_k_value_l792_79247


namespace NUMINAMATH_CALUDE_min_product_constrained_l792_79276

theorem min_product_constrained (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 3/125 := by
sorry

end NUMINAMATH_CALUDE_min_product_constrained_l792_79276


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l792_79275

-- (1)
theorem simplify_sqrt_expression : 
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = (5 * Real.sqrt 2) / 4 := by sorry

-- (2)
theorem simplify_algebraic_expression : 
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = Real.sqrt 3 - 1 := by sorry

-- (3)
theorem simplify_complex_sqrt_expression : 
  (Real.sqrt 15 + Real.sqrt 60) / Real.sqrt 3 - 3 * Real.sqrt 5 = -5 * Real.sqrt 5 := by sorry

-- (4)
theorem simplify_difference_of_squares : 
  (Real.sqrt 7 + Real.sqrt 3) * (Real.sqrt 7 - Real.sqrt 3) - Real.sqrt 36 = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l792_79275


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_96_l792_79204

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The shorter base of the trapezoid
  short_base : ℝ
  -- The perimeter of the trapezoid
  perimeter : ℝ
  -- The diagonal bisects the obtuse angle
  diagonal_bisects_obtuse_angle : Bool

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that an isosceles trapezoid with given properties has an area of 96 -/
theorem isosceles_trapezoid_area_is_96 (t : IsoscelesTrapezoid) 
  (h1 : t.short_base = 3)
  (h2 : t.perimeter = 42)
  (h3 : t.diagonal_bisects_obtuse_angle = true) :
  area t = 96 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_96_l792_79204


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_61_l792_79227

theorem smallest_x_multiple_of_61 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(61 ∣ (3*y)^2 + 3*58*3*y + 58^2)) ∧ 
  (61 ∣ (3*x)^2 + 3*58*3*x + 58^2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_61_l792_79227


namespace NUMINAMATH_CALUDE_trees_in_yard_l792_79283

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 434) (h2 : tree_distance = 14) :
  (yard_length / tree_distance) + 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l792_79283


namespace NUMINAMATH_CALUDE_balloon_count_l792_79285

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 :=
by sorry

end NUMINAMATH_CALUDE_balloon_count_l792_79285


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l792_79221

def sequence_term (n : ℕ) (b : ℝ) : ℝ := n^2 + b*n

theorem increasing_sequence_condition (b : ℝ) : 
  (∀ n : ℕ, sequence_term (n + 1) b > sequence_term n b) → b > -3 :=
by
  sorry

#check increasing_sequence_condition

end NUMINAMATH_CALUDE_increasing_sequence_condition_l792_79221


namespace NUMINAMATH_CALUDE_initial_water_percentage_l792_79266

theorem initial_water_percentage (container_capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  container_capacity = 40 →
  added_water = 18 →
  final_fraction = 3/4 →
  (container_capacity * final_fraction - added_water) / container_capacity * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l792_79266


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l792_79249

theorem binomial_coefficient_19_13 
  (h1 : (20 : ℕ).choose 13 = 77520)
  (h2 : (20 : ℕ).choose 14 = 38760)
  (h3 : (18 : ℕ).choose 13 = 18564) :
  (19 : ℕ).choose 13 = 37128 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l792_79249


namespace NUMINAMATH_CALUDE_fraction_evaluation_l792_79297

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l792_79297


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l792_79206

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 : ℝ) → 
  ∃ (n : ℤ), n = 10 ∧ 
  ∀ (m : ℤ), |x^(1/3) - (n : ℝ)| ≤ |x^(1/3) - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l792_79206


namespace NUMINAMATH_CALUDE_harmonious_point_in_third_quadrant_l792_79222

/-- A point (x, y) is harmonious if 3x = 2y + 5 -/
def IsHarmonious (x y : ℝ) : Prop := 3 * x = 2 * y + 5

/-- The x-coordinate of point M -/
def Mx (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point M -/
def My (m : ℝ) : ℝ := 3 * m + 2

theorem harmonious_point_in_third_quadrant :
  ∀ m : ℝ, IsHarmonious (Mx m) (My m) → Mx m < 0 ∧ My m < 0 := by
  sorry

end NUMINAMATH_CALUDE_harmonious_point_in_third_quadrant_l792_79222


namespace NUMINAMATH_CALUDE_product_sum_relation_l792_79263

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 11 → b = 7 → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l792_79263


namespace NUMINAMATH_CALUDE_correct_algebraic_operation_l792_79270

variable (x y : ℝ)

theorem correct_algebraic_operation : y * x - 3 * x * y = -2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_operation_l792_79270


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_given_hcf_lcm_l792_79229

theorem ratio_of_numbers_with_given_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 84 → Nat.lcm a b = 21 → max a b = 84 → 
  (max a b : ℚ) / (min a b) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_given_hcf_lcm_l792_79229
