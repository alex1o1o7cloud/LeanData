import Mathlib

namespace NUMINAMATH_CALUDE_f_13_equals_219_l3163_316383

def f (n : ℕ) : ℕ := n^2 + 3*n + 11

theorem f_13_equals_219 : f 13 = 219 := by sorry

end NUMINAMATH_CALUDE_f_13_equals_219_l3163_316383


namespace NUMINAMATH_CALUDE_system_of_equations_l3163_316337

/-- Given a system of equations with parameters n and m, prove specific values of m for different conditions. -/
theorem system_of_equations (n m x y : ℤ) : 
  (n * x + (n + 1) * y = n + 2) → 
  (x - 2 * y + m * x = -5) →
  (
    (n = 1 ∧ x + 2 * y = 3 ∧ x + y = 2 → m = -4) ∧
    (n = 3 ∧ ∃ (x y : ℤ), n * x + (n + 1) * y = n + 2 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0)
  ) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_l3163_316337


namespace NUMINAMATH_CALUDE_city_population_ratio_l3163_316302

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) 
  (h1 : pop_x = 5 * pop_y) 
  (h2 : pop_x / pop_z = 10) : 
  pop_y / pop_z = 2 := by
sorry

end NUMINAMATH_CALUDE_city_population_ratio_l3163_316302


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3163_316355

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3163_316355


namespace NUMINAMATH_CALUDE_existence_of_n_consecutive_with_one_prime_l3163_316356

theorem existence_of_n_consecutive_with_one_prime (n : ℕ) : 
  ∃ k : ℕ, ∃! i : Fin n, Nat.Prime ((k : ℕ) + i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_consecutive_with_one_prime_l3163_316356


namespace NUMINAMATH_CALUDE_add_base_seven_example_l3163_316308

/-- Represents a number in base 7 --/
def BaseSevenNum (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Addition in base 7 --/
def addBaseSeven (a b : List Nat) : List Nat :=
  sorry

theorem add_base_seven_example :
  addBaseSeven [2, 1] [2, 5, 4] = [5, 0, 5] :=
by sorry

end NUMINAMATH_CALUDE_add_base_seven_example_l3163_316308


namespace NUMINAMATH_CALUDE_sector_area_l3163_316312

/-- Given a circular sector with circumference 8 and central angle 2 radians, its area is 4. -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 8 →
  central_angle = 2 →
  area = (1/2) * central_angle * ((circumference - central_angle) / 2)^2 →
  area = 4 := by
  sorry


end NUMINAMATH_CALUDE_sector_area_l3163_316312


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3163_316367

theorem p_sufficient_not_necessary_for_q :
  (∃ x, 0 < x ∧ x < 5 ∧ ¬(-1 < x ∧ x < 5)) = False ∧
  (∃ x, -1 < x ∧ x < 5 ∧ ¬(0 < x ∧ x < 5)) = True := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3163_316367


namespace NUMINAMATH_CALUDE_fraction_equality_l3163_316389

theorem fraction_equality (a b : ℝ) (h : 2/a - 1/b = 1/(a + 2*b)) :
  4/a^2 - 1/b^2 = 1/(a*b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3163_316389


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3163_316347

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there exists exactly one positive integer n > 1 
    satisfying the given conditions -/
theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    greatest_prime_factor n = Real.sqrt n ∧
    greatest_prime_factor (n + 72) = Real.sqrt (n + 72) :=
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3163_316347


namespace NUMINAMATH_CALUDE_number_ordering_l3163_316340

theorem number_ordering : 
  let a := Real.log 0.32
  let b := Real.log 0.33
  let c := 20.3
  let d := 0.32
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_number_ordering_l3163_316340


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l3163_316368

/-- Proves that the certain price of frisbees is $4 given the problem conditions -/
theorem frisbee_price_problem (total_frisbees : ℕ) (price_some : ℝ) (price_rest : ℝ) 
  (total_receipts : ℝ) (min_at_price_rest : ℕ) :
  total_frisbees = 60 →
  price_some = 3 →
  total_receipts = 200 →
  min_at_price_rest = 20 →
  price_rest = 4 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l3163_316368


namespace NUMINAMATH_CALUDE_non_equilateral_triangle_coverage_l3163_316310

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- Define coverage of a triangle by two other triangles
def covers (t1 t2 t : Triangle) : Prop :=
  sorry

-- Define non-equilateral triangle
def nonEquilateral (t : Triangle) : Prop :=
  sorry

-- Define smaller triangle
def smaller (t1 t2 : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem non_equilateral_triangle_coverage (t : Triangle) :
  nonEquilateral t →
  ∃ (t1 t2 : Triangle), smaller t1 t ∧ smaller t2 t ∧ similar t1 t ∧ similar t2 t ∧ covers t1 t2 t :=
sorry

end NUMINAMATH_CALUDE_non_equilateral_triangle_coverage_l3163_316310


namespace NUMINAMATH_CALUDE_inequality_solution_l3163_316301

theorem inequality_solution (c : ℝ) : 
  (4 * c / 3 ≤ 8 + 4 * c ∧ 8 + 4 * c < -3 * (1 + c)) ↔ 
  (c ≥ -3 ∧ c < -11/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3163_316301


namespace NUMINAMATH_CALUDE_roommate_payment_is_757_l3163_316342

/-- Calculates the total payment for one roommate given the costs for rent, utilities, and groceries -/
def roommateTotalPayment (rent utilities groceries : ℕ) : ℚ :=
  (rent + utilities + groceries : ℚ) / 2

/-- Proves that one roommate's total payment is $757 given the specified costs -/
theorem roommate_payment_is_757 :
  roommateTotalPayment 1100 114 300 = 757 := by
  sorry

end NUMINAMATH_CALUDE_roommate_payment_is_757_l3163_316342


namespace NUMINAMATH_CALUDE_perfect_square_implies_zero_a_l3163_316324

theorem perfect_square_implies_zero_a (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_implies_zero_a_l3163_316324


namespace NUMINAMATH_CALUDE_base_salary_per_week_l3163_316316

def past_week_incomes : List ℝ := [406, 413, 420, 436, 395]
def num_past_weeks : ℕ := 5
def num_future_weeks : ℕ := 2
def total_weeks : ℕ := num_past_weeks + num_future_weeks
def average_commission_future : ℝ := 345
def average_weekly_income : ℝ := 500

def total_past_income : ℝ := past_week_incomes.sum
def total_income : ℝ := average_weekly_income * total_weeks
def total_future_income : ℝ := total_income - total_past_income
def total_future_commission : ℝ := average_commission_future * num_future_weeks
def total_future_base_salary : ℝ := total_future_income - total_future_commission

theorem base_salary_per_week : 
  total_future_base_salary / num_future_weeks = 370 := by sorry

end NUMINAMATH_CALUDE_base_salary_per_week_l3163_316316


namespace NUMINAMATH_CALUDE_exist_k_m_with_prime_divisor_diff_l3163_316366

/-- The number of prime divisors of a positive integer -/
def num_prime_divisors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers k and m such that 
    k - m = n and the number of prime divisors of k is exactly one more than 
    the number of prime divisors of m -/
theorem exist_k_m_with_prime_divisor_diff (n : ℕ+) : 
  ∃ (k m : ℕ+), k - m = n ∧ num_prime_divisors k = num_prime_divisors m + 1 := by sorry

end NUMINAMATH_CALUDE_exist_k_m_with_prime_divisor_diff_l3163_316366


namespace NUMINAMATH_CALUDE_equation_solution_l3163_316300

theorem equation_solution : 
  let x : ℝ := 14.8 / 0.13
  0.05 * x + 0.04 * (30 + 2 * x) = 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3163_316300


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l3163_316364

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ 
  (s^2 + k*s + 12 = 0) ∧
  ((r-3)^2 - k*(r-3) + 12 = 0) ∧ 
  ((s-3)^2 - k*(s-3) + 12 = 0) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l3163_316364


namespace NUMINAMATH_CALUDE_terriers_groomed_count_l3163_316378

/-- Represents the time in minutes to groom a poodle -/
def poodle_groom_time : ℕ := 30

/-- Represents the time in minutes to groom a terrier -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Represents the number of poodles groomed -/
def poodles_groomed : ℕ := 3

/-- Represents the total grooming time in minutes -/
def total_groom_time : ℕ := 210

/-- Proves that the number of terriers groomed is 8 -/
theorem terriers_groomed_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_terriers_groomed_count_l3163_316378


namespace NUMINAMATH_CALUDE_smallest_solution_5x2_eq_3y5_l3163_316363

theorem smallest_solution_5x2_eq_3y5 :
  ∃! (x y : ℕ), 
    (5 * x^2 = 3 * y^5) ∧ 
    (∀ (a b : ℕ), (5 * a^2 = 3 * b^5) → (x ≤ a ∧ y ≤ b)) ∧
    x = 675 ∧ y = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_solution_5x2_eq_3y5_l3163_316363


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l3163_316399

theorem salary_savings_percentage (prev_salary : ℝ) (prev_savings_rate : ℝ) 
  (h1 : prev_savings_rate > 0) 
  (h2 : prev_savings_rate < 1) : 
  let new_salary : ℝ := prev_salary * 1.1
  let new_savings_rate : ℝ := 0.1
  let new_savings : ℝ := new_salary * new_savings_rate
  let prev_savings : ℝ := prev_salary * prev_savings_rate
  new_savings = prev_savings * 1.8333333333333331 → prev_savings_rate = 0.06 := by
sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l3163_316399


namespace NUMINAMATH_CALUDE_inequality_solution_l3163_316372

theorem inequality_solution (x : ℝ) : 
  (3 - 1 / (3 * x + 2) < 5) ↔ (x < -5/3 ∨ x > -2/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3163_316372


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3163_316341

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total number of employees -/
def totalEmployees (ec : EmployeeCount) : Nat :=
  ec.young + ec.middleAged + ec.elderly

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total sample size -/
def totalSampleSize (ss : SampleSize) : Nat :=
  ss.young + ss.middleAged + ss.elderly

theorem stratified_sampling_theorem (ec : EmployeeCount) (ss : SampleSize) :
  totalEmployees ec = 750 ∧
  ec.young = 350 ∧
  ec.middleAged = 250 ∧
  ec.elderly = 150 ∧
  ss.young = 7 →
  totalSampleSize ss = 15 :=
sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3163_316341


namespace NUMINAMATH_CALUDE_scout_troop_profit_l3163_316358

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit :
  -- Number of candy bars
  let n : ℕ := 1500
  -- Buy price (in cents) for 3 bars
  let buy_price : ℕ := 150
  -- Sell price (in cents) for 3 bars
  let sell_price : ℕ := 200
  -- All candy bars are sold (implied in the problem)
  -- Profit calculation (in cents)
  let profit : ℚ := n * sell_price / 3 - n * buy_price / 3
  -- The theorem: profit equals 25050 cents (250.50 dollars)
  profit = 25050 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l3163_316358


namespace NUMINAMATH_CALUDE_samantha_overall_percentage_l3163_316377

/-- Represents an exam with its number of questions, weight per question, and percentage correct --/
structure Exam where
  questions : ℕ
  weight : ℕ
  percentCorrect : ℚ

/-- Calculates the total weighted questions for an exam --/
def totalWeightedQuestions (e : Exam) : ℚ :=
  (e.questions * e.weight : ℚ)

/-- Calculates the number of weighted questions answered correctly for an exam --/
def weightedQuestionsCorrect (e : Exam) : ℚ :=
  e.percentCorrect * totalWeightedQuestions e

/-- Calculates the overall percentage of weighted questions answered correctly across multiple exams --/
def overallPercentageCorrect (exams : List Exam) : ℚ :=
  let totalCorrect := (exams.map weightedQuestionsCorrect).sum
  let totalQuestions := (exams.map totalWeightedQuestions).sum
  totalCorrect / totalQuestions

/-- The three exams Samantha took --/
def samanthasExams : List Exam :=
  [{ questions := 30, weight := 1, percentCorrect := 75/100 },
   { questions := 50, weight := 1, percentCorrect := 80/100 },
   { questions := 20, weight := 2, percentCorrect := 65/100 }]

theorem samantha_overall_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |overallPercentageCorrect samanthasExams - 74/100| < ε :=
sorry

end NUMINAMATH_CALUDE_samantha_overall_percentage_l3163_316377


namespace NUMINAMATH_CALUDE_letter_count_theorem_l3163_316314

structure LetterCounts where
  china : ℕ
  italy : ℕ
  india : ℕ

def january : LetterCounts := { china := 6, italy := 8, india := 4 }
def february : LetterCounts := { china := 9, italy := 5, india := 7 }

def percentageChange (old new : ℕ) : ℚ :=
  (new - old : ℚ) / old * 100

def tripleCount (count : LetterCounts) : LetterCounts :=
  { china := 3 * count.china,
    italy := 3 * count.italy,
    india := 3 * count.india }

def totalLetters (a b c : LetterCounts) : ℕ :=
  a.china + a.italy + a.india +
  b.china + b.italy + b.india +
  c.china + c.italy + c.india

theorem letter_count_theorem :
  percentageChange january.china february.china = 50 ∧
  percentageChange january.italy february.italy = -37.5 ∧
  percentageChange january.india february.india = 75 ∧
  totalLetters january february (tripleCount january) = 93 := by
  sorry

end NUMINAMATH_CALUDE_letter_count_theorem_l3163_316314


namespace NUMINAMATH_CALUDE_triangle_circle_area_l3163_316369

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let base := a
  let angle1 := Real.pi / 4  -- 45 degrees in radians
  let angle2 := Real.pi / 12 -- 15 degrees in radians
  let height := a / (1 + Real.tan (Real.pi / 12))
  let circle_area := Real.pi * height^2
  let sector_angle := 2 * Real.pi / 3 -- 120 degrees in radians
  sector_angle / (2 * Real.pi) * circle_area = (Real.pi * a^2 * (2 - Real.sqrt 3)) / 18
  := by sorry

end NUMINAMATH_CALUDE_triangle_circle_area_l3163_316369


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l3163_316359

/-- The number of ways to arrange 5 people in a row with two specific people having exactly one person between them -/
def arrangement_count : ℕ := 36

/-- The number of people in the row -/
def total_people : ℕ := 5

/-- The number of people that can be placed between the two specific people -/
def middle_choices : ℕ := 3

/-- The number of ways to arrange the two specific people with one person between them -/
def specific_arrangement : ℕ := 2

/-- The number of ways to arrange the group of three (two specific people and the one between them) with the other two people -/
def group_arrangement : ℕ := 6

theorem correct_arrangement_count :
  arrangement_count = middle_choices * specific_arrangement * group_arrangement :=
sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l3163_316359


namespace NUMINAMATH_CALUDE_balloons_left_after_sharing_l3163_316309

def blue_balloons : ℕ := 303
def purple_balloons : ℕ := 453

theorem balloons_left_after_sharing :
  (blue_balloons + purple_balloons) / 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_balloons_left_after_sharing_l3163_316309


namespace NUMINAMATH_CALUDE_pq_length_l3163_316344

/-- Two similar triangles PQR and STU with given side lengths and angles -/
structure SimilarTriangles where
  -- Side lengths of triangle PQR
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Side lengths of triangle STU
  ST : ℝ
  TU : ℝ
  SU : ℝ
  -- Angles
  angle_P : ℝ
  angle_S : ℝ
  -- Conditions
  h1 : angle_P = 120
  h2 : angle_S = 120
  h3 : PR = 15
  h4 : SU = 15
  h5 : ST = 4.5
  h6 : TU = 10.5

/-- The length of PQ in similar triangles PQR and STU is 9 -/
theorem pq_length (t : SimilarTriangles) : t.PQ = 9 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_l3163_316344


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_15_l3163_316357

theorem lcm_of_5_6_10_15 : 
  Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 15)) = 30 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_15_l3163_316357


namespace NUMINAMATH_CALUDE_f_equals_g_l3163_316370

-- Define the functions f and g
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 5 * x^5

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3163_316370


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3163_316348

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 2501 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3163_316348


namespace NUMINAMATH_CALUDE_gold_bars_worth_l3163_316353

/-- Calculate the total worth of gold bars in a safe -/
theorem gold_bars_worth (rows : ℕ) (bars_per_row : ℕ) (worth_per_bar : ℕ) :
  rows = 4 →
  bars_per_row = 20 →
  worth_per_bar = 20000 →
  rows * bars_per_row * worth_per_bar = 1600000 := by
  sorry

#check gold_bars_worth

end NUMINAMATH_CALUDE_gold_bars_worth_l3163_316353


namespace NUMINAMATH_CALUDE_popcorn_selling_price_l3163_316315

/-- Calculate the selling price per bag of popcorn -/
theorem popcorn_selling_price 
  (cost_price : ℝ) 
  (num_bags : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 4)
  (h2 : num_bags = 30)
  (h3 : total_profit = 120) : 
  (cost_price * num_bags + total_profit) / num_bags = 8 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_selling_price_l3163_316315


namespace NUMINAMATH_CALUDE_candy_distribution_l3163_316373

/-- Given that Frank has a total of 16 pieces of candy and divides them equally into 2 bags,
    prove that each bag contains 8 pieces of candy. -/
theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → num_bags = 2 → total_candy = num_bags * candy_per_bag → candy_per_bag = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3163_316373


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3163_316328

theorem system_of_equations_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 49)
  (eq3 : z^2 + x*z + x^2 = 124) :
  x*y + y*z + x*z = 70 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3163_316328


namespace NUMINAMATH_CALUDE_closest_axis_of_symmetry_l3163_316354

theorem closest_axis_of_symmetry (ω : ℝ) (h1 : 0 < ω) (h2 : ω < π) :
  let f := fun x ↦ Real.sin (ω * x + 5 * π / 6)
  (f 0 = 1 / 2) →
  (f (1 / 2) = 0) →
  (∃ k : ℤ, -1 = 3 * k - 1 ∧ 
    ∀ m : ℤ, m ≠ k → |3 * m - 1| > |3 * k - 1|) :=
by sorry

end NUMINAMATH_CALUDE_closest_axis_of_symmetry_l3163_316354


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3163_316385

theorem matrix_equation_solution (x : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, x]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; -1, 1]
  B * A = !![2, 4; -1, -2] → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3163_316385


namespace NUMINAMATH_CALUDE_no_nonzero_real_solutions_l3163_316327

theorem no_nonzero_real_solutions :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 2 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_real_solutions_l3163_316327


namespace NUMINAMATH_CALUDE_max_popsicles_for_eight_dollars_l3163_316322

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3

/-- Returns the number of popsicles in a box given its size -/
def boxCount (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  singles : ℕ
  threes : ℕ
  fives : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * boxCost BoxSize.Single +
  p.threes * boxCost BoxSize.Three +
  p.fives * boxCost BoxSize.Five

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * boxCount BoxSize.Single +
  p.threes * boxCount BoxSize.Three +
  p.fives * boxCount BoxSize.Five

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_eight_dollars :
  (∃ p : Purchase, totalCost p = 8 ∧ totalPopsicles p = 13) ∧
  (∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13) := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_for_eight_dollars_l3163_316322


namespace NUMINAMATH_CALUDE_min_n_plus_d_l3163_316335

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1949
  nth_term : ∃ n : ℕ, a n = 2009
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The minimum value of n + d for the given arithmetic sequence -/
theorem min_n_plus_d (seq : ArithmeticSequence) : 
  ∃ n d : ℕ, seq.d = d ∧ (∃ k, seq.a k = 2009) ∧ 
  (∀ m e : ℕ, seq.d = e ∧ (∃ j, seq.a j = 2009) → n + d ≤ m + e) ∧
  n + d = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_n_plus_d_l3163_316335


namespace NUMINAMATH_CALUDE_carnation_bouquets_l3163_316376

theorem carnation_bouquets (b1 b2 b3 : ℝ) (total_bouquets : ℕ) (avg : ℝ) :
  b1 = 9.5 →
  b2 = 14.25 →
  b3 = 18.75 →
  total_bouquets = 6 →
  avg = 16 →
  ∃ b4 b5 b6 : ℝ, b4 + b5 + b6 = total_bouquets * avg - (b1 + b2 + b3) ∧
                  b4 + b5 + b6 = 53.5 :=
by sorry

end NUMINAMATH_CALUDE_carnation_bouquets_l3163_316376


namespace NUMINAMATH_CALUDE_shared_angle_measure_l3163_316338

/-- A configuration of a regular pentagon sharing a side with an equilateral triangle -/
structure PentagonTriangleConfig where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the equilateral triangle in degrees -/
  triangle_angle : ℝ
  /-- The condition that the pentagon is regular -/
  pentagon_regular : pentagon_angle = 108
  /-- The condition that the triangle is equilateral -/
  triangle_equilateral : triangle_angle = 60

/-- The theorem stating that the angle formed by the shared side and the adjacent sides is 6 degrees -/
theorem shared_angle_measure (config : PentagonTriangleConfig) :
  let total_angle := config.pentagon_angle + config.triangle_angle
  let shared_angle := (180 - total_angle) / 2
  shared_angle = 6 := by sorry

end NUMINAMATH_CALUDE_shared_angle_measure_l3163_316338


namespace NUMINAMATH_CALUDE_not_all_two_equal_sides_congruent_l3163_316388

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Statement to be proven false
theorem not_all_two_equal_sides_congruent :
  ¬ (∀ t1 t2 : RightTriangle,
    (t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) ∨
    (t1.leg1 = t2.leg1 ∧ t1.hypotenuse = t2.hypotenuse) ∨
    (t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse)
    → congruent t1 t2) :=
  sorry

end NUMINAMATH_CALUDE_not_all_two_equal_sides_congruent_l3163_316388


namespace NUMINAMATH_CALUDE_unique_natural_number_l3163_316371

theorem unique_natural_number : ∃! n : ℕ, 
  (∃ a : ℕ, n - 45 = a^2) ∧ 
  (∃ b : ℕ, n + 44 = b^2) ∧ 
  n = 1981 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_l3163_316371


namespace NUMINAMATH_CALUDE_ernesto_extra_distance_l3163_316351

/-- Given that Renaldo drove 15 kilometers, Ernesto drove some kilometers more than one-third of Renaldo's distance, and the total distance driven by both men is 27 kilometers, prove that Ernesto drove 7 kilometers more than one-third of Renaldo's distance. -/
theorem ernesto_extra_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) (total_distance : ℝ)
  (h1 : renaldo_distance = 15)
  (h2 : ernesto_distance > (1/3) * renaldo_distance)
  (h3 : total_distance = renaldo_distance + ernesto_distance)
  (h4 : total_distance = 27) :
  ernesto_distance - (1/3) * renaldo_distance = 7 := by
  sorry

end NUMINAMATH_CALUDE_ernesto_extra_distance_l3163_316351


namespace NUMINAMATH_CALUDE_max_distance_sum_l3163_316352

/-- Given m ∈ ℝ, and lines l₁ and l₂ passing through points A and B respectively,
    and intersecting at point P ≠ A, B, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, 3)
  let l₁ := {(x, y) : ℝ × ℝ | x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m * x - y - 2 * m + 3 = 0}
  ∀ P : ℝ × ℝ, P ∈ l₁ ∩ l₂ → P ≠ A → P ≠ B →
    ‖P - A‖ + ‖P - B‖ ≤ 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_max_distance_sum_l3163_316352


namespace NUMINAMATH_CALUDE_remainder_sum_l3163_316374

theorem remainder_sum (p q : ℤ) (hp : p % 80 = 75) (hq : q % 120 = 115) : (p + q) % 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3163_316374


namespace NUMINAMATH_CALUDE_total_wheels_at_station_l3163_316317

/-- Calculates the total number of wheels at a train station -/
theorem total_wheels_at_station
  (num_trains : ℕ)
  (carriages_per_train : ℕ)
  (wheel_rows_per_carriage : ℕ)
  (wheels_per_row : ℕ)
  (h1 : num_trains = 4)
  (h2 : carriages_per_train = 4)
  (h3 : wheel_rows_per_carriage = 3)
  (h4 : wheels_per_row = 5) :
  num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_wheels_at_station_l3163_316317


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3163_316390

/-- Given that a² varies inversely with b², prove that a² = 25/16 when b = 8, given a = 5 when b = 2 -/
theorem inverse_variation_problem (a b : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, x^2 * y^2 = k) 
  (h1 : 5^2 * 2^2 = a^2 * b^2) : 
  8^2 * (25/16) = a^2 * 8^2 := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3163_316390


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_62575_99_l3163_316380

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem least_subtraction_62575_99 :
  ∃ (k : ℕ), k < 99 ∧ (62575 - k) % 99 = 0 ∧ ∀ (m : ℕ), m < k → (62575 - m) % 99 ≠ 0 ∧ k = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_62575_99_l3163_316380


namespace NUMINAMATH_CALUDE_calculation_proof_inequalities_solution_l3163_316325

-- Problem 1
theorem calculation_proof :
  Real.pi ^ 0 + |3 - Real.sqrt 2| - (1/3)⁻¹ = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem inequalities_solution (x : ℝ) :
  (2*x > x - 2 ∧ x + 1 < 2) ↔ (-2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequalities_solution_l3163_316325


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l3163_316350

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (c₁ ≠ c₂) ∧
    (∀ c : ℕ+, (∃ x : ℚ, 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l3163_316350


namespace NUMINAMATH_CALUDE_radical_simplification_l3163_316362

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (40 * q) * Real.sqrt (20 * q) * Real.sqrt (10 * q) = 40 * q * Real.sqrt (5 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3163_316362


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3163_316319

theorem triangle_side_ratio (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are side lengths opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sine rule
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A + Real.cos A - 2 / (Real.sin B + Real.cos B) = 0 →
  -- Conclusion
  (a + b) / c = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3163_316319


namespace NUMINAMATH_CALUDE_fixed_point_on_tangency_line_l3163_316313

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency types
inductive TangencyType
  | ExternalExternal
  | ExternalInternal
  | InternalExternal
  | InternalInternal

-- Define the similarity point
def similarityPoint (k₁ k₂ : Circle) (t : TangencyType) : ℝ × ℝ :=
  sorry

-- Define the line connecting tangency points
def tangencyLine (k k₁ k₂ : Circle) : Set (ℝ × ℝ) :=
  sorry

-- Main theorem
theorem fixed_point_on_tangency_line
  (k₁ k₂ : Circle)
  (h : k₁.radius ≠ k₂.radius)
  (t : TangencyType) :
  ∃ (p : ℝ × ℝ), ∀ (k : Circle),
    p ∈ tangencyLine k k₁ k₂ ∧ p = similarityPoint k₁ k₂ t :=
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_tangency_line_l3163_316313


namespace NUMINAMATH_CALUDE_inequality_preservation_l3163_316384

theorem inequality_preservation (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3163_316384


namespace NUMINAMATH_CALUDE_last_date_2011_divisible_by_101_l3163_316346

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31 ∧
  (month ∈ [4, 6, 9, 11] → day ≤ 30) ∧
  (month = 2 → day ≤ 28)

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_2011_divisible_by_101 :
  ∀ year month day : ℕ,
    is_valid_date year month day →
    date_to_number year month day ≤ 20111221 →
    date_to_number year month day % 101 = 0 →
    date_to_number year month day = 20111221 :=
sorry

end NUMINAMATH_CALUDE_last_date_2011_divisible_by_101_l3163_316346


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3163_316321

theorem polynomial_expansion :
  (fun z : ℝ => 3 * z^3 + 4 * z^2 - 8 * z - 5) *
  (fun z : ℝ => 2 * z^4 - 3 * z^2 + 1) =
  (fun z : ℝ => 6 * z^7 + 12 * z^6 - 25 * z^5 - 20 * z^4 + 34 * z^2 - 8 * z - 5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3163_316321


namespace NUMINAMATH_CALUDE_sum_product_bound_l3163_316307

theorem sum_product_bound (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α*β + β*γ + γ*α ∧ α*β + β*γ + γ*α ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bound_l3163_316307


namespace NUMINAMATH_CALUDE_f_properties_l3163_316386

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -(Real.sin x)^2 + a * Real.sin x - 1

theorem f_properties :
  (∀ x, f 1 x ≥ -3) ∧
  (∀ x, f 1 x = -3 → ∃ y, f 1 y = -3) ∧
  (∀ a, (∀ x, f a x ≤ 1/2) ∧ (∃ y, f a y = 1/2) ↔ a = -5/2 ∨ a = 5/2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3163_316386


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3163_316345

theorem absolute_value_equation (x y : ℝ) : 
  |x^2 - Real.log y| = x^2 + Real.log y → x * (y - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3163_316345


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3163_316379

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (h₆ : x₆ > 0) (h₇ : x₇ > 0) (h₈ : x₈ > 0) (h₉ : x₉ > 0) :
  (x₁ - x₃) / (x₁ * x₃ + 2 * x₂ * x₃ + x₂^2) +
  (x₂ - x₄) / (x₂ * x₄ + 2 * x₃ * x₄ + x₃^2) +
  (x₃ - x₅) / (x₃ * x₅ + 2 * x₄ * x₅ + x₄^2) +
  (x₄ - x₆) / (x₄ * x₆ + 2 * x₅ * x₆ + x₅^2) +
  (x₅ - x₇) / (x₅ * x₇ + 2 * x₆ * x₇ + x₆^2) +
  (x₆ - x₈) / (x₆ * x₈ + 2 * x₇ * x₈ + x₇^2) +
  (x₇ - x₉) / (x₇ * x₉ + 2 * x₈ * x₉ + x₈^2) +
  (x₈ - x₁) / (x₈ * x₁ + 2 * x₉ * x₁ + x₉^2) +
  (x₉ - x₂) / (x₉ * x₂ + 2 * x₁ * x₂ + x₁^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3163_316379


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3163_316365

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3163_316365


namespace NUMINAMATH_CALUDE_sticker_distribution_l3163_316391

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into 5 or fewer parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3163_316391


namespace NUMINAMATH_CALUDE_equal_area_division_l3163_316393

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A figure on a grid --/
structure GridFigure where
  area : ℚ
  points : Set GridPoint

/-- A ray on a grid --/
structure GridRay where
  start : GridPoint
  direction : GridPoint

/-- Theorem: There exists a ray that divides a figure of area 9 into two equal parts --/
theorem equal_area_division (fig : GridFigure) (A : GridPoint) :
  fig.area = 9 →
  ∃ (B : GridPoint) (ray : GridRay),
    B ≠ A ∧
    ray.start = A ∧
    (∃ (t : ℚ), ray.start.x + t * ray.direction.x = B.x ∧ ray.start.y + t * ray.direction.y = B.y) ∧
    ∃ (left_area right_area : ℚ),
      left_area = right_area ∧
      left_area + right_area = fig.area := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l3163_316393


namespace NUMINAMATH_CALUDE_vegetables_minus_fruits_l3163_316329

def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

def vegetables : ℕ := cucumbers + tomatoes
def fruits : ℕ := apples + bananas

theorem vegetables_minus_fruits : vegetables - fruits = 8 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_minus_fruits_l3163_316329


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3163_316304

theorem quadratic_equation_roots (a b c : ℤ) : 
  a ≠ 0 → 
  (∃ x : ℚ, a * x^2 + b * x + c = 0) → 
  ¬(Odd a ∧ Odd b ∧ Odd c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3163_316304


namespace NUMINAMATH_CALUDE_sum_of_odds_l3163_316396

theorem sum_of_odds (sum_of_evens : ℕ) (n : ℕ) :
  (n = 70) →
  (sum_of_evens = n / 2 * (2 + n * 2)) →
  (sum_of_evens = 4970) →
  (n / 2 * (1 + (n * 2 - 1)) = 4900) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odds_l3163_316396


namespace NUMINAMATH_CALUDE_problem_2a_l3163_316392

theorem problem_2a (a b x y : ℝ) 
  (eq1 : a * x + b * y = 7)
  (eq2 : a * x^2 + b * y^2 = 49)
  (eq3 : a * x^3 + b * y^3 = 133)
  (eq4 : a * x^4 + b * y^4 = 406) :
  2014 * (x + y - x * y) - 100 * (a + b) = 6889.33 := by
sorry

end NUMINAMATH_CALUDE_problem_2a_l3163_316392


namespace NUMINAMATH_CALUDE_triangle_arctans_sum_l3163_316360

theorem triangle_arctans_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b^2 + c^2 = a^2) (h5 : Real.arcsin (1/2) + Real.arcsin (1/2) = Real.pi/2) :
  Real.arctan (b/(c+a)) + Real.arctan (c/(b+a)) = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_arctans_sum_l3163_316360


namespace NUMINAMATH_CALUDE_sum_b_plus_d_l3163_316305

theorem sum_b_plus_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : a + c = 7) : 
  b + d = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_b_plus_d_l3163_316305


namespace NUMINAMATH_CALUDE_max_vertical_distance_is_sqrt2_over_2_l3163_316323

/-- Represents a square with side length 1 inch -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : List UnitSquare
  rotated_square : UnitSquare

/-- The maximum vertical distance from the original line to any point on the rotated square -/
def max_vertical_distance (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating the maximum vertical distance is √2/2 -/
theorem max_vertical_distance_is_sqrt2_over_2 (config : SquareConfiguration) :
  max_vertical_distance config = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_vertical_distance_is_sqrt2_over_2_l3163_316323


namespace NUMINAMATH_CALUDE_product_of_five_primes_with_491_l3163_316331

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_abc_abc (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)

theorem product_of_five_primes_with_491 :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime 491 ∧
    is_abc_abc (p₁ * p₂ * p₃ * p₄ * 491) ∧
    p₁ * p₂ * p₃ * p₄ * 491 = 982982 :=
  sorry

end NUMINAMATH_CALUDE_product_of_five_primes_with_491_l3163_316331


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3163_316320

theorem complex_equation_solution (z : ℂ) :
  Complex.I * z = 4 + 3 * Complex.I → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3163_316320


namespace NUMINAMATH_CALUDE_truck_distance_l3163_316382

/-- Proves that a truck traveling at a rate of 2 miles per 4 minutes will cover 90 miles in 3 hours -/
theorem truck_distance (rate : ℚ) (time : ℚ) : 
  rate = 2 / 4 → time = 3 * 60 → rate * time = 90 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l3163_316382


namespace NUMINAMATH_CALUDE_fixed_distance_point_l3163_316397

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, if p satisfies ‖p - b‖ = 3 ‖p - a‖, 
    then p is at a fixed distance from (9/8)a - (1/8)b -/
theorem fixed_distance_point (a b p : V) 
  (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : V), 
    (‖q - b‖ = 3 * ‖q - a‖) → 
    ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l3163_316397


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3163_316336

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 2 * x + y = 11) 
  (h2 : x + 2 * y = 13) : 
  10 * x^2 - 6 * x * y + y^2 = 530 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3163_316336


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3163_316398

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = Complex.mk (3 / 4) (Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3163_316398


namespace NUMINAMATH_CALUDE_nonzero_real_number_problem_l3163_316349

theorem nonzero_real_number_problem (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_number_problem_l3163_316349


namespace NUMINAMATH_CALUDE_batsman_average_l3163_316339

/-- Calculates the average runs for a batsman given two sets of matches --/
def average_runs (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the average runs for 45 matches is 42 given the specified conditions --/
theorem batsman_average :
  average_runs 30 50 15 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l3163_316339


namespace NUMINAMATH_CALUDE_jacobs_february_bill_l3163_316306

/-- Calculates the total cell phone bill given the plan details and usage --/
def calculate_bill (base_cost : ℚ) (included_hours : ℚ) (cost_per_text : ℚ) 
  (cost_per_extra_minute : ℚ) (texts_sent : ℚ) (hours_talked : ℚ) : ℚ :=
  let text_cost := texts_sent * cost_per_text
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  let extra_cost := extra_minutes * cost_per_extra_minute
  base_cost + text_cost + extra_cost

/-- Theorem stating that Jacob's cell phone bill for February is $83.80 --/
theorem jacobs_february_bill :
  calculate_bill 25 25 0.08 0.13 150 31 = 83.80 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_february_bill_l3163_316306


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3163_316334

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x < 1 ∨ x > b}

-- Define the inequality function
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - (c + 2) * x + 2 * x

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ (a b : ℝ), (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 0 → {x | g c x < 0} = Set.Ioo 0 c) ∧
      (c = 0 → {x | g c x < 0} = ∅) ∧
      (c < 0 → {x | g c x < 0} = Set.Ioo c 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3163_316334


namespace NUMINAMATH_CALUDE_sum_congruence_l3163_316387

theorem sum_congruence : (1 + 23 + 456 + 7890) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l3163_316387


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l3163_316311

-- Define the ellipse G
def G (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for point M
def M_condition (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  (xA - m)^2 + yA^2 = (xB - m)^2 + yB^2

-- Main theorem
theorem ellipse_intersection_range :
  ∀ (k : ℝ) (A B : ℝ × ℝ) (m : ℝ),
  (∃ (xA yA xB yB : ℝ), A = (xA, yA) ∧ B = (xB, yB) ∧
    G xA yA ∧ G xB yB ∧
    line k xA yA ∧ line k xB yB ∧
    A ≠ B ∧
    M_condition m A B) →
  m ∈ Set.Icc (- Real.sqrt 6 / 12) (Real.sqrt 6 / 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l3163_316311


namespace NUMINAMATH_CALUDE_secretary_work_ratio_l3163_316326

/-- Represents the work hours of three secretaries on a project. -/
structure SecretaryWork where
  total : ℝ
  longest : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of work hours for three secretaries. -/
theorem secretary_work_ratio (work : SecretaryWork) 
  (h_total : work.total = 120)
  (h_longest : work.longest = 75)
  (h_sum : work.second + work.third = work.total - work.longest) :
  ∃ (b c : ℝ), work.second = b ∧ work.third = c ∧ b + c = 45 := by
  sorry

#check secretary_work_ratio

end NUMINAMATH_CALUDE_secretary_work_ratio_l3163_316326


namespace NUMINAMATH_CALUDE_project_selection_count_l3163_316330

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 3 additional projects --/
def selectProjects : ℕ :=
  choose 4 1 * choose 6 1 * choose 4 1 +
  choose 6 2 * choose 4 1 +
  choose 6 1 * choose 4 2

theorem project_selection_count :
  selectProjects = 192 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l3163_316330


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l3163_316375

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 5) 
  (h3 : outer_revolutions = 15) : 
  ∃ inner_revolutions : ℕ, 
    (2 * Real.pi * outer_radius * outer_revolutions) = 
    (2 * Real.pi * inner_radius * inner_revolutions) ∧ 
    inner_revolutions = 90 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l3163_316375


namespace NUMINAMATH_CALUDE_total_books_l3163_316381

theorem total_books (keith_books jason_books megan_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) 
  (h3 : megan_books = 15) : 
  keith_books + jason_books + megan_books = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3163_316381


namespace NUMINAMATH_CALUDE_worker_completion_time_l3163_316343

/-- Given two workers A and B, where A is thrice as fast as B, 
    and together they can complete a job in 18 days,
    prove that A alone can complete the job in 24 days. -/
theorem worker_completion_time 
  (speed_A speed_B : ℝ) 
  (combined_time : ℝ) :
  speed_A = 3 * speed_B →
  1 / speed_A + 1 / speed_B = 1 / combined_time →
  combined_time = 18 →
  1 / speed_A = 1 / 24 :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3163_316343


namespace NUMINAMATH_CALUDE_anniversary_18_months_ago_proof_l3163_316361

/-- The anniversary Bella and Bob celebrated 18 months ago -/
def anniversary_18_months_ago : ℕ := 2

/-- The number of months until their 4th anniversary -/
def months_until_4th_anniversary : ℕ := 6

/-- The current duration of their relationship in months -/
def current_relationship_duration : ℕ := 4 * 12 - months_until_4th_anniversary

/-- The duration of their relationship 18 months ago in months -/
def relationship_duration_18_months_ago : ℕ := current_relationship_duration - 18

theorem anniversary_18_months_ago_proof :
  anniversary_18_months_ago = relationship_duration_18_months_ago / 12 :=
by sorry

end NUMINAMATH_CALUDE_anniversary_18_months_ago_proof_l3163_316361


namespace NUMINAMATH_CALUDE_valid_permutations_64420_l3163_316394

def digits : List Nat := [6, 4, 4, 2, 0]

/-- The number of permutations of the digits that form a 5-digit number not starting with 0 -/
def valid_permutations (ds : List Nat) : Nat :=
  let non_zero_digits := ds.filter (· ≠ 0)
  let zero_digits := ds.filter (· = 0)
  non_zero_digits.length * (ds.length - 1).factorial / (non_zero_digits.map (λ d => (ds.filter (· = d)).length)).prod

theorem valid_permutations_64420 :
  valid_permutations digits = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_permutations_64420_l3163_316394


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3163_316318

theorem disjunction_false_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3163_316318


namespace NUMINAMATH_CALUDE_correct_land_equation_l3163_316332

/-- Represents the relationship between arable land and forest land areas -/
def land_relationship (x y : ℝ) : Prop :=
  x + y = 2000 ∧ y = x * (30 / 100)

/-- The correct system of equations for the land areas -/
theorem correct_land_equation :
  ∀ x y : ℝ,
  (x + y = 2000 ∧ y = x * (30 / 100)) ↔ land_relationship x y :=
by sorry

end NUMINAMATH_CALUDE_correct_land_equation_l3163_316332


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3163_316395

/-- A function satisfying the given functional equation is constant and equal to 2 -/
theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x > 0, f x > 0) → 
  (∀ x y, x > 0 → y > 0 → f x * f y = 2 * f (x + y * f x)) → 
  (∀ x > 0, f x = 2) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3163_316395


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3163_316333

/-- An isosceles triangle with side lengths 6 and 9 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 9 ∨ side2 = 9)
  (has_length_6 : side1 = 6 ∨ side2 = 6)
  (has_length_9 : side1 = 9 ∨ side2 = 9)

/-- The perimeter of an isosceles triangle with side lengths 6 and 9 is either 21 or 24 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 21 ∨ 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3163_316333


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l3163_316303

theorem revenue_change_after_price_and_sales_change
  (initial_price initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease : ℝ)
  (h1 : price_increase = 0.3)
  (h2 : sales_decrease = 0.2)
  : (((initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) - initial_price * initial_sales) / (initial_price * initial_sales)) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l3163_316303
