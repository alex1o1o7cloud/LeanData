import Mathlib

namespace circle_center_coordinate_sum_l1346_134638

/-- The sum of the coordinates of the center of the circle defined by x^2 + y^2 = -4x - 6y + 5 is -5 -/
theorem circle_center_coordinate_sum :
  ∃ (x y : ℝ), x^2 + y^2 = -4*x - 6*y + 5 ∧ x + y = -5 := by
  sorry

end circle_center_coordinate_sum_l1346_134638


namespace r_2011_equals_2_l1346_134673

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def r (n : ℕ) : ℕ := fib n % 3

theorem r_2011_equals_2 : r 2011 = 2 := by
  sorry

end r_2011_equals_2_l1346_134673


namespace age_of_25th_student_l1346_134648

/-- The age of the 25th student in a class with specific age distributions -/
theorem age_of_25th_student (total_students : ℕ) (avg_age : ℝ) 
  (group1_count : ℕ) (group1_avg : ℝ) (group2_count : ℕ) (group2_avg : ℝ) :
  total_students = 25 →
  avg_age = 25 →
  group1_count = 10 →
  group1_avg = 22 →
  group2_count = 14 →
  group2_avg = 28 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg) = 13 :=
by sorry

end age_of_25th_student_l1346_134648


namespace smallest_integer_with_divisibility_prove_smallest_integer_l1346_134663

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i ∧ i ≤ b → is_divisible n i

theorem smallest_integer_with_divisibility (n : ℕ) : Prop :=
  (n = 1225224000) ∧
  (is_divisible_by_range n 1 26) ∧
  (is_divisible_by_range n 30 30) ∧
  (¬ is_divisible n 27) ∧
  (¬ is_divisible n 28) ∧
  (¬ is_divisible n 29) ∧
  (∀ m : ℕ, m < n →
    ¬(is_divisible_by_range m 1 26 ∧
      is_divisible_by_range m 30 30 ∧
      ¬ is_divisible m 27 ∧
      ¬ is_divisible m 28 ∧
      ¬ is_divisible m 29))

theorem prove_smallest_integer : ∃ n : ℕ, smallest_integer_with_divisibility n :=
  sorry

end smallest_integer_with_divisibility_prove_smallest_integer_l1346_134663


namespace prince_gvidon_descendants_l1346_134699

/-- The total number of descendants of Prince Gvidon -/
def total_descendants : ℕ := 189

/-- The number of sons Prince Gvidon had -/
def initial_sons : ℕ := 3

/-- The number of descendants who had two sons each -/
def descendants_with_sons : ℕ := 93

/-- The number of sons each descendant with sons had -/
def sons_per_descendant : ℕ := 2

theorem prince_gvidon_descendants :
  total_descendants = initial_sons + descendants_with_sons * sons_per_descendant :=
by sorry

end prince_gvidon_descendants_l1346_134699


namespace quinton_cupcakes_l1346_134629

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := sorry

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := 16

/-- The number of staff members who received a cupcake -/
def staff_members : ℕ := 4

/-- The number of cupcakes left over -/
def leftover_cupcakes : ℕ := 2

/-- Theorem stating that the total number of cupcakes Quinton brought to school is 40 -/
theorem quinton_cupcakes : 
  total_cupcakes = delmont_students + donnelly_students + staff_members + leftover_cupcakes :=
by sorry

end quinton_cupcakes_l1346_134629


namespace book_distribution_l1346_134613

theorem book_distribution (total_books : ℕ) (girls boys non_binary : ℕ) 
  (h1 : total_books = 840)
  (h2 : girls = 20)
  (h3 : boys = 15)
  (h4 : non_binary = 5)
  (h5 : ∃ (x : ℕ), 
    girls * (2 * x) + boys * x + non_binary * x = total_books ∧ 
    x > 0) :
  ∃ (books_per_boy : ℕ),
    books_per_boy = 14 ∧
    girls * (2 * books_per_boy) + boys * books_per_boy + non_binary * books_per_boy = total_books :=
by sorry

end book_distribution_l1346_134613


namespace number_problem_l1346_134628

theorem number_problem (x : ℝ) : 2 * x - x / 2 = 45 → x = 30 := by
  sorry

end number_problem_l1346_134628


namespace bisection_method_step_next_interval_is_1_5_to_2_l1346_134669

def f (x : ℝ) := x^3 - x - 5

theorem bisection_method_step (a b : ℝ) (hab : a < b) (hf : f a * f b < 0) :
  let m := (a + b) / 2
  (f a * f m < 0 ∧ (m, b) = (1.5, 2)) ∨
  (f m * f b < 0 ∧ (a, m) = (1.5, 2)) :=
sorry

theorem next_interval_is_1_5_to_2 :
  let a := 1
  let b := 2
  let m := (a + b) / 2
  m = 1.5 ∧ f a * f b < 0 →
  (1.5, 2) = (let m := (a + b) / 2; if f a * f m < 0 then (a, m) else (m, b)) :=
sorry

end bisection_method_step_next_interval_is_1_5_to_2_l1346_134669


namespace cyclic_inequality_l1346_134642

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end cyclic_inequality_l1346_134642


namespace six_eight_ten_pythagorean_triple_l1346_134692

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end six_eight_ten_pythagorean_triple_l1346_134692


namespace library_fee_calculation_l1346_134666

/-- Calculates the total amount paid for borrowing books from a library. -/
def calculate_library_fee (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (book3_days : ℕ) : ℚ :=
  daily_rate * (book1_days + book2_days + book3_days)

theorem library_fee_calculation :
  let daily_rate : ℚ := 1/2
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let book3_days : ℕ := 31
  calculate_library_fee daily_rate book1_days book2_days book3_days = 41 := by
  sorry

#eval calculate_library_fee (1/2) 20 31 31

end library_fee_calculation_l1346_134666


namespace range_of_a_l1346_134688

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 1 → f x a ≥ 0) → 
  a ∈ Set.Icc (-6 : ℝ) (-2 : ℝ) := by sorry

end range_of_a_l1346_134688


namespace browser_usage_inconsistency_l1346_134625

theorem browser_usage_inconsistency (total_A : ℕ) (total_B : ℕ) (both : ℕ) (only_one : ℕ) :
  total_A = 316 →
  total_B = 478 →
  both = 104 →
  only_one = 567 →
  (total_A - both) + (total_B - both) ≠ only_one :=
by
  sorry

end browser_usage_inconsistency_l1346_134625


namespace special_pentagon_theorem_l1346_134614

/-- A pentagon with two right angles and three known angles -/
structure SpecialPentagon where
  -- The measures of the three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- The measures of the two unknown angles
  angle_U : ℝ
  angle_V : ℝ
  -- Conditions
  angle_P_eq : angle_P = 42
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 38
  -- The pentagon has two right angles
  has_two_right_angles : True
  -- The sum of all interior angles of a pentagon is 540°
  sum_of_angles : angle_P + angle_Q + angle_R + angle_U + angle_V + 180 = 540

theorem special_pentagon_theorem (p : SpecialPentagon) : p.angle_U + p.angle_V = 40 := by
  sorry

end special_pentagon_theorem_l1346_134614


namespace diagonal_cuboids_count_l1346_134643

def cuboid_count (a b c L : ℕ) : ℕ :=
  L / a + L / b + L / c - L / (a * b) - L / (a * c) - L / (b * c) + L / (a * b * c)

theorem diagonal_cuboids_count : 
  let a : ℕ := 2
  let b : ℕ := 7
  let c : ℕ := 13
  let L : ℕ := 2002
  let lcm : ℕ := a * b * c
  (L / lcm) * cuboid_count a b c lcm = 1210 := by sorry

end diagonal_cuboids_count_l1346_134643


namespace imaginary_unit_power_l1346_134655

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2018 = -1 := by sorry

end imaginary_unit_power_l1346_134655


namespace largest_valid_number_sum_of_digits_l1346_134661

def is_valid_remainder (r : ℕ) (m : ℕ) : Prop :=
  r > 1 ∧ r < m

def form_geometric_progression (r1 r2 r3 : ℕ) : Prop :=
  (r2 * r2 = r1 * r3) ∧ r1 ≠ r2

def satisfies_conditions (n : ℕ) : Prop :=
  ∃ (r1 r2 r3 : ℕ),
    is_valid_remainder r1 9 ∧
    is_valid_remainder r2 10 ∧
    is_valid_remainder r3 11 ∧
    form_geometric_progression r1 r2 r3 ∧
    n % 9 = r1 ∧
    n % 10 = r2 ∧
    n % 11 = r3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_valid_number_sum_of_digits :
  ∃ (N : ℕ), N < 990 ∧ satisfies_conditions N ∧
  (∀ (m : ℕ), m < 990 → satisfies_conditions m → m ≤ N) ∧
  sum_of_digits N = 13 :=
sorry

end largest_valid_number_sum_of_digits_l1346_134661


namespace algebraic_expression_value_l1346_134691

theorem algebraic_expression_value (x y : ℝ) : 
  5 * x^2 - 4 * x * y - 1 = -11 → -10 * x^2 + 8 * x * y + 5 = 25 := by
  sorry

end algebraic_expression_value_l1346_134691


namespace units_digit_of_quotient_l1346_134677

theorem units_digit_of_quotient (h : 7 ∣ (4^2065 + 6^2065)) :
  (4^2065 + 6^2065) / 7 % 10 = 0 := by sorry

end units_digit_of_quotient_l1346_134677


namespace compound_statement_falsity_l1346_134605

theorem compound_statement_falsity (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end compound_statement_falsity_l1346_134605


namespace binary_arithmetic_proof_l1346_134678

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_arithmetic_proof :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, false, true]              -- 101₂
  let c := [false, true, false, true]       -- 1010₂
  let product := binary_to_decimal a * binary_to_decimal b
  let result := product - binary_to_decimal c
  decimal_to_binary result = [true, false, true, true, true, true, true] -- 1111101₂
  := by sorry

end binary_arithmetic_proof_l1346_134678


namespace expansion_theorem_l1346_134682

-- Define the sum of coefficients for (3x + √x)^n
def sumCoefficients (n : ℕ) : ℝ := 4^n

-- Define the sum of binomial coefficients
def sumBinomialCoefficients (n : ℕ) : ℝ := 2^n

-- Define the condition M - N = 240
def conditionSatisfied (n : ℕ) : Prop :=
  sumCoefficients n - sumBinomialCoefficients n = 240

-- Define the rational terms in the expansion
def rationalTerms (n : ℕ) : List (ℝ × ℕ) :=
  [(81, 4), (54, 3), (1, 2)]

theorem expansion_theorem :
  ∃ n : ℕ, conditionSatisfied n ∧ 
  n = 4 ∧
  rationalTerms n = [(81, 4), (54, 3), (1, 2)] :=
sorry

end expansion_theorem_l1346_134682


namespace prob_six_odd_in_eight_rolls_l1346_134659

/-- A fair 6-sided die -/
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 6

/-- The probability of getting exactly 6 odd results in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls :
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd^target_odd * (1 - prob_odd)^(num_rolls - target_odd) = 28/256 := by
  sorry

end prob_six_odd_in_eight_rolls_l1346_134659


namespace original_average_score_l1346_134650

/-- Proves that the original average score of a class is 37, given the conditions. -/
theorem original_average_score (num_students : ℕ) (grace_marks : ℕ) (new_average : ℕ) :
  num_students = 35 →
  grace_marks = 3 →
  new_average = 40 →
  (num_students * new_average - num_students * grace_marks) / num_students = 37 :=
by sorry

end original_average_score_l1346_134650


namespace minimum_houses_with_more_than_five_floors_l1346_134671

theorem minimum_houses_with_more_than_five_floors (n : ℕ) : 
  (n > 0) → 
  (∃ x : ℕ, x < n ∧ (n - x : ℚ) / n > 47/50) → 
  (∀ m : ℕ, m < n → ∃ y : ℕ, y < m ∧ (m - y : ℚ) / m ≤ 47/50) → 
  n = 20 := by
sorry

end minimum_houses_with_more_than_five_floors_l1346_134671


namespace partition_five_elements_l1346_134631

/-- The number of ways to partition a set of 5 elements into two non-empty subsets, 
    where two specific elements must be in the same subset -/
def partitionWays : ℕ := 6

/-- A function that calculates the number of ways to partition a set of n elements into two non-empty subsets,
    where two specific elements must be in the same subset -/
def partitionFunction (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2)

theorem partition_five_elements :
  partitionWays = partitionFunction 5 :=
by sorry

end partition_five_elements_l1346_134631


namespace test_score_result_l1346_134652

/-- Represents the score calculation for a test with specific conditions -/
def test_score (total_questions : ℕ) 
               (single_answer_questions : ℕ) 
               (multiple_answer_questions : ℕ) 
               (single_answer_marks : ℕ) 
               (multiple_answer_marks : ℕ) 
               (single_answer_penalty : ℕ) 
               (multiple_answer_penalty : ℕ) 
               (jose_wrong_single : ℕ) 
               (jose_wrong_multiple : ℕ) 
               (meghan_diff : ℕ) 
               (alisson_diff : ℕ) : ℕ := 
  sorry

theorem test_score_result : 
  test_score 70 50 20 2 4 1 2 10 5 30 50 = 280 :=
sorry

end test_score_result_l1346_134652


namespace first_day_of_month_l1346_134609

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day after n days
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (dayAfter d n)

-- Theorem statement
theorem first_day_of_month (d : DayOfWeek) :
  dayAfter d 22 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by sorry

end first_day_of_month_l1346_134609


namespace isosceles_triangle_side_length_l1346_134684

/-- An isosceles triangle with perimeter 26 and one side 12 has the other side length either 12 or 7 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 26 → -- perimeter is 26
  (a = b ∨ b = c ∨ a = c) → -- isosceles condition
  (a = 12 ∨ b = 12 ∨ c = 12) → -- one side is 12
  (a = 7 ∨ b = 7 ∨ c = 7) ∨ (a = 12 ∧ b = 12) ∨ (b = 12 ∧ c = 12) ∨ (a = 12 ∧ c = 12) :=
by sorry


end isosceles_triangle_side_length_l1346_134684


namespace selection_theorem_l1346_134683

/-- Represents the number of students with each skill -/
structure StudentGroup where
  total : ℕ
  singers : ℕ
  dancers : ℕ
  both : ℕ

/-- Represents the selection requirements -/
structure SelectionRequirement where
  singersToSelect : ℕ
  dancersToSelect : ℕ

/-- Calculates the number of ways to select students given a student group and selection requirements -/
def numberOfWaysToSelect (group : StudentGroup) (req : SelectionRequirement) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem selection_theorem (group : StudentGroup) (req : SelectionRequirement) :
  group.total = 6 ∧ 
  group.singers = 3 ∧ 
  group.dancers = 2 ∧ 
  group.both = 1 ∧
  req.singersToSelect = 2 ∧
  req.dancersToSelect = 1 →
  numberOfWaysToSelect group req = 15 :=
by sorry

end selection_theorem_l1346_134683


namespace distinct_prime_factors_count_l1346_134697

def number := 102 * 104 * 107 * 108

theorem distinct_prime_factors_count :
  Nat.card (Nat.factors number).toFinset = 5 := by sorry

end distinct_prime_factors_count_l1346_134697


namespace tangency_line_parallel_to_common_tangent_l1346_134667

/-- Given three parabolas p₁, p₂, and p₃, where p₁ and p₂ both touch p₃,
    the line connecting the points of tangency of p₁ and p₂ with p₃
    is parallel to the common tangent of p₁ and p₂. -/
theorem tangency_line_parallel_to_common_tangent
  (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let p₁ := fun x => -x^2 + b₁ * x + c₁
  let p₂ := fun x => -x^2 + b₂ * x + c₂
  let p₃ := fun x => x^2 + b₃ * x + c₃
  let x₁ := (b₁ - b₃) / 4
  let y₁ := p₃ x₁
  let x₂ := (b₂ - b₃) / 4
  let y₂ := p₃ x₂
  let m_tangency := (y₂ - y₁) / (x₂ - x₁)
  let m_common_tangent := (4 * (c₁ - c₂) - 2 * b₃ * (b₁ - b₂)) / (2 * (b₂ - b₁))
  (b₃ - b₁)^2 = 8 * (c₃ - c₁) →
  (b₃ - b₂)^2 = 8 * (c₃ - c₂) →
  m_tangency = m_common_tangent :=
by sorry

end tangency_line_parallel_to_common_tangent_l1346_134667


namespace total_money_calculation_l1346_134645

def hundred_bills : ℕ := 2
def fifty_bills : ℕ := 5
def ten_bills : ℕ := 10

def hundred_value : ℕ := 100
def fifty_value : ℕ := 50
def ten_value : ℕ := 10

theorem total_money_calculation : 
  (hundred_bills * hundred_value) + (fifty_bills * fifty_value) + (ten_bills * ten_value) = 550 := by
  sorry

end total_money_calculation_l1346_134645


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1346_134695

/-- A curve y = sin(x + φ) is symmetric about the y-axis if and only if sin(x + φ) = sin(-x + φ) for all x ∈ ℝ -/
def symmetric_about_y_axis (φ : ℝ) : Prop :=
  ∀ x : ℝ, Real.sin (x + φ) = Real.sin (-x + φ)

/-- φ = π/2 is a sufficient condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_condition (φ : ℝ) (h : φ = π / 2) : symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is not a necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem not_necessary_condition : ∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is a sufficient but not necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_but_not_necessary : 
  (∀ φ : ℝ, φ = π / 2 → symmetric_about_y_axis φ) ∧ 
  (∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ) := by
  sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1346_134695


namespace geometric_sequence_ratio_l1346_134604

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {an} is a geometric sequence with common ratio q
  q > 0 →                       -- q is positive
  a 2 = 1 →                     -- a2 = 1
  a 4 = 4 →                     -- a4 = 4
  q = 2 :=                      -- prove q = 2
by sorry

end geometric_sequence_ratio_l1346_134604


namespace project_completion_days_l1346_134634

/-- Represents the time in days for a worker to complete the project alone -/
structure WorkerRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the project completion scenario -/
structure ProjectCompletion where
  worker_a : WorkerRate
  worker_b : WorkerRate
  worker_c : WorkerRate
  a_quit_before_end : ℕ

/-- Calculates the total days to complete the project -/
def total_days (p : ProjectCompletion) : ℕ := 
  sorry

/-- Theorem stating that the project will be completed in 18 days -/
theorem project_completion_days (p : ProjectCompletion) 
  (h1 : p.worker_a.days = 20)
  (h2 : p.worker_b.days = 30)
  (h3 : p.worker_c.days = 40)
  (h4 : p.a_quit_before_end = 18) :
  total_days p = 18 := by
  sorry

end project_completion_days_l1346_134634


namespace justin_flower_gathering_time_l1346_134639

/-- Calculates the additional time needed for Justin to gather flowers for his classmates -/
def additional_time_needed (
  classmates : ℕ)
  (average_time_per_flower : ℕ)
  (gathering_time_hours : ℕ)
  (lost_flowers : ℕ) : ℕ :=
  let gathering_time_minutes := gathering_time_hours * 60
  let flowers_gathered := gathering_time_minutes / average_time_per_flower
  let flowers_remaining := flowers_gathered - lost_flowers
  let additional_flowers_needed := classmates - flowers_remaining
  additional_flowers_needed * average_time_per_flower

theorem justin_flower_gathering_time :
  additional_time_needed 30 10 2 3 = 210 := by
  sorry

end justin_flower_gathering_time_l1346_134639


namespace megan_country_albums_l1346_134627

/-- The number of country albums Megan bought -/
def num_country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def num_pop_albums : ℕ := 8

/-- The number of songs per album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 70

/-- Proof that Megan bought 2 country albums -/
theorem megan_country_albums :
  num_country_albums * songs_per_album + num_pop_albums * songs_per_album = total_songs :=
by sorry

end megan_country_albums_l1346_134627


namespace expression_equals_ten_to_twelve_l1346_134620

theorem expression_equals_ten_to_twelve : (2 * 5 * 10^5) * 10^6 = 10^12 := by
  sorry

end expression_equals_ten_to_twelve_l1346_134620


namespace mason_bricks_used_l1346_134690

/-- Calculates the total number of bricks used by a mason given the following conditions:
  * The mason needs to build 6 courses per wall
  * Each course has 10 bricks
  * He needs to build 4 walls
  * He can't finish two courses of the last wall due to lack of bricks
-/
def total_bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (unfinished_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course * complete_walls
  let incomplete_wall_bricks := (courses_per_wall - unfinished_courses) * bricks_per_course
  complete_wall_bricks + incomplete_wall_bricks

theorem mason_bricks_used :
  total_bricks_used 6 10 4 2 = 220 := by
  sorry

end mason_bricks_used_l1346_134690


namespace oz_language_word_loss_l1346_134610

theorem oz_language_word_loss :
  let total_letters : ℕ := 64
  let forbidden_letters : ℕ := 1
  let max_word_length : ℕ := 2

  let one_letter_words_lost : ℕ := forbidden_letters
  let two_letter_words_lost : ℕ := 
    total_letters * forbidden_letters + 
    forbidden_letters * total_letters - 
    forbidden_letters * forbidden_letters

  one_letter_words_lost + two_letter_words_lost = 128 :=
by sorry

end oz_language_word_loss_l1346_134610


namespace sports_club_members_l1346_134644

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 17)
  (h2 : tennis = 19)
  (h3 : both = 8)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end sports_club_members_l1346_134644


namespace cards_traded_count_l1346_134617

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ)
  (robert_second_trade : ℕ) (padma_second_trade : ℕ) : ℕ :=
  (padma_first_trade + robert_first_trade) + (robert_second_trade + padma_second_trade)

/-- Theorem stating the total number of cards traded between Padma and Robert -/
theorem cards_traded_count :
  total_cards_traded 75 88 2 10 8 15 = 35 := by
  sorry

end cards_traded_count_l1346_134617


namespace average_gas_mileage_round_trip_l1346_134611

/-- Calculates the average gas mileage for a round trip with different distances and fuel efficiencies -/
theorem average_gas_mileage_round_trip 
  (distance_outgoing : ℝ) 
  (distance_return : ℝ)
  (efficiency_outgoing : ℝ)
  (efficiency_return : ℝ) :
  let total_distance := distance_outgoing + distance_return
  let total_fuel := distance_outgoing / efficiency_outgoing + distance_return / efficiency_return
  let average_mileage := total_distance / total_fuel
  (distance_outgoing = 150 ∧ 
   distance_return = 180 ∧ 
   efficiency_outgoing = 25 ∧ 
   efficiency_return = 50) →
  (34 < average_mileage ∧ average_mileage < 35) :=
by sorry

end average_gas_mileage_round_trip_l1346_134611


namespace max_value_of_complex_distance_l1346_134657

theorem max_value_of_complex_distance (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 5 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≤ max_val :=
sorry

end max_value_of_complex_distance_l1346_134657


namespace quinary_444_equals_octal_174_l1346_134640

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinary_to_decimal (q : ℕ) : ℕ := 
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimal_to_octal (d : ℕ) : ℕ := 
  1 * 8^2 + 7 * 8^1 + 4 * 8^0

/-- Theorem stating that 444₅ in quinary is equal to 174₈ in octal --/
theorem quinary_444_equals_octal_174 : 
  quinary_to_decimal 444 = decimal_to_octal 174 := by
  sorry

end quinary_444_equals_octal_174_l1346_134640


namespace rectangle_covers_ellipse_l1346_134612

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of an ellipse -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ

/-- Checks if a rectangle can cover an ellipse -/
def can_cover (r : Rectangle) (e : Ellipse) : Prop :=
  r.length ≥ e.minor_axis ∧
  r.width ≥ e.minor_axis ∧
  r.length^2 + r.width^2 ≥ e.major_axis^2 + e.minor_axis^2

/-- The specific rectangle and ellipse from the problem -/
def problem_rectangle : Rectangle := ⟨140, 130⟩
def problem_ellipse : Ellipse := ⟨160, 100⟩

/-- Theorem stating that the problem_rectangle can cover the problem_ellipse -/
theorem rectangle_covers_ellipse : can_cover problem_rectangle problem_ellipse :=
  sorry

end rectangle_covers_ellipse_l1346_134612


namespace percent_of_percent_l1346_134619

theorem percent_of_percent (x : ℝ) :
  (20 / 100) * (x / 100) = 80 / 100 → x = 400 := by
  sorry

end percent_of_percent_l1346_134619


namespace distribute_five_balls_three_boxes_l1346_134681

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 balls into 3 boxes with one box always empty -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 3 :=
sorry

end distribute_five_balls_three_boxes_l1346_134681


namespace gcf_180_270_l1346_134680

theorem gcf_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcf_180_270_l1346_134680


namespace set_equality_implies_values_l1346_134658

theorem set_equality_implies_values (a b : ℝ) : 
  ({1, a, b} : Set ℝ) = {a, a^2, a*b} → a = -1 ∧ b = 0 := by
  sorry

end set_equality_implies_values_l1346_134658


namespace max_area_at_150_l1346_134698

/-- Represents a rectangular pasture with a fence on three sides and a barn on the fourth side. -/
structure Pasture where
  fenceLength : ℝ  -- Total length of fence available
  barnLength : ℝ   -- Length of the barn side

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn. -/
def Pasture.area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fenceLength - 2 * x)

/-- Theorem stating that the maximum area of the pasture occurs when the side parallel to the barn is 150 feet. -/
theorem max_area_at_150 (p : Pasture) (h1 : p.fenceLength = 300) (h2 : p.barnLength = 350) :
  ∃ (x : ℝ), x > 0 ∧ x < p.barnLength ∧
  (∀ (y : ℝ), y > 0 → y < p.barnLength → p.area x ≥ p.area y) ∧
  p.fenceLength - 2 * x = 150 := by
  sorry


end max_area_at_150_l1346_134698


namespace prob_at_most_one_eq_seven_twenty_sevenths_l1346_134633

/-- The probability of making a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The probability of making exactly k successful shots in n attempts -/
def binomial_prob (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of making at most 1 successful shot in 3 attempts -/
def prob_at_most_one : ℚ :=
  binomial_prob 0 + binomial_prob 1

theorem prob_at_most_one_eq_seven_twenty_sevenths : 
  prob_at_most_one = 7/27 := by sorry

end prob_at_most_one_eq_seven_twenty_sevenths_l1346_134633


namespace fixed_point_satisfies_line_fixed_point_unique_l1346_134637

/-- A line that passes through a fixed point for all values of m -/
def line (m x y : ℝ) : Prop :=
  (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6 = 0

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all m -/
theorem fixed_point_satisfies_line :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ m : ℝ, line m x y) → (x, y) = fixed_point :=
by sorry

end fixed_point_satisfies_line_fixed_point_unique_l1346_134637


namespace tangent_line_at_x_squared_l1346_134696

theorem tangent_line_at_x_squared (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := 2 * x₀
  (λ x ↦ m * (x - x₀) + y₀) = (λ x ↦ 4 * x - 4) := by sorry

end tangent_line_at_x_squared_l1346_134696


namespace boat_weight_problem_l1346_134636

theorem boat_weight_problem (initial_average : ℝ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_average = 60 →
  new_person_weight = 45 →
  new_average = 55 →
  ∃ n : ℕ, n * initial_average + new_person_weight = (n + 1) * new_average ∧ n = 2 :=
by sorry

end boat_weight_problem_l1346_134636


namespace log_four_eighteen_l1346_134654

theorem log_four_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 4 = (a + 2*b) / (2*a) := by sorry

end log_four_eighteen_l1346_134654


namespace fourth_power_sum_l1346_134641

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 18.5 := by
  sorry

end fourth_power_sum_l1346_134641


namespace cube_root_of_negative_64_l1346_134622

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end cube_root_of_negative_64_l1346_134622


namespace covered_boards_l1346_134624

/-- Represents a modified checkerboard with one corner removed. -/
structure ModifiedBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Checks if a modified board can be completely covered by dominoes. -/
def can_be_covered (board : ModifiedBoard) : Prop :=
  let total_squares := board.rows * board.cols - 1
  (total_squares % 2 = 0) ∧ 
  (board.rows ≥ 2) ∧ 
  (board.cols ≥ 2)

/-- Theorem stating which modified boards can be covered. -/
theorem covered_boards :
  (can_be_covered ⟨5, 5⟩) ∧
  (can_be_covered ⟨7, 3⟩) ∧
  ¬(can_be_covered ⟨4, 5⟩) ∧
  ¬(can_be_covered ⟨6, 5⟩) ∧
  ¬(can_be_covered ⟨5, 4⟩) :=
sorry

end covered_boards_l1346_134624


namespace not_q_sufficient_not_necessary_for_p_l1346_134606

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1/x < 1

-- Theorem stating that ¬q is a sufficient but not necessary condition for p
theorem not_q_sufficient_not_necessary_for_p :
  (∀ x : ℝ, ¬(q x) → p x) ∧ 
  (∃ x : ℝ, p x ∧ q x) :=
sorry

end not_q_sufficient_not_necessary_for_p_l1346_134606


namespace coordinate_change_l1346_134670

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v : Fin 3 → V) : Prop :=
  LinearIndependent ℝ v ∧ Submodule.span ℝ (Set.range v) = ⊤

theorem coordinate_change
  (a b c : V)
  (h1 : is_basis (![a, b, c]))
  (h2 : is_basis (![a - b, a + b, c]))
  (p : V)
  (h3 : p = 4 • a + 2 • b + (-1) • c) :
  ∃ (x y z : ℝ), p = x • (a - b) + y • (a + b) + z • c ∧ x = 1 ∧ y = 3 ∧ z = -1 :=
sorry

end coordinate_change_l1346_134670


namespace tank_flow_rate_l1346_134679

/-- Represents the flow rate problem for a water tank -/
theorem tank_flow_rate 
  (tank_capacity : ℝ) 
  (initial_level : ℝ) 
  (fill_time : ℝ) 
  (drain1_rate : ℝ) 
  (drain2_rate : ℝ) 
  (h1 : tank_capacity = 8000)
  (h2 : initial_level = tank_capacity / 2)
  (h3 : fill_time = 48)
  (h4 : drain1_rate = 1000 / 4)
  (h5 : drain2_rate = 1000 / 6)
  : ∃ (flow_rate : ℝ), 
    flow_rate = 500 ∧ 
    (flow_rate - (drain1_rate + drain2_rate)) * fill_time = tank_capacity - initial_level :=
by sorry


end tank_flow_rate_l1346_134679


namespace watermelon_price_in_units_l1346_134649

/-- The price of a watermelon in won -/
def watermelon_price : ℝ := 5000 - 200

/-- The conversion factor from won to units of 1000 won -/
def conversion_factor : ℝ := 1000

theorem watermelon_price_in_units : watermelon_price / conversion_factor = 4.8 := by
  sorry

end watermelon_price_in_units_l1346_134649


namespace bridge_length_calculation_l1346_134647

theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 250 →
  crossing_time = 25 →
  train_speed_kmh = 57.6 →
  ∃ (bridge_length : ℝ),
    bridge_length = 150 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end bridge_length_calculation_l1346_134647


namespace leftover_fraction_l1346_134665

def fractions : List ℚ := [5/4, 17/6, -5/4, 10/7, 2/3, 14/8, -1/3, 5/3, -3/2]

def has_sum_5_2 (a b : ℚ) : Prop := a + b = 5/2
def has_diff_5_2 (a b : ℚ) : Prop := a - b = 5/2
def has_prod_5_2 (a b : ℚ) : Prop := a * b = 5/2
def has_quot_5_2 (a b : ℚ) : Prop := a / b = 5/2

def is_in_pair (x : ℚ) : Prop :=
  ∃ y ∈ fractions, x ≠ y ∧ (has_sum_5_2 x y ∨ has_diff_5_2 x y ∨ has_prod_5_2 x y ∨ has_quot_5_2 x y)

theorem leftover_fraction :
  ∀ x ∈ fractions, x ≠ -3/2 → is_in_pair x :=
sorry

end leftover_fraction_l1346_134665


namespace cubic_equation_solution_l1346_134660

theorem cubic_equation_solution (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 25 * y^3) 
  (h3 : a - b = y) : 
  b = -(1 - Real.sqrt 33) / 2 * y ∨ b = -(1 + Real.sqrt 33) / 2 * y := by
  sorry

end cubic_equation_solution_l1346_134660


namespace find_a_range_of_t_l1346_134689

-- Define the function f
def f (x a : ℝ) := |2 * x - a| + a

-- Theorem 1
theorem find_a : 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) → 
  (∃! a, ∀ x, f x a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
sorry

-- Theorem 2
theorem range_of_t :
  (∀ t : ℝ, (∃ n : ℝ, |2 * n - 1| + 1 ≤ t - (|2 * (-n) - 1| + 1)) ↔ t ≥ 4) :=
sorry

end find_a_range_of_t_l1346_134689


namespace triathlon_speed_l1346_134693

/-- Triathlon problem -/
theorem triathlon_speed (total_time swim_dist swim_speed run_dist run_speed rest_time bike_dist : ℝ) 
  (h_total : total_time = 2.25)
  (h_swim : swim_dist = 0.5)
  (h_swim_speed : swim_speed = 2)
  (h_run : run_dist = 4)
  (h_run_speed : run_speed = 8)
  (h_rest : rest_time = 1/6)
  (h_bike : bike_dist = 20) :
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed + rest_time)) = 15 := by
  sorry

end triathlon_speed_l1346_134693


namespace fathers_age_l1346_134664

theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 8 = (father_age + 8) / 2 →
  father_age = 40 := by
sorry

end fathers_age_l1346_134664


namespace inequality_proof_l1346_134601

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ (3*(a+1)*(b+1)*(c+1))/(a*b*c+1) := by
  sorry

end inequality_proof_l1346_134601


namespace expand_product_l1346_134672

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + 6) = 3*x^3 - 6*x^2 + 18*x - 36 := by
  sorry

end expand_product_l1346_134672


namespace hyperbola_equation_l1346_134651

/-- Definition of a hyperbola with given foci and distance property -/
structure Hyperbola where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  dist_diff : ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: The standard equation of the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_f1 : h.f1 = (-5, 0))
    (h_f2 : h.f2 = (5, 0))
    (h_dist : h.dist_diff = 8) :
    ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ‖p - h.f1‖ - ‖p - h.f2‖ = h.dist_diff} →
    standard_equation 4 3 x y := by
  sorry

end hyperbola_equation_l1346_134651


namespace events_mutually_exclusive_and_complementary_l1346_134674

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Represents the event "at least 1 girl" -/
def at_least_one_girl : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event "all boys" -/
def all_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- Proves that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_girl ∩ all_boys = ∅) ∧
  (at_least_one_girl ∪ all_boys = Set.univ) :=
sorry

end events_mutually_exclusive_and_complementary_l1346_134674


namespace tan_cos_expression_equals_negative_one_l1346_134626

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_cos_expression_equals_negative_one_l1346_134626


namespace ratio_change_l1346_134662

theorem ratio_change (x y : ℕ) (n : ℕ) (h1 : y = 24) (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : n = 6 := by
  sorry

end ratio_change_l1346_134662


namespace symmetric_function_theorem_l1346_134694

/-- A function f: ℝ → ℝ is symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem symmetric_function_theorem (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 - x)) :
  ∀ x ≤ 0, f x = x * (x + 1) := by
  sorry

end symmetric_function_theorem_l1346_134694


namespace fraction_of_satisfactory_grades_l1346_134685

-- Define the grades
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def is_satisfactory (g : Grade) : Prop :=
  g = Grade.B ∨ g = Grade.C ∨ g = Grade.D

-- Define the number of students for each grade
def num_students (g : Grade) : ℕ :=
  match g with
  | Grade.A => 8
  | Grade.B => 6
  | Grade.C => 5
  | Grade.D => 4
  | Grade.F => 7

-- Define the total number of students
def total_students : ℕ :=
  num_students Grade.A + num_students Grade.B + num_students Grade.C +
  num_students Grade.D + num_students Grade.F

-- Define the number of students with satisfactory grades
def satisfactory_students : ℕ :=
  num_students Grade.B + num_students Grade.C + num_students Grade.D

-- Theorem to prove
theorem fraction_of_satisfactory_grades :
  (satisfactory_students : ℚ) / total_students = 1 / 2 := by
  sorry

end fraction_of_satisfactory_grades_l1346_134685


namespace sum_minus_k_equals_ten_l1346_134608

theorem sum_minus_k_equals_ten (n k : ℕ) (a : ℕ) (h1 : 1 < k) (h2 : k < n) 
  (h3 : (n * (n + 1) / 2 - k) / (n - 1) = 10) (h4 : n + k = a) : a = 29 := by
  sorry

end sum_minus_k_equals_ten_l1346_134608


namespace inverse_of_5_mod_34_l1346_134630

theorem inverse_of_5_mod_34 : ∃ x : ℕ, x < 34 ∧ (5 * x) % 34 = 1 := by
  sorry

end inverse_of_5_mod_34_l1346_134630


namespace movie_tickets_difference_l1346_134687

theorem movie_tickets_difference (x y : ℕ) : 
  x + y = 30 →
  10 * x + 20 * y = 500 →
  y > x →
  y - x = 10 :=
by sorry

end movie_tickets_difference_l1346_134687


namespace sqrt_sum_equals_seven_l1346_134618

theorem sqrt_sum_equals_seven (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end sqrt_sum_equals_seven_l1346_134618


namespace banana_arrangements_l1346_134623

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60) := by
sorry

end banana_arrangements_l1346_134623


namespace kenneth_remaining_money_l1346_134621

-- Define the initial amount Kenneth has
def initial_amount : ℕ := 50

-- Define the number of baguettes and bottles of water
def num_baguettes : ℕ := 2
def num_water_bottles : ℕ := 2

-- Define the cost of each baguette and bottle of water
def cost_baguette : ℕ := 2
def cost_water : ℕ := 1

-- Define the total cost of purchases
def total_cost : ℕ := num_baguettes * cost_baguette + num_water_bottles * cost_water

-- Define the remaining money after purchases
def remaining_money : ℕ := initial_amount - total_cost

-- Theorem statement
theorem kenneth_remaining_money :
  remaining_money = 44 :=
by sorry

end kenneth_remaining_money_l1346_134621


namespace greatest_power_of_200_dividing_100_factorial_l1346_134646

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem greatest_power_of_200_dividing_100_factorial :
  (∃ k : ℕ, 200^k ∣ factorial 100 ∧ ∀ m : ℕ, m > k → ¬(200^m ∣ factorial 100)) ∧
  (∀ k : ℕ, 200^k ∣ factorial 100 → k ≤ 12) ∧
  (200^12 ∣ factorial 100) :=
sorry

end greatest_power_of_200_dividing_100_factorial_l1346_134646


namespace consecutive_draw_probability_l1346_134656

def num_purple_chips : ℕ := 4
def num_orange_chips : ℕ := 3
def num_green_chips : ℕ := 5
def total_chips : ℕ := num_purple_chips + num_orange_chips + num_green_chips

def probability_consecutive_draw : ℚ :=
  (Nat.factorial 2 * Nat.factorial num_purple_chips * Nat.factorial num_orange_chips * Nat.factorial num_green_chips) /
  Nat.factorial total_chips

theorem consecutive_draw_probability :
  probability_consecutive_draw = 1 / 13860 := by
  sorry

end consecutive_draw_probability_l1346_134656


namespace quadratic_inequality_solution_l1346_134607

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
sorry

end quadratic_inequality_solution_l1346_134607


namespace arithmetic_sequence_property_l1346_134603

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition for the arithmetic sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 6 + a 10 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SequenceCondition a) : 
  a 3 + a 9 = 60 := by
  sorry

end arithmetic_sequence_property_l1346_134603


namespace water_bottle_consumption_l1346_134675

/-- Proves that given a 24-pack of bottled water, if 1/3 is consumed on the first day
    and 1/2 of the remainder is consumed on the second day, then 8 bottles remain after 2 days. -/
theorem water_bottle_consumption (initial_bottles : ℕ) 
  (h1 : initial_bottles = 24)
  (first_day_consumption : ℚ) 
  (h2 : first_day_consumption = 1/3)
  (second_day_consumption : ℚ) 
  (h3 : second_day_consumption = 1/2) :
  initial_bottles - 
  (↑initial_bottles * first_day_consumption).floor - 
  ((↑initial_bottles - (↑initial_bottles * first_day_consumption).floor) * second_day_consumption).floor = 8 :=
by sorry

end water_bottle_consumption_l1346_134675


namespace x_plus_reciprocal_three_l1346_134615

theorem x_plus_reciprocal_three (x : ℝ) (h : x ≠ 0) :
  x + 1/x = 3 →
  (x - 1)^2 + 16/(x - 1)^2 = x + 16/x :=
by
  sorry

end x_plus_reciprocal_three_l1346_134615


namespace diophantine_equation_solution_l1346_134600

theorem diophantine_equation_solution :
  ∀ (p a b c : ℕ),
    p.Prime →
    0 < a ∧ 0 < b ∧ 0 < c →
    73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 →
    ((p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4)) :=
by sorry

end diophantine_equation_solution_l1346_134600


namespace parabola_vertex_l1346_134632

/-- The vertex of a parabola defined by y^2 + 8y + 4x + 5 = 0 is (11/4, -4) -/
theorem parabola_vertex : 
  let f (x y : ℝ) := y^2 + 8*y + 4*x + 5
  ∃! (vx vy : ℝ), (∀ (x y : ℝ), f x y = 0 → (x - vx)^2 ≥ 0) ∧ vx = 11/4 ∧ vy = -4 := by
  sorry

end parabola_vertex_l1346_134632


namespace c_worked_four_days_l1346_134676

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 125

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1850

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ
  | 0 => 3  -- a's ratio
  | 1 => 4  -- b's ratio
  | 2 => 5  -- c's ratio

/-- Calculates the daily wage for a given person based on c's wage and the ratio -/
def daily_wage (person : Fin 3) : ℕ :=
  wage_c * wage_ratio person / wage_ratio 2

/-- Theorem stating that person c worked for 4 days -/
theorem c_worked_four_days :
  ∃ (days_c : ℕ), 
    days_c * daily_wage 2 + 
    days_a * daily_wage 0 + 
    days_b * daily_wage 1 = total_earnings ∧
    days_c = 4 := by
  sorry

end c_worked_four_days_l1346_134676


namespace perpendicular_line_proof_l1346_134616

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2

def point : ℝ × ℝ := (1, 2)

def tangent_slope : ℝ := 4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 9 = 0

theorem perpendicular_line_proof :
  perpendicular_line point.1 point.2 ∧
  (∃ k : ℝ, k * tangent_slope = -1 ∧
    ∀ x y : ℝ, perpendicular_line x y ↔ y - point.2 = k * (x - point.1)) :=
sorry

end perpendicular_line_proof_l1346_134616


namespace quadratic_properties_l1346_134653

-- Define the quadratic function
def quadratic (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  -- Part 1
  (quadratic b c (-1) = 0 ∧ quadratic b c 3 = 0 →
    ∃ x, ∀ y, quadratic b c y ≤ quadratic b c x ∧ quadratic b c x = 4) ∧
  -- Part 2
  (c = -5 ∧ (∃! x, quadratic b c x = 1) →
    b = 2 * Real.sqrt 6 ∨ b = -2 * Real.sqrt 6) ∧
  -- Part 3
  (c = b^2 ∧ (∃ x, b ≤ x ∧ x ≤ b + 3 ∧
    ∀ y, b ≤ y ∧ y ≤ b + 3 → quadratic b c y ≤ quadratic b c x) ∧
    quadratic b c x = 20 →
    b = 2 * Real.sqrt 5 ∨ b = -4) :=
by sorry

end quadratic_properties_l1346_134653


namespace sqrt_product_equality_l1346_134635

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l1346_134635


namespace probability_not_above_x_axis_is_half_l1346_134668

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- The probability of a point not being above the x-axis in a given parallelogram -/
def probabilityNotAboveXAxis (para : Parallelogram) : ℝ := sorry

/-- The specific parallelogram PQRS from the problem -/
def pqrs : Parallelogram :=
  { p := { x := 4, y := 4 }
  , q := { x := -2, y := -2 }
  , r := { x := -8, y := -2 }
  , s := { x := -2, y := 4 }
  }

theorem probability_not_above_x_axis_is_half :
  probabilityNotAboveXAxis pqrs = 1/2 := by sorry

end probability_not_above_x_axis_is_half_l1346_134668


namespace one_third_to_fifth_power_l1346_134602

theorem one_third_to_fifth_power : (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end one_third_to_fifth_power_l1346_134602


namespace union_of_A_and_B_l1346_134686

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 2} := by sorry

end union_of_A_and_B_l1346_134686
