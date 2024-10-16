import Mathlib

namespace NUMINAMATH_CALUDE_families_left_near_mountain_l2220_222009

/-- The number of bird families initially living near the mountain. -/
def initial_families : ℕ := 41

/-- The number of bird families that flew away for the winter. -/
def families_flew_away : ℕ := 27

/-- Theorem: The number of bird families left near the mountain is 14. -/
theorem families_left_near_mountain :
  initial_families - families_flew_away = 14 := by
  sorry

end NUMINAMATH_CALUDE_families_left_near_mountain_l2220_222009


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l2220_222019

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 435

/-- Calculates the number of handshakes between players given the number of players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the number of handshakes the coach had -/
def coach_handshakes (n : ℕ) : ℕ := total_handshakes - player_handshakes n

theorem min_coach_handshakes :
  ∃ (n : ℕ), n > 1 ∧ player_handshakes n ≤ total_handshakes ∧ coach_handshakes n = 0 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l2220_222019


namespace NUMINAMATH_CALUDE_password_from_polynomial_factorization_password_for_given_values_l2220_222007

/-- Generates a password from the factors of x^3 - xy^2 --/
def generate_password (x y : ℕ) : ℕ :=
  x * 10000 + (x + y) * 100 + (x - y)

/-- The polynomial x^3 - xy^2 factors as x(x-y)(x+y) --/
theorem password_from_polynomial_factorization (x y : ℕ) :
  x^3 - x*y^2 = x * (x - y) * (x + y) :=
sorry

/-- The password generated from x^3 - xy^2 with x=18 and y=5 is 181323 --/
theorem password_for_given_values :
  generate_password 18 5 = 181323 :=
sorry

end NUMINAMATH_CALUDE_password_from_polynomial_factorization_password_for_given_values_l2220_222007


namespace NUMINAMATH_CALUDE_largest_even_from_powerful_digits_l2220_222076

/-- A natural number is powerful if n + (n+1) + (n+2) has no carrying over --/
def isPowerful (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → (n / 10^d % 10 + (n+1) / 10^d % 10 + (n+2) / 10^d % 10) < 10

/-- The set of powerful numbers less than 1000 --/
def powerfulSet : Set ℕ := {n | n < 1000 ∧ isPowerful n}

/-- The set of digits from powerful numbers less than 1000 --/
def powerfulDigits : Set ℕ := {d | ∃ n ∈ powerfulSet, ∃ k, n / 10^k % 10 = d}

/-- An even number formed by non-repeating digits from powerfulDigits --/
def validNumber (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 
  (∀ d, d ∈ powerfulDigits → (∃! k, n / 10^k % 10 = d)) ∧
  (∀ k, n / 10^k % 10 ∈ powerfulDigits)

theorem largest_even_from_powerful_digits :
  ∃ n, validNumber n ∧ ∀ m, validNumber m → m ≤ n ∧ n = 43210 :=
sorry

end NUMINAMATH_CALUDE_largest_even_from_powerful_digits_l2220_222076


namespace NUMINAMATH_CALUDE_complex_fraction_product_l2220_222063

theorem complex_fraction_product (a b : ℝ) :
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b →
  a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l2220_222063


namespace NUMINAMATH_CALUDE_grade_ratio_l2220_222089

/-- Proves that the ratio of Bob's grade to Jason's grade is 1:2 -/
theorem grade_ratio (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = 35 →
  (bob_grade : ℚ) / jason_grade = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_grade_ratio_l2220_222089


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2220_222051

theorem rectangle_perimeter (area : ℝ) (length width : ℝ) : 
  area = 450 ∧ length = 2 * width ∧ area = length * width → 
  2 * (length + width) = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2220_222051


namespace NUMINAMATH_CALUDE_four_lines_intersect_l2220_222003

-- Define the basic structures
structure Point := (x y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)

structure Parallelogram :=
  (E F G H : Point)

-- Define the properties
def is_inscribed (p : Parallelogram) (q : Quadrilateral) : Prop := sorry

def sides_parallel_to_diagonals (p : Parallelogram) (q : Quadrilateral) : Prop := sorry

def is_midpoint (M : Point) (A B : Point) : Prop := sorry

def line_through (P Q : Point) : Set Point := sorry

-- Define the centroid
def centroid (q : Quadrilateral) : Point := sorry

-- Main theorem
theorem four_lines_intersect (ABCD : Quadrilateral) (EFGH : Parallelogram)
  (E' F' G' H' : Point) :
  is_inscribed EFGH ABCD →
  sides_parallel_to_diagonals EFGH ABCD →
  is_midpoint E' ABCD.A ABCD.B →
  is_midpoint F' ABCD.B ABCD.C →
  is_midpoint G' ABCD.C ABCD.D →
  is_midpoint H' ABCD.D ABCD.A →
  ∃ O, O ∈ line_through E EFGH.E ∩
        line_through F EFGH.F ∩
        line_through G EFGH.G ∩
        line_through H EFGH.H ∧
     O = centroid ABCD :=
sorry

end NUMINAMATH_CALUDE_four_lines_intersect_l2220_222003


namespace NUMINAMATH_CALUDE_subtracted_number_l2220_222041

theorem subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 6 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2220_222041


namespace NUMINAMATH_CALUDE_long_division_unique_solution_l2220_222042

theorem long_division_unique_solution :
  ∃! (dividend divisor quotient : ℕ),
    dividend ≥ 100000 ∧ dividend < 1000000 ∧
    divisor ≥ 100 ∧ divisor < 1000 ∧
    quotient ≥ 100 ∧ quotient < 1000 ∧
    quotient % 10 = 8 ∧
    (divisor * (quotient / 100)) % 10 = 5 ∧
    dividend = divisor * quotient :=
by sorry

end NUMINAMATH_CALUDE_long_division_unique_solution_l2220_222042


namespace NUMINAMATH_CALUDE_square_root_expression_equals_256_l2220_222059

theorem square_root_expression_equals_256 :
  Real.sqrt ((16^12 + 2^36) / (16^5 + 2^42)) = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_equals_256_l2220_222059


namespace NUMINAMATH_CALUDE_sum_of_roots_l2220_222027

theorem sum_of_roots (h b x₁ x₂ : ℝ) 
  (hx : x₁ ≠ x₂) 
  (eq₁ : 3 * x₁^2 - h * x₁ = b) 
  (eq₂ : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2220_222027


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l2220_222095

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 108)
  (h2 : eaten_candy = 36)
  (h3 : num_piles = 8) :
  (initial_candy - eaten_candy) / num_piles = 9 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l2220_222095


namespace NUMINAMATH_CALUDE_quadratic_form_value_l2220_222098

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    prove that when written in the form (ax + b)^2 + c,
    the value of 2b - 3c is -20 -/
theorem quadratic_form_value (a b c : ℝ) : 
  (∀ x, 4 * x^2 - 40 * x + 100 = (a * x + b)^2 + c) →
  2 * b - 3 * c = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_value_l2220_222098


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2220_222049

theorem contrapositive_equivalence (m : ℝ) : 
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔ 
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2220_222049


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l2220_222044

/-- In a triangle ABC, if b cos C = (3a - c) cos B, then cos B = 1/3 -/
theorem cosine_value_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.cos C = (3 * a - c) * Real.cos B) →
  Real.cos B = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_cosine_value_in_triangle_l2220_222044


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2220_222026

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2220_222026


namespace NUMINAMATH_CALUDE_donut_holes_problem_l2220_222014

/-- Given the number of mini-cupcakes, students, and desserts per student,
    calculate the number of donut holes needed. -/
def donut_holes_needed (mini_cupcakes : ℕ) (students : ℕ) (desserts_per_student : ℕ) : ℕ :=
  students * desserts_per_student - mini_cupcakes

/-- Theorem stating that given 14 mini-cupcakes, 13 students, and 2 desserts per student,
    the number of donut holes needed is 12. -/
theorem donut_holes_problem :
  donut_holes_needed 14 13 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_donut_holes_problem_l2220_222014


namespace NUMINAMATH_CALUDE_raft_sticks_difference_l2220_222034

theorem raft_sticks_difference (simon_sticks : ℕ) (total_sticks : ℕ) : 
  simon_sticks = 36 →
  total_sticks = 129 →
  let gerry_sticks := (2 * simon_sticks) / 3
  let simon_and_gerry_sticks := simon_sticks + gerry_sticks
  let micky_sticks := total_sticks - simon_and_gerry_sticks
  micky_sticks - simon_and_gerry_sticks = 9 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_difference_l2220_222034


namespace NUMINAMATH_CALUDE_radar_coverage_l2220_222074

noncomputable def n : ℕ := 7
def r : ℝ := 41
def w : ℝ := 18

theorem radar_coverage (n : ℕ) (r w : ℝ) 
  (h_n : n = 7) 
  (h_r : r = 41) 
  (h_w : w = 18) :
  ∃ (max_distance area : ℝ),
    max_distance = 40 / Real.sin (180 / n * π / 180) ∧
    area = 1440 * π / Real.tan (180 / n * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_radar_coverage_l2220_222074


namespace NUMINAMATH_CALUDE_equation_solution_l2220_222078

theorem equation_solution (x : ℝ) : 
  x^2 + 3*x + 2 ≠ 0 →
  (-x^2 = (4*x + 2) / (x^2 + 3*x + 2)) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2220_222078


namespace NUMINAMATH_CALUDE_richards_walking_ratio_l2220_222017

/-- Proves that the ratio of Richard's second day walking distance to his first day walking distance is 1/5 --/
theorem richards_walking_ratio :
  let total_distance : ℝ := 70
  let first_day : ℝ := 20
  let third_day : ℝ := 10
  let remaining : ℝ := 36
  let second_day := total_distance - remaining - first_day - third_day
  second_day / first_day = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_richards_walking_ratio_l2220_222017


namespace NUMINAMATH_CALUDE_fifth_root_of_x_fourth_root_of_x_l2220_222050

theorem fifth_root_of_x_fourth_root_of_x (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_x_fourth_root_of_x_l2220_222050


namespace NUMINAMATH_CALUDE_valid_parameterization_l2220_222054

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- Checks if a given vector parameterization is valid for the line y = 2x - 4 -/
def isValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 4 ∧ ∃ k : ℝ, p.a = k * 1 ∧ p.b = k * 2

/-- The theorem stating the conditions for a valid vector parameterization -/
theorem valid_parameterization (p : VectorParam) : 
  isValidParam p ↔ 
  (∀ t : ℝ, (p.x₀ + t * p.a, p.y₀ + t * p.b) ∈ {(x, y) : ℝ × ℝ | y = 2 * x - 4}) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l2220_222054


namespace NUMINAMATH_CALUDE_opposite_is_negation_l2220_222032

-- Define the concept of opposite number
def opposite (a : ℝ) : ℝ := -a

-- Theorem stating that the opposite of a is -a
theorem opposite_is_negation (a : ℝ) : opposite a = -a := by
  sorry

end NUMINAMATH_CALUDE_opposite_is_negation_l2220_222032


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2220_222021

theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2220_222021


namespace NUMINAMATH_CALUDE_bug_ends_on_two_l2220_222088

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five
| six

/-- Defines the movement rules for the bug -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.four
  | Point.three => Point.four
  | Point.four => Point.one
  | Point.five => Point.six
  | Point.six => Point.two

/-- Simulates the bug's movement for a given number of jumps -/
def bug_position (start : Point) (jumps : Nat) : Point :=
  match jumps with
  | 0 => start
  | n + 1 => next_point (bug_position start n)

/-- The main theorem to prove -/
theorem bug_ends_on_two :
  bug_position Point.six 2000 = Point.two := by
  sorry

end NUMINAMATH_CALUDE_bug_ends_on_two_l2220_222088


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2220_222082

theorem quadratic_roots_sum_of_squares (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^2 + q^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2220_222082


namespace NUMINAMATH_CALUDE_sin_cos_sum_2023_17_l2220_222023

theorem sin_cos_sum_2023_17 :
  Real.sin (2023 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (2023 * π / 180) * Real.sin (17 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_2023_17_l2220_222023


namespace NUMINAMATH_CALUDE_students_with_average_age_16_l2220_222092

theorem students_with_average_age_16 (total_students : ℕ) (total_avg_age : ℕ) 
  (students_avg_14 : ℕ) (age_15th_student : ℕ) :
  total_students = 15 →
  total_avg_age = 15 →
  students_avg_14 = 5 →
  age_15th_student = 11 →
  ∃ (students_avg_16 : ℕ),
    students_avg_16 = 9 ∧
    students_avg_16 * 16 = total_students * total_avg_age - students_avg_14 * 14 - age_15th_student :=
by sorry

end NUMINAMATH_CALUDE_students_with_average_age_16_l2220_222092


namespace NUMINAMATH_CALUDE_baker_donuts_l2220_222073

theorem baker_donuts (total_donuts : ℕ) (boxes : ℕ) (extra_donuts : ℕ) : 
  boxes = 7 → 
  extra_donuts = 6 → 
  ∃ (n : ℕ), n > 0 ∧ total_donuts = 7 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_donuts_l2220_222073


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l2220_222037

theorem no_solution_in_interval : ¬∃ x : ℝ, 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l2220_222037


namespace NUMINAMATH_CALUDE_contractor_problem_l2220_222046

theorem contractor_problem (total_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) :
  total_days = 6 →
  absent_workers = 7 →
  actual_days = 10 →
  ∃ (original_workers : ℕ), 
    original_workers * total_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 18 :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l2220_222046


namespace NUMINAMATH_CALUDE_power_of_product_squared_l2220_222000

theorem power_of_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l2220_222000


namespace NUMINAMATH_CALUDE_remaining_time_formula_l2220_222043

/-- Represents the exam scenario for Jessica -/
structure ExamScenario where
  totalTime : ℕ  -- Total time for the exam in minutes
  totalQuestions : ℕ  -- Total number of questions
  answeredQuestions : ℕ  -- Number of questions answered so far
  timeUsed : ℕ  -- Time used so far in minutes
  penaltyPerIncorrect : ℕ  -- Time penalty for each incorrect answer in minutes

/-- Calculates the remaining time after penalties -/
def remainingTimeAfterPenalties (scenario : ExamScenario) (incorrectAnswers : ℕ) : ℤ :=
  scenario.totalTime - scenario.timeUsed - 
  (scenario.totalQuestions - scenario.answeredQuestions) * 
  (scenario.timeUsed / scenario.answeredQuestions) -
  incorrectAnswers * scenario.penaltyPerIncorrect

/-- Theorem stating that the remaining time after penalties is 15 - 2x -/
theorem remaining_time_formula (incorrectAnswers : ℕ) : 
  remainingTimeAfterPenalties 
    { totalTime := 90
    , totalQuestions := 100
    , answeredQuestions := 20
    , timeUsed := 15
    , penaltyPerIncorrect := 2 } 
    incorrectAnswers = 15 - 2 * incorrectAnswers :=
by sorry

end NUMINAMATH_CALUDE_remaining_time_formula_l2220_222043


namespace NUMINAMATH_CALUDE_forester_count_impossible_l2220_222093

/-- Represents a circle in the forest --/
structure Circle where
  id : Nat
  trees : Finset Nat

/-- Represents the forest with circles and pine trees --/
structure Forest where
  circles : Finset Circle
  total_trees : Finset Nat

/-- The property that each circle contains exactly 3 distinct trees --/
def validCount (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees.card = 3

/-- The property that all trees in circles are from the total set of trees --/
def validTrees (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees ⊆ f.total_trees

/-- The main theorem stating the impossibility of the forester's count --/
theorem forester_count_impossible (f : Forest) :
  f.circles.card = 5 → validCount f → validTrees f → False := by
  sorry

end NUMINAMATH_CALUDE_forester_count_impossible_l2220_222093


namespace NUMINAMATH_CALUDE_course_assessment_probabilities_l2220_222004

/-- Represents a student in the course -/
inductive Student := | A | B | C

/-- Represents the type of assessment -/
inductive AssessmentType := | Theory | Experimental

/-- The probability of a student passing a specific assessment type -/
def passProbability (s : Student) (t : AssessmentType) : ℝ :=
  match s, t with
  | Student.A, AssessmentType.Theory => 0.9
  | Student.B, AssessmentType.Theory => 0.8
  | Student.C, AssessmentType.Theory => 0.7
  | Student.A, AssessmentType.Experimental => 0.8
  | Student.B, AssessmentType.Experimental => 0.7
  | Student.C, AssessmentType.Experimental => 0.9

/-- The probability of at least two students passing the theory assessment -/
def atLeastTwoPassTheory : ℝ := sorry

/-- The probability of all three students passing both assessments -/
def allPassBoth : ℝ := sorry

theorem course_assessment_probabilities :
  (atLeastTwoPassTheory = 0.902) ∧ (allPassBoth = 0.254) := by sorry

end NUMINAMATH_CALUDE_course_assessment_probabilities_l2220_222004


namespace NUMINAMATH_CALUDE_inequality_proof_l2220_222033

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (3 : ℝ) / 2 < 1 / (a^3 + 1) + 1 / (b^3 + 1) ∧ 1 / (a^3 + 1) + 1 / (b^3 + 1) ≤ 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2220_222033


namespace NUMINAMATH_CALUDE_error_probability_theorem_l2220_222094

-- Define the probability of error
def probability_of_error : ℝ := 0.01

-- Define the observed value of K²
def observed_k_squared : ℝ := 6.635

-- Define the relationship between variables
def relationship_exists : Prop := True

-- Define the conclusion of the statistical test
def statistical_conclusion (p : ℝ) (relationship : Prop) : Prop :=
  p ≤ probability_of_error ∧ relationship

-- Theorem statement
theorem error_probability_theorem 
  (h : statistical_conclusion probability_of_error relationship_exists) :
  probability_of_error = 0.01 := by sorry

end NUMINAMATH_CALUDE_error_probability_theorem_l2220_222094


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2220_222018

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 * a * (-1) - b * 2 + 2 = 0) → 
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ c d : ℝ, c > 0 → d > 0 → (2 * c * (-1) - d * 2 + 2 = 0) → 1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2220_222018


namespace NUMINAMATH_CALUDE_rectangle_to_total_height_ratio_l2220_222053

/-- Represents an octagon with specific properties -/
structure Octagon :=
  (area : ℝ)
  (rectangle_width : ℝ)
  (triangle_base : ℝ)

/-- Properties of the octagon -/
axiom octagon_properties (o : Octagon) :
  o.area = 12 ∧
  o.rectangle_width = 3 ∧
  o.triangle_base = 3

/-- The diagonal bisects the area of the octagon -/
axiom diagonal_bisects (o : Octagon) (rectangle_height : ℝ) :
  o.rectangle_width * rectangle_height = o.area / 2

/-- The total height of the octagon -/
def total_height (o : Octagon) (rectangle_height : ℝ) : ℝ :=
  2 * rectangle_height

/-- Theorem: The ratio of rectangle height to total height is 1/2 -/
theorem rectangle_to_total_height_ratio (o : Octagon) (rectangle_height : ℝ) :
  rectangle_height / (total_height o rectangle_height) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_total_height_ratio_l2220_222053


namespace NUMINAMATH_CALUDE_triangle_inequality_l2220_222052

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  -- Add triangle inequality conditions
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.area ∧
  (t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * t.area ↔ isEquilateral t) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2220_222052


namespace NUMINAMATH_CALUDE_work_absence_problem_l2220_222065

theorem work_absence_problem (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_received : ℕ) :
  total_days = 30 →
  daily_wage = 10 →
  daily_fine = 2 →
  total_received = 216 →
  ∃ (absent_days : ℕ),
    absent_days = 7 ∧
    total_received = daily_wage * (total_days - absent_days) - daily_fine * absent_days :=
by sorry

end NUMINAMATH_CALUDE_work_absence_problem_l2220_222065


namespace NUMINAMATH_CALUDE_final_answer_is_67_l2220_222038

/-- Ben's operations on a number -/
def ben_operations (x : ℕ) : ℕ := ((x + 2) * 3) + 5

/-- Sue's operations on a number -/
def sue_operations (x : ℕ) : ℕ := ((x - 3) * 3) + 7

/-- Theorem: If Ben thinks of 4 and they perform their operations, Sue's final answer is 67 -/
theorem final_answer_is_67 : sue_operations (ben_operations 4) = 67 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_is_67_l2220_222038


namespace NUMINAMATH_CALUDE_handshake_theorem_l2220_222091

theorem handshake_theorem (n : ℕ) (h : n = 40) : 
  (n * (n - 1)) / 2 = 780 := by
  sorry

#check handshake_theorem

end NUMINAMATH_CALUDE_handshake_theorem_l2220_222091


namespace NUMINAMATH_CALUDE_james_fleet_capacity_l2220_222071

/-- Represents the fleet of gas transportation vans --/
structure Fleet :=
  (total_vans : ℕ)
  (large_vans : ℕ)
  (medium_vans : ℕ)
  (small_van : ℕ)
  (medium_capacity : ℕ)
  (small_capacity : ℕ)
  (large_capacity : ℕ)

/-- Calculates the total capacity of the fleet --/
def total_capacity (f : Fleet) : ℕ :=
  f.large_vans * f.medium_capacity +
  f.medium_vans * f.medium_capacity +
  f.small_van * f.small_capacity +
  (f.total_vans - f.large_vans - f.medium_vans - f.small_van) * f.large_capacity

/-- Theorem stating the total capacity of James' fleet --/
theorem james_fleet_capacity :
  ∃ (f : Fleet),
    f.total_vans = 6 ∧
    f.medium_vans = 2 ∧
    f.small_van = 1 ∧
    f.medium_capacity = 8000 ∧
    f.small_capacity = (7 * f.medium_capacity) / 10 ∧
    f.large_capacity = (3 * f.medium_capacity) / 2 ∧
    total_capacity f = 57600 :=
  sorry

end NUMINAMATH_CALUDE_james_fleet_capacity_l2220_222071


namespace NUMINAMATH_CALUDE_square_difference_plus_square_l2220_222010

theorem square_difference_plus_square : 5^2 - 4^2 + 3^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_square_l2220_222010


namespace NUMINAMATH_CALUDE_right_triangle_subdivision_l2220_222040

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  hypotenuse_bounds : 100 < AB ∧ AB < 101
  leg_bounds : 99 < AC ∧ AC < 100

-- Define a subdivision of a triangle
structure Subdivision where
  num_triangles : ℕ
  has_unit_side : Bool

-- Theorem statement
theorem right_triangle_subdivision (t : RightTriangle) :
  ∃ (s : Subdivision), s.num_triangles ≤ 21 ∧ s.has_unit_side = true := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_subdivision_l2220_222040


namespace NUMINAMATH_CALUDE_triangle_middle_side_bound_l2220_222022

theorem triangle_middle_side_bound (a b c : ℝ) (h_area : 1 = (1/2) * b * c * Real.sin α) 
  (h_order : a ≥ b ∧ b ≥ c) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a < b + c) :
  b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_middle_side_bound_l2220_222022


namespace NUMINAMATH_CALUDE_distance_to_origin_of_fourth_point_on_circle_l2220_222068

/-- Given four points on a circle, prove that the distance from the fourth point to the origin is √13 -/
theorem distance_to_origin_of_fourth_point_on_circle 
  (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 1)) 
  (hB : B = (-1, 0)) 
  (hC : C = (2, 3)) 
  (hD : D.2 = 3) 
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 ∧
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = radius^2) :
  Real.sqrt (D.1^2 + D.2^2) = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_origin_of_fourth_point_on_circle_l2220_222068


namespace NUMINAMATH_CALUDE_undefined_values_count_l2220_222096

theorem undefined_values_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + 2*x - 3) * (x - 3) = 0) ∧ Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_count_l2220_222096


namespace NUMINAMATH_CALUDE_bianca_cupcakes_theorem_l2220_222085

/-- Represents the number of cupcakes Bianca made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Bianca made 17 cupcakes after selling the first batch -/
theorem bianca_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final : ℕ)
  (h1 : initial = 14)
  (h2 : sold = 6)
  (h3 : final = 25) :
  cupcakes_made_after initial sold final = 17 := by
  sorry

end NUMINAMATH_CALUDE_bianca_cupcakes_theorem_l2220_222085


namespace NUMINAMATH_CALUDE_girls_in_ritas_class_l2220_222012

/-- Calculates the number of girls in a class given the total number of students and the ratio of girls to boys -/
def girls_in_class (total_students : ℕ) (girl_ratio : ℕ) (boy_ratio : ℕ) : ℕ :=
  (total_students * girl_ratio) / (girl_ratio + boy_ratio)

/-- Theorem stating that in a class with 35 students and a 3:4 ratio of girls to boys, there are 15 girls -/
theorem girls_in_ritas_class :
  girls_in_class 35 3 4 = 15 := by
  sorry

#eval girls_in_class 35 3 4

end NUMINAMATH_CALUDE_girls_in_ritas_class_l2220_222012


namespace NUMINAMATH_CALUDE_max_garden_area_l2220_222039

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 180
  length_constraint : length ≥ 100
  width_constraint : width ≥ 60

/-- Calculates the area of a garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∀ g : Garden, garden_area g ≤ 8000 ∧
  ∃ g_max : Garden, g_max.length = 100 ∧ g_max.width = 80 ∧ garden_area g_max = 8000 := by
  sorry

#check max_garden_area

end NUMINAMATH_CALUDE_max_garden_area_l2220_222039


namespace NUMINAMATH_CALUDE_ellipse_and_hyperbola_equations_l2220_222020

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of a line (asymptote) -/
structure Line where
  m : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = m * x

def foci : (Point × Point) := ⟨⟨-5, 0⟩, ⟨5, 0⟩⟩
def intersectionPoint : Point := ⟨4, 3⟩

/-- Theorem stating the equations of the ellipse and hyperbola -/
theorem ellipse_and_hyperbola_equations 
  (e : Ellipse) 
  (h : Hyperbola) 
  (l : Line) 
  (hfoci : e.a^2 - e.b^2 = h.a^2 + h.b^2 ∧ e.a^2 - e.b^2 = 25) 
  (hpoint_on_ellipse : e.equation intersectionPoint.x intersectionPoint.y) 
  (hpoint_on_line : l.equation intersectionPoint.x intersectionPoint.y) 
  (hline_is_asymptote : l.m = h.b / h.a) :
  e.a^2 = 40 ∧ e.b^2 = 15 ∧ h.a^2 = 16 ∧ h.b^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_hyperbola_equations_l2220_222020


namespace NUMINAMATH_CALUDE_expand_binomials_l2220_222097

theorem expand_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2220_222097


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2220_222055

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + 3*x^2 + 3*x + 7 = 0 ∧ x = -1 - Real.rpow 6 (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2220_222055


namespace NUMINAMATH_CALUDE_total_lives_calculation_l2220_222013

theorem total_lives_calculation (initial_players additional_players lives_per_player : ℕ) :
  initial_players = 4 →
  additional_players = 5 →
  lives_per_player = 3 →
  (initial_players + additional_players) * lives_per_player = 27 :=
by sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l2220_222013


namespace NUMINAMATH_CALUDE_tangent_equality_implies_angle_l2220_222072

theorem tangent_equality_implies_angle (x : Real) : 
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_tangent_equality_implies_angle_l2220_222072


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_solution_l2220_222090

theorem quadratic_and_inequality_solution :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    (x₁^2 - 2*x₁ - 4 = 0) ∧ (x₂^2 - 2*x₂ - 4 = 0)) ∧
  (∀ x : ℝ, (2*(x-1) ≥ -4 ∧ (3*x-6)/2 < x-1) ↔ (-1 ≤ x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_solution_l2220_222090


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2220_222006

theorem regular_polygon_sides (b : ℕ) (h : b ≥ 3) : (180 * (b - 2) = 1080) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2220_222006


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2220_222062

/-- Given a geometric sequence of positive integers where the first term is 5 and the third term is 120, 
    prove that the fifth term is 2880. -/
theorem geometric_sequence_fifth_term : 
  ∀ (a : ℕ → ℕ), 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 3 = 120 →                          -- Third term is 120
  a 5 = 2880 :=                        -- Fifth term is 2880
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2220_222062


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2220_222079

theorem largest_constant_inequality (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  ∃ m : ℝ, m = 2 ∧ 
  (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + 
    Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + 
    Real.sqrt (d / (a + b + c + e)) > m) ∧
  (∀ m' : ℝ, m' > m → 
    ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
      Real.sqrt (a / (b + c + d + e)) + 
      Real.sqrt (b / (a + c + d + e)) + 
      Real.sqrt (c / (a + b + d + e)) + 
      Real.sqrt (d / (a + b + c + e)) ≤ m') :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2220_222079


namespace NUMINAMATH_CALUDE_kelly_string_cheese_problem_l2220_222064

/-- The number of string cheeses Kelly's youngest child eats per day -/
def youngest_daily_cheese : ℕ := by sorry

theorem kelly_string_cheese_problem :
  let days_per_week : ℕ := 5
  let oldest_daily_cheese : ℕ := 2
  let cheeses_per_pack : ℕ := 30
  let weeks : ℕ := 4
  let packs_needed : ℕ := 2

  youngest_daily_cheese = 1 := by sorry

end NUMINAMATH_CALUDE_kelly_string_cheese_problem_l2220_222064


namespace NUMINAMATH_CALUDE_rebuild_points_l2220_222086

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define symmetry with respect to a point
def symmetric (p1 p2 center : Point) : Prop :=
  center.x = (p1.x + p2.x) / 2 ∧ center.y = (p1.y + p2.y) / 2

theorem rebuild_points (A' B' C' D' : Point) :
  ∃! (A B C D : Point),
    symmetric A A' B ∧
    symmetric B B' C ∧
    symmetric C C' D ∧
    symmetric D D' A :=
  sorry

end NUMINAMATH_CALUDE_rebuild_points_l2220_222086


namespace NUMINAMATH_CALUDE_mice_breeding_experiment_l2220_222028

/-- Calculates the number of mice after two generations of breeding and some pups being eaten --/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten_per_adult : ℕ) : ℕ :=
  let first_gen_total := initial_mice + initial_mice * pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pups_per_mouse
  second_gen_total - (first_gen_total * pups_eaten_per_adult)

/-- Theorem stating that under the given conditions, the final number of mice is 280 --/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mice_breeding_experiment_l2220_222028


namespace NUMINAMATH_CALUDE_polynomial_independent_implies_m_plus_n_squared_l2220_222047

/-- A polynomial that is independent of x -/
def polynomial (m n x y : ℝ) : ℝ := 4*m*x^2 + 5*x - 2*y^2 + 8*x^2 - n*x + y - 1

/-- The polynomial is independent of x -/
def independent_of_x (m n : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ c : ℝ, ∀ x' : ℝ, polynomial m n x' y = c

/-- The main theorem -/
theorem polynomial_independent_implies_m_plus_n_squared (m n : ℝ) :
  independent_of_x m n → (m + n)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independent_implies_m_plus_n_squared_l2220_222047


namespace NUMINAMATH_CALUDE_unique_prime_multiple_of_11_l2220_222036

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

theorem unique_prime_multiple_of_11 :
  ∃! n : ℕ, n ≥ 1 ∧ n ≤ 100 ∧ is_prime n ∧ is_multiple_of_11 n :=
sorry

end NUMINAMATH_CALUDE_unique_prime_multiple_of_11_l2220_222036


namespace NUMINAMATH_CALUDE_symmetry_about_xoz_plane_l2220_222045

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry operation about the xOz plane
def symmetryAboutXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem symmetry_about_xoz_plane :
  let A : Point3D := { x := 3, y := -2, z := 5 }
  let A_sym : Point3D := symmetryAboutXOZ A
  A_sym = { x := 3, y := 2, z := 5 } := by sorry

end NUMINAMATH_CALUDE_symmetry_about_xoz_plane_l2220_222045


namespace NUMINAMATH_CALUDE_mens_wages_l2220_222070

theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 →
  5 * women = men * 8 →
  total_earnings = 60 →
  total_earnings / (3 * men) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mens_wages_l2220_222070


namespace NUMINAMATH_CALUDE_quartet_characterization_l2220_222056

def is_valid_quartet (a b c d : ℕ+) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c * d ∧ a * b = c + d

def valid_quartets : List (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  [(1, 5, 3, 2), (1, 5, 2, 3), (5, 1, 3, 2), (5, 1, 2, 3),
   (2, 3, 1, 5), (3, 2, 1, 5), (2, 3, 5, 1), (3, 2, 5, 1)]

theorem quartet_characterization (a b c d : ℕ+) :
  is_valid_quartet a b c d ↔ (a, b, c, d) ∈ valid_quartets :=
sorry

end NUMINAMATH_CALUDE_quartet_characterization_l2220_222056


namespace NUMINAMATH_CALUDE_fraction_reducibility_fraction_reducibility_2_l2220_222024

theorem fraction_reducibility (n : ℤ) :
  (∃ k : ℤ, n = 3 * k - 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^2 + 2*n + 4) = a * (n^2 + n + 3) :=
sorry

theorem fraction_reducibility_2 (n : ℤ) :
  (∃ k : ℤ, n = 3 * k ∨ n = 3 * k + 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^3 - n^2 - 3*n) = a * (n^2 - n + 3) :=
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_fraction_reducibility_2_l2220_222024


namespace NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2220_222005

theorem decimal_digits_of_fraction : ∃ (n : ℚ), 
  n = (5^7 : ℚ) / ((10^5 : ℚ) * 125) ∧ 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (m : ℕ), n = (m : ℚ) / (10^d : ℚ) ∧ 
   m % 10 ≠ 0 ∧ 
   (∀ (k : ℕ), k > d → (m * 10^(k-d)) % 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2220_222005


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_constraint_l2220_222058

theorem max_sum_with_lcm_gcd_constraint (m n : ℕ) : 
  m + 3*n - 5 = 2*(Nat.lcm m n) - 11*(Nat.gcd m n) → 
  m + n ≤ 70 ∧ ∃ (m₀ n₀ : ℕ), m₀ + 3*n₀ - 5 = 2*(Nat.lcm m₀ n₀) - 11*(Nat.gcd m₀ n₀) ∧ m₀ + n₀ = 70 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_constraint_l2220_222058


namespace NUMINAMATH_CALUDE_prime_relation_l2220_222001

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_relation (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h : 13 * p + 1 = q + 2) : 
  q = 39 := by sorry

end NUMINAMATH_CALUDE_prime_relation_l2220_222001


namespace NUMINAMATH_CALUDE_kimberly_skittles_l2220_222080

def skittles_problem (initial_skittles : ℚ) 
                     (eaten_skittles : ℚ) 
                     (given_skittles : ℚ) 
                     (promotion_skittles : ℚ) 
                     (exchange_skittles : ℚ) : Prop :=
  let remaining_after_eating := initial_skittles - eaten_skittles
  let remaining_after_giving := remaining_after_eating - given_skittles
  let after_promotion := remaining_after_giving + promotion_skittles
  let final_skittles := after_promotion + exchange_skittles
  final_skittles = 18

theorem kimberly_skittles : 
  skittles_problem 7.5 2.25 1.5 3.75 10.5 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l2220_222080


namespace NUMINAMATH_CALUDE_optimal_prevention_plan_l2220_222081

/-- Represents a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given a set of preventive measures -/
def totalCost (baseProbability : ℝ) (baseLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let preventionCost := measures.foldl (fun acc m => acc + m.cost) 0
  let incidentProbability := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) baseProbability
  preventionCost + incidentProbability * baseLoss

theorem optimal_prevention_plan 
  (baseProbability : ℝ)
  (baseLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h1 : baseProbability = 0.3)
  (h2 : baseLoss = 400)
  (h3 : measureA.cost = 45)
  (h4 : measureA.effectiveness = 0.9)
  (h5 : measureB.cost = 30)
  (h6 : measureB.effectiveness = 0.85) :
  totalCost baseProbability baseLoss [measureA, measureB] < 
  min 
    (totalCost baseProbability baseLoss [])
    (min 
      (totalCost baseProbability baseLoss [measureA])
      (totalCost baseProbability baseLoss [measureB])) := by
  sorry

#check optimal_prevention_plan

end NUMINAMATH_CALUDE_optimal_prevention_plan_l2220_222081


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2220_222008

theorem zoo_animal_ratio (parrots : ℕ) (snakes : ℕ) (elephants : ℕ) (zebras : ℕ) (monkeys : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  elephants = (parrots + snakes) / 2 →
  zebras = elephants - 3 →
  monkeys - zebras = 35 →
  monkeys / snakes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2220_222008


namespace NUMINAMATH_CALUDE_total_project_hours_l2220_222011

/-- Represents the time spent on an activity --/
structure ActivityTime where
  hoursPerDay : ℕ
  days : ℕ

/-- Calculates the total hours for an activity --/
def totalHours (a : ActivityTime) : ℕ := a.hoursPerDay * a.days

/-- Represents the time spent on various activities for a song --/
structure SongTime where
  vocals : ActivityTime
  instrument : ActivityTime
  mixing : ActivityTime

/-- Represents the overall project time --/
structure ProjectTime where
  song1 : SongTime
  song2 : SongTime
  song3 : SongTime
  videoProduction : ActivityTime
  marketing : ActivityTime

/-- The given project time data --/
def givenProjectTime : ProjectTime :=
  { song1 := { vocals := { hoursPerDay := 8, days := 12 },
               instrument := { hoursPerDay := 2, days := 6 },
               mixing := { hoursPerDay := 4, days := 3 } },
    song2 := { vocals := { hoursPerDay := 10, days := 9 },
               instrument := { hoursPerDay := 3, days := 4 },
               mixing := { hoursPerDay := 5, days := 2 } },
    song3 := { vocals := { hoursPerDay := 6, days := 15 },
               instrument := { hoursPerDay := 1, days := 5 },
               mixing := { hoursPerDay := 3, days := 4 } },
    videoProduction := { hoursPerDay := 5, days := 7 },
    marketing := { hoursPerDay := 4, days := 10 } }

/-- Calculates the total hours spent on the project --/
def calculateTotalHours (p : ProjectTime) : ℕ :=
  totalHours p.song1.vocals + totalHours p.song1.instrument + totalHours p.song1.mixing +
  totalHours p.song2.vocals + totalHours p.song2.instrument + totalHours p.song2.mixing +
  totalHours p.song3.vocals + totalHours p.song3.instrument + totalHours p.song3.mixing +
  totalHours p.videoProduction + totalHours p.marketing

/-- Theorem: The total hours spent on the project is 414 --/
theorem total_project_hours : calculateTotalHours givenProjectTime = 414 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l2220_222011


namespace NUMINAMATH_CALUDE_october_order_theorem_l2220_222075

/-- Represents the order quantities for a specific month -/
structure MonthOrder where
  clawHammers : ℕ
  ballPeenHammers : ℕ
  sledgehammers : ℕ

/-- Calculates the next month's order based on the pattern -/
def nextMonthOrder (current : MonthOrder) : MonthOrder := sorry

/-- Calculates the total number of hammers in an order -/
def totalHammers (order : MonthOrder) : ℕ :=
  order.clawHammers + order.ballPeenHammers + order.sledgehammers

/-- Applies the seasonal increase to the total order -/
def applySeasonalIncrease (total : ℕ) (increase : Rat) : ℕ := sorry

/-- The order data for June, July, August, and September -/
def juneOrder : MonthOrder := ⟨3, 2, 1⟩
def julyOrder : MonthOrder := ⟨4, 3, 2⟩
def augustOrder : MonthOrder := ⟨6, 7, 3⟩
def septemberOrder : MonthOrder := ⟨9, 11, 4⟩

/-- The seasonal increase percentage -/
def seasonalIncrease : Rat := 7 / 100

theorem october_order_theorem :
  let octoberOrder := nextMonthOrder septemberOrder
  let totalBeforeIncrease := totalHammers octoberOrder
  let finalTotal := applySeasonalIncrease totalBeforeIncrease seasonalIncrease
  finalTotal = 32 := by sorry

end NUMINAMATH_CALUDE_october_order_theorem_l2220_222075


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l2220_222061

theorem no_solutions_for_equation : ¬∃ (x y : ℕ), 2^(2*x) - 3^(2*y) = 58 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l2220_222061


namespace NUMINAMATH_CALUDE_sphere_volume_constant_l2220_222035

theorem sphere_volume_constant (K : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_volume : ℝ := (K * Real.sqrt 18) / Real.sqrt Real.pi
  sphere_surface_area = 4 * Real.pi * ((3 * Real.sqrt 3) / Real.sqrt Real.pi)^2 ∧
  sphere_volume = (4 / 3) * Real.pi * ((3 * Real.sqrt 3) / Real.sqrt Real.pi)^3 →
  K = 54 := by
sorry


end NUMINAMATH_CALUDE_sphere_volume_constant_l2220_222035


namespace NUMINAMATH_CALUDE_six_people_three_events_outcomes_l2220_222048

/-- The number of possible outcomes for champions in a competition. -/
def championOutcomes (people : ℕ) (events : ℕ) : ℕ :=
  people ^ events

/-- Theorem stating the number of possible outcomes for 6 people in 3 events. -/
theorem six_people_three_events_outcomes :
  championOutcomes 6 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_six_people_three_events_outcomes_l2220_222048


namespace NUMINAMATH_CALUDE_paths_amc9_count_l2220_222067

/-- Represents the number of paths to spell "AMC9" in the grid -/
def pathsAMC9 (m_from_a : Nat) (c_from_m : Nat) (nine_from_c : Nat) : Nat :=
  m_from_a * c_from_m * nine_from_c

/-- Theorem stating that the number of paths to spell "AMC9" is 36 -/
theorem paths_amc9_count :
  pathsAMC9 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_paths_amc9_count_l2220_222067


namespace NUMINAMATH_CALUDE_equidistant_point_l2220_222087

theorem equidistant_point (x y : ℝ) : 
  let d_y_axis := |x|
  let d_line1 := |x + y - 1| / Real.sqrt 2
  let d_line2 := |y - 3*x| / Real.sqrt 10
  (d_y_axis = d_line1 ∧ d_y_axis = d_line2) → 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_l2220_222087


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2220_222060

/-- Theorem: The polynomial division of 8x^4 + 7x^3 - 2x^2 - 9x + 5 by x - 1
    yields a quotient of 8x^3 + 15x^2 + 13x + 4 with a remainder of 9. -/
theorem polynomial_division_theorem (x : ℝ) :
  (8*x^3 + 15*x^2 + 13*x + 4) * (x - 1) + 9 = 8*x^4 + 7*x^3 - 2*x^2 - 9*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2220_222060


namespace NUMINAMATH_CALUDE_grocery_solution_l2220_222077

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (pasta_price : ℝ) (pasta_amount : ℝ) 
  (sauce_price : ℝ) (sauce_amount : ℝ) (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (mustard_oil_amount : ℝ),
    mustard_oil_amount ≥ 0 ∧
    mustard_oil_price > 0 ∧
    pasta_price > 0 ∧
    pasta_amount > 0 ∧
    sauce_price > 0 ∧
    sauce_amount > 0 ∧
    initial_money > 0 ∧
    remaining_money ≥ 0 ∧
    initial_money - remaining_money = 
      mustard_oil_amount * mustard_oil_price + pasta_amount * pasta_price + sauce_amount * sauce_price ∧
    mustard_oil_amount = 2

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 4 3 5 1 50 7 :=
by sorry

end NUMINAMATH_CALUDE_grocery_solution_l2220_222077


namespace NUMINAMATH_CALUDE_range_of_a_l2220_222083

def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def p (a : ℝ) : Prop := A a ⊆ B
def q (a : ℝ) : Prop := ∀ x > (1/2 : ℝ), Monotone (f a)

theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((1/2 < a ∧ a ≤ 2) ∨ a > 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2220_222083


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2220_222084

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : 
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2220_222084


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2220_222025

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2220_222025


namespace NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l2220_222066

/-- Given a sphere with initial surface area 400π cm² and radius increased by 2 cm, 
    prove that the new volume is 2304π cm³ -/
theorem sphere_volume_after_radius_increase :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 400 * π) →  -- Initial surface area condition
    ((4 / 3) * π * (r + 2)^3 = 2304 * π) -- New volume after radius increase
:= by sorry

end NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l2220_222066


namespace NUMINAMATH_CALUDE_f_divisibility_by_3_smallest_n_for_2017_l2220_222015

def f : ℕ → ℤ
  | 0 => 0  -- base case
  | n + 1 => if n.succ % 2 = 0 then -f (n.succ / 2) else f n + 1

theorem f_divisibility_by_3 (n : ℕ) : 3 ∣ f n ↔ 3 ∣ n := by sorry

def geometric_sum (n : ℕ) : ℕ := (4^(n+1) - 1) / 3

theorem smallest_n_for_2017 : 
  f (geometric_sum 1008) = 2017 ∧ 
  ∀ m : ℕ, m < geometric_sum 1008 → f m ≠ 2017 := by sorry

end NUMINAMATH_CALUDE_f_divisibility_by_3_smallest_n_for_2017_l2220_222015


namespace NUMINAMATH_CALUDE_problem_solution_l2220_222031

theorem problem_solution (x y : ℝ) (h : 2 * x = Real.log (x + y - 1) + Real.log (x - y - 1) + 4) :
  2015 * x^2 + 2016 * y^3 = 8060 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2220_222031


namespace NUMINAMATH_CALUDE_students_in_front_of_yuna_l2220_222030

theorem students_in_front_of_yuna (total_students : ℕ) (students_behind_yuna : ℕ) : 
  total_students = 25 → students_behind_yuna = 9 → total_students - (students_behind_yuna + 1) = 15 := by
  sorry


end NUMINAMATH_CALUDE_students_in_front_of_yuna_l2220_222030


namespace NUMINAMATH_CALUDE_product_and_sum_of_roots_l2220_222029

theorem product_and_sum_of_roots : 
  (16 : ℝ) ^ (1/4 : ℝ) * (32 : ℝ) ^ (1/5 : ℝ) + (64 : ℝ) ^ (1/6 : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_roots_l2220_222029


namespace NUMINAMATH_CALUDE_prime_sum_squares_l2220_222069

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 → q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l2220_222069


namespace NUMINAMATH_CALUDE_farmer_apples_l2220_222057

/-- The number of apples remaining after giving some away -/
def applesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ := initial - givenAway

/-- Theorem: A farmer with 127 apples who gives away 88 apples has 39 apples remaining -/
theorem farmer_apples : applesRemaining 127 88 = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l2220_222057


namespace NUMINAMATH_CALUDE_magazine_clients_count_l2220_222099

/-- The number of clients using magazines in an advertising agency --/
def clients_using_magazines (total : ℕ) (tv : ℕ) (radio : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) : ℕ :=
  total + all_three - (tv + radio - tv_radio)

/-- Theorem stating the number of clients using magazines --/
theorem magazine_clients_count : 
  clients_using_magazines 180 115 110 85 75 95 80 = 130 := by
  sorry

end NUMINAMATH_CALUDE_magazine_clients_count_l2220_222099


namespace NUMINAMATH_CALUDE_composite_function_equation_solution_l2220_222016

theorem composite_function_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 8
  ∃! x : ℝ, (δ ∘ φ) x = 10 ∧ x = -31 / 36 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_function_equation_solution_l2220_222016


namespace NUMINAMATH_CALUDE_factory_x_bulb_longevity_l2220_222002

theorem factory_x_bulb_longevity 
  (supply_x : ℝ) 
  (supply_y : ℝ) 
  (longevity_y : ℝ) 
  (overall_longevity : ℝ) 
  (h1 : supply_x = 0.60)
  (h2 : supply_y = 1 - supply_x)
  (h3 : longevity_y = 0.65)
  (h4 : overall_longevity = 0.62) :
  supply_x * (supply_x * overall_longevity - supply_y * longevity_y) / supply_x = 0.60 :=
by sorry

end NUMINAMATH_CALUDE_factory_x_bulb_longevity_l2220_222002
