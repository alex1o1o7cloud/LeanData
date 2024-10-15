import Mathlib

namespace NUMINAMATH_CALUDE_exam_day_percentage_l3850_385000

/-- Represents the percentage of students who took the exam on the assigned day -/
def assigned_day_percentage : ℝ := 70

/-- Represents the total number of students in the class -/
def total_students : ℕ := 100

/-- Represents the average score of students who took the exam on the assigned day -/
def assigned_day_score : ℝ := 60

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_day_score : ℝ := 80

/-- Represents the average score for the entire class -/
def class_average_score : ℝ := 66

theorem exam_day_percentage :
  assigned_day_percentage * assigned_day_score / 100 +
  (100 - assigned_day_percentage) * makeup_day_score / 100 =
  class_average_score :=
sorry

end NUMINAMATH_CALUDE_exam_day_percentage_l3850_385000


namespace NUMINAMATH_CALUDE_fruit_display_total_l3850_385024

/-- Proves that the total number of fruits on a display is 35, given specific conditions --/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l3850_385024


namespace NUMINAMATH_CALUDE_seashells_left_l3850_385091

def initial_seashells : ℕ := 62
def seashells_given : ℕ := 49

theorem seashells_left : initial_seashells - seashells_given = 13 := by
  sorry

end NUMINAMATH_CALUDE_seashells_left_l3850_385091


namespace NUMINAMATH_CALUDE_series_sum_l3850_385002

/-- The sum of the infinite series Σ(n=1 to ∞) of n/(3^n) equals 9/4 -/
theorem series_sum : ∑' n : ℕ, (n : ℝ) / (3 : ℝ) ^ n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3850_385002


namespace NUMINAMATH_CALUDE_both_correct_undetermined_l3850_385081

/-- Represents a class of students and their test performance -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Predicate to check if the number of students who answered both questions correctly is determinable -/
def both_correct_determinable (c : ClassTestResults) : Prop :=
  ∃ (n : ℕ), n ≤ c.correct_q1 ∧ n ≤ c.correct_q2 ∧ n = c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that the number of students who answered both questions correctly is undetermined -/
theorem both_correct_undetermined (c : ClassTestResults)
  (h1 : c.total_students = 25)
  (h2 : c.correct_q1 = 22)
  (h3 : c.absent = 3)
  (h4 : c.correct_q2 ≤ c.total_students - c.absent)
  (h5 : c.correct_q2 > 0) :
  ¬ both_correct_determinable c := by
  sorry


end NUMINAMATH_CALUDE_both_correct_undetermined_l3850_385081


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3850_385043

theorem expand_and_simplify (x : ℝ) : (x - 1) * (x + 3) - x * (x - 2) = 4 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3850_385043


namespace NUMINAMATH_CALUDE_total_prime_factors_is_27_l3850_385064

/-- The total number of prime factors in the expression (4)^11 * (7)^3 * (11)^2 -/
def totalPrimeFactors : ℕ :=
  let four_factorization := 2 * 2
  let four_exponent := 11
  let seven_exponent := 3
  let eleven_exponent := 2
  (four_factorization * four_exponent) + seven_exponent + eleven_exponent

/-- Theorem stating that the total number of prime factors in the given expression is 27 -/
theorem total_prime_factors_is_27 : totalPrimeFactors = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_prime_factors_is_27_l3850_385064


namespace NUMINAMATH_CALUDE_max_value_of_a_l3850_385095

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ b) → b ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3850_385095


namespace NUMINAMATH_CALUDE_ball_count_proof_l3850_385083

/-- Proves that given 9 yellow balls in a box and a 30% probability of drawing a yellow ball,
    the total number of balls in the box is 30. -/
theorem ball_count_proof (yellow_balls : ℕ) (probability : ℚ) (total_balls : ℕ) : 
  yellow_balls = 9 → probability = 3/10 → (yellow_balls : ℚ) / total_balls = probability → total_balls = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l3850_385083


namespace NUMINAMATH_CALUDE_test_scores_l3850_385006

/-- Represents the score of a test taker -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  total_questions : Nat
  h_sum : correct + unanswered + incorrect = total_questions

/-- Calculates the score based on the test results -/
def calculate_score (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Checks if a score is possible given the test parameters -/
def is_possible_score (score : Nat) : Prop :=
  ∃ (ts : TestScore), ts.total_questions = 30 ∧ calculate_score ts = score

theorem test_scores :
  is_possible_score 116 ∧
  ¬is_possible_score 117 ∧
  is_possible_score 118 ∧
  ¬is_possible_score 119 ∧
  is_possible_score 120 :=
sorry

end NUMINAMATH_CALUDE_test_scores_l3850_385006


namespace NUMINAMATH_CALUDE_product_of_complex_numbers_l3850_385057

/-- Represents a complex number in polar form -/
structure PolarComplex where
  r : ℝ
  θ : ℝ
  h_r_pos : r > 0
  h_θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi

/-- Multiplies two complex numbers in polar form -/
def polar_multiply (z₁ z₂ : PolarComplex) : PolarComplex :=
  { r := z₁.r * z₂.r,
    θ := z₁.θ + z₂.θ,
    h_r_pos := by sorry,
    h_θ_range := by sorry }

theorem product_of_complex_numbers :
  let z₁ : PolarComplex := ⟨5, 30 * Real.pi / 180, by sorry, by sorry⟩
  let z₂ : PolarComplex := ⟨4, 140 * Real.pi / 180, by sorry, by sorry⟩
  let result := polar_multiply z₁ z₂
  result.r = 20 ∧ result.θ = 170 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_product_of_complex_numbers_l3850_385057


namespace NUMINAMATH_CALUDE_trishas_walk_distance_l3850_385046

theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_distance_l3850_385046


namespace NUMINAMATH_CALUDE_like_terms_mn_value_l3850_385051

theorem like_terms_mn_value (n m : ℕ) :
  (∃ (a b : ℝ) (x y : ℝ), a * x^n * y^3 = b * x^3 * y^m) →
  m^n = 27 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_mn_value_l3850_385051


namespace NUMINAMATH_CALUDE_johns_max_correct_answers_l3850_385096

/-- Represents an exam with a given number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if an exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.total_score = result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score

/-- Theorem: The maximum number of correctly answered questions for John's exam is 12. -/
theorem johns_max_correct_answers (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 20 ∧
  john_exam.correct_score = 5 ∧
  john_exam.incorrect_score = -2 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 48 ∧
  is_valid_result john_result →
  ∀ (other_result : ExamResult),
    is_valid_result other_result ∧
    other_result.exam = john_exam ∧
    other_result.total_score = 48 →
    other_result.correct ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_johns_max_correct_answers_l3850_385096


namespace NUMINAMATH_CALUDE_square_perimeter_is_96_l3850_385029

/-- A square ABCD with side lengths expressed in terms of x -/
structure Square (x : ℝ) where
  AB : ℝ := x + 16
  BC : ℝ := 3 * x
  is_square : AB = BC

/-- The perimeter of the square ABCD is 96 -/
theorem square_perimeter_is_96 (x : ℝ) (ABCD : Square x) : 
  4 * ABCD.AB = 96 := by
  sorry

#check square_perimeter_is_96

end NUMINAMATH_CALUDE_square_perimeter_is_96_l3850_385029


namespace NUMINAMATH_CALUDE_solve_for_a_l3850_385025

/-- Given that x + 2a - 6 = 0 and x = -2, prove that a = 4 -/
theorem solve_for_a (x a : ℝ) (h1 : x + 2*a - 6 = 0) (h2 : x = -2) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3850_385025


namespace NUMINAMATH_CALUDE_relation_implications_l3850_385003

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary (P Q : Prop) : Prop :=
  Q → P

-- State the theorem
theorem relation_implications :
  sufficient_not_necessary A B →
  necessary B C →
  sufficient_not_necessary C D →
  (sufficient_not_necessary A C ∧ 
   ¬(sufficient_not_necessary D A ∨ necessary D A)) :=
by sorry

end NUMINAMATH_CALUDE_relation_implications_l3850_385003


namespace NUMINAMATH_CALUDE_alice_notebook_savings_l3850_385089

/-- The amount Alice saves when buying notebooks during a sale -/
theorem alice_notebook_savings (number_of_notebooks : ℕ) (original_price : ℚ) (discount_rate : ℚ) :
  number_of_notebooks = 8 →
  original_price = 375/100 →
  discount_rate = 25/100 →
  (number_of_notebooks * original_price) - (number_of_notebooks * (original_price * (1 - discount_rate))) = 75/10 := by
  sorry

end NUMINAMATH_CALUDE_alice_notebook_savings_l3850_385089


namespace NUMINAMATH_CALUDE_decompose_power_l3850_385023

theorem decompose_power (a : ℝ) (h : a > 0) : 
  ∃ (x y z w : ℝ), 
    a^(3/4) = a^x * a^y * a^z * a^w ∧ 
    y = x + 1/6 ∧ 
    z = y + 1/6 ∧ 
    w = z + 1/6 ∧
    x = -1/16 ∧ 
    y = 5/48 ∧ 
    z = 13/48 ∧ 
    w = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_decompose_power_l3850_385023


namespace NUMINAMATH_CALUDE_triangle_problem_l3850_385061

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b^2 = a^2 + c^2 - Real.sqrt 3 * a * c →
  B = π/6 ∧
  Real.sqrt 3 / 2 < Real.cos A + Real.sin C ∧ 
  Real.cos A + Real.sin C < 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3850_385061


namespace NUMINAMATH_CALUDE_factorial_inequality_l3850_385013

theorem factorial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n ≤ m) :
  (2^n : ℝ) * (n.factorial : ℝ) ≤ ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ∧
  ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ≤ ((m^2 + m : ℝ)^n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l3850_385013


namespace NUMINAMATH_CALUDE_factorization_equality_l3850_385005

theorem factorization_equality (x : ℝ) :
  (x^2 + 3*x - 3) * (x^2 + 3*x + 1) - 5 = (x + 1) * (x + 2) * (x + 4) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3850_385005


namespace NUMINAMATH_CALUDE_squares_below_line_l3850_385063

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement --/
theorem squares_below_line :
  countPointsBelowLine problemLine = 984 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l3850_385063


namespace NUMINAMATH_CALUDE_other_number_is_64_l3850_385018

/-- Given two positive integers with specific LCM and HCF, prove that one is 64 -/
theorem other_number_is_64 (A B : ℕ+) (h1 : A = 48) 
  (h2 : Nat.lcm A B = 192) (h3 : Nat.gcd A B = 16) : B = 64 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_64_l3850_385018


namespace NUMINAMATH_CALUDE_multiple_problem_l3850_385097

theorem multiple_problem (n m : ℝ) : n = 5 → n + m * n = 20 → m = 3 := by sorry

end NUMINAMATH_CALUDE_multiple_problem_l3850_385097


namespace NUMINAMATH_CALUDE_profit_percentage_60_percent_l3850_385054

/-- Profit percentage for 60% of apples given total apples, profit percentages, and sales distribution --/
theorem profit_percentage_60_percent (total_apples : ℝ) (profit_40_percent : ℝ) (total_profit_percent : ℝ) :
  total_apples = 280 →
  profit_40_percent = 10 →
  total_profit_percent = 22.000000000000007 →
  let profit_60_percent := 
    (total_profit_percent * total_apples - profit_40_percent * (0.4 * total_apples)) / (0.6 * total_apples) * 100
  profit_60_percent = 30 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_60_percent_l3850_385054


namespace NUMINAMATH_CALUDE_total_legs_equals_1564_l3850_385033

/-- Calculates the total number of legs of all animals owned by Mark -/
def totalLegs (numKangaroos : ℕ) : ℕ :=
  let numGoats := 3 * numKangaroos
  let numSpiders := 2 * numGoats
  let numBirds := numSpiders / 2
  let kangarooLegs := 2 * numKangaroos
  let goatLegs := 4 * numGoats
  let spiderLegs := 8 * numSpiders
  let birdLegs := 2 * numBirds
  kangarooLegs + goatLegs + spiderLegs + birdLegs

/-- Theorem stating that the total number of legs of all Mark's animals is 1564 -/
theorem total_legs_equals_1564 : totalLegs 23 = 1564 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_equals_1564_l3850_385033


namespace NUMINAMATH_CALUDE_normal_to_curve_l3850_385087

-- Define the curve
def curve (x y a : ℝ) : Prop := x^(2/3) + y^(2/3) = a^(2/3)

-- Define the normal equation
def normal_equation (x y a θ : ℝ) : Prop := y * Real.cos θ - x * Real.sin θ = a * Real.cos (2 * θ)

-- Theorem statement
theorem normal_to_curve (x y a θ : ℝ) :
  curve x y a →
  (∃ (p q : ℝ), curve p q a ∧ 
    -- The point (p, q) is on the curve and the normal at this point makes an angle θ with the X-axis
    (y - q) * Real.cos θ = (x - p) * Real.sin θ) →
  normal_equation x y a θ :=
by sorry

end NUMINAMATH_CALUDE_normal_to_curve_l3850_385087


namespace NUMINAMATH_CALUDE_dividend_calculation_l3850_385012

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3850_385012


namespace NUMINAMATH_CALUDE_min_lines_for_31_segments_l3850_385085

/-- A broken line represented by its number of segments -/
structure BrokenLine where
  segments : ℕ
  no_self_intersections : Bool
  distinct_endpoints : Bool

/-- The minimum number of straight lines formed by extending all segments of a broken line -/
def min_straight_lines (bl : BrokenLine) : ℕ :=
  (bl.segments + 1) / 2

/-- Theorem stating the minimum number of straight lines for a specific broken line -/
theorem min_lines_for_31_segments :
  ∀ (bl : BrokenLine),
    bl.segments = 31 →
    bl.no_self_intersections = true →
    bl.distinct_endpoints = true →
    min_straight_lines bl = 16 := by
  sorry

#eval min_straight_lines { segments := 31, no_self_intersections := true, distinct_endpoints := true }

end NUMINAMATH_CALUDE_min_lines_for_31_segments_l3850_385085


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3850_385079

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : T > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/6) * T) :
  (T - G) / G = 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3850_385079


namespace NUMINAMATH_CALUDE_grumpy_not_orange_l3850_385058

structure Lizard where
  orange : Prop
  grumpy : Prop
  can_swim : Prop
  can_jump : Prop

def Cathys_lizards : Set Lizard := sorry

theorem grumpy_not_orange :
  ∀ (total : ℕ) (orange_count : ℕ) (grumpy_count : ℕ),
  total = 15 →
  orange_count = 6 →
  grumpy_count = 7 →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.grumpy → l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.orange → ¬l.can_jump) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬l.can_jump → ¬l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬(l.grumpy ∧ l.orange)) :=
by sorry

end NUMINAMATH_CALUDE_grumpy_not_orange_l3850_385058


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l3850_385078

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l3850_385078


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3850_385076

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3850_385076


namespace NUMINAMATH_CALUDE_triangle_area_l3850_385065

theorem triangle_area (b c : ℝ) (angle_C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : angle_C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3850_385065


namespace NUMINAMATH_CALUDE_absolute_value_fraction_sum_not_one_l3850_385015

theorem absolute_value_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_sum_not_one_l3850_385015


namespace NUMINAMATH_CALUDE_equation_solution_set_l3850_385040

theorem equation_solution_set : 
  {x : ℝ | |x^2 - 5*x + 6| = x + 2} = {3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3850_385040


namespace NUMINAMATH_CALUDE_f_difference_bound_l3850_385010

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_difference_bound (a x : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * |a| + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_bound_l3850_385010


namespace NUMINAMATH_CALUDE_cody_initial_tickets_cody_initial_tickets_proof_l3850_385036

/-- Theorem: Cody's initial number of tickets
Given:
- Cody lost 6.0 tickets
- Cody spent 25.0 tickets
- Cody has 18 tickets left
Prove: Cody's initial number of tickets was 49.0
-/
theorem cody_initial_tickets : ℝ → Prop :=
  fun initial_tickets =>
    let lost_tickets : ℝ := 6.0
    let spent_tickets : ℝ := 25.0
    let remaining_tickets : ℝ := 18.0
    initial_tickets = lost_tickets + spent_tickets + remaining_tickets
    ∧ initial_tickets = 49.0

/-- Proof of the theorem -/
theorem cody_initial_tickets_proof : cody_initial_tickets 49.0 := by
  sorry

end NUMINAMATH_CALUDE_cody_initial_tickets_cody_initial_tickets_proof_l3850_385036


namespace NUMINAMATH_CALUDE_A_n_squared_value_l3850_385019

theorem A_n_squared_value (n : ℕ) : (n.choose 2 = 15) → (n * (n - 1) = 30) := by
  sorry

end NUMINAMATH_CALUDE_A_n_squared_value_l3850_385019


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3850_385050

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_roots : a 3 + a 15 = 6 ∧ a 3 * a 15 = 8) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3850_385050


namespace NUMINAMATH_CALUDE_virus_length_scientific_notation_l3850_385011

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

end NUMINAMATH_CALUDE_virus_length_scientific_notation_l3850_385011


namespace NUMINAMATH_CALUDE_monopoly_wins_ratio_l3850_385035

/-- 
Proves that given the conditions of the Monopoly game wins, 
the ratio of Susan's wins to Betsy's wins is 3:1
-/
theorem monopoly_wins_ratio :
  ∀ (betsy helen susan : ℕ),
  betsy = 5 →
  helen = 2 * betsy →
  betsy + helen + susan = 30 →
  susan / betsy = 3 := by
sorry

end NUMINAMATH_CALUDE_monopoly_wins_ratio_l3850_385035


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3850_385069

theorem quadratic_rewrite (x : ℝ) : 
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 18 = (a * x + b)^2 + c ∧ a * b = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3850_385069


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3850_385045

theorem algebraic_simplification (b : ℝ) : 3*b*(3*b^2 + 2*b - 1) - 2*b^2 = 9*b^3 + 4*b^2 - 3*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3850_385045


namespace NUMINAMATH_CALUDE_product_of_nonreal_roots_l3850_385066

theorem product_of_nonreal_roots : ∃ (r₁ r₂ : ℂ),
  (r₁ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  (r₂ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 1 + Real.sqrt 2006 :=
by sorry

end NUMINAMATH_CALUDE_product_of_nonreal_roots_l3850_385066


namespace NUMINAMATH_CALUDE_equation_solution_l3850_385070

theorem equation_solution :
  ∃ m : ℝ, (m - 5) ^ 3 = (1 / 16)⁻¹ ∧ m = 5 + 2 ^ (4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3850_385070


namespace NUMINAMATH_CALUDE_mrs_martin_coffee_cups_l3850_385021

-- Define the cost of a bagel
def bagel_cost : ℝ := 1.5

-- Define Mrs. Martin's purchase
def mrs_martin_total : ℝ := 12.75
def mrs_martin_bagels : ℕ := 2

-- Define Mr. Martin's purchase
def mr_martin_total : ℝ := 14.00
def mr_martin_coffee : ℕ := 2
def mr_martin_bagels : ℕ := 5

-- Theorem to prove
theorem mrs_martin_coffee_cups : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mrs_martin_coffee_cups_l3850_385021


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3850_385074

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3850_385074


namespace NUMINAMATH_CALUDE_bob_eats_one_more_than_george_l3850_385093

/-- Represents the number of slices in different pizza sizes and quantities purchased --/
structure PizzaOrder where
  small_slices : ℕ := 4
  large_slices : ℕ := 8
  small_count : ℕ := 3
  large_count : ℕ := 2

/-- Represents the pizza consumption of different people --/
structure PizzaConsumption where
  george : ℕ := 3
  bill : ℕ := 3
  fred : ℕ := 3
  mark : ℕ := 3
  leftover : ℕ := 10

/-- Theorem stating that Bob eats one more slice than George --/
theorem bob_eats_one_more_than_george (order : PizzaOrder) (consumption : PizzaConsumption) : 
  ∃ (bob : ℕ) (susie : ℕ), 
    susie = bob / 2 ∧ 
    bob = consumption.george + 1 ∧
    order.small_slices * order.small_count + order.large_slices * order.large_count = 
      consumption.george + bob + susie + consumption.bill + consumption.fred + consumption.mark + consumption.leftover :=
by
  sorry

end NUMINAMATH_CALUDE_bob_eats_one_more_than_george_l3850_385093


namespace NUMINAMATH_CALUDE_square_field_area_l3850_385009

theorem square_field_area (wire_length : ℝ) (wire_rounds : ℕ) (field_area : ℝ) : 
  wire_length = 7348 →
  wire_rounds = 11 →
  wire_length = 4 * wire_rounds * Real.sqrt field_area →
  field_area = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3850_385009


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3850_385038

def P (x : ℝ) : ℝ := (2*x^2 - 2*x + 1)^17 * (3*x^2 - 3*x + 1)^17

theorem polynomial_coefficient_sums :
  (∀ x, P x = P 1) ∧
  (∀ x, (P x + P (-x)) / 2 = (1 + 35^17) / 2) ∧
  (∀ x, (P x - P (-x)) / 2 = (1 - 35^17) / 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3850_385038


namespace NUMINAMATH_CALUDE_nickel_count_l3850_385022

def total_cents : ℕ := 400
def num_quarters : ℕ := 10
def num_dimes : ℕ := 12
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

theorem nickel_count : 
  (total_cents - (num_quarters * quarter_value + num_dimes * dime_value)) / nickel_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_nickel_count_l3850_385022


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3850_385090

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 90 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 28 + 2 * Real.sqrt 106 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3850_385090


namespace NUMINAMATH_CALUDE_class_size_l3850_385031

theorem class_size (S : ℚ) 
  (basketball : S / 2 = S * (1 / 2))
  (volleyball : S * (2 / 5) = S * (2 / 5))
  (both : S / 10 = S * (1 / 10))
  (neither : S * (1 / 5) = 4) : S = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3850_385031


namespace NUMINAMATH_CALUDE_min_sum_4x4x4_dice_cube_l3850_385048

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  total_dice : Nat
  opposite_face_sum : Nat

/-- Calculates the minimum visible sum on the large cube -/
def min_visible_sum (c : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum visible sum for a 4x4x4 cube of dice -/
theorem min_sum_4x4x4_dice_cube :
  ∀ c : LargeCube, 
    c.size = 4 → 
    c.total_dice = 64 → 
    c.opposite_face_sum = 7 → 
    min_visible_sum c = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_4x4x4_dice_cube_l3850_385048


namespace NUMINAMATH_CALUDE_optimal_strategy_minimizes_cost_l3850_385008

/-- Represents the bookstore's ordering strategy --/
structure OrderStrategy where
  numOrders : ℕ
  copiesPerOrder : ℕ

/-- Calculates the total cost for a given order strategy --/
def totalCost (s : OrderStrategy) : ℝ :=
  let handlingCost := 30 * s.numOrders
  let storageCost := 40 * (s.copiesPerOrder / 1000) * s.numOrders / 2
  handlingCost + storageCost

/-- The optimal order strategy --/
def optimalStrategy : OrderStrategy :=
  { numOrders := 10, copiesPerOrder := 15000 }

/-- Theorem stating that the optimal strategy minimizes total cost --/
theorem optimal_strategy_minimizes_cost :
  ∀ s : OrderStrategy,
    s.numOrders * s.copiesPerOrder = 150000 →
    totalCost optimalStrategy ≤ totalCost s :=
by sorry

#check optimal_strategy_minimizes_cost

end NUMINAMATH_CALUDE_optimal_strategy_minimizes_cost_l3850_385008


namespace NUMINAMATH_CALUDE_f_composition_at_pi_l3850_385041

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1 else Real.sin x - 2

theorem f_composition_at_pi : f (f Real.pi) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_pi_l3850_385041


namespace NUMINAMATH_CALUDE_constant_c_value_l3850_385020

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l3850_385020


namespace NUMINAMATH_CALUDE_green_peaches_count_l3850_385026

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 1

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 4

/-- The total number of peaches in all baskets -/
def total_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := total_peaches - (num_baskets * red_peaches_per_basket)

theorem green_peaches_count : green_peaches_per_basket = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l3850_385026


namespace NUMINAMATH_CALUDE_equal_intercept_line_theorem_tangent_circle_theorem_l3850_385086

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the line with equal intercepts passing through P
def equal_intercept_line (x y : ℝ) : Prop := x + y = 3

-- Define the circle
def tangent_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem for the line with equal intercepts
theorem equal_intercept_line_theorem :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, equal_intercept_line x y ↔ x / a + y / a = 1) ∧
  equal_intercept_line point_P.1 point_P.2 :=
sorry

-- Theorem for the tangent circle
theorem tangent_circle_theorem :
  ∃ A B : ℝ × ℝ,
  (line_l A.1 A.2 ∧ A.2 = 0) ∧
  (line_l B.1 B.2 ∧ B.1 = 0) ∧
  (∀ x y : ℝ, tangent_circle x y →
    (x = 0 ∨ y = 0 ∨ line_l x y)) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_theorem_tangent_circle_theorem_l3850_385086


namespace NUMINAMATH_CALUDE_circles_tangency_l3850_385059

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

-- Define the condition of having exactly one common point
def have_one_common_point (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 a

-- State the theorem
theorem circles_tangency (a : ℝ) :
  have_one_common_point a → a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_tangency_l3850_385059


namespace NUMINAMATH_CALUDE_equation_solution_l3850_385072

theorem equation_solution (t : ℝ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * Real.sin t ^ 2 - Real.sin (2 * t) + 3 * Real.cos t ^ 2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3850_385072


namespace NUMINAMATH_CALUDE_domain_of_g_l3850_385077

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l3850_385077


namespace NUMINAMATH_CALUDE_mario_blossoms_l3850_385001

/-- The number of hibiscus plants Mario has -/
def num_plants : ℕ := 3

/-- The number of flowers on the first hibiscus plant -/
def flowers_first : ℕ := 2

/-- The number of flowers on the second hibiscus plant -/
def flowers_second : ℕ := 2 * flowers_first

/-- The number of flowers on the third hibiscus plant -/
def flowers_third : ℕ := 4 * flowers_second

/-- The total number of blossoms Mario has -/
def total_blossoms : ℕ := flowers_first + flowers_second + flowers_third

theorem mario_blossoms : total_blossoms = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_blossoms_l3850_385001


namespace NUMINAMATH_CALUDE_product_of_powers_l3850_385034

theorem product_of_powers : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l3850_385034


namespace NUMINAMATH_CALUDE_fan_weight_l3850_385067

/-- Given a box with fans, calculate the weight of a single fan. -/
theorem fan_weight (total_weight : ℝ) (num_fans : ℕ) (empty_box_weight : ℝ) 
  (h1 : total_weight = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_fans = 0.76 := by
  sorry

#check fan_weight

end NUMINAMATH_CALUDE_fan_weight_l3850_385067


namespace NUMINAMATH_CALUDE_nathan_blanket_warmth_l3850_385062

theorem nathan_blanket_warmth (total_blankets : ℕ) (warmth_per_blanket : ℕ) (fraction_used : ℚ) : 
  total_blankets = 14 → 
  warmth_per_blanket = 3 → 
  fraction_used = 1/2 →
  (↑total_blankets * fraction_used : ℚ).floor * warmth_per_blanket = 21 := by
sorry

end NUMINAMATH_CALUDE_nathan_blanket_warmth_l3850_385062


namespace NUMINAMATH_CALUDE_monomial_properties_l3850_385042

/-- The coefficient of a monomial -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The degree of a monomial -/
def degree (x y : ℚ) : ℕ := 2 + 1

theorem monomial_properties :
  let m : ℚ := -π / 7
  let x : ℚ := 0  -- Placeholder value, not used in computation
  let y : ℚ := 0  -- Placeholder value, not used in computation
  (coefficient m x y = -π / 7) ∧ (degree x y = 3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3850_385042


namespace NUMINAMATH_CALUDE_prob_different_suits_modified_deck_l3850_385014

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The modified 40-card deck -/
def modified_deck : Deck :=
  { total_cards := 40
  , num_suits := 4
  , cards_per_suit := 10
  , h_total := rfl }

theorem prob_different_suits_modified_deck :
  prob_different_suits modified_deck = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_modified_deck_l3850_385014


namespace NUMINAMATH_CALUDE_antonette_overall_score_l3850_385047

/-- Calculates the overall score percentage on a combined test, given individual test scores and problem counts. -/
def overall_score (score1 score2 score3 : ℚ) (problems1 problems2 problems3 : ℕ) : ℚ :=
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / (problems1 + problems2 + problems3)

/-- Rounds a rational number to the nearest integer. -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem antonette_overall_score :
  let score1 : ℚ := 70/100
  let score2 : ℚ := 80/100
  let score3 : ℚ := 90/100
  let problems1 : ℕ := 10
  let problems2 : ℕ := 20
  let problems3 : ℕ := 30
  round_to_nearest (overall_score score1 score2 score3 problems1 problems2 problems3 * 100) = 83 := by
  sorry

end NUMINAMATH_CALUDE_antonette_overall_score_l3850_385047


namespace NUMINAMATH_CALUDE_episodes_per_season_l3850_385082

theorem episodes_per_season 
  (days : ℕ) 
  (episodes_per_day : ℕ) 
  (seasons : ℕ) 
  (h1 : days = 10)
  (h2 : episodes_per_day = 6)
  (h3 : seasons = 4)
  (h4 : (days * episodes_per_day) % seasons = 0) : 
  (days * episodes_per_day) / seasons = 15 := by
sorry

end NUMINAMATH_CALUDE_episodes_per_season_l3850_385082


namespace NUMINAMATH_CALUDE_g_composition_two_roots_l3850_385092

def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

theorem g_composition_two_roots (d : ℝ) : 
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g d (g d x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_two_roots_l3850_385092


namespace NUMINAMATH_CALUDE_maximize_product_l3850_385049

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 100) :
  x^4 * y^6 ≤ 40^4 * 60^6 ∧ 
  (x^4 * y^6 = 40^4 * 60^6 ↔ x = 40 ∧ y = 60) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l3850_385049


namespace NUMINAMATH_CALUDE_cubic_repeated_root_condition_l3850_385071

/-- A cubic polynomial with a repeated root -/
def has_repeated_root (b : ℝ) : Prop :=
  ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧
           (3 * b * x^2 + 30 * x + 9 = 0)

/-- Theorem stating that if a nonzero b makes the cubic have a repeated root, then b = 100 -/
theorem cubic_repeated_root_condition (b : ℝ) (hb : b ≠ 0) :
  has_repeated_root b → b = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_repeated_root_condition_l3850_385071


namespace NUMINAMATH_CALUDE_equation_solution_l3850_385044

theorem equation_solution (x : Real) :
  x ∈ Set.Ioo (-π / 2) 0 →
  (Real.sqrt 3 / Real.sin x) + (1 / Real.cos x) = 4 →
  x = -4 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3850_385044


namespace NUMINAMATH_CALUDE_square_of_negative_l3850_385075

theorem square_of_negative (a : ℝ) : (-a)^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_l3850_385075


namespace NUMINAMATH_CALUDE_distance_is_600_l3850_385068

/-- The distance between two points A and B, given specific train travel conditions. -/
def distance_between_points : ℝ :=
  let forward_speed : ℝ := 200
  let return_speed : ℝ := 100
  let time_difference : ℝ := 3
  600

/-- Theorem stating that the distance between points A and B is 600 km under given conditions. -/
theorem distance_is_600 (forward_speed return_speed time_difference : ℝ)
  (h1 : forward_speed = 200)
  (h2 : return_speed = 100)
  (h3 : time_difference = 3)
  : distance_between_points = 600 :=
by sorry

end NUMINAMATH_CALUDE_distance_is_600_l3850_385068


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3850_385028

/-- Given a triangle ABC with sides a, b, c, this structure represents the triangle and related points --/
structure TriangleWithIntersections where
  -- The lengths of the sides of triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- The area of triangle ABC
  S_ABC : ℝ
  -- The area of hexagon PQRSTF
  S_PQRSTF : ℝ
  -- Assumption that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating the relationship between the areas --/
theorem area_ratio_theorem (t : TriangleWithIntersections) :
  t.S_PQRSTF / t.S_ABC = 1 - (t.a * t.b + t.b * t.c + t.c * t.a) / (t.a + t.b + t.c)^2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3850_385028


namespace NUMINAMATH_CALUDE_project_bolts_boxes_l3850_385080

/-- The number of bolts in each box of bolts -/
def bolts_per_box : ℕ := 11

/-- The number of boxes of nuts purchased -/
def boxes_of_nuts : ℕ := 3

/-- The number of nuts in each box of nuts -/
def nuts_per_box : ℕ := 15

/-- The number of bolts left over -/
def bolts_leftover : ℕ := 3

/-- The number of nuts left over -/
def nuts_leftover : ℕ := 6

/-- The total number of bolts and nuts used for the project -/
def total_used : ℕ := 113

/-- The minimum number of boxes of bolts purchased -/
def min_boxes_of_bolts : ℕ := 7

theorem project_bolts_boxes :
  ∃ (boxes_of_bolts : ℕ),
    boxes_of_bolts * bolts_per_box ≥
      total_used - (boxes_of_nuts * nuts_per_box - nuts_leftover) + bolts_leftover ∧
    boxes_of_bolts = min_boxes_of_bolts :=
by sorry

end NUMINAMATH_CALUDE_project_bolts_boxes_l3850_385080


namespace NUMINAMATH_CALUDE_palindrome_existence_l3850_385037

/-- A number is a palindrome if it reads the same backwards and forwards in its decimal representation -/
def IsPalindrome (m : ℕ) : Prop :=
  ∃ (digits : List ℕ), m = digits.foldl (fun acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number n, there exists a natural number N such that 9 * 5^n * N is a palindrome -/
theorem palindrome_existence (n : ℕ) : ∃ (N : ℕ), IsPalindrome (9 * 5^n * N) := by
  sorry

end NUMINAMATH_CALUDE_palindrome_existence_l3850_385037


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_is_five_l3850_385098

/-- The cost of a Ferris wheel ride that satisfies the given conditions -/
def ferris_wheel_cost : ℕ → Prop := fun cost =>
  ∃ (total_children ferris_children : ℕ),
    total_children = 5 ∧
    ferris_children = 3 ∧
    total_children * (2 * 8 + 3) + ferris_children * cost = 110

/-- The cost of the Ferris wheel ride is $5 per child -/
theorem ferris_wheel_cost_is_five : ferris_wheel_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_is_five_l3850_385098


namespace NUMINAMATH_CALUDE_group_work_problem_l3850_385016

theorem group_work_problem (n : ℕ) (W_total : ℝ) : 
  (n : ℝ) * (W_total / 55) = ((n : ℝ) - 15) * (W_total / 60) → n = 165 := by
  sorry

end NUMINAMATH_CALUDE_group_work_problem_l3850_385016


namespace NUMINAMATH_CALUDE_garrison_problem_l3850_385004

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 2000

/-- Represents the number of reinforcement men -/
def reinforcement : ℕ := 1600

/-- Represents the initial number of days the provisions would last -/
def initial_days : ℕ := 54

/-- Represents the number of days passed before reinforcement -/
def days_before_reinforcement : ℕ := 18

/-- Represents the number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_problem :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement :=
by sorry

end NUMINAMATH_CALUDE_garrison_problem_l3850_385004


namespace NUMINAMATH_CALUDE_inequality_solution_l3850_385053

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3850_385053


namespace NUMINAMATH_CALUDE_smallest_x_value_l3850_385056

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) → x ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3850_385056


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3850_385017

theorem fraction_evaluation (a b : ℝ) (h : a ≠ b) : (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3850_385017


namespace NUMINAMATH_CALUDE_regular_polygon_with_900_degree_sum_l3850_385039

theorem regular_polygon_with_900_degree_sum (n : ℕ) : 
  n > 2 → (n - 2) * 180 = 900 → n = 7 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_900_degree_sum_l3850_385039


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l3850_385099

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the polynomial
def f (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + 45

-- State the theorem
theorem cubic_root_theorem (p q : ℚ) :
  f p q (2 - 3 * cubeRoot 5) = 0 → p = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l3850_385099


namespace NUMINAMATH_CALUDE_trapezoid_height_l3850_385032

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  let height := a * b / (b - a)
  ∃ (x y : ℝ), 
    (x^2 + y^2 = a^2 + b^2) ∧ 
    ((b - a)^2 = x^2 + y^2 - x*y*Real.sqrt 2) ∧
    (x * y * Real.sqrt 2 = 2 * (b - a) * height) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l3850_385032


namespace NUMINAMATH_CALUDE_function_symmetry_l3850_385084

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
def passes_through_point (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem function_symmetry 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : passes_through_point (log a) 2 (-1)) 
  (f : ℝ → ℝ) 
  (h4 : symmetric_wrt_y_eq_x f (log a)) : 
  f = fun x ↦ (1/2)^x := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l3850_385084


namespace NUMINAMATH_CALUDE_max_consecutive_sum_45_l3850_385052

/-- The sum of consecutive integers starting from a given integer -/
def sum_consecutive (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + count - 1) / 2

/-- The property that a sequence of consecutive integers sums to 45 -/
def sums_to_45 (start : ℤ) (count : ℕ) : Prop :=
  sum_consecutive start count = 45

/-- The theorem stating that 90 is the maximum number of consecutive integers that sum to 45 -/
theorem max_consecutive_sum_45 :
  (∃ start : ℤ, sums_to_45 start 90) ∧
  (∀ count : ℕ, count > 90 → ∀ start : ℤ, ¬ sums_to_45 start count) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_45_l3850_385052


namespace NUMINAMATH_CALUDE_milk_calculation_l3850_385007

/-- The amount of milk Yuna's family drank as a fraction of the total -/
def milk_drunk : ℝ := 0.4

/-- The amount of leftover milk in liters -/
def leftover_milk : ℝ := 0.69

/-- The initial amount of milk in liters -/
def initial_milk : ℝ := 1.15

theorem milk_calculation (milk_drunk : ℝ) (leftover_milk : ℝ) (initial_milk : ℝ) :
  milk_drunk = 0.4 →
  leftover_milk = 0.69 →
  initial_milk = 1.15 →
  initial_milk * (1 - milk_drunk) = leftover_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_calculation_l3850_385007


namespace NUMINAMATH_CALUDE_initial_men_count_l3850_385094

/-- Given a group of men where:
  * The average age increases by 2 years when two women replace two men
  * The replaced men are 10 and 12 years old
  * The average age of the women is 21 years
  Prove that the initial number of men in the group is 10 -/
theorem initial_men_count (M : ℕ) (A : ℚ) : 
  (M * A - 22 + 42 = M * (A + 2)) → M = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l3850_385094


namespace NUMINAMATH_CALUDE_solve_equation_l3850_385030

theorem solve_equation (x : ℝ) : (x - 5) ^ 4 = 16 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3850_385030


namespace NUMINAMATH_CALUDE_some_number_value_l3850_385027

theorem some_number_value : ∀ some_number : ℝ, 
  (54 / some_number) * (54 / 162) = 1 → some_number = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3850_385027


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3850_385073

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 4*y + 3 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3850_385073


namespace NUMINAMATH_CALUDE_max_value_of_a_l3850_385055

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3850_385055


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l3850_385060

def library_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def hospital_interval : ℕ := 30

theorem next_simultaneous_ring : 
  Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l3850_385060


namespace NUMINAMATH_CALUDE_tooth_fairy_calculation_l3850_385088

theorem tooth_fairy_calculation (total_amount : ℕ) (total_teeth : ℕ) (lost_teeth : ℕ) (first_tooth_amount : ℕ) :
  total_teeth = 20 →
  total_amount = 54 →
  lost_teeth = 2 →
  first_tooth_amount = 20 →
  (total_amount - first_tooth_amount) / (total_teeth - lost_teeth - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_tooth_fairy_calculation_l3850_385088
