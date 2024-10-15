import Mathlib

namespace NUMINAMATH_CALUDE_ice_melting_volume_l2772_277212

theorem ice_melting_volume (ice_volume : ℝ) (h1 : ice_volume = 2) :
  let water_volume := ice_volume * (10/11)
  water_volume = 20/11 :=
by sorry

end NUMINAMATH_CALUDE_ice_melting_volume_l2772_277212


namespace NUMINAMATH_CALUDE_complex_exponential_thirteen_pi_over_two_equals_i_l2772_277227

theorem complex_exponential_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_thirteen_pi_over_two_equals_i_l2772_277227


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l2772_277274

def particle_position (t : ℕ) : ℕ × ℕ :=
  let n := (Nat.sqrt (t / 4) : ℕ)
  let remaining_time := t - 4 * n^2
  let side_length := 2 * n + 1
  if remaining_time ≤ side_length then
    (remaining_time, 0)
  else if remaining_time ≤ 2 * side_length then
    (side_length, remaining_time - side_length)
  else if remaining_time ≤ 3 * side_length then
    (3 * side_length - remaining_time, side_length)
  else
    (0, 4 * side_length - remaining_time)

theorem particle_position_after_2023_minutes :
  particle_position 2023 = (87, 0) := by
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l2772_277274


namespace NUMINAMATH_CALUDE_square_root_of_product_l2772_277225

theorem square_root_of_product : Real.sqrt (64 * Real.sqrt 49) = 8 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_l2772_277225


namespace NUMINAMATH_CALUDE_quadratic_m_gt_n_l2772_277203

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The y-value of a quadratic function at a given x -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_m_gt_n 
  (f : QuadraticFunction)
  (h1 : eval f (-1) = 0)
  (h2 : eval f 0 = 2)
  (h3 : eval f 3 = 0)
  (m n : ℝ)
  (hm : eval f 1 = m)
  (hn : eval f 2 = n) :
  m > n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_m_gt_n_l2772_277203


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l2772_277283

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧  -- positive length and width
  2 * (l + w) = 240 ∧  -- perimeter is 240 feet
  l * w = 8 * 240 →  -- area is eight times perimeter
  max l w = 101 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l2772_277283


namespace NUMINAMATH_CALUDE_base_conversion_156_to_234_l2772_277284

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_conversion_156_to_234 :
  156 = base8ToBase10 [4, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_156_to_234_l2772_277284


namespace NUMINAMATH_CALUDE_range_of_x_l2772_277215

theorem range_of_x (x : ℝ) : 
  (∃ m : ℝ, m ∈ Set.Icc 1 3 ∧ x + 3 * m + 5 > 0) → x > -14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2772_277215


namespace NUMINAMATH_CALUDE_log_expression_evaluation_l2772_277222

theorem log_expression_evaluation :
  2 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (2 * Real.log 3 / Real.log 5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_evaluation_l2772_277222


namespace NUMINAMATH_CALUDE_average_rate_of_change_cubic_l2772_277237

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Theorem statement
theorem average_rate_of_change_cubic (a b : ℝ) (h : a = 1 ∧ b = 2) :
  (f b - f a) / (b - a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_cubic_l2772_277237


namespace NUMINAMATH_CALUDE_infinite_good_pairs_l2772_277205

/-- A number is "good" if every prime factor in its prime factorization appears with an exponent of at least 2 -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p ^ k ∣ n)

/-- The sequence of "good" numbers -/
def good_sequence : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * good_sequence n * (good_sequence n + 1)

/-- Theorem stating the existence of infinitely many pairs of consecutive "good" numbers -/
theorem infinite_good_pairs :
  ∀ n : ℕ, is_good (good_sequence n) ∧ is_good (good_sequence n + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_good_pairs_l2772_277205


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2772_277249

/-- Represents the number of students in the classroom -/
def total_students : ℕ := 25

/-- Represents the number of students who scored 'A' on both exams -/
def students_a : ℕ := 3

/-- Represents the number of students who scored 'B' on both exams -/
def students_b : ℕ := 2

/-- Represents the number of students who scored 'C' on both exams -/
def students_c : ℕ := 1

/-- Represents the number of students who scored 'D' on both exams -/
def students_d : ℕ := 3

/-- Calculates the total number of students who received the same grade on both exams -/
def same_grade_students : ℕ := students_a + students_b + students_c + students_d

/-- Theorem: The percentage of students who received the same grade on both exams is 36% -/
theorem same_grade_percentage :
  (same_grade_students : ℚ) / total_students * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2772_277249


namespace NUMINAMATH_CALUDE_newspaper_conference_max_neither_l2772_277258

theorem newspaper_conference_max_neither (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) (N : ℕ) :
  total = 90 →
  writers = 45 →
  editors ≥ 39 →
  writers + editors - x + N = total →
  N = 2 * x →
  N ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_neither_l2772_277258


namespace NUMINAMATH_CALUDE_prob_first_second_given_two_fail_l2772_277259

-- Define the failure probabilities for each component
def p1 : ℝ := 0.2
def p2 : ℝ := 0.4
def p3 : ℝ := 0.3

-- Define the probability of two components failing
def prob_two_fail : ℝ := p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the probability of the first and second components failing
def prob_first_second_fail : ℝ := p1 * p2 * (1 - p3)

-- Theorem statement
theorem prob_first_second_given_two_fail : 
  prob_first_second_fail / prob_two_fail = 0.3 := by sorry

end NUMINAMATH_CALUDE_prob_first_second_given_two_fail_l2772_277259


namespace NUMINAMATH_CALUDE_part1_part2_l2772_277255

-- Part 1
theorem part1 (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x = Real.exp x + Real.sin x + b) →
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  b ≥ -1 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (b m : ℝ) :
  (∀ x : ℝ, f x = Real.exp x + b) →
  (f 0 = 1 ∧ (deriv f) 0 = 1) →
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = (m - 2*x₁) / x₁ ∧ f x₂ = (m - 2*x₂) / x₂) →
  -1 / Real.exp 1 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2772_277255


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2772_277263

theorem sum_of_fractions : (3 / 10 : ℚ) + (29 / 5 : ℚ) = 61 / 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2772_277263


namespace NUMINAMATH_CALUDE_min_sticks_to_remove_8x8_l2772_277236

/-- Represents a chessboard with sticks on edges -/
structure Chessboard :=
  (size : Nat)
  (sticks : Nat)

/-- The minimum number of sticks that must be removed to avoid rectangles -/
def min_sticks_to_remove (board : Chessboard) : Nat :=
  Nat.ceil (2 / 3 * (board.size * board.size))

/-- Theorem stating the minimum number of sticks to remove for an 8x8 chessboard -/
theorem min_sticks_to_remove_8x8 :
  let board : Chessboard := ⟨8, 144⟩
  min_sticks_to_remove board = 43 := by
  sorry

#eval min_sticks_to_remove ⟨8, 144⟩

end NUMINAMATH_CALUDE_min_sticks_to_remove_8x8_l2772_277236


namespace NUMINAMATH_CALUDE_bead_arrangement_theorem_l2772_277256

/-- Represents a bead with a color --/
structure Bead where
  color : Nat

/-- Represents a necklace of beads --/
def Necklace := List Bead

/-- Checks if a segment of beads contains at least k different colors --/
def hasAtLeastKColors (segment : List Bead) (k : Nat) : Prop :=
  (segment.map (·.color)).toFinset.card ≥ k

/-- The property we want to prove --/
theorem bead_arrangement_theorem (total_beads : Nat) (num_colors : Nat) (beads_per_color : Nat)
    (h1 : total_beads = 1000)
    (h2 : num_colors = 50)
    (h3 : beads_per_color = 20)
    (h4 : total_beads = num_colors * beads_per_color) :
    ∃ (n : Nat),
      (∀ (necklace : Necklace),
        necklace.length = total_beads →
        (∀ (i : Nat),
          i + n ≤ necklace.length →
          hasAtLeastKColors (necklace.take n) 25)) ∧
      (∀ (m : Nat),
        m < n →
        ∃ (necklace : Necklace),
          necklace.length = total_beads ∧
          ∃ (i : Nat),
            i + m ≤ necklace.length ∧
            ¬hasAtLeastKColors (necklace.take m) 25) :=
  sorry

#check bead_arrangement_theorem

end NUMINAMATH_CALUDE_bead_arrangement_theorem_l2772_277256


namespace NUMINAMATH_CALUDE_cyclists_speed_problem_l2772_277265

theorem cyclists_speed_problem (total_distance : ℝ) (speed_difference : ℝ) :
  total_distance = 270 →
  speed_difference = 1.5 →
  ∃ (speed1 speed2 time : ℝ),
    speed1 > 0 ∧
    speed2 > 0 ∧
    speed1 = speed2 + speed_difference ∧
    time = speed1 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed1 = 12 ∧
    speed2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_problem_l2772_277265


namespace NUMINAMATH_CALUDE_sqrt_180_equals_6_sqrt_5_l2772_277221

theorem sqrt_180_equals_6_sqrt_5 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_equals_6_sqrt_5_l2772_277221


namespace NUMINAMATH_CALUDE_x_equals_one_l2772_277244

theorem x_equals_one :
  ∀ x : ℝ,
  ((x^31) / (5^31)) * ((x^16) / (4^16)) = 1 / (2 * (10^31)) →
  x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l2772_277244


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l2772_277223

/-- Represents the scoring system for a test -/
structure TestScore where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculate_score (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

/-- Theorem stating the number of correct answers given the conditions -/
theorem correct_answers_for_given_score
  (test : TestScore)
  (h1 : test.total_questions = 100)
  (h2 : test.correct_answers + test.incorrect_answers = test.total_questions)
  (h3 : test.score = calculate_score test.correct_answers test.incorrect_answers)
  (h4 : test.score = 76) :
  test.correct_answers = 92 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_score_l2772_277223


namespace NUMINAMATH_CALUDE_robot_energy_cells_l2772_277298

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The robot's reported energy cell count in base 7 -/
def robotReport : List Nat := [1, 2, 3]

theorem robot_energy_cells :
  base7ToBase10 robotReport = 162 := by
  sorry

end NUMINAMATH_CALUDE_robot_energy_cells_l2772_277298


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2772_277272

theorem smallest_winning_number : ∃ N : ℕ, 
  N ≤ 499 ∧ 
  27 * N + 360 < 500 ∧ 
  (∀ k : ℕ, k < N → 27 * k + 360 ≥ 500) ∧
  N = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2772_277272


namespace NUMINAMATH_CALUDE_product_digit_sum_l2772_277277

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  hundreds_digit product + units_digit product = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2772_277277


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l2772_277289

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ (k : ℕ), k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l2772_277289


namespace NUMINAMATH_CALUDE_garbage_collection_theorem_l2772_277207

/-- The amount of garbage collected by four people given specific relationships between their collections. -/
def total_garbage_collected (daliah_amount : ℝ) : ℝ := 
  let dewei_amount := daliah_amount - 2
  let zane_amount := 4 * dewei_amount
  let bela_amount := zane_amount + 3.75
  daliah_amount + dewei_amount + zane_amount + bela_amount

/-- Theorem stating that the total amount of garbage collected is 160.75 pounds when Daliah collects 17.5 pounds. -/
theorem garbage_collection_theorem : 
  total_garbage_collected 17.5 = 160.75 := by
  sorry

#eval total_garbage_collected 17.5

end NUMINAMATH_CALUDE_garbage_collection_theorem_l2772_277207


namespace NUMINAMATH_CALUDE_negation_of_universal_inequality_l2772_277260

theorem negation_of_universal_inequality : 
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_inequality_l2772_277260


namespace NUMINAMATH_CALUDE_max_reciprocal_negative_l2772_277286

theorem max_reciprocal_negative (B : Set ℝ) (a₀ : ℝ) :
  (B.Nonempty) →
  (0 ∉ B) →
  (∀ x ∈ B, x ≤ a₀) →
  (a₀ ∈ B) →
  (a₀ < 0) →
  (∀ x ∈ B, -x⁻¹ ≤ -a₀⁻¹) ∧ (-a₀⁻¹ ∈ {-x⁻¹ | x ∈ B}) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_negative_l2772_277286


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2772_277246

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 18 →
  (men : ℚ) / women = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2772_277246


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2772_277262

theorem ratio_sum_problem (a b c : ℝ) 
  (ratio : a / 4 = b / 5 ∧ b / 5 = c / 7)
  (sum : a + b + c = 240) :
  2 * b - a + c = 195 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2772_277262


namespace NUMINAMATH_CALUDE_television_price_proof_l2772_277291

theorem television_price_proof (discount_rate : ℝ) (final_price : ℝ) (num_tvs : ℕ) :
  discount_rate = 0.25 →
  final_price = 975 →
  num_tvs = 2 →
  ∃ (original_price : ℝ),
    original_price = 650 ∧
    final_price = (1 - discount_rate) * (num_tvs * original_price) :=
by sorry

end NUMINAMATH_CALUDE_television_price_proof_l2772_277291


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l2772_277229

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH 
    when GE is 3 times EH and GF is 8 times FH. -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F lie on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l2772_277229


namespace NUMINAMATH_CALUDE_stating_student_marks_theorem_l2772_277287

/-- 
A function that calculates the total marks secured in an examination
given the following parameters:
- total_questions: The total number of questions in the exam
- correct_answers: The number of questions answered correctly
- marks_per_correct: The number of marks awarded for each correct answer
- marks_per_wrong: The number of marks deducted for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_per_wrong)

/-- 
Theorem stating that given the specific conditions of the exam,
the student secures 140 marks in total.
-/
theorem student_marks_theorem :
  calculate_total_marks 60 40 4 1 = 140 := by
  sorry

end NUMINAMATH_CALUDE_stating_student_marks_theorem_l2772_277287


namespace NUMINAMATH_CALUDE_stratified_sample_third_group_l2772_277251

/-- Given a population with three groups in the ratio 2:5:3 and a stratified sample of size 120,
    the number of items from the third group in the sample is 36. -/
theorem stratified_sample_third_group : 
  ∀ (total_ratio : ℕ) (group1_ratio group2_ratio group3_ratio : ℕ) (sample_size : ℕ),
    total_ratio = group1_ratio + group2_ratio + group3_ratio →
    group1_ratio = 2 →
    group2_ratio = 5 →
    group3_ratio = 3 →
    sample_size = 120 →
    (sample_size * group3_ratio) / total_ratio = 36 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_third_group_l2772_277251


namespace NUMINAMATH_CALUDE_father_son_work_time_work_completed_in_three_days_l2772_277292

/-- Given a task that takes 6 days for either a man or his son to complete alone,
    prove that they can complete it together in 3 days. -/
theorem father_son_work_time : ℝ → Prop :=
  fun total_work =>
    let man_rate := total_work / 6
    let son_rate := total_work / 6
    let combined_rate := man_rate + son_rate
    (total_work / combined_rate) = 3

/-- The main theorem stating that the work will be completed in 3 days -/
theorem work_completed_in_three_days (total_work : ℝ) (h : total_work > 0) :
  father_son_work_time total_work := by
  sorry

#check work_completed_in_three_days

end NUMINAMATH_CALUDE_father_son_work_time_work_completed_in_three_days_l2772_277292


namespace NUMINAMATH_CALUDE_unique_friend_groups_l2772_277234

theorem unique_friend_groups (n : ℕ) (h : n = 10) : 
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by
  sorry

end NUMINAMATH_CALUDE_unique_friend_groups_l2772_277234


namespace NUMINAMATH_CALUDE_product_cost_reduction_l2772_277266

theorem product_cost_reduction (original_selling_price : ℝ) 
  (original_profit_rate : ℝ) (new_profit_rate : ℝ) (additional_profit : ℝ) :
  original_selling_price = 659.9999999999994 →
  original_profit_rate = 0.1 →
  new_profit_rate = 0.3 →
  additional_profit = 42 →
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_cost := (original_selling_price + additional_profit) / (1 + new_profit_rate)
  (original_cost - new_cost) / original_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_product_cost_reduction_l2772_277266


namespace NUMINAMATH_CALUDE_jared_age_difference_l2772_277282

/-- Given three friends Jared, Hakimi, and Molly, this theorem proves that Jared is 10 years older than Hakimi -/
theorem jared_age_difference (jared_age hakimi_age molly_age : ℕ) : 
  hakimi_age = 40 →
  molly_age = 30 →
  (jared_age + hakimi_age + molly_age) / 3 = 40 →
  jared_age - hakimi_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jared_age_difference_l2772_277282


namespace NUMINAMATH_CALUDE_surface_area_of_special_rectangular_solid_l2772_277214

/-- A function that checks if a number is prime or a square of a prime -/
def isPrimeOrSquareOfPrime (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p, Nat.Prime p ∧ n = p^2

/-- Definition of a rectangular solid with the given properties -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_valid : isPrimeOrSquareOfPrime length
  width_valid : isPrimeOrSquareOfPrime width
  height_valid : isPrimeOrSquareOfPrime height
  volume_is_1155 : length * width * height = 1155

/-- The theorem to be proved -/
theorem surface_area_of_special_rectangular_solid (r : RectangularSolid) :
  2 * (r.length * r.width + r.width * r.height + r.height * r.length) = 814 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_special_rectangular_solid_l2772_277214


namespace NUMINAMATH_CALUDE_equation_solution_l2772_277293

theorem equation_solution (x : ℝ) :
  x ≠ 5 ∧ x ≠ 6 →
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2772_277293


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2772_277216

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    left focus F₁ and right focus F₂ on the x-axis,
    point P(3,4) on an asymptote, and |PF₁ + PF₂| = |F₁F₂|,
    prove that the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : ∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0))
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ k : ℝ, k * 3 = 4 ∧ (∀ x y : ℝ, y = k * x → x^2/a^2 - y^2/b^2 = 1))
  (h_vector_sum : ‖P - F₁ + (P - F₂)‖ = ‖F₂ - F₁‖) :
  ∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2772_277216


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2772_277231

theorem adult_ticket_cost (num_adults num_children : ℕ) (total_bill child_ticket_cost : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  total_bill = 124 →
  child_ticket_cost = 4 →
  (total_bill - num_children * child_ticket_cost) / num_adults = 8 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2772_277231


namespace NUMINAMATH_CALUDE_parabola_c_value_l2772_277261

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 3 = -5 →   -- Vertex at (3, -5)
  p.y_at 4 = -3 →   -- Passes through (4, -3)
  p.c = 13 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2772_277261


namespace NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l2772_277235

/-- The number of children who got on the bus at a stop -/
def ChildrenWhoGotOn (initial final : ℕ) : ℕ := final - initial

theorem children_who_got_on_bus_stop (initial final : ℕ) 
  (h1 : initial = 52) 
  (h2 : final = 76) : 
  ChildrenWhoGotOn initial final = 24 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l2772_277235


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l2772_277275

theorem sqrt_of_nine (x : ℝ) : x = Real.sqrt 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l2772_277275


namespace NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2772_277206

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates_in_third_quadrant :
  ∃ (p : Point), ThirdQuadrant p ∧ DistToXAxis p = 2 ∧ DistToYAxis p = 3 → p = Point.mk (-3) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2772_277206


namespace NUMINAMATH_CALUDE_number_of_rolls_not_random_variable_l2772_277273

/-- A type representing the outcome of a single die roll -/
inductive DieRoll
| one | two | three | four | five | six

/-- A type representing a random variable on die rolls -/
def RandomVariable := DieRoll → ℝ

/-- The number of times the die is rolled -/
def numberOfRolls : ℕ := 2

theorem number_of_rolls_not_random_variable :
  ¬ ∃ (f : RandomVariable), ∀ (r₁ r₂ : DieRoll), f r₁ = numberOfRolls ∧ f r₂ = numberOfRolls :=
sorry

end NUMINAMATH_CALUDE_number_of_rolls_not_random_variable_l2772_277273


namespace NUMINAMATH_CALUDE_rachels_reading_homework_l2772_277230

/-- Given that Rachel had 9 pages of math homework and 7 more pages of math homework than reading homework, prove that she had 2 pages of reading homework. -/
theorem rachels_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 9)
  (h2 : math_homework = reading_homework + 7) :
  reading_homework = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachels_reading_homework_l2772_277230


namespace NUMINAMATH_CALUDE_coupon_usage_theorem_l2772_277213

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isCouponDay (startDay : DayOfWeek) (n : Nat) : Prop :=
  ∃ k : Nat, k < 8 ∧ advanceDays startDay (7 * k) = DayOfWeek.Monday

theorem coupon_usage_theorem (startDay : DayOfWeek) :
  startDay = DayOfWeek.Sunday ↔
    ¬(isCouponDay startDay 8) ∧
    ∀ d : DayOfWeek, d ≠ DayOfWeek.Sunday → isCouponDay d 8 :=
by sorry

end NUMINAMATH_CALUDE_coupon_usage_theorem_l2772_277213


namespace NUMINAMATH_CALUDE_widget_earnings_proof_l2772_277285

/-- Calculates the earnings per widget given the hourly rate, weekly hours, target earnings, and required widget production. -/
def earnings_per_widget (hourly_rate : ℚ) (weekly_hours : ℕ) (target_earnings : ℚ) (widget_production : ℕ) : ℚ :=
  (target_earnings - hourly_rate * weekly_hours) / widget_production

/-- Proves that the earnings per widget is $0.16 given the specified conditions. -/
theorem widget_earnings_proof :
  let hourly_rate : ℚ := 25 / 2
  let weekly_hours : ℕ := 40
  let target_earnings : ℚ := 620
  let widget_production : ℕ := 750
  earnings_per_widget hourly_rate weekly_hours target_earnings widget_production = 16 / 100 := by
  sorry

#eval earnings_per_widget (25/2) 40 620 750

end NUMINAMATH_CALUDE_widget_earnings_proof_l2772_277285


namespace NUMINAMATH_CALUDE_malfunctioning_mix_sum_l2772_277250

/-- Represents the fractional composition of Papaya Splash ingredients -/
structure PapayaSplash :=
  (soda_water : ℚ)
  (lemon_juice : ℚ)
  (sugar : ℚ)
  (papaya_puree : ℚ)
  (secret_spice : ℚ)
  (lime_extract : ℚ)

/-- The standard formula for Papaya Splash -/
def standard_formula : PapayaSplash :=
  { soda_water := 8/21,
    lemon_juice := 4/21,
    sugar := 3/21,
    papaya_puree := 3/21,
    secret_spice := 2/21,
    lime_extract := 1/21 }

/-- The malfunctioning machine's mixing ratios -/
def malfunction_ratios : PapayaSplash :=
  { soda_water := 1/2,
    lemon_juice := 3,
    sugar := 2,
    papaya_puree := 1,
    secret_spice := 1/5,
    lime_extract := 1 }

/-- Applies the malfunction ratios to the standard formula -/
def apply_malfunction (formula : PapayaSplash) (ratios : PapayaSplash) : PapayaSplash :=
  { soda_water := formula.soda_water * ratios.soda_water,
    lemon_juice := formula.lemon_juice * ratios.lemon_juice,
    sugar := formula.sugar * ratios.sugar,
    papaya_puree := formula.papaya_puree * ratios.papaya_puree,
    secret_spice := formula.secret_spice * ratios.secret_spice,
    lime_extract := formula.lime_extract * ratios.lime_extract }

/-- Calculates the sum of soda water, sugar, and secret spice blend fractions -/
def sum_selected_ingredients (mix : PapayaSplash) : ℚ :=
  mix.soda_water + mix.sugar + mix.secret_spice

/-- Theorem stating that the sum of selected ingredients in the malfunctioning mix is 52/105 -/
theorem malfunctioning_mix_sum :
  sum_selected_ingredients (apply_malfunction standard_formula malfunction_ratios) = 52/105 :=
sorry

end NUMINAMATH_CALUDE_malfunctioning_mix_sum_l2772_277250


namespace NUMINAMATH_CALUDE_alligator_count_theorem_l2772_277211

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friends_count : ℕ) (friends_average : ℕ) : ℕ :=
  samara_count + friends_count * friends_average

/-- Theorem stating the total number of alligators seen by Samara and her friends -/
theorem alligator_count_theorem : 
  total_alligators 35 6 15 = 125 := by
  sorry

#eval total_alligators 35 6 15

end NUMINAMATH_CALUDE_alligator_count_theorem_l2772_277211


namespace NUMINAMATH_CALUDE_sin_pi_over_six_l2772_277295

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_six_l2772_277295


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2772_277268

theorem complex_fraction_simplification :
  let z : ℂ := (3 - I) / (1 - I)
  z = 2 + I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2772_277268


namespace NUMINAMATH_CALUDE_cricketer_score_percentage_l2772_277218

/-- A cricketer's score breakdown and calculation of runs made by running between wickets --/
theorem cricketer_score_percentage (total_score : ℕ) (boundaries : ℕ) (sixes : ℕ)
  (singles : ℕ) (twos : ℕ) (threes : ℕ) :
  total_score = 138 →
  boundaries = 12 →
  sixes = 2 →
  singles = 25 →
  twos = 7 →
  threes = 3 →
  (((singles * 1 + twos * 2 + threes * 3) : ℚ) / total_score) * 100 = 48 / 138 * 100 := by
  sorry

#eval (48 : ℚ) / 138 * 100

end NUMINAMATH_CALUDE_cricketer_score_percentage_l2772_277218


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2772_277288

theorem negation_of_proposition (p : Prop) :
  (∃ n : ℕ, n^2 > 2*n - 1) → (¬p ↔ ∀ n : ℕ, n^2 ≤ 2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2772_277288


namespace NUMINAMATH_CALUDE_number_divisible_by_56_l2772_277240

/-- The number formed by concatenating digits a, 7, 8, 3, and b -/
def number (a b : ℕ) : ℕ := a * 10000 + 7000 + 800 + 30 + b

/-- Theorem stating that 47832 is divisible by 56 -/
theorem number_divisible_by_56 : 
  number 4 2 % 56 = 0 :=
sorry

end NUMINAMATH_CALUDE_number_divisible_by_56_l2772_277240


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2772_277271

theorem race_speed_ratio (L : ℝ) (h_L : L > 0) : 
  let head_start := 0.35 * L
  let winning_distance := 0.25 * L
  let a_distance := L + head_start
  let b_distance := L + winning_distance
  ∃ R : ℝ, R * (L / b_distance) = a_distance / b_distance ∧ R = 1.08 :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2772_277271


namespace NUMINAMATH_CALUDE_probability_point_between_F_and_G_l2772_277243

/-- Given a line segment AB with points A, E, F, G, B placed consecutively,
    where AB = 4AE and AB = 8BF, the probability that a randomly selected
    point on AB lies between F and G is 1/2. -/
theorem probability_point_between_F_and_G (AB AE BF FG : ℝ) : 
  AB > 0 → AB = 4 * AE → AB = 8 * BF → FG = AB / 2 → FG / AB = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_point_between_F_and_G_l2772_277243


namespace NUMINAMATH_CALUDE_max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2772_277294

/-- A quadrilateral with one angle of 120 degrees can have at most 3 obtuse angles. -/
theorem max_obtuse_angles_in_quadrilateral_with_120_degree_angle :
  ∀ (a b c d : ℝ),
  a = 120 →
  a + b + c + d = 360 →
  a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90 →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2772_277294


namespace NUMINAMATH_CALUDE_critical_point_of_cubic_l2772_277224

/-- The function f(x) = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem critical_point_of_cubic (x : ℝ) : 
  (f' x = 0 ↔ x = 0) :=
sorry

#check critical_point_of_cubic

end NUMINAMATH_CALUDE_critical_point_of_cubic_l2772_277224


namespace NUMINAMATH_CALUDE_survival_rate_definition_correct_l2772_277264

/-- Survival rate is defined as the percentage of living seedlings out of the total seedlings -/
def survival_rate (living_seedlings total_seedlings : ℕ) : ℚ :=
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100

/-- The given definition of survival rate is correct -/
theorem survival_rate_definition_correct :
  ∀ (living_seedlings total_seedlings : ℕ),
  survival_rate living_seedlings total_seedlings =
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_survival_rate_definition_correct_l2772_277264


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l2772_277239

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l2772_277239


namespace NUMINAMATH_CALUDE_simplify_expression_l2772_277248

theorem simplify_expression (x : ℝ) (h : x^8 ≠ 1) :
  4 / (1 + x^4) + 2 / (1 + x^2) + 1 / (1 + x) + 1 / (1 - x) = 8 / (1 - x^8) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2772_277248


namespace NUMINAMATH_CALUDE_total_stars_l2772_277254

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 124) 
  (h2 : stars_per_student = 3) : 
  num_students * stars_per_student = 372 := by
sorry

end NUMINAMATH_CALUDE_total_stars_l2772_277254


namespace NUMINAMATH_CALUDE_blood_expiration_time_l2772_277217

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  hle24 : hours < 24
  mle59 : minutes < 60

/-- Represents a date with a day and a time -/
structure Date where
  day : Nat
  time : TimeOfDay

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def secondsToTime (seconds : Nat) : TimeOfDay :=
  let totalMinutes := seconds / 60
  let hours := totalMinutes / 60
  let minutes := totalMinutes % 60
  ⟨hours % 24, minutes, by sorry, by sorry⟩

def addTimeToDate (d : Date) (seconds : Nat) : Date :=
  let newTime := secondsToTime ((d.time.hours * 60 + d.time.minutes) * 60 + seconds)
  ⟨d.day + (if newTime.hours < d.time.hours then 1 else 0), newTime⟩

theorem blood_expiration_time :
  let donationDate : Date := ⟨1, ⟨12, 0, by sorry, by sorry⟩⟩
  let expirationSeconds := factorial 8
  let expirationDate := addTimeToDate donationDate expirationSeconds
  expirationDate = ⟨1, ⟨23, 13, by sorry, by sorry⟩⟩ :=
by sorry

end NUMINAMATH_CALUDE_blood_expiration_time_l2772_277217


namespace NUMINAMATH_CALUDE_power_sum_equality_l2772_277208

theorem power_sum_equality : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2772_277208


namespace NUMINAMATH_CALUDE_people_in_line_l2772_277252

/-- The number of people in a line is equal to the number of people behind the first passenger plus one. -/
theorem people_in_line (people_behind : ℕ) : ℕ := 
  people_behind + 1

#check people_in_line 10 -- Should evaluate to 11

end NUMINAMATH_CALUDE_people_in_line_l2772_277252


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l2772_277257

theorem multiples_of_four_between_100_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 100 ∧ n < 300) (Finset.range 300)).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l2772_277257


namespace NUMINAMATH_CALUDE_max_reflections_l2772_277241

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 12

/-- The maximum angle of incidence before reflection becomes impossible -/
def max_angle : ℝ := 90

/-- The number of reflections -/
def n : ℕ := 7

/-- Theorem stating that 7 is the maximum number of reflections possible -/
theorem max_reflections :
  (n : ℝ) * angle_between_lines ≤ max_angle ∧
  ((n + 1) : ℝ) * angle_between_lines > max_angle :=
sorry

end NUMINAMATH_CALUDE_max_reflections_l2772_277241


namespace NUMINAMATH_CALUDE_rhombus_c_coordinate_sum_l2772_277290

/-- A rhombus with vertices A, B, C, and D in 2D space -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The property that A and D are diagonally opposite in the rhombus -/
def diagonallyOpposite (r : Rhombus) : Prop :=
  (r.A.1 + r.D.1) / 2 = (r.B.1 + r.C.1) / 2 ∧
  (r.A.2 + r.D.2) / 2 = (r.B.2 + r.C.2) / 2

/-- The theorem stating that for a rhombus ABCD with given coordinates,
    the sum of coordinates of C is 9 -/
theorem rhombus_c_coordinate_sum :
  ∀ (r : Rhombus),
  r.A = (-3, -2) →
  r.B = (1, -5) →
  r.D = (9, 1) →
  diagonallyOpposite r →
  r.C.1 + r.C.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_c_coordinate_sum_l2772_277290


namespace NUMINAMATH_CALUDE_solution_set_of_f_geq_1_l2772_277209

def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

theorem solution_set_of_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_geq_1_l2772_277209


namespace NUMINAMATH_CALUDE_training_end_time_l2772_277281

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem training_end_time :
  let startTime : Time := ⟨8, 0, sorry⟩
  let sessionDuration : ℕ := 40
  let breakDuration : ℕ := 15
  let numSessions : ℕ := 4
  let totalDuration := numSessions * sessionDuration + (numSessions - 1) * breakDuration
  let endTime := addMinutes startTime totalDuration
  endTime = ⟨11, 25, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_training_end_time_l2772_277281


namespace NUMINAMATH_CALUDE_child_b_share_l2772_277279

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 5400 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 1800 := by
sorry

end NUMINAMATH_CALUDE_child_b_share_l2772_277279


namespace NUMINAMATH_CALUDE_summer_break_length_l2772_277233

/-- Represents the summer break reading scenario --/
structure SummerReading where
  deshaun_books : ℕ
  avg_pages_per_book : ℕ
  second_person_percentage : ℚ
  second_person_daily_pages : ℕ

/-- Calculates the number of days in the summer break --/
def summer_break_days (sr : SummerReading) : ℚ :=
  (sr.deshaun_books * sr.avg_pages_per_book * sr.second_person_percentage) / sr.second_person_daily_pages

/-- Theorem stating that the summer break is 80 days long --/
theorem summer_break_length (sr : SummerReading) 
  (h1 : sr.deshaun_books = 60)
  (h2 : sr.avg_pages_per_book = 320)
  (h3 : sr.second_person_percentage = 3/4)
  (h4 : sr.second_person_daily_pages = 180) :
  summer_break_days sr = 80 := by
  sorry

#eval summer_break_days { 
  deshaun_books := 60, 
  avg_pages_per_book := 320, 
  second_person_percentage := 3/4, 
  second_person_daily_pages := 180 
}

end NUMINAMATH_CALUDE_summer_break_length_l2772_277233


namespace NUMINAMATH_CALUDE_cost_of_apples_l2772_277210

/-- The cost of apples given the total cost of groceries and the costs of other items -/
theorem cost_of_apples (total cost_bananas cost_bread cost_milk : ℕ) 
  (h1 : total = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7) :
  total - (cost_bananas + cost_bread + cost_milk) = 14 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_apples_l2772_277210


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2772_277201

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2772_277201


namespace NUMINAMATH_CALUDE_product_of_roots_l2772_277253

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 24 → ∃ y : ℝ, (x + 2) * (x - 3) = 24 ∧ (y + 2) * (y - 3) = 24 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2772_277253


namespace NUMINAMATH_CALUDE_daughter_and_child_weight_l2772_277276

/-- The combined weight of a daughter and her daughter (child) given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5) * mother_weight →
  daughter_weight = 46 →
  total_weight = 130 →
  daughter_weight + child_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_daughter_and_child_weight_l2772_277276


namespace NUMINAMATH_CALUDE_smallest_dual_base_number_l2772_277200

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digit1 digit2 : ℕ), 
    n = digit1 * base + digit2 ∧
    digit1 < base ∧
    digit2 < base

/-- The smallest number representable in both base 6 and base 8 as AA and BB respectively -/
def SmallestDualBaseNumber : ℕ := 63

theorem smallest_dual_base_number :
  (BaseRepresentation SmallestDualBaseNumber 6) ∧
  (BaseRepresentation SmallestDualBaseNumber 8) ∧
  (∀ m : ℕ, m < SmallestDualBaseNumber →
    ¬(BaseRepresentation m 6 ∧ BaseRepresentation m 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_number_l2772_277200


namespace NUMINAMATH_CALUDE_janet_pages_per_day_l2772_277278

/-- Prove that Janet reads 80 pages a day given the conditions -/
theorem janet_pages_per_day :
  let belinda_pages_per_day : ℕ := 30
  let weeks : ℕ := 6
  let days_per_week : ℕ := 7
  let extra_pages : ℕ := 2100
  ∀ (janet_pages_per_day : ℕ),
    janet_pages_per_day * (weeks * days_per_week) = 
      belinda_pages_per_day * (weeks * days_per_week) + extra_pages →
    janet_pages_per_day = 80 := by
  sorry

end NUMINAMATH_CALUDE_janet_pages_per_day_l2772_277278


namespace NUMINAMATH_CALUDE_place_value_difference_l2772_277226

def number : ℝ := 135.21

def hundreds_place_value : ℝ := 100
def tenths_place_value : ℝ := 0.1

theorem place_value_difference : 
  hundreds_place_value - tenths_place_value = 99.9 := by
  sorry

end NUMINAMATH_CALUDE_place_value_difference_l2772_277226


namespace NUMINAMATH_CALUDE_urban_road_network_renovation_l2772_277296

theorem urban_road_network_renovation (a m k : ℝ) (ha : a > 0) (hm : m > 0) (hk : k ≥ 3) :
  let x := 0.2 * a
  let P := (m * x) / (m * k * (a * x + 5))
  P ≤ 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_urban_road_network_renovation_l2772_277296


namespace NUMINAMATH_CALUDE_bridge_length_l2772_277247

/-- The length of a bridge given specific train parameters -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2772_277247


namespace NUMINAMATH_CALUDE_removal_gives_desired_average_l2772_277299

def original_list : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def removed_number : ℕ := 5
def desired_average : ℚ := 10.5

theorem removal_gives_desired_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = desired_average := by
  sorry

end NUMINAMATH_CALUDE_removal_gives_desired_average_l2772_277299


namespace NUMINAMATH_CALUDE_power_equation_solution_l2772_277269

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2772_277269


namespace NUMINAMATH_CALUDE_arithmetic_heptagon_angle_l2772_277238

/-- Represents a heptagon with angles in arithmetic progression -/
structure ArithmeticHeptagon where
  -- First angle of the progression
  a : ℝ
  -- Common difference of the progression
  d : ℝ
  -- Constraint: Sum of angles in a heptagon is 900°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) + (a + 6*d) = 900

/-- Theorem: In a heptagon with angles in arithmetic progression, one of the angles can be 128.57° -/
theorem arithmetic_heptagon_angle (h : ArithmeticHeptagon) : 
  ∃ k : Fin 7, h.a + k * h.d = 128.57 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_heptagon_angle_l2772_277238


namespace NUMINAMATH_CALUDE_main_theorem_l2772_277228

/-- Definition of H function -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ + f x₂) / 2 > f ((x₁ + x₂) / 2)

/-- Definition of even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The main theorem -/
theorem main_theorem (c : ℝ) (f : ℝ → ℝ) (h : f = fun x ↦ x^2 + c*x) 
  (h_even : is_even f) : c = 0 ∧ is_H_function f := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l2772_277228


namespace NUMINAMATH_CALUDE_candy_distribution_l2772_277242

/-- The number of candy pieces in the bag -/
def total_candy : ℕ := 120

/-- Predicate to check if a number is a valid count of students -/
def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (total_candy - 1) % n = 0

/-- The theorem stating the possible number of students -/
theorem candy_distribution :
  ∃ (n : ℕ), is_valid_student_count n ∧ (n = 7 ∨ n = 17) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2772_277242


namespace NUMINAMATH_CALUDE_first_15_prime_sums_l2772_277297

/-- Returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Returns the sum of the first n prime numbers -/
def sumOfFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The number of prime sums among the first 15 sums of consecutive primes -/
def numPrimeSums : ℕ := sorry

theorem first_15_prime_sums : 
  numPrimeSums = 6 := by sorry

end NUMINAMATH_CALUDE_first_15_prime_sums_l2772_277297


namespace NUMINAMATH_CALUDE_inequality_proof_l2772_277245

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x < 1) (hy1 : y < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2772_277245


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l2772_277280

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 4th term of the sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 6th term of the sequence -/
def a_6 (a : ℕ → ℝ) : ℝ := a 6

/-- The 8th term of the sequence -/
def a_8 (a : ℕ → ℝ) : ℝ := a 8

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a_4 a)^2 - 3*(a_4 a) + 2 = 0 →
  (a_8 a)^2 - 3*(a_8 a) + 2 = 0 →
  a_6 a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l2772_277280


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2772_277267

theorem decimal_equivalent_of_one_fifth_squared : (1 / 5 : ℚ) ^ 2 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2772_277267


namespace NUMINAMATH_CALUDE_attendance_difference_l2772_277220

def football_game_attendance (saturday monday wednesday friday thursday expected_total : ℕ) : Prop :=
  let total := saturday + monday + wednesday + friday + thursday
  saturday = 80 ∧
  monday = saturday - 20 ∧
  wednesday = monday + 50 ∧
  friday = saturday + monday ∧
  thursday = 45 ∧
  expected_total = 350 ∧
  total - expected_total = 85

theorem attendance_difference :
  ∃ (saturday monday wednesday friday thursday expected_total : ℕ),
    football_game_attendance saturday monday wednesday friday thursday expected_total :=
by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l2772_277220


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_purchase_cost_l2772_277232

/-- The cost of purchasing sandwiches and sodas at Joe's Fast Food -/
theorem joes_fast_food_cost : ℕ → ℕ → ℕ
  | sandwich_count, soda_count => 
    4 * sandwich_count + 3 * soda_count

/-- Proof that purchasing 7 sandwiches and 9 sodas costs $55 -/
theorem purchase_cost : joes_fast_food_cost 7 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_purchase_cost_l2772_277232


namespace NUMINAMATH_CALUDE_steve_shared_oranges_l2772_277204

/-- The number of oranges Steve shared with Patrick -/
def oranges_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem steve_shared_oranges :
  oranges_shared 46 42 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steve_shared_oranges_l2772_277204


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2772_277219

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2772_277219


namespace NUMINAMATH_CALUDE_yanni_toy_cost_l2772_277202

/-- The cost of the toy Yanni bought -/
def toy_cost (initial_money mother_gift found_money money_left : ℚ) : ℚ :=
  initial_money + mother_gift + found_money - money_left

/-- Theorem stating the cost of the toy Yanni bought -/
theorem yanni_toy_cost :
  toy_cost 0.85 0.40 0.50 0.15 = 1.60 := by
  sorry

end NUMINAMATH_CALUDE_yanni_toy_cost_l2772_277202


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l2772_277270

/-- An increasing geometric sequence with specific conditions -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ q > 1, ∀ n, a (n + 1) = q * a n
  sum_condition : a 1 + a 4 = 9
  product_condition : a 2 * a 3 = 8

/-- The common ratio of the special geometric sequence is 2 -/
theorem special_geometric_sequence_ratio (seq : SpecialGeometricSequence) :
  ∃ q > 1, (∀ n, seq.a (n + 1) = q * seq.a n) ∧ q = 2 := by sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l2772_277270
