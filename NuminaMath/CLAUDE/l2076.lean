import Mathlib

namespace NUMINAMATH_CALUDE_snail_meets_minute_hand_l2076_207661

/-- Represents the position on a clock face in minutes past 12 -/
def ClockPosition := ℕ

/-- Calculates the position of the snail at a given time -/
def snail_position (time : ℕ) : ClockPosition :=
  (3 * time) % 60

/-- Calculates the position of the minute hand at a given time -/
def minute_hand_position (time : ℕ) : ClockPosition :=
  time % 60

/-- Checks if the snail and minute hand meet at a given time -/
def meets_at (time : ℕ) : Prop :=
  snail_position time = minute_hand_position time

theorem snail_meets_minute_hand :
  meets_at 40 ∧ meets_at 80 :=
sorry

end NUMINAMATH_CALUDE_snail_meets_minute_hand_l2076_207661


namespace NUMINAMATH_CALUDE_ken_summit_time_l2076_207666

/-- Represents the climbing scenario of Sari and Ken -/
structure ClimbingScenario where
  sari_start_time : ℕ  -- in hours after midnight
  ken_start_time : ℕ   -- in hours after midnight
  initial_distance : ℝ  -- distance Sari is ahead when Ken starts
  ken_pace : ℝ          -- Ken's climbing pace in meters per hour
  final_distance : ℝ    -- distance Sari is behind when Ken reaches summit

/-- The time it takes Ken to reach the summit -/
def time_to_summit (scenario : ClimbingScenario) : ℝ :=
  sorry

/-- Theorem stating that Ken reaches the summit 5 hours after starting -/
theorem ken_summit_time (scenario : ClimbingScenario) 
  (h1 : scenario.sari_start_time = 8)
  (h2 : scenario.ken_start_time = 10)
  (h3 : scenario.initial_distance = 700)
  (h4 : scenario.ken_pace = 500)
  (h5 : scenario.final_distance = 50) :
  time_to_summit scenario = 5 :=
sorry

end NUMINAMATH_CALUDE_ken_summit_time_l2076_207666


namespace NUMINAMATH_CALUDE_number_transformation_l2076_207653

theorem number_transformation (x : ℝ) : (x * (5/6) / 10 + 2/3) = 3/4 * x + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l2076_207653


namespace NUMINAMATH_CALUDE_robert_claire_photo_difference_l2076_207642

/-- 
Given that:
- Lisa and Robert have taken the same number of photos
- Lisa has taken 3 times as many photos as Claire
- Claire has taken 6 photos

Prove that Robert has taken 12 more photos than Claire.
-/
theorem robert_claire_photo_difference : 
  ∀ (lisa robert claire : ℕ),
  robert = lisa →
  lisa = 3 * claire →
  claire = 6 →
  robert - claire = 12 :=
by sorry

end NUMINAMATH_CALUDE_robert_claire_photo_difference_l2076_207642


namespace NUMINAMATH_CALUDE_f_max_value_l2076_207611

def f (x : ℝ) := x^3 - x^2 - x + 2

theorem f_max_value :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) →
  (∃ x, f x = 59/27 ∧ ∀ y, f y ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l2076_207611


namespace NUMINAMATH_CALUDE_max_value_of_f_l2076_207640

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f c) ∧
  f c = (1 : ℝ) / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2076_207640


namespace NUMINAMATH_CALUDE_log_equality_implies_value_l2076_207626

theorem log_equality_implies_value (p q : ℝ) (c : ℝ) (h : 0 < p ∧ 0 < q ∧ 0 < 5) :
  Real.log p / Real.log 5 = c - Real.log q / Real.log 5 → p = 5^c / q := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_value_l2076_207626


namespace NUMINAMATH_CALUDE_selma_has_fifty_marbles_l2076_207656

-- Define the number of marbles each person has
def merill_marbles : ℕ := 30
def elliot_marbles : ℕ := merill_marbles / 2
def selma_marbles : ℕ := merill_marbles + elliot_marbles + 5

-- State the theorem
theorem selma_has_fifty_marbles :
  selma_marbles = 50 := by sorry

end NUMINAMATH_CALUDE_selma_has_fifty_marbles_l2076_207656


namespace NUMINAMATH_CALUDE_probability_no_university_in_further_analysis_l2076_207671

/-- Represents the types of schools in the region -/
inductive SchoolType
  | Elementary
  | Middle
  | University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
  | SchoolType.Elementary => 21
  | SchoolType.Middle => 14
  | SchoolType.University => 7

/-- The total number of schools in the region -/
def totalAllSchools : Nat := 
  totalSchools SchoolType.Elementary + 
  totalSchools SchoolType.Middle + 
  totalSchools SchoolType.University

/-- The number of schools selected in the stratified sample -/
def sampleSize : Nat := 6

/-- The number of schools of each type in the stratified sample -/
def stratifiedSample : SchoolType → Nat
  | SchoolType.Elementary => 3
  | SchoolType.Middle => 2
  | SchoolType.University => 1

/-- The number of schools selected for further analysis -/
def furtherAnalysisSize : Nat := 2

theorem probability_no_university_in_further_analysis : 
  (Nat.choose (stratifiedSample SchoolType.Elementary + stratifiedSample SchoolType.Middle) furtherAnalysisSize : ℚ) / 
  (Nat.choose sampleSize furtherAnalysisSize : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_university_in_further_analysis_l2076_207671


namespace NUMINAMATH_CALUDE_product_units_digit_base8_l2076_207635

theorem product_units_digit_base8 : ∃ (n : ℕ), 
  (505 * 71) % 8 = n ∧ n = ((505 % 8) * (71 % 8)) % 8 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_base8_l2076_207635


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2076_207670

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the gain percent after discount
def gain_percent : ℝ := 0.171875

-- Define the relationship between cost price and marked price
theorem cost_price_percentage (marked_price cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : marked_price * (1 - discount_rate) = cost_price * (1 + gain_percent)) : 
  cost_price / marked_price = 0.64 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_percentage_l2076_207670


namespace NUMINAMATH_CALUDE_red_tint_percentage_l2076_207619

/-- Given a paint mixture, calculate the percentage of red tint after adding more red tint -/
theorem red_tint_percentage (original_volume : ℝ) (original_red_percent : ℝ) (added_red_volume : ℝ) :
  original_volume = 40 →
  original_red_percent = 20 →
  added_red_volume = 10 →
  let original_red_volume := original_red_percent / 100 * original_volume
  let new_red_volume := original_red_volume + added_red_volume
  let new_total_volume := original_volume + added_red_volume
  (new_red_volume / new_total_volume) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_red_tint_percentage_l2076_207619


namespace NUMINAMATH_CALUDE_function_correspondence_l2076_207684

-- Case 1
def A1 : Set ℕ := {1, 2, 3}
def B1 : Set ℕ := {7, 8, 9}
def f1 : ℕ → ℕ
  | 1 => 7
  | 2 => 7
  | 3 => 8
  | _ => 0  -- default case for completeness

-- Case 2
def A2 : Set ℕ := {1, 2, 3}
def B2 : Set ℕ := {1, 2, 3}
def f2 : ℕ → ℕ
  | x => 2 * x - 1

-- Case 3
def A3 : Set ℝ := {x : ℝ | x ≥ -1}
def B3 : Set ℝ := {x : ℝ | x ≥ -1}
def f3 : ℝ → ℝ
  | x => 2 * x + 1

-- Case 4
def A4 : Set ℤ := Set.univ
def B4 : Set ℤ := {-1, 1}
def f4 : ℤ → ℤ
  | n => if n % 2 = 0 then 1 else -1

theorem function_correspondence :
  (∀ x ∈ A1, f1 x ∈ B1) ∧
  (¬∀ x ∈ A2, f2 x ∈ B2) ∧
  (∀ x ∈ A3, f3 x ∈ B3) ∧
  (∀ x ∈ A4, f4 x ∈ B4) :=
by sorry

end NUMINAMATH_CALUDE_function_correspondence_l2076_207684


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2076_207691

theorem triangle_angle_proof (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- angles are positive
  A + B + C = π ∧ -- sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- sides are positive
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B → -- given condition
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2076_207691


namespace NUMINAMATH_CALUDE_inequality_solution_l2076_207643

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 1 / (x - 2) + 3 / 4) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2076_207643


namespace NUMINAMATH_CALUDE_sum_is_negative_l2076_207633

theorem sum_is_negative (x y : ℝ) (hx : x > 0) (hy : y < 0) (hxy : |x| < |y|) : x + y < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_negative_l2076_207633


namespace NUMINAMATH_CALUDE_square_of_binomial_l2076_207685

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16*x^2 + 40*x + a = (4*x + b)^2) → a = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2076_207685


namespace NUMINAMATH_CALUDE_sum_in_base8_l2076_207610

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 :
  let a := 53
  let b := 27
  let sum := base10_to_base8 (base8_to_base10 a + base8_to_base10 b)
  sum = 102 := by sorry

end NUMINAMATH_CALUDE_sum_in_base8_l2076_207610


namespace NUMINAMATH_CALUDE_train_crossing_time_l2076_207672

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300) 
  (h2 : signal_pole_time = 18) 
  (h3 : platform_length = 600.0000000000001) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 54.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2076_207672


namespace NUMINAMATH_CALUDE_zero_occurrences_in_900_pages_l2076_207657

/-- Count the occurrences of '0' in page numbers from 1 to n -/
def countZeros (n : ℕ) : ℕ :=
  sorry

theorem zero_occurrences_in_900_pages :
  countZeros 900 = 172 :=
sorry

end NUMINAMATH_CALUDE_zero_occurrences_in_900_pages_l2076_207657


namespace NUMINAMATH_CALUDE_corresponding_angles_random_l2076_207659

-- Define the concept of an event
def Event : Type := Unit

-- Define the concept of a random event
def RandomEvent (e : Event) : Prop := sorry

-- Define the given events
def sunRisesWest : Event := sorry
def triangleAngleSum : Event := sorry
def correspondingAngles : Event := sorry
def drawRedBall : Event := sorry

-- State the theorem
theorem corresponding_angles_random : RandomEvent correspondingAngles := by sorry

end NUMINAMATH_CALUDE_corresponding_angles_random_l2076_207659


namespace NUMINAMATH_CALUDE_sum_bound_l2076_207687

theorem sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  a + b > 9 ∧ ∀ ε > 0, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 1 ∧ a' + b' < 9 + ε :=
sorry

end NUMINAMATH_CALUDE_sum_bound_l2076_207687


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2076_207665

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2076_207665


namespace NUMINAMATH_CALUDE_A_suff_not_nec_D_l2076_207607

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_suff_not_nec_D_l2076_207607


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2076_207612

/-- Given a cylinder with base area 4π and a lateral surface that unfolds into a square,
    prove that its lateral surface area is 16π. -/
theorem cylinder_lateral_surface_area (r h : ℝ) : 
  (π * r^2 = 4 * π) →  -- base area condition
  (2 * π * r = h) →    -- lateral surface unfolds into a square condition
  (2 * π * r * h = 16 * π) := by 
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2076_207612


namespace NUMINAMATH_CALUDE_max_correct_answers_15_l2076_207639

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct_answers : ℕ
  incorrect_answers : ℕ
  blank_answers : ℕ
  total_score : ℤ

/-- Calculates the total score for an exam result. -/
def calculate_score (result : ExamResult) : ℤ :=
  result.correct_answers * result.exam.correct_score +
  result.incorrect_answers * result.exam.incorrect_score

/-- Verifies if an exam result is valid. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct_answers + result.incorrect_answers + result.blank_answers = result.exam.total_questions ∧
  calculate_score result = result.total_score

/-- Theorem: The maximum number of correct answers for John's exam is 15. -/
theorem max_correct_answers_15 (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 25 ∧
  john_exam.correct_score = 6 ∧
  john_exam.incorrect_score = -3 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 60 ∧
  is_valid_result john_result →
  john_result.correct_answers ≤ 15 ∧
  ∃ (valid_result : ExamResult),
    valid_result.exam = john_exam ∧
    valid_result.total_score = 60 ∧
    is_valid_result valid_result ∧
    valid_result.correct_answers = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_15_l2076_207639


namespace NUMINAMATH_CALUDE_tuesday_wednesday_thursday_avg_l2076_207646

def tuesday_temp : ℝ := 38
def friday_temp : ℝ := 44
def wed_thur_fri_avg : ℝ := 34

theorem tuesday_wednesday_thursday_avg :
  let wed_thur_sum := 3 * wed_thur_fri_avg - friday_temp
  (tuesday_temp + wed_thur_sum) / 3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_tuesday_wednesday_thursday_avg_l2076_207646


namespace NUMINAMATH_CALUDE_data_average_problem_l2076_207621

theorem data_average_problem (x : ℝ) : 
  (6 + x + 2 + 4) / 4 = 5 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_data_average_problem_l2076_207621


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2076_207677

theorem quadratic_transformation (x : ℝ) : 
  (4 * x^2 - 16 * x - 400 = 0) → 
  (∃ p q : ℝ, (x + p)^2 = q ∧ q = 104) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2076_207677


namespace NUMINAMATH_CALUDE_same_color_probability_l2076_207608

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def green_plates : ℕ := 5
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose green_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2076_207608


namespace NUMINAMATH_CALUDE_inequality_proof_l2076_207695

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) :
  a / b + b / c + c / a ≥ a^2 + b^2 + c^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2076_207695


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_l2076_207632

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- Theorem stating the condition for f(x) to be non-negative for all real x -/
theorem f_nonnegative_iff (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_l2076_207632


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2076_207655

theorem smaller_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  x = 270 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2076_207655


namespace NUMINAMATH_CALUDE_students_in_range_estimate_l2076_207613

/-- Represents a normal distribution of scores -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  isNormal : Bool

/-- Represents the student population and their score distribution -/
structure StudentPopulation where
  totalStudents : ℕ
  scoreDistribution : ScoreDistribution

/-- Calculates the number of students within a given score range -/
def studentsInRange (pop : StudentPopulation) (lowerBound upperBound : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem students_in_range_estimate 
  (pop : StudentPopulation) 
  (h1 : pop.totalStudents = 3000) 
  (h2 : pop.scoreDistribution.isNormal = true) : 
  ∃ (ε : ℕ), ε ≤ 10 ∧ 
  (studentsInRange pop 70 80 = 408 + ε ∨ studentsInRange pop 70 80 = 408 - ε) :=
sorry

end NUMINAMATH_CALUDE_students_in_range_estimate_l2076_207613


namespace NUMINAMATH_CALUDE_heart_nested_calculation_l2076_207664

def heart (a b : ℝ) : ℝ := (a + 2*b) * (a - b)

theorem heart_nested_calculation : heart 2 (heart 3 4) = -260 := by
  sorry

end NUMINAMATH_CALUDE_heart_nested_calculation_l2076_207664


namespace NUMINAMATH_CALUDE_frank_candy_count_l2076_207603

/-- Given a number of bags and pieces per bag, calculates the total number of pieces -/
def totalPieces (n m : ℕ) : ℕ := n * m

/-- Theorem: For 2 bags with 21 pieces each, the total number of pieces is 42 -/
theorem frank_candy_count : totalPieces 2 21 = 42 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l2076_207603


namespace NUMINAMATH_CALUDE_smallest_number_l2076_207634

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = 1 → b = -2 → c = 0 → d = -1/2 → 
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2076_207634


namespace NUMINAMATH_CALUDE_solve_for_T_l2076_207674

theorem solve_for_T : ∃ T : ℚ, (1/3 : ℚ) * (1/6 : ℚ) * T = (1/4 : ℚ) * (1/8 : ℚ) * 120 ∧ T = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l2076_207674


namespace NUMINAMATH_CALUDE_f_of_g_10_l2076_207644

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 262 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l2076_207644


namespace NUMINAMATH_CALUDE_max_value_of_f_l2076_207699

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the interval
def I : Set ℝ := Set.Icc (-4) 3

-- State the theorem
theorem max_value_of_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = 15 ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2076_207699


namespace NUMINAMATH_CALUDE_y_equation_implies_expression_equals_two_l2076_207647

theorem y_equation_implies_expression_equals_two (y : ℝ) (h : y + 2/y = 2) :
  y^6 + 3*y^4 - 4*y^2 + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_implies_expression_equals_two_l2076_207647


namespace NUMINAMATH_CALUDE_probability_of_two_red_books_l2076_207622

-- Define the number of red and blue books
def red_books : ℕ := 4
def blue_books : ℕ := 4
def total_books : ℕ := red_books + blue_books

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_of_two_red_books :
  (combination red_books books_selected : ℚ) / (combination total_books books_selected) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_red_books_l2076_207622


namespace NUMINAMATH_CALUDE_unique_line_count_for_p_2_l2076_207600

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for lines in a plane
def Line : Type := Point → Point → Prop

-- Define a function to count the number of intersection points
def count_intersections (lines : List Line) : ℕ := sorry

-- Define a function to check if lines intersect at exactly p points
def intersect_at_p_points (lines : List Line) (p : ℕ) : Prop :=
  count_intersections lines = p

-- Theorem: When p = 2, there is a unique number of lines (3) that intersect at exactly p points
theorem unique_line_count_for_p_2 :
  ∃! n : ℕ, ∃ lines : List Line, intersect_at_p_points lines 2 ∧ lines.length = n :=
sorry

end NUMINAMATH_CALUDE_unique_line_count_for_p_2_l2076_207600


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l2076_207615

theorem arithmetic_mean_geq_geometric_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l2076_207615


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2076_207669

theorem max_value_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2076_207669


namespace NUMINAMATH_CALUDE_sports_and_literature_enthusiasts_l2076_207628

theorem sports_and_literature_enthusiasts
  (total_students : ℕ)
  (sports_enthusiasts : ℕ)
  (literature_enthusiasts : ℕ)
  (h_total : total_students = 100)
  (h_sports : sports_enthusiasts = 60)
  (h_literature : literature_enthusiasts = 65) :
  ∃ (m n : ℕ),
    m = max sports_enthusiasts literature_enthusiasts ∧
    n = max 0 (sports_enthusiasts + literature_enthusiasts - total_students) ∧
    m + n = 85 :=
by sorry

end NUMINAMATH_CALUDE_sports_and_literature_enthusiasts_l2076_207628


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2076_207602

/-- The complex number (1-i)^2 / (1+i) lies in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2076_207602


namespace NUMINAMATH_CALUDE_complex_symmetric_product_l2076_207620

theorem complex_symmetric_product (z₁ z₂ : ℂ) :
  z₁.im = -z₂.im → z₁.re = z₂.re → z₁ = 2 - I → z₁ * z₂ = 5 := by sorry

end NUMINAMATH_CALUDE_complex_symmetric_product_l2076_207620


namespace NUMINAMATH_CALUDE_white_surface_fraction_l2076_207697

/-- Represents a cube with its properties -/
structure Cube where
  edge_length : ℕ
  total_subcubes : ℕ
  white_subcubes : ℕ
  black_subcubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the number of exposed faces of subcubes at diagonal ends -/
def exposed_diagonal_faces (c : Cube) : ℕ := 3 * c.black_subcubes

/-- Theorem: The fraction of white surface area in the given cube configuration is 1/2 -/
theorem white_surface_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_subcubes = 64)
  (h3 : c.white_subcubes = 48)
  (h4 : c.black_subcubes = 16)
  (h5 : exposed_diagonal_faces c = c.black_subcubes * 3) :
  (surface_area c - exposed_diagonal_faces c) / surface_area c = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_l2076_207697


namespace NUMINAMATH_CALUDE_function_properties_l2076_207650

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f is not always zero
def not_always_zero (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ 0

-- Define the functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)

-- Theorem statement
theorem function_properties
  (h1 : not_always_zero f)
  (h2 : satisfies_equation f) :
  f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2076_207650


namespace NUMINAMATH_CALUDE_calvins_haircuts_l2076_207698

/-- The number of haircuts Calvin has gotten so far -/
def haircuts_gotten : ℕ := 8

/-- The number of additional haircuts Calvin needs to reach his goal -/
def haircuts_needed : ℕ := 2

/-- The percentage of progress Calvin has made towards his goal -/
def progress_percentage : ℚ := 80 / 100

theorem calvins_haircuts : 
  (haircuts_gotten : ℚ) / (haircuts_gotten + haircuts_needed) = progress_percentage := by
  sorry

end NUMINAMATH_CALUDE_calvins_haircuts_l2076_207698


namespace NUMINAMATH_CALUDE_cuboid_s_value_l2076_207675

/-- Represents a cuboid with adjacent face areas a, b, and s, 
    whose vertices lie on a sphere with surface area sa -/
structure Cuboid where
  a : ℝ
  b : ℝ
  s : ℝ
  sa : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < s ∧ 0 < sa
  h_sphere : sa = 152 * Real.pi
  h_face1 : a * b = 6
  h_face2 : b * (s / b) = 10
  h_vertices_on_sphere : ∃ (r : ℝ), 
    a^2 + b^2 + (s / b)^2 = 4 * r^2 ∧ sa = 4 * Real.pi * r^2

/-- The theorem stating that for a cuboid satisfying the given conditions, s must equal 15 -/
theorem cuboid_s_value (c : Cuboid) : c.s = 15 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_s_value_l2076_207675


namespace NUMINAMATH_CALUDE_polynomial_identity_l2076_207636

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

def polynomial_equation : Prop :=
  ∀ x : ℝ, (1 + x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5

theorem polynomial_identity (h : polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅) :
  a₀ = 1 ∧ (a₀ / 1 + a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 21 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2076_207636


namespace NUMINAMATH_CALUDE_star_example_l2076_207683

-- Define the star operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_example : star (star 3 8) 6 = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l2076_207683


namespace NUMINAMATH_CALUDE_celeste_candy_theorem_l2076_207606

/-- Represents the state of candies on the table -/
structure CandyState (n : ℕ+) where
  counts : Fin n → ℕ

/-- Represents the operations that can be performed on the candy state -/
inductive Operation (n : ℕ+)
  | split : Fin n → Operation n
  | take : Fin n → Operation n

/-- Applies an operation to a candy state -/
def apply_operation {n : ℕ+} (state : CandyState n) (op : Operation n) : CandyState n :=
  sorry

/-- Checks if a candy state is empty -/
def is_empty {n : ℕ+} (state : CandyState n) : Prop :=
  ∀ i, state.counts i = 0

/-- Main theorem: Celeste can empty the table for any initial configuration
    if and only if n is not divisible by 3 -/
theorem celeste_candy_theorem (n : ℕ+) :
  (∀ (m : ℕ+) (initial_state : CandyState n),
    ∃ (ops : List (Operation n)), is_empty (ops.foldl apply_operation initial_state))
  ↔ ¬(n : ℕ) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_celeste_candy_theorem_l2076_207606


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2076_207616

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n ∧ a n > 0

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16 →
  a 3 + a 6 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2076_207616


namespace NUMINAMATH_CALUDE_line_slope_is_two_l2076_207617

/-- Given a line ax + 3my + 2a = 0 with m ≠ 0 and the sum of its intercepts on the coordinate axes is 2, prove that its slope is 2 -/
theorem line_slope_is_two (m a : ℝ) (hm : m ≠ 0) :
  (∃ (x y : ℝ), a * x + 3 * m * y + 2 * a = 0 ∧ 
   (a * 0 + 3 * m * y + 2 * a = 0 → y = -2 * a / (3 * m)) ∧
   (a * x + 3 * m * 0 + 2 * a = 0 → x = -2) ∧
   y + x = 2) →
  (∃ (k b : ℝ), ∀ x y, a * x + 3 * m * y + 2 * a = 0 ↔ y = k * x + b) ∧
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l2076_207617


namespace NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l2076_207645

theorem largest_sum_is_five_sixths :
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/7, 1/3 + 1/9]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 ∈ sums) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l2076_207645


namespace NUMINAMATH_CALUDE_tournament_ordered_victories_l2076_207652

/-- A round-robin tournament with 2^n players -/
def Tournament (n : ℕ) := Fin (2^n)

/-- The result of a match between two players -/
def Defeats (t : Tournament n) : Tournament n → Tournament n → Prop := sorry

/-- The property that player i defeats player j if and only if i < j -/
def OrderedVictories (t : Tournament n) (s : Fin (n+1) → Tournament n) : Prop :=
  ∀ i j, i < j → Defeats t (s i) (s j)

/-- The main theorem: In any tournament of 2^n players, there exists an ordered sequence of n+1 players -/
theorem tournament_ordered_victories (n : ℕ) :
  ∀ t : Tournament n, ∃ s : Fin (n+1) → Tournament n, OrderedVictories t s := by
  sorry

end NUMINAMATH_CALUDE_tournament_ordered_victories_l2076_207652


namespace NUMINAMATH_CALUDE_apples_in_basket_l2076_207605

def apples_remaining (initial : ℕ) (ricki_removes : ℕ) : ℕ :=
  initial - (ricki_removes + 2 * ricki_removes)

theorem apples_in_basket (initial : ℕ) (ricki_removes : ℕ) 
  (h1 : initial = 74) (h2 : ricki_removes = 14) : 
  apples_remaining initial ricki_removes = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l2076_207605


namespace NUMINAMATH_CALUDE_equation_solution_l2076_207662

theorem equation_solution : ∃ x : ℝ, -200 * x = 1600 ∧ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2076_207662


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2076_207638

/-- Given two points on a quadratic function, prove the value of b -/
theorem quadratic_coefficient (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -12 →
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2076_207638


namespace NUMINAMATH_CALUDE_min_value_theorem_l2076_207609

theorem min_value_theorem (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 18 ∧ (x^2 / (x - 6) = 18 ↔ x = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2076_207609


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2076_207604

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem cloth_selling_price 
  (quantity : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) :
  quantity = 85 →
  profit_per_meter = 35 →
  cost_price_per_meter = 70 →
  quantity * (profit_per_meter + cost_price_per_meter) = 8925 := by
  sorry

#check cloth_selling_price

end NUMINAMATH_CALUDE_cloth_selling_price_l2076_207604


namespace NUMINAMATH_CALUDE_fraction_of_number_l2076_207668

theorem fraction_of_number (N : ℝ) (h : N = 180) : 
  6 + (1/2) * (1/3) * (1/5) * N = (1/25) * N := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l2076_207668


namespace NUMINAMATH_CALUDE_circle_symmetry_axis_l2076_207693

theorem circle_symmetry_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 2*y₀ + 1 = 0 ∧
      m*x₀ + y₀ - 1 = 0 ∧
      ∀ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 →
        (m*x' + y' - 1 = 0 ↔ m*(2*x₀ - x') + (2*y₀ - y') - 1 = 0))) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_axis_l2076_207693


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l2076_207682

/-- The number of ways to arrange 6 students in two rows of three, 
    with the taller student in each column in the back row -/
def arrangement_count : ℕ := 90

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students in each row -/
def students_per_row : ℕ := 3

theorem student_arrangement_theorem :
  (num_students = 6) →
  (students_per_row = 3) →
  (∀ n : ℕ, n ≤ num_students → n > 0 → ∃! h : ℕ, h = n) →  -- All students have different heights
  arrangement_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l2076_207682


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l2076_207629

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of base AD
  ad : ℝ
  -- Length of base BC
  bc : ℝ
  -- Length of diagonal AC
  ac : ℝ
  -- Circles on AB, BC, CD as diameters intersect at a single point
  circles_intersect : Prop

/-- The theorem stating that under given conditions, diagonal BD has length 24 -/
theorem trapezoid_diagonal_length (t : Trapezoid) 
  (h1 : t.ad = 16)
  (h2 : t.bc = 10)
  (h3 : t.ac = 10)
  (h4 : t.circles_intersect) :
  ∃ (bd : ℝ), bd = 24 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l2076_207629


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2076_207618

def y : ℕ := 2^3 * 3^4 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_factor_for_perfect_square :
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬ ∃ k : ℕ, m * y = k^2) ∧
  ∃ k : ℕ, 2 * y = k^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2076_207618


namespace NUMINAMATH_CALUDE_distance_to_sea_world_l2076_207630

/-- Calculates the distance to Sea World based on given conditions --/
theorem distance_to_sea_world 
  (savings : ℕ) 
  (parking_cost : ℕ) 
  (entrance_cost : ℕ) 
  (meal_pass_cost : ℕ) 
  (car_efficiency : ℕ) 
  (gas_price : ℕ) 
  (additional_savings_needed : ℕ) 
  (h1 : savings = 28)
  (h2 : parking_cost = 10)
  (h3 : entrance_cost = 55)
  (h4 : meal_pass_cost = 25)
  (h5 : car_efficiency = 30)
  (h6 : gas_price = 3)
  (h7 : additional_savings_needed = 95)
  : ℕ := by
  sorry

#check distance_to_sea_world

end NUMINAMATH_CALUDE_distance_to_sea_world_l2076_207630


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2076_207679

theorem complex_fraction_equals_i : (3 + 2*I) / (2 - 3*I) = I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2076_207679


namespace NUMINAMATH_CALUDE_angle_from_coordinates_l2076_207654

theorem angle_from_coordinates (α : Real) 
  (h1 : α > 0) (h2 : α < 2 * Real.pi)
  (h3 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ 
                        y = Real.cos (5 * Real.pi / 6) ∧
                        x = Real.sin α ∧
                        y = Real.cos α) :
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_from_coordinates_l2076_207654


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l2076_207637

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degrees :
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l2076_207637


namespace NUMINAMATH_CALUDE_base_eight_1423_equals_787_l2076_207601

/-- Converts a base-8 digit to its base-10 equivalent -/
def baseEightDigitToBaseTen (d : Nat) : Nat :=
  if d < 8 then d else 0

/-- Converts a four-digit base-8 number to base-10 -/
def baseEightToBaseTen (a b c d : Nat) : Nat :=
  (baseEightDigitToBaseTen a) * 512 + 
  (baseEightDigitToBaseTen b) * 64 + 
  (baseEightDigitToBaseTen c) * 8 + 
  (baseEightDigitToBaseTen d)

theorem base_eight_1423_equals_787 : 
  baseEightToBaseTen 1 4 2 3 = 787 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1423_equals_787_l2076_207601


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2076_207696

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2076_207696


namespace NUMINAMATH_CALUDE_train_speed_l2076_207627

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 400.032 →
  time = 9 →
  speed = (length / 1000) / time * 3600 →
  speed = 160.0128 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2076_207627


namespace NUMINAMATH_CALUDE_average_first_5_subjects_l2076_207678

-- Define the given conditions
def total_subjects : ℕ := 6
def average_6_subjects : ℚ := 77
def marks_6th_subject : ℕ := 92

-- Define the theorem to prove
theorem average_first_5_subjects :
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  (marks_5_subjects / (total_subjects - 1) : ℚ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_first_5_subjects_l2076_207678


namespace NUMINAMATH_CALUDE_range_of_function_l2076_207667

theorem range_of_function (x : ℝ) (h : x ≥ -1) :
  let y := (12 * Real.sqrt (x + 1)) / (3 * x + 4)
  0 ≤ y ∧ y ≤ 2 * Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_range_of_function_l2076_207667


namespace NUMINAMATH_CALUDE_factorial_division_l2076_207686

theorem factorial_division : 
  (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2076_207686


namespace NUMINAMATH_CALUDE_altitude_sum_less_than_perimeter_l2076_207649

/-- For any triangle, the sum of its altitudes is less than its perimeter -/
theorem altitude_sum_less_than_perimeter (a b c h_a h_b h_c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)
  (altitude_a : h_a ≤ b ∧ h_a ≤ c)
  (altitude_b : h_b ≤ a ∧ h_b ≤ c)
  (altitude_c : h_c ≤ a ∧ h_c ≤ b)
  (non_degenerate : h_a < b ∨ h_a < c ∨ h_b < a ∨ h_b < c ∨ h_c < a ∨ h_c < b) :
  h_a + h_b + h_c < a + b + c := by
  sorry

end NUMINAMATH_CALUDE_altitude_sum_less_than_perimeter_l2076_207649


namespace NUMINAMATH_CALUDE_sum_seven_is_thirtyfive_l2076_207676

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_property : a 2 + a 10 = 16
  eighth_term : a 8 = 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The main theorem to prove -/
theorem sum_seven_is_thirtyfive (seq : ArithmeticSequence) : 
  sum_n seq 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_is_thirtyfive_l2076_207676


namespace NUMINAMATH_CALUDE_division_problem_l2076_207623

theorem division_problem :
  let dividend : ℕ := 16698
  let divisor : ℝ := 187.46067415730337
  let quotient : ℕ := 89
  let remainder : ℕ := 14
  (dividend : ℝ) = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2076_207623


namespace NUMINAMATH_CALUDE_cuboid_face_area_l2076_207692

theorem cuboid_face_area (small_face_area : ℝ) 
  (h1 : small_face_area > 0)
  (h2 : ∃ (large_face_area : ℝ), large_face_area = 4 * small_face_area)
  (h3 : 2 * small_face_area + 4 * (4 * small_face_area) = 72) :
  ∃ (large_face_area : ℝ), large_face_area = 16 := by
sorry

end NUMINAMATH_CALUDE_cuboid_face_area_l2076_207692


namespace NUMINAMATH_CALUDE_slope_of_line_l2076_207625

theorem slope_of_line (x y : ℝ) :
  4 * y = -6 * x + 12 → (y - 3 = -3/2 * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2076_207625


namespace NUMINAMATH_CALUDE_max_value_a_plus_2b_l2076_207694

theorem max_value_a_plus_2b (a b : ℝ) (h : a^2 + 2*b^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 1 → x + 2*y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_a_plus_2b_l2076_207694


namespace NUMINAMATH_CALUDE_smallest_k_for_convergence_l2076_207660

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^3

def L : ℚ := 1/3

theorem smallest_k_for_convergence :
  ∀ k : ℕ, k ≥ 1 → |u k - L| ≤ 1 / 3^300 ∧
  ∀ m : ℕ, m < k → |u m - L| > 1 / 3^300 →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_convergence_l2076_207660


namespace NUMINAMATH_CALUDE_salary_percentage_decrease_l2076_207651

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_percentage_decrease 
  (initial_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : initial_salary = 6000)
  (h2 : increase_percentage = 10)
  (h3 : final_salary = 6270) :
  let increased_salary := initial_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_salary_percentage_decrease_l2076_207651


namespace NUMINAMATH_CALUDE_cat_catches_24_birds_l2076_207658

def birds_problem (day_catch : ℕ) (night_multiplier : ℕ) : Prop :=
  let night_catch := night_multiplier * day_catch
  let total_catch := day_catch + night_catch
  total_catch = 24

theorem cat_catches_24_birds : birds_problem 8 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_catches_24_birds_l2076_207658


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_l2076_207641

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^8 : ℚ) / (2^5 * 10^6) = (n : ℚ) / 10^11 ∧ 
  0 < n ∧ 
  n < 10^11 := by
sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_l2076_207641


namespace NUMINAMATH_CALUDE_twentyFifthInBase6_l2076_207624

/-- Converts a natural number to its representation in base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem twentyFifthInBase6 : fromBase6 [4, 1] = 25 := by
  sorry

#eval toBase6 25  -- Should output [4, 1]
#eval fromBase6 [4, 1]  -- Should output 25

end NUMINAMATH_CALUDE_twentyFifthInBase6_l2076_207624


namespace NUMINAMATH_CALUDE_smallest_b_value_l2076_207663

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 - b^3) / (a - b)) (a * b) = 4) : 
  b = 2 ∧ ∀ (c : ℕ+), c < b → ¬(∃ (d : ℕ+), d - c = 4 ∧ 
    Nat.gcd ((d^3 - c^3) / (d - c)) (d * c) = 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2076_207663


namespace NUMINAMATH_CALUDE_blue_notes_scattered_l2076_207648

def red_rows : ℕ := 5
def red_notes_per_row : ℕ := 6
def blue_notes_under_each_red : ℕ := 2
def total_notes : ℕ := 100

theorem blue_notes_scattered (red_rows : ℕ) (red_notes_per_row : ℕ) (blue_notes_under_each_red : ℕ) (total_notes : ℕ) :
  red_rows = 5 →
  red_notes_per_row = 6 →
  blue_notes_under_each_red = 2 →
  total_notes = 100 →
  total_notes - (red_rows * red_notes_per_row + red_rows * red_notes_per_row * blue_notes_under_each_red) = 10 :=
by sorry

end NUMINAMATH_CALUDE_blue_notes_scattered_l2076_207648


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l2076_207631

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def rhombus_diagonals (r : Rhombus) : ℝ × ℝ := sorry

/-- Theorem: The diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ∃ r : Rhombus, (rhombus_diagonals r).1 ≠ (rhombus_diagonals r).2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l2076_207631


namespace NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l2076_207673

theorem negation_of_divisible_by_two_is_even :
  ¬(∀ n : ℤ, 2 ∣ n → Even n) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬Even n :=
by sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l2076_207673


namespace NUMINAMATH_CALUDE_meet_twice_l2076_207688

/-- Represents the meeting scenario between Michael and the garbage truck -/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly twice -/
theorem meet_twice (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 6)
  (h2 : scenario.truck_speed = 12)
  (h3 : scenario.pail_distance = 240)
  (h4 : scenario.truck_stop_time = 40)
  (h5 : scenario.initial_distance = 240) :
  number_of_meetings scenario = 2 :=
sorry

end NUMINAMATH_CALUDE_meet_twice_l2076_207688


namespace NUMINAMATH_CALUDE_trigonometric_inequality_and_supremum_l2076_207614

theorem trigonometric_inequality_and_supremum 
  (x y z : ℝ) (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (Real.sin x)^m * (Real.cos y)^n + 
  (Real.sin y)^m * (Real.cos z)^n + 
  (Real.sin z)^m * (Real.cos x)^n ≤ 1 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), 
    (Real.sin x₀)^m * (Real.cos y₀)^n + 
    (Real.sin y₀)^m * (Real.cos z₀)^n + 
    (Real.sin z₀)^m * (Real.cos x₀)^n = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_and_supremum_l2076_207614


namespace NUMINAMATH_CALUDE_smallest_factor_correct_l2076_207690

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_factor : ℕ := 1848

/-- Theorem stating that the smallest_factor is correct -/
theorem smallest_factor_correct :
  (∀ k : ℕ, k > 0 → 112 ∣ (k * 43 * 62 * 1311) → 33 ∣ (k * 43 * 62 * 1311) → k ≥ smallest_factor) ∧
  (112 ∣ (smallest_factor * 43 * 62 * 1311)) ∧
  (33 ∣ (smallest_factor * 43 * 62 * 1311)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_correct_l2076_207690


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2076_207681

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 5 * x^3 - 10 * x^2 + x - 24
  ∃ (r₁ r₂ r₃ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2076_207681


namespace NUMINAMATH_CALUDE_expression_evaluation_l2076_207680

theorem expression_evaluation : -(-2) + 2 * Real.cos (60 * π / 180) + (-1/8)⁻¹ + (Real.pi - 3.14)^0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2076_207680


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_four_l2076_207689

/-- Given that the mean of 8, 15, and 21 is equal to the mean of 16, 24, and y, prove that y = 4 -/
theorem mean_equality_implies_y_equals_four :
  (((8 + 15 + 21) / 3) = ((16 + 24 + y) / 3)) → y = 4 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_four_l2076_207689
