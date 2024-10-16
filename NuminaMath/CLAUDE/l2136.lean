import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_parallel_sides_distance_l2136_213622

/-- The distance between parallel sides of a rhombus given its diagonals -/
theorem rhombus_parallel_sides_distance (AC BD : ℝ) (h1 : AC = 3) (h2 : BD = 4) :
  let area := (1 / 2) * AC * BD
  let side := Real.sqrt ((AC / 2)^2 + (BD / 2)^2)
  area / side = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_rhombus_parallel_sides_distance_l2136_213622


namespace NUMINAMATH_CALUDE_orange_ribbons_l2136_213688

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 12 →
  black = 40 →
  yellow + purple + orange + black = total →
  orange = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l2136_213688


namespace NUMINAMATH_CALUDE_original_number_equation_l2136_213607

/-- Given a number x, prove that when it's doubled, 15 is added, and the result is trebled, it equals 75 -/
theorem original_number_equation (x : ℝ) : 3 * (2 * x + 15) = 75 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_equation_l2136_213607


namespace NUMINAMATH_CALUDE_square_k_ascending_range_l2136_213606

/-- A function f is k-ascending on a set M if for all x in M, f(x + k) ≥ f(x) --/
def IsKAscending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem square_k_ascending_range {k : ℝ} (hk : k ≠ 0) :
  IsKAscending (fun x ↦ x^2) k (Set.Ioi (-1)) → k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_k_ascending_range_l2136_213606


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l2136_213621

theorem smallest_n_with_conditions : ∃ (n : ℕ), 
  (n = 46656) ∧ 
  (∀ m : ℕ, m < n → 
    (36 ∣ m ∧ ∃ k : ℕ, m^2 = k^3 ∧ ∃ l : ℕ, m^3 = l^2) → False) ∧
  (36 ∣ n) ∧ 
  (∃ k : ℕ, n^2 = k^3) ∧ 
  (∃ l : ℕ, n^3 = l^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l2136_213621


namespace NUMINAMATH_CALUDE_parallelepiped_volume_theorem_l2136_213624

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  -- Side lengths of the base parallelogram
  side1 : ℝ
  side2 : ℝ
  -- Acute angle of the base parallelogram in radians
  angle : ℝ
  -- Length of the longest diagonal
  diagonal : ℝ

/-- Calculate the volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  -- This function will be defined in the proof
  sorry

/-- The main theorem to prove -/
theorem parallelepiped_volume_theorem (p : RightParallelepiped) 
  (h1 : p.side1 = 1)
  (h2 : p.side2 = 4)
  (h3 : p.angle = π / 3)  -- 60 degrees in radians
  (h4 : p.diagonal = 5) :
  volume p = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_theorem_l2136_213624


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2136_213686

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x + 15 = 51 ↔ x = p ∨ x = q) →
  p ≥ q →
  3*p + 2*q = 15 + 3*Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2136_213686


namespace NUMINAMATH_CALUDE_odd_painted_faces_6_4_2_l2136_213699

/-- Represents a 3D rectangular block of cubes -/
structure Block :=
  (length : Nat) (width : Nat) (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a block -/
def oddPaintedFaces (b : Block) : Nat :=
  sorry

/-- The main theorem: In a 6x4x2 block, 16 cubes have an odd number of painted faces -/
theorem odd_painted_faces_6_4_2 : 
  oddPaintedFaces (Block.mk 6 4 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_6_4_2_l2136_213699


namespace NUMINAMATH_CALUDE_unique_starting_number_l2136_213681

def operation (n : ℕ) : ℕ :=
  if n % 3 = 0 then n / 3 else n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => iterate_operation (operation n) k

theorem unique_starting_number : 
  ∃! n : ℕ, iterate_operation n 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_starting_number_l2136_213681


namespace NUMINAMATH_CALUDE_sum_zero_not_all_negative_l2136_213603

theorem sum_zero_not_all_negative (a b c : ℝ) (h : a + b + c = 0) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_not_all_negative_l2136_213603


namespace NUMINAMATH_CALUDE_sam_coupons_l2136_213684

/-- Calculates the number of coupons Sam used when buying tuna cans. -/
def calculate_coupons (num_cans : ℕ) (can_cost : ℕ) (coupon_value : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let total_cost := num_cans * can_cost
  let savings := total_cost - total_spent
  savings / coupon_value

/-- Proves that Sam had 5 coupons given the problem conditions. -/
theorem sam_coupons :
  calculate_coupons 9 175 25 2000 550 = 5 := by
  sorry

#eval calculate_coupons 9 175 25 2000 550

end NUMINAMATH_CALUDE_sam_coupons_l2136_213684


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2136_213661

theorem log_sum_equals_two (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2136_213661


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2136_213611

/-- A hexagon ABCDEF with specific side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon ABCDEF is 13 -/
theorem hexagon_perimeter :
  ∃ (h : Hexagon),
    h.AB = 2 ∧ h.BC = 2 ∧ h.CD = 2 ∧ h.DE = 2 ∧ h.EF = 2 ∧ h.FA = 3 ∧
    perimeter h = 13 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2136_213611


namespace NUMINAMATH_CALUDE_student_club_distribution_l2136_213695

-- Define the number of students and clubs
def num_students : ℕ := 5
def num_clubs : ℕ := 3

-- Define a function to calculate the number of ways to distribute students into clubs
def distribute_students (students : ℕ) (clubs : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem student_club_distribution :
  distribute_students num_students num_clubs = 150 :=
sorry

end NUMINAMATH_CALUDE_student_club_distribution_l2136_213695


namespace NUMINAMATH_CALUDE_orange_groups_count_l2136_213618

/-- The number of oranges in Philip's collection -/
def total_oranges : ℕ := 356

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := total_oranges / oranges_per_group

theorem orange_groups_count : orange_groups = 178 := by
  sorry

end NUMINAMATH_CALUDE_orange_groups_count_l2136_213618


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2136_213664

-- Define set A as a subset of real numbers
variable (A : Set ℝ)

-- Define proposition p
def p (A : Set ℝ) : Prop :=
  ∃ x ∈ A, x^2 - 2*x - 3 < 0

-- Define proposition q
def q (A : Set ℝ) : Prop :=
  ∀ x ∈ A, x^2 - 2*x - 3 < 0

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ A : Set ℝ, q A → p A) ∧ (∃ A : Set ℝ, p A ∧ ¬q A) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2136_213664


namespace NUMINAMATH_CALUDE_right_triangle_side_c_l2136_213666

theorem right_triangle_side_c (a b c : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_c_l2136_213666


namespace NUMINAMATH_CALUDE_students_favoring_both_issues_l2136_213612

/-- The number of students who voted in favor of both issues in a school referendum -/
theorem students_favoring_both_issues 
  (total_students : ℕ) 
  (favor_first : ℕ) 
  (favor_second : ℕ) 
  (against_both : ℕ) 
  (h1 : total_students = 215)
  (h2 : favor_first = 160)
  (h3 : favor_second = 132)
  (h4 : against_both = 40) : 
  favor_first + favor_second - (total_students - against_both) = 117 :=
by sorry

end NUMINAMATH_CALUDE_students_favoring_both_issues_l2136_213612


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2136_213619

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) :
  selling_price = 0.88 * marked_price →
  selling_price = 1.375 * cost_price →
  cost_price / marked_price = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2136_213619


namespace NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l2136_213617

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The number of whole boxes Nathan consumed -/
def boxes_consumed : ℕ := 4

/-- The total number of gumballs Nathan ate -/
def gumballs_eaten : ℕ := gumballs_per_package * boxes_consumed

theorem nathan_ate_twenty_gumballs : gumballs_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l2136_213617


namespace NUMINAMATH_CALUDE_correct_grooming_time_l2136_213655

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  nailClipTime : ℕ  -- Time to clip one nail in seconds
  earCleanTime : ℕ  -- Time to clean one ear in seconds
  totalTime : ℕ     -- Total grooming time in seconds

/-- Calculates the total grooming time for a cat -/
def calculateGroomingTime (gt : GroomingTime) (numClaws numFeet numEars : ℕ) : ℕ :=
  (gt.nailClipTime * numClaws * numFeet) + (gt.earCleanTime * numEars) + 
  (gt.totalTime - (gt.nailClipTime * numClaws * numFeet) - (gt.earCleanTime * numEars))

/-- Theorem stating that the total grooming time is correct -/
theorem correct_grooming_time (gt : GroomingTime) :
  gt.nailClipTime = 10 → 
  gt.earCleanTime = 90 → 
  gt.totalTime = 640 → 
  calculateGroomingTime gt 4 4 2 = 640 := by
  sorry

#eval calculateGroomingTime { nailClipTime := 10, earCleanTime := 90, totalTime := 640 } 4 4 2

end NUMINAMATH_CALUDE_correct_grooming_time_l2136_213655


namespace NUMINAMATH_CALUDE_z_range_in_parallelogram_l2136_213692

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)
def D : ℝ × ℝ := (4, 0)

-- Define a function to check if a point is within or on the boundary of the parallelogram
def isInParallelogram (p : ℝ × ℝ) : Prop := sorry

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := 2 * p.1 - 5 * p.2

-- State the theorem
theorem z_range_in_parallelogram :
  ∀ p : ℝ × ℝ, isInParallelogram p → -14 ≤ z p ∧ z p ≤ 18 := by sorry

end NUMINAMATH_CALUDE_z_range_in_parallelogram_l2136_213692


namespace NUMINAMATH_CALUDE_B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l2136_213696

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x^2 - (m+2)*x + 2 < 0}

-- State the theorems
theorem B_subset_complement_A_iff_m_in_range :
  ∀ m : ℝ, B m ⊆ (Set.univ \ A) ↔ m ∈ Set.Icc 1 2 := by sorry

theorem A_intersect_B_nonempty_iff_m_in_range :
  ∀ m : ℝ, (A ∩ B m).Nonempty ↔ m ∈ Set.Iic 1 ∪ Set.Ioi 2 := by sorry

theorem A_union_B_eq_A_iff_m_in_range :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l2136_213696


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2136_213687

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 45 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2136_213687


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2136_213675

/-- The measure of angle P in a pentagon PQRST where ∠P = 2∠Q = 4∠R = 3∠S = 6∠T is 240° -/
theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 → -- sum of angles in a pentagon
  P = 2 * Q →              -- ∠P = 2∠Q
  P = 4 * R →              -- ∠P = 4∠R
  P = 3 * S →              -- ∠P = 3∠S
  P = 6 * T →              -- ∠P = 6∠T
  P = 240 := by            -- ∠P = 240°
sorry


end NUMINAMATH_CALUDE_pentagon_angle_measure_l2136_213675


namespace NUMINAMATH_CALUDE_perfect_score_l2136_213640

theorem perfect_score (perfect_score : ℕ) (h : 3 * perfect_score = 63) : perfect_score = 21 := by
  sorry

end NUMINAMATH_CALUDE_perfect_score_l2136_213640


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l2136_213658

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ 
  (∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l2136_213658


namespace NUMINAMATH_CALUDE_third_shift_participation_rate_l2136_213628

-- Define the total number of employees in each shift
def first_shift : ℕ := 60
def second_shift : ℕ := 50
def third_shift : ℕ := 40

-- Define the participation rates for the first two shifts
def first_shift_rate : ℚ := 1/5
def second_shift_rate : ℚ := 2/5

-- Define the total participation rate
def total_participation_rate : ℚ := 6/25

-- Theorem statement
theorem third_shift_participation_rate :
  let total_employees := first_shift + second_shift + third_shift
  let total_participants := total_employees * total_participation_rate
  let first_shift_participants := first_shift * first_shift_rate
  let second_shift_participants := second_shift * second_shift_rate
  let third_shift_participants := total_participants - first_shift_participants - second_shift_participants
  third_shift_participants / third_shift = 1/10 := by
sorry

end NUMINAMATH_CALUDE_third_shift_participation_rate_l2136_213628


namespace NUMINAMATH_CALUDE_prism_volume_l2136_213608

/-- The volume of a right rectangular prism with given face areas and one side length -/
theorem prism_volume (side_area front_area bottom_area : ℝ) (known_side : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 15)
  (h_known : known_side = 5) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    b = known_side ∧
    a * b * c = 75 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2136_213608


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_six_l2136_213610

theorem sum_reciprocals_equals_six (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) :
  1 / a + 1 / b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_six_l2136_213610


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l2136_213656

def fruit_arrangements (total : ℕ) (apples oranges bananas : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas)

theorem cory_fruit_arrangements :
  fruit_arrangements 9 4 2 2 = 3780 :=
by sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l2136_213656


namespace NUMINAMATH_CALUDE_function_inequality_l2136_213652

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), (deriv f x) / tan x < f x) →
  f (π / 3) < Real.sqrt 3 * f (π / 6) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2136_213652


namespace NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l2136_213670

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 128 ways to distribute 8 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_8_balls_3_boxes : distribute_balls 8 3 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l2136_213670


namespace NUMINAMATH_CALUDE_one_and_quarter_of_what_is_forty_l2136_213678

theorem one_and_quarter_of_what_is_forty : ∃ x : ℝ, 1.25 * x = 40 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_one_and_quarter_of_what_is_forty_l2136_213678


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l2136_213663

theorem binomial_difference_divisibility (p n : ℕ) (hp : Prime p) (hn : n > p) :
  ∃ k : ℤ, (Nat.choose (n + p - 1) p : ℤ) - (Nat.choose n p : ℤ) = k * n :=
sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l2136_213663


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2136_213629

theorem trigonometric_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2136_213629


namespace NUMINAMATH_CALUDE_xiaoliang_step_count_l2136_213679

/-- Represents the number of steps a person climbs to reach their floor -/
structure StepCount where
  floor : ℕ
  steps : ℕ

/-- Represents the building with information about Xiaoping and Xiaoliang -/
structure Building where
  xiaoping : StepCount
  xiaoliang : StepCount

/-- The theorem stating the number of steps Xiaoliang climbs -/
theorem xiaoliang_step_count (b : Building) 
  (h1 : b.xiaoping.floor = 5)
  (h2 : b.xiaoliang.floor = 4)
  (h3 : b.xiaoping.steps = 80) :
  b.xiaoliang.steps = 60 := by
  sorry

end NUMINAMATH_CALUDE_xiaoliang_step_count_l2136_213679


namespace NUMINAMATH_CALUDE_number_problem_l2136_213649

theorem number_problem (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 40 → x = 160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2136_213649


namespace NUMINAMATH_CALUDE_product_equals_48_l2136_213667

theorem product_equals_48 : 12 * (1 / 7) * 14 * 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_48_l2136_213667


namespace NUMINAMATH_CALUDE_correct_answers_l2136_213605

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  totalScore : ℤ
  attemptedAll : Bool

/-- Theorem stating the number of correctly answered questions given the exam conditions. -/
theorem correct_answers (e : Exam) (a : ExamAttempt) 
    (h1 : e.totalQuestions = 60)
    (h2 : e.correctScore = 4)
    (h3 : e.wrongScore = -1)
    (h4 : a.exam = e)
    (h5 : a.totalScore = 150)
    (h6 : a.attemptedAll = true) :
    ∃ (c : ℕ), c = 42 ∧ 
    c * e.correctScore + (e.totalQuestions - c) * e.wrongScore = a.totalScore :=
  sorry

end NUMINAMATH_CALUDE_correct_answers_l2136_213605


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_meaningful_l2136_213633

theorem sqrt_x_minus_5_meaningful (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 5 ↔ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_meaningful_l2136_213633


namespace NUMINAMATH_CALUDE_f_properties_l2136_213620

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_period : ∀ x, f (x + 6) = f x + f 3
axiom f_increasing_on_0_3 : ∀ x₁ x₂, x₁ ∈ Set.Icc 0 3 → x₂ ∈ Set.Icc 0 3 → x₁ ≠ x₂ → 
  (f x₁ - f x₂) / (x₁ - x₂) > 0

-- Theorem to prove
theorem f_properties :
  (∀ x, f (x - 6) = f (-x)) ∧ 
  (¬ ∀ x₁ x₂, x₁ ∈ Set.Icc (-9) (-6) → x₂ ∈ Set.Icc (-9) (-6) → x₁ < x₂ → f x₁ < f x₂) ∧
  (¬ ∃ x₁ x₂ x₃ x₄ x₅, x₁ ∈ Set.Icc (-9) 9 ∧ x₂ ∈ Set.Icc (-9) 9 ∧ x₃ ∈ Set.Icc (-9) 9 ∧ 
    x₄ ∈ Set.Icc (-9) 9 ∧ x₅ ∈ Set.Icc (-9) 9 ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ 
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ 
    x₄ ≠ x₅) :=
by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2136_213620


namespace NUMINAMATH_CALUDE_fraction_simplification_l2136_213616

theorem fraction_simplification :
  (21 : ℚ) / 25 * 35 / 45 * 75 / 63 = 35 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2136_213616


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2136_213636

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2136_213636


namespace NUMINAMATH_CALUDE_value_of_P_l2136_213694

theorem value_of_P (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 10) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_P_l2136_213694


namespace NUMINAMATH_CALUDE_problem_solution_l2136_213662

theorem problem_solution : 
  (99^2 = 9801) ∧ 
  ((-8)^2009 * (-1/8)^2008 = -8) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2136_213662


namespace NUMINAMATH_CALUDE_product_of_primes_l2136_213676

theorem product_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  2 < p → p < 6 → 
  8 < q → q < 24 → 
  15 < p * q → p * q < 36 → 
  p * q = 33 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_l2136_213676


namespace NUMINAMATH_CALUDE_two_eggs_remain_l2136_213683

/-- The number of eggs remaining unsold when packaging a given number of eggs into cartons of a specific size -/
def remaining_eggs (debra_eggs eli_eggs fiona_eggs carton_size : ℕ) : ℕ :=
  (debra_eggs + eli_eggs + fiona_eggs) % carton_size

/-- Theorem stating that given the specified number of eggs and carton size, 2 eggs will remain unsold -/
theorem two_eggs_remain :
  remaining_eggs 45 58 19 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_eggs_remain_l2136_213683


namespace NUMINAMATH_CALUDE_circle_c_equation_l2136_213643

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- Define the line l: x - y - 1 = 0
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_c_equation :
  ∃ (c : Circle),
    (circle_equation c A.1 A.2) ∧
    (line_l B.1 B.2) ∧
    (∀ (x y : ℝ), line_l x y → (x - B.1)^2 + (y - B.2)^2 ≤ c.radius^2) ∧
    (circle_equation c x y ↔ (x - 3)^2 + y^2 = 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_c_equation_l2136_213643


namespace NUMINAMATH_CALUDE_factorization_equality_simplification_equality_system_of_inequalities_l2136_213647

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  x^2 * (x - 3) + y^2 * (3 - x) = (x - 3) * (x + y) * (x - y) := by sorry

-- Problem 2
theorem simplification_equality (x : ℝ) (h1 : x ≠ 3/5) (h2 : x ≠ -3/5) :
  (2*x / (5*x - 3)) / (3 / (25*x^2 - 9)) * (x / (5*x + 3)) = 2/3 * x^2 := by sorry

-- Problem 3
theorem system_of_inequalities (x : ℝ) :
  ((x - 3) / 2 + 3 ≥ x + 1) ∧ (1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_simplification_equality_system_of_inequalities_l2136_213647


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l2136_213657

/-- A solid formed by unit cubes -/
structure CubeSolid where
  num_cubes : ℕ
  height : ℕ
  width : ℕ

/-- The surface area of a CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ := by sorry

/-- The theorem stating the surface area of the specific solid -/
theorem surface_area_of_specific_solid :
  ∃ (solid : CubeSolid),
    solid.num_cubes = 10 ∧
    solid.height = 3 ∧
    solid.width = 4 ∧
    surface_area solid = 34 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l2136_213657


namespace NUMINAMATH_CALUDE_invalid_period_pair_l2136_213635

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 48) ≡ a n [ZMOD 35]

def least_period_mod5 (a : ℕ → ℤ) (i : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + i) ≡ a n [ZMOD 5]) ∧
  (∀ k, k < i → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 5]))

def least_period_mod7 (a : ℕ → ℤ) (j : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + j) ≡ a n [ZMOD 7]) ∧
  (∀ k, k < j → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 7]))

theorem invalid_period_pair :
  ∀ a : ℕ → ℤ,
  is_valid_sequence a →
  ∀ i j : ℕ,
  least_period_mod5 a i →
  least_period_mod7 a j →
  (i, j) ≠ (16, 4) :=
by sorry

end NUMINAMATH_CALUDE_invalid_period_pair_l2136_213635


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l2136_213646

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hq_bound : q < 100) :
  M * (1 + p / 100) * (1 - q / 100) = 1.1 * M ↔ 
  p = (10 + 100 * q) / (100 - q) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l2136_213646


namespace NUMINAMATH_CALUDE_exists_unreachable_grid_l2136_213644

/-- Represents an 8x8 grid of natural numbers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Represents a subgrid selection, either 3x3 or 4x4 -/
inductive Subgrid
| three : Fin 6 → Fin 6 → Subgrid
| four : Fin 5 → Fin 5 → Subgrid

/-- Applies the increment operation to a subgrid -/
def applyOperation (g : Grid) (s : Subgrid) : Grid :=
  sorry

/-- Checks if all numbers in the grid are divisible by 10 -/
def allDivisibleBy10 (g : Grid) : Prop :=
  ∀ i j, (g i j) % 10 = 0

/-- The main theorem statement -/
theorem exists_unreachable_grid :
  ∃ (initial : Grid), ¬∃ (ops : List Subgrid), allDivisibleBy10 (ops.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unreachable_grid_l2136_213644


namespace NUMINAMATH_CALUDE_k_increasing_range_l2136_213693

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the domain of f
def D : Set ℝ := { x | x ≥ -1 }

-- Define the property of being k-increasing on a set
def is_k_increasing (f : ℝ → ℝ) (k : ℝ) (S : Set ℝ) : Prop :=
  k ≠ 0 ∧ ∀ x ∈ S, (x + k) ∈ S → f (x + k) ≥ f x

-- State the theorem
theorem k_increasing_range (k : ℝ) :
  is_k_increasing f k D → k ≥ 2 := by sorry

end NUMINAMATH_CALUDE_k_increasing_range_l2136_213693


namespace NUMINAMATH_CALUDE_missing_score_l2136_213609

theorem missing_score (scores : List ℕ) (mean : ℚ) : 
  scores = [73, 83, 86, 73] ∧ 
  mean = 79.2 ∧ 
  (scores.sum + (missing : ℕ)) / 5 = mean → 
  missing = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_score_l2136_213609


namespace NUMINAMATH_CALUDE_uncle_ben_chickens_l2136_213639

/-- Represents Uncle Ben's farm --/
structure Farm where
  roosters : Nat
  nonLayingHens : Nat
  eggsPerLayingHen : Nat
  totalEggs : Nat

/-- Calculates the total number of chickens on the farm --/
def totalChickens (f : Farm) : Nat :=
  let layingHens := f.totalEggs / f.eggsPerLayingHen
  f.roosters + f.nonLayingHens + layingHens

/-- Theorem stating that Uncle Ben has 440 chickens --/
theorem uncle_ben_chickens :
  ∀ (f : Farm),
    f.roosters = 39 →
    f.nonLayingHens = 15 →
    f.eggsPerLayingHen = 3 →
    f.totalEggs = 1158 →
    totalChickens f = 440 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_chickens_l2136_213639


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l2136_213641

theorem quadratic_expression_equality (x : ℝ) (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l2136_213641


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l2136_213615

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (a b : Line) (α β : Plane)
  (distinct_lines : a ≠ b)
  (distinct_planes : α ≠ β)
  (a_perp_α : perpendicular a α)
  (b_perp_β : perpendicular b β)
  (a_parallel_b : parallel a b) :
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l2136_213615


namespace NUMINAMATH_CALUDE_wire_length_between_poles_l2136_213654

theorem wire_length_between_poles (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) 
  (h1 : base_distance = 20)
  (h2 : short_pole_height = 10)
  (h3 : tall_pole_height = 22) :
  Real.sqrt (base_distance ^ 2 + (tall_pole_height - short_pole_height) ^ 2) = Real.sqrt 544 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_between_poles_l2136_213654


namespace NUMINAMATH_CALUDE_smallest_number_l2136_213632

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases --/
def number_A : List Nat := [0, 2]
def number_B : List Nat := [0, 3]
def number_C : List Nat := [3, 2]
def number_D : List Nat := [1, 3]

/-- The bases of the given numbers --/
def base_A : Nat := 7
def base_B : Nat := 5
def base_C : Nat := 6
def base_D : Nat := 4

theorem smallest_number :
  to_decimal number_D base_D < to_decimal number_A base_A ∧
  to_decimal number_D base_D < to_decimal number_B base_B ∧
  to_decimal number_D base_D < to_decimal number_C base_C :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2136_213632


namespace NUMINAMATH_CALUDE_parallel_condition_l2136_213672

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + (a + 2) * y + 1
  l₂ : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- The condition for two lines to be parallel -/
def parallel (lines : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ lines.a = k ∧ lines.a + 2 = k * lines.a

/-- The statement to be proved -/
theorem parallel_condition (lines : TwoLines) :
  (parallel lines → lines.a = -1) ∧ ¬(lines.a = -1 → parallel lines) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2136_213672


namespace NUMINAMATH_CALUDE_triangle_area_l2136_213600

/-- Given a triangle with perimeter 36, inradius 2.5, and sides in ratio 3:4:5, its area is 45 -/
theorem triangle_area (a b c : ℝ) (perimeter inradius : ℝ) : 
  perimeter = 36 →
  inradius = 2.5 →
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →
  a + b + c = perimeter →
  (a + b + c) / 2 * inradius = 45 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l2136_213600


namespace NUMINAMATH_CALUDE_defective_rate_is_twenty_percent_l2136_213601

variable (n : ℕ)  -- number of defective items among 10 products

-- Define the probability of selecting one defective item out of two random selections
def prob_one_defective (n : ℕ) : ℚ :=
  (n * (10 - n)) / (10 * 9)

-- Theorem statement
theorem defective_rate_is_twenty_percent :
  n ≤ 10 ∧                     -- n is at most 10 (total number of products)
  prob_one_defective n = 16/45 ∧ -- probability of selecting one defective item is 16/45
  n ≤ 4 →                      -- defective rate does not exceed 40%
  n = 2                        -- implies that n = 2, which means 20% defective rate
  := by sorry

end NUMINAMATH_CALUDE_defective_rate_is_twenty_percent_l2136_213601


namespace NUMINAMATH_CALUDE_expression_evaluation_l2136_213659

theorem expression_evaluation : 4 * (5^2 + 5^2 + 5^2 + 5^2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2136_213659


namespace NUMINAMATH_CALUDE_corner_removed_cube_surface_area_l2136_213613

/-- Represents a cube with corner cubes removed -/
structure CornerRemovedCube where
  side_length : ℝ
  corner_size : ℝ

/-- Calculates the surface area of a cube with corner cubes removed -/
def surface_area (cube : CornerRemovedCube) : ℝ :=
  6 * cube.side_length^2

/-- Theorem stating that a 4x4x4 cube with corner cubes removed has surface area 96 sq.cm -/
theorem corner_removed_cube_surface_area :
  let cube : CornerRemovedCube := ⟨4, 1⟩
  surface_area cube = 96 := by
  sorry

end NUMINAMATH_CALUDE_corner_removed_cube_surface_area_l2136_213613


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l2136_213665

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the number of people eating pizza -/
def num_people : ℕ := 4

/-- Represents the number of people eating both types of pizza -/
def num_people_both_types : ℕ := 3

/-- Represents the number of cheese slices left -/
def cheese_slices_left : ℕ := 7

/-- Represents the number of pepperoni slices left -/
def pepperoni_slices_left : ℕ := 1

/-- Represents the total number of slices each person eats -/
def slices_per_person : ℕ := 6

theorem pizza_slices_theorem :
  slices_per_person * num_people = 
    2 * slices_per_pizza - cheese_slices_left - pepperoni_slices_left :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_theorem_l2136_213665


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l2136_213660

/-- Joel's current age -/
def joel_current_age : ℕ := 8

/-- Joel's dad's current age -/
def dad_current_age : ℕ := 37

/-- The number of years until Joel's dad is twice Joel's age -/
def years_until_double : ℕ := dad_current_age - 2 * joel_current_age

/-- Joel's age when his dad is twice as old as him -/
def joel_future_age : ℕ := joel_current_age + years_until_double

theorem joel_age_when_dad_twice_as_old :
  joel_future_age = 29 :=
sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l2136_213660


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2136_213626

/-- An isosceles triangle with congruent sides of length 6 and perimeter 20 has a base of length 8 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 6
    let perimeter := 20
    (2 * congruent_side + base = perimeter) → base = 8

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2136_213626


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l2136_213645

theorem polynomial_roots_problem (c d : ℤ) (h1 : c ≠ 0) (h2 : d ≠ 0) 
  (h3 : ∃ p q : ℤ, (X - p)^2 * (X - q) = X^3 + c*X^2 + d*X + 12*c) : 
  |c * d| = 192 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_problem_l2136_213645


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2136_213625

theorem min_sum_of_squares (x y z k : ℝ) 
  (h1 : (x + 8) * (y - 8) = 0) 
  (h2 : x + y + z = k) : 
  x^2 + y^2 + z^2 ≥ 64 + k^2/2 - 4*k + 32 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2136_213625


namespace NUMINAMATH_CALUDE_intersection_line_not_through_point_l2136_213668

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition for M being on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line equation passing through intersection points
def line_AB (a b m n : ℝ) : Prop := 2*m*a + 2*n*b - (2*m + 3) = 0

-- Theorem statement
theorem intersection_line_not_through_point :
  ∀ (a b : ℝ), M_on_C a b →
  ¬(line_AB a b (1/2) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_not_through_point_l2136_213668


namespace NUMINAMATH_CALUDE_leftover_value_l2136_213627

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the dollar value of a given number of quarters and dimes --/
def dollarValue (quarters dimes : Nat) : ℚ :=
  (quarters : ℚ) * (1 / 4) + (dimes : ℚ) * (1 / 10)

/-- Theorem stating the dollar value of leftover coins --/
theorem leftover_value (roll_size : RollSize) (ana_coins ben_coins : CoinCollection) :
  roll_size.quarters = 30 →
  roll_size.dimes = 40 →
  ana_coins.quarters = 95 →
  ana_coins.dimes = 183 →
  ben_coins.quarters = 104 →
  ben_coins.dimes = 219 →
  dollarValue 
    ((ana_coins.quarters + ben_coins.quarters) % roll_size.quarters)
    ((ana_coins.dimes + ben_coins.dimes) % roll_size.dimes) = 695 / 100 := by
  sorry

#eval dollarValue 19 22

end NUMINAMATH_CALUDE_leftover_value_l2136_213627


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l2136_213671

theorem quadratic_real_root_condition (a b c : ℝ) : 
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a - b + c) * x + 3 = 0) →
  (a = c ∧ a = -b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l2136_213671


namespace NUMINAMATH_CALUDE_factoring_quadratic_l2136_213651

theorem factoring_quadratic (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_quadratic_l2136_213651


namespace NUMINAMATH_CALUDE_tan_equality_implies_160_degrees_l2136_213614

theorem tan_equality_implies_160_degrees (x : Real) :
  0 ≤ x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
                                   (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 160 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_160_degrees_l2136_213614


namespace NUMINAMATH_CALUDE_stack_map_views_l2136_213674

def StackMap : Type := List (List Nat)

def frontView (sm : StackMap) : List Nat :=
  sm.map (List.foldl max 0)

def rightSideView (sm : StackMap) : List Nat :=
  List.map (List.foldl max 0) (List.transpose sm)

theorem stack_map_views (sm : StackMap) 
  (h1 : sm = [[3, 1, 2], [2, 4, 3], [1, 1, 3]]) : 
  frontView sm = [3, 4, 3] ∧ rightSideView sm = [3, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_views_l2136_213674


namespace NUMINAMATH_CALUDE_carla_liquid_consumption_l2136_213634

/-- The amount of water Carla drank in ounces -/
def water : ℝ := 15

/-- The amount of soda Carla drank in ounces -/
def soda : ℝ := 3 * water - 6

/-- The total amount of liquid Carla drank in ounces -/
def total_liquid : ℝ := water + soda

/-- Theorem stating the total amount of liquid Carla drank -/
theorem carla_liquid_consumption : total_liquid = 54 := by
  sorry

end NUMINAMATH_CALUDE_carla_liquid_consumption_l2136_213634


namespace NUMINAMATH_CALUDE_magnitude_relationship_l2136_213682

theorem magnitude_relationship :
  let α : ℝ := Real.cos 4
  let b : ℝ := Real.cos (4 * π / 5)
  let c : ℝ := Real.sin (7 * π / 6)
  b < α ∧ α < c := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l2136_213682


namespace NUMINAMATH_CALUDE_height_radius_ratio_is_2pi_l2136_213689

/-- A cylinder with a square lateral surface -/
structure SquareLateralCylinder where
  radius : ℝ
  height : ℝ
  lateral_surface_is_square : height = 2 * Real.pi * radius

/-- The ratio of height to radius for a cylinder with a square lateral surface is 2π -/
theorem height_radius_ratio_is_2pi (c : SquareLateralCylinder) :
  c.height / c.radius = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_height_radius_ratio_is_2pi_l2136_213689


namespace NUMINAMATH_CALUDE_citric_acid_weight_l2136_213685

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_molecular_weight : ℝ := 192.12

/-- Theorem stating that the molecular weight of Citric acid is 192.12 g/mol -/
theorem citric_acid_weight : citric_acid_molecular_weight = 192.12 := by sorry

end NUMINAMATH_CALUDE_citric_acid_weight_l2136_213685


namespace NUMINAMATH_CALUDE_geometric_sum_property_l2136_213680

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_property (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q = 2 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_property_l2136_213680


namespace NUMINAMATH_CALUDE_painted_cubes_theorem_l2136_213650

/-- Represents the dimensions of a parallelepiped -/
structure Parallelepiped where
  m : ℕ
  n : ℕ
  k : ℕ
  h1 : 0 < k
  h2 : k ≤ n
  h3 : n ≤ m

/-- The set of possible numbers of painted cubes -/
def PaintedCubesCounts : Set ℕ := {60, 72, 84, 90, 120}

/-- 
  Given a parallelepiped where three faces sharing a common vertex are painted,
  if half of all cubes have at least one painted face, then the number of
  painted cubes is in the set PaintedCubesCounts
-/
theorem painted_cubes_theorem (p : Parallelepiped) :
  (p.m - 1) * (p.n - 1) * (p.k - 1) = p.m * p.n * p.k / 2 →
  (p.m * p.n * p.k - (p.m - 1) * (p.n - 1) * (p.k - 1)) ∈ PaintedCubesCounts := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_theorem_l2136_213650


namespace NUMINAMATH_CALUDE_negation_of_log_inequality_l2136_213677

theorem negation_of_log_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, Real.log x > 1) → 
  (¬p ↔ ∃ x₀ : ℝ, Real.log x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_log_inequality_l2136_213677


namespace NUMINAMATH_CALUDE_fair_coin_same_side_five_tosses_l2136_213630

/-- A fair coin is a coin with equal probability of landing on either side -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a sequence of independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The number of tosses -/
def num_tosses : ℕ := 5

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_same_side_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_sequence p num_tosses = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_same_side_five_tosses_l2136_213630


namespace NUMINAMATH_CALUDE_expression_evaluation_l2136_213653

theorem expression_evaluation : 120 * (120 - 5) - (120 * 120 - 10 + 2) = -592 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2136_213653


namespace NUMINAMATH_CALUDE_adidas_to_skechers_ratio_l2136_213698

/-- Proves the ratio of Adidas to Skechers sneakers spending is 1:5 --/
theorem adidas_to_skechers_ratio
  (total_spent : ℕ)
  (nike_to_adidas_ratio : ℕ)
  (adidas_cost : ℕ)
  (clothes_cost : ℕ)
  (h1 : total_spent = 8000)
  (h2 : nike_to_adidas_ratio = 3)
  (h3 : adidas_cost = 600)
  (h4 : clothes_cost = 2600) :
  (adidas_cost : ℚ) / (total_spent - clothes_cost - nike_to_adidas_ratio * adidas_cost - adidas_cost) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_adidas_to_skechers_ratio_l2136_213698


namespace NUMINAMATH_CALUDE_scallop_dinner_cost_l2136_213631

/-- Represents the problem of calculating the cost of scallops for Nate's dinner. -/
theorem scallop_dinner_cost :
  let scallops_per_pound : ℕ := 8
  let cost_per_pound : ℚ := 24
  let scallops_per_person : ℕ := 2
  let number_of_people : ℕ := 8
  
  let total_scallops : ℕ := scallops_per_person * number_of_people
  let pounds_needed : ℚ := total_scallops / scallops_per_pound
  let total_cost : ℚ := pounds_needed * cost_per_pound
  
  total_cost = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_scallop_dinner_cost_l2136_213631


namespace NUMINAMATH_CALUDE_total_cost_separate_tickets_l2136_213642

def adult_ticket_cost : ℕ := 35
def child_ticket_cost : ℕ := 20
def num_adults : ℕ := 2
def num_children : ℕ := 5

theorem total_cost_separate_tickets :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = 170 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_separate_tickets_l2136_213642


namespace NUMINAMATH_CALUDE_average_of_combined_results_l2136_213602

theorem average_of_combined_results :
  let n₁ : ℕ := 30
  let avg₁ : ℚ := 20
  let n₂ : ℕ := 20
  let avg₂ : ℚ := 30
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l2136_213602


namespace NUMINAMATH_CALUDE_custom_deck_probability_l2136_213638

theorem custom_deck_probability : 
  let total_cards : ℕ := 65
  let spades : ℕ := 14
  let other_suits : ℕ := 13
  let aces : ℕ := 4
  let kings : ℕ := 4
  (aces : ℚ) / total_cards * kings / (total_cards - 1) = 1 / 260 :=
by sorry

end NUMINAMATH_CALUDE_custom_deck_probability_l2136_213638


namespace NUMINAMATH_CALUDE_longest_side_length_l2136_213690

/-- The polygonal region defined by the given system of inequalities -/
def PolygonalRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 5 ∧ 3 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The vertices of the polygonal region -/
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (1, 0), (0, 5)}

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The length of the longest side of the polygonal region is √26 -/
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
  ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices →
  squaredDistance p q ≥ squaredDistance r s ∧
  squaredDistance p q = 26 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_length_l2136_213690


namespace NUMINAMATH_CALUDE_complex_division_equality_l2136_213637

theorem complex_division_equality : (2 : ℂ) / (2 - I) = 4/5 + 2/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l2136_213637


namespace NUMINAMATH_CALUDE_diagonal_path_cubes_3_4_5_l2136_213623

/-- The number of cubes a diagonal path crosses in a cuboid -/
def cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd a c + Nat.gcd a (Nat.gcd b c)

/-- Theorem: In a 3 × 4 × 5 cuboid, a diagonal path from one corner to the opposite corner
    that doesn't intersect the edges of any small cube inside the cuboid passes through 10 small cubes -/
theorem diagonal_path_cubes_3_4_5 :
  cubes_crossed 3 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_path_cubes_3_4_5_l2136_213623


namespace NUMINAMATH_CALUDE_log_inequality_l2136_213669

theorem log_inequality (x : Real) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2136_213669


namespace NUMINAMATH_CALUDE_total_is_100_l2136_213673

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (1 / 4) * (s.b + s.c) ∧
  s.b = (3 / 5) * (s.a + s.c) ∧
  s.a = 20

/-- The theorem stating that the total amount is 100 -/
theorem total_is_100 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_is_100_l2136_213673


namespace NUMINAMATH_CALUDE_max_value_of_function_l2136_213604

theorem max_value_of_function (x : ℝ) : 1 + 1 / (x^2 + 2*x + 2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2136_213604


namespace NUMINAMATH_CALUDE_quadratic_roots_irrational_l2136_213648

theorem quadratic_roots_irrational (k : ℝ) (h1 : k^2 = 16/3) (h2 : ∀ x, x^2 - 5*k*x + 3*k^2 = 0 → ∃ y, x^2 - 5*k*x + 3*k^2 = 0 ∧ x * y = 16) :
  ∃ x y : ℝ, x^2 - 5*k*x + 3*k^2 = 0 ∧ y^2 - 5*k*y + 3*k^2 = 0 ∧ x * y = 16 ∧ (¬ ∃ m n : ℤ, x = m / n ∨ y = m / n) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_irrational_l2136_213648


namespace NUMINAMATH_CALUDE_slope_product_negative_half_exists_line_equal_distances_l2136_213691

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and Q
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (-2, 0)

-- Theorem for part (I)
theorem slope_product_negative_half (x y : ℝ) :
  C x y → x ≠ 0 → (y - A.2) / (x - A.1) * (y - B.2) / (x - B.1) = -1/2 := by sorry

-- Theorem for part (II)
theorem exists_line_equal_distances :
  ∃ (M N : ℝ × ℝ), 
    C M.1 M.2 ∧ C N.1 N.2 ∧ 
    M ≠ N ∧
    M.2 = 0 ∧ N.2 = 0 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (N.1 - B.1)^2 + (N.2 - B.2)^2 := by sorry

end NUMINAMATH_CALUDE_slope_product_negative_half_exists_line_equal_distances_l2136_213691


namespace NUMINAMATH_CALUDE_trig_identity_l2136_213697

theorem trig_identity : 
  Real.sin (46 * π / 180) * Real.cos (16 * π / 180) - 
  Real.cos (314 * π / 180) * Real.sin (16 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2136_213697
