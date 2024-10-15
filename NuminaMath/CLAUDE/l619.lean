import Mathlib

namespace NUMINAMATH_CALUDE_sixth_term_is_three_l619_61963

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The sequence
  sum_first_three : (a 1) + (a 2) + (a 3) = 168
  diff_two_five : (a 2) - (a 5) = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l619_61963


namespace NUMINAMATH_CALUDE_down_payment_calculation_l619_61918

def cash_price : ℕ := 400
def monthly_payment : ℕ := 30
def num_months : ℕ := 12
def cash_savings : ℕ := 80

theorem down_payment_calculation : 
  cash_price + cash_savings - monthly_payment * num_months = 120 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l619_61918


namespace NUMINAMATH_CALUDE_fraction_change_l619_61934

theorem fraction_change (a b p q x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a^2 + x^2) / (b^2 + x^2) = p / q) 
  (h4 : q ≠ p) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l619_61934


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l619_61987

def total_red_eggs : ℕ := 30
def total_blue_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket :
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧
             n ∣ total_red_eggs ∧
             n ∣ total_blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs_per_basket ∧
                        m ∣ total_red_eggs ∧
                        m ∣ total_blue_eggs →
                        m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l619_61987


namespace NUMINAMATH_CALUDE_fraction_equality_l619_61958

theorem fraction_equality (x : ℚ) (f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l619_61958


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l619_61981

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 125 / 1 → a / b = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l619_61981


namespace NUMINAMATH_CALUDE_high_school_population_l619_61930

theorem high_school_population (total_sample : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) (third_grade_population : ℕ) : 
  total_sample = 36 → 
  first_grade_sample = 15 → 
  second_grade_sample = 12 → 
  third_grade_population = 900 → 
  (total_sample : ℚ) / (first_grade_sample + second_grade_sample + (total_sample - first_grade_sample - second_grade_sample)) = 
  (total_sample - first_grade_sample - second_grade_sample : ℚ) / third_grade_population → 
  (total_sample : ℕ) * (third_grade_population / (total_sample - first_grade_sample - second_grade_sample)) = 3600 :=
by sorry

end NUMINAMATH_CALUDE_high_school_population_l619_61930


namespace NUMINAMATH_CALUDE_fraction_irreducible_l619_61960

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l619_61960


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l619_61916

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (20 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  2 * (Real.sin (35 * π / 180) - Real.sin (10 * π / 180)) / 
  (1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l619_61916


namespace NUMINAMATH_CALUDE_arrangements_not_head_tail_six_arrangements_not_adjacent_six_l619_61908

/-- The number of students in the row -/
def n : ℕ := 6

/-- The number of arrangements where one student doesn't stand at the head or tail -/
def arrangements_not_head_tail (n : ℕ) : ℕ := sorry

/-- The number of arrangements where three specific students are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem for the first question -/
theorem arrangements_not_head_tail_six : 
  arrangements_not_head_tail n = 480 := by sorry

/-- Theorem for the second question -/
theorem arrangements_not_adjacent_six : 
  arrangements_not_adjacent n = 144 := by sorry

end NUMINAMATH_CALUDE_arrangements_not_head_tail_six_arrangements_not_adjacent_six_l619_61908


namespace NUMINAMATH_CALUDE_unique_solution_l619_61985

theorem unique_solution (p q n : ℕ+) (h1 : Nat.gcd p.val q.val = 1)
  (h2 : p + q^2 = (n^2 + 1) * p^2 + q) :
  p = n + 1 ∧ q = n^2 + n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l619_61985


namespace NUMINAMATH_CALUDE_quadratic_intersection_range_l619_61988

/-- The range of a for which the intersection of A and B is non-empty -/
theorem quadratic_intersection_range (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 - 2 * x - 2 * a
  let A : Set ℝ := {x | f x > 0}
  let B : Set ℝ := {x | 1 < x ∧ x < 3}
  (A ∩ B).Nonempty → a < -2 ∨ a > 6/7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_range_l619_61988


namespace NUMINAMATH_CALUDE_sum_of_possible_radii_l619_61907

/-- Given a circle with center C(r,r) that is tangent to the positive x-axis,
    positive y-axis, and externally tangent to a circle centered at (4,0) with radius 1,
    the sum of all possible radii of circle C is 10. -/
theorem sum_of_possible_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 1)^2) →
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 1)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 1)^2) ∧
    r₁ + r₂ = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_radii_l619_61907


namespace NUMINAMATH_CALUDE_probability_of_target_plate_l619_61922

structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char

def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']
def hex_digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first ∈ vowels ∧
  plate.second ∈ non_vowels ∧
  plate.third ∈ non_vowels ∧
  plate.second ≠ plate.third ∧
  plate.fourth ∈ hex_digits

def total_valid_plates : ℕ := vowels.length * non_vowels.length * (non_vowels.length - 1) * hex_digits.length

def target_plate : LicensePlate := ⟨'E', 'Y', 'B', '5'⟩

theorem probability_of_target_plate :
  (1 : ℚ) / total_valid_plates = 1 / 44352 :=
sorry

end NUMINAMATH_CALUDE_probability_of_target_plate_l619_61922


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l619_61947

theorem imaginary_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l619_61947


namespace NUMINAMATH_CALUDE_bill_true_discount_l619_61993

/-- Given a bill with face value and banker's discount, calculate the true discount -/
def true_discount (face_value banker_discount : ℚ) : ℚ :=
  (banker_discount * face_value) / (banker_discount + face_value)

/-- Theorem stating that for a bill with face value 270 and banker's discount 54, 
    the true discount is 45 -/
theorem bill_true_discount : 
  true_discount 270 54 = 45 := by sorry

end NUMINAMATH_CALUDE_bill_true_discount_l619_61993


namespace NUMINAMATH_CALUDE_methodC_cannot_eliminate_variables_l619_61952

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 5 * x + 2 * y = 4
def equation2 (x y : ℝ) : Prop := 2 * x - 3 * y = 10

-- Define the method C
def methodC (x y : ℝ) : Prop := 1.5 * (5 * x + 2 * y) - (2 * x - 3 * y) = 1.5 * 4 - 10

-- Theorem stating that method C cannot eliminate variables
theorem methodC_cannot_eliminate_variables :
  ∀ x y : ℝ, methodC x y ↔ (5.5 * x + 6 * y = -4) :=
by sorry

end NUMINAMATH_CALUDE_methodC_cannot_eliminate_variables_l619_61952


namespace NUMINAMATH_CALUDE_chores_to_cartoons_ratio_l619_61945

/-- Given that 2 hours (120 minutes) of cartoons requires 96 minutes of chores,
    prove that the ratio of chores to cartoons is 8 minutes of chores
    for every 10 minutes of cartoons. -/
theorem chores_to_cartoons_ratio :
  ∀ (cartoon_time chore_time : ℕ),
    cartoon_time = 120 →
    chore_time = 96 →
    (chore_time : ℚ) / (cartoon_time : ℚ) * 10 = 8 := by
  sorry

#check chores_to_cartoons_ratio

end NUMINAMATH_CALUDE_chores_to_cartoons_ratio_l619_61945


namespace NUMINAMATH_CALUDE_mike_passing_percentage_l619_61980

/-- The percentage Mike needs to pass, given his score, shortfall, and maximum possible marks. -/
theorem mike_passing_percentage
  (mike_score : ℕ)
  (shortfall : ℕ)
  (max_marks : ℕ)
  (h1 : mike_score = 212)
  (h2 : shortfall = 16)
  (h3 : max_marks = 760) :
  (((mike_score + shortfall : ℚ) / max_marks) * 100 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_passing_percentage_l619_61980


namespace NUMINAMATH_CALUDE_student_score_l619_61964

theorem student_score (total_questions : Nat) (correct_responses : Nat) : 
  total_questions = 100 →
  correct_responses = 88 →
  let incorrect_responses := total_questions - correct_responses
  let score := correct_responses - 2 * incorrect_responses
  score = 64 := by
sorry

end NUMINAMATH_CALUDE_student_score_l619_61964


namespace NUMINAMATH_CALUDE_semicircle_radius_l619_61935

/-- Given a semi-circle with perimeter 180 cm, its radius is 180 / (π + 2) cm. -/
theorem semicircle_radius (P : ℝ) (h : P = 180) :
  P = π * r + 2 * r → r = 180 / (π + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l619_61935


namespace NUMINAMATH_CALUDE_decimal_point_shift_l619_61973

theorem decimal_point_shift (x : ℝ) : (x / 10 = x - 0.72) → x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l619_61973


namespace NUMINAMATH_CALUDE_trigonometric_identities_l619_61900

theorem trigonometric_identities : 
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1/2) ∧ 
  (Real.sin (45 * π / 180) * Real.cos (15 * π / 180) - 
   Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l619_61900


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l619_61920

theorem number_puzzle_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l619_61920


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l619_61974

/-- A die is represented as a finite type with 6 elements -/
def Die : Type := Fin 6

/-- The sample space of rolling two dice -/
def SampleSpace : Type := Die × Die

/-- Event A: the number on the first die is a multiple of 3 -/
def EventA (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) % 3 = 0

/-- Event B: the sum of the numbers on the two dice is greater than 7 -/
def EventB (outcome : SampleSpace) : Prop :=
  outcome.1.val + outcome.2.val + 2 > 7

/-- The probability measure on the sample space -/
def P : Set SampleSpace → ℝ := sorry

/-- Theorem: The conditional probability P(B|A) is 7/12 -/
theorem conditional_probability_B_given_A :
  P {outcome | EventB outcome ∧ EventA outcome} / P {outcome | EventA outcome} = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l619_61974


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l619_61984

theorem simplify_fraction_expression : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l619_61984


namespace NUMINAMATH_CALUDE_min_gennadys_for_festival_l619_61959

/-- Represents the number of people with a given name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadys required -/
def min_gennadys (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- The theorem stating the minimum number of Gennadys required -/
theorem min_gennadys_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  min_gennadys counts = 49 := by
  sorry

#eval min_gennadys { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadys_for_festival_l619_61959


namespace NUMINAMATH_CALUDE_min_sum_distances_l619_61957

open Real

/-- The minimum sum of distances between four points in a Cartesian plane -/
theorem min_sum_distances :
  let A : ℝ × ℝ := (-2, -3)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ → ℝ × ℝ := λ m ↦ (m, 0)
  let D : ℝ → ℝ × ℝ := λ n ↦ (n, n)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m n : ℝ), ∀ (m' n' : ℝ),
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A ≤
    distance A B + distance B (C m') + distance (C m') (D n') + distance (D n') A ∧
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A = 58 + 2 * Real.sqrt 10 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_distances_l619_61957


namespace NUMINAMATH_CALUDE_problem_solution_l619_61940

theorem problem_solution (x : ℝ) : (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l619_61940


namespace NUMINAMATH_CALUDE_boundary_length_of_modified_square_l619_61901

-- Define the square's area
def square_area : ℝ := 256

-- Define the number of divisions per side
def divisions : ℕ := 4

-- Theorem statement
theorem boundary_length_of_modified_square :
  let side_length := Real.sqrt square_area
  let segment_length := side_length / divisions
  let arc_length := 2 * Real.pi * segment_length
  let straight_segments_length := 2 * divisions * segment_length
  abs ((arc_length + straight_segments_length) - 57.1) < 0.05 := by
sorry

end NUMINAMATH_CALUDE_boundary_length_of_modified_square_l619_61901


namespace NUMINAMATH_CALUDE_line_parallel_contained_in_plane_l619_61995

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_contained_in_plane 
  (a b : Line) (α : Plane) :
  parallel a b → containedIn b α → 
  parallelToPlane a α ∨ containedIn a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_contained_in_plane_l619_61995


namespace NUMINAMATH_CALUDE_outfits_count_l619_61932

/-- The number of different outfits that can be made with a given number of shirts, ties, and shoes. -/
def num_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ := shirts * ties * shoes

/-- Theorem: Given 8 shirts, 7 ties, and 4 pairs of shoes, the total number of different possible outfits is 224. -/
theorem outfits_count : num_outfits 8 7 4 = 224 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l619_61932


namespace NUMINAMATH_CALUDE_quadratic_max_value_l619_61998

/-- A quadratic function that takes specific values for consecutive natural numbers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 14 ∧ f (n + 2) = 14

/-- The theorem stating the maximum value of the quadratic function -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 15 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l619_61998


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l619_61937

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 + Nat.factorial 5 = 5160 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l619_61937


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l619_61946

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (d t : ℝ) :
  arithmetic_sequence a d →
  d > 0 →
  a 1 = 1 →
  (∀ n, 2 * (a n * a (n + 1) + 1) = t * (1 + a n)) →
  ∀ n, a n = 2 * n - 1 + (-1)^n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l619_61946


namespace NUMINAMATH_CALUDE_combination_square_numbers_examples_find_m_l619_61943

def is_combination_square_numbers (a b c : Int) : Prop :=
  a < 0 ∧ b < 0 ∧ c < 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y z : Int), x^2 = a * b ∧ y^2 = b * c ∧ z^2 = a * c

theorem combination_square_numbers_examples :
  (is_combination_square_numbers (-4) (-16) (-25)) ∧
  (is_combination_square_numbers (-3) (-48) (-12)) ∧
  (is_combination_square_numbers (-2) (-18) (-72)) := by sorry

theorem find_m :
  ∀ m : Int, is_combination_square_numbers (-3) m (-12) ∧ 
  (∃ (x : Int), x^2 = -3 * m ∨ x^2 = m * (-12) ∨ x^2 = -3 * (-12)) ∧
  x = 12 → m = -48 := by sorry

end NUMINAMATH_CALUDE_combination_square_numbers_examples_find_m_l619_61943


namespace NUMINAMATH_CALUDE_ascending_order_l619_61966

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_ascending_order_l619_61966


namespace NUMINAMATH_CALUDE_problem_solution_l619_61926

theorem problem_solution : 
  let P : ℕ := 2007 / 5
  let Q : ℕ := P / 4
  let Y : ℕ := 2 * (P - Q)
  Y = 602 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l619_61926


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l619_61933

theorem no_function_satisfies_equation :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + y) = x * f x + y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l619_61933


namespace NUMINAMATH_CALUDE_total_time_in_work_week_l619_61921

/-- Represents the days of the work week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Commute time for each day of the week -/
def commute_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 35
  | Weekday.Tuesday => 45
  | Weekday.Wednesday => 25
  | Weekday.Thursday => 40
  | Weekday.Friday => 30

/-- Additional delay for each day of the week -/
def additional_delay (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 5
  | Weekday.Wednesday => 10
  | Weekday.Friday => 8
  | _ => 0

/-- Security check time for each day of the week -/
def security_check_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Tuesday => 30
  | Weekday.Thursday => 10
  | _ => 15

/-- Constant time for parking and walking -/
def parking_and_walking_time : ℕ := 8

/-- Total time spent on a given day -/
def daily_total_time (day : Weekday) : ℕ :=
  commute_time day + additional_delay day + security_check_time day + parking_and_walking_time

/-- List of all work days in a week -/
def work_week : List Weekday := [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, Weekday.Friday]

/-- Theorem stating the total time spent in a work week -/
theorem total_time_in_work_week : (work_week.map daily_total_time).sum = 323 := by
  sorry

end NUMINAMATH_CALUDE_total_time_in_work_week_l619_61921


namespace NUMINAMATH_CALUDE_room_length_proof_l619_61989

theorem room_length_proof (width : ℝ) (area_covered : ℝ) (area_needed : ℝ) :
  width = 15 →
  area_covered = 16 →
  area_needed = 149 →
  (area_covered + area_needed) / width = 11 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l619_61989


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l619_61965

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 8 →
  (2 * a 4 - a 3 = a 3 - 4 * a 5) →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 31) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l619_61965


namespace NUMINAMATH_CALUDE_martin_initial_fruits_l619_61910

/-- The number of fruits Martin initially had --/
def initial_fruits : ℕ := 288

/-- The number of oranges Martin has after eating half his fruits --/
def oranges_after : ℕ := 50

/-- The number of apples Martin has after eating half his fruits --/
def apples_after : ℕ := 72

/-- The number of limes Martin has after eating half his fruits --/
def limes_after : ℕ := 24

theorem martin_initial_fruits :
  (initial_fruits / 2 = oranges_after + apples_after + limes_after) ∧
  (oranges_after = 2 * limes_after) ∧
  (apples_after = 3 * limes_after) ∧
  (oranges_after = 50) ∧
  (apples_after = 72) :=
by sorry

end NUMINAMATH_CALUDE_martin_initial_fruits_l619_61910


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l619_61971

theorem root_reciprocal_sum (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → 
  n^2 + 3*n - 1 = 0 → 
  m ≠ n →
  1/m + 1/n = 3 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l619_61971


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l619_61961

/-- Given line L1: 4x + 5y = 10 and line L2 perpendicular to L1 with y-intercept -3,
    prove that the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 10
  let L2 : ℝ → ℝ → Prop := λ x y ↦ ∃ m : ℝ, y = m * x - 3 ∧ m * (-4/5) = -1
  ∃ x : ℝ, L2 x 0 ∧ x = 12/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l619_61961


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l619_61978

theorem discount_percentage_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 128)
  (h2 : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l619_61978


namespace NUMINAMATH_CALUDE_gomoku_pieces_count_l619_61975

theorem gomoku_pieces_count :
  ∀ (initial_black : ℕ) (added_black : ℕ),
    initial_black > 0 →
    initial_black ≤ 5 →
    initial_black + added_black + (initial_black + (20 - added_black)) ≤ 30 →
    7 * (initial_black + (20 - added_black)) = 8 * (initial_black + added_black) →
    initial_black + added_black = 16 := by
  sorry

end NUMINAMATH_CALUDE_gomoku_pieces_count_l619_61975


namespace NUMINAMATH_CALUDE_simplify_expression_l619_61909

theorem simplify_expression (a : ℝ) : (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l619_61909


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l619_61991

/-- Given a right triangle ABC in the x-y plane where:
    - Angle B is 90 degrees
    - Length of AC is 25
    - Slope of line segment AC is 4/3
    Prove that the length of AB is 15 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- B is a right angle
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25) -- Length of AC is 25
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l619_61991


namespace NUMINAMATH_CALUDE_alices_favorite_number_l619_61941

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ,
    90 < n ∧ n < 150 ∧
    n % 13 = 0 ∧
    n % 4 ≠ 0 ∧
    digit_sum n % 4 = 0 ∧
    n = 143 := by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l619_61941


namespace NUMINAMATH_CALUDE_power_two_geq_square_l619_61979

theorem power_two_geq_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_geq_square_l619_61979


namespace NUMINAMATH_CALUDE_finite_rational_points_with_finite_orbit_l619_61927

noncomputable def f (C : ℚ) (x : ℚ) : ℚ := x^2 - C

def has_finite_orbit (C : ℚ) (x : ℚ) : Prop :=
  ∃ (S : Finset ℚ), ∀ n : ℕ, (f C)^[n] x ∈ S

theorem finite_rational_points_with_finite_orbit (C : ℚ) :
  {x : ℚ | has_finite_orbit C x}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_rational_points_with_finite_orbit_l619_61927


namespace NUMINAMATH_CALUDE_sum_reciprocals_factors_of_12_l619_61915

def factors_of_12 : List ℕ := [1, 2, 3, 4, 6, 12]

theorem sum_reciprocals_factors_of_12 :
  (factors_of_12.map (λ n => (1 : ℚ) / n)).sum = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_factors_of_12_l619_61915


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l619_61939

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 8 / 1 → a / b = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l619_61939


namespace NUMINAMATH_CALUDE_average_growth_rate_l619_61969

/-- The average monthly growth rate of CPI food prices -/
def x : ℝ := sorry

/-- The food price increase in January -/
def january_increase : ℝ := 0.028

/-- The predicted food price increase in February -/
def february_increase : ℝ := 0.02

/-- Theorem stating the relationship between the monthly increases and the average growth rate -/
theorem average_growth_rate : 
  (1 + january_increase) * (1 + february_increase) = (1 + x)^2 := by sorry

end NUMINAMATH_CALUDE_average_growth_rate_l619_61969


namespace NUMINAMATH_CALUDE_vacation_savings_time_l619_61925

/-- 
Given:
- goal_amount: The total amount needed for the vacation
- current_savings: The amount currently saved
- monthly_savings: The amount that can be saved each month

Prove that the number of months needed to reach the goal is 3.
-/
theorem vacation_savings_time (goal_amount current_savings monthly_savings : ℕ) 
  (h1 : goal_amount = 5000)
  (h2 : current_savings = 2900)
  (h3 : monthly_savings = 700) :
  (goal_amount - current_savings + monthly_savings - 1) / monthly_savings = 3 := by
  sorry


end NUMINAMATH_CALUDE_vacation_savings_time_l619_61925


namespace NUMINAMATH_CALUDE_extra_bananas_proof_l619_61929

/-- Calculates the number of extra bananas each child receives when some children are absent -/
def extra_bananas (total_children : ℕ) (absent_children : ℕ) : ℕ :=
  absent_children

theorem extra_bananas_proof (total_children : ℕ) (absent_children : ℕ) 
  (h1 : total_children = 700) 
  (h2 : absent_children = 350) :
  extra_bananas total_children absent_children = absent_children :=
by
  sorry

#eval extra_bananas 700 350

end NUMINAMATH_CALUDE_extra_bananas_proof_l619_61929


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l619_61968

theorem quadratic_roots_imply_m (m : ℚ) : 
  (∃ x : ℂ, 9 * x^2 + 5 * x + m = 0 ∧ 
   (x = (-5 + Complex.I * Real.sqrt 391) / 18 ∨ 
    x = (-5 - Complex.I * Real.sqrt 391) / 18)) → 
  m = 104 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l619_61968


namespace NUMINAMATH_CALUDE_sin_210_degrees_l619_61949

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l619_61949


namespace NUMINAMATH_CALUDE_equal_tire_usage_l619_61923

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  tire_miles : ℕ

/-- Theorem stating the correct tire usage for the given scenario -/
theorem equal_tire_usage (usage : TireUsage) 
  (h1 : usage.total_tires = 5)
  (h2 : usage.active_tires = 4)
  (h3 : usage.total_miles = 45000)
  (h4 : usage.tire_miles = usage.total_miles * usage.active_tires / usage.total_tires) :
  usage.tire_miles = 36000 := by
  sorry

#check equal_tire_usage

end NUMINAMATH_CALUDE_equal_tire_usage_l619_61923


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l619_61994

def p (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)
def q (x : ℝ) : Prop := x = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l619_61994


namespace NUMINAMATH_CALUDE_train_speed_problem_l619_61931

/-- Proves that the speed of the first train is 20 kmph given the problem conditions -/
theorem train_speed_problem (distance : ℝ) (speed_second : ℝ) (time_first : ℝ) (time_second : ℝ) 
  (h1 : distance = 200)
  (h2 : speed_second = 25)
  (h3 : time_first = 5)
  (h4 : time_second = 4) :
  ∃ (speed_first : ℝ), speed_first * time_first + speed_second * time_second = distance ∧ speed_first = 20 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l619_61931


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l619_61905

/-- The fixed point theorem for a parabola -/
theorem fixed_point_parabola 
  (p a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : b^2 ≠ 2*p*a) :
  ∃ C : ℝ × ℝ, 
    ∀ (M M₁ M₂ : ℝ × ℝ),
      (M.2)^2 = 2*p*M.1 →  -- M is on the parabola
      (M₁.2)^2 = 2*p*M₁.1 →  -- M₁ is on the parabola
      (M₂.2)^2 = 2*p*M₂.1 →  -- M₂ is on the parabola
      M₁ ≠ M →
      M₂ ≠ M →
      M₁ ≠ M₂ →
      (∃ t : ℝ, M₁.2 - b = t * (M₁.1 - a)) →  -- M₁ is on line AM
      (∃ t : ℝ, M₂.2 = t * (M₂.1 + a)) →  -- M₂ is on line BM
      (∃ t : ℝ, M₂.2 - M₁.2 = t * (M₂.1 - M₁.1)) →  -- M₁M₂ is a line
      C = (a, 2*p*a/b) ∧ 
      ∃ t : ℝ, C.2 - M₁.2 = t * (C.1 - M₁.1)  -- C is on line M₁M₂
  := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l619_61905


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l619_61982

def P : Set Nat := {1, 3, 6, 9}
def Q : Set Nat := {1, 2, 4, 6, 8}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l619_61982


namespace NUMINAMATH_CALUDE_intersection_of_sets_l619_61983

theorem intersection_of_sets : 
  let M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}
  let N : Set ℤ := {x | x^2 = x}
  M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l619_61983


namespace NUMINAMATH_CALUDE_mirror_area_l619_61912

/-- The area of a rectangular mirror that fits exactly inside a frame with given dimensions. -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) : 
  frame_length = 70 ∧ frame_width = 90 ∧ frame_thickness = 15 → 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 2400 :=
by
  sorry

#check mirror_area

end NUMINAMATH_CALUDE_mirror_area_l619_61912


namespace NUMINAMATH_CALUDE_cat_whiskers_count_l619_61944

/-- The number of whiskers on Princess Puff's face -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers on Catman Do's face -/
def catman_do_whiskers : ℕ := 2 * princess_puff_whiskers - 6

/-- The number of whiskers on Sir Whiskerson's face -/
def sir_whiskerson_whiskers : ℕ := princess_puff_whiskers + catman_do_whiskers + 8

/-- Theorem stating the correct number of whiskers for each cat -/
theorem cat_whiskers_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  sir_whiskerson_whiskers = 44 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_count_l619_61944


namespace NUMINAMATH_CALUDE_proportion_with_added_number_l619_61967

theorem proportion_with_added_number : 
  ∃ (x : ℚ), (1 : ℚ) / 3 = 4 / x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_proportion_with_added_number_l619_61967


namespace NUMINAMATH_CALUDE_propositions_truth_l619_61942

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a > b) → (a^2 > b^2)
def proposition2 (a b : ℝ) : Prop := (Real.log a = Real.log b) → (a = b)
def proposition3 (x y : ℝ) : Prop := (|x| = |y|) ↔ (x^2 = y^2)
def proposition4 (A B : ℝ) : Prop := (Real.sin A > Real.sin B) ↔ (A > B)

-- Theorem statement
theorem propositions_truth : 
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧ 
  (∃ a b : ℝ, Real.log a = Real.log b ∧ a ≠ b) ∧
  (∀ x y : ℝ, (|x| = |y|) ↔ (x^2 = y^2)) ∧
  (∀ A B : ℝ, 0 < A ∧ A < π ∧ 0 < B ∧ B < π → ((Real.sin A > Real.sin B) ↔ (A > B))) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l619_61942


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_243_l619_61996

theorem inverse_of_three_mod_243 : ∃ x : ℕ, x < 243 ∧ (3 * x) % 243 = 1 :=
by
  use 324
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_243_l619_61996


namespace NUMINAMATH_CALUDE_decorative_object_height_correct_l619_61903

/-- Represents a circular fountain with water jets -/
structure Fountain where
  diameter : ℝ
  max_height : ℝ
  max_height_distance : ℝ
  decorative_object_height : ℝ

/-- Properties of the specific fountain described in the problem -/
def problem_fountain : Fountain where
  diameter := 20
  max_height := 8
  max_height_distance := 2
  decorative_object_height := 7.5

/-- Theorem stating that the decorative object height is correct for the given fountain parameters -/
theorem decorative_object_height_correct (f : Fountain) 
  (h1 : f.diameter = 20)
  (h2 : f.max_height = 8)
  (h3 : f.max_height_distance = 2) :
  f.decorative_object_height = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_decorative_object_height_correct_l619_61903


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l619_61953

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x + y ≥ 2) ∧
  (∃ x y : ℝ, x + y ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l619_61953


namespace NUMINAMATH_CALUDE_sqrt_2_minus_x_real_range_l619_61914

theorem sqrt_2_minus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2_minus_x_real_range_l619_61914


namespace NUMINAMATH_CALUDE_max_cookies_in_class_l619_61948

/-- Represents the maximum number of cookies one student could have taken in a class. -/
def max_cookies_for_one_student (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ) : ℕ :=
  num_students * avg_cookies - (num_students - 1) * min_cookies

/-- Theorem stating the maximum number of cookies one student could have taken. -/
theorem max_cookies_in_class (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ)
    (h_num_students : num_students = 20)
    (h_avg_cookies : avg_cookies = 6)
    (h_min_cookies : min_cookies = 2) :
    max_cookies_for_one_student num_students avg_cookies min_cookies = 82 := by
  sorry

#eval max_cookies_for_one_student 20 6 2

end NUMINAMATH_CALUDE_max_cookies_in_class_l619_61948


namespace NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l619_61970

/-- The function f(x) = ax^2 + ax - 1 is always negative over the real numbers
    if and only if a is in the range (-4, 0]. -/
theorem function_always_negative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l619_61970


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l619_61992

theorem product_and_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a * b = 4) (h2 : 1 / a = 3 / b) : a + b = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l619_61992


namespace NUMINAMATH_CALUDE_no_valid_solution_l619_61962

/-- Represents the conditions of the age problem -/
structure AgeProblem where
  jane_current_age : ℕ
  dick_current_age : ℕ
  n : ℕ
  jane_future_age : ℕ
  dick_future_age : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfies_conditions (problem : AgeProblem) : Prop :=
  problem.jane_current_age = 30 ∧
  problem.dick_current_age = problem.jane_current_age + 5 ∧
  problem.n > 0 ∧
  problem.jane_future_age = problem.jane_current_age + problem.n ∧
  problem.dick_future_age = problem.dick_current_age + problem.n ∧
  10 ≤ problem.jane_future_age ∧ problem.jane_future_age ≤ 99 ∧
  10 ≤ problem.dick_future_age ∧ problem.dick_future_age ≤ 99 ∧
  (problem.jane_future_age / 10 = problem.dick_future_age % 10) ∧
  (problem.jane_future_age % 10 = problem.dick_future_age / 10)

/-- The main theorem stating that no valid solution exists -/
theorem no_valid_solution : ¬∃ (problem : AgeProblem), satisfies_conditions problem := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l619_61962


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l619_61976

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + k*x₁ + 1 = 3*x₁ + k) ∧ 
  (x₂^2 + k*x₂ + 1 = 3*x₂ + k) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l619_61976


namespace NUMINAMATH_CALUDE_chris_hockey_stick_cost_l619_61999

/-- The cost of a hockey stick, given the total spent, helmet cost, and number of sticks. -/
def hockey_stick_cost (total_spent helmet_cost num_sticks : ℚ) : ℚ :=
  (total_spent - helmet_cost) / num_sticks

/-- Theorem stating the cost of one hockey stick given Chris's purchase. -/
theorem chris_hockey_stick_cost :
  let total_spent : ℚ := 68
  let helmet_cost : ℚ := 25
  let num_sticks : ℚ := 2
  hockey_stick_cost total_spent helmet_cost num_sticks = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_chris_hockey_stick_cost_l619_61999


namespace NUMINAMATH_CALUDE_garden_length_proof_l619_61951

/-- Represents a rectangular garden with its dimensions. -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_length_proof (g : RectangularGarden) 
  (h1 : perimeter g = 600) 
  (h2 : g.breadth = 95) : 
  g.length = 205 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_proof_l619_61951


namespace NUMINAMATH_CALUDE_compound_weight_l619_61919

theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 352 → moles = 8 → moles * molecular_weight = 2816 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l619_61919


namespace NUMINAMATH_CALUDE_sum_of_F_at_4_and_neg_2_l619_61954

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 2)) + (10 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem sum_of_F_at_4_and_neg_2 : F 4 + F (-2) = Real.sqrt 6 + 3.529 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_F_at_4_and_neg_2_l619_61954


namespace NUMINAMATH_CALUDE_triangle_condition_implies_linear_l619_61997

/-- A function satisfying the triangle condition -/
def TriangleCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a ≠ b → b ≠ c → a ≠ c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- The main theorem statement -/
theorem triangle_condition_implies_linear (f : ℝ → ℝ) (h : TriangleCondition f) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x = c * x :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_linear_l619_61997


namespace NUMINAMATH_CALUDE_mold_cost_is_250_l619_61911

/-- The cost of a mold for handmade shoes --/
def mold_cost (hourly_rate : ℝ) (hours : ℝ) (work_percentage : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - work_percentage * hourly_rate * hours

/-- Proves that the cost of the mold is $250 given the problem conditions --/
theorem mold_cost_is_250 :
  mold_cost 75 8 0.8 730 = 250 := by
  sorry

end NUMINAMATH_CALUDE_mold_cost_is_250_l619_61911


namespace NUMINAMATH_CALUDE_inequality_proof_l619_61928

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l619_61928


namespace NUMINAMATH_CALUDE_greatest_sum_36_l619_61977

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that n is the greatest number of consecutive positive integers starting from 1 whose sum is 36 -/
def is_greatest_sum_36 (n : ℕ) : Prop :=
  sum_first_n n = 36 ∧ ∀ m : ℕ, m > n → sum_first_n m > 36

theorem greatest_sum_36 : is_greatest_sum_36 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_36_l619_61977


namespace NUMINAMATH_CALUDE_functional_equation_solution_l619_61936

/-- A function satisfying the given functional equation and differentiability condition -/
class FunctionalEquationSolution (f : ℝ → ℝ) : Prop where
  equation : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
  smooth : ContDiff ℝ ⊤ f

/-- The main theorem stating the form of the solution -/
theorem functional_equation_solution (f : ℝ → ℝ) [FunctionalEquationSolution f] :
  ∃ a : ℝ, ∀ x : ℝ, f x = x^2 + a * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l619_61936


namespace NUMINAMATH_CALUDE_swimming_pool_payment_analysis_l619_61986

/-- Represents the swimming pool payment methods -/
structure SwimmingPoolPayment where
  membershipCost : ℕ
  memberSwimCost : ℕ
  nonMemberSwimCost : ℕ

/-- Calculates the cost for a given number of swims using Method 1 -/
def method1Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.membershipCost + p.memberSwimCost * swims

/-- Calculates the cost for a given number of swims using Method 2 -/
def method2Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.nonMemberSwimCost * swims

/-- Calculates the maximum number of swims possible with a given budget using Method 1 -/
def maxSwimMethod1 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  (budget - p.membershipCost) / p.memberSwimCost

/-- Calculates the maximum number of swims possible with a given budget using Method 2 -/
def maxSwimMethod2 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  budget / p.nonMemberSwimCost

theorem swimming_pool_payment_analysis 
  (p : SwimmingPoolPayment) 
  (h1 : p.membershipCost = 200)
  (h2 : p.memberSwimCost = 10)
  (h3 : p.nonMemberSwimCost = 30) :
  (method1Cost p 3 = 230) ∧
  (method2Cost p 9 < method1Cost p 9) ∧
  (maxSwimMethod1 p 600 > maxSwimMethod2 p 600) := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_payment_analysis_l619_61986


namespace NUMINAMATH_CALUDE_mitzel_spending_l619_61904

/-- Proves that Mitzel spent $14, given the conditions of the problem -/
theorem mitzel_spending (allowance : ℝ) (spent_percentage : ℝ) (remaining : ℝ) : 
  spent_percentage = 0.35 →
  remaining = 26 →
  (1 - spent_percentage) * allowance = remaining →
  spent_percentage * allowance = 14 := by
  sorry

end NUMINAMATH_CALUDE_mitzel_spending_l619_61904


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l619_61972

theorem smallest_cube_root_with_small_fraction (n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (∃ m : ℕ, (n + r)^3 = m) →
  (∀ k < n, ∀ s : ℝ, (0 < s) → (s < 1 / 500) → (∃ l : ℕ, (k + s)^3 = l) → False) →
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l619_61972


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l619_61950

/-- The value of a cow in dollars -/
def cow_value : ℕ := 400

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 280

/-- A debt is resolvable if it can be expressed as a linear combination of cow and sheep values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (c s : ℤ), debt = c * cow_value + s * sheep_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l619_61950


namespace NUMINAMATH_CALUDE_kaleb_remaining_chocolates_l619_61902

def boxes_bought : ℕ := 14
def pieces_per_box : ℕ := 6
def boxes_given_away : ℕ := 5 + 2 + 3

def remaining_boxes : ℕ := boxes_bought - boxes_given_away
def remaining_pieces : ℕ := remaining_boxes * pieces_per_box

def eaten_pieces : ℕ := (remaining_pieces * 10) / 100

theorem kaleb_remaining_chocolates :
  remaining_pieces - eaten_pieces = 22 := by sorry

end NUMINAMATH_CALUDE_kaleb_remaining_chocolates_l619_61902


namespace NUMINAMATH_CALUDE_expression_simplification_l619_61917

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 2/3) : 
  ((x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 3*x*(2*x - y)) / (2*x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l619_61917


namespace NUMINAMATH_CALUDE_cats_needed_to_reach_goal_l619_61906

theorem cats_needed_to_reach_goal (current_cats goal_cats : ℕ) : 
  current_cats = 11 → goal_cats = 43 → goal_cats - current_cats = 32 := by
sorry

end NUMINAMATH_CALUDE_cats_needed_to_reach_goal_l619_61906


namespace NUMINAMATH_CALUDE_sqrt_3x_lt_5x_iff_l619_61955

theorem sqrt_3x_lt_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt (3 * x) < 5 * x ↔ x > 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_lt_5x_iff_l619_61955


namespace NUMINAMATH_CALUDE_mixture_weight_l619_61924

/-- The weight of a mixture of green tea and coffee given specific price changes and costs. -/
theorem mixture_weight (june_cost green_tea_july coffee_july mixture_cost : ℝ) : 
  june_cost > 0 →
  green_tea_july = 0.1 * june_cost →
  coffee_july = 2 * june_cost →
  mixture_cost = 3.15 →
  (mixture_cost / ((green_tea_july + coffee_july) / 2)) = 3 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l619_61924


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l619_61956

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (h1 : l > w) (h2 : l > 0) (h3 : w > 0) (h4 : h > 0) :
  let box_base_area := l * w
  let box_side_area := 2 * (l * h + w * h)
  let corner_area := 4 * h^2
  box_base_area + box_side_area + corner_area = wrapping_paper_area l w h :=
by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l619_61956


namespace NUMINAMATH_CALUDE_simplify_fraction_l619_61938

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l619_61938


namespace NUMINAMATH_CALUDE_exists_line_with_three_colors_l619_61913

/-- A color type with four possible values -/
inductive Color
  | One
  | Two
  | Three
  | Four

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- A predicate that checks if a coloring uses all four colors -/
def uses_all_colors (f : Coloring) : Prop :=
  (∃ p : Point, f p = Color.One) ∧
  (∃ p : Point, f p = Color.Two) ∧
  (∃ p : Point, f p = Color.Three) ∧
  (∃ p : Point, f p = Color.Four)

/-- A predicate that checks if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem exists_line_with_three_colors (f : Coloring) (h : uses_all_colors f) :
  ∃ l : Line, ∃ p₁ p₂ p₃ : Point,
    on_line p₁ l ∧ on_line p₂ l ∧ on_line p₃ l ∧
    f p₁ ≠ f p₂ ∧ f p₁ ≠ f p₃ ∧ f p₂ ≠ f p₃ :=
sorry

end NUMINAMATH_CALUDE_exists_line_with_three_colors_l619_61913


namespace NUMINAMATH_CALUDE_f_min_at_3_l619_61990

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_3 : ∀ x : ℝ, f 3 ≤ f x := by sorry

end NUMINAMATH_CALUDE_f_min_at_3_l619_61990
