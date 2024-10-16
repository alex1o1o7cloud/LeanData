import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_problem_l2120_212014

theorem certain_number_problem (a : ℕ) (certain_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * 45 * certain_number) :
  certain_number = 49 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2120_212014


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l2120_212055

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 6's when rolling three standard dice -/
def expected_sixes : ℚ :=
  0 * (prob_not_six ^ 3) +
  1 * (3 * prob_six * prob_not_six ^ 2) +
  2 * (3 * prob_six ^ 2 * prob_not_six) +
  3 * (prob_six ^ 3)

theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_half_l2120_212055


namespace NUMINAMATH_CALUDE_number_of_boys_l2120_212076

theorem number_of_boys (M W B : ℕ) : 
  M = W → 
  W = B → 
  M * 8 = 120 → 
  B = 15 := by sorry

end NUMINAMATH_CALUDE_number_of_boys_l2120_212076


namespace NUMINAMATH_CALUDE_least_possible_QR_length_l2120_212078

theorem least_possible_QR_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7)
  (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 24) :
  ∃ (QR : ℕ), QR ≥ 14 ∧ ∀ (n : ℕ), n ≥ 14 → 
  (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > QS - SR := by
  sorry

end NUMINAMATH_CALUDE_least_possible_QR_length_l2120_212078


namespace NUMINAMATH_CALUDE_two_green_then_red_probability_l2120_212034

/-- The number of traffic checkpoints -/
def num_checkpoints : ℕ := 6

/-- The probability of encountering a red light at each checkpoint -/
def red_light_prob : ℚ := 1/3

/-- The probability of passing exactly two checkpoints before encountering a red light -/
def prob_two_green_then_red : ℚ := 4/27

theorem two_green_then_red_probability :
  (1 - red_light_prob)^2 * red_light_prob = prob_two_green_then_red :=
sorry

end NUMINAMATH_CALUDE_two_green_then_red_probability_l2120_212034


namespace NUMINAMATH_CALUDE_line_point_distance_condition_l2120_212012

theorem line_point_distance_condition (a : ℝ) : 
  (∃ x y : ℝ, a * x + y + 2 = 0 ∧ 
    ((x + 3)^2 + y^2)^(1/2) = 2 * (x^2 + y^2)^(1/2)) → 
  a ≤ 0 ∨ a ≥ 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_point_distance_condition_l2120_212012


namespace NUMINAMATH_CALUDE_johns_new_total_capacity_l2120_212073

/-- Represents the lifting capacities of a weightlifter -/
structure LiftingCapacities where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improvedCapacities (initial : LiftingCapacities) : LiftingCapacities :=
  { cleanAndJerk := initial.cleanAndJerk * 2,
    snatch := initial.snatch * 1.8 }

/-- Calculates the total lifting capacity -/
def totalCapacity (capacities : LiftingCapacities) : ℝ :=
  capacities.cleanAndJerk + capacities.snatch

/-- John's initial lifting capacities -/
def johnsInitialCapacities : LiftingCapacities :=
  { cleanAndJerk := 80,
    snatch := 50 }

theorem johns_new_total_capacity :
  totalCapacity (improvedCapacities johnsInitialCapacities) = 250 := by
  sorry


end NUMINAMATH_CALUDE_johns_new_total_capacity_l2120_212073


namespace NUMINAMATH_CALUDE_horner_first_step_value_v₁_equals_30_l2120_212017

/-- Horner's Rule first step for polynomial evaluation -/
def horner_first_step (a₄ a₃ : ℝ) (x : ℝ) : ℝ :=
  a₄ * x + a₃

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ :=
  3 * x^4 + 2 * x^2 + x + 4

theorem horner_first_step_value :
  horner_first_step 3 0 10 = 30 :=
by sorry

theorem v₁_equals_30 :
  horner_first_step 3 0 10 = 30 :=
by sorry

end NUMINAMATH_CALUDE_horner_first_step_value_v₁_equals_30_l2120_212017


namespace NUMINAMATH_CALUDE_trapezoid_is_plane_figure_l2120_212022

-- Define a trapezoid
structure Trapezoid :=
  (hasParallelLines : Bool)

-- Define a plane figure
structure PlaneFigure

-- Theorem: A trapezoid is a plane figure
theorem trapezoid_is_plane_figure (t : Trapezoid) (h : t.hasParallelLines = true) : PlaneFigure :=
sorry

end NUMINAMATH_CALUDE_trapezoid_is_plane_figure_l2120_212022


namespace NUMINAMATH_CALUDE_dice_product_six_prob_l2120_212025

/-- The probability of rolling a specific number on a standard die -/
def die_prob : ℚ := 1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The set of favorable outcomes where the product of the three numbers is 6 -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The probability of rolling three dice such that their product is 6 -/
theorem dice_product_six_prob : 
  (Finset.card favorable_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_dice_product_six_prob_l2120_212025


namespace NUMINAMATH_CALUDE_biology_marks_calculation_l2120_212033

def english_marks : ℕ := 96
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem biology_marks_calculation :
  ∃ (biology_marks : ℕ),
    biology_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks) ∧
    biology_marks = 85 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_calculation_l2120_212033


namespace NUMINAMATH_CALUDE_calculator_game_sum_l2120_212031

/-- Represents the state of the calculators -/
structure CalculatorState :=
  (calc1 : ℤ)
  (calc2 : ℤ)
  (calc3 : ℤ)

/-- The operation performed on the calculators in each turn -/
def squareOperation (state : CalculatorState) : CalculatorState :=
  { calc1 := state.calc1 ^ 2,
    calc2 := state.calc2 ^ 2,
    calc3 := state.calc3 ^ 2 }

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2,
    calc2 := -2,
    calc3 := 0 }

/-- The theorem to be proved -/
theorem calculator_game_sum (n : ℕ) (h : n ≥ 1) :
  (squareOperation^[n] initialState).calc1 +
  (squareOperation^[n] initialState).calc2 +
  (squareOperation^[n] initialState).calc3 = 8 :=
sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l2120_212031


namespace NUMINAMATH_CALUDE_combinatorial_sum_equality_l2120_212083

theorem combinatorial_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (λ j => Nat.choose k j * Nat.choose n (m - j)) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_sum_equality_l2120_212083


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l2120_212074

/-- Calculates Bhanu's petrol expense given his house rent expense and spending percentages -/
theorem bhanu_petrol_expense (house_rent : ℝ) (petrol_percent : ℝ) (rent_percent : ℝ) : 
  house_rent = 140 → 
  petrol_percent = 0.3 → 
  rent_percent = 0.2 → 
  ∃ (total_income : ℝ), 
    total_income > 0 ∧ 
    rent_percent * (1 - petrol_percent) * total_income = house_rent ∧
    petrol_percent * total_income = 300 :=
by sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l2120_212074


namespace NUMINAMATH_CALUDE_only_234_not_right_triangle_l2120_212002

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 3, 4) is not a right triangle --/
theorem only_234_not_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 1 1 (Real.sqrt 2)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 3) (Real.sqrt 5)) ∧
  (is_right_triangle 3 4 5) :=
by sorry


end NUMINAMATH_CALUDE_only_234_not_right_triangle_l2120_212002


namespace NUMINAMATH_CALUDE_fair_coin_prob_diff_l2120_212072

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The probability of getting exactly 3 heads in 4 flips of a fair coin -/
def prob_3_heads : ℚ := prob_k_heads 4 3

/-- The probability of getting 4 heads in 4 flips of a fair coin -/
def prob_4_heads : ℚ := prob_k_heads 4 4

/-- The positive difference between the probability of exactly 3 heads
    and the probability of 4 heads in 4 flips of a fair coin -/
theorem fair_coin_prob_diff : prob_3_heads - prob_4_heads = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_prob_diff_l2120_212072


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2120_212084

/-- Given a triangle ABC with an arbitrary point inside it, and three lines drawn through
    this point parallel to the sides of the triangle, dividing it into six parts including
    three triangles with areas S₁, S₂, and S₃, the area of triangle ABC is (√S₁ + √S₂ + √S₃)². -/
theorem triangle_area_theorem (S₁ S₂ S₃ : ℝ) (h₁ : 0 < S₁) (h₂ : 0 < S₂) (h₃ : 0 < S₃) :
  ∃ (S : ℝ), S > 0 ∧ S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2120_212084


namespace NUMINAMATH_CALUDE_distance_of_problem_lines_l2120_212021

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on the first line
  b : ℝ × ℝ  -- Point on the second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { a := (3, -2)
    b := (5, -1)
    d := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = 2 * Real.sqrt 109 / 29 :=
sorry

end NUMINAMATH_CALUDE_distance_of_problem_lines_l2120_212021


namespace NUMINAMATH_CALUDE_justin_total_pages_justin_first_book_pages_justin_second_book_pages_l2120_212054

/-- Represents the reading schedule for a week -/
structure ReadingSchedule where
  firstBookDay1 : ℕ
  secondBookDay1 : ℕ
  firstBookIncrement : ℕ → ℕ
  secondBookIncrement : ℕ
  firstBookBreakDay : ℕ
  secondBookBreakDay : ℕ

/-- Calculates the total pages read for both books in a week -/
def totalPagesRead (schedule : ReadingSchedule) : ℕ := 
  let firstBookPages := schedule.firstBookDay1 + 
    (schedule.firstBookDay1 * 2) + 
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 3) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 4) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 5) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 6)
  let secondBookPages := schedule.secondBookDay1 + 
    (schedule.secondBookDay1 + schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 2 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 3 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 4 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 5 * schedule.secondBookIncrement)
  firstBookPages + secondBookPages

/-- Justin's reading schedule -/
def justinSchedule : ReadingSchedule := {
  firstBookDay1 := 10,
  secondBookDay1 := 15,
  firstBookIncrement := λ n => 5 * (n - 2),
  secondBookIncrement := 3,
  firstBookBreakDay := 7,
  secondBookBreakDay := 4
}

/-- Theorem stating that Justin reads 295 pages in total -/
theorem justin_total_pages : totalPagesRead justinSchedule = 295 := by
  sorry

/-- Theorem stating that Justin reads 160 pages of the first book -/
theorem justin_first_book_pages : 
  justinSchedule.firstBookDay1 + 
  (justinSchedule.firstBookDay1 * 2) + 
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 3) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 4) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 5) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 6) = 160 := by
  sorry

/-- Theorem stating that Justin reads 135 pages of the second book -/
theorem justin_second_book_pages :
  justinSchedule.secondBookDay1 + 
  (justinSchedule.secondBookDay1 + justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 2 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 3 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 4 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 5 * justinSchedule.secondBookIncrement) = 135 := by
  sorry

end NUMINAMATH_CALUDE_justin_total_pages_justin_first_book_pages_justin_second_book_pages_l2120_212054


namespace NUMINAMATH_CALUDE_polynomial_integer_solutions_l2120_212063

theorem polynomial_integer_solutions :
  ∀ n : ℤ, n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_solutions_l2120_212063


namespace NUMINAMATH_CALUDE_books_obtained_l2120_212060

/-- The number of additional books obtained by the class -/
def additional_books (initial final : ℕ) : ℕ := final - initial

/-- Proves that the number of additional books is 23 given the initial and final counts -/
theorem books_obtained (initial final : ℕ) 
  (h_initial : initial = 54)
  (h_final : final = 77) :
  additional_books initial final = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_obtained_l2120_212060


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2120_212086

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 3*x - (1 - 2*x)
  (f 1 = 0) ∧ (f (-1/2) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2120_212086


namespace NUMINAMATH_CALUDE_magazine_cost_l2120_212097

/-- The cost of a magazine and pencil, given specific conditions -/
theorem magazine_cost (pencil_cost coupon_value total_spent : ℚ) :
  pencil_cost = 0.5 →
  coupon_value = 0.35 →
  total_spent = 1 →
  ∃ (magazine_cost : ℚ),
    magazine_cost + pencil_cost - coupon_value = total_spent ∧
    magazine_cost = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_l2120_212097


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2120_212070

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_diff : a 2 - a 1 = 2)
  (h_arithmetic : 2 * a 2 = (3 * a 1 + a 3) / 2) :
  a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2120_212070


namespace NUMINAMATH_CALUDE_least_x_for_integer_fraction_l2120_212019

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem least_x_for_integer_fraction :
  ∀ x : ℝ, (is_integer (24 / (x - 4)) ∧ x < -20) → False :=
by sorry

end NUMINAMATH_CALUDE_least_x_for_integer_fraction_l2120_212019


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2120_212043

theorem solution_of_linear_equation (x y m : ℝ) : 
  x = 1 → y = m → 3 * x - 4 * y = 7 → m = -1 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2120_212043


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2120_212089

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2120_212089


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l2120_212020

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the octal number
def octal_num : ℕ := 0o55

-- Theorem statement
theorem binary_to_octal_conversion :
  binary_num = octal_num := by sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l2120_212020


namespace NUMINAMATH_CALUDE_projection_vector_l2120_212008

/-- Given vectors a and b in ℝ², prove that the projection of a onto b is equal to the expected result. -/
theorem projection_vector (a b : ℝ × ℝ) (ha : a = (2, 4)) (hb : b = (-1, 2)) :
  let proj := (((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.1,
               ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.2)
  proj = (-6/5, 12/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l2120_212008


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2120_212067

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2120_212067


namespace NUMINAMATH_CALUDE_camilo_kenny_difference_l2120_212010

def paint_house_problem (judson_contribution kenny_contribution camilo_contribution total_cost : ℕ) : Prop :=
  judson_contribution = 500 ∧
  kenny_contribution = judson_contribution + judson_contribution / 5 ∧
  camilo_contribution > kenny_contribution ∧
  total_cost = 1900 ∧
  judson_contribution + kenny_contribution + camilo_contribution = total_cost

theorem camilo_kenny_difference :
  ∀ judson_contribution kenny_contribution camilo_contribution total_cost,
    paint_house_problem judson_contribution kenny_contribution camilo_contribution total_cost →
    camilo_contribution - kenny_contribution = 200 := by
  sorry

end NUMINAMATH_CALUDE_camilo_kenny_difference_l2120_212010


namespace NUMINAMATH_CALUDE_eight_times_one_seventh_squared_l2120_212098

theorem eight_times_one_seventh_squared : 8 * (1 / 7)^2 = 8 / 49 := by
  sorry

end NUMINAMATH_CALUDE_eight_times_one_seventh_squared_l2120_212098


namespace NUMINAMATH_CALUDE_partition_6_5_l2120_212066

/-- The number of ways to partition n into at most k non-negative integer parts -/
def num_partitions (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition 6 into at most 5 non-negative integer parts -/
theorem partition_6_5 : num_partitions 6 5 = 11 := by sorry

end NUMINAMATH_CALUDE_partition_6_5_l2120_212066


namespace NUMINAMATH_CALUDE_seashells_count_l2120_212081

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l2120_212081


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l2120_212079

theorem base_4_9_digit_difference :
  let n : ℕ := 1234
  let base_4_digits := (Nat.log n 4).succ
  let base_9_digits := (Nat.log n 9).succ
  base_4_digits = base_9_digits + 2 :=
by sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l2120_212079


namespace NUMINAMATH_CALUDE_division_problem_l2120_212024

theorem division_problem (dividend divisor : ℕ) (h1 : dividend + divisor = 136) (h2 : dividend / divisor = 7) : divisor = 17 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2120_212024


namespace NUMINAMATH_CALUDE_solve_for_x_l2120_212016

theorem solve_for_x (x y : ℝ) (eq1 : x + 3 * y = 10) (eq2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2120_212016


namespace NUMINAMATH_CALUDE_correct_payments_l2120_212075

/-- Represents the weekly payments to three employees --/
structure EmployeePayments where
  total : ℕ
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given payments satisfy the problem conditions --/
def isValidPayment (p : EmployeePayments) : Prop :=
  p.total = 1500 ∧
  p.a = (150 * p.b) / 100 ∧
  p.c = (80 * p.b) / 100 ∧
  p.a + p.b + p.c = p.total

/-- The theorem stating the correct payments --/
theorem correct_payments :
  ∃ (p : EmployeePayments), isValidPayment p ∧ p.a = 682 ∧ p.b = 454 ∧ p.c = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_payments_l2120_212075


namespace NUMINAMATH_CALUDE_simson_line_l2120_212028

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the properties and relations
variable (incircle : Point → Point → Point → Point → Prop)
variable (on_circle : Point → Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (collinear : Point → Point → Point → Prop)

-- Define the theorem
theorem simson_line 
  (A B C P U V W : Point) 
  (circle : Line) 
  (BC CA AB : Line) :
  incircle A B C P →
  on_circle A B C P →
  perpendicular P U BC →
  perpendicular P V CA →
  perpendicular P W AB →
  on_line U BC →
  on_line V CA →
  on_line W AB →
  collinear U V W :=
sorry

end NUMINAMATH_CALUDE_simson_line_l2120_212028


namespace NUMINAMATH_CALUDE_equation_solution_l2120_212004

theorem equation_solution (a b : ℝ) :
  (∀ x : ℝ, (a*x^2 + b*x - 5)*(a*x^2 + b*x + 25) + c = (a*x^2 + b*x + 10)^2) →
  c = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2120_212004


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2120_212035

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 4*a + 1

theorem quadratic_root_range (a : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁ < -1 ∧ r₂ > 3 ∧ f a r₁ = 0 ∧ f a r₂ = 0) →
  a > 4/5 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2120_212035


namespace NUMINAMATH_CALUDE_sin_negative_780_degrees_l2120_212036

theorem sin_negative_780_degrees : 
  Real.sin ((-780 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_780_degrees_l2120_212036


namespace NUMINAMATH_CALUDE_students_without_portraits_l2120_212093

theorem students_without_portraits (total_students : ℕ) 
  (before_break : ℕ) (during_break : ℕ) (after_lunch : ℕ) : 
  total_students = 60 →
  before_break = total_students / 4 →
  during_break = (total_students - before_break) / 3 →
  after_lunch = 10 →
  total_students - (before_break + during_break + after_lunch) = 20 := by
sorry

end NUMINAMATH_CALUDE_students_without_portraits_l2120_212093


namespace NUMINAMATH_CALUDE_three_number_problem_l2120_212040

theorem three_number_problem (a b c : ℤ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 47)
  (sum_ca : c + a = 52) :
  (a + b + c = 67) ∧ (a * b * c = 9600) := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l2120_212040


namespace NUMINAMATH_CALUDE_cycle_price_problem_l2120_212032

theorem cycle_price_problem (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1170)
  (h2 : gain_percent = 30) :
  let original_price := selling_price / (1 + gain_percent / 100)
  original_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cycle_price_problem_l2120_212032


namespace NUMINAMATH_CALUDE_total_is_527_given_shares_inconsistent_l2120_212029

/-- Represents the shares of money for three individuals --/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Calculates the total amount from given shares --/
def total_amount (s : Shares) : ℕ := s.a + s.b + s.c

/-- The given shares --/
def given_shares : Shares := ⟨372, 93, 62⟩

/-- Theorem stating that the total amount is 527 --/
theorem total_is_527 : total_amount given_shares = 527 := by
  sorry

/-- Property that should hold for the shares based on the problem statement --/
def shares_property (s : Shares) : Prop :=
  s.a = (2 * s.b) / 3 ∧ s.b = s.c / 4

/-- Theorem stating that the given shares do not satisfy the problem's conditions --/
theorem given_shares_inconsistent : ¬ shares_property given_shares := by
  sorry

end NUMINAMATH_CALUDE_total_is_527_given_shares_inconsistent_l2120_212029


namespace NUMINAMATH_CALUDE_power_inequality_l2120_212096

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a*b)^((a+b)/2) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2120_212096


namespace NUMINAMATH_CALUDE_chim_tu_survival_days_l2120_212077

/-- The number of distinct T-shirts --/
def n : ℕ := 4

/-- The number of days between outfit changes --/
def days_per_outfit : ℕ := 3

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of distinct outfits with exactly k T-shirts --/
def outfits_with_k (k : ℕ) : ℕ := choose n k * factorial k

/-- The total number of distinct outfits --/
def total_outfits : ℕ := outfits_with_k 3 + outfits_with_k 4

/-- The number of days Chim Tu can wear a unique outfit --/
def survival_days : ℕ := total_outfits * days_per_outfit

theorem chim_tu_survival_days : survival_days = 144 := by
  sorry

end NUMINAMATH_CALUDE_chim_tu_survival_days_l2120_212077


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l2120_212005

theorem set_equality_implies_sum (a b : ℝ) (ha : a ≠ 0) :
  ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0} →
  a^2015 + b^2016 = -1 := by sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l2120_212005


namespace NUMINAMATH_CALUDE_diagonal_length_l2120_212059

/-- A quadrilateral with specific side lengths and an integer diagonal -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  AC : ℤ
  h1 : AB = 9
  h2 : BC = 2
  h3 : CD = 14
  h4 : DA = 5

/-- The diagonal AC of the quadrilateral is 10 -/
theorem diagonal_length (q : Quadrilateral) : q.AC = 10 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l2120_212059


namespace NUMINAMATH_CALUDE_handshakes_count_l2120_212046

/-- Represents the number of handshakes in a gathering of twins and triplets -/
def handshakes (twin_sets triplet_sets : ℕ) : ℕ :=
  let twins := 2 * twin_sets
  let triplets := 3 * triplet_sets
  let twin_handshakes := twins * (twins - 2) / 2
  let cross_handshakes := twins * triplets
  twin_handshakes + cross_handshakes

/-- Theorem stating that the number of handshakes in the given scenario is 352 -/
theorem handshakes_count : handshakes 8 5 = 352 := by
  sorry

#eval handshakes 8 5

end NUMINAMATH_CALUDE_handshakes_count_l2120_212046


namespace NUMINAMATH_CALUDE_wheat_flour_amount_l2120_212092

/-- The amount of wheat flour used by the bakery -/
def wheat_flour : ℝ := sorry

/-- The amount of white flour used by the bakery -/
def white_flour : ℝ := 0.1

/-- The total amount of flour used by the bakery -/
def total_flour : ℝ := 0.3

/-- Theorem stating that the amount of wheat flour used is 0.2 bags -/
theorem wheat_flour_amount : wheat_flour = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_wheat_flour_amount_l2120_212092


namespace NUMINAMATH_CALUDE_second_fold_perpendicular_l2120_212087

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a sheet of paper with one straight edge -/
structure Paper :=
  (straight_edge : Line)

/-- Represents a fold on the paper -/
structure Fold :=
  (line : Line)
  (paper : Paper)

/-- Checks if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Theorem: The second fold creates a line perpendicular to the initial crease -/
theorem second_fold_perpendicular 
  (paper : Paper) 
  (initial_fold : Fold)
  (A : Point)
  (second_fold : Fold)
  (h1 : point_on_line A paper.straight_edge)
  (h2 : point_on_line A initial_fold.line)
  (h3 : point_on_line A second_fold.line)
  (h4 : ∃ (p q : Point), 
    point_on_line p paper.straight_edge ∧ 
    point_on_line q paper.straight_edge ∧
    point_on_line p second_fold.line ∧
    point_on_line q initial_fold.line) :
  perpendicular initial_fold.line second_fold.line :=
sorry

end NUMINAMATH_CALUDE_second_fold_perpendicular_l2120_212087


namespace NUMINAMATH_CALUDE_maria_coin_difference_l2120_212094

/-- Represents the number of coins of each denomination -/
structure CoinCollection where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ
  twenty_five_cent : ℕ

/-- The conditions of Maria's coin collection -/
def maria_collection (c : CoinCollection) : Prop :=
  c.five_cent + c.ten_cent + c.twenty_cent + c.twenty_five_cent = 30 ∧
  c.ten_cent = 2 * c.five_cent ∧
  5 * c.five_cent + 10 * c.ten_cent + 20 * c.twenty_cent + 25 * c.twenty_five_cent = 410

theorem maria_coin_difference (c : CoinCollection) : 
  maria_collection c → c.twenty_five_cent - c.twenty_cent = 1 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_difference_l2120_212094


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l2120_212011

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- Defines the conditions given in the problem -/
def satisfiesConditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- Theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : BrotherAges) :
  satisfiesConditions ages → ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l2120_212011


namespace NUMINAMATH_CALUDE_inverse_of_complex_l2120_212065

theorem inverse_of_complex (z : ℂ) : z = 1 - 2 * I → z⁻¹ = (1 / 5 : ℂ) + (2 / 5 : ℂ) * I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_l2120_212065


namespace NUMINAMATH_CALUDE_subset_condition_l2120_212057

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_condition (a : ℝ) : A a ⊆ (A a ∩ B) ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2120_212057


namespace NUMINAMATH_CALUDE_factorization_equality_l2120_212000

theorem factorization_equality (a b : ℝ) : (a - b)^2 - (b - a) = (a - b) * ((a - b) + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2120_212000


namespace NUMINAMATH_CALUDE_sphere_in_cone_angle_l2120_212018

/-- Given a sphere inscribed in a cone, if the circle of tangency divides the surface
    of the sphere in the ratio of 1:4, then the angle between the generatrix of the cone
    and its base plane is arccos(3/5). -/
theorem sphere_in_cone_angle (R : ℝ) (α : ℝ) :
  R > 0 →  -- Radius is positive
  (2 * π * R^2 * (1 - Real.cos α)) / (4 * π * R^2) = 1/5 →  -- Surface area ratio condition
  α = Real.arccos (3/5) :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_angle_l2120_212018


namespace NUMINAMATH_CALUDE_die_roll_probability_l2120_212061

/-- The probability of getting a specific number on a standard six-sided die -/
def prob_match : ℚ := 1 / 6

/-- The probability of not getting a specific number on a standard six-sided die -/
def prob_no_match : ℚ := 5 / 6

/-- The number of rolls -/
def n : ℕ := 12

/-- The number of ways to choose the position of the first pair of consecutive matches -/
def ways_to_choose_first_pair : ℕ := n - 2

theorem die_roll_probability :
  (ways_to_choose_first_pair : ℚ) * prob_no_match^(n - 3) * prob_match^2 = 19531250 / 362797056 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2120_212061


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_seven_sixtyfourth_l2120_212053

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_seven_sixtyfourth :
  f⁻¹ (-7/64) = (9/16)^(1/7) :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_seven_sixtyfourth_l2120_212053


namespace NUMINAMATH_CALUDE_cole_fence_cost_l2120_212095

theorem cole_fence_cost (side_length : ℝ) (back_length : ℝ) (cost_per_foot : ℝ)
  (h_side : side_length = 9)
  (h_back : back_length = 18)
  (h_cost : cost_per_foot = 3)
  (h_neighbor_back : ∃ (x : ℝ), x = back_length * cost_per_foot / 2)
  (h_neighbor_left : ∃ (y : ℝ), y = side_length * cost_per_foot / 3) :
  ∃ (total_cost : ℝ), total_cost = 72 ∧
    total_cost = side_length * cost_per_foot + 
                 (2/3) * side_length * cost_per_foot + 
                 back_length * cost_per_foot / 2 :=
by sorry

end NUMINAMATH_CALUDE_cole_fence_cost_l2120_212095


namespace NUMINAMATH_CALUDE_convex_polygon_angle_sum_l2120_212026

theorem convex_polygon_angle_sum (n : ℕ) (angle_sum : ℝ) : n = 17 → angle_sum = 2610 → ∃ (missing_angle : ℝ), 
  0 < missing_angle ∧ 
  missing_angle < 180 ∧ 
  (180 * (n - 2) : ℝ) = angle_sum + missing_angle := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_angle_sum_l2120_212026


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2120_212037

/-- Represents the squaring operation on a calculator --/
def square (n : ℕ) : ℕ := n * n

/-- Applies the squaring operation n times to the initial value --/
def repeated_square (initial : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | n + 1 => square (repeated_square initial n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 3 k ≤ 1000) ∧
  repeated_square 3 3 > 1000 :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2120_212037


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l2120_212044

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through given x -/
def RegressionLine.point_at (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect_at_average_point 
  (l₁ l₂ : RegressionLine) (s t : ℝ) : 
  (∀ (x : ℝ), l₁.point_at s = (s, t) ∧ l₂.point_at s = (s, t)) → 
  l₁.point_at s = l₂.point_at s := by
  sorry

#check regression_lines_intersect_at_average_point

end NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l2120_212044


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2120_212048

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2120_212048


namespace NUMINAMATH_CALUDE_go_square_side_count_l2120_212062

/-- Represents a square arrangement of Go stones -/
structure GoSquare where
  side_length : ℕ
  perimeter_stones : ℕ

/-- The number of stones on one side of a GoSquare -/
def stones_on_side (square : GoSquare) : ℕ := square.side_length

/-- The number of stones on the perimeter of a GoSquare -/
def perimeter_count (square : GoSquare) : ℕ := square.perimeter_stones

theorem go_square_side_count (square : GoSquare) 
  (h : perimeter_count square = 84) : 
  stones_on_side square = 22 := by
  sorry

end NUMINAMATH_CALUDE_go_square_side_count_l2120_212062


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2120_212042

theorem min_value_of_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2120_212042


namespace NUMINAMATH_CALUDE_jorges_clay_rich_soil_fraction_l2120_212080

theorem jorges_clay_rich_soil_fraction (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_rich_soil_yield : ℝ) (total_yield : ℝ) 
  (h1 : total_land = 60)
  (h2 : good_soil_yield = 400)
  (h3 : clay_rich_soil_yield = good_soil_yield / 2)
  (h4 : total_yield = 20000) :
  let clay_rich_fraction := (total_land * good_soil_yield - total_yield) / 
    (total_land * (good_soil_yield - clay_rich_soil_yield))
  clay_rich_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jorges_clay_rich_soil_fraction_l2120_212080


namespace NUMINAMATH_CALUDE_train_car_speed_ratio_l2120_212045

/-- Given a bus, a train, and a car with the following properties:
  * The speed of the bus is 3/4 of the speed of the train
  * The bus travels 480 km in 8 hours
  * The car travels 450 km in 6 hours
  Prove that the ratio of the speed of the train to the speed of the car is 16:15 -/
theorem train_car_speed_ratio : 
  ∀ (bus_speed train_speed car_speed : ℝ),
  bus_speed = (3/4) * train_speed →
  bus_speed = 480 / 8 →
  car_speed = 450 / 6 →
  train_speed / car_speed = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_train_car_speed_ratio_l2120_212045


namespace NUMINAMATH_CALUDE_john_sleep_for_target_score_l2120_212030

/-- Represents the relationship between sleep hours and exam score -/
structure ExamPerformance where
  sleep : ℝ
  score : ℝ

/-- The inverse relationship between sleep and score -/
def inverseRelation (e1 e2 : ExamPerformance) : Prop :=
  e1.sleep * e1.score = e2.sleep * e2.score

theorem john_sleep_for_target_score 
  (e1 : ExamPerformance) 
  (e2 : ExamPerformance) 
  (h1 : e1.sleep = 6) 
  (h2 : e1.score = 80) 
  (h3 : inverseRelation e1 e2) 
  (h4 : (e1.score + e2.score) / 2 = 85) : 
  e2.sleep = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_john_sleep_for_target_score_l2120_212030


namespace NUMINAMATH_CALUDE_expected_value_is_three_halves_l2120_212049

/-- The number of white balls in the bag -/
def white_balls : ℕ := 1

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- X represents the number of red balls drawn -/
def X : Finset ℕ := {1, 2}

/-- The probability mass function of X -/
def prob_X (x : ℕ) : ℚ :=
  if x = 1 then 1/2
  else if x = 2 then 1/2
  else 0

/-- The expected value of X -/
def expected_value_X : ℚ := (1 : ℚ) * (prob_X 1) + (2 : ℚ) * (prob_X 2)

theorem expected_value_is_three_halves :
  expected_value_X = 3/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_three_halves_l2120_212049


namespace NUMINAMATH_CALUDE_total_marbles_is_240_l2120_212013

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_240_l2120_212013


namespace NUMINAMATH_CALUDE_ab_gt_b_squared_l2120_212071

theorem ab_gt_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_gt_b_squared_l2120_212071


namespace NUMINAMATH_CALUDE_rice_seedling_stats_l2120_212003

def dataset : List Nat := [25, 26, 27, 26, 27, 28, 29, 26, 29]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem rice_seedling_stats :
  mode dataset = 26 ∧ median dataset = 27 := by sorry

end NUMINAMATH_CALUDE_rice_seedling_stats_l2120_212003


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l2120_212050

theorem triangle_area_in_circle (R : ℝ) (α β : ℝ) (h_R : R = 2) 
  (h_α : α = π / 3) (h_β : β = π / 4) :
  let γ : ℝ := π - α - β
  let a : ℝ := 2 * R * Real.sin α
  let b : ℝ := 2 * R * Real.sin β
  let c : ℝ := 2 * R * Real.sin γ
  let S : ℝ := (Real.sqrt 3 + 3 : ℝ)
  S = (1 / 2) * a * b * Real.sin γ := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l2120_212050


namespace NUMINAMATH_CALUDE_square_area_error_l2120_212058

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x + 0.38 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let percentage_error := (area_error / actual_area) * 100
  percentage_error = 90.44 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2120_212058


namespace NUMINAMATH_CALUDE_car_trip_speed_l2120_212090

/-- Given a 6-hour trip with an average speed of 38 miles per hour,
    where the speed for the last 2 hours is 44 miles per hour,
    prove that the average speed for the first 4 hours is 35 miles per hour. -/
theorem car_trip_speed :
  ∀ (first_4_hours_speed : ℝ),
    (first_4_hours_speed * 4 + 44 * 2) / 6 = 38 →
    first_4_hours_speed = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l2120_212090


namespace NUMINAMATH_CALUDE_equation_roots_l2120_212068

theorem equation_roots (m : ℝ) :
  ((m - 2) ≠ 0) →  -- Condition for linear equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ * (x₁ + 2*m) + m * (1 - x₁) - 1 = 0 ∧ 
    x₂ * (x₂ + 2*m) + m * (1 - x₂) - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2120_212068


namespace NUMINAMATH_CALUDE_function_zero_point_implies_a_range_l2120_212047

theorem function_zero_point_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 1 ∧ 4 * |a| * x₀ - 2 * a + 1 = 0) →
  a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_point_implies_a_range_l2120_212047


namespace NUMINAMATH_CALUDE_exist_expression_with_fewer_sevens_l2120_212015

-- Define a datatype for arithmetic expressions
inductive Expr
  | Const : ℕ → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr
  | Div : Expr → Expr → Expr

-- Function to count the number of sevens in an expression
def countSevens : Expr → ℕ
  | Expr.Const n => if n = 7 then 1 else 0
  | Expr.Add e1 e2 => countSevens e1 + countSevens e2
  | Expr.Sub e1 e2 => countSevens e1 + countSevens e2
  | Expr.Mul e1 e2 => countSevens e1 + countSevens e2
  | Expr.Div e1 e2 => countSevens e1 + countSevens e2

-- Function to evaluate an expression
def eval : Expr → ℚ
  | Expr.Const n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2
  | Expr.Div e1 e2 => eval e1 / eval e2

-- Theorem statement
theorem exist_expression_with_fewer_sevens :
  ∃ e : Expr, countSevens e < 10 ∧ eval e = 100 := by
  sorry

end NUMINAMATH_CALUDE_exist_expression_with_fewer_sevens_l2120_212015


namespace NUMINAMATH_CALUDE_family_member_bites_l2120_212069

-- Define the number of mosquito bites Cyrus got on arms and legs
def cyrus_arms_legs_bites : ℕ := 14

-- Define the number of mosquito bites Cyrus got on his body
def cyrus_body_bites : ℕ := 10

-- Define the number of other family members
def family_members : ℕ := 6

-- Define Cyrus' total bites
def cyrus_total_bites : ℕ := cyrus_arms_legs_bites + cyrus_body_bites

-- Define the family's total bites
def family_total_bites : ℕ := cyrus_total_bites / 2

-- Theorem to prove
theorem family_member_bites :
  family_total_bites / family_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_family_member_bites_l2120_212069


namespace NUMINAMATH_CALUDE_megan_shirt_payment_l2120_212041

/-- The amount Megan pays for a shirt after discount -/
def shirt_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: Megan pays $16 for the shirt -/
theorem megan_shirt_payment : shirt_price 22 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_megan_shirt_payment_l2120_212041


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2120_212039

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Soap box dimensions -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2120_212039


namespace NUMINAMATH_CALUDE_ratio_of_sides_l2120_212064

/-- A rectangle with a point inside dividing it into four triangles -/
structure DividedRectangle where
  -- The lengths of the sides of the rectangle
  AB : ℝ
  BC : ℝ
  -- The areas of the four triangles
  area_APD : ℝ
  area_BPA : ℝ
  area_CPB : ℝ
  area_DPC : ℝ
  -- Conditions
  positive_AB : 0 < AB
  positive_BC : 0 < BC
  diagonal_condition : AB^2 + BC^2 = (2*AB)^2
  area_condition : area_APD = 1 ∧ area_BPA = 2 ∧ area_CPB = 3 ∧ area_DPC = 4

/-- The theorem stating the ratio of sides in the divided rectangle -/
theorem ratio_of_sides (r : DividedRectangle) : r.AB / r.BC = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sides_l2120_212064


namespace NUMINAMATH_CALUDE_number_problem_l2120_212052

theorem number_problem : ∃ x : ℝ, (x - 5) / 3 = 4 ∧ x = 17 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2120_212052


namespace NUMINAMATH_CALUDE_total_paper_pieces_l2120_212091

theorem total_paper_pieces : 
  let olivia_pieces : ℕ := 127
  let edward_pieces : ℕ := 345
  let sam_pieces : ℕ := 518
  olivia_pieces + edward_pieces + sam_pieces = 990 :=
by sorry

end NUMINAMATH_CALUDE_total_paper_pieces_l2120_212091


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2120_212027

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ q : ℤ, 4^m - 4^n = 3^(k+1) * q) ↔ (∃ p : ℤ, m - n = 3^k * p) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2120_212027


namespace NUMINAMATH_CALUDE_equation1_no_solution_equation2_unique_solution_l2120_212007

-- Define the equations
def equation1 (x : ℝ) : Prop := (4 - x) / (x - 3) + 1 / (3 - x) = 1
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_no_solution : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem equation2_unique_solution : ∃! x : ℝ, equation2 x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation1_no_solution_equation2_unique_solution_l2120_212007


namespace NUMINAMATH_CALUDE_probability_at_least_one_strike_l2120_212038

theorem probability_at_least_one_strike (p : ℝ) (h : p = 2/5) :
  1 - (1 - p)^2 = 16/25 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_strike_l2120_212038


namespace NUMINAMATH_CALUDE_difference_before_l2120_212023

/-- The number of battle cards Sang-cheol had originally -/
def S : ℕ := sorry

/-- The number of battle cards Byeong-ji had originally -/
def B : ℕ := sorry

/-- Sang-cheol gave Byeong-ji 2 battle cards -/
axiom exchange : S ≥ 2

/-- After the exchange, the difference between Byeong-ji and Sang-cheol was 6 -/
axiom difference_after : B + 2 - (S - 2) = 6

/-- Byeong-ji has more cards than Sang-cheol -/
axiom byeongji_has_more : B > S

/-- The difference between Byeong-ji and Sang-cheol before the exchange was 2 -/
theorem difference_before : B - S = 2 := by sorry

end NUMINAMATH_CALUDE_difference_before_l2120_212023


namespace NUMINAMATH_CALUDE_julian_needs_1100_more_legos_l2120_212006

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_models : ℕ := 4

/-- The number of legos required for each airplane model -/
def legos_per_model : ℕ := 375

/-- The number of additional legos Julian needs -/
def additional_legos_needed : ℕ := 1100

/-- Theorem stating that Julian needs 1100 more legos to make 4 identical airplane models -/
theorem julian_needs_1100_more_legos :
  (num_models * legos_per_model) - julian_legos = additional_legos_needed := by
  sorry

end NUMINAMATH_CALUDE_julian_needs_1100_more_legos_l2120_212006


namespace NUMINAMATH_CALUDE_margies_change_is_6_25_l2120_212082

/-- The amount of change Margie received after buying apples -/
def margies_change (num_apples : ℕ) (cost_per_apple : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples : ℚ) * cost_per_apple

/-- Theorem stating that Margie's change is $6.25 given the problem conditions -/
theorem margies_change_is_6_25 :
  margies_change 5 (75 / 100) 10 = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_margies_change_is_6_25_l2120_212082


namespace NUMINAMATH_CALUDE_night_shift_arrangements_count_l2120_212051

/-- The number of days in the shift schedule -/
def num_days : ℕ := 6

/-- The number of people available for shifts -/
def num_people : ℕ := 4

/-- The number of scenarios for arranging consecutive shifts -/
def num_scenarios : ℕ := 6

/-- Calculates the number of different night shift arrangements -/
def night_shift_arrangements : ℕ := 
  num_scenarios * (num_people.factorial / (num_people - 2).factorial) * 
  ((num_people - 2).factorial / (num_people - 4).factorial)

/-- Theorem stating the number of different night shift arrangements -/
theorem night_shift_arrangements_count : night_shift_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_night_shift_arrangements_count_l2120_212051


namespace NUMINAMATH_CALUDE_remaining_speed_calculation_l2120_212085

/-- Calculates the average speed for the remaining part of a trip given:
    - The fraction of the trip completed in the first part
    - The speed of the first part of the trip
    - The average speed for the entire trip
-/
theorem remaining_speed_calculation 
  (first_part_fraction : Real) 
  (first_part_speed : Real) 
  (total_average_speed : Real) :
  first_part_fraction = 0.4 →
  first_part_speed = 40 →
  total_average_speed = 50 →
  (1 - first_part_fraction) * total_average_speed / 
    (1 - first_part_fraction * total_average_speed / first_part_speed) = 60 := by
  sorry

#check remaining_speed_calculation

end NUMINAMATH_CALUDE_remaining_speed_calculation_l2120_212085


namespace NUMINAMATH_CALUDE_count_special_numbers_is_279_l2120_212088

/-- A function that counts the number of positive integers less than 100,000 
    with at most two different digits, where one of the digits must be 1. -/
def count_special_numbers : ℕ :=
  let max_number := 100000
  let required_digit := 1
  -- Implementation details are omitted
  279

/-- Theorem stating that the count of special numbers is 279. -/
theorem count_special_numbers_is_279 : count_special_numbers = 279 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_279_l2120_212088


namespace NUMINAMATH_CALUDE_ten_balls_distribution_l2120_212056

/-- The number of ways to distribute n identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
def distributionWays (n : ℕ) : ℕ :=
  let remainingBalls := n - (1 + 2 + 3)
  (remainingBalls + 3 - 1).choose 2

/-- Theorem: There are 15 ways to distribute 10 identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
theorem ten_balls_distribution : distributionWays 10 = 15 := by
  sorry

#eval distributionWays 10  -- Should output 15

end NUMINAMATH_CALUDE_ten_balls_distribution_l2120_212056


namespace NUMINAMATH_CALUDE_birthday_problem_solution_l2120_212001

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Represents the ages of Alice, Bob, and Carl -/
structure FamilyAges :=
  (alice : Age)
  (bob : Age)
  (carl : Age)

/-- Checks if one age is a multiple of another -/
def isMultipleOf (a b : Age) : Prop :=
  ∃ k : ℕ, a.value = k * b.value

/-- Represents the conditions of the problem -/
structure BirthdayProblem :=
  (ages : FamilyAges)
  (aliceOlderThanBob : ages.alice.value = ages.bob.value + 2)
  (carlAgeToday : ages.carl.value = 3)
  (bobMultipleOfCarl : isMultipleOf ages.bob ages.carl)
  (firstOfFourBirthdays : ∀ n : ℕ, n < 4 → isMultipleOf ⟨ages.bob.value + n⟩ ⟨ages.carl.value + n⟩)

/-- The main theorem to prove -/
theorem birthday_problem_solution (problem : BirthdayProblem) :
  ∃ (futureAliceAge : ℕ),
    futureAliceAge > problem.ages.alice.value ∧
    isMultipleOf ⟨futureAliceAge⟩ ⟨problem.ages.carl.value + (futureAliceAge - problem.ages.alice.value)⟩ ∧
    sumOfDigits futureAliceAge = 6 :=
  sorry

end NUMINAMATH_CALUDE_birthday_problem_solution_l2120_212001


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2120_212099

theorem inequality_system_solution_set :
  {x : ℝ | x - 1 < 0 ∧ x + 1 > 0} = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2120_212099


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2120_212009

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (15 < a ∧ a ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2120_212009
