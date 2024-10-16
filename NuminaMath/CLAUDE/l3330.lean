import Mathlib

namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3330_333020

/-- Proves that the profit percent is 26% when selling an article at a certain price,
    given that selling it at 2/3 of that price results in a 16% loss. -/
theorem profit_percent_calculation (P C : ℝ) 
  (h : (2/3) * P = 0.84 * C) : 
  (P - C) / C * 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3330_333020


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3330_333045

/-- Given a line with equation 2x + 3y = 6, the slope of a perpendicular line is 3/2 -/
theorem perpendicular_slope (x y : ℝ) :
  (2 * x + 3 * y = 6) →
  ∃ m : ℝ, m = 3 / 2 ∧ m * (-2 / 3) = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3330_333045


namespace NUMINAMATH_CALUDE_intersection_point_of_problem_lines_l3330_333070

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The specific two lines from the problem -/
def problemLines : TwoLines where
  line1 := λ x y ↦ x + y + 3 = 0
  line2 := λ x y ↦ x - 2*y + 3 = 0

/-- Definition of an intersection point -/
def isIntersectionPoint (lines : TwoLines) (x y : ℝ) : Prop :=
  lines.line1 x y ∧ lines.line2 x y

/-- Theorem stating that (-3, 0) is the intersection point of the given lines -/
theorem intersection_point_of_problem_lines :
  isIntersectionPoint problemLines (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_problem_lines_l3330_333070


namespace NUMINAMATH_CALUDE_finite_rule2_applications_l3330_333080

/-- Represents the state of the blackboard -/
def Blackboard := List ℤ

/-- Rule 1: If there's a pair of equal numbers, add a to one and b to the other -/
def applyRule1 (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Rule 2: If there's no pair of equal numbers, write two zeros -/
def applyRule2 : Blackboard := [0, 0]

/-- Applies either Rule 1 or Rule 2 based on the current board state -/
def applyRule (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Represents a sequence of rule applications -/
def RuleSequence := List (Blackboard → Blackboard)

/-- Counts the number of times Rule 2 is applied in a sequence -/
def countRule2Applications (seq : RuleSequence) : ℕ :=
  sorry

/-- The main theorem: Rule 2 is applied only finitely many times -/
theorem finite_rule2_applications (a b : ℕ) (h : a ≠ b) :
  ∃ N : ℕ, ∀ seq : RuleSequence, countRule2Applications seq ≤ N :=
  sorry

end NUMINAMATH_CALUDE_finite_rule2_applications_l3330_333080


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3330_333083

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let circumference := (3/4) * (2 * π * r)
  let base_radius := circumference / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * height = 3 * π * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3330_333083


namespace NUMINAMATH_CALUDE_bird_families_count_l3330_333002

/-- The number of bird families that flew away for winter -/
def flew_away : ℕ := 32

/-- The number of bird families that stayed near the mountain -/
def stayed : ℕ := 35

/-- The initial number of bird families living near the mountain -/
def initial_families : ℕ := flew_away + stayed

theorem bird_families_count : initial_families = 67 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l3330_333002


namespace NUMINAMATH_CALUDE_divisibility_problem_l3330_333069

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ k : ℕ, abc - 1 = k * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3330_333069


namespace NUMINAMATH_CALUDE_exponent_sum_l3330_333023

theorem exponent_sum (m n : ℕ) (h : 2^m * 2^n = 16) : m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3330_333023


namespace NUMINAMATH_CALUDE_two_cars_meeting_l3330_333075

/-- Two cars meeting on a highway problem -/
theorem two_cars_meeting (highway_length : ℝ) (car1_speed : ℝ) (meeting_time : ℝ) :
  highway_length = 45 →
  car1_speed = 14 →
  meeting_time = 1.5 →
  ∃ car2_speed : ℝ,
    car2_speed = 16 ∧
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length :=
by sorry

end NUMINAMATH_CALUDE_two_cars_meeting_l3330_333075


namespace NUMINAMATH_CALUDE_five_from_second_row_wading_l3330_333073

/-- Represents the beach scenario with people in rows and some wading in the water -/
structure BeachScenario where
  initial_first_row : ℕ
  initial_second_row : ℕ
  third_row : ℕ
  first_row_wading : ℕ
  remaining_on_beach : ℕ

/-- Calculates the number of people from the second row who joined those wading in the water -/
def second_row_wading (scenario : BeachScenario) : ℕ :=
  scenario.initial_first_row + scenario.initial_second_row + scenario.third_row
  - scenario.first_row_wading - scenario.remaining_on_beach

/-- Theorem stating that 5 people from the second row joined those wading in the water -/
theorem five_from_second_row_wading (scenario : BeachScenario)
  (h1 : scenario.initial_first_row = 24)
  (h2 : scenario.initial_second_row = 20)
  (h3 : scenario.third_row = 18)
  (h4 : scenario.first_row_wading = 3)
  (h5 : scenario.remaining_on_beach = 54) :
  second_row_wading scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_from_second_row_wading_l3330_333073


namespace NUMINAMATH_CALUDE_recursive_sequence_solution_l3330_333035

/-- A sequence of real numbers satisfying the given recursion -/
def RecursiveSequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)

theorem recursive_sequence_solution 
  (b : ℕ → ℝ) 
  (h_recursive : RecursiveSequence b) 
  (h_b1 : b 1 = 2 + Real.sqrt 8) 
  (h_b1980 : b 1980 = 15 + Real.sqrt 8) : 
  b 2013 = -1/6 + 13 * Real.sqrt 8 / 6 := by
  sorry

end NUMINAMATH_CALUDE_recursive_sequence_solution_l3330_333035


namespace NUMINAMATH_CALUDE_power_equality_l3330_333039

theorem power_equality : (4 : ℝ) ^ 10 = 16 ^ 5 := by sorry

end NUMINAMATH_CALUDE_power_equality_l3330_333039


namespace NUMINAMATH_CALUDE_five_in_set_A_l3330_333015

theorem five_in_set_A : 5 ∈ {x : ℕ | 1 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_five_in_set_A_l3330_333015


namespace NUMINAMATH_CALUDE_inequality_proof_l3330_333094

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3330_333094


namespace NUMINAMATH_CALUDE_circle_radius_l3330_333093

theorem circle_radius (A : ℝ) (h : A = 196 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3330_333093


namespace NUMINAMATH_CALUDE_remainder_theorem_l3330_333086

theorem remainder_theorem : ∃ q : ℕ, 3^202 + 303 = (3^101 + 3^51 + 1) * q + 302 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3330_333086


namespace NUMINAMATH_CALUDE_wooden_block_volume_l3330_333076

/-- Represents a rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  baseArea : ℝ

/-- Calculates the volume of a rectangular wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.baseArea

/-- Theorem: The volume of the wooden block is 864 cubic decimeters -/
theorem wooden_block_volume :
  ∀ (block : WoodenBlock),
    block.length = 72 →
    (3 - 1) * 2 * block.baseArea = 48 →
    volume block = 864 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_volume_l3330_333076


namespace NUMINAMATH_CALUDE_distribute_5_4_l3330_333014

/-- The number of ways to distribute n different books to k students,
    with each student receiving at least one book. -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n - (k.choose 3) * (k-3)^n

/-- Theorem stating that distributing 5 different books to 4 students,
    with each student receiving at least one book, results in 240 different schemes. -/
theorem distribute_5_4 : distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3330_333014


namespace NUMINAMATH_CALUDE_amp_2_neg1_1_l3330_333096

-- Define the & operation
def amp (a b c : ℝ) : ℝ := 3 * b^2 - 4 * a * c

-- Theorem statement
theorem amp_2_neg1_1 : amp 2 (-1) 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_amp_2_neg1_1_l3330_333096


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3330_333066

theorem trigonometric_product_equals_one :
  let α : Real := 15 * π / 180  -- 15 degrees in radians
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin (π/2 - α)) *
  (1 - 1 / Real.sin α) * (1 + 1 / Real.cos (π/2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3330_333066


namespace NUMINAMATH_CALUDE_square_divided_into_rectangles_l3330_333016

theorem square_divided_into_rectangles (square_perimeter : ℝ) 
  (h1 : square_perimeter = 200) : 
  let side_length := square_perimeter / 4
  let rectangle_length := side_length
  let rectangle_width := side_length / 2
  2 * (rectangle_length + rectangle_width) = 150 := by
sorry

end NUMINAMATH_CALUDE_square_divided_into_rectangles_l3330_333016


namespace NUMINAMATH_CALUDE_digit_concatenation_divisibility_l3330_333057

theorem digit_concatenation_divisibility (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a) (h3 : a < 10^n) :
  let b := a * (10^n + 1)
  ∃! k : ℕ, k > 1 ∧ k ≤ 10 ∧ b = k * a^2 ∧ k = 7 :=
sorry

end NUMINAMATH_CALUDE_digit_concatenation_divisibility_l3330_333057


namespace NUMINAMATH_CALUDE_supermarket_can_display_l3330_333084

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem supermarket_can_display :
  let a₁ : ℕ := 28  -- first term
  let aₙ : ℕ := 1   -- last term
  let n : ℕ := 10   -- number of terms
  arithmeticSequenceSum a₁ aₙ n = 145 := by
  sorry


end NUMINAMATH_CALUDE_supermarket_can_display_l3330_333084


namespace NUMINAMATH_CALUDE_special_op_is_addition_l3330_333030

/-- A binary operation on real numbers satisfying (a * b) * c = a + b + c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property of the special operation -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by sorry

end NUMINAMATH_CALUDE_special_op_is_addition_l3330_333030


namespace NUMINAMATH_CALUDE_complex_magnitude_l3330_333024

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3330_333024


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l3330_333051

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1DailyConsumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2DailyConsumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3DailyConsumption : ℕ := 9

/-- The total weekly water consumption of all siblings -/
def totalWeeklyConsumption : ℕ :=
  (sibling1DailyConsumption + sibling2DailyConsumption + sibling3DailyConsumption) * daysInWeek

theorem siblings_weekly_water_consumption :
  totalWeeklyConsumption = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l3330_333051


namespace NUMINAMATH_CALUDE_existence_of_a_and_b_l3330_333022

/-- The number of positive divisors of a natural number -/
noncomputable def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem existence_of_a_and_b (k l c : ℕ+) :
  ∃ (a b : ℕ+),
    (b - a = c * Nat.gcd a b) ∧
    (tau a * tau (b / Nat.gcd a b) * l = tau b * tau (a / Nat.gcd a b) * k) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_and_b_l3330_333022


namespace NUMINAMATH_CALUDE_chantel_bracelets_l3330_333004

def bracelet_problem (
  days1 : ℕ) (bracelets_per_day1 : ℕ) (give_away1 : ℕ)
  (days2 : ℕ) (bracelets_per_day2 : ℕ) (give_away2 : ℕ)
  (days3 : ℕ) (bracelets_per_day3 : ℕ)
  (days4 : ℕ) (bracelets_per_day4 : ℕ) (give_away3 : ℕ) : ℕ :=
  (days1 * bracelets_per_day1 - give_away1 +
   days2 * bracelets_per_day2 - give_away2 +
   days3 * bracelets_per_day3 +
   days4 * bracelets_per_day4 - give_away3)

theorem chantel_bracelets :
  bracelet_problem 7 4 8 10 5 12 4 6 2 3 10 = 78 := by
  sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l3330_333004


namespace NUMINAMATH_CALUDE_button_comparison_l3330_333003

theorem button_comparison (mari_buttons sue_buttons : ℕ) 
  (h1 : mari_buttons = 8)
  (h2 : sue_buttons = 22)
  (h3 : ∃ kendra_buttons : ℕ, sue_buttons = kendra_buttons / 2)
  (h4 : ∃ kendra_buttons : ℕ, kendra_buttons > 5 * mari_buttons) :
  ∃ kendra_buttons : ℕ, kendra_buttons - (5 * mari_buttons) = 4 :=
by sorry

end NUMINAMATH_CALUDE_button_comparison_l3330_333003


namespace NUMINAMATH_CALUDE_key_pairs_and_drawers_l3330_333010

/-- Given 10 distinct keys, prove the following:
1. The number of possible pairs of keys
2. The number of copies of each key needed to form all possible pairs
3. The minimum number of drawers to open to ensure possession of all 10 different keys
-/
theorem key_pairs_and_drawers (n : ℕ) (h : n = 10) :
  let num_pairs := n.choose 2
  let copies_per_key := n - 1
  let total_drawers := num_pairs
  let min_drawers := total_drawers - copies_per_key + 1
  (num_pairs = 45) ∧ (copies_per_key = 9) ∧ (min_drawers = 37) := by
  sorry


end NUMINAMATH_CALUDE_key_pairs_and_drawers_l3330_333010


namespace NUMINAMATH_CALUDE_june_decrease_percentage_l3330_333099

-- Define the price changes for each month
def january_change : ℝ := 0.15
def february_change : ℝ := -0.10
def march_change : ℝ := 0.20
def april_change : ℝ := -0.30
def may_change : ℝ := 0.10

-- Function to calculate the price after a change
def apply_change (price : ℝ) (change : ℝ) : ℝ :=
  price * (1 + change)

-- Theorem stating the required decrease in June
theorem june_decrease_percentage (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let final_price := apply_change (apply_change (apply_change (apply_change (apply_change initial_price january_change) february_change) march_change) april_change) may_change
  ∃ (june_decrease : ℝ), 
    (apply_change final_price june_decrease = initial_price) ∧ 
    (abs (june_decrease + 0.0456) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_june_decrease_percentage_l3330_333099


namespace NUMINAMATH_CALUDE_complex_simplification_l3330_333095

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex fraction in the problem -/
noncomputable def z : ℂ := (2 + 3*i) / (2 - 3*i)

/-- The main theorem -/
theorem complex_simplification : z^8 * 3 = 3 := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3330_333095


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l3330_333047

open Real

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π

/-- The sides of the triangle form an arithmetic sequence -/
def SidesArithmeticSequence (t : Triangle) : Prop :=
  2 / t.b = 1 / t.a + 1 / t.c

/-- The angles of the triangle form an arithmetic sequence -/
def AnglesArithmeticSequence (t : Triangle) : Prop :=
  2 * t.β = t.α + t.γ

/-- A triangle is equilateral if all its sides are equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral
  (t : Triangle)
  (h_sides : SidesArithmeticSequence t)
  (h_angles : AnglesArithmeticSequence t) :
  IsEquilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l3330_333047


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3330_333028

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3330_333028


namespace NUMINAMATH_CALUDE_garden_fence_columns_l3330_333072

theorem garden_fence_columns (S C : ℕ) : 
  S * C + (S - 1) / 2 = 1223 → 
  S = 2 * C + 5 → 
  C = 23 := by sorry

end NUMINAMATH_CALUDE_garden_fence_columns_l3330_333072


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3330_333034

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3 / 4 = s / 2) → (3 * s = 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3330_333034


namespace NUMINAMATH_CALUDE_fraction_difference_l3330_333013

theorem fraction_difference (n : ℝ) : 
  let simplified := (n * (n + 3)) / (n^2 + 3*n + 1)
  (n^2 + 3*n + 1) - (n^2 + 3*n) = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l3330_333013


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_problem_l3330_333044

theorem consecutive_even_numbers_problem :
  ∀ (x y z : ℕ),
  (y = x + 2) →
  (z = y + 2) →
  (3 * x = 2 * z + 14) →
  z = 26 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_problem_l3330_333044


namespace NUMINAMATH_CALUDE_symmetric_polynomial_n_l3330_333097

/-- A polynomial p(x) is symmetric about x = m if p(m + k) = p(m - k) for all real k -/
def is_symmetric_about (p : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ k, p (m + k) = p (m - k)

/-- The polynomial p(x) = x^2 + 2nx + 3 -/
def p (n : ℝ) (x : ℝ) : ℝ := x^2 + 2*n*x + 3

theorem symmetric_polynomial_n (n : ℝ) :
  is_symmetric_about (p n) 5 → n = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_n_l3330_333097


namespace NUMINAMATH_CALUDE_simplify_expression_l3330_333029

theorem simplify_expression (r s : ℝ) : 120*r - 32*r + 50*s - 20*s = 88*r + 30*s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3330_333029


namespace NUMINAMATH_CALUDE_franks_sunday_bags_l3330_333053

/-- Given that Frank filled 5 bags on Saturday, each bag contains 5 cans,
    and Frank collected a total of 40 cans over the weekend,
    prove that Frank filled 3 bags on Sunday. -/
theorem franks_sunday_bags (saturday_bags : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ)
    (h1 : saturday_bags = 5)
    (h2 : cans_per_bag = 5)
    (h3 : total_cans = 40) :
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_franks_sunday_bags_l3330_333053


namespace NUMINAMATH_CALUDE_increase_540_by_10_six_times_l3330_333059

/-- The result of increasing a number by a fixed amount multiple times -/
def increase_multiple_times (start : ℕ) (increment : ℕ) (times : ℕ) : ℕ :=
  start + increment * times

/-- Theorem stating that increasing 540 by 10 six times results in 600 -/
theorem increase_540_by_10_six_times :
  increase_multiple_times 540 10 6 = 600 := by
  sorry

end NUMINAMATH_CALUDE_increase_540_by_10_six_times_l3330_333059


namespace NUMINAMATH_CALUDE_class_average_age_l3330_333041

theorem class_average_age (initial_students : ℕ) (leaving_student_age : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  initial_students = 30 →
  leaving_student_age = 11 →
  teacher_age = 41 →
  new_average = 11 →
  (initial_students * (initial_average : ℝ) - leaving_student_age + teacher_age) / initial_students = new_average →
  initial_average = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_age_l3330_333041


namespace NUMINAMATH_CALUDE_system_solution_l3330_333052

theorem system_solution (a b : ℝ) : 
  (2 * a * 1 + b * 1 = 3) → 
  (a * 1 - b * 1 = 1) → 
  a + 2 * b = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3330_333052


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3330_333036

theorem fourteenth_root_of_unity : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (4 * π / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3330_333036


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l3330_333092

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
inductive Coin
| Real : Coin
| Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := CoinSet → CoinSet → WeighingResult

/-- The maximum number of weighings allowed -/
def MaxWeighings : Nat := 4

/-- The number of unknown coins -/
def UnknownCoins : Nat := 12

/-- The number of known real coins -/
def KnownRealCoins : Nat := 5

/-- The number of known counterfeit coins -/
def KnownCounterfeitCoins : Nat := 5

/-- A strategy is a function that takes the current state and returns the next weighing to perform -/
def Strategy := List WeighingResult → Weighing

/-- Determines if a strategy is successful in identifying the number of counterfeit coins -/
def IsSuccessfulStrategy (s : Strategy) : Prop := sorry

/-- The main theorem: There exists a successful strategy to determine the number of counterfeit coins -/
theorem counterfeit_coin_identification :
  ∃ (s : Strategy), IsSuccessfulStrategy s := by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l3330_333092


namespace NUMINAMATH_CALUDE_equation_is_hyperbola_l3330_333063

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 25y^2 - 10x + 50 = 0 represents a hyperbola --/
theorem equation_is_hyperbola :
  determineConicSection 1 (-25) 0 (-10) 0 50 = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_is_hyperbola_l3330_333063


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3330_333042

theorem sum_of_three_squares (n : ℕ+) (h : ∃ m : ℕ, 3 * n + 1 = m^2) :
  ∃ a b c : ℕ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3330_333042


namespace NUMINAMATH_CALUDE_fraction_addition_l3330_333078

theorem fraction_addition : (2 : ℚ) / 520 + 23 / 40 = 301 / 520 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l3330_333078


namespace NUMINAMATH_CALUDE_lamp_height_difference_example_l3330_333031

/-- The height difference between two lamps -/
def lamp_height_difference (new_height old_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem: The height difference between a new lamp of 2.33 feet and an old lamp of 1 foot is 1.33 feet -/
theorem lamp_height_difference_example :
  lamp_height_difference 2.33 1 = 1.33 := by
  sorry

end NUMINAMATH_CALUDE_lamp_height_difference_example_l3330_333031


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3330_333019

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3330_333019


namespace NUMINAMATH_CALUDE_solution_set_is_ray_iff_l3330_333081

/-- The polynomial function representing the left side of the inequality -/
def f (a x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions to the inequality for a given a -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- A set is a ray if it's of the form [c, ∞) or (-∞, c] for some c ∈ ℝ -/
def IsRay (S : Set ℝ) : Prop :=
  ∃ c : ℝ, S = {x : ℝ | x ≥ c} ∨ S = {x : ℝ | x ≤ c}

/-- The main theorem: The solution set is a ray iff a = 1 or a = -1 -/
theorem solution_set_is_ray_iff (a : ℝ) :
  IsRay (SolutionSet a) ↔ a = 1 ∨ a = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_ray_iff_l3330_333081


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3330_333040

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| + |x - 2| ≤ 4} = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3330_333040


namespace NUMINAMATH_CALUDE_probability_arrives_before_l3330_333008

/-- Represents a student -/
structure Student :=
  (name : String)

/-- Represents the arrival order of students -/
def ArrivalOrder := List Student

/-- Given a list of students, generates all possible arrival orders -/
def allPossibleArrivals (students : List Student) : List ArrivalOrder :=
  sorry

/-- Checks if student1 arrives before student2 in a given arrival order -/
def arrivesBeforeIn (student1 student2 : Student) (order : ArrivalOrder) : Bool :=
  sorry

/-- Counts the number of arrival orders where student1 arrives before student2 -/
def countArrivesBeforeOrders (student1 student2 : Student) (orders : List ArrivalOrder) : Nat :=
  sorry

theorem probability_arrives_before (student1 student2 student3 : Student) :
  let students := [student1, student2, student3]
  let allOrders := allPossibleArrivals students
  let favorableOrders := countArrivesBeforeOrders student1 student2 allOrders
  favorableOrders = allOrders.length / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_arrives_before_l3330_333008


namespace NUMINAMATH_CALUDE_negative_two_fourth_power_l3330_333079

theorem negative_two_fourth_power :
  ∃ (base : ℤ) (exponent : ℕ), ((-2 : ℤ) ^ 4 = base ^ exponent) ∧ (base = -2) ∧ (exponent = 4) :=
by sorry

end NUMINAMATH_CALUDE_negative_two_fourth_power_l3330_333079


namespace NUMINAMATH_CALUDE_balog_theorem_l3330_333005

theorem balog_theorem (q : ℕ+) (A : Finset ℤ) :
  ∃ (C_q : ℕ), (A.card + q * A.card : ℤ) ≥ ((q + 1) * A.card : ℤ) - C_q :=
by sorry

end NUMINAMATH_CALUDE_balog_theorem_l3330_333005


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3330_333017

/-- The area of the triangle formed by y = x, x = -5, and the x-axis --/
def triangle_area : ℝ := 12.5

/-- The x-coordinate of the vertical line --/
def vertical_line_x : ℝ := -5

/-- Theorem: The area of the triangle formed by y = x, x = -5, and the x-axis is 12.5 --/
theorem triangle_area_proof :
  let intersection_point := (vertical_line_x, vertical_line_x)
  let base := -vertical_line_x
  let height := -vertical_line_x
  (1/2 : ℝ) * base * height = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3330_333017


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3330_333058

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | f a b c x > 0}

-- State the theorem
theorem quadratic_inequality_properties (a b c : ℝ) :
  solution_set a b c = Set.Ioo (-1/2 : ℝ) 2 →
  a < 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3330_333058


namespace NUMINAMATH_CALUDE_ship_passengers_l3330_333012

theorem ship_passengers (total : ℝ) (round_trip_with_car : ℝ) 
  (h1 : 0 < total) 
  (h2 : 0 ≤ round_trip_with_car) 
  (h3 : round_trip_with_car ≤ total) 
  (h4 : round_trip_with_car / total = 0.2 * (round_trip_with_car / 0.2) / total) : 
  round_trip_with_car / 0.2 = total := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_l3330_333012


namespace NUMINAMATH_CALUDE_total_fuel_needed_l3330_333064

def fuel_consumption : ℝ := 5
def trip1_distance : ℝ := 30
def trip2_distance : ℝ := 20

theorem total_fuel_needed : 
  fuel_consumption * (trip1_distance + trip2_distance) = 250 := by
sorry

end NUMINAMATH_CALUDE_total_fuel_needed_l3330_333064


namespace NUMINAMATH_CALUDE_earphone_cost_l3330_333077

/-- The cost of an earphone given weekly expenditure data -/
theorem earphone_cost (mean_expenditure : ℕ) (mon tue wed thu sat sun : ℕ) (pen notebook : ℕ) :
  mean_expenditure = 500 →
  mon = 450 →
  tue = 600 →
  wed = 400 →
  thu = 500 →
  sat = 550 →
  sun = 300 →
  pen = 30 →
  notebook = 50 →
  ∃ (earphone : ℕ), earphone = 7 * mean_expenditure - (mon + tue + wed + thu + sat + sun) - pen - notebook :=
by
  sorry

end NUMINAMATH_CALUDE_earphone_cost_l3330_333077


namespace NUMINAMATH_CALUDE_bell_interval_problem_l3330_333087

theorem bell_interval_problem (x : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n * 5 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 8 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * x = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 15 = 1320) →
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l3330_333087


namespace NUMINAMATH_CALUDE_no_f_iteration_to_one_l3330_333060

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^2 + 1 else n / 2 + 3

def iterateF (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => f (iterateF n k)

theorem no_f_iteration_to_one :
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 100 → ∀ k : ℕ, iterateF n k ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_no_f_iteration_to_one_l3330_333060


namespace NUMINAMATH_CALUDE_debt_doubling_time_l3330_333074

def interest_rate : ℝ := 0.07

theorem debt_doubling_time : 
  ∀ t : ℕ, t < 10 → (1 + interest_rate) ^ t ≤ 2 ∧ 
  (1 + interest_rate) ^ 10 > 2 := by sorry

end NUMINAMATH_CALUDE_debt_doubling_time_l3330_333074


namespace NUMINAMATH_CALUDE_subsets_containing_neither_A_nor_B_l3330_333056

variable (X : Finset ℕ)
variable (A B : Finset ℕ)

theorem subsets_containing_neither_A_nor_B :
  X.card = 10 →
  A ⊆ X →
  B ⊆ X →
  A.card = 3 →
  B.card = 4 →
  Disjoint A B →
  (X.powerset.filter (λ S => ¬(A ⊆ S) ∧ ¬(B ⊆ S))).card = 840 :=
by sorry

end NUMINAMATH_CALUDE_subsets_containing_neither_A_nor_B_l3330_333056


namespace NUMINAMATH_CALUDE_initial_chicken_wings_chef_initial_wings_l3330_333068

theorem initial_chicken_wings (num_friends : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) : ℕ :=
  num_friends * wings_per_friend - additional_wings

theorem chef_initial_wings : initial_chicken_wings 4 7 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_chicken_wings_chef_initial_wings_l3330_333068


namespace NUMINAMATH_CALUDE_isabella_total_items_l3330_333048

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    and Alexis bought 21 pairs of pants and 18 dresses,
    prove that Isabella bought a total of 13 items (pants and dresses combined). -/
theorem isabella_total_items (alexis_pants : ℕ) (alexis_dresses : ℕ) 
    (h1 : alexis_pants = 21) 
    (h2 : alexis_dresses = 18) 
    (h3 : ∃ (isabella_pants isabella_dresses : ℕ), 
      alexis_pants = 3 * isabella_pants ∧ 
      alexis_dresses = 3 * isabella_dresses) : 
  ∃ (isabella_total : ℕ), isabella_total = 13 := by
  sorry

end NUMINAMATH_CALUDE_isabella_total_items_l3330_333048


namespace NUMINAMATH_CALUDE_polynomial_negative_values_l3330_333000

theorem polynomial_negative_values (a x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + 5/2*a^2*(a - x)^4 - 1/2*a^4*(a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_negative_values_l3330_333000


namespace NUMINAMATH_CALUDE_contrapositive_at_least_one_even_l3330_333088

theorem contrapositive_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_at_least_one_even_l3330_333088


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3330_333089

theorem polygon_interior_angles (n : ℕ) (h : n = 14) : 
  (n - 2) * 180 - 180 = 2000 :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3330_333089


namespace NUMINAMATH_CALUDE_prime_divisors_of_p_cubed_plus_three_l3330_333054

theorem prime_divisors_of_p_cubed_plus_three (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p^2 + 2)) :
  ∃ (a b c : ℕ), Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ p^3 + 3 = a * b * c :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_p_cubed_plus_three_l3330_333054


namespace NUMINAMATH_CALUDE_second_vessel_ratio_l3330_333061

/-- Represents the ratio of milk to water in a mixture -/
structure MilkWaterRatio where
  milk : ℚ
  water : ℚ

/-- The mixture in a vessel -/
structure Mixture where
  volume : ℚ
  ratio : MilkWaterRatio

theorem second_vessel_ratio 
  (v1 v2 : Mixture) 
  (h1 : v1.volume = v2.volume) 
  (h2 : v1.ratio = MilkWaterRatio.mk 4 2) 
  (h3 : let combined_ratio := MilkWaterRatio.mk 
          (v1.ratio.milk * v1.volume + v2.ratio.milk * v2.volume) 
          (v1.ratio.water * v1.volume + v2.ratio.water * v2.volume)
        combined_ratio = MilkWaterRatio.mk 3 1) :
  v2.ratio = MilkWaterRatio.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_second_vessel_ratio_l3330_333061


namespace NUMINAMATH_CALUDE_six_disks_common_point_implies_center_inside_l3330_333011

-- Define a disk in 2D space
structure Disk :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define what it means for a point to be inside a disk
def isInside (p : ℝ × ℝ) (d : Disk) : Prop :=
  let (x, y) := p
  let (cx, cy) := d.center
  (x - cx)^2 + (y - cy)^2 < d.radius^2

-- Define a set of six disks
def SixDisks := Fin 6 → Disk

-- The theorem statement
theorem six_disks_common_point_implies_center_inside
  (disks : SixDisks)
  (common_point : ℝ × ℝ)
  (h : ∀ i : Fin 6, isInside common_point (disks i)) :
  ∃ i j : Fin 6, i ≠ j ∧ isInside (disks j).center (disks i) :=
sorry

end NUMINAMATH_CALUDE_six_disks_common_point_implies_center_inside_l3330_333011


namespace NUMINAMATH_CALUDE_solve_family_income_problem_l3330_333007

def family_income_problem (initial_members : ℕ) (initial_average : ℝ) 
  (final_members : ℕ) (final_average : ℝ) : Prop :=
  let initial_total := initial_members * initial_average
  let final_total := final_members * final_average
  let deceased_income := initial_total - final_total
  initial_members = 4 ∧ 
  final_members = 3 ∧ 
  initial_average = 782 ∧ 
  final_average = 650 ∧ 
  deceased_income = 1178

theorem solve_family_income_problem : 
  ∃ (initial_members final_members : ℕ) (initial_average final_average : ℝ),
    family_income_problem initial_members initial_average final_members final_average :=
by
  sorry

end NUMINAMATH_CALUDE_solve_family_income_problem_l3330_333007


namespace NUMINAMATH_CALUDE_quarter_percent_of_120_l3330_333021

theorem quarter_percent_of_120 : (1 / 4 : ℚ) / 100 * 120 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_quarter_percent_of_120_l3330_333021


namespace NUMINAMATH_CALUDE_division_property_l3330_333055

theorem division_property (n : ℕ) : 
  (n / 5 = 248) ∧ (n % 5 = 4) → (n / 9 + n % 9 = 140) := by
  sorry

end NUMINAMATH_CALUDE_division_property_l3330_333055


namespace NUMINAMATH_CALUDE_fraction_sum_difference_equals_half_l3330_333018

theorem fraction_sum_difference_equals_half : 
  (3 : ℚ) / 9 + 5 / 12 - 1 / 4 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_equals_half_l3330_333018


namespace NUMINAMATH_CALUDE_sum_of_digits_in_period_l3330_333043

def period_length (n : ℕ) : ℕ := sorry

def decimal_expansion (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_in_period (n : ℕ) (h : n = 98^2) :
  let m := period_length n
  let digits := decimal_expansion n
  List.sum (List.take m digits) = 900 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_period_l3330_333043


namespace NUMINAMATH_CALUDE_special_function_property_l3330_333091

/-- A real-valued function on rational numbers satisfying specific properties -/
def special_function (f : ℚ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ α, α ≠ 0 → f α > 0) ∧
  (∀ α β, f (α * β) = f α * f β) ∧
  (∀ α β, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

/-- Theorem stating that f(α + β) = max{f(α), f(β)} when f(α) ≠ f(β) -/
theorem special_function_property (f : ℚ → ℝ) (h : special_function f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l3330_333091


namespace NUMINAMATH_CALUDE_root_sum_pq_l3330_333037

theorem root_sum_pq (p q : ℝ) : 
  (2 * Complex.I ^ 2 + p * Complex.I + q = 0) →
  (2 * (-3 + 2 * Complex.I) ^ 2 + p * (-3 + 2 * Complex.I) + q = 0) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_root_sum_pq_l3330_333037


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_increase_decrease_intervals_l3330_333090

noncomputable section

-- Define the function f(x) = ln x - ax
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem for the tangent line equation when a = -2
theorem tangent_line_at_x_1 (a : ℝ) (h : a = -2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → m * x + b = y) ∧ 
  (m * x - y + b = 0 ↔ 3 * x - y - 1 = 0) :=
sorry

-- Theorem for the intervals of increase and decrease
theorem increase_decrease_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂ : ℝ, 1/a < x₁ → x₁ < x₂ → f a x₂ < f a x₁)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_increase_decrease_intervals_l3330_333090


namespace NUMINAMATH_CALUDE_xy_product_l3330_333032

theorem xy_product (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (25:ℝ)^(x+y) / (5:ℝ)^(7*y) = 3125) : 
  x * y = 75 := by sorry

end NUMINAMATH_CALUDE_xy_product_l3330_333032


namespace NUMINAMATH_CALUDE_square_diff_equals_32_l3330_333025

theorem square_diff_equals_32 (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_diff_equals_32_l3330_333025


namespace NUMINAMATH_CALUDE_min_value_of_function_l3330_333082

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3330_333082


namespace NUMINAMATH_CALUDE_ivan_always_wins_l3330_333006

/-- Represents the state of the game -/
structure GameState where
  ivan_bars : ℕ  -- Number of bars Ivan has
  chest_bars : ℕ  -- Number of bars in the chest
  last_move : ℕ  -- Number of bars moved in the last turn

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (bars : ℕ) : Prop :=
  bars ≠ 0 ∧ bars ≠ state.last_move ∧ 
  (state.ivan_bars + state.chest_bars = 13 ∨ state.ivan_bars + state.chest_bars = 14)

/-- Defines the game's winning condition for Ivan -/
def ivan_wins (state : GameState) : Prop :=
  state.ivan_bars = 13

/-- Theorem stating that Ivan can always win the game -/
theorem ivan_always_wins (initial_bars : ℕ) 
  (h : initial_bars = 13 ∨ initial_bars = 14) : 
  ∃ (strategy : GameState → ℕ), 
    ∀ (koschei_strategy : GameState → ℕ),
      ∃ (final_state : GameState),
        ivan_wins final_state ∧
        (∀ (state : GameState), 
          valid_move state (strategy state) ∧
          valid_move state (koschei_strategy state)) :=
sorry

end NUMINAMATH_CALUDE_ivan_always_wins_l3330_333006


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3330_333033

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 3 ∨ x = 1)) ∧
  (∀ y : ℝ, 4*y^2 - 3*y ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3330_333033


namespace NUMINAMATH_CALUDE_no_always_largest_l3330_333049

theorem no_always_largest (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5) :
  ¬(∀ x y : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_no_always_largest_l3330_333049


namespace NUMINAMATH_CALUDE_definite_integral_2x_l3330_333038

theorem definite_integral_2x : ∫ x in (1:ℝ)..2, 2*x = 3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_l3330_333038


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3330_333001

theorem fraction_sum_zero (a b c : ℝ) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3330_333001


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3330_333009

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem linear_function_not_in_second_quadrant :
  ¬ ∃ x : ℝ, in_second_quadrant x (f x) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3330_333009


namespace NUMINAMATH_CALUDE_nuts_per_bag_l3330_333065

theorem nuts_per_bag (bags : ℕ) (students : ℕ) (nuts_per_student : ℕ) 
  (h1 : bags = 65)
  (h2 : students = 13)
  (h3 : nuts_per_student = 75) :
  (students * nuts_per_student) / bags = 15 := by
sorry

end NUMINAMATH_CALUDE_nuts_per_bag_l3330_333065


namespace NUMINAMATH_CALUDE_rectangular_formation_perimeter_l3330_333046

theorem rectangular_formation_perimeter (area : ℝ) (num_squares : ℕ) :
  area = 512 →
  num_squares = 8 →
  let square_side : ℝ := Real.sqrt (area / num_squares)
  let perimeter : ℝ := 2 * (4 * square_side + 3 * square_side)
  perimeter = 152 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_formation_perimeter_l3330_333046


namespace NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3330_333026

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

#eval binary_to_decimal [true, false, true, true]
#eval decimal_to_base5 13

end NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3330_333026


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3330_333071

theorem min_value_sum_reciprocals (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hsum : p + q + r = 3) :
  (1 / (p + 3*q) + 1 / (q + 3*r) + 1 / (r + 3*p)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3330_333071


namespace NUMINAMATH_CALUDE_car_hire_payment_l3330_333067

/-- Represents the car hiring scenario -/
structure CarHire where
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  payment_b : ℚ

/-- Calculates the total amount paid for hiring the car -/
def total_payment (hire : CarHire) : ℚ :=
  let rate := hire.payment_b / hire.hours_b
  rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total payment for the given scenario -/
theorem car_hire_payment :
  ∀ (hire : CarHire),
    hire.hours_a = 9 ∧
    hire.hours_b = 10 ∧
    hire.hours_c = 13 ∧
    hire.payment_b = 225 →
    total_payment hire = 720 := by
  sorry


end NUMINAMATH_CALUDE_car_hire_payment_l3330_333067


namespace NUMINAMATH_CALUDE_sum_divided_by_ten_l3330_333098

theorem sum_divided_by_ten : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_divided_by_ten_l3330_333098


namespace NUMINAMATH_CALUDE_slower_speed_percentage_l3330_333027

theorem slower_speed_percentage (D : ℝ) (S : ℝ) (S_slow : ℝ) 
    (h1 : D = S * 16)
    (h2 : D = S_slow * 40) :
  S_slow / S = 0.4 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_percentage_l3330_333027


namespace NUMINAMATH_CALUDE_race_course_length_correct_l3330_333062

/-- The length of a race course where two runners finish at the same time -/
def race_course_length : ℝ :=
  let speed_ratio : ℝ := 7
  let head_start : ℝ := 120
  140

theorem race_course_length_correct :
  let speed_ratio : ℝ := 7  -- A is 7 times faster than B
  let head_start : ℝ := 120 -- B starts 120 meters ahead
  let course_length := race_course_length
  course_length / speed_ratio = (course_length - head_start) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_course_length_correct_l3330_333062


namespace NUMINAMATH_CALUDE_distribute_two_four_x_minus_one_l3330_333085

theorem distribute_two_four_x_minus_one (x : ℝ) : 2 * (4 * x - 1) = 8 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_two_four_x_minus_one_l3330_333085


namespace NUMINAMATH_CALUDE_P_Q_disjoint_l3330_333050

def P : Set ℕ := {n | ∃ k, n = 2 * k^2 - 2 * k + 1}

def Q : Set ℕ := {n | n > 0 ∧ (Complex.I + 1)^(2*n) = 2^n * Complex.I}

theorem P_Q_disjoint : P ∩ Q = ∅ := by sorry

end NUMINAMATH_CALUDE_P_Q_disjoint_l3330_333050
