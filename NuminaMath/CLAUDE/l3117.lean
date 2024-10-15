import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_three_l3117_311724

theorem divisible_by_three (x y : ℤ) (h : 3 ∣ (x^2 + y^2)) : 3 ∣ x ∧ 3 ∣ y := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3117_311724


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_area_of_triangle_l3117_311745

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The determinant condition from the problem -/
def determinant_condition (t : Triangle) : Prop :=
  2 * t.c * Real.sin t.C = (2 * t.a - t.b) * Real.sin t.A + (2 * t.b - t.a) * Real.sin t.B

/-- Theorem 1: If the determinant condition holds, then C = π/3 -/
theorem angle_C_is_pi_third (t : Triangle) 
  (h : determinant_condition t) : t.C = Real.pi / 3 := by
  sorry

/-- Theorem 2: Area of the triangle under given conditions -/
theorem area_of_triangle (t : Triangle) 
  (h1 : Real.sin t.A = 4/5)
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.c = Real.sqrt 3) : 
  (1/2) * t.a * t.c * Real.sin t.B = (18 - 8 * Real.sqrt 3) / 25 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_angle_C_is_pi_third_area_of_triangle_l3117_311745


namespace NUMINAMATH_CALUDE_ned_video_game_earnings_l3117_311785

/-- Given the total number of games, non-working games, and price per working game,
    calculates the total earnings from selling the working games. -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63. -/
theorem ned_video_game_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

#eval calculate_earnings 15 6 7

end NUMINAMATH_CALUDE_ned_video_game_earnings_l3117_311785


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l3117_311771

/-- Represents the amount of money Chris had before his birthday -/
def money_before_birthday : ℕ := sorry

/-- Represents the amount of money Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- Represents the amount of money Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- Represents the amount of money Chris received from his parents -/
def parents_gift : ℕ := 75

/-- Represents the total amount of money Chris has now -/
def total_money_now : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before_birthday = 159 :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l3117_311771


namespace NUMINAMATH_CALUDE_range_of_a_l3117_311706

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - a*x + 2*a > 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  always_positive a ↔ (0 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3117_311706


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3117_311712

/-- Given a cube with surface area 6x^2, its volume is x^3 -/
theorem cube_volume_from_surface_area (x : ℝ) :
  let surface_area := 6 * x^2
  let side_length := Real.sqrt (surface_area / 6)
  let volume := side_length^3
  volume = x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3117_311712


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3117_311740

/-- A triangle with angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- A right triangle is a triangle with one 90-degree angle. -/
def RightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition that angles A and B are equal and twice angle C. -/
def AngleCondition (t : Triangle) : Prop :=
  t.A = t.B ∧ t.A = 2 * t.C

theorem not_necessarily_right_triangle :
  ∃ t : Triangle, AngleCondition t ∧ ¬RightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3117_311740


namespace NUMINAMATH_CALUDE_exists_unreachable_positive_configuration_l3117_311760

/-- Represents a cell in the grid -/
inductive Cell
| Plus
| Minus

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents the allowed operations -/
inductive Operation
| Flip3x3 (row col : Fin 6)  -- Top-left corner of 3x3 square
| Flip4x4 (row col : Fin 5)  -- Top-left corner of 4x4 square

/-- Applies an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Checks if a grid is all positive -/
def isAllPositive (g : Grid) : Prop :=
  ∀ i j, g i j = Cell.Plus

/-- Theorem: There exists an initial grid configuration that cannot be transformed to all positive -/
theorem exists_unreachable_positive_configuration :
  ∃ (initial : Grid), ¬∃ (ops : List Operation), isAllPositive (ops.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unreachable_positive_configuration_l3117_311760


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3117_311722

theorem fraction_to_decimal : 13 / 243 = 0.00416 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3117_311722


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_l3117_311793

theorem four_digit_perfect_square : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  (∃ m : ℕ, n = m^2) ∧      -- perfect square
  (n / 100 = n % 100 + 1)   -- first two digits are one more than last two digits
  := by
  use 8281
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_l3117_311793


namespace NUMINAMATH_CALUDE_f_zero_at_three_l3117_311791

def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + r

theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -276 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l3117_311791


namespace NUMINAMATH_CALUDE_two_painter_time_l3117_311747

/-- The time taken for two painters to complete a wall together, given their individual rates -/
theorem two_painter_time (harish_rate ganpat_rate : ℝ) (harish_time ganpat_time : ℝ) :
  harish_rate = 1 / harish_time →
  ganpat_rate = 1 / ganpat_time →
  harish_time = 3 →
  ganpat_time = 6 →
  1 / (harish_rate + ganpat_rate) = 2 := by
  sorry

#check two_painter_time

end NUMINAMATH_CALUDE_two_painter_time_l3117_311747


namespace NUMINAMATH_CALUDE_problem_solution_l3117_311750

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3117_311750


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l3117_311719

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_decreasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (h1 : is_even f) (h2 : is_decreasing_on_positive f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) := by sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l3117_311719


namespace NUMINAMATH_CALUDE_problem_solution_l3117_311751

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2-x}

theorem problem_solution (x : ℝ) (C : Set ℝ) 
  (h1 : B x ⊆ A x) 
  (h2 : B x ∪ C = A x) : 
  x = -2 ∧ C = {3} := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l3117_311751


namespace NUMINAMATH_CALUDE_visit_either_not_both_l3117_311796

/-- The probability of visiting either Chile or Madagascar, but not both -/
theorem visit_either_not_both (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) :
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_visit_either_not_both_l3117_311796


namespace NUMINAMATH_CALUDE_machine_input_l3117_311708

theorem machine_input (x : ℝ) : 
  1.2 * ((3 * (x + 15) - 6) / 2)^2 = 35 → x = -9.4 := by
  sorry

end NUMINAMATH_CALUDE_machine_input_l3117_311708


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3117_311766

theorem min_value_squared_sum (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  ∃ (m : ℝ), m = (Real.sqrt 5 + 1) / 4 ∧ ∀ (x y : ℝ), x^2 + 2*x*y - 3*y^2 = 1 → x^2 + y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3117_311766


namespace NUMINAMATH_CALUDE_initial_rabbits_forest_rabbits_l3117_311763

theorem initial_rabbits (initial_weasels : ℕ) (foxes : ℕ) (weeks : ℕ) 
  (weasels_caught_per_fox_per_week : ℕ) (rabbits_caught_per_fox_per_week : ℕ)
  (remaining_animals : ℕ) : ℕ :=
  let total_caught := foxes * weeks * (weasels_caught_per_fox_per_week + rabbits_caught_per_fox_per_week)
  let total_initial := remaining_animals + total_caught
  total_initial - initial_weasels

theorem forest_rabbits : 
  initial_rabbits 100 3 3 4 2 96 = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_rabbits_forest_rabbits_l3117_311763


namespace NUMINAMATH_CALUDE_largest_A_k_l3117_311755

def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

theorem largest_A_k : 
  ∃ (k : ℕ), k = 166 ∧ 
  (∀ (j : ℕ), j ≤ 1000 → A k ≥ A j) := by
sorry

end NUMINAMATH_CALUDE_largest_A_k_l3117_311755


namespace NUMINAMATH_CALUDE_james_socks_l3117_311705

/-- The total number of socks James has -/
def total_socks (red_pairs black_pairs white_socks : ℕ) : ℕ :=
  2 * red_pairs + 2 * black_pairs + white_socks

/-- Theorem stating the total number of socks James has -/
theorem james_socks : 
  ∀ (red_pairs black_pairs white_socks : ℕ),
    red_pairs = 20 →
    black_pairs = red_pairs / 2 →
    white_socks = 2 * (2 * red_pairs + 2 * black_pairs) →
    total_socks red_pairs black_pairs white_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_l3117_311705


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l3117_311764

theorem consecutive_integers_fourth_power_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 12 * (3 * x + 3) - 24 →
  x^4 + (x + 1)^4 + (x + 2)^4 = 98 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l3117_311764


namespace NUMINAMATH_CALUDE_child_height_end_of_year_l3117_311790

/-- Calculates the child's height at the end of the school year given initial height and growth rates -/
def final_height (initial_height : ℝ) (rate1 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  initial_height + (3 * rate1) + (3 * rate2) + (6 * rate3)

/-- Theorem stating that the child's height at the end of the school year is 43.3 inches -/
theorem child_height_end_of_year :
  final_height 38.5 0.5 0.3 0.4 = 43.3 := by
  sorry

#eval final_height 38.5 0.5 0.3 0.4

end NUMINAMATH_CALUDE_child_height_end_of_year_l3117_311790


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_possible_values_complete_l3117_311743

-- Define the set of possible values for |a + b + c|
def PossibleValues : Set ℝ := {1, 2, 3}

-- Main theorem
theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 →
  Complex.abs (a + b + c) ∈ PossibleValues := by
  sorry

-- Completeness of the set of possible values
theorem possible_values_complete (x : ℝ) :
  x ∈ PossibleValues →
  ∃ (a b c : ℂ), Complex.abs a = 1 ∧
                  Complex.abs b = 1 ∧
                  Complex.abs c = 1 ∧
                  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 ∧
                  Complex.abs (a + b + c) = x := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_possible_values_complete_l3117_311743


namespace NUMINAMATH_CALUDE_prob_white_balls_same_color_l3117_311768

/-- The number of white balls in the box -/
def white_balls : ℕ := 6

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of balls drawn -/
def balls_drawn : ℕ := 3

/-- The probability that the drawn balls are white, given they are the same color -/
def prob_white_given_same_color : ℚ := 2/3

theorem prob_white_balls_same_color :
  let total_same_color := Nat.choose white_balls balls_drawn + Nat.choose black_balls balls_drawn
  let prob := (Nat.choose white_balls balls_drawn : ℚ) / total_same_color
  prob = prob_white_given_same_color := by sorry

end NUMINAMATH_CALUDE_prob_white_balls_same_color_l3117_311768


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3117_311702

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def is_multiple_of_odd_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ Odd p ∧ ∃ k : ℕ, n = k * (p^2)

def is_internal_angle (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 2 ∧ n = (m - 2) * 180 / m

def is_proper_factor (a b : ℕ) : Prop :=
  a ≠ 1 ∧ a ≠ b ∧ b % a = 0

theorem cross_number_puzzle :
  ∀ (across_1 across_3 across_5 down_1 down_2 down_4 : ℕ),
    across_1 > 0 ∧ across_3 > 0 ∧ across_5 > 0 ∧
    down_1 > 0 ∧ down_2 > 0 ∧ down_4 > 0 →
    is_multiple_of_7 across_1 →
    across_5 > 10 →
    is_multiple_of_odd_prime_square down_1 ∧ ¬(∃ k : ℕ, down_1 = k^2) ∧ ¬(∃ k : ℕ, down_1 = k^3) →
    is_internal_angle down_2 ∧ 170 < down_2 ∧ down_2 < 180 →
    is_proper_factor down_4 across_5 ∧ ¬is_proper_factor down_4 down_1 →
    across_3 = 961 := by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3117_311702


namespace NUMINAMATH_CALUDE_find_number_l3117_311799

theorem find_number : ∃ n : ℕ, 72519 * n = 724827405 ∧ n = 10005 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3117_311799


namespace NUMINAMATH_CALUDE_inequality_proof_l3117_311742

theorem inequality_proof (x : ℝ) : 
  (|(7 - x) / 4| < 3) ∧ (x ≥ 0) → (0 ≤ x ∧ x < 19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3117_311742


namespace NUMINAMATH_CALUDE_jerry_lawn_mowing_money_l3117_311733

/-- The amount of money Jerry made mowing lawns -/
def M : ℝ := sorry

/-- The amount of money Jerry made from weed eating -/
def weed_eating_money : ℝ := 31

/-- The number of weeks Jerry's money would last -/
def weeks : ℝ := 9

/-- The amount Jerry would spend per week -/
def weekly_spending : ℝ := 5

theorem jerry_lawn_mowing_money :
  M = 14 :=
by
  have total_money : M + weed_eating_money = weeks * weekly_spending := by sorry
  sorry

end NUMINAMATH_CALUDE_jerry_lawn_mowing_money_l3117_311733


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3117_311717

/-- Represents a school with a given number of students -/
structure School where
  students : ℕ

/-- Calculates the number of students to be sampled from a school in a stratified sample -/
def stratifiedSampleSize (school : School) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (school.students * sampleSize) / totalStudents

theorem stratified_sample_theorem (schoolA schoolB schoolC : School) 
    (h1 : schoolA.students = 3600)
    (h2 : schoolB.students = 5400)
    (h3 : schoolC.students = 1800)
    (totalSampleSize : ℕ)
    (h4 : totalSampleSize = 90) :
  let totalStudents := schoolA.students + schoolB.students + schoolC.students
  (stratifiedSampleSize schoolA totalStudents totalSampleSize = 30) ∧
  (stratifiedSampleSize schoolB totalStudents totalSampleSize = 45) ∧
  (stratifiedSampleSize schoolC totalStudents totalSampleSize = 15) := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_theorem_l3117_311717


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3117_311727

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3117_311727


namespace NUMINAMATH_CALUDE_outer_digits_swap_l3117_311781

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  units_range : units ≥ 0 ∧ units ≤ 9

/-- Convert a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem outer_digits_swap (n : ThreeDigitNumber) 
  (h1 : n.toNum + 45 = 100 * n.hundreds + 10 * n.units + n.tens)
  (h2 : n.toNum = 100 * n.tens + 10 * n.hundreds + n.units + 270) :
  100 * n.units + 10 * n.tens + n.hundreds = n.toNum + 198 := by
  sorry

#check outer_digits_swap

end NUMINAMATH_CALUDE_outer_digits_swap_l3117_311781


namespace NUMINAMATH_CALUDE_hours_worked_l3117_311770

def hourly_wage : ℝ := 3.25
def total_earned : ℝ := 26

theorem hours_worked : 
  (total_earned / hourly_wage : ℝ) = 8 := by sorry

end NUMINAMATH_CALUDE_hours_worked_l3117_311770


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3117_311789

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3117_311789


namespace NUMINAMATH_CALUDE_find_k_l3117_311709

theorem find_k : ∃ k : ℕ, (1/2)^16 * (1/81)^k = 1/(18^16) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3117_311709


namespace NUMINAMATH_CALUDE_sum_of_roots_l3117_311752

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → 
  let M : Set ℝ := {a^2 - 4*a, -1}
  let N : Set ℝ := {b^2 - 4*b + 1, -2}
  ∃ f : ℝ → ℝ, (∀ x ∈ M, f x = x ∧ f x ∈ N) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3117_311752


namespace NUMINAMATH_CALUDE_average_age_problem_l3117_311732

theorem average_age_problem (age_a age_b age_c : ℝ) :
  age_b = 20 →
  (age_a + age_c) / 2 = 29 →
  (age_a + age_b + age_c) / 3 = 26 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l3117_311732


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3117_311756

/-- The common ratio of the geometric series 2/3 + 4/9 + 8/27 + ... is 2/3 -/
theorem geometric_series_common_ratio : 
  let a : ℕ → ℚ := fun n => (2 / 3) * (2 / 3)^n
  ∀ n : ℕ, a (n + 1) / a n = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3117_311756


namespace NUMINAMATH_CALUDE_building_shadow_length_l3117_311758

/-- Given a flagstaff and a building with their respective heights and the flagstaff's shadow length,
    calculate the length of the building's shadow under similar conditions. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5) :
  (building_height * flagstaff_shadow) / flagstaff_height = 28.75 :=
by sorry

end NUMINAMATH_CALUDE_building_shadow_length_l3117_311758


namespace NUMINAMATH_CALUDE_set_union_problem_l3117_311787

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {0, a}
  let B : Set ℕ := {2^a, b}
  A ∪ B = {0, 1, 2} → b = 0 ∨ b = 1 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3117_311787


namespace NUMINAMATH_CALUDE_right_triangle_exists_l3117_311726

/-- Checks if three line segments can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_exists :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧
  (a = 3 ∧ b = 4 ∧ c = 5) ∧
  ¬(is_right_triangle 2 3 4) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_exists_l3117_311726


namespace NUMINAMATH_CALUDE_guitar_center_discount_l3117_311700

/-- The discount offered by Guitar Center for a guitar with a suggested retail price of $1000,
    given that Guitar Center has a $100 shipping fee, Sweetwater has a 10% discount with free shipping,
    and the difference in final price between the two stores is $50. -/
theorem guitar_center_discount (suggested_price : ℕ) (gc_shipping : ℕ) (sw_discount_percent : ℕ) (price_difference : ℕ) :
  suggested_price = 1000 →
  gc_shipping = 100 →
  sw_discount_percent = 10 →
  price_difference = 50 →
  ∃ (gc_discount : ℕ), gc_discount = 150 :=
by sorry

end NUMINAMATH_CALUDE_guitar_center_discount_l3117_311700


namespace NUMINAMATH_CALUDE_brahmagupta_formula_l3117_311767

/-- Represents a convex quadrilateral ABCD with side lengths a, b, c, d and diagonal lengths m, n -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The Brahmagupta's formula for a convex quadrilateral -/
theorem brahmagupta_formula (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
by sorry

end NUMINAMATH_CALUDE_brahmagupta_formula_l3117_311767


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3117_311715

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).re = 0 → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3117_311715


namespace NUMINAMATH_CALUDE_mismatched_pairs_count_l3117_311798

/-- Represents a sock with a color and a pattern -/
structure Sock :=
  (color : String)
  (pattern : String)

/-- Represents a pair of socks -/
def SockPair := Sock × Sock

/-- Checks if two socks are mismatched (different color and pattern) -/
def isMismatched (s1 s2 : Sock) : Bool :=
  s1.color ≠ s2.color ∧ s1.pattern ≠ s2.pattern

/-- The set of all sock pairs -/
def allPairs : List SockPair := [
  (⟨"Red", "Striped"⟩, ⟨"Red", "Striped"⟩),
  (⟨"Green", "Polka-dotted"⟩, ⟨"Green", "Polka-dotted"⟩),
  (⟨"Blue", "Checked"⟩, ⟨"Blue", "Checked"⟩),
  (⟨"Yellow", "Floral"⟩, ⟨"Yellow", "Floral"⟩),
  (⟨"Purple", "Plaid"⟩, ⟨"Purple", "Plaid"⟩)
]

/-- Theorem: The number of unique mismatched pairs is 10 -/
theorem mismatched_pairs_count :
  (List.length (List.filter
    (fun (p : Sock × Sock) => isMismatched p.1 p.2)
    (List.join (List.map
      (fun (p1 : SockPair) => List.map
        (fun (p2 : SockPair) => (p1.1, p2.2))
        allPairs)
      allPairs)))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mismatched_pairs_count_l3117_311798


namespace NUMINAMATH_CALUDE_system_solution_l3117_311741

theorem system_solution (x y : ℝ) 
  (h1 : Real.log (x + y) - Real.log 5 = Real.log x + Real.log y - Real.log 6)
  (h2 : Real.log x / (Real.log (y + 6) - (Real.log y + Real.log 6)) = -1)
  (hx : x > 0)
  (hy : y > 0)
  (hny : y ≠ 6/5)
  (hyb : y > -6) :
  x = 2 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3117_311741


namespace NUMINAMATH_CALUDE_y_derivative_l3117_311730

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + 
  (1 / 3) * (3 * x - 1) / (3 * x^2 - 2 * x + 1)

theorem y_derivative (x : ℝ) :
  deriv y x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l3117_311730


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3117_311794

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) (h3 : x > y) :
  y = 14 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3117_311794


namespace NUMINAMATH_CALUDE_inequality_proof_l3117_311720

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3117_311720


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l3117_311754

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l3117_311754


namespace NUMINAMATH_CALUDE_coin_counting_fee_percentage_l3117_311780

def coinValue (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

def totalValue (quarters dimes nickels pennies : ℕ) : ℚ :=
  quarters * coinValue "quarter" + 
  dimes * coinValue "dime" + 
  nickels * coinValue "nickel" + 
  pennies * coinValue "penny"

theorem coin_counting_fee_percentage 
  (quarters dimes nickels pennies : ℕ) 
  (amountAfterFee : ℚ) : 
  quarters = 76 → 
  dimes = 85 → 
  nickels = 20 → 
  pennies = 150 → 
  amountAfterFee = 27 → 
  (totalValue quarters dimes nickels pennies - amountAfterFee) / 
  (totalValue quarters dimes nickels pennies) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_counting_fee_percentage_l3117_311780


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3117_311769

/-- An arithmetic sequence with first four terms 1, x, a, and 2x has x = 2 -/
theorem arithmetic_sequence_x_value (x a : ℝ) : 
  (∃ d : ℝ, x = 1 + d ∧ a = x + d ∧ 2*x = a + d) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3117_311769


namespace NUMINAMATH_CALUDE_second_number_is_seventeen_l3117_311788

theorem second_number_is_seventeen (first_number second_number third_number : ℕ) :
  first_number = 16 →
  third_number = 20 →
  3 * first_number + 3 * second_number + 3 * third_number + 11 = 170 →
  second_number = 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_seventeen_l3117_311788


namespace NUMINAMATH_CALUDE_cone_base_radius_l3117_311731

/-- A cone with surface area 3π whose lateral surface unfolds into a semicircle has base radius 1. -/
theorem cone_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem)
  3 * π * r^2 = 3 * π → -- surface area condition
  π * (2 * r) = 2 * π * r → -- lateral surface unfolds into semicircle condition
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3117_311731


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3117_311783

theorem divisibility_equivalence (n : ℕ+) :
  (n.val^5 + 5^n.val) % 11 = 0 ↔ (n.val^5 * 5^n.val + 1) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3117_311783


namespace NUMINAMATH_CALUDE_books_sold_l3117_311713

/-- Given Kaleb's initial and final book counts, along with the number of new books bought,
    prove the number of books he sold. -/
theorem books_sold (initial : ℕ) (new_bought : ℕ) (final : ℕ) 
    (h1 : initial = 34) 
    (h2 : new_bought = 7) 
    (h3 : final = 24) : 
  initial - final + new_bought = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l3117_311713


namespace NUMINAMATH_CALUDE_triangle_projection_shapes_l3117_311704

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- Possible projection shapes -/
inductive ProjectionShape
  | Angle
  | Strip
  | TwoAngles
  | Triangle
  | CompositeShape

/-- Function to project a triangle onto a plane from a point -/
def project (t : Triangle3D) (p : Plane3D) (o : Point3D) : ProjectionShape :=
  sorry

/-- Theorem stating the possible projection shapes -/
theorem triangle_projection_shapes (t : Triangle3D) (p : Plane3D) (o : Point3D) 
  (h : o ∉ {x : Point3D | t.a.z = t.b.z ∧ t.b.z = t.c.z}) :
  ∃ (shape : ProjectionShape), project t p o = shape :=
sorry

end NUMINAMATH_CALUDE_triangle_projection_shapes_l3117_311704


namespace NUMINAMATH_CALUDE_normal_force_wooden_blocks_l3117_311718

/-- The normal force from a system of wooden blocks to a table -/
theorem normal_force_wooden_blocks
  (M m : ℝ)  -- Masses of the larger block and smaller cubes
  (α β : ℝ)  -- Angles of the sides of the larger block
  (hM : M > 0)  -- Mass of larger block is positive
  (hm : m > 0)  -- Mass of smaller cubes is positive
  (hα : 0 < α ∧ α < π/2)  -- α is between 0 and π/2
  (hβ : 0 < β ∧ β < π/2)  -- β is between 0 and π/2
  (g : ℝ)  -- Gravitational acceleration
  (hg : g > 0)  -- Gravitational acceleration is positive
  : ℝ :=
  M * g + m * g * (Real.cos α ^ 2 + Real.cos β ^ 2)

#check normal_force_wooden_blocks

end NUMINAMATH_CALUDE_normal_force_wooden_blocks_l3117_311718


namespace NUMINAMATH_CALUDE_unique_digit_sum_l3117_311792

theorem unique_digit_sum (A₁₂ B C D : ℕ) : 
  (∃! (B C D : ℕ), 
    (10 > A₁₂ ∧ A₁₂ > B ∧ B > C ∧ C > D ∧ D > 0) ∧
    (1000 * A₁₂ + 100 * B + 10 * C + D) - (1000 * D + 100 * C + 10 * B + A₁₂) = 
    (1000 * B + 100 * D + 10 * A₁₂ + C)) →
  B + C + D = 11 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_sum_l3117_311792


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3117_311776

theorem x_plus_y_value (x y : ℝ) (h1 : x + Real.cos y = 3005) 
  (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y) (h4 : y ≤ π) : 
  x + y = 3004 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3117_311776


namespace NUMINAMATH_CALUDE_not_p_and_q_l3117_311737

-- Define proposition p
def p : Prop := ∀ a b : ℝ, a > b → a > b^2

-- Define proposition q
def q : Prop := (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → x ≤ 1) ∧ 
                (∃ x : ℝ, x ≤ 1 ∧ x^2 + 2*x - 3 > 0)

-- Theorem to prove
theorem not_p_and_q : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_l3117_311737


namespace NUMINAMATH_CALUDE_prime_power_cube_plus_one_l3117_311773

theorem prime_power_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) :
  p ^ (x : ℕ) = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_power_cube_plus_one_l3117_311773


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l3117_311725

/-- Proves that an employee's pay after a raise and subsequent cut is 75% of their pay after the raise -/
theorem employee_pay_percentage (initial_pay : ℝ) (raise_percentage : ℝ) (final_pay : ℝ) : 
  initial_pay = 10 →
  raise_percentage = 20 →
  final_pay = 9 →
  final_pay / (initial_pay * (1 + raise_percentage / 100)) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l3117_311725


namespace NUMINAMATH_CALUDE_new_york_to_new_england_ratio_l3117_311735

/-- The population of New England -/
def new_england_population : ℕ := 2100000

/-- The combined population of New York and New England -/
def combined_population : ℕ := 3500000

/-- The population of New York -/
def new_york_population : ℕ := combined_population - new_england_population

/-- The ratio of New York's population to New England's population -/
theorem new_york_to_new_england_ratio :
  (new_york_population : ℚ) / (new_england_population : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_new_york_to_new_england_ratio_l3117_311735


namespace NUMINAMATH_CALUDE_beatrice_tv_ratio_l3117_311778

/-- Proves that the ratio of TVs Beatrice looked at in the online store to the first store is 3:1 -/
theorem beatrice_tv_ratio : 
  ∀ (first_store online_store auction_site total : ℕ),
  first_store = 8 →
  auction_site = 10 →
  total = 42 →
  first_store + online_store + auction_site = total →
  online_store / first_store = 3 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_ratio_l3117_311778


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l3117_311748

/-- Given that points (-4, y₁), (-1, y₂), and (5/3, y₃) lie on the graph of y = -x² - 4x + 5,
    prove that y₂ > y₁ > y₃ -/
theorem parabola_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -(-4)^2 - 4*(-4) + 5 →
  y₂ = -(-1)^2 - 4*(-1) + 5 →
  y₃ = -(5/3)^2 - 4*(5/3) + 5 →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_parabola_y_relationship_l3117_311748


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l3117_311711

theorem remainder_17_63_mod_7 : 17^63 ≡ 6 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l3117_311711


namespace NUMINAMATH_CALUDE_hcf_of_210_and_517_l3117_311729

theorem hcf_of_210_and_517 (lcm_value : ℕ) (a b : ℕ) (h_lcm : Nat.lcm a b = lcm_value) 
  (h_a : a = 210) (h_b : b = 517) (h_lcm_value : lcm_value = 2310) : Nat.gcd a b = 47 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_210_and_517_l3117_311729


namespace NUMINAMATH_CALUDE_linear_equation_with_integer_roots_l3117_311703

theorem linear_equation_with_integer_roots 
  (m : ℤ) (n : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : n = 1) 
  (h3 : ∃ x : ℤ, (m - 1) * x - 3 = 0) :
  m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_with_integer_roots_l3117_311703


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3117_311784

/-- The retail price of a machine given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  let selling_price := (1 - discount_rate) * retail_price
  let profit := profit_rate * wholesale_price
  wholesale_price = 81 ∧ discount_rate = 0.1 ∧ profit_rate = 0.2 →
  ∃ retail_price : ℝ, selling_price = wholesale_price + profit ∧ retail_price = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3117_311784


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3117_311707

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3117_311707


namespace NUMINAMATH_CALUDE_triangle_max_area_l3117_311739

/-- Given a triangle ABC with the following properties:
    1. (cos A / sin B) + (cos B / sin A) = 2
    2. The perimeter of the triangle is 12
    The maximum possible area of the triangle is 36(3 - 2√2) -/
theorem triangle_max_area (A B C : ℝ) (h1 : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2)
  (h2 : A + B + C = π) (h3 : Real.sin A > 0) (h4 : Real.sin B > 0) (h5 : Real.sin C > 0)
  (a b c : ℝ) (h6 : a + b + c = 12) (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : a / Real.sin A = b / Real.sin B) (h11 : b / Real.sin B = c / Real.sin C) :
  (1/2) * a * b * Real.sin C ≤ 36 * (3 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3117_311739


namespace NUMINAMATH_CALUDE_parabola_sum_l3117_311721

def parabola (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

theorem parabola_sum (a b c : ℝ) :
  parabola a b c (-6) = 7 →
  parabola a b c (-4) = 5 →
  a + b + c = -35/2 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l3117_311721


namespace NUMINAMATH_CALUDE_bamboo_problem_l3117_311757

theorem bamboo_problem (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l3117_311757


namespace NUMINAMATH_CALUDE_equation_positive_root_l3117_311753

theorem equation_positive_root (n : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ n / (x - 1) + 2 / (1 - x) = 1) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l3117_311753


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3117_311775

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + x^2 + 6 * x - 8) = x^3 + 3 * x^2 + 3 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3117_311775


namespace NUMINAMATH_CALUDE_cube_opposite_color_l3117_311772

/-- Represents the colors of the squares --/
inductive Color
  | P | C | M | S | L | K

/-- Represents the faces of a cube --/
inductive Face
  | Top | Bottom | Front | Back | Left | Right

/-- Represents a cube formed by six hinged squares --/
structure Cube where
  faces : Face → Color

/-- Defines the opposite face relationship --/
def opposite_face : Face → Face
  | Face.Top    => Face.Bottom
  | Face.Bottom => Face.Top
  | Face.Front  => Face.Back
  | Face.Back   => Face.Front
  | Face.Left   => Face.Right
  | Face.Right  => Face.Left

theorem cube_opposite_color (c : Cube) :
  c.faces Face.Top = Color.M →
  c.faces Face.Front = Color.L →
  c.faces (opposite_face Face.Front) = Color.K :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_color_l3117_311772


namespace NUMINAMATH_CALUDE_remainder_of_b_86_mod_50_l3117_311762

theorem remainder_of_b_86_mod_50 : (7^86 + 9^86) % 50 = 40 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_b_86_mod_50_l3117_311762


namespace NUMINAMATH_CALUDE_not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l3117_311734

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬is_neighboring_root_equation 1 1 (-6) :=
sorry

/-- Theorem for the second equation -/
theorem is_neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l3117_311734


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3117_311777

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3117_311777


namespace NUMINAMATH_CALUDE_popcorn_per_serving_l3117_311716

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn : ℕ := 90

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn : ℕ := 60

/-- The number of Jared's friends -/
def num_friends : ℕ := 3

/-- The number of servings Jared should order -/
def num_servings : ℕ := 9

/-- Theorem stating that the number of pieces of popcorn in a serving is 30 -/
theorem popcorn_per_serving : 
  (jared_popcorn + num_friends * friend_popcorn) / num_servings = 30 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_per_serving_l3117_311716


namespace NUMINAMATH_CALUDE_circle_equation_implies_a_eq_neg_one_l3117_311786

/-- A circle equation in the form x^2 + by^2 + cx + d = 0 --/
structure CircleEquation where
  b : ℝ
  c : ℝ
  d : ℝ

/-- Condition for an equation to represent a circle --/
def is_circle (eq : CircleEquation) : Prop :=
  eq.b = 1 ∧ eq.b ≠ 0

/-- The given equation x^2 + (a+2)y^2 + 2ax + a = 0 --/
def given_equation (a : ℝ) : CircleEquation :=
  { b := a + 2
  , c := 2 * a
  , d := a }

theorem circle_equation_implies_a_eq_neg_one :
  ∀ a : ℝ, is_circle (given_equation a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_a_eq_neg_one_l3117_311786


namespace NUMINAMATH_CALUDE_uniform_transform_l3117_311795

/-- A uniform random number between 0 and 1 -/
def uniform_random_01 : Set ℝ := Set.Icc 0 1

/-- The transformation function -/
def transform (x : ℝ) : ℝ := x * 5 - 2

/-- The set of numbers between -2 and 3 -/
def target_set : Set ℝ := Set.Icc (-2) 3

theorem uniform_transform :
  ∀ (a₁ : ℝ), a₁ ∈ uniform_random_01 → transform a₁ ∈ target_set :=
sorry

end NUMINAMATH_CALUDE_uniform_transform_l3117_311795


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_all_even_digits_l3117_311701

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  100000000 > n ∧ n ≥ 10000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_with_all_even_digits :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_all_even_digits_l3117_311701


namespace NUMINAMATH_CALUDE_total_handshakes_l3117_311714

/-- The total number of handshakes in a group of boys with specific conditions -/
theorem total_handshakes (n : ℕ) (l : ℕ) (f : ℕ) (h : ℕ) : 
  n = 15 → l = 5 → f = 3 → h = 2 → 
  (n * (n - 1)) / 2 - (l * (n - l)) - f * h = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l3117_311714


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3117_311749

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3117_311749


namespace NUMINAMATH_CALUDE_china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l3117_311723

-- Define the data types
structure GDPData where
  year_code : ℕ
  gdp : ℝ

-- Define the given data
def china_gdp_data : List GDPData := [
  ⟨1, 8.5⟩, ⟨2, 9.6⟩, ⟨3, 10.4⟩, ⟨4, 11⟩, ⟨5, 11.1⟩, ⟨6, 12.1⟩, ⟨7, 13.6⟩
]

-- Define the sums given in the problem
def sum_y : ℝ := 76.3
def sum_xy : ℝ := 326.2

-- Define the US GDP in 2018
def us_gdp_2018 : ℝ := 20.5

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ := 0.75 * x + 7.9

-- Theorem statement
theorem china_gdp_surpass_us (n : ℕ) :
  (linear_regression (n + 7 : ℝ) ≥ us_gdp_2018) ↔ (n + 2021 ≥ 2028) := by
  sorry

-- Prove that the linear regression equation is correct
theorem correct_regression_equation :
  ∀ x, linear_regression x = 0.75 * x + 7.9 := by
  sorry

-- Prove that China's GDP will surpass US 2018 GDP in 2028
theorem china_gdp_surpass_us_in_2028 :
  ∃ n : ℕ, n + 2021 = 2028 ∧ linear_regression (n + 7 : ℝ) ≥ us_gdp_2018 := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l3117_311723


namespace NUMINAMATH_CALUDE_pancake_price_l3117_311710

/-- Janina's pancake stand problem -/
theorem pancake_price (daily_rent : ℝ) (daily_supplies : ℝ) (pancakes_to_cover_expenses : ℕ) :
  daily_rent = 30 ∧ daily_supplies = 12 ∧ pancakes_to_cover_expenses = 21 →
  (daily_rent + daily_supplies) / pancakes_to_cover_expenses = 2 :=
by sorry

end NUMINAMATH_CALUDE_pancake_price_l3117_311710


namespace NUMINAMATH_CALUDE_simplify_expression_l3117_311728

theorem simplify_expression (a : ℝ) (h : -1 < a ∧ a < 0) :
  Real.sqrt ((a + 1/a)^2 - 4) + Real.sqrt ((a - 1/a)^2 + 4) = -2/a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3117_311728


namespace NUMINAMATH_CALUDE_vector_problem_l3117_311765

/-- Given vectors a, b, c in ℝ², prove the coordinates of c and the cosine of the angle between a and b -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (-Real.sqrt 2, 1) →
  (c.1 * c.1 + c.2 * c.2 = 4) →
  (∃ (k : ℝ), c = k • a) →
  (b.1 * b.1 + b.2 * b.2 = 2) →
  ((a.1 + 3 * b.1) * (a.1 - b.1) + (a.2 + 3 * b.2) * (a.2 - b.2) = 0) →
  ((c = (-2 * Real.sqrt 6 / 3, 2 * Real.sqrt 3 / 3)) ∨ 
   (c = (2 * Real.sqrt 6 / 3, -2 * Real.sqrt 3 / 3))) ∧
  ((a.1 * b.1 + a.2 * b.2) / 
   (Real.sqrt (a.1 * a.1 + a.2 * a.2) * Real.sqrt (b.1 * b.1 + b.2 * b.2)) = Real.sqrt 6 / 4) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3117_311765


namespace NUMINAMATH_CALUDE_tv_price_reduction_l3117_311797

/-- Proves that the price reduction percentage is 10% given the conditions of the problem -/
theorem tv_price_reduction (x : ℝ) : 
  (1 - x / 100) * 1.85 = 1.665 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l3117_311797


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l3117_311759

theorem square_sum_geq_product (x y z : ℝ) (h : x + y + z ≥ x * y * z) : x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l3117_311759


namespace NUMINAMATH_CALUDE_buses_needed_l3117_311738

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def seats_per_bus : ℕ := 72

def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def total_people : ℕ := total_students + total_chaperones

theorem buses_needed : 
  ∃ n : ℕ, n * seats_per_bus ≥ total_people ∧ 
  ∀ m : ℕ, m * seats_per_bus ≥ total_people → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l3117_311738


namespace NUMINAMATH_CALUDE_consequences_of_only_some_A_are_B_l3117_311746

-- Define sets A and B
variable (A B : Set α)

-- Define the premise "Only some A are B"
def only_some_A_are_B : Prop := ∃ x ∈ A, x ∈ B ∧ ∃ y ∈ A, y ∉ B

-- Theorem stating the consequences
theorem consequences_of_only_some_A_are_B (h : only_some_A_are_B A B) :
  (¬ ∀ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) ∧
  (∃ x ∈ B, x ∈ A) ∧
  (∃ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_consequences_of_only_some_A_are_B_l3117_311746


namespace NUMINAMATH_CALUDE_complex_magnitude_bounds_l3117_311779

/-- Given a complex number z satisfying 2|z-3-3i| = |z|, prove that the maximum value of |z| is 6√2 and the minimum value of |z| is 2√2. -/
theorem complex_magnitude_bounds (z : ℂ) (h : 2 * Complex.abs (z - (3 + 3*I)) = Complex.abs z) :
  (∃ (w : ℂ), 2 * Complex.abs (w - (3 + 3*I)) = Complex.abs w ∧ Complex.abs w = 6 * Real.sqrt 2) ∧
  (∃ (v : ℂ), 2 * Complex.abs (v - (3 + 3*I)) = Complex.abs v ∧ Complex.abs v = 2 * Real.sqrt 2) ∧
  (∀ (u : ℂ), 2 * Complex.abs (u - (3 + 3*I)) = Complex.abs u → 
    2 * Real.sqrt 2 ≤ Complex.abs u ∧ Complex.abs u ≤ 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_bounds_l3117_311779


namespace NUMINAMATH_CALUDE_dormitory_problem_l3117_311744

theorem dormitory_problem (rooms : ℕ) (students : ℕ) : 
  (students % 4 = 19) ∧ 
  (0 < students - 6 * (rooms - 1)) ∧ 
  (students - 6 * (rooms - 1) < 6) →
  ((rooms = 10 ∧ students = 59) ∨ 
   (rooms = 11 ∧ students = 63) ∨ 
   (rooms = 12 ∧ students = 67)) :=
by sorry

end NUMINAMATH_CALUDE_dormitory_problem_l3117_311744


namespace NUMINAMATH_CALUDE_million_factorizations_l3117_311736

def million : ℕ := 1000000

/-- The number of ways to represent 1,000,000 as a product of three factors when order matters -/
def distinct_factorizations : ℕ := 784

/-- The number of ways to represent 1,000,000 as a product of three factors when order doesn't matter -/
def identical_factorizations : ℕ := 139

/-- Function to count the number of ways to represent a number as a product of three factors -/
def count_factorizations (n : ℕ) (order_matters : Bool) : ℕ := sorry

theorem million_factorizations :
  (count_factorizations million true = distinct_factorizations) ∧
  (count_factorizations million false = identical_factorizations) := by sorry

end NUMINAMATH_CALUDE_million_factorizations_l3117_311736


namespace NUMINAMATH_CALUDE_thompson_purchase_cost_l3117_311761

/-- The total cost of chickens and potatoes -/
def total_cost (num_chickens : ℕ) (chicken_price : ℝ) (potato_price : ℝ) : ℝ :=
  (num_chickens : ℝ) * chicken_price + potato_price

/-- Theorem: The total cost of 3 chickens at $3 each and a bag of potatoes at $6 is $15 -/
theorem thompson_purchase_cost : total_cost 3 3 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_thompson_purchase_cost_l3117_311761


namespace NUMINAMATH_CALUDE_same_terminal_side_l3117_311774

theorem same_terminal_side (k : ℤ) : ∃ k, (11 * π) / 6 = 2 * k * π - π / 6 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3117_311774


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l3117_311782

/-- A quadratic function f(x) = ax^2 + bx + c, where a ≠ 0 and a, b, c are constants. -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_solution_existence (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c h
  (f 6.17 = -0.03) →
  (f 6.18 = -0.01) →
  (f 6.19 = 0.02) →
  (f 6.20 = 0.04) →
  ∃ x : ℝ, (f x = 0) ∧ (6.18 < x) ∧ (x < 6.19) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l3117_311782
