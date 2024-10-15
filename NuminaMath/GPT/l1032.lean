import Mathlib

namespace NUMINAMATH_GPT_smallest_number_of_students_l1032_103218

theorem smallest_number_of_students (a b c : ℕ) (h1 : 4 * c = 3 * a) (h2 : 7 * b = 5 * a) (h3 : 10 * c = 9 * b) : a + b + c = 66 := sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1032_103218


namespace NUMINAMATH_GPT_arrangement_plans_l1032_103207

-- Definition of the problem conditions
def numChineseTeachers : ℕ := 2
def numMathTeachers : ℕ := 4
def numTeachersPerSchool : ℕ := 3

-- Definition of the problem statement
theorem arrangement_plans
  (c : ℕ) (m : ℕ) (s : ℕ)
  (h1 : numChineseTeachers = c)
  (h2 : numMathTeachers = m)
  (h3 : numTeachersPerSchool = s)
  (h4 : ∀ a b : ℕ, a + b = numChineseTeachers → a = 1 ∧ b = 1)
  (h5 : ∀ a b : ℕ, a + b = numMathTeachers → a = 2 ∧ b = 2) :
  (c * (1 / 2 * m * (m - 1) / 2)) = 12 :=
sorry

end NUMINAMATH_GPT_arrangement_plans_l1032_103207


namespace NUMINAMATH_GPT_eliminating_y_l1032_103240

theorem eliminating_y (x y : ℝ) (h1 : y = x + 3) (h2 : 2 * x - y = 5) : 2 * x - x - 3 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_eliminating_y_l1032_103240


namespace NUMINAMATH_GPT_sum_abc_of_quadrilateral_l1032_103262

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_abc_of_quadrilateral :
  let p1 := (0, 0)
  let p2 := (4, 3)
  let p3 := (5, 2)
  let p4 := (4, -1)
  let perimeter := 
    distance p1 p2 + distance p2 p3 + distance p3 p4 + distance p4 p1
  let a : ℤ := 1    -- corresponding to the equivalent simplified distances to √5 parts
  let b : ℤ := 2    -- corresponding to the equivalent simplified distances to √2 parts
  let c : ℤ := 9    -- rest constant integer simplified part
  a + b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_abc_of_quadrilateral_l1032_103262


namespace NUMINAMATH_GPT_xiao_zhao_physical_education_grade_l1032_103291

def classPerformanceScore : ℝ := 40
def midtermExamScore : ℝ := 50
def finalExamScore : ℝ := 45

def classPerformanceWeight : ℝ := 0.3
def midtermExamWeight : ℝ := 0.2
def finalExamWeight : ℝ := 0.5

def overallGrade : ℝ :=
  (classPerformanceScore * classPerformanceWeight) +
  (midtermExamScore * midtermExamWeight) +
  (finalExamScore * finalExamWeight)

theorem xiao_zhao_physical_education_grade : overallGrade = 44.5 := by
  sorry

end NUMINAMATH_GPT_xiao_zhao_physical_education_grade_l1032_103291


namespace NUMINAMATH_GPT_calculate_expression_l1032_103294

def thirteen_power_thirteen_div_thirteen_power_twelve := 13 ^ 13 / 13 ^ 12
def expression := (thirteen_power_thirteen_div_thirteen_power_twelve ^ 3) * (3 ^ 3)
/- We define the main statement to be proven -/
theorem calculate_expression : (expression / 2 ^ 6) = 926 := sorry

end NUMINAMATH_GPT_calculate_expression_l1032_103294


namespace NUMINAMATH_GPT_square_side_length_l1032_103230

-- Definition of the problem (statements)
theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s * s) (hA : A = 49) : s = 7 := 
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l1032_103230


namespace NUMINAMATH_GPT_two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l1032_103256

variable (n : ℕ) (F : ℕ → ℕ) (p : ℕ)

-- Condition: F_n = 2^{2^n} + 1
def F_n (n : ℕ) : ℕ := 2^(2^n) + 1

-- Assuming n >= 2
def n_ge_two (n : ℕ) : Prop := n ≥ 2

-- Assuming p is a prime factor of F_n
def prime_factor_of_F_n (p : ℕ) (n : ℕ) : Prop := p ∣ (F_n n) ∧ Prime p

-- Part a: 2 is a quadratic residue modulo p
theorem two_quadratic_residue_mod_p (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  ∃ x : ℕ, x^2 ≡ 2 [MOD p] := sorry

-- Part b: p ≡ 1 (mod 2^(n+2))
theorem p_congruent_one_mod_2_pow_n_plus_two (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  p ≡ 1 [MOD 2^(n+2)] := sorry

end NUMINAMATH_GPT_two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l1032_103256


namespace NUMINAMATH_GPT_a_gt_b_iff_a_ln_a_gt_b_ln_b_l1032_103215

theorem a_gt_b_iff_a_ln_a_gt_b_ln_b {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (a > b) ↔ (a + Real.log a > b + Real.log b) :=
by sorry

end NUMINAMATH_GPT_a_gt_b_iff_a_ln_a_gt_b_ln_b_l1032_103215


namespace NUMINAMATH_GPT_sixty_three_times_fifty_seven_l1032_103285

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_GPT_sixty_three_times_fifty_seven_l1032_103285


namespace NUMINAMATH_GPT_percentage_increase_of_x_l1032_103214

theorem percentage_increase_of_x (C x y : ℝ) (P : ℝ) (h1 : x * y = C) (h2 : (x * (1 + P / 100)) * (y * (5 / 6)) = C) :
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_of_x_l1032_103214


namespace NUMINAMATH_GPT_solve_for_x_l1032_103200

theorem solve_for_x (x : ℝ) (h1 : 1 - x^2 = 0) (h2 : x ≠ 1) : x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1032_103200


namespace NUMINAMATH_GPT_number_of_20_paise_coins_l1032_103268

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 336) (h2 : (20 / 100 : ℚ) * x + (25 / 100 : ℚ) * y = 71) :
    x = 260 :=
by
  sorry

end NUMINAMATH_GPT_number_of_20_paise_coins_l1032_103268


namespace NUMINAMATH_GPT_incorrect_square_root_0_2_l1032_103245

theorem incorrect_square_root_0_2 :
  (0.45)^2 = 0.2 ∧ (0.02)^2 ≠ 0.2 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_square_root_0_2_l1032_103245


namespace NUMINAMATH_GPT_range_of_m_l1032_103210

theorem range_of_m (m : ℝ) :
  (∀ x: ℝ, |x| + |x - 1| > m) ∨ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y) 
  → ¬ ((∀ x: ℝ, |x| + |x - 1| > m) ∧ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y)) 
  ↔ (1 ≤ m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1032_103210


namespace NUMINAMATH_GPT_solve_for_p_l1032_103203

def cubic_eq_has_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  5*(a:ℝ)^3 - 5*(p + 1)*(a:ℝ)^2 + (71*p - 1)*(a:ℝ) + 1 = 66*p ∧
  5*(b:ℝ)^3 - 5*(p + 1)*(b:ℝ)^2 + (71*p - 1)*(b:ℝ) + 1 = 66*p ∧
  5*(c:ℝ)^3 - 5*(p + 1)*(c:ℝ)^2 + (71*p - 1)*(c:ℝ) + 1 = 66*p

theorem solve_for_p : ∀ (p : ℝ), cubic_eq_has_natural_roots p → p = 76 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_p_l1032_103203


namespace NUMINAMATH_GPT_intersection_M_N_l1032_103233

def M : Set ℝ := { x | -5 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 3 } := 
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1032_103233


namespace NUMINAMATH_GPT_percentage_of_students_70_79_l1032_103264

def tally_90_100 := 6
def tally_80_89 := 9
def tally_70_79 := 8
def tally_60_69 := 6
def tally_50_59 := 3
def tally_below_50 := 1

def total_students := tally_90_100 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50

theorem percentage_of_students_70_79 : (tally_70_79 : ℚ) / total_students = 8 / 33 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_70_79_l1032_103264


namespace NUMINAMATH_GPT_squirrel_acorns_l1032_103205

theorem squirrel_acorns :
  ∀ (total_acorns : ℕ)
    (first_month_percent second_month_percent third_month_percent : ℝ)
    (first_month_consumed second_month_consumed third_month_consumed : ℝ),
    total_acorns = 500 →
    first_month_percent = 0.40 →
    second_month_percent = 0.30 →
    third_month_percent = 0.30 →
    first_month_consumed = 0.20 →
    second_month_consumed = 0.25 →
    third_month_consumed = 0.15 →
    let first_month_acorns := total_acorns * first_month_percent
    let second_month_acorns := total_acorns * second_month_percent
    let third_month_acorns := total_acorns * third_month_percent
    let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
    let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
    let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
    remaining_first_month + remaining_second_month + remaining_third_month = 400 := 
by
  intros 
    total_acorns
    first_month_percent second_month_percent third_month_percent
    first_month_consumed second_month_consumed third_month_consumed
    h_total
    h_first_percent
    h_second_percent
    h_third_percent
    h_first_consumed
    h_second_consumed
    h_third_consumed
  let first_month_acorns := total_acorns * first_month_percent
  let second_month_acorns := total_acorns * second_month_percent
  let third_month_acorns := total_acorns * third_month_percent
  let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
  let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
  let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
  sorry

end NUMINAMATH_GPT_squirrel_acorns_l1032_103205


namespace NUMINAMATH_GPT_shakes_indeterminable_l1032_103231

variable {B S C x : ℝ}

theorem shakes_indeterminable (h1 : 3 * B + x * S + C = 130) (h2 : 4 * B + 10 * S + C = 164.5) : 
  ¬ (∃ x, 3 * B + x * S + C = 130 ∧ 4 * B + 10 * S + C = 164.5) :=
by
  sorry

end NUMINAMATH_GPT_shakes_indeterminable_l1032_103231


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1032_103297

theorem solve_eq1 (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2 / 3) :=
by sorry

theorem solve_eq2 (x : ℝ) : x^2 - 4 * x - 5 = 0 ↔ (x = 5 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1032_103297


namespace NUMINAMATH_GPT_can_choose_P_l1032_103298

-- Define the objects in the problem,
-- types, constants, and assumptions as per the problem statement.

theorem can_choose_P (cube : ℝ) (P Q R S T A B C D : ℝ)
  (edge_length : cube = 10)
  (AR_RB_eq_CS_SB : ∀ AR RB CS SB, (AR / RB = 7 / 3) ∧ (CS / SB = 7 / 3))
  : ∃ P, 2 * (Q - R) = (P - Q) + (R - S) := by
  sorry

end NUMINAMATH_GPT_can_choose_P_l1032_103298


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1032_103206

variable {x y : ℝ}

theorem value_of_x_plus_y (h1 : 1 / x + 1 / y = 1) (h2 : 1 / x - 1 / y = 9) : x + y = -1 / 20 := 
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1032_103206


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1032_103246

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₁ : a 3 = 4) (h₂ : a 2 + a 4 = -10) (h₃ : |q| > 1) : 
  (a 0 + a 1 + a 2 + a 3 = -5) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1032_103246


namespace NUMINAMATH_GPT_number_of_officers_l1032_103237

theorem number_of_officers
  (avg_all : ℝ := 120)
  (avg_officer : ℝ := 420)
  (avg_non_officer : ℝ := 110)
  (num_non_officer : ℕ := 450) :
  ∃ O : ℕ, avg_all * (O + num_non_officer) = avg_officer * O + avg_non_officer * num_non_officer ∧ O = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_officers_l1032_103237


namespace NUMINAMATH_GPT_Q_has_negative_root_l1032_103250

def Q (x : ℝ) : ℝ := x^7 + 2 * x^5 + 5 * x^3 - x + 12

theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 :=
by
  sorry

end NUMINAMATH_GPT_Q_has_negative_root_l1032_103250


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1032_103284

theorem ratio_of_x_to_y (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 4 / 7) : x / y = 23 / 12 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1032_103284


namespace NUMINAMATH_GPT_total_teaching_hours_l1032_103243

-- Define the durations of the classes
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2

def math_class_duration : ℕ := 1
def science_class_duration : ℚ := 1.5
def history_class_duration : ℕ := 2

-- Define Eduardo's teaching time
def eduardo_total_time : ℚ :=
  eduardo_math_classes * math_class_duration +
  eduardo_science_classes * science_class_duration +
  eduardo_history_classes * history_class_duration

-- Define Frankie's teaching time (double the classes of Eduardo)
def frankie_total_time : ℚ :=
  2 * (eduardo_math_classes * math_class_duration) +
  2 * (eduardo_science_classes * science_class_duration) +
  2 * (eduardo_history_classes * history_class_duration)

-- Define the total teaching time for both Eduardo and Frankie
def total_teaching_time : ℚ :=
  eduardo_total_time + frankie_total_time

-- Theorem statement that both their total teaching time is 39 hours
theorem total_teaching_hours : total_teaching_time = 39 :=
by
  -- skipping the proof using sorry
  sorry

end NUMINAMATH_GPT_total_teaching_hours_l1032_103243


namespace NUMINAMATH_GPT_shaded_area_of_squares_is_20_l1032_103251

theorem shaded_area_of_squares_is_20 :
  ∀ (a b : ℝ), a = 2 → b = 6 → 
    (1/2) * a * a + (1/2) * b * b = 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_shaded_area_of_squares_is_20_l1032_103251


namespace NUMINAMATH_GPT_determine_x_l1032_103239

theorem determine_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  have : ∀ y : ℝ, (5 * y + 1) * (2 * x - 3) = 0 := 
    sorry
  have : (2 * x - 3) = 0 := 
    sorry
  show x = 3 / 2
  sorry

end NUMINAMATH_GPT_determine_x_l1032_103239


namespace NUMINAMATH_GPT_nails_for_smaller_planks_l1032_103232

def total_large_planks := 13
def nails_per_plank := 17
def total_nails := 229

def nails_for_large_planks : ℕ :=
  total_large_planks * nails_per_plank

theorem nails_for_smaller_planks :
  total_nails - nails_for_large_planks = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_nails_for_smaller_planks_l1032_103232


namespace NUMINAMATH_GPT_inequality_proof_l1032_103202

-- Define the main theorem to be proven.
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1032_103202


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1032_103253

theorem largest_angle_in_triangle : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A + B = 105 ∧ (A = B + 40)
  → (C = 75) :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1032_103253


namespace NUMINAMATH_GPT_total_sticks_used_l1032_103290

-- Define the number of sides an octagon has
def octagon_sides : ℕ := 8

-- Define the number of sticks each subsequent octagon needs, sharing one side with the previous one
def additional_sticks_per_octagon : ℕ := 7

-- Define the total number of octagons in the row
def total_octagons : ℕ := 700

-- Define the total number of sticks used
def total_sticks : ℕ := 
  let first_sticks := octagon_sides
  let additional_sticks := additional_sticks_per_octagon * (total_octagons - 1)
  first_sticks + additional_sticks

-- Statement to prove
theorem total_sticks_used : total_sticks = 4901 := by
  sorry

end NUMINAMATH_GPT_total_sticks_used_l1032_103290


namespace NUMINAMATH_GPT_A_work_days_l1032_103235

theorem A_work_days {total_wages B_share : ℝ} (B_work_days : ℝ) (total_wages_eq : total_wages = 5000) 
    (B_share_eq : B_share = 3333) (B_rate : ℝ) (correct_rate : B_rate = 1 / B_work_days) :
    ∃x : ℝ, B_share / (total_wages - B_share) = B_rate / (1 / x) ∧ total_wages - B_share = 5000 - B_share ∧ B_work_days = 10 -> x = 20 :=
by
  sorry

end NUMINAMATH_GPT_A_work_days_l1032_103235


namespace NUMINAMATH_GPT_average_leaves_per_hour_l1032_103252

theorem average_leaves_per_hour :
  let leaves_first_hour := 7
  let leaves_second_hour := 4
  let leaves_third_hour := 4
  let total_hours := 3
  let total_leaves := leaves_first_hour + leaves_second_hour + leaves_third_hour
  let average_leaves_per_hour := total_leaves / total_hours
  average_leaves_per_hour = 5 := by
  sorry

end NUMINAMATH_GPT_average_leaves_per_hour_l1032_103252


namespace NUMINAMATH_GPT_paint_remaining_after_two_days_l1032_103226

-- Define the conditions
def original_paint_amount := 1
def paint_used_day1 := original_paint_amount * (1/4)
def remaining_paint_after_day1 := original_paint_amount - paint_used_day1
def paint_used_day2 := remaining_paint_after_day1 * (1/2)
def remaining_paint_after_day2 := remaining_paint_after_day1 - paint_used_day2

-- Theorem to be proved
theorem paint_remaining_after_two_days :
  remaining_paint_after_day2 = (3/8) * original_paint_amount := sorry

end NUMINAMATH_GPT_paint_remaining_after_two_days_l1032_103226


namespace NUMINAMATH_GPT_non_neg_int_solutions_m_value_integer_values_of_m_l1032_103270

-- 1. Non-negative integer solutions of x + 2y = 3
theorem non_neg_int_solutions (x y : ℕ) :
  x + 2 * y = 3 ↔ (x = 3 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
sorry

-- 2. If (x, y) = (1, 1) satisfies both x + 2y = 3 and x + y = 2, then m = -4
theorem m_value (m : ℝ) :
  (1 + 2 * 1 = 3) ∧ (1 + 1 = 2) ∧ (1 - 2 * 1 + m * 1 = -5) → m = -4 :=
sorry

-- 3. Given n = 3, integer values of m are -2 or 0
theorem integer_values_of_m (m : ℤ) :
  ∃ x y : ℤ, 3 * x + 4 * y = 5 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0 :=
sorry

end NUMINAMATH_GPT_non_neg_int_solutions_m_value_integer_values_of_m_l1032_103270


namespace NUMINAMATH_GPT_percentage_increase_l1032_103229

variable (x y p : ℝ)

theorem percentage_increase (h : x = y + (p / 100) * y) : p = 100 * ((x - y) / y) := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l1032_103229


namespace NUMINAMATH_GPT_monthlyShoeSales_l1032_103272

-- Defining the conditions
def pairsSoldLastWeek := 27
def pairsSoldThisWeek := 12
def pairsNeededToMeetGoal := 41

-- Defining the question as a statement to prove
theorem monthlyShoeSales : pairsSoldLastWeek + pairsSoldThisWeek + pairsNeededToMeetGoal = 80 := by
  sorry

end NUMINAMATH_GPT_monthlyShoeSales_l1032_103272


namespace NUMINAMATH_GPT_find_angle_C_l1032_103254

variable {A B C a b c : ℝ}

theorem find_angle_C (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π) (h8 : a > 0) (h9 : b > 0) (h10 : c > 0) 
  (h11 : (a + b - c) * (a + b + c) = a * b) : C = 2 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_l1032_103254


namespace NUMINAMATH_GPT_count_color_patterns_l1032_103282

def regions := 6
def colors := 3

theorem count_color_patterns (h1 : regions = 6) (h2 : colors = 3) :
  3^6 - 3 * 2^6 + 3 * 1^6 = 540 := by
  sorry

end NUMINAMATH_GPT_count_color_patterns_l1032_103282


namespace NUMINAMATH_GPT_subset_property_l1032_103275

theorem subset_property : {2} ⊆ {x | x ≤ 10} := 
by 
  sorry

end NUMINAMATH_GPT_subset_property_l1032_103275


namespace NUMINAMATH_GPT_inequality_solution_set_l1032_103222

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x)} :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1032_103222


namespace NUMINAMATH_GPT_triangle_area_l1032_103289

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end NUMINAMATH_GPT_triangle_area_l1032_103289


namespace NUMINAMATH_GPT_range_of_m_l1032_103273

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1032_103273


namespace NUMINAMATH_GPT_order_of_f_l1032_103277

variable (f : ℝ → ℝ)

/-- Conditions:
1. f is an even function for all x ∈ ℝ
2. f is increasing on [0, +∞)
Question:
Prove that the order of f(-2), f(-π), f(3) is f(-2) < f(3) < f(-π) -/
theorem order_of_f (h_even : ∀ x : ℝ, f (-x) = f x)
                   (h_incr : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y) : 
                   f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry

end NUMINAMATH_GPT_order_of_f_l1032_103277


namespace NUMINAMATH_GPT_find_student_hourly_rate_l1032_103281

-- Definitions based on conditions
def janitor_work_time : ℝ := 8  -- Janitor can clean the school in 8 hours
def student_work_time : ℝ := 20  -- Student can clean the school in 20 hours
def janitor_hourly_rate : ℝ := 21  -- Janitor is paid $21 per hour
def cost_difference : ℝ := 8  -- The cost difference between janitor alone and both together is $8

-- The value we need to prove
def student_hourly_rate := 7

theorem find_student_hourly_rate
  (janitor_work_time : ℝ)
  (student_work_time : ℝ)
  (janitor_hourly_rate : ℝ)
  (cost_difference : ℝ) :
  S = 7 :=
by
  -- Calculations and logic can be filled here to prove the theorem
  sorry

end NUMINAMATH_GPT_find_student_hourly_rate_l1032_103281


namespace NUMINAMATH_GPT_intersect_inverse_l1032_103228

theorem intersect_inverse (c d : ℤ) (h1 : 2 * (-4) + c = d) (h2 : 2 * d + c = -4) : d = -4 := 
by
  sorry

end NUMINAMATH_GPT_intersect_inverse_l1032_103228


namespace NUMINAMATH_GPT_john_books_nights_l1032_103286

theorem john_books_nights (n : ℕ) (cost_per_night discount amount_paid : ℕ) 
  (h1 : cost_per_night = 250)
  (h2 : discount = 100)
  (h3 : amount_paid = 650)
  (h4 : amount_paid = cost_per_night * n - discount) : 
  n = 3 :=
by
  sorry

end NUMINAMATH_GPT_john_books_nights_l1032_103286


namespace NUMINAMATH_GPT_problem_statement_l1032_103276

theorem problem_statement (a b c x y z : ℂ)
  (h1 : a = (b + c) / (x - 2))
  (h2 : b = (c + a) / (y - 2))
  (h3 : c = (a + b) / (z - 2))
  (h4 : x * y + y * z + z * x = 67)
  (h5 : x + y + z = 2010) :
  x * y * z = -5892 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1032_103276


namespace NUMINAMATH_GPT_Ellen_won_17_legos_l1032_103227

theorem Ellen_won_17_legos (initial_legos : ℕ) (current_legos : ℕ) (h₁ : initial_legos = 2080) (h₂ : current_legos = 2097) : 
  current_legos - initial_legos = 17 := 
  by 
    sorry

end NUMINAMATH_GPT_Ellen_won_17_legos_l1032_103227


namespace NUMINAMATH_GPT_angle_A_is_120_max_sin_B_plus_sin_C_l1032_103271

-- Define the measures in degrees using real numbers
variable (a b c R : Real)
variable (A B C : ℝ) (sin cos : ℝ → ℝ)

-- Question 1: Prove A = 120 degrees given the initial condition
theorem angle_A_is_120
  (H1 : 2 * a * (sin A) = (2 * b + c) * (sin B) + (2 * c + b) * (sin C)) :
  A = 120 :=
by
  sorry

-- Question 2: Given the angles sum to 180 degrees and A = 120 degrees, prove the max value of sin B + sin C is 1
theorem max_sin_B_plus_sin_C
  (H2 : A + B + C = 180)
  (H3 : A = 120) :
  (sin B) + (sin C) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_is_120_max_sin_B_plus_sin_C_l1032_103271


namespace NUMINAMATH_GPT_triangle_perimeter_l1032_103234

-- Let the lengths of the sides of the triangle be a, b, c.
variables (a b c : ℕ)
-- To represent the sides with specific lengths as stated in the problem.
def side1 := 2
def side2 := 5

-- The condition that the third side must be an odd integer.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Setting up the third side based on the given conditions.
def third_side_odd (c : ℕ) : Prop := 3 < c ∧ c < 7 ∧ is_odd c

-- The perimeter of the triangle.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove.
theorem triangle_perimeter (c : ℕ) (h_odd : third_side_odd c) : perimeter side1 side2 c = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1032_103234


namespace NUMINAMATH_GPT_find_three_digit_number_l1032_103221

-- Define the function that calculates the total number of digits required
def total_digits (x : ℕ) : ℕ :=
  (if x >= 1 then 9 else 0) +
  (if x >= 10 then 90 * 2 else 0) +
  (if x >= 100 then 3 * (x - 99) else 0)

theorem find_three_digit_number : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ 2 * x = total_digits x := by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l1032_103221


namespace NUMINAMATH_GPT_x1_x2_lt_one_l1032_103223

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

theorem x1_x2_lt_one (k : ℝ) (x1 x2 : ℝ) (h : f x1 1 + g x1 - k = 0) (h2 : f x2 1 + g x2 - k = 0) (hx1 : 0 < x1) (hx2 : x1 < x2) : x1 * x2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_x1_x2_lt_one_l1032_103223


namespace NUMINAMATH_GPT_maximum_sum_of_factors_exists_maximum_sum_of_factors_l1032_103224

theorem maximum_sum_of_factors {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 2023) : A + B + C ≤ 297 :=
sorry

theorem exists_maximum_sum_of_factors : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2023 ∧ A + B + C = 297 :=
sorry

end NUMINAMATH_GPT_maximum_sum_of_factors_exists_maximum_sum_of_factors_l1032_103224


namespace NUMINAMATH_GPT_common_chord_of_circles_l1032_103216

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x + 2 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_common_chord_of_circles_l1032_103216


namespace NUMINAMATH_GPT_bookstore_earnings_difference_l1032_103274

def base_price_TOP := 8.0
def base_price_ABC := 23.0
def discount_TOP := 0.10
def discount_ABC := 0.05
def sales_tax := 0.07
def num_TOP_sold := 13
def num_ABC_sold := 4

def discounted_price (base_price discount : Float) : Float :=
  base_price * (1.0 - discount)

def final_price (discounted_price tax : Float) : Float :=
  discounted_price * (1.0 + tax)

def total_earnings (final_price : Float) (quantity : Nat) : Float :=
  final_price * (quantity.toFloat)

theorem bookstore_earnings_difference :
  let discounted_price_TOP := discounted_price base_price_TOP discount_TOP
  let discounted_price_ABC := discounted_price base_price_ABC discount_ABC
  let final_price_TOP := final_price discounted_price_TOP sales_tax
  let final_price_ABC := final_price discounted_price_ABC sales_tax
  let total_earnings_TOP := total_earnings final_price_TOP num_TOP_sold
  let total_earnings_ABC := total_earnings final_price_ABC num_ABC_sold
  total_earnings_TOP - total_earnings_ABC = 6.634 :=
by
  sorry

end NUMINAMATH_GPT_bookstore_earnings_difference_l1032_103274


namespace NUMINAMATH_GPT_count_two_digit_integers_remainder_3_div_9_l1032_103258

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end NUMINAMATH_GPT_count_two_digit_integers_remainder_3_div_9_l1032_103258


namespace NUMINAMATH_GPT_joan_took_marbles_l1032_103292

-- Each condition is used as a definition.
def original_marbles : ℕ := 86
def remaining_marbles : ℕ := 61

-- The theorem states that the number of marbles Joan took equals 25.
theorem joan_took_marbles : (original_marbles - remaining_marbles) = 25 := by
  sorry    -- Add sorry to skip the proof.

end NUMINAMATH_GPT_joan_took_marbles_l1032_103292


namespace NUMINAMATH_GPT_measure_of_angleA_l1032_103299

theorem measure_of_angleA (A B : ℝ) 
  (h1 : ∀ (x : ℝ), x ≠ A → x ≠ B → x ≠ (3 * B - 20) → (3 * x - 20 ≠ A)) 
  (h2 : A = 3 * B - 20) :
  A = 10 ∨ A = 130 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angleA_l1032_103299


namespace NUMINAMATH_GPT_moles_CH3COOH_equiv_l1032_103242

theorem moles_CH3COOH_equiv (moles_NaOH moles_NaCH3COO : ℕ)
    (h1 : moles_NaOH = 1)
    (h2 : moles_NaCH3COO = 1) :
    moles_NaOH = moles_NaCH3COO :=
by
  sorry

end NUMINAMATH_GPT_moles_CH3COOH_equiv_l1032_103242


namespace NUMINAMATH_GPT_mabel_tomatoes_l1032_103293

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end NUMINAMATH_GPT_mabel_tomatoes_l1032_103293


namespace NUMINAMATH_GPT_first_year_after_2023_with_digit_sum_8_l1032_103220

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2023_with_digit_sum_8 : ∃ (y : ℕ), y > 2023 ∧ sum_of_digits y = 8 ∧ ∀ z, (z > 2023 ∧ sum_of_digits z = 8) → y ≤ z :=
by sorry

end NUMINAMATH_GPT_first_year_after_2023_with_digit_sum_8_l1032_103220


namespace NUMINAMATH_GPT_sum_MN_MK_eq_14_sqrt4_3_l1032_103241

theorem sum_MN_MK_eq_14_sqrt4_3
  (MN MK : ℝ)
  (area: ℝ)
  (angle_LMN : ℝ)
  (h_area : area = 49)
  (h_angle_LMN : angle_LMN = 30) :
  MN + MK = 14 * (Real.sqrt (Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_sum_MN_MK_eq_14_sqrt4_3_l1032_103241


namespace NUMINAMATH_GPT_four_racers_meet_l1032_103247

/-- In a circular auto race, four racers participate. Their cars start simultaneously from 
the same point and move at constant speeds, and for any three cars, there is a moment 
when they meet. Prove that after the start of the race, there will be a moment when all 
four cars meet. (Assume the race continues indefinitely in time.) -/
theorem four_racers_meet (V1 V2 V3 V4 : ℝ) (L : ℝ) (t : ℝ) 
  (h1 : 0 ≤ t) 
  (h2 : V1 ≤ V2 ∧ V2 ≤ V3 ∧ V3 ≤ V4)
  (h3 : ∀ t1 t2 t3, ∃ t, t1 * V1 = t ∧ t2 * V2 = t ∧ t3 * V3 = t) :
  ∃ t, t > 0 ∧ ∃ t', V1 * t' % L = 0 ∧ V2 * t' % L = 0 ∧ V3 * t' % L = 0 ∧ V4 * t' % L = 0 :=
sorry

end NUMINAMATH_GPT_four_racers_meet_l1032_103247


namespace NUMINAMATH_GPT_bound_c_n_l1032_103208

theorem bound_c_n (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, a (n + 1) = a n * (a n - 1)) →
  (∀ n, 2^b n = a n) →
  (∀ n, 2^(n - c n) = b n) →
  ∃ (m M : ℝ), (m = 0) ∧ (M = 1) ∧ ∀ n > 0, m ≤ c n ∧ c n ≤ M :=
by
  intro h1 h2 h3 h4
  use 0
  use 1
  sorry

end NUMINAMATH_GPT_bound_c_n_l1032_103208


namespace NUMINAMATH_GPT_probability_consecutive_computer_scientists_l1032_103255

theorem probability_consecutive_computer_scientists :
  let n := 12
  let k := 5
  let total_permutations := Nat.factorial (n - 1)
  let consecutive_permutations := Nat.factorial (7) * Nat.factorial (5)
  let probability := consecutive_permutations / total_permutations
  probability = (1 / 66) :=
by
  sorry

end NUMINAMATH_GPT_probability_consecutive_computer_scientists_l1032_103255


namespace NUMINAMATH_GPT_product_value_l1032_103204

theorem product_value (x : ℝ) (h : (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8)) : (6 + x) * (21 - x) = 1369 / 4 :=
by
  sorry

end NUMINAMATH_GPT_product_value_l1032_103204


namespace NUMINAMATH_GPT_area_PCD_eq_l1032_103280

/-- Define the points P, D, and C as given in the conditions. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 18⟩
def D : Point := ⟨3, 18⟩
def C (q : ℝ) : Point := ⟨0, q⟩

/-- Define the function to compute the area of triangle PCD given q. -/
noncomputable def area_triangle_PCD (q : ℝ) : ℝ :=
  1 / 2 * (D.x - P.x) * (P.y - q)

theorem area_PCD_eq (q : ℝ) : 
  area_triangle_PCD q = 27 - 3 / 2 * q := 
by 
  sorry

end NUMINAMATH_GPT_area_PCD_eq_l1032_103280


namespace NUMINAMATH_GPT_hens_count_l1032_103296

theorem hens_count
  (H C : ℕ)
  (heads_eq : H + C = 48)
  (feet_eq : 2 * H + 4 * C = 136) :
  H = 28 :=
by
  sorry

end NUMINAMATH_GPT_hens_count_l1032_103296


namespace NUMINAMATH_GPT_ball_hits_ground_at_5_over_2_l1032_103248

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 + 40 * t + 60

theorem ball_hits_ground_at_5_over_2 :
  ∃ t : ℝ, t = 5 / 2 ∧ ball_height t = 0 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_at_5_over_2_l1032_103248


namespace NUMINAMATH_GPT_polynomial_remainder_l1032_103288

theorem polynomial_remainder 
  (y: ℤ) 
  (root_cond: y^3 + y^2 + y + 1 = 0) 
  (beta_is_root: ∃ β: ℚ, β^3 + β^2 + β + 1 = 0) 
  (beta_four: ∀ β: ℚ, β^3 + β^2 + β + 1 = 0 → β^4 = 1) : 
  ∃ q r, (y^20 + y^15 + y^10 + y^5 + 1) = q * (y^3 + y^2 + y + 1) + r ∧ (r = 1) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1032_103288


namespace NUMINAMATH_GPT_number_of_outfits_l1032_103219

theorem number_of_outfits (num_shirts : ℕ) (num_pants : ℕ) (num_shoe_types : ℕ) (shoe_styles_per_type : ℕ) (h_shirts : num_shirts = 4) (h_pants : num_pants = 4) (h_shoes : num_shoe_types = 2) (h_styles : shoe_styles_per_type = 2) :
  num_shirts * num_pants * (num_shoe_types * shoe_styles_per_type) = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_outfits_l1032_103219


namespace NUMINAMATH_GPT_new_job_larger_than_original_l1032_103244

theorem new_job_larger_than_original (original_workers original_days new_workers new_days : ℕ) 
  (h_original_workers : original_workers = 250)
  (h_original_days : original_days = 16)
  (h_new_workers : new_workers = 600)
  (h_new_days : new_days = 20) :
  (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry

end NUMINAMATH_GPT_new_job_larger_than_original_l1032_103244


namespace NUMINAMATH_GPT_remainder_when_dividing_l1032_103263

theorem remainder_when_dividing (c d : ℕ) (p q : ℕ) :
  c = 60 * p + 47 ∧ d = 45 * q + 14 → (c + d) % 15 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_l1032_103263


namespace NUMINAMATH_GPT_boat_crossing_l1032_103257

theorem boat_crossing (students teacher trips people_in_boat : ℕ) (h_students : students = 13) (h_teacher : teacher = 1) (h_boat_capacity : people_in_boat = 5) :
  trips = (students + teacher + people_in_boat - 1) / (people_in_boat - 1) :=
by
  sorry

end NUMINAMATH_GPT_boat_crossing_l1032_103257


namespace NUMINAMATH_GPT_base_conversion_sum_correct_l1032_103279

theorem base_conversion_sum_correct :
  (253 / 8 / 13 / 3 + 245 / 7 / 35 / 6 : ℚ) = 339 / 23 := sorry

end NUMINAMATH_GPT_base_conversion_sum_correct_l1032_103279


namespace NUMINAMATH_GPT_distance_from_Q_to_AD_l1032_103265

-- Define the square $ABCD$ with side length 6
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 6) ∧ B = (6, 6) ∧ C = (6, 0) ∧ D = (0, 0)

-- Define point $N$ as the midpoint of $\overline{CD}$
def midpoint_CD (C D N : ℝ × ℝ) : Prop :=
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the intersection condition of the circles centered at $N$ and $A$
def intersect_circles (N A Q D : ℝ × ℝ) : Prop :=
  (Q = D ∨ (∃ r₁ r₂, (Q.1 - N.1)^2 + Q.2^2 = r₁ ∧ Q.1^2 + (Q.2 - A.2)^2 = r₂))

-- Prove the distance from $Q$ to $\overline{AD}$ equals 12/5
theorem distance_from_Q_to_AD (A B C D N Q : ℝ × ℝ)
  (h_square : square_ABCD A B C D)
  (h_midpoint : midpoint_CD C D N)
  (h_intersect : intersect_circles N A Q D) :
  Q.2 = 12 / 5 :=
sorry

end NUMINAMATH_GPT_distance_from_Q_to_AD_l1032_103265


namespace NUMINAMATH_GPT_range_of_a_l1032_103261

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 64 > 0) → -16 < a ∧ a < 16 :=
by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_range_of_a_l1032_103261


namespace NUMINAMATH_GPT_factorable_polynomial_l1032_103259

theorem factorable_polynomial (m : ℤ) :
  (∃ A B C D E F : ℤ, 
    (A * D = 1 ∧ E + B = 4 ∧ C + F = 2 ∧ F + 3 * E + C = m + m^2 - 16)
    ∧ ((A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + 2 * x + m * y + m^2 - 16)) ↔
  (m = 5 ∨ m = -6) :=
by
  sorry

end NUMINAMATH_GPT_factorable_polynomial_l1032_103259


namespace NUMINAMATH_GPT_largest_y_l1032_103295

theorem largest_y (y : ℝ) (h : (⌊y⌋ / y) = 8 / 9) : y ≤ 63 / 8 :=
sorry

end NUMINAMATH_GPT_largest_y_l1032_103295


namespace NUMINAMATH_GPT_valve_rate_difference_l1032_103238

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end NUMINAMATH_GPT_valve_rate_difference_l1032_103238


namespace NUMINAMATH_GPT_marbles_lost_l1032_103283

theorem marbles_lost (initial_marbles lost_marbles gifted_marbles remaining_marbles : ℕ) 
  (h_initial : initial_marbles = 85)
  (h_gifted : gifted_marbles = 25)
  (h_remaining : remaining_marbles = 43)
  (h_before_gifting : remaining_marbles + gifted_marbles = initial_marbles - lost_marbles) :
  lost_marbles = 17 :=
by
  sorry

end NUMINAMATH_GPT_marbles_lost_l1032_103283


namespace NUMINAMATH_GPT_slope_of_line_l1032_103211

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end NUMINAMATH_GPT_slope_of_line_l1032_103211


namespace NUMINAMATH_GPT_additional_toothpicks_needed_l1032_103236

theorem additional_toothpicks_needed 
  (t : ℕ → ℕ)
  (h1 : t 1 = 4)
  (h2 : t 2 = 10)
  (h3 : t 3 = 18)
  (h4 : t 4 = 28)
  (h5 : t 5 = 40)
  (h6 : t 6 = 54) :
  t 6 - t 4 = 26 :=
by
  sorry

end NUMINAMATH_GPT_additional_toothpicks_needed_l1032_103236


namespace NUMINAMATH_GPT_complex_fraction_simplification_l1032_103287

theorem complex_fraction_simplification (a b c d : ℂ) (h₁ : a = 3 + i) (h₂ : b = 1 + i) (h₃ : c = 1 - i) (h₄ : d = 2 - i) : (a / b) = d := by
  sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l1032_103287


namespace NUMINAMATH_GPT_range_of_a_l1032_103249

noncomputable def line_eq (a : ℝ) (x y : ℝ) : ℝ := 3 * x - 2 * y + a 

def pointA : ℝ × ℝ := (3, 1)
def pointB : ℝ × ℝ := (-4, 6)

theorem range_of_a :
  (line_eq a pointA.1 pointA.2) * (line_eq a pointB.1 pointB.2) < 0 ↔ -7 < a ∧ a < 24 := sorry

end NUMINAMATH_GPT_range_of_a_l1032_103249


namespace NUMINAMATH_GPT_john_games_l1032_103212

variables (G_f G_g B G G_t : ℕ)

theorem john_games (h1: G_f = 21) (h2: B = 23) (h3: G = 6) 
(h4: G_t = G_f + G_g) (h5: G + B = G_t) : G_g = 8 :=
by sorry

end NUMINAMATH_GPT_john_games_l1032_103212


namespace NUMINAMATH_GPT_magnitude_product_complex_l1032_103266

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_product_complex_l1032_103266


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1032_103213

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (S_10_eq : S 10 = 20) (S_20_eq : S 20 = 15) :
  S 30 = -15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1032_103213


namespace NUMINAMATH_GPT_count_red_balls_l1032_103278

/-- Given conditions:
  - The total number of balls in the bag is 100.
  - There are 50 white, 20 green, 10 yellow, and 3 purple balls.
  - The probability that a ball will be neither red nor purple is 0.8.
  Prove that the number of red balls is 17. -/
theorem count_red_balls (total_balls white_balls green_balls yellow_balls purple_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls = 50)
  (h3 : green_balls = 20)
  (h4 : yellow_balls = 10)
  (h5 : purple_balls = 3)
  (h6 : (white_balls + green_balls + yellow_balls) = 80)
  (h7 : (white_balls + green_balls + yellow_balls) / (total_balls : ℝ) = 0.8) :
  red_balls = 17 :=
by
  sorry

end NUMINAMATH_GPT_count_red_balls_l1032_103278


namespace NUMINAMATH_GPT_triangle_range_condition_l1032_103209

def triangle_side_range (x : ℝ) : Prop :=
  (1 < x) ∧ (x < 17)

theorem triangle_range_condition (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 8) → (b = 9) → triangle_side_range x :=
by
  intros h1 h2
  dsimp [triangle_side_range]
  sorry

end NUMINAMATH_GPT_triangle_range_condition_l1032_103209


namespace NUMINAMATH_GPT_tangent_line_eq_l1032_103267

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def M : ℝ×ℝ := (2, -3)

theorem tangent_line_eq (x y : ℝ) (h : y = f x) (h' : (x, y) = M) :
  2 * x - y - 7 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1032_103267


namespace NUMINAMATH_GPT_intersection_M_N_eq_l1032_103269

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_l1032_103269


namespace NUMINAMATH_GPT_calculate_train_speed_l1032_103217

def speed_train_excluding_stoppages (distance_per_hour_including_stoppages : ℕ) (stoppage_minutes_per_hour : ℕ) : ℕ :=
  let effective_running_time_per_hour := 60 - stoppage_minutes_per_hour
  let effective_running_time_in_hours := effective_running_time_per_hour / 60
  distance_per_hour_including_stoppages / effective_running_time_in_hours

theorem calculate_train_speed :
  speed_train_excluding_stoppages 42 4 = 45 :=
by
  sorry

end NUMINAMATH_GPT_calculate_train_speed_l1032_103217


namespace NUMINAMATH_GPT_perpendicular_line_equation_l1032_103201

theorem perpendicular_line_equation 
  (p : ℝ × ℝ)
  (L1 : ℝ → ℝ → Prop)
  (L2 : ℝ → ℝ → ℝ → Prop) 
  (hx : p = (1, -1)) 
  (hL1 : ∀ x y, L1 x y ↔ 3 * x - 2 * y = 0) 
  (hL2 : ∀ x y m, L2 x y m ↔ 2 * x + 3 * y + m = 0) :
  ∃ m : ℝ, L2 (p.1) (p.2) m ∧ 2 * p.1 + 3 * p.2 + m = 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_equation_l1032_103201


namespace NUMINAMATH_GPT_simplify_expression_l1032_103225

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1032_103225


namespace NUMINAMATH_GPT_problem_proof_l1032_103260

-- Define the conditions
def a (n : ℕ) : Real := sorry  -- a is some real number, so it's non-deterministic here

def a_squared (n : ℕ) : Real := a n ^ (2 * n)  -- a^(2n)

-- Main theorem to prove
theorem problem_proof (n : ℕ) (h : a_squared n = 3) : 2 * (a n ^ (6 * n)) - 1 = 53 :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_problem_proof_l1032_103260
