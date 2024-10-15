import Mathlib

namespace NUMINAMATH_GPT_find_C_work_rate_l216_21648

-- Conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 6

-- Combined work rate of A and B
def AB_work_rate := A_work_rate + B_work_rate

-- Total work rate when C is assisting, completing in 2 days
def total_work_rate_of_ABC := 1 / 2

theorem find_C_work_rate : ∃ c : ℕ, (AB_work_rate + 1 / c = total_work_rate_of_ABC) ∧ c = 12 :=
by
  -- To complete the proof, we solve the equation for c
  sorry

end NUMINAMATH_GPT_find_C_work_rate_l216_21648


namespace NUMINAMATH_GPT_part1_inequality_part2_range_l216_21666

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

-- Part 1: Prove that f(x) ≥ f(0) for all x
theorem part1_inequality : ∀ x : ℝ, f x ≥ f 0 :=
sorry

-- Part 2: Prove that the range of a satisfying 2f(x) ≥ f(a+1) for all x is -4.5 ≤ a ≤ 1.5
theorem part2_range (a : ℝ) (h : ∀ x : ℝ, 2 * f x ≥ f (a + 1)) : -4.5 ≤ a ∧ a ≤ 1.5 :=
sorry

end NUMINAMATH_GPT_part1_inequality_part2_range_l216_21666


namespace NUMINAMATH_GPT_solution_set_for_x_l216_21605

theorem solution_set_for_x (x : ℝ) (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 :=
sorry

end NUMINAMATH_GPT_solution_set_for_x_l216_21605


namespace NUMINAMATH_GPT_triangle_is_isosceles_l216_21602

theorem triangle_is_isosceles (A B C : ℝ)
  (h : Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) :
  ∃ a b c : ℝ, a = b ∨ b = c ∨ a = c := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l216_21602


namespace NUMINAMATH_GPT_seth_initial_boxes_l216_21662

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_seth_initial_boxes_l216_21662


namespace NUMINAMATH_GPT_arc_length_l216_21657

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 7) : (α * r) = 2 * π / 7 := by
  sorry

end NUMINAMATH_GPT_arc_length_l216_21657


namespace NUMINAMATH_GPT_optimal_solution_for_z_is_1_1_l216_21665

def x := 1
def y := 1
def z (x y : ℝ) := 2 * x + y

theorem optimal_solution_for_z_is_1_1 :
  ∀ (x y : ℝ), z x y ≥ z 1 1 := 
by
  simp [z]
  sorry

end NUMINAMATH_GPT_optimal_solution_for_z_is_1_1_l216_21665


namespace NUMINAMATH_GPT_solve_for_b_l216_21673

theorem solve_for_b (b : ℝ) (hb : b + ⌈b⌉ = 17.8) : b = 8.8 := 
by sorry

end NUMINAMATH_GPT_solve_for_b_l216_21673


namespace NUMINAMATH_GPT_find_higher_selling_price_l216_21641

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end NUMINAMATH_GPT_find_higher_selling_price_l216_21641


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l216_21643

noncomputable def first_21_sum (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  let a1 := a 1
  let a21 := a 21
  21 * (a1 + a21) / 2

theorem arithmetic_sequence_sum
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_symmetry : ∀ x, f (x + 1) = f (-(x + 1)))
  (h_monotonic : ∀ x y, 1 < x → x < y → f x < f y)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_f_eq : f (a 4) = f (a 18))
  (h_non_zero_diff : d ≠ 0) :
  first_21_sum f a d = 21 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l216_21643


namespace NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l216_21655

-- Case 1
theorem quadratic_function_expression 
  (a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = 3) : 
  by {exact (a = -2 ∧ b = 3)} := sorry

theorem quadratic_function_range 
  (x : ℝ) 
  (h : -1 ≤ x ∧ x ≤ 2) : 
  (-3 ≤ -2*x^2 + 3*x + 2 ∧ -2*x^2 + 3*x + 2 ≤ 25/8) := sorry

-- Case 2
theorem quadratic_function_m_range 
  (m a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = m) 
  (h₃ : a > 0) : 
  m < 1 := sorry

end NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l216_21655


namespace NUMINAMATH_GPT_geometric_series_sum_l216_21639

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l216_21639


namespace NUMINAMATH_GPT_power_of_three_l216_21615

theorem power_of_three (a b : ℕ) (h1 : 360 = (2^3) * (3^2) * (5^1))
  (h2 : 2^a ∣ 360 ∧ ∀ n, 2^n ∣ 360 → n ≤ a)
  (h3 : 5^b ∣ 360 ∧ ∀ n, 5^n ∣ 360 → n ≤ b) :
  (1/3 : ℝ)^(b - a) = 9 :=
by sorry

end NUMINAMATH_GPT_power_of_three_l216_21615


namespace NUMINAMATH_GPT_g_s_difference_l216_21695

def g (n : ℤ) : ℤ := n^3 + 3 * n^2 + 3 * n + 1

theorem g_s_difference (s : ℤ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end NUMINAMATH_GPT_g_s_difference_l216_21695


namespace NUMINAMATH_GPT_find_a7_a8_l216_21679

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (g : geometric_sequence a q)

def sum_1_2 : ℝ := a 1 + a 2
def sum_3_4 : ℝ := a 3 + a 4

theorem find_a7_a8
  (h1 : sum_1_2 = 30)
  (h2 : sum_3_4 = 60)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 7 + a 8 = (a 1 + a 2) * (q ^ 6) := 
sorry

end NUMINAMATH_GPT_find_a7_a8_l216_21679


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l216_21671

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  {x | 1 / x ≤ 1} ⊆ {x | Real.log x ≥ 0} ∧ 
  ¬ ({x | Real.log x ≥ 0} ⊆ {x | 1 / x ≤ 1}) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l216_21671


namespace NUMINAMATH_GPT_find_k_l216_21608

-- Definitions based on the problem conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Property of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u.1 = c * v.1 ∧ u.2 = c * v.2

-- Theorem statement equivalent to the problem
theorem find_k (k : ℝ) (h : parallel vector_a (vector_b k)) : k = -2 :=
sorry

end NUMINAMATH_GPT_find_k_l216_21608


namespace NUMINAMATH_GPT_average_value_f_l216_21640

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end NUMINAMATH_GPT_average_value_f_l216_21640


namespace NUMINAMATH_GPT_garden_perimeter_l216_21685

theorem garden_perimeter (width_garden length_playground width_playground : ℕ) 
  (h1 : width_garden = 12) 
  (h2 : length_playground = 16) 
  (h3 : width_playground = 12) 
  (area_playground : ℕ)
  (h4 : area_playground = length_playground * width_playground) 
  (area_garden : ℕ) 
  (h5 : area_garden = area_playground) 
  (length_garden : ℕ) 
  (h6 : area_garden = length_garden * width_garden) :
  2 * length_garden + 2 * width_garden = 56 := by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l216_21685


namespace NUMINAMATH_GPT_mass_of_compound_l216_21678

-- Constants as per the conditions
def molecular_weight : ℕ := 444           -- The molecular weight in g/mol.
def number_of_moles : ℕ := 6             -- The number of moles.

-- Defining the main theorem we want to prove.
theorem mass_of_compound : (number_of_moles * molecular_weight) = 2664 := by 
  sorry

end NUMINAMATH_GPT_mass_of_compound_l216_21678


namespace NUMINAMATH_GPT_father_age_is_30_l216_21610

theorem father_age_is_30 {M F : ℝ} 
  (h1 : M = (2 / 5) * F) 
  (h2 : M + 6 = (1 / 2) * (F + 6)) :
  F = 30 :=
sorry

end NUMINAMATH_GPT_father_age_is_30_l216_21610


namespace NUMINAMATH_GPT_total_kilometers_ridden_l216_21606

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_total_kilometers_ridden_l216_21606


namespace NUMINAMATH_GPT_probability_not_snowing_l216_21691

variable (P_snowing : ℚ)
variable (h : P_snowing = 2/5)

theorem probability_not_snowing (P_not_snowing : ℚ) : 
  P_not_snowing = 3 / 5 :=
by 
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_probability_not_snowing_l216_21691


namespace NUMINAMATH_GPT_perimeter_of_ABCD_l216_21622

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_perimeter_of_ABCD_l216_21622


namespace NUMINAMATH_GPT_tate_total_years_proof_l216_21604

def highSchoolYears: ℕ := 4 - 1
def gapYear: ℕ := 2
def bachelorYears (highSchoolYears: ℕ): ℕ := 2 * highSchoolYears
def workExperience: ℕ := 1
def phdYears (highSchoolYears: ℕ) (bachelorYears: ℕ): ℕ := 3 * (highSchoolYears + bachelorYears)
def totalYears (highSchoolYears: ℕ) (gapYear: ℕ) (bachelorYears: ℕ) (workExperience: ℕ) (phdYears: ℕ): ℕ :=
  highSchoolYears + gapYear + bachelorYears + workExperience + phdYears

theorem tate_total_years_proof : totalYears highSchoolYears gapYear (bachelorYears highSchoolYears) workExperience (phdYears highSchoolYears (bachelorYears highSchoolYears)) = 39 := by
  sorry

end NUMINAMATH_GPT_tate_total_years_proof_l216_21604


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l216_21693

open Set

def A : Set ℝ := { x | 3 * x + 2 > 0 }
def B : Set ℝ := { x | (x + 1) * (x - 3) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | x > 3 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l216_21693


namespace NUMINAMATH_GPT_value_of_a_l216_21613

theorem value_of_a :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - -1)^2 + (y - 1)^2 = 4) := sorry

end NUMINAMATH_GPT_value_of_a_l216_21613


namespace NUMINAMATH_GPT_incorrect_statement_l216_21629

-- Define the relationship between the length of the spring and the mass of the object
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

-- Formalize statements A, B, C, and D
def statementA : Prop := spring_length 0 = 10

def statementB : Prop :=
  ¬ ∃ (length : ℝ) (mass : ℝ), (spring_length mass = length ∧ mass = (length - 10) / 2.5)

def statementC : Prop :=
  ∀ m : ℝ, spring_length (m + 1) = spring_length m + 2.5

def statementD : Prop := spring_length 4 = 20

-- The Lean statement to prove that statement B is incorrect
theorem incorrect_statement (hA : statementA) (hC : statementC) (hD : statementD) : ¬ statementB := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l216_21629


namespace NUMINAMATH_GPT_ratio_A_to_B_l216_21632

/--
Proof problem statement:
Given that A and B together can finish the work in 4 days,
and B alone can finish the work in 24 days,
prove that the ratio of the time A takes to finish the work to the time B takes to finish the work is 1:5.
-/
theorem ratio_A_to_B
  (A_time B_time working_together_time : ℝ) 
  (h1 : working_together_time = 4)
  (h2 : B_time = 24)
  (h3 : 1 / A_time + 1 / B_time = 1 / working_together_time) :
  A_time / B_time = 1 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_A_to_B_l216_21632


namespace NUMINAMATH_GPT_merchant_discount_percentage_l216_21670

theorem merchant_discount_percentage
  (CP MP SP : ℝ)
  (h1 : MP = CP + 0.40 * CP)
  (h2 : SP = CP + 0.26 * CP)
  : ((MP - SP) / MP) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_merchant_discount_percentage_l216_21670


namespace NUMINAMATH_GPT_largest_house_number_l216_21646

theorem largest_house_number (house_num : ℕ) : 
  house_num ≤ 981 :=
  sorry

end NUMINAMATH_GPT_largest_house_number_l216_21646


namespace NUMINAMATH_GPT_sum_of_coefficients_l216_21650

noncomputable def problem_expr (d : ℝ) := (16 * d + 15 + 18 * d^2 + 3 * d^3) + (4 * d + 2 + d^2 + 2 * d^3)
noncomputable def simplified_expr (d : ℝ) := 5 * d^3 + 19 * d^2 + 20 * d + 17

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : 
  problem_expr d = simplified_expr d ∧ (5 + 19 + 20 + 17 = 61) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l216_21650


namespace NUMINAMATH_GPT_ratio_PR_QS_l216_21611

/-- Given points P, Q, R, and S on a straight line in that order with
    distances PQ = 3 units, QR = 7 units, and PS = 20 units,
    the ratio of PR to QS is 1. -/
theorem ratio_PR_QS (P Q R S : ℝ) (PQ QR PS : ℝ) (hPQ : PQ = 3) (hQR : QR = 7) (hPS : PS = 20) :
  let PR := PQ + QR
  let QS := PS - PQ - QR
  PR / QS = 1 :=
by
  -- Definitions from conditions
  let PR := PQ + QR
  let QS := PS - PQ - QR
  -- Proof not required, hence sorry
  sorry

end NUMINAMATH_GPT_ratio_PR_QS_l216_21611


namespace NUMINAMATH_GPT_solve_equation_l216_21633

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8) ∨ (x = 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l216_21633


namespace NUMINAMATH_GPT_pencils_left_l216_21609

def initial_pencils : Nat := 127
def pencils_from_joyce : Nat := 14
def pencils_per_friend : Nat := 7

theorem pencils_left : ((initial_pencils + pencils_from_joyce) % pencils_per_friend) = 1 := by
  sorry

end NUMINAMATH_GPT_pencils_left_l216_21609


namespace NUMINAMATH_GPT_possible_marks_l216_21617

theorem possible_marks (n : ℕ) : n = 3 ∨ n = 6 ↔
  ∃ (m : ℕ), n = (m * (m - 1)) / 2 ∧ (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ∃ (i j : ℕ), i < j ∧ j - i = k ∧ (∀ (x y : ℕ), x < y → x ≠ i ∨ y ≠ j)) :=
by sorry

end NUMINAMATH_GPT_possible_marks_l216_21617


namespace NUMINAMATH_GPT_granddaughter_fraction_l216_21668

noncomputable def betty_age : ℕ := 60
def fraction_younger (p : ℕ) : ℕ := (p * 40) / 100
noncomputable def daughter_age : ℕ := betty_age - fraction_younger betty_age
def granddaughter_age : ℕ := 12
def fraction (a b : ℕ) : ℚ := a / b

theorem granddaughter_fraction :
  fraction granddaughter_age daughter_age = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_granddaughter_fraction_l216_21668


namespace NUMINAMATH_GPT_inequality_proof_l216_21674

variables {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l216_21674


namespace NUMINAMATH_GPT_intersection_M_N_l216_21626

open Set

def M : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def N : Set ℝ := { x | x >= 1 }

theorem intersection_M_N : M ∩ N = { x | 1 <= x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l216_21626


namespace NUMINAMATH_GPT_average_of_remaining_three_numbers_l216_21645

noncomputable def avg_remaining_three_numbers (avg_12 : ℝ) (avg_4 : ℝ) (avg_3 : ℝ) (avg_2 : ℝ) : ℝ :=
  let sum_12 := 12 * avg_12
  let sum_4 := 4 * avg_4
  let sum_3 := 3 * avg_3
  let sum_2 := 2 * avg_2
  let sum_9 := sum_4 + sum_3 + sum_2
  let sum_remaining_3 := sum_12 - sum_9
  sum_remaining_3 / 3

theorem average_of_remaining_three_numbers :
  avg_remaining_three_numbers 6.30 5.60 4.90 7.25 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_of_remaining_three_numbers_l216_21645


namespace NUMINAMATH_GPT_percentage_increase_bears_with_assistant_l216_21680

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end NUMINAMATH_GPT_percentage_increase_bears_with_assistant_l216_21680


namespace NUMINAMATH_GPT_set_intersection_example_l216_21689

theorem set_intersection_example :
  let A := { y | ∃ x, y = Real.log x / Real.log 2 ∧ x ≥ 3 }
  let B := { x | x^2 - 4 * x + 3 = 0 }
  A ∩ B = {3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_example_l216_21689


namespace NUMINAMATH_GPT_scientific_notation_470000000_l216_21620

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_470000000_l216_21620


namespace NUMINAMATH_GPT_sum_remainder_l216_21663

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 5) (h3 : c % 30 = 18) : 
  (a + b + c) % 30 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_l216_21663


namespace NUMINAMATH_GPT_fraction_sum_equals_zero_l216_21616

theorem fraction_sum_equals_zero :
  (1 / 12) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_equals_zero_l216_21616


namespace NUMINAMATH_GPT_calc_result_l216_21684

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end NUMINAMATH_GPT_calc_result_l216_21684


namespace NUMINAMATH_GPT_functional_equation_solution_l216_21675

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2) →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l216_21675


namespace NUMINAMATH_GPT_avg_hamburgers_per_day_l216_21659

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 49) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_avg_hamburgers_per_day_l216_21659


namespace NUMINAMATH_GPT_solve_y_l216_21664

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end NUMINAMATH_GPT_solve_y_l216_21664


namespace NUMINAMATH_GPT_part1_part2_l216_21644

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | abs (x - a) <= 2}

def symdiff (A B : Set ℝ) : Set ℝ := A ∩ (U \ B)

theorem part1 :
  symdiff M N = {x | 1 < x ∧ x < 2} ∧
  symdiff N M = {x | 3 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem part2 (a : ℝ) :
  symdiff (symdiff N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then {x | a + 2 < x ∧ x < 2}
    else ∅ :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l216_21644


namespace NUMINAMATH_GPT_johnny_marble_choice_l216_21627

/-- Johnny has 9 different colored marbles and always chooses 1 specific red marble.
    Prove that the number of ways to choose four marbles from his bag is 56. -/
theorem johnny_marble_choice : (Nat.choose 8 3) = 56 := 
by
  sorry

end NUMINAMATH_GPT_johnny_marble_choice_l216_21627


namespace NUMINAMATH_GPT_average_rainfall_per_hour_in_June_1882_l216_21677

open Real

theorem average_rainfall_per_hour_in_June_1882 
  (total_rainfall : ℝ) (days_in_June : ℕ) (hours_per_day : ℕ)
  (H1 : total_rainfall = 450) (H2 : days_in_June = 30) (H3 : hours_per_day = 24) :
  total_rainfall / (days_in_June * hours_per_day) = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_per_hour_in_June_1882_l216_21677


namespace NUMINAMATH_GPT_cost_of_swim_trunks_is_14_l216_21697

noncomputable def cost_of_swim_trunks : Real :=
  let flat_rate_shipping := 5.00
  let shipping_rate := 0.20
  let price_shirt := 12.00
  let price_socks := 5.00
  let price_shorts := 15.00
  let cost_known_items := 3 * price_shirt + price_socks + 2 * price_shorts
  let total_bill := 102.00
  let x := (total_bill - 0.20 * cost_known_items - cost_known_items) / 1.20
  x

theorem cost_of_swim_trunks_is_14 : cost_of_swim_trunks = 14 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_cost_of_swim_trunks_is_14_l216_21697


namespace NUMINAMATH_GPT_frequency_of_second_group_l216_21676

theorem frequency_of_second_group (total_capacity : ℕ) (freq_percentage : ℝ)
    (h_capacity : total_capacity = 80)
    (h_percentage : freq_percentage = 0.15) :
    total_capacity * freq_percentage = 12 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_second_group_l216_21676


namespace NUMINAMATH_GPT_each_nap_duration_l216_21686

-- Definitions based on the problem conditions
def BillProjectDurationInDays : ℕ := 4
def HoursPerDay : ℕ := 24
def TotalProjectHours : ℕ := BillProjectDurationInDays * HoursPerDay
def WorkHours : ℕ := 54
def NapsTaken : ℕ := 6

-- Calculate the time spent on naps and the duration of each nap
def NapHoursTotal : ℕ := TotalProjectHours - WorkHours
def DurationEachNap : ℕ := NapHoursTotal / NapsTaken

-- The theorem stating the expected answer
theorem each_nap_duration :
  DurationEachNap = 7 := by
  sorry

end NUMINAMATH_GPT_each_nap_duration_l216_21686


namespace NUMINAMATH_GPT_part1_part2_l216_21690

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l216_21690


namespace NUMINAMATH_GPT_coefficients_sum_l216_21642

theorem coefficients_sum (a0 a1 a2 a3 a4 : ℝ) (h : (1 - 2*x)^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) : 
  a0 + a4 = 17 :=
by
  sorry

end NUMINAMATH_GPT_coefficients_sum_l216_21642


namespace NUMINAMATH_GPT_concentration_of_concentrated_kola_is_correct_l216_21630

noncomputable def concentration_of_concentrated_kola_added 
  (initial_volume : ℝ) (initial_pct_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ)
  (required_pct_sugar : ℝ) (new_sugar_volume : ℝ) : ℝ :=
  let initial_sugar := initial_volume * initial_pct_sugar / 100
  let total_sugar := initial_sugar + sugar_added
  let new_total_volume := initial_volume + sugar_added + water_added
  let total_volume_with_kola := new_total_volume + (new_sugar_volume / required_pct_sugar * 100 - total_sugar) / (100 / required_pct_sugar - 1)
  total_volume_with_kola - new_total_volume

noncomputable def problem_kola : ℝ :=
  concentration_of_concentrated_kola_added 340 7 3.2 10 7.5 27

theorem concentration_of_concentrated_kola_is_correct : 
  problem_kola = 6.8 :=
by
  unfold problem_kola concentration_of_concentrated_kola_added
  sorry

end NUMINAMATH_GPT_concentration_of_concentrated_kola_is_correct_l216_21630


namespace NUMINAMATH_GPT_blue_to_red_marble_ratio_l216_21618

-- Define the given conditions and the result.
theorem blue_to_red_marble_ratio (total_marble yellow_marble : ℕ) 
  (h1 : total_marble = 19)
  (h2 : yellow_marble = 5)
  (red_marble : ℕ)
  (h3 : red_marble = yellow_marble + 3) : 
  ∃ blue_marble : ℕ, (blue_marble = total_marble - (yellow_marble + red_marble)) 
  ∧ (blue_marble / (gcd blue_marble red_marble)) = 3 
  ∧ (red_marble / (gcd blue_marble red_marble)) = 4 :=
by {
  --existence of blue_marble and the ratio
  sorry
}

end NUMINAMATH_GPT_blue_to_red_marble_ratio_l216_21618


namespace NUMINAMATH_GPT_find_x_l216_21631

noncomputable def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

theorem find_x :
  ∃ x : ℝ, x + Real.sqrt 68 = 24 ∧ approx_equal x 15.753788749 0.0001 :=
sorry

end NUMINAMATH_GPT_find_x_l216_21631


namespace NUMINAMATH_GPT_daliah_garbage_l216_21638

theorem daliah_garbage (D : ℝ) (h1 : 4 * (D - 2) = 62) : D = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_daliah_garbage_l216_21638


namespace NUMINAMATH_GPT_boxwoods_shaped_into_spheres_l216_21696

theorem boxwoods_shaped_into_spheres :
  ∀ (total_boxwoods : ℕ) (cost_trimming : ℕ) (cost_shaping : ℕ) (total_charge : ℕ) (x : ℕ),
    total_boxwoods = 30 →
    cost_trimming = 5 →
    cost_shaping = 15 →
    total_charge = 210 →
    30 * 5 + x * 15 = 210 →
    x = 4 :=
by
  intros total_boxwoods cost_trimming cost_shaping total_charge x
  rintro rfl rfl rfl rfl h
  sorry

end NUMINAMATH_GPT_boxwoods_shaped_into_spheres_l216_21696


namespace NUMINAMATH_GPT_simplify_expression_l216_21614

theorem simplify_expression (x y : ℝ) (h : x = -3) : 
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6 * x + 9) + 5 * x^3 * y^2 / (x^2 * y^2) = -66 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l216_21614


namespace NUMINAMATH_GPT_fraction_operation_correct_l216_21636

theorem fraction_operation_correct (a b : ℝ) (h : 0.2 * a + 0.5 * b ≠ 0) : 
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
sorry

end NUMINAMATH_GPT_fraction_operation_correct_l216_21636


namespace NUMINAMATH_GPT_fraction_problem_l216_21621

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l216_21621


namespace NUMINAMATH_GPT_grandpa_uncle_ratio_l216_21651

def initial_collection := 150
def dad_gift := 10
def mum_gift := dad_gift + 5
def auntie_gift := 6
def uncle_gift := auntie_gift - 1
def final_collection := 196
def total_cars_needed := final_collection - initial_collection
def other_gifts := dad_gift + mum_gift + auntie_gift + uncle_gift
def grandpa_gift := total_cars_needed - other_gifts

theorem grandpa_uncle_ratio : grandpa_gift = 2 * uncle_gift := by
  sorry

end NUMINAMATH_GPT_grandpa_uncle_ratio_l216_21651


namespace NUMINAMATH_GPT_sin_double_angle_l216_21637

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2*θ) = 24/25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l216_21637


namespace NUMINAMATH_GPT_gamma_max_two_day_success_ratio_l216_21653

theorem gamma_max_two_day_success_ratio :
  ∃ (e g f h : ℕ), 0 < e ∧ 0 < g ∧
  e + g = 335 ∧ 
  e < f ∧ g < h ∧ 
  f + h = 600 ∧ 
  (e : ℚ) / f < (180 : ℚ) / 360 ∧ 
  (g : ℚ) / h < (150 : ℚ) / 240 ∧ 
  (e + g) / 600 = 67 / 120 :=
by
  sorry

end NUMINAMATH_GPT_gamma_max_two_day_success_ratio_l216_21653


namespace NUMINAMATH_GPT_lucy_withdrawal_l216_21688

-- Given conditions
def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

-- Define balance before withdrawal
def balance_before_withdrawal := initial_balance + deposit

-- Theorem to prove
theorem lucy_withdrawal : balance_before_withdrawal - final_balance = 4 :=
by sorry

end NUMINAMATH_GPT_lucy_withdrawal_l216_21688


namespace NUMINAMATH_GPT_fields_fertilized_in_25_days_l216_21681

-- Definitions from conditions
def fertilizer_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def fertilizer_needed_per_acre : ℕ := 400
def number_of_acres : ℕ := 20
def acres_fertilized_per_day : ℕ := 4

-- Total fertilizer produced per day
def total_fertilizer_per_day : ℕ := fertilizer_per_horse_per_day * number_of_horses

-- Total fertilizer needed
def total_fertilizer_needed : ℕ := fertilizer_needed_per_acre * number_of_acres

-- Days to collect enough fertilizer
def days_to_collect_fertilizer : ℕ := total_fertilizer_needed / total_fertilizer_per_day

-- Days to spread fertilizer
def days_to_spread_fertilizer : ℕ := number_of_acres / acres_fertilized_per_day

-- Calculate the total time until all fields are fertilized
def total_days : ℕ := days_to_collect_fertilizer + days_to_spread_fertilizer

-- Theorem statement
theorem fields_fertilized_in_25_days : total_days = 25 :=
by
  sorry

end NUMINAMATH_GPT_fields_fertilized_in_25_days_l216_21681


namespace NUMINAMATH_GPT_average_after_17th_inning_l216_21699

theorem average_after_17th_inning (A : ℝ) (total_runs_16th_inning : ℝ) 
  (average_before_17th : A * 16 = total_runs_16th_inning) 
  (increased_average_by_3 : (total_runs_16th_inning + 83) / 17 = A + 3) :
  (A + 3) = 35 := 
sorry

end NUMINAMATH_GPT_average_after_17th_inning_l216_21699


namespace NUMINAMATH_GPT_factorization_of_expression_l216_21660

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l216_21660


namespace NUMINAMATH_GPT_original_population_l216_21661

variable (n : ℝ)

theorem original_population
  (h1 : n + 1500 - 0.15 * (n + 1500) = n - 45) :
  n = 8800 :=
sorry

end NUMINAMATH_GPT_original_population_l216_21661


namespace NUMINAMATH_GPT_max_product_production_l216_21652

theorem max_product_production (C_mats A_mats C_ship A_ship B_mats B_ship : ℝ)
  (cost_A cost_B ship_A ship_B : ℝ) (prod_A prod_B max_cost_mats max_cost_ship prod_max : ℝ)
  (h_prod_A : prod_A = 90)
  (h_cost_A : cost_A = 1000)
  (h_ship_A : ship_A = 500)
  (h_prod_B : prod_B = 100)
  (h_cost_B : cost_B = 1500)
  (h_ship_B : ship_B = 400)
  (h_max_cost_mats : max_cost_mats = 6000)
  (h_max_cost_ship : max_cost_ship = 2000)
  (h_prod_max : prod_max = 440)
  (H_C_mats : C_mats = cost_A * A_mats + cost_B * B_mats)
  (H_C_ship : C_ship = ship_A * A_ship + ship_B * B_ship)
  (H_A_mats_ship : A_mats = A_ship)
  (H_B_mats_ship : B_mats = B_ship)
  (H_C_mats_le : C_mats ≤ max_cost_mats)
  (H_C_ship_le : C_ship ≤ max_cost_ship) :
  prod_A * A_mats + prod_B * B_mats ≤ prod_max :=
by {
  sorry
}

end NUMINAMATH_GPT_max_product_production_l216_21652


namespace NUMINAMATH_GPT_tan_seven_pi_over_four_l216_21628

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end NUMINAMATH_GPT_tan_seven_pi_over_four_l216_21628


namespace NUMINAMATH_GPT_math_problem_l216_21601

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem math_problem
  (omega phi : ℝ)
  (h1 : omega > 0)
  (h2 : |phi| < Real.pi / 2)
  (h3 : ∀ x, f x = Real.sin (omega * x + phi))
  (h4 : ∀ k : ℤ, f (k * Real.pi) = f 0) 
  (h5 : f 0 = 1 / 2) :
  (omega = 2) ∧
  (∀ x, f (x + Real.pi / 6) = f (-x + Real.pi / 6)) ∧
  (∀ k : ℤ, 
    ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    x < y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l216_21601


namespace NUMINAMATH_GPT_fraction_checked_by_worker_y_l216_21647

-- Definitions of conditions given in the problem
variable (P Px Py : ℝ)
variable (h1 : Px + Py = P)
variable (h2 : 0.005 * Px = defective_x)
variable (h3 : 0.008 * Py = defective_y)
variable (defective_x defective_y : ℝ)
variable (total_defective : ℝ)
variable (h4 : defective_x + defective_y = total_defective)
variable (h5 : total_defective = 0.0065 * P)

-- The fraction of products checked by worker y
theorem fraction_checked_by_worker_y (h : Px + Py = P) (h2 : 0.005 * Px = 0.0065 * P) (h3 : 0.008 * Py = 0.0065 * P) :
  Py / P = 1 / 2 := 
  sorry

end NUMINAMATH_GPT_fraction_checked_by_worker_y_l216_21647


namespace NUMINAMATH_GPT_Marley_fruit_count_l216_21658

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Marley_fruit_count_l216_21658


namespace NUMINAMATH_GPT_vacant_seats_l216_21654

theorem vacant_seats (total_seats filled_percentage : ℕ) (h_filled_percentage : filled_percentage = 62) (h_total_seats : total_seats = 600) : 
  (total_seats - total_seats * filled_percentage / 100) = 228 :=
by
  sorry

end NUMINAMATH_GPT_vacant_seats_l216_21654


namespace NUMINAMATH_GPT_sqrt_360000_eq_600_l216_21600

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := 
sorry

end NUMINAMATH_GPT_sqrt_360000_eq_600_l216_21600


namespace NUMINAMATH_GPT_range_of_first_person_l216_21667

variable (R1 R2 R3 : ℕ)
variable (min_range : ℕ)
variable (condition1 : min_range = 25)
variable (condition2 : R2 = 25)
variable (condition3 : R3 = 30)
variable (condition4 : min_range ≤ R1 ∧ min_range ≤ R2 ∧ min_range ≤ R3)

theorem range_of_first_person : R1 = 25 :=
by
  sorry

end NUMINAMATH_GPT_range_of_first_person_l216_21667


namespace NUMINAMATH_GPT_probability_coin_covers_black_region_l216_21683

open Real

noncomputable def coin_cover_black_region_probability : ℝ :=
  let side_length_square := 10
  let triangle_leg := 3
  let diamond_side_length := 3 * sqrt 2
  let smaller_square_side := 1
  let coin_diameter := 1
  -- The derived probability calculation
  (32 + 9 * sqrt 2 + π) / 81

theorem probability_coin_covers_black_region :
  coin_cover_black_region_probability = (32 + 9 * sqrt 2 + π) / 81 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_coin_covers_black_region_l216_21683


namespace NUMINAMATH_GPT_no_divisibility_condition_by_all_others_l216_21625

theorem no_divisibility_condition_by_all_others 
  {p : ℕ → ℕ} 
  (h_distinct_odd_primes : ∀ i j, i ≠ j → Nat.Prime (p i) ∧ Nat.Prime (p j) ∧ p i ≠ p j ∧ p i % 2 = 1 ∧ p j % 2 = 1)
  (h_ordered : ∀ i j, i < j → p i < p j) :
  ¬ ∀ i j, i ≠ j → (∀ k ≠ i, k ≠ j → p k ∣ (p i ^ 8 - p j ^ 8)) :=
by
  sorry

end NUMINAMATH_GPT_no_divisibility_condition_by_all_others_l216_21625


namespace NUMINAMATH_GPT_arithmetic_progressions_count_l216_21687

theorem arithmetic_progressions_count (d : ℕ) (h_d : d = 2) (S : ℕ) (h_S : S = 200) : 
  ∃ n : ℕ, n = 6 := sorry

end NUMINAMATH_GPT_arithmetic_progressions_count_l216_21687


namespace NUMINAMATH_GPT_incorrect_statement_D_l216_21624

/-
Define the conditions for the problem:
- A prism intersected by a plane.
- The intersection of a sphere and a plane when the plane is less than the radius.
- The intersection of a plane parallel to the base of a circular cone.
- The geometric solid formed by rotating a right triangle around one of its sides.
- The incorrectness of statement D.
-/

noncomputable def intersect_prism_with_plane (prism : Type) (plane : Type) : Prop := sorry

noncomputable def sphere_intersection (sphere_radius : ℝ) (distance_to_plane : ℝ) : Type := sorry

noncomputable def cone_intersection (cone : Type) (plane : Type) : Type := sorry

noncomputable def rotation_result (triangle : Type) (side : Type) : Type := sorry

theorem incorrect_statement_D :
  ¬(rotation_result RightTriangle Side = Cone) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_D_l216_21624


namespace NUMINAMATH_GPT_jose_is_21_l216_21619

-- Define the ages of the individuals based on the conditions
def age_of_inez := 12
def age_of_zack := age_of_inez + 4
def age_of_jose := age_of_zack + 5

-- State the proposition we want to prove
theorem jose_is_21 : age_of_jose = 21 := 
by 
  sorry

end NUMINAMATH_GPT_jose_is_21_l216_21619


namespace NUMINAMATH_GPT_relatively_prime_m_n_l216_21672

noncomputable def probability_of_distinct_real_solutions : ℝ :=
  let b := (1 : ℝ)
  if 1 ≤ b ∧ b ≤ 25 then 1 else 0

theorem relatively_prime_m_n : ∃ m n : ℕ, 
  Nat.gcd m n = 1 ∧ 
  (1 : ℝ) = (m : ℝ) / (n : ℝ) ∧ m + n = 2 := 
by
  sorry

end NUMINAMATH_GPT_relatively_prime_m_n_l216_21672


namespace NUMINAMATH_GPT_trapezium_area_l216_21635

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 10) : 
  (1 / 2) * (a + b) * h = 190 := 
by
  -- We provide the conditions:
  rw [h_a, h_b, h_h]
  -- The proof steps will be skipped using 'sorry'
  sorry

end NUMINAMATH_GPT_trapezium_area_l216_21635


namespace NUMINAMATH_GPT_increasing_on_1_to_infty_min_value_on_1_to_e_l216_21634

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := (2 * x^2 - a) / x

-- Proof that f(x) is increasing on (1, +∞) when a = 2
theorem increasing_on_1_to_infty (x : ℝ) (h : x > 1) : f' x 2 > 0 := 
  sorry

-- Proof for minimum value of f(x) on [1, e]
theorem min_value_on_1_to_e (a : ℝ) :
  if a ≤ 2 then ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = 1
  else if 2 < a ∧ a < 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = a / 2 - (a / 2) * Real.log (a / 2)
  else if a ≥ 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = Real.exp 2 - a
  else False := 
  sorry

end NUMINAMATH_GPT_increasing_on_1_to_infty_min_value_on_1_to_e_l216_21634


namespace NUMINAMATH_GPT_find_x_l216_21682

theorem find_x (x : ℝ) (h : (3 * x) / 4 = 24) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l216_21682


namespace NUMINAMATH_GPT_maximize_ab_l216_21656

theorem maximize_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ab + a + b = 1) : 
  ab ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximize_ab_l216_21656


namespace NUMINAMATH_GPT_min_value_l216_21603

theorem min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
(h5 : m + 4 * n = 1) : 
  1 / m + 4 / n ≥ 25 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l216_21603


namespace NUMINAMATH_GPT_next_term_in_geom_sequence_l216_21698

   /- Define the given geometric sequence as a function in Lean -/

   def geom_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ n

   theorem next_term_in_geom_sequence (x : ℤ) (n : ℕ) 
     (h₁ : geom_sequence 3 (-3*x) 0 = 3)
     (h₂ : geom_sequence 3 (-3*x) 1 = -9*x)
     (h₃ : geom_sequence 3 (-3*x) 2 = 27*(x^2))
     (h₄ : geom_sequence 3 (-3*x) 3 = -81*(x^3)) :
     geom_sequence 3 (-3*x) 4 = 243*(x^4) := 
   sorry
   
end NUMINAMATH_GPT_next_term_in_geom_sequence_l216_21698


namespace NUMINAMATH_GPT_red_peaches_l216_21607

theorem red_peaches (R G : ℕ) (h1 : G = 11) (h2 : G = R + 6) : R = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_peaches_l216_21607


namespace NUMINAMATH_GPT_division_result_l216_21692

def n : ℕ := 16^1024

theorem division_result : n / 8 = 2^4093 :=
by sorry

end NUMINAMATH_GPT_division_result_l216_21692


namespace NUMINAMATH_GPT_rohan_food_percentage_l216_21694

noncomputable def rohan_salary : ℝ := 7500
noncomputable def rohan_savings : ℝ := 1500
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def entertainment_percentage : ℝ := 0.10
noncomputable def conveyance_percentage : ℝ := 0.10
noncomputable def total_spent : ℝ := rohan_salary - rohan_savings
noncomputable def known_percentage : ℝ := house_rent_percentage + entertainment_percentage + conveyance_percentage

theorem rohan_food_percentage (F : ℝ) :
  total_spent = rohan_salary * (1 - known_percentage - F) →
  F = 0.20 :=
sorry

end NUMINAMATH_GPT_rohan_food_percentage_l216_21694


namespace NUMINAMATH_GPT_polygon_interior_angles_540_implies_pentagon_l216_21612

theorem polygon_interior_angles_540_implies_pentagon
  (n : ℕ) (H: 180 * (n - 2) = 540) : n = 5 :=
sorry

end NUMINAMATH_GPT_polygon_interior_angles_540_implies_pentagon_l216_21612


namespace NUMINAMATH_GPT_mulch_price_per_pound_l216_21669

noncomputable def price_per_pound (total_cost : ℝ) (total_tons : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  total_cost / (total_tons * pounds_per_ton)

theorem mulch_price_per_pound :
  price_per_pound 15000 3 2000 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_mulch_price_per_pound_l216_21669


namespace NUMINAMATH_GPT_grace_pennies_l216_21623

theorem grace_pennies :
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  dimes * dime_value + coins * coin_value = 150 :=
by
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  sorry

end NUMINAMATH_GPT_grace_pennies_l216_21623


namespace NUMINAMATH_GPT_marbles_shared_equally_l216_21649

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end NUMINAMATH_GPT_marbles_shared_equally_l216_21649
