import Mathlib

namespace NUMINAMATH_GPT_average_weight_of_class_l1571_157104

theorem average_weight_of_class (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ)
  (h_students_a : students_a = 24)
  (h_students_b : students_b = 16)
  (h_avg_weight_a : avg_weight_a = 40)
  (h_avg_weight_b : avg_weight_b = 35) :
  ((students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b)) = 38 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_class_l1571_157104


namespace NUMINAMATH_GPT_calculation_l1571_157157

def operation_e (x y z : ℕ) : ℕ := 3 * x * y * z

theorem calculation :
  operation_e 3 (operation_e 4 5 6) 1 = 3240 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l1571_157157


namespace NUMINAMATH_GPT_initial_water_percentage_l1571_157134

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end NUMINAMATH_GPT_initial_water_percentage_l1571_157134


namespace NUMINAMATH_GPT_range_of_a_l1571_157122

theorem range_of_a (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1571_157122


namespace NUMINAMATH_GPT_sequence_inequality_l1571_157159

theorem sequence_inequality (a : ℕ → ℤ) (h₀ : a 1 > a 0) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n+1) = 3 * a n - 2 * a (n-1)) : 
  a 100 > 2^99 := 
sorry

end NUMINAMATH_GPT_sequence_inequality_l1571_157159


namespace NUMINAMATH_GPT_no_integer_roots_of_quadratic_l1571_157135

theorem no_integer_roots_of_quadratic (n : ℤ) : 
  ¬ ∃ (x : ℤ), x^2 - 16 * n * x + 7^5 = 0 := by
  sorry

end NUMINAMATH_GPT_no_integer_roots_of_quadratic_l1571_157135


namespace NUMINAMATH_GPT_bobs_share_l1571_157171

theorem bobs_share 
  (r : ℕ → ℕ → ℕ → Prop) (s : ℕ) 
  (h_ratio : r 1 2 3) 
  (bill_share : s = 300) 
  (hr : ∃ p, s = 2 * p) :
  ∃ b, b = 3 * (s / 2) ∧ b = 450 := 
by
  sorry

end NUMINAMATH_GPT_bobs_share_l1571_157171


namespace NUMINAMATH_GPT_original_six_digit_number_l1571_157111

theorem original_six_digit_number :
  ∃ a b c d e : ℕ, 
  (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e = 142857) ∧ 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 1 = 64 * (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e)) :=
by
  sorry

end NUMINAMATH_GPT_original_six_digit_number_l1571_157111


namespace NUMINAMATH_GPT_minimum_rows_required_l1571_157179

theorem minimum_rows_required (n : ℕ) : (3 * n * (n + 1)) / 2 ≥ 150 ↔ n ≥ 10 := 
by
  sorry

end NUMINAMATH_GPT_minimum_rows_required_l1571_157179


namespace NUMINAMATH_GPT_henry_classical_cds_l1571_157161

variable (R C : ℕ)

theorem henry_classical_cds :
  (23 - 3 = R) →
  (R = 2 * C) →
  C = 10 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_henry_classical_cds_l1571_157161


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1571_157182

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | 1 ≤ x ∧ x < 4}

-- The theorem stating the problem
theorem intersection_of_A_and_B : A ∩ B = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1571_157182


namespace NUMINAMATH_GPT_find_positive_n_l1571_157120

def arithmetic_sequence (a d : ℤ) (n : ℤ) := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_positive_n :
  ∃ (n : ℕ), n > 0 ∧ ∀ a d : ℤ, a = -12 → sum_of_first_n_terms a d 13 = 0 → arithmetic_sequence a d n > 0 ∧ n = 8 := 
sorry

end NUMINAMATH_GPT_find_positive_n_l1571_157120


namespace NUMINAMATH_GPT_player_match_count_l1571_157131

open Real

theorem player_match_count (n : ℕ) : 
  (∃ T, T = 32 * n ∧ (T + 98) / (n + 1) = 38) → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_player_match_count_l1571_157131


namespace NUMINAMATH_GPT_total_hotdogs_sold_l1571_157178

theorem total_hotdogs_sold : 
  let small := 58.3
  let medium := 21.7
  let large := 35.9
  let extra_large := 15.4
  small + medium + large + extra_large = 131.3 :=
by 
  sorry

end NUMINAMATH_GPT_total_hotdogs_sold_l1571_157178


namespace NUMINAMATH_GPT_find_radius_l1571_157132

theorem find_radius (r : ℝ) :
  (135 * r * Real.pi) / 180 = 3 * Real.pi → r = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l1571_157132


namespace NUMINAMATH_GPT_value_of_h_otimes_h_otimes_h_l1571_157125

variable (h x y : ℝ)

-- Define the new operation
def otimes (x y : ℝ) := x^3 - x * y + y^2

-- Prove that h ⊗ (h ⊗ h) = h^6 - h^4 + h^3
theorem value_of_h_otimes_h_otimes_h :
  otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end NUMINAMATH_GPT_value_of_h_otimes_h_otimes_h_l1571_157125


namespace NUMINAMATH_GPT_parts_repetition_cycle_l1571_157113

noncomputable def parts_repetition_condition (t : ℕ) : Prop := sorry
def parts_initial_condition : Prop := sorry

theorem parts_repetition_cycle :
  parts_initial_condition →
  parts_repetition_condition 2 ∧
  parts_repetition_condition 4 ∧
  parts_repetition_condition 38 ∧
  parts_repetition_condition 76 :=
sorry


end NUMINAMATH_GPT_parts_repetition_cycle_l1571_157113


namespace NUMINAMATH_GPT_find_solution_l1571_157147

theorem find_solution (x : ℝ) (h : (5 + x / 3)^(1/3) = -4) : x = -207 :=
sorry

end NUMINAMATH_GPT_find_solution_l1571_157147


namespace NUMINAMATH_GPT_Keiko_speed_l1571_157199

theorem Keiko_speed (a b s : ℝ) (h1 : 8 = 8) 
  (h2 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : 
  s = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_Keiko_speed_l1571_157199


namespace NUMINAMATH_GPT_find_ratio_l1571_157139

-- Define the geometric sequence properties and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions stated in the problem
axiom h₁ : a 5 * a 11 = 3
axiom h₂ : a 3 + a 13 = 4

-- The goal is to find the values of a_15 / a_5
theorem find_ratio (h₁ : a 5 * a 11 = 3) (h₂ : a 3 + a 13 = 4) :
  ∃ r : ℝ, r = a 15 / a 5 ∧ (r = 3 ∨ r = 1 / 3) :=
sorry

end NUMINAMATH_GPT_find_ratio_l1571_157139


namespace NUMINAMATH_GPT_wire_cut_problem_l1571_157143

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end NUMINAMATH_GPT_wire_cut_problem_l1571_157143


namespace NUMINAMATH_GPT_proposition_D_is_true_l1571_157195

-- Define the propositions
def proposition_A : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def proposition_B : Prop := ∀ x : ℝ, 2^x > x^2
def proposition_C : Prop := ∀ a b : ℝ, (a + b = 0 ↔ a / b = -1)
def proposition_D : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1

-- Problem statement: Proposition D is true
theorem proposition_D_is_true : proposition_D := 
by sorry

end NUMINAMATH_GPT_proposition_D_is_true_l1571_157195


namespace NUMINAMATH_GPT_power_of_power_l1571_157172

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end NUMINAMATH_GPT_power_of_power_l1571_157172


namespace NUMINAMATH_GPT_ordered_triples_eq_l1571_157168

theorem ordered_triples_eq :
  ∃! (x y z : ℤ), x + y = 4 ∧ xy - z^2 = 3 ∧ (x = 2 ∧ y = 2 ∧ z = 0) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ordered_triples_eq_l1571_157168


namespace NUMINAMATH_GPT_deluxe_stereo_time_fraction_l1571_157158

theorem deluxe_stereo_time_fraction (S : ℕ) (B : ℝ)
  (H1 : 2 / 3 > 0)
  (H2 : 1.6 > 0) :
  (1.6 / 3 * S * B) / (1.2 * S * B) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_deluxe_stereo_time_fraction_l1571_157158


namespace NUMINAMATH_GPT_find_a_b_l1571_157101

def z := Complex.ofReal 3 + Complex.I * 4
def z_conj := Complex.ofReal 3 - Complex.I * 4

theorem find_a_b 
  (a b : ℝ) 
  (h : z + Complex.ofReal a * z_conj + Complex.I * b = Complex.ofReal 9) : 
  a = 2 ∧ b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_b_l1571_157101


namespace NUMINAMATH_GPT_eggs_per_day_second_store_l1571_157196

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozen eggs supplied to the first store each day
def dozen_per_day_first_store : ℕ := 5

-- Define the number of eggs supplied to the first store each day
def eggs_per_day_first_store : ℕ := dozen_per_day_first_store * eggs_in_a_dozen

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Calculate the weekly supply to the first store
def weekly_supply_first_store : ℕ := eggs_per_day_first_store * days_in_week

-- Define the total weekly supply to both stores
def total_weekly_supply : ℕ := 630

-- Calculate the weekly supply to the second store
def weekly_supply_second_store : ℕ := total_weekly_supply - weekly_supply_first_store

-- Define the theorem to prove the number of eggs supplied to the second store each day
theorem eggs_per_day_second_store : weekly_supply_second_store / days_in_week = 30 := by
  sorry

end NUMINAMATH_GPT_eggs_per_day_second_store_l1571_157196


namespace NUMINAMATH_GPT_probability_XiaoYu_group_A_l1571_157141

theorem probability_XiaoYu_group_A :
  ∀ (students : Fin 48) (groups : Fin 4) (groupAssignment : Fin 48 → Fin 4)
    (student : Fin 48) (groupA : Fin 4),
    (∀ (s : Fin 48), ∃ (g : Fin 4), groupAssignment s = g) → 
    (∀ (g : Fin 4), ∃ (count : ℕ), (0 < count ∧ count ≤ 12) ∧
       (∃ (groupMembers : List (Fin 48)), groupMembers.length = count ∧
        (∀ (m : Fin 48), m ∈ groupMembers → groupAssignment m = g))) →
    (groupAssignment student = groupA) →
  ∃ (p : ℚ), p = (1/4) ∧ ∀ (s : Fin 48), groupAssignment s = groupA → p = (1/4) :=
by
  sorry

end NUMINAMATH_GPT_probability_XiaoYu_group_A_l1571_157141


namespace NUMINAMATH_GPT_log_sqrt2_bounds_l1571_157130

theorem log_sqrt2_bounds :
  10^3 = 1000 →
  10^4 = 10000 →
  2^11 = 2048 →
  2^12 = 4096 →
  2^13 = 8192 →
  2^14 = 16384 →
  3 / 22 < Real.log 2 / Real.log 10 / 2 ∧ Real.log 2 / Real.log 10 / 2 < 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_log_sqrt2_bounds_l1571_157130


namespace NUMINAMATH_GPT_find_a_b_sum_l1571_157118

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_b_sum (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 24) : a + b = 6 :=
  sorry

end NUMINAMATH_GPT_find_a_b_sum_l1571_157118


namespace NUMINAMATH_GPT_interest_earned_l1571_157173

-- Define the principal, interest rate, and number of years
def principal : ℝ := 1200
def annualInterestRate : ℝ := 0.12
def numberOfYears : ℕ := 4

-- Define the compound interest formula
def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the total interest earned
def totalInterest (P A : ℝ) : ℝ :=
  A - P

-- State the theorem
theorem interest_earned :
  totalInterest principal (compoundInterest principal annualInterestRate numberOfYears) = 688.224 :=
by
  sorry

end NUMINAMATH_GPT_interest_earned_l1571_157173


namespace NUMINAMATH_GPT_isosceles_trapezoid_ratio_l1571_157170

theorem isosceles_trapezoid_ratio (a b h : ℝ) 
  (h1: h = b / 2)
  (h2: a = 1 - ((1 - b) / 2))
  (h3 : 1 = ((a + 1) / 2)^2 + (b / 2)^2) :
  b / a = (-1 + Real.sqrt 7) / 2 := 
sorry

end NUMINAMATH_GPT_isosceles_trapezoid_ratio_l1571_157170


namespace NUMINAMATH_GPT_combined_score_210_l1571_157189

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end NUMINAMATH_GPT_combined_score_210_l1571_157189


namespace NUMINAMATH_GPT_quadratic_coefficients_l1571_157164

theorem quadratic_coefficients (x : ℝ) : 
  let a := 3
  let b := -5
  let c := 1
  3 * x^2 + 1 = 5 * x → a * x^2 + b * x + c = 0 := by
sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1571_157164


namespace NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1571_157176

theorem least_possible_value_of_smallest_integer 
  (A B C D : ℤ) 
  (H_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (H_avg : (A + B + C + D) / 4 = 74)
  (H_max : D = 90) :
  A ≥ 31 :=
by sorry

end NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1571_157176


namespace NUMINAMATH_GPT_nine_chapters_problem_l1571_157127

variable (m n : ℕ)

def horses_condition_1 : Prop := m + n = 100
def horses_condition_2 : Prop := 3 * m + n / 3 = 100

theorem nine_chapters_problem (h1 : horses_condition_1 m n) (h2 : horses_condition_2 m n) :
  (m + n = 100 ∧ 3 * m + n / 3 = 100) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_nine_chapters_problem_l1571_157127


namespace NUMINAMATH_GPT_paint_amount_third_day_l1571_157186

theorem paint_amount_third_day : 
  let initial_paint := 80
  let first_day_usage := initial_paint / 2
  let paint_after_first_day := initial_paint - first_day_usage
  let added_paint := 20
  let new_total_paint := paint_after_first_day + added_paint
  let second_day_usage := new_total_paint / 2
  let paint_after_second_day := new_total_paint - second_day_usage
  paint_after_second_day = 30 :=
by
  sorry

end NUMINAMATH_GPT_paint_amount_third_day_l1571_157186


namespace NUMINAMATH_GPT_number_of_parallelograms_l1571_157133

-- Problem's condition
def side_length (n : ℕ) : Prop := n > 0

-- Required binomial coefficient (combination formula)
def binom (n k : ℕ) : ℕ := n.choose k

-- Total number of parallelograms in the tiling
theorem number_of_parallelograms (n : ℕ) (h : side_length n) : 
  3 * binom (n + 2) 4 = 3 * (n+2).choose 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_parallelograms_l1571_157133


namespace NUMINAMATH_GPT_min_value_x_plus_y_l1571_157149

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l1571_157149


namespace NUMINAMATH_GPT_probability_of_no_shaded_square_l1571_157105

noncomputable def rectangles_without_shaded_square_probability : ℚ :=
  let n := 502 * 1003
  let m := 502 ^ 2
  1 - (m : ℚ) / n 

theorem probability_of_no_shaded_square : rectangles_without_shaded_square_probability = 501 / 1003 :=
  sorry

end NUMINAMATH_GPT_probability_of_no_shaded_square_l1571_157105


namespace NUMINAMATH_GPT_mass_percentage_Al_in_AlBr₃_l1571_157183

theorem mass_percentage_Al_in_AlBr₃ :
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  (Al_mass / M_AlBr₃ * 100) = 10.11 :=
by 
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  have : (Al_mass / M_AlBr₃ * 100) = 10.11 := sorry
  assumption

end NUMINAMATH_GPT_mass_percentage_Al_in_AlBr₃_l1571_157183


namespace NUMINAMATH_GPT_exponent_evaluation_problem_l1571_157140

theorem exponent_evaluation_problem (m : ℕ) : 
  (m^2 * m^3 ≠ m^6) → 
  (m^2 + m^4 ≠ m^6) → 
  ((m^3)^3 ≠ m^6) → 
  (m^7 / m = m^6) :=
by
  intros hA hB hC
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_exponent_evaluation_problem_l1571_157140


namespace NUMINAMATH_GPT_sum_of_angles_l1571_157177

variables (A B C D E F : ℝ)

theorem sum_of_angles 
  (h : E = 30) :
  A + B + C + D + E + F = 420 :=
sorry

end NUMINAMATH_GPT_sum_of_angles_l1571_157177


namespace NUMINAMATH_GPT_acute_angle_probability_l1571_157154

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_probability_l1571_157154


namespace NUMINAMATH_GPT_emails_received_afternoon_is_one_l1571_157191

-- Define the number of emails received by Jack in the morning
def emails_received_morning : ℕ := 4

-- Define the total number of emails received by Jack in a day
def total_emails_received : ℕ := 5

-- Define the number of emails received by Jack in the afternoon
def emails_received_afternoon : ℕ := total_emails_received - emails_received_morning

-- Prove the number of emails received by Jack in the afternoon
theorem emails_received_afternoon_is_one : emails_received_afternoon = 1 :=
by 
  -- Proof is neglected as per instructions.
  sorry

end NUMINAMATH_GPT_emails_received_afternoon_is_one_l1571_157191


namespace NUMINAMATH_GPT_find_a_l1571_157115

def star (a b : ℕ) : ℕ := 3 * a - b ^ 2

theorem find_a (a : ℕ) (b : ℕ) (h : star a b = 14) : a = 10 :=
by sorry

end NUMINAMATH_GPT_find_a_l1571_157115


namespace NUMINAMATH_GPT_largest_divisor_l1571_157184

theorem largest_divisor (x : ℤ) (hx : x % 2 = 1) : 180 ∣ (15 * x + 3) * (15 * x + 9) * (10 * x + 5) := 
by
  sorry

end NUMINAMATH_GPT_largest_divisor_l1571_157184


namespace NUMINAMATH_GPT_rent_cost_l1571_157108

-- Definitions based on conditions
def daily_supplies_cost : ℕ := 12
def price_per_pancake : ℕ := 2
def pancakes_sold_per_day : ℕ := 21

-- Proving the daily rent cost
theorem rent_cost (total_sales : ℕ) (rent : ℕ) :
  total_sales = pancakes_sold_per_day * price_per_pancake →
  rent = total_sales - daily_supplies_cost →
  rent = 30 :=
by
  intro h_total_sales h_rent
  sorry

end NUMINAMATH_GPT_rent_cost_l1571_157108


namespace NUMINAMATH_GPT_distance_between_centers_of_intersecting_circles_l1571_157106

theorem distance_between_centers_of_intersecting_circles
  {r R d : ℝ} (hrR : r < R) (hr : 0 < r) (hR : 0 < R)
  (h_intersect : d < r + R ∧ d > R - r) :
  R - r < d ∧ d < r + R := by
  sorry

end NUMINAMATH_GPT_distance_between_centers_of_intersecting_circles_l1571_157106


namespace NUMINAMATH_GPT_part1_general_formula_part2_sum_S_l1571_157174

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => a n + 1

theorem part1_general_formula (n : ℕ) : a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (↑n * ↑(n + 2))

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

theorem part2_sum_S (n : ℕ) : 
  S n = (1/2) * ((3/2) - (1 / (n + 1)) - (1 / (n + 2))) := by
  sorry

end NUMINAMATH_GPT_part1_general_formula_part2_sum_S_l1571_157174


namespace NUMINAMATH_GPT_abs_inequality_solution_l1571_157124

theorem abs_inequality_solution (x : ℝ) :
  |x + 2| + |x - 2| ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1571_157124


namespace NUMINAMATH_GPT_percentage_problem_l1571_157146

theorem percentage_problem (X : ℝ) (h : 0.28 * X + 0.45 * 250 = 224.5) : X = 400 :=
sorry

end NUMINAMATH_GPT_percentage_problem_l1571_157146


namespace NUMINAMATH_GPT_find_q_l1571_157151

theorem find_q (a b m p q : ℚ) (h1 : a * b = 3) (h2 : a + b = m) 
  (h3 : (a + 1/b) * (b + 1/a) = q) : 
  q = 13 / 3 := by
  sorry

end NUMINAMATH_GPT_find_q_l1571_157151


namespace NUMINAMATH_GPT_length_of_bridge_l1571_157156

theorem length_of_bridge
  (T : ℕ) (t : ℕ) (s : ℕ)
  (hT : T = 250)
  (ht : t = 20)
  (hs : s = 20) :
  ∃ L : ℕ, L = 150 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1571_157156


namespace NUMINAMATH_GPT_range_of_f_l1571_157126

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2)

theorem range_of_f (x : ℝ) : -1 < f x ∧ f x < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1571_157126


namespace NUMINAMATH_GPT_twice_joan_more_than_karl_l1571_157163

-- Define the conditions
def J : ℕ := 158
def total : ℕ := 400
def K : ℕ := total - J

-- Define the theorem to be proven
theorem twice_joan_more_than_karl :
  2 * J - K = 74 := by
    -- Skip the proof steps using 'sorry'
    sorry

end NUMINAMATH_GPT_twice_joan_more_than_karl_l1571_157163


namespace NUMINAMATH_GPT_functional_eq_is_odd_function_l1571_157110

theorem functional_eq_is_odd_function (f : ℝ → ℝ)
  (hf_nonzero : ∃ x : ℝ, f x ≠ 0)
  (hf_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end NUMINAMATH_GPT_functional_eq_is_odd_function_l1571_157110


namespace NUMINAMATH_GPT_vova_gave_pavlik_three_nuts_l1571_157165

variable {V P k : ℕ}
variable (h1 : V > P)
variable (h2 : V - P = 2 * P)
variable (h3 : k ≤ 5)
variable (h4 : ∃ m : ℕ, V - k = 3 * m)

theorem vova_gave_pavlik_three_nuts (h1 : V > P) (h2 : V - P = 2 * P) (h3 : k ≤ 5) (h4 : ∃ m : ℕ, V - k = 3 * m) : k = 3 := by
  sorry

end NUMINAMATH_GPT_vova_gave_pavlik_three_nuts_l1571_157165


namespace NUMINAMATH_GPT_GCF_30_90_75_l1571_157162

theorem GCF_30_90_75 : Nat.gcd (Nat.gcd 30 90) 75 = 15 := by
  sorry

end NUMINAMATH_GPT_GCF_30_90_75_l1571_157162


namespace NUMINAMATH_GPT_range_of_a_l1571_157148

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 4 * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1571_157148


namespace NUMINAMATH_GPT_time_period_is_12_hours_l1571_157190

-- Define the conditions in the problem
def birth_rate := 8 / 2 -- people per second
def death_rate := 6 / 2 -- people per second
def net_increase := 86400 -- people

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Total time period in seconds
def time_period_seconds := net_increase / net_increase_per_second

-- Convert the time period to hours
def time_period_hours := time_period_seconds / 3600

-- The theorem we want to state and prove
theorem time_period_is_12_hours : time_period_hours = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_time_period_is_12_hours_l1571_157190


namespace NUMINAMATH_GPT_geometric_then_sum_geometric_l1571_157150

variable {a b c d : ℝ}

def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def forms_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

theorem geometric_then_sum_geometric (h : geometric_sequence a b c d) :
  forms_geometric_sequence (a + b) (b + c) (c + d) :=
sorry

end NUMINAMATH_GPT_geometric_then_sum_geometric_l1571_157150


namespace NUMINAMATH_GPT_find_f_value_l1571_157166

noncomputable def f (x y z : ℝ) : ℝ := 2 * x^3 * Real.sin y + Real.log (z^2)

theorem find_f_value :
  f 1 (Real.pi / 2) (Real.exp 2) = 8 →
  f 2 Real.pi (Real.exp 3) = 6 :=
by
  intro h
  unfold f
  sorry

end NUMINAMATH_GPT_find_f_value_l1571_157166


namespace NUMINAMATH_GPT_domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l1571_157114

open Real

noncomputable def f (x : ℝ) : ℝ := log (9 - x^2)

theorem domain_of_f : Set.Ioo (-3 : ℝ) 3 = {x : ℝ | -3 < x ∧ x < 3} :=
by
  sorry

theorem range_of_f : ∃ y : ℝ, y ∈ Set.Iic (2 * log 3) :=
by
  sorry

theorem monotonic_increasing_interval_of_f : 
  {x : ℝ | -3 < x} ∩ {x : ℝ | 0 ≥ x} = Set.Ioc (-3 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l1571_157114


namespace NUMINAMATH_GPT_find_S2017_l1571_157128

-- Setting up the given conditions and sequences
def a1 : ℤ := -2014
def S (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * 2 -- Using the provided sum formula

theorem find_S2017
  (h1 : a1 = -2014)
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) :
  S 2017 = 4034 := 
sorry

end NUMINAMATH_GPT_find_S2017_l1571_157128


namespace NUMINAMATH_GPT_inverse_proportion_graph_l1571_157103

theorem inverse_proportion_graph (m n : ℝ) (h : n = -2 / m) : m = -2 / n :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_graph_l1571_157103


namespace NUMINAMATH_GPT_distance_on_dirt_section_distance_on_mud_section_l1571_157160

noncomputable def v_highway : ℝ := 120 -- km/h
noncomputable def v_dirt : ℝ := 40 -- km/h
noncomputable def v_mud : ℝ := 10 -- km/h
noncomputable def initial_distance : ℝ := 0.6 -- km

theorem distance_on_dirt_section : 
  ∃ s_1 : ℝ, 
  (s_1 = 0.2 * 1000 ∧ -- converting km to meters
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

theorem distance_on_mud_section : 
  ∃ s_2 : ℝ, 
  (s_2 = 50 ∧
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

end NUMINAMATH_GPT_distance_on_dirt_section_distance_on_mud_section_l1571_157160


namespace NUMINAMATH_GPT_scientific_notation_100000_l1571_157153

theorem scientific_notation_100000 : ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (100000 = a * 10 ^ n) :=
by
  use 1, 5
  repeat { split }
  repeat { sorry }

end NUMINAMATH_GPT_scientific_notation_100000_l1571_157153


namespace NUMINAMATH_GPT_sum_of_possible_n_values_l1571_157180

theorem sum_of_possible_n_values (m n : ℕ) 
  (h : 0 < m ∧ 0 < n)
  (eq1 : 1/m + 1/n = 1/5) : 
  n = 6 ∨ n = 10 ∨ n = 30 → 
  m = 30 ∨ m = 10 ∨ m = 6 ∨ m = 5 ∨ m = 25 ∨ m = 1 →
  (6 + 10 + 30 = 46) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_possible_n_values_l1571_157180


namespace NUMINAMATH_GPT_bruce_can_buy_11_bags_l1571_157123

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end NUMINAMATH_GPT_bruce_can_buy_11_bags_l1571_157123


namespace NUMINAMATH_GPT_correct_statements_about_C_l1571_157137

-- Conditions: Curve C is defined by the equation x^4 + y^2 = 1
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Prove the properties of curve C
theorem correct_statements_about_C :
  (-- 1. Symmetric about the x-axis
    (∀ x y : ℝ, C x y → C x (-y)) ∧
    -- 2. Symmetric about the y-axis
    (∀ x y : ℝ, C x y → C (-x) y) ∧
    -- 3. Symmetric about the origin
    (∀ x y : ℝ, C x y → C (-x) (-y)) ∧
    -- 6. A closed figure with an area greater than π
    (∃ (area : ℝ), area > π)) := sorry

end NUMINAMATH_GPT_correct_statements_about_C_l1571_157137


namespace NUMINAMATH_GPT_contrapositive_example_l1571_157155

theorem contrapositive_example (a b : ℕ) (h : a = 0 → ab = 0) : ab ≠ 0 → a ≠ 0 :=
by sorry

end NUMINAMATH_GPT_contrapositive_example_l1571_157155


namespace NUMINAMATH_GPT_total_wheels_eq_90_l1571_157117

def total_wheels (num_bicycles : Nat) (wheels_per_bicycle : Nat) (num_tricycles : Nat) (wheels_per_tricycle : Nat) :=
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_90 : total_wheels 24 2 14 3 = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_eq_90_l1571_157117


namespace NUMINAMATH_GPT_inequality_ge_one_l1571_157194

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end NUMINAMATH_GPT_inequality_ge_one_l1571_157194


namespace NUMINAMATH_GPT_problem_statement_l1571_157187

theorem problem_statement (a b c m : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0) (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1571_157187


namespace NUMINAMATH_GPT_parking_monthly_charge_l1571_157145

theorem parking_monthly_charge :
  ∀ (M : ℕ), (52 * 10 - 12 * M = 100) → M = 35 :=
by
  intro M h
  sorry

end NUMINAMATH_GPT_parking_monthly_charge_l1571_157145


namespace NUMINAMATH_GPT_Jeanine_has_more_pencils_than_Clare_l1571_157116

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end NUMINAMATH_GPT_Jeanine_has_more_pencils_than_Clare_l1571_157116


namespace NUMINAMATH_GPT_range_a_sub_b_mul_c_l1571_157142

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end NUMINAMATH_GPT_range_a_sub_b_mul_c_l1571_157142


namespace NUMINAMATH_GPT_part1_part2_l1571_157198

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (x : ℝ) : f x 1 >= f x 1 := sorry

theorem part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) : b / a ≥ 0 := sorry

end NUMINAMATH_GPT_part1_part2_l1571_157198


namespace NUMINAMATH_GPT_length_of_FD_l1571_157138

-- Define the conditions
def is_square (ABCD : ℝ) (side_length : ℝ) : Prop :=
  side_length = 8 ∧ ABCD = 4 * side_length

def point_E (x : ℝ) : Prop :=
  x = 8 / 3

def point_F (CD : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 8

-- State the theorem
theorem length_of_FD (side_length : ℝ) (x : ℝ) (CD ED FD : ℝ) :
  is_square 4 side_length → 
  point_E ED → 
  point_F CD x → 
  FD = 20 / 9 :=
by
  sorry

end NUMINAMATH_GPT_length_of_FD_l1571_157138


namespace NUMINAMATH_GPT_divisor_of_1058_l1571_157175

theorem divisor_of_1058 :
  ∃ (d : ℕ), (∃ (k : ℕ), 1058 = d * k) ∧ (¬ ∃ (d : ℕ), (∃ (l : ℕ), 1 < d ∧ d < 1058 ∧ 1058 = d * l)) :=
by {
  sorry
}

end NUMINAMATH_GPT_divisor_of_1058_l1571_157175


namespace NUMINAMATH_GPT_points_same_color_separed_by_two_l1571_157136

theorem points_same_color_separed_by_two (circle : Fin 239 → Bool) : 
  ∃ i j : Fin 239, i ≠ j ∧ (i + 2) % 239 = j ∧ circle i = circle j :=
by
  sorry

end NUMINAMATH_GPT_points_same_color_separed_by_two_l1571_157136


namespace NUMINAMATH_GPT_minimum_value_of_function_l1571_157167

theorem minimum_value_of_function :
  ∃ x y : ℝ, 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x + y = 3.7391 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1571_157167


namespace NUMINAMATH_GPT_range_of_a_zeros_of_g_l1571_157192

-- Definitions for the original functions f and g and their corresponding conditions
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (x x2 a : ℝ) : ℝ := f x a - (x2 / 2)

-- Proving the range of a
theorem range_of_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  0 < a ∧ a < 1 := 
sorry

-- Proving the number of zeros of g based on the value of a
theorem zeros_of_g (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  (0 < a ∧ a < 3 / Real.exp 2 → ∃ x3 x4, x3 ≠ x4 ∧ g x3 x2 a = 0 ∧ g x4 x2 a = 0) ∧
  (a = 3 / Real.exp 2 → ∃ x3, g x3 x2 a = 0) ∧
  (3 / Real.exp 2 < a ∧ a < 1 → ∀ x, g x x2 a ≠ 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_zeros_of_g_l1571_157192


namespace NUMINAMATH_GPT_trigonometric_identities_l1571_157185

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identities_l1571_157185


namespace NUMINAMATH_GPT_problem1_problem2_l1571_157144

-- Problem 1
theorem problem1 (x: ℚ) (h: x + 1 / 4 = 7 / 4) : x = 3 / 2 :=
by sorry

-- Problem 2
theorem problem2 (x: ℚ) (h: 2 / 3 + x = 3 / 4) : x = 1 / 12 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1571_157144


namespace NUMINAMATH_GPT_distance_from_A_to_y_axis_is_2_l1571_157129

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_A_to_y_axis_is_2_l1571_157129


namespace NUMINAMATH_GPT_part1_l1571_157119

theorem part1 (f : ℝ → ℝ) (m n : ℝ) (cond1 : m + n > 0) (cond2 : ∀ x, f x = |x - m| + |x + n|) (cond3 : ∀ x, f x ≥ m + n) (minimum : ∃ x, f x = 2) :
    m + n = 2 := sorry

end NUMINAMATH_GPT_part1_l1571_157119


namespace NUMINAMATH_GPT_probability_of_odd_divisor_l1571_157169

noncomputable def prime_factorization_15! : ℕ :=
  (2 ^ 11) * (3 ^ 6) * (5 ^ 3) * (7 ^ 2) * 11 * 13

def total_factors_15! : ℕ :=
  (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def odd_factors_15! : ℕ :=
  (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def probability_odd_divisor_15! : ℚ :=
  odd_factors_15! / total_factors_15!

theorem probability_of_odd_divisor : probability_odd_divisor_15! = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_divisor_l1571_157169


namespace NUMINAMATH_GPT_grunters_win_all_five_l1571_157193

theorem grunters_win_all_five (p : ℚ) (games : ℕ) (win_prob : ℚ) :
  games = 5 ∧ win_prob = 3 / 5 → 
  p = (win_prob) ^ games ∧ p = 243 / 3125 := 
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_grunters_win_all_five_l1571_157193


namespace NUMINAMATH_GPT_tims_total_earnings_l1571_157181

theorem tims_total_earnings (days_of_week : ℕ) (tasks_per_day : ℕ) (tasks_40_rate : ℕ) (tasks_30_rate1 : ℕ) (tasks_30_rate2 : ℕ)
    (rate_40 : ℝ) (rate_30_1 : ℝ) (rate_30_2 : ℝ) (bonus_per_50 : ℝ) (performance_bonus : ℝ)
    (total_earnings : ℝ) :
  days_of_week = 6 →
  tasks_per_day = 100 →
  tasks_40_rate = 40 →
  tasks_30_rate1 = 30 →
  tasks_30_rate2 = 30 →
  rate_40 = 1.2 →
  rate_30_1 = 1.5 →
  rate_30_2 = 2.0 →
  bonus_per_50 = 10 →
  performance_bonus = 20 →
  total_earnings = 1058 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tims_total_earnings_l1571_157181


namespace NUMINAMATH_GPT_condition_1_condition_2_l1571_157109

theorem condition_1 (m : ℝ) : (m^2 - 2*m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

theorem condition_2 (m : ℝ) : (2*m^2 + 3*m - 9 = 0) ∧ (7*m + 21 ≠ 0) ↔ (m = 3/2) :=
sorry

end NUMINAMATH_GPT_condition_1_condition_2_l1571_157109


namespace NUMINAMATH_GPT_trig_expression_value_l1571_157197

theorem trig_expression_value (θ : ℝ) (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_trig_expression_value_l1571_157197


namespace NUMINAMATH_GPT_exists_integers_for_linear_combination_l1571_157188

theorem exists_integers_for_linear_combination 
  (a b c d b1 b2 : ℤ)
  (h1 : ad - bc ≠ 0)
  (h2 : ∃ k : ℤ, b1 = (ad - bc) * k)
  (h3 : ∃ q : ℤ, b2 = (ad - bc) * q) :
  ∃ x y : ℤ, a * x + b * y = b1 ∧ c * x + d * y = b2 :=
sorry

end NUMINAMATH_GPT_exists_integers_for_linear_combination_l1571_157188


namespace NUMINAMATH_GPT_measure_angle_F_l1571_157152

theorem measure_angle_F :
  ∃ (F : ℝ), F = 18 ∧
  ∃ (D E : ℝ),
  D = 75 ∧
  E = 15 + 4 * F ∧
  D + E + F = 180 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_F_l1571_157152


namespace NUMINAMATH_GPT_total_dogs_equation_l1571_157100

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end NUMINAMATH_GPT_total_dogs_equation_l1571_157100


namespace NUMINAMATH_GPT_box_office_collection_l1571_157107

open Nat

/-- Define the total tickets sold -/
def total_tickets : ℕ := 1500

/-- Define the price of an adult ticket -/
def price_adult_ticket : ℕ := 12

/-- Define the price of a student ticket -/
def price_student_ticket : ℕ := 6

/-- Define the number of student tickets sold -/
def student_tickets : ℕ := 300

/-- Define the number of adult tickets sold -/
def adult_tickets : ℕ := total_tickets - student_tickets

/-- Define the revenue from adult tickets -/
def revenue_adult_tickets : ℕ := adult_tickets * price_adult_ticket

/-- Define the revenue from student tickets -/
def revenue_student_tickets : ℕ := student_tickets * price_student_ticket

/-- Define the total amount collected -/
def total_amount_collected : ℕ := revenue_adult_tickets + revenue_student_tickets

/-- Theorem to prove the total amount collected at the box office -/
theorem box_office_collection : total_amount_collected = 16200 := by
  sorry

end NUMINAMATH_GPT_box_office_collection_l1571_157107


namespace NUMINAMATH_GPT_fraction_squared_0_0625_implies_value_l1571_157102

theorem fraction_squared_0_0625_implies_value (x : ℝ) (hx : x^2 = 0.0625) : x = 0.25 :=
sorry

end NUMINAMATH_GPT_fraction_squared_0_0625_implies_value_l1571_157102


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1571_157112

theorem arithmetic_sequence_nth_term (x n : ℝ) 
  (h1 : 3*x - 4 = a1)
  (h2 : 7*x - 14 = a2)
  (h3 : 4*x + 6 = a3)
  (h4 : a_n = 3012) :
n = 392 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1571_157112


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1571_157121

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1571_157121
